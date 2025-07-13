import torch
import functools
from itertools import repeat
from easydict import EasyDict
import torch.distributed as dist
from torch.cuda.amp import autocast
from trainer.utils import build_optimizer
from utils import logger, AverageMeter
from datasets.editing_dataset import ImageEditingDataloader
from trainer.utils import TrainerBase, print_model_param_num
from models import VLChatProcessor, MultiModalityCausalLM, MultiModalityConfig

def repeater(data_loader):
    for i, loader in enumerate(repeat(data_loader)):
        sampler = getattr(loader, "sampler", None)
        if sampler is not None:
            sampler.set_epoch(i)
        for data in loader:
            yield data

def train_setup(model: MultiModalityCausalLM):
    for n, p in model.language_model.named_parameters():
        p.requires_grad = True
    model.language_model.train()
    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    for n, p in model.gen_embed.named_parameters():
        p.requires_grad = True
    model.gen_embed.train()
    for n, p in model.gen_head.named_parameters():
        p.requires_grad = True
    model.gen_head.train()
    for n, p in model.gen_aligner.named_parameters():
        p.requires_grad = True
    model.gen_aligner.train()
    for n, p in model.aligner.named_parameters():
        p.requires_grad = True
    model.aligner.train()
    for n, p in model.vision_model.named_parameters():
        p.requires_grad = False
    model.vision_model.eval()
    for n, p in model.gen_vision_model.named_parameters():
        p.requires_grad = False
    model.gen_vision_model.eval()

class ImageEditingTrainer(TrainerBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_loader = ImageEditingDataloader(cfg)
        self.cfg.dataloader_len = len(self.data_loader)
        self._data_loader_iter = iter(repeater(self.data_loader))
        self.totals = EasyDict()
        self.totals.epochs = self.cfg.optimize.max_epochs
        self.totals.iter_per_epoch = len(self.data_loader)
        self.totals.total_iters = self.cfg.optimize.max_epochs * self.totals.iter_per_epoch 
        self.dist = EasyDict()
        self.dist.rank = dist.get_rank()
        self.dist.world_size = dist.get_world_size()
        self.pretrain_model = self.cfg.model.get("pretrain_model", None)
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.cfg.model.processor_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.model_before_ddp = MultiModalityCausalLM.from_pretrained(cfg.model.model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
        train_setup(self.model_before_ddp)

        if self.pretrain_model is not None:
            state_dict = torch.load(self.pretrain_model, "cpu")
            if "ema" in state_dict:
                module_dict = state_dict["ema"]
            elif "module" in state_dict:
                module_dict = state_dict["module"]
            else:
                module_dict = state_dict
            missing, unexpected = self.model_before_ddp.load_state_dict(module_dict, strict=False)
            del state_dict
            print(f"GPT: Restored from {self.pretrain_model} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            if len(missing) > 0:
                print(f"Missing Keys: {missing}")
                print(f"Unexpected Keys: {unexpected}")

        print_model_param_num(cfg.model, self.model_before_ddp)

        if not self.cfg.common.use_fsdp:
            raise NotImplementedError
        print("USING FSDP from Pytorch...")
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            BackwardPrefetch,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.llama.modeling_llama  import LlamaDecoderLayer 
        my_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )
        self.use_bf16 = cfg.common.use_bf16
        self.use_fp16 = cfg.common.use_fp16
        if not self.use_bf16:
            raise NotImplementedError
        logger.info("Use bfloat16 training...")
        fpSixteen = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        self.model = FSDP(self.model_before_ddp,
                            auto_wrap_policy=my_auto_wrap_policy,
                            mixed_precision=fpSixteen if self.use_bf16 else None,
                            device_id=torch.cuda.current_device(),
                            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                            use_orig_params=True,
                            limit_all_gathers=True)
        print('model already load')
        self.optimizer = build_optimizer(cfg, self.model)
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        self._scaler = ShardedGradScaler(enabled=self.use_fp16)
        dist.barrier()
        self.model.train()
        self.gradient_accumulation_steps = self.cfg.optimize.get("gradient_accumulation_steps", 1)

        self.input_token_max_len = 700
        self.meters = EasyDict()
        self.meters["batch_time"] = AverageMeter(self.cfg.common.log_interval, fstr="%.3f")
        self.meters["loss"] = AverageMeter(self.cfg.common.log_interval, fstr="%.5f")
        self.meters["grad_norm"] = AverageMeter(self.cfg.common.log_interval, fstr="%.5f")
        self.meters["lr"] = AverageMeter(self.cfg.common.log_interval, fstr="%.3e")

    def get_next_data(self):
        input_data = next(self._data_loader_iter)
        batch_size = len(input_data)
        input_token_max_len = self.input_token_max_len
        batched_input_ids = torch.full(
            (batch_size, input_token_max_len), self.vl_chat_processor.pad_id
        ).long()
        batched_attention_mask = torch.zeros((batch_size, input_token_max_len)).long()
        batched_images_seq_mask = torch.zeros((batch_size, input_token_max_len)).bool()
        image1 = torch.stack([input_data[k]['image1'] for k in range(batch_size)], dim=0)
        image2 = torch.stack([input_data[k]['image2'] for k in range(batch_size)], dim=0)
        for k in range(batch_size):
            input_ids = input_data[k]['input_ids']
            seq_len = len(input_ids)
            batched_attention_mask[k, -seq_len:] = 1
            batched_input_ids[k, -seq_len:] = torch.LongTensor(input_ids)  
            batched_images_seq_mask[k, -seq_len:] = input_ids == self.vl_chat_processor.image_id
        return {'input_ids': batched_input_ids.cuda(), 'attention_mask': batched_attention_mask.cuda(), 'image1':image1.cuda(), 'image2':image2.cuda(), 'image_seq_mask': batched_images_seq_mask.cuda(), 'task_type': 2}

    def run_step(self):
        model_input = self.get_next_data()
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=torch.bfloat16, cache_enabled=False):
            self._loss = self.model(**model_input)
        self._loss.backward()
        total_norm = self._clip_grad_norm(max_norm=1.0)
        self.optimizer.step()
        reduced_loss = self._loss.clone().detach() / self.dist.world_size
        reduced_grad_norm = total_norm.clone().detach() / self.dist.world_size
        self.meters.loss.reduce_update(reduced_loss)
        self.meters.grad_norm.reduce_update(reduced_grad_norm)
        self.meters.lr.reduce_update(torch.tensor(self.optimizer.param_groups[0]['lr']).cuda() / self.dist.world_size)

    def _clip_grad_norm(self, max_norm):
        if hasattr(self.cfg.common, "use_fsdp") and self.cfg.common.use_fsdp:
            total_norm = self.model.clip_grad_norm_(max_norm)
        else:
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
        return total_norm