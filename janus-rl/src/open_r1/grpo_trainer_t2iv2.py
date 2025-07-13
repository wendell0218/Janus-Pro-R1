# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from unittest.mock import patch

import torch
import torch.utils.data
import math
import torch.optim as optim
import transformers
from accelerate.utils import broadcast_object_list, gather_object
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.import_utils import is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad
from llama import JanusLLamaModel
from models import VLChatProcessor

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

from transformers import AutoProcessor, AutoModel

def lr_linear_early_drop_with_warm_up(x, *, warm_up_steps=45, convert_steps=500, total_steps=2000, max_lr=5e-6, convert_lr=1e-6, min_lr=2e-7):
    # return the factor
    max_factor = 1
    min_factor = min_lr / max_lr
    convert_factor = convert_lr / max_lr

    if x < warm_up_steps:
        # warm up steps
        k = max_factor / warm_up_steps
        lr = k * x
    elif warm_up_steps <= x < convert_steps:
        # high lr stage
        k = (convert_factor - max_factor) / (convert_steps - warm_up_steps)
        lr = k * x - k * warm_up_steps + max_factor
    else:
        k = (min_factor - convert_factor) / (total_steps - convert_steps)
        lr = k * x - k * convert_steps + convert_factor
    
    return lr

class GRPOTrainer(Trainer):
    
    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        # self.set_special_tokens()
        # self.set_model()
        self.args = args
        self.apply_api = None
        model_name = model
        self.model_name = model_name
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        # Models
        # Trained model
        # from llama import MyLLamaModel
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            # model_init_kwargs["use_cache"] = (
            #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            # )
            model_init_kwargs["use_cache"] = None
            model_init_kwargs['attn_implementation'] = 'flash_attention_2'
            print(model_init_kwargs)
            model = JanusLLamaModel.from_pretrained(model_name, revision='main', trust_remote_code=False, torch_dtype=torch.bfloat16)
            if args.pretrain_path is not None:
                state_dict = torch.load(f"{args.pretrain_path}", map_location="cpu")
                model.load_state_dict(state_dict)
            else:
                state_dict = None
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            print('set peft!!!!!!!!')
            model = get_peft_model(model, peft_config)

        self.ref_model = JanusLLamaModel.from_pretrained(model_name, revision='main', trust_remote_code=False, torch_dtype=torch.bfloat16)
        if args.ref_pretrain_path is not None:
            state_dict = torch.load(f"{args.ref_pretrain_path}", map_location="cpu")
            self.ref_model.load_state_dict(state_dict)
        else:
            state_dict = None
        parameter_names = [n for n, _ in self.ref_model.named_parameters()]
        for param_name in parameter_names:
            param = self.ref_model.get_parameter(param_name)
            param.requires_grad = False 
        self.ref_model.eval()

        # Processing class
        if processing_class is None:
            # specify the path to the model
            processor_path = self.model_name

            vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(processor_path)
            processing_class  = vl_chat_processor.tokenizer

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm

        self.beta = args.beta
        self.epsilon = args.epsilon

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also used for training. This may lead to unexpected "
                        "behavior. It is recommended to use a dedicated device for vLLM."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        # dtype=self.args.vllm_dtype,
                        # # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # # This is particularly useful here because we generate completions from the same prompts.
                        # enable_prefix_caching=True,
                        # max_model_len=self.args.vllm_max_model_len,
                    )
                self.sampling_params = SamplingParams(
                    n=self.num_generations,
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad checkpointing

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=self.num_generations,
                pad_token_id=processing_class.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
        
        self.set_special_tokens()
        self.set_model()

    def set_special_tokens(self, task_type='t2i'):
        model_path = self.model_name
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.tokenizer.padding_side = 'left'

        self.guidance_scale = self.args.guidance_scale
        self.generate_with_cfg = self.args.generate_with_cfg
        self.set_epsilon = self.args.set_epsilon
        
    def set_model(self):
        if '8' in self.args.internvl_tp:
            from internvl_img import InternVLReward
            self.reward_model = InternVLReward()
            self.apply_api=False
        else:
            self.apply_api=True

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_embeds, output_ids, attention_mask, logits_to_keep=0, addcfg=True, visual=True):
        if visual:
            if output_ids.shape[0] < input_embeds.shape[0]:
                new_img_ids = torch.repeat_interleave(output_ids, 2, dim=0) 
            else:
                new_img_ids = output_ids
            output_embeds = model.gen_aligner(model.gen_embed(new_img_ids))
        else:
            addcfg = False
            output_embeds = model.language_model.get_input_embeddings()(output_ids)

        inputs_embeds = torch.cat([input_embeds, output_embeds], dim=1)

        if addcfg == False:
            if visual:
                outputs = model.language_model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)  # (B, L, V)
                hidden_states = outputs.last_hidden_state
                logits = model.gen_head(hidden_states)
                logits = logits[:, -1-logits_to_keep:-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            else:
                logits = model.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
                logits = logits[:, -1-logits_to_keep:-1, :]
            input_ids = output_ids.long()  # (B, L-1), exclude the first input ID since we don't have logits for it
        
        else:
            outputs = model.language_model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)  # (B, L, V)
            hidden_states = outputs.last_hidden_state
            logits = model.gen_head(hidden_states)
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_cond - (self.guidance_scale-1) / self.guidance_scale *logit_uncond
            logits = logits[:, -1-logits_to_keep:-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = output_ids.long()

        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        
        ins_prompts = [inp['text'] for inp in inputs]
        prompts = ins_prompts

        allprompts = []
        for prompt in prompts:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": prompt,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.vl_chat_processor.sft_format,
                system_prompt="",
            )
            prompt = sft_format + self.vl_chat_processor.image_start_tag
            allprompts.append(prompt)
        
        instruction = self.tokenizer(
            allprompts,
            return_tensors="pt",
            padding='longest',
            max_length=200, truncation=True
        ).to(device)

        # bsz, L, dtype = instruction['input_ids'].size(0), instruction['input_ids'].size(1), instruction['input_ids'].dtype
        prompt_ids = instruction['input_ids']
        prompt_mask = instruction['attention_mask']

        # Generate completions using either vLLM or regular generation
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        if self.args.use_vllm:
            raise NotImplementedError
        else:
            prompt_ids = torch.repeat_interleave(prompt_ids, self.num_generations, dim=0) 
            prompt_mask = torch.repeat_interleave(prompt_mask, self.num_generations, dim=0) 
            if self.guidance_scale is not None and self.generate_with_cfg:
                set_cfg = True
                my_guidance_scale = self.guidance_scale
            else:
                set_cfg = False
                my_guidance_scale = None
            self.model.eval()

            with unwrap_model_for_generation(self.model, self.accelerator, gather_deepspeed3_params=False) as unwrapped_model:
                (img_ids_1, all_imgs_1), (img_ids_2, all_imgs_2), (output_text_ids, selfcheck, attention_mask_txt), (embeds_1, attention_mask_1), (embeds_2, attention_mask_2), (embeds_3, attention_mask_3) = \
                    unwrapped_model.generate_with_refine(
                        vl_chat_processor=self.vl_chat_processor,
                        input_ids=prompt_ids, attention_mask=prompt_mask, 
                        cfg_weight=my_guidance_scale,
                        cur_step=self.state.global_step,
                    )

                img_ids = torch.zeros(img_ids_1.shape, dtype=img_ids_1.dtype, device=img_ids_1.device)
                imgs_1 = all_imgs_1
                imgs_2 = []
                for cc in range(selfcheck.size(0)):
                  if selfcheck[cc]==True:
                    imgs_2.append(all_imgs_1[cc])
                    img_ids[cc, :] = img_ids_1[cc, :]
                  else:
                    imgs_2.append(all_imgs_2[cc])
                    img_ids[cc, :] = img_ids_2[cc, :]
                        
            self.model.train()

        # Mask everything after the first EOS token
        completion_mask_1 = torch.ones((img_ids_1.size(0), img_ids_1.size(1)), dtype=torch.long, device=device)
        completion_mask_1_cfg = torch.repeat_interleave(completion_mask_1, 2, dim=0) 
        completion_mask_2 = attention_mask_txt
        completion_mask_3 = torch.ones((img_ids.size(0), img_ids.size(1)), dtype=torch.long, device=device)
        completion_mask_3_cfg = torch.repeat_interleave(completion_mask_3, 2, dim=0) 
       
        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask_1 = torch.cat((attention_mask_1, completion_mask_1_cfg), dim=1)
        attention_mask_2 = torch.cat((attention_mask_2, completion_mask_2), dim=1)
        attention_mask_3 = torch.cat((attention_mask_3, completion_mask_3_cfg), dim=1)
        
        logits_to_keep_1 = img_ids_1.size(1)
        logits_to_keep_2 = output_text_ids.size(1)
        logits_to_keep_3 = img_ids.size(1)
        
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps_1 = self._get_per_token_logps(
                    self.ref_model, input_embeds=embeds_1, output_ids=img_ids_1, attention_mask=attention_mask_1, logits_to_keep=logits_to_keep_1, addcfg=True, visual=True
                )
            else:
                raise NotImplementedError
            
            if self.ref_model is not None:
                ref_per_token_logps_2 = self._get_per_token_logps(
                    self.ref_model, input_embeds=embeds_2, output_ids=output_text_ids, attention_mask=attention_mask_2, logits_to_keep=logits_to_keep_2, addcfg=False, visual=False
                )
            else:
                raise NotImplementedError
            
            if self.ref_model is not None:
                ref_per_token_logps_3 = self._get_per_token_logps(
                    self.ref_model, input_embeds=embeds_3, output_ids=img_ids, attention_mask=attention_mask_3, logits_to_keep=logits_to_keep_3, addcfg=True, visual=True
                )
            else:
                raise NotImplementedError
          
        comp_reward_len = len(imgs_1)
        all_imgs = imgs_1 + imgs_2
        all_prompts = prompts + prompts
        
        with torch.no_grad():
            rewards_per_func = torch.zeros(len(prompts)*2, 1, device=device)
            if self.apply_api:
                gather_imgs = gather_object(all_imgs)
                gather_prompts = gather_object(all_prompts)
                if self.accelerator.is_main_process:
                    print(len(gather_prompts))
                    score1 = self.reward_funcs[0](None, gather_imgs, gather_prompts, self.apply_api)
                else:
                    score1 = [None] * len(gather_prompts)
                score1 = broadcast_object_list(score1, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts) * 2,
                    (self.accelerator.process_index + 1) * len(prompts) * 2,
                )
                score1 = score1[process_slice]
                rewards_per_func[:, 0] = torch.tensor(score1, dtype=torch.float32, device=device)
        
        allrewards = rewards_per_func.sum(dim=1)

        all_self_check = self.accelerator.gather_for_metrics(selfcheck.float()).mean().item()
        self._metrics[f"rewards/selfcheck"].append(all_self_check)
        rewards1 = allrewards[:comp_reward_len]
        rewards3 = allrewards[comp_reward_len:]
        rewards2 = 1.0-torch.abs(allrewards[:comp_reward_len]-selfcheck.int())
        names = ['first_gen', 'compre', 'final_gen']
        
        all_rewards_list = [rewards1, rewards2, rewards3]
        all_advantages = []
        
        tt = 0
        for j, rewards in enumerate(all_rewards_list):
            # Compute grouped-wise rewards
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            all_advantages.append(advantages)

            # Log the metrics
            rr = self.accelerator.gather_for_metrics(rewards).mean().item()
            tt+=rr
            self._metrics[names[j]].append(rr)
            self._metrics[f'{names[j]}_std'].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        
        self._metrics["tot_reward"].append(tt)
        

        return [
            {
                "prompt_embeds": embeds_1,
                "prompt_mask": attention_mask_1,
                "completion_ids": img_ids_1,
                'completion_mask':completion_mask_1,
                "ref_per_token_logps": ref_per_token_logps_1,
                "advantages": all_advantages[0],
            },{
                "prompt_embeds": embeds_2,
                "prompt_mask": attention_mask_2,
                "completion_ids": output_text_ids,
                'completion_mask':completion_mask_2,
                "ref_per_token_logps": ref_per_token_logps_2,
                "advantages": all_advantages[1],
            },{
                "prompt_embeds": embeds_3,
                "prompt_mask": attention_mask_3,
                "completion_ids": img_ids,
                'completion_mask':completion_mask_3,
                "ref_per_token_logps": ref_per_token_logps_3,
                "advantages": all_advantages[2],
          }
        ]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        loss_sum = 0.0
        for j, curinputs in enumerate(inputs):
           
            prompt_embeds, prompt_mask = curinputs["prompt_embeds"], curinputs["prompt_mask"]
            completion_ids, completion_mask = curinputs["completion_ids"], curinputs['completion_mask']

            logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

            if j!=1:
                addcfg = True
                visual = True
            else:
                addcfg = False
                visual = False

            per_token_logps = self._get_per_token_logps(
                model, input_embeds=prompt_embeds, output_ids=completion_ids, attention_mask=prompt_mask, logits_to_keep=logits_to_keep, addcfg=addcfg, visual=visual
            )
            
            # Compute the KL divergence between the model and the reference model
            ref_per_token_logps = curinputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

            # x - x.detach() allows for preserving gradients from x
            advantages = curinputs["advantages"]
            if self.set_epsilon==False:
                per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
                per_token_loss = -(per_token_loss - self.beta * per_token_kl)

            else:
                coef_1 = torch.exp(per_token_logps - per_token_logps.detach())
                coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
                per_token_loss1 = coef_1 * advantages.unsqueeze(1)
                per_token_loss2 = coef_2 * advantages.unsqueeze(1)
                per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
                if self.beta != 0.0:
                    per_token_loss = per_token_loss + self.beta * per_token_kl

            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

            # Log the metrics
            completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
            self._metrics[f"completion_length_{j}"].append(completion_length)

            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics[f"kl_{j}"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        
            loss_sum += loss
        loss_sum = loss_sum / len(inputs)

        return loss_sum

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if not self.args.use_self_lr_scheduler:
            return super().create_scheduler(num_training_steps, optimizer)
        from functools import partial
        if self.lr_scheduler is None:
            lr_lambda = partial(lr_linear_early_drop_with_warm_up,
                                warm_up_steps=self.args.warm_up_steps,
                                convert_steps=self.args.convert_steps,
                                total_steps=num_training_steps,
                                max_lr=self.args.learning_rate,
                                convert_lr=self.args.convert_lr,
                                min_lr=self.args.min_lr
                                )
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer if optimizer is None else optimizer, 
                                                            lr_lambda
                                                            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

