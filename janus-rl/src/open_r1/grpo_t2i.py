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

import logging
import os
from dataclasses import dataclass, field
import datasets
import torch
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from configs import GRPOConfig
from utils.wandb_logging import init_wandb_training
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from grpo_trainer_t2iv1 import GRPOTrainer
import base64
import io
from tqdm import tqdm
import torch.distributed as dist
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )

    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )

def send_request(body, idx):
    url = "XXX"
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "MPS-app-name": "your-app-name",
        "MPS-http-version": "1.0",
        "MPS-trace-id": "your-trace-id",
    }
    curtime = 0
    while True:
        try:
            curtime+=1
            r = requests.post(url=url, json=body, headers=headers)
          
            res =  r.json()
            if res['success']:
                score = res['resultMap']['scores']
                return (score, idx)
            else:
                if curtime>=5:
                    print('error!')
                    return (0.0, idx)
        except Exception as e:
            if curtime>=5:
                print('error!')
                
                return (0.0, idx)

def intern_reward(reward_model, images, prompts, apply_api=True):
    if apply_api == False:
        with torch.no_grad():
            score = reward_model.evaluate(images, prompts)
            return score
    else:
        pts = []
        for i in range(len(images)):
            input_image = images[i]
            prompt = prompts[i]
            input_bytes = io.BytesIO()
            input_image.save(input_bytes, format='PNG')
            curdata = {"features": {}, "tensorFeatures": {
                "images": {"shapes": [1], "byteValues": base64.b64encode(input_bytes.getvalue()).decode("utf-8")},
                "captions": {"shapes": [1], "stringValues": [prompt]}
            }}
            pts.append((i, curdata))
    
        concurrent_requests = 16
        results = []
        scores  = [0.0] * len(prompts)

        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = {executor.submit(send_request, body, i): body for i, body in pts}
            for future in tqdm(as_completed(futures), total=len(pts), desc="Processing Requests"):
                result = future.result()
                results.append(result)
        
        for res in results:
            scores[res[1]] = res[0]
        for ss in scores:
            assert ss>=0.0
        
        return scores
        
class TxtLoggingCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if dist.get_rank()==0:
            with open(self.log_file, "a") as f:
                f.write(f'{state.global_step}: '+str(logs) + "\n")

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    nickname = training_args.output_dir.split('/')[-1]
    os.makedirs('logs/', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"logs/{nickname}.log"),
            logging.StreamHandler()
        ]
    ) 
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    from mydataset import MyT2IDataset
    train_dataset = MyT2IDataset(data_path=training_args.datap)

    reward_funcs = [intern_reward]

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs
    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=training_args.model_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        # callbacks=get_callbacks(training_args, model_args),
        callbacks=[TxtLoggingCallback(log_file=f"logs/{nickname}_log.txt")]
    )
    print(trainer.optimizer)
    print("*" * 30)
    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    # metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print(training_args)
    main(script_args, training_args, model_args)
