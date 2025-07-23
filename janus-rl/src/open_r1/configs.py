# coding=utf-8
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

from dataclasses import dataclass, field
from typing import Optional

from sympy import Float
import trl


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None, metadata={"help": "The optional system prompt to use for benchmarking."}
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    weight : Optional[float] = field(
        default=0.5,
        metadata={'help': 'help'}
    )
    datap : Optional[str] = field(
        default='',
        metadata={'help': 'help'}
    )
    guidance_scale : Optional[float] = field(
        default=4.0,
        metadata={'help': 'help'}
    )
    preserving : Optional[float] = field(
        default=1.0,
        metadata={'help': 'help'}
    )
    following : Optional[float] = field(
        default=1.0,
        metadata={'help': 'help'}
    )
    generate_with_cfg: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    internvl_tp: Optional[str] = field(
        default='26b',
        metadata={'help': 'use clip score in reward model, default to be true'}
    )
    pretrain_path: Optional[str] = field(
        default=None,
        metadata={'help': 'use clip score in reward model, default to be true'}
    )
    ref_pretrain_path: Optional[str] = field(
        default=None,
        metadata={'help': 'use clip score in reward model, default to be true'}
    )
    reverse: bool = field(default=False, metadata={"help": "XXX."})
    addselft2i: bool = field(default=False, metadata={"help": "XXX."})
    set_epsilon: bool = field(default=False, metadata={"help": "XXX."})
    epsilon: Float = field(
        default=0.2,
        metadata={'help': 'coefficient that clip the loss'}
    )
    warm_up_steps: int = field(
        default=45,
        metadata={'help': 'warm up steps'}
    )
    convert_steps: int = field(
        default=500,
        metadata={'help': 'steps that lr begins to drop'}
    )
    convert_lr: Float = field(
        default=1e-6,
        metadata={'help': 'lr that the optimizer drops to after the convert_steps'}
    )
    min_lr: Float = field(
        default=2e-7,
        metadata={'help': 'min lr'}
    )
    model_path: str = field(
        default='/cache/llama3-hf/iter_5000',
        metadata={'help': 'min lr'}
    )
    use_self_lr_scheduler: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
