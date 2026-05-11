"""Train RiC on SafeRLHF reward-conditioned data.

This script does not take a preference weight. RiC is trained once on examples
conditioned by their own reward-vector scores. Preference weights are used later
by ``evaluation_ric_saferlhf.py`` to choose target score conditions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser

from training_ric_saferlhf import train_ric_saferlhf
from utils import clean_gpu_memory, save_configs


@dataclass
class ScriptArguments:
    log_with: Optional[str] = field(default=None)
    disable_wandb: bool = field(default=False)
    save_directory: str = field(default="./logs_ric_saferlhf")
    wandb_name: str = field(default="ric_saferlhf_helpful_harmless")

    base_model_name: str = field(default="PKU-Alignment/alpaca-7b-reproduced")
    peft_name: str = field(default="")
    train_dataset_path: str = field(default="./datasets/ric_saferlhf_helpful_harmless.hf")

    training_steps: int = field(default=20000)
    learning_rate: float = field(default=1e-5)
    batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    max_grad_norm: float = field(default=1.0)
    load_in_8bit: bool = field(default=True)
    response_template: str = field(default="ASSISTANT:")

    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)


args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]

if args.disable_wandb:
    os.environ["WANDB_DISABLED"] = "true"

save_root = os.path.join(args.save_directory, args.wandb_name)
os.makedirs(save_root, exist_ok=True)

save_configs(
    {
        "method": "RiC-SafeRLHF",
        "base_model_name": args.base_model_name,
        "train_dataset_path": args.train_dataset_path,
        "training_steps": args.training_steps,
        "learning_rate": args.learning_rate,
        "response_template": args.response_template,
    },
    save_root,
)

train_ric_saferlhf(
    base_model_name=args.base_model_name,
    train_dataset=args.train_dataset_path,
    save_path=os.path.join(save_root, "model_iter0"),
    tokenizer_name=args.base_model_name,
    peft_name=args.peft_name if len(args.peft_name) > 0 else None,
    training_steps=args.training_steps,
    learning_rate=args.learning_rate,
    args=args,
    response_template=args.response_template,
)

clean_gpu_memory()
