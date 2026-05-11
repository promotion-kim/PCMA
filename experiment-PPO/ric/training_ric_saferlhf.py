"""RiC SFT training for SafeRLHF prompt format.

The original RiC training code uses the HH-RLHF response delimiter
``\n\nAssistant:``. SafeRLHF experiments in this project use
``BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:``.
This training helper keeps the original SFT objective but uses a configurable
response template, defaulting to ``ASSISTANT:``.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_from_disk, disable_caching
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, set_seed

from utils import load_main_tokenizer, print_trainable_parameters, save_configs


disable_caching()


def train_ric_saferlhf(
    base_model_name: str,
    train_dataset: str,
    save_path: str,
    tokenizer_name: Optional[str] = None,
    peft_name: Optional[str] = None,
    training_steps: int = 20000,
    learning_rate: float = 1e-5,
    iter: int = 0,
    lr_scheduler_type: str = "linear",
    args: Optional[object] = None,
    response_template: str = "ASSISTANT:",
):
    set_seed(8888 + int(iter))
    tokenizer_name = tokenizer_name or base_model_name

    if args is None:
        raise ValueError("args must be provided; it supplies batch and logging configuration.")

    training_args = TrainingArguments(
        max_steps=training_steps,
        output_dir=os.path.join(args.save_directory, args.wandb_name),
        dataloader_drop_last=True,
        eval_steps=max(training_steps * 2, 1),
        save_steps=max(training_steps * 2, 1),
        save_strategy="steps",
        logging_steps=10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=0,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=False,
        weight_decay=0.01,
        bf16=True,
        run_name=args.wandb_name,
        report_to="none" if getattr(args, "disable_wandb", False) else getattr(args, "log_with", "wandb"),
        ddp_find_unused_parameters=False,
    )

    save_configs(training_args, save_path)

    accelerator = Accelerator()
    process_id = accelerator.local_process_index
    gpu_id = process_id
    print(f"base model: {base_model_name}")
    print(f"process: {process_id}, model gpu id: {gpu_id}")

    tokenizer = load_main_tokenizer(tokenizer_name)
    dataset = load_from_disk(train_dataset) if isinstance(train_dataset, str) else train_dataset
    selected_index = np.arange(0, len(dataset))
    np.random.shuffle(selected_index)
    dataset = dataset.select(selected_index)
    print(f"Size of the RiC SafeRLHF train set: {len(dataset)}")

    if training_steps <= 0:
        print("training_steps <= 0: skipping training and returning dataset")
        return dataset

    lora_config = LoraConfig(
        r=getattr(args, "lora_r", 64),
        lora_alpha=getattr(args, "lora_alpha", 128),
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )

    if getattr(args, "load_in_8bit", True):
        model = AutoModelForCausalLM.from_pretrained(base_model_name, load_in_8bit=True, device_map=gpu_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=gpu_id if torch.cuda.is_available() else None,
        )

    model.resize_token_embeddings(len(tokenizer))
    if peft_name:
        model = PeftModel.from_pretrained(model, peft_name, is_trainable=True)

    print_trainable_parameters(model)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        packing=False,
        dataset_text_field="query",
        data_collator=collator,
    )
    trainer.train()

    if process_id == 0:
        print("Saving last checkpoint of the model")
        trainer.model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    accelerator.wait_for_everyone()
    return dataset
