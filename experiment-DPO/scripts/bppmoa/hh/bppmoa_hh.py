#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train 3D HH BPP-MOA policy by soft-label DPO-style distillation.

This is stage 3 of the paper-faithful pipeline.  It expects a target dataset
created by build_bpp_targets_hh.py containing raw_prompt/chosen/rejected/bpp_rho.

By construction, raw_prompt is already fully formatted during target building,
so the default prompt_template is the identity template: {raw_prompt}.
"""

from __future__ import annotations

import argparse
import inspect
import os
from typing import List

import torch
from accelerate import Accelerator
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.trainer.bppmoa_trainer import BPPMOATrainer
from scripts.bppmoa.hh.bppmoa_hh_utils import ensure_tokenizer, str2bool, str_to_torch_dtype


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs = dict(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
        run_name=args.run_name,
        report_to=args.report_to,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_delay=args.eval_delay,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = args.evaluation_strategy
        kwargs["save_strategy"] = args.save_strategy
    elif "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = args.evaluation_strategy
        kwargs["save_strategy"] = args.save_strategy

    # Remove kwargs unsupported by old transformers versions.
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return TrainingArguments(**kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 3D HH BPP-MOA policy distillation.")
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--bpp_dataset_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--prompt_template", type=str, default="{raw_prompt}")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--num_proc", type=int, default=4)
    p.add_argument("--generate_during_eval", type=str2bool, default=False)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--torch_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--use_flash_attention_2", type=str2bool, default=False)
    p.add_argument("--trust_remote_code", type=str2bool, default=True)
    p.add_argument("--gradient_checkpointing", type=str2bool, default=False)

    # Training args
    p.add_argument("--run_name", type=str, default="bppmoa_hh")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--fp16", type=str2bool, default=False)
    p.add_argument("--bf16", type=str2bool, default=True)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=float, default=0.25)
    p.add_argument("--eval_steps", type=float, default=0.25)
    p.add_argument("--eval_delay", type=float, default=0.25)
    p.add_argument("--evaluation_strategy", type=str, default="steps")
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--load_best_model_at_end", type=str2bool, default=True)

    # LoRA args
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    training_args = make_training_args(args)
    dtype = str_to_torch_dtype(args.torch_dtype)

    accelerator = Accelerator()
    device_map = {"": accelerator.local_process_index} if torch.cuda.is_available() else None

    if accelerator.is_local_main_process:
        print("============================================================")
        print("[BPP-MOA-HH policy distillation]")
        print(f"model          = {args.model_name}")
        print(f"bpp_dataset    = {args.bpp_dataset_dir}")
        print(f"output_dir     = {args.output_dir}")
        print(f"beta           = {args.beta}")
        print(f"prompt_template= {args.prompt_template}")
        print("============================================================")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        use_flash_attention_2=bool(args.use_flash_attention_2),
        trust_remote_code=bool(args.trust_remote_code),
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model.config.use_cache = False
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id

    if args.gradient_checkpointing:
        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=bool(args.trust_remote_code))
    ensure_tokenizer(tokenizer)

    train_dataset = load_from_disk(os.path.join(args.bpp_dataset_dir, "train"))
    eval_dataset = load_from_disk(os.path.join(args.bpp_dataset_dir, "validation"))
    required_cols = {"raw_prompt", "chosen", "rejected", "bpp_rho"}
    missing = required_cols - set(train_dataset.column_names)
    if missing:
        raise KeyError(f"Target dataset missing columns: {sorted(missing)}")

    if accelerator.is_local_main_process and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    trainer = BPPMOATrainer(
        model=model,
        beta=args.beta,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_proc=args.num_proc,
        generate_during_eval=args.generate_during_eval,
        prompt_template=args.prompt_template,
    )
    trainer.train()

    save_name = "best_checkpoint" if training_args.load_best_model_at_end else "final_checkpoint"
    save_dir = os.path.join(training_args.output_dir, save_name)
    trainer.model.save_pretrained(save_dir)
    trainer.tokenizer.save_pretrained(save_dir)
    if accelerator.is_local_main_process:
        print(f"[BPP-MOA-HH] saved: {save_dir}")


if __name__ == "__main__":
    main()
