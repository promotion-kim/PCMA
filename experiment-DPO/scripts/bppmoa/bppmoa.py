#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import os
from typing import List

import torch
from accelerate import Accelerator
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.trainer.bppmoa_trainer import BPPMOATrainer
from src.data.configs import DEFAULT_PROMPT_TEMPLATE
from src.utils import (
    disable_progress_bar_non_local_main,
    param_sharding_enabled,
    prepare_model_for_peft,
    print_local_main,
    set_seed,
)


disable_progress_bar_non_local_main()


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).lower()
    if x in {"true", "1", "yes", "y"}:
        return True
    if x in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {x}")


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    """
    Build TrainingArguments manually.

    We avoid tyro because recent transformers TrainingArguments contains
    nested fields such as AcceleratorConfig that can break tyro parsing.
    """

    kwargs = dict(
        output_dir=args.training_args_output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.training_args_per_device_train_batch_size,
        per_device_eval_batch_size=args.training_args_per_device_eval_batch_size,
        gradient_accumulation_steps=args.training_args_gradient_accumulation_steps,
        learning_rate=args.training_args_learning_rate,
        lr_scheduler_type=args.training_args_lr_scheduler_type,
        warmup_steps=args.training_args_warmup_steps,
        weight_decay=args.training_args_weight_decay,
        fp16=args.training_args_fp16,
        bf16=args.training_args_bf16,
        remove_unused_columns=False,
        run_name=args.training_args_run_name,
        report_to=args.training_args_report_to,
        num_train_epochs=args.training_args_num_train_epochs,
        logging_steps=args.training_args_logging_steps,
        save_steps=args.training_args_save_steps,
        eval_steps=args.training_args_eval_steps,
        eval_delay=args.training_args_eval_delay,
        save_total_limit=args.training_args_save_total_limit,
        load_best_model_at_end=args.training_args_load_best_model_at_end,
    )

    sig = inspect.signature(TrainingArguments.__init__)

    # transformers version compatibility:
    # older versions use evaluation_strategy, newer versions may use eval_strategy.
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = args.training_args_evaluation_strategy
    elif "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = args.training_args_evaluation_strategy

    # Some versions complain if save/eval strategy mismatch when
    # load_best_model_at_end=True. Keep the same step-based behavior.
    return TrainingArguments(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BPP-MOA policy distillation without tyro.")

    # Core args
    parser.add_argument("--sft_model_name", type=str, required=True)
    parser.add_argument("--bpp_dataset_dir", type=str, required=True)
    parser.add_argument("--use_flash_attention_2", action="store_true")
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--generate_during_eval", type=str2bool, default=True)

    # TrainingArguments-like args.
    # These names intentionally match run_bppmoa.sh, e.g. --training_args.output_dir.
    parser.add_argument("--training_args.output_dir", dest="training_args_output_dir", type=str, default="./output/dev/bppmoa")
    parser.add_argument("--training_args.run_name", dest="training_args_run_name", type=str, default="dev_bppmoa")
    parser.add_argument("--training_args.per_device_train_batch_size", dest="training_args_per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--training_args.per_device_eval_batch_size", dest="training_args_per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--training_args.gradient_accumulation_steps", dest="training_args_gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--training_args.learning_rate", dest="training_args_learning_rate", type=float, default=1e-4)
    parser.add_argument("--training_args.lr_scheduler_type", dest="training_args_lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--training_args.warmup_steps", dest="training_args_warmup_steps", type=float, default=0.1)
    parser.add_argument("--training_args.weight_decay", dest="training_args_weight_decay", type=float, default=0.05)
    parser.add_argument("--training_args.fp16", dest="training_args_fp16", type=str2bool, default=True)
    parser.add_argument("--training_args.bf16", dest="training_args_bf16", type=str2bool, default=False)
    parser.add_argument("--training_args.report_to", dest="training_args_report_to", type=str, default="wandb")
    parser.add_argument("--training_args.num_train_epochs", dest="training_args_num_train_epochs", type=float, default=3.0)
    parser.add_argument("--training_args.logging_steps", dest="training_args_logging_steps", type=int, default=10)
    parser.add_argument("--training_args.save_steps", dest="training_args_save_steps", type=float, default=0.25)
    parser.add_argument("--training_args.eval_steps", dest="training_args_eval_steps", type=float, default=0.25)
    parser.add_argument("--training_args.eval_delay", dest="training_args_eval_delay", type=float, default=0.25)
    parser.add_argument("--training_args.evaluation_strategy", dest="training_args_evaluation_strategy", type=str, default="steps")
    parser.add_argument("--training_args.save_total_limit", dest="training_args_save_total_limit", type=int, default=3)
    parser.add_argument("--training_args.load_best_model_at_end", dest="training_args_load_best_model_at_end", type=str2bool, default=True)

    # PEFT args.
    parser.add_argument("--peft_config.r", dest="peft_r", type=int, default=16)
    parser.add_argument("--peft_config.lora_alpha", dest="peft_lora_alpha", type=int, default=32)
    parser.add_argument("--peft_config.lora_dropout", dest="peft_lora_dropout", type=float, default=0.05)
    parser.add_argument("--peft_config.target_modules", dest="peft_target_modules", nargs="+", default=None)

    args, unknown = parser.parse_known_args()
    if unknown:
        print_local_main(f"[warning] ignoring unknown args: {unknown}")

    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    training_args = make_training_args(args)

    peft_config = LoraConfig(
        r=args.peft_r,
        lora_alpha=args.peft_lora_alpha,
        lora_dropout=args.peft_lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.peft_target_modules,
    )

    print_local_main("loading model...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_name,
        use_flash_attention_2=args.use_flash_attention_2,
        torch_dtype=torch.bfloat16,
        **({"device_map": {"": Accelerator().local_process_index}} if not param_sharding_enabled() else {}),
    )
    sft_model.config.update(
        {
            "use_cache": False,
            "pad_token_id": sft_model.config.eos_token_id,
        }
    )
    print_local_main(sft_model)
    print_local_main(peft_config)

    model = prepare_model_for_peft(
        sft_model,
        peft_config=peft_config,
        args=training_args,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_path = os.path.join(args.bpp_dataset_dir, "train")
    eval_path = os.path.join(args.bpp_dataset_dir, "validation")

    print_local_main(f"loading BPP-MOA target dataset from {args.bpp_dataset_dir}")
    train_dataset = load_from_disk(train_path)
    eval_dataset = load_from_disk(eval_path)

    required_cols = {"raw_prompt", "chosen", "rejected", "bpp_rho"}
    missing = required_cols - set(train_dataset.column_names)
    if missing:
        raise KeyError(f"{train_path} missing required columns: {sorted(missing)}")

    print_local_main("start BPP-MOA policy distillation...")
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

    if Accelerator().is_local_main_process and hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    trainer.train()

    save_name = "best_checkpoint" if training_args.load_best_model_at_end else "final_checkpoint"
    save_dir = os.path.join(training_args.output_dir, save_name)

    trainer.model.save_pretrained(save_dir)
    trainer.tokenizer.save_pretrained(save_dir)
    print_local_main(f"saved: {save_dir}")


if __name__ == "__main__":
    main()
