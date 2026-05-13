#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRL/DeepSpeed-free CPC-MODPO entry point, with MODPO-aligned defaults.

This script is intended for fair comparison with scripts/modpo/modpo.py and
scripts/modpo/run.sh in environments where the original DPOTrainer import chain
fails due to TRL/DeepSpeed/Torch version conflicts.

Fair-comparison choices:
  - same SFT model / safety margin adapter
  - same train/eval split names
  - same max_length, batch sizes, gradient accumulation, LR, epochs
  - same LoRA target modules/r/alpha/dropout as run.sh
  - same W&B project style
  - same MODPO algebra, replacing w_i with CPC coefficients c_i
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, disable_caching, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.trainer.cpcmodpo_trainer import CPCMODPOTrainer, CPCPreferenceDataCollator


DEFAULT_PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"


def split_csv(x: str) -> List[str]:
    return [v.strip() for v in str(x).split(",") if v.strip()]


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).lower()
    if x in {"true", "1", "yes", "y"}:
        return True
    if x in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean: {x}")


def common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def strip_objective_suffix(dataset_name: str) -> Tuple[str, Optional[str]]:
    if dataset_name.endswith("-better"):
        return dataset_name[:-len("-better")], "better"
    if dataset_name.endswith("-safer"):
        return dataset_name[:-len("-safer")], "safer"
    return dataset_name, None


def load_any_split(dataset_name: str, split: str):
    """Load the same underlying HF dataset/split as MODPO.

    For names such as PKU-Alignment/PKU-SafeRLHF-10K-better, HF does not have a
    separate dataset with that suffix. We load PKU-Alignment/PKU-SafeRLHF-10K and
    use the suffix only to decide which response is preferred.
    """
    base_name, _ = strip_objective_suffix(dataset_name)
    names = [dataset_name]
    if base_name != dataset_name:
        names.append(base_name)

    last_error = None
    for name in names:
        try:
            return load_dataset(name, split=split), name
        except Exception as e:
            last_error = e

    # Fallback for datasets without validation split.
    if split in {"validation", "eval", "dev"}:
        for name in names:
            try:
                return load_dataset(name, split="train[:5%]"), name
            except Exception as e:
                last_error = e

    raise RuntimeError(f"Failed to load {dataset_name} split={split}. Last error: {last_error}")


def format_prompt_and_raw(ex: dict, prompt_template: str) -> Tuple[str, str]:
    """Return formatted prompt and raw prompt.

    MODPO's DATASET_CONFIGS applies DEFAULT_PROMPT_TEMPLATE. To match it as
    closely as possible without importing src.data.configs, we apply the template
    unless the prompt is already formatted.
    """
    if ex.get("raw_prompt") is not None:
        raw = str(ex["raw_prompt"])
    elif ex.get("prompt") is not None:
        raw = str(ex["prompt"])
    elif ex.get("instruction") is not None:
        raw = str(ex["instruction"])
    else:
        raise KeyError(f"No prompt/raw_prompt/instruction in keys={list(ex.keys())}")

    if "BEGINNING OF CONVERSATION:" in raw and "ASSISTANT:" in raw:
        return raw, raw
    return prompt_template.format(raw_prompt=raw), raw


def pick_pair(ex: dict, objective_kind: Optional[str]) -> Optional[Tuple[str, str]]:
    """Return chosen/rejected strings for processed or PKU-like pair format."""
    if ex.get("chosen") is not None and ex.get("rejected") is not None:
        return str(ex["chosen"]), str(ex["rejected"])

    r0 = ex.get("response_0", ex.get("answer_0", ex.get("output_0", None)))
    r1 = ex.get("response_1", ex.get("answer_1", ex.get("output_1", None)))
    if r0 is None or r1 is None:
        return None

    if objective_kind == "better":
        id_keys = ["better_response_id", "chosen_response_id", "preference"]
        bool0 = ["is_response_0_better", "response_0_better", "is_response_0_preferred"]
        bool1 = ["is_response_1_better", "response_1_better", "is_response_1_preferred"]
    elif objective_kind == "safer":
        id_keys = ["safer_response_id", "chosen_response_id", "preference"]
        bool0 = ["is_response_0_safe", "response_0_safe", "is_response_0_safer"]
        bool1 = ["is_response_1_safe", "response_1_safe", "is_response_1_safer"]
    else:
        id_keys = ["chosen_response_id", "better_response_id", "safer_response_id", "preference"]
        bool0 = ["is_response_0_better", "is_response_0_safe", "response_0_better", "response_0_safe"]
        bool1 = ["is_response_1_better", "is_response_1_safe", "response_1_better", "response_1_safe"]

    for k in id_keys:
        if k in ex and ex[k] is not None:
            try:
                idx = int(ex[k])
                if idx == 0:
                    return str(r0), str(r1)
                if idx == 1:
                    return str(r1), str(r0)
            except Exception:
                pass

    k0 = next((k for k in bool0 if k in ex), None)
    k1 = next((k for k in bool1 if k in ex), None)
    if k0 is not None and k1 is not None:
        b0, b1 = bool(ex[k0]), bool(ex[k1])
        if b0 and not b1:
            return str(r0), str(r1)
        if b1 and not b0:
            return str(r1), str(r0)
        return None

    return None


def load_pairs(
    dataset_name: str,
    split: str,
    prompt_template: str,
    sanity_check: bool,
    max_examples: Optional[int],
) -> List[dict]:
    _, objective_kind = strip_objective_suffix(dataset_name)
    ds, loaded_name = load_any_split(dataset_name, split)

    if sanity_check:
        ds = ds.select(range(min(len(ds), 128)))

    pairs = []
    skipped = 0
    for ex in ds:
        picked = pick_pair(ex, objective_kind)
        if picked is None:
            skipped += 1
            continue
        try:
            prompt, raw_prompt = format_prompt_and_raw(ex, prompt_template)
        except Exception:
            skipped += 1
            continue
        chosen, rejected = picked
        pairs.append(
            {
                "prompt": prompt,
                "raw_prompt": raw_prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        if max_examples is not None and len(pairs) >= max_examples:
            break

    print(
        f"[load_pairs] requested={dataset_name} loaded={loaded_name} split={split} "
        f"pairs={len(pairs)} skipped={skipped} objective_kind={objective_kind}",
        flush=True,
    )
    if len(pairs) == 0:
        raise RuntimeError(f"No valid preference pairs found for {dataset_name}")
    return pairs


def tokenize_pair(tokenizer, ex: dict, max_length: int) -> dict:
    prompt_toks = tokenizer(ex["prompt"], add_special_tokens=False)

    def tok_side(response: str):
        full_toks = tokenizer(ex["prompt"] + response, add_special_tokens=False)
        input_ids = full_toks["input_ids"] + [tokenizer.eos_token_id]
        attention_mask = full_toks["attention_mask"] + [1]
        prompt_len = common_prefix_length(prompt_toks["input_ids"], input_ids)
        labels = list(input_ids)
        labels[:prompt_len] = [-100] * prompt_len
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        return input_ids, attention_mask, labels

    ci, ca, cl = tok_side(ex["chosen"])
    ri, ra, rl = tok_side(ex["rejected"])
    return {
        "chosen_input_ids": ci,
        "chosen_attention_mask": ca,
        "chosen_labels": cl,
        "rejected_input_ids": ri,
        "rejected_attention_mask": ra,
        "rejected_labels": rl,
        "raw_prompt": ex["raw_prompt"],
        "prompt": ex["prompt"],
        "chosen": ex["chosen"],
        "rejected": ex["rejected"],
    }


def load_cpc_log_precisions(path: str) -> List[float]:
    p = Path(path)
    if p.is_dir():
        p = p / "cpc_calibrator.json"
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    vals = payload.get("objective_log_precisions", None)
    if vals is None:
        raise KeyError(f"objective_log_precisions not found in {p}")
    return [float(x) for x in vals]


def build_parser():
    p = argparse.ArgumentParser()

    # MODPO-matched core args.
    p.add_argument("--sft_model_name", required=True)
    p.add_argument("--margin_reward_model_name", required=True, help="comma-separated non-anchor objective adapters")
    p.add_argument("--cpc_calibrator_path", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    p.add_argument("--dataset_caching", type=str2bool, default=False)
    p.add_argument("--train_split", default="train")
    p.add_argument("--eval_split", default="validation")
    p.add_argument("--sanity_check", type=str2bool, default=False)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--w", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--anchor_objective_idx", type=int, default=0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--num_proc", type=int, default=4)  # kept for CLI parity
    p.add_argument("--generate_during_eval", type=str2bool, default=True)
    p.add_argument("--use_flash_attention_2", type=str2bool, default=False)
    p.add_argument("--torch_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    # TrainingArguments equivalents.
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", default="cpcmodpo")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--warmup_steps", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=float, default=0.25)
    p.add_argument("--eval_steps", type=float, default=0.25)
    p.add_argument("--eval_delay", type=float, default=0.25)
    p.add_argument("--evaluation_strategy", default="steps")
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--load_best_model_at_end", type=str2bool, default=True)
    p.add_argument("--report_to", default="wandb")
    p.add_argument("--fp16", type=str2bool, default=True)
    p.add_argument("--bf16", type=str2bool, default=False)
    p.add_argument("--remove_unused_columns", type=str2bool, default=False)

    # LoRA, matched to MODPO run.sh through CLI.
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--policy_adapter_name", default="default")
    p.add_argument("--margin_adapter_prefix", default="margin")

    # CPC-specific diagnostics/generation.
    p.add_argument("--coefficient_floor", type=float, default=1e-6)
    p.add_argument("--debug_print_every", type=int, default=10)
    p.add_argument("--wandb_eval_generation_n", type=int, default=5)
    p.add_argument("--wandb_eval_generation_max_new_tokens", type=int, default=512)
    p.add_argument("--wandb_eval_generation_do_sample", type=str2bool, default=False)
    p.add_argument("--wandb_eval_generation_temperature", type=float, default=0.7)
    p.add_argument("--wandb_eval_generation_top_p", type=float, default=0.9)

    return p


def main():
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.dataset_caching:
        disable_caching()

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.torch_dtype]
    base = AutoModelForCausalLM.from_pretrained(
        args.sft_model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_flash_attention_2=args.use_flash_attention_2,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    base.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=split_csv(args.lora_target_modules),
    )

    try:
        model = get_peft_model(base, lora_config, adapter_name=args.policy_adapter_name)
    except TypeError:
        model = get_peft_model(base, lora_config)
        args.policy_adapter_name = "default"

    margin_paths = split_csv(args.margin_reward_model_name)
    margin_adapter_names = []
    for j, path in enumerate(margin_paths):
        name = f"{args.margin_adapter_prefix}{j}"
        model.load_adapter(path, adapter_name=name, is_trainable=False)
        margin_adapter_names.append(name)

    model.set_adapter(args.policy_adapter_name)
    model.print_trainable_parameters()

    train_pairs = load_pairs(
        args.dataset_name,
        args.train_split,
        args.prompt_template,
        args.sanity_check,
        args.max_examples,
    )
    eval_pairs = load_pairs(
        args.dataset_name,
        args.eval_split,
        args.prompt_template,
        args.sanity_check,
        args.max_examples,
    )

    train_data = [tokenize_pair(tokenizer, ex, args.max_length) for ex in train_pairs]
    eval_data = [tokenize_pair(tokenizer, ex, args.max_length) for ex in eval_pairs]
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    cpc_log_precisions = load_cpc_log_precisions(args.cpc_calibrator_path)
    num_obj = len(cpc_log_precisions)
    if num_obj != 2:
        raise ValueError("This CLI currently assumes two objectives. Extend base_w parsing for >2 objectives.")
    base_w = [args.w, 1.0 - args.w]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_delay=args.eval_delay,
        evaluation_strategy=args.evaluation_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        remove_unused_columns=args.remove_unused_columns,
        report_to=args.report_to,
        run_name=args.run_name,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    trainer = CPCMODPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=CPCPreferenceDataCollator(tokenizer),
        beta=args.beta,
        cpc_log_precisions=cpc_log_precisions,
        base_w=base_w,
        gamma=args.gamma,
        anchor_objective_idx=args.anchor_objective_idx,
        margin_adapter_names=margin_adapter_names,
        policy_adapter_name=args.policy_adapter_name,
        coefficient_floor=args.coefficient_floor,
        debug_print_every=args.debug_print_every,
        prompt_template=args.prompt_template,
        generate_during_eval=args.generate_during_eval,
        wandb_eval_generation_n=args.wandb_eval_generation_n,
        wandb_eval_generation_max_new_tokens=args.wandb_eval_generation_max_new_tokens,
        wandb_eval_generation_do_sample=args.wandb_eval_generation_do_sample,
        wandb_eval_generation_temperature=args.wandb_eval_generation_temperature,
        wandb_eval_generation_top_p=args.wandb_eval_generation_top_p,
    )

    print("[CPC] log_precisions:", cpc_log_precisions, flush=True)
    print("[CPC] base_w:", base_w, "gamma:", args.gamma, flush=True)
    print("[CPC] coefficients:", trainer.cpc_coefficients.tolist(), flush=True)
    print("[fairness] dataset:", args.dataset_name, "train_split:", args.train_split, "eval_split:", args.eval_split, flush=True)
    print("[fairness] lr:", args.learning_rate, "batch:", args.per_device_train_batch_size,
          "grad_accum:", args.gradient_accumulation_steps, "epochs:", args.num_train_epochs, flush=True)

    trainer.train()

    save_name = "best_checkpoint" if args.load_best_model_at_end else "final_checkpoint"
    save_dir = Path(args.output_dir) / save_name
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print("saved checkpoint to", save_dir, flush=True)


if __name__ == "__main__":
    main()
