#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Reward Soup generation for SafeRLHF/MOD-style DPO adapters.

This script avoids importing DPOTrainer_Light / TRL / DeepSpeed.
It only:
  1) loads the SFT base model,
  2) loads DPO LoRA adapters,
  3) linearly interpolates adapter weights,
  4) generates responses for evaluation prompts,
  5) saves txt/jsonl outputs.

Assumption:
  dpo_model_1 = better/helpful adapter
  dpo_model_2 = safer/harmless adapter
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import tyro
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
except Exception:
    DATASET_CONFIGS = {}
    DEFAULT_PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"


@dataclass
class ScriptArguments:
    sft_model_name: str = field(default="PKU-Alignment/alpaca-7b-reproduced")
    dpo_model_1_name: str = field(default="/ext_hdd/sjkim/mod/dpo/dpo-better/best_checkpoint")
    dpo_model_2_name: str = field(default="/ext_hdd/sjkim/mod/dpo/dpo-safer/best_checkpoint")
    dpo_model_3_name: Optional[str] = field(default=None)

    weight_1: float = field(default=0.5)
    weight_2: float = field(default=0.5)
    weight_3: float = field(default=0.0)

    dataset_name: str = field(default="PKU-Alignment/PKU-SafeRLHF-10K-better")
    split: str = field(default="validation")
    prompt_template: str = field(default=DEFAULT_PROMPT_TEMPLATE)

    output_dir: str = field(default="outputs/generation/rs")
    output_name: Optional[str] = field(default=None)

    max_eval_samples: Optional[int] = field(default=None)
    batch_size: int = field(default=4)
    max_prompt_length: int = field(default=512)
    max_new_tokens: int = field(default=512)
    num_beams: int = field(default=1)
    do_sample: bool = field(default=False)
    temperature: float = field(default=0.7)
    top_p: float = field(default=1.0)

    torch_dtype: str = field(default="bf16", metadata={"help": "bf16, fp16, fp32"})
    seed: int = field(default=42)
    trust_remote_code: bool = field(default=True)


def _dtype(name: str):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32



def resolve_hf_dataset_name(dataset_name: str) -> str:
    """
    Convert project-specific dataset aliases to actual Hugging Face names.
    Example:
      PKU-Alignment/PKU-SafeRLHF-10K-better -> PKU-Alignment/PKU-SafeRLHF-10K
    """
    for suffix in ["-better", "-safer"]:
        if dataset_name.endswith(suffix):
            return dataset_name[: -len(suffix)]
    return dataset_name


def is_pku_saferlhf_10k(dataset_name: str) -> bool:
    return resolve_hf_dataset_name(dataset_name) == "PKU-Alignment/PKU-SafeRLHF-10K"


def is_pku_saferlhf_full(dataset_name: str) -> bool:
    return resolve_hf_dataset_name(dataset_name) == "PKU-Alignment/PKU-SafeRLHF"


def load_pku_split_like_training_code(dataset_name: str, split: str):
    """
    Reproduce the project wrapper split logic.

    For PKU-SafeRLHF-10K:
      train      = 90% of HF train, seed=0
      validation = held-out 10% of HF train, seed=0
      test       = not available
    """
    hf_dataset_name = resolve_hf_dataset_name(dataset_name)
    split = split.lower()

    if is_pku_saferlhf_10k(dataset_name):
        raw_train = load_dataset(hf_dataset_name, split="train")
        split_ds = raw_train.train_test_split(test_size=0.1, seed=0)

        if split == "train":
            print("[rs] using PKU-SafeRLHF-10K train = 90% of HF train, seed=0")
            return split_ds["train"]

        if split in {"validation", "valid", "val", "eval", "dev"}:
            print("[rs] using PKU-SafeRLHF-10K validation = held-out 10% of HF train, seed=0")
            return split_ds["test"]

        if split == "test":
            raise NotImplementedError(
                "PKU-SafeRLHF-10K has no official test split. "
                "Use --split validation."
            )

        raise NotImplementedError(f"Unsupported split for PKU-SafeRLHF-10K: {split}")

    if is_pku_saferlhf_full(dataset_name):
        if split == "train":
            raw_train = load_dataset(hf_dataset_name, split="train")
            print("[rs] using PKU-SafeRLHF train = 90% of HF train, seed=0")
            return raw_train.train_test_split(test_size=0.1, seed=0)["train"]

        if split in {"validation", "valid", "val", "eval", "dev"}:
            raw_train = load_dataset(hf_dataset_name, split="train")
            print("[rs] using PKU-SafeRLHF validation = held-out 10% of HF train, seed=0")
            return raw_train.train_test_split(test_size=0.1, seed=0)["test"]

        if split == "test":
            print("[rs] using PKU-SafeRLHF official HF test split")
            return load_dataset(hf_dataset_name, split="test")

        raise NotImplementedError(f"Unsupported split for PKU-SafeRLHF: {split}")

    print(f"[rs] generic load_dataset path={hf_dataset_name}, split={split}")
    return load_dataset(hf_dataset_name, split=split)


def extract_prompt_from_example(example: dict) -> str | None:
    for key in ["raw_prompt", "prompt", "input", "question", "instruction"]:
        if key in example and example[key] is not None:
            return str(example[key])
    return None


def format_model_input(raw_prompt: str, prompt_template: str) -> str:
    if "BEGINNING OF CONVERSATION:" in raw_prompt and "ASSISTANT:" in raw_prompt:
        return raw_prompt

    if prompt_template is not None and "{raw_prompt}" in prompt_template:
        return prompt_template.format(raw_prompt=raw_prompt)

    return raw_prompt


def _format_prompt(example: Dict[str, Any], prompt_template: str) -> tuple[str, str]:
    """Return (raw_prompt, formatted_prompt)."""
    if example.get("raw_prompt") is not None:
        raw = str(example["raw_prompt"])
        return raw, prompt_template.format(raw_prompt=raw)

    if example.get("prompt") is not None:
        p = str(example["prompt"])
        if "{raw_prompt}" in prompt_template and "BEGINNING OF CONVERSATION" not in p:
            return p, prompt_template.format(raw_prompt=p)
        return p, p

    if example.get("query") is not None:
        q = str(example["query"])
        return q, q

    if example.get("input") is not None:
        raw = str(example["input"])
        return raw, prompt_template.format(raw_prompt=raw)

    # Raw PKU-SafeRLHF sometimes uses only `prompt`.
    raise KeyError(f"Cannot find prompt field. Available keys={list(example.keys())}")


def load_eval_prompts(dataset_name: str, split: str, prompt_template: str, max_eval_samples: Optional[int]):
    rows = []

    print(
        f"[rs] requested dataset={dataset_name}, "
        f"resolved_hf_dataset={resolve_hf_dataset_name(dataset_name)}, split={split}"
    )

    # Use the same PKU split behavior as MODPO/PCMODPO generation.
    dataset = load_pku_split_like_training_code(dataset_name, split)

    seen = set()
    for idx, ex in enumerate(dataset):
        raw_prompt = extract_prompt_from_example(ex)
        if raw_prompt is None:
            if idx == 0:
                print("[rs] available keys:", list(ex.keys()))
            continue

        prompt = format_model_input(raw_prompt, prompt_template)

        if raw_prompt in seen:
            continue
        seen.add(raw_prompt)

        rows.append({"raw_prompt": raw_prompt, "prompt": prompt})

        if max_eval_samples is not None and len(rows) >= max_eval_samples:
            break

    if len(rows) == 0:
        raise RuntimeError(f"No prompts loaded from dataset={dataset_name}, split={split}")

    print(f"[rs] loaded prompts: {len(rows)}")
    return rows


def weighted_soup_adapters(model, args: ScriptArguments):
    """Load LoRA adapters and overwrite adapter_rs with weighted average."""
    model = PeftModel.from_pretrained(
        model,
        args.dpo_model_1_name,
        adapter_name="adapter_rs",
        is_trainable=False,
    )
    model.load_adapter(args.dpo_model_2_name, adapter_name="adapter_aux_1", is_trainable=False)

    use_third = args.dpo_model_3_name is not None and args.weight_3 > 0
    if use_third:
        model.load_adapter(args.dpo_model_3_name, adapter_name="adapter_aux_2", is_trainable=False)

    sd = model.state_dict()
    new_state = {}

    missing = 0
    for key, value in sd.items():
        if "adapter_rs" not in key:
            continue

        key2 = key.replace("adapter_rs", "adapter_aux_1")
        if key2 not in sd:
            missing += 1
            continue

        mixed = args.weight_1 * value + args.weight_2 * sd[key2]

        if use_third:
            key3 = key.replace("adapter_rs", "adapter_aux_2")
            if key3 not in sd:
                missing += 1
                continue
            mixed = mixed + args.weight_3 * sd[key3]

        new_state[key] = mixed

    if not new_state:
        raise RuntimeError(
            "No adapter_rs parameters were found for interpolation. "
            "Check that the checkpoint paths are PEFT LoRA adapter directories."
        )

    if missing:
        print(f"[warning] missing matching adapter keys: {missing}")

    model.load_state_dict(new_state, strict=False)
    model.set_adapter("adapter_rs")
    return model


def clean_response(prompt: str, full_text: str) -> str:
    text = full_text
    if text.startswith(prompt):
        text = text[len(prompt):]
    text = text.strip()
    for sep in ["\n\nHuman:", "\nHuman:", "\n\nUSER:", "\nUSER:", "BEGINNING OF CONVERSATION: USER:"]:
        if sep in text:
            text = text.split(sep)[0].strip()
    if "</s>" in text:
        text = text.split("</s>")[0].strip()
    return text


def main():
    args = tyro.cli(ScriptArguments)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.output_name is None:
        args.output_name = f"rs_output_h{args.weight_1:.1f}_s{args.weight_2:.1f}.jsonl"

    output_jsonl = os.path.join(args.output_dir, args.output_name)
    output_txt = output_jsonl.rsplit(".", 1)[0] + ".txt"

    print("=" * 100)
    print("[Reward Soup Standalone]")
    print(f"sft_model_name={args.sft_model_name}")
    print(f"dpo_model_1_name={args.dpo_model_1_name}")
    print(f"dpo_model_2_name={args.dpo_model_2_name}")
    print(f"weights=({args.weight_1}, {args.weight_2}, {args.weight_3})")
    print(f"dataset_name={args.dataset_name}, split={args.split}")
    print(f"output_jsonl={output_jsonl}")
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(
        args.sft_model_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_name,
        torch_dtype=_dtype(args.torch_dtype),
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id

    model = weighted_soup_adapters(model, args)
    model.eval()

    prompts = load_eval_prompts(args.dataset_name, args.split, args.prompt_template, args.max_eval_samples)
    print(f"[data] num prompts={len(prompts)}")

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})

    records = []
    with torch.no_grad():
        for start in tqdm(range(0, len(prompts), args.batch_size), dynamic_ncols=True):
            batch = prompts[start:start + args.batch_size]
            prompt_texts = [r["prompt"] for r in batch]

            toks = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
            )
            toks = {k: v.to(model.device) for k, v in toks.items()}

            out = model.generate(**toks, **generation_kwargs)
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

            for row, full in zip(batch, decoded):
                response = clean_response(row["prompt"], full)
                records.append({
                    "raw_prompt": row["raw_prompt"],
                    "prompt": row["prompt"],
                    "response": response,
                    "full_text": full,
                    "weight_helpful": args.weight_1,
                    "weight_harmless": args.weight_2,
                })

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(output_txt, "w", encoding="utf-8") as f:
        for r in records:
            f.write("\nPrompt and response\n")
            f.write(r["prompt"] + " " + r["response"])
            f.write("\n")

    print(f"[saved] {output_jsonl}")
    print(f"[saved] {output_txt}")


if __name__ == "__main__":
    main()
