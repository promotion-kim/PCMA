#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""vLLM generation for MORLHF and PC-MORLHF LoRA checkpoints.

This script is specialized for PPO/MORLHF-style checkpoints saved as LoRA
adapters, for example:
  /ext_hdd/sjkim/mod/morlhf_saferlhf/.../batch_697
  /ext_hdd/sjkim/mod/pcmorlhf_saferlhf/.../batch_697

It follows the same PKU-SafeRLHF-10K split convention as the project wrapper:
train      = 90% of HF train split, seed=0
validation = held-out 10% of HF train split, seed=0
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

DEFAULT_PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


@dataclass
class Args:
    model_type: str
    sft_model_name: str
    adapter_path: str
    dataset_name: str
    split: str
    prompt_template: str
    output_path: str
    limit: int
    batch_size: int
    max_input_length: int
    max_new_tokens: int
    max_model_len: int
    do_sample: bool
    temperature: float
    top_p: float
    seed: int
    dtype: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_lora_rank: int
    trust_remote_code: bool
    deduplicate_prompts: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["morlhf", "pcmorlhf"], required=True)
    p.add_argument("--sft_model_name", default="PKU-Alignment/alpaca-7b-reproduced")
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--dataset_name", default="PKU-Alignment/PKU-SafeRLHF-10K-better")
    p.add_argument("--split", default="validation")
    p.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    p.add_argument("--output_path", required=True)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_model_len", type=int, default=1024)
    p.add_argument("--do_sample", type=str2bool, default=False)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_lora_rank", type=int, default=64)
    p.add_argument("--trust_remote_code", type=str2bool, default=True)
    p.add_argument("--deduplicate_prompts", type=str2bool, default=True)
    return Args(**vars(p.parse_args()))


def resolve_adapter_path(path: str) -> str:
    """Resolve direct adapter, best_checkpoint, latest batch_*, or latest checkpoint-*."""
    path = os.path.abspath(path)
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path

    best = os.path.join(path, "best_checkpoint")
    if os.path.isfile(os.path.join(best, "adapter_config.json")):
        return best

    if os.path.isdir(path):
        candidates = []
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if not os.path.isdir(full):
                continue
            if not (name.startswith("batch_") or name.startswith("checkpoint-")):
                continue
            if not os.path.isfile(os.path.join(full, "adapter_config.json")):
                continue
            m = re.match(r"(?:batch_|checkpoint-)(\d+)$", name)
            step = int(m.group(1)) if m else -1
            candidates.append((step, full))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]

    raise FileNotFoundError(
        f"Cannot find adapter_config.json under {path}. "
        "Pass an adapter dir, best_checkpoint dir, batch_XXX dir, or parent output dir."
    )


def resolve_hf_dataset_name(dataset_name: str) -> str:
    for suffix in ["-better", "-safer"]:
        if dataset_name.endswith(suffix):
            return dataset_name[: -len(suffix)]
    return dataset_name


def is_pku_saferlhf_10k(dataset_name: str) -> bool:
    return resolve_hf_dataset_name(dataset_name) == "PKU-Alignment/PKU-SafeRLHF-10K"


def is_pku_saferlhf_full(dataset_name: str) -> bool:
    return resolve_hf_dataset_name(dataset_name) == "PKU-Alignment/PKU-SafeRLHF"


def load_pku_split_like_training_code(dataset_name: str, split: str):
    hf_dataset_name = resolve_hf_dataset_name(dataset_name)
    split = split.lower()

    if is_pku_saferlhf_10k(dataset_name):
        raw_train = load_dataset(hf_dataset_name, split="train")
        split_ds = raw_train.train_test_split(test_size=0.1, seed=0)
        if split == "train":
            print("[generation] using PKU-SafeRLHF-10K train = 90% of HF train, seed=0")
            return split_ds["train"]
        if split in {"validation", "valid", "val", "eval", "dev"}:
            print("[generation] using PKU-SafeRLHF-10K validation = held-out 10% of HF train, seed=0")
            return split_ds["test"]
        if split == "test":
            raise NotImplementedError(
                "PKU-Alignment/PKU-SafeRLHF-10K has no official test split. "
                "Use --split validation for the held-out 10% validation split."
            )
        raise NotImplementedError(f"Unsupported split for PKU-SafeRLHF-10K: {split}")

    if is_pku_saferlhf_full(dataset_name):
        if split == "train":
            raw_train = load_dataset(hf_dataset_name, split="train")
            print("[generation] using PKU-SafeRLHF train = 90% of HF train, seed=0")
            return raw_train.train_test_split(test_size=0.1, seed=0)["train"]
        if split in {"validation", "valid", "val", "eval", "dev"}:
            raw_train = load_dataset(hf_dataset_name, split="train")
            print("[generation] using PKU-SafeRLHF validation = held-out 10% of HF train, seed=0")
            return raw_train.train_test_split(test_size=0.1, seed=0)["test"]
        if split == "test":
            print("[generation] using PKU-SafeRLHF official HF test split")
            return load_dataset(hf_dataset_name, split="test")
        raise NotImplementedError(f"Unsupported split for PKU-SafeRLHF: {split}")

    print(f"[generation] generic load_dataset path={hf_dataset_name}, split={split}")
    return load_dataset(hf_dataset_name, split=split)


def extract_prompt_from_example(example: dict) -> str | None:
    for key in ["raw_prompt", "prompt", "input", "query", "question", "instruction"]:
        if key in example and example[key] is not None:
            return str(example[key])
    return None


def format_model_input(raw_prompt: str, prompt_template: str) -> str:
    if "BEGINNING OF CONVERSATION:" in raw_prompt and "ASSISTANT:" in raw_prompt:
        return raw_prompt
    if prompt_template is not None and "{raw_prompt}" in prompt_template:
        return prompt_template.format(raw_prompt=raw_prompt)
    return raw_prompt


def maybe_truncate_prompt(model_input: str, tokenizer, max_input_length: int) -> str:
    """Left-truncate only if needed, keeping the end of the prompt near ASSISTANT."""
    if max_input_length is None or max_input_length <= 0:
        return model_input
    encoded = tokenizer(model_input, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if len(input_ids) <= max_input_length:
        return model_input
    input_ids = input_ids[-max_input_length:]
    return tokenizer.decode(input_ids, skip_special_tokens=True)


def load_eval_prompts(args: Args, tokenizer) -> List[Dict[str, Any]]:
    hf_dataset_name = resolve_hf_dataset_name(args.dataset_name)
    print(
        f"[generation] requested dataset={args.dataset_name}, "
        f"resolved_hf_dataset={hf_dataset_name}, split={args.split}"
    )

    ds = load_pku_split_like_training_code(args.dataset_name, args.split)
    rows: List[Dict[str, Any]] = []
    seen = set()

    for idx, ex in enumerate(ds):
        raw_prompt = extract_prompt_from_example(ex)
        if raw_prompt is None:
            if idx == 0:
                print("[generation] available keys:", list(ex.keys()))
            continue

        model_input = format_model_input(raw_prompt, args.prompt_template)
        model_input = maybe_truncate_prompt(model_input, tokenizer, args.max_input_length)

        if args.deduplicate_prompts:
            if raw_prompt in seen:
                continue
            seen.add(raw_prompt)

        rows.append({"eval_index": int(idx), "raw_prompt": raw_prompt, "model_input": model_input})
        if args.limit is not None and args.limit > 0 and len(rows) >= args.limit:
            break

    if not rows:
        raise RuntimeError(f"No prompts were loaded from dataset={args.dataset_name}, split={args.split}")

    print(f"[generation] loaded prompts: {len(rows)}")
    return rows


def main() -> None:
    args = parse_args()
    adapter_path = resolve_adapter_path(args.adapter_path)
    temperature = args.temperature if args.do_sample else 0.0
    top_p = args.top_p if args.do_sample else 1.0

    print("=" * 100)
    print("[generation_morlhf_vllm] config")
    for k, v in asdict(args).items():
        print(f"{k}: {v}")
    print(f"resolved_adapter_path: {adapter_path}")
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_eval_prompts(args, tokenizer)
    prompts = [r["model_input"] for r in rows]

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    llm = LLM(
        model=args.sft_model_name,
        tokenizer=args.sft_model_name,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        trust_remote_code=args.trust_remote_code,
    )

    lora_request = LoRARequest(
        lora_name=args.model_type,
        lora_int_id=1,
        lora_path=adapter_path,
    )

    outputs_all = []
    for start in tqdm(range(0, len(prompts), args.batch_size), desc="vLLM generating", dynamic_ncols=True):
        batch_rows = rows[start : start + args.batch_size]
        batch_prompts = prompts[start : start + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_request, use_tqdm=False)

        for row, out in zip(batch_rows, outputs):
            text = out.outputs[0].text if out.outputs else ""
            item = dict(row)
            item["generation"] = text.strip()
            outputs_all.append(item)

    result = {
        "metadata": {**asdict(args), "resolved_adapter_path": adapter_path, "num_generations": len(outputs_all), "backend": "vllm"},
        "data": outputs_all,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[generation_morlhf_vllm] saved to: {args.output_path}")


if __name__ == "__main__":
    main()
