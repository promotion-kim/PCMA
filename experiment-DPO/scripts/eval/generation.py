#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

#from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from datasets import load_dataset

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
    sanity_check: bool
    deduplicate_prompts: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser()

    p.add_argument("--model_type", choices=["modpo", "pcmodpo"], required=True)
    p.add_argument("--sft_model_name", default="PKU-Alignment/alpaca-7b-reproduced")
    p.add_argument("--adapter_path", required=True)

    p.add_argument("--dataset_name", default="PKU-Alignment/PKU-SafeRLHF-10K-better")
    p.add_argument("--split", default="test")
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
    p.add_argument("--sanity_check", type=str2bool, default=False)
    p.add_argument("--deduplicate_prompts", type=str2bool, default=True)

    return Args(**vars(p.parse_args()))


def resolve_adapter_path(path: str) -> str:
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
            if not name.startswith("checkpoint-"):
                continue
            if os.path.isfile(os.path.join(full, "adapter_config.json")):
                m = re.match(r"checkpoint-(\d+)", name)
                step = int(m.group(1)) if m else -1
                candidates.append((step, full))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]

    raise FileNotFoundError(
        f"Cannot find adapter_config.json under {path}. "
        f"Pass checkpoint dir, best_checkpoint dir, or training output_dir."
    )


def resolve_hf_dataset_name(dataset_name: str) -> str:
    """
    Convert project-specific dataset aliases to actual Hugging Face dataset names.

    Examples:
      PKU-Alignment/PKU-SafeRLHF-10K-better -> PKU-Alignment/PKU-SafeRLHF-10K
      PKU-Alignment/PKU-SafeRLHF-10K-safer  -> PKU-Alignment/PKU-SafeRLHF-10K
      PKU-Alignment/PKU-SafeRLHF-better     -> PKU-Alignment/PKU-SafeRLHF
      PKU-Alignment/PKU-SafeRLHF-safer      -> PKU-Alignment/PKU-SafeRLHF
    """
    for suffix in ["-better", "-safer"]:
        if dataset_name.endswith(suffix):
            return dataset_name[: -len(suffix)]
    return dataset_name


def is_pku_saferlhf_10k(dataset_name: str) -> bool:
    base = resolve_hf_dataset_name(dataset_name)
    return base == "PKU-Alignment/PKU-SafeRLHF-10K"


def is_pku_saferlhf_full(dataset_name: str) -> bool:
    base = resolve_hf_dataset_name(dataset_name)
    return base == "PKU-Alignment/PKU-SafeRLHF"


def load_pku_split_like_training_code(dataset_name: str, split: str):
    """
    Reproduce the project wrapper split logic without importing src.data.configs.

    Original logic:

    PKUSafeRlhfRDP:
      train      = load_dataset(path, split="train").train_test_split(test_size=0.1, seed=0)["train"]
      validation = load_dataset(path, split="train").train_test_split(test_size=0.1, seed=0)["test"]
      test       = load_dataset(path, split="test")

    PKUSafeRlhf10KRDP:
      train      = load_dataset(path, split="train").train_test_split(test_size=0.1, seed=0)["train"]
      validation = load_dataset(path, split="train").train_test_split(test_size=0.1, seed=0)["test"]
      test       = not available
    """
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
                "Use --split validation to reproduce the project's held-out 10% validation split, "
                "or use --dataset_name PKU-Alignment/PKU-SafeRLHF --split test for the full dataset test split."
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

    # Generic fallback for other datasets.
    print(f"[generation] generic load_dataset path={hf_dataset_name}, split={split}")
    return load_dataset(hf_dataset_name, split=split)


def extract_prompt_from_example(example: dict) -> str | None:
    """
    Try common prompt column names.
    PKU-SafeRLHF uses `prompt`.
    """
    for key in ["raw_prompt", "prompt", "input", "question", "instruction"]:
        if key in example and example[key] is not None:
            return str(example[key])
    return None


def format_model_input(raw_prompt: str, prompt_template: str) -> str:
    """
    Avoid double-formatting if the dataset already contains the full conversation template.
    """
    if "BEGINNING OF CONVERSATION:" in raw_prompt and "ASSISTANT:" in raw_prompt:
        return raw_prompt

    if prompt_template is not None and "{raw_prompt}" in prompt_template:
        return prompt_template.format(raw_prompt=raw_prompt)

    return raw_prompt


def load_eval_prompts(args) -> List[Dict[str, Any]]:
    """
    Load prompts for generation while reproducing the project's PKU split behavior.

    Important:
      - PKU-SafeRLHF-10K has no official test split.
      - For 10K, use --split validation to get the same held-out 10% split
        used by the training/eval wrapper.
      - `-better` / `-safer` suffixes are project aliases, not HF dataset names.
    """
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

        if args.deduplicate_prompts:
            if raw_prompt in seen:
                continue
            seen.add(raw_prompt)

        rows.append(
            {
                "eval_index": int(idx),
                "raw_prompt": raw_prompt,
                "model_input": model_input,
            }
        )

        if args.limit is not None and args.limit > 0 and len(rows) >= args.limit:
            break

    if len(rows) == 0:
        raise RuntimeError(
            f"No prompts were loaded from dataset={args.dataset_name}, split={args.split}. "
            "Check dataset columns with:\n"
            f"python - <<'PY'\n"
            f"from datasets import load_dataset\n"
            f"ds = load_dataset('{hf_dataset_name}', split='train')\n"
            f"print(ds[0].keys())\n"
            f"print(ds[0])\n"
            f"PY"
        )

    print(f"[generation] loaded prompts: {len(rows)}")
    return rows


def main() -> None:
    args = parse_args()
    adapter_path = resolve_adapter_path(args.adapter_path)

    if not args.do_sample:
        temperature = 0.0
        top_p = 1.0
    else:
        temperature = args.temperature
        top_p = args.top_p

    print("=" * 100)
    print("[generation_vllm] config")
    for k, v in asdict(args).items():
        print(f"{k}: {v}")
    print(f"resolved_adapter_path: {adapter_path}")
    print("=" * 100)

    rows = load_eval_prompts(args)
    prompts = [r["model_input"] for r in rows]
    print(f"[generation_vllm] loaded prompts: {len(prompts)}")

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

    # vLLM internally batches, but chunking avoids very large Python-side request lists.
    for start in tqdm(range(0, len(prompts), args.batch_size), desc="vLLM generating", dynamic_ncols=True):
        batch_rows = rows[start : start + args.batch_size]
        batch_prompts = prompts[start : start + args.batch_size]

        outputs = llm.generate(
            batch_prompts,
            sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
        )

        # vLLM returns outputs in input order.
        for row, out in zip(batch_rows, outputs):
            text = out.outputs[0].text if out.outputs else ""
            item = dict(row)
            item["generation"] = text.strip()
            outputs_all.append(item)

    result = {
        "metadata": {
            **asdict(args),
            "resolved_adapter_path": adapter_path,
            "num_generations": len(outputs_all),
            "backend": "vllm",
        },
        "data": outputs_all,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[generation_vllm] saved to: {args.output_path}")


if __name__ == "__main__":
    main()