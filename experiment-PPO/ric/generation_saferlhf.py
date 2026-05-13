#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""vLLM generation for SafeRLHF-10K validation prompts from a trained RiC LoRA model.

This is the vLLM counterpart of ric/generation_saferlhf.py.

Key behavior:
  1. Uses the same SafeRLHF-10K validation split convention:
       raw HF train split -> train_test_split(test_size=0.1, seed=0)["test"]
     for PKU-Alignment/PKU-SafeRLHF-10K(-better/-safer).
  2. Uses validation prompts only, not the prepared RiC disk dataset.
  3. Takes helpfulness weight w_h and inserts RiC score tokens.
  4. Uses vLLM + LoRARequest for fast decoding.
  5. Saves JSON and JSONL records compatible with the previous generation script.

Example:
  CUDA_VISIBLE_DEVICES=0 python ric/generation_saferlhf_vllm.py \
    --base_model_name PKU-Alignment/alpaca-7b-reproduced \
    --peft_name /ext_hdd/sjkim/mod/ric_saferlhf/logs/ric_saferlhf_helpful_harmless/model_iter0 \
    --preference 0.5 \
    --output_dir outputs/generation/ric_saferlhf_vllm
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils import Instructions_n, map_rewards_from_preference


DEFAULT_PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _split_csv(x: str) -> List[str]:
    return [v.strip() for v in str(x).split(",") if v.strip()]


def resolve_hf_dataset_name(dataset_name: str) -> str:
    """Convert project-specific aliases to actual HF dataset names."""
    for suffix in ["-better", "-safer"]:
        if dataset_name.endswith(suffix):
            return dataset_name[: -len(suffix)]
    return dataset_name


def is_pku_saferlhf_10k(dataset_name: str) -> bool:
    return resolve_hf_dataset_name(dataset_name) == "PKU-Alignment/PKU-SafeRLHF-10K"


def is_pku_saferlhf_full(dataset_name: str) -> bool:
    return resolve_hf_dataset_name(dataset_name) == "PKU-Alignment/PKU-SafeRLHF"


def load_pku_split_like_training_code(dataset_name: str, split: str):
    """Reproduce the MODPO/PCMODPO split convention."""
    hf_name = resolve_hf_dataset_name(dataset_name)
    split = split.lower()

    if is_pku_saferlhf_10k(dataset_name):
        raw_train = load_dataset(hf_name, split="train")
        split_ds = raw_train.train_test_split(test_size=0.1, seed=0)

        if split == "train":
            print("[data] PKU-SafeRLHF-10K train = 90% of HF train, seed=0")
            return split_ds["train"]

        if split in {"validation", "valid", "val", "eval", "dev"}:
            print("[data] PKU-SafeRLHF-10K validation = held-out 10% of HF train, seed=0")
            return split_ds["test"]

        if split == "test":
            raise NotImplementedError(
                "PKU-SafeRLHF-10K has no official test split. Use --split validation."
            )

        raise NotImplementedError(f"Unsupported split for PKU-SafeRLHF-10K: {split}")

    if is_pku_saferlhf_full(dataset_name):
        if split == "train":
            raw_train = load_dataset(hf_name, split="train")
            print("[data] PKU-SafeRLHF train = 90% of HF train, seed=0")
            return raw_train.train_test_split(test_size=0.1, seed=0)["train"]

        if split in {"validation", "valid", "val", "eval", "dev"}:
            raw_train = load_dataset(hf_name, split="train")
            print("[data] PKU-SafeRLHF validation = held-out 10% of HF train, seed=0")
            return raw_train.train_test_split(test_size=0.1, seed=0)["test"]

        if split == "test":
            print("[data] PKU-SafeRLHF official HF test split")
            return load_dataset(hf_name, split="test")

        raise NotImplementedError(f"Unsupported split for PKU-SafeRLHF: {split}")

    print(f"[data] generic load_dataset path={hf_name}, split={split}")
    return load_dataset(hf_name, split=split)


def _get_first(example: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    return None


def _format_prompt(raw_prompt: str, prompt_template: str) -> str:
    raw_prompt = str(raw_prompt)
    if "BEGINNING OF CONVERSATION:" in raw_prompt and "ASSISTANT:" in raw_prompt:
        return raw_prompt
    if "{raw_prompt}" in prompt_template:
        return prompt_template.format(raw_prompt=raw_prompt)
    return prompt_template + raw_prompt


def load_eval_rows(dataset_name: str, split: str, prompt_template: str, max_eval_samples: int) -> List[Dict[str, str]]:
    print(
        f"[data] requested dataset={dataset_name}, "
        f"resolved={resolve_hf_dataset_name(dataset_name)}, split={split}"
    )
    ds = load_pku_split_like_training_code(dataset_name, split)

    rows: List[Dict[str, str]] = []
    seen = set()

    for ex in ds:
        raw_prompt = _get_first(ex, ["raw_prompt", "prompt", "input", "query", "question", "instruction"])
        if raw_prompt is None:
            continue

        raw_prompt = str(raw_prompt)
        if raw_prompt in seen:
            continue
        seen.add(raw_prompt)

        rows.append(
            {
                "raw_prompt": raw_prompt,
                "prompt": _format_prompt(raw_prompt, prompt_template),
            }
        )

        if max_eval_samples and max_eval_samples > 0 and len(rows) >= max_eval_samples:
            break

    if not rows:
        raise RuntimeError("No validation prompts were loaded.")

    print(f"[data] loaded unique prompts: {len(rows)}")
    return rows


def _insert_score_tokens(prompt: str, scores: Sequence[float], instructions: Instructions_n) -> str:
    p = str(prompt).strip()

    if p.endswith("ASSISTANT:"):
        prefix = p[: -len("ASSISTANT:")].rstrip()
        suffix = "ASSISTANT:"
    elif p.endswith("Assistant:"):
        prefix = p[: -len("Assistant:")].rstrip()
        suffix = "Assistant:"
    else:
        prefix = p
        suffix = "ASSISTANT:"

    score_text = " ".join(
        f"{instructions.score_splits[i]} {round(float(scores[i]), 1)}"
        for i in range(len(scores))
    )
    return f"{prefix} {score_text} {suffix}"


def _clean_vllm_text(text: str) -> str:
    text = str(text).strip()
    if "</s>" in text:
        text = text.split("</s>", 1)[0].strip()
    for sep in ["\n\nHuman:", "\nHuman:", "\n\nUSER:", "\nUSER:", "\n\nAssistant:", "\nAssistant:", "###"]:
        if sep in text:
            text = text.split(sep, 1)[0].strip()
    return text


@dataclass
class Args:
    # Model / checkpoint
    base_model_name: str
    peft_name: str

    # Data
    dataset_name: str
    split: str
    prompt_template: str
    max_eval_samples: int

    # RiC preference condition
    preference: Optional[float]
    preferences: str
    target_map_method: str
    target_rewards: str

    # Output
    output_dir: str
    output_prefix: str

    # Generation hyperparameters
    max_prompt_length: int
    max_new_tokens: int
    batch_size: int
    seed: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int

    # vLLM config
    dtype: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    max_lora_rank: int
    trust_remote_code: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser()

    p.add_argument("--base_model_name", default="PKU-Alignment/alpaca-7b-reproduced")
    p.add_argument("--peft_name", default="", help="trained RiC LoRA adapter path")

    p.add_argument("--dataset_name", default="PKU-Alignment/PKU-SafeRLHF-10K-better")
    p.add_argument("--split", default="validation")
    p.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    p.add_argument("--max_eval_samples", type=int, default=-1, help="-1 means use all validation prompts")

    p.add_argument("--preference", type=float, default=None)
    p.add_argument("--preferences", default="0.0,0.3,0.5,0.7,1.0")
    p.add_argument("--target_map_method", default="l2", choices=["linf", "l2", "linear"])
    p.add_argument("--target_rewards", default="", help="optional comma-separated manual target rewards")

    p.add_argument("--output_dir", default="outputs/generation/ric_saferlhf_vllm")
    p.add_argument("--output_prefix", default="ric")

    p.add_argument("--max_prompt_length", type=int, default=384)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=8888)
    p.add_argument("--do_sample", type=str2bool, default=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)

    p.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=1024)
    p.add_argument("--max_lora_rank", type=int, default=64)
    p.add_argument("--trust_remote_code", type=str2bool, default=True)

    return Args(**vars(p.parse_args()))


def get_pref_values(args: Args) -> List[float]:
    if args.preference is not None:
        return [float(args.preference)]
    return [float(x) for x in _split_csv(args.preferences)]


def get_target(args: Args, w_helpful: float, num_rewards: int, rewards_reference_list: Sequence[np.ndarray]) -> np.ndarray:
    if args.target_rewards:
        target = np.array([float(x) for x in _split_csv(args.target_rewards)], dtype=np.float32)
        if len(target) != num_rewards:
            raise ValueError(f"--target_rewards must contain {num_rewards} values.")
        return target

    if num_rewards != 2:
        raise NotImplementedError("This generation script currently assumes two objectives: helpful, harmless.")

    preference_vec = np.array([w_helpful, 1.0 - w_helpful], dtype=np.float32)
    return map_rewards_from_preference(
        rewards_reference_list,
        preference_vec,
        method=args.target_map_method,
    ).reshape(-1)


def truncate_prompt_if_needed(prompt: str, max_prompt_length: int, tokenizer=None) -> str:
    """Optional text-level fallback.

    vLLM tokenizes internally. We avoid an extra transformers tokenizer dependency here.
    The previous HF script used tokenizer truncation. In practice, SafeRLHF prompts are
    short enough; this function keeps the interface explicit.
    """
    return prompt


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    num_rewards = 2
    instructions = Instructions_n(num_rewards)

    rows = load_eval_rows(
        dataset_name=args.dataset_name,
        split=args.split,
        prompt_template=args.prompt_template,
        max_eval_samples=args.max_eval_samples,
    )

    rng = np.random.default_rng(args.seed)
    rewards_reference_list = [rng.standard_normal(50000) for _ in range(num_rewards)]

    if not args.do_sample:
        temperature = 0.0
        top_p = 1.0
    else:
        temperature = args.temperature
        top_p = args.top_p

    # vLLM uses top_k=-1 for disabled in many versions.
    top_k = args.top_k if int(args.top_k) > 0 else -1

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    enable_lora = bool(args.peft_name)
    lora_request = None
    if enable_lora:
        lora_request = LoRARequest(
            lora_name="ric",
            lora_int_id=1,
            lora_path=args.peft_name,
        )

    print("=" * 100)
    print("[RiC SafeRLHF vLLM generation config]")
    for k, v in asdict(args).items():
        print(f"{k}: {v}")
    print(f"num_prompts: {len(rows)}")
    print(f"enable_lora: {enable_lora}")
    print(f"sampling_params: {sampling_params}")
    print("=" * 100)

    llm = LLM(
        model=args.base_model_name,
        tokenizer=args.base_model_name,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        enable_lora=enable_lora,
        max_lora_rank=args.max_lora_rank,
        trust_remote_code=args.trust_remote_code,
    )

    pref_values = get_pref_values(args)

    for w_helpful in pref_values:
        if not (0.0 <= w_helpful <= 1.0):
            raise ValueError(f"preference must be in [0, 1], got {w_helpful}")

        w_harmless = 1.0 - w_helpful
        target = get_target(args, w_helpful, num_rewards, rewards_reference_list)

        print("-" * 100)
        print(f"[generate] helpful={w_helpful:.1f}, harmless={w_harmless:.1f}")
        print(f"[generate] target rewards={target.tolist()}")
        print("-" * 100)

        conditioned_rows = []
        prompts = []
        for r in rows:
            prompt_with_score = _insert_score_tokens(r["prompt"], target, instructions)
            prompt_with_score = truncate_prompt_if_needed(prompt_with_score, args.max_prompt_length)

            conditioned_rows.append(
                {
                    "raw_prompt": r["raw_prompt"],
                    "prompt": r["prompt"],
                    "prompt_with_score": prompt_with_score,
                }
            )
            prompts.append(prompt_with_score)

        records = []

        # vLLM internally batches requests. Chunking prevents huge request lists
        # and keeps progress logs readable.
        for start in tqdm(range(0, len(prompts), args.batch_size), desc=f"vLLM h{w_helpful:.1f}", dynamic_ncols=True):
            batch_rows = conditioned_rows[start : start + args.batch_size]
            batch_prompts = prompts[start : start + args.batch_size]

            if lora_request is not None:
                outputs = llm.generate(
                    batch_prompts,
                    sampling_params,
                    lora_request=lora_request,
                    use_tqdm=False,
                )
            else:
                outputs = llm.generate(
                    batch_prompts,
                    sampling_params,
                    use_tqdm=False,
                )

            for row, out in zip(batch_rows, outputs):
                response = out.outputs[0].text if out.outputs else ""
                response = _clean_vllm_text(response)

                records.append(
                    {
                        "raw_prompt": row["raw_prompt"],
                        "prompt": row["prompt"],
                        "prompt_with_score": row["prompt_with_score"],
                        "response": response,
                        "weight_helpful": float(w_helpful),
                        "weight_harmless": float(w_harmless),
                        "desired_score1": float(target[0]),
                        "desired_score2": float(target[1]),
                        "method": "ric",
                        "backend": "vllm",
                    }
                )

        tag = f"h{w_helpful:.1f}_s{w_harmless:.1f}"
        json_path = os.path.join(args.output_dir, f"{args.output_prefix}_{tag}.json")
        jsonl_path = os.path.join(args.output_dir, f"{args.output_prefix}_{tag}.jsonl")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[save] {json_path}")
        print(f"[save] {jsonl_path}")

    print("[done] RiC SafeRLHF vLLM generation finished.")


if __name__ == "__main__":
    main()
