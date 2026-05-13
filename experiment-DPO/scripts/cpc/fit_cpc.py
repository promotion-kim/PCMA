#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone CPC calibrator fitting script.

Why this file exists
--------------------
This script intentionally avoids importing project modules such as
`src.data.configs` and `src.utils`, because those imports may pull in
TRL/DeepSpeed and fail under version-mismatched environments.

It only needs:
  - datasets
  - transformers
  - peft
  - accelerate
  - torch

It computes objective-specific implicit reward gaps from DPO adapters and
fits prompt-independent CPC log-precisions:

    p(z_i = 1 | gap_i) = sigmoid(exp(a_i) * gap_i)

The saved file is:
    <output_dir>/cpc_calibrator.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"


# ---------------------------------------------------------------------
# Small local utilities, replacing src.utils dependencies.
# ---------------------------------------------------------------------

def split_csv(x: str) -> List[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


def common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_main(accelerator: Accelerator, *args, **kwargs) -> None:
    if accelerator.is_local_main_process:
        print(*args, **kwargs, flush=True)


def str2bool(x: str | bool) -> bool:
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x in {"true", "1", "yes", "y"}:
        return True
    if x in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {x}")


# ---------------------------------------------------------------------
# CPC objective-level calibrator.
# ---------------------------------------------------------------------

@dataclass
class CPCFitResult:
    objective_log_precisions: List[float]
    objective_precisions: List[float]
    prior_sigma_a: float
    num_objectives: int
    num_examples: int
    train_nll: float
    config: Dict


class ObjectiveCPCalibrator:
    def __init__(self, objective_log_precisions: Sequence[float], prior_sigma_a: float = 2.0):
        self.objective_log_precisions = [float(x) for x in objective_log_precisions]
        self.prior_sigma_a = float(prior_sigma_a)

    @property
    def objective_precisions(self) -> List[float]:
        return [float(math.exp(x)) for x in self.objective_log_precisions]

    def coefficients(self, w: Sequence[float], gamma: float = 1.0) -> List[float]:
        if len(w) != len(self.objective_log_precisions):
            raise ValueError(
                f"len(w)={len(w)} but num_objectives={len(self.objective_log_precisions)}"
            )
        return [
            float(wi) * float(math.exp(float(gamma) * ai))
            for wi, ai in zip(w, self.objective_log_precisions)
        ]

    def save_pretrained(self, output_dir: str | os.PathLike, extra: Optional[Dict] = None) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "objective_level_cpc_v1",
            "objective_log_precisions": self.objective_log_precisions,
            "objective_precisions": self.objective_precisions,
            "prior_sigma_a": self.prior_sigma_a,
        }
        if extra is not None:
            payload["extra"] = extra
        with open(output_dir / "cpc_calibrator.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str | os.PathLike) -> "ObjectiveCPCalibrator":
        path = Path(path)
        json_path = path / "cpc_calibrator.json" if path.is_dir() else path
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            objective_log_precisions=payload["objective_log_precisions"],
            prior_sigma_a=payload.get("prior_sigma_a", 2.0),
        )


def fit_objective_cpc(
    objective_idx: torch.Tensor,
    signed_gaps: torch.Tensor,
    num_objectives: int,
    prior_sigma_a: float = 2.0,
    num_steps: int = 2000,
    lr: float = 3e-3,
    batch_size: int = 512,
    seed: int = 42,
    device: str = "cuda",
    log_every: int = 100,
    accelerator: Optional[Accelerator] = None,
) -> Tuple[ObjectiveCPCalibrator, float]:
    set_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    objective_idx = objective_idx.to(dev).long()
    signed_gaps = signed_gaps.to(dev).float()

    a = torch.zeros(num_objectives, device=dev, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)

    n = signed_gaps.numel()
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    last_nll = float("nan")
    for step in range(num_steps):
        if batch_size is not None and batch_size > 0 and batch_size < n:
            batch_ids = torch.randint(0, n, (batch_size,), generator=g, device="cpu").to(dev)
            obj_b = objective_idx[batch_ids]
            gap_b = signed_gaps[batch_ids]
        else:
            obj_b = objective_idx
            gap_b = signed_gaps

        logits = torch.exp(a[obj_b]) * gap_b
        nll = F.softplus(-logits).mean()
        prior = torch.sum(a ** 2) / (2.0 * float(n) * (prior_sigma_a ** 2))
        loss = nll + prior

        opt.zero_grad()
        loss.backward()
        opt.step()

        last_nll = float(nll.detach().cpu().item())
        if accelerator is not None and accelerator.is_local_main_process:
            if log_every > 0 and (step % log_every == 0 or step == num_steps - 1):
                vals = a.detach().cpu().tolist()
                print(f"[CPC fit] step={step:04d} nll={last_nll:.6f} a={vals}", flush=True)

    with torch.no_grad():
        full_logits = torch.exp(a[objective_idx]) * signed_gaps
        full_nll = float(F.softplus(-full_logits).mean().detach().cpu().item())
        a_cpu = a.detach().cpu().tolist()

    return ObjectiveCPCalibrator(a_cpu, prior_sigma_a=prior_sigma_a), full_nll


# ---------------------------------------------------------------------
# Dataset parsing.
# ---------------------------------------------------------------------

def strip_objective_suffix(dataset_name: str) -> Tuple[str, Optional[str]]:
    """
    Handles names like:
      PKU-Alignment/PKU-SafeRLHF-10K-better
      PKU-Alignment/PKU-SafeRLHF-10K-safer

    Returns:
      base_name, objective_kind
    """
    if dataset_name.endswith("-better"):
        return dataset_name[: -len("-better")], "better"
    if dataset_name.endswith("-safer"):
        return dataset_name[: -len("-safer")], "safer"
    return dataset_name, None


def format_prompt(example: dict, prompt_template: str) -> str:
    if "prompt" in example and example["prompt"] is not None:
        prompt = example["prompt"]
        # Some datasets already contain fully formatted prompts. Keep as-is.
        return str(prompt)
    if "raw_prompt" in example and example["raw_prompt"] is not None:
        return prompt_template.format(raw_prompt=example["raw_prompt"])
    if "instruction" in example and example["instruction"] is not None:
        return prompt_template.format(raw_prompt=example["instruction"])
    raise KeyError(f"Cannot find prompt/raw_prompt/instruction keys. Keys={list(example.keys())}")


def pick_response_pair_from_pku_like(example: dict, objective_kind: Optional[str]) -> Optional[Tuple[str, str]]:
    """
    Robust parser for PKU SafeRLHF-like rows. Returns chosen, rejected or None if ambiguous.
    """
    # Already processed preference dataset.
    if "chosen" in example and "rejected" in example and example["chosen"] is not None and example["rejected"] is not None:
        return str(example["chosen"]), str(example["rejected"])

    # Generic pair fields.
    r0 = example.get("response_0", None)
    r1 = example.get("response_1", None)
    if r0 is None or r1 is None:
        # Other common names.
        r0 = example.get("answer_0", example.get("output_0", None))
        r1 = example.get("answer_1", example.get("output_1", None))
    if r0 is None or r1 is None:
        return None

    # Direct id fields, if available.
    id_keys = []
    if objective_kind == "better":
        id_keys = ["better_response_id", "chosen_response_id", "preference"]
    elif objective_kind == "safer":
        id_keys = ["safer_response_id", "chosen_response_id", "preference"]
    else:
        id_keys = ["chosen_response_id", "better_response_id", "safer_response_id", "preference"]

    for k in id_keys:
        if k in example and example[k] is not None:
            val = example[k]
            try:
                idx = int(val)
                if idx == 0:
                    return str(r0), str(r1)
                if idx == 1:
                    return str(r1), str(r0)
            except Exception:
                pass

    # Boolean flags.
    if objective_kind == "better":
        k0s = ["is_response_0_better", "response_0_better", "is_response_0_preferred"]
        k1s = ["is_response_1_better", "response_1_better", "is_response_1_preferred"]
    elif objective_kind == "safer":
        k0s = ["is_response_0_safe", "response_0_safe", "is_response_0_safer"]
        k1s = ["is_response_1_safe", "response_1_safe", "is_response_1_safer"]
    else:
        k0s = ["is_response_0_better", "is_response_0_safe", "response_0_better", "response_0_safe"]
        k1s = ["is_response_1_better", "is_response_1_safe", "response_1_better", "response_1_safe"]

    k0 = next((k for k in k0s if k in example), None)
    k1 = next((k for k in k1s if k in example), None)
    if k0 is not None and k1 is not None:
        b0 = bool(example[k0])
        b1 = bool(example[k1])
        if b0 and not b1:
            return str(r0), str(r1)
        if b1 and not b0:
            return str(r1), str(r0)
        # ambiguous tie, skip
        return None

    return None


def load_preference_pairs(
    dataset_name: str,
    split: str,
    prompt_template: str,
    max_examples: Optional[int],
    sanity_check: bool,
    accelerator: Accelerator,
) -> List[dict]:
    """
    Direct HF dataset loader. Supports:
      - datasets already containing chosen/rejected
      - PKU-like response_0/response_1 plus better/safer labels
    """
    base_name, objective_kind = strip_objective_suffix(dataset_name)

    try_names = [dataset_name]
    if base_name != dataset_name:
        try_names.append(base_name)

    last_err = None
    ds = None
    loaded_name = None
    for name in try_names:
        try:
            ds = load_dataset(name, split=split)
            loaded_name = name
            break
        except Exception as e:
            last_err = e

    if ds is None:
        raise RuntimeError(f"Failed to load dataset {dataset_name}. Last error: {last_err}")

    if sanity_check:
        ds = ds.select(range(min(len(ds), 128)))

    pairs: List[dict] = []
    skipped = 0
    for ex in ds:
        try:
            prompt = format_prompt(ex, prompt_template=prompt_template)
            picked = pick_response_pair_from_pku_like(ex, objective_kind=objective_kind)
            if picked is None:
                skipped += 1
                continue
            chosen, rejected = picked
            raw_prompt = ex.get("raw_prompt", ex.get("prompt", ex.get("instruction", prompt)))
            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "raw_prompt": str(raw_prompt),
                }
            )
        except Exception:
            skipped += 1
            continue

        if max_examples is not None and len(pairs) >= max_examples:
            break

    print_main(
        accelerator,
        f"[dataset] requested={dataset_name} loaded={loaded_name} split={split} "
        f"pairs={len(pairs)} skipped={skipped} objective_kind={objective_kind}",
    )

    if len(pairs) == 0:
        raise RuntimeError(
            f"No valid pairs found for dataset={dataset_name}. "
            f"Try checking field names with datasets.load_dataset(...)."
        )

    return pairs


# ---------------------------------------------------------------------
# Log-probability computation.
# ---------------------------------------------------------------------

def tokenize_prompt_response(tokenizer, prompt: str, response: str, max_length: int) -> dict:
    prompt_toks = tokenizer(prompt, add_special_tokens=False)
    full_toks = tokenizer(prompt + response, add_special_tokens=False)
    input_ids = full_toks["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = full_toks["attention_mask"] + [1]
    prompt_len = common_prefix_length(prompt_toks["input_ids"], input_ids)

    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_logp(tokenizer, features: Sequence[dict]) -> dict:
    batch = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
    padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
    max_len = padded["input_ids"].shape[1]

    labels = []
    for f in features:
        lab = f["labels"] + [-100] * (max_len - len(f["labels"]))
        labels.append(lab)
    padded["labels"] = torch.tensor(labels, dtype=torch.long)
    return padded


@torch.no_grad()
def sequence_logps(model, batch: dict, device: torch.device) -> torch.Tensor:
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    labels = batch["labels"][:, 1:].clone()
    mask = labels.ne(-100)
    labels = labels.masked_fill(~mask, 0)

    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1).detach().cpu()


def compute_objective_gaps(
    model,
    tokenizer,
    adapter_name: str,
    pairs: List[dict],
    max_length: int,
    batch_size: int,
    num_workers: int,
    accelerator: Accelerator,
) -> torch.Tensor:
    model.eval()
    model.set_adapter(adapter_name)

    items = []
    for p in pairs:
        items.append(tokenize_prompt_response(tokenizer, p["prompt"], p["chosen"], max_length=max_length))
        items.append(tokenize_prompt_response(tokenizer, p["prompt"], p["rejected"], max_length=max_length))

    loader = DataLoader(
        items,
        batch_size=batch_size * 2,
        collate_fn=lambda fs: collate_logp(tokenizer, fs),
        num_workers=num_workers,
    )

    policy_logps_all: List[torch.Tensor] = []
    ref_logps_all: List[torch.Tensor] = []

    for step, batch in enumerate(
        tqdm(
            loader,
            desc=f"{adapter_name}: logp gaps",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
        )
    ):
        policy_logps_all.append(sequence_logps(model, batch, accelerator.device))
        with model.disable_adapter():
            ref_logps_all.append(sequence_logps(model, batch, accelerator.device))

        if accelerator.is_local_main_process and (step + 1) % 50 == 0:
            processed_items = min((step + 1) * loader.batch_size, len(items))
            print(
                f"[{adapter_name}] processed {processed_items // 2}/{len(pairs)} pairs",
                flush=True,
            )

    policy_logps = torch.cat(policy_logps_all, dim=0).view(-1, 2)
    ref_logps = torch.cat(ref_logps_all, dim=0).view(-1, 2)

    gap = (policy_logps[:, 0] - ref_logps[:, 0]) - (policy_logps[:, 1] - ref_logps[:, 1])
    return gap.float()


# ---------------------------------------------------------------------
# CLI / main.
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fit objective-level CPC calibrator without importing src/trl/deepspeed.")

    p.add_argument("--sft_model_name", type=str, required=True)
    p.add_argument("--objective_adapter_names", type=str, required=True)
    p.add_argument("--objective_dataset_names", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--sanity_check", type=str2bool, default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_examples_per_objective", type=int, default=None)
    p.add_argument("--use_flash_attention_2", type=str2bool, default=False)
    p.add_argument("--torch_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    p.add_argument("--cpc_steps", type=int, default=2000)
    p.add_argument("--cpc_lr", type=float, default=3e-3)
    p.add_argument("--cpc_batch_size", type=int, default=512)
    p.add_argument("--cpc_log_every", type=int, default=100)
    p.add_argument("--prior_sigma_a", type=float, default=2.0)

    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    accelerator = Accelerator()

    objective_adapters = split_csv(args.objective_adapter_names)
    objective_datasets = split_csv(args.objective_dataset_names)
    if len(objective_adapters) != len(objective_datasets):
        raise ValueError("objective_adapter_names and objective_dataset_names must have the same length.")
    num_objectives = len(objective_adapters)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    print_main(accelerator, "loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print_main(accelerator, "loading SFT/reference model and adapters...")
    base = AutoModelForCausalLM.from_pretrained(
        args.sft_model_name,
        use_flash_attention_2=args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    base.config.update({"use_cache": False, "pad_token_id": tokenizer.pad_token_id})

    model = PeftModel.from_pretrained(base, objective_adapters[0], adapter_name="obj0", is_trainable=False)
    for i, path in enumerate(objective_adapters[1:], start=1):
        model.load_adapter(path, adapter_name=f"obj{i}", is_trainable=False)

    model.to(accelerator.device)
    model.eval()

    all_gaps: List[torch.Tensor] = []
    all_obj_idx: List[torch.Tensor] = []
    dataset_counts: Dict[str, int] = {}

    for i, dataset_name in enumerate(objective_datasets):
        pairs = load_preference_pairs(
            dataset_name=dataset_name,
            split=args.dataset_split,
            prompt_template=args.prompt_template,
            max_examples=args.max_examples_per_objective,
            sanity_check=args.sanity_check,
            accelerator=accelerator,
        )
        dataset_counts[dataset_name] = len(pairs)

        print_main(accelerator, f"computing gaps for objective {i} using adapter obj{i}")
        gaps = compute_objective_gaps(
            model=model,
            tokenizer=tokenizer,
            adapter_name=f"obj{i}",
            pairs=pairs,
            max_length=args.max_length,
            batch_size=args.per_device_batch_size,
            num_workers=args.num_workers,
            accelerator=accelerator,
        )
        all_gaps.append(gaps)
        all_obj_idx.append(torch.full((len(gaps),), i, dtype=torch.long))

    signed_gaps = torch.cat(all_gaps, dim=0)
    objective_idx = torch.cat(all_obj_idx, dim=0)

    print_main(
        accelerator,
        f"fitting objective-level CPC: num_objectives={num_objectives}, total_pairs={signed_gaps.numel()}",
    )
    calibrator, train_nll = fit_objective_cpc(
        objective_idx=objective_idx,
        signed_gaps=signed_gaps,
        num_objectives=num_objectives,
        prior_sigma_a=args.prior_sigma_a,
        num_steps=args.cpc_steps,
        lr=args.cpc_lr,
        batch_size=args.cpc_batch_size,
        seed=args.seed,
        device=f"cuda:{accelerator.local_process_index}" if torch.cuda.is_available() else "cpu",
        log_every=args.cpc_log_every,
        accelerator=accelerator,
    )

    if accelerator.is_local_main_process:
        extra = {
            "script": "fit_cpc_standalone.py",
            "train_nll": train_nll,
            "dataset_counts": dataset_counts,
            "objective_adapter_names": objective_adapters,
            "objective_dataset_names": objective_datasets,
            "sft_model_name": args.sft_model_name,
            "max_length": args.max_length,
            "seed": args.seed,
            "cpc_steps": args.cpc_steps,
            "cpc_lr": args.cpc_lr,
            "cpc_batch_size": args.cpc_batch_size,
        }
        calibrator.save_pretrained(args.output_dir, extra=extra)
        print("saved CPC calibrator to:", args.output_dir, flush=True)
        print("objective_log_precisions:", calibrator.objective_log_precisions, flush=True)
        print("objective_precisions:", calibrator.objective_precisions, flush=True)
        print("train_nll:", train_nll, flush=True)


if __name__ == "__main__":
    main()
