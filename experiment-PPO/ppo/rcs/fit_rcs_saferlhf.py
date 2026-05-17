#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit objective-level RCS log precisions for SafeRLHF reward/cost models.

This script fits one log precision a_i per objective using Bradley--Terry style
likelihood on objective-specific pairwise SafeRLHF data:

    p(z_i = 1 | gap_i) = sigmoid(exp(a_i) * gap_i)

where gap_i = signed_score_i(chosen) - signed_score_i(rejected), and signed_score
already applies reward_signs, e.g. reward_sign=-1 for a cost model so that the
harmlessness score is larger-is-better.

The output directory contains both ``rcs_calibrator.json`` and
``cpc_calibrator.json`` for compatibility with existing CPC-MODPO utilities.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from safe_rlhf.models import AutoModelForScore
from tqdm import tqdm
from transformers import AutoTokenizer


DEFAULT_PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"


def split_csv(x: str) -> List[str]:
    return [v.strip() for v in str(x).split(",") if v.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_objective_suffix(dataset_name: str) -> Tuple[str, Optional[str]]:
    if dataset_name.endswith("-better"):
        return dataset_name[: -len("-better")], "better"
    if dataset_name.endswith("-safer"):
        return dataset_name[: -len("-safer")], "safer"
    return dataset_name, None


def load_any_split(dataset_name: str, split: str):
    base_name, _ = strip_objective_suffix(dataset_name)
    names = [dataset_name]
    if base_name != dataset_name:
        names.append(base_name)

    last_error = None
    for name in names:
        try:
            return load_dataset(name, split=split), name
        except Exception as exc:
            last_error = exc

    if split in {"validation", "eval", "dev"}:
        for name in names:
            try:
                return load_dataset(name, split="train[:5%]"), name
            except Exception as exc:
                last_error = exc

    raise RuntimeError(f"Failed to load dataset={dataset_name} split={split}. Last error: {last_error}")


def format_prompt_and_raw(ex: dict, prompt_template: str) -> Tuple[str, str]:
    if ex.get("raw_prompt") is not None:
        raw = str(ex["raw_prompt"])
    elif ex.get("prompt") is not None:
        raw = str(ex["prompt"])
    elif ex.get("instruction") is not None:
        raw = str(ex["instruction"])
    elif ex.get("query") is not None:
        raw = str(ex["query"])
    else:
        raise KeyError(f"No prompt/raw_prompt/instruction/query in keys={list(ex.keys())}")

    if "BEGINNING OF CONVERSATION:" in raw and "ASSISTANT:" in raw:
        return raw, raw
    return prompt_template.format(raw_prompt=raw), raw


def pick_pair(ex: dict, objective_kind: Optional[str]) -> Optional[Tuple[str, str]]:
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


def load_preference_pairs(
    dataset_name: str,
    split: str,
    prompt_template: str,
    max_examples: int,
    sanity_check: bool,
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
        pairs.append({"prompt": prompt, "raw_prompt": raw_prompt, "chosen": chosen, "rejected": rejected})
        if max_examples is not None and max_examples > 0 and len(pairs) >= max_examples:
            break

    print(
        f"[load_pairs] requested={dataset_name} loaded={loaded_name} split={split} "
        f"objective={objective_kind} pairs={len(pairs)} skipped={skipped}",
        flush=True,
    )
    if not pairs:
        raise RuntimeError(f"No valid preference pairs found for {dataset_name}")
    return pairs


class ScoreModel:
    def __init__(self, model_name: str, device: torch.device, max_length: int):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
        if device.type == "cuda":
            kwargs["device_map"] = {"": int(device.index or 0)}
        self.model = AutoModelForScore.from_pretrained(model_name, **kwargs)
        if device.type != "cuda":
            self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def score_texts(self, texts: List[str], batch_size: int) -> List[float]:
        out: List[float] = []
        for start in tqdm(range(0, len(texts), batch_size), desc=f"score {self.model_name}"):
            batch_texts = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            scores = getattr(outputs, "end_scores", None)
            if scores is None:
                scores = getattr(outputs, "scores", None)
            if scores is None:
                raise RuntimeError(f"AutoModelForScore output has no end_scores/scores: {outputs}")
            if scores.ndim == 2 and scores.shape[-1] == 1:
                scores = scores.squeeze(-1)
            elif scores.ndim >= 2:
                scores = scores[:, -1]
            out.extend(scores.float().detach().cpu().tolist())
        return out


def compute_signed_gaps_for_objective(
    pairs: List[dict],
    model_name: str,
    reward_sign: float,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> torch.Tensor:
    scorer = ScoreModel(model_name, device=device, max_length=max_length)
    chosen_texts = [ex["prompt"] + ex["chosen"] for ex in pairs]
    rejected_texts = [ex["prompt"] + ex["rejected"] for ex in pairs]

    chosen_scores = scorer.score_texts(chosen_texts, batch_size=batch_size)
    rejected_scores = scorer.score_texts(rejected_texts, batch_size=batch_size)

    gaps = [float(reward_sign) * (float(c) - float(r)) for c, r in zip(chosen_scores, rejected_scores)]

    # Explicitly release the model before moving to the next objective.
    del scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.tensor(gaps, dtype=torch.float32)


def fit_objective_rcs(
    objective_idx: torch.Tensor,
    signed_gaps: torch.Tensor,
    num_objectives: int,
    prior_sigma_a: float,
    steps: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: torch.device,
    log_every: int,
) -> Tuple[List[float], float]:
    set_seed(seed)
    objective_idx = objective_idx.to(device).long()
    signed_gaps = signed_gaps.to(device).float()

    a = torch.zeros(num_objectives, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)
    n = int(signed_gaps.numel())
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    for step in range(steps):
        if batch_size is not None and batch_size > 0 and batch_size < n:
            ids = torch.randint(0, n, (batch_size,), generator=gen, device="cpu").to(device)
            obj_b = objective_idx[ids]
            gap_b = signed_gaps[ids]
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

        if log_every > 0 and (step % log_every == 0 or step == steps - 1):
            print(f"[RCS fit] step={step:04d} nll={float(nll):.6f} a={a.detach().cpu().tolist()}", flush=True)

    with torch.no_grad():
        full_nll = float(F.softplus(-torch.exp(a[objective_idx]) * signed_gaps).mean().detach().cpu())
        a_list = [float(x) for x in a.detach().cpu().tolist()]
    return a_list, full_nll


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective_dataset_names", default="PKU-Alignment/PKU-SafeRLHF-10K-better,PKU-Alignment/PKU-SafeRLHF-10K-safer")
    parser.add_argument("--score_model_names", default="PKU-Alignment/beaver-7b-v1.0-reward,PKU-Alignment/beaver-7b-v1.0-cost")
    parser.add_argument("--reward_signs", default="1,-1")
    parser.add_argument("--reward_names", default="helpful,harmless")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--max_examples_per_objective", type=int, default=-1)
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--rcs_steps", type=int, default=2000)
    parser.add_argument("--rcs_lr", type=float, default=3e-3)
    parser.add_argument("--rcs_batch_size", type=int, default=512)
    parser.add_argument("--prior_sigma_a", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    objective_datasets = split_csv(args.objective_dataset_names)
    score_models = split_csv(args.score_model_names)
    reward_signs = [float(x) for x in split_csv(args.reward_signs)]
    reward_names = split_csv(args.reward_names)

    if not (len(objective_datasets) == len(score_models) == len(reward_signs) == len(reward_names)):
        raise ValueError(
            "objective_dataset_names, score_model_names, reward_signs, and reward_names must have the same length. "
            f"Got {len(objective_datasets)}, {len(score_models)}, {len(reward_signs)}, {len(reward_names)}"
        )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    all_gaps: List[torch.Tensor] = []
    all_obj_idx: List[torch.Tensor] = []
    dataset_counts: Dict[str, int] = {}
    gap_stats: Dict[str, Dict[str, float]] = {}

    for i, (dataset_name, model_name, sign, obj_name) in enumerate(zip(objective_datasets, score_models, reward_signs, reward_names)):
        pairs = load_preference_pairs(
            dataset_name=dataset_name,
            split=args.dataset_split,
            prompt_template=args.prompt_template,
            max_examples=args.max_examples_per_objective,
            sanity_check=args.sanity_check,
        )
        dataset_counts[dataset_name] = len(pairs)
        print(f"[objective {i}] name={obj_name} dataset={dataset_name} model={model_name} sign={sign}", flush=True)
        gaps = compute_signed_gaps_for_objective(
            pairs=pairs,
            model_name=model_name,
            reward_sign=sign,
            device=device,
            max_length=args.max_length,
            batch_size=args.per_device_batch_size,
        )
        all_gaps.append(gaps)
        all_obj_idx.append(torch.full((len(gaps),), i, dtype=torch.long))
        gap_stats[obj_name] = {
            "mean": float(gaps.mean()),
            "std": float(gaps.std(unbiased=False)),
            "min": float(gaps.min()),
            "max": float(gaps.max()),
            "agreement_rate_gap_positive": float((gaps > 0).float().mean()),
        }
        print(f"[objective {i}] gap_stats={gap_stats[obj_name]}", flush=True)

    signed_gaps = torch.cat(all_gaps, dim=0)
    objective_idx = torch.cat(all_obj_idx, dim=0)

    a_list, train_nll = fit_objective_rcs(
        objective_idx=objective_idx,
        signed_gaps=signed_gaps,
        num_objectives=len(reward_names),
        prior_sigma_a=args.prior_sigma_a,
        steps=args.rcs_steps,
        lr=args.rcs_lr,
        batch_size=args.rcs_batch_size,
        seed=args.seed,
        device=device,
        log_every=args.log_every,
    )
    lambdas = [float(math.exp(a)) for a in a_list]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "objective_level_rcs_v1",
        "objective_names": reward_names,
        "objective_log_precisions": a_list,
        "objective_precisions": lambdas,
        "prior_sigma_a": float(args.prior_sigma_a),
        "num_objectives": len(reward_names),
        "num_examples": int(signed_gaps.numel()),
        "train_nll": float(train_nll),
        "dataset_counts": dataset_counts,
        "gap_stats": gap_stats,
        "config": vars(args),
    }
    with open(out_dir / "rcs_calibrator.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    # Compatibility with older CPC utilities.
    with open(out_dir / "cpc_calibrator.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("saved RCS calibrator to:", out_dir, flush=True)
    print("objective_log_precisions:", a_list, flush=True)
    print("objective_precisions:", lambdas, flush=True)
    print("train_nll:", train_nll, flush=True)


if __name__ == "__main__":
    main()
