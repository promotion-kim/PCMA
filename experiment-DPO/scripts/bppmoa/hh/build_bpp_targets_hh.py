#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build 3D posterior-pooled BPP-MOA soft targets for HH.

This is stage 2 of the paper-faithful pipeline.  It loads three objective-wise
Laplace heads and computes, for each candidate pair (response_0, response_1),

  mu_i = Delta h^T theta_i
  v_i  = Delta h^T Sigma_i Delta h
  ell_i = mu_i / sqrt(1 + pi v_i / 8)
  bpp_logit = sum_i w_i ell_i
  bpp_rho = sigmoid(bpp_logit)

The output dataset is saved with HuggingFace `save_to_disk` layout:
  <output_dir>/train
  <output_dir>/validation
  <output_dir>/metadata.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
from datasets import Dataset, disable_caching
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.bppmoa.hh.bppmoa_hh_utils import (
    OBJECTIVES,
    build_candidate_pair_rows,
    collate_pair_text,
    ensure_tokenizer,
    laplace_pair_stats,
    load_laplace_head,
    load_split,
    pair_features,
    str2bool,
    str_to_torch_dtype,
    validate_weights,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 3D HH BPP-MOA posterior-pooled target dataset.")
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--head_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--helpful_head_path", type=str, default=None)
    p.add_argument("--harmless_head_path", type=str, default=None)
    p.add_argument("--humor_head_path", type=str, default=None)

    p.add_argument("--w_helpful", type=float, default=1.0 / 3.0)
    p.add_argument("--w_harmless", type=float, default=1.0 / 3.0)
    p.add_argument("--w_humor", type=float, default=1.0 / 3.0)
    p.add_argument("--normalize_weights", type=str2bool, default=True)

    p.add_argument("--variance_scale", type=float, default=1.0)
    p.add_argument("--use_posterior_variance", type=str2bool, default=True)
    p.add_argument("--constant_variance", type=str2bool, default=False)
    p.add_argument("--constant_variance_value", type=float, default=1.0)

    p.add_argument("--prompt_format", type=str, default="chat", choices=["chat", "hh", "raw", "modpo"])
    p.add_argument("--prompt_template", type=str, default="{raw_prompt}")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--torch_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--use_flash_attention_2", type=str2bool, default=False)
    p.add_argument("--trust_remote_code", type=str2bool, default=True)
    p.add_argument("--dataset_caching", type=str2bool, default=False)
    p.add_argument("--sanity_check", action="store_true")
    p.add_argument("--limit_examples", type=int, default=None)
    return p.parse_args()


def default_head_path(head_root: str, objective: str) -> str:
    return os.path.join(head_root, objective, "laplace_head.pt")


@torch.no_grad()
def add_targets_for_split(
    split_name: str,
    raw_rows: List[Dict],
    model,
    tokenizer,
    heads: Dict[str, Dict],
    weights: Dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
) -> Dataset:
    rows = build_candidate_pair_rows(
        raw_rows,
        tokenizer=tokenizer,
        prompt_format=args.prompt_format,
        sanity_check=args.sanity_check,
    )
    if args.limit_examples is not None:
        rows = rows[: min(args.limit_examples, len(rows))]

    loader = DataLoader(
        rows,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_pair_text,
    )

    cols = {
        "bpp_rho": [],
        "bpp_logit": [],
    }
    for obj in OBJECTIVES:
        cols[f"bpp_mu_{obj}"] = []
        cols[f"bpp_var_{obj}"] = []
        cols[f"bpp_ell_{obj}"] = []

    for batch in tqdm(loader, desc=f"targets:{split_name}"):
        _, _, delta_h = pair_features(model, tokenizer, batch, args.prompt_template, args.max_length, device)
        pooled_logit = torch.zeros(delta_h.size(0), device=device, dtype=torch.float32)

        for obj in OBJECTIVES:
            theta = heads[obj]["theta"].to(device)
            prec = heads[obj]["diag_precision"].to(device)
            if not args.use_posterior_variance:
                # MAP-only ablation: make posterior variance approximately zero.
                prec = torch.full_like(prec, 1e30)

            mu, var, ell = laplace_pair_stats(
                delta_h,
                theta,
                prec,
                variance_scale=float(args.variance_scale),
            )
            if args.constant_variance:
                # Constant-variance ablation: keep mean but replace pair-specific v.
                const_v = torch.full_like(var, float(args.constant_variance_value))
                ell = mu / torch.sqrt(1.0 + (torch.pi / 8.0) * const_v.clamp_min(0.0))
                var = const_v

            pooled_logit = pooled_logit + float(weights[obj]) * ell
            cols[f"bpp_mu_{obj}"].extend(mu.detach().cpu().float().tolist())
            cols[f"bpp_var_{obj}"].extend(var.detach().cpu().float().tolist())
            cols[f"bpp_ell_{obj}"].extend(ell.detach().cpu().float().tolist())

        rho = torch.sigmoid(pooled_logit)
        cols["bpp_logit"].extend(pooled_logit.detach().cpu().float().tolist())
        cols["bpp_rho"].extend(rho.detach().cpu().float().tolist())

    out = Dataset.from_list(rows)
    for name, values in cols.items():
        out = out.add_column(name, values)
    return out


def main() -> None:
    args = parse_args()
    if not args.dataset_caching:
        disable_caching()

    weights = validate_weights(
        {"helpful": args.w_helpful, "harmless": args.w_harmless, "humor": args.w_humor},
        normalize=bool(args.normalize_weights),
    )

    head_paths = {
        "helpful": args.helpful_head_path or default_head_path(args.head_root, "helpful"),
        "harmless": args.harmless_head_path or default_head_path(args.head_root, "harmless"),
        "humor": args.humor_head_path or default_head_path(args.head_root, "humor"),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = str_to_torch_dtype(args.torch_dtype)

    heads = {obj: load_laplace_head(path, map_location=device) for obj, path in head_paths.items()}

    print(f"[BPP-MOA-HH] loading frozen feature model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        use_flash_attention_2=bool(args.use_flash_attention_2),
        trust_remote_code=bool(args.trust_remote_code),
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=bool(args.trust_remote_code))
    ensure_tokenizer(tokenizer)

    train_rows = load_split(args.data_dir, "train")
    eval_rows = load_split(args.data_dir, "validation")
    if args.sanity_check:
        train_rows = train_rows[:128]
        eval_rows = eval_rows[:128]

    train_dataset = add_targets_for_split("train", train_rows, model, tokenizer, heads, weights, args, device)
    eval_dataset = add_targets_for_split("validation", eval_rows, model, tokenizer, heads, weights, args, device)

    os.makedirs(args.output_dir, exist_ok=True)
    train_dataset.save_to_disk(os.path.join(args.output_dir, "train"))
    eval_dataset.save_to_disk(os.path.join(args.output_dir, "validation"))

    metadata = {
        "model_name": args.model_name,
        "data_dir": args.data_dir,
        "head_paths": head_paths,
        "weights": weights,
        "prompt_format": args.prompt_format,
        "prompt_template": args.prompt_template,
        "variance_scale": args.variance_scale,
        "use_posterior_variance": bool(args.use_posterior_variance),
        "constant_variance": bool(args.constant_variance),
        "constant_variance_value": args.constant_variance_value,
        "num_train": len(train_dataset),
        "num_validation": len(eval_dataset),
        "formula": "bpp_rho=sigmoid(sum_i w_i * mu_i/sqrt(1+pi*v_i/8)) for i in helpful,harmless,humor",
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[BPP-MOA-HH] saved target dataset: {args.output_dir}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
