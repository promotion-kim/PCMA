#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit objective-wise last-layer Laplace reward heads for 3D HH BPP-MOA.

This is stage 1 of the paper-faithful pipeline:
  For each objective i in {helpful, harmless, humor}, construct objective-specific
  pairwise labels from score columns, fit a MAP linear reward head on frozen LM
  features, then save diagonal Laplace precision.

Output:
  <output_root>/<objective>/laplace_head.pt
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.bppmoa.hh.bppmoa_hh_utils import (
    DEFAULT_DIRECTIONS,
    OBJECTIVES,
    build_objective_preference_rows,
    collate_pair_text,
    ensure_tokenizer,
    load_split,
    pair_features,
    save_laplace_head,
    str2bool,
    str_to_torch_dtype,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit 3D HH objective-wise last-layer Laplace heads.")
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--objective", type=str, default="all", choices=["all", *OBJECTIVES])

    p.add_argument("--prompt_format", type=str, default="chat", choices=["chat", "hh", "raw", "modpo"])
    # target dataset rows already store fully formatted prompts, so keep this identity template.
    p.add_argument("--prompt_template", type=str, default="{raw_prompt}")

    p.add_argument("--direction_helpful", type=str, default=DEFAULT_DIRECTIONS["helpful"], choices=["higher_is_better", "lower_is_better"])
    p.add_argument("--direction_harmless", type=str, default=DEFAULT_DIRECTIONS["harmless"], choices=["higher_is_better", "lower_is_better"])
    p.add_argument("--direction_humor", type=str, default=DEFAULT_DIRECTIONS["humor"], choices=["higher_is_better", "lower_is_better"])
    p.add_argument("--min_abs_score_gap", type=float, default=0.0)

    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--prior_precision", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--torch_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--use_flash_attention_2", type=str2bool, default=False)
    p.add_argument("--trust_remote_code", type=str2bool, default=True)
    p.add_argument("--sanity_check", action="store_true")
    p.add_argument("--limit_train_examples", type=int, default=None)
    p.add_argument("--limit_eval_examples", type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def evaluate_head(
    model,
    tokenizer,
    theta: torch.Tensor,
    dataloader: DataLoader,
    prompt_template: str,
    max_length: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total = 0
    for batch in tqdm(dataloader, desc="eval", leave=False):
        _, _, delta_h = pair_features(model, tokenizer, batch, prompt_template, max_length, device)
        logits = delta_h.matmul(theta)
        loss = -F.logsigmoid(logits)
        total_loss += loss.sum().item()
        total_correct += (logits > 0).float().sum().item()
        total += logits.numel()
    return {"nll": total_loss / max(total, 1), "accuracy": total_correct / max(total, 1)}


@torch.no_grad()
def compute_diag_precision(
    model,
    tokenizer,
    theta: torch.Tensor,
    dataloader: DataLoader,
    prompt_template: str,
    max_length: int,
    device: torch.device,
    prior_precision: float,
) -> torch.Tensor:
    """Diagonal GGN/Laplace precision for BT last-layer reward head.

    H_diag = prior_precision + sum_n sigmoid(u_n)(1-sigmoid(u_n)) * Delta h_n^2
    """
    model.eval()
    diag = torch.zeros_like(theta, device=device)
    for batch in tqdm(dataloader, desc="curvature", leave=False):
        _, _, delta_h = pair_features(model, tokenizer, batch, prompt_template, max_length, device)
        logits = delta_h.matmul(theta)
        weight = torch.sigmoid(logits) * torch.sigmoid(-logits)
        diag += (weight.unsqueeze(-1) * delta_h.pow(2)).sum(dim=0)
    diag += float(prior_precision)
    return diag.clamp_min(1e-8)


def fit_one_objective(
    objective: str,
    train_rows: List[Dict],
    eval_rows: List[Dict],
    model,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    directions: Dict[str, str],
) -> Dict[str, float]:
    train_pairs = build_objective_preference_rows(
        train_rows,
        objective=objective,
        tokenizer=tokenizer,
        prompt_format=args.prompt_format,
        directions=directions,
        min_abs_score_gap=args.min_abs_score_gap,
        sanity_check=args.sanity_check,
    )
    eval_pairs = build_objective_preference_rows(
        eval_rows,
        objective=objective,
        tokenizer=tokenizer,
        prompt_format=args.prompt_format,
        directions=directions,
        min_abs_score_gap=args.min_abs_score_gap,
        sanity_check=args.sanity_check,
    )
    if args.limit_train_examples is not None:
        train_pairs = train_pairs[: min(args.limit_train_examples, len(train_pairs))]
    if args.limit_eval_examples is not None:
        eval_pairs = eval_pairs[: min(args.limit_eval_examples, len(eval_pairs))]

    train_loader = DataLoader(
        train_pairs,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_pair_text,
    )
    train_loader_noshuffle = DataLoader(
        train_pairs,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=collate_pair_text,
    )
    eval_loader = DataLoader(
        eval_pairs,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_pair_text,
    )

    # Infer hidden size.
    first_batch = next(iter(train_loader_noshuffle))
    _, _, first_delta = pair_features(model, tokenizer, first_batch, args.prompt_template, args.max_length, device)
    hidden_size = first_delta.size(-1)
    theta = torch.zeros(hidden_size, device=device, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.AdamW([theta], lr=args.learning_rate, weight_decay=0.0)
    num_train = len(train_pairs)
    global_step = 0

    print(f"\n[BPP-MOA-HH] fitting objective={objective}, train={len(train_pairs)}, eval={len(eval_pairs)}")
    for epoch in range(args.num_train_epochs):
        running = 0.0
        running_n = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"{objective}:epoch{epoch}"), start=1):
            _, _, delta_h = pair_features(model, tokenizer, batch, args.prompt_template, args.max_length, device)
            logits = delta_h.matmul(theta)
            nll = -F.logsigmoid(logits).mean()
            # MAP objective: sum NLL + 0.5 * prior_precision * ||theta||^2.
            # We optimize mean NLL, so divide the prior term by N.
            prior = 0.5 * float(args.prior_precision) * theta.pow(2).sum() / max(num_train, 1)
            loss = nll + prior
            (loss / args.gradient_accumulation_steps).backward()

            running += nll.detach().item() * logits.numel()
            running_n += logits.numel()

            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_([theta], args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.logging_steps == 0:
                    print(f"[{objective}] epoch={epoch} step={global_step} train_nll={running / max(running_n, 1):.4f}")

        metrics = evaluate_head(model, tokenizer, theta.detach(), eval_loader, args.prompt_template, args.max_length, device)
        print(f"[{objective}] epoch={epoch} eval_nll={metrics['nll']:.4f} eval_acc={metrics['accuracy']:.4f}")

    final_metrics = evaluate_head(model, tokenizer, theta.detach(), eval_loader, args.prompt_template, args.max_length, device)
    diag_precision = compute_diag_precision(
        model,
        tokenizer,
        theta.detach(),
        train_loader_noshuffle,
        args.prompt_template,
        args.max_length,
        device,
        prior_precision=args.prior_precision,
    )

    out_dir = os.path.join(args.output_root, objective)
    os.makedirs(out_dir, exist_ok=True)
    metadata = {
        "objective": objective,
        "model_name": args.model_name,
        "prompt_format": args.prompt_format,
        "prompt_template": args.prompt_template,
        "directions": directions,
        "min_abs_score_gap": args.min_abs_score_gap,
        "num_train_pairs": len(train_pairs),
        "num_eval_pairs": len(eval_pairs),
        "prior_precision": args.prior_precision,
        "final_eval": final_metrics,
        "formula": "p(y_chosen > y_rejected)=sigmoid(theta^T Delta h), q(theta)=N(theta_map,H_diag^-1)",
    }
    save_laplace_head(os.path.join(out_dir, "laplace_head.pt"), theta.detach(), diag_precision, metadata)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[{objective}] saved {os.path.join(out_dir, 'laplace_head.pt')}")
    return final_metrics


def main() -> None:
    args = parse_args()
    directions = {
        "helpful": args.direction_helpful,
        "harmless": args.direction_harmless,
        "humor": args.direction_humor,
    }

    train_rows = load_split(args.data_dir, "train")
    eval_rows = load_split(args.data_dir, "validation")
    if args.sanity_check:
        train_rows = train_rows[:128]
        eval_rows = eval_rows[:128]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = str_to_torch_dtype(args.torch_dtype)

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

    objectives = list(OBJECTIVES) if args.objective == "all" else [args.objective]
    summary = {}
    for objective in objectives:
        summary[objective] = fit_one_objective(objective, train_rows, eval_rows, model, tokenizer, args, device, directions)

    os.makedirs(args.output_root, exist_ok=True)
    with open(os.path.join(args.output_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[BPP-MOA-HH] done. summary={os.path.join(args.output_root, 'summary.json')}")


if __name__ == "__main__":
    main()
