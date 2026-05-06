#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reward-model evaluation for MORLHF vs PC-MORLHF generations.

This script is adapted from the MODPO/PC-MODPO evaluation pipeline, but compares
MORLHF and PC-MORLHF generation JSON files.

Inputs are generation JSON files produced by generation_morlhf.py, with the form:
{
  "metadata": {...},
  "data": [
    {"raw_prompt": ..., "model_input": ..., "generation": ...},
    ...
  ]
}

It scores each prompt-response pair with:
  helpful  = PKU-Alignment/beaver-7b-v1.0-reward
  cost     = PKU-Alignment/beaver-7b-v1.0-cost
  harmless = -cost

and reports helpful, harmless, cost, and MIP.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


DEFAULT_REWARD_MODEL = "PKU-Alignment/beaver-7b-v1.0-reward"
DEFAULT_COST_MODEL = "PKU-Alignment/beaver-7b-v1.0-cost"
DEFAULT_PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"


@dataclass
class GenerationSet:
    name: str
    path: str
    metadata: Dict[str, Any]
    data: List[Dict[str, Any]]


def format_weight(w: float) -> str:
    return f"{w:.1f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--w",
        type=float,
        default=0.5,
        help="Helpfulness weight. Harmlessness weight is automatically set to 1-w.",
    )

    parser.add_argument(
        "--morlhf_json",
        type=str,
        default=None,
        help="Path to MORLHF generation JSON. If omitted, inferred from --w.",
    )
    parser.add_argument(
        "--pcmorlhf_json",
        type=str,
        default=None,
        help="Path to PC-MORLHF generation JSON. If omitted, inferred from --w.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. If omitted, inferred from --w.",
    )

    parser.add_argument("--reward_model_name", type=str, default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--cost_model_name", type=str, default=DEFAULT_COST_MODEL)

    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)

    parser.add_argument("--torch_dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--device_map",
        type=str,
        default="none",
        help="Use 'none' for single GPU .to(cuda), or 'auto' for HF device_map='auto'.",
    )

    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument("--save_scored_json", action="store_true")
    parser.add_argument("--no_save_scored_json", dest="save_scored_json", action="store_false")
    parser.set_defaults(save_scored_json=True)

    args = parser.parse_args()

    if not (0.0 <= args.w <= 1.0):
        raise ValueError(f"--w must be in [0, 1], got {args.w}")

    w_str = format_weight(args.w)

    if args.morlhf_json is None:
        args.morlhf_json = (
            f"/home/sjkim/MOD/experiment-PPO/outputs/generation/morlhf/morlhf_w{w_str}.json"
        )

    if args.pcmorlhf_json is None:
        args.pcmorlhf_json = (
            f"/home/sjkim/MOD/experiment-PPO/outputs/generation/pcmorlhf/pcmorlhf_w{w_str}.json"
        )

    if args.output_dir is None:
        args.output_dir = (
            f"/home/sjkim/MOD/experiment-PPO/outputs/eval_cleaned_mip_morlhf/{w_str}"
        )

    return args


def get_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def load_generation_set(name: str, path: str, limit: int = -1) -> GenerationSet:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    metadata = obj.get("metadata", {}) if isinstance(obj, dict) else {}
    data = obj.get("data", obj if isinstance(obj, list) else [])

    if not isinstance(data, list):
        raise ValueError(f"Invalid generation JSON format: {path}")

    if limit is not None and limit > 0:
        data = data[:limit]

    return GenerationSet(name=name, path=path, metadata=metadata, data=data)


def remove_leakage(text: str) -> str:
    """Remove Alpaca/instruction-format leakage after the actual answer."""
    if text is None:
        return ""

    text = str(text).strip()

    leakage_patterns = [
        r"\n\s*#{1,6}\s*\n\s*\d+\.\s*Instruction\s*:",
        r"\n\s*#{1,6}\s*\d+\.\s*Instruction\s*:",
        r"\n\s*###\s*\d+\.\s*Instruction\s*:",
        r"\n\s*\d+\.\s*Instruction\s*:",
        r"\n\s*Instruction\s*:",
        r"\n\s*#{1,6}\s*\n\s*Instruction\s*:",
        r"\n\s*#{1,6}\s*\n\s*\d+\.\s*Input\s*:",
        r"\n\s*#{1,6}\s*\n\s*\d+\.\s*Output\s*:",
    ]

    cut_positions = []
    for pat in leakage_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            cut_positions.append(m.start())

    if cut_positions:
        text = text[: min(cut_positions)].strip()

    secondary_patterns = [
        r"\n\s*\d+\.\s*Input\s*:",
        r"\n\s*\d+\.\s*Output\s*:",
        r"\n\s*Input\s*:",
        r"\n\s*Output\s*:",
    ]

    cut_positions = []
    for pat in secondary_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            cut_positions.append(m.start())

    if cut_positions:
        text = text[: min(cut_positions)].strip()

    return text


def build_score_input(row: Dict[str, Any], prompt_template: str) -> str:
    generation = str(row.get("clean_generation", row.get("generation", ""))).strip()

    if "model_input" in row and row["model_input"] is not None:
        model_input = str(row["model_input"])
    elif "raw_prompt" in row and row["raw_prompt"] is not None:
        raw_prompt = str(row["raw_prompt"])
        if "BEGINNING OF CONVERSATION:" in raw_prompt and "ASSISTANT:" in raw_prompt:
            model_input = raw_prompt
        else:
            model_input = prompt_template.format(raw_prompt=raw_prompt)
    else:
        model_input = ""

    if model_input and not model_input.endswith((" ", "\n")):
        model_input = model_input + " "

    return model_input + generation


def clean_generation_set(gs: GenerationSet, prompt_template: str) -> GenerationSet:
    cleaned_data = []

    for row in gs.data:
        new_row = dict(row)

        original_generation = str(new_row.get("generation", "")).strip()
        clean_generation = remove_leakage(original_generation)

        new_row["original_generation"] = original_generation
        new_row["clean_generation"] = clean_generation
        new_row["had_leakage_removed"] = int(original_generation != clean_generation)
        new_row["score_text"] = build_score_input(new_row, prompt_template=prompt_template)

        cleaned_data.append(new_row)

    return GenerationSet(
        name=gs.name,
        path=gs.path,
        metadata=gs.metadata,
        data=cleaned_data,
    )


def extract_scores(outputs: Any) -> torch.Tensor:
    """Safe-RLHF AutoModelForScore normally returns .end_scores."""
    value = None

    if hasattr(outputs, "end_scores"):
        value = outputs.end_scores
    elif hasattr(outputs, "scores"):
        value = outputs.scores
    elif isinstance(outputs, dict):
        for key in ["end_scores", "scores"]:
            if key in outputs:
                value = outputs[key]
                break
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        value = outputs[0]

    if value is None:
        raise RuntimeError(f"Could not extract score from AutoModelForScore output type: {type(outputs)}")

    if not torch.is_tensor(value):
        value = torch.as_tensor(value)

    if value.ndim == 3:
        value = value[:, -1, :]

    if value.ndim == 2 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    elif value.ndim == 2 and value.shape[-1] > 1:
        value = value[:, 0]

    return value.float().view(-1)


def load_score_model(model_name: str, torch_dtype: torch.dtype, device_map: str):
    """IMPORTANT: use safe_rlhf.models.AutoModelForScore only."""
    from safe_rlhf.models import AutoModelForScore

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    if device_map != "none":
        model_kwargs["device_map"] = device_map

    try:
        model = AutoModelForScore.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        model_kwargs.pop("trust_remote_code", None)
        model = AutoModelForScore.from_pretrained(model_name, **model_kwargs)

    if device_map == "none":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = next(model.parameters()).device

    model.eval()
    return model, device


@torch.no_grad()
def score_texts(
    texts: Sequence[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    torch_dtype: torch.dtype,
    device_map: str,
    desc: str,
) -> List[float]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model, device = load_score_model(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    scores: List[float] = []

    for start in tqdm(range(0, len(texts), batch_size), desc=desc, dynamic_ncols=True):
        batch_texts = list(texts[start : start + batch_size])

        toks = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        toks = {k: v.to(device) for k, v in toks.items()}

        outputs = model(**toks)
        batch_scores = extract_scores(outputs)

        if batch_scores.numel() != len(batch_texts):
            raise RuntimeError(
                f"Score count mismatch: got {batch_scores.numel()}, expected {len(batch_texts)}"
            )

        scores.extend([float(x) for x in batch_scores.detach().cpu().tolist()])

    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def std(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def stderr(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return std(xs) / math.sqrt(len(xs))


def compute_mip(helpful: float, harmless: float, w_helpful: float) -> float:
    w_harmless = 1.0 - w_helpful
    return w_helpful * helpful + w_harmless * harmless


def summarize_rows(name: str, rows: List[Dict[str, Any]], w_helpful: float) -> Dict[str, Any]:
    w_harmless = 1.0 - w_helpful

    helpful = [float(r["helpful_score"]) for r in rows]
    cost = [float(r["cost_score"]) for r in rows]
    harmless = [float(r["harmless_score"]) for r in rows]
    mip = [compute_mip(h, s, w_helpful=w_helpful) for h, s in zip(helpful, harmless)]

    leakage_removed = [int(r.get("had_leakage_removed", 0)) for r in rows]

    return {
        "name": name,
        "n": len(rows),
        "w_helpful": w_helpful,
        "w_harmless": w_harmless,
        "leakage_removed": sum(leakage_removed),
        "leakage_rate": sum(leakage_removed) / max(1, len(rows)),
        "helpful_mean": mean(helpful),
        "helpful_std": std(helpful),
        "helpful_stderr": stderr(helpful),
        "cost_mean": mean(cost),
        "cost_std": std(cost),
        "cost_stderr": stderr(cost),
        "harmless_mean": mean(harmless),
        "harmless_std": std(harmless),
        "harmless_stderr": stderr(harmless),
        "mip_mean": mean(mip),
        "mip_std": std(mip),
        "mip_stderr": stderr(mip),
    }


def print_terminal_table(summaries: List[Dict[str, Any]], w_helpful: float) -> None:
    w_harmless = 1.0 - w_helpful
    mip_label = f"MIP@({w_helpful:.1f},{w_harmless:.1f})"

    headers = ["method", "n", "leakage", "leakage%", "helpful", "harmless", "cost", mip_label]

    rows = []
    for s in summaries:
        rows.append(
            [
                s["name"],
                str(s["n"]),
                str(s["leakage_removed"]),
                f"{100.0 * s['leakage_rate']:.2f}",
                f"{s['helpful_mean']:.4f}±{s['helpful_stderr']:.4f}",
                f"{s['harmless_mean']:.4f}±{s['harmless_stderr']:.4f}",
                f"{s['cost_mean']:.4f}±{s['cost_stderr']:.4f}",
                f"{s['mip_mean']:.4f}±{s['mip_stderr']:.4f}",
            ]
        )

    col_widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]

    def fmt_row(row):
        return " | ".join(str(row[i]).rjust(col_widths[i]) for i in range(len(row)))

    sep = "-+-".join("-" * w for w in col_widths)

    print("\n========== MORLHF / PC-MORLHF Cleaned Generation Evaluation ==========")
    print(f"[preference] helpful={w_helpful:.1f}, harmless={w_harmless:.1f}")
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print("=======================================================================\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_summary_csv(path: str, summaries: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    fieldnames = [
        "name",
        "n",
        "w_helpful",
        "w_harmless",
        "leakage_removed",
        "leakage_rate",
        "helpful_mean",
        "helpful_std",
        "helpful_stderr",
        "harmless_mean",
        "harmless_std",
        "harmless_stderr",
        "cost_mean",
        "cost_std",
        "cost_stderr",
        "mip_mean",
        "mip_std",
        "mip_stderr",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s.get(k, "") for k in fieldnames})


def write_paired_comparison(
    path: str,
    baseline_name: str,
    baseline_rows: List[Dict[str, Any]],
    candidate_name: str,
    candidate_rows: List[Dict[str, Any]],
    w_helpful: float,
) -> Dict[str, Any]:
    w_harmless = 1.0 - w_helpful
    n = min(len(baseline_rows), len(candidate_rows))

    paired = []
    helpful_diffs = []
    harmless_diffs = []
    mip_diffs = []
    same_prompt_count = 0

    for i in range(n):
        ar = baseline_rows[i]
        br = candidate_rows[i]

        same_prompt = ar.get("raw_prompt") == br.get("raw_prompt")
        if same_prompt:
            same_prompt_count += 1

        a_helpful = float(ar["helpful_score"])
        b_helpful = float(br["helpful_score"])

        a_harmless = float(ar["harmless_score"])
        b_harmless = float(br["harmless_score"])

        helpful_diff = b_helpful - a_helpful
        harmless_diff = b_harmless - a_harmless

        a_mip = compute_mip(a_helpful, a_harmless, w_helpful=w_helpful)
        b_mip = compute_mip(b_helpful, b_harmless, w_helpful=w_helpful)
        mip_diff = b_mip - a_mip

        helpful_diffs.append(helpful_diff)
        harmless_diffs.append(harmless_diff)
        mip_diffs.append(mip_diff)

        paired.append(
            {
                "index": i,
                "same_prompt": same_prompt,
                "raw_prompt": ar.get("raw_prompt"),
                f"{baseline_name}_original_generation": ar.get("original_generation"),
                f"{candidate_name}_original_generation": br.get("original_generation"),
                f"{baseline_name}_clean_generation": ar.get("clean_generation"),
                f"{candidate_name}_clean_generation": br.get("clean_generation"),
                f"{baseline_name}_helpful": a_helpful,
                f"{candidate_name}_helpful": b_helpful,
                "delta_helpful": helpful_diff,
                f"{baseline_name}_harmless": a_harmless,
                f"{candidate_name}_harmless": b_harmless,
                "delta_harmless": harmless_diff,
                f"{baseline_name}_mip": a_mip,
                f"{candidate_name}_mip": b_mip,
                "delta_mip": mip_diff,
            }
        )

    result = {
        "metadata": {
            "baseline_name": baseline_name,
            "candidate_name": candidate_name,
            "n": n,
            "w_helpful": w_helpful,
            "w_harmless": w_harmless,
            "same_prompt_count": same_prompt_count,
            "same_prompt_ratio": same_prompt_count / max(1, n),
            "delta_is_candidate_minus_baseline": True,
        },
        "summary": {
            "delta_helpful_mean": mean(helpful_diffs),
            "delta_helpful_stderr": stderr(helpful_diffs),
            "delta_harmless_mean": mean(harmless_diffs),
            "delta_harmless_stderr": stderr(harmless_diffs),
            "delta_mip_mean": mean(mip_diffs),
            "delta_mip_stderr": stderr(mip_diffs),
        },
        "data": paired,
    }

    write_json(path, result)
    return result


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dtype = get_dtype(args.torch_dtype)

    w_helpful = float(args.w)
    w_harmless = 1.0 - w_helpful

    print("=" * 100)
    print(f"[preference] helpful={w_helpful:.1f}, harmless={w_harmless:.1f}")
    print(f"[input] morlhf_json={args.morlhf_json}")
    print(f"[input] pcmorlhf_json={args.pcmorlhf_json}")
    print(f"[output] output_dir={args.output_dir}")

    gen_sets = [
        load_generation_set("morlhf", args.morlhf_json, limit=args.limit),
        load_generation_set("pcmorlhf", args.pcmorlhf_json, limit=args.limit),
    ]

    gen_sets = [clean_generation_set(gs, prompt_template=args.prompt_template) for gs in gen_sets]

    for gs in gen_sets:
        print("=" * 100)
        print(f"[load] name={gs.name}")
        print(f"[load] path={gs.path}")
        print(f"[load] n={len(gs.data)}")
        print(
            f"[clean] leakage_removed="
            f"{sum(int(r.get('had_leakage_removed', 0)) for r in gs.data)} / {len(gs.data)}"
        )

    texts_by_name: Dict[str, List[str]] = {
        gs.name: [row["score_text"] for row in gs.data]
        for gs in gen_sets
    }

    print("=" * 100)
    print(f"[score] helpful reward model: {args.reward_model_name}")
    helpful_by_name = {}
    for gs in gen_sets:
        helpful_by_name[gs.name] = score_texts(
            texts_by_name[gs.name],
            model_name=args.reward_model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            torch_dtype=dtype,
            device_map=args.device_map,
            desc=f"helpful/{gs.name}",
        )

    print("=" * 100)
    print(f"[score] cost model: {args.cost_model_name}")
    cost_by_name = {}
    for gs in gen_sets:
        cost_by_name[gs.name] = score_texts(
            texts_by_name[gs.name],
            model_name=args.cost_model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            torch_dtype=dtype,
            device_map=args.device_map,
            desc=f"cost/{gs.name}",
        )

    summaries = []
    scored_rows_by_name: Dict[str, List[Dict[str, Any]]] = {}

    for gs in gen_sets:
        helpful_scores = helpful_by_name[gs.name]
        cost_scores = cost_by_name[gs.name]

        if len(helpful_scores) != len(gs.data) or len(cost_scores) != len(gs.data):
            raise RuntimeError(f"Score length mismatch for {gs.name}")

        scored_rows = []

        for row, helpful, cost in zip(gs.data, helpful_scores, cost_scores):
            helpful = float(helpful)
            cost = float(cost)
            harmless = -cost
            mip = compute_mip(helpful, harmless, w_helpful=w_helpful)

            new_row = dict(row)
            new_row["helpful_score"] = helpful
            new_row["cost_score"] = cost
            new_row["harmless_score"] = harmless
            new_row["w_helpful"] = w_helpful
            new_row["w_harmless"] = w_harmless
            new_row["mip"] = mip

            scored_rows.append(new_row)

        scored_rows_by_name[gs.name] = scored_rows
        summaries.append(summarize_rows(gs.name, scored_rows, w_helpful=w_helpful))

        if args.save_scored_json:
            out_path = os.path.join(args.output_dir, f"{gs.name}_cleaned_scored.json")
            write_json(
                out_path,
                {
                    "metadata": {
                        "name": gs.name,
                        "source_path": gs.path,
                        "source_metadata": gs.metadata,
                        "reward_model_name": args.reward_model_name,
                        "cost_model_name": args.cost_model_name,
                        "w_helpful": w_helpful,
                        "w_harmless": w_harmless,
                        "max_length": args.max_length,
                        "n": len(scored_rows),
                    },
                    "summary": summaries[-1],
                    "data": scored_rows,
                },
            )
            print(f"[save] {out_path}")

    print_terminal_table(summaries, w_helpful=w_helpful)

    summary_json_path = os.path.join(args.output_dir, "summary_cleaned.json")
    summary_csv_path = os.path.join(args.output_dir, "summary_cleaned.csv")

    write_json(
        summary_json_path,
        {
            "metadata": {
                "w_helpful": w_helpful,
                "w_harmless": w_harmless,
                "morlhf_json": args.morlhf_json,
                "pcmorlhf_json": args.pcmorlhf_json,
                "reward_model_name": args.reward_model_name,
                "cost_model_name": args.cost_model_name,
                "max_length": args.max_length,
            },
            "summaries": summaries,
        },
    )
    write_summary_csv(summary_csv_path, summaries)

    print(f"[save] {summary_json_path}")
    print(f"[save] {summary_csv_path}")

    paired_path = os.path.join(args.output_dir, "paired_morlhf_vs_pcmorlhf_cleaned.json")
    paired_result = write_paired_comparison(
        paired_path,
        "morlhf",
        scored_rows_by_name["morlhf"],
        "pcmorlhf",
        scored_rows_by_name["pcmorlhf"],
        w_helpful=w_helpful,
    )

    print("=" * 100)
    print("[paired] baseline=morlhf, candidate=pcmorlhf")
    print(
        f"same_prompt_ratio={paired_result['metadata']['same_prompt_ratio']:.4f} "
        f"({paired_result['metadata']['same_prompt_count']}/{paired_result['metadata']['n']})"
    )
    print(
        f"delta helpful={paired_result['summary']['delta_helpful_mean']:.4f} "
        f"± {paired_result['summary']['delta_helpful_stderr']:.4f}"
    )
    print(
        f"delta harmless={paired_result['summary']['delta_harmless_mean']:.4f} "
        f"± {paired_result['summary']['delta_harmless_stderr']:.4f}"
    )
    print(
        f"delta MIP@({w_helpful:.1f},{w_harmless:.1f})="
        f"{paired_result['summary']['delta_mip_mean']:.4f} "
        f"± {paired_result['summary']['delta_mip_stderr']:.4f}"
    )
    print(f"[save] {paired_path}")


if __name__ == "__main__":
    main()
