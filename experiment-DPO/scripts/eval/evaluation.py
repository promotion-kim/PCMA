#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm


DEFAULT_REWARD_MODEL = "PKU-Alignment/beaver-7b-v1.0-reward"
DEFAULT_COST_MODEL = "PKU-Alignment/beaver-7b-v1.0-cost"
DEFAULT_PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"
DEFAULT_METHODS = ["modpo", "pcmodpo", "rs"]
DEFAULT_WEIGHTS = ["0.0", "0.3", "0.5", "0.7", "1.0"]


@dataclass
class GenerationSet:
    method: str
    weight: float
    path: str
    metadata: Dict[str, Any]
    data: List[Dict[str, Any]]


def fmt_weight(w: float) -> str:
    return f"{float(w):.1f}"


def parse_weights(values: Sequence[str]) -> List[float]:
    weights = [float(v) for v in values]
    for w in weights:
        if not (0.0 <= w <= 1.0):
            raise ValueError(f"weight must be in [0, 1], got {w}")
    return weights


def get_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected dict JSONL row at {path}:{lineno}, got {type(obj)}")
            rows.append(obj)
    return rows


def find_one(candidates: Iterable[Path], glob_patterns: Iterable[str]) -> Path:
    checked: List[Path] = []

    for path in candidates:
        checked.append(path)
        if path.exists():
            return path

    matches: List[Path] = []
    for pattern in glob_patterns:
        found = sorted(Path().glob(pattern)) if not pattern.startswith("/") else sorted(Path("/").glob(pattern[1:]))
        matches.extend([p for p in found if p.exists()])

    # De-duplicate while preserving order.
    unique_matches: List[Path] = []
    seen = set()
    for p in matches:
        rp = str(p.resolve())
        if rp not in seen:
            unique_matches.append(p)
            seen.add(rp)

    if len(unique_matches) == 1:
        return unique_matches[0]
    if len(unique_matches) > 1:
        msg = "\n".join(str(p) for p in unique_matches)
        raise FileExistsError(f"Multiple candidate files found:\n{msg}")

    checked_msg = "\n".join(str(p) for p in checked)
    glob_msg = "\n".join(glob_patterns)
    raise FileNotFoundError(f"No generation file found. Checked:\n{checked_msg}\nGlob patterns:\n{glob_msg}")


def resolve_generation_path(generation_root: Path, method: str, weight: float) -> Path:
    """
    Expected directory structure:
      outputs/generation/modpo/modpo_0.0.json
      outputs/generation/pcmodpo/pcmodpo_0.0.json
      outputs/generation/rs/rs_output_h0.0_s1.0.jsonl

    Also supports flat layout for quick tests:
      generation_root/modpo_0.0.json
      generation_root/pcmodpo_0.0.json
      generation_root/rs_output_h0.0_s1.0.jsonl
    """
    w = fmt_weight(weight)
    s = fmt_weight(1.0 - weight)
    method_dir = generation_root / method

    if method == "modpo":
        candidates = [
            method_dir / f"modpo_{w}.json",
            generation_root / f"modpo_{w}.json",
        ]
        globs = [
            str(method_dir / f"*{w}*.json"),
            str(generation_root / f"modpo*{w}*.json"),
        ]
    elif method == "pcmodpo":
        candidates = [
            method_dir / f"pcmodpo_{w}.json",
            generation_root / f"pcmodpo_{w}.json",
        ]
        globs = [
            str(method_dir / f"*{w}*.json"),
            str(generation_root / f"pcmodpo*{w}*.json"),
        ]
    elif method == "rs":
        candidates = [
            method_dir / f"rs_output_h{w}_s{s}.jsonl",
            generation_root / f"rs_output_h{w}_s{s}.jsonl",
        ]
        globs = [
            str(method_dir / f"*h{w}*s{s}*.jsonl"),
            str(generation_root / f"rs*h{w}*s{s}*.jsonl"),
        ]
    else:
        # Generic fallback for additional methods.
        candidates = [
            method_dir / f"{method}_{w}.json",
            method_dir / f"{method}_{w}.jsonl",
            generation_root / f"{method}_{w}.json",
            generation_root / f"{method}_{w}.jsonl",
        ]
        globs = [
            str(method_dir / f"*{w}*.json"),
            str(method_dir / f"*{w}*.jsonl"),
            str(generation_root / f"{method}*{w}*.json"),
            str(generation_root / f"{method}*{w}*.jsonl"),
        ]

    return find_one(candidates, globs)


def load_raw_generation_file(method: str, weight: float, path: Path, limit: int = -1) -> GenerationSet:
    metadata: Dict[str, Any] = {}

    if path.suffix == ".jsonl":
        data = read_jsonl(path)
        metadata = {"format": "jsonl"}
    elif path.suffix == ".json":
        obj = read_json(path)
        if isinstance(obj, dict):
            metadata = obj.get("metadata", {}) if isinstance(obj.get("metadata", {}), dict) else {}
            data = obj.get("data", [])
        elif isinstance(obj, list):
            metadata = {"format": "json_list"}
            data = obj
        else:
            raise ValueError(f"Invalid JSON root type for {path}: {type(obj)}")
    else:
        raise ValueError(f"Unsupported generation file extension: {path}")

    if not isinstance(data, list):
        raise ValueError(f"Invalid generation data format: {path}")

    if limit is not None and limit > 0:
        data = data[:limit]

    return GenerationSet(
        method=method,
        weight=weight,
        path=str(path),
        metadata=metadata,
        data=[dict(r) for r in data],
    )


def remove_leakage(text: Any) -> str:
    """Remove common Alpaca/instruction-format leakage after the actual answer."""
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


def first_nonempty(row: Mapping[str, Any], keys: Sequence[str], default: str = "") -> str:
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip() != "":
            return str(v)
    return default


def normalize_row(row: Dict[str, Any], prompt_template: str) -> Dict[str, Any]:
    """
    Normalize MODPO/PCMODPO JSON rows and Reward Soup JSONL rows into one schema.

    Supported input fields:
      - MODPO/PCMODPO: raw_prompt, model_input, generation
      - RS: raw_prompt, prompt, response, full_text, weight_helpful, weight_harmless
    """
    raw_prompt = first_nonempty(row, ["raw_prompt", "prompt_text", "instruction"])

    prompt = first_nonempty(row, ["model_input", "prompt"])
    if not prompt and raw_prompt:
        prompt = prompt_template.format(raw_prompt=raw_prompt)

    generation = first_nonempty(row, ["generation", "response", "output", "completion", "answer"])
    original_generation = generation.strip()
    clean_generation = remove_leakage(original_generation)

    if prompt and not prompt.endswith((" ", "\n")):
        prompt_for_score = prompt + " "
    else:
        prompt_for_score = prompt

    score_text = prompt_for_score + clean_generation

    new_row = dict(row)
    new_row.update(
        {
            "raw_prompt": raw_prompt,
            "model_input": prompt,
            "original_generation": original_generation,
            "clean_generation": clean_generation,
            "had_leakage_removed": int(original_generation != clean_generation),
            "score_text": score_text,
        }
    )
    return new_row


def clean_generation_set(gs: GenerationSet, prompt_template: str) -> GenerationSet:
    return GenerationSet(
        method=gs.method,
        weight=gs.weight,
        path=gs.path,
        metadata=gs.metadata,
        data=[normalize_row(r, prompt_template=prompt_template) for r in gs.data],
    )


def extract_scores(outputs: Any) -> torch.Tensor:
    value = None

    if hasattr(outputs, "end_scores"):
        value = outputs.end_scores
    elif hasattr(outputs, "scores"):
        value = outputs.scores
    elif isinstance(outputs, dict):
        for key in ["end_scores", "scores", "logits"]:
            if key in outputs:
                value = outputs[key]
                break
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        value = outputs[0]

    if value is None:
        raise RuntimeError(f"Could not extract score from model output type: {type(outputs)}")

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
    """Use Safe-RLHF AutoModelForScore for Beaver reward/cost models."""
    from transformers import AutoTokenizer
    from safe_rlhf.models import AutoModelForScore

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": True}
    if device_map != "none":
        kwargs["device_map"] = device_map

    try:
        model = AutoModelForScore.from_pretrained(model_name, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        model = AutoModelForScore.from_pretrained(model_name, **kwargs)

    if device_map == "none":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = next(model.parameters()).device

    model.eval()
    return tokenizer, model, device


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
    tokenizer, model, device = load_score_model(
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
        scores.extend(float(x) for x in batch_scores.detach().cpu().tolist())

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores


def score_all_groups(
    groups: Dict[Tuple[str, float], List[str]],
    model_name: str,
    batch_size: int,
    max_length: int,
    torch_dtype: torch.dtype,
    device_map: str,
    desc: str,
) -> Dict[Tuple[str, float], List[float]]:
    all_texts: List[str] = []
    spans: Dict[Tuple[str, float], Tuple[int, int]] = {}
    cursor = 0

    for key, texts in groups.items():
        spans[key] = (cursor, cursor + len(texts))
        all_texts.extend(texts)
        cursor += len(texts)

    all_scores = score_texts(
        all_texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        torch_dtype=torch_dtype,
        device_map=device_map,
        desc=desc,
    )

    out: Dict[Tuple[str, float], List[float]] = {}
    for key, (s, e) in spans.items():
        out[key] = all_scores[s:e]
    return out


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
    return float(w_helpful * helpful + (1.0 - w_helpful) * harmless)


def is_non_dominated(points: Sequence[Tuple[float, float]]) -> List[bool]:
    """Return non-dominated mask for 2D maximization points."""
    mask: List[bool] = []
    for i, (x_i, y_i) in enumerate(points):
        dominated = False
        for j, (x_j, y_j) in enumerate(points):
            if i == j:
                continue
            if (x_j >= x_i and y_j >= y_i) and (x_j > x_i or y_j > y_i):
                dominated = True
                break
        mask.append(not dominated)
    return mask


def hypervolume_2d(points: Sequence[Tuple[float, float]], reference_point: Tuple[float, float]) -> float:
    """2D maximization hypervolume over non-dominated points."""
    ref_x, ref_y = reference_point
    valid = [(x, y) for x, y in points if x > ref_x and y > ref_y]
    if not valid:
        return 0.0

    nd_mask = is_non_dominated(valid)
    frontier = [p for p, keep in zip(valid, nd_mask) if keep]
    frontier = sorted(frontier, key=lambda p: p[0], reverse=True)

    hv = 0.0
    current_y = ref_y
    for x, y in frontier:
        if y > current_y:
            hv += (x - ref_x) * (y - current_y)
            current_y = y
    return float(hv)


def default_reference_point(score_df: pd.DataFrame) -> Tuple[float, float]:
    min_helpful = float(score_df["helpful_mean"].min())
    max_helpful = float(score_df["helpful_mean"].max())
    min_harmless = float(score_df["harmless_mean"].min())
    max_harmless = float(score_df["harmless_mean"].max())

    helpful_range = max(max_helpful - min_helpful, 1e-6)
    harmless_range = max(max_harmless - min_harmless, 1e-6)
    return (min_helpful - 0.1 * helpful_range, min_harmless - 0.1 * harmless_range)


def summarize_scored_rows(method: str, weight: float, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    helpful = [float(r["helpful_score"]) for r in rows]
    cost = [float(r["cost_score"]) for r in rows]
    harmless = [float(r["harmless_score"]) for r in rows]
    mip = [float(r["mip"]) for r in rows]
    leakage = [int(r.get("had_leakage_removed", 0)) for r in rows]

    return {
        "method": method,
        "weight": fmt_weight(weight),
        "w_helpful": float(weight),
        "w_harmless": float(1.0 - weight),
        "n": len(rows),
        "leakage_removed": sum(leakage),
        "leakage_rate": sum(leakage) / max(1, len(rows)),
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


def add_scores_to_rows(
    gs: GenerationSet,
    helpful_scores: Sequence[float],
    cost_scores: Sequence[float],
) -> List[Dict[str, Any]]:
    if len(helpful_scores) != len(gs.data) or len(cost_scores) != len(gs.data):
        raise RuntimeError(f"Score length mismatch for {gs.method}@{fmt_weight(gs.weight)}")

    out: List[Dict[str, Any]] = []
    for row, helpful, cost in zip(gs.data, helpful_scores, cost_scores):
        helpful = float(helpful)
        cost = float(cost)
        harmless = -cost
        mip = compute_mip(helpful, harmless, w_helpful=gs.weight)

        new_row = dict(row)
        new_row.update(
            {
                "method": gs.method,
                "weight": fmt_weight(gs.weight),
                "w_helpful": float(gs.weight),
                "w_harmless": float(1.0 - gs.weight),
                "helpful_score": helpful,
                "cost_score": cost,
                "harmless_score": harmless,
                "mip": mip,
            }
        )
        out.append(new_row)
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_pairwise_deltas(
    scored: Dict[Tuple[str, float], List[Dict[str, Any]]],
    methods: Sequence[str],
    weights: Sequence[float],
    baseline: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if baseline not in methods:
        return pd.DataFrame(rows)

    for w in weights:
        base_key = (baseline, w)
        if base_key not in scored:
            continue
        base_rows = scored[base_key]

        for method in methods:
            if method == baseline:
                continue
            key = (method, w)
            if key not in scored:
                continue

            cand_rows = scored[key]
            n = min(len(base_rows), len(cand_rows))
            if n == 0:
                continue

            same_prompt = 0
            d_helpful: List[float] = []
            d_harmless: List[float] = []
            d_mip: List[float] = []

            for br, cr in zip(base_rows[:n], cand_rows[:n]):
                if br.get("raw_prompt") == cr.get("raw_prompt"):
                    same_prompt += 1
                d_helpful.append(float(cr["helpful_score"]) - float(br["helpful_score"]))
                d_harmless.append(float(cr["harmless_score"]) - float(br["harmless_score"]))
                d_mip.append(float(cr["mip"]) - float(br["mip"]))

            rows.append(
                {
                    "baseline": baseline,
                    "candidate": method,
                    "weight": fmt_weight(w),
                    "w_helpful": float(w),
                    "w_harmless": float(1.0 - w),
                    "n": n,
                    "same_prompt_ratio": same_prompt / max(1, n),
                    "delta_helpful_mean": mean(d_helpful),
                    "delta_helpful_stderr": stderr(d_helpful),
                    "delta_harmless_mean": mean(d_harmless),
                    "delta_harmless_stderr": stderr(d_harmless),
                    "delta_mip_mean": mean(d_mip),
                    "delta_mip_stderr": stderr(d_mip),
                }
            )

    return pd.DataFrame(rows)


def build_method_summary(score_df: pd.DataFrame, methods: Sequence[str], reference_point: Tuple[float, float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for method in methods:
        method_df = score_df[score_df["method"] == method].copy()
        if method_df.empty:
            continue
        points = list(zip(method_df["helpful_mean"].tolist(), method_df["harmless_mean"].tolist()))
        nd_mask = is_non_dominated(points)
        rows.append(
            {
                "method": method,
                "num_points": len(points),
                "num_non_dominated": int(sum(nd_mask)),
                "mean_mip_across_weights": float(method_df["mip_mean"].mean()),
                "best_mip": float(method_df["mip_mean"].max()),
                "hypervolume": hypervolume_2d(points, reference_point),
                "reference_helpful": reference_point[0],
                "reference_harmless": reference_point[1],
            }
        )
    return pd.DataFrame(rows)


def plot_pareto_frontier(score_df: pd.DataFrame, methods: Sequence[str], output_path: Path) -> None:
    plt.figure(figsize=(8, 6))

    for method in methods:
        method_df = score_df[score_df["method"] == method].copy()
        if method_df.empty:
            continue

        method_df = method_df.sort_values("w_helpful")
        points = list(zip(method_df["helpful_mean"].tolist(), method_df["harmless_mean"].tolist()))
        nd_mask = is_non_dominated(points)

        # Plot all weight-wise points.
        plt.scatter(method_df["helpful_mean"], method_df["harmless_mean"], label=f"{method} points")

        for _, row in method_df.iterrows():
            plt.annotate(
                row["weight"],
                (row["helpful_mean"], row["harmless_mean"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        # Connect only non-dominated points as the Pareto frontier.
        nd_df = method_df[nd_mask].sort_values("helpful_mean")
        if len(nd_df) >= 2:
            plt.plot(
                nd_df["helpful_mean"],
                nd_df["harmless_mean"],
                marker="o",
                linewidth=2,
                label=f"{method} Pareto frontier",
            )
        elif len(nd_df) == 1:
            plt.scatter(
                nd_df["helpful_mean"],
                nd_df["harmless_mean"],
                marker="*",
                s=120,
                label=f"{method} Pareto point",
            )

    plt.xlabel("Helpful reward")
    plt.ylabel("Harmless score = - Cost")
    plt.title("Reward / Harmlessness Pareto Frontier")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=250)
    plt.close()


def plot_mip_by_weight(score_df: pd.DataFrame, methods: Sequence[str], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for method in methods:
        method_df = score_df[score_df["method"] == method].copy().sort_values("w_helpful")
        if method_df.empty:
            continue
        plt.plot(method_df["w_helpful"], method_df["mip_mean"], marker="o", label=method)
    plt.xlabel("Helpfulness weight")
    plt.ylabel("MIP")
    plt.title("MIP by Preference Weight")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=250)
    plt.close()


def print_score_table(score_df: pd.DataFrame) -> None:
    show = score_df[
        [
            "method",
            "weight",
            "n",
            "leakage_removed",
            "leakage_rate",
            "helpful_mean",
            "harmless_mean",
            "cost_mean",
            "mip_mean",
        ]
    ].copy()
    show["leakage_rate"] = show["leakage_rate"].map(lambda x: f"{100*x:.2f}%")
    for col in ["helpful_mean", "harmless_mean", "cost_mean", "mip_mean"]:
        show[col] = show[col].map(lambda x: f"{x:.4f}")
    print("\n========== Method × Weight Summary ==========")
    print(show.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation_root",
        type=str,
        default="/home/sjkim/MOD/experiment-DPO/outputs/generation",
        help="Root directory containing modpo/, pcmodpo/, rs/ folders.",
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--weights", nargs="+", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output_dir", type=str, default="/home/sjkim/MOD/experiment-DPO/outputs/eval_generation_methods")

    parser.add_argument("--reward_model_name", type=str, default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--cost_model_name", type=str, default=DEFAULT_COST_MODEL)
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--torch_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--device_map",
        type=str,
        default="none",
        help="Use 'none' for single GPU .to(cuda), or 'auto' for HF device_map='auto'.",
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--baseline", type=str, default="modpo", help="Baseline method for delta_vs_baseline.csv")
    parser.add_argument(
        "--reference_point",
        type=float,
        nargs=2,
        default=None,
        help="Optional HV reference point: --reference_point REF_HELPFUL REF_HARMLESS",
    )
    parser.add_argument("--save_scored_json", action="store_true", default=True)
    parser.add_argument("--no_save_scored_json", dest="save_scored_json", action="store_false")
    parser.add_argument(
        "--dry_run_load_only",
        action="store_true",
        help="Only resolve/load/clean generation files without loading reward/cost models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_root = Path(args.generation_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = list(args.methods)
    weights = parse_weights(args.weights)
    dtype = get_dtype(args.torch_dtype)

    print("=" * 100)
    print(f"[generation_root] {generation_root}")
    print(f"[methods] {methods}")
    print(f"[weights] {[fmt_weight(w) for w in weights]}")
    print(f"[output_dir] {output_dir}")

    gen_sets: Dict[Tuple[str, float], GenerationSet] = {}
    for method in methods:
        for w in weights:
            path = resolve_generation_path(generation_root, method, w)
            gs = load_raw_generation_file(method, w, path, limit=args.limit)
            gs = clean_generation_set(gs, prompt_template=args.prompt_template)
            gen_sets[(method, w)] = gs
            leakage = sum(int(r.get("had_leakage_removed", 0)) for r in gs.data)
            print(f"[load] method={method:8s} w={fmt_weight(w)} n={len(gs.data):5d} leakage={leakage:5d} path={path}")

    if args.dry_run_load_only:
        print("[dry_run_load_only] File loading/cleaning succeeded. Scoring skipped.")
        return

    text_groups: Dict[Tuple[str, float], List[str]] = {
        key: [r["score_text"] for r in gs.data]
        for key, gs in gen_sets.items()
    }

    print("=" * 100)
    print(f"[score] helpful reward model: {args.reward_model_name}")
    helpful_scores = score_all_groups(
        groups=text_groups,
        model_name=args.reward_model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch_dtype=dtype,
        device_map=args.device_map,
        desc="helpful/all_methods",
    )

    print("=" * 100)
    print(f"[score] cost model: {args.cost_model_name}")
    cost_scores = score_all_groups(
        groups=text_groups,
        model_name=args.cost_model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch_dtype=dtype,
        device_map=args.device_map,
        desc="cost/all_methods",
    )

    scored_by_key: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    summary_rows: List[Dict[str, Any]] = []
    all_scored_rows: List[Dict[str, Any]] = []

    for key, gs in gen_sets.items():
        scored_rows = add_scores_to_rows(gs, helpful_scores[key], cost_scores[key])
        scored_by_key[key] = scored_rows
        all_scored_rows.extend(scored_rows)
        summary_rows.append(summarize_scored_rows(gs.method, gs.weight, scored_rows))

        if args.save_scored_json:
            out_path = output_dir / "scored" / gs.method / f"{gs.method}_{fmt_weight(gs.weight)}_cleaned_scored.json"
            write_json(
                out_path,
                {
                    "metadata": {
                        "method": gs.method,
                        "weight": fmt_weight(gs.weight),
                        "source_path": gs.path,
                        "source_metadata": gs.metadata,
                        "reward_model_name": args.reward_model_name,
                        "cost_model_name": args.cost_model_name,
                        "max_length": args.max_length,
                        "n": len(scored_rows),
                    },
                    "summary": summary_rows[-1],
                    "data": scored_rows,
                },
            )
            print(f"[save] {out_path}")

    score_df = pd.DataFrame(summary_rows).sort_values(["method", "w_helpful"])
    score_csv = output_dir / "method_weight_scores.csv"
    score_json = output_dir / "method_weight_scores.json"
    score_df.to_csv(score_csv, index=False)
    write_json(score_json, {"summaries": summary_rows})

    if args.save_scored_json:
        write_jsonl(output_dir / "all_cleaned_scored.jsonl", all_scored_rows)

    reference_point = tuple(args.reference_point) if args.reference_point is not None else default_reference_point(score_df)
    method_summary = build_method_summary(score_df, methods=methods, reference_point=reference_point)
    method_summary_csv = output_dir / "method_summary_hv_mip.csv"
    method_summary.to_csv(method_summary_csv, index=False)

    delta_df = compute_pairwise_deltas(scored_by_key, methods=methods, weights=weights, baseline=args.baseline)
    delta_csv = output_dir / f"delta_vs_{args.baseline}.csv"
    delta_df.to_csv(delta_csv, index=False)

    pareto_plot = output_dir / "pareto_frontier_all_methods.png"
    mip_plot = output_dir / "mip_by_weight.png"
    plot_pareto_frontier(score_df, methods=methods, output_path=pareto_plot)
    plot_mip_by_weight(score_df, methods=methods, output_path=mip_plot)

    print_score_table(score_df)

    print("\n========== Method Summary: HV / Mean MIP ==========")
    print(method_summary.to_string(index=False))

    if not delta_df.empty:
        print(f"\n========== Delta vs {args.baseline} ==========")
        print(delta_df.to_string(index=False))

    print("\n========== Saved files ==========")
    print(f"Method × weight scores: {score_csv}")
    print(f"Method summary:          {method_summary_csv}")
    print(f"Delta vs baseline:       {delta_csv}")
    print(f"Pareto plot:             {pareto_plot}")
    print(f"MIP plot:                {mip_plot}")
    print(f"Reference point:         helpful={reference_point[0]:.6f}, harmless={reference_point[1]:.6f}")


if __name__ == "__main__":
    main()
