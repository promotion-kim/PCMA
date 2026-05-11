#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_jsons",
        type=str,
        required=True,
        help="Comma-separated generation json paths. Example: modpo.json,pcmodpo.json",
    )
    parser.add_argument(
        "--names",
        type=str,
        required=True,
        help="Comma-separated names matching input_jsons. Example: modpo,pcmodpo",
    )

    parser.add_argument("--output_dir", type=str, required=True)

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

    return parser.parse_args()


def get_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def split_csv_arg(x: str) -> List[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


def load_generation_set(name: str, path: str, limit: int = -1) -> GenerationSet:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    metadata = obj.get("metadata", {})
    data = obj.get("data", obj if isinstance(obj, list) else [])

    if not isinstance(data, list):
        raise ValueError(f"Invalid generation JSON format: {path}")

    if limit is not None and limit > 0:
        data = data[:limit]

    return GenerationSet(name=name, path=path, metadata=metadata, data=data)


def build_score_input(row: Dict[str, Any], prompt_template: str) -> str:
    generation = str(row.get("generation", "")).strip()

    if "model_input" in row and row["model_input"] is not None:
        model_input = str(row["model_input"])
    elif "raw_prompt" in row and row["raw_prompt"] is not None:
        model_input = prompt_template.format(raw_prompt=str(row["raw_prompt"]))
    else:
        model_input = ""

    # Most prompts end with "ASSISTANT:"; add a space before the generated answer.
    if model_input and not model_input.endswith((" ", "\n")):
        model_input = model_input + " "

    return model_input + generation


def extract_scores(outputs: Any) -> torch.Tensor:
    """
    Supports Safe-RLHF AutoModelForScore and common HF sequence-classification outputs.
    Expected output shape: (B,)
    """
    value = None

    if hasattr(outputs, "end_scores"):
        value = outputs.end_scores
    elif hasattr(outputs, "scores"):
        value = outputs.scores
    elif hasattr(outputs, "logits"):
        value = outputs.logits
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

    # Typical shapes: (B, 1), (B,), or sometimes (B, T, 1).
    if value.ndim == 3:
        value = value[:, -1, :]
    if value.ndim == 2 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    elif value.ndim == 2 and value.shape[-1] > 1:
        # Fallback for unexpected multi-logit heads.
        value = value[:, 0]

    return value.float().view(-1)


def load_score_model(model_name: str, torch_dtype: torch.dtype, device_map: str):
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    if device_map != "none":
        model_kwargs["device_map"] = device_map

    from safe_rlhf.models import AutoModelForScore

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

    # Score models generally work fine with right padding; end_scores should use attention_mask.
    tokenizer.padding_side = "right"

    model, device = load_score_model(model_name, torch_dtype=torch_dtype, device_map=device_map)

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


def summarize_rows(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    helpful = [float(r["helpful_score"]) for r in rows]
    cost = [float(r["cost_score"]) for r in rows]
    harmless = [float(r["harmless_score"]) for r in rows]
    mip_05 = [0.5 * h + 0.5 * s for h, s in zip(helpful, harmless)]

    return {
        "name": name,
        "n": len(rows),
        "helpful_mean": mean(helpful),
        "helpful_std": std(helpful),
        "helpful_stderr": stderr(helpful),
        "cost_mean": mean(cost),
        "cost_std": std(cost),
        "cost_stderr": stderr(cost),
        "harmless_mean": mean(harmless),
        "harmless_std": std(harmless),
        "harmless_stderr": stderr(harmless),
        "mip_0p5_mean": mean(mip_05),
        "mip_0p5_std": std(mip_05),
        "mip_0p5_stderr": stderr(mip_05),
    }


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_summary_csv(path: str, summaries: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    fieldnames = [
        "name",
        "n",
        "helpful_mean",
        "helpful_std",
        "helpful_stderr",
        "harmless_mean",
        "harmless_std",
        "harmless_stderr",
        "cost_mean",
        "cost_std",
        "cost_stderr",
        "mip_0p5_mean",
        "mip_0p5_std",
        "mip_0p5_stderr",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s.get(k, "") for k in fieldnames})


def write_paired_comparison(
    path: str,
    a_name: str,
    a_rows: List[Dict[str, Any]],
    b_name: str,
    b_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    n = min(len(a_rows), len(b_rows))
    paired = []

    helpful_diffs = []
    harmless_diffs = []
    mip_diffs = []

    same_prompt_count = 0

    for i in range(n):
        ar = a_rows[i]
        br = b_rows[i]

        same_prompt = ar.get("raw_prompt") == br.get("raw_prompt")
        if same_prompt:
            same_prompt_count += 1

        a_helpful = float(ar["helpful_score"])
        b_helpful = float(br["helpful_score"])
        a_harmless = float(ar["harmless_score"])
        b_harmless = float(br["harmless_score"])

        helpful_diff = b_helpful - a_helpful
        harmless_diff = b_harmless - a_harmless

        a_mip = 0.5 * a_helpful + 0.5 * a_harmless
        b_mip = 0.5 * b_helpful + 0.5 * b_harmless
        mip_diff = b_mip - a_mip

        helpful_diffs.append(helpful_diff)
        harmless_diffs.append(harmless_diff)
        mip_diffs.append(mip_diff)

        paired.append(
            {
                "index": i,
                "same_prompt": same_prompt,
                "raw_prompt": ar.get("raw_prompt"),
                f"{a_name}_generation": ar.get("generation"),
                f"{b_name}_generation": br.get("generation"),
                f"{a_name}_helpful": a_helpful,
                f"{b_name}_helpful": b_helpful,
                "delta_helpful": helpful_diff,
                f"{a_name}_harmless": a_harmless,
                f"{b_name}_harmless": b_harmless,
                "delta_harmless": harmless_diff,
                f"{a_name}_mip_0p5": a_mip,
                f"{b_name}_mip_0p5": b_mip,
                "delta_mip_0p5": mip_diff,
            }
        )

    result = {
        "metadata": {
            "baseline_name": a_name,
            "candidate_name": b_name,
            "n": n,
            "same_prompt_count": same_prompt_count,
            "same_prompt_ratio": same_prompt_count / max(1, n),
            "delta_is_candidate_minus_baseline": True,
        },
        "summary": {
            "delta_helpful_mean": mean(helpful_diffs),
            "delta_helpful_stderr": stderr(helpful_diffs),
            "delta_harmless_mean": mean(harmless_diffs),
            "delta_harmless_stderr": stderr(harmless_diffs),
            "delta_mip_0p5_mean": mean(mip_diffs),
            "delta_mip_0p5_stderr": stderr(mip_diffs),
        },
        "data": paired,
    }

    write_json(path, result)
    return result


def main() -> None:
    args = parse_args()

    input_jsons = split_csv_arg(args.input_jsons)
    names = split_csv_arg(args.names)

    if len(input_jsons) != len(names):
        raise ValueError(
            f"--input_jsons and --names must have same length. "
            f"Got {len(input_jsons)} input_jsons and {len(names)} names."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = get_dtype(args.torch_dtype)

    gen_sets = [
        load_generation_set(name=name, path=path, limit=args.limit)
        for name, path in zip(names, input_jsons)
    ]

    for gs in gen_sets:
        print("=" * 100)
        print(f"[load] name={gs.name}")
        print(f"[load] path={gs.path}")
        print(f"[load] n={len(gs.data)}")

    # Build all texts.
    texts_by_name: Dict[str, List[str]] = {}
    for gs in gen_sets:
        texts_by_name[gs.name] = [
            build_score_input(row, prompt_template=args.prompt_template)
            for row in gs.data
        ]

    # Score helpful reward.
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

    # Score cost.
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
            new_row = dict(row)
            new_row["helpful_score"] = float(helpful)
            new_row["cost_score"] = float(cost)
            new_row["harmless_score"] = float(-cost)
            new_row["mip_0p5"] = 0.5 * float(helpful) + 0.5 * float(-cost)
            scored_rows.append(new_row)

        scored_rows_by_name[gs.name] = scored_rows
        summaries.append(summarize_rows(gs.name, scored_rows))

        if args.save_scored_json:
            out_path = os.path.join(args.output_dir, f"{gs.name}_scored.json")
            write_json(
                out_path,
                {
                    "metadata": {
                        "name": gs.name,
                        "source_path": gs.path,
                        "source_metadata": gs.metadata,
                        "reward_model_name": args.reward_model_name,
                        "cost_model_name": args.cost_model_name,
                        "max_length": args.max_length,
                        "n": len(scored_rows),
                    },
                    "summary": summaries[-1],
                    "data": scored_rows,
                },
            )
            print(f"[save] {out_path}")

    summary_json_path = os.path.join(args.output_dir, "summary.json")
    summary_csv_path = os.path.join(args.output_dir, "summary.csv")

    write_json(summary_json_path, {"summaries": summaries})
    write_summary_csv(summary_csv_path, summaries)

    print("=" * 100)
    print("[summary]")
    for s in summaries:
        print(
            f"{s['name']:>12s} | "
            f"n={s['n']} | "
            f"helpful={s['helpful_mean']:.4f} ± {s['helpful_stderr']:.4f} | "
            f"harmless={s['harmless_mean']:.4f} ± {s['harmless_stderr']:.4f} | "
            f"cost={s['cost_mean']:.4f} | "
            f"MIP@0.5={s['mip_0p5_mean']:.4f} ± {s['mip_0p5_stderr']:.4f}"
        )

    print(f"[save] {summary_json_path}")
    print(f"[save] {summary_csv_path}")

    # If exactly two models, write paired comparison.
    if len(gen_sets) == 2:
        a = gen_sets[0]
        b = gen_sets[1]
        paired_path = os.path.join(args.output_dir, f"paired_{a.name}_vs_{b.name}.json")
        paired_result = write_paired_comparison(
            paired_path,
            a.name,
            scored_rows_by_name[a.name],
            b.name,
            scored_rows_by_name[b.name],
        )

        print("=" * 100)
        print(f"[paired] baseline={a.name}, candidate={b.name}")
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
            f"delta MIP@0.5={paired_result['summary']['delta_mip_0p5_mean']:.4f} "
            f"± {paired_result['summary']['delta_mip_0p5_stderr']:.4f}"
        )
        print(f"[save] {paired_path}")


if __name__ == "__main__":
    main()