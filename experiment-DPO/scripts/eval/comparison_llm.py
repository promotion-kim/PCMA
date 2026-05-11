#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_WEIGHTS = ["0.0", "0.3", "0.5", "0.7", "1.0"]


def infer_weight_from_path(path: Path) -> Tuple[float, float]:
    """
    Infer helpful/harmless weights from parent directory name.
    Example:
      .../llm_judge_eval/0.3/judge_summary.json -> (0.3, 0.7)
    """
    try:
        w_helpful = float(path.parent.name)
        return w_helpful, 1.0 - w_helpful
    except Exception as exc:
        raise ValueError(
            f"Cannot infer weight from path={path}. "
            "Expected parent directory name like 0.0, 0.3, 0.5, 0.7, 1.0."
        ) from exc


def load_judge_summary(path: Path) -> Tuple[List[Dict], Dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported judge summary format: {path}")

    metadata = obj.get("metadata", {})
    method_summaries = obj.get("method_summaries", [])
    pairwise_summary = obj.get("pairwise_summary", {})

    if not method_summaries:
        raise ValueError(f"No method_summaries found in {path}")

    if "w_helpful" in metadata and "w_harmless" in metadata:
        default_w_helpful = float(metadata["w_helpful"])
        default_w_harmless = float(metadata["w_harmless"])
    else:
        default_w_helpful, default_w_harmless = infer_weight_from_path(path)

    rows = []

    for s in method_summaries:
        method = s.get("method")
        if method is None:
            raise ValueError(f"Cannot find method name in {path}: {s}")

        w_helpful = default_w_helpful
        w_harmless = default_w_harmless

        helpful = float(s["helpfulness_mean"])
        harmless = float(s["harmlessness_mean"])

        # Recompute MIP for consistency.
        mip = w_helpful * helpful + w_harmless * harmless

        rows.append(
            {
                "weight": f"{w_helpful:.1f},{w_harmless:.1f}",
                "w_helpful": w_helpful,
                "w_harmless": w_harmless,
                "method": method,
                "n": int(s.get("n", metadata.get("valid_n", 0))),
                "helpful_mean": helpful,
                "helpful_stderr": float(s.get("helpfulness_stderr", 0.0)),
                "harmless_mean": harmless,
                "harmless_stderr": float(s.get("harmlessness_stderr", 0.0)),
                "mip_mean": mip,
                "mip_stderr": float(s.get("mip_stderr", 0.0)),
            }
        )

    pairwise_row = {
        "weight": f"{default_w_helpful:.1f},{default_w_harmless:.1f}",
        "w_helpful": default_w_helpful,
        "w_harmless": default_w_harmless,
        "n": int(pairwise_summary.get("n", metadata.get("valid_n", 0))),
        "helpfulness_modpo_win_rate": float(pairwise_summary.get("helpfulness_modpo_win_rate", 0.0)),
        "helpfulness_pcmodpo_win_rate": float(pairwise_summary.get("helpfulness_pcmodpo_win_rate", 0.0)),
        "helpfulness_tie_rate": float(pairwise_summary.get("helpfulness_tie_rate", 0.0)),
        "harmlessness_modpo_win_rate": float(pairwise_summary.get("harmlessness_modpo_win_rate", 0.0)),
        "harmlessness_pcmodpo_win_rate": float(pairwise_summary.get("harmlessness_pcmodpo_win_rate", 0.0)),
        "harmlessness_tie_rate": float(pairwise_summary.get("harmlessness_tie_rate", 0.0)),
        "overall_modpo_win_rate": float(pairwise_summary.get("overall_modpo_win_rate", 0.0)),
        "overall_pcmodpo_win_rate": float(pairwise_summary.get("overall_pcmodpo_win_rate", 0.0)),
        "overall_tie_rate": float(pairwise_summary.get("overall_tie_rate", 0.0)),
    }

    return rows, pairwise_row


def collect_results(eval_root: Path, weights: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    pairwise_rows = []

    for w in weights:
        path = eval_root / w / "judge_summary.json"

        if not path.exists():
            print(f"[WARN] missing file: {path}")
            continue

        print(f"[load] {path}")
        rows, pairwise = load_judge_summary(path)
        all_rows.extend(rows)
        pairwise_rows.append(pairwise)

    if not all_rows:
        raise FileNotFoundError(f"No judge_summary.json files found under {eval_root}")

    df = pd.DataFrame(all_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)

    required = [
        "weight",
        "w_helpful",
        "w_harmless",
        "method",
        "helpful_mean",
        "harmless_mean",
        "mip_mean",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after loading: {missing}")

    df = df.sort_values(["method", "w_helpful"]).reset_index(drop=True)
    pairwise_df = pairwise_df.sort_values("w_helpful").reset_index(drop=True)

    return df, pairwise_df


def pareto_nondominated(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Maximization Pareto frontier for 2D points.
    x = helpfulness, y = harmlessness.
    Larger is better for both.
    """
    nondominated = []

    for i, (x_i, y_i) in enumerate(points):
        dominated = False

        for j, (x_j, y_j) in enumerate(points):
            if i == j:
                continue

            if (x_j >= x_i and y_j >= y_i) and (x_j > x_i or y_j > y_i):
                dominated = True
                break

        if not dominated:
            nondominated.append((x_i, y_i))

    return sorted(set(nondominated), key=lambda p: p[0])


def hypervolume_2d(
    points: List[Tuple[float, float]],
    reference: Tuple[float, float],
) -> float:
    """
    2D hypervolume for maximization.

    Computes area dominated by non-dominated points relative to the reference point.
    The reference point should be worse than all plotted points.
    """
    ref_x, ref_y = reference
    frontier = pareto_nondominated(points)

    if not frontier:
        return 0.0

    frontier = sorted(frontier, key=lambda p: p[0])

    hv = 0.0
    prev_x = ref_x

    for x, y in frontier:
        width = max(0.0, x - prev_x)
        height = max(0.0, y - ref_y)
        hv += width * height
        prev_x = max(prev_x, x)

    return float(hv)


def compute_reference_point(df: pd.DataFrame, margin_ratio: float = 0.05) -> Tuple[float, float]:
    min_x = float(df["helpful_mean"].min())
    max_x = float(df["helpful_mean"].max())
    min_y = float(df["harmless_mean"].min())
    max_y = float(df["harmless_mean"].max())

    x_range = max(max_x - min_x, 1.0)
    y_range = max(max_y - min_y, 1.0)

    ref_x = min_x - margin_ratio * x_range
    ref_y = min_y - margin_ratio * y_range

    return float(ref_x), float(ref_y)


def build_method_comparison_table(
    df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    reference: Tuple[float, float],
) -> pd.DataFrame:
    rows = []

    avg_pairwise = {}
    if not pairwise_df.empty:
        avg_pairwise = {
            "modpo": {
                "avg_helpfulness_win_rate": float(pairwise_df["helpfulness_modpo_win_rate"].mean()),
                "avg_harmlessness_win_rate": float(pairwise_df["harmlessness_modpo_win_rate"].mean()),
                "avg_overall_win_rate": float(pairwise_df["overall_modpo_win_rate"].mean()),
            },
            "pcmodpo": {
                "avg_helpfulness_win_rate": float(pairwise_df["helpfulness_pcmodpo_win_rate"].mean()),
                "avg_harmlessness_win_rate": float(pairwise_df["harmlessness_pcmodpo_win_rate"].mean()),
                "avg_overall_win_rate": float(pairwise_df["overall_pcmodpo_win_rate"].mean()),
            },
        }

    for method, g in df.groupby("method"):
        g = g.sort_values("w_helpful")
        points = list(zip(g["helpful_mean"], g["harmless_mean"]))

        hv = hypervolume_2d(points, reference=reference)
        frontier = pareto_nondominated(points)

        row = {
            "method": method,
            "num_points": len(points),
            "num_pareto_points": len(frontier),
            "avg_mip": float(g["mip_mean"].mean()),
            "avg_helpful": float(g["helpful_mean"].mean()),
            "avg_harmless": float(g["harmless_mean"].mean()),
            "hypervolume": hv,
        }

        if method in avg_pairwise:
            row.update(avg_pairwise[method])

        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values("hypervolume", ascending=False).reset_index(drop=True)
    return comparison


def plot_pareto(df: pd.DataFrame, output_path: Path, reference: Tuple[float, float]) -> None:
    plt.figure(figsize=(8, 6))

    for method, g in df.groupby("method"):
        g = g.sort_values("w_helpful")

        x = g["helpful_mean"].tolist()
        y = g["harmless_mean"].tolist()

        plt.plot(x, y, marker="o", linewidth=2, label=method)

        for _, row in g.iterrows():
            label = f"({row['w_helpful']:.1f},{row['w_harmless']:.1f})"
            plt.annotate(
                label,
                (row["helpful_mean"], row["harmless_mean"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    ref_x, ref_y = reference
    plt.scatter([ref_x], [ref_y], marker="x", label="HV reference")

    plt.xlabel("Judge Helpfulness score ↑")
    plt.ylabel("Judge Harmlessness score ↑")
    plt.title("LLM-Judge Pareto Frontier: Helpfulness vs. Harmlessness")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[save] {output_path}")


def print_tables(
    df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    comparison: pd.DataFrame,
    reference: Tuple[float, float],
) -> None:
    print("\n========== LLM-Judge Weight-wise scores ==========")

    display_cols = [
        "weight",
        "method",
        "helpful_mean",
        "harmless_mean",
        "mip_mean",
    ]

    df_print = df.sort_values(["w_helpful", "method"])

    print(
        df_print[display_cols]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n========== LLM-Judge Pairwise win rates ==========")

    pairwise_display_cols = [
        "weight",
        "helpfulness_modpo_win_rate",
        "helpfulness_pcmodpo_win_rate",
        "harmlessness_modpo_win_rate",
        "harmlessness_pcmodpo_win_rate",
        "overall_modpo_win_rate",
        "overall_pcmodpo_win_rate",
    ]

    if not pairwise_df.empty:
        print(
            pairwise_df[pairwise_display_cols]
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    print("\n========== LLM-Judge Method comparison ==========")
    print(f"HV reference point: helpful={reference[0]:.4f}, harmless={reference[1]:.4f}")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_root",
        type=str,
        default="/home/sjkim/MOD/experiment-DPO/outputs/llm_judge_eval",
        help="Root directory containing weight subdirectories, e.g. 0.0/judge_summary.json",
    )

    parser.add_argument(
        "--weights",
        type=str,
        nargs="+",
        default=DEFAULT_WEIGHTS,
        help="Weight directory names to load.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sjkim/MOD/experiment-DPO/outputs/llm_judge_eval/pareto_analysis",
        help="Directory to save plot and tables.",
    )

    parser.add_argument(
        "--reference_helpful",
        type=float,
        default=None,
        help="Optional fixed HV reference helpfulness score.",
    )

    parser.add_argument(
        "--reference_harmless",
        type=float,
        default=None,
        help="Optional fixed HV reference harmlessness score.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    eval_root = Path(args.eval_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, pairwise_df = collect_results(eval_root, args.weights)

    if args.reference_helpful is not None and args.reference_harmless is not None:
        reference = (float(args.reference_helpful), float(args.reference_harmless))
    else:
        reference = compute_reference_point(df)

    comparison = build_method_comparison_table(
        df=df,
        pairwise_df=pairwise_df,
        reference=reference,
    )

    scores_csv = output_dir / "llm_judge_weightwise_scores.csv"
    pairwise_csv = output_dir / "llm_judge_pairwise_winrates.csv"
    comparison_csv = output_dir / "llm_judge_method_comparison_hv_mip.csv"
    plot_path = output_dir / "llm_judge_pareto_frontier_helpful_harmless.png"

    df.to_csv(scores_csv, index=False)
    pairwise_df.to_csv(pairwise_csv, index=False)
    comparison.to_csv(comparison_csv, index=False)

    print(f"[save] {scores_csv}")
    print(f"[save] {pairwise_csv}")
    print(f"[save] {comparison_csv}")

    plot_pareto(df, output_path=plot_path, reference=reference)
    print_tables(df, pairwise_df, comparison, reference=reference)


if __name__ == "__main__":
    main()
