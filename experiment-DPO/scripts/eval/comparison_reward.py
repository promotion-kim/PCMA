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
      .../eval_cleaned_mip/0.3/summary_cleaned.json -> (0.3, 0.7)
    """
    try:
        w_helpful = float(path.parent.name)
        return w_helpful, 1.0 - w_helpful
    except Exception as exc:
        raise ValueError(
            f"Cannot infer weight from path={path}. "
            "Expected parent directory name like 0.0, 0.3, 0.5, 0.7, 1.0."
        ) from exc


def load_summary(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "summaries" in obj:
        summaries = obj["summaries"]
        metadata = obj.get("metadata", {})
    elif isinstance(obj, list):
        summaries = obj
        metadata = {}
    else:
        raise ValueError(f"Unsupported summary format: {path}")

    if "w_helpful" in metadata and "w_harmless" in metadata:
        default_w_helpful = float(metadata["w_helpful"])
        default_w_harmless = float(metadata["w_harmless"])
    else:
        default_w_helpful, default_w_harmless = infer_weight_from_path(path)

    rows = []

    for s in summaries:
        w_helpful = float(s.get("w_helpful", default_w_helpful))
        w_harmless = float(s.get("w_harmless", default_w_harmless))

        helpful = float(s["helpful_mean"])
        harmless = float(s["harmless_mean"])

        # Recompute MIP to avoid using an old fixed-0.5 MIP stored in older files.
        mip = w_helpful * helpful + w_harmless * harmless

        method = s.get("name", s.get("method"))
        if method is None:
            raise ValueError(f"Cannot find method name in summary item from {path}: {s}")

        rows.append(
            {
                "weight": f"{w_helpful:.1f},{w_harmless:.1f}",
                "w_helpful": w_helpful,
                "w_harmless": w_harmless,
                "method": method,
                "n": int(s.get("n", 0)),
                "leakage_rate": float(s.get("leakage_rate", 0.0)),
                "leakage_percent": 100.0 * float(s.get("leakage_rate", 0.0)),
                "helpful_mean": helpful,
                "helpful_stderr": float(s.get("helpful_stderr", 0.0)),
                "harmless_mean": harmless,
                "harmless_stderr": float(s.get("harmless_stderr", 0.0)),
                "mip_mean": mip,
                "mip_stderr": float(s.get("mip_stderr", 0.0)),
            }
        )

    return rows


def collect_results(eval_root: Path, weights: List[str]) -> pd.DataFrame:
    all_rows = []

    for w in weights:
        path = eval_root / w / "summary_cleaned.json"

        if not path.exists():
            print(f"[WARN] missing file: {path}")
            continue

        print(f"[load] {path}")
        all_rows.extend(load_summary(path))

    if not all_rows:
        raise FileNotFoundError(f"No summary_cleaned.json files found under {eval_root}")

    df = pd.DataFrame(all_rows)

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
    return df


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

    Computes the area dominated by the non-dominated points relative to the reference point.
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


def build_comparison_table(df: pd.DataFrame, reference: Tuple[float, float]) -> pd.DataFrame:
    rows = []

    for method, g in df.groupby("method"):
        g = g.sort_values("w_helpful")
        points = list(zip(g["helpful_mean"], g["harmless_mean"]))

        hv = hypervolume_2d(points, reference=reference)
        frontier = pareto_nondominated(points)

        rows.append(
            {
                "method": method,
                "num_points": len(points),
                "num_pareto_points": len(frontier),
                "avg_mip": float(g["mip_mean"].mean()),
                "avg_helpful": float(g["helpful_mean"].mean()),
                "avg_harmless": float(g["harmless_mean"].mean()),
                "avg_leakage_percent": float(g["leakage_percent"].mean()),
                "hypervolume": hv,
            }
        )

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values("hypervolume", ascending=False).reset_index(drop=True)
    return comparison


def plot_pareto(df: pd.DataFrame, output_path: Path, reference: Tuple[float, float]) -> None:
    plt.figure(figsize=(8, 6))

    for method, g in df.groupby("method"):
        g = g.sort_values("w_helpful")

        # All evaluated points
        points = list(zip(g["helpful_mean"], g["harmless_mean"]))

        x_all = g["helpful_mean"].tolist()
        y_all = g["harmless_mean"].tolist()

        # First, draw all evaluated solutions as points
        plt.scatter(
            x_all,
            y_all,
            marker="o",
            s=35,
            alpha=0.45,
        )

        # Then, connect only Pareto non-dominated points
        frontier = pareto_nondominated(points)

        if frontier:
            x_frontier = [p[0] for p in frontier]
            y_frontier = [p[1] for p in frontier]

            plt.plot(
                x_frontier,
                y_frontier,
                marker="o",
                linewidth=2,
                markersize=7,
                label=method,
            )

        # Annotate all evaluated weight points
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

    plt.xlabel("Helpfulness score ↑")
    plt.ylabel("Harmlessness score ↑")
    plt.title("Pareto Frontier: Helpfulness vs. Harmlessness")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[save] {output_path}")


def print_tables(df: pd.DataFrame, comparison: pd.DataFrame, reference: Tuple[float, float]) -> None:
    print("\n========== Weight-wise scores ==========")

    df_print = df.copy()

    display_cols = [
        "weight",
        "method",
        "leakage_percent",
        "helpful_mean",
        "harmless_mean",
        "mip_mean",
    ]

    # Sort first, then select display columns.
    df_print = df_print.sort_values(["w_helpful", "method"])

    print(
        df_print[display_cols]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n========== Method comparison ==========")
    print(f"HV reference point: helpful={reference[0]:.4f}, harmless={reference[1]:.4f}")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_root",
        type=str,
        default="/home/sjkim/MOD/experiment-DPO/outputs/eval_cleaned_mip",
        help="Root directory containing weight subdirectories, e.g. 0.0/summary_cleaned.json",
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
        default="/home/sjkim/MOD/experiment-DPO/outputs/eval_cleaned_mip/pareto_analysis",
        help="Directory to save plot and tables.",
    )

    parser.add_argument(
        "--reference_helpful",
        type=float,
        default=None,
        help="Optional fixed HV reference helpful score.",
    )

    parser.add_argument(
        "--reference_harmless",
        type=float,
        default=None,
        help="Optional fixed HV reference harmless score.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    eval_root = Path(args.eval_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_results(eval_root, args.weights)

    if args.reference_helpful is not None and args.reference_harmless is not None:
        reference = (float(args.reference_helpful), float(args.reference_harmless))
    else:
        reference = compute_reference_point(df)

    comparison = build_comparison_table(df, reference=reference)

    scores_csv = output_dir / "weightwise_scores.csv"
    comparison_csv = output_dir / "method_comparison_hv_mip.csv"
    plot_path = output_dir / "pareto_frontier_helpful_harmless.png"

    df.to_csv(scores_csv, index=False)
    comparison.to_csv(comparison_csv, index=False)

    print(f"[save] {scores_csv}")
    print(f"[save] {comparison_csv}")

    plot_pareto(df, output_path=plot_path, reference=reference)
    print_tables(df, comparison, reference=reference)


if __name__ == "__main__":
    main()
