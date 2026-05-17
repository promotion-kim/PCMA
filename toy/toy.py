#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Toy experiment:
Compare standard BT without uncertainty vs objective-specific uncertainty-aware BT.

Model cases:
  1. No uncertainty BT:
       p_hat = sigmoid(gap)
  2. Objective-specific uncertainty-aware BT:
       p_hat = sigmoid(exp(b_i) * gap)
     where b_i is learned by MAP.

Data cases:
  1. BT with objective-specific uncertainty:
       p_star = sigmoid(exp(b_i^*) * gap)
  2. Label flip noise:
       p_star = (1 - eps_i) * sigmoid(beta * gap) + eps_i * 0.5

Metrics:
  - Test NLL against sampled labels
  - Probability MSE/MAE against ground-truth p_star
  - Precision MAE against true b_i^* when available
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def nll_from_prob(p_hat: np.ndarray, z: np.ndarray, eps: float = 1e-8) -> float:
    p_hat = np.clip(p_hat, eps, 1.0 - eps)
    return float(-np.mean(z * np.log(p_hat) + (1.0 - z) * np.log(1.0 - p_hat)))


def probability_errors(p_hat: np.ndarray, p_star: np.ndarray) -> Dict[str, float]:
    return {
        "prob_mse": float(np.mean((p_hat - p_star) ** 2)),
        "prob_mae": float(np.mean(np.abs(p_hat - p_star))),
    }


def make_gap_distribution(
    rng: np.random.Generator,
    n: int,
    mixture: bool = True,
) -> np.ndarray:
    """
    Generate scorer gaps.

    mixture=True creates both clear and ambiguous pairs:
      - 70% clear-ish gaps from N(0, 1)
      - 30% ambiguous gaps from N(0, 0.2^2)
    """
    if not mixture:
        return rng.normal(loc=0.0, scale=1.0, size=n)

    is_ambiguous = rng.binomial(1, 0.3, size=n).astype(bool)
    gaps = rng.normal(loc=0.0, scale=1.0, size=n)
    gaps[is_ambiguous] = rng.normal(loc=0.0, scale=0.2, size=is_ambiguous.sum())
    return gaps


@dataclass
class ToyConfig:
    num_objectives: int = 2
    n_train_per_objective: int = 1000
    n_test_per_objective: int = 5000

    # True log precisions for BT-with-objective-uncertainty DGP.
    # Objective 0: clean/reliable. Objective 1: noisy/unreliable.
    b_true: Tuple[float, float] = (1.0, -1.0)

    # Label-flip noise rates for label-flip DGP.
    # Objective 0: low noise. Objective 1: high noise.
    flip_eps: Tuple[float, float] = (0.05, 0.35)

    # Base beta for label-flip DGP.
    label_flip_beta: float = 1.5

    # MAP prior scale for learned b_i.
    prior_sigma_b: float = 2.0

    # Optimization.
    lr: float = 5e-2
    steps: int = 2000

    seed: int = 0


# -----------------------------
# Data generation
# -----------------------------

def generate_dataset(
    dgp: str,
    n_per_objective: int,
    cfg: ToyConfig,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Returns:
      objective_ids: shape [N]
      gaps: shape [N]
      p_star: shape [N], ground-truth preference probability
      z: shape [N], sampled binary label
      b_true_per_obj: optional true log precision, available for bt_objective_uncertainty
    """
    rng = np.random.default_rng(seed)

    obj_ids_list = []
    gaps_list = []
    p_star_list = []

    for i in range(cfg.num_objectives):
        gaps_i = make_gap_distribution(rng, n_per_objective, mixture=True)

        if dgp == "bt_objective_uncertainty":
            b_i_star = cfg.b_true[i]
            kappa_i_star = math.exp(b_i_star)
            p_i_star = sigmoid_np(kappa_i_star * gaps_i)

        elif dgp == "label_flip_noise":
            eps_i = cfg.flip_eps[i]
            p_clean = sigmoid_np(cfg.label_flip_beta * gaps_i)
            p_i_star = (1.0 - eps_i) * p_clean + eps_i * 0.5

        else:
            raise ValueError(f"Unknown dgp: {dgp}")

        obj_ids_list.append(np.full(n_per_objective, i, dtype=np.int64))
        gaps_list.append(gaps_i.astype(np.float64))
        p_star_list.append(p_i_star.astype(np.float64))

    objective_ids = np.concatenate(obj_ids_list, axis=0)
    gaps = np.concatenate(gaps_list, axis=0)
    p_star = np.concatenate(p_star_list, axis=0)

    z = rng.binomial(1, p_star).astype(np.float64)

    # Shuffle all objectives together.
    perm = rng.permutation(len(gaps))
    objective_ids = objective_ids[perm]
    gaps = gaps[perm]
    p_star = p_star[perm]
    z = z[perm]

    return {
        "objective_ids": objective_ids,
        "gaps": gaps,
        "p_star": p_star,
        "z": z,
    }


# -----------------------------
# Models
# -----------------------------

def predict_no_uncertainty_bt(gaps: np.ndarray) -> np.ndarray:
    """
    Standard BT with fixed beta = 1:
      p_hat = sigmoid(gap)
    """
    return sigmoid_np(gaps)


def fit_objective_uncertainty_bt_map(
    train: Dict[str, np.ndarray],
    cfg: ToyConfig,
    verbose: bool = False,
) -> np.ndarray:
    """
    Learn objective-specific log precision b_i by MAP:

      p_hat = sigmoid(exp(b_i) * gap)

    Gaussian prior:
      b_i ~ N(0, sigma_b^2)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obj = torch.tensor(train["objective_ids"], dtype=torch.long, device=device)
    gaps = torch.tensor(train["gaps"], dtype=torch.float32, device=device)
    z = torch.tensor(train["z"], dtype=torch.float32, device=device)

    b = torch.zeros(cfg.num_objectives, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([b], lr=cfg.lr)

    n = float(len(gaps))

    for step in range(cfg.steps):
        logits = torch.exp(b[obj]) * gaps
        nll = F.binary_cross_entropy_with_logits(logits, z, reduction="mean")

        # Since nll is averaged over data, this matches:
        # (1 / (2 N sigma_b^2)) * sum_i b_i^2
        prior_penalty = torch.sum(b ** 2) / (2.0 * n * (cfg.prior_sigma_b ** 2))

        loss = nll + prior_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (step % 500 == 0 or step == cfg.steps - 1):
            print(
                f"[step {step:04d}] loss={loss.item():.6f}, "
                f"nll={nll.item():.6f}, b={b.detach().cpu().numpy()}"
            )

    return b.detach().cpu().numpy()


def predict_objective_uncertainty_bt(
    gaps: np.ndarray,
    objective_ids: np.ndarray,
    b_hat: np.ndarray,
) -> np.ndarray:
    logits = np.exp(b_hat[objective_ids]) * gaps
    return sigmoid_np(logits)


# Optional fairer baseline: shared/global uncertainty.
def fit_global_uncertainty_bt_map(
    train: Dict[str, np.ndarray],
    cfg: ToyConfig,
) -> float:
    """
    Learn a single shared log precision b:

      p_hat = sigmoid(exp(b) * gap)

    This is useful as a stronger baseline than fixed beta=1.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gaps = torch.tensor(train["gaps"], dtype=torch.float32, device=device)
    z = torch.tensor(train["z"], dtype=torch.float32, device=device)

    b = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([b], lr=cfg.lr)

    n = float(len(gaps))

    for _ in range(cfg.steps):
        logits = torch.exp(b) * gaps
        nll = F.binary_cross_entropy_with_logits(logits, z, reduction="mean")
        prior_penalty = (b ** 2) / (2.0 * n * (cfg.prior_sigma_b ** 2))
        loss = nll + prior_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(b.detach().cpu().item())


def predict_global_uncertainty_bt(gaps: np.ndarray, b_global: float) -> np.ndarray:
    return sigmoid_np(np.exp(b_global) * gaps)


# -----------------------------
# Evaluation
# -----------------------------

def evaluate_predictions(
    model_name: str,
    dgp: str,
    p_hat: np.ndarray,
    test: Dict[str, np.ndarray],
    b_hat: Optional[np.ndarray] = None,
    cfg: Optional[ToyConfig] = None,
) -> Dict[str, float | str]:
    z = test["z"]
    p_star = test["p_star"]

    row: Dict[str, float | str] = {
        "dgp": dgp,
        "model": model_name,
        "test_nll": nll_from_prob(p_hat, z),
    }

    row.update(probability_errors(p_hat, p_star))

    if dgp == "bt_objective_uncertainty" and b_hat is not None and cfg is not None:
        b_true = np.asarray(cfg.b_true, dtype=np.float64)
        row["precision_mae"] = float(np.mean(np.abs(b_hat - b_true)))
    else:
        row["precision_mae"] = np.nan

    return row


def run_one_experiment(
    dgp: str,
    cfg: ToyConfig,
    include_global_baseline: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray | float]]:
    train = generate_dataset(
        dgp=dgp,
        n_per_objective=cfg.n_train_per_objective,
        cfg=cfg,
        seed=cfg.seed,
    )
    test = generate_dataset(
        dgp=dgp,
        n_per_objective=cfg.n_test_per_objective,
        cfg=cfg,
        seed=cfg.seed + 10_000,
    )

    rows = {}

    # Model 1: no uncertainty BT.
    p_no = predict_no_uncertainty_bt(test["gaps"])
    rows["no_uncertainty_bt"] = evaluate_predictions(
        model_name="No uncertainty BT: sigmoid(gap)",
        dgp=dgp,
        p_hat=p_no,
        test=test,
    )

    # Optional baseline: global uncertainty.
    learned: Dict[str, np.ndarray | float] = {}

    if include_global_baseline:
        b_global = fit_global_uncertainty_bt_map(train, cfg)
        p_global = predict_global_uncertainty_bt(test["gaps"], b_global)
        rows["global_uncertainty_bt"] = evaluate_predictions(
            model_name="Global uncertainty BT: sigmoid(exp(b) gap)",
            dgp=dgp,
            p_hat=p_global,
            test=test,
        )
        learned["b_global"] = b_global

    # Model 2: objective-specific uncertainty BT.
    b_obj = fit_objective_uncertainty_bt_map(train, cfg)
    p_obj = predict_objective_uncertainty_bt(
        gaps=test["gaps"],
        objective_ids=test["objective_ids"],
        b_hat=b_obj,
    )
    rows["objective_uncertainty_bt"] = evaluate_predictions(
        model_name="Objective-specific uncertainty BT: sigmoid(exp(b_i) gap)",
        dgp=dgp,
        p_hat=p_obj,
        test=test,
        b_hat=b_obj,
        cfg=cfg,
    )
    learned["b_objective"] = b_obj

    # Oracle reference.
    rows["oracle"] = evaluate_predictions(
        model_name="Oracle p*",
        dgp=dgp,
        p_hat=test["p_star"],
        test=test,
        b_hat=np.asarray(cfg.b_true) if dgp == "bt_objective_uncertainty" else None,
        cfg=cfg,
    )

    df = pd.DataFrame(list(rows.values()))
    return df, learned


def run_repeated_experiments(
    dgp_list: List[str],
    seeds: List[int],
    base_cfg: ToyConfig,
    include_global_baseline: bool = True,
) -> pd.DataFrame:
    all_rows = []

    for dgp in dgp_list:
        for seed in seeds:
            cfg = ToyConfig(**{**base_cfg.__dict__, "seed": seed})
            df, learned = run_one_experiment(
                dgp=dgp,
                cfg=cfg,
                include_global_baseline=include_global_baseline,
            )
            df["seed"] = seed

            # Store learned parameters for inspection.
            if "b_global" in learned:
                df["learned_b_global"] = learned["b_global"]
            else:
                df["learned_b_global"] = np.nan

            b_obj = learned["b_objective"]
            df["learned_b_obj0"] = b_obj[0]
            df["learned_b_obj1"] = b_obj[1]

            all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)



def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "test_nll",
        "prob_mse",
        "prob_mae",
        "precision_mae",
        "learned_b_global",
        "learned_b_obj0",
        "learned_b_obj1",
    ]

    summary = (
        df.groupby(["dgp", "model"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten multi-index columns.
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary


# -----------------------------
# CSV / table export helpers
# -----------------------------

DGP_ORDER = [
    "bt_objective_uncertainty",
    "label_flip_noise",
]

MODEL_ORDER = [
    "No uncertainty BT: sigmoid(gap)",
    "Global uncertainty BT: sigmoid(exp(b) gap)",
    "Objective-specific uncertainty BT: sigmoid(exp(b_i) gap)",
    "Oracle p*",
]

DGP_DISPLAY = {
    "bt_objective_uncertainty": "BT with objective-specific uncertainty",
    "label_flip_noise": "Objective-specific label-flip noise",
}

MODEL_DISPLAY = {
    "No uncertainty BT: sigmoid(gap)": "Fixed-$\\beta$ BT",
    "Global uncertainty BT: sigmoid(exp(b) gap)": "Shared-precision BT",
    "Objective-specific uncertainty BT: sigmoid(exp(b_i) gap)": "Objective-specific BT",
    "Oracle p*": "Oracle",
}


def sort_for_table(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows in the same order as the paper table."""
    out = df.copy()
    out["dgp_order"] = out["dgp"].map({v: i for i, v in enumerate(DGP_ORDER)})
    out["model_order"] = out["model"].map({v: i for i, v in enumerate(MODEL_ORDER)})
    out = out.sort_values(["dgp_order", "model_order"]).drop(
        columns=["dgp_order", "model_order"]
    )
    return out


def format_mean_std(
    mean_value: float,
    std_value: float,
    decimals: int = 4,
    latex: bool = True,
    zero_without_std: bool = True,
) -> str:
    """Format mean ± std for direct table filling."""
    if pd.isna(mean_value):
        return "--"

    if zero_without_std and abs(float(mean_value)) < 10 ** (-(decimals + 1)):
        if pd.isna(std_value) or abs(float(std_value)) < 10 ** (-(decimals + 1)):
            return f"{0.0:.{decimals}f}"

    if pd.isna(std_value):
        return f"{float(mean_value):.{decimals}f}"

    if latex:
        return f"${float(mean_value):.{decimals}f} \\pm {float(std_value):.{decimals}f}$"
    return f"{float(mean_value):.{decimals}f} ± {float(std_value):.{decimals}f}"


def make_table_ready_summary(
    summary: pd.DataFrame,
    decimals: int = 4,
    latex: bool = True,
) -> pd.DataFrame:
    """
    Create a compact CSV that can be copied directly into the LaTeX table.

    Columns match the paper table:
      Data-generating process, Model, Test NLL, Prob. MSE to p*, Precision MAE
    """
    rows = []
    summary = sort_for_table(summary)

    for _, row in summary.iterrows():
        rows.append(
            {
                "Data-generating process": DGP_DISPLAY.get(row["dgp"], row["dgp"]),
                "Model": MODEL_DISPLAY.get(row["model"], row["model"]),
                "Test NLL": format_mean_std(
                    row["test_nll_mean"], row["test_nll_std"], decimals=decimals, latex=latex
                ),
                "Prob. MSE to p_star": format_mean_std(
                    row["prob_mse_mean"], row["prob_mse_std"], decimals=decimals, latex=latex
                ),
                "Precision MAE": format_mean_std(
                    row["precision_mae_mean"],
                    row["precision_mae_std"],
                    decimals=decimals,
                    latex=latex,
                ),
            }
        )

    return pd.DataFrame(rows)


def save_experiment_outputs(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: ToyConfig,
    output_dir: str | Path,
    prefix: str = "toy_calibration",
    decimals: int = 4,
) -> Dict[str, Path]:
    """Save raw, numeric summary, and table-ready CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_sorted = sort_for_table(df)
    summary_sorted = sort_for_table(summary)
    table_ready = make_table_ready_summary(summary_sorted, decimals=decimals, latex=True)
    table_ready_plain = make_table_ready_summary(summary_sorted, decimals=decimals, latex=False)

    paths = {
        "raw": output_dir / f"{prefix}_raw_by_seed.csv",
        "summary_numeric": output_dir / f"{prefix}_summary_numeric.csv",
        "table_ready_latex": output_dir / f"{prefix}_table_ready_latex.csv",
        "table_ready_plain": output_dir / f"{prefix}_table_ready_plain.csv",
        "config": output_dir / f"{prefix}_config.json",
    }

    df_sorted.to_csv(paths["raw"], index=False)
    summary_sorted.to_csv(paths["summary_numeric"], index=False)
    table_ready.to_csv(paths["table_ready_latex"], index=False)
    table_ready_plain.to_csv(paths["table_ready_plain"], index=False)

    with open(paths["config"], "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    return paths


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run toy BT calibration experiments and save CSV outputs."
    )
    parser.add_argument("--output_dir", type=str, default="toy_outputs")
    parser.add_argument("--prefix", type=str, default="toy_calibration")
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_train_per_objective", type=int, default=1000)
    parser.add_argument("--n_test_per_objective", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--prior_sigma_b", type=float, default=2.0)
    parser.add_argument("--label_flip_beta", type=float, default=1.5)
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument(
        "--no_global_baseline",
        action="store_true",
        help="Disable the shared/global precision BT baseline.",
    )
    args = parser.parse_args()

    base_cfg = ToyConfig(
        num_objectives=2,
        n_train_per_objective=args.n_train_per_objective,
        n_test_per_objective=args.n_test_per_objective,
        b_true=(1.0, -1.0),
        flip_eps=(0.05, 0.35),
        label_flip_beta=args.label_flip_beta,
        prior_sigma_b=args.prior_sigma_b,
        lr=args.lr,
        steps=args.steps,
        seed=args.seed_start,
    )

    dgp_list = [
        "bt_objective_uncertainty",
        "label_flip_noise",
    ]

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    df = run_repeated_experiments(
        dgp_list=dgp_list,
        seeds=seeds,
        base_cfg=base_cfg,
        include_global_baseline=not args.no_global_baseline,
    )

    summary = summarize_results(df)

    paths = save_experiment_outputs(
        df=df,
        summary=summary,
        cfg=base_cfg,
        output_dir=args.output_dir,
        prefix=args.prefix,
        decimals=args.decimals,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n=== Per-seed raw results ===")
    print(df)

    print("\n=== Summary over seeds ===")
    print(summary)

    print("\n=== Table-ready summary ===")
    print(make_table_ready_summary(summary, decimals=args.decimals, latex=True))

    print("\n=== Saved CSV / metadata files ===")
    for name, path in paths.items():
        print(f"[{name}] {path}")

    print("\n=== True parameters ===")
    print(f"BT uncertainty DGP true b*: {base_cfg.b_true}")
    print(f"Label flip DGP eps: {base_cfg.flip_eps}")
