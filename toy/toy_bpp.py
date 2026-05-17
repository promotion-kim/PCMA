#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OOD toy experiment for BPP-MOA vs MODPO-style score scalarization.

Goal
----
Show the mechanism BPP-MOA is designed for:

  When an objective-specific reward model has poor coverage in some feature
  directions, plug-in score scalarization can become overconfident because it
  only uses the posterior mean reward gap.

  BPP-MOA uses the posterior predictive variance:
      ell_i = mu_i / sqrt(1 + pi * v_i / 8)
  so objective logits are shrunk toward 0 in low-evidence / OOD regions.

Synthetic setup
---------------
There are two objectives:
  1. helpful
  2. harmless/safety

Each objective has a true linear Bradley-Terry reward gap:
    z_i = theta_i^T delta_h
    y_i ~ Bernoulli(sigmoid(z_i))

The helpful objective has broad training coverage:
    delta_h_help_train ~ N(0, I)

The safety objective has narrow coverage:
    first id_dim features      ~ N(0, I)
    remaining tail features   ~ N(0, safe_train_tail_scale^2 I)

But the safety true reward heavily depends on the tail dimensions.
Therefore, the safety reward head is uncertain on tail directions.

We evaluate on:
  ID test  : same narrow coverage as safety train
  OOD test : tail dimensions are active, delta_h_tail ~ N(0, I)

Methods
-------
MODPO-style plug-in:
    rho_modpo = sigmoid(w_h * mu_h + w_s * mu_s)

BPP-MOA:
    ell_h = mu_h / sqrt(1 + pi * v_h / 8)
    ell_s = mu_s / sqrt(1 + pi * v_s / 8)
    rho_bpp = sigmoid(w_h * ell_h + w_s * ell_s)

Metrics
-------
NLL/Brier/Accuracy/Confidence are reported separately on ID and OOD test sets.
The most important diagnostic is that in OOD:
    safe_abs_ell << safe_abs_mu
because BPP-MOA shrinks the uncertain safety logit.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def binary_nll(y: np.ndarray, p: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Simple binary ECE using confidence=max(p,1-p)."""
    pred = (p >= 0.5).astype(np.float64)
    conf = np.maximum(p, 1.0 - p)
    correct = (pred == y).astype(np.float64)

    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def make_unit_vector(rng: np.random.Generator, dim: int, norm: float) -> np.ndarray:
    v = rng.normal(size=dim)
    v = v / (np.linalg.norm(v) + 1e-12)
    return norm * v


def make_safety_theta(
    rng: np.random.Generator,
    dim: int,
    id_dim: int,
    true_norm: float,
    tail_ratio: float,
) -> np.ndarray:
    """
    Make safety objective mostly depend on tail dimensions.

    tail_ratio controls the fraction of the norm allocated to tail dimensions.
    Example: tail_ratio=0.9 means 90% of theta norm is in tail dimensions.
    """
    assert 0.0 <= tail_ratio <= 1.0
    theta = np.zeros(dim, dtype=np.float64)

    head_norm = true_norm * math.sqrt(max(0.0, 1.0 - tail_ratio**2))
    tail_norm = true_norm * tail_ratio

    if id_dim > 0 and head_norm > 0:
        theta[:id_dim] = make_unit_vector(rng, id_dim, head_norm)
    if dim - id_dim > 0 and tail_norm > 0:
        theta[id_dim:] = make_unit_vector(rng, dim - id_dim, tail_norm)

    return theta


def sample_id_features(
    rng: np.random.Generator,
    n: int,
    dim: int,
    id_dim: int,
    tail_scale: float,
) -> np.ndarray:
    """Feature distribution with weak tail dimensions."""
    x_id = rng.normal(size=(n, id_dim))
    x_tail = tail_scale * rng.normal(size=(n, dim - id_dim))
    return np.concatenate([x_id, x_tail], axis=1)


def sample_ood_features(
    rng: np.random.Generator,
    n: int,
    dim: int,
    id_dim: int,
) -> np.ndarray:
    """Feature distribution where tail dimensions become active."""
    x_id = rng.normal(size=(n, id_dim))
    x_tail = rng.normal(size=(n, dim - id_dim))
    return np.concatenate([x_id, x_tail], axis=1)


def fit_map_and_diag_laplace(
    X: np.ndarray,
    y: np.ndarray,
    prior_precision: float,
) -> Dict[str, np.ndarray]:
    """
    Ridge logistic MAP + diagonal GGN/Laplace precision.

    MAP objective:
        - sum log Bernoulli(y | sigmoid(X theta))
        + 0.5 * prior_precision * ||theta||^2

    Diagonal GGN precision:
        H_diag = prior_precision + sum_n p_n(1-p_n) x_n^2
    """
    if len(np.unique(y)) < 2:
        # Avoid sklearn failure for extremely small datasets.
        y = y.copy()
        y[0] = 1 - y[0]

    # sklearn objective scale is not exactly the same as the written MAP objective,
    # but C=1/prior_precision gives a controlled ridge strength for this toy.
    clf = LogisticRegression(
        penalty="l2",
        C=1.0 / prior_precision,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=2000,
    )
    clf.fit(X, y)

    theta = clf.coef_.reshape(-1)
    p = sigmoid(X @ theta)
    diag_precision = prior_precision + ((p * (1.0 - p))[:, None] * (X**2)).sum(axis=0)

    return {
        "theta": theta,
        "diag_precision": diag_precision,
    }


def evaluate_predictions(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "nll": binary_nll(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "acc": float(np.mean((p >= 0.5) == y)),
        "ece": expected_calibration_error(y, p),
        "confidence": float(np.mean(np.abs(p - 0.5)) * 2.0),
    }


def evaluate_one_split(
    rng: np.random.Generator,
    X: np.ndarray,
    theta_help_true: np.ndarray,
    theta_safe_true: np.ndarray,
    help_head: Dict[str, np.ndarray],
    safe_head: Dict[str, np.ndarray],
    w_help: float,
) -> Dict[str, float]:
    w_safe = 1.0 - w_help

    true_logit = w_help * (X @ theta_help_true) + w_safe * (X @ theta_safe_true)
    y = rng.binomial(1, sigmoid(true_logit))

    mu_help = X @ help_head["theta"]
    mu_safe = X @ safe_head["theta"]

    var_help = ((X**2) / help_head["diag_precision"]).sum(axis=1)
    var_safe = ((X**2) / safe_head["diag_precision"]).sum(axis=1)

    pred_modpo = sigmoid(w_help * mu_help + w_safe * mu_safe)

    ell_help = mu_help / np.sqrt(1.0 + math.pi * var_help / 8.0)
    ell_safe = mu_safe / np.sqrt(1.0 + math.pi * var_safe / 8.0)
    pred_bpp = sigmoid(w_help * ell_help + w_safe * ell_safe)

    modpo = evaluate_predictions(y, pred_modpo)
    bpp = evaluate_predictions(y, pred_bpp)

    out = {}
    for k, v in modpo.items():
        out[f"modpo_{k}"] = v
    for k, v in bpp.items():
        out[f"bpp_{k}"] = v

    out.update(
        {
            "safe_var_mean": float(np.mean(var_safe)),
            "safe_abs_mu_mean": float(np.mean(np.abs(mu_safe))),
            "safe_abs_ell_mean": float(np.mean(np.abs(ell_safe))),
            "help_var_mean": float(np.mean(var_help)),
            "help_abs_mu_mean": float(np.mean(np.abs(mu_help))),
            "help_abs_ell_mean": float(np.mean(np.abs(ell_help))),
        }
    )
    return out


def run_one_seed(args: argparse.Namespace, seed: int, n_safe: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    theta_help_true = make_unit_vector(rng, args.dim, args.true_norm)
    theta_safe_true = make_safety_theta(
        rng=rng,
        dim=args.dim,
        id_dim=args.id_dim,
        true_norm=args.true_norm,
        tail_ratio=args.safe_tail_ratio,
    )

    # Objective-specific datasets.
    X_help = sample_ood_features(rng, args.n_help, args.dim, args.id_dim)
    y_help = rng.binomial(1, sigmoid(X_help @ theta_help_true))

    X_safe = sample_id_features(
        rng,
        n_safe,
        args.dim,
        args.id_dim,
        args.safe_train_tail_scale,
    )
    y_safe = rng.binomial(1, sigmoid(X_safe @ theta_safe_true))

    help_head = fit_map_and_diag_laplace(X_help, y_help, args.prior_precision)
    safe_head = fit_map_and_diag_laplace(X_safe, y_safe, args.prior_precision)

    X_id = sample_id_features(
        rng,
        args.n_test,
        args.dim,
        args.id_dim,
        args.safe_train_tail_scale,
    )
    X_ood = sample_ood_features(rng, args.n_test, args.dim, args.id_dim)

    id_metrics = evaluate_one_split(
        rng, X_id, theta_help_true, theta_safe_true, help_head, safe_head, args.w_help
    )
    ood_metrics = evaluate_one_split(
        rng, X_ood, theta_help_true, theta_safe_true, help_head, safe_head, args.w_help
    )

    row = {
        "seed": seed,
        "n_safe": n_safe,
        "n_help": args.n_help,
        "dim": args.dim,
        "id_dim": args.id_dim,
        "safe_train_tail_scale": args.safe_train_tail_scale,
        "safe_tail_ratio": args.safe_tail_ratio,
        "prior_precision": args.prior_precision,
        "w_help": args.w_help,
        "w_safe": 1.0 - args.w_help,
    }

    for k, v in id_metrics.items():
        row[f"id_{k}"] = v
    for k, v in ood_metrics.items():
        row[f"ood_{k}"] = v

    return row


def plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    # OOD NLL
    plt.figure()
    plt.plot(summary["n_safe"], summary["ood_modpo_nll"], marker="o", label="MODPO-style plug-in")
    plt.plot(summary["n_safe"], summary["ood_bpp_nll"], marker="o", label="BPP-MOA")
    plt.xscale("log")
    plt.xlabel("Number of safety-objective training pairs")
    plt.ylabel("OOD mixed-objective NLL")
    plt.title("OOD low-coverage region: BPP-MOA reduces overconfidence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ood_nll_vs_safe_data.png", dpi=220)

    # ID NLL
    plt.figure()
    plt.plot(summary["n_safe"], summary["id_modpo_nll"], marker="o", label="MODPO-style plug-in")
    plt.plot(summary["n_safe"], summary["id_bpp_nll"], marker="o", label="BPP-MOA")
    plt.xscale("log")
    plt.xlabel("Number of safety-objective training pairs")
    plt.ylabel("ID mixed-objective NLL")
    plt.title("ID region")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "id_nll_vs_safe_data.png", dpi=220)

    # Safety logit shrinkage diagnostic
    plt.figure()
    plt.plot(summary["n_safe"], summary["ood_safe_abs_mu_mean"], marker="o", label="Raw safety |mu|")
    plt.plot(summary["n_safe"], summary["ood_safe_abs_ell_mean"], marker="o", label="BPP safety |ell|")
    plt.xscale("log")
    plt.xlabel("Number of safety-objective training pairs")
    plt.ylabel("Mean absolute safety logit on OOD")
    plt.title("Posterior predictive shrinkage diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ood_safety_logit_shrinkage.png", dpi=220)

    # Confidence diagnostic
    plt.figure()
    plt.plot(summary["n_safe"], summary["ood_modpo_confidence"], marker="o", label="MODPO-style confidence")
    plt.plot(summary["n_safe"], summary["ood_bpp_confidence"], marker="o", label="BPP-MOA confidence")
    plt.xscale("log")
    plt.xlabel("Number of safety-objective training pairs")
    plt.ylabel("Mean confidence, 2|p-0.5|")
    plt.title("OOD confidence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ood_confidence.png", dpi=220)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_help", type=int, default=2000)
    parser.add_argument("--n_safe_list", type=int, nargs="+", default=[10, 20, 50, 100, 500])
    parser.add_argument("--n_test", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=50)

    parser.add_argument("--dim", type=int, default=80)
    parser.add_argument("--id_dim", type=int, default=10)
    parser.add_argument("--true_norm", type=float, default=3.0)
    parser.add_argument("--safe_tail_ratio", type=float, default=0.90)
    parser.add_argument("--safe_train_tail_scale", type=float, default=0.05)

    parser.add_argument("--prior_precision", type=float, default=0.10)
    parser.add_argument("--w_help", type=float, default=0.5)

    parser.add_argument("--output_dir", type=str, default="toy_bppmoa_ood_outputs")
    args = parser.parse_args()

    if not (0 < args.id_dim < args.dim):
        raise ValueError("--id_dim must satisfy 0 < id_dim < dim")
    if not (0.0 <= args.safe_tail_ratio <= 1.0):
        raise ValueError("--safe_tail_ratio must be in [0, 1]")
    if not (0.0 <= args.w_help <= 1.0):
        raise ValueError("--w_help must be in [0, 1]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    for n_safe in args.n_safe_list:
        for seed in range(args.seeds):
            rows.append(run_one_seed(args, seed=seed, n_safe=n_safe))

    raw = pd.DataFrame(rows)
    raw_path = output_dir / "raw_results.csv"
    raw.to_csv(raw_path, index=False)

    summary = raw.groupby("n_safe").mean(numeric_only=True).reset_index()

    # Gains: positive means BPP is better.
    summary["id_nll_gain"] = summary["id_modpo_nll"] - summary["id_bpp_nll"]
    summary["ood_nll_gain"] = summary["ood_modpo_nll"] - summary["ood_bpp_nll"]
    summary["id_brier_gain"] = summary["id_modpo_brier"] - summary["id_bpp_brier"]
    summary["ood_brier_gain"] = summary["ood_modpo_brier"] - summary["ood_bpp_brier"]
    summary["id_ece_gain"] = summary["id_modpo_ece"] - summary["id_bpp_ece"]
    summary["ood_ece_gain"] = summary["ood_modpo_ece"] - summary["ood_bpp_ece"]

    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_results(summary, output_dir)

    cols = [
        "n_safe",
        "id_modpo_nll", "id_bpp_nll", "id_nll_gain",
        "ood_modpo_nll", "ood_bpp_nll", "ood_nll_gain",
        "ood_modpo_brier", "ood_bpp_brier", "ood_brier_gain",
        "ood_modpo_ece", "ood_bpp_ece", "ood_ece_gain",
        "ood_safe_var_mean", "ood_safe_abs_mu_mean", "ood_safe_abs_ell_mean",
        "ood_modpo_confidence", "ood_bpp_confidence",
    ]

    print("\n=== Mean over seeds ===")
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nSaved raw results:     {raw_path}")
    print(f"Saved summary:         {summary_path}")
    print(f"Saved plots under:     {output_dir}")


if __name__ == "__main__":
    main()
