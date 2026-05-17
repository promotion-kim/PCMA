#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
toy_mo.py

Controlled multi-objective toy experiment for objective-specific uncertainty.

Goal
----
Show that when objective scorers have unequal reliability, learning
objective-specific precision from objective-wise preferences can improve
multi-objective scalarized preference prediction.

The experiment separates three layers:

1. Latent objective gaps:
     Δq_i = q_i(y_A) - q_i(y_B)
   These define the ground-truth objective qualities.

2. Noisy scorer gaps:
     Δs_i = Δq_i + ε_i
   Models only observe Δs_i.

3. Multi-objective user preference:
     p_w^* = sigmoid(beta_mo * sum_i w_i Δq_i)
   Evaluation checks whether each model can predict p_w^* using only Δs_i.

Models
------
- Standard scalarized BT:
    sigmoid(beta_mo * sum_i w_i Δs_i)

- Shared-precision scalarized BT:
    sigmoid(beta_mo * exp(gamma b) * sum_i w_i Δs_i)

- PCMA-normalized scalarization:
    alpha_i(w) = w_i exp(gamma b_i) / sum_j w_j exp(gamma b_j)
    sigmoid(beta_mo * sum_i alpha_i(w) Δs_i)

- Confidence-preserving PCMA:
    alpha_i(w) = w_i exp(gamma b_i) / sum_j w_j exp(gamma b_j)
    rho(w) = sum_j w_j exp(gamma b_j)
    sigmoid(beta_mo * rho(w) * sum_i alpha_i(w) Δs_i)
  Equivalently:
    sigmoid(beta_mo * sum_i w_i exp(gamma b_i) Δs_i)

- Oracle latent utility:
    sigmoid(beta_mo * sum_i w_i Δq_i)

Outputs
-------
Given --output_dir, saves:
- <prefix>_raw_by_seed.csv
- <prefix>_summary_numeric.csv
- <prefix>_table_ready_latex.csv
- <prefix>_mo_per_weight_raw.csv
- <prefix>_mo_per_weight_summary_numeric.csv
- <prefix>_config.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    # Numerically stable enough for this toy setting.
    return 1.0 / (1.0 + np.exp(-x))


def binary_nll_from_prob(p_hat: np.ndarray, z: np.ndarray, eps: float = 1e-8) -> float:
    p_hat = np.clip(p_hat, eps, 1.0 - eps)
    return float(-np.mean(z * np.log(p_hat) + (1.0 - z) * np.log(1.0 - p_hat)))


def prob_mse(p_hat: np.ndarray, p_star: np.ndarray) -> float:
    return float(np.mean((p_hat - p_star) ** 2))


def prob_mae(p_hat: np.ndarray, p_star: np.ndarray) -> float:
    return float(np.mean(np.abs(p_hat - p_star)))


def format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"${mean:.{digits}f}$"
    return f"${mean:.{digits}f} \\pm {std:.{digits}f}$"


def parse_float_tuple(s: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class MOConfig:
    num_objectives: int = 3

    # Calibration train/test pair counts.
    n_train_per_objective: int = 1000
    n_test_per_objective: int = 5000

    # Response-selection regret evaluation.
    n_selection_prompts: int = 2000
    n_candidates: int = 5

    # Latent objective covariance.
    # The default covariance for 3 objectives is:
    # [[1.0, conflict_corr, 0.2],
    #  [conflict_corr, 1.0, 0.1],
    #  [0.2, 0.1, 1.0]]
    conflict_corr: float = -0.4

    # Scorer noise.
    scorer_noise_sigmas: Tuple[float, ...] = (0.3, 1.2, 0.7)
    equal_noise_sigma: float = 0.7

    # BT temperature/scales.
    beta_obj: float = 1.0
    beta_mo: float = 1.0

    # Downstream calibration strength.
    # gamma=0 recovers standard scalarization; gamma=1 uses full learned precision.
    calibration_strength_gamma: float = 1.0

    # MAP prior and optimizer.
    prior_sigma_b: float = 2.0
    lr: float = 5e-2
    steps: int = 1000

    # Randomness.
    seed: int = 0

    # Device.
    device: str = "auto"


def make_covariance(cfg: MOConfig) -> np.ndarray:
    m = cfg.num_objectives
    if m != 3:
        # Generic fallback: identity with mild negative correlation
        # between the first two objectives.
        cov = np.eye(m)
        if m >= 2:
            cov[0, 1] = cov[1, 0] = cfg.conflict_corr
        return cov

    cov = np.array(
        [
            [1.0, cfg.conflict_corr, 0.2],
            [cfg.conflict_corr, 1.0, 0.1],
            [0.2, 0.1, 1.0],
        ],
        dtype=np.float64,
    )

    # Ensure positive definiteness. The default is positive definite.
    eig_min = np.linalg.eigvalsh(cov).min()
    if eig_min <= 1e-6:
        cov = cov + np.eye(3) * (1e-6 - eig_min + 1e-6)
    return cov


def get_noise_sigmas(dgp: str, cfg: MOConfig) -> np.ndarray:
    if dgp == "objective_specific_noise":
        sigmas = np.asarray(cfg.scorer_noise_sigmas, dtype=np.float64)
        if len(sigmas) != cfg.num_objectives:
            raise ValueError(
                f"Expected {cfg.num_objectives} scorer noise sigmas, got {len(sigmas)}: {sigmas}"
            )
        return sigmas

    if dgp == "equal_noise_control":
        return np.full(cfg.num_objectives, cfg.equal_noise_sigma, dtype=np.float64)

    raise ValueError(f"Unknown DGP: {dgp}")


def default_weight_grid(num_objectives: int) -> np.ndarray:
    if num_objectives == 3:
        weights = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [0.2, 0.6, 0.2],
                [0.6, 0.3, 0.1],
            ],
            dtype=np.float64,
        )
        return weights

    # Generic fallback: endpoints plus uniform.
    weights = []
    for i in range(num_objectives):
        w = np.zeros(num_objectives, dtype=np.float64)
        w[i] = 1.0
        weights.append(w)
    weights.append(np.full(num_objectives, 1.0 / num_objectives, dtype=np.float64))
    return np.stack(weights, axis=0)


def weights_to_name(w: np.ndarray) -> str:
    return "(" + ",".join(f"{x:.2f}" for x in w.tolist()) + ")"


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ---------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------

def generate_pair_data(
    dgp: str,
    n_pairs: int,
    cfg: MOConfig,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Generate pairwise latent objective gaps Δq and noisy scorer gaps Δs.

    Returns:
      dq: shape [N, m], latent objective gaps
      ds: shape [N, m], observed noisy scorer gaps
      p_obj_star: shape [N, m], objective-wise true preference probabilities
      z_obj: shape [N, m], sampled objective-wise binary labels
    """
    rng = np.random.default_rng(seed)

    m = cfg.num_objectives
    cov = make_covariance(cfg)
    sigmas = get_noise_sigmas(dgp, cfg)

    dq = rng.multivariate_normal(
        mean=np.zeros(m, dtype=np.float64),
        cov=cov,
        size=n_pairs,
    )
    eps = rng.normal(
        loc=0.0,
        scale=sigmas.reshape(1, m),
        size=(n_pairs, m),
    )
    ds = dq + eps

    p_obj_star = sigmoid_np(cfg.beta_obj * dq)
    z_obj = rng.binomial(1, p_obj_star).astype(np.float64)

    return {
        "dq": dq.astype(np.float64),
        "ds": ds.astype(np.float64),
        "p_obj_star": p_obj_star.astype(np.float64),
        "z_obj": z_obj.astype(np.float64),
    }


def generate_selection_data(
    dgp: str,
    n_prompts: int,
    n_candidates: int,
    cfg: MOConfig,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Generate response-candidate data for decision regret.

    q: latent objective quality, shape [P, K, m]
    s: noisy scorer quality, shape [P, K, m]
    """
    rng = np.random.default_rng(seed)

    m = cfg.num_objectives
    cov = make_covariance(cfg)
    sigmas = get_noise_sigmas(dgp, cfg)

    q = rng.multivariate_normal(
        mean=np.zeros(m, dtype=np.float64),
        cov=cov,
        size=(n_prompts, n_candidates),
    )
    eps = rng.normal(
        loc=0.0,
        scale=sigmas.reshape(1, 1, m),
        size=(n_prompts, n_candidates, m),
    )
    s = q + eps

    return {
        "q": q.astype(np.float64),
        "s": s.astype(np.float64),
    }


# ---------------------------------------------------------------------
# Fitting calibration models
# ---------------------------------------------------------------------

def fit_shared_precision_map(
    train: Dict[str, np.ndarray],
    cfg: MOConfig,
) -> float:
    """
    Fit shared log precision b by MAP:
      p(z_i = 1) = sigmoid(beta_obj * exp(b) * Δs_i)
    """
    device = resolve_device(cfg.device)

    ds = torch.tensor(train["ds"], dtype=torch.float32, device=device)
    z = torch.tensor(train["z_obj"], dtype=torch.float32, device=device)

    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    b = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([b], lr=cfg.lr)

    n_total = float(ds.numel())

    for _ in range(cfg.steps):
        logits = cfg.beta_obj * torch.exp(b) * ds
        nll = F.binary_cross_entropy_with_logits(logits, z, reduction="mean")
        prior = (b ** 2) / (2.0 * n_total * (cfg.prior_sigma_b ** 2))
        loss = nll + prior

        opt.zero_grad()
        loss.backward()
        opt.step()

    return float(b.detach().cpu().item())


def fit_objective_precision_map(
    train: Dict[str, np.ndarray],
    cfg: MOConfig,
) -> np.ndarray:
    """
    Fit objective-specific log precisions b_i by MAP:
      p(z_i = 1) = sigmoid(beta_obj * exp(b_i) * Δs_i)
    """
    device = resolve_device(cfg.device)

    ds = torch.tensor(train["ds"], dtype=torch.float32, device=device)
    z = torch.tensor(train["z_obj"], dtype=torch.float32, device=device)

    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    b = torch.zeros(cfg.num_objectives, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([b], lr=cfg.lr)

    n_total = float(ds.numel())

    for _ in range(cfg.steps):
        logits = cfg.beta_obj * torch.exp(b.reshape(1, -1)) * ds
        nll = F.binary_cross_entropy_with_logits(logits, z, reduction="mean")
        prior = torch.sum(b ** 2) / (2.0 * n_total * (cfg.prior_sigma_b ** 2))
        loss = nll + prior

        opt.zero_grad()
        loss.backward()
        opt.step()

    return b.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------

def objective_predictions(
    test: Dict[str, np.ndarray],
    cfg: MOConfig,
    model: str,
    b_global: Optional[float] = None,
    b_obj: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Return objective-wise probabilities, shape [N, m].
    Some MO-only models return None.
    """
    ds = test["ds"]

    if model == "standard":
        return sigmoid_np(cfg.beta_obj * ds)

    if model == "shared_precision":
        if b_global is None:
            raise ValueError("b_global is required for shared_precision.")
        return sigmoid_np(cfg.beta_obj * math.exp(b_global) * ds)

    if model == "objective_precision":
        if b_obj is None:
            raise ValueError("b_obj is required for objective_precision.")
        return sigmoid_np(cfg.beta_obj * ds * np.exp(b_obj).reshape(1, -1))

    return None


def mo_prediction_for_weight(
    test: Dict[str, np.ndarray],
    cfg: MOConfig,
    w: np.ndarray,
    model: str,
    b_global: Optional[float] = None,
    b_obj: Optional[np.ndarray] = None,
) -> np.ndarray:
    dq = test["dq"]
    ds = test["ds"]

    if model == "standard":
        return sigmoid_np(cfg.beta_mo * (ds @ w))

    if model == "shared_precision":
        if b_global is None:
            raise ValueError("b_global is required for shared_precision.")
        return sigmoid_np(cfg.beta_mo * math.exp(cfg.calibration_strength_gamma * b_global) * (ds @ w))

    if model == "objective_precision_scaled":
        if b_obj is None:
            raise ValueError("b_obj is required for objective_precision_scaled.")
        kappa = np.exp(cfg.calibration_strength_gamma * b_obj)
        return sigmoid_np(cfg.beta_mo * (ds @ (w * kappa)))

    if model == "pcma_normalized":
        if b_obj is None:
            raise ValueError("b_obj is required for pcma_normalized.")
        kappa = np.exp(cfg.calibration_strength_gamma * b_obj)
        alpha = w * kappa
        denom = alpha.sum()
        if denom <= 0.0:
            # This happens only for all-zero weights, which should not be used.
            alpha = w
        else:
            alpha = alpha / denom
        return sigmoid_np(cfg.beta_mo * (ds @ alpha))

    if model == "confidence_preserving_pcma":
        if b_obj is None:
            raise ValueError("b_obj is required for confidence_preserving_pcma.")
        kappa = np.exp(cfg.calibration_strength_gamma * b_obj)
        alpha_unnormalized = w * kappa
        rho = alpha_unnormalized.sum()
        if rho <= 0.0:
            # This happens only for all-zero weights, which should not be used.
            alpha = w
            rho = 1.0
        else:
            alpha = alpha_unnormalized / rho

        # Confidence-preserving PCMA keeps both relative reliability (alpha)
        # and total reliability mass (rho). This is equivalent to ds @ (w * kappa).
        return sigmoid_np(cfg.beta_mo * rho * (ds @ alpha))

    if model == "oracle":
        return sigmoid_np(cfg.beta_mo * (dq @ w))

    raise ValueError(f"Unknown MO model: {model}")


def model_display_name(model: str) -> str:
    return {
        "standard": "Standard scalarized BT",
        "shared_precision": "Shared-precision BT",
        "objective_precision": "Objective-specific precision BT",
        "objective_precision_scaled": "Objective-specific precision BT",
        "pcma_normalized": "PCMA-normalized scalarization",
        "confidence_preserving_pcma": "Confidence-preserving PCMA",
        "oracle": "Oracle latent utility",
    }.get(model, model)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_objective_level(
    dgp: str,
    test: Dict[str, np.ndarray],
    cfg: MOConfig,
    model: str,
    b_global: Optional[float] = None,
    b_obj: Optional[np.ndarray] = None,
) -> Dict[str, float | str]:
    p_hat = objective_predictions(test, cfg, model, b_global=b_global, b_obj=b_obj)

    row: Dict[str, float | str] = {
        "dgp": dgp,
        "model_key": model,
        "model": model_display_name(model),
    }

    if p_hat is None:
        row.update(
            {
                "obj_nll": np.nan,
                "obj_prob_mse": np.nan,
                "obj_prob_mae": np.nan,
            }
        )
        return row

    z = test["z_obj"]
    p_star = test["p_obj_star"]

    row.update(
        {
            "obj_nll": binary_nll_from_prob(p_hat.reshape(-1), z.reshape(-1)),
            "obj_prob_mse": prob_mse(p_hat, p_star),
            "obj_prob_mae": prob_mae(p_hat, p_star),
        }
    )
    return row


def evaluate_mo_level(
    dgp: str,
    test: Dict[str, np.ndarray],
    cfg: MOConfig,
    weights: np.ndarray,
    model: str,
    b_global: Optional[float],
    b_obj: Optional[np.ndarray],
    seed: int,
) -> Tuple[Dict[str, float | str], pd.DataFrame]:
    rng = np.random.default_rng(seed)

    per_weight_rows = []

    mo_nlls = []
    mo_mses = []
    mo_maes = []

    for w in weights:
        p_star = sigmoid_np(cfg.beta_mo * (test["dq"] @ w))
        z = rng.binomial(1, p_star).astype(np.float64)

        p_hat = mo_prediction_for_weight(
            test=test,
            cfg=cfg,
            w=w,
            model=model,
            b_global=b_global,
            b_obj=b_obj,
        )

        nll = binary_nll_from_prob(p_hat, z)
        mse = prob_mse(p_hat, p_star)
        mae = prob_mae(p_hat, p_star)

        mo_nlls.append(nll)
        mo_mses.append(mse)
        mo_maes.append(mae)

        per_weight_rows.append(
            {
                "dgp": dgp,
                "model_key": model,
                "model": model_display_name(model),
                "weight": weights_to_name(w),
                "mo_nll": nll,
                "mo_prob_mse": mse,
                "mo_prob_mae": mae,
            }
        )

    row: Dict[str, float | str] = {
        "dgp": dgp,
        "model_key": model,
        "model": model_display_name(model),
        "mo_nll": float(np.mean(mo_nlls)),
        "mo_prob_mse": float(np.mean(mo_mses)),
        "mo_prob_mae": float(np.mean(mo_maes)),
    }

    return row, pd.DataFrame(per_weight_rows)


def selection_scores(
    selection: Dict[str, np.ndarray],
    w: np.ndarray,
    model: str,
    b_global: Optional[float],
    b_obj: Optional[np.ndarray],
    calibration_strength_gamma: float = 1.0,
) -> np.ndarray:
    q = selection["q"]
    s = selection["s"]

    if model == "standard":
        return s @ w

    if model == "shared_precision":
        # Positive global scaling does not change argmax, but we keep it for completeness.
        if b_global is None:
            raise ValueError("b_global is required for shared_precision.")
        return math.exp(calibration_strength_gamma * b_global) * (s @ w)

    if model == "objective_precision_scaled":
        if b_obj is None:
            raise ValueError("b_obj is required for objective_precision_scaled.")
        kappa = np.exp(calibration_strength_gamma * b_obj)
        return s @ (w * kappa)

    if model == "pcma_normalized":
        if b_obj is None:
            raise ValueError("b_obj is required for pcma_normalized.")
        kappa = np.exp(calibration_strength_gamma * b_obj)
        alpha = w * kappa
        denom = alpha.sum()
        alpha = alpha / denom if denom > 0.0 else w
        return s @ alpha

    if model == "confidence_preserving_pcma":
        if b_obj is None:
            raise ValueError("b_obj is required for confidence_preserving_pcma.")
        kappa = np.exp(calibration_strength_gamma * b_obj)
        alpha_unnormalized = w * kappa
        rho = alpha_unnormalized.sum()
        alpha = alpha_unnormalized / rho if rho > 0.0 else w
        return rho * (s @ alpha)

    if model == "oracle":
        return q @ w

    raise ValueError(f"Unknown selection model: {model}")


def evaluate_decision_regret(
    selection: Dict[str, np.ndarray],
    weights: np.ndarray,
    model: str,
    b_global: Optional[float],
    b_obj: Optional[np.ndarray],
    calibration_strength_gamma: float = 1.0,
) -> float:
    q = selection["q"]

    regrets = []
    for w in weights:
        true_scores = q @ w  # [P, K]
        best_true_idx = np.argmax(true_scores, axis=1)
        best_true_value = true_scores[np.arange(true_scores.shape[0]), best_true_idx]

        model_scores = selection_scores(
            selection=selection,
            w=w,
            model=model,
            b_global=b_global,
            b_obj=b_obj,
            calibration_strength_gamma=calibration_strength_gamma,
        )
        chosen_idx = np.argmax(model_scores, axis=1)
        chosen_true_value = true_scores[np.arange(true_scores.shape[0]), chosen_idx]

        regrets.append(float(np.mean(best_true_value - chosen_true_value)))

    return float(np.mean(regrets))


def run_one_seed(
    dgp: str,
    seed: int,
    base_cfg: MOConfig,
    weights: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = MOConfig(**{**asdict(base_cfg), "seed": seed})

    train = generate_pair_data(
        dgp=dgp,
        n_pairs=cfg.n_train_per_objective,
        cfg=cfg,
        seed=seed,
    )
    test = generate_pair_data(
        dgp=dgp,
        n_pairs=cfg.n_test_per_objective,
        cfg=cfg,
        seed=seed + 10_000,
    )
    selection = generate_selection_data(
        dgp=dgp,
        n_prompts=cfg.n_selection_prompts,
        n_candidates=cfg.n_candidates,
        cfg=cfg,
        seed=seed + 20_000,
    )

    b_global = fit_shared_precision_map(train, cfg)
    b_obj = fit_objective_precision_map(train, cfg)

    model_keys = [
        "standard",
        "shared_precision",
        "pcma_normalized",
        "confidence_preserving_pcma",
        "oracle",
    ]

    combined_rows = []
    per_weight_all = []

    for model in model_keys:
        obj_model_for_eval: str
        if model in {"objective_precision_scaled", "confidence_preserving_pcma"}:
            obj_model_for_eval = "objective_precision"
        elif model in {"standard", "shared_precision"}:
            obj_model_for_eval = model
        else:
            obj_model_for_eval = model

        obj_row = evaluate_objective_level(
            dgp=dgp,
            test=test,
            cfg=cfg,
            model=obj_model_for_eval,
            b_global=b_global,
            b_obj=b_obj,
        )

        mo_row, per_weight_df = evaluate_mo_level(
            dgp=dgp,
            test=test,
            cfg=cfg,
            weights=weights,
            model=model,
            b_global=b_global,
            b_obj=b_obj,
            seed=seed + 30_000,
        )

        regret = evaluate_decision_regret(
            selection=selection,
            weights=weights,
            model=model,
            b_global=b_global,
            b_obj=b_obj,
            calibration_strength_gamma=cfg.calibration_strength_gamma,
        )

        row = {
            "seed": seed,
            "dgp": dgp,
            "model_key": model,
            "model": model_display_name(model),
            "obj_nll": obj_row["obj_nll"],
            "obj_prob_mse": obj_row["obj_prob_mse"],
            "obj_prob_mae": obj_row["obj_prob_mae"],
            "mo_nll": mo_row["mo_nll"],
            "mo_prob_mse": mo_row["mo_prob_mse"],
            "mo_prob_mae": mo_row["mo_prob_mae"],
            "decision_regret": regret,
            "learned_b_global": b_global,
        }

        for i in range(cfg.num_objectives):
            row[f"learned_b_obj{i}"] = b_obj[i]
            row[f"scorer_noise_sigma{i}"] = get_noise_sigmas(dgp, cfg)[i]

        combined_rows.append(row)

        per_weight_df["seed"] = seed
        per_weight_df["decision_regret"] = np.nan  # Aggregate regret is in combined rows.
        per_weight_all.append(per_weight_df)

    return pd.DataFrame(combined_rows), pd.concat(per_weight_all, ignore_index=True)


def run_experiments(
    dgp_list: Sequence[str],
    seeds: Sequence[int],
    cfg: MOConfig,
    weights: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    all_per_weight = []

    for dgp in dgp_list:
        for seed in seeds:
            rows, per_weight = run_one_seed(
                dgp=dgp,
                seed=seed,
                base_cfg=cfg,
                weights=weights,
            )
            all_rows.append(rows)
            all_per_weight.append(per_weight)

    return (
        pd.concat(all_rows, ignore_index=True),
        pd.concat(all_per_weight, ignore_index=True),
    )


def summarize(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    metric_cols = [
        c
        for c in df.columns
        if c not in set(group_cols)
        and c not in {"seed", "model_key", "model", "dgp", "weight"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    summary = df.groupby(list(group_cols), dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def make_table_ready_latex(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compact table-ready output for the main multi-objective table.
    """
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            {
                "dgp": r["dgp"],
                "model": r["model"],
                "Obj. NLL": format_mean_std(r.get("obj_nll_mean", np.nan), r.get("obj_nll_std", np.nan)),
                "MO NLL": format_mean_std(r.get("mo_nll_mean", np.nan), r.get("mo_nll_std", np.nan)),
                "MSE($\\hat p_w,p_w^\\star$)": format_mean_std(
                    r.get("mo_prob_mse_mean", np.nan),
                    r.get("mo_prob_mse_std", np.nan),
                ),
                "Decision Regret": format_mean_std(
                    r.get("decision_regret_mean", np.nan),
                    r.get("decision_regret_std", np.nan),
                ),
            }
        )
    return pd.DataFrame(rows)


def save_outputs(
    output_dir: Path,
    prefix: str,
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    table_ready: pd.DataFrame,
    per_weight_raw: pd.DataFrame,
    per_weight_summary: pd.DataFrame,
    cfg: MOConfig,
    dgp_list: Sequence[str],
    seeds: Sequence[int],
    weights: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw.to_csv(output_dir / f"{prefix}_raw_by_seed.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_summary_numeric.csv", index=False)
    table_ready.to_csv(output_dir / f"{prefix}_table_ready_latex.csv", index=False)
    per_weight_raw.to_csv(output_dir / f"{prefix}_mo_per_weight_raw.csv", index=False)
    per_weight_summary.to_csv(output_dir / f"{prefix}_mo_per_weight_summary_numeric.csv", index=False)

    config_payload = {
        "config": asdict(cfg),
        "dgp_list": list(dgp_list),
        "seeds": list(seeds),
        "weights": weights.tolist(),
    }
    with open(output_dir / f"{prefix}_config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled multi-objective uncertainty toy experiment."
    )

    parser.add_argument("--output_dir", type=str, default="toy_mo_outputs")
    parser.add_argument("--prefix", type=str, default="toy_mo")

    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument(
        "--dgp_list",
        type=str,
        nargs="+",
        default=["objective_specific_noise", "equal_noise_control"],
        choices=["objective_specific_noise", "equal_noise_control"],
    )

    parser.add_argument("--num_objectives", type=int, default=3)
    parser.add_argument("--n_train_per_objective", type=int, default=1000)
    parser.add_argument("--n_test_per_objective", type=int, default=5000)
    parser.add_argument("--n_selection_prompts", type=int, default=2000)
    parser.add_argument("--n_candidates", type=int, default=5)

    parser.add_argument("--conflict_corr", type=float, default=-0.4)
    parser.add_argument(
        "--scorer_noise_sigmas",
        type=str,
        default="0.3,1.2,0.7",
        help="Comma-separated scorer noise sigmas. Length must match num_objectives.",
    )
    parser.add_argument("--equal_noise_sigma", type=float, default=0.7)

    parser.add_argument("--beta_obj", type=float, default=1.0)
    parser.add_argument("--beta_mo", type=float, default=1.0)
    parser.add_argument(
        "--calibration_strength_gamma",
        type=float,
        default=1.0,
        help="Downstream calibration strength. gamma=0 recovers standard scalarization; gamma=1 uses full learned precision.",
    )
    parser.add_argument("--prior_sigma_b", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")

    return parser


def main() -> None:
    args = build_argparser().parse_args()

    scorer_noise_sigmas = parse_float_tuple(args.scorer_noise_sigmas)

    cfg = MOConfig(
        num_objectives=args.num_objectives,
        n_train_per_objective=args.n_train_per_objective,
        n_test_per_objective=args.n_test_per_objective,
        n_selection_prompts=args.n_selection_prompts,
        n_candidates=args.n_candidates,
        conflict_corr=args.conflict_corr,
        scorer_noise_sigmas=scorer_noise_sigmas,
        equal_noise_sigma=args.equal_noise_sigma,
        beta_obj=args.beta_obj,
        beta_mo=args.beta_mo,
        calibration_strength_gamma=args.calibration_strength_gamma,
        prior_sigma_b=args.prior_sigma_b,
        lr=args.lr,
        steps=args.steps,
        seed=0,
        device=args.device,
    )

    weights = default_weight_grid(cfg.num_objectives)
    seeds = list(range(args.num_seeds))

    raw, per_weight_raw = run_experiments(
        dgp_list=args.dgp_list,
        seeds=seeds,
        cfg=cfg,
        weights=weights,
    )

    summary = summarize(raw, group_cols=["dgp", "model_key", "model"])
    table_ready = make_table_ready_latex(summary)

    per_weight_summary = summarize(
        per_weight_raw,
        group_cols=["dgp", "model_key", "model", "weight"],
    )

    save_outputs(
        output_dir=Path(args.output_dir),
        prefix=args.prefix,
        raw=raw,
        summary=summary,
        table_ready=table_ready,
        per_weight_raw=per_weight_raw,
        per_weight_summary=per_weight_summary,
        cfg=cfg,
        dgp_list=args.dgp_list,
        seeds=seeds,
        weights=weights,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("\n=== Summary over seeds ===")
    print(summary)

    print("\n=== Table-ready LaTeX cells ===")
    print(table_ready)

    print("\n=== Saved files ===")
    out = Path(args.output_dir)
    for name in [
        f"{args.prefix}_raw_by_seed.csv",
        f"{args.prefix}_summary_numeric.csv",
        f"{args.prefix}_table_ready_latex.csv",
        f"{args.prefix}_mo_per_weight_raw.csv",
        f"{args.prefix}_mo_per_weight_summary_numeric.csv",
        f"{args.prefix}_config.json",
    ]:
        print(out / name)


if __name__ == "__main__":
    main()
