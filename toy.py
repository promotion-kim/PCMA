# toy_noisy_utility_ablation.py
#
# Run examples:
#   python toy_noisy_utility_ablation.py
#   python toy_noisy_utility_ablation.py --num_objectives 4 --dim 16 --num_seeds 50
#   python toy_noisy_utility_ablation.py --num_objectives 8 --dim 32 --num_seeds 50
#   python toy_noisy_utility_ablation.py --n_obj 50 --num_seeds 100
#
# Main idea:
#   Hidden oracle:
#       z_w = sum_i w_i (d_i + eps_i)
#       eps_i ~ N(0, s_i^2(x))
#       P(y1 > y2) = E[sigmoid(z_w)]
#                  ~= sigmoid(mu_w / sqrt(1 + pi v_w / 8))
#
#   Proposed target:
#       Estimate objective-specific reward heads from finite data.
#       Use posterior mean/variance:
#           mu_i = x^T theta_i_hat
#           v_i  = x^T Sigma_i x
#       Then scalarize noisy utilities:
#           rho = sigmoid(sum_i w_i mu_i / sqrt(1 + pi sum_i w_i^2 v_i / 8))
#
#   Ablations:
#       no_uncertainty:      remove v_i
#       logit_pooling:       old BPP-MOA style, sum_i w_i ell_i
#       linear_pooling:      sum_i w_i sigmoid(ell_i)
#       hard_target:         binarize proposed target

import argparse
import math
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm


@dataclass
class Head:
    theta: torch.Tensor
    diag_precision: torch.Tensor


def set_fast_cpu():
    torch.set_num_threads(1)


def sigmoid(x):
    return torch.sigmoid(x)


def normalize_rows(x, scale=1.0, eps=1e-8):
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps) * scale


def make_true_objectives(num_objectives, dim, true_scale=3.0, seed=0):
    """
    Hidden deterministic objective-specific utilities.

    true_thetas[i] defines objective i:
        d_i = delta_x^T true_thetas[i]

    Each objective mainly cares about its own feature block.
    Other blocks mildly conflict.
    """
    g = torch.Generator().manual_seed(seed)
    true_thetas = torch.randn(num_objectives, dim, generator=g) * 0.4

    block = max(dim // num_objectives, 1)

    for i in range(num_objectives):
        start = i * block
        end = min((i + 1) * block, dim)

        # Objective i likes its own block.
        true_thetas[i, start:end] += 2.0

        # Mild conflict with other objective blocks.
        for j in range(num_objectives):
            if j == i:
                continue
            s = j * block
            e = min((j + 1) * block, dim)
            true_thetas[i, s:e] -= 0.5 / max(num_objectives - 1, 1)

    return normalize_rows(true_thetas, scale=true_scale)


def make_objective_coverages(num_objectives, dim, low_coverage=0.12):
    """
    Objective i has good data coverage on its own block and weak coverage elsewhere.
    This produces objective-dependent uncertainty.
    """
    coverages = torch.full((num_objectives, dim), float(low_coverage))
    block = max(dim // num_objectives, 1)

    for i in range(num_objectives):
        start = i * block
        end = min((i + 1) * block, dim)
        coverages[i, start:end] = 1.0

    return coverages


def true_noise_variance(delta_x, coverage, noise_base=0.15, noise_scale=1.0):
    """
    Pair-specific true objective noise variance.

    If objective i has poor coverage on dimensions active in delta_x,
    then s_i^2(delta_x) is larger.

    This is the hidden data-generating noise, not the learner's posterior variance.
    """
    uncovered = 1.0 - coverage
    raw = (delta_x.pow(2) * uncovered.pow(2)).mean(dim=-1)
    return noise_base**2 + noise_scale**2 * raw


def noisy_bt_prob(mean, var):
    """
    Logistic-Gaussian approximation:
        E_{z ~ N(mean,var)} sigmoid(z)
        ~= sigmoid(mean / sqrt(1 + pi var / 8))
    """
    return sigmoid(mean / torch.sqrt(1.0 + (math.pi / 8.0) * var.clamp_min(0.0)))


def true_objective_prob(delta_x, true_theta, coverage, noise_base, noise_scale):
    """
    Objective-specific preference probability:
        d_i = delta_x^T theta_i
        eps_i ~ N(0, s_i^2(delta_x))
        P_i(y1 > y2) = E sigmoid(d_i + eps_i)
    """
    mean = delta_x @ true_theta
    var = true_noise_variance(
        delta_x,
        coverage=coverage,
        noise_base=noise_base,
        noise_scale=noise_scale,
    )
    return noisy_bt_prob(mean, var)


def true_multiobjective_oracle_prob(
    delta_x,
    true_thetas,
    coverages,
    weights,
    noise_base,
    noise_scale,
):
    """
    Utility-weighted noisy oracle:

        z_w = sum_i w_i (d_i + eps_i)

    where:
        d_i = delta_x^T theta_i
        eps_i ~ N(0, s_i^2(delta_x))

    If eps_i are independent:
        mean_w = sum_i w_i d_i
        var_w  = sum_i w_i^2 s_i^2(delta_x)

    Then:
        P(y1 > y2 | w) = E sigmoid(z_w)
                       ~= sigmoid(mean_w / sqrt(1 + pi var_w / 8))
    """
    weights = weights / weights.sum()
    m = true_thetas.shape[0]

    means = []
    variances = []

    for i in range(m):
        mean_i = delta_x @ true_thetas[i]
        var_i = true_noise_variance(
            delta_x,
            coverage=coverages[i],
            noise_base=noise_base,
            noise_scale=noise_scale,
        )
        means.append(mean_i)
        variances.append(var_i)

    means = torch.stack(means, dim=1)      # [n, m]
    variances = torch.stack(variances, dim=1)  # [n, m]

    mean_w = (means * weights.view(1, -1)).sum(dim=1)
    var_w = (variances * weights.view(1, -1).pow(2)).sum(dim=1)

    return noisy_bt_prob(mean_w, var_w)


def make_objective_dataset(
    n,
    dim,
    true_theta,
    coverage,
    seed,
    noise_base,
    noise_scale,
):
    """
    Generate objective-specific pairwise preference data.

    delta_x = phi(x,y1) - phi(x,y2)
    label = 1 means y1 is preferred to y2 for this objective.

    Data distribution also uses coverage:
    objective i observes richer variation on dimensions it covers well.
    """
    g = torch.Generator().manual_seed(seed)

    delta_x = torch.randn(n, dim, generator=g) * coverage
    prob = true_objective_prob(
        delta_x,
        true_theta=true_theta,
        coverage=coverage,
        noise_base=noise_base,
        noise_scale=noise_scale,
    )
    label = torch.bernoulli(prob, generator=g)

    return delta_x, label


def fit_map_logistic_laplace(
    delta_x,
    label,
    prior_precision=1.0,
    lr=0.05,
    steps=500,
):
    """
    Fit objective-specific reward head using BT/logistic likelihood.

    MAP:
        min_theta sum_n BCEWithLogits(delta_x_n @ theta, label_n)
                  + 0.5 * prior_precision * ||theta||^2

    Diagonal Laplace precision:
        H_diag = prior_precision
               + sum_n sigmoid(u_n)(1-sigmoid(u_n)) delta_x_n^2
    """
    dim = delta_x.shape[-1]
    theta = torch.zeros(dim, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)

    for _ in range(steps):
        logits = delta_x @ theta
        nll = F.binary_cross_entropy_with_logits(logits, label, reduction="sum")
        prior = 0.5 * float(prior_precision) * theta.pow(2).sum()
        loss = nll + prior

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = delta_x @ theta
        p = sigmoid(logits)
        curvature_weight = p * (1.0 - p)
        diag_precision = float(prior_precision) + (
            curvature_weight.unsqueeze(-1) * delta_x.pow(2)
        ).sum(dim=0)

    return Head(theta=theta.detach(), diag_precision=diag_precision.detach())


def posterior_mean_var(delta_x, head, use_uncertainty=True):
    """
    Learner's posterior predictive mean and variance:
        mu_i = delta_x^T theta_hat
        v_i  = delta_x^T Sigma_i delta_x

    With diagonal precision:
        Sigma_diag = 1 / diag_precision
    """
    mu = delta_x @ head.theta

    if not use_uncertainty:
        var = torch.zeros_like(mu)
    else:
        var = (delta_x.pow(2) / head.diag_precision.clamp_min(1e-8)).sum(dim=-1)

    return mu, var


def build_training_target(
    delta_x,
    heads,
    weights,
    variant,
):
    """
    Variants:

    proposed:
        scalarize posterior utilities and aggregate uncertainty:
            mu_w = sum_i w_i mu_i
            v_w  = sum_i w_i^2 v_i
            rho  = sigmoid(mu_w / sqrt(1 + pi v_w / 8))

    no_uncertainty:
        same but v_i = 0:
            rho = sigmoid(sum_i w_i mu_i)

    logit_pooling:
        old BPP-MOA-style:
            ell_i = mu_i / sqrt(1 + pi v_i / 8)
            rho = sigmoid(sum_i w_i ell_i)

    linear_pooling:
        probability pooling:
            rho = sum_i w_i sigmoid(ell_i)

    hard_target:
        binarize proposed target:
            rho = I[rho_proposed > 0.5]
    """
    weights = weights / weights.sum()
    m = len(heads)

    use_uncertainty = variant != "no_uncertainty"

    mus = []
    vars_ = []
    ells = []

    for i in range(m):
        mu_i, var_i = posterior_mean_var(
            delta_x,
            heads[i],
            use_uncertainty=use_uncertainty,
        )
        ell_i = mu_i / torch.sqrt(1.0 + (math.pi / 8.0) * var_i.clamp_min(0.0))

        mus.append(mu_i)
        vars_.append(var_i)
        ells.append(ell_i)

    mus = torch.stack(mus, dim=1)      # [n, m]
    vars_ = torch.stack(vars_, dim=1)  # [n, m]
    ells = torch.stack(ells, dim=1)    # [n, m]

    if variant in {"proposed", "no_uncertainty", "hard_target"}:
        mu_w = (mus * weights.view(1, -1)).sum(dim=1)
        var_w = (vars_ * weights.view(1, -1).pow(2)).sum(dim=1)
        rho = noisy_bt_prob(mu_w, var_w)

    elif variant == "logit_pooling":
        pooled_logit = (ells * weights.view(1, -1)).sum(dim=1)
        rho = sigmoid(pooled_logit)

    elif variant == "linear_pooling":
        rho_i = sigmoid(ells)
        rho = (rho_i * weights.view(1, -1)).sum(dim=1)

    else:
        raise ValueError(f"Unknown variant={variant}")

    if variant == "hard_target":
        rho = (rho > 0.5).float()

    return rho.clamp(1e-6, 1.0 - 1e-6)


def train_toy_policy(delta_x, target, lr=0.1, steps=400):
    """
    Toy policy:
        policy_logit = delta_x^T theta_policy

    Trained with soft-label BCE.
    """
    dim = delta_x.shape[-1]
    theta = torch.zeros(dim, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)

    for _ in range(steps):
        logits = delta_x @ theta
        loss = F.binary_cross_entropy_with_logits(logits, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return theta.detach()


def evaluate_policy(delta_x, policy_theta, oracle_prob):
    """
    Evaluate policy against hidden noisy utility oracle.
    """
    pred_prob = sigmoid(delta_x @ policy_theta).clamp(1e-6, 1.0 - 1e-6)
    oracle_prob = oracle_prob.clamp(1e-6, 1.0 - 1e-6)

    nll = -(
        oracle_prob * torch.log(pred_prob)
        + (1.0 - oracle_prob) * torch.log(1.0 - pred_prob)
    ).mean()

    brier = (pred_prob - oracle_prob).pow(2).mean()

    hard_agreement = (
        (pred_prob > 0.5) == (oracle_prob > 0.5)
    ).float().mean()

    return {
        "NLL": float(nll),
        "Brier": float(brier),
        "HardAgreement": float(hard_agreement),
    }


def make_weights(num_objectives, mode="uniform", seed=0):
    if mode == "uniform":
        return torch.ones(num_objectives) / num_objectives

    if mode == "random":
        g = torch.Generator().manual_seed(seed)
        w = torch.rand(num_objectives, generator=g)
        return w / w.sum()

    if mode == "first_heavy":
        w = torch.ones(num_objectives)
        w[0] = num_objectives
        return w / w.sum()

    raise ValueError(f"Unknown weight_mode={mode}")


def run_one_seed(args, seed):
    dim = args.dim
    m = args.num_objectives

    weights = make_weights(
        num_objectives=m,
        mode=args.weight_mode,
        seed=seed * 100 + 7,
    )

    true_thetas = make_true_objectives(
        num_objectives=m,
        dim=dim,
        true_scale=args.true_scale,
        seed=seed * 100 + 1,
    )

    coverages = make_objective_coverages(
        num_objectives=m,
        dim=dim,
        low_coverage=args.low_coverage,
    )

    heads = []

    for i in range(m):
        x_i, y_i = make_objective_dataset(
            n=args.n_obj,
            dim=dim,
            true_theta=true_thetas[i],
            coverage=coverages[i],
            seed=seed * 1000 + 10 + i,
            noise_base=args.noise_base,
            noise_scale=args.noise_scale,
        )

        head_i = fit_map_logistic_laplace(
            x_i,
            y_i,
            prior_precision=args.prior_precision,
            lr=args.reward_lr,
            steps=args.reward_steps,
        )
        heads.append(head_i)

    g_train = torch.Generator().manual_seed(seed * 1000 + 101)
    g_test = torch.Generator().manual_seed(seed * 1000 + 102)

    # Policy training/eval distribution.
    # This is broader than objective-specific data, so uncertainty matters.
    x_train = torch.randn(args.n_policy_train, dim, generator=g_train)
    x_test = torch.randn(args.n_test, dim, generator=g_test)

    oracle_prob = true_multiobjective_oracle_prob(
        x_test,
        true_thetas=true_thetas,
        coverages=coverages,
        weights=weights,
        noise_base=args.noise_base,
        noise_scale=args.noise_scale,
    )

    variants = [
        "proposed",
        "no_uncertainty",
        "logit_pooling",
        "linear_pooling",
        "hard_target",
    ]

    rows = []

    for variant in variants:
        target = build_training_target(
            x_train,
            heads=heads,
            weights=weights,
            variant=variant,
        )

        policy_theta = train_toy_policy(
            x_train,
            target,
            lr=args.policy_lr,
            steps=args.policy_steps,
        )

        metrics = evaluate_policy(
            x_test,
            policy_theta=policy_theta,
            oracle_prob=oracle_prob,
        )

        rows.append(
            {
                "seed": seed,
                "variant": variant,
                "target_mean": float(target.mean()),
                **metrics,
            }
        )

    return rows


def summarize(df):
    metric_cols = ["target_mean", "NLL", "Brier", "HardAgreement"]

    summary = (
        df.groupby("variant")[metric_cols]
        .agg(["mean", "std"])
        .round(4)
    )

    pivot_nll = df.pivot(index="seed", columns="variant", values="NLL")
    pivot_brier = df.pivot(index="seed", columns="variant", values="Brier")
    pivot_agree = df.pivot(index="seed", columns="variant", values="HardAgreement")

    win_rows = []

    for variant in pivot_nll.columns:
        if variant == "proposed":
            continue

        win_rows.append(
            {
                "comparison": f"proposed vs {variant}",
                "NLL_win_rate": float((pivot_nll["proposed"] < pivot_nll[variant]).mean()),
                "Brier_win_rate": float((pivot_brier["proposed"] < pivot_brier[variant]).mean()),
                "HardAgreement_win_rate": float((pivot_agree["proposed"] > pivot_agree[variant]).mean()),
            }
        )

    win_df = pd.DataFrame(win_rows).round(4)

    return summary, win_df


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_seeds", type=int, default=50)
    parser.add_argument("--num_objectives", type=int, default=2)
    parser.add_argument("--dim", type=int, default=8)

    parser.add_argument("--weight_mode", type=str, default="uniform",
                        choices=["uniform", "random", "first_heavy"])

    # objective-specific labels per objective
    parser.add_argument("--n_obj", type=int, default=100)

    # policy distillation and evaluation sample sizes
    parser.add_argument("--n_policy_train", type=int, default=1000)
    parser.add_argument("--n_test", type=int, default=3000)

    # hidden true utility/noise
    parser.add_argument("--true_scale", type=float, default=3.0)
    parser.add_argument("--low_coverage", type=float, default=0.12)
    parser.add_argument("--noise_base", type=float, default=0.15)
    parser.add_argument("--noise_scale", type=float, default=1.0)

    # Bayesian reward head fitting
    parser.add_argument("--prior_precision", type=float, default=1.0)
    parser.add_argument("--reward_lr", type=float, default=0.05)
    parser.add_argument("--reward_steps", type=int, default=500)

    # toy policy training
    parser.add_argument("--policy_lr", type=float, default=0.1)
    parser.add_argument("--policy_steps", type=int, default=400)

    parser.add_argument("--save_csv", type=str, default=None)

    return parser.parse_args()


def main():
    set_fast_cpu()
    args = parse_args()

    if args.dim < args.num_objectives:
        raise ValueError("--dim should be >= --num_objectives.")

    all_rows = []

    for seed in tqdm(range(args.num_seeds)):
        all_rows.extend(run_one_seed(args, seed))

    df = pd.DataFrame(all_rows)
    summary, win_df = summarize(df)

    print("\n=== Noisy Utility Multi-Objective Toy Ablation ===")
    print(f"num_objectives   = {args.num_objectives}")
    print(f"dim              = {args.dim}")
    print(f"weight_mode      = {args.weight_mode}")
    print(f"num_seeds        = {args.num_seeds}")
    print(f"n_obj/objective  = {args.n_obj}")
    print(f"noise_base       = {args.noise_base}")
    print(f"noise_scale      = {args.noise_scale}")

    print("\nOracle:")
    print("  z_w = sum_i w_i (d_i + eps_i)")
    print("  eps_i ~ N(0, s_i^2(x))")
    print("  P(y1 > y2) ~= sigmoid(mu_w / sqrt(1 + pi v_w / 8))")

    print("\nProposed target:")
    print("  rho = sigmoid(sum_i w_i mu_i / sqrt(1 + pi sum_i w_i^2 v_i / 8))")

    print("\n[Mean / Std over seeds]")
    print(summary)

    print("\n[How often proposed beats each ablation]")
    print(win_df.to_string(index=False))

    print("\nInterpretation:")
    print("- NLL/Brier: lower is better.")
    print("- HardAgreement: higher is better, but ignores calibration.")
    print("- target_mean is diagnostic, not a performance metric.")
    print("- n_obj is the number of objective-specific preference labels per objective.")

    if args.save_csv is not None:
        df.to_csv(args.save_csv, index=False)
        print(f"\nSaved raw per-seed results to: {args.save_csv}")


if __name__ == "__main__":
    main()
