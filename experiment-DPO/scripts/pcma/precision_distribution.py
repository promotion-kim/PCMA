import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


jsonl_path = Path("outputs/pcma/calibration_nll/direct_a_prompt_diagnostics.jsonl")
output_dir = Path("outputs/pcma/calibration_nll/figures")
output_dir.mkdir(parents=True, exist_ok=True)

rows = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

df = pd.DataFrame(rows)

print("Loaded rows:", len(df))
print(df.columns.tolist())

# ------------------------------------------------------------
# 1. Objective별 현재 row의 a_i 분포
# ------------------------------------------------------------
summary = (
    df.groupby("objective_name")["a_i"]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .reset_index()
)

print("\nObjective-wise a_i summary")
print(summary.to_string(index=False))

for obj_name, sub in df.groupby("objective_name"):
    values = sub["a_i"].dropna().to_numpy()

    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=40, density=True, alpha=0.75)
    plt.axvline(values.mean(), linestyle="--", linewidth=2, label=f"mean={values.mean():.4f}")
    plt.axvline(np.median(values), linestyle=":", linewidth=2, label=f"median={np.median(values):.4f}")
    plt.xlabel(r"$a_i(x)$")
    plt.ylabel("Density")
    plt.title(f"Distribution of direct $a_i(x)$ for objective: {obj_name}")
    plt.legend()
    plt.tight_layout()

    save_path = output_dir / f"a_i_distribution_{obj_name}.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")


# ------------------------------------------------------------
# 2. Objective별 a_i boxplot
# ------------------------------------------------------------
objective_names = sorted(df["objective_name"].unique())
data = [
    df.loc[df["objective_name"] == obj, "a_i"].dropna().to_numpy()
    for obj in objective_names
]

plt.figure(figsize=(7, 4))
plt.boxplot(data, labels=objective_names, showmeans=True)
plt.ylabel(r"$a_i(x)$")
plt.title(r"Objective-wise distribution of direct $a_i(x)$")
plt.tight_layout()

save_path = output_dir / "a_i_boxplot_by_objective.png"
plt.savefig(save_path, dpi=200)
plt.close()

print(f"Saved: {save_path}")


# ------------------------------------------------------------
# 3. 같은 prompt에 대해 a_help와 a_safe 분포 비교
# ------------------------------------------------------------
# JSONL에는 각 row마다 a_help, a_safe가 저장되어 있으므로,
# row 단위로 보면 중복 prompt가 있을 수 있음.
# prompt_index 기준으로 중복 제거해서 prompt-level distribution을 봄.
# ------------------------------------------------------------
required_cols = {"prompt_index", "a_help", "a_safe"}
if required_cols.issubset(set(df.columns)):
    prompt_df = df.drop_duplicates(subset=["prompt_index"]).copy()

    print("\nPrompt-level a_help/a_safe summary")
    for col in ["a_help", "a_safe"]:
        values = prompt_df[col].dropna().to_numpy()
        print(
            f"{col}: "
            f"count={len(values)}, "
            f"mean={values.mean():.4f}, "
            f"std={values.std():.4f}, "
            f"min={values.min():.4f}, "
            f"median={np.median(values):.4f}, "
            f"max={values.max():.4f}"
        )

    plt.figure(figsize=(7, 4))
    plt.hist(prompt_df["a_help"].dropna().to_numpy(), bins=40, density=True, alpha=0.5, label="help")
    plt.hist(prompt_df["a_safe"].dropna().to_numpy(), bins=40, density=True, alpha=0.5, label="safe")
    plt.xlabel(r"$a_i(x)$")
    plt.ylabel("Density")
    plt.title(r"Prompt-level distributions of $a_{\mathrm{help}}(x)$ and $a_{\mathrm{safe}}(x)$")
    plt.legend()
    plt.tight_layout()

    save_path = output_dir / "a_help_vs_a_safe_distribution.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")


# ------------------------------------------------------------
# 4. Relative reliability: delta_safe = a_safe - mean(a_help, a_safe)
# ------------------------------------------------------------
if {"delta_safe", "alpha_safe"}.issubset(set(df.columns)):
    prompt_df = df.drop_duplicates(subset=["prompt_index"]).copy()

    plt.figure(figsize=(7, 4))
    plt.hist(prompt_df["delta_safe"].dropna().to_numpy(), bins=40, density=True, alpha=0.75)
    plt.axvline(0.0, linestyle="--", linewidth=2, label="no relative shift")
    plt.xlabel(r"$\delta_{\mathrm{safe}}(x)$")
    plt.ylabel("Density")
    plt.title(r"Distribution of relative safety reliability $\delta_{\mathrm{safe}}(x)$")
    plt.legend()
    plt.tight_layout()

    save_path = output_dir / "delta_safe_distribution.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")

    plt.figure(figsize=(7, 4))
    plt.hist(prompt_df["alpha_safe"].dropna().to_numpy(), bins=40, density=True, alpha=0.75)
    plt.axvline(0.5, linestyle="--", linewidth=2, label="original weight=0.5")
    plt.xlabel(r"$\alpha_{\mathrm{safe}}(x)$")
    plt.ylabel("Density")
    plt.title(r"Distribution of calibrated safety weight $\alpha_{\mathrm{safe}}(x)$")
    plt.legend()
    plt.tight_layout()

    save_path = output_dir / "alpha_safe_distribution.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")