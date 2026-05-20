from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import torch
import tyro
from datasets import Dataset, disable_caching
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils import set_seed
from scripts.bppmoa.bppmoa_last_layer import (
    laplace_pair_stats,
    load_laplace_head,
    pair_features,
    str_to_torch_dtype,
)


TargetMode = Literal["full", "constant_variance", "map_only"]


@dataclass
class ScriptArguments:
    sft_model_name: str = field(default="PKU-Alignment/alpaca-7b-reproduced")
    target_dataset_name: str = field(default="PKU-Alignment/PKU-SafeRLHF-10K-better")
    helpful_head_path: str = field(default="./output/bppmoa/reward_heads/better/laplace_head.pt")
    harmless_head_path: str = field(default="./output/bppmoa/reward_heads/safer/laplace_head.pt")
    output_dir: str = field(default="./output/bppmoa/targets/w0.5")

    w_helpful: float = field(default=0.5)
    w_harmless: float = field(default=0.5)
    variance_scale: float = field(default=1.0)

    # full: original BPP-MOA, using pair-specific posterior variance v_i(x,y+,y-)
    # constant_variance: replace v_i(x,y+,y-) by split-level mean \bar{v}_i
    # map_only: remove posterior variance entirely, ell_i = mu_i
    target_mode: TargetMode = field(default="full")

    # Backward-compatible switch. If an old script passes
    # --use_posterior_variance False while target_mode is left as full, we map it
    # to target_mode="map_only" in main(). Prefer --target_mode for new runs.
    use_posterior_variance: bool = field(default=True)

    prompt_template: str = field(default=DEFAULT_PROMPT_TEMPLATE)
    sanity_check: bool = field(default=False)
    dataset_caching: bool = field(default=False)
    seed: int = field(default=0)
    max_length: int = field(default=512)
    per_device_eval_batch_size: int = field(default=4)
    torch_dtype: str = field(default="bf16")
    use_flash_attention_2: bool = field(default=False)
    trust_remote_code: bool = field(default=True)
    limit_examples: Optional[int] = field(default=None)


def collate_preference(batch: List[Dict]) -> Dict[str, List[str]]:
    return {
        "raw_prompt": [ex["raw_prompt"] for ex in batch],
        "chosen": [ex["chosen"] for ex in batch],
        "rejected": [ex["rejected"] for ex in batch],
    }


def _attenuate_logits(
    mu: torch.Tensor,
    raw_var: torch.Tensor,
    target_mode: TargetMode,
    const_var: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return uncertainty-attenuated logit and the variance actually used.

    raw_var is already scaled by args.variance_scale because it comes from
    laplace_pair_stats(..., variance_scale=args.variance_scale).
    """
    if target_mode == "full":
        used_var = raw_var.clamp_min(0.0)
    elif target_mode == "constant_variance":
        if const_var is None:
            raise ValueError("const_var must be provided for target_mode='constant_variance'.")
        used_var = torch.full_like(raw_var, float(const_var)).clamp_min(0.0)
    elif target_mode == "map_only":
        used_var = torch.zeros_like(raw_var)
    else:
        raise ValueError(f"Unknown target_mode={target_mode}")

    ell = mu / torch.sqrt(1.0 + (math.pi / 8.0) * used_var)
    return ell, used_var


@torch.no_grad()
def add_targets_for_split(
    split_name: str,
    dataset: Dataset,
    model,
    tokenizer,
    helpful_head: Dict,
    harmless_head: Dict,
    args: ScriptArguments,
    device: torch.device,
    constant_variance_override: Optional[Tuple[float, float]] = None,
) -> Tuple[Dataset, Dict[str, float]]:
    if args.limit_examples is not None:
        dataset = dataset.select(range(min(args.limit_examples, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_preference,
    )

    theta_h = helpful_head["theta"].to(device)
    prec_h = helpful_head["diag_precision"].to(device)
    theta_s = harmless_head["theta"].to(device)
    prec_s = harmless_head["diag_precision"].to(device)

    # First pass: compute all MAP gaps and raw posterior variances.
    # We need split-level mean variances for the constant_variance ablation.
    batch_stats: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    var_h_sum = 0.0
    var_s_sum = 0.0
    num_examples = 0

    for batch in tqdm(loader, desc=f"stats:{split_name}"):
        _, _, delta_h = pair_features(model, tokenizer, batch, args.prompt_template, args.max_length, device)
        mu_h, var_h, _ = laplace_pair_stats(delta_h, theta_h, prec_h, args.variance_scale)
        mu_s, var_s, _ = laplace_pair_stats(delta_h, theta_s, prec_s, args.variance_scale)

        batch_stats.append((
            mu_h.detach().cpu().float(),
            var_h.detach().cpu().float(),
            mu_s.detach().cpu().float(),
            var_s.detach().cpu().float(),
        ))
        var_h_sum += float(var_h.sum().item())
        var_s_sum += float(var_s.sum().item())
        num_examples += int(var_h.numel())

    if num_examples == 0:
        raise ValueError(f"Empty dataset split: {split_name}")

    raw_const_var_h = var_h_sum / num_examples
    raw_const_var_s = var_s_sum / num_examples

    # For strict ablations, estimate the constant variance on the training split
    # and reuse it for validation if an override is provided.
    if constant_variance_override is None:
        const_var_h = raw_const_var_h
        const_var_s = raw_const_var_s
    else:
        const_var_h, const_var_s = constant_variance_override

    cols = {
        "bpp_rho": [],
        "bpp_logit": [],
        "bpp_mu_helpful": [],
        # Raw posterior variance from the Laplace head, useful for diagnostics.
        "bpp_var_helpful": [],
        # Variance actually used in the target formula: raw, constant mean, or zero.
        "bpp_used_var_helpful": [],
        "bpp_ell_helpful": [],
        "bpp_mu_harmless": [],
        "bpp_var_harmless": [],
        "bpp_used_var_harmless": [],
        "bpp_ell_harmless": [],
    }

    # Second pass over cached scalar statistics: construct the chosen ablation target.
    for mu_h, var_h, mu_s, var_s in tqdm(batch_stats, desc=f"targets:{split_name}"):
        ell_h, used_var_h = _attenuate_logits(
            mu_h,
            var_h,
            args.target_mode,
            const_var=const_var_h,
        )
        ell_s, used_var_s = _attenuate_logits(
            mu_s,
            var_s,
            args.target_mode,
            const_var=const_var_s,
        )

        pooled_logit = float(args.w_helpful) * ell_h + float(args.w_harmless) * ell_s
        rho = torch.sigmoid(pooled_logit)

        cols["bpp_rho"].extend(rho.float().tolist())
        cols["bpp_logit"].extend(pooled_logit.float().tolist())
        cols["bpp_mu_helpful"].extend(mu_h.float().tolist())
        cols["bpp_var_helpful"].extend(var_h.float().tolist())
        cols["bpp_used_var_helpful"].extend(used_var_h.float().tolist())
        cols["bpp_ell_helpful"].extend(ell_h.float().tolist())
        cols["bpp_mu_harmless"].extend(mu_s.float().tolist())
        cols["bpp_var_harmless"].extend(var_s.float().tolist())
        cols["bpp_used_var_harmless"].extend(used_var_s.float().tolist())
        cols["bpp_ell_harmless"].extend(ell_s.float().tolist())

    out = dataset
    for name, values in cols.items():
        if name in out.column_names:
            out = out.remove_columns(name)
        out = out.add_column(name, values)

    split_meta = {
        "num_examples": num_examples,
        "raw_var_helpful_mean": raw_const_var_h,
        "raw_var_harmless_mean": raw_const_var_s,
        "const_var_helpful_used_for_constant_variance": const_var_h,
        "const_var_harmless_used_for_constant_variance": const_var_s,
        "used_var_helpful_mean": const_var_h if args.target_mode == "constant_variance" else (0.0 if args.target_mode == "map_only" else raw_const_var_h),
        "used_var_harmless_mean": const_var_s if args.target_mode == "constant_variance" else (0.0 if args.target_mode == "map_only" else raw_const_var_s),
    }
    return out, split_meta


def main() -> None:
    args = tyro.cli(ScriptArguments)

    # Backward compatibility with the previous boolean ablation.
    if args.target_mode == "full" and not args.use_posterior_variance:
        args.target_mode = "map_only"

    set_seed(args.seed)
    if not args.dataset_caching:
        disable_caching()

    w_sum = args.w_helpful + args.w_harmless
    if abs(w_sum - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1. Got w_helpful+w_harmless={w_sum}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = str_to_torch_dtype(args.torch_dtype)

    helpful_head = load_laplace_head(args.helpful_head_path, map_location=device)
    harmless_head = load_laplace_head(args.harmless_head_path, map_location=device)

    print(f"[BPP-MOA] target_mode={args.target_mode}")
    print(f"[BPP-MOA] loading frozen feature model: {args.sft_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_name,
        torch_dtype=dtype,
        use_flash_attention_2=args.use_flash_attention_2,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_name, trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    rdp = DATASET_CONFIGS[args.target_dataset_name](
        prompt_template=args.prompt_template,
        sanity_check=args.sanity_check,
    )
    train_dataset = rdp.get_preference_dataset(split="train")
    eval_dataset = rdp.get_preference_dataset(split="validation")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out, train_meta = add_targets_for_split("train", train_dataset, model, tokenizer, helpful_head, harmless_head, args, device)

    constant_variance_override = None
    if args.target_mode == "constant_variance":
        constant_variance_override = (
            float(train_meta["raw_var_helpful_mean"]),
            float(train_meta["raw_var_harmless_mean"]),
        )

    eval_out, eval_meta = add_targets_for_split(
        "validation",
        eval_dataset,
        model,
        tokenizer,
        helpful_head,
        harmless_head,
        args,
        device,
        constant_variance_override=constant_variance_override,
    )

    train_out.save_to_disk(str(out_dir / "train"))
    eval_out.save_to_disk(str(out_dir / "validation"))

    if args.target_mode == "full":
        formula = "rho=sigmoid(w_h*mu_h/sqrt(1+pi*v_h(x,y+,y-)/8)+w_s*mu_s/sqrt(1+pi*v_s(x,y+,y-)/8))"
    elif args.target_mode == "constant_variance":
        formula = "rho=sigmoid(w_h*mu_h/sqrt(1+pi*mean(v_h)/8)+w_s*mu_s/sqrt(1+pi*mean(v_s)/8))"
    else:
        formula = "rho=sigmoid(w_h*mu_h+w_s*mu_s)"

    meta = {
        "target_dataset_name": args.target_dataset_name,
        "sft_model_name": args.sft_model_name,
        "helpful_head_path": args.helpful_head_path,
        "harmless_head_path": args.harmless_head_path,
        "w_helpful": args.w_helpful,
        "w_harmless": args.w_harmless,
        "variance_scale": args.variance_scale,
        "target_mode": args.target_mode,
        "use_posterior_variance": args.target_mode != "map_only",
        "formula": formula,
        "split_stats": {
            "train": train_meta,
            "validation": eval_meta,
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[BPP-MOA] saved target dataset: {out_dir}")


if __name__ == "__main__":
    main()
