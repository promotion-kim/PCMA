from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
) -> Dataset:
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
    if not args.use_posterior_variance:
        # Very large precision makes v approximately zero, giving the MAP ablation.
        prec_h = torch.full_like(prec_h, 1e30)
        prec_s = torch.full_like(prec_s, 1e30)

    cols = {
        "bpp_rho": [],
        "bpp_logit": [],
        "bpp_mu_helpful": [],
        "bpp_var_helpful": [],
        "bpp_ell_helpful": [],
        "bpp_mu_harmless": [],
        "bpp_var_harmless": [],
        "bpp_ell_harmless": [],
    }

    for batch in tqdm(loader, desc=f"targets:{split_name}"):
        _, _, delta_h = pair_features(model, tokenizer, batch, args.prompt_template, args.max_length, device)
        mu_h, var_h, ell_h = laplace_pair_stats(delta_h, theta_h, prec_h, args.variance_scale)
        mu_s, var_s, ell_s = laplace_pair_stats(delta_h, theta_s, prec_s, args.variance_scale)
        pooled_logit = float(args.w_helpful) * ell_h + float(args.w_harmless) * ell_s
        rho = torch.sigmoid(pooled_logit)

        cols["bpp_rho"].extend(rho.cpu().float().tolist())
        cols["bpp_logit"].extend(pooled_logit.cpu().float().tolist())
        cols["bpp_mu_helpful"].extend(mu_h.cpu().float().tolist())
        cols["bpp_var_helpful"].extend(var_h.cpu().float().tolist())
        cols["bpp_ell_helpful"].extend(ell_h.cpu().float().tolist())
        cols["bpp_mu_harmless"].extend(mu_s.cpu().float().tolist())
        cols["bpp_var_harmless"].extend(var_s.cpu().float().tolist())
        cols["bpp_ell_harmless"].extend(ell_s.cpu().float().tolist())

    out = dataset
    for name, values in cols.items():
        if name in out.column_names:
            out = out.remove_columns(name)
        out = out.add_column(name, values)
    return out


def main() -> None:
    args = tyro.cli(ScriptArguments)
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

    train_out = add_targets_for_split("train", train_dataset, model, tokenizer, helpful_head, harmless_head, args, device)
    eval_out = add_targets_for_split("validation", eval_dataset, model, tokenizer, helpful_head, harmless_head, args, device)

    train_out.save_to_disk(str(out_dir / "train"))
    eval_out.save_to_disk(str(out_dir / "validation"))

    meta = {
        "target_dataset_name": args.target_dataset_name,
        "sft_model_name": args.sft_model_name,
        "helpful_head_path": args.helpful_head_path,
        "harmless_head_path": args.harmless_head_path,
        "w_helpful": args.w_helpful,
        "w_harmless": args.w_harmless,
        "variance_scale": args.variance_scale,
        "use_posterior_variance": args.use_posterior_variance,
        "formula": "rho=sigmoid(w_h*mu_h/sqrt(1+pi*v_h/8)+w_s*mu_s/sqrt(1+pi*v_s/8))",
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[BPP-MOA] saved target dataset: {out_dir}")


if __name__ == "__main__":
    main()
