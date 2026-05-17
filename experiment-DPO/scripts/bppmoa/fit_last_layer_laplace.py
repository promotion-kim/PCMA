from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import tyro
from datasets import disable_caching
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils import set_seed
from scripts.bppmoa.bppmoa_last_layer import (
    pair_features,
    save_laplace_head,
    str_to_torch_dtype,
)


@dataclass
class ScriptArguments:
    sft_model_name: str = field(default="PKU-Alignment/alpaca-7b-reproduced")
    dataset_name: str = field(default="PKU-Alignment/PKU-SafeRLHF-10K-better")
    output_dir: str = field(default="./output/bppmoa/reward_head")
    prompt_template: str = field(default=DEFAULT_PROMPT_TEMPLATE)
    sanity_check: bool = field(default=False)
    dataset_caching: bool = field(default=False)
    seed: int = field(default=0)

    max_length: int = field(default=512)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    num_train_epochs: int = field(default=3)
    learning_rate: float = field(default=1e-3)
    prior_precision: float = field(default=1.0)
    max_grad_norm: float = field(default=1.0)
    logging_steps: int = field(default=20)

    torch_dtype: str = field(default="bf16")
    use_flash_attention_2: bool = field(default=False)
    trust_remote_code: bool = field(default=True)

    limit_train_examples: Optional[int] = field(default=None)
    limit_eval_examples: Optional[int] = field(default=None)


def collate_preference(batch: List[Dict]) -> Dict[str, List[str]]:
    return {
        "raw_prompt": [ex["raw_prompt"] for ex in batch],
        "chosen": [ex["chosen"] for ex in batch],
        "rejected": [ex["rejected"] for ex in batch],
    }


@torch.no_grad()
def evaluate_head(
    model,
    tokenizer,
    theta: torch.Tensor,
    dataloader: DataLoader,
    prompt_template: str,
    max_length: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total = 0
    for batch in tqdm(dataloader, desc="eval", leave=False):
        _, _, delta_h = pair_features(model, tokenizer, batch, prompt_template, max_length, device)
        logits = delta_h.matmul(theta)
        loss = -F.logsigmoid(logits)
        total_loss += loss.sum().item()
        total_correct += (logits > 0).float().sum().item()
        total += logits.numel()
    return {
        "nll": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
    }


@torch.no_grad()
def compute_diag_precision(
    model,
    tokenizer,
    theta: torch.Tensor,
    dataloader: DataLoader,
    prompt_template: str,
    max_length: int,
    device: torch.device,
    prior_precision: float,
) -> torch.Tensor:
    """Diagonal GGN curvature for the last-layer BT reward model.

    H_diag = prior_precision + sum_n sigmoid(u_n)(1-sigmoid(u_n)) * Delta h_n^2.
    """
    model.eval()
    diag = torch.zeros_like(theta, device=device)
    for batch in tqdm(dataloader, desc="curvature", leave=False):
        _, _, delta_h = pair_features(model, tokenizer, batch, prompt_template, max_length, device)
        logits = delta_h.matmul(theta)
        weight = torch.sigmoid(logits) * torch.sigmoid(-logits)
        diag += (weight.unsqueeze(-1) * delta_h.pow(2)).sum(dim=0)
    diag += float(prior_precision)
    return diag.clamp_min(1e-8)


def main() -> None:
    args = tyro.cli(ScriptArguments)
    set_seed(args.seed)
    if not args.dataset_caching:
        disable_caching()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = str_to_torch_dtype(args.torch_dtype)

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

    rdp = DATASET_CONFIGS[args.dataset_name](
        prompt_template=args.prompt_template,
        sanity_check=args.sanity_check,
    )
    train_dataset = rdp.get_preference_dataset(split="train")
    eval_dataset = rdp.get_preference_dataset(split="validation")
    if args.limit_train_examples is not None:
        train_dataset = train_dataset.select(range(min(args.limit_train_examples, len(train_dataset))))
    if args.limit_eval_examples is not None:
        eval_dataset = eval_dataset.select(range(min(args.limit_eval_examples, len(eval_dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_preference,
    )
    train_loader_noshuffle = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=collate_preference,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_preference,
    )

    # Infer hidden size from the first batch.
    first_batch = next(iter(train_loader_noshuffle))
    _, _, first_delta = pair_features(model, tokenizer, first_batch, args.prompt_template, args.max_length, device)
    hidden_size = first_delta.size(-1)
    theta = torch.zeros(hidden_size, device=device, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.AdamW([theta], lr=args.learning_rate, weight_decay=0.0)
    num_train = len(train_dataset)
    global_step = 0

    print(f"[BPP-MOA] fitting MAP last-layer reward head on {args.dataset_name}")
    for epoch in range(args.num_train_epochs):
        running = 0.0
        running_n = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}"), start=1):
            _, _, delta_h = pair_features(model, tokenizer, batch, args.prompt_template, args.max_length, device)
            logits = delta_h.matmul(theta)
            nll = -F.logsigmoid(logits).mean()
            # Full MAP objective is sum NLL + 0.5 * prior_precision * ||theta||^2.
            # Since minibatch training uses mean NLL, divide prior by N.
            prior = 0.5 * float(args.prior_precision) * theta.pow(2).sum() / max(num_train, 1)
            loss = nll + prior
            (loss / args.gradient_accumulation_steps).backward()

            running += nll.detach().item() * logits.numel()
            running_n += logits.numel()

            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_([theta], args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.logging_steps == 0:
                    print(f"[train] epoch={epoch} step={global_step} nll={running / max(running_n, 1):.4f}")
                    running = 0.0
                    running_n = 0

        metrics = evaluate_head(model, tokenizer, theta.detach(), eval_loader, args.prompt_template, args.max_length, device)
        print(f"[eval] epoch={epoch} nll={metrics['nll']:.4f} acc={metrics['accuracy']:.4f}")

    print("[BPP-MOA] computing diagonal Laplace curvature")
    theta_detached = theta.detach()
    diag_precision = compute_diag_precision(
        model=model,
        tokenizer=tokenizer,
        theta=theta_detached,
        dataloader=train_loader_noshuffle,
        prompt_template=args.prompt_template,
        max_length=args.max_length,
        device=device,
        prior_precision=args.prior_precision,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sft_model_name": args.sft_model_name,
        "dataset_name": args.dataset_name,
        "prompt_template": args.prompt_template,
        "max_length": args.max_length,
        "hidden_size": hidden_size,
        "prior_precision": args.prior_precision,
        "num_train_examples": len(train_dataset),
        "num_eval_examples": len(eval_dataset),
        "feature": "last_nonpad_token_hidden_state",
        "posterior": "diagonal_GGN_Laplace_last_layer",
    }
    save_laplace_head(str(out_dir / "laplace_head.pt"), theta_detached, diag_precision, metadata)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[BPP-MOA] saved: {out_dir / 'laplace_head.pt'}")


if __name__ == "__main__":
    main()
