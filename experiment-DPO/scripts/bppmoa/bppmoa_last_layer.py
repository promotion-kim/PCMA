from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def str_to_torch_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half", "torch.float16"}:
        return torch.float16
    if name in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype={name}. Use bf16, fp16, or fp32.")


def format_prompt_response(prompt_template: str, raw_prompt: str, response: str) -> str:
    """Format a single prompt-response text exactly once.

    The MODPO scripts pass raw_prompt separately and use the template
    "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:".
    We keep that convention here.
    """
    return prompt_template.format(raw_prompt=raw_prompt) + response


@torch.no_grad()
def last_token_features(
    model,
    tokenizer,
    raw_prompts: List[str],
    responses: List[str],
    prompt_template: str,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return frozen LM last-token hidden states for prompt+response texts.

    Shape: (batch, hidden_size), dtype float32.  The model itself can run in
    bf16/fp16; the Bayesian linear head is kept in float32 for numerical safety.
    """
    texts = [format_prompt_response(prompt_template, p, r) for p, r in zip(raw_prompts, responses)]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[-1]
    lengths = encoded["attention_mask"].sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    feats = hidden[batch_idx, lengths]
    return feats.float()


def pair_features(
    model,
    tokenizer,
    batch: Dict[str, List[str]],
    prompt_template: str,
    max_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_prompts = batch["raw_prompt"]
    chosen_h = last_token_features(model, tokenizer, raw_prompts, batch["chosen"], prompt_template, max_length, device)
    rejected_h = last_token_features(model, tokenizer, raw_prompts, batch["rejected"], prompt_template, max_length, device)
    delta_h = chosen_h - rejected_h
    return chosen_h, rejected_h, delta_h


def laplace_pair_stats(
    delta_h: torch.Tensor,
    theta: torch.Tensor,
    diag_precision: torch.Tensor,
    variance_scale: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mu, v, and uncertainty-attenuated logit for last-layer Laplace.

    For r_theta(x,y)=theta^T h_phi(x,y), the paper gives
      mu = theta^T Delta h,
      g = Delta h,
      v = Delta h^T Sigma Delta h.
    With diagonal precision H_diag, Sigma_diag = 1 / H_diag.
    """
    theta = theta.to(delta_h.device, dtype=delta_h.dtype)
    diag_precision = diag_precision.to(delta_h.device, dtype=delta_h.dtype).clamp_min(eps)
    mu = delta_h.matmul(theta)
    v = variance_scale * (delta_h.pow(2) / diag_precision).sum(dim=-1)
    ell = mu / torch.sqrt(1.0 + (math.pi / 8.0) * v.clamp_min(0.0))
    return mu, v, ell


def save_laplace_head(
    output_path: str,
    theta: torch.Tensor,
    diag_precision: torch.Tensor,
    metadata: Optional[Dict] = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "theta": theta.detach().cpu().float(),
        "diag_precision": diag_precision.detach().cpu().float(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_laplace_head(path: str, map_location: str | torch.device = "cpu") -> Dict:
    payload = torch.load(path, map_location=map_location)
    if "theta" not in payload or "diag_precision" not in payload:
        raise KeyError(f"{path} must contain 'theta' and 'diag_precision'.")
    payload["theta"] = payload["theta"].float()
    payload["diag_precision"] = payload["diag_precision"].float()
    return payload
