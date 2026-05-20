#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for paper-faithful 3D BPP-MOA on HH-style scored pairs.

This file is intentionally self-contained.  It implements the pieces that
should be shared by
  1) objective-wise last-layer Laplace fitting,
  2) posterior-pooled target construction, and
  3) BPP-MOA policy distillation.

Expected raw JSON/JSONL row format:
    {
      "prompt": "Human: ...\n\nAssistant:",
      "response_0": "...",
      "response_1": "...",
      "help_score_0": float,
      "help_score_1": float,
      "harm_score_0": float,      # lower is safer by default
      "harm_score_1": float,
      "humor_score_0": float,
      "humor_score_1": float
    }

The exact score key names are configurable by convention below.  If your file
uses e.g. harmless_score_0 instead of harm_score_0, this utility will also find
it automatically.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

OBJECTIVES: Tuple[str, str, str] = ("helpful", "harmless", "humor")

# Candidate score keys.  The first matching pair is used.
SCORE_KEY_CANDIDATES: Dict[str, List[Tuple[str, str]]] = {
    "helpful": [
        ("help_score_0", "help_score_1"),
        ("helpful_score_0", "helpful_score_1"),
        ("helpfulness_score_0", "helpfulness_score_1"),
        ("helpsteer_score_0", "helpsteer_score_1"),
    ],
    "harmless": [
        ("harmless_score_0", "harmless_score_1"),
        ("safety_score_0", "safety_score_1"),
        ("safe_score_0", "safe_score_1"),
        # Common in reward/cost pipelines: higher harm/cost means worse.
        ("harm_score_0", "harm_score_1"),
        ("cost_score_0", "cost_score_1"),
    ],
    "humor": [
        ("humor_score_0", "humor_score_1"),
        ("humorous_score_0", "humorous_score_1"),
        ("funny_score_0", "funny_score_1"),
    ],
}

DIRECTION_CHOICES = {"higher_is_better", "lower_is_better"}
DEFAULT_DIRECTIONS = {
    "helpful": "higher_is_better",
    # The current HH script uses harm_score_0/1.  For harm/cost, lower is safer.
    "harmless": "lower_is_better",
    "humor": "higher_is_better",
}


def str_to_torch_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half", "torch.float16"}:
        return torch.float16
    if name in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype={name}. Use bf16, fp16, or fp32.")


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).lower()
    if x in {"true", "1", "yes", "y"}:
        return True
    if x in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean, got {x}")


def load_json_or_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        # Accept {"data": [...]} or split dictionaries.
        for key in ["data", "rows", "examples"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        raise ValueError(f"{path} is a dict, but no list-valued data/rows/examples key was found.")
    if not isinstance(obj, list):
        raise ValueError(f"{path} must be a JSON list or JSONL file.")
    return obj


def find_split_path(data_dir: str, split: str) -> str:
    """Find train/dev/validation/test JSON or JSONL file under data_dir."""
    aliases = {
        "train": ["train"],
        "validation": ["validation", "valid", "val", "dev", "test"],
        "dev": ["dev", "validation", "valid", "val", "test"],
        "test": ["test", "dev", "validation", "valid", "val"],
    }
    names = aliases.get(split, [split])
    for name in names:
        for ext in ["json", "jsonl"]:
            p = os.path.join(data_dir, f"{name}.{ext}")
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"Could not find split={split} under {data_dir}. Tried aliases={names}.")


def load_split(data_dir: str, split: str) -> List[Dict]:
    return load_json_or_jsonl(find_split_path(data_dir, split))


def get_prompt(row: Dict) -> str:
    for key in ["prompt", "raw_prompt", "instruction", "query"]:
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError("Row is missing prompt/raw_prompt/instruction/query.")


def get_response(row: Dict, response_id: int) -> str:
    candidates = [
        f"response_{response_id}",
        f"answer_{response_id}",
        f"output_{response_id}",
        f"completion_{response_id}",
    ]
    for key in candidates:
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError(f"Row is missing response_{response_id}/answer_{response_id}/output_{response_id}.")


def hh_prompt_to_messages(prompt_hh: str) -> List[Dict[str, str]]:
    """Convert Anthropic HH text into chat-template messages."""
    text = str(prompt_hh).strip()
    if text.endswith("Assistant:"):
        text = text[: -len("Assistant:")].strip()

    parts = text.split("\n\n")
    messages: List[Dict[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("Human:"):
            messages.append({"role": "user", "content": part[len("Human:") :].strip()})
        elif part.startswith("Assistant:"):
            content = part[len("Assistant:") :].strip()
            if content:
                messages.append({"role": "assistant", "content": content})
    if len(messages) == 0:
        # Fallback: treat the entire prompt as one user message.
        messages = [{"role": "user", "content": text}]
    return messages


def format_prompt(prompt: str, tokenizer=None, prompt_format: str = "chat") -> str:
    """Return the exact prompt prefix that will be concatenated with a response.

    prompt_format:
      - chat: convert HH prompt to messages and apply tokenizer.chat_template
      - hh:   keep HH-style prompt unchanged
      - raw:  keep prompt unchanged
      - modpo: use the MODPO-style template
    """
    prompt = str(prompt)
    if prompt_format in {"raw", "hh"}:
        return prompt
    if prompt_format == "modpo":
        return f"BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:"
    if prompt_format == "chat":
        if tokenizer is None:
            raise ValueError("prompt_format='chat' requires tokenizer.")
        if getattr(tokenizer, "chat_template", None) is None:
            raise ValueError(
                "tokenizer.chat_template is missing. Use --prompt_format hh/raw/modpo "
                "or set a chat_template on the tokenizer."
            )
        messages = hh_prompt_to_messages(prompt)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raise ValueError(f"Unknown prompt_format={prompt_format}")


def format_prompt_response(prompt_template: str, raw_prompt: str, response: str) -> str:
    return prompt_template.format(raw_prompt=raw_prompt) + str(response)


def find_score_keys(row: Dict, objective: str) -> Tuple[str, str]:
    if objective not in SCORE_KEY_CANDIDATES:
        raise ValueError(f"Unknown objective={objective}")
    for k0, k1 in SCORE_KEY_CANDIDATES[objective]:
        if k0 in row and k1 in row:
            return k0, k1
    raise KeyError(
        f"Could not find score keys for objective={objective}. "
        f"Tried {SCORE_KEY_CANDIDATES[objective]}. Available keys={sorted(row.keys())[:50]}"
    )


def canonical_score(row: Dict, objective: str, response_id: int, directions: Dict[str, str]) -> float:
    """Return a score where larger always means better for the objective."""
    k0, k1 = find_score_keys(row, objective)
    key = k0 if response_id == 0 else k1
    raw = float(row[key])
    direction = directions.get(objective, DEFAULT_DIRECTIONS[objective])
    if direction not in DIRECTION_CHOICES:
        raise ValueError(f"Invalid direction for {objective}: {direction}")
    return raw if direction == "higher_is_better" else -raw


def validate_weights(weights: Dict[str, float], normalize: bool = True) -> Dict[str, float]:
    out = {k: float(weights[k]) for k in OBJECTIVES}
    total = sum(out.values())
    if total <= 0:
        raise ValueError(f"Weights must have positive sum. Got {out}")
    if normalize:
        out = {k: v / total for k, v in out.items()}
    elif abs(total - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1. Got {out}, sum={total}")
    return out


def build_objective_preference_rows(
    rows: Sequence[Dict],
    objective: str,
    tokenizer=None,
    prompt_format: str = "chat",
    directions: Optional[Dict[str, str]] = None,
    min_abs_score_gap: float = 0.0,
    sanity_check: bool = False,
) -> List[Dict]:
    """Convert scored response pairs into objective-specific BT preference pairs.

    The resulting rows contain raw_prompt/chosen/rejected, where raw_prompt is
    already formatted for the model.  Use prompt_template='{raw_prompt}' later.
    """
    if directions is None:
        directions = DEFAULT_DIRECTIONS
    if sanity_check:
        rows = rows[: min(len(rows), 128)]

    out: List[Dict] = []
    skipped_tie = 0
    skipped_empty = 0
    for idx, row in enumerate(rows):
        try:
            r0 = get_response(row, 0).strip()
            r1 = get_response(row, 1).strip()
            if not r0 or not r1:
                skipped_empty += 1
                continue
            s0 = canonical_score(row, objective, 0, directions)
            s1 = canonical_score(row, objective, 1, directions)
        except Exception:
            skipped_empty += 1
            continue

        gap = s0 - s1
        if abs(gap) <= float(min_abs_score_gap):
            skipped_tie += 1
            continue

        prompt = format_prompt(get_prompt(row), tokenizer=tokenizer, prompt_format=prompt_format)
        if gap > 0:
            chosen, rejected = r0, r1
            chosen_score, rejected_score = s0, s1
        else:
            chosen, rejected = r1, r0
            chosen_score, rejected_score = s1, s0

        out.append(
            {
                "raw_prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "objective": objective,
                "score_chosen": float(chosen_score),
                "score_rejected": float(rejected_score),
                "abs_score_gap": float(abs(gap)),
                "source_idx": int(idx),
            }
        )
    if len(out) == 0:
        raise ValueError(
            f"No objective preference rows were built for {objective}. "
            f"skipped_empty={skipped_empty}, skipped_tie={skipped_tie}."
        )
    return out


def build_candidate_pair_rows(
    rows: Sequence[Dict],
    tokenizer=None,
    prompt_format: str = "chat",
    sanity_check: bool = False,
) -> List[Dict]:
    """Build candidate pairs whose event is response_0 > response_1.

    We save them as chosen=response_0/rejected=response_1 because the soft target
    bpp_rho may be smaller than 0.5.  This is valid for soft-label DPO.
    """
    if sanity_check:
        rows = rows[: min(len(rows), 128)]
    out: List[Dict] = []
    for idx, row in enumerate(rows):
        try:
            r0 = get_response(row, 0).strip()
            r1 = get_response(row, 1).strip()
        except Exception:
            continue
        if not r0 or not r1:
            continue
        prompt = format_prompt(get_prompt(row), tokenizer=tokenizer, prompt_format=prompt_format)
        item = {
            "raw_prompt": prompt,
            "chosen": r0,
            "rejected": r1,
            "source_idx": int(idx),
        }
        # Preserve original scores when available for debugging.
        for obj in OBJECTIVES:
            try:
                k0, k1 = find_score_keys(row, obj)
                item[k0] = float(row[k0])
                item[k1] = float(row[k1])
            except Exception:
                pass
        out.append(item)
    if len(out) == 0:
        raise ValueError("No candidate pairs were built.")
    return out


def collate_pair_text(batch: List[Dict]) -> Dict[str, List[str]]:
    return {
        "raw_prompt": [ex["raw_prompt"] for ex in batch],
        "chosen": [ex["chosen"] for ex in batch],
        "rejected": [ex["rejected"] for ex in batch],
    }


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
    """Return frozen LM last-token hidden states for prompt+response texts."""
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
    """Compute mu, v, and uncertainty-attenuated logit.

    mu = Delta h^T theta
    v  = Delta h^T Sigma Delta h, with Sigma_diag = 1 / diag_precision
    ell = mu / sqrt(1 + pi v / 8)
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
    payload.setdefault("metadata", {})
    return payload


def ensure_tokenizer(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
