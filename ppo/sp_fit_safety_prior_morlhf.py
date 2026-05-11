"""Fit a prompt-only safety-prior model for Safety-Prior PC-MORLHF.

This script trains rho(x) = P(safety-critical | x) from fixed prompt
features.  The trained rho is used only as a prompt-level safety prior during
PPO; it does not look at generated responses at training time.

The simplest runnable mode is dataset_labels:

  --risk_dataset_names PKU-Alignment/PKU-SafeRLHF-10K-safer,PKU-Alignment/PKU-SafeRLHF-10K-better \
  --risk_dataset_labels 1,0

For a cleaner experiment, use auto_flags when the dataset exposes safety
annotations such as is_response_0_safe/is_response_1_safe or
is_chosen_safe/is_rejected_safe. In all modes, rho is trained on prompt x only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import tyro
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils import disable_progress_bar_non_local_main, param_sharding_enabled, print_local_main, set_seed
from src.utils.posterior_calibration import (
    FrozenCausalLMPromptFeatureExtractor,
    HFPromptFeatureExtractor,
)


disable_progress_bar_non_local_main()


def _split_csv(x: str) -> List[str]:
    return [v.strip() for v in str(x).split(",") if v.strip()]


def _to_bool(v) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "safe", "harmless", "benign"}:
        return True
    if s in {"0", "false", "no", "unsafe", "harmful", "malicious"}:
        return False
    return None


@dataclass
class ScriptArguments:
    # Base / data
    sft_model_name: str = field(default="PKU-Alignment/alpaca-7b-reproduced")
    risk_dataset_names: str = field(
        default="PKU-Alignment/PKU-SafeRLHF-10K",
        metadata={"help": "comma-separated datasets used to train prompt risk model. Prefer raw PKU-SafeRLHF/10K so safety flags are available."},
    )
    risk_dataset_labels: str = field(
        default="",
        metadata={"help": "comma-separated labels used only when risk_label_mode=dataset_labels"},
    )
    risk_label_mode: str = field(
        default="auto_flags",
        metadata={"help": "auto_flags or dataset_labels. auto_flags uses is_response_*_safe when available."},
    )
    output_dir: str = field(default="./safety_prior_morlhf")
    prompt_template: str = field(default=DEFAULT_PROMPT_TEMPLATE)
    train_split: str = field(default="train")
    eval_ratio: float = field(default=0.1)
    max_examples_per_dataset: Optional[int] = field(default=None)
    seed: int = field(default=42)

    # Feature extraction
    feature_source: str = field(default="sft_hidden", metadata={"help": "sft_hidden or hf_model"})
    feature_model_name: str = field(default="sentence-transformers/all-MiniLM-L6-v2")
    feature_pooling: str = field(default="mean")
    feature_max_length: int = field(default=256)
    feature_batch_size: int = field(default=8)
    use_flash_attention_2: bool = field(default=False)

    # Risk classifier training
    num_steps: int = field(default=1000)
    lr: float = field(default=1e-3)
    batch_size: int = field(default=256)
    weight_decay: float = field(default=0.0)
    log_every: int = field(default=100)


script_args = tyro.cli(ScriptArguments)
set_seed(script_args.seed)
accelerator = Accelerator()

risk_dataset_names = _split_csv(script_args.risk_dataset_names)
risk_dataset_labels = [float(x) for x in _split_csv(script_args.risk_dataset_labels)]

if script_args.risk_label_mode == "dataset_labels" and len(risk_dataset_names) != len(risk_dataset_labels):
    raise ValueError("risk_dataset_names and risk_dataset_labels must have the same length when risk_label_mode=dataset_labels.")


def _format_prompt(example: dict) -> Tuple[str, str]:
    """Return (formatted_prompt, raw_prompt).

    Raw PKU-SafeRLHF uses `prompt`; local wrappers may use `raw_prompt`.
    """
    if "raw_prompt" in example and example["raw_prompt"] is not None:
        raw_prompt = str(example["raw_prompt"])
        formatted = script_args.prompt_template.format(raw_prompt=raw_prompt)
        return formatted, raw_prompt
    if "prompt" in example and example["prompt"] is not None:
        raw_prompt = str(example["prompt"])
        # Raw PKU prompt is not formatted; wrap it for consistency with features.
        if "{raw_prompt}" in script_args.prompt_template and "BEGINNING OF CONVERSATION" not in raw_prompt:
            return script_args.prompt_template.format(raw_prompt=raw_prompt), raw_prompt
        return raw_prompt, raw_prompt
    if "query" in example and example["query"] is not None:
        query = str(example["query"])
        return query, query
    if "input" in example and example["input"] is not None:
        raw_prompt = str(example["input"])
        return script_args.prompt_template.format(raw_prompt=raw_prompt), raw_prompt
    raise KeyError(f"Cannot find prompt key. Available keys: {list(example.keys())}")


def _auto_flag_label(example: dict) -> Optional[float]:
    """Create z_risk(x) from dataset safety annotations.

    The intended PKU-SafeRLHF definition is:
        z_risk(x) = 1 if both candidate responses are unsafe.
        z_risk(x) = 0 otherwise.

    The classifier still receives only the prompt x as input.  Response-level
    safety flags are used only to create supervision.
    """
    # Raw PKU-SafeRLHF / PKU-SafeRLHF-10K fields.
    if "is_response_0_safe" in example and "is_response_1_safe" in example:
        b0 = _to_bool(example.get("is_response_0_safe"))
        b1 = _to_bool(example.get("is_response_1_safe"))
        # risk=1 only when both candidate responses are unsafe
        if b0 is not None and b1 is not None:
            return 1.0 if ((not b0) and (not b1)) else 0.0

    # Prompt-level flags, if a processed dataset provides them.
    for key in ["is_prompt_safe", "prompt_safe", "is_safe", "safe", "is_harmless", "harmless"]:
        if key in example:
            b = _to_bool(example.get(key))
            if b is not None:
                return 0.0 if b else 1.0

    for key in ["is_prompt_harmful", "prompt_harmful", "is_harmful", "harmful", "unsafe"]:
        if key in example:
            b = _to_bool(example.get(key))
            if b is not None:
                return 1.0 if b else 0.0

    # Common processed response-level variants.
    safe_keys = [
        ("response_0_safe", "response_1_safe"),
        ("is_chosen_safe", "is_rejected_safe"),
        ("chosen_safe", "rejected_safe"),
        ("safer_chosen_safe", "safer_rejected_safe"),
    ]
    for k0, k1 in safe_keys:
        if k0 in example and k1 in example:
            b0 = _to_bool(example.get(k0))
            b1 = _to_bool(example.get(k1))
            if b0 is not None and b1 is not None:
                # risk=1 only when both candidate responses are unsafe
                return 1.0 if ((not b0) and (not b1)) else 0.0

    unsafe_keys = [
        ("is_response_0_unsafe", "is_response_1_unsafe"),
        ("response_0_unsafe", "response_1_unsafe"),
        ("is_chosen_unsafe", "is_rejected_unsafe"),
        ("chosen_unsafe", "rejected_unsafe"),
    ]
    for k0, k1 in unsafe_keys:
        if k0 in example and k1 in example:
            b0 = _to_bool(example.get(k0))
            b1 = _to_bool(example.get(k1))
            if b0 is not None and b1 is not None:
                return 1.0 if (b0 or b1) else 0.0

    return None


def _iter_dataset(dataset_name: str):
    """Yield examples from either a local DATASET_CONFIGS wrapper or raw HF dataset."""
    if dataset_name in DATASET_CONFIGS:
        rdp = DATASET_CONFIGS[dataset_name](
            prompt_template=script_args.prompt_template,
            sanity_check=False,
        )
        return rdp.get_preference_dataset(split=script_args.train_split)
    # Prefer raw HF dataset for safety-prior fitting because it exposes
    # is_response_0_safe/is_response_1_safe, better_response_id, safer_response_id.
    return load_dataset(dataset_name, split=script_args.train_split)


def _load_prompt_labels() -> Tuple[List[str], torch.Tensor]:
    prompt_to_label: Dict[str, float] = {}

    for dataset_id, dataset_name in enumerate(risk_dataset_names):
        print_local_main(f"loading risk dataset {dataset_id}: {dataset_name}")
        dataset = _iter_dataset(dataset_name)

        fixed_label = None
        if script_args.risk_label_mode == "dataset_labels":
            fixed_label = float(risk_dataset_labels[dataset_id])

        count = 0
        skipped = 0
        for ex in dataset:
            _, raw_prompt = _format_prompt(ex)
            if fixed_label is None:
                label = _auto_flag_label(ex)
                if label is None:
                    skipped += 1
                    continue
            else:
                label = fixed_label

            # If a prompt appears multiple times, keep the conservative label.
            prompt_to_label[raw_prompt] = max(prompt_to_label.get(raw_prompt, 0.0), float(label))

            count += 1
            if script_args.max_examples_per_dataset and count >= script_args.max_examples_per_dataset:
                break
        print_local_main(
            f"dataset {dataset_name}: collected {count} labeled rows, skipped {skipped} rows without usable flags"
        )

    if not prompt_to_label:
        raise RuntimeError(
            "No prompt labels were collected. For PKU-SafeRLHF use "
            "--risk_dataset_names PKU-Alignment/PKU-SafeRLHF-10K --risk_label_mode auto_flags."
        )

    prompts = list(prompt_to_label.keys())
    labels = torch.tensor([prompt_to_label[p] for p in prompts], dtype=torch.float32)
    print_local_main(
        f"unique prompts: {len(prompts)}, positive risk ratio: {labels.mean().item():.4f}"
    )
    return prompts, labels

prompts, labels = _load_prompt_labels()

print_local_main("extracting prompt features for safety prior...")
if script_args.feature_source == "sft_hidden":
    tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_name,
        use_flash_attention_2=script_args.use_flash_attention_2,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        **({"device_map": {"": accelerator.local_process_index}} if torch.cuda.is_available() and not param_sharding_enabled() else {}),
    )
    base.config.update({"use_cache": False, "pad_token_id": base.config.eos_token_id})
    base.eval()

    try:
        feature_extractor = FrozenCausalLMPromptFeatureExtractor(
            model=base,
            tokenizer=tokenizer,
            max_length=script_args.feature_max_length,
            device=accelerator.device,
            pooling=script_args.feature_pooling,
            disable_adapter=True,
            prompt_template=script_args.prompt_template,
        )
    except TypeError:
        feature_extractor = FrozenCausalLMPromptFeatureExtractor(
            model=base,
            tokenizer=tokenizer,
            max_length=script_args.feature_max_length,
            device=accelerator.device,
            pooling=script_args.feature_pooling,
            disable_adapter=True,
        )
    feature_id = f"sft_hidden::{script_args.sft_model_name}::{script_args.feature_pooling}"
    features = feature_extractor.encode(
        prompts,
        batch_size=max(1, script_args.feature_batch_size),
        device="cpu",
    )
elif script_args.feature_source == "hf_model":
    feature_extractor = HFPromptFeatureExtractor(
        script_args.feature_model_name,
        max_length=script_args.feature_max_length,
        device=f"cuda:{accelerator.local_process_index}" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    feature_id = script_args.feature_model_name
    features = feature_extractor.encode(prompts, batch_size=32, device="cpu")
else:
    raise ValueError("feature_source must be either 'sft_hidden' or 'hf_model'.")

features = features.float()
labels = labels.float()
num_examples, feature_dim = features.shape

perm = torch.randperm(num_examples, generator=torch.Generator().manual_seed(script_args.seed))
num_eval = int(round(num_examples * script_args.eval_ratio))
num_eval = min(max(num_eval, 1), num_examples - 1) if num_examples > 1 else 0
eval_idx = perm[:num_eval]
train_idx = perm[num_eval:]

train_x = features[train_idx]
train_y = labels[train_idx]
eval_x = features[eval_idx] if num_eval > 0 else features[:0]
eval_y = labels[eval_idx] if num_eval > 0 else labels[:0]

model = nn.Linear(feature_dim, 1)
model.to(accelerator.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=script_args.lr, weight_decay=script_args.weight_decay)
criterion = nn.BCEWithLogitsLoss()

dataset = TensorDataset(train_x, train_y)
loader = DataLoader(dataset, batch_size=script_args.batch_size, shuffle=True, drop_last=False)
loader_iter = iter(loader)

print_local_main("training prompt-only safety prior...")
for step in tqdm(
    range(1, script_args.num_steps + 1),
    disable=not accelerator.is_local_main_process,
    dynamic_ncols=True,
):
    try:
        xb, yb = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        xb, yb = next(loader_iter)

    xb = xb.to(accelerator.device)
    yb = yb.to(accelerator.device)
    logits = model(xb).squeeze(-1)
    loss = criterion(logits, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if accelerator.is_local_main_process and script_args.log_every > 0 and step % script_args.log_every == 0:
        with torch.no_grad():
            train_prob = torch.sigmoid(model(train_x.to(accelerator.device)).squeeze(-1)).cpu()
            train_pred = (train_prob >= 0.5).float()
            train_acc = (train_pred == train_y).float().mean().item()
            if num_eval > 0:
                eval_prob = torch.sigmoid(model(eval_x.to(accelerator.device)).squeeze(-1)).cpu()
                eval_pred = (eval_prob >= 0.5).float()
                eval_loss = criterion(
                    model(eval_x.to(accelerator.device)).squeeze(-1),
                    eval_y.to(accelerator.device),
                ).item()
                eval_acc = (eval_pred == eval_y).float().mean().item()
            else:
                eval_loss = float("nan")
                eval_acc = float("nan")
        print(
            f"step={step} loss={loss.item():.4f} train_acc={train_acc:.4f} "
            f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f}",
            flush=True,
        )

if accelerator.is_local_main_process:
    os.makedirs(script_args.output_dir, exist_ok=True)
    with torch.no_grad():
        train_logits = model(train_x.to(accelerator.device)).squeeze(-1)
        train_loss = criterion(train_logits, train_y.to(accelerator.device)).item()
        train_prob = torch.sigmoid(train_logits).cpu()
        train_acc = ((train_prob >= 0.5).float() == train_y).float().mean().item()
        if num_eval > 0:
            eval_logits = model(eval_x.to(accelerator.device)).squeeze(-1)
            eval_loss = criterion(eval_logits, eval_y.to(accelerator.device)).item()
            eval_prob = torch.sigmoid(eval_logits).cpu()
            eval_acc = ((eval_prob >= 0.5).float() == eval_y).float().mean().item()
        else:
            eval_loss = float("nan")
            eval_acc = float("nan")

    state = {
        "weight": model.weight.detach().cpu(),
        "bias": model.bias.detach().cpu(),
        "feature_dim": feature_dim,
        "feature_source": script_args.feature_source,
        "feature_model_name": feature_id,
        "feature_pooling": script_args.feature_pooling,
        "feature_max_length": script_args.feature_max_length,
        "prompt_template": script_args.prompt_template,
        "risk_label_mode": script_args.risk_label_mode,
        "risk_dataset_names": risk_dataset_names,
        "risk_dataset_labels": risk_dataset_labels,
    }
    torch.save(state, os.path.join(script_args.output_dir, "safety_prior.pt"))

    metrics = {
        "num_examples": int(num_examples),
        "num_train": int(len(train_idx)),
        "num_eval": int(num_eval),
        "positive_ratio": float(labels.mean().item()),
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "eval_loss": float(eval_loss),
        "eval_acc": float(eval_acc),
    }
    with open(os.path.join(script_args.output_dir, "safety_prior_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"saved safety prior to {script_args.output_dir}")
    print(json.dumps(metrics, indent=2), flush=True)
