# scripts/pcma/eval_calibration_direct_prompt_pca_fixed.py

from __future__ import annotations

# ---------------------------------------------------------------------
# Compatibility shim for old TRL + newer Transformers
# ---------------------------------------------------------------------
try:
    import transformers

    if not hasattr(transformers, "top_k_top_p_filtering"):
        from transformers.generation.logits_process import (
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        def top_k_top_p_filtering(
            logits,
            top_k=0,
            top_p=1.0,
            filter_value=-float("Inf"),
            min_tokens_to_keep=1,
        ):
            if top_k is not None and top_k > 0:
                logits = TopKLogitsWarper(
                    top_k=top_k,
                    filter_value=filter_value,
                    min_tokens_to_keep=min_tokens_to_keep,
                )(None, logits)

            if top_p is not None and top_p < 1.0:
                logits = TopPLogitsWarper(
                    top_p=top_p,
                    filter_value=filter_value,
                    min_tokens_to_keep=min_tokens_to_keep,
                )(None, logits)

            return logits

        transformers.top_k_top_p_filtering = top_k_top_p_filtering

except Exception as e:
    print(f"[warning] transformers compatibility shim failed: {e}", flush=True)
# ---------------------------------------------------------------------

import gc
import hashlib
import importlib.util
import json
import random
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from accelerate import Accelerator
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------
# Local compatibility stub:
# Avoid importing full TRL -> DeepSpeed during this evaluation script.
# This script only needs src.utils to resolve trl.import_utils.is_peft_available.
# It does not use TRL Trainer or DeepSpeed.
# ---------------------------------------------------------------------
if "trl.import_utils" not in sys.modules:
    trl_stub = types.ModuleType("trl")
    trl_import_utils_stub = types.ModuleType("trl.import_utils")

    def is_peft_available():
        return importlib.util.find_spec("peft") is not None

    trl_import_utils_stub.is_peft_available = is_peft_available
    trl_stub.import_utils = trl_import_utils_stub

    sys.modules.setdefault("trl", trl_stub)
    sys.modules.setdefault("trl.import_utils", trl_import_utils_stub)
# ---------------------------------------------------------------------

from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils import (
    common_prefix_length,
    disable_progress_bar_non_local_main,
    param_sharding_enabled,
    print_local_main,
    set_seed,
)
from src.utils.posterior_calibration import (
    FrozenCausalLMPromptFeatureExtractor,
    HFPromptFeatureExtractor,
)


disable_progress_bar_non_local_main()


def _split_csv(x: str) -> List[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


@dataclass
class ScriptArguments:
    sft_model_name: str = field(metadata={"help": "SFT/reference model"})

    objective_adapter_names: str = field(
        default="",
        metadata={
            "help": (
                "comma-separated objective adapter paths in same order as datasets. "
                "Required only when --reward_type implicit. Example: better_adapter,safer_adapter"
            )
        },
    )
    objective_dataset_names: str = field(
        default="",
        metadata={"help": "comma-separated objective preference datasets in same order"},
    )

    reward_type: str = field(
        default="implicit",
        metadata={
            "help": (
                "Which scorer gaps to calibrate: 'implicit' uses "
                "beta*(log pi_adapter-log pi_ref); 'explicit' uses reward/cost score models. "
                "For explicit with two objectives, obj0=reward model and obj1=-cost model."
            )
        },
    )

    output_path: str = field(
        default="outputs/pcma/calibration_nll/simple_calibration_eval.json",
        metadata={"help": "where to save NLL/ECE comparison results"},
    )

    # Prompt features
    feature_source: str = field(default="sft_hidden", metadata={"help": "sft_hidden or hf_model"})
    feature_model_name: str = field(default="sentence-transformers/all-MiniLM-L6-v2")
    feature_pooling: str = field(default="last", metadata={"help": "mean or last"})
    normalize_features: bool = field(
        default=True,
        metadata={"help": "standardize prompt features using train split mean/std before fitting a_i(x)"},
    )
    feature_reduce_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If set, reduce prompt hidden-state features with train-only PCA before fitting "
                "prompt-dependent log-precision models. Example: --feature_reduce_dim 128"
            )
        },
    )
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE)

    seed: int = 42
    max_length: int = 512
    feature_max_length: int = 256
    per_device_batch_size: int = 2
    num_workers: int = 0
    max_examples_per_objective: Optional[int] = None
    valid_ratio: float = 0.2
    use_flash_attention_2: bool = False
    allow_tf32: bool = True

    # implicit reward scale: s_i^imp = beta * (log pi_i - log pi_ref)
    implicit_beta: float = 1.0

    # explicit reward scorer configuration
    reward_model_name: str = "PKU-Alignment/beaver-7b-v1.0-reward"
    cost_model_name: str = "PKU-Alignment/beaver-7b-v1.0-cost"
    explicit_reward_scale: float = field(
        default=1.0,
        metadata={
            "help": (
                "Optional global scale for explicit reward/cost score gaps. "
                "For harmlessness, the scorer is -cost, then multiplied by this scale."
            )
        },
    )

    # Optional controlled preference label noise. Default is clean setting.
    noise_ratio: float = 0.0
    noise_objectives: str = "all"
    noise_seed: int = 1234
    apply_noise_to_valid: bool = False

    # Calibration fitting.
    # Compared models:
    #   raw_scorer:
    #       no calibration, logit = gap
    #   fixed_objective:
    #       a_i(x)=u_i
    #   prompt_dependent:
    #       a_i(x)=rho(x)+delta_i(x), sum_i delta_i(x)=0
    #   direct_prompt:
    #       a_i(x)=b_i+v_i^T phi(x)
    calib_steps: int = 3000
    calib_lr: float = 3e-3
    calib_batch_size: int = 512
    calib_log_every: int = 200
    early_select_by: str = field(default="nll", metadata={"help": "nll or ece"})

    # Gaussian prior scales for MAP fitting.
    #mu_std: float = 0.1 #0.5
    #c_std: float = 0.1 #0.03

    # Gaussian prior scales for MAP fitting.
    # fixed_objective:
    #   u_i ~ N(0, u_std^2)
    # direct_prompt:
    #   b_i ~ N(0, u_std^2), v_ij ~ N(0, W_std^2)
    u_std: float = 0.5
    W_std: float = 0.02

    # numerical stability for log precision a_i(x)
    clamp_a_min: float = -6.0
    clamp_a_max: float = 6.0

    # Speed/cache options
    shard_gaps_across_processes: bool = True
    use_gap_cache: bool = True
    force_recompute_gaps: bool = False
    gap_cache_dir: str = field(
        default="outputs/pcma/calibration_nll/gap_cache",
        metadata={"help": "cache directory for expensive scorer gaps"},
    )
    use_feature_cache: bool = True
    force_recompute_features: bool = False
    feature_cache_dir: str = field(
        default="outputs/pcma/calibration_nll/feature_cache",
        metadata={"help": "cache directory for prompt hidden-state features"},
    )

    # diagnostic for direct a_i(x)
    dump_direct_diagnostics: bool = True
    diagnostic_weight: str = "0.5,0.5"
    direct_diagnostic_path: str = "outputs/pcma/calibration_nll/direct_a_prompt_diagnostics.jsonl"
    direct_topk: int = 20

    # Full Laplace ablation for direct_prompt.
    # This uses the full Hessian of the negative log posterior.
    # No diagonal Hessian approximation and no damping are applied.
    run_laplace_ablation: bool = True
    laplace_mc_samples: int = 10
    laplace_seed: int = 42


args = tyro.cli(ScriptArguments)
set_seed(args.seed)
accelerator = Accelerator()

if args.allow_tf32 and torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


tokenizer = AutoTokenizer.from_pretrained(args.sft_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def _json_hash(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _all_gather_object(obj: Any) -> List[Any]:
    """Gather arbitrary Python objects from all distributed ranks."""
    if accelerator.num_processes == 1:
        return [obj]

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return [obj]

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


def _format_prompt(example: dict) -> str:
    if "prompt" in example and example["prompt"] is not None:
        return example["prompt"]
    if "raw_prompt" in example and example["raw_prompt"] is not None:
        return args.prompt_template.format(raw_prompt=example["raw_prompt"])
    raise KeyError("Dataset example must contain `prompt` or `raw_prompt`.")


def _raw_prompt_for_feature(example: dict, formatted_prompt: str) -> str:
    return example.get("raw_prompt", formatted_prompt)


def _tokenize_prompt_response(prompt: str, response: str) -> dict:
    prompt_toks = tokenizer(prompt, add_special_tokens=False)
    full_toks = tokenizer(prompt + response, add_special_tokens=False)

    input_ids = full_toks["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = full_toks["attention_mask"] + [1]

    prompt_len = common_prefix_length(prompt_toks["input_ids"], input_ids)

    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len

    if len(input_ids) > args.max_length:
        input_ids = input_ids[: args.max_length]
        attention_mask = attention_mask[: args.max_length]
        labels = labels[: args.max_length]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _collate_logp(features: Sequence[dict]) -> dict:
    batch = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
    padded = tokenizer.pad(batch, padding=True, return_tensors="pt")

    max_len = padded["input_ids"].shape[1]
    labels = []
    for f in features:
        lab = f["labels"] + [-100] * (max_len - len(f["labels"]))
        labels.append(lab)

    padded["labels"] = torch.tensor(labels, dtype=torch.long)
    return padded


@torch.inference_mode()
def _sequence_logps(model, batch: dict) -> torch.Tensor:
    batch = {k: v.to(accelerator.device, non_blocking=True) for k, v in batch.items()}

    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]

    labels = batch["labels"][:, 1:].clone()
    mask = labels.ne(-100)
    labels = labels.masked_fill(~mask, 0)

    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    return (token_logps * mask).sum(dim=-1).detach().cpu()


def _build_pairs(dataset, max_examples: Optional[int]) -> List[dict]:
    pairs = []

    for ex in dataset:
        prompt = _format_prompt(ex)
        chosen = ex.get("chosen")
        rejected = ex.get("rejected")

        if chosen is None or rejected is None:
            raise KeyError("Preference dataset must contain `chosen` and `rejected` fields.")

        pairs.append(
            {
                "prompt": prompt,
                "feature_prompt": _raw_prompt_for_feature(ex, prompt),
                "chosen": chosen,
                "rejected": rejected,
            }
        )

        if max_examples is not None and len(pairs) >= max_examples:
            break

    return pairs


def _prompt_level_split(pairs: List[dict], valid_ratio: float, seed: int):
    rng = random.Random(seed)

    unique_prompts = sorted(set(p["feature_prompt"] for p in pairs))
    rng.shuffle(unique_prompts)

    n_valid = max(1, int(len(unique_prompts) * valid_ratio))
    valid_set = set(unique_prompts[:n_valid])

    train_pairs = [p for p in pairs if p["feature_prompt"] not in valid_set]
    valid_pairs = [p for p in pairs if p["feature_prompt"] in valid_set]

    return train_pairs, valid_pairs


def _gap_cache_path(dataset_name: str, scorer_name: str, split_name: str, pairs: List[dict]) -> Path:
    payload = {
        "type": "scorer_gaps_v4",
        "reward_type": args.reward_type,
        "dataset_name": dataset_name,
        "scorer_name": scorer_name,
        "reward_model_name": args.reward_model_name,
        "cost_model_name": args.cost_model_name,
        "explicit_reward_scale": args.explicit_reward_scale,
        "split_name": split_name,
        "num_pairs": len(pairs),
        "max_length": args.max_length,
        "implicit_beta": args.implicit_beta,
        "sft_model_name": args.sft_model_name,
        "prompt_template": args.prompt_template,
        "pair_hash": _json_hash(
            {
                "prompts": [p["feature_prompt"] for p in pairs[:1000]],
                "chosen_head": [p["chosen"][:128] for p in pairs[:50]],
                "rejected_head": [p["rejected"][:128] for p in pairs[:50]],
                "n": len(pairs),
            }
        ),
    }
    safe_dataset = dataset_name.replace("/", "_").replace(":", "_")
    return Path(args.gap_cache_dir) / f"{_json_hash(payload)}_{safe_dataset}_{split_name}.pt"


def _load_gap_cache(path: Path) -> Optional[Tuple[List[str], torch.Tensor]]:
    if not args.use_gap_cache or args.force_recompute_gaps or not path.exists():
        return None
    obj = torch.load(path, map_location="cpu")
    prompts = list(obj["feature_prompts"])
    gaps = obj["signed_gaps"].float()
    print_local_main(f"[gap cache] loaded: {path} ({len(gaps)} pairs)")
    return prompts, gaps


def _save_gap_cache(path: Path, feature_prompts: List[str], signed_gaps: torch.Tensor) -> None:
    if not args.use_gap_cache or not accelerator.is_main_process:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"feature_prompts": feature_prompts, "signed_gaps": signed_gaps.detach().cpu()},
        path,
    )
    print_local_main(f"[gap cache] saved: {path}")


def _compute_implicit_objective_gaps_no_cache(
    model,
    adapter_name: str,
    pairs: List[dict],
    desc: str,
) -> Tuple[List[str], torch.Tensor]:
    """
    Compute signed implicit reward gaps.

    If launched by Accelerate with multiple processes, this shards preference pairs
    across ranks and gathers the results back in original pair order.
    """
    model.eval()
    model.set_adapter(adapter_name)

    world = accelerator.num_processes if args.shard_gaps_across_processes else 1
    rank = accelerator.process_index if args.shard_gaps_across_processes else 0

    indexed_local_pairs = [(idx, pairs[idx]) for idx in range(rank, len(pairs), world)]

    local_indices: List[int] = []
    local_feature_prompts: List[str] = []
    local_items: List[dict] = []

    for idx, p in indexed_local_pairs:
        local_indices.append(idx)
        local_feature_prompts.append(p["feature_prompt"])
        local_items.append(_tokenize_prompt_response(p["prompt"], p["chosen"]))
        local_items.append(_tokenize_prompt_response(p["prompt"], p["rejected"]))

    if len(local_items) == 0:
        local_records: List[Tuple[int, str, float]] = []
    else:
        loader = DataLoader(
            local_items,
            batch_size=args.per_device_batch_size * 2,
            collate_fn=_collate_logp,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        policy_logps_all = []
        ref_logps_all = []

        for batch in tqdm(
            loader,
            desc=f"{desc} [rank {accelerator.process_index}]",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
        ):
            policy_logps_all.append(_sequence_logps(model, batch))

            with model.disable_adapter():
                ref_logps_all.append(_sequence_logps(model, batch))

        policy_logps = torch.cat(policy_logps_all, dim=0).view(-1, 2)
        ref_logps = torch.cat(ref_logps_all, dim=0).view(-1, 2)

        chosen_score = args.implicit_beta * (policy_logps[:, 0] - ref_logps[:, 0])
        rejected_score = args.implicit_beta * (policy_logps[:, 1] - ref_logps[:, 1])
        signed_gap = (chosen_score - rejected_score).float()

        local_records = [
            (int(idx), prompt, float(gap))
            for idx, prompt, gap in zip(local_indices, local_feature_prompts, signed_gap.tolist())
        ]

    gathered = _all_gather_object(local_records)
    flat_records: List[Tuple[int, str, float]] = []
    for part in gathered:
        flat_records.extend(part)

    flat_records.sort(key=lambda x: x[0])
    feature_prompts = [r[1] for r in flat_records]
    signed_gaps = torch.tensor([r[2] for r in flat_records], dtype=torch.float32)

    if len(signed_gaps) != len(pairs):
        raise RuntimeError(
            f"Gap gathering failed: got {len(signed_gaps)} records for {len(pairs)} pairs. "
            f"num_processes={accelerator.num_processes}, rank={accelerator.process_index}"
        )

    return feature_prompts, signed_gaps


def _compute_implicit_objective_gaps(
    model,
    adapter_name: str,
    dataset_name: str,
    split_name: str,
    pairs: List[dict],
    desc: str,
) -> Tuple[List[str], torch.Tensor]:
    cache_path = _gap_cache_path(dataset_name, adapter_name, split_name, pairs)
    cached = _load_gap_cache(cache_path)
    if cached is not None:
        return cached

    feature_prompts, signed_gaps = _compute_implicit_objective_gaps_no_cache(
        model=model,
        adapter_name=adapter_name,
        pairs=pairs,
        desc=desc,
    )

    _save_gap_cache(cache_path, feature_prompts, signed_gaps)
    accelerator.wait_for_everyone()
    return feature_prompts, signed_gaps


def _load_score_model_and_tokenizer(model_name: str):
    """
    Load an explicit reward/cost model.

    PKU-Alignment/beaver-7b-v1.0-reward and -cost are Safe-RLHF score models
    in many environments, so AutoModelForScore is tried first. A sequence
    classification fallback is included for portability.
    """
    score_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if score_tokenizer.pad_token is None:
        score_tokenizer.pad_token = score_tokenizer.eos_token
    score_tokenizer.padding_side = "right"

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = (
        {"": accelerator.local_process_index}
        if torch.cuda.is_available() and not param_sharding_enabled()
        else None
    )

    try:
        from safe_rlhf.models import AutoModelForScore

        kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": True}
        if device_map is not None:
            kwargs["device_map"] = device_map
        score_model = AutoModelForScore.from_pretrained(model_name, **kwargs)
        model_kind = "safe_rlhf.AutoModelForScore"
    except Exception as e:
        print_local_main(
            f"[explicit scorer] AutoModelForScore load failed for {model_name}: {e}\n"
            f"[explicit scorer] falling back to AutoModelForSequenceClassification"
        )
        from transformers import AutoModelForSequenceClassification

        kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": True}
        if device_map is not None:
            kwargs["device_map"] = device_map
        score_model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        model_kind = "transformers.AutoModelForSequenceClassification"

    if hasattr(score_model, "config"):
        if getattr(score_model.config, "pad_token_id", None) is None:
            score_model.config.pad_token_id = score_tokenizer.pad_token_id
        if hasattr(score_model.config, "use_cache"):
            score_model.config.use_cache = False

    score_model.to(accelerator.device)
    score_model.eval()
    print_local_main(f"[explicit scorer] loaded {model_kind}: {model_name}")
    return score_tokenizer, score_model


def _tokenize_score_prompt_response(score_tokenizer, prompt: str, response: str) -> dict:
    toks = score_tokenizer(
        prompt + response,
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = toks["input_ids"]
    attention_mask = toks["attention_mask"]

    eos_id = getattr(score_tokenizer, "eos_token_id", None)
    if eos_id is not None and len(input_ids) < args.max_length:
        input_ids = input_ids + [eos_id]
        attention_mask = attention_mask + [1]

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _collate_score(features: Sequence[dict], score_tokenizer) -> dict:
    return score_tokenizer.pad(features, padding=True, return_tensors="pt")


def _extract_scalar_scores_from_output(output, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert various reward-model outputs into one scalar per sequence.

    Safe-RLHF score models typically expose `end_scores`; sequence classifiers
    expose `logits`. If token-level scores are returned, use the last non-padding
    position according to attention_mask.
    """
    score = None

    if isinstance(output, dict):
        for key in ("end_scores", "scores", "logits", "reward", "rewards"):
            if key in output:
                score = output[key]
                break
    else:
        for key in ("end_scores", "scores", "logits", "reward", "rewards"):
            if hasattr(output, key):
                score = getattr(output, key)
                break
        if score is None and isinstance(output, (tuple, list)) and len(output) > 0:
            score = output[0]

    if score is None:
        raise RuntimeError(
            "Could not extract scalar scores from reward model output. "
            "Expected one of: end_scores, scores, logits, reward, rewards."
        )

    if not isinstance(score, torch.Tensor):
        score = torch.as_tensor(score, device=attention_mask.device)

    if score.dim() == 0:
        score = score.view(1)
    elif score.dim() == 1:
        pass
    elif score.dim() == 2:
        if score.shape[1] == 1:
            score = score[:, 0]
        elif score.shape[1] == attention_mask.shape[1]:
            last_idx = attention_mask.long().sum(dim=1).clamp_min(1) - 1
            score = score.gather(1, last_idx[:, None]).squeeze(1)
        else:
            score = score[:, 0]
    elif score.dim() == 3:
        if score.shape[-1] == 1:
            score = score.squeeze(-1)
        if score.dim() == 2 and score.shape[1] == attention_mask.shape[1]:
            last_idx = attention_mask.long().sum(dim=1).clamp_min(1) - 1
            score = score.gather(1, last_idx[:, None]).squeeze(1)
        else:
            score = score.reshape(score.shape[0], -1)[:, -1]
    else:
        score = score.reshape(score.shape[0], -1)[:, -1]

    return score.float()


@torch.inference_mode()
def _sequence_scores(score_model, batch: dict) -> torch.Tensor:
    batch = {k: v.to(accelerator.device, non_blocking=True) for k, v in batch.items()}
    output = score_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    scores = _extract_scalar_scores_from_output(output, batch["attention_mask"])
    return scores.detach().cpu()


def _compute_explicit_objective_gaps_no_cache(
    score_model,
    score_tokenizer,
    pairs: List[dict],
    scorer_sign: float,
    desc: str,
) -> Tuple[List[str], torch.Tensor]:
    """
    Compute signed explicit scorer gaps.

    scorer_sign=+1: helpfulness/reward scorer s(x,y)=reward(x,y)
    scorer_sign=-1: harmlessness scorer s(x,y)=-cost(x,y)
    """
    score_model.eval()

    world = accelerator.num_processes if args.shard_gaps_across_processes else 1
    rank = accelerator.process_index if args.shard_gaps_across_processes else 0
    indexed_local_pairs = [(idx, pairs[idx]) for idx in range(rank, len(pairs), world)]

    local_indices: List[int] = []
    local_feature_prompts: List[str] = []
    local_items: List[dict] = []

    for idx, p in indexed_local_pairs:
        local_indices.append(idx)
        local_feature_prompts.append(p["feature_prompt"])
        local_items.append(_tokenize_score_prompt_response(score_tokenizer, p["prompt"], p["chosen"]))
        local_items.append(_tokenize_score_prompt_response(score_tokenizer, p["prompt"], p["rejected"]))

    if len(local_items) == 0:
        local_records: List[Tuple[int, str, float]] = []
    else:
        loader = DataLoader(
            local_items,
            batch_size=args.per_device_batch_size * 2,
            collate_fn=lambda xs: _collate_score(xs, score_tokenizer),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        scores_all = []
        for batch in tqdm(
            loader,
            desc=f"{desc} [rank {accelerator.process_index}]",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
        ):
            scores_all.append(_sequence_scores(score_model, batch))

        scores = torch.cat(scores_all, dim=0).view(-1, 2)
        chosen_score = scorer_sign * scores[:, 0]
        rejected_score = scorer_sign * scores[:, 1]
        signed_gap = args.explicit_reward_scale * (chosen_score - rejected_score)
        signed_gap = signed_gap.float()

        local_records = [
            (int(idx), prompt, float(gap))
            for idx, prompt, gap in zip(local_indices, local_feature_prompts, signed_gap.tolist())
        ]

    gathered = _all_gather_object(local_records)
    flat_records: List[Tuple[int, str, float]] = []
    for part in gathered:
        flat_records.extend(part)

    flat_records.sort(key=lambda x: x[0])
    feature_prompts = [r[1] for r in flat_records]
    signed_gaps = torch.tensor([r[2] for r in flat_records], dtype=torch.float32)

    if len(signed_gaps) != len(pairs):
        raise RuntimeError(
            f"Explicit gap gathering failed: got {len(signed_gaps)} records for {len(pairs)} pairs. "
            f"num_processes={accelerator.num_processes}, rank={accelerator.process_index}"
        )

    return feature_prompts, signed_gaps


def _compute_explicit_objective_gaps(
    score_model,
    score_tokenizer,
    scorer_name: str,
    scorer_sign: float,
    dataset_name: str,
    split_name: str,
    pairs: List[dict],
    desc: str,
) -> Tuple[List[str], torch.Tensor]:
    cache_path = _gap_cache_path(dataset_name, scorer_name, split_name, pairs)
    cached = _load_gap_cache(cache_path)
    if cached is not None:
        return cached

    feature_prompts, signed_gaps = _compute_explicit_objective_gaps_no_cache(
        score_model=score_model,
        score_tokenizer=score_tokenizer,
        pairs=pairs,
        scorer_sign=scorer_sign,
        desc=desc,
    )
    _save_gap_cache(cache_path, feature_prompts, signed_gaps)
    accelerator.wait_for_everyone()
    return feature_prompts, signed_gaps


def _release_torch_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _parse_noise_objectives(spec: str, num_objectives: int) -> set[int]:
    spec = spec.strip().lower()

    if spec == "all":
        return set(range(num_objectives))

    name_to_idx = {
        "better": 0,
        "help": 0,
        "helpful": 0,
        "helpfulness": 0,
        "safer": 1,
        "safe": 1,
        "safety": 1,
        "harmless": 1,
        "harmlessness": 1,
    }

    out = set()
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue

        if token in name_to_idx:
            idx = name_to_idx[token]
        else:
            idx = int(token)

        if idx < 0 or idx >= num_objectives:
            raise ValueError(f"Invalid noise objective index: {idx}")

        out.add(idx)

    if not out:
        raise ValueError(f"Could not parse noise_objectives={spec!r}")

    return out


def _apply_preference_flip_noise(
    signed_gaps: torch.Tensor,
    objective_idx: torch.Tensor,
    noise_ratio: float,
    noise_objectives: str,
    num_objectives: int,
    seed: int,
    split_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Apply controlled preference flip noise.

    Since signed_gap = s(chosen) - s(rejected) and the training label is chosen > rejected,
    flipping the preference is equivalent to signed_gap <- -signed_gap.
    """
    if noise_ratio < 0.0 or noise_ratio > 1.0:
        raise ValueError(f"noise_ratio must be in [0, 1], got {noise_ratio}")

    noisy_gaps = signed_gaps.clone()
    flip_mask = torch.zeros_like(signed_gaps, dtype=torch.bool)
    target_objectives = _parse_noise_objectives(noise_objectives, num_objectives)

    if noise_ratio == 0.0:
        stats = {
            "split": split_name,
            "target_objectives": sorted(target_objectives),
            "requested_noise_ratio": float(noise_ratio),
            "num_eligible": 0,
            "num_flipped": 0,
            "actual_noise_ratio_among_eligible": 0.0,
            "actual_noise_ratio_among_all": 0.0,
        }
        return noisy_gaps, flip_mask, stats

    eligible = torch.zeros_like(signed_gaps, dtype=torch.bool)
    for obj in target_objectives:
        eligible |= objective_idx.eq(obj)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    rand = torch.rand(signed_gaps.shape[0], generator=generator)
    flip_mask = eligible.cpu() & (rand < noise_ratio)
    flip_mask = flip_mask.to(signed_gaps.device)

    noisy_gaps[flip_mask] = -noisy_gaps[flip_mask]

    num_eligible = int(eligible.sum().item())
    num_flipped = int(flip_mask.sum().item())
    actual_ratio_eligible = num_flipped / max(num_eligible, 1)
    actual_ratio_all = num_flipped / max(int(signed_gaps.numel()), 1)

    stats = {
        "split": split_name,
        "target_objectives": sorted(target_objectives),
        "requested_noise_ratio": float(noise_ratio),
        "num_eligible": num_eligible,
        "num_flipped": num_flipped,
        "actual_noise_ratio_among_eligible": float(actual_ratio_eligible),
        "actual_noise_ratio_among_all": float(actual_ratio_all),
    }

    print_local_main(
        f"[noise:{split_name}] target_objectives={sorted(target_objectives)}, "
        f"requested_ratio={noise_ratio:.3f}, eligible={num_eligible}, flipped={num_flipped}, "
        f"actual_ratio_eligible={actual_ratio_eligible:.4f}, actual_ratio_all={actual_ratio_all:.4f}"
    )

    return noisy_gaps, flip_mask, stats


class SimpleLogPrecisionCalibrator(nn.Module):
    """
    MAP-only log-precision calibrator.

    Modes:
      fixed_objective:
        a_i(x) = u_i

      direct_prompt:
        a_i(x) = b_i + v_i^T phi(x)
    """

    def __init__(self, feature_dim: int, num_objectives: int, mode: str):
        super().__init__()

        if mode not in {"fixed_objective", "direct_prompt"}:
            raise ValueError(f"Unknown calibration mode: {mode}")

        self.feature_dim = feature_dim
        self.num_objectives = num_objectives
        self.mode = mode

        # fixed_objective parameters: a_i = fixed_u_i
        self.fixed_u = nn.Parameter(torch.zeros(num_objectives))

        # direct_prompt parameters: a_i(x)=direct_b_i + direct_V_i^T phi(x)
        self.direct_b = nn.Parameter(torch.zeros(num_objectives))
        self.direct_V = nn.Parameter(torch.zeros(num_objectives, feature_dim))

    def direct_a_all(self, features: torch.Tensor) -> torch.Tensor:
        """
        Return all objective-wise direct log-precisions.

        Shape:
            features: [N, d]
            output:   [N, m]
        """
        return self.direct_b.unsqueeze(0) + features @ self.direct_V.T

    def log_precision_all(self, features: torch.Tensor) -> torch.Tensor:
        if self.mode == "fixed_objective":
            a_all = self.fixed_u.unsqueeze(0).expand(
                features.shape[0],
                self.num_objectives,
            )

        elif self.mode == "direct_prompt":
            a_all = self.direct_a_all(features)

        else:
            raise ValueError(f"Unknown calibration mode: {self.mode}")

        return torch.clamp(a_all, args.clamp_a_min, args.clamp_a_max)

    def log_precision(self, features: torch.Tensor, objective_idx: torch.Tensor) -> torch.Tensor:
        a_all = self.log_precision_all(features)
        return a_all.gather(1, objective_idx[:, None]).squeeze(1)

    def forward(
        self,
        features: torch.Tensor,
        objective_idx: torch.Tensor,
        signed_gaps: torch.Tensor,
    ) -> torch.Tensor:
        a = self.log_precision(features, objective_idx)
        return torch.exp(a) * signed_gaps

    def prior_penalty(self) -> torch.Tensor:
        penalty = torch.zeros((), device=self.fixed_u.device)

        if self.mode == "fixed_objective":
            penalty = penalty + 0.5 * torch.sum((self.fixed_u / args.u_std) ** 2)

        elif self.mode == "direct_prompt":
            penalty = penalty + 0.5 * torch.sum((self.direct_b / args.u_std) ** 2)
            penalty = penalty + 0.5 * torch.sum((self.direct_V / args.W_std) ** 2)

        else:
            raise ValueError(f"Unknown calibration mode: {self.mode}")

        return penalty


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _balanced_logits_and_labels(logits: np.ndarray):
    """
    Original preference pair is always chosen > rejected, so y=1.
    For standard binary metrics, add the reversed pair:
        logit' = -logit, y'=0.
    This keeps per-pair NLL identical while giving a balanced binary set for ECE.
    """
    logits = np.asarray(logits, dtype=np.float64)
    all_logits = np.concatenate([logits, -logits], axis=0)
    labels = np.concatenate([np.ones_like(logits), np.zeros_like(logits)], axis=0)
    return all_logits, labels


def _binary_metrics_from_signed_logits(logits: np.ndarray, n_bins: int = 10) -> dict:
    
    logits_bal, labels = _balanced_logits_and_labels(logits)
    probs = _sigmoid_np(logits_bal)

    eps = 1e-12
    nll = -np.mean(labels * np.log(probs + eps) + (1.0 - labels) * np.log(1.0 - probs + eps))
    brier = np.mean((probs - labels) ** 2)
    acc = np.mean((probs >= 0.5) == labels)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    conf = np.maximum(probs, 1.0 - probs)
    pred = (probs >= 0.5).astype(np.float64)
    correct = (pred == labels).astype(np.float64)

    for b, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if b == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())

    return {
        "nll": float(nll),
        "brier": float(brier),
        "ece": float(ece),
        "acc": float(acc),
        "mean_prob_chosen": float(_sigmoid_np(np.asarray(logits)).mean()),
    }

def _binary_metrics_from_chosen_probs(p_chosen: np.ndarray, n_bins: int = 10) -> dict:
    """
    Metrics from posterior predictive probabilities for the original pair.

    p_chosen[n] = P(y_w > y_l | x_n), where the observed label is y=1.
    For balanced ECE, add the reversed pair with probability 1 - p_chosen
    and label 0.
    """

    p_chosen = np.asarray(p_chosen, dtype=np.float64)
    
    probs = np.concatenate([p_chosen, 1.0 - p_chosen], axis=0)
    labels = np.concatenate([np.ones_like(p_chosen), np.zeros_like(p_chosen)], axis=0)

    eps = 1e-12
    nll = -np.mean(labels * np.log(probs + eps) + (1.0 - labels) * np.log(1.0 - probs + eps))
    brier = np.mean((probs - labels) ** 2)
    acc = np.mean((probs >= 0.5) == labels)

    bins =  np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    confidence = np.maximum(probs, 1.0 - probs)
    pred = (probs >= 0.5).astype(np.float64)
    correct = (pred == labels).astype(np.float64)

    for b, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if b == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        
        if mask.sum() == 0:
            continue
        
        ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    
    return {
        "nll": float(nll),
        "brier": float(brier),
        "ece": float(ece),
        "acc": float(acc),
        "mean_prob_chosen": float(p_chosen.mean()),
    }


def _fit_calibrator(
    mode: str,
    train_features: torch.Tensor,
    train_obj_idx: torch.Tensor,
    train_gaps: torch.Tensor,
    valid_features: torch.Tensor,
    valid_obj_idx: torch.Tensor,
    valid_gaps: torch.Tensor,
    num_objectives: int,
) -> Tuple[SimpleLogPrecisionCalibrator, dict]:
    device = accelerator.device
    feature_dim = train_features.shape[1]

    model = SimpleLogPrecisionCalibrator(feature_dim, num_objectives, mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.calib_lr, weight_decay=0.0)

    train_features = train_features.to(device)
    train_obj_idx = train_obj_idx.to(device)
    train_gaps = train_gaps.to(device)

    valid_features = valid_features.to(device)
    valid_obj_idx = valid_obj_idx.to(device)
    valid_gaps = valid_gaps.to(device)

    n = train_gaps.shape[0]
    best_state = None
    best_metric = float("inf")

    if args.early_select_by not in {"nll", "ece"}:
        raise ValueError("--early_select_by must be either 'nll' or 'ece'.")

    for step in range(args.calib_steps):
        idx = torch.randint(0, n, (min(args.calib_batch_size, n),), device=device)

        logits = model(train_features[idx], train_obj_idx[idx], train_gaps[idx])

        # signed_gap assumes label=1, so BCE-with-logits reduces to softplus(-logit).
        nll = F.softplus(-logits).mean()

        # MAP objective. Divide prior by n to keep scale similar to average NLL.
        loss = nll + model.prior_penalty() / n

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % args.calib_log_every == 0 or (step + 1) == args.calib_steps:
            model.eval()
            with torch.no_grad():
                valid_logits = model(valid_features, valid_obj_idx, valid_gaps).detach().cpu().numpy()
                metrics = _binary_metrics_from_signed_logits(valid_logits)
            model.train()

            score = metrics[args.early_select_by]
            if score < best_metric:
                best_metric = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            print_local_main(
                f"[{mode}] step={step+1} train_nll={nll.item():.4f} "
                f"valid_nll={metrics['nll']:.4f} valid_ece={metrics['ece']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        valid_logits = model(valid_features, valid_obj_idx, valid_gaps).detach().cpu().numpy()
        train_logits = model(train_features, train_obj_idx, train_gaps).detach().cpu().numpy()

    metrics = _binary_metrics_from_signed_logits(valid_logits)
    train_metrics = _binary_metrics_from_signed_logits(train_logits)
    metrics["train_nll"] = train_metrics["nll"]
    metrics["train_ece"] = train_metrics["ece"]

    with torch.no_grad():
        valid_features_device = valid_features.to(device)
        valid_obj_idx_device = valid_obj_idx.to(device)
        a_valid = model.log_precision(valid_features_device, valid_obj_idx_device).detach().cpu()
        metrics["valid_log_precision_mean"] = float(a_valid.mean())
        metrics["valid_log_precision_std"] = float(a_valid.std(unbiased=False))

        for obj in range(num_objectives):
            mask = valid_obj_idx.detach().cpu().eq(obj)
            if mask.any():
                metrics[f"valid_log_precision_obj{obj}_mean"] = float(a_valid[mask].mean())
                metrics[f"valid_precision_obj{obj}_mean"] = float(torch.exp(a_valid[mask]).mean())

        if mode == "fixed_objective":
            metrics["learned_log_precision"] = model.fixed_u.detach().cpu().tolist()
            metrics["learned_precision"] = torch.exp(model.fixed_u.detach().cpu()).tolist()

        if mode == "direct_prompt":
            metrics["direct_b"] = model.direct_b.detach().cpu().tolist()
            metrics["direct_b_precision"] = torch.exp(model.direct_b.detach().cpu()).tolist()

    return model, metrics

def _flatten_direct_theta(model: SimpleLogPrecisionCalibrator) -> torch.Tensor:
    """
    Flatten direct_prompt parameters in the order:
        theta = [direct_b, vec(direct_V)]
    where direct_V is flattened objective-major.
    """
    return torch.cat(
        [
            model.direct_b.detach().cpu().double().reshape(-1),
            model.direct_V.detach().cpu().double().reshape(-1),
        ],
        dim=0,
    )

def _make_direct_design_matrix(
    features: torch.Tensor,
    objective_idx: torch.Tensor,
    num_objectives: int,
    ) -> torch.Tensor:
    """
    Construct design matrix Psi for direct_prompt model.

    direct model:
        a_i(x) = b_i + v_i^T phi(x)

    theta order:
        [b_0, ..., b_{m-1}, V_0[0:d], V_1[0:d], ..., V_{m-1}[0:d]]

    Then:
        a_i(x_n) = Psi[n]^T theta.
    """
    X = features.detach().cpu().to(dtype=torch.float64)
    obj = objective_idx.detach().cpu().reshape(-1).to(dtype=torch.long)

    n, d = X.shape
    p = num_objectives + num_objectives * d

    if obj.numel() != n:
        raise RuntimeError(
            f"objective_idx length mismatch: obj={obj.numel()}, features={n}"
        )

    if obj.min().item() < 0 or obj.max().item() >= num_objectives:
        raise RuntimeError(
            f"objective_idx out of range: min={obj.min().item()}, "
            f"max={obj.max().item()}, num_objectives={num_objectives}"
        )

    Psi = torch.zeros((n, p), dtype=torch.float64)

    row_idx = torch.arange(n, dtype=torch.long)

    # bias block: b_i column
    Psi[row_idx, obj] = 1.0

    # feature block: V_i^T phi(x)
    feat_offset = num_objectives + obj[:, None] * d
    feat_cols = feat_offset + torch.arange(d, dtype=torch.long)[None, :]

    Psi[row_idx[:, None], feat_cols] = X

    return Psi

def _direct_prior_precision_vector(
    feature_dim: int,
    num_objectives: int,
    ) -> torch.Tensor:
    """
    Prior precision vector for direct_prompt parameters.

    prior:
        direct_b_i ~ N(0, u_std^2)
        direct_V_ij ~ N(0, W_std^2)
    """

    b_prec = torch.full((num_objectives,), 1.0 / (args.u_std ** 2), dtype=torch.float64)
    v_prec = torch.full(
        (num_objectives * feature_dim,),
        1.0 / (args.W_std ** 2),
        dtype=torch.float64,
    )
    return torch.cat([b_prec, v_prec], dim=0)

def _fit_full_laplace_for_direct_prompt(
    direct_model: SimpleLogPrecisionCalibrator,
    train_features: torch.Tensor,
    train_obj_idx: torch.Tensor,
    train_gaps: torch.Tensor,
    num_objectives: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Textbook full Laplace approximation for direct_prompt.

    Computes:
        H = Hessian of the negative log posterior at theta_hat.

    No diagonal approximation.
    No damping.
    No jitter.

    If H is not positive definite, Cholesky will fail. That is intentional:
    without damping, textbook Laplace requires a positive definite local
    curvature matrix.
    """
    if direct_model.mode != "direct_prompt":
        raise ValueError("Full Laplace ablation is implemented only for direct_prompt mode.")
    
    X = train_features.detach().cpu().to(dtype=torch.float64)
    obj = train_obj_idx.detach().cpu().reshape(-1).to(dtype=torch.long)
    gaps = train_gaps.detach().cpu().reshape(-1).to(dtype=torch.float64)

    n, d = X.shape
    theta_hat = _flatten_direct_theta(direct_model)

    Psi = _make_direct_design_matrix(
        features=X,
        objective_idx=obj,
        num_objectives=num_objectives,
    )

    # a = Psi theta, t = exp(a) * gap
    # For textbook Laplace, use the unclamped linear a_i(x).
    a = Psi @ theta_hat
    t = torch.exp(a) * gaps

    sig_pos = torch.sigmoid(t)
    sig_neg = torch.sigmoid(-t)

    # Negative log-likelihood curvature wrt a.
    # loglik second derivative:
    #   l''(a) = t sigma(-t) [1 - t sigma(t)]
    # negative loglik Hessian:
    #   -l''(a) = t sigma(-t) [t sigma(t) - 1]
    h = t * sig_neg * (t * sig_pos - 1.0)

    # H_lik = Psi^T diag(h) Psi
    H_lik = Psi.T @ (Psi * h[:, None])

    prior_prec = _direct_prior_precision_vector(
        feature_dim=d,
        num_objectives=num_objectives,
    )

    H = H_lik + torch.diag(prior_prec)
    H = 0.5 * (H + H.T)

    eigvals = torch.linalg.eigvalsh(H)
    min_eig = float(eigvals.min().item())
    max_eig = float(eigvals.max().item())

    # No damping / no jitter. This will fail if H is not PD.
    L = torch.linalg.cholesky(H)

    info = {
        "laplace_type": "full_hessian_no_damping",
        "num_train": int(n),
        "feature_dim": int(d),
        "num_objectives": int(num_objectives),
        "num_params": int(theta_hat.numel()),
        "hessian_min_eig": min_eig,
        "hessian_max_eig": max_eig,
        "hessian_condition_number": float(max_eig / max(min_eig, 1e-300)),
    }

    return theta_hat, L, info

def _evaluate_direct_laplace_mc(
    direct_model: SimpleLogPrecisionCalibrator,
    train_features: torch.Tensor,
    train_obj_idx: torch.Tensor,
    train_gaps: torch.Tensor,
    valid_features: torch.Tensor,
    valid_obj_idx: torch.Tensor,
    valid_gaps: torch.Tensor,
    num_objectives: int,
    num_samples: int,
    seed: int,
) -> Tuple[dict, dict]:
    """
    Evaluate posterior predictive probabilities using full Laplace MC.

    theta_s ~ N(theta_hat, H^{-1})
    p_n = 1/S sum_s sigmoid(exp(a_i(x_n; theta_s)) * gap_n)

    Metrics are computed from p_n rather than from a single logit.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    theta_hat, L, info = _fit_full_laplace_for_direct_prompt(
        direct_model=direct_model,
        train_features=train_features,
        train_obj_idx=train_obj_idx,
        train_gaps=train_gaps,
        num_objectives=num_objectives,
    )

    Xv = valid_features.detach().cpu().double()
    objv = valid_obj_idx.detach().cpu().long()
    gapsv = valid_gaps.detach().cpu().double()

    Psi_v = _make_direct_design_matrix(
        features=Xv,
        objective_idx=objv,
        num_objectives=num_objectives,
    )

    p = theta_hat.numel()

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    # z: [p, S]
    z = torch.randn((p, num_samples), dtype=torch.float64, generator=gen)

    # H = L L^T, covariance = H^{-1}.
    # epsilon = L^{-T} z has covariance H^{-1}.
    eps = torch.linalg.solve_triangular(L.T, z, upper=True).T  # [S, p]
    theta_samples = theta_hat[None, :] + eps                  # [S, p]

    # a_valid: [N, S]
    a_valid = Psi_v @ theta_samples.T
    a_valid = torch.clamp(a_valid, args.clamp_a_min, args.clamp_a_max)

    logits = torch.exp(a_valid) * gapsv[:, None]
    probs = torch.sigmoid(logits)

    p_chosen = probs.mean(dim=1).detach().cpu().numpy()
    metrics = _binary_metrics_from_chosen_probs(p_chosen)

    info = dict(info)
    info.update(
        {
            "mc_samples": int(num_samples),
            "mc_seed": int(seed),
            "posterior_predictive": "mean_sigmoid_probability",
        }
    )

    metrics["laplace_info"] = info

    return metrics, info

def _feature_cache_path(feature_prompts: List[str], split_name: str) -> Path:
    payload = {
        "type": "prompt_features_v2",
        "split_name": split_name,
        "sft_model_name": args.sft_model_name,
        "feature_source": args.feature_source,
        "feature_model_name": args.feature_model_name,
        "feature_pooling": args.feature_pooling,
        "feature_max_length": args.feature_max_length,
        "prompt_template": args.prompt_template,
        "num_prompts": len(feature_prompts),
        "prompt_hash": _json_hash({"prompts": feature_prompts[:2000], "n": len(feature_prompts)}),
    }
    return Path(args.feature_cache_dir) / f"{_json_hash(payload)}_{split_name}.pt"


def _parse_weight_vector(spec: str, num_objectives: int) -> torch.Tensor:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if len(vals) != num_objectives:
        raise ValueError(
            f"diagnostic_weight must have {num_objectives} values, got {len(vals)}: {spec}"
        )

    w = torch.tensor(vals, dtype=torch.float32)
    if torch.any(w < 0):
        raise ValueError(f"diagnostic_weight must be non-negative: {spec}")

    s = w.sum()
    if s <= 0:
        raise ValueError(f"diagnostic_weight sum must be positive: {spec}")

    return w / s


@torch.no_grad()
def _dump_direct_a_diagnostics(
    direct_model: SimpleLogPrecisionCalibrator,
    valid_features: torch.Tensor,
    valid_obj_idx: torch.Tensor,
    valid_gaps: torch.Tensor,
    valid_prompts: List[str],
    valid_chosen: List[str],
    valid_rejected: List[str],
    num_objectives: int,
    objective_names: List[str],
    output_path: str,
    diagnostic_weight: str,
    topk: int,
) -> None:
    """
    Save prompt-level diagnostics for direct a_i(x).

    direct model:
        a_i(x) = b_i + v_i^T phi(x)

    canonical decomposition:
        rho(x) = mean_i a_i(x)
        delta_i(x) = a_i(x) - rho(x)

    calibrated weight:
        alpha_i(x,w) = w_i exp(delta_i(x)) / sum_j w_j exp(delta_j(x))
    """
    if not (
        len(valid_prompts)
        == len(valid_chosen)
        == len(valid_rejected)
        == valid_features.shape[0]
        == valid_obj_idx.shape[0]
        == valid_gaps.shape[0]
    ):
        raise RuntimeError(
            "Diagnostic inputs are misaligned: "
            f"prompts={len(valid_prompts)}, "
            f"chosen={len(valid_chosen)}, "
            f"rejected={len(valid_rejected)}, "
            f"features={valid_features.shape[0]}, "
            f"obj_idx={valid_obj_idx.shape[0]}, "
            f"gaps={valid_gaps.shape[0]}"
        )
    if not accelerator.is_main_process:
        return

    device = accelerator.device
    direct_model.eval()

    features = valid_features.to(device)
    obj_idx = valid_obj_idx.to(device)
    gaps = valid_gaps.to(device)

    # a_all: [N, m]
    a_all = direct_model.direct_a_all(features)
    a_all = torch.clamp(a_all, args.clamp_a_min, args.clamp_a_max)

    a_mean = a_all.mean(dim=1)
    centered_a_all = a_all - a_mean[:, None]

    w = _parse_weight_vector(diagnostic_weight, num_objectives).to(device)
    alpha_num = w[None, :] * torch.exp(centered_a_all)
    alpha_all = alpha_num / alpha_num.sum(dim=1, keepdim=True)

    a_i = a_all.gather(1, obj_idx[:, None]).squeeze(1)
    centered_a_i = centered_a_all.gather(1, obj_idx[:, None]).squeeze(1)
    alpha_i = alpha_all.gather(1, obj_idx[:, None]).squeeze(1)

    raw_prob = torch.sigmoid(gaps)
    cal_prob = torch.sigmoid(torch.exp(a_i) * gaps)

    a_all_np = a_all.detach().cpu().float().numpy()
    centered_a_all_np = centered_a_all.detach().cpu().float().numpy()
    alpha_all_np = alpha_all.detach().cpu().float().numpy()

    a_mean_np = a_mean.detach().cpu().float().numpy()
    a_i_np = a_i.detach().cpu().float().numpy()
    centered_a_i_np = centered_a_i.detach().cpu().float().numpy()
    alpha_i_np = alpha_i.detach().cpu().float().numpy()
    gaps_np = gaps.detach().cpu().float().numpy()
    raw_prob_np = raw_prob.detach().cpu().float().numpy()
    cal_prob_np = cal_prob.detach().cpu().float().numpy()
    obj_idx_np = valid_obj_idx.detach().cpu().numpy()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for n in range(len(valid_prompts)):
            row = {
                "prompt_index": int(n),
                "prompt": valid_prompts[n],
                "chosen": valid_chosen[n], 
                "rejected": valid_rejected[n],
                "objective_idx": int(obj_idx_np[n]),
                "objective_name": objective_names[int(obj_idx_np[n])]
                if int(obj_idx_np[n]) < len(objective_names)
                else f"obj{int(obj_idx_np[n])}",
                "gap": float(gaps_np[n]),
                "raw_prob_chosen": float(raw_prob_np[n]),
                "cal_prob_chosen": float(cal_prob_np[n]),
                "a_mean": float(a_mean_np[n]),
                "a_i": float(a_i_np[n]),
                "centered_a_i": float(centered_a_i_np[n]),
                "alpha_i": float(alpha_i_np[n]),
                "diagnostic_weight": diagnostic_weight,
            }

            for j in range(num_objectives):
                name = objective_names[j] if j < len(objective_names) else f"obj{j}"
                row[f"a_{name}"] = float(a_all_np[n, j])
                row[f"centered_a_{name}"] = float(centered_a_all_np[n, j])
                row[f"alpha_{name}"] = float(alpha_all_np[n, j])

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nsaved direct a_i(x) diagnostics to: {out}")

    print("\n[direct a_i(x) summary]")
    print(f"a_mean mean/std: {a_mean_np.mean():.4f} / {a_mean_np.std():.4f}")
    for j in range(num_objectives):
        name = objective_names[j] if j < len(objective_names) else f"obj{j}"
        print(
            f"{name}: "
            f"a mean/std={a_all_np[:, j].mean():.4f}/{a_all_np[:, j].std():.4f}, "
            f"centered_a mean/std={centered_a_all_np[:, j].mean():.4f}/{centered_a_all_np[:, j].std():.4f}, "
            f"alpha mean/std={alpha_all_np[:, j].mean():.4f}/{alpha_all_np[:, j].std():.4f}"
        )

    if num_objectives >= 2:
        target_j = 1
        target_name = objective_names[target_j] if target_j < len(objective_names) else f"obj{target_j}"
        w_cpu = _parse_weight_vector(diagnostic_weight, num_objectives)
        shifts = alpha_all_np[:, target_j] - float(w_cpu[target_j])

        top_pos = np.argsort(-shifts)[:topk]
        top_neg = np.argsort(shifts)[:topk]

        print(f"\n[top +{topk}] largest alpha_{target_name} - w_{target_name}")
        for idx in top_pos:
            print(
                f"[+shift={shifts[idx]:+.4f}] "
                f"a_mean={a_mean_np[idx]:+.4f}, "
                f"a0={a_all_np[idx,0]:+.4f}, a1={a_all_np[idx,1]:+.4f}, "
                f"alpha0={alpha_all_np[idx,0]:.4f}, alpha1={alpha_all_np[idx,1]:.4f} | "
                f"{valid_prompts[idx][:180].replace(chr(10), ' ')}"
            )

        print(f"\n[top -{topk}] smallest alpha_{target_name} - w_{target_name}")
        for idx in top_neg:
            print(
                f"[-shift={shifts[idx]:+.4f}] "
                f"a_mean={a_mean_np[idx]:+.4f}, "
                f"a0={a_all_np[idx,0]:+.4f}, a1={a_all_np[idx,1]:+.4f}, "
                f"alpha0={alpha_all_np[idx,0]:.4f}, alpha1={alpha_all_np[idx,1]:.4f} | "
                f"{valid_prompts[idx][:180].replace(chr(10), ' ')}"
            )


def _extract_features(model, feature_prompts: List[str]) -> torch.Tensor:
    if args.feature_source == "sft_hidden":
        feature_extractor = FrozenCausalLMPromptFeatureExtractor(
            model=model,
            tokenizer=tokenizer,
            max_length=args.feature_max_length,
            device=accelerator.device,
            pooling=args.feature_pooling,
            disable_adapter=True,
            prompt_template=args.prompt_template,
        )
        features = feature_extractor.encode(
            feature_prompts,
            batch_size=max(1, args.per_device_batch_size),
            device="cpu",
        )

    elif args.feature_source == "hf_model":
        feature_extractor = HFPromptFeatureExtractor(
            args.feature_model_name,
            max_length=args.feature_max_length,
            device=f"cuda:{accelerator.local_process_index}" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        )
        features = feature_extractor.encode(feature_prompts, batch_size=32, device="cpu")

    else:
        raise ValueError("feature_source must be either 'sft_hidden' or 'hf_model'.")

    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features)

    return features.float()


def _extract_features_cached(model, feature_prompts: List[str], split_name: str) -> torch.Tensor:
    path = _feature_cache_path(feature_prompts, split_name)
    if args.use_feature_cache and not args.force_recompute_features and path.exists():
        obj = torch.load(path, map_location="cpu")
        print_local_main(f"[feature cache] loaded: {path} {tuple(obj['features'].shape)}")
        return obj["features"].float()

    features = _extract_features(model, feature_prompts)

    if args.use_feature_cache and accelerator.is_main_process:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features.detach().cpu()}, path)
        print_local_main(f"[feature cache] saved: {path} {tuple(features.shape)}")

    accelerator.wait_for_everyone()
    return features.float()


def _standardize_train_valid_features(
    train_features: torch.Tensor,
    valid_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    if not args.normalize_features:
        return train_features, valid_features, {"enabled": False}

    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train_z = (train_features - mean) / std
    valid_z = (valid_features - mean) / std
    stats = {
        "enabled": True,
        "feature_dim": int(train_features.shape[1]),
        "train_feature_mean_abs_after_norm": float(train_z.mean(dim=0).abs().mean()),
        "train_feature_std_mean_after_norm": float(train_z.std(dim=0, unbiased=False).mean()),
    }
    return train_z.float(), valid_z.float(), stats


def _reduce_features_train_only_pca(
    train_features: torch.Tensor,
    valid_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Reduce prompt features using PCA fitted on train features only.

    Important:
      - PCA mean/components are estimated only from train features.
      - Valid features are projected using the train-fitted PCA basis.
      - After projection, reduced features are standardized using train statistics.
    """
    if args.feature_reduce_dim is None:
        return train_features, valid_features, {"enabled": False}

    reduce_dim = int(args.feature_reduce_dim)
    if reduce_dim <= 0:
        return train_features, valid_features, {"enabled": False, "requested_dim": reduce_dim}

    X_train = train_features.float().cpu()
    X_valid = valid_features.float().cpu()

    n_train, original_dim = X_train.shape
    max_rank = min(n_train, original_dim)
    k = min(reduce_dim, max_rank)

    if k < reduce_dim:
        print_local_main(
            f"[feature PCA] requested_dim={reduce_dim} is larger than max_rank={max_rank}; using k={k}."
        )

    pca_mean = X_train.mean(dim=0, keepdim=True)
    Xc = X_train - pca_mean

    # Exact SVD is deterministic. For current sanity-check sizes this is usually
    # cheap compared with computing scorer gaps through LLMs.
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    components = Vh[:k].contiguous()

    train_reduced = (X_train - pca_mean) @ components.T
    valid_reduced = (X_valid - pca_mean) @ components.T

    red_mean = train_reduced.mean(dim=0, keepdim=True)
    red_std = train_reduced.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train_reduced = (train_reduced - red_mean) / red_std
    valid_reduced = (valid_reduced - red_mean) / red_std

    total_var = torch.sum(S**2).clamp_min(1e-12)
    explained = torch.sum(S[:k] ** 2) / total_var

    stats = {
        "enabled": True,
        "original_dim": int(original_dim),
        "requested_dim": int(reduce_dim),
        "reduced_dim": int(k),
        "num_train_features": int(n_train),
        "explained_var_ratio_sum": float(explained.item()),
        "reduced_train_mean_abs_after_norm": float(train_reduced.mean(dim=0).abs().mean()),
        "reduced_train_std_mean_after_norm": float(train_reduced.std(dim=0, unbiased=False).mean()),
    }

    print_local_main(
        f"[feature PCA] {original_dim} -> {k} "
        f"(explained_var_sum={stats['explained_var_ratio_sum']:.4f})"
    )

    return train_reduced.float(), valid_reduced.float(), stats


def _print_terminal_table(results: Dict[str, dict]) -> None:
    rows = []
    raw_name = "Implicit reward only" if args.reward_type == "implicit" else "Explicit reward only"
    order = [
        ("raw_scorer", raw_name),
        ("fixed_objective", "Fixed objective uncertainty"),
        ("direct_prompt", "Prompt-dependent direct a_i(x) MAP"),
        (
            "direct_prompt_laplace_mc",
            f"Direct a_i(x) + full Laplace MC-{args.laplace_mc_samples}",
        ),
    ]

    for key, name in order:
        if key not in results:
            continue
        m = results[key]
        rows.append(
            [
                name,
                f"{m['nll']:.4f}",
                f"{m['ece']:.4f}",
                f"{m['brier']:.4f}",
                f"{m['acc']:.4f}",
                f"{m['mean_prob_chosen']:.4f}",
            ]
        )

    headers = ["Model", "Valid NLL ↓", "ECE ↓", "Brier ↓", "Acc ↑", "Mean P(chosen) ↑"]
    widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))

    def fmt_row(row):
        return " | ".join(
            str(cell).ljust(widths[j]) if j == 0 else str(cell).rjust(widths[j])
            for j, cell in enumerate(row)
        )

    sep = "-+-".join("-" * w for w in widths)
    print("\n" + fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print("")


def _load_sft_base_model_for_features():
    print_local_main("loading SFT/reference model for prompt features...")
    base = AutoModelForCausalLM.from_pretrained(
        args.sft_model_name,
        use_flash_attention_2=args.use_flash_attention_2,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        **(
            {"device_map": {"": accelerator.local_process_index}}
            if torch.cuda.is_available() and not param_sharding_enabled()
            else {}
        ),
    )
    base.config.update({"use_cache": False, "pad_token_id": base.config.eos_token_id})
    base.to(accelerator.device)
    base.eval()
    return base


def _load_implicit_peft_model(objective_adapters: List[str]):
    print_local_main("loading SFT/reference model and objective adapters for implicit reward gaps...")
    base = AutoModelForCausalLM.from_pretrained(
        args.sft_model_name,
        use_flash_attention_2=args.use_flash_attention_2,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        **(
            {"device_map": {"": accelerator.local_process_index}}
            if torch.cuda.is_available() and not param_sharding_enabled()
            else {}
        ),
    )
    base.config.update({"use_cache": False, "pad_token_id": base.config.eos_token_id})

    model = PeftModel.from_pretrained(
        base,
        objective_adapters[0],
        adapter_name="obj0",
        is_trainable=False,
    )
    for i, path in enumerate(objective_adapters[1:], start=1):
        model.load_adapter(path, adapter_name=f"obj{i}", is_trainable=False)

    model.to(accelerator.device)
    model.eval()
    return model


def _objective_names_from_dataset_names(dataset_names: List[str]) -> List[str]:
    objective_names = []
    for ds_name in dataset_names:
        low = ds_name.lower()
        if "better" in low:
            objective_names.append("help")
        elif "safer" in low or "safe" in low:
            objective_names.append("safe")
        else:
            objective_names.append(f"obj{len(objective_names)}")
    return objective_names


def main():
    args.reward_type = args.reward_type.strip().lower()
    if args.reward_type not in {"implicit", "explicit"}:
        raise ValueError("reward_type must be either 'implicit' or 'explicit'.")

    objective_adapters = _split_csv(args.objective_adapter_names) if args.objective_adapter_names.strip() else []
    objective_datasets = _split_csv(args.objective_dataset_names)

    if not objective_datasets:
        raise ValueError("objective_dataset_names must be a non-empty comma-separated list.")

    if args.reward_type == "implicit":
        if len(objective_adapters) != len(objective_datasets):
            raise ValueError(
                "When --reward_type implicit, objective_adapter_names and objective_dataset_names "
                "must have the same length."
            )
    else:
        if len(objective_datasets) != 2:
            raise ValueError(
                "The current explicit reward setup expects exactly two objective datasets: "
                "obj0=helpfulness/reward and obj1=harmlessness/-cost."
            )
        if objective_adapters:
            print_local_main(
                "[explicit reward] objective_adapter_names was provided but will be ignored. "
                "Explicit gaps use reward_model_name and cost_model_name."
            )

    num_objectives = len(objective_datasets)
    objective_names = _objective_names_from_dataset_names(objective_datasets)

    print_local_main(
        f"reward_type={args.reward_type}, num_processes={accelerator.num_processes}, "
        f"process_index={accelerator.process_index}, feature_pooling={args.feature_pooling}, "
        f"feature_reduce_dim={args.feature_reduce_dim}, shard_gaps={args.shard_gaps_across_processes}"
    )
    if args.reward_type == "explicit":
        print_local_main(
            f"explicit scorer mapping: obj0 = reward({args.reward_model_name}), "
            f"obj1 = -cost({args.cost_model_name})"
        )

    scorer_model_for_features = None
    implicit_model = None

    if args.reward_type == "implicit":
        implicit_model = _load_implicit_peft_model(objective_adapters)
        scorer_model_for_features = implicit_model

    all_train_prompts: List[str] = []
    all_valid_prompts: List[str] = []
    all_train_gaps: List[torch.Tensor] = []
    all_valid_gaps: List[torch.Tensor] = []
    all_train_obj_idx: List[torch.Tensor] = []
    all_valid_obj_idx: List[torch.Tensor] = []
    all_valid_chosen: List[str] = []
    all_valid_rejected: List[str] = []

    for i, dataset_name in enumerate(objective_datasets):
        print_local_main(f"loading dataset for objective {i}: {dataset_name}")

        rdp = DATASET_CONFIGS[dataset_name](
            prompt_template=args.prompt_template,
            sanity_check=False,
        )
        dataset = rdp.get_preference_dataset(split="train")
        pairs = _build_pairs(dataset, args.max_examples_per_objective)

        train_pairs, valid_pairs = _prompt_level_split(
            pairs,
            valid_ratio=args.valid_ratio,
            seed=args.seed + i,
        )

        print_local_main(
            f"objective={i}: train_pairs={len(train_pairs)}, valid_pairs={len(valid_pairs)}"
        )

        if args.reward_type == "implicit":
            train_prompts, train_gaps = _compute_implicit_objective_gaps(
                implicit_model,
                adapter_name=f"obj{i}",
                dataset_name=dataset_name,
                split_name=f"implicit_obj{i}_train",
                pairs=train_pairs,
                desc=f"implicit obj{i}: train gaps",
            )
            valid_prompts, valid_gaps = _compute_implicit_objective_gaps(
                implicit_model,
                adapter_name=f"obj{i}",
                dataset_name=dataset_name,
                split_name=f"implicit_obj{i}_valid",
                pairs=valid_pairs,
                desc=f"implicit obj{i}: valid gaps",
            )

        else:
            if i == 0:
                scorer_name = args.reward_model_name
                scorer_sign = +1.0
                scorer_desc = "explicit_reward"
            elif i == 1:
                scorer_name = args.cost_model_name
                scorer_sign = -1.0  # harmlessness = -cost
                scorer_desc = "explicit_neg_cost"
            else:
                raise ValueError("Explicit reward mode currently supports only two objectives.")

            score_tokenizer, score_model = _load_score_model_and_tokenizer(scorer_name)
            train_prompts, train_gaps = _compute_explicit_objective_gaps(
                score_model=score_model,
                score_tokenizer=score_tokenizer,
                scorer_name=f"{scorer_desc}:{scorer_name}",
                scorer_sign=scorer_sign,
                dataset_name=dataset_name,
                split_name=f"explicit_obj{i}_train",
                pairs=train_pairs,
                desc=f"{scorer_desc} obj{i}: train gaps",
            )
            valid_prompts, valid_gaps = _compute_explicit_objective_gaps(
                score_model=score_model,
                score_tokenizer=score_tokenizer,
                scorer_name=f"{scorer_desc}:{scorer_name}",
                scorer_sign=scorer_sign,
                dataset_name=dataset_name,
                split_name=f"explicit_obj{i}_valid",
                pairs=valid_pairs,
                desc=f"{scorer_desc} obj{i}: valid gaps",
            )
            _release_torch_model(score_model)
            del score_tokenizer
            gc.collect()

        all_train_prompts.extend(train_prompts)
        all_valid_prompts.extend(valid_prompts)

        all_train_gaps.append(train_gaps)
        all_valid_gaps.append(valid_gaps)

        all_train_obj_idx.append(torch.full((len(train_gaps),), i, dtype=torch.long))
        all_valid_obj_idx.append(torch.full((len(valid_gaps),), i, dtype=torch.long))

        all_valid_chosen.extend([p["chosen"] for p in valid_pairs])
        all_valid_rejected.extend([p["rejected"] for p in valid_pairs])

    train_gaps_clean = torch.cat(all_train_gaps, dim=0)
    valid_gaps_clean = torch.cat(all_valid_gaps, dim=0)

    train_obj_idx = torch.cat(all_train_obj_idx, dim=0)
    valid_obj_idx = torch.cat(all_valid_obj_idx, dim=0)

    train_gaps, train_flip_mask, train_noise_stats = _apply_preference_flip_noise(
        signed_gaps=train_gaps_clean,
        objective_idx=train_obj_idx,
        noise_ratio=args.noise_ratio,
        noise_objectives=args.noise_objectives,
        num_objectives=num_objectives,
        seed=args.noise_seed,
        split_name="train",
    )

    if args.apply_noise_to_valid:
        valid_gaps_for_eval, valid_flip_mask, valid_noise_stats = _apply_preference_flip_noise(
            signed_gaps=valid_gaps_clean,
            objective_idx=valid_obj_idx,
            noise_ratio=args.noise_ratio,
            noise_objectives=args.noise_objectives,
            num_objectives=num_objectives,
            seed=args.noise_seed + 999,
            split_name="valid",
        )
        eval_split_name = "noisy_valid"
    else:
        valid_gaps_for_eval = valid_gaps_clean
        valid_flip_mask = torch.zeros_like(valid_gaps_clean, dtype=torch.bool)
        valid_noise_stats = {
            "split": "valid",
            "target_objectives": [],
            "requested_noise_ratio": 0.0,
            "num_eligible": 0,
            "num_flipped": 0,
            "actual_noise_ratio_among_eligible": 0.0,
            "actual_noise_ratio_among_all": 0.0,
        }
        eval_split_name = "clean_valid"

    # Load SFT/reference model only if needed for features. In implicit mode,
    # reuse the PEFT model with adapters disabled inside the feature extractor.
    if scorer_model_for_features is None:
        scorer_model_for_features = _load_sft_base_model_for_features()

    print_local_main("extracting train prompt features...")
    train_features = _extract_features_cached(
        scorer_model_for_features,
        all_train_prompts,
        split_name=f"{args.reward_type}_train",
    )

    print_local_main("extracting valid prompt features...")
    valid_features = _extract_features_cached(
        scorer_model_for_features,
        all_valid_prompts,
        split_name=f"{args.reward_type}_valid",
    )

    train_features, valid_features, feature_norm_stats = _standardize_train_valid_features(
        train_features,
        valid_features,
    )

    train_features, valid_features, feature_pca_stats = _reduce_features_train_only_pca(
        train_features,
        valid_features,
    )

    results: Dict[str, dict] = {}
    fitted_models: Dict[str, SimpleLogPrecisionCalibrator] = {}

    # 1) Raw scorer baseline: implicit reward gap or explicit reward/-cost gap.
    raw_valid_logits = valid_gaps_for_eval.detach().cpu().numpy()
    raw_train_logits = train_gaps.detach().cpu().numpy()
    results["raw_scorer"] = _binary_metrics_from_signed_logits(raw_valid_logits)
    train_raw_metrics = _binary_metrics_from_signed_logits(raw_train_logits)
    results["raw_scorer"]["train_nll"] = train_raw_metrics["nll"]
    results["raw_scorer"]["train_ece"] = train_raw_metrics["ece"]

    # 2) Objective-wise fixed uncertainty: a_i is learned but does not depend on prompt.
    for mode in ["fixed_objective", "direct_prompt"]:
        print_local_main(f"fitting calibration mode: {mode}")
        fitted_model, metrics = _fit_calibrator(
            mode=mode,
            train_features=train_features,
            train_obj_idx=train_obj_idx,
            train_gaps=train_gaps,
            valid_features=valid_features,
            valid_obj_idx=valid_obj_idx,
            valid_gaps=valid_gaps_for_eval,
            num_objectives=num_objectives,
        )
        results[mode] = metrics
        fitted_models[mode] = fitted_model

    # 3) Full Laplace posterior predictive ablation for direct_prompt.
    # Textbook setting: full Hessian, no diagonal approximation, no damping.
    # We evaluate posterior predictive probabilities with MC samples.
    if (
        accelerator.is_main_process
        and args.run_laplace_ablation
        and "direct_prompt" in fitted_models
    ):
        print_local_main(
            f"fitting full Laplace approximation for direct_prompt "
            f"and evaluating MC-{args.laplace_mc_samples} posterior predictive..."
        )

        try:
            laplace_metrics, laplace_info = _evaluate_direct_laplace_mc(
                direct_model=fitted_models["direct_prompt"],
                train_features=train_features,
                train_obj_idx=train_obj_idx,
                train_gaps=train_gaps,
                valid_features=valid_features,
                valid_obj_idx=valid_obj_idx,
                valid_gaps=valid_gaps_for_eval,
                num_objectives=num_objectives,
                num_samples=args.laplace_mc_samples,
                seed=args.laplace_seed,
            )
            results["direct_prompt_laplace_mc"] = laplace_metrics

            print_local_main(
                "[direct_prompt_laplace_mc] "
                f"valid_nll={laplace_metrics['nll']:.4f} "
                f"valid_ece={laplace_metrics['ece']:.4f} "
                f"valid_brier={laplace_metrics['brier']:.4f} "
                f"min_eig={laplace_info['hessian_min_eig']:.6e} "
                f"cond={laplace_info['hessian_condition_number']:.6e}"
            )

        except RuntimeError as e:
            # With no damping, Cholesky can fail if the full Hessian is not PD.
            # We record the failure rather than silently adding damping.
            print_local_main(
                "[direct_prompt_laplace_mc] failed because full Hessian "
                f"was not positive definite or Cholesky failed: {repr(e)}"
            )
            results["direct_prompt_laplace_mc_failure"] = {
                "error": repr(e),
                "laplace_type": "full_hessian_no_damping",
                "note": "No damping or diagonal approximation was applied.",
            }

    if args.dump_direct_diagnostics and "direct_prompt" in fitted_models:
        _dump_direct_a_diagnostics(
            direct_model=fitted_models["direct_prompt"],
            valid_features=valid_features,
            valid_obj_idx=valid_obj_idx,
            valid_gaps=valid_gaps_for_eval,
            valid_prompts=all_valid_prompts,
            valid_chosen=all_valid_chosen,
            valid_rejected=all_valid_rejected,
            num_objectives=num_objectives,
            objective_names=objective_names,
            output_path=args.direct_diagnostic_path,
            diagnostic_weight=args.diagnostic_weight,
            topk=args.direct_topk,
        )

    if accelerator.is_main_process:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "args": vars(args),
            "reward_type": args.reward_type,
            "explicit_scorer_mapping": (
                {
                    "obj0": {"score": "reward", "model": args.reward_model_name, "sign": +1.0},
                    "obj1": {"score": "-cost", "model": args.cost_model_name, "sign": -1.0},
                }
                if args.reward_type == "explicit"
                else None
            ),
            "objective_datasets": objective_datasets,
            "objective_names": objective_names,
            "num_train_pairs": int(train_gaps.numel()),
            "num_valid_pairs": int(valid_gaps_for_eval.numel()),
            "eval_split": eval_split_name,
            "noise": {
                "train": train_noise_stats,
                "valid": valid_noise_stats,
                "num_train_flipped": int(train_flip_mask.sum().item()),
                "num_valid_flipped": int(valid_flip_mask.sum().item()),
            },
            "feature_norm": feature_norm_stats,
            "feature_pca": feature_pca_stats,
            "posterior": {
                "used": False,
                "laplace": False,
                "monte_carlo": False,
                "damping": None,
                "note": "MAP-only calibration. No Laplace approximation or posterior sampling.",
            },
            "results": results,
        }

        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"\nsaved results to: {out}")
        print(f"eval_split: {eval_split_name}")
        print(f"train_noise: {train_noise_stats}")
        print(f"valid_noise: {valid_noise_stats}\n")

        _print_terminal_table(results)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
