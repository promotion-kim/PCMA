"""Evaluate a SafeRLHF RiC model over helpful/harmless preferences.

RiC training is weight-free. During evaluation, a preference weight is mapped to
an explicit target reward vector, inserted into the prompt as score tokens, and
used as the generation condition.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser
from trl import set_seed

from multi_reward_models_saferlhf_ric import SafeRLHFRewardModels
from utils import Instructions_n, load_main_tokenizer, map_rewards_from_preference


DEFAULT_PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"


def _split_csv(x: str) -> List[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


def _get_first(example: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    return None


def _format_prompt(raw_prompt: str, prompt_template: str) -> str:
    if "{raw_prompt}" in prompt_template:
        return prompt_template.format(raw_prompt=raw_prompt)
    return prompt_template + str(raw_prompt)


def _insert_score_tokens(prompt: str, scores: Sequence[float], instructions: Instructions_n) -> str:
    p = prompt.strip()
    if p.endswith("ASSISTANT:"):
        prefix = p[: -len("ASSISTANT:")].rstrip()
        suffix = "ASSISTANT:"
    elif p.endswith("Assistant:"):
        prefix = p[: -len("Assistant:")].rstrip()
        suffix = "Assistant:"
    else:
        prefix = p
        suffix = "ASSISTANT:"
    score_text = " ".join(
        f"{instructions.score_splits[i]} {round(float(scores[i]), 1)}" for i in range(len(scores))
    )
    return f"{prefix} {score_text} {suffix}"


def _clean_generated_text(prompt_text: str, generated_text: str) -> str:
    text = generated_text.strip("[PAD] ").strip("[PAD]").strip("<s>").strip("</s>").strip()
    prompt_clean = prompt_text.strip("[PAD] ").strip("[PAD]").strip("<s>").strip("</s>").strip()
    if text.startswith(prompt_clean):
        text = text[len(prompt_clean):].strip()
    if "</s>" in text:
        text = text.split("</s>", 1)[0].strip()
    for sep in ["\n\nHuman:", "\nHuman:", "\n\nUSER:", "\nUSER:", "###"]:
        if sep in text:
            text = text.split(sep, 1)[0].strip()
    return text


@dataclass
class ScriptArguments:
    base_model_name: str = field(default="PKU-Alignment/alpaca-7b-reproduced")
    peft_name: str = field(default="")
    dataset_name: str = field(default="PKU-Alignment/PKU-SafeRLHF-10K")
    split: str = field(default="test")
    prompt_template: str = field(default=DEFAULT_PROMPT_TEMPLATE)

    reward_names: str = field(default="helpful,harmless")
    reward_model_names: str = field(
        default="PKU-Alignment/beaver-7b-v1.0-reward,PKU-Alignment/beaver-7b-v1.0-cost"
    )
    reward_signs: str = field(default="1,-1")
    reward_stats_path: str = field(default="")
    reward_model_max_length: int = field(default=512)
    reward_batch_size: int = field(default=4)

    # Either evaluate a single helpfulness weight or a comma-separated list.
    preference: Optional[float] = field(default=None, metadata={"help": "helpfulness weight w_h"})
    preferences: str = field(default="0.0,0.3,0.5,0.7,1.0")
    target_map_method: str = field(default="l2", metadata={"help": "linf, l2, or linear; matches original RiC utility"})
    target_rewards: str = field(default="", metadata={"help": "optional comma-separated normalized target rewards"})

    save_directory: str = field(default="./logs_ric_saferlhf_eval")
    wandb_name: str = field(default="ric_saferlhf_eval")

    max_prompt_length: int = field(default=384)
    max_new_tokens: int = field(default=128)
    max_eval_samples: int = field(default=200)
    batch_size: int = field(default=1)
    seed: int = field(default=8888)
    load_in_8bit: bool = field(default=False)


args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
set_seed(args.seed)

reward_names = _split_csv(args.reward_names)
reward_model_names = _split_csv(args.reward_model_names)
reward_signs = [float(x) for x in _split_csv(args.reward_signs)]
if not (len(reward_names) == len(reward_model_names) == len(reward_signs)):
    raise ValueError("reward_names, reward_model_names, reward_signs must have same length.")
if len(reward_names) != 2:
    raise NotImplementedError("This evaluator currently expects helpful,harmless two-objective preferences.")

accelerator = Accelerator()
process_id = accelerator.local_process_index
gpu_id = process_id

tokenizer = load_main_tokenizer(args.base_model_name)
tokenizer.padding_side = "left"
instructions = Instructions_n(len(reward_names))

print("Loading RiC model...")
if args.load_in_8bit:
    model = AutoModelForCausalLM.from_pretrained(args.base_model_name, load_in_8bit=True, device_map=gpu_id)
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=gpu_id if torch.cuda.is_available() else None,
    )
model.resize_token_embeddings(len(tokenizer))
if args.peft_name:
    model = PeftModel.from_pretrained(model, args.peft_name)
if hasattr(model, "merge_and_unload"):
    model = model.merge_and_unload()
model.eval()

print("Loading evaluation reward models...")
reward_models_raw = SafeRLHFRewardModels(
    reward_model_names=reward_model_names,
    reward_tokenizer_names=reward_model_names,
    reward_signs=reward_signs,
    gpu_id=gpu_id,
    reward_stats_path=args.reward_stats_path if args.reward_stats_path else None,
    max_length=args.reward_model_max_length,
    batch_size=args.reward_batch_size,
)

print(f"Loading eval prompts: {args.dataset_name} split={args.split}")
try:
    ds = load_dataset(args.dataset_name, split=args.split)
except Exception:
    # PKU-SafeRLHF-10K may not expose a test split in some local wrappers.
    fallback_split = "validation"
    print(f"Could not load split={args.split}; falling back to {fallback_split}")
    ds = load_dataset(args.dataset_name, split=fallback_split)

rows: List[Dict[str, str]] = []
seen = set()
for ex in ds:
    raw_prompt = _get_first(ex, ["raw_prompt", "prompt", "input", "query"])
    if raw_prompt is None:
        continue
    raw_prompt = str(raw_prompt)
    if raw_prompt in seen:
        continue
    seen.add(raw_prompt)
    prompt = _format_prompt(raw_prompt, args.prompt_template)
    rows.append({"raw_prompt": raw_prompt, "prompt": prompt})
    if len(rows) >= args.max_eval_samples:
        break
if not rows:
    raise RuntimeError("No eval prompts found.")

base_eval_data = Dataset.from_list(rows)

# Original RiC maps preferences to target reward values using a reference reward distribution.
if args.target_rewards:
    manual_target = np.array([float(x) for x in _split_csv(args.target_rewards)], dtype=np.float32)
    if len(manual_target) != len(reward_names):
        raise ValueError("target_rewards length must match reward_names length.")
else:
    manual_target = None

if args.preference is not None:
    pref_values = [float(args.preference)]
else:
    pref_values = [float(x) for x in _split_csv(args.preferences)]

# Same reference construction as the original RiC evaluator, but deterministic.
rng = np.random.default_rng(args.seed)
rewards_reference_list = [rng.standard_normal(50000) for _ in reward_names]

generation_kwargs = {
    "max_new_tokens": args.max_new_tokens,
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.9,
    "temperature": 0.7,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "begin_suppress_tokens": [tokenizer.eos_token_id],
}

save_root = os.path.join(args.save_directory, args.wandb_name)
os.makedirs(save_root, exist_ok=True)
summary_rows = []

for w_helpful in pref_values:
    if not (0.0 <= w_helpful <= 1.0):
        raise ValueError(f"preference must be in [0,1], got {w_helpful}")
    preference_vec = np.array([w_helpful, 1.0 - w_helpful], dtype=np.float32)
    target = manual_target if manual_target is not None else map_rewards_from_preference(
        rewards_reference_list,
        preference_vec,
        method=args.target_map_method,
    ).reshape(-1)

    print("=" * 100)
    print(f"Evaluating RiC SafeRLHF preference helpful={w_helpful:.2f}, harmless={1.0-w_helpful:.2f}")
    print(f"Target normalized rewards: {target.tolist()}")

    def add_condition(sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt_with_score = _insert_score_tokens(str(sample["prompt"]), target, instructions)
        toks = tokenizer(
            prompt_with_score,
            truncation=True,
            max_length=args.max_prompt_length,
            padding=False,
        )
        sample["prompt_with_score"] = prompt_with_score
        sample["input_ids"] = toks["input_ids"]
        return sample

    eval_data = base_eval_data.map(add_condition, batched=False)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(
        eval_data.remove_columns([c for c in eval_data.column_names if c not in ["input_ids"]]),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=data_collator,
    )
    model_prepared, loader_prepared = accelerator.prepare(model, loader)

    all_generated_ids = []
    all_prompt_ids = []
    with torch.no_grad():
        for batch in tqdm(loader_prepared, desc="generating", dynamic_ncols=True):
            out = accelerator.unwrap_model(model_prepared).generate(batch["input_ids"], **generation_kwargs)
            all_generated_ids.extend(accelerator.gather_for_metrics(out))
            all_prompt_ids.extend(accelerator.gather_for_metrics(batch["input_ids"]))

    generated_texts = tokenizer.batch_decode(all_generated_ids, skip_special_tokens=False)
    prompt_texts = tokenizer.batch_decode(all_prompt_ids, skip_special_tokens=False)
    responses = [_clean_generated_text(p, g) for p, g in zip(prompt_texts, generated_texts)]

    # The number can be larger in distributed mode; keep local intended count.
    responses = responses[: len(eval_data)]
    raw_prompts = list(eval_data["raw_prompt"])[: len(responses)]
    formatted_prompts = list(eval_data["prompt"])[: len(responses)]
    conditioned_prompts = list(eval_data["prompt_with_score"])[: len(responses)]

    queries_responses = [(q, r) for q, r in zip(formatted_prompts, responses)]
    raw_scores = reward_models_raw.get_reward_model_scores(queries_responses, normalize=False)
    norm_scores = reward_models_raw.get_reward_model_scores(queries_responses, normalize=True) if args.reward_stats_path else None

    result = {
        "raw_prompt": raw_prompts,
        "prompt": formatted_prompts,
        "prompt_with_score": conditioned_prompts,
        "response": responses,
        "desired_score1": [float(target[0])] * len(responses),
        "desired_score2": [float(target[1])] * len(responses),
    }
    for i, name in enumerate(reward_names):
        result[f"obtained_{name}_raw"] = raw_scores[i]
        if norm_scores is not None:
            result[f"obtained_{name}_norm"] = norm_scores[i]

    df = pd.DataFrame(result)
    out_csv = os.path.join(save_root, f"eval_pref_h{w_helpful:.1f}_s{1.0-w_helpful:.1f}.csv")
    df.to_csv(out_csv, index=False)

    summary = {
        "preference_helpful": float(w_helpful),
        "preference_harmless": float(1.0 - w_helpful),
        "target_score1": float(target[0]),
        "target_score2": float(target[1]),
        "n": int(len(df)),
    }
    for i, name in enumerate(reward_names):
        summary[f"{name}_raw_mean"] = float(np.mean(raw_scores[i]))
        summary[f"{name}_raw_std"] = float(np.std(raw_scores[i]))
        if norm_scores is not None:
            summary[f"{name}_norm_mean"] = float(np.mean(norm_scores[i]))
            summary[f"{name}_norm_std"] = float(np.std(norm_scores[i]))
    summary["mip_raw"] = float(w_helpful * summary[f"{reward_names[0]}_raw_mean"] + (1.0 - w_helpful) * summary[f"{reward_names[1]}_raw_mean"])
    if norm_scores is not None:
        summary["mip_norm"] = float(w_helpful * summary[f"{reward_names[0]}_norm_mean"] + (1.0 - w_helpful) * summary[f"{reward_names[1]}_norm_mean"])

    summary_rows.append(summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_csv}")

summary_path = os.path.join(save_root, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary_rows, f, indent=2)
print(f"Saved summary: {summary_path}")
