"""Prepare a SafeRLHF reward-conditioned dataset for RiC.

This is the SafeRLHF counterpart of the original HH-RLHF
``prepare_dataset_with_rewards.py``. It keeps the RiC idea unchanged:

  1. collect prompt-response examples,
  2. score each response with objective reward models,
  3. normalize objective scores on the training set,
  4. insert the normalized scores into the prompt as condition tokens,
  5. save a dataset whose ``query`` field is used for SFT.

No preference weight is used during training-data construction. Preference
weights are used only at evaluation/inference time to choose target reward
conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from transformers import HfArgumentParser

from multi_reward_models_saferlhf_ric import SafeRLHFRewardModels
from utils import Instructions_n, load_main_tokenizer


DEFAULT_PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"


def _split_csv(x: str) -> List[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


def _get_first(example: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    return None


def _strip_prompt_from_full_text(full_text: str) -> tuple[str, str]:
    """Fallback for HH-style chosen/rejected strings."""
    if "\n\nAssistant:" in full_text:
        parts = full_text.split("\n\nAssistant:")
        prompt = "\n\nAssistant:".join(parts[:-1]).strip()
        response = parts[-1].strip()
        return prompt, response
    return "", full_text.strip()


def _format_prompt(raw_prompt: str, prompt_template: str) -> str:
    if "{raw_prompt}" in prompt_template:
        return prompt_template.format(raw_prompt=raw_prompt)
    return prompt_template + str(raw_prompt)


def _insert_score_tokens(prompt: str, scores: List[float], instructions: Instructions_n) -> str:
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


@dataclass
class ScriptArguments:
    dataset_name: str = field(default="PKU-Alignment/PKU-SafeRLHF-10K")
    split: str = field(default="train")
    base_model_name: str = field(default="PKU-Alignment/alpaca-7b-reproduced")
    prompt_template: str = field(default=DEFAULT_PROMPT_TEMPLATE)
    save_directory: str = field(default="./datasets/ric_saferlhf_helpful_harmless.hf")

    reward_names: str = field(default="helpful,harmless")
    reward_model_names: str = field(
        default="PKU-Alignment/beaver-7b-v1.0-reward,PKU-Alignment/beaver-7b-v1.0-cost"
    )
    reward_signs: str = field(default="1,-1")
    reward_model_max_length: int = field(default=512)
    reward_batch_size: int = field(default=4)

    max_length: int = field(default=512)
    max_examples: Optional[int] = field(default=None, metadata={"help": "number of prompts before doubling responses"})
    sanity_check: bool = field(default=False)
    num_proc: int = field(default=8)


args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]

reward_names = _split_csv(args.reward_names)
reward_model_names = _split_csv(args.reward_model_names)
reward_signs = [float(x) for x in _split_csv(args.reward_signs)]
if not (len(reward_names) == len(reward_model_names) == len(reward_signs)):
    raise ValueError("reward_names, reward_model_names, and reward_signs must have the same length.")

accelerator = Accelerator()
gpu_id = accelerator.local_process_index

tokenizer = load_main_tokenizer(args.base_model_name)
instructions = Instructions_n(len(reward_names))

reward_models = SafeRLHFRewardModels(
    reward_model_names=reward_model_names,
    reward_tokenizer_names=reward_model_names,
    reward_signs=reward_signs,
    gpu_id=gpu_id,
    reward_stats_path=None,
    max_length=args.reward_model_max_length,
    batch_size=args.reward_batch_size,
)

print(f"Loading dataset: {args.dataset_name} split={args.split}")
ds = load_dataset(args.dataset_name, split=args.split)
if args.sanity_check:
    ds = ds.select(range(min(100, len(ds))))
elif args.max_examples is not None:
    ds = ds.select(range(min(int(args.max_examples), len(ds))))

rows: List[Dict[str, str]] = []
for ex in ds:
    raw_prompt = _get_first(ex, ["raw_prompt", "prompt", "input", "query"])

    response_0 = _get_first(ex, ["response_0", "answer_0", "output_0"])
    response_1 = _get_first(ex, ["response_1", "answer_1", "output_1"])

    if raw_prompt is not None and response_0 is not None and response_1 is not None:
        raw_prompt = str(raw_prompt)
        prompt = _format_prompt(raw_prompt, args.prompt_template)
        rows.append({"raw_prompt": raw_prompt, "prompt": prompt, "response": str(response_0)})
        rows.append({"raw_prompt": raw_prompt, "prompt": prompt, "response": str(response_1)})
        continue

    # Robust fallback: if a wrapper exposes chosen/rejected full conversations.
    chosen = _get_first(ex, ["chosen"])
    rejected = _get_first(ex, ["rejected"])
    if chosen is not None and rejected is not None:
        for full_text in [str(chosen), str(rejected)]:
            fallback_prompt, response = _strip_prompt_from_full_text(full_text)
            if raw_prompt is not None:
                raw = str(raw_prompt)
                prompt = _format_prompt(raw, args.prompt_template)
            else:
                raw = fallback_prompt
                prompt = fallback_prompt if fallback_prompt else _format_prompt("", args.prompt_template)
            rows.append({"raw_prompt": raw, "prompt": prompt, "response": response})

if not rows:
    raise RuntimeError(
        "No prompt-response rows were built. Expected response_0/response_1 or chosen/rejected fields."
    )

train_data = Dataset.from_list(rows)
print(f"Built {len(train_data)} prompt-response rows before length filtering.")


def tokenize_for_length(sample: Dict[str, Any]) -> Dict[str, Any]:
    text = str(sample["prompt"]) + str(sample["response"])
    sample["input_ids"] = tokenizer.encode(text)
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample


train_data = train_data.map(tokenize_for_length, batched=False, num_proc=args.num_proc)
train_data = train_data.filter(lambda x: args.max_length >= len(x["input_ids"]) >= 8)
print(f"After length filtering: {len(train_data)} examples")

queries_responses = [(q, r) for q, r in zip(train_data["prompt"], train_data["response"])]
raw_rewards = reward_models.get_reward_model_scores(queries_responses, normalize=False)

for i, scores in enumerate(raw_rewards):
    train_data = train_data.add_column(f"raw_score{i+1}", [float(s) for s in scores])

stats = []
for i in range(len(raw_rewards)):
    arr = np.asarray(train_data[f"raw_score{i+1}"], dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    if std == 0.0:
        raise ValueError(f"Reward std for objective {i+1} is zero; cannot normalize RiC score tokens.")
    stats.append([mean, std])
    norm = ((arr - mean) / std).astype(np.float32)
    train_data = train_data.add_column(f"score{i+1}", [float(x) for x in norm])

# Remove temporary tokenized fields before reconstructing the final RiC query.
for col in ["input_ids", "query"]:
    if col in train_data.column_names:
        train_data = train_data.remove_columns(col)


def add_score_prompt(sample: Dict[str, Any]) -> Dict[str, Any]:
    norm_scores = [float(sample[f"score{i+1}"]) for i in range(len(reward_names))]
    prompt_with_score = _insert_score_tokens(str(sample["prompt"]), norm_scores, instructions)
    sample["prompt_with_score"] = prompt_with_score
    sample["prompt_with_score_ids"] = tokenizer.encode(prompt_with_score)
    sample["input_ids"] = tokenizer.encode(prompt_with_score + " " + str(sample["response"]))
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample


train_data = train_data.map(add_score_prompt, batched=False, num_proc=args.num_proc)
train_data = train_data.filter(lambda x: args.max_length >= len(x["input_ids"]) >= 8)
train_data.set_format(type="torch")
train_data.save_to_disk(args.save_directory)

stats_array = np.asarray(stats, dtype=np.float32)
np.save(args.save_directory + "/all_reward_stat.npy", stats_array)

print(train_data)
print(f"Saved RiC SafeRLHF dataset to: {args.save_directory}")
print("Reward stats [mean, std] per objective:")
print(stats_array)
