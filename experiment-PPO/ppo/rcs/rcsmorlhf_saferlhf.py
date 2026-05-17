#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RCS-MORLHF training on PKU-SafeRLHF prompts with Beaver reward/cost models.

This is a minimal, drop-in variant of ``morlhf_saferlhf.py``.  The only
algorithmic change is the scalar reward used by PPO:

    MORLHF:      R(x,y;w)     = sum_i w_i r_i(x,y)
    RCS-MORLHF:  R_RCS(x,y;w) = sum_i w_i lambda_i r_i(x,y),
                 lambda_i = exp(gamma * a_i)

where a_i is the objective-level log precision fitted by RCS/CPC calibration.
The sign conversion for a cost model should be included in ``reward_signs``;
for SafeRLHF Beaver models, use ``1,-1`` so that both signed scores are
"larger is better" before applying RCS coefficients.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed

from multi_reward_models import RewardModels
from safe_rlhf.models import AutoModelForScore
from src.data.configs import DATASET_CONFIGS
from utils import print_trainable_parameters


tqdm.pandas()


@dataclass
class ScriptArguments:
    # Logging / output
    log_with: Optional[str] = field(default="wandb", metadata={"help": "Use 'wandb' to log with wandb. Set to None if disabled."})
    disable_wandb: bool = field(default=False, metadata={"help": "Whether to disable wandb."})
    save_directory: str = field(default="./logs_rcsmorlhf/", metadata={"help": "Root directory for checkpoints/logs."})
    wandb_name: str = field(default="rcsmorlhf_saferlhf", metadata={"help": "Name for this experiment."})

    # PPO hyperparameters
    epochs: int = field(default=1, metadata={"help": "Number of training epochs."})
    learning_rate: float = field(default=1e-5, metadata={"help": "Learning rate."})
    mini_batch_size: int = field(default=1, metadata={"help": "PPO minibatch size."})
    batch_size: int = field(default=4, metadata={"help": "PPO batch size."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Gradient accumulation steps."})
    early_stopping: bool = field(default=True, metadata={"help": "Whether to early stop based on KL target."})
    target: float = field(default=3.0, metadata={"help": "Target KL divergence for adaptive KL control."})
    init_kl_coef: float = field(default=0.2, metadata={"help": "Initial KL penalty coefficient."})
    max_grad_norm: float = field(default=0.5, metadata={"help": "Maximum gradient norm for clipping."})

    # Models / data
    base_model_name: str = field(
        default="PKU-Alignment/alpaca-7b-reproduced",
        metadata={"help": "Base/SFT model path or HF model name."},
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Load policy model in 8-bit."})
    dataset_name: str = field(
        default="PKU-Alignment/PKU-SafeRLHF-10K-better",
        metadata={"help": "Dataset name used by src.data.configs.DATASET_CONFIGS."},
    )
    train_split: str = field(default="train", metadata={"help": "Split used for PPO training prompts."})
    eval_split: str = field(default="validation", metadata={"help": "Split used for periodic validation generation."})
    prompt_template: str = field(
        default="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:",
        metadata={"help": "Prompt template. Must contain {raw_prompt} for raw prompts."},
    )
    max_prompt_length: int = field(default=384, metadata={"help": "Maximum tokenized prompt length."})
    max_new_tokens: int = field(default=512, metadata={"help": "Maximum generated response length."})
    max_train_samples: int = field(default=-1, metadata={"help": "Maximum number of training prompts. -1 means all."})

    # Multi-objective reward/cost setup
    reward_names: str = field(default="helpful,harmless", metadata={"help": "Comma-separated reward names."})
    reward_model_names: str = field(
        default="PKU-Alignment/beaver-7b-v1.0-reward,PKU-Alignment/beaver-7b-v1.0-cost",
        metadata={"help": "Comma-separated reward/cost model names, aligned with reward_names."},
    )
    reward_signs: str = field(
        default="1,-1",
        metadata={"help": "Comma-separated signs. Use -1 for cost model so larger harmlessness is better."},
    )
    reward_model_max_length: int = field(default=512, metadata={"help": "Reward model max sequence length."})
    reward_batch_size: int = field(default=4, metadata={"help": "Reward model scoring batch size."})
    preference: float = field(default=0.5, metadata={"help": "Weight for the first objective. For helpful,harmless this is w_helpful."})

    # RCS setup
    rcs_calibrator_path: str = field(
        default="",
        metadata={"help": "Path to rcs_calibrator.json/cpc_calibrator.json or a directory containing one. Empty means lambda_i=1."},
    )
    rcs_gamma: float = field(
        default=1.0,
        metadata={"help": "Damping exponent for lambda_i = exp(gamma * a_i). Core RCS uses gamma=1."},
    )
    rcs_normalize_coefficients: bool = field(
        default=False,
        metadata={"help": "If true, normalize w_i*lambda_i to sum to one. This is NOT core RCS; use only as ablation."},
    )
    rcs_reward_scale: float = field(
        default=1.0,
        metadata={"help": "Optional global multiplier applied after RCS scalarization."},
    )
    rcs_clip_reward: float = field(
        default=0.0,
        metadata={"help": "If >0, clip scalar PPO reward to [-value, value] as an engineering safeguard."},
    )

    # Validation / logging cadence
    eval_steps: int = field(default=500, metadata={"help": "Run validation every N PPO steps. <=0 disables validation."})
    eval_num_prompts: int = field(default=5, metadata={"help": "Number of validation prompts to generate."})
    eval_max_new_tokens: int = field(default=128, metadata={"help": "Max new tokens for validation generation."})
    save_every: int = field(default=100, metadata={"help": "Save checkpoint every N PPO steps. <=0 disables periodic saving."})
    log_every: int = field(default=1, metadata={"help": "Write local CSV/plot every N PPO steps."})

    # Misc
    seed: int = field(default=42, metadata={"help": "Random seed."})
    exp_type: str = field(default="assistant", metadata={"help": "Kept for compatibility with the original script."})
    sanity_check: bool = field(default=False, metadata={"help": "Pass sanity_check to DATASET_CONFIGS when supported."})


def split_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def build_preference_vector(first_weight: float, num_rewards: int) -> List[float]:
    if num_rewards == 1:
        return [1.0]
    if num_rewards == 2:
        return [round(float(first_weight), 6), round(1.0 - float(first_weight), 6)]
    return [round(1.0 / num_rewards, 6) for _ in range(num_rewards)]


def _candidate_calibrator_files(path: str) -> List[Path]:
    p = Path(path)
    if p.is_dir():
        return [p / "rcs_calibrator.json", p / "cpc_calibrator.json", p / "calibrator.json"]
    return [p]


def load_objective_log_precisions(path: str, num_objectives: int) -> Tuple[List[float], Dict]:
    """Load objective-level log precisions a_i.

    The loader accepts the new RCS filename and the existing CPC filename so that
    previously fitted ``cpc_calibrator.json`` files can be reused.
    """
    if path is None or str(path).strip() == "":
        return [0.0 for _ in range(num_objectives)], {"source": "identity", "note": "lambda_i=1; identical to MORLHF scalarization"}

    chosen_path = None
    for cand in _candidate_calibrator_files(path):
        if cand.exists():
            chosen_path = cand
            break
    if chosen_path is None:
        raise FileNotFoundError(
            f"Could not find calibrator under {path}. Tried: "
            + ", ".join(str(x) for x in _candidate_calibrator_files(path))
        )

    with open(chosen_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    vals = payload.get("objective_log_precisions")
    if vals is None:
        vals = payload.get("rcs_log_precisions")
    if vals is None:
        raise KeyError(f"No objective_log_precisions/rcs_log_precisions in {chosen_path}")

    vals = [float(x) for x in vals]
    if len(vals) != num_objectives:
        raise ValueError(f"Calibrator has {len(vals)} objectives, but reward_names has {num_objectives}: {chosen_path}")

    payload["_loaded_from"] = str(chosen_path)
    return vals, payload


def build_rcs_coefficients(preference: List[float], log_precisions: List[float], gamma: float, normalize: bool) -> Tuple[List[float], List[float], float]:
    lambdas = [float(math.exp(float(gamma) * a)) for a in log_precisions]
    coeffs = [float(w) * lam for w, lam in zip(preference, lambdas)]
    rho = float(sum(coeffs))
    if normalize:
        if rho <= 0.0:
            raise ValueError(f"Cannot normalize non-positive RCS coefficient sum rho={rho}")
        coeffs = [c / rho for c in coeffs]
    return lambdas, coeffs, rho


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

set_seed(script_args.seed)

if script_args.disable_wandb:
    os.environ["WANDB_DISABLED"] = "true"
    script_args.log_with = None

base_model_name = script_args.base_model_name
tokenizer_name = script_args.base_model_name
print("base model:", base_model_name, flush=True)

reward_names = split_csv(script_args.reward_names)
reward_model_path_list = split_csv(script_args.reward_model_names)
rm_tokenizer_path_list = reward_model_path_list
reward_sign_list = [float(x) for x in split_csv(script_args.reward_signs)]
num_rewards = len(reward_names)
preference = build_preference_vector(script_args.preference, num_rewards)

if len(reward_model_path_list) != num_rewards:
    raise ValueError(
        f"reward_names has {num_rewards} items ({reward_names}), but reward_model_names has "
        f"{len(reward_model_path_list)} items ({reward_model_path_list})."
    )
if len(reward_sign_list) != num_rewards:
    raise ValueError(
        f"reward_names has {num_rewards} items ({reward_names}), but reward_signs has "
        f"{len(reward_sign_list)} items ({reward_sign_list})."
    )

rcs_log_precisions, rcs_payload = load_objective_log_precisions(script_args.rcs_calibrator_path, num_rewards)
rcs_lambdas, rcs_coefficients, rcs_rho = build_rcs_coefficients(
    preference=preference,
    log_precisions=rcs_log_precisions,
    gamma=script_args.rcs_gamma,
    normalize=script_args.rcs_normalize_coefficients,
)

script_args.wandb_name = (
    f"{script_args.wandb_name}"
)
if script_args.rcs_normalize_coefficients:
    script_args.wandb_name += "_normcoef"
run_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(run_dir, exist_ok=True)

if os.environ.get("LOCAL_RANK", "0") in {"0", "-1"}:
    with open(os.path.join(run_dir, "rcs_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "reward_names": reward_names,
                "reward_model_names": reward_model_path_list,
                "reward_signs": reward_sign_list,
                "preference": preference,
                "rcs_log_precisions": rcs_log_precisions,
                "rcs_lambdas": rcs_lambdas,
                "rcs_coefficients": rcs_coefficients,
                "rcs_rho_before_normalization": rcs_rho,
                "rcs_gamma": script_args.rcs_gamma,
                "rcs_normalize_coefficients": script_args.rcs_normalize_coefficients,
                "rcs_reward_scale": script_args.rcs_reward_scale,
                "rcs_clip_reward": script_args.rcs_clip_reward,
                "calibrator_payload": rcs_payload,
            },
            f,
            indent=2,
        )

print("reward names:", reward_names, flush=True)
print("reward model paths:", reward_model_path_list, flush=True)
print("reward signs:", reward_sign_list, flush=True)
print("preference w:", preference, flush=True)
print("RCS log precisions a:", rcs_log_precisions, flush=True)
print("RCS lambdas exp(gamma*a):", rcs_lambdas, flush=True)
print("RCS coefficients c=w*lambda:", rcs_coefficients, flush=True)
print("RCS rho before normalization:", rcs_rho, flush=True)
print("save dir:", run_dir, flush=True)

accelerator = Accelerator()
process_id = accelerator.local_process_index
gpu_id = process_id
device = accelerator.device
print(f"process: {process_id}, accelerator device: {device}, model gpu id: {gpu_id}", flush=True)


def local_device_map():
    if torch.cuda.is_available():
        return {"": gpu_id}
    return None


class BeaverScoreModels:
    """Wrapper for PKU Beaver reward/cost models loaded through AutoModelForScore."""

    def __init__(self, model_paths, tokenizer_paths, gpu_id, max_length=1024, batch_size=4):
        self.model_paths = model_paths
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.tokenizers = []

        print("Loading Beaver score models with AutoModelForScore...", flush=True)
        for model_path, tokenizer_path in zip(model_paths, tokenizer_paths):
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                use_fast=True,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            kwargs = dict(
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            dev_map = local_device_map()
            if dev_map is not None:
                kwargs["device_map"] = dev_map

            model = AutoModelForScore.from_pretrained(model_path, **kwargs)
            if dev_map is None:
                model.to(self.device)
            model.eval()

            self.tokenizers.append(tokenizer)
            self.models.append(model)

        print("Loaded Beaver score models:", model_paths, flush=True)

    @torch.no_grad()
    def get_reward_model_scores(self, queries_responses, *args, **kwargs):
        texts = [str(query) + str(response) for query, response in queries_responses]
        all_scores = []

        for model, tokenizer in zip(self.models, self.tokenizers):
            model_scores = []
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = model(**inputs)

                scores = getattr(outputs, "end_scores", None)
                if scores is None:
                    scores = getattr(outputs, "scores", None)
                if scores is None:
                    raise RuntimeError(
                        "AutoModelForScore output has no end_scores/scores. "
                        f"Available fields: {outputs}"
                    )

                if scores.ndim == 2 and scores.shape[-1] == 1:
                    scores = scores.squeeze(-1)
                elif scores.ndim >= 2:
                    scores = scores[:, -1]

                model_scores.extend(scores.float().detach().cpu().tolist())
            all_scores.append(model_scores)

        return all_scores


# Load reward/cost models.
if any("beaver-7b" in path for path in reward_model_path_list):
    reward_model = BeaverScoreModels(
        reward_model_path_list,
        rm_tokenizer_path_list,
        gpu_id,
        max_length=script_args.reward_model_max_length,
        batch_size=script_args.reward_batch_size,
    )
else:
    reward_model = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    use_fast=True,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


def build_pku_prompt_dataset(args, tokenizer, split="train"):
    if args.dataset_name not in DATASET_CONFIGS:
        raise KeyError(f"Unknown dataset_name={args.dataset_name}. Available keys: {list(DATASET_CONFIGS.keys())}")

    rdp = DATASET_CONFIGS[args.dataset_name](
        prompt_template=args.prompt_template,
        sanity_check=args.sanity_check,
    )
    pref_dataset = rdp.get_preference_dataset(split=split)

    rows = []
    seen = set()
    for ex in pref_dataset:
        raw_prompt = ex.get("raw_prompt") or ex.get("prompt") or ex.get("query") or ex.get("input")
        if raw_prompt is None:
            raw_prompt = ex.get("text")
        if raw_prompt is None:
            raise KeyError(f"Cannot find prompt key. Available keys: {list(ex.keys())}")

        raw_prompt = str(raw_prompt)
        query = args.prompt_template.format(raw_prompt=raw_prompt) if "{raw_prompt}" in args.prompt_template else raw_prompt

        if query in seen:
            continue
        seen.add(query)

        toks = tokenizer(
            query,
            truncation=True,
            max_length=args.max_prompt_length,
            padding=False,
        )
        rows.append({"query": query, "input_ids": toks["input_ids"]})

    return Dataset.from_list(rows)


train_dataset = build_pku_prompt_dataset(script_args, tokenizer, split=script_args.train_split)
train_dataset = train_dataset.shuffle(seed=script_args.seed)
if script_args.max_train_samples is not None and script_args.max_train_samples > 0:
    n_train = min(script_args.max_train_samples, len(train_dataset))
    train_dataset = train_dataset.select(range(n_train))

# Build eval dataset only when periodic validation is enabled.
eval_dataset = Dataset.from_list([])
if script_args.eval_steps is not None and script_args.eval_steps > 0:
    try:
        eval_dataset = build_pku_prompt_dataset(script_args, tokenizer, split=script_args.eval_split)
        if len(eval_dataset) > 0:
            eval_dataset = eval_dataset.shuffle(seed=script_args.seed)
    except Exception as exc:
        print(f"[warn] Could not build eval split '{script_args.eval_split}': {exc}", flush=True)
        print("[warn] Disabling periodic validation.", flush=True)
        script_args.eval_steps = 0

print(f"Size of the train set ({script_args.train_split}): {len(train_dataset)}", flush=True)
print(f"Size of the eval set ({script_args.eval_split}): {len(eval_dataset)}", flush=True)


class PKUInstructions:
    def get_input(self, text):
        if "ASSISTANT:" in text:
            return text.split("ASSISTANT:")[0].strip() + "ASSISTANT:"
        return text

    def get_response(self, text):
        if "ASSISTANT:" in text:
            return text.split("ASSISTANT:", 1)[1].strip()
        return text


instructions = PKUInstructions()

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

config = PPOConfig(
    model_name=base_model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    early_stopping=script_args.early_stopping,
    target=script_args.target,
    max_grad_norm=script_args.max_grad_norm,
    optimize_cuda_cache=True,
    init_kl_coef=script_args.init_kl_coef,
    tracker_project_name=os.environ.get("WANDB_PROJECT", "pcma"),
    tracker_kwargs={"wandb": {"name": script_args.wandb_name}},
)

model_kwargs = dict(
    peft_config=lora_config,
    trust_remote_code=True,
)
if torch.cuda.is_available():
    model_kwargs["device_map"] = local_device_map()

if script_args.load_in_8bit:
    model_kwargs["load_in_8bit"] = True
else:
    model_kwargs["torch_dtype"] = torch.bfloat16

model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_name, **model_kwargs)

print_trainable_parameters(model)
model.pretrained_model.resize_token_embeddings(len(tokenizer))
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

ppo_trainer = PPOTrainer(
    config,
    model,
    tokenizer=tokenizer,
    dataset=train_dataset,
    data_collator=collator,
    optimizer=optimizer,
)

generation_kwargs = {
    "max_new_tokens": script_args.max_new_tokens,
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "temperature": 0.7,
    "pad_token_id": tokenizer.eos_token_id,
    "begin_suppress_tokens": [tokenizer.eos_token_id],
}


def _clean_responses(decoded_responses):
    cleaned = []
    for response in decoded_responses:
        response = response.strip("[PAD] ")
        response = response.strip("<unk>")
        temp_resp = response.strip("<s>").strip("</s>")
        temp_resp = temp_resp.split("\n\nHuman:")[0].strip()
        temp_resp = temp_resp.split("\nHuman:")[0].strip()
        temp_resp = temp_resp.split("\n\nAssistant:")[0].strip()
        temp_resp = temp_resp.split("\nAssistant:")[0].strip()
        temp_resp = temp_resp.split("\n\n\n")[0].strip()
        temp_resp = temp_resp.split("###")[0].strip()
        cleaned.append(temp_resp)
    return cleaned


def _compute_scalar_rewards(queries, responses):
    texts_merge = [q + r for q, r in zip(queries, responses)]
    queries_responses = [
        (instructions.get_input(text), instructions.get_response(text))
        for text in texts_merge
    ]

    if hasattr(instructions, "get_post"):
        rewards_list = reward_model.get_reward_model_scores(queries_responses, instructions.get_post)
    else:
        rewards_list = reward_model.get_reward_model_scores(queries_responses)

    scalar_rewards = []
    signed_scores_by_objective = []
    contribution_by_objective = []

    for j in range(len(queries_responses)):
        scalar_reward = 0.0
        signed_row = []
        contrib_row = []
        for k in range(num_rewards):
            signed_score = reward_sign_list[k] * float(rewards_list[k][j])
            contribution = rcs_coefficients[k] * signed_score
            scalar_reward += contribution
            signed_row.append(float(signed_score))
            contrib_row.append(float(contribution))

        scalar_reward *= float(script_args.rcs_reward_scale)
        if script_args.rcs_clip_reward is not None and script_args.rcs_clip_reward > 0:
            clip = float(script_args.rcs_clip_reward)
            scalar_reward = max(-clip, min(clip, scalar_reward))
        scalar_rewards.append(float(round(scalar_reward, 4)))
        signed_scores_by_objective.append(signed_row)
        contribution_by_objective.append(contrib_row)

    diagnostics = {
        "signed_scores": signed_scores_by_objective,
        "rcs_contributions": contribution_by_objective,
    }
    return rewards_list, scalar_rewards, diagnostics


@torch.no_grad()
def run_periodic_validation(global_step):
    if script_args.eval_steps is None or script_args.eval_steps <= 0:
        return
    if len(eval_dataset) == 0:
        return
    if not accelerator.is_local_main_process:
        return

    n_eval = min(script_args.eval_num_prompts, len(eval_dataset))
    eval_batch = [eval_dataset[idx] for idx in range(n_eval)]
    eval_queries = [ex["query"] for ex in eval_batch]
    eval_query_tensors = [
        torch.as_tensor(ex["input_ids"], dtype=torch.long, device=device)
        for ex in eval_batch
    ]

    was_training = model.training
    model.eval()
    model.gradient_checkpointing_disable()
    model.pretrained_model.config.use_cache = True

    eval_generation_kwargs = dict(generation_kwargs)
    eval_generation_kwargs["max_new_tokens"] = script_args.eval_max_new_tokens

    response_tensors = ppo_trainer.generate(
        eval_query_tensors,
        return_prompt=False,
        **eval_generation_kwargs,
    )
    decoded = tokenizer.batch_decode(response_tensors)
    eval_responses = _clean_responses(decoded)

    rewards_list, scalar_rewards, diagnostics = _compute_scalar_rewards(eval_queries, eval_responses)

    if not script_args.disable_wandb:
        columns = ["step", "idx", "prompt", "response"]
        for name in reward_names:
            columns.append(f"{name}_raw")
            columns.append(f"{name}_signed")
            columns.append(f"{name}_lambda")
            columns.append(f"{name}_coefficient")
            columns.append(f"{name}_rcs_contribution")
        columns.append("scalar_reward")

        table = wandb.Table(columns=columns)
        for row_idx, (query, response, scalar_reward) in enumerate(zip(eval_queries, eval_responses, scalar_rewards)):
            row = [global_step, row_idx, query, response]
            for k, _name in enumerate(reward_names):
                raw_score = float(rewards_list[k][row_idx])
                row.append(raw_score)
                row.append(float(diagnostics["signed_scores"][row_idx][k]))
                row.append(float(rcs_lambdas[k]))
                row.append(float(rcs_coefficients[k]))
                row.append(float(diagnostics["rcs_contributions"][row_idx][k]))
            row.append(float(scalar_reward))
            table.add_data(*row)

        if wandb.run is None:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "pcma"),
                entity=os.environ.get("WANDB_ENTITY", None),
                name=script_args.wandb_name,
            )
        log_dict = {
            f"validation_samples/step_{global_step}": table,
            "validation/rcs_scalar_reward_mean": float(np.mean(scalar_rewards)),
            "validation/rcs_scalar_reward_std": float(np.std(scalar_rewards)),
            "rcs/rho_before_normalization": float(rcs_rho),
        }
        for k, name in enumerate(reward_names):
            log_dict[f"rcs/lambda_{name}"] = float(rcs_lambdas[k])
            log_dict[f"rcs/coefficient_{name}"] = float(rcs_coefficients[k])
        wandb.log(log_dict, step=global_step)

    print(f"[eval] step={global_step}, generated {n_eval} validation samples", flush=True)

    if was_training:
        model.train()
    model.pretrained_model.config.use_cache = False


def save_local_training_curves(mean_scores, std_scores, save_data):
    if len(mean_scores) == 0:
        return
    save_path = os.path.join(run_dir, "scores.png")
    x = np.arange(len(mean_scores))
    mean_arr = np.array(mean_scores)
    std_arr = np.array(std_scores)

    plt.figure()
    plt.plot(mean_scores)
    plt.fill_between(x, mean_arr - std_arr, mean_arr + std_arr, alpha=0.5)
    plt.xlabel("logged PPO step")
    plt.ylabel("RCS scalar reward")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    pd.DataFrame(save_data).to_csv(os.path.join(run_dir, "data.csv"), index=False)


print("Training........", flush=True)
model.gradient_checkpointing_disable()
model.pretrained_model.config.use_cache = True

mean_scores = []
std_scores = []
save_data = {
    "kl_mean": [],
    "reward_mean": [],
    "reward_std": [],
    "text_sample": [],
    "batch_time": [],
    "total_time": [],
}
for name in reward_names:
    save_data[f"{name}_signed_mean"] = []
    save_data[f"{name}_rcs_contribution_mean"] = []


t_start = time.time()
global_step = 0
last_batch_idx = -1

for epoch in range(script_args.epochs):
    try:
        pbar_total = len(ppo_trainer.dataloader)
    except TypeError:
        pbar_total = max(1, len(train_dataset) // max(1, script_args.batch_size * accelerator.num_processes))

    pbar = tqdm(total=pbar_total, disable=not accelerator.is_local_main_process)

    for i, batch in enumerate(ppo_trainer.dataloader):
        last_batch_idx = i
        t_epoch_start = time.time()
        print(f"epoch {epoch}, batch {i}", flush=True)

        query_tensors = [
            torch.as_tensor(q, dtype=torch.long, device=device)
            for q in batch["input_ids"]
        ]

        model.gradient_checkpointing_disable()
        model.pretrained_model.config.use_cache = True

        with torch.no_grad():
            response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)

        full_responses = tokenizer.batch_decode(response_tensors)
        clean_texts = _clean_responses(full_responses)
        clean_response_tensors = [tokenizer.encode(text, add_special_tokens=False) for text in clean_texts]
        lengths = [len(x) for x in clean_response_tensors]
        response_tensors = [
            response_tensors[j][: max(lengths[j], 2)]
            for j in range(len(response_tensors))
        ]
        batch["response"] = clean_texts

        texts_merge = [q + r for q, r in zip(batch["query"], batch["response"])]
        _rewards_list, rewards, diagnostics = _compute_scalar_rewards(batch["query"], batch["response"])
        rewards_tensor = [torch.tensor(float(r), dtype=torch.float32, device=device) for r in rewards]

        print(
            f"iter {epoch}, batch {i}, mean RCS score: {float(torch.tensor(rewards).mean()):.4f}",
            flush=True,
        )

        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)
        ppo_trainer.log_stats(stats, batch, rewards_tensor)

        global_step += 1
        policy_kl_value = float(stats.get("objective/kl", 0.0))

        if script_args.eval_steps > 0 and global_step % script_args.eval_steps == 0:
            run_periodic_validation(global_step)
        accelerator.wait_for_everyone()

        rewards_for_gather = torch.tensor(rewards, dtype=torch.float32, device=device)
        kl_for_gather = torch.tensor([policy_kl_value], dtype=torch.float32, device=device)
        all_rewards = accelerator.gather_for_metrics(rewards_for_gather).detach().cpu().numpy()
        all_policy_kl = accelerator.gather_for_metrics(kl_for_gather).detach().cpu().numpy()

        signed_np = np.asarray(diagnostics["signed_scores"], dtype=np.float32)
        contrib_np = np.asarray(diagnostics["rcs_contributions"], dtype=np.float32)
        signed_tensor = torch.tensor(signed_np, dtype=torch.float32, device=device)
        contrib_tensor = torch.tensor(contrib_np, dtype=torch.float32, device=device)
        all_signed = accelerator.gather_for_metrics(signed_tensor).detach().cpu().numpy()
        all_contrib = accelerator.gather_for_metrics(contrib_tensor).detach().cpu().numpy()

        if process_id == 0 and script_args.log_every > 0 and global_step % script_args.log_every == 0:
            mean_scores.append(float(np.mean(all_rewards)))
            std_scores.append(float(np.std(all_rewards)))
            t_epoch_end = time.time()
            save_data["batch_time"].append(float(t_epoch_end - t_epoch_start))
            save_data["total_time"].append(float(t_epoch_end - t_start))
            save_data["kl_mean"].append(float(np.mean(all_policy_kl)))
            save_data["reward_mean"].append(mean_scores[-1])
            save_data["reward_std"].append(std_scores[-1])
            save_data["text_sample"].append(texts_merge[0] if len(texts_merge) > 0 else "")
            for k, name in enumerate(reward_names):
                save_data[f"{name}_signed_mean"].append(float(np.mean(all_signed[:, k])))
                save_data[f"{name}_rcs_contribution_mean"].append(float(np.mean(all_contrib[:, k])))
            save_local_training_curves(mean_scores, std_scores, save_data)
            print(f"iter {epoch}, batch {i}: local log finish", flush=True)

        accelerator.wait_for_everyone()
        pbar.update(1)

        if (
            ppo_trainer.accelerator.is_main_process
            and script_args.save_every > 0
            and global_step % script_args.save_every == 0
        ):
            save_path = os.path.join(run_dir, f"step_{global_step}")
            ppo_trainer.save_pretrained(save_path)
            print(f"iter {epoch}, batch {i}: model saved to {save_path}", flush=True)

    pbar.close()

# Final save.
if ppo_trainer.accelerator.is_main_process:
    save_path = os.path.join(run_dir, f"final_step_{global_step}_batch_{last_batch_idx}")
    ppo_trainer.save_pretrained(save_path)
    print(f"final model saved to {save_path}", flush=True)
