"""Safety-Prior PC-MORLHF: PPO with risk-aware calibrated scalar rewards.

This script keeps the PPO loop from MORLHF and the posterior-calibrated
weights from PC-MORLHF, but adds a prompt-only safety prior rho(x).

First, PCMA gives reliability-calibrated weights

    alpha_pc(x, w).

Then Safety-Prior PC-MORLHF uses

    alpha_sp(x, w) = (1 - rho(x)) * alpha_pc(x, w) + rho(x) * q_safe.

Here q_safe is a fixed conservative objective distribution, e.g. (0, 1) for
(helpful, harmless).  The risk model observes only prompt x, not the generated
response, so it is a prompt-level safety prior rather than a reward-gap gate.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import tyro
import wandb
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed

from safe_rlhf.models import AutoModelForScore
from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils.posterior_calibration import (
    FrozenCausalLMPromptFeatureExtractor,
    HFPromptFeatureExtractor,
    LaplacePosteriorCalibrator,
)
from utils import print_trainable_parameters


@dataclass
class ScriptArguments:
    # Model / data
    base_model_name: str = field(default="PKU-Alignment/alpaca-7b-reproduced")
    posterior_calibrator_path: str = field(default="")
    safety_prior_path: str = field(default="", metadata={"help": "directory containing safety_prior.pt, or the .pt path itself"})
    dataset_name: str = field(default="PKU-Alignment/PKU-SafeRLHF-10K-better")
    prompt_template: str = field(default=DEFAULT_PROMPT_TEMPLATE)
    train_split: str = field(default="train")
    eval_split: str = field(default="validation")
    seed: int = field(default=8888)

    # PPO
    save_directory: str = field(default="./logs_sp_pcmorlhf")
    epochs: int = field(default=1)
    learning_rate: float = field(default=1e-5)
    batch_size: int = field(default=4)
    mini_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    early_stopping: bool = field(default=True)
    target: float = field(default=3.0)
    init_kl_coef: float = field(default=0.2)
    max_grad_norm: float = field(default=0.5)
    load_in_8bit: bool = field(default=False)

    # Generation / length
    max_prompt_length: int = field(default=384)
    max_new_tokens: int = field(default=128)
    reward_model_max_length: int = field(default=512)

    # Explicit objective scorers
    reward_names: str = field(default="helpful,harmless")
    reward_model_names: str = field(
        default="PKU-Alignment/beaver-7b-v1.0-reward,PKU-Alignment/beaver-7b-v1.0-cost"
    )
    reward_signs: str = field(default="1,-1")

    # Scalarization / calibration
    preference: float = field(default=0.5, metadata={"help": "base helpfulness weight w_H for 2 objectives"})
    preference_weights: str = field(default="", metadata={"help": "optional comma-separated base weights for m objectives"})
    posterior_num_samples: int = field(default=10, metadata={"help": "0=MAP plug-in; >0=Laplace MC average"})
    q_safe: str = field(default="", metadata={"help": "comma-separated conservative distribution; default is 0,1 for 2 objectives"})
    rho_fixed: Optional[float] = field(default=None, metadata={"help": "optional constant rho for ablation; if set, safety_prior_path is not required"})
    feature_source: str = field(default="sft_hidden", metadata={"help": "sft_hidden or hf_model"})
    feature_model_name: Optional[str] = field(default=None)
    feature_pooling: str = field(default="mean")
    feature_max_length: int = field(default=256)

    # W&B / monitoring
    log_with: str = field(default="wandb")
    disable_wandb: bool = field(default=False)
    wandb_name: str = field(default="sp_pcmorlhf")
    eval_steps: int = field(default=100)
    eval_num_prompts: int = field(default=5)
    eval_max_new_tokens: int = field(default=128)
    debug_print_alpha_every: int = field(default=50)
    debug_print_alpha_n: int = field(default=3)

    # LoRA
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)


def _split_csv(x: str) -> List[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


script_args = tyro.cli(ScriptArguments)
set_seed(script_args.seed)
accelerator = Accelerator()
process_id = accelerator.local_process_index
gpu_id = process_id

if script_args.disable_wandb:
    os.environ["WANDB_DISABLED"] = "true"

reward_names = _split_csv(script_args.reward_names)
reward_model_names = _split_csv(script_args.reward_model_names)
reward_sign_list = [float(x) for x in _split_csv(script_args.reward_signs)]
if not (len(reward_names) == len(reward_model_names) == len(reward_sign_list)):
    raise ValueError("reward_names, reward_model_names, and reward_signs must have the same length.")

def _parse_distribution(csv: str, *, name: str, expected_len: int) -> torch.Tensor:
    values = [float(x) for x in _split_csv(csv)]
    if len(values) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values, got {len(values)}: {csv}")
    t = torch.tensor(values, dtype=torch.float32)
    if torch.any(t < 0):
        raise ValueError(f"{name} must be non-negative: {values}")
    total = t.sum()
    if float(total) <= 0.0:
        raise ValueError(f"{name} must have positive mass: {values}")
    return t / total


num_objectives = len(reward_names)
if script_args.preference_weights.strip():
    base_w = _parse_distribution(script_args.preference_weights, name="preference_weights", expected_len=num_objectives)
else:
    if num_objectives != 2:
        raise ValueError("For more than 2 objectives, pass --preference_weights explicitly.")
    base_w = torch.tensor([script_args.preference, 1.0 - script_args.preference], dtype=torch.float32)
    if torch.any(base_w < 0) or float(base_w.sum()) <= 0.0:
        raise ValueError(f"Invalid two-objective preference: {base_w.tolist()}")
    base_w = base_w / base_w.sum()

if script_args.q_safe.strip():
    q_safe = _parse_distribution(script_args.q_safe, name="q_safe", expected_len=num_objectives)
else:
    if num_objectives != 2:
        raise ValueError("For more than 2 objectives, pass --q_safe explicitly.")
    q_safe = torch.tensor([0.0, 1.0], dtype=torch.float32)

pref_tag = "_".join([f"{reward_names[i]}{base_w[i].item():.2f}" for i in range(num_objectives)])
q_tag = "_".join([f"{reward_names[i]}{q_safe[i].item():.2f}" for i in range(num_objectives)])
script_args.wandb_name = f"{script_args.wandb_name}_pref_{pref_tag}_qsafe_{q_tag}"

os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)

print(f"base model: {script_args.base_model_name}")
print(f"reward_names: {reward_names}")
print(f"base_w: {base_w.tolist()}")
print(f"q_safe: {q_safe.tolist()}")
print(f"process: {process_id}, model gpu id: {gpu_id}")


class BeaverScoreModels:
    def __init__(self, model_paths: Sequence[str], signs: Sequence[float], gpu_id: int):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model_paths = list(model_paths)
        self.signs = [float(s) for s in signs]
        self.models = []
        self.tokenizers = []

        print("Loading explicit score models with AutoModelForScore...")
        for model_path in self.model_paths:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            model = AutoModelForScore.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map={"": gpu_id} if torch.cuda.is_available() else None,
            )
            model.eval()
            self.tokenizers.append(tokenizer)
            self.models.append(model)
        print("Loaded score models:", self.model_paths)

    @torch.no_grad()
    def get_scores(self, queries: Sequence[str], responses: Sequence[str]) -> torch.Tensor:
        texts = [str(q) + str(r) for q, r in zip(queries, responses)]
        cols = []
        for model, tokenizer, sign in zip(self.models, self.tokenizers, self.signs):
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=script_args.reward_model_max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            scores = getattr(outputs, "end_scores", None)
            if scores is None:
                scores = getattr(outputs, "scores", None)
            if scores is None:
                raise RuntimeError(f"AutoModelForScore output has no end_scores/scores: {outputs}")
            if scores.ndim == 2 and scores.shape[-1] == 1:
                scores = scores.squeeze(-1)
            elif scores.ndim >= 2:
                scores = scores[:, -1]
            cols.append(scores.float() * float(sign))
        return torch.stack(cols, dim=-1)  # (B, m), signed scores; higher is better.


reward_model = BeaverScoreModels(reward_model_names, reward_sign_list, gpu_id)

tokenizer = AutoTokenizer.from_pretrained(
    script_args.base_model_name,
    use_fast=True,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


def build_pku_prompt_dataset(split: str) -> Dataset:
    rdp = DATASET_CONFIGS[script_args.dataset_name](
        prompt_template=script_args.prompt_template,
        sanity_check=False,
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
        if "{raw_prompt}" in script_args.prompt_template and "BEGINNING OF CONVERSATION" not in raw_prompt:
            query = script_args.prompt_template.format(raw_prompt=raw_prompt)
        else:
            query = raw_prompt

        if query in seen:
            continue
        seen.add(query)
        toks = tokenizer(
            query,
            truncation=True,
            max_length=script_args.max_prompt_length,
            padding=False,
        )
        rows.append({"raw_prompt": raw_prompt, "query": query, "input_ids": toks["input_ids"]})
    return Dataset.from_list(rows)


train_dataset = build_pku_prompt_dataset(script_args.train_split).shuffle(seed=script_args.seed)
eval_dataset = build_pku_prompt_dataset(script_args.eval_split).shuffle(seed=script_args.seed)
print(f"Size of train prompt set ({script_args.train_split}): {len(train_dataset)}")
print(f"Size of eval prompt set ({script_args.eval_split}): {len(eval_dataset)}")


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


def _recover_raw_prompt_from_query(query: str) -> str:
    """Recover raw prompt if PPOTrainer drops the raw_prompt column."""
    q = str(query)

    # General case for templates containing {raw_prompt}.
    if "{raw_prompt}" in script_args.prompt_template:
        prefix, suffix = script_args.prompt_template.split("{raw_prompt}", 1)
        if q.startswith(prefix) and q.endswith(suffix):
            return q[len(prefix): len(q) - len(suffix)].strip()

    # Fallback for the PKU/SafeRLHF conversation template.
    if "USER:" in q and "ASSISTANT:" in q:
        return q.split("USER:", 1)[1].rsplit("ASSISTANT:", 1)[0].strip()

    return q.strip()


def _get_batch_raw_prompts(batch) -> list:
    if "raw_prompt" in batch and batch["raw_prompt"] is not None:
        return list(batch["raw_prompt"])
    if "query" not in batch:
        raise KeyError(f"Batch has neither raw_prompt nor query. Available keys: {list(batch.keys())}")
    return [_recover_raw_prompt_from_query(q) for q in batch["query"]]


lora_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

config = PPOConfig(
    model_name=script_args.base_model_name,
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
    tracker_kwargs={"wandb": {"name": script_args.wandb_name, "entity": os.environ.get("WANDB_ENTITY", None)}},
)

if script_args.load_in_8bit:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.base_model_name,
        load_in_8bit=True,
        peft_config=lora_config,
        device_map=gpu_id,
    )
else:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        peft_config=lora_config,
        device_map=gpu_id if torch.cuda.is_available() else None,
    )

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

posterior_calibrator = LaplacePosteriorCalibrator.from_pretrained(script_args.posterior_calibrator_path)


class LinearSafetyPrior:
    """Prompt-only rho(x) = sigmoid(w^T phi(x) + b)."""

    def __init__(self, path: str, device: torch.device):
        if os.path.isdir(path):
            path = os.path.join(path, "safety_prior.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find safety prior checkpoint: {path}")
        state = torch.load(path, map_location="cpu")
        self.weight = state["weight"].float().to(device)
        self.bias = state["bias"].float().to(device)
        self.feature_dim = int(state.get("feature_dim", self.weight.shape[-1]))
        self.path = path

    @torch.no_grad()
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Safety prior feature dim mismatch: got {features.shape[-1]}, "
                f"expected {self.feature_dim}. Use the same feature extractor as fitting."
            )
        logits = features.float().matmul(self.weight.t()).squeeze(-1) + self.bias.squeeze()
        return torch.sigmoid(logits)


safety_prior = None
if script_args.rho_fixed is None:
    if not script_args.safety_prior_path:
        raise ValueError("Pass --safety_prior_path or set --rho_fixed for ablation.")
    safety_prior = LinearSafetyPrior(script_args.safety_prior_path, accelerator.device)
    print(f"loaded safety prior: {safety_prior.path}")
else:
    if not (0.0 <= float(script_args.rho_fixed) <= 1.0):
        raise ValueError("rho_fixed must be in [0, 1].")
    print(f"using constant rho_fixed={script_args.rho_fixed}")


def _build_feature_extractor():
    if script_args.feature_source == "sft_hidden":
        # AutoModelForCausalLMWithValueHead wraps the PEFT CausalLM in `.pretrained_model`.
        feature_model = model.pretrained_model
        try:
            return FrozenCausalLMPromptFeatureExtractor(
                model=feature_model,
                tokenizer=tokenizer,
                max_length=script_args.feature_max_length,
                device=accelerator.device,
                pooling=script_args.feature_pooling,
                disable_adapter=True,
                prompt_template=script_args.prompt_template,
            )
        except TypeError:
            return FrozenCausalLMPromptFeatureExtractor(
                model=feature_model,
                tokenizer=tokenizer,
                max_length=script_args.feature_max_length,
                device=accelerator.device,
                pooling=script_args.feature_pooling,
                disable_adapter=True,
            )

    if script_args.feature_source == "hf_model":
        feature_model_name = script_args.feature_model_name or posterior_calibrator.feature_model_name
        if feature_model_name is None:
            raise ValueError("feature_model_name must be provided when feature_source='hf_model'.")
        return HFPromptFeatureExtractor(
            feature_model_name,
            max_length=script_args.feature_max_length,
            device=f"cuda:{accelerator.local_process_index}" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        )

    raise ValueError("feature_source must be either 'sft_hidden' or 'hf_model'.")


prompt_feature_extractor = _build_feature_extractor()

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


def _clean_responses(decoded_responses: Sequence[str]) -> List[str]:
    cleaned = []
    for response in decoded_responses:
        response = response.strip("[PAD] ").strip("<unk>")
        temp = response.strip("<s>").strip("</s>")
        temp = temp.split("\n\nHuman:")[0].strip()
        temp = temp.split("\nHuman:")[0].strip()
        temp = temp.split("\n\nAssistant:")[0].strip()
        temp = temp.split("\nAssistant:")[0].strip()
        temp = temp.split("\n\n\n")[0].strip()
        temp = temp.split("###")[0].strip()
        cleaned.append(temp)
    return cleaned


@torch.no_grad()
def _batch_sp_weights(raw_prompts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = prompt_feature_extractor.encode(
        list(raw_prompts),
        batch_size=min(32, max(1, len(raw_prompts))),
        device=accelerator.device,
    )
    alpha_pc = posterior_calibrator.expected_alpha(
        features,
        base_w=base_w.detach().float().cpu(),
        num_samples=script_args.posterior_num_samples,
        device=accelerator.device,
    )
    if script_args.rho_fixed is None:
        rho = safety_prior(features).to(alpha_pc.device)
    else:
        rho = torch.full((alpha_pc.shape[0],), float(script_args.rho_fixed), device=alpha_pc.device)

    q = q_safe.to(alpha_pc.device, dtype=alpha_pc.dtype).view(1, -1)
    alpha_sp = (1.0 - rho.view(-1, 1)) * alpha_pc + rho.view(-1, 1) * q
    return alpha_pc, rho, alpha_sp


@torch.no_grad()
def _compute_rewards(raw_prompts: Sequence[str], queries: Sequence[str], responses: Sequence[str]):
    signed_scores = reward_model.get_scores(queries, responses).to(accelerator.device)
    alpha_pc, rho, alpha_sp = _batch_sp_weights(raw_prompts)
    scalar_rewards = (alpha_sp.to(signed_scores.dtype) * signed_scores).sum(dim=-1)
    return signed_scores, alpha_pc, rho, alpha_sp, scalar_rewards


def _maybe_print_alpha(
    global_step: int,
    raw_prompts: Sequence[str],
    alpha_pc: torch.Tensor,
    rho: torch.Tensor,
    alpha_sp: torch.Tensor,
):
    if not accelerator.is_local_main_process:
        return
    if script_args.debug_print_alpha_every <= 0 or global_step % script_args.debug_print_alpha_every != 0:
        return
    n = min(script_args.debug_print_alpha_n, len(raw_prompts), alpha_sp.shape[0])
    print("\n" + "=" * 100, flush=True)
    print(f"[Safety-Prior PC-MORLHF alpha samples] global_step={global_step}", flush=True)
    for idx in range(n):
        prompt = str(raw_prompts[idx]).replace("\n", " ")
        if len(prompt) > 220:
            prompt = prompt[:217] + "..."
        pc_str = " ".join([f"pc_{reward_names[j]}={alpha_pc[idx, j].item():.4f}" for j in range(alpha_pc.shape[-1])])
        sp_str = " ".join([f"sp_{reward_names[j]}={alpha_sp[idx, j].item():.4f}" for j in range(alpha_sp.shape[-1])])
        print(f"[sample {idx}] prompt: {prompt}", flush=True)
        print(f"           rho={rho[idx].item():.4f}", flush=True)
        print(f"           {pc_str}", flush=True)
        print(f"           {sp_str}", flush=True)
    print("=" * 100 + "\n", flush=True)

@torch.no_grad()
def run_periodic_validation(global_step: int):
    if script_args.eval_steps <= 0 or len(eval_dataset) == 0:
        return
    if not accelerator.is_local_main_process:
        return

    n_eval = min(script_args.eval_num_prompts, len(eval_dataset))
    examples = [eval_dataset[idx] for idx in range(n_eval)]
    raw_prompts = [ex["raw_prompt"] for ex in examples]
    queries = [ex["query"] for ex in examples]
    query_tensors = [torch.as_tensor(ex["input_ids"], dtype=torch.long, device=gpu_id) for ex in examples]

    was_training = model.training
    model.eval()
    model.gradient_checkpointing_disable()
    model.pretrained_model.config.use_cache = True

    eval_generation_kwargs = dict(generation_kwargs)
    eval_generation_kwargs["max_new_tokens"] = script_args.eval_max_new_tokens

    response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **eval_generation_kwargs)
    responses = _clean_responses(tokenizer.batch_decode(response_tensors))

    signed_scores, alpha_pc, rho, alpha, scalar_rewards = _compute_rewards(raw_prompts, queries, responses)

    columns = ["step", "idx", "raw_prompt", "response"]
    columns.append("rho_safety")
    for name in reward_names:
        columns.append(f"pc_alpha_{name}")
    for name in reward_names:
        columns.append(f"sp_alpha_{name}")
    for name in reward_names:
        columns.append(f"{name}_signed_score")
    columns.append("scalar_reward")

    table = wandb.Table(columns=columns)
    for row_idx in range(n_eval):
        row = [global_step, row_idx, raw_prompts[row_idx], responses[row_idx]]
        row += [float(rho[row_idx].detach().cpu())]
        row += [float(alpha_pc[row_idx, k].detach().cpu()) for k in range(alpha_pc.shape[-1])]
        row += [float(alpha[row_idx, k].detach().cpu()) for k in range(alpha.shape[-1])]
        row += [float(signed_scores[row_idx, k].detach().cpu()) for k in range(signed_scores.shape[-1])]
        row += [float(scalar_rewards[row_idx].detach().cpu())]
        table.add_data(*row)

    if not script_args.disable_wandb:
        if wandb.run is None:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "pcma"),
                entity=os.environ.get("WANDB_ENTITY", None),
                name=script_args.wandb_name,
            )
        wandb.log(
            {
                f"validation_samples/step_{global_step}": table,
                "validation/scalar_reward_mean": float(scalar_rewards.mean().detach().cpu()),
                "validation/scalar_reward_std": float(scalar_rewards.std(unbiased=False).detach().cpu()),
                "validation/rho_safety_mean": float(rho.mean().detach().cpu()),
                "validation/pc_alpha_helpful_mean": float(alpha_pc[:, 0].mean().detach().cpu()),
                "validation/pc_alpha_harmless_mean": float(alpha_pc[:, 1].mean().detach().cpu()) if alpha_pc.shape[-1] > 1 else 0.0,
                "validation/sp_alpha_helpful_mean": float(alpha[:, 0].mean().detach().cpu()),
                "validation/sp_alpha_harmless_mean": float(alpha[:, 1].mean().detach().cpu()) if alpha.shape[-1] > 1 else 0.0,
            },
            step=global_step,
        )

    print(f"[eval] step={global_step}, logged {n_eval} validation generations", flush=True)

    if was_training:
        model.train()
    model.pretrained_model.config.use_cache = False


def _to_float_scalar(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().mean().cpu())
    if isinstance(x, np.ndarray):
        return float(np.asarray(x).mean())
    return float(x)


print("Training........")
model.gradient_checkpointing_disable()
model.pretrained_model.config.use_cache = True

mean_scores, std_scores = [], []
save_data = {
    "kl_mean": [],
    "reward_mean": [],
    "reward_std": [],
    "rho_safety_mean": [],
    "pc_alpha_helpful_mean": [],
    "pc_alpha_harmless_mean": [],
    "sp_alpha_helpful_mean": [],
    "sp_alpha_harmless_mean": [],
    "text_sample": [],
    "batch_time": [],
    "total_time": [],
}

t_start = time.time()
global_step = 0

for epoch in range(script_args.epochs):
    pbar = tqdm(total=len(train_dataset) // script_args.batch_size // accelerator.num_processes)
    for i, batch in enumerate(ppo_trainer.dataloader):
        t_batch_start = time.time()
        print(f"epoch {epoch}, batch {i}")

        query_tensors = [torch.as_tensor(q, dtype=torch.long, device=gpu_id) for q in batch["input_ids"]]

        model.gradient_checkpointing_disable()
        model.pretrained_model.config.use_cache = True
        with torch.no_grad():
            response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)

        responses = _clean_responses(tokenizer.batch_decode(response_tensors))
        response_tensors = [tokenizer.encode(r, return_tensors="pt").squeeze(0).to(gpu_id) for r in responses]
        response_tensors = [rt[: max(int(rt.numel()), 2)] for rt in response_tensors]
        batch["response"] = responses

        raw_prompts = _get_batch_raw_prompts(batch)
        signed_scores, alpha_pc, rho, alpha, scalar_rewards = _compute_rewards(raw_prompts, batch["query"], responses)
        rewards_tensor = [r.detach().to(gpu_id) for r in scalar_rewards]

        print(
            f"iter {epoch}, batch {i}, mean reward: {scalar_rewards.mean().item():.4f}, "
            f"rho_mean={rho.mean().item():.4f}, sp_alpha_mean={alpha.mean(dim=0).detach().cpu().tolist()}",
            flush=True,
        )

        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)
        ppo_trainer.log_stats(stats, batch, [float(r.detach().cpu()) for r in scalar_rewards])

        global_step += 1
        _maybe_print_alpha(global_step, raw_prompts, alpha_pc, rho, alpha)

        all_rewards = accelerator.gather_for_metrics(scalar_rewards.detach())
        kl_value = _to_float_scalar(stats.get("objective/kl", 0.0))
        all_kl = accelerator.gather_for_metrics(
            torch.tensor([kl_value], dtype=torch.float32, device=accelerator.device)
        )
        all_alpha = accelerator.gather_for_metrics(alpha.detach())
        all_alpha_pc = accelerator.gather_for_metrics(alpha_pc.detach())
        all_rho = accelerator.gather_for_metrics(rho.detach())

        if accelerator.is_local_main_process:
            reward_mean = float(all_rewards.mean().detach().cpu())
            reward_std = float(all_rewards.std(unbiased=False).detach().cpu())
            alpha_mean = all_alpha.mean(dim=0).detach().cpu()
            alpha_pc_mean = all_alpha_pc.mean(dim=0).detach().cpu()
            rho_mean = float(all_rho.mean().detach().cpu())
            mean_scores.append(reward_mean)
            std_scores.append(reward_std)

            batch_time = time.time() - t_batch_start
            total_time = time.time() - t_start
            save_data["batch_time"].append(batch_time)
            save_data["total_time"].append(total_time)
            save_data["kl_mean"].append(float(all_kl.mean().detach().cpu()))
            save_data["reward_mean"].append(reward_mean)
            save_data["reward_std"].append(reward_std)
            save_data["rho_safety_mean"].append(rho_mean)
            save_data["pc_alpha_helpful_mean"].append(float(alpha_pc_mean[0]))
            save_data["pc_alpha_harmless_mean"].append(float(alpha_pc_mean[1]) if alpha_pc_mean.numel() > 1 else 0.0)
            save_data["sp_alpha_helpful_mean"].append(float(alpha_mean[0]))
            save_data["sp_alpha_harmless_mean"].append(float(alpha_mean[1]) if alpha_mean.numel() > 1 else 0.0)
            save_data["text_sample"].append((batch["query"][0] + responses[0]) if len(responses) > 0 else "")

            pd.DataFrame(save_data).to_csv(
                os.path.join(script_args.save_directory, script_args.wandb_name, "data.csv"),
                index=False,
            )

            if wandb.run is not None:
                wandb.log(
                    {
                        "train/reward_mean": reward_mean,
                        "train/reward_std": reward_std,
                        "train/kl_mean": float(all_kl.mean().detach().cpu()),
                        "train/rho_safety_mean": rho_mean,
                        "train/pc_alpha_helpful_mean": float(alpha_pc_mean[0]),
                        "train/pc_alpha_harmless_mean": float(alpha_pc_mean[1]) if alpha_pc_mean.numel() > 1 else 0.0,
                        "train/sp_alpha_helpful_mean": float(alpha_mean[0]),
                        "train/sp_alpha_harmless_mean": float(alpha_mean[1]) if alpha_mean.numel() > 1 else 0.0,
                        "train/batch_time": batch_time,
                        "train/total_time": total_time,
                    },
                    step=global_step,
                )

            print(f"iter {epoch}, batch {i}: log finish", flush=True)

        if script_args.eval_steps > 0 and global_step % script_args.eval_steps == 0:
            run_periodic_validation(global_step)

        accelerator.wait_for_everyone()
        pbar.update(1)

        if ppo_trainer.accelerator.is_main_process and i % 100 == 0 and i != 0:
            save_path = os.path.join(script_args.save_directory, script_args.wandb_name, f"batch_{i}")
            ppo_trainer.save_pretrained(save_path)
            print(f"iter {epoch}, batch {i}: model saved", flush=True)

    if ppo_trainer.accelerator.is_main_process:
        save_path = os.path.join(script_args.save_directory, script_args.wandb_name, f"batch_{i}")
        ppo_trainer.save_pretrained(save_path)
        print(f"iter {epoch}, batch {i}: model saved", flush=True)
