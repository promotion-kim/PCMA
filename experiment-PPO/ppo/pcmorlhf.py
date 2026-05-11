"""PC-MORLHF: PPO training with posterior-calibrated scalar rewards.

This script is the explicit-reward counterpart of PC-MODPO. It keeps the PPO
training loop from MORLHF, but replaces fixed scalarization

    w_help * reward + (1-w_help) * (-cost)

with prompt-dependent posterior-calibrated weights

    alpha_help(x,w) * reward + alpha_safe(x,w) * (-cost).

It also periodically generates from validation prompts and logs a W&B table.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

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
    dataset_name: str = field(default="PKU-Alignment/PKU-SafeRLHF-10K-better")
    prompt_template: str = field(default=DEFAULT_PROMPT_TEMPLATE)
    train_split: str = field(default="train")
    eval_split: str = field(default="validation")
    seed: int = field(default=8888)

    # PPO
    save_directory: str = field(default="./logs_pcmorlhf")
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
    preference: float = field(default=0.5, metadata={"help": "base helpfulness weight w_H"})
    posterior_num_samples: int = field(default=10, metadata={"help": "0=MAP plug-in; >0=Laplace MC average"})
    alpha_min: float = field(default=0.05)
    feature_source: str = field(default="sft_hidden", metadata={"help": "sft_hidden or hf_model"})
    feature_model_name: Optional[str] = field(default=None)
    feature_pooling: str = field(default="mean")
    feature_max_length: int = field(default=256)

    # W&B / monitoring
    log_with: str = field(default="wandb")
    disable_wandb: bool = field(default=False)
    wandb_name: str = field(default="pcmorlhf")
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

base_w = torch.tensor([script_args.preference, 1.0 - script_args.preference], dtype=torch.float32)
base_w = base_w / base_w.sum().clamp_min(1e-12)
script_args.wandb_name = f"{script_args.wandb_name}_pref{base_w[0].item():.1f}_{base_w[1].item():.1f}"

os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)

print(f"base model: {script_args.base_model_name}")
print(f"reward_names: {reward_names}")
print(f"base_w: {base_w.tolist()}")
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
def _batch_alpha(raw_prompts: Sequence[str]) -> torch.Tensor:
    features = prompt_feature_extractor.encode(
        list(raw_prompts),
        batch_size=min(32, max(1, len(raw_prompts))),
        device=accelerator.device,
    )
    alpha = posterior_calibrator.expected_alpha(
        features,
        base_w=base_w.detach().float().cpu(),
        num_samples=script_args.posterior_num_samples,
        device=accelerator.device,
    )
    if script_args.alpha_min and script_args.alpha_min > 0:
        alpha = alpha.clamp_min(script_args.alpha_min)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return alpha


@torch.no_grad()
def _compute_rewards(raw_prompts: Sequence[str], queries: Sequence[str], responses: Sequence[str]):
    signed_scores = reward_model.get_scores(queries, responses).to(accelerator.device)
    alpha = _batch_alpha(raw_prompts)
    scalar_rewards = (alpha.to(signed_scores.dtype) * signed_scores).sum(dim=-1)
    return signed_scores, alpha, scalar_rewards


def _maybe_print_alpha(global_step: int, raw_prompts: Sequence[str], alpha: torch.Tensor):
    if not accelerator.is_local_main_process:
        return
    if script_args.debug_print_alpha_every <= 0 or global_step % script_args.debug_print_alpha_every != 0:
        return
    n = min(script_args.debug_print_alpha_n, len(raw_prompts), alpha.shape[0])
    print("\n" + "=" * 100, flush=True)
    print(f"[PC-MORLHF alpha samples] global_step={global_step}", flush=True)
    for idx in range(n):
        prompt = str(raw_prompts[idx]).replace("\n", " ")
        if len(prompt) > 220:
            prompt = prompt[:217] + "..."
        alpha_str = " ".join([f"{reward_names[j]}={alpha[idx, j].item():.4f}" for j in range(alpha.shape[-1])])
        print(f"[sample {idx}] prompt: {prompt}", flush=True)
        print(f"           alpha: {alpha_str}", flush=True)
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

    signed_scores, alpha, scalar_rewards = _compute_rewards(raw_prompts, queries, responses)

    columns = ["step", "idx", "raw_prompt", "response"]
    for name in reward_names:
        columns.append(f"alpha_{name}")
    for name in reward_names:
        columns.append(f"{name}_signed_score")
    columns.append("scalar_reward")

    table = wandb.Table(columns=columns)
    for row_idx in range(n_eval):
        row = [global_step, row_idx, raw_prompts[row_idx], responses[row_idx]]
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
                "validation/alpha_helpful_mean": float(alpha[:, 0].mean().detach().cpu()),
                "validation/alpha_harmless_mean": float(alpha[:, 1].mean().detach().cpu()) if alpha.shape[-1] > 1 else 0.0,
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
    "alpha_helpful_mean": [],
    "alpha_harmless_mean": [],
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
        signed_scores, alpha, scalar_rewards = _compute_rewards(raw_prompts, batch["query"], responses)
        rewards_tensor = [r.detach().to(gpu_id) for r in scalar_rewards]

        print(
            f"iter {epoch}, batch {i}, mean reward: {scalar_rewards.mean().item():.4f}, "
            f"alpha_mean={alpha.mean(dim=0).detach().cpu().tolist()}",
            flush=True,
        )

        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)
        ppo_trainer.log_stats(stats, batch, [float(r.detach().cpu()) for r in scalar_rewards])

        global_step += 1
        _maybe_print_alpha(global_step, raw_prompts, alpha)

        all_rewards = accelerator.gather_for_metrics(scalar_rewards.detach())
        kl_value = _to_float_scalar(stats.get("objective/kl", 0.0))
        all_kl = accelerator.gather_for_metrics(
            torch.tensor([kl_value], dtype=torch.float32, device=accelerator.device)
        )
        all_alpha = accelerator.gather_for_metrics(alpha.detach())

        if accelerator.is_local_main_process:
            reward_mean = float(all_rewards.mean().detach().cpu())
            reward_std = float(all_rewards.std(unbiased=False).detach().cpu())
            alpha_mean = all_alpha.mean(dim=0).detach().cpu()
            mean_scores.append(reward_mean)
            std_scores.append(reward_std)

            batch_time = time.time() - t_batch_start
            total_time = time.time() - t_start
            save_data["batch_time"].append(batch_time)
            save_data["total_time"].append(total_time)
            save_data["kl_mean"].append(float(all_kl.mean().detach().cpu()))
            save_data["reward_mean"].append(reward_mean)
            save_data["reward_std"].append(reward_std)
            save_data["alpha_helpful_mean"].append(float(alpha_mean[0]))
            save_data["alpha_harmless_mean"].append(float(alpha_mean[1]) if alpha_mean.numel() > 1 else 0.0)
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
                        "train/alpha_helpful_mean": float(alpha_mean[0]),
                        "train/alpha_harmless_mean": float(alpha_mean[1]) if alpha_mean.numel() > 1 else 0.0,
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
