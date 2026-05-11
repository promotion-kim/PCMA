"""Fit the posterior calibration model for PC-MODPO.

Example:

PYTHONPATH=. accelerate launch scripts/pcma/fit_posterior.py \
  --sft_model_name PKU-Alignment/alpaca-7b-reproduced \
  --objective_adapter_names /path/dpo-better/best_checkpoint,/path/dpo-safer/best_checkpoint \
  --objective_dataset_names PKU-Alignment/PKU-SafeRLHF-10K-better,PKU-Alignment/PKU-SafeRLHF-10K-safer \
  --feature_source sft_hidden \
  --output_dir /ext_hdd/sjkim/mod/output/pcma/posterior
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import tyro
from accelerate import Accelerator
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils import common_prefix_length, disable_progress_bar_non_local_main, param_sharding_enabled, print_local_main, set_seed
from src.utils.posterior_calibration import (
    CalibrationFitConfig,
    CalibrationPriorConfig,
    FrozenCausalLMPromptFeatureExtractor,
    HFPromptFeatureExtractor,
    LaplacePosteriorCalibrator,
)


disable_progress_bar_non_local_main()


def _split_csv(x: str) -> List[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


@dataclass
class ScriptArguments:
    sft_model_name: str = field(metadata={"help": "SFT/reference model"})
    objective_adapter_names: str = field(metadata={"help": "comma-separated objective adapter paths, e.g. better,safer"})
    objective_dataset_names: str = field(metadata={"help": "comma-separated objective preference datasets in same order"})
    output_dir: str = field(metadata={"help": "where to save posterior_calibrator.json"})
    feature_source: str = field(default="sft_hidden", metadata={"help": "sft_hidden or hf_model"})
    feature_model_name: str = field(default="sentence-transformers/all-MiniLM-L6-v2", metadata={"help": "used only when feature_source=hf_model"})
    feature_pooling: str = field(default="mean", metadata={"help": "mean or last; used when feature_source=sft_hidden"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE)
    sanity_check: Optional[bool] = field(default=False)
    seed: int = field(default=42)
    max_length: int = field(default=512)
    feature_max_length: int = field(default=256)
    per_device_batch_size: int = field(default=2)
    num_workers: int = field(default=0)
    max_examples_per_objective: Optional[int] = field(default=None)
    use_flash_attention_2: Optional[bool] = field(default=False)

    # MAP/Laplace config
    posterior_steps: int = field(default=2000)
    posterior_lr: float = field(default=3e-3)
    posterior_batch_size: int = field(default=512)
    posterior_log_every: int = field(default=100)
    mu_std: float = field(default=0.5) #1.0
    c_std: float = field(default=0.03) #1.0
    u_std: float = field(default=0.1) #0.5
    W_std: float = field(default=0.005) #0.5
    laplace_damping: float = field(default=1e-3)


script_args = tyro.cli(ScriptArguments)
set_seed(script_args.seed)
accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def _format_prompt(example: dict) -> str:
    if "prompt" in example and example["prompt"] is not None:
        return example["prompt"]
    if "raw_prompt" in example and example["raw_prompt"] is not None:
        return script_args.prompt_template.format(raw_prompt=example["raw_prompt"])
    raise KeyError("Dataset example must contain `prompt` or `raw_prompt`.")


def _tokenize_prompt_response(prompt: str, response: str) -> dict:
    prompt_toks = tokenizer(prompt, add_special_tokens=False)
    full_toks = tokenizer(prompt + response, add_special_tokens=False)
    input_ids = full_toks["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = full_toks["attention_mask"] + [1]
    prompt_len = common_prefix_length(prompt_toks["input_ids"], input_ids)
    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len
    if len(input_ids) > script_args.max_length:
        input_ids = input_ids[:script_args.max_length]
        attention_mask = attention_mask[:script_args.max_length]
        labels = labels[:script_args.max_length]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _collate_logp(features: Sequence[dict]) -> dict:
    batch = []
    for f in features:
        batch.append({"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]})
    padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
    max_len = padded["input_ids"].shape[1]
    labels = []
    for f in features:
        lab = f["labels"] + [-100] * (max_len - len(f["labels"]))
        labels.append(lab)
    padded["labels"] = torch.tensor(labels, dtype=torch.long)
    return padded


@torch.no_grad()
def _sequence_logps(model, batch: dict) -> torch.Tensor:
    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    labels = batch["labels"][:, 1:].clone()
    mask = labels.ne(-100)
    labels = labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1).detach().cpu()


def _build_pairs(dataset) -> List[dict]:
    pairs = []
    for ex in dataset:
        prompt = _format_prompt(ex)
        chosen = ex.get("chosen")
        rejected = ex.get("rejected")
        if chosen is None or rejected is None:
            raise KeyError("Preference dataset must contain `chosen` and `rejected` fields.")
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "raw_prompt": ex.get("raw_prompt", prompt)})
        if script_args.max_examples_per_objective and len(pairs) >= script_args.max_examples_per_objective:
            break
    return pairs


def _compute_objective_gaps(model, adapter_name: str, pairs: List[dict]) -> Tuple[List[str], torch.Tensor]:
    prompts: List[str] = []
    gaps: List[torch.Tensor] = []
    model.eval()
    model.set_adapter(adapter_name)

    items = []
    for p in pairs:
        prompts.append(p["raw_prompt"])
        items.append(_tokenize_prompt_response(p["prompt"], p["chosen"]))
        items.append(_tokenize_prompt_response(p["prompt"], p["rejected"]))

    loader = DataLoader(items, batch_size=script_args.per_device_batch_size * 2, collate_fn=_collate_logp, num_workers=script_args.num_workers)
    policy_logps_all = []
    ref_logps_all = []
    num_pairs = len(pairs)
    num_batches = len(loader)

    for step, batch in enumerate(
        tqdm(
            loader,
            total=num_batches,
            desc=f"{adapter_name}: computing logp gaps",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
        )
    ):
        policy_logps_all.append(_sequence_logps(model, batch))
        with model.disable_adapter():
            ref_logps_all.append(_sequence_logps(model, batch))

        if accelerator.is_local_main_process and (step + 1) % 50 == 0:
            processed_items = min((step + 1) * loader.batch_size, len(items))
            processed_pairs = processed_items // 2
            print(
                f"[{adapter_name}] processed {processed_pairs}/{num_pairs} pairs "
                f"({100.0 * processed_pairs / max(num_pairs, 1):.1f}%)",
                flush=True,
            )
    policy_logps = torch.cat(policy_logps_all, dim=0).view(-1, 2)
    ref_logps = torch.cat(ref_logps_all, dim=0).view(-1, 2)

    # g_i(y) = log pi_i(y|x) - log pi_ref(y|x). Chosen is preferred in each objective dataset.
    gap = (policy_logps[:, 0] - ref_logps[:, 0]) - (policy_logps[:, 1] - ref_logps[:, 1])
    return prompts, gap.float()


objective_adapters = _split_csv(script_args.objective_adapter_names)
objective_datasets = _split_csv(script_args.objective_dataset_names)
if len(objective_adapters) != len(objective_datasets):
    raise ValueError("objective_adapter_names and objective_dataset_names must have the same length.")
num_objectives = len(objective_adapters)

print_local_main("loading SFT/reference model and objective adapters...")
base = AutoModelForCausalLM.from_pretrained(
    script_args.sft_model_name,
    use_flash_attention_2=script_args.use_flash_attention_2,
    torch_dtype=torch.bfloat16,
    **({"device_map": {"": accelerator.local_process_index}} if not param_sharding_enabled() else {}),
)
base.config.update({"use_cache": False, "pad_token_id": base.config.eos_token_id})
model = PeftModel.from_pretrained(base, objective_adapters[0], adapter_name="obj0", is_trainable=False)
for i, path in enumerate(objective_adapters[1:], start=1):
    model.load_adapter(path, adapter_name=f"obj{i}", is_trainable=False)
model.to(accelerator.device)
model.eval()

all_prompts: List[str] = []
all_features_prompts: List[str] = []
all_gaps: List[torch.Tensor] = []
all_obj_idx: List[torch.Tensor] = []

for i, dataset_name in enumerate(objective_datasets):
    print_local_main(f"loading dataset for objective {i}: {dataset_name}")
    rdp = DATASET_CONFIGS[dataset_name](prompt_template=script_args.prompt_template, sanity_check=script_args.sanity_check)
    dataset = rdp.get_preference_dataset(split="train")
    pairs = _build_pairs(dataset)
    print_local_main(f"computing policy-induced gaps for objective {i} with {len(pairs)} examples")
    prompts, gaps = _compute_objective_gaps(model, f"obj{i}", pairs)
    all_features_prompts.extend(prompts)
    all_gaps.append(gaps)
    all_obj_idx.append(torch.full((len(gaps),), i, dtype=torch.long))

signed_gaps = torch.cat(all_gaps, dim=0)
objective_idx = torch.cat(all_obj_idx, dim=0)

print_local_main("extracting prompt features...")
if script_args.feature_source == "sft_hidden":
    # Self-contained feature extractor: frozen adapter-free SFT/reference hidden states.
    feature_extractor = FrozenCausalLMPromptFeatureExtractor(
        model=model,
        tokenizer=tokenizer,
        max_length=script_args.feature_max_length,
        device=accelerator.device,
        pooling=script_args.feature_pooling,
        disable_adapter=True,
        prompt_template=script_args.prompt_template,
    )
    feature_id = f"sft_hidden::{script_args.sft_model_name}::{script_args.feature_pooling}"
    features = feature_extractor.encode(
        all_features_prompts,
        batch_size=max(1, script_args.per_device_batch_size),
        device="cpu",
    )
elif script_args.feature_source == "hf_model":
    # Prototype/ablation option: external sentence encoder such as all-MiniLM-L6-v2.
    feature_extractor = HFPromptFeatureExtractor(
        script_args.feature_model_name,
        max_length=script_args.feature_max_length,
        device=f"cuda:{accelerator.local_process_index}" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    feature_id = script_args.feature_model_name
    features = feature_extractor.encode(all_features_prompts, batch_size=32, device="cpu")
else:
    raise ValueError("feature_source must be either 'sft_hidden' or 'hf_model'.")

print_local_main("fitting MAP + diagonal Laplace posterior...")
prior = CalibrationPriorConfig(
    mu_std=script_args.mu_std,
    c_std=script_args.c_std,
    u_std=script_args.u_std,
    w_std=script_args.W_std,
    laplace_damping=script_args.laplace_damping,
)
fit_cfg = CalibrationFitConfig(
    num_steps=script_args.posterior_steps,
    lr=script_args.posterior_lr,
    batch_size=script_args.posterior_batch_size,
    log_every=script_args.posterior_log_every,
    seed=script_args.seed,
    device=f"cuda:{accelerator.local_process_index}" if torch.cuda.is_available() else "cpu",
)
calibrator = LaplacePosteriorCalibrator.fit(
    features=features,
    objective_idx=objective_idx,
    signed_gaps=signed_gaps,
    num_objectives=num_objectives,
    prior=prior,
    fit_cfg=fit_cfg,
    feature_model_name=feature_id,
)
if accelerator.is_local_main_process:
    calibrator.save_pretrained(script_args.output_dir)
    print(f"saved posterior calibrator to {script_args.output_dir}")
