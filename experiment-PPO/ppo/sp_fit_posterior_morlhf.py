"""Fit the PCMA posterior calibrator used by Safety-Prior PC-MORLHF.

This is the MORLHF counterpart of PC-MODPO's posterior fitting stage.
Instead of policy-induced implicit rewards from objective-specific adapters, this
script uses explicit objective scorers:

  helpful  : PKU-Alignment/beaver-7b-v1.0-reward
  harmless : - PKU-Alignment/beaver-7b-v1.0-cost

For each objective dataset, the chosen response is assumed to be preferred for
that objective. We compute signed scorer gaps

    Delta s_i = s_i(x, y_chosen) - s_i(x, y_rejected)

and fit the same LaplacePosteriorCalibrator used by PC-MODPO.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import tyro
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from safe_rlhf.models import AutoModelForScore
from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from src.utils import disable_progress_bar_non_local_main, param_sharding_enabled, print_local_main, set_seed
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
    # Base / data
    sft_model_name: str = field(metadata={"help": "SFT/reference model used for prompt features"})
    objective_dataset_names: str = field(
        metadata={"help": "comma-separated objective datasets, e.g. better,safer"}
    )
    output_dir: str = field(metadata={"help": "where to save posterior_calibrator.json"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE)
    sanity_check: Optional[bool] = field(default=False)
    seed: int = field(default=42)

    # Explicit objective scorers.
    objective_names: str = field(default="helpful,harmless")
    objective_model_names: str = field(
        default="PKU-Alignment/beaver-7b-v1.0-reward,PKU-Alignment/beaver-7b-v1.0-cost"
    )
    objective_signs: str = field(default="1,-1", metadata={"help": "1 for reward, -1 for cost-to-reward"})

    # Scoring / feature extraction
    scorer_max_length: int = field(default=512)
    scorer_batch_size: int = field(default=2)
    feature_source: str = field(default="sft_hidden", metadata={"help": "sft_hidden or hf_model"})
    feature_model_name: str = field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        metadata={"help": "used only when feature_source=hf_model"},
    )
    feature_pooling: str = field(default="mean", metadata={"help": "mean or last; used when feature_source=sft_hidden"})
    feature_max_length: int = field(default=256)
    feature_batch_size: int = field(default=8)
    max_examples_per_objective: Optional[int] = field(default=None)
    use_flash_attention_2: Optional[bool] = field(default=False)

    # MAP/Laplace config
    posterior_steps: int = field(default=2000)
    posterior_lr: float = field(default=3e-3)
    posterior_batch_size: int = field(default=512)
    posterior_log_every: int = field(default=100)
    mu_std: float = field(default=0.5)
    c_std: float = field(default=0.03)
    u_std: float = field(default=0.1)
    W_std: float = field(default=0.005)
    laplace_damping: float = field(default=1e-3)


script_args = tyro.cli(ScriptArguments)
set_seed(script_args.seed)
accelerator = Accelerator()

objective_datasets = _split_csv(script_args.objective_dataset_names)
objective_names = _split_csv(script_args.objective_names)
objective_model_names = _split_csv(script_args.objective_model_names)
objective_signs = [float(x) for x in _split_csv(script_args.objective_signs)]

if not (len(objective_datasets) == len(objective_names) == len(objective_model_names) == len(objective_signs)):
    raise ValueError(
        "objective_dataset_names, objective_names, objective_model_names, and objective_signs "
        "must have the same length."
    )
num_objectives = len(objective_names)


def _format_prompt(example: dict) -> Tuple[str, str]:
    """Return (formatted_prompt, raw_prompt_for_features)."""
    if "raw_prompt" in example and example["raw_prompt"] is not None:
        raw_prompt = str(example["raw_prompt"])
        formatted = script_args.prompt_template.format(raw_prompt=raw_prompt)
        return formatted, raw_prompt

    if "prompt" in example and example["prompt"] is not None:
        prompt = str(example["prompt"])
        return prompt, prompt

    raise KeyError("Dataset example must contain `prompt` or `raw_prompt`.")


def _build_pairs(dataset) -> List[dict]:
    pairs = []
    for ex in dataset:
        prompt, raw_prompt = _format_prompt(ex)
        chosen = ex.get("chosen")
        rejected = ex.get("rejected")
        if chosen is None or rejected is None:
            raise KeyError("Preference dataset must contain `chosen` and `rejected` fields.")
        pairs.append(
            {
                "prompt": prompt,
                "raw_prompt": raw_prompt,
                "chosen": str(chosen),
                "rejected": str(rejected),
            }
        )
        if script_args.max_examples_per_objective and len(pairs) >= script_args.max_examples_per_objective:
            break
    return pairs


class ExplicitScoreModel:
    """Wrapper around Safe-RLHF AutoModelForScore."""

    def __init__(self, model_name: str, sign: float, device: torch.device):
        self.model_name = model_name
        self.sign = float(sign)
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForScore.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map={"": accelerator.local_process_index} if torch.cuda.is_available() else None,
        )
        self.model.eval()

    @torch.no_grad()
    def score_texts(self, texts: Sequence[str]) -> torch.Tensor:
        scores = []
        loader = DataLoader(list(texts), batch_size=script_args.scorer_batch_size, shuffle=False)
        for batch_texts in tqdm(
            loader,
            desc=f"scoring with {self.model_name}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
        ):
            toks = self.tokenizer(
                list(batch_texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=script_args.scorer_max_length,
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            out = self.model(**toks)
            s = getattr(out, "end_scores", None)
            if s is None:
                s = getattr(out, "scores", None)
            if s is None:
                raise RuntimeError(f"AutoModelForScore output has no end_scores/scores: {out}")
            if s.ndim == 2 and s.shape[-1] == 1:
                s = s.squeeze(-1)
            elif s.ndim >= 2:
                s = s[:, -1]
            scores.append(s.float().detach().cpu())
        raw = torch.cat(scores, dim=0)
        return raw * self.sign


all_feature_prompts: List[str] = []
all_gaps: List[torch.Tensor] = []
all_obj_idx: List[torch.Tensor] = []

for i, dataset_name in enumerate(objective_datasets):
    print_local_main(f"loading objective dataset {i} ({objective_names[i]}): {dataset_name}")
    rdp = DATASET_CONFIGS[dataset_name](
        prompt_template=script_args.prompt_template,
        sanity_check=script_args.sanity_check,
    )
    dataset = rdp.get_preference_dataset(split="train")
    pairs = _build_pairs(dataset)
    print_local_main(f"objective {i}: {len(pairs)} preference pairs")

    scorer = ExplicitScoreModel(
        objective_model_names[i],
        sign=objective_signs[i],
        device=accelerator.device,
    )

    chosen_texts = [p["prompt"] + p["chosen"] for p in pairs]
    rejected_texts = [p["prompt"] + p["rejected"] for p in pairs]

    chosen_scores = scorer.score_texts(chosen_texts)
    rejected_scores = scorer.score_texts(rejected_texts)
    gaps = (chosen_scores - rejected_scores).float()

    all_feature_prompts.extend([p["raw_prompt"] for p in pairs])
    all_gaps.append(gaps)
    all_obj_idx.append(torch.full((len(gaps),), i, dtype=torch.long))

    # Free the 7B scorer before loading the next one.
    del scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

signed_gaps = torch.cat(all_gaps, dim=0)
objective_idx = torch.cat(all_obj_idx, dim=0)

print_local_main("extracting prompt features...")
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

    # Different local versions of FrozenCausalLMPromptFeatureExtractor may or may
    # not accept prompt_template, so use a small compatibility fallback.
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
        all_feature_prompts,
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
    features = feature_extractor.encode(all_feature_prompts, batch_size=32, device="cpu")
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
