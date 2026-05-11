"""
Command-line entry point for VI-MOD.

Modes:
  1. train_calibrator
  2. generate

Fair MOD-comparison generation:
  --prompt_source mod_eval

This makes VI-MOD draw prompts in the same way as the MOD baseline:
  - DATASET_CONFIGS[dataset_name](prompt_template=...)
  - validation split
  - DPOTrainer_Light data_collator(..., generate=True)
  - random.sample(range(num_samples), k=eval_batch_size)
  - iters=50 by default
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.utils.vi_util import (
    FeatureConfig,
    LLMPromptFeatureExtractor,
    VIPriorConfig,
    VIMODFusionModel,
    apply_prompt_template,
    compute_signed_margins_for_rows,
    group_rows_by_objective,
    load_calibration_jsonl,
    load_vi_calibrator,
    save_vi_calibrator,
    train_vi_calibrator_from_margins,
)


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_str(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def str_to_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype={name}")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# TRL compatibility patch
# -----------------------------------------------------------------------------

def patch_trl_compat() -> None:
    """
    Some versions of this repo expect older TRL symbols such as PeftSavingCallback.
    In trl==0.7.11, this symbol may not exist. For generation/eval, a no-op
    callback is sufficient.
    """
    try:
        import trl.trainer.utils as trl_utils
    except Exception:
        return

    if not hasattr(trl_utils, "PeftSavingCallback"):
        from transformers import TrainerCallback

        class PeftSavingCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                return control

        trl_utils.PeftSavingCallback = PeftSavingCallback

    if not hasattr(trl_utils, "disable_dropout_in_model"):
        import torch.nn as nn

        def disable_dropout_in_model(model):
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0

        trl_utils.disable_dropout_in_model = disable_dropout_in_model

    if not hasattr(trl_utils, "pad_to_length"):
        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            pad_size = list(tensor.shape)
            pad_size[dim] = length - tensor.size(dim)
            pad_tensor = tensor.new_full(pad_size, pad_value)
            return torch.cat([tensor, pad_tensor], dim=dim)

        trl_utils.pad_to_length = pad_to_length

    if not hasattr(trl_utils, "compute_accuracy"):
        def compute_accuracy(eval_pred):
            predictions, labels = eval_pred
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            predictions = torch.as_tensor(predictions)
            labels = torch.as_tensor(labels)
            preds = predictions.argmax(dim=-1)
            mask = labels != -100
            if mask.sum() == 0:
                return {"accuracy": 0.0}
            return {"accuracy": float((preds[mask] == labels[mask]).float().mean())}

        trl_utils.compute_accuracy = compute_accuracy


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_model_and_tokenizer(args):
    print("[stage] loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[stage] loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=str_to_dtype(args.torch_dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = True

    adapter_paths = parse_csv_str(args.adapter_paths)
    if len(adapter_paths) == 0:
        raise ValueError("--adapter_paths must contain at least one adapter path.")

    print(f"[stage] loading adapter model_0: {adapter_paths[0]}", flush=True)
    model = PeftModel.from_pretrained(
        model,
        adapter_paths[0],
        adapter_name="model_0",
        is_trainable=False,
    )

    for i, path in enumerate(adapter_paths[1:], start=1):
        print(f"[stage] loading adapter model_{i}: {path}", flush=True)
        model.load_adapter(path, adapter_name=f"model_{i}", is_trainable=False)

    model.eval()
    return model, tokenizer, adapter_paths


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------

def normalize_margins(margins: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return margins / torch.sqrt(torch.mean(margins.float() ** 2) + eps)


def train_calibrator(args) -> None:
    model, tokenizer, adapter_paths = load_model_and_tokenizer(args)
    num_objectives = len(adapter_paths)

    rows = load_calibration_jsonl(args.calibration_jsonl)
    grouped = group_rows_by_objective(rows, num_objectives=num_objectives)
    if any(len(g) == 0 for g in grouped):
        counts = [len(g) for g in grouped]
        raise ValueError(f"Every objective must have at least one calibration row. counts={counts}")

    feature_config = FeatureConfig(
        projection_dim=args.feature_dim,
        normalize=not args.no_feature_normalize,
        seed=args.feature_seed,
        batch_size=args.feature_batch_size,
        max_length=args.feature_max_length,
        pooling=args.feature_pooling,
    )
    extractor = LLMPromptFeatureExtractor(model, tokenizer, feature_config)

    features_by_obj = []
    margins_by_obj = []

    for i, rows_i in enumerate(grouped):
        print(f"[calibration] objective={i}, rows={len(rows_i)}", flush=True)

        prompts_i = [
            apply_prompt_template(
                r.get("prompt", r.get("raw_prompt", r.get("input", ""))),
                args.prompt_template,
            )
            for r in rows_i
        ]

        feats_i = extractor.encode(prompts_i)

        margins_i = compute_signed_margins_for_rows(
            model=model,
            tokenizer=tokenizer,
            rows=rows_i,
            objective_adapter_name=f"model_{i}",
            prompt_template=args.prompt_template,
            batch_size=args.logprob_batch_size,
            max_length=args.max_length,
            length_normalize=not args.no_length_normalize,
        )

        if args.normalize_margins:
            margins_i = normalize_margins(margins_i)

        features_by_obj.append(feats_i)
        margins_by_obj.append(margins_i)

        print(
            f"[calibration] objective={i} "
            f"margin_mean={float(margins_i.mean()):.4f}, "
            f"margin_rms={float(torch.sqrt((margins_i ** 2).mean())):.4f}",
            flush=True,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    feature_state_path = os.path.join(args.output_dir, "features.pt")
    calibrator_path = os.path.join(args.output_dir, "calibrator.pt")

    extractor.save_projector(feature_state_path)

    prior = VIPriorConfig(
        m0=args.m0,
        s0=args.s0,
        sigma_u=args.sigma_u,
        sigma_b=args.sigma_b,
    )

    calibrator = train_vi_calibrator_from_margins(
        features_by_obj=features_by_obj,
        margins_by_obj=margins_by_obj,
        num_steps=args.vi_steps,
        lr=args.vi_lr,
        num_mc_samples=args.vi_mc_samples,
        prior=prior,
        device=args.vi_device,
        print_every=args.print_every,
    )

    save_vi_calibrator(calibrator, calibrator_path)
    print(f"[saved] calibrator: {calibrator_path}", flush=True)
    print(f"[saved] feature state: {feature_state_path}", flush=True)


# -----------------------------------------------------------------------------
# Optional prompt-file mode
# -----------------------------------------------------------------------------

@torch.no_grad()
def read_prompts(args) -> List[str]:
    if args.prompt:
        return [args.prompt]

    if args.prompts_file:
        prompts = []
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("{"):
                    obj = json.loads(line)
                    prompts.append(
                        obj.get("prompt", obj.get("raw_prompt", obj.get("input", "")))
                    )
                else:
                    prompts.append(line)

        return prompts

    raise ValueError("Provide --prompt or --prompts_file.")


# -----------------------------------------------------------------------------
# MOD-like prompt sampler
# -----------------------------------------------------------------------------

def get_mod_prompt_batches(args, model, tokenizer) -> Iterator[Tuple[int, List[int], Dict[str, torch.Tensor]]]:
    """
    Draw prompt batches in the same way as scripts/eval/mod.py.

    This intentionally follows the MOD baseline:
      - DATASET_CONFIGS[args.dataset_name]
      - split='validation'
      - DPOTrainer_Light
      - trainer.data_collator(..., generate=True)
      - random.sample(range(num_samples), k=trainer.args.eval_batch_size)
    """
    patch_trl_compat()

    from transformers import TrainingArguments
    from src.trainer.light_dpo_trainer import DPOTrainer_Light
    from src.data.configs import DATASET_CONFIGS

    if not args.dataset_caching:
        from datasets import disable_caching
        disable_caching()

    if args.dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset_name={args.dataset_name}. "
            f"Available examples={list(DATASET_CONFIGS.keys())[:20]}"
        )

    print(f"[MOD prompt sampler] dataset_name={args.dataset_name}", flush=True)
    print(f"[MOD prompt sampler] split=validation", flush=True)
    print(f"[MOD prompt sampler] prompt_template={args.prompt_template}", flush=True)

    rdp = DATASET_CONFIGS[args.dataset_name](
        prompt_template=args.prompt_template,
        sanity_check=args.sanity_check,
    )

    train_dataset = rdp.get_preference_dataset(split="train")
    eval_dataset = rdp.get_preference_dataset(split="validation")

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "tmp_vi_mod_prompt_sampler"),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.generation_batch_size,
        per_device_eval_batch_size=args.generation_batch_size,
        gradient_accumulation_steps=1,
        fp16=(args.torch_dtype == "fp16"),
        bf16=(args.torch_dtype == "bf16"),
        remove_unused_columns=False,
        report_to="none",
        run_name="vi_mod_prompt_sampler",
        dataloader_drop_last=False,
    )

    trainer = DPOTrainer_Light(
        ref_model=model,
        beta=args.beta,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_proc=args.num_proc,
        generate_during_eval=True,
    )

    num_samples = len(trainer.eval_dataset)

    print(f"[MOD prompt sampler] num_samples={num_samples}", flush=True)
    print(f"[MOD prompt sampler] eval_batch_size={trainer.args.eval_batch_size}", flush=True)
    print(f"[MOD prompt sampler] iters={args.mod_iters}", flush=True)
    print(
        f"[MOD prompt sampler] total generations={args.mod_iters * trainer.args.eval_batch_size}",
        flush=True,
    )

    # Same style as MOD baseline: seed then random.sample.
    random.seed(args.seed)

    for iter_idx in tqdm(range(args.mod_iters), desc="[MOD-like prompt sampling]"):
        random_indices = random.sample(
            range(num_samples),
            k=trainer.args.eval_batch_size,
        )

        print(f"sampled: {len(random_indices)}/{num_samples}", flush=True)

        dataloader = trainer.get_eval_dataloader(trainer.eval_dataset)
        random_batch_dataset = dataloader.dataset.select(random_indices)

        batch = trainer.data_collator(random_batch_dataset, generate=True)
        batch = trainer._prepare_inputs(batch)

        yield iter_idx, random_indices, batch


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------

def build_generation_config(args, tokenizer) -> GenerationConfig:
    return GenerationConfig(
        max_length=args.max_length,
        do_sample=False,
        num_beams=args.num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

def print_generation_preview(
    iter_idx: int,
    local_idx: int,
    dataset_idx: Optional[int],
    prompt: str,
    output: str,
    vi_weight: Optional[List[float]] = None,
    max_chars: int = 1500,
) -> None:
    print("\n" + "=" * 100, flush=True)
    print(
        f"[GENERATION] iter={iter_idx} local={local_idx} dataset_idx={dataset_idx}",
        flush=True,
    )

    if vi_weight is not None:
        print(f"[VI weights] {vi_weight}", flush=True)

    print("-" * 100, flush=True)
    print("[PROMPT]", flush=True)
    print(prompt[:max_chars], flush=True)

    print("-" * 100, flush=True)
    print("[OUTPUT]", flush=True)
    print(output[:max_chars], flush=True)
    print("=" * 100 + "\n", flush=True)

@torch.no_grad()
def generate(args) -> None:
    set_all_seeds(args.seed)

    model, tokenizer, adapter_paths = load_model_and_tokenizer(args)

    user_weights = parse_csv_floats(args.user_weights)
    if len(user_weights) != len(adapter_paths):
        raise ValueError(
            f"len(user_weights)={len(user_weights)} must match num adapters={len(adapter_paths)}"
        )

    print("[stage] loading VI calibrator...", flush=True)
    calibrator = load_vi_calibrator(args.calibrator_path, map_location=args.vi_device).to(args.vi_device)

    if args.feature_state_path:
        feature_config, projector = LLMPromptFeatureExtractor.load_projector(args.feature_state_path)
    else:
        feature_config = FeatureConfig(
            projection_dim=calibrator.feature_dim,
            normalize=not args.no_feature_normalize,
            seed=args.feature_seed,
            batch_size=args.feature_batch_size,
            max_length=args.feature_max_length,
            pooling=args.feature_pooling,
        )
        projector = None

    extractor = LLMPromptFeatureExtractor(
        model,
        tokenizer,
        feature_config,
        projector=projector,
    )

    fusion = VIMODFusionModel(
        model=model,
        user_weights=user_weights,
        f_type=args.divergence_type,
        vi_calibrator=calibrator,
        rho=args.rho,
        num_vi_samples=args.num_vi_samples,
    )

    gen_config = build_generation_config(args, tokenizer)
    results = []

    if args.prompt_source == "mod_eval":
        for iter_idx, random_indices, batch in get_mod_prompt_batches(args, model, tokenizer):
            prompt_input_ids = batch["prompt_input_ids"]
            prompt_attention_mask = batch["prompt_attention_mask"]

            prompt_texts = tokenizer.batch_decode(
                prompt_input_ids,
                skip_special_tokens=True,
            )

            prompt_features = extractor.encode(prompt_texts).to(args.vi_device)

            out = fusion.generate(
                inputs=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                prompt_features=prompt_features.to(model.device),
                generation_config=gen_config,
            )

            decoded_outputs = tokenizer.batch_decode(out, skip_special_tokens=True)

            vi_weights = None
            if getattr(fusion, "_vi_weights", None) is not None:
                vi_weights = fusion._vi_weights.detach().float().cpu().tolist()

            for local_idx, output in enumerate(decoded_outputs):
                this_vi_weight = None
                if vi_weights is not None and local_idx < len(vi_weights):
                    this_vi_weight = vi_weights[local_idx]

                if args.print_generations:
                    print_generation_preview(
                        iter_idx=int(iter_idx),
                        local_idx=int(local_idx),
                        dataset_idx=int(random_indices[local_idx]),
                        prompt=prompt_texts[local_idx],
                        output=output,
                        vi_weight=this_vi_weight,
                        max_chars=args.print_max_chars,
                    )

                row = {
                    "method": "vi_mod",
                    "iter_idx": int(iter_idx),
                    "local_idx": int(local_idx),
                    "dataset_idx": int(random_indices[local_idx]),
                    "prompt": prompt_texts[local_idx],
                    "output": output,
                    "user_weights": user_weights,
                    "weight_1": float(user_weights[0]),
                    "weight_2": float(user_weights[1]) if len(user_weights) > 1 else None,
                    "rho": float(args.rho),
                    "f_type": args.divergence_type,
                    "seed": int(args.seed),
                    "max_length": int(args.max_length),
                    "num_beams": int(args.num_beams),
                }

                if this_vi_weight is not None:
                    row["vi_weights"] = this_vi_weight

                results.append(row)

    elif args.prompt_source == "file":
        raw_prompts = read_prompts(args)
        prompts = raw_prompts if args.prompts_are_templated else [
            apply_prompt_template(p, args.prompt_template)
            for p in raw_prompts
        ]

        bs = args.generation_batch_size

        for start in tqdm(range(0, len(prompts), bs), desc="[VI-MOD file generate]"):
            raw_batch = raw_prompts[start:start + bs]
            prompt_batch = prompts[start:start + bs]

            prompt_features = extractor.encode(prompt_batch).to(args.vi_device)

            enc = tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.feature_max_length,
            ).to(model.device)

            out = fusion.generate(
                inputs=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                prompt_features=prompt_features.to(model.device),
                generation_config=gen_config,
            )

            decoded_outputs = tokenizer.batch_decode(out, skip_special_tokens=True)

            vi_weights = None
            if getattr(fusion, "_vi_weights", None) is not None:
                vi_weights = fusion._vi_weights.detach().float().cpu().tolist()

            for local_idx, (raw, prompt, output) in enumerate(
                zip(raw_batch, prompt_batch, decoded_outputs)
            ):
                row = {
                    "method": "vi_mod",
                    "local_idx": int(local_idx),
                    "raw_prompt": raw,
                    "prompt": prompt,
                    "output": output,
                    "user_weights": user_weights,
                    "rho": float(args.rho),
                    "f_type": args.divergence_type,
                    "seed": int(args.seed),
                    "max_length": int(args.max_length),
                    "num_beams": int(args.num_beams),
                }
                if vi_weights is not None and local_idx < len(vi_weights):
                    row["vi_weights"] = vi_weights[local_idx]
                results.append(row)

    else:
        raise ValueError(f"Unknown prompt_source={args.prompt_source}")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[saved] generations: {args.output_file}", flush=True)
        print(f"[saved] num rows: {len(results)}", flush=True)

        if any("dataset_idx" in r for r in results):
            unique = len(set(r["dataset_idx"] for r in results if "dataset_idx" in r))
            print(f"[saved] num unique prompts: {unique}", flush=True)

    else:
        for i, row in enumerate(results):
            print("=" * 80)
            print(f"PROMPT {i}: {row['prompt']}")
            print("-" * 80)
            print(row["output"])


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train_calibrator", "generate"], required=True)

    # model/adapters
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_paths", required=True, help="Comma-separated LoRA adapter paths for objectives, in MOD order.")
    p.add_argument("--torch_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--divergence_type", default="reverse_kl")
    p.add_argument("--prompt_template", default="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:")

    # calibration data/training
    p.add_argument("--calibration_jsonl")
    p.add_argument("--output_dir", default="./vi_mod_out")
    p.add_argument("--calibrator_path", default="./vi_mod_out/calibrator.pt")
    p.add_argument("--feature_state_path", default="./vi_mod_out/features.pt")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--logprob_batch_size", type=int, default=1)
    p.add_argument("--no_length_normalize", action="store_true")
    p.add_argument("--normalize_margins", action="store_true")

    # LLM prompt features
    p.add_argument("--feature_dim", type=int, default=128)
    p.add_argument("--feature_seed", type=int, default=42)
    p.add_argument("--feature_batch_size", type=int, default=4)
    p.add_argument("--feature_max_length", type=int, default=512)
    p.add_argument("--feature_pooling", choices=["mean", "last"], default="mean")
    p.add_argument("--no_feature_normalize", action="store_true")

    # VI hyperparams
    p.add_argument("--m0", type=float, default=0.0)
    p.add_argument("--s0", type=float, default=1.0)
    p.add_argument("--sigma_u", type=float, default=0.5)
    p.add_argument("--sigma_b", type=float, default=0.1)
    p.add_argument("--vi_steps", type=int, default=2000)
    p.add_argument("--vi_lr", type=float, default=1e-2)
    p.add_argument("--vi_mc_samples", type=int, default=1)
    p.add_argument("--vi_device", default="cuda")
    p.add_argument("--print_every", type=int, default=100)

    # generation common
    p.add_argument("--prompt_source", choices=["mod_eval", "file"], default="mod_eval")
    p.add_argument("--user_weights", default="0.5,0.5")
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--num_vi_samples", type=int, default=32)
    p.add_argument("--output_file")
    p.add_argument("--generation_batch_size", type=int, default=4)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_generations", action="store_true")
    p.add_argument("--print_max_chars", type=int, default=1500)

    # file prompt mode
    p.add_argument("--prompt")
    p.add_argument("--prompts_file")
    p.add_argument("--prompts_are_templated", action="store_true")

    # MOD-like prompt sampler mode
    p.add_argument("--dataset_name", default="PKU-Alignment/PKU-SafeRLHF-10K-better")
    p.add_argument("--dataset_caching", action="store_true")
    p.add_argument("--sanity_check", default=False)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--num_proc", type=int, default=4)
    p.add_argument("--mod_iters", type=int, default=50)

    return p


def main() -> None:
    args = build_parser().parse_args()

    if isinstance(args.sanity_check, str):
        args.sanity_check = args.sanity_check.lower() in {"true", "1", "yes"}

    if args.mode == "train_calibrator":
        if not args.calibration_jsonl:
            raise ValueError("--calibration_jsonl is required for train_calibrator mode.")
        train_calibrator(args)

    elif args.mode == "generate":
        generate(args)

    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()