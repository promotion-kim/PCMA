import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import tyro
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.trainer.dpo_trainer import f_DPOTrainer
from src.trainer.modpo_trainer import MODPOTrainer
from src.utils import (
    print_local_main,
    disable_progress_bar_non_local_main,
    prepare_model_for_peft,
    param_sharding_enabled,
    PeftAsPreTrained,
    RewardWrapperList,
    ImplicitRewardWrapper,
    set_seed,
)

disable_progress_bar_non_local_main()

OBJECTIVES = ["helpful", "harmless", "humor"]
OBJECTIVE_TO_ID_FIELD = {
    "helpful": "better_response_id",
    "harmless": "safer_response_id",
    "humor": "funnier_response_id",
}


def parse_bool_like(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in ["1", "true", "yes", "y"]
    return bool(x)


def load_json_or_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    if text[0] == "[":
        return json.loads(text)

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def hh_prompt_to_messages(prompt_hh: str) -> List[Dict[str, str]]:
    """Convert Anthropic HH prompt to chat messages."""
    text = prompt_hh.strip()
    if text.endswith("Assistant:"):
        text = text[: -len("Assistant:")].strip()

    messages = []
    for part in text.split("\n\n"):
        part = part.strip()
        if not part:
            continue
        if part.startswith("Human:"):
            messages.append({"role": "user", "content": part[len("Human:") :].strip()})
        elif part.startswith("Assistant:"):
            messages.append({"role": "assistant", "content": part[len("Assistant:") :].strip()})
    return messages


def format_prompt(raw_hh_prompt: str, tokenizer, prompt_format: str, prompt_template: str) -> str:
    if prompt_format == "hh":
        prompt = raw_hh_prompt
    elif prompt_format == "chat":
        messages = hh_prompt_to_messages(raw_hh_prompt)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        raise ValueError(f"Unknown prompt_format={prompt_format}. Use 'hh' or 'chat'.")

    # Existing MOD code expects a prompt string in the `prompt` column.
    # If prompt_template='{raw_prompt}', this is identity.
    return prompt_template.format(raw_prompt=prompt)


def get_chosen_rejected_from_parm_row(row: Dict, objective: str) -> Tuple[str, str]:
    id_field = OBJECTIVE_TO_ID_FIELD[objective]
    chosen_id = int(row[id_field])
    if chosen_id not in [0, 1]:
        raise ValueError(f"Invalid {id_field}={chosen_id}; expected 0 or 1.")

    rejected_id = 1 - chosen_id
    chosen = row[f"response_{chosen_id}"]
    rejected = row[f"response_{rejected_id}"]
    return chosen, rejected


def make_parm_preference_dataset(
    path: str,
    objective: str,
    tokenizer,
    prompt_format: str,
    prompt_template: str,
    sanity_check: bool = False,
    skip_same_response: bool = True,
) -> Dataset:
    rows = load_json_or_jsonl(path)
    out = []
    skipped = 0

    for idx, row in enumerate(rows):
        try:
            prompt = format_prompt(
                raw_hh_prompt=row["prompt"],
                tokenizer=tokenizer,
                prompt_format=prompt_format,
                prompt_template=prompt_template,
            )
            chosen, rejected = get_chosen_rejected_from_parm_row(row, objective)
        except Exception as e:
            skipped += 1
            continue

        if skip_same_response and chosen.strip() == rejected.strip():
            skipped += 1
            continue

        ex = {
            # DPODataMapFunc uses `prompt`, `chosen`, `rejected`.
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            # MODPODataMapFunc additionally keeps these raw strings.
            "raw_prompt": prompt,
            "source_idx": idx,
            "objective": objective,
        }

        # Keep scores for debugging; they are removed by map preprocessing later.
        for k in [
            "help_score_0",
            "help_score_1",
            "harm_score_0",
            "harm_score_1",
            "humor_score_0",
            "humor_score_1",
            "better_response_id",
            "safer_response_id",
            "funnier_response_id",
        ]:
            if k in row:
                ex[k] = row[k]

        out.append(ex)

    if sanity_check:
        out = out[: min(len(out), 128)]

    print_local_main(
        f"Loaded PARM dataset: path={path}, objective={objective}, "
        f"num_rows={len(out)}, skipped={skipped}"
    )
    return Dataset.from_list(out)


def objective_weight_dict(args) -> Dict[str, float]:
    return {
        "helpful": float(args.w_helpful),
        "harmless": float(args.w_harmless),
        "humor": float(args.w_humor),
    }


def select_primary_objective(args) -> str:
    if args.train_objective == "auto":
        w = objective_weight_dict(args)
        positive = {k: v for k, v in w.items() if v > 0}
        if not positive:
            raise ValueError("At least one objective weight must be positive.")
        return max(positive, key=positive.get)

    if args.train_objective not in OBJECTIVES:
        raise ValueError(f"train_objective must be one of {OBJECTIVES} or auto.")
    return args.train_objective


@dataclass
class ScriptArguments:
    # Mode
    mode: str = field(default="modpo", metadata={"help": "dpo or modpo"})

    # Base / reference policy. For pilot, use meta-llama/Llama-3.1-8B-Instruct.
    sft_model_name: str = field(default="meta-llama/Llama-3.1-8B-Instruct")
    use_flash_attention_2: Optional[bool] = field(default=False)

    # PARM data paths. train/dev/test are JSON lists or JSONL.
    parm_train_path: str = field(default="/ext_hdd/sjkim/mod/data/parm/train.json")
    parm_eval_path: str = field(default="/ext_hdd/sjkim/mod/data/parm/dev.json")

    # Prompt formatting.
    # hh   : keep Anthropic HH prompt exactly.
    # chat : convert HH turns to the base model's chat template.
    prompt_format: str = field(default="chat")
    prompt_template: str = field(default="{raw_prompt}")

    # DPO objective or MODPO primary objective.
    # For MODPO, `auto` chooses the objective with the largest positive weight.
    train_objective: str = field(default="auto")

    # Objective-specific DPO adapters used as implicit margin reward models for MODPO.
    helpful_adapter_name: Optional[str] = field(default=None)
    harmless_adapter_name: Optional[str] = field(default=None)
    humor_adapter_name: Optional[str] = field(default=None)

    # 3-objective weights.
    w_helpful: float = field(default=1.0 / 3.0)
    w_harmless: float = field(default=1.0 / 3.0)
    w_humor: float = field(default=1.0 / 3.0)

    sanity_check: Optional[bool] = field(default=False)
    seed: int = field(default=42)
    beta: Optional[float] = field(default=0.1)
    max_length: Optional[int] = field(default=1024)
    num_proc: Optional[int] = field(default=4)
    generate_during_eval: Optional[bool] = field(default=False)
    skip_same_response: Optional[bool] = field(default=True)

    # Used only by mode=dpo, because the repo's f_DPOTrainer supports variants.
    divergence_type: str = field(default="reverse_kl")

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="./output/dev/modpo_hh",
            overwrite_output_dir=True,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=5e-4,
            lr_scheduler_type="cosine",
            warmup_steps=0.1,
            weight_decay=0.05,
            bf16=True,
            fp16=False,
            remove_unused_columns=False,
            run_name="dev_modpo_hh",
            report_to="wandb",
            num_train_epochs=3,
            logging_steps=10,
            save_steps=0.25,
            eval_steps=0.25,
            eval_delay=0.25,
            evaluation_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
        )
    )

    peft_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=64,
            lora_alpha=1,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
    )


def load_base_model_and_tokenizer(args):
    print_local_main(f"loading base/reference model: {args.sft_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_name,
        use_flash_attention_2=args.use_flash_attention_2,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        **({"device_map": {"": Accelerator().local_process_index}} if not param_sharding_enabled() else {}),
    )
    model.config.update({"use_cache": False, "pad_token_id": model.config.eos_token_id})

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def run_dpo(args):
    objective = select_primary_objective(args)
    if objective == "auto":
        raise ValueError("mode=dpo requires train_objective to be helpful/harmless/humor, not auto.")

    model, tokenizer = load_base_model_and_tokenizer(args)

    train_dataset = make_parm_preference_dataset(
        path=args.parm_train_path,
        objective=objective,
        tokenizer=tokenizer,
        prompt_format=args.prompt_format,
        prompt_template=args.prompt_template,
        sanity_check=parse_bool_like(args.sanity_check),
        skip_same_response=parse_bool_like(args.skip_same_response),
    )
    eval_dataset = make_parm_preference_dataset(
        path=args.parm_eval_path,
        objective=objective,
        tokenizer=tokenizer,
        prompt_format=args.prompt_format,
        prompt_template=args.prompt_template,
        sanity_check=parse_bool_like(args.sanity_check),
        skip_same_response=parse_bool_like(args.skip_same_response),
    )

    print_local_main(f"start DPO adapter training for objective={objective}")
    trainer = f_DPOTrainer(
        divergence_type=args.divergence_type,
        model=model,
        ref_model=model,
        beta=args.beta,
        args=args.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=args.peft_config,
        max_length=args.max_length,
        num_proc=args.num_proc,
        generate_during_eval=parse_bool_like(args.generate_during_eval),
    )

    if Accelerator().is_local_main_process and args.peft_config:
        trainer.model.print_trainable_parameters()

    trainer.train()
    save_name = "best_checkpoint" if args.training_args.load_best_model_at_end else "final_checkpoint"
    save_path = os.path.join(args.training_args.output_dir, save_name)
    trainer.model.save_pretrained(save_path)
    trainer.tokenizer.save_pretrained(save_path)
    print_local_main(f"saved DPO adapter to {save_path}")


def run_modpo(args):
    primary = select_primary_objective(args)
    w_dict = objective_weight_dict(args)

    if w_dict[primary] <= 0:
        raise ValueError(
            f"Primary objective weight must be positive because MODPOTrainer divides by w[0]. "
            f"primary={primary}, weight={w_dict[primary]}"
        )

    adapter_paths = {
        "helpful": args.helpful_adapter_name,
        "harmless": args.harmless_adapter_name,
        "humor": args.humor_adapter_name,
    }
    for obj, path in adapter_paths.items():
        if obj != primary and (path is None or not path):
            raise ValueError(f"Missing adapter path for margin objective={obj}.")

    margin_objectives = [obj for obj in OBJECTIVES if obj != primary]
    ordered_weights = [w_dict[primary]] + [w_dict[obj] for obj in margin_objectives]

    print_local_main(f"MODPO primary objective: {primary}")
    print_local_main(f"MODPO margin objectives: {margin_objectives}")
    print_local_main(f"MODPO ordered weights: {ordered_weights}")

    sft_model, tokenizer = load_base_model_and_tokenizer(args)

    # Main trainable policy adapter.
    model = prepare_model_for_peft(sft_model, peft_config=args.peft_config, args=args.training_args)

    # Load only margin adapters. The primary objective is represented by the hard preference dataset.
    for obj in margin_objectives:
        adapter_name = f"margin_{obj}"
        print_local_main(f"loading margin adapter {adapter_name}: {adapter_paths[obj]}")
        try:
            model.load_adapter(adapter_paths[obj], adapter_name=adapter_name, is_trainable=False)
        except TypeError:
            model.load_adapter(adapter_paths[obj], adapter_name=adapter_name)

    try:
        model.set_adapter("default")
    except Exception:
        pass

    train_dataset = make_parm_preference_dataset(
        path=args.parm_train_path,
        objective=primary,
        tokenizer=tokenizer,
        prompt_format=args.prompt_format,
        prompt_template=args.prompt_template,
        sanity_check=parse_bool_like(args.sanity_check),
        skip_same_response=parse_bool_like(args.skip_same_response),
    )
    eval_dataset = make_parm_preference_dataset(
        path=args.parm_eval_path,
        objective=primary,
        tokenizer=tokenizer,
        prompt_format=args.prompt_format,
        prompt_template=args.prompt_template,
        sanity_check=parse_bool_like(args.sanity_check),
        skip_same_response=parse_bool_like(args.skip_same_response),
    )

    print_local_main("start MODPO training")
    trainer = MODPOTrainer(
        model=model,
        beta=args.beta,
        args=args.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_proc=args.num_proc,
        generate_during_eval=parse_bool_like(args.generate_during_eval),
    )

    if Accelerator().is_local_main_process:
        trainer.model.print_trainable_parameters()

    reward_wrappers = []
    for obj in margin_objectives:
        adapter_name = f"margin_{obj}"
        reward_wrappers.append(
            ImplicitRewardWrapper(
                model=PeftAsPreTrained(trainer.model, adapter_name),
                ref_model=PeftAsPreTrained(trainer.model),
                tokenizer=tokenizer,
                beta=args.beta,
                prompt_template=args.prompt_template,
            )
        )

    trainer.set_wrapped_margin_reward_model_list(
        RewardWrapperList(reward_wrappers),
        w=ordered_weights,
        prepare=False,
    )

    trainer.train()
    save_name = "best_checkpoint" if args.training_args.load_best_model_at_end else "final_checkpoint"
    save_path = os.path.join(args.training_args.output_dir, save_name)
    trainer.model.save_pretrained(save_path)
    trainer.tokenizer.save_pretrained(save_path)
    print_local_main(f"saved MODPO adapter to {save_path}")


def main():
    args = tyro.cli(ScriptArguments)
    set_seed(args.seed)

    mode = args.mode.lower()
    if mode == "dpo":
        run_dpo(args)
    elif mode == "modpo":
        run_modpo(args)
    else:
        raise ValueError("mode must be either 'dpo' or 'modpo'.")


if __name__ == "__main__":
    main()
