import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch

# ---------------------------------------------------------------------
# Compatibility patch:
# Older TRL imports `top_k_top_p_filtering` from the top-level
# `transformers` package. Newer transformers versions removed that
# top-level symbol. We restore it before importing TRL-dependent code.
# This does NOT modify installed packages or dependencies.
# ---------------------------------------------------------------------
def _patch_transformers_top_k_top_p_filtering():
    import transformers

    if hasattr(transformers, "top_k_top_p_filtering"):
        return

    def top_k_top_p_filtering(
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        """
        Backward-compatible implementation used by older TRL versions.
        Adapted from the historical Hugging Face filtering behavior.

        Args:
            logits: Tensor of shape (..., vocab_size)
            top_k: Keep only top-k tokens with highest probability.
            top_p: Keep the smallest token set whose cumulative prob >= top_p.
            filter_value: Value assigned to filtered logits.
            min_tokens_to_keep: Always keep at least this many tokens.
        """
        if top_k is not None and top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
            kth_values = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            indices_to_remove = logits < kth_values
            logits = logits.masked_fill(indices_to_remove, filter_value)

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p

            if min_tokens_to_keep > 1:
                sorted_indices_to_remove[..., :min_tokens_to_keep] = False

            # Shift right so that the first token above threshold is kept.
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
            indices_to_remove.scatter_(
                dim=-1,
                index=sorted_indices,
                src=sorted_indices_to_remove,
            )

            logits = logits.masked_fill(indices_to_remove, filter_value)

        return logits

    transformers.top_k_top_p_filtering = top_k_top_p_filtering


_patch_transformers_top_k_top_p_filtering()

def _patch_llama3_rope_scaling_compat():
    """
    Backward-compat for older transformers that expect:
      rope_scaling = {"type": ..., "factor": ...}
    but Llama-3 configs may include extra keys and `rope_type`.
    """
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
    except Exception:
        return

    original_validation = LlamaConfig._rope_scaling_validation

    def _patched_validation(self):
        rope = getattr(self, "rope_scaling", None)
        if isinstance(rope, dict) and "type" not in rope and "rope_type" in rope and "factor" in rope:
            rope_type = rope.get("rope_type")
            mapped_type = "dynamic" if rope_type == "llama3" else rope_type
            self.rope_scaling = {"type": mapped_type, "factor": float(rope["factor"])}
        return original_validation(self)

    LlamaConfig._rope_scaling_validation = _patched_validation


_patch_llama3_rope_scaling_compat()

import tyro
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.trainer.dpo_trainer import f_DPOTrainer
from src.utils import (
    print_local_main,
    disable_progress_bar_non_local_main,
    param_sharding_enabled,
)

disable_progress_bar_non_local_main()


OBJECTIVE_TO_ID_KEY = {
    "helpful": "better_response_id",
    "harmless": "safer_response_id",
    "humor": "funnier_response_id",
}


def load_json_or_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        raise ValueError(f"Expected a JSON list in {path}, got {type(obj)}")

    raise ValueError(f"Unsupported file extension: {path}")


def hh_prompt_to_messages(prompt_hh: str) -> List[Dict[str, str]]:
    """
    Convert Anthropic HH style prompt into chat messages.

    Input example:
        \\n\\nHuman: hi\\n\\nAssistant: hello\\n\\nHuman: question\\n\\nAssistant:

    Output:
        [
          {"role": "user", "content": "hi"},
          {"role": "assistant", "content": "hello"},
          {"role": "user", "content": "question"}
        ]
    """
    text = prompt_hh.strip()

    if text.endswith("Assistant:"):
        text = text[: -len("Assistant:")].strip()

    parts = text.split("\n\n")
    messages = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("Human:"):
            messages.append(
                {
                    "role": "user",
                    "content": part[len("Human:"):].strip(),
                }
            )
        elif part.startswith("Assistant:"):
            messages.append(
                {
                    "role": "assistant",
                    "content": part[len("Assistant:"):].strip(),
                }
            )

    return messages


def format_prompt(prompt: str, tokenizer: AutoTokenizer, prompt_format: str) -> str:
    """
    prompt_format:
      - hh:   keep original HH prompt.
      - chat: convert HH prompt to model-specific chat template.
    """
    if prompt_format == "hh":
        return prompt

    if prompt_format == "chat":
        messages = hh_prompt_to_messages(prompt)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    raise ValueError(f"Unknown prompt_format: {prompt_format}")


def convert_parm_rows_to_dpo_dataset(
    rows: List[Dict],
    objective: str,
    tokenizer: AutoTokenizer,
    prompt_format: str,
    sanity_check: bool = False,
) -> Dataset:
    if objective not in OBJECTIVE_TO_ID_KEY:
        raise ValueError(
            f"objective must be one of {list(OBJECTIVE_TO_ID_KEY.keys())}, got {objective}"
        )

    id_key = OBJECTIVE_TO_ID_KEY[objective]
    converted = []

    for idx, row in enumerate(rows):
        if "prompt" not in row:
            continue
        if "response_0" not in row or "response_1" not in row:
            continue
        if id_key not in row:
            continue

        chosen_id = int(row[id_key])
        if chosen_id not in [0, 1]:
            continue

        rejected_id = 1 - chosen_id

        raw_prompt = format_prompt(
            prompt=row["prompt"],
            tokenizer=tokenizer,
            prompt_format=prompt_format,
        )

        chosen = row[f"response_{chosen_id}"]
        rejected = row[f"response_{rejected_id}"]

        if not isinstance(chosen, str) or not isinstance(rejected, str):
            continue
        if len(chosen.strip()) == 0 or len(rejected.strip()) == 0:
            continue

        example = {
            # Keep both keys for compatibility with possible trainer/data-map assumptions.
            "prompt": raw_prompt,
            "raw_prompt": raw_prompt,
            "chosen": chosen.strip(),
            "rejected": rejected.strip(),
            "objective": objective,
            "source_idx": idx,
            "chosen_id": chosen_id,
            "rejected_id": rejected_id,
        }

        # Optional debug fields, harmless if trainer removes unused columns.
        for key in [
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
            if key in row:
                example[key] = row[key]

        converted.append(example)

    if sanity_check:
        converted = converted[: min(len(converted), 128)]

    if len(converted) == 0:
        raise ValueError(
            f"No examples were converted for objective={objective}. "
            f"Check data format and id key={id_key}."
        )

    print_local_main(
        f"[dataset] objective={objective}, prompt_format={prompt_format}, n={len(converted)}"
    )

    return Dataset.from_list(converted)


def get_split_path(data_dir: str, split: str) -> str:
    """
    Expected files:
      data/train.json
      data/dev.json
      data/test.json

    For eval, prefer dev.json. If not found, fall back to test.json.
    """
    if split == "train":
        candidates = ["train.json", "train.jsonl"]
    elif split in ["validation", "eval", "dev"]:
        candidates = ["dev.json", "dev.jsonl", "validation.json", "validation.jsonl", "test.json", "test.jsonl"]
    elif split == "test":
        candidates = ["test.json", "test.jsonl"]
    else:
        raise ValueError(f"Unknown split: {split}")

    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not find split={split} in {data_dir}. Tried: {candidates}"
    )


@dataclass
class ScriptArguments:
    divergence_type: str = field(default="reverse_kl", metadata={"help": "DPO divergence type"})
    sft_model_name: str = field(default="meta-llama/Llama-3.1-8B-Instruct")
    data_dir: str = field(default="data", metadata={"help": "Directory containing train/dev/test JSON files"})
    objective: str = field(default="helpful", metadata={"help": "helpful, harmless, or humor"})

    prompt_format: str = field(
        default="chat",
        metadata={"help": "chat for Llama/Qwen instruct template, hh for raw HH prompt"},
    )

    use_flash_attention_2: Optional[bool] = field(default=False)
    dataset_caching: Optional[bool] = field(default=False)
    sanity_check: Optional[bool] = field(default=False)

    beta: Optional[float] = field(default=0.1)
    max_length: Optional[int] = field(default=1024)
    num_proc: Optional[int] = field(default=4)
    generate_during_eval: Optional[bool] = field(default=True)

    @dataclass
    class _TrainingArgsConfig:
        output_dir: str = "./output/dev/dpo_hh"
        overwrite_output_dir: bool = True
        per_device_train_batch_size: int = 1
        per_device_eval_batch_size: int = 1
        gradient_accumulation_steps: int = 2
        learning_rate: float = 5e-4
        lr_scheduler_type: str = "cosine"
        warmup_ratio: float = 0.1
        weight_decay: float = 0.05
        bf16: bool = True
        fp16: bool = False
        remove_unused_columns: bool = False
        run_name: str = "dev_dpo_hh"
        report_to: str = "wandb"
        num_train_epochs: float = 3.0
        logging_steps: int = 10
        save_steps: float = 0.25
        eval_steps: float = 0.25
        eval_delay: float = 0.25
        evaluation_strategy: str = "steps"
        save_total_limit: int = 3
        load_best_model_at_end: bool = True

    training_args: _TrainingArgsConfig = field(
        default_factory=lambda: ScriptArguments._TrainingArgsConfig(
            output_dir="./output/dev/dpo_hh",
            overwrite_output_dir=True,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=5e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            weight_decay=0.05,
            bf16=True,
            fp16=False,
            remove_unused_columns=False,
            run_name="dev_dpo_hh",
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

    peft: Optional[bool] = field(default=True)
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


script_args = tyro.cli(ScriptArguments)
script_args.training_args = TrainingArguments(**asdict(script_args.training_args))

if not script_args.peft:
    script_args.peft_config = None

if script_args.objective not in OBJECTIVE_TO_ID_KEY:
    raise ValueError(
        f"--objective must be one of {list(OBJECTIVE_TO_ID_KEY.keys())}, "
        f"got {script_args.objective}"
    )

if not script_args.dataset_caching:
    from datasets import disable_caching

    disable_caching()


# tokenizer first, because we may need chat template for local JSON data
print_local_main("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# dataset
train_path = get_split_path(script_args.data_dir, "train")
eval_path = get_split_path(script_args.data_dir, "validation")

print_local_main(f"loading train data from: {train_path}")
print_local_main(f"loading eval data from:  {eval_path}")

train_rows = load_json_or_jsonl(train_path)
eval_rows = load_json_or_jsonl(eval_path)

train_dataset = convert_parm_rows_to_dpo_dataset(
    rows=train_rows,
    objective=script_args.objective,
    tokenizer=tokenizer,
    prompt_format=script_args.prompt_format,
    sanity_check=script_args.sanity_check,
)

eval_dataset = convert_parm_rows_to_dpo_dataset(
    rows=eval_rows,
    objective=script_args.objective,
    tokenizer=tokenizer,
    prompt_format=script_args.prompt_format,
    sanity_check=script_args.sanity_check,
)


# base/reference model
print_local_main("loading model...")
sft_model = AutoModelForCausalLM.from_pretrained(
    script_args.sft_model_name,
    use_flash_attention_2=script_args.use_flash_attention_2,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    **({"device_map": {"": Accelerator().local_process_index}} if not param_sharding_enabled() else {}),
)

sft_model.config.update(
    {
        "use_cache": False,
        "pad_token_id": sft_model.config.eos_token_id,
    }
)

print_local_main(sft_model)
print_local_main(script_args.peft_config)


print_local_main("start DPO training...")
trainer = f_DPOTrainer(
    divergence_type=script_args.divergence_type,
    model=sft_model,
    ref_model=sft_model,
    beta=script_args.beta,
    args=script_args.training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    peft_config=script_args.peft_config,
    max_length=script_args.max_length,
    num_proc=script_args.num_proc,
    generate_during_eval=script_args.generate_during_eval,
)

if Accelerator().is_local_main_process and script_args.peft_config:
    trainer.model.print_trainable_parameters()

trainer.train()

save_name = "best_checkpoint" if script_args.training_args.load_best_model_at_end else "final_checkpoint"
save_path = os.path.join(script_args.training_args.output_dir, save_name)

trainer.model.save_pretrained(save_path)
trainer.tokenizer.save_pretrained(save_path)

print_local_main(f"saved checkpoint to: {save_path}")
