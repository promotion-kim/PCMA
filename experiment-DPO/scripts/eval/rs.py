import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import random

import torch
import tyro
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from tqdm import tqdm

# Make project root importable regardless of current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.configs import DEFAULT_PROMPT_TEMPLATE
from src.utils import print_local_main, disable_progress_bar_non_local_main, prepare_model_for_peft, param_sharding_enabled, set_seed

os.environ["WANDB_MODE"] = "dryrun"
disable_progress_bar_non_local_main()


@dataclass
class ScriptArguments:
    sft_model_name: str = field(default="PKU-Alignment/alpaca-7b-reproduced", metadata={"help": "the sft model name"})
    dpo_model_1_name: str = field(default="/ext_hdd/sjkim/mod/dpo/dpo-better/best_checkpoint",metadata={"help": "the dpo model 1 name"})
    dpo_model_2_name: str = field(default="/ext_hdd/sjkim/mod/dpo/dpo-safer/best_checkpoint",metadata={"help": "the dpo model 2 name"})
    dpo_model_3_name: Optional[str] = field(default=None, metadata={"help": "the dpo model 3 name"})

    weight_1: float = field(default=0.5, metadata={"help": "the weight for dpo model 1"})
    weight_2: float = field(default=0.5, metadata={"help": "the weight for dpo model 2"})
    weight_3: float = field(default=0.0, metadata={"help": "the weight for dpo model 3"})

    num_beams: int = field(default=1, metadata={"help": "the number of beams"})
    seed: int = field(default=42, metadata={"help": "the seed"})
    f_type: str = field(default="reverse_kl")

    use_flash_attention_2: Optional[bool] = field(default=True, metadata={"help": "whether to use flash attention 2"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE, metadata={"help": "prompt template"})

    dataset_name: str = field(default="PKU-Alignment/PKU-SafeRLHF-10K-better", metadata={"help": "HF dataset or project alias"})
    split: str = field(default="validation", metadata={"help": "train/validation/test"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "number of prompts; None means all"})

    dataset_caching: Optional[bool] = field(default=False, metadata={"help": "used cached dataset"})

    beta: Optional[float] = field(default=0.1, metadata={"help": "beta for kl control"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "max input+output length"})
    max_new_tokens: Optional[int] = field(default=512, metadata={"help": "max generated tokens"})
    batch_size: int = field(default=4, metadata={"help": "generation batch size"})

    training_args: Optional[TrainingArguments] = field(default=None, init=False)

    peft: Optional[bool] = field(default=True, metadata={"help": "whether to use peft"})
    peft_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    )


def resolve_hf_dataset_name(dataset_name: str) -> str:
    alias = dataset_name.strip()
    alias_lower = alias.lower()
    # Convenience aliases used in local scripts.
    if alias_lower in {"pku", "beavertail"}:
        alias = "PKU-Alignment/PKU-SafeRLHF-10K-better"
    elif alias_lower in {"pku-safer"}:
        alias = "PKU-Alignment/PKU-SafeRLHF-10K-safer"
    elif alias_lower in {"pku-10k", "pku-saferlhf-10k"}:
        alias = "PKU-Alignment/PKU-SafeRLHF-10K"

    for suffix in ["-better", "-safer"]:
        if alias.endswith(suffix):
            return alias[: -len(suffix)]
    return alias


def load_pku_split_like_modpo(dataset_name: str, split: str):
    hf_dataset_name = resolve_hf_dataset_name(dataset_name)
    split = split.lower()

    if hf_dataset_name == "PKU-Alignment/PKU-SafeRLHF-10K":
        raw_train = load_dataset(hf_dataset_name, split="train")
        split_ds = raw_train.train_test_split(test_size=0.1, seed=0)
        if split == "train":
            return split_ds["train"]
        if split in {"validation", "valid", "val", "eval", "dev"}:
            return split_ds["test"]
        raise NotImplementedError("PKU-SafeRLHF-10K supports train/validation only in this script")

    if hf_dataset_name == "PKU-Alignment/PKU-SafeRLHF":
        if split == "train":
            raw_train = load_dataset(hf_dataset_name, split="train")
            return raw_train.train_test_split(test_size=0.1, seed=0)["train"]
        if split in {"validation", "valid", "val", "eval", "dev"}:
            raw_train = load_dataset(hf_dataset_name, split="train")
            return raw_train.train_test_split(test_size=0.1, seed=0)["test"]
        if split == "test":
            return load_dataset(hf_dataset_name, split="test")
        raise NotImplementedError(f"Unsupported split={split}")

    return load_dataset(hf_dataset_name, split=split)


def extract_raw_prompt(example: Dict) -> Optional[str]:
    for k in ["raw_prompt", "prompt", "input", "question", "instruction"]:
        if k in example and example[k] is not None:
            return str(example[k])
    return None


def format_model_input(raw_prompt: str, prompt_template: str) -> str:
    if "BEGINNING OF CONVERSATION:" in raw_prompt and "ASSISTANT:" in raw_prompt:
        return raw_prompt
    if prompt_template is not None and "{raw_prompt}" in prompt_template:
        return prompt_template.format(raw_prompt=raw_prompt)
    return raw_prompt


def load_eval_prompts(dataset_name: str, split: str, prompt_template: str, max_eval_samples: Optional[int]):
    rows = []
    dataset = load_pku_split_like_modpo(dataset_name, split)
    seen = set()

    for ex in dataset:
        raw_prompt = extract_raw_prompt(ex)
        if raw_prompt is None or raw_prompt in seen:
            continue
        seen.add(raw_prompt)
        rows.append(format_model_input(raw_prompt, prompt_template))
        if max_eval_samples is not None and len(rows) >= max_eval_samples:
            break

    if len(rows) == 0:
        raise RuntimeError(f"No prompts loaded from dataset={dataset_name}, split={split}")
    return rows


def chunked(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


script_args = tyro.cli(ScriptArguments)
script_args.training_args = TrainingArguments(output_dir="./output/dev/dpo", remove_unused_columns=False)
if not script_args.peft:
    script_args.peft_config = None

set_seed(script_args.seed)

if not script_args.dataset_caching:
    from datasets import disable_caching
    disable_caching()

print_local_main("loading model...")
model_load_kwargs = {
    "torch_dtype": torch.bfloat16,
    **({"device_map": "auto"} if not param_sharding_enabled() else {}),
}
if script_args.use_flash_attention_2:
    model_load_kwargs["attn_implementation"] = "flash_attention_2"

try:
    sft_model = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_name,
        **model_load_kwargs,
    )
except ImportError as e:
    # Keep environment untouched: fall back to default attention if flash-attn is unavailable.
    if script_args.use_flash_attention_2 or "FlashAttention2" in str(e) or "flash_attn" in str(e):
        print_local_main("flash-attn unavailable; falling back to default attention implementation.")
        model_load_kwargs.pop("attn_implementation", None)
        sft_model = AutoModelForCausalLM.from_pretrained(
            script_args.sft_model_name,
            **model_load_kwargs,
        )
    else:
        raise
sft_model.config.update({"use_cache": True, "pad_token_id": sft_model.config.eos_token_id})
sft_model = prepare_model_for_peft(sft_model, peft_config=script_args.peft_config, args=script_args.training_args)

sft_model.load_adapter(script_args.dpo_model_1_name, "adapter_rs")
sft_model.load_adapter(script_args.dpo_model_2_name, "adapter_aux_1")
if script_args.dpo_model_3_name is not None and script_args.weight_3 > 0:
    sft_model.load_adapter(script_args.dpo_model_3_name, "adapter_aux_2")

new_state_dict = {}
sd = sft_model.state_dict()
for key in sd:
    if "adapter_rs" not in key:
        continue
    if script_args.dpo_model_3_name is not None and script_args.weight_3 > 0:
        new_state_dict[key] = (
            script_args.weight_1 * sd[key]
            + script_args.weight_2 * sd[key.replace("adapter_rs", "adapter_aux_1")]
            + script_args.weight_3 * sd[key.replace("adapter_rs", "adapter_aux_2")]
        )
    else:
        new_state_dict[key] = (
            script_args.weight_1 * sd[key]
            + script_args.weight_2 * sd[key.replace("adapter_rs", "adapter_aux_1")]
        )
sft_model.load_state_dict(new_state_dict, strict=False)
sft_model.set_adapter("adapter_rs")
sft_model.eval()

tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

prompts = load_eval_prompts(
    dataset_name=script_args.dataset_name,
    split=script_args.split,
    prompt_template=script_args.prompt_template,
    max_eval_samples=script_args.max_eval_samples,
)
print_local_main(f"loaded prompts: {len(prompts)}")

results = []
for prompt_batch in tqdm(list(chunked(prompts, script_args.batch_size))):
    inputs = tokenizer(
        prompt_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=script_args.max_length,
    )
    inputs = {k: v.to(sft_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sft_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=script_args.max_new_tokens,
            do_sample=False,
            num_beams=script_args.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    for i in range(outputs.size(0)):
        results.append(tokenizer.decode(outputs[i], skip_special_tokens=True))

output_prefix = "" if script_args.num_beams == 1 else f"{script_args.num_beams}_"
file_path = f"results_beavertail/outputs/{output_prefix}rs_output_{script_args.weight_1}_{script_args.weight_2}_{script_args.f_type}.txt"
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, "w", encoding="utf-8") as f:
    for result in results:
        f.write("\nPrompt and response\n")
        f.write(result)
        f.write("\n")

print_local_main(f"saved: {file_path}")
