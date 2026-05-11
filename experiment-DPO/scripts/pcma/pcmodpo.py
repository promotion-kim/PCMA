import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.trainer.pcmodpo_trainer import PCMODPOTrainer
from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
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
from src.utils.posterior_calibration import FrozenCausalLMPromptFeatureExtractor, HFPromptFeatureExtractor, LaplacePosteriorCalibrator


disable_progress_bar_non_local_main()


@dataclass
class ScriptArguments:
    sft_model_name: str = field(metadata={"help": "the sft/reference model name"})
    margin_reward_model_name: str = field(metadata={"help": "non-anchor objective adapter path, e.g. safer DPO adapter"})
    posterior_calibrator_path: str = field(metadata={"help": "directory containing posterior_calibrator.json"})
    feature_source: str = field(default="sft_hidden", metadata={"help": "sft_hidden or hf_model"})
    feature_model_name: Optional[str] = field(default=None, metadata={"help": "used only when feature_source=hf_model"})
    feature_pooling: str = field(default="mean", metadata={"help": "mean or last; used when feature_source=sft_hidden"})
    use_flash_attention_2: Optional[bool] = field(default=False)
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE)
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf")
    dataset_caching: Optional[bool] = field(default=False)
    sanity_check: Optional[bool] = field(default=False)
    seed: int = field(default=42)
    w: Optional[float] = field(default=0.5, metadata={"help": "anchor objective base weight; for two objectives alpha base is (w, 1-w)"})
    beta: Optional[float] = field(default=0.1)
    max_length: Optional[int] = field(default=1024)
    feature_max_length: Optional[int] = field(default=256)
    posterior_num_samples: Optional[int] = field(default=0, metadata={"help": "0 uses MAP plug-in alpha; >0 Monte Carlo averages Laplace samples"})
    anchor_objective_idx: Optional[int] = field(default=0)
    num_proc: Optional[int] = field(default=4)
    generate_during_eval: Optional[bool] = field(default=True)
    alpha_min: float = field(default=0.05)
    debug_print_alpha_every: int = field(default=50)
    debug_print_alpha_n: int = field(default=3)

    wandb_eval_generation_n: int = field(default=5)
    wandb_eval_generation_max_new_tokens: int = field(default=512)
    wandb_eval_generation_do_sample: bool = field(default=True)
    wandb_eval_generation_temperature: float = field(default=0.7)
    wandb_eval_generation_top_p: float = field(default=0.9)

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="./output/dev/pcmodpo",
            overwrite_output_dir=True,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=0.1,
            weight_decay=0.05,
            fp16=True,
            remove_unused_columns=False,
            run_name="dev_pcmodpo",
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
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    )


script_args = tyro.cli(ScriptArguments)
set_seed(script_args.seed)

print_local_main("loading posterior calibrator...")
posterior_calibrator = LaplacePosteriorCalibrator.from_pretrained(script_args.posterior_calibrator_path)
print_local_main("loading base model...")
sft_model = AutoModelForCausalLM.from_pretrained(
    script_args.sft_model_name,
    use_flash_attention_2=script_args.use_flash_attention_2,
    torch_dtype=torch.bfloat16,
    **({"device_map": {"": Accelerator().local_process_index}} if not param_sharding_enabled() else {}),
)
sft_model.config.update({"use_cache": False, "pad_token_id": sft_model.config.eos_token_id})

model = prepare_model_for_peft(sft_model, peft_config=script_args.peft_config, args=script_args.training_args)
model.load_adapter(script_args.margin_reward_model_name, adapter_name="margin_reward")

tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

if not script_args.dataset_caching:
    from datasets import disable_caching
    disable_caching()
rdp = DATASET_CONFIGS[script_args.dataset_name](
    prompt_template=script_args.prompt_template,
    sanity_check=script_args.sanity_check,
)
train_dataset = rdp.get_preference_dataset(split="train")
eval_dataset = rdp.get_preference_dataset(split="validation")

print_local_main("start PC-MODPO training...")
trainer = PCMODPOTrainer(
    model=model,
    beta=script_args.beta,
    args=script_args.training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=script_args.max_length,
    num_proc=script_args.num_proc,
    generate_during_eval=script_args.generate_during_eval,

    alpha_min=script_args.alpha_min,
    debug_print_alpha_every=script_args.debug_print_alpha_every,
    debug_print_alpha_n=script_args.debug_print_alpha_n,
    wandb_eval_generation_n=script_args.wandb_eval_generation_n,
    wandb_eval_generation_max_new_tokens=script_args.wandb_eval_generation_max_new_tokens,
    wandb_eval_generation_do_sample=script_args.wandb_eval_generation_do_sample,
    wandb_eval_generation_temperature=script_args.wandb_eval_generation_temperature,
    wandb_eval_generation_top_p=script_args.wandb_eval_generation_top_p,
    prompt_template=script_args.prompt_template,
)
if Accelerator().is_local_main_process:
    trainer.model.print_trainable_parameters()

print_local_main("loading prompt feature extractor...")
if script_args.feature_source == "sft_hidden":
    # Use the same frozen SFT/reference LM representation that was used during posterior fitting.
    # disable_adapter=True makes the feature come from the adapter-free reference model.
    feature_extractor = FrozenCausalLMPromptFeatureExtractor(
        model=trainer.model,
        tokenizer=tokenizer,
        max_length=script_args.feature_max_length,
        device=Accelerator().device,
        pooling=script_args.feature_pooling,
        disable_adapter=True,
        prompt_template=script_args.prompt_template,
    )
elif script_args.feature_source == "hf_model":
    feature_model_name = script_args.feature_model_name or posterior_calibrator.feature_model_name
    if feature_model_name is None:
        raise ValueError("feature_model_name must be provided when feature_source='hf_model'.")
    feature_extractor = HFPromptFeatureExtractor(
        feature_model_name,
        max_length=script_args.feature_max_length,
        device=f"cuda:{Accelerator().local_process_index}" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
else:
    raise ValueError("feature_source must be either 'sft_hidden' or 'hf_model'.")

trainer.set_wrapped_margin_reward_model_list(
    RewardWrapperList([
        ImplicitRewardWrapper(
            model=PeftAsPreTrained(trainer.model, "margin_reward"),
            ref_model=PeftAsPreTrained(trainer.model),
            tokenizer=tokenizer,
            beta=script_args.beta,
            prompt_template=script_args.prompt_template,
        )
    ]),
    w=(script_args.w, 1.0 - script_args.w),
    prepare=False,
)
trainer.set_posterior_calibrator(
    posterior_calibrator=posterior_calibrator,
    prompt_feature_extractor=feature_extractor,
    posterior_num_samples=script_args.posterior_num_samples,
    anchor_objective_idx=script_args.anchor_objective_idx,
)

trainer.train()

save_name = "best_checkpoint" if script_args.training_args.load_best_model_at_end else "final_checkpoint"
trainer.model.save_pretrained(os.path.join(script_args.training_args.output_dir, save_name))
trainer.tokenizer.save_pretrained(os.path.join(script_args.training_args.output_dir, save_name))
