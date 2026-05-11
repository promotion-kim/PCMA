#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export TRITON_CACHE_DIR=/ext_hdd/sjkim/mod/triton_cache
mkdir -p "$TRITON_CACHE_DIR"

# Make W&B write to your personal workspace/project.
export WANDB_ENTITY=${WANDB_ENTITY:-promotion-kim}
export WANDB_PROJECT=${WANDB_PROJECT:-pcma}
export WANDB_MODE=${WANDB_MODE:-online}

LAUNCH="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=2 --main_process_port 29501"

sft_model_name="PKU-Alignment/alpaca-7b-reproduced"
prompt_template="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"
dataset_name="PKU-Alignment/PKU-SafeRLHF-10K"
san_check=False
output_root="/ext_hdd/sjkim/mod"

max_length=512
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=2

# PC-MODPO는 scale이 튈 수 있으니 처음에는 1e-4 또는 5e-5도 권장
learning_rate=1e-4

seed=0
w=${1:-0.5}

# Alpha stabilization
alpha_min=0.05

# Debug / monitoring
debug_print_alpha_every=20
debug_print_alpha_n=3

# Eval generation for collapse inspection
wandb_eval_generation_n=5
wandb_eval_generation_max_new_tokens=512
wandb_eval_generation_do_sample=False

# Existing objective-specific DPO adapters.
better_adapter="${output_root}/dpo/dpo-better/best_checkpoint"
safer_adapter="${output_root}/dpo/dpo-safer/best_checkpoint"

# Used only when feature_source=hf_model.
# Since we use feature_source=sft_hidden, this is mostly ignored.
feature_model_name="sentence-transformers/all-MiniLM-L6-v2"

posterior_dir="${output_root}/${dataset_name}/pcma/posterior/better_safer"

# -----------------------------------------------------------------------------
# Stage 1: fit posterior calibration q(theta | D_better, D_safer)
# -----------------------------------------------------------------------------
#PYTHONPATH=. $LAUNCH scripts/pcma/fit_posterior.py \
#    --sft_model_name "${sft_model_name}" \
#    --objective_adapter_names "${better_adapter},${safer_adapter}" \
#    --objective_dataset_names "${dataset_name}-better,${dataset_name}-safer" \
#    --feature_model_name "${feature_model_name}" \
#    --prompt_template "${prompt_template}" \
#    --sanity_check ${san_check} \
#    --seed ${seed} \
#    --max_length ${max_length} \
#    --output_dir "${posterior_dir}" \
#    --feature_source sft_hidden \
#    --feature_pooling mean \
#    --per_device_batch_size 1 \
#    --posterior_steps 2000 \
#    --posterior_lr 3e-3 \
#    --posterior_batch_size 512 \
#    --posterior_log_every 50

# -----------------------------------------------------------------------------
# Stage 2: PC-MODPO on anchor objective D_better with safety as margin reward.
# -----------------------------------------------------------------------------
lm_run_name="${dataset_name}/pcmodpo/lm/pc_alpha_w${w}_better_safer"

PYTHONPATH=. $LAUNCH scripts/pcma/pcmodpo.py \
    --sft_model_name "${sft_model_name}" \
    --margin_reward_model_name "${safer_adapter}" \
    --posterior_calibrator_path "${posterior_dir}" \
    --feature_source sft_hidden \
    --feature_pooling mean \
    --feature_model_name "${feature_model_name}" \
    --prompt_template "${prompt_template}" \
    --dataset_name "${dataset_name}-better" \
    --sanity_check ${san_check} \
    --seed ${seed} \
    --w ${w} \
    --max_length ${max_length} \
    --posterior_num_samples 10 \
    --alpha_min ${alpha_min} \
    --debug_print_alpha_every ${debug_print_alpha_every} \
    --debug_print_alpha_n ${debug_print_alpha_n} \
    --wandb_eval_generation_n ${wandb_eval_generation_n} \
    --wandb_eval_generation_max_new_tokens ${wandb_eval_generation_max_new_tokens} \
    --wandb_eval_generation_do_sample \
    --training_args.output_dir "${output_root}/${lm_run_name}" \
    --training_args.run_name "${lm_run_name}" \
    --training_args.per_device_train_batch_size ${per_device_train_batch_size} \
    --training_args.per_device_eval_batch_size ${per_device_eval_batch_size} \
    --training_args.gradient_accumulation_steps ${gradient_accumulation_steps} \
    --training_args.learning_rate ${learning_rate} \
    --peft_config.r 64 \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha 1 \
    --peft_config.lora_dropout 0