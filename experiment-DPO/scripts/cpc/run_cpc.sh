#!/usr/bin/env bash
set -euo pipefail

# Fair CPC-MODPO run script.
# This mirrors the attached MODPO run.sh as closely as possible, while keeping
# the CPC coefficient c_i = w_i exp(gamma a_i).

export PYTHONPATH=.:scripts/cpc
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Match MODPO W&B settings.
export WANDB_ENTITY=${WANDB_ENTITY:-promotion-kim}
export WANDB_PROJECT=${WANDB_PROJECT:-pcma}
export WANDB_MODE=${WANDB_MODE:-online}

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/ext_hdd/sjkim/mod/triton_cache}
mkdir -p "$TRITON_CACHE_DIR"

LAUNCH="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=2 --main_process_port 29501"

sft_model_name="PKU-Alignment/alpaca-7b-reproduced"
prompt_template="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"
dataset_name="PKU-Alignment/PKU-SafeRLHF-10K"
sanity_check=False
output_dir="/ext_hdd/sjkim/mod"

# Match attached MODPO run.sh.
max_length=512
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=2
learning_rate=5e-4
seed=0

# CPC-specific.
w=${1:-0.5}
gamma=${2:-1.0}

# Existing objective-specific DPO adapter used as safety margin reward.
safer_adapter="${output_dir}/dpo/dpo-safer/best_checkpoint"

# Fitted CPC calibrator.
cpc_calibrator_path="${output_dir}/safe_rlhf/cpc/calibrator"

# Name includes gamma, but otherwise follows MODPO run naming.
lm_run_name="${dataset_name}/cpc/lm/(${w})*r_better+(1-${w})*r_safer/gamma_${gamma}"

PYTHONPATH=.:scripts/cpc $LAUNCH scripts/cpc/cpcmodpo.py \
    --sft_model_name "${sft_model_name}" \
    --margin_reward_model_name "${safer_adapter}" \
    --cpc_calibrator_path "${cpc_calibrator_path}" \
    --prompt_template "${prompt_template}" \
    --dataset_name "${dataset_name}-better" \
    --dataset_caching false \
    --sanity_check ${sanity_check} \
    --seed ${seed} \
    --w ${w} \
    --gamma ${gamma} \
    --beta 0.1 \
    --max_length ${max_length} \
    --generate_during_eval true \
    --output_dir "${output_dir}/${lm_run_name}" \
    --run_name "${lm_run_name}" \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --lr_scheduler_type cosine \
    --warmup_steps 0.1 \
    --weight_decay 0.05 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 0.25 \
    --eval_steps 0.25 \
    --eval_delay 0.25 \
    --evaluation_strategy steps \
    --save_total_limit 3 \
    --load_best_model_at_end true \
    --report_to wandb \
    --fp16 true \
    --bf16 false \
    --remove_unused_columns false \
    --lora_r 64 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --lora_alpha 1 \
    --lora_dropout 0 \
    --wandb_eval_generation_n 5 \
    --wandb_eval_generation_max_new_tokens 512 \
    --wandb_eval_generation_do_sample false
