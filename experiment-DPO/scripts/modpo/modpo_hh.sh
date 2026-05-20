#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   STAGE=adapters CUDA_VISIBLE_DEVICES=0,1 bash scripts/modpo/modpo_hh.sh
#   STAGE=modpo   CUDA_VISIBLE_DEVICES=0,1 bash scripts/modpo/modpo_hh.sh 0.33 0.33 0.34 auto
#   STAGE=all     CUDA_VISIBLE_DEVICES=0,1 bash scripts/modpo/modpo_hh.sh 0.60 0.20 0.20 auto

export WANDB_ENTITY=${WANDB_ENTITY:-promotion-kim}
export WANDB_PROJECT=${WANDB_PROJECT:-pcma}
export WANDB_MODE=${WANDB_MODE:-online}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/ext_hdd/sjkim/mod/triton_cache}
mkdir -p "$TRITON_CACHE_DIR"

PORT=${PORT:-29501}
NUM_PROCESSES=${NUM_PROCESSES:-2}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-scripts/accelerate_configs/multi_gpu.yaml}
LAUNCH="accelerate launch --config_file ${ACCELERATE_CONFIG} --num_processes=${NUM_PROCESSES} --main_process_port ${PORT}"

# Stage: adapters | modpo | all
STAGE=${STAGE:-modpo}

# Weight order: helpful harmless humor
w_helpful=${1:-0.33}
w_harmless=${2:-0.33}
w_humor=${3:-0.34}

# helpful | harmless | humor | auto
# auto chooses the largest positive weight as the MODPO primary objective.
train_objective=${4:-auto}

# Base/reference policy.
# You need HF access for Llama-3.1. If not available, replace with another 7~8B instruct model.
sft_model_name=${SFT_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}

# PARM dataset paths.
PARM_DIR=${PARM_DIR:-/home/sjkim/MOD/experiment-DPO/data}
parm_train_path=${PARM_TRAIN_PATH:-${PARM_DIR}/train.json}
parm_eval_path=${PARM_EVAL_PATH:-${PARM_DIR}/dev.json}

# Output root.
output_dir=${OUTPUT_DIR:-/ext_hdd/sjkim/mod}

# Prompt formatting.
# chat: convert HH prompt into Llama/Qwen chat template.
# hh:   keep Anthropic HH prompt string as-is.
prompt_format=${PROMPT_FORMAT:-chat}
prompt_template=${PROMPT_TEMPLATE:-'{raw_prompt}'}

max_length=${MAX_LENGTH:-1024}
per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
per_device_eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE:-1}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-2}
learning_rate=${LEARNING_RATE:-5e-4}
num_train_epochs=${NUM_TRAIN_EPOCHS:-3}
seed=${SEED:-42}
beta=${BETA:-0.1}
sanity_check=${SANITY_CHECK:-False}
use_flash_attention_2=${USE_FLASH_ATTENTION_2:-False}
generate_during_eval=${GENERATE_DURING_EVAL:-False}
divergence_type=${DIVERGENCE_TYPE:-reverse_kl}

script_path=${SCRIPT_PATH:-scripts/modpo/modpo_hh.py}

adapter_root=${ADAPTER_ROOT:-${output_dir}/dpo/parm_llama31_8b}
helpful_adapter=${HELPFUL_ADAPTER:-${adapter_root}/helpful/best_checkpoint}
harmless_adapter=${HARMLESS_ADAPTER:-${adapter_root}/harmless/best_checkpoint}
humor_adapter=${HUMOR_ADAPTER:-${adapter_root}/humor/best_checkpoint}

run_dpo_adapter() {
    local obj=$1
    local run_name="dpo/parm_llama31_8b/${obj}"

    echo "[stage=adapters] training objective adapter: ${obj}"
    PYTHONPATH=. ${LAUNCH} "${script_path}" \
        --mode dpo \
        --sft_model_name "${sft_model_name}" \
        --use_flash_attention_2 ${use_flash_attention_2} \
        --parm_train_path "${parm_train_path}" \
        --parm_eval_path "${parm_eval_path}" \
        --prompt_format "${prompt_format}" \
        --prompt_template "${prompt_template}" \
        --train_objective "${obj}" \
        --sanity_check ${sanity_check} \
        --seed ${seed} \
        --beta ${beta} \
        --divergence_type "${divergence_type}" \
        --max_length ${max_length} \
        --generate_during_eval ${generate_during_eval} \
        --training_args.output_dir "${output_dir}/${run_name}" \
        --training_args.run_name "${run_name}" \
        --training_args.per_device_train_batch_size ${per_device_train_batch_size} \
        --training_args.per_device_eval_batch_size ${per_device_eval_batch_size} \
        --training_args.gradient_accumulation_steps ${gradient_accumulation_steps} \
        --training_args.learning_rate ${learning_rate} \
        --training_args.num_train_epochs ${num_train_epochs} \
        --peft_config.r 64 \
        --peft_config.target_modules q_proj k_proj v_proj o_proj \
        --peft_config.lora_alpha 1 \
        --peft_config.lora_dropout 0
}

run_modpo() {
    local run_name="modpo/parm_llama31_8b/h${w_helpful}_s${w_harmless}_u${w_humor}/primary_${train_objective}"

    echo "[stage=modpo] weights: helpful=${w_helpful}, harmless=${w_harmless}, humor=${w_humor}, primary=${train_objective}"
    PYTHONPATH=. ${LAUNCH} "${script_path}" \
        --mode modpo \
        --sft_model_name "${sft_model_name}" \
        --use_flash_attention_2 ${use_flash_attention_2} \
        --parm_train_path "${parm_train_path}" \
        --parm_eval_path "${parm_eval_path}" \
        --prompt_format "${prompt_format}" \
        --prompt_template "${prompt_template}" \
        --train_objective "${train_objective}" \
        --helpful_adapter_name "${helpful_adapter}" \
        --harmless_adapter_name "${harmless_adapter}" \
        --humor_adapter_name "${humor_adapter}" \
        --w_helpful ${w_helpful} \
        --w_harmless ${w_harmless} \
        --w_humor ${w_humor} \
        --sanity_check ${sanity_check} \
        --seed ${seed} \
        --beta ${beta} \
        --max_length ${max_length} \
        --generate_during_eval ${generate_during_eval} \
        --training_args.output_dir "${output_dir}/${run_name}" \
        --training_args.run_name "${run_name}" \
        --training_args.per_device_train_batch_size ${per_device_train_batch_size} \
        --training_args.per_device_eval_batch_size ${per_device_eval_batch_size} \
        --training_args.gradient_accumulation_steps ${gradient_accumulation_steps} \
        --training_args.learning_rate ${learning_rate} \
        --training_args.num_train_epochs ${num_train_epochs} \
        --peft_config.r 64 \
        --peft_config.target_modules q_proj k_proj v_proj o_proj \
        --peft_config.lora_alpha 1 \
        --peft_config.lora_dropout 0
}

if [[ "${STAGE}" == "adapters" || "${STAGE}" == "all" ]]; then
    run_dpo_adapter helpful
    run_dpo_adapter harmless
    run_dpo_adapter humor
fi

if [[ "${STAGE}" == "modpo" || "${STAGE}" == "all" ]]; then
    run_modpo
fi
