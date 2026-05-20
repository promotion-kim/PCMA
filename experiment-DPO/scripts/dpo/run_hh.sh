#!/usr/bin/env bash
set -euo pipefail

export WANDB_ENTITY=${WANDB_ENTITY:-promotion-kim}
export WANDB_PROJECT=${WANDB_PROJECT:-pcma}
export WANDB_MODE=${WANDB_MODE:-online}

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/ext_hdd/sjkim/mod/triton_cache}
mkdir -p "$TRITON_CACHE_DIR"

# Usage:
#   bash scripts/dpo/run_hh_dpo.sh helpful
#   bash scripts/dpo/run_hh_dpo.sh harmless
#   bash scripts/dpo/run_hh_dpo.sh humor
#   bash scripts/dpo/run_hh_dpo.sh all

objective=${1:-all}

if [[ "${objective}" != "helpful" && "${objective}" != "harmless" && "${objective}" != "humor" && "${objective}" != "all" ]]; then
  echo "[error] objective must be one of: helpful, harmless, humor, all"
  exit 1
fi

SFT_MODEL_NAME=${SFT_MODEL_NAME:-"meta-llama/Llama-3.1-8B-Instruct"}

# You said preference data are saved under data/.
DATA_DIR=${DATA_DIR:-"data"}

OUTPUT_DIR=${OUTPUT_DIR:-"/ext_hdd/sjkim/mod"}
RUN_GROUP=${RUN_GROUP:-"dpo_hh"}

PROMPT_FORMAT=${PROMPT_FORMAT:-"chat"}
DIVERGENCE_TYPE=${DIVERGENCE_TYPE:-"reverse_kl"}

MAX_LENGTH=${MAX_LENGTH:-1024}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
BETA=${BETA:-0.1}
SANITY_CHECK=${SANITY_CHECK:-False}
NUM_PROCESSES=${NUM_PROCESSES:-2}
PORT=${PORT:-29501}

LAUNCH="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=${NUM_PROCESSES} --main_process_port ${PORT}"

run_one_objective () {
  local obj=$1

  local run_name="dpo/${RUN_GROUP}/${obj}"
  local out_dir="${OUTPUT_DIR}/${run_name}"

  echo "============================================================"
  echo "[DPO-HH] objective      = ${obj}"
  echo "[DPO-HH] model          = ${SFT_MODEL_NAME}"
  echo "[DPO-HH] data_dir       = ${DATA_DIR}"
  echo "[DPO-HH] prompt_format  = ${PROMPT_FORMAT}"
  echo "[DPO-HH] output_dir     = ${out_dir}"
  echo "============================================================"

  PYTHONPATH="$PWD/compat:$PWD:${PYTHONPATH:-}" ${LAUNCH} scripts/dpo/dpo_hh.py \
    --divergence_type "${DIVERGENCE_TYPE}" \
    --sft_model_name "${SFT_MODEL_NAME}" \
    --data_dir "${DATA_DIR}" \
    --objective "${obj}" \
    --prompt_format "${PROMPT_FORMAT}" \
    --sanity_check ${SANITY_CHECK} \
    --beta ${BETA} \
    --max_length ${MAX_LENGTH} \
    --training_args.output_dir "${out_dir}" \
    --training_args.run_name "${run_name}" \
    --training_args.per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --training_args.per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
    --training_args.gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --training_args.learning_rate ${LEARNING_RATE} \
    --peft_config.r 64 \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha 1 \
    --peft_config.lora_dropout 0
}

if [[ "${objective}" == "all" ]]; then
  run_one_objective helpful
  run_one_objective harmless
  run_one_objective humor
else
  run_one_objective "${objective}"
fi