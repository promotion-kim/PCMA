#!/usr/bin/env bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

export WANDB_MODE=${WANDB_MODE:-online}
export TOKENIZERS_PARALLELISM=false

# Paper-faithful 3D BPP-MOA pipeline for HH Helpful Assistant task.
#
# Usage examples:
#   # full pipeline: fit heads -> build posterior targets -> train policy
#   bash scripts/bppmoa/hh/bppmoa_hh.sh 0.33 0.33 0.34
#
#   # run only one stage
#   MODE=fit   bash scripts/bppmoa/hh/bppmoa_hh.sh
#   MODE=build bash scripts/bppmoa/hh/bppmoa_hh.sh 0.60 0.20 0.20
#   MODE=train bash scripts/bppmoa/hh/bppmoa_hh.sh 0.60 0.20 0.20

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_MODE=${WANDB_MODE:-online}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/storm_hdd/sjkim/mod/triton_cache}
mkdir -p "${TRITON_CACHE_DIR}"

MODE=${MODE:-all}  # all | fit | build | train

w_helpful=${1:-0.33}
w_harmless=${2:-0.33}
w_humor=${3:-0.34}
run_name="h${w_helpful}_s${w_harmless}_u${w_humor}"

MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.1-8B-Instruct"}
DATA_DIR=${DATA_DIR:-"data"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/storm_hdd/sjkim/mod/bppmoa_hh"}
HEAD_ROOT=${HEAD_ROOT:-"${OUTPUT_ROOT}/reward_heads"}
TARGET_ROOT=${TARGET_ROOT:-"${OUTPUT_ROOT}/targets"}
MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT:-"${OUTPUT_ROOT}/models"}

TARGET_DIR="${TARGET_ROOT}/${run_name}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_ROOT}/${run_name}"

PROMPT_FORMAT=${PROMPT_FORMAT:-"chat"}     # chat | hh | raw | modpo
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"{raw_prompt}"}
TORCH_DTYPE=${TORCH_DTYPE:-"bf16"}
USE_FLASH_ATTENTION_2=${USE_FLASH_ATTENTION_2:-False}

# Score direction after canonicalization.  For harm_score/cost_score, lower is safer.
DIRECTION_HELPFUL=${DIRECTION_HELPFUL:-"higher_is_better"}
DIRECTION_HARMLESS=${DIRECTION_HARMLESS:-"lower_is_better"}
DIRECTION_HUMOR=${DIRECTION_HUMOR:-"higher_is_better"}
MIN_ABS_SCORE_GAP=${MIN_ABS_SCORE_GAP:-0.0}

MAX_LENGTH=${MAX_LENGTH:-1024}
HEAD_BATCH_SIZE=${HEAD_BATCH_SIZE:-4}
HEAD_EVAL_BATCH_SIZE=${HEAD_EVAL_BATCH_SIZE:-4}
HEAD_EPOCHS=${HEAD_EPOCHS:-3}
HEAD_LR=${HEAD_LR:-1e-3}
PRIOR_PRECISION=${PRIOR_PRECISION:-1.0}

VARIANCE_SCALE=${VARIANCE_SCALE:-1.0}
USE_POSTERIOR_VARIANCE=${USE_POSTERIOR_VARIANCE:-True}
CONSTANT_VARIANCE=${CONSTANT_VARIANCE:-False}
CONSTANT_VARIANCE_VALUE=${CONSTANT_VARIANCE_VALUE:-1.0}
TARGET_BATCH_SIZE=${TARGET_BATCH_SIZE:-4}

NUM_PROCESSES=${NUM_PROCESSES:-2}
PORT=${PORT:-29511}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
BETA=${BETA:-0.1}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}

REPORT_TO=${REPORT_TO:-wandb}
WANDB_ENTITY=${WANDB_ENTITY:-promotion-kim}
WANDB_PROJECT=${WANDB_PROJECT:-pcma}
WANDB_MODE=${WANDB_MODE:-online}

export WANDB_ENTITY
export WANDB_PROJECT
export WANDB_MODE
FORCE_RERUN=${FORCE_RERUN:-False}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-False}

HELPFUL_HEAD_PATH="${HEAD_ROOT}/helpful/laplace_head.pt"
HARMLESS_HEAD_PATH="${HEAD_ROOT}/harmless/laplace_head.pt"
HUMOR_HEAD_PATH="${HEAD_ROOT}/humor/laplace_head.pt"

BEST_CKPT_DIR="${MODEL_OUTPUT_DIR}/best_checkpoint"
FINAL_CKPT_DIR="${MODEL_OUTPUT_DIR}/final_checkpoint"

model_done() {
  [[ -f "${BEST_CKPT_DIR}/adapter_model.safetensors" ]] || \
  [[ -f "${BEST_CKPT_DIR}/adapter_model.bin" ]] || \
  [[ -f "${FINAL_CKPT_DIR}/adapter_model.safetensors" ]] || \
  [[ -f "${FINAL_CKPT_DIR}/adapter_model.bin" ]]
}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-False}

SANITY_ARG=""
if [[ "${SANITY_CHECK:-False}" == "True" || "${SANITY_CHECK:-False}" == "true" ]]; then
  SANITY_ARG="--sanity_check"
fi

echo "============================================================"
echo "[BPP-MOA-HH paper-faithful pipeline]"
echo "mode          = ${MODE}"
echo "model         = ${MODEL_NAME}"
echo "data_dir      = ${DATA_DIR}"
echo "weights       = (${w_helpful}, ${w_harmless}, ${w_humor})"
echo "head_root     = ${HEAD_ROOT}"
echo "target_dir    = ${TARGET_DIR}"
echo "model_out     = ${MODEL_OUTPUT_DIR}"
echo "prompt_format = ${PROMPT_FORMAT}"
echo "============================================================"

if [[ "${MODE}" == "all" || "${MODE}" == "fit" ]]; then
  if [[ "${FORCE_RERUN}" != "True" && "${FORCE_RERUN}" != "true" && \
        -f "${HELPFUL_HEAD_PATH}" && -f "${HARMLESS_HEAD_PATH}" && -f "${HUMOR_HEAD_PATH}" ]]; then
    echo "[skip] all reward heads exist:"
    echo "       ${HELPFUL_HEAD_PATH}"
    echo "       ${HARMLESS_HEAD_PATH}"
    echo "       ${HUMOR_HEAD_PATH}"
  else
    python scripts/bppmoa/hh/fit_last_layer_laplace_hh.py \
      --model_name "${MODEL_NAME}" \
      --data_dir "${DATA_DIR}" \
      --output_root "${HEAD_ROOT}" \
      --objective all \
      --prompt_format "${PROMPT_FORMAT}" \
      --prompt_template "${PROMPT_TEMPLATE}" \
      --direction_helpful "${DIRECTION_HELPFUL}" \
      --direction_harmless "${DIRECTION_HARMLESS}" \
      --direction_humor "${DIRECTION_HUMOR}" \
      --min_abs_score_gap "${MIN_ABS_SCORE_GAP}" \
      --max_length "${MAX_LENGTH}" \
      --per_device_train_batch_size "${HEAD_BATCH_SIZE}" \
      --per_device_eval_batch_size "${HEAD_EVAL_BATCH_SIZE}" \
      --num_train_epochs "${HEAD_EPOCHS}" \
      --learning_rate "${HEAD_LR}" \
      --prior_precision "${PRIOR_PRECISION}" \
      --torch_dtype "${TORCH_DTYPE}" \
      --use_flash_attention_2 "${USE_FLASH_ATTENTION_2}" \
      ${SANITY_ARG}
  fi
fi

if [[ "${MODE}" == "all" || "${MODE}" == "build" ]]; then
  if [[ "${FORCE_RERUN}" != "True" && "${FORCE_RERUN}" != "true" && \
        -d "${TARGET_DIR}/train" && -d "${TARGET_DIR}/validation" ]]; then
    echo "[skip] BPP target dataset exists: ${TARGET_DIR}"
  else
    python scripts/bppmoa/hh/build_bpp_targets_hh.py \
      --model_name "${MODEL_NAME}" \
      --data_dir "${DATA_DIR}" \
      --head_root "${HEAD_ROOT}" \
      --output_dir "${TARGET_DIR}" \
      --w_helpful "${w_helpful}" \
      --w_harmless "${w_harmless}" \
      --w_humor "${w_humor}" \
      --variance_scale "${VARIANCE_SCALE}" \
      --use_posterior_variance "${USE_POSTERIOR_VARIANCE}" \
      --constant_variance "${CONSTANT_VARIANCE}" \
      --constant_variance_value "${CONSTANT_VARIANCE_VALUE}" \
      --prompt_format "${PROMPT_FORMAT}" \
      --prompt_template "${PROMPT_TEMPLATE}" \
      --max_length "${MAX_LENGTH}" \
      --per_device_eval_batch_size "${TARGET_BATCH_SIZE}" \
      --torch_dtype "${TORCH_DTYPE}" \
      --use_flash_attention_2 "${USE_FLASH_ATTENTION_2}" \
      ${SANITY_ARG}
  fi
fi

if [[ "${MODE}" == "all" || "${MODE}" == "train" ]]; then
  if [[ "${FORCE_RERUN}" != "True" && "${FORCE_RERUN}" != "true" ]] && model_done; then
    echo "[skip] trained policy already exists:"
    if [[ -d "${BEST_CKPT_DIR}" ]]; then
      echo "       ${BEST_CKPT_DIR}"
    fi
    if [[ -d "${FINAL_CKPT_DIR}" ]]; then
      echo "       ${FINAL_CKPT_DIR}"
    fi
  else
    if [[ "${NUM_PROCESSES}" == "1" ]]; then
      LAUNCH="python"
    else
      LAUNCH="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=${NUM_PROCESSES} --main_process_port ${PORT}"
    fi

    PYTHONPATH="$PWD:${PYTHONPATH:-}" ${LAUNCH} scripts/bppmoa/hh/bppmoa_hh.py \
      --model_name "${MODEL_NAME}" \
      --bpp_dataset_dir "${TARGET_DIR}" \
      --output_dir "${MODEL_OUTPUT_DIR}" \
      --prompt_template "${PROMPT_TEMPLATE}" \
      --beta "${BETA}" \
      --max_length "${MAX_LENGTH}" \
      --torch_dtype "${TORCH_DTYPE}" \
      --use_flash_attention_2 "${USE_FLASH_ATTENTION_2}" \
      --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
      --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
      --gradient_accumulation_steps "${GRAD_ACCUM}" \
      --learning_rate "${LEARNING_RATE}" \
      --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
      --lora_r "${LORA_R}" \
      --lora_alpha "${LORA_ALPHA}" \
      --lora_dropout "${LORA_DROPOUT}" \
      --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
      --bf16 True \
      --fp16 False \
      --report_to "${REPORT_TO}" \
      --run_name "bppmoa_hh_${run_name}"
  fi
fi

echo "[BPP-MOA-HH] done: ${MODE}"
