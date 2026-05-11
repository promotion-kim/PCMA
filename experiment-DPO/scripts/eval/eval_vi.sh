#!/usr/bin/env bash
set -euo pipefail

cd /home/sjkim/MOD/experiment-DPO

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=dryrun

# Usage:
#   bash scripts/eval/eval_vi.sh GPU_ID f_type preference
#
# Example:
#   bash scripts/eval/eval_vi.sh 0 reverse_kl better
#   bash scripts/eval/eval_vi.sh 0 reverse_kl safer

GPU_ID="${1:-0}"
F_TYPE="${2:-reverse_kl}"
PREFERENCE="${3:-better}"

PROMPT_TEMPLATE="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"
BASE_MODEL="PKU-Alignment/alpaca-7b-reproduced"

DATASET_BASE="PKU-Alignment/PKU-SafeRLHF-10K"
DATASET_NAME="${DATASET_BASE}-${PREFERENCE}"

OUTPUT_ROOT="/ext_hdd/sjkim/mod"
ADAPTER_ROOT="${OUTPUT_ROOT}/dpo"
ADAPTER_BETTER="${ADAPTER_ROOT}/dpo-better/best_checkpoint"
ADAPTER_SAFER="${ADAPTER_ROOT}/dpo-safer/best_checkpoint"
ADAPTER_PATHS="${ADAPTER_BETTER},${ADAPTER_SAFER}"

OUT_DIR="${OUTPUT_ROOT}/output/vi_mod"
CALIBRATOR_PATH="${OUT_DIR}/calibrator.pt"
FEATURE_STATE_PATH="${OUT_DIR}/features.pt"

SEED=42
MAX_LENGTH=512
NUM_BEAMS=1
GEN_BATCH_SIZE=4
MOD_ITERS=50
RHO=1.0
NUM_VI_SAMPLES=32
SANITY_CHECK=False

if [ "${F_TYPE}" != "reverse_kl" ]; then
    TYPE_STR="-${F_TYPE}"
else
    TYPE_STR=""
fi

if [ ! -f "${CALIBRATOR_PATH}" ]; then
  echo "[error] calibrator not found: ${CALIBRATOR_PATH}"
  exit 1
fi

if [ ! -f "${FEATURE_STATE_PATH}" ]; then
  echo "[error] feature state not found: ${FEATURE_STATE_PATH}"
  exit 1
fi

echo "[run] VI-MOD iterative generation"
echo "[gpu]             ${GPU_ID}"
echo "[dataset]         ${DATASET_NAME}"
echo "[seed]            ${SEED}"
echo "[mod iters]       ${MOD_ITERS}"
echo "[batch size]      ${GEN_BATCH_SIZE}"
echo "[max length]      ${MAX_LENGTH}"
echo "[f_type]          ${F_TYPE}"
echo "[rho]             ${RHO}"
echo "[num_vi_samples]  ${NUM_VI_SAMPLES}"
echo "[calibrator]      ${CALIBRATOR_PATH}"
echo "[features]        ${FEATURE_STATE_PATH}"

for weights in \
    "0.7 0.3" \
    "1.0 0.0"
do
    WEIGHT_1=$(echo ${weights} | awk '{print $1}')
    WEIGHT_2=$(echo ${weights} | awk '{print $2}')

    OUTPUT_FILE="${OUT_DIR}/vi_records_${PREFERENCE}_w_${WEIGHT_1}_${WEIGHT_2}_${F_TYPE}.jsonl"

    echo "=========================================="
    echo "[RUN] preference=${PREFERENCE}"
    echo "[RUN] weight_1=${WEIGHT_1}, weight_2=${WEIGHT_2}"
    echo "[RUN] output=${OUTPUT_FILE}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/eval/vi_mod.py \
      --mode generate \
      --prompt_source mod_eval \
      --base_model "${BASE_MODEL}" \
      --adapter_paths "${ADAPTER_PATHS}" \
      --divergence_type "${F_TYPE}" \
      --prompt_template "${PROMPT_TEMPLATE}" \
      --calibrator_path "${CALIBRATOR_PATH}" \
      --feature_state_path "${FEATURE_STATE_PATH}" \
      --output_dir "${OUT_DIR}" \
      --user_weights "${WEIGHT_1},${WEIGHT_2}" \
      --rho "${RHO}" \
      --num_vi_samples "${NUM_VI_SAMPLES}" \
      --dataset_name "${DATASET_NAME}" \
      --mod_iters "${MOD_ITERS}" \
      --generation_batch_size "${GEN_BATCH_SIZE}" \
      --max_length "${MAX_LENGTH}" \
      --num_beams "${NUM_BEAMS}" \
      --seed "${SEED}" \
      --beta 0.1 \
      --num_proc 4 \
      --sanity_check "${SANITY_CHECK}" \
      --output_file "${OUTPUT_FILE}" \
      --print_generations \
      --print_max_chars 2000
done