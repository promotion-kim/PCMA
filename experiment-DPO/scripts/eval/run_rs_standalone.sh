#!/usr/bin/env bash
set -euo pipefail

# Standalone Reward Soup runner.
# This avoids importing TRL/DeepSpeed by using scripts/eval/rs_standalone.py.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/eval/run_rs_standalone.sh 0.5 outputs/generation/rs
#   CUDA_VISIBLE_DEVICES=1 bash scripts/eval/run_rs_standalone.sh all outputs/generation/rs

W_ARG=${1:-0.5}
OUTPUT_DIR=${2:-outputs/generation/rs}

cd /home/sjkim/MOD/experiment-DPO

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

SFT_MODEL_NAME=${SFT_MODEL_NAME:-PKU-Alignment/alpaca-7b-reproduced}
DPO_MODEL_1_NAME=${DPO_MODEL_1_NAME:-/ext_hdd/sjkim/mod/dpo/dpo-better/best_checkpoint}
DPO_MODEL_2_NAME=${DPO_MODEL_2_NAME:-/ext_hdd/sjkim/mod/dpo/dpo-safer/best_checkpoint}
DATASET_NAME=${DATASET_NAME:-PKU-Alignment/PKU-SafeRLHF-10K}
SPLIT=${SPLIT:-validation}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
NUM_BEAMS=${NUM_BEAMS:-1}
DO_SAMPLE=${DO_SAMPLE:-False}
TORCH_DTYPE=${TORCH_DTYPE:-bf16}

mkdir -p "${OUTPUT_DIR}"

if [[ "${W_ARG}" == "all" ]]; then
  WEIGHTS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
else
  WEIGHTS=("${W_ARG}")
fi

echo "===================================================================================================="
echo "[Reward Soup Standalone config]"
echo "PWD=${PWD}"
echo "SFT_MODEL_NAME=${SFT_MODEL_NAME}"
echo "DPO_MODEL_1_NAME=${DPO_MODEL_1_NAME}"
echo "DPO_MODEL_2_NAME=${DPO_MODEL_2_NAME}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "SPLIT=${SPLIT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "===================================================================================================="

for W1 in "${WEIGHTS[@]}"; do
  W2=$(python - <<PY
w = float("${W1}")
print(f"{1.0 - w:.1f}")
PY
)

  echo "----------------------------------------------------------------------------------------------------"
  echo "[Reward Soup Standalone] weight_helpful=${W1}, weight_harmless=${W2}"
  echo "----------------------------------------------------------------------------------------------------"

  EXTRA_ARGS=()
  if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
    EXTRA_ARGS+=(--max_eval_samples "${MAX_EVAL_SAMPLES}")
  fi

  SAMPLE_ARGS=()
  if [[ "${DO_SAMPLE}" == "True" || "${DO_SAMPLE}" == "true" || "${DO_SAMPLE}" == "1" ]]; then
    SAMPLE_ARGS+=(--do_sample)
  fi

  python scripts/eval/rs_standalone.py \
    --sft_model_name "${SFT_MODEL_NAME}" \
    --dpo_model_1_name "${DPO_MODEL_1_NAME}" \
    --dpo_model_2_name "${DPO_MODEL_2_NAME}" \
    --weight_1 "${W1}" \
    --weight_2 "${W2}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${SPLIT}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_name "rs_output_h${W1}_s${W2}.jsonl" \
    --batch_size "${BATCH_SIZE}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --num_beams "${NUM_BEAMS}" \
    --torch_dtype "${TORCH_DTYPE}" \
    "${SAMPLE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
done

echo "[done] Reward Soup standalone generation finished."
