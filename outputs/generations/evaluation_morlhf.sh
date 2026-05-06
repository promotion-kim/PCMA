#!/usr/bin/env bash
set -euo pipefail

# Reward-model evaluation for MORLHF vs PC-MORLHF generations.
#
# Example:
#   bash scripts/eval/evaluation_morlhf.sh \
#     --w 0.5 \
#     --morlhf_json /home/sjkim/MOD/experiment-PPO/outputs/generation/morlhf/morlhf_w0.5.json \
#     --pcmorlhf_json /home/sjkim/MOD/experiment-PPO/outputs/generation/pcmorlhf/pcmorlhf_w0.5.json \
#     --output_dir /home/sjkim/MOD/experiment-PPO/outputs/eval_cleaned_mip_morlhf/0.5

cd /home/sjkim/MOD/experiment-PPO

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export DPO_ROOT=${DPO_ROOT:-/home/sjkim/MOD/experiment-DPO}
export SAFE_RLHF_ROOT=${SAFE_RLHF_ROOT:-/home/sjkim/MOD/experiment-DPO/safe-rlhf}
export PYTHONPATH="${SAFE_RLHF_ROOT}:${DPO_ROOT}:${PWD}:${PWD}/ppo:${PYTHONPATH:-}"

GPU_ID=${GPU_ID:-0}
W=0.5

MORLHF_JSON=""
PCMORLHF_JSON=""
OUTPUT_DIR=""

REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-PKU-Alignment/beaver-7b-v1.0-reward}
COST_MODEL_NAME=${COST_MODEL_NAME:-PKU-Alignment/beaver-7b-v1.0-cost}

BATCH_SIZE=${BATCH_SIZE:-4}
MAX_LENGTH=${MAX_LENGTH:-1024}
TORCH_DTYPE=${TORCH_DTYPE:-bf16}
DEVICE_MAP=${DEVICE_MAP:-none}
LIMIT=${LIMIT:--1}
SAVE_SCORED_JSON=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu_id) GPU_ID="$2"; shift 2 ;;
    --w) W="$2"; shift 2 ;;
    --morlhf_json) MORLHF_JSON="$2"; shift 2 ;;
    --pcmorlhf_json) PCMORLHF_JSON="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --reward_model_name) REWARD_MODEL_NAME="$2"; shift 2 ;;
    --cost_model_name) COST_MODEL_NAME="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --max_length) MAX_LENGTH="$2"; shift 2 ;;
    --torch_dtype) TORCH_DTYPE="$2"; shift 2 ;;
    --device_map) DEVICE_MAP="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --save_scored_json) SAVE_SCORED_JSON=true; shift 1 ;;
    --no_save_scored_json) SAVE_SCORED_JSON=false; shift 1 ;;
    -h|--help)
      cat <<EOF
Usage:
  bash scripts/eval/evaluation_morlhf.sh [options]

Required if defaults are not used:
  --morlhf_json PATH
  --pcmorlhf_json PATH

Options:
  --w FLOAT
  --output_dir PATH
  --gpu_id GPU_ID
  --batch_size INT
  --max_length INT
  --limit INT
EOF
      exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
  esac
done

W_STR=$(python - <<PY
w = float("${W}")
print(f"{w:.1f}")
PY
)

if [[ -z "${MORLHF_JSON}" ]]; then
  MORLHF_JSON="/home/sjkim/MOD/experiment-PPO/outputs/generations/morlhf/morlhf_w${W_STR}.json"
fi

if [[ -z "${PCMORLHF_JSON}" ]]; then
  PCMORLHF_JSON="/home/sjkim/MOD/experiment-PPO/outputs/generations/pcmorlhf/pcmorlhf_w${W_STR}.json"
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="/home/sjkim/MOD/experiment-PPO/outputs/eval_cleaned_mip_morlhf/${W_STR}"
fi

if [[ ! -f "${MORLHF_JSON}" ]]; then
  echo "[ERROR] MORLHF_JSON not found: ${MORLHF_JSON}"
  exit 1
fi

if [[ ! -f "${PCMORLHF_JSON}" ]]; then
  echo "[ERROR] PCMORLHF_JSON not found: ${PCMORLHF_JSON}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

EXTRA_SAVE_FLAG="--save_scored_json"
if [[ "${SAVE_SCORED_JSON}" != "true" ]]; then
  EXTRA_SAVE_FLAG="--no_save_scored_json"
fi

echo "================================================================================"
echo "[evaluation_morlhf.sh]"
echo "GPU_ID=${GPU_ID}"
echo "W=${W_STR}"
echo "MORLHF_JSON=${MORLHF_JSON}"
echo "PCMORLHF_JSON=${PCMORLHF_JSON}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "REWARD_MODEL_NAME=${REWARD_MODEL_NAME}"
echo "COST_MODEL_NAME=${COST_MODEL_NAME}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "LIMIT=${LIMIT}"
echo "================================================================================"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python outputs/generations/evaluation_morlhf.py \
  --w "${W_STR}" \
  --morlhf_json "${MORLHF_JSON}" \
  --pcmorlhf_json "${PCMORLHF_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --reward_model_name "${REWARD_MODEL_NAME}" \
  --cost_model_name "${COST_MODEL_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --device_map "${DEVICE_MAP}" \
  --limit "${LIMIT}" \
  ${EXTRA_SAVE_FLAG}

echo "[evaluation_morlhf.sh] Done."
echo "[evaluation_morlhf.sh] Output: ${OUTPUT_DIR}"
