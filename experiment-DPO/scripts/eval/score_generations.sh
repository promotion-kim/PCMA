#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/safe-rlhf:$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

GPU_ID=0

MODPO_JSON="/home/sjkim/MOD/experiment-DPO/outputs/generation/modpo/modpo_0.5.json"
PCMODPO_JSON="/home/sjkim/MOD/experiment-DPO/outputs/generation/pcmodpo/pcmodpo_0.5.json"
OUTPUT_DIR="/home/sjkim/MOD/experiment-DPO/outputs/scores/modpo_vs_pcmodpo_w0p5"

REWARD_MODEL="PKU-Alignment/beaver-7b-v1.0-reward"
COST_MODEL="PKU-Alignment/beaver-7b-v1.0-cost"

BATCH_SIZE=2
MAX_LENGTH=1024
TORCH_DTYPE="bf16"
DEVICE_MAP="none"
LIMIT=-1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu_id) GPU_ID="$2"; shift 2 ;;
    --modpo_json) MODPO_JSON="$2"; shift 2 ;;
    --pcmodpo_json) PCMODPO_JSON="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;

    --reward_model) REWARD_MODEL="$2"; shift 2 ;;
    --cost_model) COST_MODEL="$2"; shift 2 ;;

    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --max_length) MAX_LENGTH="$2"; shift 2 ;;
    --torch_dtype) TORCH_DTYPE="$2"; shift 2 ;;
    --device_map) DEVICE_MAP="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;

    -h|--help)
      cat <<EOF
Usage:
  bash scripts/eval/score_generations.sh \\
    --gpu_id 0 \\
    --modpo_json /path/to/modpo.json \\
    --pcmodpo_json /path/to/pcmodpo.json \\
    --output_dir /path/to/output_dir \\
    --batch_size 2 \\
    --max_length 1024

Outputs:
  output_dir/summary.json
  output_dir/summary.csv
  output_dir/modpo_scored.json
  output_dir/pcmodpo_scored.json
  output_dir/paired_modpo_vs_pcmodpo.json
EOF
      exit 0 ;;

    *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODPO_JSON" ]]; then
  echo "[ERROR] --modpo_json is required"
  exit 1
fi

if [[ -z "$PCMODPO_JSON" ]]; then
  echo "[ERROR] --pcmodpo_json is required"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "================================================================================"
echo "[score_generations.sh]"
echo "GPU_ID=${GPU_ID}"
echo "MODPO_JSON=${MODPO_JSON}"
echo "PCMODPO_JSON=${PCMODPO_JSON}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "REWARD_MODEL=${REWARD_MODEL}"
echo "COST_MODEL=${COST_MODEL}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "TORCH_DTYPE=${TORCH_DTYPE}"
echo "DEVICE_MAP=${DEVICE_MAP}"
echo "LIMIT=${LIMIT}"
echo "================================================================================"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONPATH="$REPO_ROOT/safe-rlhf:$REPO_ROOT:${PYTHONPATH:-}" \
python "$REPO_ROOT/scripts/eval/score_generations.py" \
  --input_jsons "${MODPO_JSON},${PCMODPO_JSON}" \
  --names "modpo,pcmodpo" \
  --output_dir "${OUTPUT_DIR}" \
  --reward_model_name "${REWARD_MODEL}" \
  --cost_model_name "${COST_MODEL}" \
  --batch_size "${BATCH_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --device_map "${DEVICE_MAP}" \
  --limit "${LIMIT}"

echo "[score_generations.sh] Done."
echo "[score_generations.sh] Output dir: ${OUTPUT_DIR}"