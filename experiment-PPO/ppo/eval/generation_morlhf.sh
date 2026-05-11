#!/usr/bin/env bash
set -euo pipefail

# vLLM generation for MORLHF / PC-MORLHF LoRA adapters.

export PYTHONPATH="${PYTHONPATH:-.}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

GPU_ID=0
MODEL_TYPE="morlhf"   # morlhf or pcmorlhf
SFT_MODEL_NAME="PKU-Alignment/alpaca-7b-reproduced"
DATASET_NAME="PKU-Alignment/PKU-SafeRLHF-10K-better"
SPLIT="validation"
PROMPT_TEMPLATE="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"

ADAPTER_PATH=""
OUTPUT_ROOT="/home/sjkim/MOD/experiment-PPO/outputs"
OUTPUT_PATH=""

BATCH_SIZE=128
MAX_INPUT_LENGTH=512
MAX_NEW_TOKENS=512
MAX_MODEL_LEN=1024
LIMIT=-1

DO_SAMPLE=false
TEMPERATURE=0.7
TOP_P=0.9
SEED=0

DTYPE="bfloat16"
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.90
MAX_LORA_RANK=64
TRUST_REMOTE_CODE=true
DEDUPLICATE_PROMPTS=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu_id) GPU_ID="$2"; shift 2 ;;
    --model_type) MODEL_TYPE="$2"; shift 2 ;;
    --sft_model_name) SFT_MODEL_NAME="$2"; shift 2 ;;
    --dataset_name) DATASET_NAME="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --prompt_template) PROMPT_TEMPLATE="$2"; shift 2 ;;
    --adapter_path) ADAPTER_PATH="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --output_path) OUTPUT_PATH="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --max_input_length) MAX_INPUT_LENGTH="$2"; shift 2 ;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --max_model_len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --do_sample) DO_SAMPLE=true; shift 1 ;;
    --no_do_sample) DO_SAMPLE=false; shift 1 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top_p) TOP_P="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --tensor_parallel_size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --max_lora_rank) MAX_LORA_RANK="$2"; shift 2 ;;
    --trust_remote_code) TRUST_REMOTE_CODE=true; shift 1 ;;
    --no_trust_remote_code) TRUST_REMOTE_CODE=false; shift 1 ;;
    --deduplicate_prompts) DEDUPLICATE_PROMPTS=true; shift 1 ;;
    --no_deduplicate_prompts) DEDUPLICATE_PROMPTS=false; shift 1 ;;
    -h|--help)
      cat <<EOF
Usage:
  bash scripts/eval/generation_morlhf.sh [options]

Required:
  --model_type morlhf|pcmorlhf
  --adapter_path PATH
  --output_path PATH

Examples:
  bash scripts/eval/generation_morlhf.sh \\
    --gpu_id 0 \\
    --model_type morlhf \\
    --adapter_path /ext_hdd/sjkim/mod/morlhf_saferlhf/morlhf_saferlhf_h0.5_s0.5_pref0.5_0.5/batch_697 \\
    --output_path /home/sjkim/MOD/experiment-PPO/outputs/generation/morlhf/morlhf_w0.5.json

  bash scripts/eval/generation_morlhf.sh \\
    --gpu_id 1 \\
    --model_type pcmorlhf \\
    --adapter_path /ext_hdd/sjkim/mod/pcmorlhf_saferlhf/PKU-Alignment/PKU-SafeRLHF-10K/pcmorlhf/pc_alpha_h0.5_s0.5_pref0.5_0.5/batch_697 \\
    --output_path /home/sjkim/MOD/experiment-PPO/outputs/generation/pcmorlhf/pcmorlhf_w0.5.json
EOF
      exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ "${MODEL_TYPE}" != "morlhf" && "${MODEL_TYPE}" != "pcmorlhf" ]]; then
  echo "[ERROR] --model_type must be morlhf or pcmorlhf"
  exit 1
fi

if [[ -z "${ADAPTER_PATH}" ]]; then
  echo "[ERROR] adapter path is empty. Pass --adapter_path PATH."
  exit 1
fi

if [[ -z "${OUTPUT_PATH}" ]]; then
  ts=$(date +%Y%m%d_%H%M%S)
  OUTPUT_PATH="${OUTPUT_ROOT}/generations/${MODEL_TYPE}_${ts}.json"
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "================================================================================"
echo "[generation_morlhf.sh]"
echo "MODEL_TYPE=${MODEL_TYPE}"
echo "GPU_ID=${GPU_ID}"
echo "SFT_MODEL_NAME=${SFT_MODEL_NAME}"
echo "ADAPTER_PATH=${ADAPTER_PATH}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "SPLIT=${SPLIT}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "MAX_INPUT_LENGTH=${MAX_INPUT_LENGTH}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "LIMIT=${LIMIT}"
echo "DO_SAMPLE=${DO_SAMPLE}"
echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
echo "MAX_LORA_RANK=${MAX_LORA_RANK}"
echo "================================================================================"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONPATH=. \
python ppo/eval/generation_morlhf.py \
  --model_type "${MODEL_TYPE}" \
  --sft_model_name "${SFT_MODEL_NAME}" \
  --adapter_path "${ADAPTER_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --split "${SPLIT}" \
  --prompt_template "${PROMPT_TEMPLATE}" \
  --output_path "${OUTPUT_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --max_input_length "${MAX_INPUT_LENGTH}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_model_len "${MAX_MODEL_LEN}" \
  --limit "${LIMIT}" \
  --do_sample "${DO_SAMPLE}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --seed "${SEED}" \
  --dtype "${DTYPE}" \
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  --max_lora_rank "${MAX_LORA_RANK}" \
  --trust_remote_code "${TRUST_REMOTE_CODE}" \
  --deduplicate_prompts "${DEDUPLICATE_PROMPTS}"

echo "[generation_morlhf.sh] Done."
echo "[generation_morlhf.sh] Output: ${OUTPUT_PATH}"
