#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

GPU_ID=0,1

MODEL_TYPE="pcmodpo"   # modpo or pcmodpo
SFT_MODEL_NAME="PKU-Alignment/alpaca-7b-reproduced"
DATASET_NAME="PKU-Alignment/PKU-SafeRLHF-10K-better"
SPLIT="test"
PROMPT_TEMPLATE="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"

ADAPTER_PATH=""
MODPO_ADAPTER_PATH=""
PCMODPO_ADAPTER_PATH=""

OUTPUT_ROOT="/home/sjkim/MOD/experiment-DPO/outputs"
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
SANITY_CHECK=false
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
    --modpo_adapter_path) MODPO_ADAPTER_PATH="$2"; shift 2 ;;
    --pcmodpo_adapter_path) PCMODPO_ADAPTER_PATH="$2"; shift 2 ;;

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

    --sanity_check) SANITY_CHECK=true; shift 1 ;;
    --no_sanity_check) SANITY_CHECK=false; shift 1 ;;
    --deduplicate_prompts) DEDUPLICATE_PROMPTS=true; shift 1 ;;
    --no_deduplicate_prompts) DEDUPLICATE_PROMPTS=false; shift 1 ;;

    -h|--help)
      cat <<EOF
Usage:
  bash scripts/eval/generation_vllm.sh [options]

Required:
  --model_type modpo|pcmodpo
  --adapter_path PATH
  --output_path PATH

Examples:
  bash scripts/eval/generation.sh \\
    --gpu_id 0 \\
    --model_type pcmodpo \\
    --adapter_path /path/to/pcmodpo/output_or_checkpoint \\
    --output_path /ext_hdd/sjkim/mod/generations/pcmodpo_w0p5_vllm.json \\
    --batch_size 128 \\
    --max_new_tokens 512

  bash scripts/eval/generation.sh \\
    --gpu_id 0,1 \\
    --tensor_parallel_size 1 \\
    --model_type modpo \\
    --adapter_path /ext_hdd/sjkim/mod/PKU-Alignment/PKU-SafeRLHF-10K/modpo/lm/(0.0)*r_better+(1-0.0)*r_safer/best_checkpoint\\
    --output_path /home/sjkim/mod/experiment-DPO/outputs/generation/modpo/modpo_0.0.json
EOF
      exit 0 ;;

    *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "${ADAPTER_PATH}" ]]; then
  if [[ "${MODEL_TYPE}" == "modpo" ]]; then
    ADAPTER_PATH="${MODPO_ADAPTER_PATH}"
  elif [[ "${MODEL_TYPE}" == "pcmodpo" ]]; then
    ADAPTER_PATH="${PCMODPO_ADAPTER_PATH}"
  else
    echo "[ERROR] --model_type must be modpo or pcmodpo"
    exit 1
  fi
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
echo "[generation_vllm.sh]"
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
python scripts/eval/generation.py \
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
  --sanity_check "${SANITY_CHECK}" \
  --deduplicate_prompts "${DEDUPLICATE_PROMPTS}"

echo "[generation_vllm.sh] Done."
echo "[generation_vllm.sh] Output: ${OUTPUT_PATH}"