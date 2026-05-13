#!/usr/bin/env bash
set -euo pipefail

# vLLM generation for SafeRLHF-10K validation prompts from a trained RiC LoRA model.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash ric/run_generation_saferlhf_vllm.sh 0.5
#   CUDA_VISIBLE_DEVICES=0 bash ric/run_generation_saferlhf_vllm.sh all
#   CUDA_VISIBLE_DEVICES=0,1 TENSOR_PARALLEL_SIZE=2 bash ric/run_generation_saferlhf_vllm.sh all

W_ARG=${1:-0.5}

cd /home/sjkim/MOD/experiment-PPO

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}:${PWD}/ric:${PYTHONPATH:-}"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/ext_hdd/sjkim/mod/triton_cache}
mkdir -p "${TRITON_CACHE_DIR}"

BASE_MODEL_NAME=${BASE_MODEL_NAME:-PKU-Alignment/alpaca-7b-reproduced}
PEFT_NAME=${PEFT_NAME:-/ext_hdd/sjkim/mod/ric_saferlhf/logs/ric_saferlhf_helpful_harmless/model_iter0}
DATASET_NAME=${DATASET_NAME:-PKU-Alignment/PKU-SafeRLHF-10K-better}
SPLIT=${SPLIT:-validation}

OUTPUT_DIR=${OUTPUT_DIR:-/home/sjkim/MOD/experiment-PPO/outputs/generations/ric_saferlhf}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-ric}

MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:--1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-384}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
BATCH_SIZE=${BATCH_SIZE:-32}
SEED=${SEED:-8888}
DO_SAMPLE=${DO_SAMPLE:-True}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-0}
TARGET_MAP_METHOD=${TARGET_MAP_METHOD:-l2}
TARGET_REWARDS=${TARGET_REWARDS:-}

DTYPE=${DTYPE:-bfloat16}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
MAX_LORA_RANK=${MAX_LORA_RANK:-64}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-True}

mkdir -p "${OUTPUT_DIR}"

if [[ "${W_ARG}" == "all" ]]; then
  PREF_ARGS=(--preferences "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
else
  PREF_ARGS=(--preference "${W_ARG}")
fi

EXTRA_TARGET_ARGS=()
if [[ -n "${TARGET_REWARDS}" ]]; then
  EXTRA_TARGET_ARGS+=(--target_rewards "${TARGET_REWARDS}")
fi

echo "===================================================================================================="
echo "[RiC SafeRLHF vLLM generation runner]"
echo "BASE_MODEL_NAME=${BASE_MODEL_NAME}"
echo "PEFT_NAME=${PEFT_NAME}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "SPLIT=${SPLIT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
echo "===================================================================================================="

python ric/generation_saferlhf.py \
  --base_model_name "${BASE_MODEL_NAME}" \
  --peft_name "${PEFT_NAME}" \
  --dataset_name "${DATASET_NAME}" \
  --split "${SPLIT}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_prefix "${OUTPUT_PREFIX}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --do_sample "${DO_SAMPLE}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --target_map_method "${TARGET_MAP_METHOD}" \
  --dtype "${DTYPE}" \
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  --max_model_len "${MAX_MODEL_LEN}" \
  --max_lora_rank "${MAX_LORA_RANK}" \
  --trust_remote_code "${TRUST_REMOTE_CODE}" \
  "${EXTRA_TARGET_ARGS[@]}" \
  "${PREF_ARGS[@]}"

echo "[done] vLLM generation complete: ${OUTPUT_DIR}"
