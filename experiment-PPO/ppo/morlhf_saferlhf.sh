#!/usr/bin/env bash
set -euo pipefail

# MORLHF-SafeRLHF training runner.
#
# Usage examples:
#   CUDA_VISIBLE_DEVICES=0 bash ppo/morlhf_saferlhf.sh 29500 0.5
#   CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 bash ppo/morlhf_saferlhf.sh 29500 all
#
# $1: main process port
# $2: helpfulness weight w_H, or "all"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "$PROJECT_ROOT"

PCMA_ROOT=${PCMA_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export DPO_ROOT=${DPO_ROOT:-${PCMA_ROOT}/experiment-DPO}
export SAFE_RLHF_ROOT=${SAFE_RLHF_ROOT:-${DPO_ROOT}/safe-rlhf}
export PYTHONPATH="${SAFE_RLHF_ROOT}:${DPO_ROOT}:${PWD}:${PWD}/ppo:${PWD}/ric:${PYTHONPATH:-}"

export WANDB_ENTITY=${WANDB_ENTITY:-promotion-kim}
export WANDB_PROJECT=${WANDB_PROJECT:-pcma}
export WANDB_MODE=${WANDB_MODE:-online}

# Avoid the previous /ext_hdd permission issue by defaulting to a local project cache.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${PROJECT_ROOT}/.triton_cache}
mkdir -p "$TRITON_CACHE_DIR"

NUM_PROCESSES=${NUM_PROCESSES:-1}
NUM_MACHINES=${NUM_MACHINES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-no}
DYNAMO_BACKEND=${DYNAMO_BACKEND:-no}
PORT_ARG=${1:-29500}
W_ARG=${2:-0.5}

LAUNCH=(
  accelerate launch
  --num_processes "$NUM_PROCESSES"
  --num_machines "$NUM_MACHINES"
  --mixed_precision "$MIXED_PRECISION"
  --dynamo_backend "$DYNAMO_BACKEND"
  --main_process_port "$PORT_ARG"
)

SFT_MODEL_NAME=${SFT_MODEL_NAME:-PKU-Alignment/alpaca-7b-reproduced}
DATASET_BASE=${DATASET_BASE:-PKU-Alignment/PKU-SafeRLHF-10K}
DATASET_NAME=${DATASET_NAME:-${DATASET_BASE}-better}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-'BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:'}

OUTPUT_ROOT=${OUTPUT_ROOT:-${PCMA_ROOT}}
SAVE_DIRECTORY=${SAVE_DIRECTORY:-${OUTPUT_ROOT}/morlhf_saferlhf}

MORLHF_SCRIPT=${MORLHF_SCRIPT:-ppo/morlhf_saferlhf.py}

# PPO hyperparameters.
EPOCHS=${EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-4}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
INIT_KL_COEF=${INIT_KL_COEF:-0.2}
TARGET_KL=${TARGET_KL:-3}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.5}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-384}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
LOAD_IN_8BIT=${LOAD_IN_8BIT:-False}
DISABLE_WANDB=${DISABLE_WANDB:-False}

# SafeRLHF reward/cost setup.
REWARD_NAMES=${REWARD_NAMES:-helpful,harmless}
REWARD_MODEL_NAMES=${REWARD_MODEL_NAMES:-PKU-Alignment/beaver-7b-v1.0-reward,PKU-Alignment/beaver-7b-v1.0-cost}
REWARD_SIGNS=${REWARD_SIGNS:-1,-1}
REWARD_MODEL_MAX_LENGTH=${REWARD_MODEL_MAX_LENGTH:-512}
REWARD_BATCH_SIZE=${REWARD_BATCH_SIZE:-4}

TRAIN_SPLIT=${TRAIN_SPLIT:-train}
EVAL_SPLIT=${EVAL_SPLIT:-validation}
EVAL_STEPS=${EVAL_STEPS:-0}
EVAL_NUM_PROMPTS=${EVAL_NUM_PROMPTS:-5}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-128}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:--1}
SAVE_EVERY=${SAVE_EVERY:-100}
LOG_EVERY=${LOG_EVERY:-1}
SEED=${SEED:-42}

if [[ "$W_ARG" == "all" ]]; then
  WEIGHTS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
else
  WEIGHTS=("$W_ARG")
fi

mkdir -p "$SAVE_DIRECTORY"

if [[ ! -f "$MORLHF_SCRIPT" ]]; then
  echo "[error] MORLHF_SCRIPT not found: $MORLHF_SCRIPT"
  echo "[hint] Current PROJECT_ROOT=$PROJECT_ROOT"
  exit 1
fi

echo "===================================================================================================="
echo "[MORLHF-SafeRLHF config]"
echo "PWD=$PWD"
echo "MORLHF_SCRIPT=$MORLHF_SCRIPT"
echo "SFT_MODEL_NAME=$SFT_MODEL_NAME"
echo "DATASET_NAME=$DATASET_NAME"
echo "PROMPT_TEMPLATE=$PROMPT_TEMPLATE"
echo "REWARD_NAMES=$REWARD_NAMES"
echo "REWARD_MODEL_NAMES=$REWARD_MODEL_NAMES"
echo "REWARD_SIGNS=$REWARD_SIGNS"
echo "SAVE_DIRECTORY=$SAVE_DIRECTORY"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "NUM_PROCESSES=$NUM_PROCESSES"
echo "NUM_MACHINES=$NUM_MACHINES"
echo "MAIN_PROCESS_PORT=$PORT_ARG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "===================================================================================================="

for W_HELPFUL in "${WEIGHTS[@]}"; do
  W_HARMLESS=$(python - <<PY
w = float("$W_HELPFUL")
print(f"{1.0 - w:.1f}")
PY
)

  RUN_NAME="morlhf_saferlhf_h${W_HELPFUL}_s${W_HARMLESS}"

  echo "----------------------------------------------------------------------------------------------------"
  echo "[run] w_helpful=${W_HELPFUL}, w_harmless=${W_HARMLESS}"
  echo "[run] wandb_name=${RUN_NAME}"
  echo "----------------------------------------------------------------------------------------------------"

  ARGS=(
    --base_model_name "$SFT_MODEL_NAME"
    --dataset_name "$DATASET_NAME"
    --prompt_template "$PROMPT_TEMPLATE"
    --train_split "$TRAIN_SPLIT"
    --eval_split "$EVAL_SPLIT"
    --reward_names "$REWARD_NAMES"
    --reward_model_names "$REWARD_MODEL_NAMES"
    --reward_signs "$REWARD_SIGNS"
    --preference "$W_HELPFUL"
    --save_directory "$SAVE_DIRECTORY"
    --wandb_name "$RUN_NAME"
    --epochs "$EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --batch_size "$BATCH_SIZE"
    --mini_batch_size "$MINI_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --init_kl_coef "$INIT_KL_COEF"
    --target "$TARGET_KL"
    --max_grad_norm "$MAX_GRAD_NORM"
    --max_prompt_length "$MAX_PROMPT_LENGTH"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --reward_model_max_length "$REWARD_MODEL_MAX_LENGTH"
    --reward_batch_size "$REWARD_BATCH_SIZE"
    --max_train_samples "$MAX_TRAIN_SAMPLES"
    --load_in_8bit "$LOAD_IN_8BIT"
    --disable_wandb "$DISABLE_WANDB"
    --eval_steps "$EVAL_STEPS"
    --eval_num_prompts "$EVAL_NUM_PROMPTS"
    --eval_max_new_tokens "$EVAL_MAX_NEW_TOKENS"
    --save_every "$SAVE_EVERY"
    --log_every "$LOG_EVERY"
    --seed "$SEED"
  )

  "${LAUNCH[@]}" "$MORLHF_SCRIPT" "${ARGS[@]}"
done
