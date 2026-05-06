#!/usr/bin/env bash
set -euo pipefail

# MORLHF-SAFERLHF training script aligned with the MODPO / PC-MODPO setting.
# Usage:
#   bash ppo/morlhf.sh 29500 0.5
#   bash ppo/morlhf.sh 29500 all
#
# $1: main process port
# $2: helpfulness weight w_H, or "all"
#
# Here, w is the helpfulness weight w_H.
# Harmlessness weight is 1 - w.
# This assumes the modified PPO script accepts:
#   --dataset_name
#   --prompt_template
#   --max_prompt_length
# and builds a prompt-only PPO dataset from PKU-SafeRLHF.

cd /home/sjkim/MOD/experiment-PPO
# Make MODPO source modules visible.
# morlhf_saferlhf.py reuses DATASET_CONFIGS from the MODPO codebase.
export DPO_ROOT="/home/sjkim/MOD/experiment-DPO"
export SAFE_RLHF_ROOT="/home/sjkim/MOD/experiment-DPO/safe-rlhf"
export PYTHONPATH="${SAFE_RLHF_ROOT}:${DPO_ROOT}:${PWD}:${PWD}/ppo:${PYTHONPATH:-}"

echo "PYTHONPATH=${PYTHONPATH}"


export PYTHONUNBUFFERED=1

export WANDB_ENTITY=${WANDB_ENTITY:-promotion-kim}
export WANDB_PROJECT=${WANDB_PROJECT:-pcma}
export WANDB_MODE=${WANDB_MODE:-online}

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/ext_hdd/sjkim/mod/triton_cache}
mkdir -p "$TRITON_CACHE_DIR"

# -----------------------------------------------------------------------------
# Accelerate / distributed setting
# -----------------------------------------------------------------------------
NUM_PROCESSES=${NUM_PROCESSES:-2}

PORT_ARG=${1:-29521}
MAIN_PROCESS_PORT=${PORT_ARG}

LAUNCH="accelerate launch --num_processes=${NUM_PROCESSES} --main_process_port ${MAIN_PROCESS_PORT}"

# -----------------------------------------------------------------------------
# Model / dataset setting: match MODPO setting as closely as possible
# -----------------------------------------------------------------------------
SFT_MODEL_NAME=${SFT_MODEL_NAME:-PKU-Alignment/alpaca-7b-reproduced}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-'BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:'}
DATASET_BASE=${DATASET_BASE:-PKU-Alignment/PKU-SafeRLHF-10K}

# PPO uses prompts only. We use the same PKU-SafeRLHF distribution as MODPO.
# Using -better is natural because MODPO stage 2 also uses ${DATASET_BASE}-better as the anchor dataset.
DATASET_NAME=${DATASET_NAME:-${DATASET_BASE}-better}

OUTPUT_ROOT=${OUTPUT_ROOT:-/ext_hdd/sjkim/mod}
SAVE_DIRECTORY=${SAVE_DIRECTORY:-${OUTPUT_ROOT}/morlhf_saferlhf}

# Path to the modified MORLHF script.
MORLHF_SCRIPT=${MORLHF_SCRIPT:-scripts/ppo/morlhf_saferlhf.py}
if [[ ! -f "$MORLHF_SCRIPT" ]]; then
  # fallback if the modified file is stored under ppo/
  MORLHF_SCRIPT=${MORLHF_SCRIPT_FALLBACK:-ppo/morlhf_saferlhf.py}
fi

# -----------------------------------------------------------------------------
# PPO hyperparameters
# -----------------------------------------------------------------------------
EPOCHS=${EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-4}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
INIT_KL_COEF=${INIT_KL_COEF:-0.2}
TARGET_KL=${TARGET_KL:-3}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.5}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-384}
LOAD_IN_8BIT=${LOAD_IN_8BIT:-True}
DISABLE_WANDB=${DISABLE_WANDB:-False}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
REWARD_MODEL_MAX_LENGTH=${REWARD_MODEL_MAX_LENGTH:-512}

# Important: put helpful first so that --preference equals helpfulness weight w_H.
REWARD_NAMES=${REWARD_NAMES:-helpful,harmless}

# One of: a single helpfulness weight, or "all".
W_ARG=${2:-0.5}

EVAL_STEPS=${EVAL_STEPS:-100}
EVAL_NUM_PROMPTS=${EVAL_NUM_PROMPTS:-5}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-128}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
EVAL_SPLIT=${EVAL_SPLIT:-validation}

if [[ "$W_ARG" == "all" ]]; then
  WEIGHTS=(0.0 0.1 0.3 0.5 0.7 1.0)
else
  WEIGHTS=("$W_ARG")
fi

mkdir -p "$SAVE_DIRECTORY"

echo "===================================================================================================="
echo "[MORLHF-SAFERLHF config]"
echo "PWD=$PWD"
echo "MORLHF_SCRIPT=$MORLHF_SCRIPT"
echo "SFT_MODEL_NAME=$SFT_MODEL_NAME"
echo "DATASET_NAME=$DATASET_NAME"
echo "PROMPT_TEMPLATE=$PROMPT_TEMPLATE"
echo "REWARD_NAMES=$REWARD_NAMES"
echo "SAVE_DIRECTORY=$SAVE_DIRECTORY"
echo "NUM_PROCESSES=$NUM_PROCESSES"
echo "MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"
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

  $LAUNCH "$MORLHF_SCRIPT" \
    --base_model_name "$SFT_MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --prompt_template "$PROMPT_TEMPLATE" \
    --reward_names "$REWARD_NAMES" \
    --preference "$W_HELPFUL" \
    --save_directory "$SAVE_DIRECTORY" \
    --wandb_name "$RUN_NAME" \
    --batch_size "$BATCH_SIZE" \
    --mini_batch_size "$MINI_BATCH_SIZE" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --reward_model_max_length "$REWARD_MODEL_MAX_LENGTH" \
    --train_split "$TRAIN_SPLIT" \
    --eval_split "$EVAL_SPLIT" \
    --eval_steps "$EVAL_STEPS" \
    --eval_num_prompts "$EVAL_NUM_PROMPTS" \
    --eval_max_new_tokens "$EVAL_MAX_NEW_TOKENS"

done
