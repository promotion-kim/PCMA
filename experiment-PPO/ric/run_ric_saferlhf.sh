#!/usr/bin/env bash
set -euo pipefail

# SafeRLHF RiC runner.
# Existing HH-RLHF RiC scripts are not modified.
#
# Usage:
#   bash ppo/run_ric_saferlhf.sh 29600 all
#   bash ppo/run_ric_saferlhf.sh 29600 0.5
#
# $1: base main-process port
# $2: helpfulness weight for evaluation, or "all"

cd ${PROJECT_ROOT:-/home/sjkim/MOD/experiment-PPO}

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export DPO_ROOT=${DPO_ROOT:-/home/sjkim/MOD/experiment-DPO}
export SAFE_RLHF_ROOT=${SAFE_RLHF_ROOT:-/home/sjkim/MOD/experiment-DPO/safe-rlhf}
export PYTHONPATH="${SAFE_RLHF_ROOT}:${DPO_ROOT}:${PWD}:${PWD}/ric:${PYTHONPATH:-}"

export WANDB_ENTITY=${WANDB_ENTITY:-promotion-kim}
export WANDB_PROJECT=${WANDB_PROJECT:-pcma}
export WANDB_MODE=${WANDB_MODE:-online}

NUM_PROCESSES=${NUM_PROCESSES:-1}
PORT_BASE=${1:-29600}
W_ARG=${2:-all}

LAUNCH_PREP="accelerate launch --num_processes=${NUM_PROCESSES} --main_process_port ${PORT_BASE}"
LAUNCH_TRAIN="accelerate launch --num_processes=${NUM_PROCESSES} --main_process_port $((PORT_BASE + 1))"
LAUNCH_EVAL="accelerate launch --num_processes=${NUM_PROCESSES} --main_process_port $((PORT_BASE + 2))"

BASE_MODEL_NAME=${BASE_MODEL_NAME:-PKU-Alignment/alpaca-7b-reproduced}
DATASET_NAME=${DATASET_NAME:-PKU-Alignment/PKU-SafeRLHF-10K}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-'BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:'}

OUTPUT_ROOT=${OUTPUT_ROOT:-/ext_hdd/sjkim/mod/ric_saferlhf}
DATASET_DIR=${DATASET_DIR:-${OUTPUT_ROOT}/datasets/helpful_harmless}
SAVE_DIRECTORY=${SAVE_DIRECTORY:-${OUTPUT_ROOT}/logs}
EVAL_DIRECTORY=${EVAL_DIRECTORY:-${OUTPUT_ROOT}/eval}
RUN_NAME=${RUN_NAME:-ric_saferlhf_helpful_harmless}

REWARD_NAMES=${REWARD_NAMES:-helpful,harmless}
REWARD_MODEL_NAMES=${REWARD_MODEL_NAMES:-PKU-Alignment/beaver-7b-v1.0-reward,PKU-Alignment/beaver-7b-v1.0-cost}
REWARD_SIGNS=${REWARD_SIGNS:-1,-1}

PREPARE_DATASET=${PREPARE_DATASET:-1}
TRAIN_RIC=${TRAIN_RIC:-1}
EVAL_RIC=${EVAL_RIC:-1}

MAX_EXAMPLES=${MAX_EXAMPLES:-}
SANITY_CHECK=${SANITY_CHECK:-False}
MAX_LENGTH=${MAX_LENGTH:-512}
REWARD_MODEL_MAX_LENGTH=${REWARD_MODEL_MAX_LENGTH:-512}
REWARD_BATCH_SIZE=${REWARD_BATCH_SIZE:-4}

TRAINING_STEPS=${TRAINING_STEPS:-20000}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
LOAD_IN_8BIT=${LOAD_IN_8BIT:-True}
DISABLE_WANDB=${DISABLE_WANDB:-False}

MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-200}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
TARGET_MAP_METHOD=${TARGET_MAP_METHOD:-l2}

PREP_SCRIPT=${PREP_SCRIPT:-ric/prepare_dataset_with_rewards_saferlhf_ric.py}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-ric/main_ric_saferlhf.py}
EVAL_SCRIPT=${EVAL_SCRIPT:-ric/evaluation_ric_saferlhf.py}

mkdir -p "$OUTPUT_ROOT" "$SAVE_DIRECTORY" "$EVAL_DIRECTORY"

echo "===================================================================================================="
echo "[RiC-SafeRLHF config]"
echo "PWD=$PWD"
echo "DATASET_NAME=$DATASET_NAME"
echo "BASE_MODEL_NAME=$BASE_MODEL_NAME"
echo "DATASET_DIR=$DATASET_DIR"
echo "SAVE_DIRECTORY=$SAVE_DIRECTORY"
echo "EVAL_DIRECTORY=$EVAL_DIRECTORY"
echo "RUN_NAME=$RUN_NAME"
echo "REWARD_NAMES=$REWARD_NAMES"
echo "REWARD_MODEL_NAMES=$REWARD_MODEL_NAMES"
echo "REWARD_SIGNS=$REWARD_SIGNS"
echo "===================================================================================================="

if [[ "$PREPARE_DATASET" == "1" ]]; then
  echo "[stage 1] Preparing SafeRLHF RiC reward-conditioned dataset..."
  PREP_ARGS=(
    "$PREP_SCRIPT"
    --dataset_name "$DATASET_NAME"
    --split train
    --base_model_name "$BASE_MODEL_NAME"
    --prompt_template "$PROMPT_TEMPLATE"
    --save_directory "$DATASET_DIR"
    --reward_names "$REWARD_NAMES"
    --reward_model_names "$REWARD_MODEL_NAMES"
    --reward_signs "$REWARD_SIGNS"
    --reward_model_max_length "$REWARD_MODEL_MAX_LENGTH"
    --reward_batch_size "$REWARD_BATCH_SIZE"
    --max_length "$MAX_LENGTH"
    --sanity_check "$SANITY_CHECK"
  )
  if [[ -n "$MAX_EXAMPLES" ]]; then
    PREP_ARGS+=(--max_examples "$MAX_EXAMPLES")
  fi
  PYTHONPATH="$PYTHONPATH" $LAUNCH_PREP "${PREP_ARGS[@]}"
else
  echo "[stage 1] Skipped dataset preparation; using DATASET_DIR=$DATASET_DIR"
fi

if [[ "$TRAIN_RIC" == "1" ]]; then
  echo "[stage 2] Training RiC-SafeRLHF once; no preference weight is used in training..."
  PYTHONPATH="$PYTHONPATH" $LAUNCH_TRAIN "$TRAIN_SCRIPT" \
    --base_model_name "$BASE_MODEL_NAME" \
    --train_dataset_path "$DATASET_DIR" \
    --save_directory "$SAVE_DIRECTORY" \
    --wandb_name "$RUN_NAME" \
    --training_steps "$TRAINING_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --load_in_8bit "$LOAD_IN_8BIT" \
    --disable_wandb "$DISABLE_WANDB"
else
  echo "[stage 2] Skipped training; using existing model under $SAVE_DIRECTORY/$RUN_NAME/model_iter0"
fi

if [[ "$EVAL_RIC" == "1" ]]; then
  MODEL_PATH=${MODEL_PATH:-${SAVE_DIRECTORY}/${RUN_NAME}/model_iter0}
  REWARD_STATS_PATH=${REWARD_STATS_PATH:-${DATASET_DIR}/all_reward_stat.npy}

  if [[ "$W_ARG" == "all" ]]; then
    PREF_ARG=(--preferences "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
  else
    PREF_ARG=(--preference "$W_ARG")
  fi

  echo "[stage 3] Evaluating RiC-SafeRLHF with weights: $W_ARG"
  PYTHONPATH="$PYTHONPATH" $LAUNCH_EVAL "$EVAL_SCRIPT" \
    --base_model_name "$BASE_MODEL_NAME" \
    --peft_name "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --split test \
    --prompt_template "$PROMPT_TEMPLATE" \
    --reward_names "$REWARD_NAMES" \
    --reward_model_names "$REWARD_MODEL_NAMES" \
    --reward_signs "$REWARD_SIGNS" \
    --reward_stats_path "$REWARD_STATS_PATH" \
    --reward_model_max_length "$REWARD_MODEL_MAX_LENGTH" \
    --reward_batch_size "$REWARD_BATCH_SIZE" \
    "${PREF_ARG[@]}" \
    --target_map_method "$TARGET_MAP_METHOD" \
    --save_directory "$EVAL_DIRECTORY" \
    --wandb_name "$RUN_NAME" \
    --max_eval_samples "$MAX_EVAL_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS"
else
  echo "[stage 3] Skipped evaluation."
fi
