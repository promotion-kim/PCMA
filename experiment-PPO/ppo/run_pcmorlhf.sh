#!/usr/bin/env bash
set -euo pipefail

# PC-MORLHF runner.
#
# Usage:
#   bash scripts/pcma/run_pcmorlhf.sh 29500 0.5
#   bash scripts/pcma/run_pcmorlhf.sh 29500 all
#
# $1: main process port
# $2: helpfulness base weight w_H, or "all"

cd /home/sjkim/MOD/experiment-PPO

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export DPO_ROOT=${DPO_ROOT:-/home/sjkim/MOD/experiment-DPO}
export SAFE_RLHF_ROOT=${SAFE_RLHF_ROOT:-/home/sjkim/MOD/experiment-DPO/safe-rlhf}
export PYTHONPATH="${SAFE_RLHF_ROOT}:${DPO_ROOT}:${PWD}:${PWD}/ppo:${PYTHONPATH:-}"

export WANDB_ENTITY=${WANDB_ENTITY:-promotion-kim}
export WANDB_PROJECT=${WANDB_PROJECT:-pcma}
export WANDB_MODE=${WANDB_MODE:-online}

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/ext_hdd/sjkim/mod/triton_cache}
mkdir -p "$TRITON_CACHE_DIR"

NUM_PROCESSES=${NUM_PROCESSES:-1}
PORT_ARG=${1:-29521}
MAIN_PROCESS_PORT=${PORT_ARG}
LAUNCH="accelerate launch --num_processes=${NUM_PROCESSES} --main_process_port ${MAIN_PROCESS_PORT}"

SFT_MODEL_NAME=${SFT_MODEL_NAME:-PKU-Alignment/alpaca-7b-reproduced}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-'BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:'}
DATASET_BASE=${DATASET_BASE:-PKU-Alignment/PKU-SafeRLHF-10K}
DATASET_NAME=${DATASET_NAME:-${DATASET_BASE}-better}

OUTPUT_ROOT=${OUTPUT_ROOT:-/ext_hdd/sjkim/mod}
POSTERIOR_DIR=${POSTERIOR_DIR:-${OUTPUT_ROOT}/${DATASET_BASE}/pcma/posterior_morlhf/better_safer}
SAVE_DIRECTORY=${SAVE_DIRECTORY:-${OUTPUT_ROOT}/pcmorlhf_saferlhf}

FIT_SCRIPT=${FIT_SCRIPT:-scripts/pcma/fit_posterior_morlhf.py}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-scripts/pcma/pcmorlhf.py}

# Fallbacks if you keep scripts under ppo/
if [[ ! -f "$FIT_SCRIPT" && -f "ppo/fit_posterior_morlhf.py" ]]; then
  FIT_SCRIPT="ppo/fit_posterior_morlhf.py"
fi
if [[ ! -f "$TRAIN_SCRIPT" && -f "ppo/pcmorlhf.py" ]]; then
  TRAIN_SCRIPT="ppo/pcmorlhf.py"
fi

# Objective datasets/scorers for posterior fitting.
OBJECTIVE_DATASET_NAMES=${OBJECTIVE_DATASET_NAMES:-${DATASET_BASE}-better,${DATASET_BASE}-safer}
OBJECTIVE_NAMES=${OBJECTIVE_NAMES:-helpful,harmless}
OBJECTIVE_MODEL_NAMES=${OBJECTIVE_MODEL_NAMES:-PKU-Alignment/beaver-7b-v1.0-reward,PKU-Alignment/beaver-7b-v1.0-cost}
OBJECTIVE_SIGNS=${OBJECTIVE_SIGNS:-1,-1}

# Posterior fitting hyperparameters.
FIT_POSTERIOR=${FIT_POSTERIOR:-0}  # set to 1 to fit q(theta) before training
FEATURE_SOURCE=${FEATURE_SOURCE:-sft_hidden}
FEATURE_POOLING=${FEATURE_POOLING:-mean}
FEATURE_MODEL_NAME=${FEATURE_MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}
FEATURE_MAX_LENGTH=${FEATURE_MAX_LENGTH:-256}
POSTERIOR_STEPS=${POSTERIOR_STEPS:-2000}
POSTERIOR_LR=${POSTERIOR_LR:-3e-3}
POSTERIOR_BATCH_SIZE=${POSTERIOR_BATCH_SIZE:-512}
POSTERIOR_LOG_EVERY=${POSTERIOR_LOG_EVERY:-50}
POSTERIOR_NUM_SAMPLES=${POSTERIOR_NUM_SAMPLES:-10}
ALPHA_MIN=${ALPHA_MIN:-0.05}

# PPO hyperparameters.
EPOCHS=${EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-4}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
INIT_KL_COEF=${INIT_KL_COEF:-1.0}
TARGET_KL=${TARGET_KL:-1.0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.5}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-384}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
REWARD_MODEL_MAX_LENGTH=${REWARD_MODEL_MAX_LENGTH:-512}
LOAD_IN_8BIT=${LOAD_IN_8BIT:-True}

# Monitoring.
EVAL_STEPS=${EVAL_STEPS:-100}
EVAL_NUM_PROMPTS=${EVAL_NUM_PROMPTS:-5}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-128}
DEBUG_PRINT_ALPHA_EVERY=${DEBUG_PRINT_ALPHA_EVERY:-50}
DEBUG_PRINT_ALPHA_N=${DEBUG_PRINT_ALPHA_N:-3}

REWARD_NAMES=${REWARD_NAMES:-helpful,harmless}
REWARD_MODEL_NAMES=${REWARD_MODEL_NAMES:-PKU-Alignment/beaver-7b-v1.0-reward,PKU-Alignment/beaver-7b-v1.0-cost}
REWARD_SIGNS=${REWARD_SIGNS:-1,-1}

SEED=${SEED:-8888}
W_ARG=${2:-0.5}

if [[ "$W_ARG" == "all" ]]; then
  WEIGHTS=(0.0 0.3 0.5 0.7 1.0)
else
  WEIGHTS=("$W_ARG")
fi

mkdir -p "$SAVE_DIRECTORY"

echo "===================================================================================================="
echo "[PC-MORLHF config]"
echo "PWD=$PWD"
echo "FIT_SCRIPT=$FIT_SCRIPT"
echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "SFT_MODEL_NAME=$SFT_MODEL_NAME"
echo "DATASET_NAME=$DATASET_NAME"
echo "POSTERIOR_DIR=$POSTERIOR_DIR"
echo "NUM_PROCESSES=$NUM_PROCESSES"
echo "MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"
echo "===================================================================================================="

if [[ "$FIT_POSTERIOR" == "1" ]]; then
  echo "[stage 1] fitting explicit-scorer posterior calibrator..."
  PYTHONPATH="$PYTHONPATH" $LAUNCH "$FIT_SCRIPT" \
    --sft_model_name "$SFT_MODEL_NAME" \
    --objective_dataset_names "$OBJECTIVE_DATASET_NAMES" \
    --objective_names "$OBJECTIVE_NAMES" \
    --objective_model_names "$OBJECTIVE_MODEL_NAMES" \
    --objective_signs "$OBJECTIVE_SIGNS" \
    --output_dir "$POSTERIOR_DIR" \
    --feature_source "$FEATURE_SOURCE" \
    --feature_pooling "$FEATURE_POOLING" \
    --feature_model_name "$FEATURE_MODEL_NAME" \
    --prompt_template "$PROMPT_TEMPLATE" \
    --seed "$SEED" \
    --scorer_max_length "$REWARD_MODEL_MAX_LENGTH" \
    --feature_max_length "$FEATURE_MAX_LENGTH" \
    --posterior_steps "$POSTERIOR_STEPS" \
    --posterior_lr "$POSTERIOR_LR" \
    --posterior_batch_size "$POSTERIOR_BATCH_SIZE" \
    --posterior_log_every "$POSTERIOR_LOG_EVERY"
else
  echo "[stage 1] skipped posterior fitting; using existing POSTERIOR_DIR=$POSTERIOR_DIR"
fi

for W_HELPFUL in "${WEIGHTS[@]}"; do
  W_HARMLESS=$(python - <<PY
w = float("$W_HELPFUL")
print(f"{1.0 - w:.1f}")
PY
)
  RUN_NAME="${DATASET_BASE}/pcmorlhf/pc_alpha_h${W_HELPFUL}_s${W_HARMLESS}"

  echo "----------------------------------------------------------------------------------------------------"
  echo "[stage 2] PC-MORLHF w_helpful=${W_HELPFUL}, w_harmless=${W_HARMLESS}"
  echo "[stage 2] wandb_name=${RUN_NAME}"
  echo "----------------------------------------------------------------------------------------------------"

  PYTHONPATH="$PYTHONPATH" $LAUNCH "$TRAIN_SCRIPT" \
    --base_model_name "$SFT_MODEL_NAME" \
    --posterior_calibrator_path "$POSTERIOR_DIR" \
    --dataset_name "$DATASET_NAME" \
    --prompt_template "$PROMPT_TEMPLATE" \
    --train_split train \
    --eval_split validation \
    --seed "$SEED" \
    --preference "$W_HELPFUL" \
    --save_directory "$SAVE_DIRECTORY" \
    --wandb_name "$RUN_NAME" \
    --reward_names "$REWARD_NAMES" \
    --reward_model_names "$REWARD_MODEL_NAMES" \
    --reward_signs "$REWARD_SIGNS" \
    --feature_source "$FEATURE_SOURCE" \
    --feature_pooling "$FEATURE_POOLING" \
    --feature_model_name "$FEATURE_MODEL_NAME" \
    --feature_max_length "$FEATURE_MAX_LENGTH" \
    --posterior_num_samples "$POSTERIOR_NUM_SAMPLES" \
    --alpha_min "$ALPHA_MIN" \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --mini_batch_size "$MINI_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --init_kl_coef "$INIT_KL_COEF" \
    --target "$TARGET_KL" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --reward_model_max_length "$REWARD_MODEL_MAX_LENGTH" \
    --eval_steps "$EVAL_STEPS" \
    --eval_num_prompts "$EVAL_NUM_PROMPTS" \
    --eval_max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
    --debug_print_alpha_every "$DEBUG_PRINT_ALPHA_EVERY" \
    --debug_print_alpha_n "$DEBUG_PRINT_ALPHA_N"
done
