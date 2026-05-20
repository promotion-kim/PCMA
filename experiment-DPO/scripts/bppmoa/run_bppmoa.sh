#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

export WANDB_ENTITY=${WANDB_ENTITY:-promotion-kim}
export WANDB_PROJECT=${WANDB_PROJECT:-pcma}
export WANDB_MODE=${WANDB_MODE:-online}

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/ext_hdd/sjkim/mod/triton_cache}
mkdir -p "$TRITON_CACHE_DIR"

port=${port:-29500}
num_processes=${num_processes:-1}
LAUNCH=${LAUNCH:-"accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes="$num_processes" --main_process_port "$port""}

sft_model_name=${sft_model_name:-"PKU-Alignment/alpaca-7b-reproduced"}
prompt_template=${prompt_template:-"BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"}
dataset_name=${dataset_name:-"PKU-Alignment/PKU-SafeRLHF-10K"}
output_dir=${output_dir:-"/ext_hdd/sjkim/mod"}
max_length=${max_length:-512}
per_device_train_batch_size=${per_device_train_batch_size:-1}
per_device_eval_batch_size=${per_device_eval_batch_size:-1}
gradient_accumulation_steps=${gradient_accumulation_steps:-2}
learning_rate=${learning_rate:-5e-4}
seed=${seed:-0}

# BPP-MOA reward-head fitting hyperparameters. These train only a small linear head
# on frozen SFT features; the final policy hyperparameters below mirror MODPO.
reward_batch_size=${reward_batch_size:-4}
reward_lr=${reward_lr:-1e-3}
reward_epochs=${reward_epochs:-3}
prior_precision=${prior_precision:-1.0}
variance_scale=${variance_scale:-1.0}

# Usage:
#   bash scripts/bppmoa/run_bppmoa.sh 0.5 full
#   bash scripts/bppmoa/run_bppmoa.sh 0.5 constant_variance
#   bash scripts/bppmoa/run_bppmoa.sh 0.5 map_only
w=${1:-0.5}
target_mode=${2:-full}

case "${target_mode}" in
  full|constant_variance|map_only) ;;
  *)
    echo "[error] target_mode must be one of: full, constant_variance, map_only"
    echo "[usage] bash $0 <w_helpful> <target_mode>"
    exit 1
    ;;
esac

w_harmless=$(python - <<PY
w = float("$w")
print(f"{1.0 - w:.10g}")
PY
)

bpp_root="${output_dir}/${dataset_name}/bppmoa"
helpful_head_dir="${bpp_root}/reward_heads/better"
harmless_head_dir="${bpp_root}/reward_heads/safer"
target_dir="${bpp_root}/targets/${target_mode}/w${w}"

mkdir -p "$bpp_root"

echo "[BPP-MOA] w_helpful=${w}, w_harmless=${w_harmless}, target_mode=${target_mode}"
echo "[BPP-MOA] target_dir=${target_dir}"

# Phase A: objective-specific Bayesian reward fitting with last-layer diagonal Laplace.
if [[ ! -f "${helpful_head_dir}/laplace_head.pt" ]]; then
  PYTHONPATH=. python scripts/bppmoa/fit_last_layer_laplace.py \
    --sft_model_name "${sft_model_name}" \
    --dataset_name "${dataset_name}-better" \
    --output_dir "${helpful_head_dir}" \
    --prompt_template "${prompt_template}" \
    --seed "${seed}" \
    --max_length "${max_length}" \
    --per_device_train_batch_size "${reward_batch_size}" \
    --per_device_eval_batch_size "${reward_batch_size}" \
    --num_train_epochs "${reward_epochs}" \
    --learning_rate "${reward_lr}" \
    --prior_precision "${prior_precision}"
else
  echo "[skip] helpful head exists: ${helpful_head_dir}/laplace_head.pt"
fi

if [[ ! -f "${harmless_head_dir}/laplace_head.pt" ]]; then
  PYTHONPATH=. python scripts/bppmoa/fit_last_layer_laplace.py \
    --sft_model_name "${sft_model_name}" \
    --dataset_name "${dataset_name}-safer" \
    --output_dir "${harmless_head_dir}" \
    --prompt_template "${prompt_template}" \
    --seed "${seed}" \
    --max_length "${max_length}" \
    --per_device_train_batch_size "${reward_batch_size}" \
    --per_device_eval_batch_size "${reward_batch_size}" \
    --num_train_epochs "${reward_epochs}" \
    --learning_rate "${reward_lr}" \
    --prior_precision "${prior_precision}"
else
  echo "[skip] harmless head exists: ${harmless_head_dir}/laplace_head.pt"
fi

# Phase B: posterior-pooled soft target construction on the same Q dataset as MODPO stage 2.
if [[ ! -d "${target_dir}/train" || ! -d "${target_dir}/validation" ]]; then
  PYTHONPATH=. python scripts/bppmoa/build_bpp_targets.py \
    --sft_model_name "${sft_model_name}" \
    --target_dataset_name "${dataset_name}-better" \
    --helpful_head_path "${helpful_head_dir}/laplace_head.pt" \
    --harmless_head_path "${harmless_head_dir}/laplace_head.pt" \
    --output_dir "${target_dir}" \
    --w_helpful "${w}" \
    --w_harmless "${w_harmless}" \
    --variance_scale "${variance_scale}" \
    --target_mode "${target_mode}" \
    --prompt_template "${prompt_template}" \
    --seed "${seed}" \
    --max_length "${max_length}" \
    --per_device_eval_batch_size "${reward_batch_size}"
else
  echo "[skip] target dataset exists: ${target_dir}"
fi

# Phase C: policy distillation. Hyperparameters mirror the attached MODPO run script.
lm_run_name="${dataset_name}/bppmoa/${target_mode}/lm/(${w})*r_better+(1-${w})*r_safer"
PYTHONPATH=. $LAUNCH scripts/bppmoa/bppmoa.py \
  --sft_model_name "${sft_model_name}" \
  --bpp_dataset_dir "${target_dir}" \
  --prompt_template "${prompt_template}" \
  --seed "${seed}" \
  --w "${w}" \
  --max_length "${max_length}" \
  --training_args.output_dir "${output_dir}/${lm_run_name}" \
  --training_args.run_name "${lm_run_name}" \
  --training_args.per_device_train_batch_size "${per_device_train_batch_size}" \
  --training_args.per_device_eval_batch_size "${per_device_eval_batch_size}" \
  --training_args.gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --training_args.learning_rate "${learning_rate}" \
  --peft_config.r 64 \
  --peft_config.target_modules q_proj k_proj v_proj o_proj \
  --peft_config.lora_alpha 1 \
  --peft_config.lora_dropout 0
