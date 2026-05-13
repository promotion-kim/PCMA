#!/usr/bin/env bash
set -euo pipefail

# Reward Soup runner for experiment-DPO/scripts/eval/rs.py
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash run_reward_soup.sh 0.5 /ext_hdd/sjkim/mod/reward_soup_outputs
#   CUDA_VISIBLE_DEVICES=1 bash run_reward_soup.sh all /ext_hdd/sjkim/mod/reward_soup_outputs
#
# Args:
#   $1: helpful/better weight w1, or "all"
#   $2: output directory where generated txt files will be copied
#
# Notes:
#   - dpo_model_1 is assumed to be the better/helpful DPO adapter.
#   - dpo_model_2 is assumed to be the safer/harmless DPO adapter.
#   - This script does not modify rs.py. It runs rs.py through a small Python
#     wrapper that patches the removed transformers.top_k_top_p_filtering symbol
#     required by older TRL versions.

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/ext_hdd/sjkim/mod/triton_cache}
mkdir -p "$TRITON_CACHE_DIR"

W_ARG=${1:-0.5}
OUTPUT_DIR=${2:-/ext_hdd/sjkim/mod/reward_soup_outputs}

cd /home/sjkim/MOD/experiment-DPO

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-dryrun}
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"

SFT_MODEL_NAME=${SFT_MODEL_NAME:-PKU-Alignment/alpaca-7b-reproduced}
DPO_MODEL_1_NAME=${DPO_MODEL_1_NAME:-/ext_hdd/sjkim/mod/dpo/dpo-better/best_checkpoint}
DPO_MODEL_2_NAME=${DPO_MODEL_2_NAME:-/ext_hdd/sjkim/mod/dpo/dpo-safer/best_checkpoint}
DATASET_NAME=${DATASET_NAME:-PKU-Alignment/PKU-SafeRLHF-10K-better}
MAX_LENGTH=${MAX_LENGTH:-512}
NUM_BEAMS=${NUM_BEAMS:-1}
F_TYPE=${F_TYPE:-reverse_kl}
RS_SCRIPT=${RS_SCRIPT:-scripts/eval/rs.py}

WRAPPER="${OUTPUT_DIR}/_run_rs_with_transformers_patch.py"

cat > "${WRAPPER}" <<'PY'
import runpy
import sys
import torch
import transformers

# DeepSpeed in some environments expects newer torch.library APIs.
# Reward Soup generation does not use these DeepSpeed custom ops, so this
# compatibility shim only lets the import finish.
if hasattr(torch, "library") and not hasattr(torch.library, "custom_op"):
    def _dummy_custom_op(name, mutates_args=(), **kwargs):
        def _decorator(fn):
            return fn
        return _decorator
    torch.library.custom_op = _dummy_custom_op

if hasattr(torch, "library") and not hasattr(torch.library, "register_fake"):
    def _dummy_register_fake(name, *args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator
    torch.library.register_fake = _dummy_register_fake

# Older TRL imports this symbol from the top-level transformers package.
# Newer transformers versions removed it, so we provide a compatible fallback.
if not hasattr(transformers, "top_k_top_p_filtering"):
    def top_k_top_p_filtering(
        logits,
        top_k=0,
        top_p=1.0,
        filter_value=-float("inf"),
        min_tokens_to_keep=1,
    ):
        if top_k is not None and top_k > 0:
            top_k = min(max(int(top_k), int(min_tokens_to_keep)), logits.size(-1))
            threshold = torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(logits < threshold, filter_value)

        if top_p is not None and 0.0 <= float(top_p) < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

            sorted_indices_to_remove = cumulative_probs > float(top_p)
            if min_tokens_to_keep > 1:
                sorted_indices_to_remove[..., :int(min_tokens_to_keep)] = 0

            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = torch.zeros_like(sorted_indices_to_remove, dtype=torch.bool)
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, filter_value)

        return logits

    transformers.top_k_top_p_filtering = top_k_top_p_filtering

target = sys.argv[1]
sys.argv = [target] + sys.argv[2:]
runpy.run_path(target, run_name="__main__")
PY

if [[ "${W_ARG}" == "all" ]]; then
  WEIGHTS=(0.0 0.3 0.5 0.7 1.0)
else
  WEIGHTS=("${W_ARG}")
fi

echo "===================================================================================================="
echo "[Reward Soup config]"
echo "PWD=${PWD}"
echo "SFT_MODEL_NAME=${SFT_MODEL_NAME}"
echo "DPO_MODEL_1_NAME=${DPO_MODEL_1_NAME}"
echo "DPO_MODEL_2_NAME=${DPO_MODEL_2_NAME}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "===================================================================================================="

for W1 in "${WEIGHTS[@]}"; do
  W2=$(python - <<PY
w = float("${W1}")
print(f"{1.0 - w:.1f}")
PY
)

  echo "----------------------------------------------------------------------------------------------------"
  echo "[Reward Soup] weight_1=${W1}, weight_2=${W2}"
  echo "----------------------------------------------------------------------------------------------------"

  python "${WRAPPER}" "${RS_SCRIPT}" \
    --sft_model_name "${SFT_MODEL_NAME}" \
    --dpo_model_1_name "${DPO_MODEL_1_NAME}" \
    --dpo_model_2_name "${DPO_MODEL_2_NAME}" \
    --weight_1 "${W1}" \
    --weight_2 "${W2}" \
    --dataset_name "${DATASET_NAME}" \
    --max_length "${MAX_LENGTH}" \
    --num_beams "${NUM_BEAMS}" \
    --f_type "${F_TYPE}"

  # rs.py writes to a hard-coded path. Copy the result to the requested output dir.
  SRC="results_beavertail/outputs/rs_output_${W1}_${W2}_${F_TYPE}.txt"
  DEST="${OUTPUT_DIR}/rs_output_h${W1}_s${W2}_${F_TYPE}.txt"

  if [[ -f "${SRC}" ]]; then
    cp "${SRC}" "${DEST}"
    echo "[saved] ${DEST}"
  else
    echo "[warning] expected output not found: ${SRC}"
    echo "[warning] check results_beavertail/outputs/"
  fi
done

echo "[done] Reward Soup generation finished."
