#!/bin/bash
set -e

cd /home/sjkim/MOD/experiment-DPO
export PYTHONPATH=$PWD:$PYTHONPATH

#!/usr/bin/env bash
set -euo pipefail

# Example runner for VI-MOD.
# 1) First train objective-specific adapters with your existing DPO script.
# 2) Prepare a calibration JSONL with rows:
#    {"objective":0,"prompt":"...","y1":"...","y2":"...","z":1}
#    {"objective":1,"prompt":"...","chosen":"...","rejected":"..."}
# 3) Run this script.

BASE_MODEL="PKU-Alignment/alpaca-7b-reproduced"
PROMPT_TEMPLATE="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"
DIVERGENCE_TYPE="reverse_kl"

# Put objective adapters in the same order as user weights.
# Example: helpful/better adapter first, safe/safer adapter second.
ADAPTER_BETTER="/ext_hdd/sjkim/mod/dpo/dpo-better/best_checkpoint"
ADAPTER_SAFER="/ext_hdd/sjkim/mod/dpo/dpo-safer/best_checkpoint"
ADAPTER_PATHS="${ADAPTER_BETTER},${ADAPTER_SAFER}"

CALIB_JSONL="./data/vi_calibration.jsonl"
OUT_DIR="/ext_hdd/sjkim/mod/output/vi_mod"

# Train VI calibrator.
python scripts/eval/vi_mod.py \
  --mode train_calibrator \
  --base_model "${BASE_MODEL}" \
  --adapter_paths "${ADAPTER_PATHS}" \
  --divergence_type "${DIVERGENCE_TYPE}" \
  --prompt_template "${PROMPT_TEMPLATE}" \
  --calibration_jsonl "${CALIB_JSONL}" \
  --output_dir "${OUT_DIR}" \
  --feature_dim 128 \
  --feature_pooling mean \
  --normalize_margins \
  --vi_steps 2000 \
  --vi_lr 1e-2 \
  --vi_mc_samples 1 \
  --sigma_u 0.5 \
  --sigma_b 0.1

