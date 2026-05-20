#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/safe-rlhf:$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

GPU_ID=0
INPUT_JSON=""
INPUT_DIR=""
METHOD_NAME="method"
OUTPUT_DIR="outputs/scores/single_method"

REWARD_TYPE="beaver"  # beaver | armorm
REWARD_MODEL="PKU-Alignment/beaver-7b-v1.0-reward"
COST_MODEL="PKU-Alignment/beaver-7b-v1.0-cost"
ARMORM_MODEL="RLHFlow/ArmoRM-Llama3-8B-v0.1"
ARMORM_HELPFUL_OBJECTIVE="helpsteer-helpfulness"
ARMORM_HARMLESS_OBJECTIVE="beavertails-is_safe"
ARMORM_SAVE_ALL_OBJECTIVES=false

BATCH_SIZE=4
MAX_LENGTH=1024
TORCH_DTYPE="bf16"
DEVICE_MAP="none"
LIMIT=-1
PREPROCESS_FLAG="--preprocess_generations"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu_id) GPU_ID="$2"; shift 2 ;;
    --input_json|--generation_json|--json|--jsonl) INPUT_JSON="$2"; shift 2 ;;
    --input_dir) INPUT_DIR="$2"; shift 2 ;;
    --name|--method_name) METHOD_NAME="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;

    --reward_type) REWARD_TYPE="$2"; shift 2 ;;
    --reward_model|--reward_model_name) REWARD_MODEL="$2"; shift 2 ;;
    --cost_model|--cost_model_name) COST_MODEL="$2"; shift 2 ;;
    --armorm_model|--armorm_model_name) ARMORM_MODEL="$2"; shift 2 ;;
    --armorm_helpful_objective) ARMORM_HELPFUL_OBJECTIVE="$2"; shift 2 ;;
    --armorm_harmless_objective) ARMORM_HARMLESS_OBJECTIVE="$2"; shift 2 ;;
    --armorm_save_all_objectives) ARMORM_SAVE_ALL_OBJECTIVES=true; shift 1 ;;

    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --max_length) MAX_LENGTH="$2"; shift 2 ;;
    --torch_dtype) TORCH_DTYPE="$2"; shift 2 ;;
    --device_map) DEVICE_MAP="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --preprocess_generations) PREPROCESS_FLAG="--preprocess_generations"; shift 1 ;;
    --no_preprocess_generations) PREPROCESS_FLAG="--no_preprocess_generations"; shift 1 ;;

    -h|--help)
      cat <<EOF
Usage:
  bash scripts/eval/score_one_generation.sh \
    --gpu_id 0 \
    --input_json /path/to/generation.json_or_jsonl \
    --name rs \
    --output_dir outputs/scores/armorm/rs_h0.0_s1.0 \
    --reward_type armorm \
    --armorm_model RLHFlow/ArmoRM-Llama3-8B-v0.1 \
    --armorm_helpful_objective helpsteer-helpfulness \
    --armorm_harmless_objective beavertails-is_safe \
    --armorm_save_all_objectives \
    --batch_size 1 \
    --max_length 1024 \
    --torch_dtype bf16 \
    --no_preprocess_generations

Supported input formats:
  1) JSON with {"data": [{"raw_prompt": ..., "generation": ...}, ...]}
  2) JSON list of rows
  3) JSONL rows, including Reward Soup format:
     {"raw_prompt": ..., "prompt": ..., "response": ..., "full_text": ..., "weight_helpful": ..., "weight_harmless": ...}

Outputs:
  output_dir/summary.json
  output_dir/summary.csv
  output_dir/<name>_scored.json
EOF
      exit 0 ;;

    *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$INPUT_JSON" && -z "$INPUT_DIR" ]]; then
  echo "[ERROR] either --input_json or --input_dir is required"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ "$ARMORM_SAVE_ALL_OBJECTIVES" == "true" ]]; then
  EXTRA_ARGS+=("--armorm_save_all_objectives")
fi

echo "================================================================================"
echo "[score_one_generation.sh]"
echo "GPU_ID=${GPU_ID}"
echo "INPUT_JSON=${INPUT_JSON}"
echo "INPUT_DIR=${INPUT_DIR}"
echo "METHOD_NAME=${METHOD_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "REWARD_TYPE=${REWARD_TYPE}"
echo "REWARD_MODEL=${REWARD_MODEL}"
echo "COST_MODEL=${COST_MODEL}"
echo "ARMORM_MODEL=${ARMORM_MODEL}"
echo "ARMORM_HELPFUL_OBJECTIVE=${ARMORM_HELPFUL_OBJECTIVE}"
echo "ARMORM_HARMLESS_OBJECTIVE=${ARMORM_HARMLESS_OBJECTIVE}"
echo "ARMORM_SAVE_ALL_OBJECTIVES=${ARMORM_SAVE_ALL_OBJECTIVES}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "TORCH_DTYPE=${TORCH_DTYPE}"
echo "DEVICE_MAP=${DEVICE_MAP}"
echo "LIMIT=${LIMIT}"
echo "PREPROCESS_FLAG=${PREPROCESS_FLAG}"
echo "================================================================================"

run_one () {
  local input_path="$1"
  local method_name="$2"
  local out_dir="$3"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONPATH="$REPO_ROOT/safe-rlhf:$REPO_ROOT:${PYTHONPATH:-}" \
  python "$REPO_ROOT/scripts/eval/score_generations.py" \
    --input_json "${input_path}" \
    --name "${method_name}" \
    --output_dir "${out_dir}" \
    --reward_type "${REWARD_TYPE}" \
    --reward_model_name "${REWARD_MODEL}" \
    --cost_model_name "${COST_MODEL}" \
    --armorm_model_name "${ARMORM_MODEL}" \
    --armorm_helpful_objective "${ARMORM_HELPFUL_OBJECTIVE}" \
    --armorm_harmless_objective "${ARMORM_HARMLESS_OBJECTIVE}" \
    "${EXTRA_ARGS[@]}" \
    --batch_size "${BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --device_map "${DEVICE_MAP}" \
    --limit "${LIMIT}" \
    "${PREPROCESS_FLAG}" 
}

if [[ -n "$INPUT_DIR" ]]; then
  shopt -s nullglob
  TXT_FILES=("${INPUT_DIR}"/*.txt)
  JSON_FILES=("${INPUT_DIR}"/*.json "${INPUT_DIR}"/*.jsonl)
  FILES=("${TXT_FILES[@]}" "${JSON_FILES[@]}")
  shopt -u nullglob

  if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "[ERROR] no .txt/.json/.jsonl files found in ${INPUT_DIR}"
    exit 1
  fi

  for f in "${FILES[@]}"; do
    stem="$(basename "$f")"
    stem="${stem%.*}"
    echo "[batch] scoring: $f"
    run_one "$f" "$stem" "$OUTPUT_DIR/$stem"
  done
else
  run_one "$INPUT_JSON" "$METHOD_NAME" "$OUTPUT_DIR"
fi

echo "[score_one_generation.sh] Done."
echo "[score_one_generation.sh] Output dir: ${OUTPUT_DIR}"
