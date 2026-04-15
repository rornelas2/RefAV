#!/bin/bash
#SBATCH --job-name=probe_pypl
#SBATCH --output=logs/probe_pypl_%j.out
#SBATCH --error=logs/probe_pypl_%j.err
#SBATCH --partition=your_partition
#SBATCH --gres=gpu:a100:1 #Request 80gb A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --mail-type=ALL

# Agentic Python-code planner probe (10 scenes).
# - Qwen-72B emits Python code that uses refAV.atomic_functions.
# - After each attempt: exec, then quantitatively score the output pkl
#   against GT for this (log, prompt). Score + prior code feed back as
#   reflection for the next attempt. Up to --max-attempts generations.
# - Successful attempts (score >= memory-threshold) persist to a long-term
#   memory jsonl and seed future prompts as few-shot examples.

set -euo pipefail

#activate your python venv

export SCRATCH="your root path"
export DATA="data directory"
export HF_HOME="${DATA}/huggingface_cache"
export TORCH_HOME="${SCRATCH}/torch_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1

#mkdir -p for_logs

DATA_ROOT="${SCRATCH}/argoverse_data"
SM_DATA="${DATA_ROOT}/scenario_mining"
REFAV_SENSOR="${DATA_ROOT}/refav_sensor"
export REFAV_AV2_DATA_DIR="${REFAV_SENSOR}"
OUTPUT_ROOT="${SCRATCH}/refav_output"

AGENTIC_PROBE="${OUTPUT_ROOT}/sm_predictions/agentic_spatiotemporal/probe_dettm_full_n10"
SELECTED_PAIRS="${AGENTIC_PROBE}/selected_log_prompt_pairs.json"

if [[ ! -f "${SELECTED_PAIRS}" ]]; then
  echo "ERROR: selected pairs not found at ${SELECTED_PAIRS}"
  exit 1
fi

PROBE_NAME="probe_python_planner_n10"
echo "=== Python planner probe (Qwen-72B, exec + quantitative feedback) ==="
echo "  experiment: ${PROBE_NAME}"
echo "  start: $(date)"

python RefAV/run/run_python_planner.py \
  --experiment-name "${PROBE_NAME}" \
  --split val \
  --log-prompt-pairs "${SELECTED_PAIRS}" \
  --log-root "${REFAV_SENSOR}/val" \
  --gt-annotations "${SM_DATA}/scenario_mining_val_annotations.feather" \
  --gt-combined-pkl "${OUTPUT_ROOT}/sm_dataset/val/${PROBE_NAME}_gt.pkl" \
  --planner-model-name Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4 \
  --model-max-memory-gpu 70GiB \
  --model-max-memory-cpu 16GiB \
  --max-items 10 \
  --max-attempts 4 \
  --target-score 0.5 \
  --memory-threshold 0.3 \
  --memory-topk 3 \
  --max-new-tokens 2048 \
  --temperature 0.2

echo "=== Done: $(date) ==="
echo "Predictions: ${OUTPUT_ROOT}/sm_predictions/python_planner/${PROBE_NAME}/"
