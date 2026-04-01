#!/bin/bash
# =============================================================================
# eval_one_experiment_all_ckpts.sh
# Evaluate ALL saved checkpoints of a given experiment on math benchmarks.
#
# Must be run from Qwen2.5-Eval/evaluation/:
#   conda activate rlvr_eval
#   cd /home/woody/iwi7/iwi7107h/One-Shot-RLVR/Qwen2.5-Eval/evaluation
#   bash sh/eval_one_experiment_all_ckpts.sh
#
# Fixed from original:
#   - CHECKPOINTS_DIR: absolute path (was undefined TODO variable)
#   - EVAL_OUTPUT_DIR: absolute path under output/eval/
#   - All experiments uncommented and selectable via EXPERIMENT variable
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Absolute paths
# ---------------------------------------------------------------------------
REPO_ROOT=/home/woody/iwi7/iwi7107h/One-Shot-RLVR
CHECKPOINTS_DIR=${REPO_ROOT}/output/checkpoints
EVAL_OUTPUT_ROOT=${REPO_ROOT}/output/eval

PROMPT_TYPE="qwen25-math-cot"
MAX_TOKENS="3072"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# ---------------------------------------------------------------------------
# Select which experiment to evaluate (uncomment exactly one block)
# ---------------------------------------------------------------------------

####### π₁ r128 — 4×A100 (recommended) #######
PROJECT_NAME="verl_few_shot"
EXPERIMENT_NAME="Qwen2.5-Math-1.5B-pi1_r128_4xa100"

# ####### π₁ r128 — 8×A100 #######
# PROJECT_NAME="verl_few_shot"
# EXPERIMENT_NAME="Qwen2.5-Math-1.5B-pi1_r128"

# ####### DeepScaleR-sub #######
# PROJECT_NAME="verl_few_shot"
# EXPERIMENT_NAME="Qwen2.5-Math-1.5B-dsr_sub"

# ---------------------------------------------------------------------------
# Loop through all saved checkpoints (save_freq=20, total_epochs=2000)
# ---------------------------------------------------------------------------
GLOBAL_STEP_LIST=($(seq 20 20 2000))

for GLOBAL_STEP in "${GLOBAL_STEP_LIST[@]}"; do
    CKPT_PATH=${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${GLOBAL_STEP}/actor
    OUTPUT_DIR=${EVAL_OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${GLOBAL_STEP}

    if [ ! -d "${CKPT_PATH}" ]; then
        echo "Checkpoint not yet saved: ${CKPT_PATH} — skipping"
        continue
    fi

    echo "======== Evaluating step ${GLOBAL_STEP}: ${CKPT_PATH} ========"
    mkdir -p "${OUTPUT_DIR}"
    bash sh/eval_all_math.sh "${PROMPT_TYPE}" "${CKPT_PATH}" "${MAX_TOKENS}" "${OUTPUT_DIR}"
done

echo "All available checkpoints evaluated."
echo "Results under: ${EVAL_OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}/"
