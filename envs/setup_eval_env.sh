#!/bin/bash
# =============================================================================
# setup_eval_env.sh
# Evaluation conda environment for One-Shot-RLVR on FAU HPC (woody).
#
# Follows the EXACT install sequence from the official repo README,
# with one change: wandb omitted (not needed for eval on HPC).
#
# Env name: rlvr_eval   (matches README)
# Usage: bash /home/woody/iwi7/iwi7107h/One-Shot-RLVR/envs/setup_eval_env.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="/home/woody/iwi7/iwi7107h/One-Shot-RLVR"
EVAL_DIR="${REPO_ROOT}/Qwen2.5-Eval/evaluation"

conda create -y -n rlvr_eval python=3.10
eval "$(conda shell.bash hook)"
conda activate rlvr_eval

# Install latex2sympy first (README does this before requirements.txt)
cd "${EVAL_DIR}/latex2sympy"
pip install -e .

# Back to eval dir, install base requirements
cd "${EVAL_DIR}"
pip install -r requirements.txt

# vLLM 0.5.1 — strictly required by eval harness
pip install vllm==0.5.1 --no-build-isolation

# Pin transformers to eval-required version
pip install transformers==4.42.3

# matplotlib (wandb omitted)
pip install matplotlib

# README upgrades transformers again after the pin — followed exactly
pip install -U transformers

# README then pins vLLM to 0.6.3 as the final step
pip install vllm==0.6.3

echo ""
echo "=== Smoke test ==="
python -c "
import torch, vllm, transformers, sympy
print(f'torch:          {torch.__version__}')
print(f'vllm:           {vllm.__version__}')
print(f'transformers:   {transformers.__version__}')
print(f'sympy:          {sympy.__version__}')
"
echo ""
echo "DONE. Activate with: conda activate rlvr_eval"
echo ""
echo "Example eval run:"
echo "  conda activate rlvr_eval"
echo "  cd ${EVAL_DIR}"
echo "  export CUDA_VISIBLE_DEVICES=0"
echo "  bash sh/eval.sh qwen25-math-cot \\"
echo "    /home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B"
