#!/bin/bash
# =============================================================================
# setup_train_env.sh
# Training conda environment for One-Shot-RLVR on FAU HPC (woody).
#
# Follows the EXACT install sequence from the official repo README,
# with two changes:
#   1. wandb replaced with tensorboard (ships with torch — no extra install)
#   2. torch index-url completed with /cu121 for CUDA 12.1 A100 nodes
#
# Env name: rlvr_train   (matches README)
# Usage: bash /home/woody/iwi7/iwi7107h/One-Shot-RLVR/envs/setup_train_env.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="/home/woody/iwi7/iwi7107h/One-Shot-RLVR"

conda create -y -n rlvr_train python=3.10
eval "$(conda shell.bash hook)"
conda activate rlvr_train

# Install the verl package first (README does this before torch)
cd "${REPO_ROOT}"
pip install -e .

# PyTorch 2.4.0 — completed with /cu121 for A100 CUDA 12.1 driver
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Ray and vLLM (as in README)
pip install ray vllm==0.6.3

# flash-attn (compiled — takes ~10 min)
pip install flash-attn --no-build-isolation

# matplotlib (wandb is intentionally omitted — TensorBoard ships with torch)
pip install matplotlib

# huggingface_hub
pip install huggingface_hub

echo ""
echo "=== Smoke test ==="
python -c "
import torch, vllm, ray, verl, transformers
from torch.utils.tensorboard import SummaryWriter
print(f'torch:          {torch.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
print(f'vllm:           {vllm.__version__}')
print(f'ray:            {ray.__version__}')
print(f'transformers:   {transformers.__version__}')
print(f'tensorboard:    OK (via torch.utils.tensorboard)')
"
echo "DONE. Activate with: conda activate rlvr_train"
