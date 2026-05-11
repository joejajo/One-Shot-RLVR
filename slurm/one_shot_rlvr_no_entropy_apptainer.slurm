#!/bin/bash
# =============================================================================
# one_shot_rlvr_no_entropy_apptainer.slurm
# SLURM job — One-Shot-RLVR training via Apptainer (verl container image)
# FAU HPC (woody / NHR@FAU), single node, 4×A100
#
# Submit:
#   sbatch slurm/one_shot_rlvr_no_entropy_apptainer.slurm
#
# Resume (pass TOTAL_EPOCHS via --export):
#   sbatch --export=ALL,TOTAL_EPOCHS=600 slurm/one_shot_rlvr_no_entropy_apptainer.slurm
# =============================================================================

#SBATCH --job-name=one_shot_rlvr_no_entropy_apptainer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# ─── GPU request ──────────────────────────────────────────────────────────────
# Check with: sinfo -o "%n %G"
#SBATCH --gres=gpu:a100:2

# ─── Time, partition, account ─────────────────────────────────────────────────
# !! ADJUST TO YOUR FAU PROJECT !!
#SBATCH --time=16:00:00
#SBATCH --partition=a100
#SBATCH --exclude=tg096


# ─── Memory ───────────────────────────────────────────────────────────────────
# (Removed --mem=200G per tinygpu cluster rules. Memory is auto-allocated per GPU)

# ─── Output logs ──────────────────────────────────────────────────────────────
#SBATCH --output=/home/woody/iwi7/iwi7107h/One-Shot-RLVR/output/run01_1000steps_no_entropy/logs/slurm-%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/One-Shot-RLVR/output/run01_1000steps_no_entropy/logs/slurm-%j.err

set -euo pipefail

# =============================================================================
# Paths — adjust if you cloned elsewhere
# =============================================================================
REPO_ROOT=/home/woody/iwi7/iwi7107h/One-Shot-RLVR
MODELS_DIR=/home/woody/iwi7/iwi7107h/models
SIF=/home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif

# How many total training steps. Override at submit time with:
#   sbatch --export=ALL,TOTAL_EPOCHS=600 ...
TOTAL_EPOCHS=${TOTAL_EPOCHS:-9999}   # step cap (total_training_steps=1000) governs

# =============================================================================
# Pre-flight: create output dirs BEFORE the container starts
# (avoids permission issues if the container runs as a different uid)
# =============================================================================
mkdir -p "${REPO_ROOT}/output/run01_1000steps_no_entropy/logs"
mkdir -p "${REPO_ROOT}/output/run01_1000steps_no_entropy/checkpoints"
mkdir -p "${REPO_ROOT}/output/run01_1000steps_no_entropy/tensorboard"
mkdir -p "${REPO_ROOT}/output/run01_1000steps_no_entropy/model_outputs"
mkdir -p "${REPO_ROOT}/.cache/huggingface"

# =============================================================================
# Apptainer is natively available on the cluster nodes
# =============================================================================
apptainer --version   # confirm it is available

# =============================================================================
# Diagnostics (logged to .out file)
# =============================================================================
echo "===== Job info ====="
echo "SLURM_JOB_ID:        ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST:  ${SLURM_JOB_NODELIST}"
echo "SLURM_GPUS_ON_NODE:  ${SLURM_GPUS_ON_NODE:-<not set>}"
echo "CUDA_VISIBLE_DEVICES:${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "SIF image:           ${SIF}"
echo "TOTAL_EPOCHS:        ${TOTAL_EPOCHS}"
echo "===================="

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "===================="

# =============================================================================
# Launch training inside the Apptainer container
#
# --nv            → pass NVIDIA GPU devices into the container
# --bind src:dst  → mount host directories into the container
#                   using the same absolute path on both sides so all
#                   hardcoded paths in scripts resolve correctly
#
# The container's Python already has verl, torch, vllm, ray, flash-attn.
# PYTHONPATH is prepended with REPO_ROOT so local patched verl/ takes precedence
# over the container's built-in verl without any pip install or image rebuild.
# =============================================================================

# Force container Python to ignore ~/.local libraries that might break ABI
mkdir -p /tmp/joel_cupti_fix
ln -sf /opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/libcupti.so.12.9 /tmp/joel_cupti_fix/libcupti.so.12

export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONPATH="${REPO_ROOT}"
export APPTAINERENV_LD_LIBRARY_PATH=/tmp/joel_cupti_fix:/.singularity.d/libs
export APPTAINERENV_NCCL_IB_DISABLE=1
export APPTAINERENV_NCCL_DEBUG=WARN

# Preload correct NVIDIA CUDA libs from the container's nvidia Python packages.
# Fixes dynamic-linker errors for libcupti / libnccl / libcublas at container startup.
_NV="/usr/local/lib/python3.12/dist-packages/nvidia"
export APPTAINERENV_LD_PRELOAD="\
${_NV}/cuda_cupti/lib/libcupti.so.12:\
${_NV}/nccl/lib/libnccl.so.2:\
${_NV}/cublas/lib/libcublas.so.12:\
${_NV}/cublas/lib/libcublasLt.so.12"

apptainer exec --cleanenv --nv \
  --bind "${REPO_ROOT}:${REPO_ROOT}" \
  --bind "${MODELS_DIR}:${MODELS_DIR}" \
  --bind /tmp/joel_cupti_fix:/tmp/joel_cupti_fix \
  "${SIF}" \
  bash -lc "
export PYTHONNOUSERSITE=1
export PYTHONPATH='${REPO_ROOT}'
export LD_LIBRARY_PATH='/tmp/joel_cupti_fix:/.singularity.d/libs'
python -s - <<'PY'
import torch, vllm, ray
import verl
print('torch =', torch.__version__)
print('vllm =', vllm.__version__)
print('ray =', ray.__version__)
print('verl file =', verl.__file__)
print('cuda available =', torch.cuda.is_available())
print('cuda device count =', torch.cuda.device_count())
PY

bash '${REPO_ROOT}/scripts/train/training_1.5b_pi1_r128_no_entropy_apptainer.sh' '${TOTAL_EPOCHS}'
"
