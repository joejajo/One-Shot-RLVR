#!/bin/bash
# =============================================================================
# SLURM sbatch script — One-Shot-RLVR, 1.5B π₁ r128, 4×A100, single node
# FAU HPC (Hochleistungsrechnen Bavaria / NHR@FAU)
#
# Submit with:  sbatch slurm/one_shot_rlvr_4xa100.slurm
# =============================================================================

#SBATCH --job-name=one_shot_rlvr_1.5b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

# ─── GPU request ─────────────────────────────────────────────────────────────
# FAU HPC uses --gres for GPU allocation. Adjust the resource string to match
# your cluster's GPU type identifier (check `sinfo -o "%n %G"` to list nodes).
#SBATCH --gres=gpu:a100:4

# ─── Time, partition, account ────────────────────────────────────────────────
# !! ADJUST THESE TO YOUR FAU PROJECT !!
#SBATCH --time=48:00:00
#SBATCH --partition=a100
#SBATCH --account=YOUR_FAU_PROJECT_ACCOUNT

# ─── Memory ──────────────────────────────────────────────────────────────────
#SBATCH --mem=200G

# ─── Output logs ─────────────────────────────────────────────────────────────
#SBATCH --output=/home/woody/iwi7/iwi7107h/One-Shot-RLVR/output/logs/slurm-%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/One-Shot-RLVR/output/logs/slurm-%j.err

# =============================================================================
# Environment setup
# =============================================================================

# Create log dir early so SLURM output redirection doesn't fail
mkdir -p /home/woody/iwi7/iwi7107h/One-Shot-RLVR/output/logs

# ─── Load HPC modules ────────────────────────────────────────────────────────
# Adjust module names to match your FAU HPC software stack.
# Common FAU HPC module names shown below; comment/uncomment as needed.
module purge
module load python/3.10-anaconda     # or: module load miniconda3
# module load cuda/12.1               # only needed if not bundled with conda env
# module load nccl/2.18-cuda12        # only if separate NCCL module exists

# ─── Activate your Python / conda environment ─────────────────────────────────
# !! REPLACE 'verl' WITH YOUR ACTUAL ENVIRONMENT NAME !!
# If using conda:
source activate rlvr_train
# If using virtualenv instead of conda, replace the line above with:
# source /home/woody/iwi7/iwi7107h/envs/verl/bin/activate

# =============================================================================
# Ray single-node setup
# We run Ray in single-node mode (no multi-node Ray cluster needed).
# ray.init() in main_ppo.py automatically detects available GPUs on the node.
# =============================================================================

# Confirm GPU visibility
echo "=== GPU Inventory ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "====================="

# Print environment info for debugging
echo "SLURM_JOB_ID:       ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set by SLURM>}"

# =============================================================================
# Run training
# The launcher script owns all path + env configuration.
# =============================================================================
exec bash /home/woody/iwi7/iwi7107h/One-Shot-RLVR/scripts/train/training_1.5b_pi1_r128_hpc.sh
