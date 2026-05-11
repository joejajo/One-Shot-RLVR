#!/bin/bash
# =============================================================================
# training_1.5b_pi1_r128_no_entropy_apptainer.sh
# One-Shot-RLVR training — Qwen2.5-Math-1.5B, π₁ r128, 4×A100, single node
#
# Designed to run INSIDE the verl Apptainer container.
# Do NOT activate conda — the container already has the full verl stack.
#
# Invoked by the SLURM wrapper:
#   slurm/one_shot_rlvr_no_entropy_apptainer.slurm
#
# Or manually on an interactive GPU node:
#   apptainer exec --nv \
#     --bind /home/woody/iwi7/iwi7107h/One-Shot-RLVR:/home/woody/iwi7/iwi7107h/One-Shot-RLVR \
#     --bind /home/woody/iwi7/iwi7107h/models:/home/woody/iwi7/iwi7107h/models \
#     /home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif \
#     bash scripts/train/training_1.5b_pi1_r128_no_entropy_apptainer.sh
#
# Resume (pass total_epochs as first arg):
#   bash training_1.5b_pi1_r128_no_entropy_apptainer.sh 600   # job 2
#   bash training_1.5b_pi1_r128_no_entropy_apptainer.sh 900   # job 3
# =============================================================================
set -euo pipefail
set -x

# ---------------------------------------------------------------------------
# 0.  Run identity — change RUN_NAME for each new experiment
# ---------------------------------------------------------------------------
RUN_NAME="run01_1000steps_no_entropy"              # <-- label for this training run
TOTAL_EPOCHS=${1:-9999}                 # effectively infinite — total_training_steps=1000 governs

# ---------------------------------------------------------------------------
# 1.  Paths  (all absolute — the container sees the host filesystem via --bind)
# ---------------------------------------------------------------------------
REPO_ROOT=/home/woody/iwi7/iwi7107h/One-Shot-RLVR
MODEL_PATH=/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B
TRAIN_FILE=${REPO_ROOT}/data/train/one_shot_rlvr/pi1_r128.parquet
VAL_FILE=${REPO_ROOT}/data/test/math500.parquet
CHECKPOINTS_DIR=${REPO_ROOT}/output/${RUN_NAME}/checkpoints
TENSORBOARD_LOG_DIR=${REPO_ROOT}/output/${RUN_NAME}/tensorboard
MODEL_OUTPUTS_JSONL=${REPO_ROOT}/output/${RUN_NAME}/model_outputs/val_generations.jsonl
LOG_FILE=${REPO_ROOT}/output/${RUN_NAME}/logs/training.log

# ---------------------------------------------------------------------------
# 2.  Create output directories (the container can write to bind-mounted paths)
# ---------------------------------------------------------------------------
mkdir -p "${CHECKPOINTS_DIR}"
mkdir -p "${TENSORBOARD_LOG_DIR}"
mkdir -p "$(dirname "${MODEL_OUTPUTS_JSONL}")"
mkdir -p "$(dirname "${LOG_FILE}")"

# ---------------------------------------------------------------------------
# 3.  Environment variables
#     - wandb fully disabled (no API key needed)
#     - XFORMERS backend for vLLM (stable on A100)
#     - HF cache pointed inside the bind-mounted repo tree so the container
#       doesn't try to write to a read-only location
# ---------------------------------------------------------------------------
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR}"
export MODEL_OUTPUTS_JSONL="${MODEL_OUTPUTS_JSONL}"

# Prevent Python inside the container from loading user's ~/.local/ site-packages
export PYTHONNOUSERSITE=1

# VLLM_ATTENTION_BACKEND=XFORMERS removed — vLLM 0.17 uses FlashAttention2 natively
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export LD_LIBRARY_PATH='/tmp/joel_cupti_fix:/.singularity.d/libs'

# Preload NVIDIA CUDA libs from the container's nvidia Python packages.
# Fixes dynamic-linker errors for libcupti / libnccl / libcublas at startup.
# APPTAINERENV_* vars are forwarded into the container automatically.
_NV="/usr/local/lib/python3.12/dist-packages/nvidia"
export APPTAINERENV_LD_PRELOAD="\
${_NV}/cuda_cupti/lib/libcupti.so.12:\
${_NV}/nccl/lib/libnccl.so.2:\
${_NV}/cublas/lib/libcublas.so.12:\
${_NV}/cublas/lib/libcublasLt.so.12"

# HPC Hygiene block
export PYTHONUNBUFFERED=1
export TF_ENABLE_ONEDNN_OPTS=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Use PYTHONPATH instead of pip install -e . — the container filesystem is
# read-only so pip cannot write to site-packages.  PYTHONPATH prepends the
# repo to Python's search path so the local patched verl/ takes precedence
# over the container's built-in verl without touching the container at all.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# HF cache — redirect to bind-mounted path; container's /root/.cache is read-only
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_HOME}"

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 5.  Launch training
#     All Hydra overrides are explicit — nothing inherited from ppo_trainer.yaml
#     Tuned for 2×A100-40GB.
#
#     IMPORTANT: Do NOT put bash comments (#) inside the python3 \ command —
#     they break the backslash line-continuation and python3 gets no arguments.
#     All section labels are kept here above the command instead.
#
#     Sections (in order below):
#       Algorithm | Data | Reward | Model | Actor optimiser | Actor loss
#       Actor KL+entropy | Actor FSDP | Ref FSDP | Critic paths
#       vLLM engine | Rollout sampling (train) | Rollout sampling (val)
#       Trainer
# ---------------------------------------------------------------------------
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.kl_penalty=kl \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=128 \
  data.val_batch_size=256 \
  data.max_prompt_length=1024 \
  data.max_response_length=3072 \
  +data.dataloader_num_workers=4 \
  reward_model.reward_manager='naive' \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.warmup_style=constant \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.model.path="${MODEL_PATH}" \
  critic.model.tokenizer_path="${MODEL_PATH}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.max_num_seqs=1024 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.disable_log_stats=True \
  +actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.ignore_eos=False \
  actor_rollout_ref.rollout.n=8 \
  +actor_rollout_ref.rollout.n_val=1 \
  +actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  +actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  +actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  +actor_rollout_ref.rollout.val_kwargs.n=1 \
  trainer.project_name='one_shot_rlvr' \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.logger=['console','tensorboard'] \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.critic_warmup=0 \
  trainer.total_training_steps=1500 \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.save_freq=20 \
  trainer.test_freq=20 \
  trainer.checkpoints_dir="${CHECKPOINTS_DIR}" \
  trainer.remove_previous_ckpt_in_save=True \
  trainer.resume_mode=auto \
  +trainer.val_before_train=True \
  +trainer.log_val_generations=0 \
  2>&1 | tee "${LOG_FILE}"
