#!/bin/bash
# =============================================================================
# training_1.5b_pi1_r128_4xa100.sh
# One-Shot-RLVR training — Qwen2.5-Math-1.5B, π₁ r128, 4×A100, single node
# (original 4×A100 script — updated for HPC, no wandb, absolute paths)
#
# Run with:
#   conda activate rlvr_train
#   bash scripts/train/training_1.5b_pi1_r128_4xa100.sh
# =============================================================================
set -euo pipefail
set -x

REPO_ROOT=/home/woody/iwi7/iwi7107h/One-Shot-RLVR
MODEL_PATH=/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B
TRAIN_FILE=${REPO_ROOT}/data/train/one_shot_rlvr/pi1_r128.parquet
VAL_FILE=${REPO_ROOT}/data/test/math500.parquet
CHECKPOINTS_DIR=${REPO_ROOT}/output/checkpoints
TENSORBOARD_LOG_DIR=${REPO_ROOT}/output/tensorboard/verl_few_shot/Qwen2.5-Math-1.5B-pi1_r128_4xa100
MODEL_OUTPUTS_JSONL=${REPO_ROOT}/output/model_outputs/val_generations_pi1_r128.jsonl
LOG_FILE=${REPO_ROOT}/output/logs/training_pi1_r128_4xa100.log

mkdir -p "${CHECKPOINTS_DIR}"
mkdir -p "${TENSORBOARD_LOG_DIR}"
mkdir -p "$(dirname "${MODEL_OUTPUTS_JSONL}")"
mkdir -p "$(dirname "${LOG_FILE}")"

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR}"
export MODEL_OUTPUTS_JSONL="${MODEL_OUTPUTS_JSONL}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN

cd "${REPO_ROOT}"

python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files="${TRAIN_FILE}" \
 data.val_files="${VAL_FILE}" \
 data.train_batch_size=128 \
 data.val_batch_size=530 \
 data.max_prompt_length=1024 \
 data.max_response_length=3072 \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path="${MODEL_PATH}" \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.temperature=0.6 \
 +actor_rollout_ref.rollout.val_temperature=0.6 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.rollout.n=8 \
 +actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.critic_warmup=0 \
 trainer.logger=['console','tensorboard'] \
 trainer.val_generations_to_log_to_wandb=0 \
 trainer.project_name='verl_few_shot' \
 trainer.experiment_name='Qwen2.5-Math-1.5B-pi1_r128_4xa100' \
 trainer.checkpoints_dir="${CHECKPOINTS_DIR}" \
 +trainer.val_before_train=True \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.save_freq=20 \
 trainer.test_freq=20 \
 trainer.default_hdfs_dir=null \
 trainer.total_epochs=2000 \
 2>&1 | tee "${LOG_FILE}"