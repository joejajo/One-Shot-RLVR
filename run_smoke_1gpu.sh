#!/bin/bash
# =============================================================================
# run_smoke_1gpu.sh
# Quick smoke test — 1×A100, 3 epochs, tiny batch — verifies the full
# training stack starts correctly on woody.
#
# Fixed from original:
#   - checkpoint dir: output/ (was outputs/ — typo)
#   - tee log path:   absolute under output/logs/
#   - data paths:     absolute
# =============================================================================
set -x

REPO_ROOT=/home/woody/iwi7/iwi7107h/One-Shot-RLVR
MODEL_PATH=/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B
SMOKE_CKPT_DIR=${REPO_ROOT}/output/smoke_checkpoints/1gpu
LOG_FILE=${REPO_ROOT}/output/logs/smoke_1gpu.log

mkdir -p "${SMOKE_CKPT_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export WANDB_MODE=disabled

cd "${REPO_ROOT}"

python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files="${REPO_ROOT}/data/train/one_shot_rlvr/pi1_r128.parquet" \
 data.val_files="${REPO_ROOT}/data/test/math500.parquet" \
 data.train_batch_size=8 \
 data.val_batch_size=16 \
 data.max_prompt_length=512 \
 data.max_response_length=512 \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path="${MODEL_PATH}" \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.temperature=0.6 \
 +actor_rollout_ref.rollout.val_temperature=0.6 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
 actor_rollout_ref.rollout.n=2 \
 +actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.critic_warmup=0 \
 trainer.logger=['console'] \
 trainer.project_name='smoke_test' \
 trainer.experiment_name='1gpu_smoke' \
 trainer.checkpoints_dir="${SMOKE_CKPT_DIR}" \
 +trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=-1 \
 trainer.test_freq=-1 \
 trainer.default_hdfs_dir=null \
 trainer.total_epochs=3 \
 2>&1 | tee "${LOG_FILE}"

echo "=================================================="
echo "1-GPU smoke test finished. Log: ${LOG_FILE}"
echo "=================================================="
