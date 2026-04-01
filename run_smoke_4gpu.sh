#!/bin/bash
# =============================================================================
# run_smoke_4gpu.sh
# Quick smoke test — 4×A100, 1 epoch, tiny batch — verifies the full
# training stack (Ray, FSDP, vLLM, reward) starts correctly on woody.
#
# Fixed from original:
#   - model path: Qwen2.5-Math-1.5B (base, matches training setup)
#   - checkpoint dir: inside repo under output/
#   - data paths:     absolute
#   - WANDB_MODE:     disabled (not just offline)
#   - added WANDB_DISABLED=true
#   - added set -euo pipefail + shebang
# =============================================================================
#!/bin/bash
set -euo pipefail

REPO_ROOT=/home/woody/iwi7/iwi7107h/One-Shot-RLVR
MODEL_PATH=/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B
SMOKE_CKPT_DIR=${REPO_ROOT}/output/smoke_checkpoints/4gpu
LOG_FILE=${REPO_ROOT}/output/logs/smoke_4gpu.log

mkdir -p "${SMOKE_CKPT_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"

ray stop -f || true
export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset VLLM_ATTENTION_BACKEND

cd "${REPO_ROOT}"

python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch cuda device count:", torch.cuda.device_count())
PY

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${REPO_ROOT}/data/train/one_shot_rlvr/pi1_r128.parquet" \
  data.val_files="${REPO_ROOT}/data/test/math500.parquet" \
  data.train_batch_size=4 \
  data.val_batch_size=4 \
  data.max_prompt_length=256 \
  data.max_response_length=256 \
  reward_model.reward_manager=naive \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_think=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.18 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.n=1 \
  +actor_rollout_ref.rollout.n_val=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  +actor_rollout_ref.ref.micro_batch_size_per_gpu=1 \
  +actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  +actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  critic.model.path="${MODEL_PATH}" \
  critic.model.tokenizer_path="${MODEL_PATH}" \
  trainer.logger="['console']" \
  trainer.project_name=one_shot_rlvr \
  trainer.experiment_name=smoke_pi1_4gpu \
  trainer.checkpoints_dir="${SMOKE_CKPT_DIR}" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  trainer.total_epochs=1 \
  2>&1 | tee "${LOG_FILE}"

echo "=================================================="
echo "4-GPU smoke test finished. Log: ${LOG_FILE}"
echo "=================================================="
