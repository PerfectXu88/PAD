#!/usr/bin/env bash
set -euo pipefail

: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT}"
: "${REAL_TOKEN_EXCLUDELIST:?Set the fixed calibration token exclusion file}"
[[ -f "$REAL_TOKEN_EXCLUDELIST" ]] || { echo "Calibration token file not found" >&2; exit 2; }

MODEL="${MODEL:-diffusiondrive}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"

case "${MODEL}" in
  diffusiondrive)
    ENTRYPOINT="run_training_diffusiondrive.py"
    AGENT="diffusiondrive_agent"
    LEARNING_RATE="${LEARNING_RATE:-6e-4}"
    CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/cache/navtrain_cache}"
    EXTRA_AGENT_ARGS=(+agent.config.ckpt_path="${EXPERIMENT_NAME:-train_diffusiondrive}")
    ;;
  ltf)
    ENTRYPOINT="run_training_transfuser.py"
    AGENT="transfuser_agent"
    LEARNING_RATE="${LEARNING_RATE:-1e-4}"
    CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/cache/ltf_navtrain_cache}"
    EXTRA_AGENT_ARGS=(agent.config.latent=true +agent.config.ckpt_path="${EXPERIMENT_NAME:-train_ltf}")
    export SYN_CACHE_PREFIX="${SYN_CACHE_PREFIX:-ltf_}"
    ;;
  meanfuser)
    ENTRYPOINT="run_training_meanfuser.py"
    AGENT="meanfuser_agent"
    LEARNING_RATE="${LEARNING_RATE:-1e-4}"
    CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/cache/meanfuser_navtrain_cache}"
    EXTRA_AGENT_ARGS=()
    export SYN_CACHE_PREFIX="${SYN_CACHE_PREFIX:-meanfuser_}"
    ;;
  *)
    echo "MODEL must be diffusiondrive, ltf, or meanfuser" >&2
    exit 2
    ;;
esac

EXPERIMENT_NAME="${EXPERIMENT_NAME:-train_${MODEL}}"
CHECKPOINT_ARGS=()
if [[ -n "${BASE_CHECKPOINT:-}" ]]; then
  [[ -f "$BASE_CHECKPOINT" ]] || { echo "Base checkpoint not found" >&2; exit 2; }
  CHECKPOINT_ARGS+=(agent.checkpoint_path="$BASE_CHECKPOINT")
fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  CHECKPOINT_ARGS+=(+resume_ckpt_path="$RESUME_CHECKPOINT")
fi

torchrun --nproc_per_node="${NUM_GPUS}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/${ENTRYPOINT}" \
  --config-name default_training \
  agent="${AGENT}" \
  "${EXTRA_AGENT_ARGS[@]}" \
  "${CHECKPOINT_ARGS[@]}" \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split=navtrain \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  +trainer.params.devices="${NUM_GPUS}" \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  agent.lr="${LEARNING_RATE}" \
  cache_path="${CACHE_PATH}" \
  use_cache_without_dataset=true \
  force_cache_computation=false
