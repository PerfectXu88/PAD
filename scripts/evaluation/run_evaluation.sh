#!/usr/bin/env bash
set -euo pipefail

: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT}"
: "${CHECKPOINT:?Set CHECKPOINT}"

MODEL="${MODEL:-diffusiondrive}"
SPLIT="${SPLIT:-navtest}"
NPROC="${NPROC:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-evaluation/${MODEL}_${SPLIT}}"

case "${MODEL}" in
  diffusiondrive)
    AGENT="diffusiondrive_agent"
    SUFFIX="diffusiondrive"
    EXTRA_AGENT_ARGS=()
    ;;
  ltf)
    AGENT="transfuser_agent"
    SUFFIX="transfuser"
    EXTRA_AGENT_ARGS=(agent.config.latent=true)
    ;;
  meanfuser)
    AGENT="meanfuser_agent"
    SUFFIX="transfuser"
    EXTRA_AGENT_ARGS=()
    ;;
  *)
    echo "MODEL must be diffusiondrive, ltf, or meanfuser" >&2
    exit 2
    ;;
esac

if [[ "${SPLIT}" == "navhard" ]]; then
  ENTRYPOINT="run_pdm_score_gpu_v2_${SUFFIX}.py"
  CONFIG_SPLIT="navhard_two_stage"
  METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/cache/navhard_two_stage_metric_cache}"
else
  ENTRYPOINT="run_pdm_score_one_stage_gpu_${SUFFIX}.py"
  CONFIG_SPLIT="${SPLIT}"
  METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/cache/${SPLIT}_metric_cache}"
fi

torchrun --nproc_per_node="${NPROC}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/${ENTRYPOINT}" \
  agent="${AGENT}" \
  "${EXTRA_AGENT_ARGS[@]}" \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  agent.checkpoint_path="${CHECKPOINT}" \
  experiment_name="${EXPERIMENT_NAME}" \
  +cache_path=null \
  metric_cache_path="${METRIC_CACHE_PATH}" \
  train_test_split="${CONFIG_SPLIT}"
