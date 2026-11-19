#!/usr/bin/env bash
set -euo pipefail

: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT}"
: "${TRAIN_TEST_SPLIT:?Set TRAIN_TEST_SPLIT}"

MODEL="${MODEL:-diffusiondrive}"
WORKERS="${WORKERS:-32}"

case "${MODEL}" in
  diffusiondrive)
    AGENT="diffusiondrive_agent"
    DEFAULT_CACHE="${NAVSIM_EXP_ROOT}/cache/${TRAIN_TEST_SPLIT}_cache"
    ;;
  ltf)
    AGENT="transfuser_agent"
    DEFAULT_CACHE="${NAVSIM_EXP_ROOT}/cache/ltf_${TRAIN_TEST_SPLIT}_cache"
    ;;
  meanfuser)
    AGENT="meanfuser_agent"
    DEFAULT_CACHE="${NAVSIM_EXP_ROOT}/cache/meanfuser_${TRAIN_TEST_SPLIT}_cache"
    ;;
  *)
    echo "MODEL must be diffusiondrive, ltf, or meanfuser" >&2
    exit 2
    ;;
esac

CACHE_PATH="${CACHE_PATH:-${DEFAULT_CACHE}}"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching.py" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  agent="${AGENT}" \
  experiment_name="cache_${MODEL}_${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  worker.threads_per_node="${WORKERS}"
