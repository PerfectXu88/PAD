#!/usr/bin/env bash
set -euo pipefail

: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT}"
: "${TRAIN_TEST_SPLIT:?Set TRAIN_TEST_SPLIT}"

METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/cache/${TRAIN_TEST_SPLIT}_metric_cache}"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  metric_cache_path="${METRIC_CACHE_PATH}"
