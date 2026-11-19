# Understanding the data format and classes

OpenScene is a compact redistribution of the large-scale [nuPlan dataset](https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/index.html), retaining only relevant annotations and sensor data at 2Hz. This reduces the dataset size by a factor of >10. The data used in NAVSIM is structured into `navsim.common.dataclasses.Scene` objects. A `Scene` is a list of `Frame` objects, each containing the required inputs and annotations for training a planning `Agent`.

**Caching.** Evaluating planners involves significant preprocessing of the raw annotation data, including accessing the global map at each ´Frame´ and converting it into a local coordinate system. You can generate the cache with:
```bash
TRAIN_TEST_SPLIT=navtest \
  bash "$NAVSIM_DEVKIT_ROOT/scripts/evaluation/run_metric_caching.sh"
```

This creates the cache under `$NAVSIM_EXP_ROOT/cache/navtest_metric_cache` by default.
