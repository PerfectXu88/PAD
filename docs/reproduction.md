# Reproducing PAD

Run from the repository root after installation in the [README](../README.md#getting-started). Commands below use DiffusionDrive. Use a separate output directory for each policy so its checkpoints, calibration outcomes, and rankings remain paired.

## Environment and data

```bash
export NAVSIM_DEVKIT_ROOT="$PWD"
export OPENSCENE_DATA_ROOT=/path/to/navsim/dataset
export NAVSIM_EXP_ROOT=/path/to/experiment/output
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
```

Prepare NAVSIM logs, sensor blobs, maps, and the synthetic rounds using the [data instructions](install.md) and [download helpers](../download/). Place the ResNet-34 backbone at `$OPENSCENE_DATA_ROOT/models/resnet34_model.bin`.

| Training setting | DiffusionDrive | LTF | MeanFuser |
| --- | --- | --- | --- |
| Optimizer | AdamW | Adam | AdamW |
| Learning rate | 6e-4 | 1e-4 | 1e-4 |
| Weight decay | 1e-4 | 0 | 0 |
| Schedule | 3-epoch warmup + cosine | Constant | 3-epoch warmup + cosine |
| Minimum learning rate | 1e-6 | - | 1e-7 |

All use 100 epochs, 64 samples per GPU, four A800 GPUs, and seed 0. Paper evaluation uses four L40 GPUs.

## 1. Fix the held-out calibration scenes

Create this manifest once, before base-policy training, and share it across all three policies:

```bash
python -m calibration.build_calibration_split \
  --n-per-source 4000 --seed 2026 \
  --manifest-out "$NAVSIM_EXP_ROOT/calibration/manifest.csv"
export REAL_TOKEN_EXCLUDELIST="$NAVSIM_EXP_ROOT/calibration/manifest.tokens.txt"
```

This samples only real `navtrain` training logs, writes `calib_C_navtrain` split/filter YAMLs into the runtime, and emits the token exclusion file. Reuse the exact manifest and generated YAMLs when running other policies. Do not resample between methods. All training commands below retain this exclusion variable.

## 2. Build policy feature caches

```bash
export MODEL=diffusiondrive
TRAIN_TEST_SPLIT=navtrain bash scripts/training/run_dataset_cache.sh
for expert in pdm recovery; do
  for round in 0 1 2 3 4; do
    TRAIN_TEST_SPLIT="navtrain_reaction_${expert}_v1.0-${round}" \
      bash scripts/training/run_dataset_cache.sh
  done
done
```

The synthetic data must already be downloaded/prepared. The training-log split YAMLs define candidate membership. The paper uses 208,413 candidates from both experts across rounds 0-4. Compare the resulting count with your release's manifest before running paper-scale experiments. Cache enumeration raises on missing configured logs and duplicate tokens; valuation also requires all feature/target files for every enumerated candidate.

| `MODEL` | Cache prefix | Valuation module | Geometry scenes |
| --- | --- | --- | ---: |
| `diffusiondrive` | empty | `influence.compute_psi` | 4000 |
| `ltf` | `ltf_` | `influence.compute_psi_ltf` | 4000 |
| `meanfuser` | `meanfuser_` | `influence.compute_psi_meanfuser` | 2000 |

For example, LTF real features are in `$NAVSIM_EXP_ROOT/cache/ltf_navtrain_cache`. Rebuild features for each policy; do not substitute another policy's cache.

## 3. Train the real-only base policy

```bash
unset SYN_IDX SYN_GT SYN_TOKEN_ALLOWLIST BASE_CHECKPOINT RESUME_CHECKPOINT
EXPERIMENT_NAME=base_dd bash scripts/training/run_training.sh
export BASE_CHECKPOINT="$NAVSIM_EXP_ROOT/base_dd/last.ckpt"
```

The launcher defaults to 4 GPUs, 64 samples per GPU, 100 epochs, and seed 0. `BASE_CHECKPOINT` initializes policy weights for a new training run; it does not resume optimizer state. To resume an interrupted run explicitly, set `RESUME_CHECKPOINT` to an existing absolute path or a path relative to `$NAVSIM_EXP_ROOT`. A missing resume checkpoint is an error. New runs never automatically resume a previous experiment.

## 4. Evaluate the base policy on calibration scenes

```bash
TRAIN_TEST_SPLIT=calib_C_navtrain \
  bash scripts/evaluation/run_metric_caching.sh
CHECKPOINT="$BASE_CHECKPOINT" SPLIT=calib_C_navtrain \
  EXPERIMENT_NAME=r002_calib_C_navtrain \
  bash scripts/evaluation/run_evaluation.sh
python -m calibration.merge_calibration_epdms \
  --eval-root "$NAVSIM_EXP_ROOT" --sources navtrain \
  --manifest "$NAVSIM_EXP_ROOT/calibration/manifest.csv" \
  --out "$NAVSIM_EXP_ROOT/calibration/C_epdms_dd.csv"
```

The merger reads the latest timestamped evaluation CSV under `r002_calib_C_navtrain/`, removes aggregate rows, and requires one finite outcome for every manifest token. Each policy must use its **own real-only checkpoint and outcomes**, even though scene identities are shared.

## 5. Compute policy-specific values

```bash
python -m influence.compute_psi \
  --checkpoint "$BASE_CHECKPOINT" \
  --c-epdms "$NAVSIM_EXP_ROOT/calibration/C_epdms_dd.csv" \
  --calib-cache "$NAVSIM_EXP_ROOT/cache/navtrain_cache" \
  --hessian-cache "$NAVSIM_EXP_ROOT/cache/navtrain_cache" \
  --d-proj 2048 --lambda-reg 0.01 --hessian-n 4000 --seed 0 \
  --out "$NAVSIM_EXP_ROOT/valuation/psi_dd.csv"
```

`--hessian-cache` and `--hessian-n` are CLI names for the **real-data gradient second-moment geometry**, not an exact Hessian. Geometry sampling uses training logs and excludes all calibration tokens. Outcome normalization is global, using population standard deviation plus `1e-8`. The calibration direction is a sum, not a mean. Projection is Gaussian with variance `1/p`, regenerated with the same seed, parameter order, and chunk size for every gradient.

For LTF/MeanFuser, change the module, checkpoint, outcomes, and cache paths according to the table. MeanFuser defaults to `--hessian-n 2000`; LTF defaults to 4000. `--max-pool` is a debugging option: partial score files cannot be used to select from the complete pool.

Each value CSV has a companion `.manifest.json` recording arguments, checkpoint SHA-256, attributed parameter names/shapes, and calibration/geometry/candidate token identities.

## 6. Select and retrain

```bash
python -m selection.select_synthetic \
  --method psi --fraction 0.25 \
  --psi-csv "$NAVSIM_EXP_ROOT/valuation/psi_dd.csv" \
  --out "$NAVSIM_EXP_ROOT/selection/dd_pad25.txt"

SYN_IDX=4 SYN_GT=pdm,recovery \
  SYN_TOKEN_ALLOWLIST="$NAVSIM_EXP_ROOT/selection/dd_pad25.txt" \
  EXPERIMENT_NAME=dd_pad25 \
  bash scripts/training/run_training.sh
```

The exported base checkpoint initializes this run, while every policy parameter remains trainable. The training dataset combines real scenes outside calibration with exactly the selected synthetic tokens. Validation remains real-only. Missing selected caches raise an error.

Use fractions `0.25`, `0.50`, and `0.75`. The budget is `round(fraction * pool_size)` (at least one), with deterministic token-order tie breaking. There is no positivity threshold or per-source quota. LTF/MeanFuser selection needs `--cache-prefix ltf_` / `--cache-prefix meanfuser_`; their training launchers set the matching prefix.

For matched baselines:

- **Random:** use `--method random --fraction ... --seed 2026`, then train with its allowlist and the same base checkpoint/protocol.
- **Full pool:** select with `--method random --fraction 1.0` and train with that allowlist.
- **PDM only:** use `--experts pdm --method random --fraction 1.0`; train with `SYN_GT=pdm`.
- **Real only:** unset synthetic variables as in step 3.

## 7. Final evaluation

```bash
TRAIN_TEST_SPLIT=navtest bash scripts/evaluation/run_metric_caching.sh
CHECKPOINT="$NAVSIM_EXP_ROOT/dd_pad25/last.ckpt" SPLIT=navtest \
  EXPERIMENT_NAME=eval_dd_pad25_navtest \
  bash scripts/evaluation/run_evaluation.sh

TRAIN_TEST_SPLIT=navhard_two_stage bash scripts/evaluation/run_metric_caching.sh
CHECKPOINT="$NAVSIM_EXP_ROOT/dd_pad25/last.ckpt" SPLIT=navhard \
  EXPERIMENT_NAME=eval_dd_pad25_navhard \
  bash scripts/evaluation/run_evaluation.sh
```

The evaluation launcher defaults to four GPU processes. NavTest uses the official one-stage entrypoint; NavHard uses the two-stage entrypoint. Paper tables report EPDMS and components on the 0-100 scale.

## Group-removal fidelity

```bash
python -m influence.make_loo_groups \
  --psi-csv "$NAVSIM_EXP_ROOT/valuation/psi_dd.csv" \
  --n-groups 5 --out-dir "$NAVSIM_EXP_ROOT/removal"
```

Train `keep_full.txt` and each `keep_remove_g0.txt` through `keep_remove_g4.txt` from the **same real-only base checkpoint**, with identical settings. Evaluate all six runs on the same evaluation protocol. Supply a CSV with `group_id,epdms`, using `-1` for the full-pool reference and `0`-`4` for removal runs:

```bash
python -m influence.analyze_spearman \
  --groups "$NAVSIM_EXP_ROOT/removal/groups.csv" \
  --epdms /path/to/removal_epdms.csv \
  --out "$NAVSIM_EXP_ROOT/removal/report.md"
```

The report uses `EPDMS_after_removal - EPDMS_full`. The expected correlation with mean group value is **negative**. It reports signed Spearman correlation, exact negative-alternative and two-sided permutation p-values, and precision among the two lowest-valued groups. It does not assign a pass/fail verdict using unreported thresholds. Exact enumeration is restricted to 2-8 groups.

## Artifact schemas

| Artifact | Contents |
| --- | --- |
| `manifest.csv` | `token,log_name,source_split,source_type,round` |
| `manifest.tokens.txt` | Calibration tokens excluded from policy training |
| `C_epdms_*.csv` | Manifest columns plus the target policy's `epdms` |
| `psi_*.csv` | `token,psi,source_type,round` |
| `psi_*.manifest.json` | Checkpoint identity, settings, parameter order, and token identities |
| Selection allowlist | One selected synthetic token per line |

Keep these artifacts with each experiment. They identify the data, policy, and settings underlying a result.

## Protocol checks

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

The checks cover global calibration normalization, forbidden calibration sources, duplicate/missing candidates, fixed-budget ranking (including negative scores), and signed removal statistics. With PyTorch installed, they also check the analytic influence solve, shared linear projection, and zero gradients for unused parameters.
