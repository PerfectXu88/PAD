# PAD documentation

Run installation, training, valuation, and evaluation commands from the repository root.

| Guide | Contents |
| --- | --- |
| [Installation](install.md) | Environment, package installation, data paths, and model assets |
| [Reproduction](reproduction.md) | Calibration, policy-specific valuation, selection, training, evaluation, and protocol checks |
| [Dataset splits](splits.md) | NAVSIM scene filters and dataset splits |
| [Caching](cache.md) | Metric cache preparation |
| [Metrics](metrics.md) | EPDMS evaluation and component metrics |

## Repository layout

```text
README.md               Project introduction and results
environment.yml         Conda environment
requirements.txt        Python dependencies
setup.py                Editable package installation
calibration/            Held-out scene construction and outcome merging
influence/              Policy adapters, projected valuation, removal analysis
selection/              Candidate-pool indexing and fixed-budget selection
navsim/                 Driving policies, simulation, training, and evaluation
scripts/                Training, caching, and evaluation launchers
tests/                  Protocol regression checks
docs/                   Setup and reproduction documentation
download/               Dataset download helpers
tools/                  Data preparation utilities and noise statistics
traj_final/             Trajectory anchors
assets/                 README figures and the standalone project-page/
```

Set `NAVSIM_DEVKIT_ROOT` to this repository root. For example:

```bash
export NAVSIM_DEVKIT_ROOT="$PWD"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m selection.select_synthetic --help
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

The valuation entrypoints are `influence.compute_psi`,
`influence.compute_psi_ltf`, and `influence.compute_psi_meanfuser`.
