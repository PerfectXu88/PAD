# Installation

Create the environment and install the local package from the repository root:

```bash
conda env create -f environment.yml
conda activate pad
pip install -e .
```

Download the NAVSIM maps, logs, sensor blobs, and SimScale synthetic sources required by your experiment. The scripts in `download/` provide the standard NAVSIM splits; SimScale synthetic data must follow the same layout documented in the [reproduction guide](reproduction.md).

Set the runtime paths:

```bash
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/path/to/dataset/maps
export NAVSIM_EXP_ROOT=/path/to/experiment/output
export NAVSIM_DEVKIT_ROOT="$PWD"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OPENSCENE_DATA_ROOT=/path/to/dataset
```

All three models use ResNet34 weights at:

```text
$OPENSCENE_DATA_ROOT/models/resnet34_model.bin
```

DiffusionDrive trajectory anchors are included at
`traj_final/kmeans_navsim_traj_20.npy`. MeanFuser noise statistics are included
at `tools/gaussian_mixed_noise/navtrain_8_mean_std.pkl`.
