"""Create a reproducible held-out calibration split and provenance manifest."""

import argparse
import csv
import os
import random
import re
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader


def source_metadata(source: str) -> tuple[str, int]:
    if "reaction_pdm" in source:
        source_type = "pdm"
    elif "reaction_recovery" in source:
        source_type = "recovery"
    else:
        source_type = "real"
    match = re.search(r"v1\.0-(\d+)", source)
    return source_type, int(match.group(1)) if match else -1


def enumerate_tokens(
    config_root: Path,
    dataset_root: Path,
    source: str,
) -> tuple[str, object, dict[str, list[str]]]:
    split_config = OmegaConf.load(config_root / "train_test_split" / f"{source}.yaml")
    scene_filter_config = OmegaConf.load(
        config_root / "train_test_split" / "scene_filter" / f"{source}.yaml"
    )
    scene_loader = SceneLoader(
        data_path=dataset_root / "navsim_logs" / split_config.data_split,
        original_sensor_path=None,
        scene_filter=instantiate(scene_filter_config),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    return split_config.data_split, scene_filter_config, scene_loader.get_tokens_list_per_log()


def sample_tokens(
    tokens_by_log: dict[str, list[str]],
    sample_count: int,
    random_generator: random.Random,
) -> list[tuple[str, str]]:
    population = sorted(
        (token, log_name)
        for log_name, tokens in tokens_by_log.items()
        for token in tokens
    )
    if sample_count > len(population):
        raise ValueError("Insufficient scenes for the fixed calibration manifest")
    return random_generator.sample(population, sample_count)


def write_split_configs(
    config_root: Path,
    source: str,
    data_split: str,
    source_filter,
    samples: list[tuple[str, str]],
) -> None:
    scene_filter = OmegaConf.create(
        {
            "_convert_": "all",
            "_target_": "navsim.common.dataclasses.SceneFilter",
            "num_history_frames": int(source_filter.get("num_history_frames", 4)),
            "num_future_frames": int(source_filter.get("num_future_frames", 10)),
            "frame_interval": int(source_filter.get("frame_interval", 1)),
            "has_route": bool(source_filter.get("has_route", True)),
            "max_scenes": None,
            "log_names": sorted({log_name for _, log_name in samples}),
            "tokens": [token for token, _ in samples],
        }
    )
    OmegaConf.save(
        scene_filter,
        config_root / "train_test_split" / "scene_filter" / f"calib_C_{source}.yaml",
    )
    OmegaConf.save(
        OmegaConf.create(
            {
                "defaults": [{"scene_filter": f"calib_C_{source}"}],
                "data_split": data_split,
            }
        ),
        config_root / "train_test_split" / f"calib_C_{source}.yaml",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", choices=["navtrain"], default="navtrain")
    parser.add_argument("--n-per-source", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--manifest-out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_per_source <= 0:
        raise ValueError("--n-per-source must be positive")

    devkit_root = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
    dataset_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    config_root = devkit_root / "navsim/planning/script/config/common"
    sources = [source.strip() for source in args.sources.split(",") if source.strip()]
    random_generator = random.Random(args.seed)

    manifest: list[dict] = []
    for source in sources:
        source_type, round_index = source_metadata(source)
        data_split, source_filter, tokens_by_log = enumerate_tokens(
            config_root,
            dataset_root,
            source,
        )
        training_split = OmegaConf.load(config_root.parent / "training/default_train_val_test_log_split.yaml")
        train_logs = set(training_split.train_logs)
        tokens_by_log = {log: tokens for log, tokens in tokens_by_log.items() if log in train_logs}
        samples = sample_tokens(tokens_by_log, args.n_per_source, random_generator)
        write_split_configs(config_root, source, data_split, source_filter, samples)
        manifest.extend(
            {
                "token": token,
                "log_name": log_name,
                "source_split": source,
                "source_type": source_type,
                "round": round_index,
            }
            for token, log_name in samples
        )
        print(f"[calibration] {source}: selected {len(samples)} scenes")

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["token", "log_name", "source_split", "source_type", "round"],
        )
        writer.writeheader()
        writer.writerows(manifest)
    manifest_path.with_suffix(".tokens.txt").write_text("\n".join(row["token"] for row in manifest) + "\n")
    print(f"[calibration] wrote {len(manifest)} scenes to {manifest_path}")


if __name__ == "__main__":
    main()
