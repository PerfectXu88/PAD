"""Enumerate the synthetic cache entries visible to the training loader."""

import os
from pathlib import Path

import yaml


def parse_rounds(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec:
        lower, upper = spec.split("-", maxsplit=1)
        return list(range(int(lower), int(upper) + 1))
    return [int(value) for value in spec.split(",") if value.strip()]


def enumerate_synthetic_pool(
    experts: list[str],
    rounds: list[int],
    split: str = "train",
    cache_prefix: str = "",
) -> dict[str, tuple[Path, str, int]]:
    """Return token-to-cache metadata constrained by each split configuration."""
    devkit_root = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
    cache_root = Path(os.environ["NAVSIM_EXP_ROOT"]) / "cache"
    config_root = devkit_root / "navsim/planning/script/config/training"

    index: dict[str, tuple[Path, str, int]] = {}
    for expert in experts:
        for round_index in rounds:
            config_path = config_root / (
                f"default_log_split_synthetic_reation_{expert}_v1.0-{round_index}.yaml"
            )
            if not config_path.is_file():
                raise FileNotFoundError(config_path)
            with config_path.open() as stream:
                log_names = yaml.safe_load(stream)[f"{split}_logs"]
            cache_dir = cache_root / (
                f"{cache_prefix}navtrain_reaction_{expert}_v1.0-{round_index}_cache"
            )
            for log_name in log_names:
                log_path = cache_dir / log_name
                if not log_path.is_dir():
                    raise FileNotFoundError(log_path)
                for token_path in sorted(log_path.iterdir()):
                    if token_path.is_dir():
                        if token_path.name in index:
                            raise ValueError(f"Duplicate synthetic token: {token_path.name}")
                        index[token_path.name] = (token_path, expert, round_index)
    return index
