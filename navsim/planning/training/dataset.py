import gzip
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

import torch
from tqdm import tqdm

from navsim.common.dataclasses import AgentInput
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

logger = logging.getLogger(__name__)


def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)


class CacheOnlyDataset(torch.utils.data.Dataset):
    """Load real and optionally selected synthetic samples from feature caches."""

    def __init__(
        self,
        cache_path: str,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: Optional[List[str]] = None,
        split: str = "train",
    ):
        super().__init__()
        self._cache_path = Path(cache_path)
        if not self._cache_path.is_dir():
            raise FileNotFoundError(f"Cache path does not exist: {cache_path}")
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        if log_names is not None:
            self.log_names = [
                Path(log_name)
                for log_name in log_names
                if (self._cache_path / log_name).is_dir()
            ]
        else:
            self.log_names = sorted(path.name for path in self._cache_path.iterdir() if path.is_dir())

        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self.split = split
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            cache_path=self._cache_path,
            feature_builders=self._feature_builders,
            target_builders=self._target_builders,
            log_names=self.log_names,
        )
        self.tokens = sorted(self._valid_cache_paths)

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self.tokens)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
        return self._load_scene_with_token(self.tokens[idx])

    @staticmethod
    def _has_required_files(
        token_path: Path,
        builders: List[AbstractFeatureBuilder],
    ) -> bool:
        return all((token_path / f"{builder.get_unique_name()}.gz").is_file() for builder in builders)

    def _load_valid_caches(
        self,
        cache_path: Path,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: List[Path],
    ) -> Dict[str, Path]:
        valid_cache_paths: Dict[str, Path] = {}
        builders = feature_builders + target_builders

        for log_name in tqdm(log_names, desc="Loading Valid Caches"):
            log_path = cache_path / log_name
            for token_path in sorted(log_path.iterdir()):
                if token_path.is_dir() and self._has_required_files(token_path, builders):
                    valid_cache_paths[token_path.name] = token_path

        excluded_path = os.environ.get("REAL_TOKEN_EXCLUDELIST")
        if excluded_path:
            excluded = set(Path(excluded_path).read_text().split())
            valid_cache_paths = {token: path for token, path in valid_cache_paths.items() if token not in excluded}

        if self.split != "train":
            return valid_cache_paths
        max_round = os.environ.get("SYN_IDX")
        if max_round is None:
            return valid_cache_paths

        expert_spec = os.environ.get("SYN_GT")
        if not expert_spec:
            raise ValueError("SYN_GT is required when SYN_IDX is set")
        allowlist = None
        allowlist_path = os.environ.get("SYN_TOKEN_ALLOWLIST")
        if allowlist_path:
            allowlist_file = Path(allowlist_path)
            if not allowlist_file.is_file():
                raise FileNotFoundError(f"Synthetic allowlist does not exist: {allowlist_file}")
            allowlist = {line.strip() for line in allowlist_file.read_text().splitlines() if line.strip()}

        config_root = Path(os.environ["NAVSIM_DEVKIT_ROOT"]) / "navsim/planning/script/config/training"
        cache_root = Path(os.environ["NAVSIM_EXP_ROOT"]) / "cache"
        cache_prefix = os.environ.get("SYN_CACHE_PREFIX", "")
        loaded_synthetic = set()
        for expert in (value.strip() for value in expert_spec.split(",") if value.strip()):
            for round_index in range(int(max_round) + 1):
                config_path = config_root / (
                    f"default_log_split_synthetic_reation_{expert}_v1.0-{round_index}.yaml"
                )
                with config_path.open() as stream:
                    synthetic_logs = yaml.safe_load(stream)[f"{self.split}_logs"]
                synthetic_cache = cache_root / (
                    f"{cache_prefix}navtrain_reaction_{expert}_v1.0-{round_index}_cache"
                )
                for log_name in tqdm(synthetic_logs, desc=f"Loading {expert} v1.0-{round_index}"):
                    log_path = synthetic_cache / log_name
                    if not log_path.is_dir():
                        raise FileNotFoundError(log_path)
                    for token_path in sorted(log_path.iterdir()):
                        if allowlist is not None and token_path.name not in allowlist:
                            continue
                        if not token_path.is_dir():
                            continue
                        if not self._has_required_files(token_path, builders):
                            raise ValueError(f"Incomplete synthetic cache: {token_path}")
                        existing = valid_cache_paths.get(token_path.name)
                        if existing is not None and existing != token_path:
                            raise ValueError(
                                f"Duplicate token across cache sources: {token_path.name}"
                            )
                        valid_cache_paths[token_path.name] = token_path
                        loaded_synthetic.add(token_path.name)

        if allowlist is not None and loaded_synthetic != allowlist:
            raise ValueError(f"Selected synthetic caches missing: {len(allowlist - loaded_synthetic)}")
        return valid_cache_paths

    def _load_scene_with_token(
        self, token: str
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return features, targets, token


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
        append_token_to_batch: bool = False,
        agent_input_only: bool = False,
        is_training: bool = True
    ):
        super().__init__()
        self.is_training = is_training
        self.append_token_to_batch = append_token_to_batch
        self.agent_input_only = agent_input_only

        self._scene_loader = scene_loader
        self._feature_builders = feature_builders
        self._target_builders = target_builders

        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            self._cache_path, feature_builders, target_builders
        )

        if self._cache_path is not None:
            self.cache_dataset()

    @staticmethod
    def _load_valid_caches(
        cache_path: Optional[Path],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        if (cache_path is not None) and cache_path.is_dir():
            for log_path in cache_path.iterdir():
                for token_path in log_path.iterdir():
                    found_caches: List[bool] = []
                    for builder in feature_builders + target_builders:
                        data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                        found_caches.append(data_dict_path.is_file())
                    if all(found_caches):
                        valid_cache_paths[token_path.name] = token_path
                        loaded_synthetic.add(token_path.name)

        return valid_cache_paths

    def _cache_scene_with_token(self, token: str) -> None:
        """
        Helper function to compute feature / targets and save in cache.
        :param token: unique identifier of scene to cache
        """

        scene = self._scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()

        metadata = scene.scene_metadata
        token_path = self._cache_path / metadata.log_name / metadata.initial_token
        os.makedirs(token_path, exist_ok=True)

        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_features(agent_input)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_targets(scene)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        self._valid_cache_paths[token] = token_path

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper function to load feature / targets from cache.
        :param token:  unique identifier of scene to load
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)

    def cache_dataset(self) -> None:
        """Caches complete dataset into cache folder."""

        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        os.makedirs(self._cache_path, exist_ok=True)

        # determine tokens to cache
        if self._force_cache_computation:
            tokens_to_cache = self._scene_loader.tokens
        else:
            tokens_to_cache = set(self._scene_loader.tokens) - set(self._valid_cache_paths.keys())
            tokens_to_cache = list(tokens_to_cache)
            logger.info(
                f"""
                Starting caching of {len(tokens_to_cache)} tokens.
                Note: Caching tokens within the training loader is slow. Only use it with a small number of tokens.
                You can cache large numbers of tokens using the `run_dataset_caching.py` python script.
                """
            )

        for token in tqdm(tokens_to_cache, desc="Caching Dataset"):
            self._cache_scene_with_token(token)

    def __len__(self) -> None:
        """
        :return: number of samples to load
        """
        return len(self._scene_loader)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
        """
        Get features or targets either from cache or computed on-the-fly.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """

        token = self._scene_loader.tokens[idx]
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}

        if self.agent_input_only:
            agent_input = AgentInput.from_scene_dict_list(
                self._scene_loader.scene_frames_dicts[token],
                self._scene_loader._sensor_blobs_path,
                num_history_frames=self._scene_loader._scene_filter.num_history_frames,
                sensor_config=self._scene_loader._sensor_config,
            )
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            return features, {'dummy': torch.zeros(1)}, token

        if self._cache_path is not None:
            assert (
                token in self._valid_cache_paths.keys()
            ), f"The token {token} has not been cached yet, please call cache_dataset first!"

            features, targets = self._load_scene_with_token(token)
        else:
            scene = self._scene_loader.get_scene_from_token(self._scene_loader.tokens[idx])
            agent_input = scene.get_agent_input()
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            if self.is_training:
                for builder in self._target_builders:
                    targets.update(builder.compute_targets(scene))

        return (features, targets, token)
