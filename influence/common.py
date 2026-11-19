"""Shared utilities for policy-specific influence estimation.

The policy adapters only define how to build a model and compute its native
planning loss. Cache traversal, projected gradients, calibration weighting,
and the regularized influence solve live here so every policy follows the same
estimation protocol.
"""

from __future__ import annotations

import csv
import os
import yaml
import json
import hashlib
import math
import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from influence.protocol import normalized_advantage

LossFunction = Callable[[Any, dict[str, Any], dict[str, Any]], torch.Tensor]


def log(message: str) -> None:
    """Print a progress message immediately."""
    print(message, flush=True)


def format_duration(seconds: float) -> str:
    """Format a duration compactly for long-running experiment logs."""
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def freeze_batchnorm(module: torch.nn.Module) -> None:
    """Use running BatchNorm statistics for single-sample gradients."""
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()


def _batch_value(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.unsqueeze(0).to(device)
    if isinstance(value, list):
        return [_batch_value(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_batch_value(item, device) for item in value)
    if isinstance(value, Mapping):
        return {key: _batch_value(item, device) for key, item in value.items()}
    return value


def batch_to_device(sample: Mapping[str, Any], device: str) -> dict[str, Any]:
    """Add a batch dimension to nested tensors and move them to ``device``."""
    return {key: _batch_value(value, device) for key, value in sample.items()}


def project_gradient(
    gradient: torch.Tensor,
    output_dim: int,
    seed: int,
    chunk_size: int,
) -> torch.Tensor:
    """Apply a reproducible chunked Gaussian projection without storing it."""
    if output_dim <= 0 or chunk_size <= 0:
        raise ValueError("Projection dimension and chunk size must be positive")
    generator = torch.Generator(device=gradient.device).manual_seed(seed)
    projected = torch.zeros(output_dim, device=gradient.device, dtype=gradient.dtype)
    for start in range(0, gradient.numel(), chunk_size):
        chunk = gradient[start : start + chunk_size]
        matrix = torch.randn(
            chunk.numel(),
            output_dim,
            generator=generator,
            device=gradient.device,
            dtype=gradient.dtype,
        )
        projected.add_(chunk @ matrix)
    return projected / math.sqrt(output_dim)


def compute_projected_gradient(
    loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    output_dim: int,
    seed: int,
    chunk_size: int,
) -> torch.Tensor:
    """Differentiate ``loss``, flatten selected parameters, and project."""
    parameters = list(parameters)
    if not parameters:
        raise ValueError("At least one parameter is required for attribution")
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    flattened = torch.cat(
        [
            gradient.reshape(-1)
            if gradient is not None
            else torch.zeros(
                parameter.numel(),
                device=parameter.device,
                dtype=parameter.dtype,
            )
            for gradient, parameter in zip(gradients, parameters)
        ]
    )
    return project_gradient(flattened, output_dim, seed, chunk_size)


def has_required_caches(token_path: Path, builders: Sequence[Any]) -> bool:
    """Return whether every feature and target builder is cached for a token."""
    return all(
        (token_path / f"{builder.get_unique_name()}.gz").is_file()
        for builder in builders
    )


def iter_cached_tokens(cache_path: Path, builders: Sequence[Any]) -> Iterable[Path]:
    """Yield complete cache entries in deterministic order."""
    for log_path in sorted(cache_path.iterdir()):
        if not log_path.is_dir():
            continue
        for token_path in sorted(log_path.iterdir()):
            if token_path.is_dir() and has_required_caches(token_path, builders):
                yield token_path


def index_cached_tokens(
    cache_path: Path,
    tokens: Sequence[str],
    builders: Sequence[Any],
) -> dict[str, Path]:
    """Index the requested token names that have complete cache entries."""
    wanted = set(tokens)
    index = {}
    for token_path in iter_cached_tokens(cache_path, builders):
        if token_path.name in wanted:
            if token_path.name in index:
                raise ValueError(f"Duplicate real token: {token_path.name}")
            index[token_path.name] = token_path
    return index


def sample_cached_tokens(
    cache_path: Path,
    sample_count: int,
    seed: int,
    builders: Sequence[Any],
    excluded_tokens: Sequence[str] = (),
) -> list[Path]:
    """Sample complete real-data cache entries reproducibly."""
    if sample_count <= 0:
        raise ValueError("Cache sample count must be positive")
    excluded = set(excluded_tokens)
    config_path = Path(os.environ["NAVSIM_DEVKIT_ROOT"]) / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
    train_logs = set(yaml.safe_load(config_path.read_text())["train_logs"])
    candidates = [p for p in iter_cached_tokens(cache_path, builders)
                  if p.name not in excluded and p.parent.name in train_logs]
    if sample_count > len(candidates):
        raise ValueError(f"Need {sample_count} geometry scenes; found {len(candidates)}")
    return random.Random(seed).sample(candidates, sample_count)


def load_cached_sample(
    token_path: Path,
    feature_builders: Sequence[Any],
    target_builders: Sequence[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the feature and target dictionaries for one cached token."""
    from navsim.planning.training.dataset import load_feature_target_from_pickle

    features: dict[str, Any] = {}
    targets: dict[str, Any] = {}
    for builder in feature_builders:
        features.update(
            load_feature_target_from_pickle(
                token_path / f"{builder.get_unique_name()}.gz"
            )
        )
    for builder in target_builders:
        targets.update(
            load_feature_target_from_pickle(
                token_path / f"{builder.get_unique_name()}.gz"
            )
        )
    return features, targets


def collect_projected_gradients(
    agent: Any,
    token_paths: Sequence[Path],
    parameters: Sequence[torch.nn.Parameter],
    loss_function: LossFunction,
    output_dim: int,
    seed: int,
    chunk_size: int,
    device: str,
    label: str,
    progress_every: int = 16,
    monte_carlo_samples: int = 1,
    log_prefix: str = "psi",
) -> torch.Tensor:
    """Compute projected per-sample gradients for one cache collection."""
    if monte_carlo_samples <= 0:
        raise ValueError("Monte Carlo sample count must be positive")
    feature_builders = agent.get_feature_builders()
    target_builders = agent.get_target_builders()
    output = torch.zeros(len(token_paths), output_dim, device=device)
    started_at = time.time()
    log(f"[{log_prefix}] {label}: start n={len(token_paths)} d_proj={output_dim}")

    for index, token_path in enumerate(token_paths):
        features, targets = load_cached_sample(
            token_path,
            feature_builders,
            target_builders,
        )
        features = batch_to_device(features, device)
        targets = batch_to_device(targets, device)
        accumulated = None
        last_loss = None
        for _ in range(monte_carlo_samples):
            loss = loss_function(agent, features, targets)
            last_loss = loss.detach()
            projected = compute_projected_gradient(
                loss,
                parameters,
                output_dim,
                seed,
                chunk_size,
            )
            accumulated = projected if accumulated is None else accumulated + projected
        output[index] = accumulated / monte_carlo_samples

        completed = index + 1
        progress_every = max(1, progress_every)
        if completed % progress_every == 0 or completed == len(token_paths):
            elapsed = time.time() - started_at
            rate = completed / max(elapsed, 1e-9)
            eta = (len(token_paths) - completed) / max(rate, 1e-9)
            log(
                f"[{log_prefix}] {label}: {completed}/{len(token_paths)} "
                f"({completed / len(token_paths) * 100:.1f}%) "
                f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
                f"rate={rate:.3f} samples/s loss={last_loss.item():.6f}"
            )
    return output




def validate_calibration_cache(rows, index):
    missing = [row["token"] for row in rows if row["token"] not in index]
    if missing:
        raise ValueError(f"Incomplete fixed calibration cache: {len(missing)} missing tokens")


def write_run_manifest(output_path, args, selected, geometry_paths, calibration_rows, pool_items):
    """Persist identities and settings, not just a ranking with an unknown origin."""
    digest = hashlib.sha256()
    with Path(args.checkpoint).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    record = {
        "arguments": vars(args), "checkpoint_sha256": digest.hexdigest(),
        "parameters": [{"name": name, "shape": list(param.shape)} for name, param in selected],
        "geometry_tokens": [path.name for path in geometry_paths],
        "calibration_tokens": [row["token"] for row in calibration_rows],
        "candidate_tokens": [token for token, _ in pool_items],
        "normalization": "global population std + 1e-8",
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(record, indent=2) + "\n")


def solve_influence_scores(
    phi_hessian: torch.Tensor,
    phi_calibration: torch.Tensor,
    phi_pool: torch.Tensor,
    advantage: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Solve the regularized projected influence system and score the pool."""
    if regularization <= 0:
        raise ValueError("Influence regularization must be positive")
    matrices = (phi_hessian, phi_calibration, phi_pool)
    if any(x.ndim != 2 or x.shape[0] == 0 for x in matrices):
        raise ValueError("Gradient matrices must be nonempty and two-dimensional")
    if len({x.shape[1] for x in matrices}) != 1 or len(advantage) != phi_calibration.shape[0]:
        raise ValueError("Inconsistent projected gradient dimensions")
    if any(not torch.isfinite(x).all() for x in matrices) or not np.isfinite(advantage).all():
        raise ValueError("Nonfinite projected gradients or advantages")
    projection_dim = phi_hessian.shape[1]
    hessian = (phi_hessian.T @ phi_hessian) / phi_hessian.shape[0]
    hessian.add_(
        regularization
        * torch.eye(
            projection_dim,
            device=phi_hessian.device,
            dtype=phi_hessian.dtype,
        )
    )
    advantage_tensor = torch.as_tensor(
        advantage,
        dtype=phi_calibration.dtype,
        device=phi_calibration.device,
    )
    weighted_target = phi_calibration.T @ advantage_tensor
    influence_direction = torch.linalg.solve(hessian, weighted_target)
    return (phi_pool @ influence_direction).detach().cpu().numpy()


def write_influence_scores(
    output_path: Path,
    pool_items: Sequence[tuple[str, tuple[Path, str, int]]],
    scores: np.ndarray,
) -> None:
    """Write policy-specific scores with synthetic source provenance."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["token", "psi", "source_type", "round"])
        for (token, metadata), score in zip(pool_items, scores):
            writer.writerow([token, f"{score:.8e}", metadata[1], metadata[2]])
