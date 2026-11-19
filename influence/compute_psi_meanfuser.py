"""Compute closed-loop-aligned influence scores for MeanFuser.

MeanFuser combines its MeanFlow and ARM planning losses while sharing PAD's
cache, projection, calibration, and regularized influence implementation.
"""

import argparse
import csv
from pathlib import Path

from influence.attribution_meanfuser import (
    build_meanfuser_agent,
    meanfuser_policy_loss,
    select_params,
)
from influence.common import (
    collect_projected_gradients,
    has_required_caches,
    index_cached_tokens,
    log,
    normalized_advantage,
    validate_calibration_cache,
    write_run_manifest,
    sample_cached_tokens,
    solve_influence_scores,
    write_influence_scores,
)
from selection.synthetic_pool import enumerate_synthetic_pool, parse_rounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--c-epdms", required=True)
    parser.add_argument("--calib-cache", required=True)
    parser.add_argument("--hessian-cache", required=True)
    parser.add_argument("--experts", default="pdm,recovery")
    parser.add_argument("--rounds", default="0-4")
    parser.add_argument("--d-proj", type=int, default=2048)
    parser.add_argument("--param-substr", default="_meanflow_head,_arm_model_head")
    parser.add_argument("--proj-chunk", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-reg", type=float, default=1e-2)
    parser.add_argument("--hessian-n", type=int, default=2000)
    parser.add_argument("--mc-samples", type=int, default=1)
    parser.add_argument("--max-pool", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if min(args.d_proj, args.proj_chunk, args.mc_samples, args.hessian_n) <= 0:
        raise ValueError("Projection, Monte Carlo, and Hessian sizes must be positive")
    if args.lambda_reg <= 0:
        raise ValueError("--lambda-reg must be positive")

    calibration_cache = Path(args.calib_cache)
    hessian_cache = Path(args.hessian_cache)
    for name, path in (("calibration", calibration_cache), ("Hessian", hessian_cache)):
        if not path.is_dir():
            raise FileNotFoundError(f"{name} cache does not exist: {path}")

    agent = build_meanfuser_agent(args.checkpoint, args.device)
    builders = agent.get_feature_builders() + agent.get_target_builders()
    selected = select_params(agent, args.param_substr)
    parameters = [parameter for _, parameter in selected]
    log(
        f"[MeanFuser-psi] selected "
        f"{sum(parameter.numel() for parameter in parameters):,} parameters; "
        f"d_proj={args.d_proj}"
    )

    with Path(args.c_epdms).open(newline="") as stream:
        calibration_rows = list(csv.DictReader(stream))
    calibration_index = index_cached_tokens(
        calibration_cache,
        [row["token"] for row in calibration_rows],
        builders,
    )
    validate_calibration_cache(calibration_rows, calibration_index)
    if not calibration_rows:
        raise ValueError("No calibration tokens have complete MeanFuser caches")
    calibration_paths = [
        calibration_index[row["token"]] for row in calibration_rows
    ]
    advantage = normalized_advantage(calibration_rows)

    pool = enumerate_synthetic_pool(
        [value.strip() for value in args.experts.split(",") if value.strip()],
        parse_rounds(args.rounds),
        cache_prefix="meanfuser_",
    )
    pool_items = [
        (token, metadata)
        for token, metadata in pool.items()
        if has_required_caches(metadata[0], builders)
    ]
    if len(pool_items) != len(pool):
        raise ValueError("Candidate cache incomplete; rebuild before valuation")
    if set(pool).intersection(row["token"] for row in calibration_rows):
        raise ValueError("Calibration and candidate tokens overlap")
    if args.max_pool > 0:
        pool_items = pool_items[: args.max_pool]
    if not pool_items:
        raise ValueError("The synthetic pool has no complete MeanFuser cache entries")

    hessian_paths = sample_cached_tokens(
        hessian_cache,
        args.hessian_n,
        args.seed,
        builders,
        excluded_tokens=[row["token"] for row in calibration_rows],
    )
    if not hessian_paths:
        raise ValueError("The Hessian cache has no complete MeanFuser entries")

    gradient_options = {
        "agent": agent,
        "parameters": parameters,
        "loss_function": meanfuser_policy_loss,
        "output_dim": args.d_proj,
        "seed": args.seed,
        "chunk_size": args.proj_chunk,
        "device": args.device,
        "monte_carlo_samples": args.mc_samples,
        "progress_every": max(1, args.progress_every),
        "log_prefix": "MeanFuser-psi",
    }
    phi_hessian = collect_projected_gradients(
        token_paths=hessian_paths,
        label="hessian",
        **gradient_options,
    )
    phi_calibration = collect_projected_gradients(
        token_paths=calibration_paths,
        label="calibration",
        **gradient_options,
    )
    phi_pool = collect_projected_gradients(
        token_paths=[metadata[0] for _, metadata in pool_items],
        label="pool",
        **gradient_options,
    )

    scores = solve_influence_scores(
        phi_hessian,
        phi_calibration,
        phi_pool,
        advantage,
        args.lambda_reg,
    )
    output_path = Path(args.out)
    write_influence_scores(output_path, pool_items, scores)
    write_run_manifest(output_path, args, selected, hessian_paths, calibration_rows, pool_items)
    log(f"[MeanFuser-psi] wrote {len(scores)} scores to {output_path}")


if __name__ == "__main__":
    main()
