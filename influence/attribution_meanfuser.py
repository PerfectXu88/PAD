"""Per-sample gradient primitives for MeanFuser influence."""

import os
from collections.abc import Iterable

import torch

from influence.common import (
    batch_to_device,
    compute_projected_gradient,
    freeze_batchnorm,
)
from navsim.agents.meanfuser.meanfuser_agent import MeanfuserAgent
from navsim.agents.meanfuser.meanfuser_config import MeanfuserConfig


def build_meanfuser_agent(checkpoint_path: str, device: str) -> MeanfuserAgent:
    config = MeanfuserConfig()
    config.latent = True
    config.lidar_seq_len = 4
    config.num_proposals = 8
    config.noise_type = "multi_gaussian"
    config.navtrain_mean_std_path = (
        f"{os.environ['NAVSIM_DEVKIT_ROOT']}/tools/gaussian_mixed_noise/"
        "navtrain_8_mean_std.pkl"
    )
    config.resnet34_ckpt_path = (
        f"{os.environ['OPENSCENE_DATA_ROOT']}/models/resnet34_model.bin"
    )
    agent = MeanfuserAgent(config=config, lr=1e-4, checkpoint_path=checkpoint_path)
    agent.initialize()
    agent.train()
    freeze_batchnorm(agent)
    return agent.to(device)


def select_params(
    agent: MeanfuserAgent, name_substr: str
) -> list[tuple[str, torch.nn.Parameter]]:
    substrings = [value.strip() for value in name_substr.split(",") if value.strip()]
    selected = [
        (name, parameter)
        for name, parameter in agent.meanfuser_model.named_parameters()
        if parameter.requires_grad and any(value in name for value in substrings)
    ]
    if not selected:
        raise ValueError(f"No parameters match: {name_substr}")
    selected_ids = {id(parameter) for _, parameter in selected}
    for parameter in agent.meanfuser_model.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)
    return selected


def meanfuser_policy_loss(
    agent: MeanfuserAgent, features: dict, targets: dict
) -> torch.Tensor:
    predictions = agent.forward(features)
    meanflow_loss = agent.meanfuser_model._meanflow_head(
        predictions["encoder_output"], targets["trajectory"]
    )
    arm_loss = agent.meanfuser_model._arm_model_head.get_arm_loss(predictions, targets)
    config = agent.meanfuser_model._config
    return config.meanflow_loss_weight * meanflow_loss + config.arm_loss_weight * arm_loss


def compute_projected_grad(
    loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    dimension: int,
    seed: int,
    chunk_size: int,
    device: str,
) -> torch.Tensor:
    """Compatibility wrapper around the shared projection implementation."""
    del device
    return compute_projected_gradient(
        loss,
        parameters,
        dimension,
        seed,
        chunk_size,
    )


to_device_batch = batch_to_device
