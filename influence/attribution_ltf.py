"""Per-sample gradient primitives for LTF/TransFuser influence."""

import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F

from influence.common import (
    batch_to_device,
    compute_projected_gradient,
    freeze_batchnorm,
)
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig


def build_ltf_agent(checkpoint_path: str, device: str) -> TransfuserAgent:
    config = TransfuserConfig()
    config.latent = True
    config.bkb_path = f"{os.environ['OPENSCENE_DATA_ROOT']}/models/resnet34_model.bin"
    agent = TransfuserAgent(config=config, lr=1e-4, checkpoint_path=checkpoint_path)
    agent.initialize()
    agent.eval()
    freeze_batchnorm(agent)
    return agent.to(device)


def select_params(
    agent: TransfuserAgent, name_substr: str
) -> list[tuple[str, torch.nn.Parameter]]:
    substrings = [value.strip() for value in name_substr.split(",") if value.strip()]
    selected = [
        (name, parameter)
        for name, parameter in agent._transfuser_model.named_parameters()
        if parameter.requires_grad and any(value in name for value in substrings)
    ]
    if not selected:
        raise ValueError(f"No parameters match: {name_substr}")
    selected_ids = {id(parameter) for _, parameter in selected}
    for parameter in agent._transfuser_model.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)
    return selected


def ltf_trajectory_loss(
    agent: TransfuserAgent, features: dict, targets: dict
) -> torch.Tensor:
    predictions = agent.forward(features)
    return agent._config.trajectory_weight * F.l1_loss(
        predictions["trajectory"], targets["trajectory"]
    )


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
