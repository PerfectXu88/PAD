"""Shared DiffusionDrive attribution primitives."""

import os

import torch

from influence.common import freeze_batchnorm
from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig


def build_agent(checkpoint_path: str, device: str) -> TransfuserAgent:
    """Build DiffusionDrive in the mode required by its denoising loss."""
    config = TransfuserConfig()
    config.bkb_path = f"{os.environ['OPENSCENE_DATA_ROOT']}/models/resnet34_model.bin"
    config.plan_anchor_path = f"{os.environ['NAVSIM_DEVKIT_ROOT']}/traj_final/kmeans_navsim_traj_20.npy"
    config.latent = True
    agent = TransfuserAgent(config=config, lr=6e-4, checkpoint_path=checkpoint_path)
    agent.initialize()
    agent.train()
    freeze_batchnorm(agent)
    return agent.to(device)


def select_parameters(agent: TransfuserAgent, name_patterns: str):
    """Select trainable model parameters matching any comma-separated pattern."""
    patterns = tuple(pattern.strip() for pattern in name_patterns.split(",") if pattern.strip())
    if not patterns:
        raise ValueError("At least one parameter name pattern is required")
    selected = [
        (name, parameter)
        for name, parameter in agent._transfuser_model.named_parameters()
        if parameter.requires_grad and any(pattern in name for pattern in patterns)
    ]
    if not selected:
        raise ValueError(f"No trainable parameters match {patterns}")
    selected_ids = {id(parameter) for _, parameter in selected}
    for parameter in agent._transfuser_model.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)
    return selected
