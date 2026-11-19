import os
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_loss import transfuser_loss
from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel as TransfuserModel
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    class_name = cfg.pop("type")
    return getattr(obj, class_name)(**cfg, **kwargs)


class TransfuserAgent(AbstractAgent):
    """NAVSIM agent interface for DiffusionDrive."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__(trajectory_sampling=config.trajectory_sampling)

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        self._transfuser_model = TransfuserModel(config)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        if not self._checkpoint_path:
            raise ValueError("A checkpoint path is required for initialization")
        checkpoint = torch.load(self._checkpoint_path, map_location="cpu")
        state_dict: Dict[str, Any] = {
            key.replace("agent.", ""): value
            for key, value in checkpoint["state_dict"].items()
        }
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing checkpoint keys: {missing}")
        if unexpected:
            print(f"Unexpected checkpoint keys: {unexpected}")

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[0, 1, 2, 3],
            cam_l0=[0, 1, 2, 3],
            cam_l1=[0, 1, 2, 3],
            cam_l2=[0, 1, 2, 3],
            cam_r0=[0, 1, 2, 3],
            cam_r1=[0, 1, 2, 3],
            cam_r2=[0, 1, 2, 3],
            cam_b0=[0, 1, 2, 3],
            lidar_pc=[],
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._transfuser_model(features, targets=targets)
        
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return transfuser_loss(targets, predictions, self._config)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return self.get_coslr_optimizers()

    def get_coslr_optimizers(self):
        optimizer_cfg = DictConfig(
            {
                "type": self._config.optimizer_type,
                "lr": self._lr,
                "weight_decay": self._config.weight_decay,
                "paramwise_cfg": self._config.opt_paramwise_cfg,
            }
        )

        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop("paramwise_cfg", None)

        if paramwise_cfg:
            params = []
            parameter_groups = [[] for _ in paramwise_cfg["name"]]

            for name, parameter in self._transfuser_model.named_parameters():
                in_param_group = True
                for index, pattern in enumerate(paramwise_cfg["name"]):
                    if pattern in name:
                        parameter_groups[index].append(parameter)
                        in_param_group = False
                if in_param_group:
                    params.append(parameter)
        else:
            params = self._transfuser_model.parameters()

        optimizer = build_from_configs(optim, optimizer_cfg, params=params)

        if paramwise_cfg:
            for parameter_group, group_config in zip(
                parameter_groups,
                paramwise_cfg["name"].values(),
            ):
                cfg = {}
                if "lr_mult" in group_config:
                    cfg["lr"] = optimizer_cfg["lr"] * group_config["lr_mult"]
                optimizer.add_param_group({"params": parameter_group, **cfg})

        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=100,
            warmup_epochs=3,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
        ckpt_callback = ModelCheckpoint(
            save_top_k=100,
            monitor="val/loss_epoch",
            mode="min",
            dirpath=f"{os.environ['NAVSIM_EXP_ROOT']}/{self._config.ckpt_path}/",
            filename="{epoch:02d}",
            enable_version_counter=False,
            save_last=True,
        )
        return [ckpt_callback, lr_monitor]
