from typing import Dict, Tuple

import pytorch_lightning as pl
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Lightning wrapper for MeanFuser training."""

    def __init__(self, agent: AbstractAgent):
        super().__init__()
        self.agent = agent

    def _step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], list[str]],
        logging_prefix: str,
    ) -> Tensor:
        features, targets, tokens = batch
        predictions = self.agent.forward(features)
        loss, metrics = self.agent.compute_loss(features, targets, predictions, tokens)
        self.log(
            f"{logging_prefix}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for name, value in metrics.items():
            self.log(
                f"{logging_prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return loss

    def training_step(self, batch, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        return self.agent.get_optimizers()
