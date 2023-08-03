from typing import Optional

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR

from ..config import TaxonCLIPConfig
from .loss import ClipLoss
from .metrics import ClipMetrics
from .textTower import TextTower
from .visionTower import VisionTower


class TaxonCLIP(pl.LightningModule):
    def __init__(
        self,
        cfg: TaxonCLIPConfig,
        datamodule: Optional[pl.LightningDataModule] = None,
        output_dict: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.visionTower = VisionTower(cfg.model.visionTower)
        self.textTower = TextTower(cfg.model.textTower)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.output_dict = output_dict
        self.steps_per_epoch = 0
        self.datamodule = datamodule
        self.loss = ClipLoss()
        self.metrics = ClipMetrics(self.device)

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(
            unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats
        )

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    def encode_image(self, image, normalize: bool = False):
        features = self.visionTower(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.textTower(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ):
        image_features = (
            self.encode_image(image, normalize=True) if image is not None else None
        )
        text_features = (
            self.encode_text(text, normalize=True) if text is not None else None
        )
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
            }
        return image_features, text_features, self.logit_scale.exp()

    def training_step(self, batch, batch_idx):
        images, texts, names, all_valid_names = batch
        image_features = self.encode_image(images, normalize=True)
        text_features = self.encode_text(texts, normalize=True)
        logit_scale = self.logit_scale.exp()

        loss = self.loss(
            image_features, text_features, logit_scale, names, all_valid_names
        )

        self.log_dict(
            {
                "train_loss": loss,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "logit_scale": self.logit_scale.exp(),
            },
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        images, texts, names, all_valid_names = batch
        image_features = self.encode_image(images, normalize=True)
        text_features = self.encode_text(texts, normalize=True)
        logit_scale = self.logit_scale.exp()
        loss = self.loss(
            image_features, text_features, logit_scale, names, all_valid_names
        )
        metrics = self.metrics(
            image_features, text_features, logit_scale, names, all_valid_names
        )
        metrics["val_loss"] = loss
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.cfg.training.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.cfg.training.learning_rate
            )
        else:
            raise NotImplementedError

        if self.cfg.training.lr_scheduler == "one_cycle":
            if self.datamodule is not None:
                self.steps_per_epoch = self.datamodule.steps_per_epoch
            assert (
                self.steps_per_epoch > 0
            ), "Need to specify datamodule if using one_cycle"
            scheduler = {
                "scheduler": OneCycleLR(
                    optimizer,
                    max_lr=self.cfg.training.learning_rate,
                    steps_per_epoch=self.steps_per_epoch,
                    epochs=self.cfg.training.num_epochs,
                ),
                "interval": "step",
                "frequency": 1,
            }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
