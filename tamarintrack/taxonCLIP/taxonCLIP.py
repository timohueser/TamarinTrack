from typing import Optional

import lightning.pytorch as pl
import torch
import torch.functional as F
from torch import nn

from ..config import TaxonCLIPConfig
from .textTower import TextTower
from .visionTower import VisionTower


class TaxonCLIP(pl.LightningModule):
    def __init__(
        self,
        cfg: TaxonCLIPConfig,
        output_dict: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.visionTower = VisionTower(cfg.model.visionTower)
        self.textTower = TextTower(cfg.model.textTower)
        self.output_dict = output_dict

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
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.cfg.training.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.cfg.training.learning_rate
            )
        else:
            raise NotImplementedError
        return optimizer
