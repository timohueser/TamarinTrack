"""
This is the demo code that uses hy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      dra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import hydra
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from .config import Config
from .datamodules.factory import create_data_module
from .taxonCLIP.taxonCLIP import TaxonCLIP


@hydra.main(config_path="../config", config_name="main", version_base=None)
def train_taxon_clip(cfg: Config):
    cfg = OmegaConf.merge(OmegaConf.structured(Config), cfg)
    print(OmegaConf.to_yaml(cfg))
    """Function to train the model"""
    print(f"Visual Model used: {cfg.taxonCLIP.model.visionTower.timm_model_name}")
    print(f"Text Model used: {cfg.taxonCLIP.model.textTower.hf_model_name}")
    print(cfg.taxonCLIP.model.textTower.embedding_dim)

    datamodule = create_data_module(cfg.taxonCLIP.training)
    model = TaxonCLIP(cfg.taxonCLIP, datamodule=datamodule)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # replace 'val_loss' with whatever metric you're using
        mode="min",  # or 'max' for metrics where higher is better
        save_top_k=1,  # only save the best model
        verbose=True,  # so we get notified
        save_last=True,  # do not save the last model implicitly
    )

    seed_everything(42, workers=True)
    trainer = Trainer(
        devices=cfg.taxonCLIP.training.num_devices,
        max_epochs=cfg.taxonCLIP.training.num_epochs,
        deterministic=True,
        callbacks=[checkpoint_callback],
        precision="16-mixed",
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train_taxon_clip()
