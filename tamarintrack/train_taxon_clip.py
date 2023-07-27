"""
This is the demo code that uses hy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      dra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import hydra
from omegaconf import OmegaConf

from .config import Config
from .taxonCLIP.taxonCLIP import TaxonCLIP


@hydra.main(config_path="../config", config_name="main", version_base=None)
def train_taxon_clip(cfg: Config):
    cfg = OmegaConf.merge(OmegaConf.structured(Config), cfg)
    print(OmegaConf.to_yaml(cfg))
    """Function to train the model"""
    print(f"Visual Model used: {cfg.taxonCLIP.model.visionTower.timm_model_name}")
    print(f"Text Model used: {cfg.taxonCLIP.model.textTower.hf_model_name}")
    print(cfg.taxonCLIP.model.textTower.embedding_dim)

    TaxonCLIP(cfg.taxonCLIP)


if __name__ == "__main__":
    train_taxon_clip()
