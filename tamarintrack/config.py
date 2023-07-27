from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class VisionTowerConfig:
    timm_model_name: str = MISSING
    pretrained: bool = True
    embedding_dim: int = 512
    patch_dropout: float = 0.0  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    timm_pool: str = (
        "avg"  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    )
    timm_proj: str = (
        "linear"  # linear projection for timm model output ('linear', 'mlp', '')
    )
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.0  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class TextTowerConfig:
    hf_model_name: str = MISSING
    hf_tokenizer_name: str = MISSING
    embedding_dim: int = 512
    pretrained: bool = True
    proj: str = "mlp"
    pooler_type: str = "mean_pooler"
    output_tokens: bool = False


@dataclass
class TaxonCLIPModelConfig:
    embedding_dim: int = 512
    visionTower: VisionTowerConfig = VisionTowerConfig()
    textTower: TextTowerConfig = TextTowerConfig()


@dataclass
class TrainingConfig:
    optimizer: str = "adam"
    learning_rate: float = 3e-4


@dataclass
class TaxonCLIPConfig:
    training: TrainingConfig = TrainingConfig()
    model: TaxonCLIPModelConfig = TaxonCLIPModelConfig()


@dataclass
class Config:
    taxonCLIP: TaxonCLIPConfig = TaxonCLIPConfig()
