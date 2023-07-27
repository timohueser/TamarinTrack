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
    image_size: int = 224


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
    dataset: str = MISSING
    image_data_dir: str = MISSING
    taxon_data_path: str = MISSING
    lr_scheduler: str = "one_cycle"
    optimizer: str = "adam"
    learning_rate: float = 3e-4
    batch_size: int = 32
    num_epochs: int = 10
    hf_tokenizer_name: str = MISSING
    image_size: int = MISSING


@dataclass
class TaxonCLIPConfig:
    training: TrainingConfig = TrainingConfig()
    model: TaxonCLIPModelConfig = TaxonCLIPModelConfig()


@dataclass
class Config:
    taxonCLIP: TaxonCLIPConfig = TaxonCLIPConfig()
