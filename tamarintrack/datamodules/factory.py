from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Normalize,
    RandomResizedCrop,
    ToTensor,
)

from ..config import TrainingConfig
from ..tokenizer import HFTokenizer
from .taxonDataModule import TaxonDataModule


def create_data_module(cfg: TrainingConfig):
    if cfg.dataset == "TaxonDataset":
        data_module = create_taxon_data_module(cfg)
    else:
        raise NotImplementedError
    return data_module


def create_taxon_data_module(cfg: TrainingConfig):
    transforms_train = Compose(
        [
            RandomResizedCrop(
                cfg.image_size,
                interpolation=InterpolationMode.BICUBIC,
            ),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    transforms_val = Compose(
        [
            RandomResizedCrop(
                cfg.image_size,
                interpolation=InterpolationMode.BICUBIC,
            ),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    tokenizer = HFTokenizer(cfg.hf_tokenizer_name)
    data_module = TaxonDataModule(
        image_data_dir=cfg.image_data_dir,
        taxon_path=cfg.taxon_data_path,
        transforms_train=transforms_train,
        transforms_val=transforms_val,
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
    )
    return data_module
