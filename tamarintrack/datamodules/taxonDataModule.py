import os

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .taxonDataset import TaxonDataset


class TaxonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_data_dir,
        taxon_path,
        transforms_train,
        transforms_val,
        tokenizer,
        batch_size: int = 32,
    ):
        super().__init__()
        self.image_data_dir = image_data_dir
        self.taxon_path = taxon_path
        self.transforms_train = transforms_train
        self.transforms_val = transforms_val
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.steps_per_epoch = 0

    def setup(self, stage: str):
        self.taxon_train = TaxonDataset(
            image_folder=os.path.join(self.image_data_dir, "train"),
            taxon_path=self.taxon_path,
            transforms=self.transforms_train,
            tokenizer=self.tokenizer,
        )
        self.taxon_val = TaxonDataset(
            image_folder=os.path.join(self.image_data_dir, "val"),
            taxon_path=self.taxon_path,
            transforms=self.transforms_val,
            tokenizer=self.tokenizer,
        )
        self.steps_per_epoch = len(self.taxon_train) // self.batch_size

    def train_dataloader(self):
        return DataLoader(self.taxon_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.taxon_val, batch_size=self.batch_size)


if __name__ == "__main__":
    taxonDM = TaxonDataModule(
        image_data_dir="data/ClassificationDatasets/iNaturalist",
        taxon_path="data/taxon2.json",
        transforms_train=None,
        transforms_val=None,
        batch_size=32,
    )
