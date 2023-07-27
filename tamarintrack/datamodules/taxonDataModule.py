import os

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .taxonDataset import TaxonDataset


class TaxonDataModule(pl.LightningDataModule):
    def __init__(self, image_data_dir, taxon_path, transforms, batch_size: int = 32):
        super().__init__()
        self.image_data_dir = image_data_dir
        self.taxon_path = taxon_path
        self.transforms = transforms

    def setup(self, stage: str):
        self.taxon_train = TaxonDataset(
            image_folder=os.path.join(self.image_data_dir, "train"),
            taxon_path=self.taxon_path,
            transforms=self.transforms,
            tokenizer=None,
        )
        self.taxon_val = TaxonDataset(
            image_folder=os.path.join(self.image_data_dir, "val"),
            taxon_path=self.taxon_path,
            transforms=self.transforms,
            tokenizer=None,
        )

    def train_dataloader(self):
        return DataLoader(self.taxon_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.taxon_val, batch_size=self.batch_size)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)


if __name__ == "__main__":
    taxonDM = TaxonDataModule(
        image_data_dir="data/ClassificationDatasets/iNaturalist",
        taxon_path="data/taxon.json",
        transforms=None,
        batch_size=32,
    )
