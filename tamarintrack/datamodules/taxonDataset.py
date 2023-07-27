import json
import logging
import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from transformers import PreTrainedTokenizerBase

from ..tokenizer import HFTokenizer


class TaxonDataset(Dataset):
    def __init__(
        self,
        image_folder: str,
        taxon_path: str,
        transforms: Compose,
        tokenizer: PreTrainedTokenizerBase = None,
    ):
        logging.debug(
            f"Loading json data from {taxon_path} and images from {image_folder}."
        )
        self.data = []
        self.taxon = json.load(open(taxon_path))
        self.max_list_len = 18
        for class_name in os.listdir(image_folder):
            species = class_name.split("-")[-1].replace("_", " ")
            taxonomy = self.taxon[species]
            valid_names = []
            for _key, val in taxonomy["common_names"].items():
                valid_names += val
            for img in os.listdir(os.path.join(image_folder, class_name)):
                item = {}
                item["valid_names"] = valid_names
                item["image"] = os.path.join(image_folder, class_name, img)
                self.data.append(item)
        self.transforms = transforms
        self.tokenize = tokenizer
        logging.debug("Done loading data.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        images = self.transforms(Image.open(str(self.data[idx]["image"])))
        Image.open(str(self.data[idx]["image"]))
        valid_names = self.data[idx]["valid_names"]
        name = random.choice(valid_names)
        valid_names = valid_names + [""] * (self.max_list_len - len(valid_names))

        tokenized_name = self.tokenize([name])[0]
        return images, tokenized_name, name, valid_names


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    tokenizer = HFTokenizer("distilbert-base-uncased")
    transforms = Compose([])

    iNaturalist_images = "data/ClassificationDatasets/iNaturalist/images/train"
    iNaturalist_taxon = "data/ClassificationDatasets/iNaturalist/taxon2.json"

    dataset = TaxonDataset(
        image_folder=iNaturalist_images,
        taxon_path=iNaturalist_taxon,
        transforms=transforms,
        tokenizer=tokenizer,
    )
    item = dataset.__getitem__(0)

    plt.imshow(item[0])
    plt.show()
