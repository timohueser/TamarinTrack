import html
import re
from typing import List, Union

import ftfy
import torch


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class HFTokenizer:
    """HuggingFace tokenizer wrapper"""

    def __init__(self, tokenizer_name: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def __call__(
        self, texts: Union[str, List[str]], context_length: int = 77
    ) -> torch.Tensor:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]
        texts = [whitespace_clean(basic_clean(text)) for text in texts]
        input_ids = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=context_length,
            padding="max_length",
            truncation=True,
        ).input_ids
        return input_ids
