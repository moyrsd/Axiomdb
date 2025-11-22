from typing import List
from transformers import AutoTokenizer
from .base import BaseTokenizer


class HFBPETokenizer(BaseTokenizer):
    """HuggingFace BPE tokenizer adapter."""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text: str) -> List[int]:
        out = self._tokenizer.encode(
            text,
            truncation=False,
            add_special_tokens=False
        )
        return out

    def tokenize_batch(self, texts: List[str]) -> List[List[int]]:
        enc = self._tokenizer(
            texts,
            truncation=False,
            add_special_tokens=False
        )
        return enc["input_ids"]

    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size
