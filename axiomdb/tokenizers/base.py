from abc import ABC, abstractmethod
from typing import List


class BaseTokenizer(ABC):
    """Abstract tokenizer interface."""

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """Tokenize a single string into token ids."""
        raise NotImplementedError

    @abstractmethod
    def tokenize_batch(self, texts: List[str]) -> List[List[int]]:
        """Tokenize a batch of strings into lists of token ids."""
        raise NotImplementedError

    @abstractmethod
    def vocab_size(self) -> int:
        """Return size of the vocabulary."""
        raise NotImplementedError
