from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEncoder(ABC):
    """Abstract encoder interface that accepts token IDs only."""

    @abstractmethod
    def embed_tokens(self, ids: List[int]) -> np.ndarray:
        """Encode one sequence of token IDs into a vector."""
        raise NotImplementedError

    @abstractmethod
    def embed_tokens_batch(self, batch_ids: List[List[int]]) -> np.ndarray:
        """Encode a batch of token ID sequences."""
        raise NotImplementedError

    @abstractmethod
    def dim(self) -> int:
        """Return embedding dimension."""
        raise NotImplementedError
