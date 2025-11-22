from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple


class BaseIndex(ABC):
    """Abstract vector index interface."""

    @abstractmethod
    def init(self, dim: int, max_elements: int) -> None:
        """Initialize index structure with dimension and capacity."""
        raise NotImplementedError

    @abstractmethod
    def add(self, vec: np.ndarray, idx: int) -> None:
        """Add a vector with an internal integer ID."""
        raise NotImplementedError

    @abstractmethod
    def add_batch(self, vecs: np.ndarray, idxs: List[int]) -> None:
        """Add multiple vectors with integer IDs."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        """Search k nearest neighbors and return (ids, distances)."""
        raise NotImplementedError

    @abstractmethod
    def size(self) -> int:
        """Return how many elements exist in the index."""
        raise NotImplementedError
