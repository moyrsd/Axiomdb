from typing import List, Tuple
import numpy as np
import hnswlib
from .base import BaseIndex


class HNSWLibIndex(BaseIndex):
    """HNSW index implementation using hnswlib with cosine similarity."""

    def __init__(self):
        self._index = None
        self._dim = None

    def init(self, dim: int, max_elements: int = 10000) -> None:
        self._dim = dim
        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(
            max_elements=max_elements,
            ef_construction=200,
            M=16
        )
        self._index.set_ef(50)

    def add(self, vec: np.ndarray, idx: int) -> None:
        vec = vec.astype(np.float32)
        self._index.add_items(vec.reshape(1, -1), [idx])

    def add_batch(self, vecs: np.ndarray, idxs: List[int]) -> None:
        vecs = vecs.astype(np.float32)
        self._index.add_items(vecs, idxs)

    def search(self, query: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        query = query.astype(np.float32).reshape(1, -1)
        labels, distances = self._index.knn_query(query, k)
        return labels[0].tolist(), distances[0].tolist()

    def size(self) -> int:
        return self._index.get_current_count()
