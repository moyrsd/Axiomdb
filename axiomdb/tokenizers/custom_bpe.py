from typing import List, Dict, Tuple
import numpy as np
from collections import Counter
import numba as nb

from .base import BaseTokenizer


@nb.njit
def _apply_merge_numba(arr, p1, p2, new_id):
    """
    Scan arr (1D int64) and replace occurrences of (p1,p2) with new_id.
    Returns a new array (int64) with merges applied.
    """
    n = arr.shape[0]
    out = np.empty(n, dtype=np.int64)
    oi = 0
    i = 0
    while i < n:
        if i < n - 1 and arr[i] == p1 and arr[i+1] == p2:
            out[oi] = new_id
            oi += 1
            i += 2
        else:
            out[oi] = arr[i]
            oi += 1
            i += 1
    return out[:oi]


def _apply_merges_seq(arr: np.ndarray, merges: List[Tuple[int,int,int]]) -> np.ndarray:
    """
    merges: list of (p1, p2, new_id) in the order they should be applied.
    arr: numpy array of dtype np.int64
    """
    cur = arr
    for (p1, p2, new_id) in merges:
        cur = _apply_merge_numba(cur, p1, p2, new_id)
    return cur


class CustomBPETokenizer(BaseTokenizer):
    """
    Byte-level BPE tokenizer with Numba-accelerated merge application.

    Key properties:
      - base vocab: 0..255 (single bytes)
      - merges: dict mapping (p1,p2) -> new_id
      - vocab: dict mapping id -> bytes sequence (used for decode)
    """

    def __init__(self):
        self.merges: Dict[Tuple[int,int], int] = {}
        self._merges_seq: List[Tuple[int,int,int]] = []
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._next_id = 256

    def train(self, text: str, num_merges: int = 1000, verbose: bool = False) -> None:
        """
        Train merges on the provided corpus text.
        This function:
          - converts corpus to byte sequence (utf-8)
          - finds most frequent adjacent byte-pair
          - applies the pair merge to the corpus sequence (in Python using numba-accelerated merge application)
          - repeats for num_merges
        """
        data_bytes = text.encode("utf-8")
        arr = np.frombuffer(data_bytes, dtype=np.uint8).astype(np.int64)

        merges_applied = []
        for m in range(num_merges):
            pairs = Counter()
            a = arr
            for i in range(a.shape[0] - 1):
                pairs[(int(a[i]), int(a[i+1]))] += 1

            if not pairs:
                break

            (p1, p2), freq = pairs.most_common(1)[0]
            new_id = self._next_id
            self._next_id += 1

            self.merges[(p1, p2)] = new_id
            merges_applied.append((p1, p2, new_id))
            arr = _apply_merge_numba(arr, p1, p2, new_id)
            self.vocab[new_id] = self.vocab[p1] + self.vocab[p2]

            if verbose and ((m + 1) % 50 == 0 or (m+1) == num_merges):
                print(f"[BPE] Applied {m+1}/{num_merges} merges; most common pair {p1},{p2} freq={freq}")

            if arr.shape[0] < 2:
                break

        self._merges_seq = merges_applied

    def tokenize(self, text: str) -> List[int]:
        """
        Encode a single string into token IDs (list[int]).
        Uses byte-level input and applies merges in the order learned during training.
        """
        b = text.encode("utf-8")
        arr = np.frombuffer(b, dtype=np.uint8).astype(np.int64)

        if self._merges_seq:
            arr = _apply_merges_seq(arr, self._merges_seq)

        return arr.tolist()

    def tokenize_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.tokenize(t) for t in texts]

    def vocab_size(self) -> int:
        return len(self.vocab)

    def decode(self, ids: List[int]) -> str:
        parts = []
        for idx in ids:
            parts.append(self.vocab.get(idx, b"?"))
        return b"".join(parts).decode("utf-8", errors="replace")

    def save_merges(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for (p1, p2, new_id) in self._merges_seq:
                f.write(f"{p1} {p2} {new_id}\n")

    def load_merges(self, path: str) -> None:
        merges = []
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._next_id = 256

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                p1_s, p2_s, new_s = line.strip().split()
                p1, p2, new_id = int(p1_s), int(p2_s), int(new_s)
                self.merges[(p1, p2)] = new_id
                self.vocab[new_id] = self.vocab[p1] + self.vocab[p2]
                merges.append((p1, p2, new_id))
                self._next_id = max(self._next_id, new_id + 1)

        self._merges_seq = merges
