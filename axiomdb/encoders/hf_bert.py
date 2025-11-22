from typing import List
import numpy as np
import torch
from transformers import AutoModel
from .base import BaseEncoder


class HFBERTEncoder(BaseEncoder):
    """HuggingFace BERT encoder that accepts token ID sequences."""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        sample = torch.zeros((1, 1), dtype=torch.long)
        with torch.no_grad():
            out = self._model(sample)
        self._dim = out.last_hidden_state.shape[-1]

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_exp = attention_mask.unsqueeze(-1).expand(last_hidden.size())
        sum_vectors = torch.sum(last_hidden * mask_exp, dim=1)
        lengths = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
        return sum_vectors / lengths

    def embed_tokens(self, ids: List[int]) -> np.ndarray:
        ids_tensor = torch.tensor([ids], dtype=torch.long)
        mask = torch.ones_like(ids_tensor)
        with torch.no_grad():
            out = self._model(ids_tensor, attention_mask=mask)
        pooled = self._pool(out.last_hidden_state, mask)
        return pooled[0].numpy()

    def embed_tokens_batch(self, batch_ids: List[List[int]]) -> np.ndarray:
        max_len = max(len(x) for x in batch_ids)
        padded = []
        masks = []
        for seq in batch_ids:
            pad_len = max_len - len(seq)
            padded.append(seq + [0] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)

        ids_tensor = torch.tensor(padded, dtype=torch.long)
        mask_tensor = torch.tensor(masks, dtype=torch.long)

        with torch.no_grad():
            out = self._model(ids_tensor, attention_mask=mask_tensor)
        pooled = self._pool(out.last_hidden_state, mask_tensor)
        return pooled.numpy()

    def dim(self) -> int:
        return self._dim
