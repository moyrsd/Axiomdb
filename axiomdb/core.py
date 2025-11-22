from typing import Any, Dict, List, Optional
from .tokenizers.base import BaseTokenizer
from .encoders.base import BaseEncoder
from .index.base import BaseIndex
from .store.base import BaseStore


class AxiomDB:
    """AxiomDB orchestrator connecting tokenizer, encoder, index, and store."""

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        encoder: BaseEncoder,
        index: BaseIndex,
        store: BaseStore,
    ):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.index = index
        self.store = store

        # internal ID counter
        self._next_internal_id = 0

        # external string ID -> internal ID
        self._ext_to_int = {}

    def add(self, external_id: str, text: str, metadata: Dict[str, Any]) -> int:
        """Add a document and return its internal ID."""
        if external_id in self._ext_to_int:
            raise ValueError("external_id already exists")

        # assign internal ID
        internal_id = self._next_internal_id
        self._next_internal_id += 1
        self._ext_to_int[external_id] = internal_id

        # store metadata
        self.store.add(internal_id, metadata)

        # tokenize and encode
        token_ids = self.tokenizer.tokenize(text)
        vec = self.encoder.embed_tokens(token_ids)

        # index vector
        self.index.add(vec, internal_id)

        return internal_id

    def search(self, text: str, k: int) -> List[str]:
        """Search nearest neighbors by text query."""
        token_ids = self.tokenizer.tokenize(text)
        vec = self.encoder.embed_tokens(token_ids)
        ids, _ = self.index.search(vec, k)

        # map internal back to external
        result = []
        for iid in ids:
            for ext, x in self._ext_to_int.items():
                if x == iid:
                    result.append(ext)
                    break
        return result

    def get_metadata(self, external_id: str) -> Optional[Dict[str, Any]]:
        if external_id not in self._ext_to_int:
            return None
        iid = self._ext_to_int[external_id]
        return self.store.get(iid)

    def count(self) -> int:
        return self.store.count()
