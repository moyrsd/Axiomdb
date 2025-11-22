from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseStore(ABC):
    """Abstract metadata store."""

    @abstractmethod
    def add(self, internal_id: int, metadata: Dict[str, Any]) -> None:
        """Store metadata for a given internal integer ID."""
        raise NotImplementedError

    @abstractmethod
    def get(self, internal_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for an internal ID."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, internal_id: int) -> None:
        """Remove metadata."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Number of entries."""
        raise NotImplementedError
