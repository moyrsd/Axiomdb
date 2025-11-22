import sqlite3
import orjson
from typing import Any, Dict, Optional
from .base import BaseStore


class SQLiteStore(BaseStore):
    """SQLite metadata store."""

    def __init__(self, path: str = "axiomdb_metadata.sqlite"):
        self._path = path
        self._conn = sqlite3.connect(self._path)
        self._create_table()

    def _create_table(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY,
                data BLOB NOT NULL
            );
            """
        )
        self._conn.commit()

    def add(self, internal_id: int, metadata: Dict[str, Any]) -> None:
        blob = orjson.dumps(metadata)
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO metadata (id, data) VALUES (?, ?)",
            (internal_id, blob),
        )
        self._conn.commit()

    def get(self, internal_id: int) -> Optional[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT data FROM metadata WHERE id = ?", (internal_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return orjson.loads(row[0])

    def delete(self, internal_id: int) -> None:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM metadata WHERE id = ?", (internal_id,))
        self._conn.commit()

    def count(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM metadata")
        return cur.fetchone()[0]
