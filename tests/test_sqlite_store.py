from axiomdb.store.sqlite_store import SQLiteStore


def test_store_basic():
    store = SQLiteStore(":memory:")

    store.add(0, {"text": "hello"})
    store.add(1, {"text": "world"})

    assert store.count() == 2
    assert store.get(0)["text"] == "hello"

    store.delete(0)
    assert store.count() == 1
