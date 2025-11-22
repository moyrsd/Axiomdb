from axiomdb.core import AxiomDB
from axiomdb.tokenizers.hf_bpe import HFBPETokenizer
from axiomdb.encoders.hf_bert import HFBERTEncoder
from axiomdb.index.hnswlib_index import HNSWLibIndex
from axiomdb.store.sqlite_store import SQLiteStore


def test_axiomdb_end_to_end():
    tok = HFBPETokenizer()
    enc = HFBERTEncoder()
    idx = HNSWLibIndex()
    idx.init(dim=enc.dim(), max_elements=100)
    store = SQLiteStore(":memory:")

    db = AxiomDB(tok, enc, idx, store)

    db.add("doc1", "hello world", {"x": 1})
    db.add("doc2", "vector search testing", {"x": 2})

    results = db.search("hello", k=1)
    assert results[0] == "doc1"

    meta = db.get_metadata("doc1")
    assert meta["x"] == 1
