from axiomdb.core import AxiomDB
from axiomdb.tokenizers.custom_bpe import CustomBPETokenizer
from axiomdb.encoders.hf_bert import HFBERTEncoder
from axiomdb.index.hnswlib_index import HNSWLibIndex
from axiomdb.store.sqlite_store import SQLiteStore

tok = CustomBPETokenizer()
tok.train("This is a training corpus for the tokenizer.", num_merges=200)

enc = HFBERTEncoder()
idx = HNSWLibIndex()
idx.init(dim=enc.dim(), max_elements=1000)
store = SQLiteStore(":memory:")

db = AxiomDB(tok, enc, idx, store)

db.add("doc1", "hello world", {"source": "test"})
db.add("doc2", "attention is all you need", {"source": "test2"})

print(db.search("attention", k=1))
