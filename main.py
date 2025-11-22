from axiomdb.core import AxiomDB
from axiomdb.tokenizers.hf_bpe import HFBPETokenizer
from axiomdb.encoders.hf_bert import HFBERTEncoder
from axiomdb.index.hnswlib_index import HNSWLibIndex
from axiomdb.store.sqlite_store import SQLiteStore
import pymupdf


def extract_pdf_text(path: str) -> str:
    doc = pymupdf.open(path)
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)


def chunk_text(text: str, max_chars: int = 500):
    chunks = []
    current = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if sum(len(x) for x in current) + len(line) > max_chars:
            chunks.append(" ".join(current))
            current = []
        current.append(line)

    if current:
        chunks.append(" ".join(current))

    return chunks


# Initialize AxiomDB
tok = HFBPETokenizer()
enc = HFBERTEncoder()
idx = HNSWLibIndex()
idx.init(dim=enc.dim(), max_elements=5000)
store = SQLiteStore("pdf_meta.sqlite")

db = AxiomDB(tok, enc, idx, store)



text = extract_pdf_text("/home/moyrsd/Dev/Axiomdb/axiomdb/example.pdf")
chunks = chunk_text(text)

# Add to DB
for i, chunk in enumerate(chunks):
    db.add(f"chunk_{i}", chunk, {"source": "pdf", "text": chunk})

print("Indexed chunks:", db.count())


# Search example
query = "What problem does the Transformer architecture solve compared to recurrent models?"
results = db.search(query, k=5)
print("Results:", results)

for ext_id in results:
    print(ext_id, db.get_metadata(ext_id))
