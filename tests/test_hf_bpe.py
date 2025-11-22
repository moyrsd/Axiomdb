from axiomdb.tokenizers.hf_bpe import HFBPETokenizer


def test_hf_tokenizer_basic():
    tok = HFBPETokenizer()

    ids = tok.tokenize("hello world")
    assert isinstance(ids, list)
    assert len(ids) > 0

    batch = tok.tokenize_batch(["hello", "world"])
    assert len(batch) == 2
    assert all(isinstance(b, list) for b in batch)
