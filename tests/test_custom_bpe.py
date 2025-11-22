from axiomdb.tokenizers.custom_bpe import CustomBPETokenizer

def test_train_and_tokenize():
    tok = CustomBPETokenizer()
    tok.train("hello hello world", num_merges=50)

    ids = tok.tokenize("hello world")
    assert isinstance(ids, list)
    assert len(ids) > 0

def test_decode_roundtrip():
    text = "attention is all you need"
    tok = CustomBPETokenizer()
    tok.train(text, num_merges=30)

    ids = tok.tokenize(text)
    decoded = tok.decode(ids)

    assert "attention" in decoded
    assert "need" in decoded
