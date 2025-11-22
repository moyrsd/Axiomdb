from axiomdb.encoders.hf_bert import HFBERTEncoder


def test_encoder_dim():
    enc = HFBERTEncoder()
    assert enc.dim() == 768 


def test_encoder_embedding_shapes():
    enc = HFBERTEncoder()

    vec = enc.embed_tokens([101, 7592, 102])  
    assert vec.shape == (enc.dim(),)

    batch = enc.embed_tokens_batch([[101, 7592], [101, 2088]])
    assert batch.shape[0] == 2
    assert batch.shape[1] == enc.dim()
