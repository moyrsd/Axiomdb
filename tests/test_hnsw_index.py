import numpy as np
from axiomdb.index.hnswlib_index import HNSWLibIndex


def test_index_add_and_search():
    dim = 4
    idx = HNSWLibIndex()
    idx.init(dim=dim, max_elements=10)

    vecs = np.random.rand(5, dim).astype(np.float32)
    ids = list(range(5))

    idx.add_batch(vecs, ids)

    q = vecs[0]
    res_ids, res_dist = idx.search(q, k=3)

    assert len(res_ids) == 3
    assert len(res_dist) == 3
    assert res_ids[0] == 0
