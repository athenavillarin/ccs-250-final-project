"""Check our grid max-flow against SciPy's reference solver and the max-flow/min-cut theorem."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from src.methods.classical import grid_maxflow


def reference_flow(s, t, right, down):
    """Max-flow value from SciPy on the same graph (test oracle only)."""
    h, w = s.shape
    n = h * w
    src, snk = n, n + 1
    rows, cols, caps = [], [], []
    idx = np.arange(n).reshape(h, w)
    for a, b, c in ((idx[:, :-1], idx[:, 1:], right), (idx[:-1, :], idx[1:, :], down)):
        rows += [a.ravel(), b.ravel()]
        cols += [b.ravel(), a.ravel()]
        caps += [c.ravel(), c.ravel()]
    rows += [np.full(n, src), idx.ravel()]
    cols += [idx.ravel(), np.full(n, snk)]
    caps += [s.ravel(), t.ravel()]
    graph = csr_matrix((np.concatenate(caps), (np.concatenate(rows), np.concatenate(cols))), shape=(n + 2, n + 2))
    return maximum_flow(graph, src, snk).flow_value


def cut_cost(mask, s, t, right, down):
    """Capacity of the cut where mask marks source-side pixels."""
    cost = s[~mask].sum() + t[mask].sum()
    cost += right[mask[:, :-1] & ~mask[:, 1:]].sum() + right[~mask[:, :-1] & mask[:, 1:]].sum()
    cost += down[mask[:-1, :] & ~mask[1:, :]].sum() + down[~mask[:-1, :] & mask[1:, :]].sum()
    return cost


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (6, 1), (5, 5), (12, 17)])
def test_matches_reference_and_cut(seed, shape):
    """Flow equals SciPy's value and the returned cut's capacity."""
    rng = np.random.default_rng(seed)
    h, w = shape
    s, t = rng.integers(0, 20, (2, h, w))
    right = rng.integers(0, 20, (h, w - 1))
    down = rng.integers(0, 20, (h - 1, w))
    flow, mask = grid_maxflow(s, t, right, down)
    assert flow == reference_flow(s, t, right, down)
    assert cut_cost(mask, s, t, right, down) == flow
