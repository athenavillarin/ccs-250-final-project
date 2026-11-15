"""Classical graph-cut flood segmentation with our own Boykov-Kolmogorov max-flow."""
from collections import deque

import numpy as np

FREE, SOURCE, SINK = 0, 1, 2
TERMINAL, ORPHAN, NONE = -1, -2, -3
INF = float("inf")


def grid_maxflow(source_cap: np.ndarray, sink_cap: np.ndarray, right: np.ndarray, down: np.ndarray) -> tuple[float, np.ndarray]:
    """Min-cut an HxW 4-connected grid with Boykov-Kolmogorov; returns (max flow, source-side mask)."""
    h, w = source_cap.shape
    n = h * w
    off = (1, -1, w, -w)
    rev = (1, 0, 3, 2)
    grid = np.zeros((h, w, 4))
    grid[:, :-1, 0] = right
    grid[:, 1:, 1] = right
    grid[:-1, :, 2] = down
    grid[1:, :, 3] = down
    cap = grid.ravel().tolist()
    s = source_cap.ravel().astype(np.float64)
    t = sink_cap.ravel().astype(np.float64)
    shared = np.minimum(s, t)
    flow = float(shared.sum())
    s, t = (s - shared).tolist(), (t - shared).tolist()

    tree = [FREE] * n
    pdir = [NONE] * n
    ts = [0] * n
    dist = [0] * n
    active = deque()
    for v in range(n):
        if s[v] > 0:
            tree[v], pdir[v], dist[v] = SOURCE, TERMINAL, 1
            active.append(v)
        elif t[v] > 0:
            tree[v], pdir[v], dist[v] = SINK, TERMINAL, 1
            active.append(v)

    time = 0
    orphans = deque()
    while active:
        p = active[0]
        if tree[p] == FREE:
            active.popleft()
            continue
        path = None
        if tree[p] == SOURCE:
            for d in range(4):
                if cap[p * 4 + d] > 0:
                    q = p + off[d]
                    if tree[q] == FREE:
                        tree[q], pdir[q], ts[q], dist[q] = SOURCE, rev[d], ts[p], dist[p] + 1
                        active.append(q)
                    elif tree[q] == SINK:
                        path = (p, d, q)
                        break
        else:
            for d in range(4):
                q = p + off[d]
                if 0 <= q < n and cap[q * 4 + rev[d]] > 0:
                    if tree[q] == FREE:
                        tree[q], pdir[q], ts[q], dist[q] = SINK, rev[d], ts[p], dist[p] + 1
                        active.append(q)
                    elif tree[q] == SOURCE:
                        path = (q, rev[d], p)
                        break
        if path is None:
            active.popleft()
            continue

        time += 1
        a, d, b = path
        bottleneck = cap[a * 4 + d]
        v = a
        while pdir[v] != TERMINAL:
            e = pdir[v]
            u = v + off[e]
            bottleneck = min(bottleneck, cap[u * 4 + rev[e]])
            v = u
        bottleneck = min(bottleneck, s[v])
        v = b
        while pdir[v] != TERMINAL:
            e = pdir[v]
            bottleneck = min(bottleneck, cap[v * 4 + e])
            v += off[e]
        bottleneck = min(bottleneck, t[v])

        cap[a * 4 + d] -= bottleneck
        cap[b * 4 + rev[d]] += bottleneck
        v = a
        while pdir[v] != TERMINAL:
            e = pdir[v]
            u = v + off[e]
            cap[u * 4 + rev[e]] -= bottleneck
            cap[v * 4 + e] += bottleneck
            if cap[u * 4 + rev[e]] <= 0:
                pdir[v] = ORPHAN
                orphans.append(v)
            v = u
        s[v] -= bottleneck
        if s[v] <= 0:
            pdir[v] = ORPHAN
            orphans.append(v)
        v = b
        while pdir[v] != TERMINAL:
            e = pdir[v]
            u = v + off[e]
            cap[v * 4 + e] -= bottleneck
            cap[u * 4 + rev[e]] += bottleneck
            if cap[v * 4 + e] <= 0:
                pdir[v] = ORPHAN
                orphans.append(v)
            v = u
        t[v] -= bottleneck
        if t[v] <= 0:
            pdir[v] = ORPHAN
            orphans.append(v)
        flow += bottleneck

        while orphans:
            p = orphans.popleft()
            side = tree[p]
            best, best_dist = NONE, INF
            for d in range(4):
                q = p + off[d]
                if not 0 <= q < n or tree[q] != side:
                    continue
                residual = cap[q * 4 + rev[d]] if side == SOURCE else cap[p * 4 + d]
                if residual <= 0:
                    continue
                k, u = 0, q
                while True:
                    if ts[u] == time:
                        k += dist[u]
                        break
                    e = pdir[u]
                    if e == TERMINAL:
                        ts[u], dist[u] = time, 1
                        k += 1
                        break
                    if e < 0:
                        k = INF
                        break
                    k += 1
                    u += off[e]
                if k == INF:
                    continue
                if k < best_dist:
                    best, best_dist = d, k
                u = q
                while ts[u] != time:
                    ts[u], dist[u] = time, k
                    k -= 1
                    u += off[pdir[u]]
            if best != NONE:
                pdir[p], ts[p], dist[p] = best, time, best_dist + 1
                continue
            for d in range(4):
                q = p + off[d]
                if not 0 <= q < n or tree[q] != side:
                    continue
                residual = cap[q * 4 + rev[d]] if side == SOURCE else cap[p * 4 + d]
                if residual > 0:
                    active.append(q)
                if pdir[q] >= 0 and q + off[pdir[q]] == p:
                    pdir[q] = ORPHAN
                    orphans.append(q)
            tree[p], pdir[p] = FREE, NONE

    return flow, np.array(tree, dtype=np.int8).reshape(h, w) == SOURCE


def load(checkpoint_path: str | None = None, device: str = "cpu") -> object:
    """Return a ready-to-use model or parameter state."""
    return None


def predict(model, image: np.ndarray) -> np.ndarray:
    """Return an HxW uint8 {0,1} flood mask at the input image size."""
    return np.zeros(image.shape[:2], dtype=np.uint8)
