"""Classical graph-cut flood segmentation with our own Boykov-Kolmogorov max-flow."""
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.paths import CHECKPOINT_DIR

FREE, SOURCE, SINK = 0, 1, 2
TERMINAL, ORPHAN, NONE = -1, -2, -3
INF = float("inf")
WORK_SIDE = 256
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "classical.npz"


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



def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert uint8 sRGB to CIE Lab (D65) as float32."""
    rgb = image.astype(np.float32) / 255
    rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = rgb @ m.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def kmeans(x: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    """Cluster rows of x with k-means++ seeding and Lloyd iterations; returns labels."""
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d2 = ((x[:, None] - np.array(centers)[None]) ** 2).sum(-1).min(1)
        centers.append(x[rng.choice(len(x), p=d2 / d2.sum())])
    centers = np.array(centers)
    for _ in range(iters):
        labels = ((x[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
        new = np.array([x[labels == j].mean(0) if np.any(labels == j) else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return labels


def fit_gmm(x: np.ndarray, k: int, seed: int = 0) -> dict:
    """Fit a full-covariance GMM by taking one Gaussian per k-means cluster."""
    labels = kmeans(x, k, seed=seed)
    comps = [x[labels == j] for j in range(k) if np.sum(labels == j) > x.shape[1]]
    return {
        "weights": np.array([len(c) / len(x) for c in comps]),
        "means": np.array([c.mean(0) for c in comps]),
        "covs": np.array([np.cov(c.T) + 1e-3 * np.eye(x.shape[1]) for c in comps]),
    }


def gmm_nll(x: np.ndarray, gmm: dict) -> np.ndarray:
    """Negative log-likelihood of each row of x under a GMM."""
    logs = []
    for w, mu, cov in zip(gmm["weights"], gmm["means"], gmm["covs"]):
        diff = x - mu
        maha = np.einsum("ni,ij,nj->n", diff, np.linalg.inv(cov), diff)
        logs.append(np.log(w) - 0.5 * (maha + np.linalg.slogdet(cov)[1] + x.shape[1] * np.log(2 * np.pi)))
    logs = np.array(logs)
    top = logs.max(0)
    return -(top + np.log(np.exp(logs - top).sum(0)))


def unary(lab: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel costs of labeling non-flood (D0) and flood (D1)."""
    x = lab.reshape(-1, 3)
    d0 = gmm_nll(x, model["background"]) - np.log(1 - model["prior"])
    d1 = gmm_nll(x, model["flood"]) - np.log(model["prior"])
    return d0.reshape(lab.shape[:2]), d1.reshape(lab.shape[:2])


def pairwise(lab: np.ndarray, lam: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Contrast-sensitive Potts weights lam * exp(-|Ii - Ij|^2 / 2 sigma^2) for right and down edges."""
    dr = ((lab[:, 1:] - lab[:, :-1]) ** 2).sum(-1)
    dd = ((lab[1:] - lab[:-1]) ** 2).sum(-1)
    return lam * np.exp(-dr / (2 * sigma**2)), lam * np.exp(-dd / (2 * sigma**2))


def resize(image: np.ndarray, side: int, resample=Image.BILINEAR) -> np.ndarray:
    """Shrink an array so its long side is at most `side`."""
    img = Image.fromarray(image)
    img.thumbnail((side, side), resample)
    return np.asarray(img)


def segment(lab: np.ndarray, model: dict) -> np.ndarray:
    """Minimize the graph-cut energy on a Lab image; returns a bool flood mask."""
    d0, d1 = unary(lab, model)
    low = np.minimum(d0, d1)
    right, down = pairwise(lab, model["lam"], model["sigma"])
    _, flood = grid_maxflow(d0 - low, d1 - low, right, down)
    return flood


def fit(pairs, k: int = 5, per_image: int = 2000, lam: float = 5.0, sigma: float = 30.0, seed: int = 0) -> dict:
    """Learn flood and background colour GMMs from (image, mask) training pairs."""
    rng = np.random.default_rng(seed)
    pixels = {0: [], 1: []}
    flood_px = total_px = 0
    for image, mask in pairs:
        lab = rgb_to_lab(resize(image, WORK_SIDE)).reshape(-1, 3)
        small = resize(mask, WORK_SIDE, Image.NEAREST).ravel()
        flood_px += int(small.sum())
        total_px += small.size
        for c in (0, 1):
            idx = np.flatnonzero(small == c)
            if len(idx):
                pixels[c].append(lab[rng.choice(idx, min(per_image, len(idx)), replace=False)])
    return {
        "background": fit_gmm(np.concatenate(pixels[0]), k, seed),
        "flood": fit_gmm(np.concatenate(pixels[1]), k, seed),
        "prior": flood_px / total_px,
        "lam": lam,
        "sigma": sigma,
    }


def save(model: dict, path: str | Path = DEFAULT_CHECKPOINT) -> None:
    """Write model parameters to an .npz file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    flat = {f"{c}_{key}": v for c in ("background", "flood") for key, v in model[c].items()}
    np.savez(path, prior=model["prior"], lam=model["lam"], sigma=model["sigma"], **flat)


def load(checkpoint_path: str | None = None, device: str = "cpu") -> dict:
    """Load fitted colour models and smoothness parameters from an .npz file."""
    data = np.load(checkpoint_path or DEFAULT_CHECKPOINT)
    model = {c: {key: data[f"{c}_{key}"] for key in ("weights", "means", "covs")} for c in ("background", "flood")}
    model.update(prior=float(data["prior"]), lam=float(data["lam"]), sigma=float(data["sigma"]))
    return model


def predict(model, image: np.ndarray) -> np.ndarray:
    """Segment at WORK_SIDE resolution and upsample the mask to the input size."""
    flood = segment(rgb_to_lab(resize(image, WORK_SIDE)), model)
    mask = Image.fromarray(flood.astype(np.uint8)).resize((image.shape[1], image.shape[0]), Image.NEAREST)
    return np.asarray(mask)
