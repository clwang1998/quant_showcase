from __future__ import annotations

import numpy as np


def build_heterogeneous_graph(
    sector_matrix: np.ndarray,
    supplychain_matrix: np.ndarray,
    style_matrix: np.ndarray,
    sc_threshold: float = 0.3,
    style_threshold: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    src, dst, w = [], [], []

    def add_edges(mat: np.ndarray, threshold: float, symmetric: bool) -> None:
        rows, cols = np.where(mat > threshold)
        for r, c in zip(rows, cols):
            if r == c:
                continue
            src.append(int(r))
            dst.append(int(c))
            w.append(float(mat[r, c]))
            if symmetric:
                src.append(int(c))
                dst.append(int(r))
                w.append(float(mat[r, c]))

    add_edges(sector_matrix, 0.5, symmetric=True)
    add_edges(supplychain_matrix, sc_threshold, symmetric=False)
    add_edges(style_matrix, style_threshold, symmetric=True)

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_weight = np.array(w, dtype=np.float32)
    return edge_index, edge_weight


def simple_multihead_gat_embedding(
    x: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    hidden_dim: int = 32,
    heads: int = 4,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, d = x.shape
    src, dst = edge_index

    outs = []
    for _ in range(heads):
        w = rng.standard_normal((d, hidden_dim)) / np.sqrt(d)
        a_src = rng.standard_normal(hidden_dim) * 0.05
        a_dst = rng.standard_normal(hidden_dim) * 0.05
        h = x @ w
        e = (h[src] @ a_src) + (h[dst] @ a_dst)
        e = np.where(e >= 0, e, 0.2 * e)
        e = e * edge_weight

        alpha = np.zeros_like(e)
        for i in range(n):
            m = dst == i
            if not np.any(m):
                continue
            scores = e[m]
            scores = scores - np.max(scores)
            exps = np.exp(scores)
            alpha[m] = exps / (np.sum(exps) + 1e-12)

        h_new = np.zeros((n, hidden_dim), dtype=np.float32)
        np.add.at(h_new, dst, alpha[:, None] * h[src])
        h_new = np.where(h_new >= 0, h_new, np.exp(h_new) - 1)
        outs.append(h_new)

    emb = np.concatenate(outs, axis=1)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb
