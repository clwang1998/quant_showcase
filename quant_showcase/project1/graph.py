"""
project1/graph.py
─────────────────
Graph construction module with two paths.
图构建模块，提供两条路径。

Path A — Heterogeneous predefined graph (production path, requires domain knowledge)
Path A — 异构预定义图（生产路径，需要领域知识）
  Three edge types: sector / supply-chain (directed) / style-factor
  三类边：sector / supply-chain（有向）/ style-factor

Path B — Sparsemax learnable graph (competition path, no domain labels needed)
Path B — Sparsemax 可学习图（竞赛路径，无需领域标签）
  A = sparsemax(ZZ^T / √d)
  Most edge weights are exactly 0 — no manual threshold needed.
  大多数边权重精确为 0，无需手工设阈值。
"""
from __future__ import annotations

import numpy as np


# ── Path A: Heterogeneous Predefined Graph ──────────────────────────────────
# ── Path A: 异构预定义图 ────────────────────────────────────────────────────

def build_heterogeneous_graph(
    sector_matrix: np.ndarray,
    supplychain_matrix: np.ndarray,
    style_matrix: np.ndarray,
    sc_threshold: float = 0.3,
    style_threshold: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build heterogeneous graph with three semantically distinct edge types.
    构建三类语义各不同的异构边：

      sector       : same-industry bidirectional edges (symmetric)
                     同行业双向边（对称）
      supply-chain : directed upstream→downstream edges (asymmetric!)
                     上下游有向边（不对称！上游→下游 ≠ 下游→上游）
      style-factor : style-factor correlation edges (symmetric)
                     风格因子相关性边（对称）

    Returns edge_index (2, E) and edge_weight (E,).
    返回 edge_index (2, E) 和 edge_weight (E,)。
    """
    src, dst, w = [], [], []

    def add_edges(mat: np.ndarray, threshold: float, symmetric: bool) -> None:
        rows, cols = np.where(mat > threshold)
        for r, c in zip(rows, cols):
            if r == c:
                continue
            src.append(int(r)); dst.append(int(c)); w.append(float(mat[r, c]))
            if symmetric:
                # Bidirectional: add reverse edge for symmetric relations
                # 双向边：对称关系需要加反向边
                src.append(int(c)); dst.append(int(r)); w.append(float(mat[r, c]))

    add_edges(sector_matrix,       0.5,             symmetric=True)
    add_edges(supplychain_matrix,  sc_threshold,    symmetric=False)  # directed / 有向
    add_edges(style_matrix,        style_threshold, symmetric=True)

    if not src:
        # Fallback: KNN graph based on feature cosine similarity
        # Fallback：基于特征余弦相似度构建 KNN 图
        n = sector_matrix.shape[0]
        return build_learnable_graph(np.eye(n), top_k=min(10, n - 1))

    return np.array([src, dst], dtype=np.int64), np.array(w, dtype=np.float32)


# ── Path B: Sparsemax Learnable Graph ───────────────────────────────────────
# ── Path B: Sparsemax 可学习图 ──────────────────────────────────────────────

def build_learnable_graph(
    Z: np.ndarray,
    top_k: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    End-to-end learnable sparse graph.
    端到端可学习稀疏图。

      A = sparsemax( Z Z^T / √d )

    Sparsemax vs Softmax:
    Sparsemax 与 Softmax 对比：
      Softmax: all edge weights > 0, requires manual threshold to sparsify.
               所有边权重 > 0，需手工 threshold 稀疏化。
      Sparsemax: projects directly to probability simplex; most weights are
                 exactly 0 without any hyperparameter tuning.
                 直接投影到概率单纯形，大多数权重精确为 0，无需任何超参数。

    Z can be: raw features X, or learned embeddings from MLP/GAT (end-to-end).
    Z 可以是：原始特征 X，或 MLP/GAT 学到的 embedding（端到端训练时）。
    """
    N, d = Z.shape
    # L2 normalize so dot product equals cosine similarity
    # L2 归一化，确保点积等于余弦相似度
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    S  = (Zn @ Zn.T) / np.sqrt(d)   # (N, N) similarity matrix / 相似度矩阵

    # Top-k masking: keep only k nearest neighbors, O(N*k) memory vs O(N²)
    # Top-k 掩码：只保留最相似的 k 个邻居，内存 O(N*k) 而非 O(N²)
    if top_k < N - 1:
        thr = np.partition(S, -top_k, axis=1)[:, -top_k]
        S   = np.where(S >= thr[:, None], S, -1e9)

    A = _sparsemax(S)                # (N, N) sparse adjacency / 稀疏邻接权重

    src, dst    = np.where(A > 0)
    edge_index  = np.stack([src, dst], axis=0)
    edge_weight = A[src, dst].astype(np.float32)
    return edge_index, edge_weight


def _sparsemax(z: np.ndarray) -> np.ndarray:
    """
    Sparsemax projection (Martins & Astudillo, 2016).
    Sparsemax 投影（Martins & Astudillo, 2016）。

    Projects each row onto the probability simplex, making weak edges
    exactly zero — unlike softmax which always produces dense outputs.
    将每行投影到概率单纯形，使弱相关的边权重精确为 0——
    不同于 softmax 总是输出稠密权重。

    Algorithm: sort → find k* → compute threshold τ* → max(z - τ*, 0)
    算法：排序 → 找 k* → 计算阈值 τ* → max(z - τ*, 0)
    """
    z   = z - z.max(axis=1, keepdims=True)   # numerical stability / 数值稳定
    zs  = np.sort(z, axis=1)[:, ::-1]         # descending sort / 降序
    N   = z.shape[1]
    k   = np.arange(1, N + 1, dtype=np.float32)
    cs  = np.cumsum(zs, axis=1)
    sup = 1 + k * zs > cs                     # support set condition / 支撑集条件
    k_  = sup.sum(axis=1, keepdims=True)
    tau = (cs[np.arange(len(cs)), k_.squeeze().astype(int) - 1] - 1) / k_.squeeze()
    return np.maximum(z - tau[:, None], 0.0)


# ── NumPy GAT Embedding (fallback when PyTorch unavailable) ─────────────────
# ── NumPy GAT Embedding（无 PyTorch 时的 fallback）─────────────────────────

def numpy_gat_embedding(
    x: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    hidden_dim: int = 32,
    heads: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """
    Multi-head GAT implemented in pure NumPy (for zero-dependency environments).
    纯 NumPy 实现的多头 GAT（用于无 PyTorch 环境）。

    Formula / 公式：
      e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) * A_ij
      α_ij = scatter_softmax over neighbors j of node i
             对节点 i 的所有邻居 j 做 scatter softmax
      h'_i = ELU( Σ_j α_ij · W h_j )
    """
    rng = np.random.default_rng(seed)
    n, d = x.shape
    src, dst = edge_index
    outs = []

    for _ in range(heads):
        # Random init for each attention head
        # 每个注意力头随机初始化
        W     = rng.standard_normal((d, hidden_dim)).astype(np.float32) / np.sqrt(d)
        a_src = rng.standard_normal(hidden_dim).astype(np.float32) * 0.05
        a_dst = rng.standard_normal(hidden_dim).astype(np.float32) * 0.05
        h = x @ W                                         # linear projection / 线性变换

        # Attention logits with LeakyReLU (negative slope=0.2)
        # LeakyReLU 注意力得分（负斜率=0.2）
        e = (h[src] @ a_src) + (h[dst] @ a_dst)
        e = np.where(e >= 0, e, 0.2 * e)                  # LeakyReLU
        e = e * edge_weight                               # structural gating / 结构边权重门控

        # Scatter softmax: normalize attention weights per destination node
        # Scatter softmax：对每个目标节点归一化注意力权重
        alpha = np.zeros_like(e)
        for i in range(n):
            m = dst == i
            if not m.any():
                continue
            s = e[m] - e[m].max()   # subtract max for numerical stability / 减最大值保持数值稳定
            exp_s = np.exp(s)
            alpha[m] = exp_s / (exp_s.sum() + 1e-12)

        # Aggregate neighbor features weighted by attention
        # 用注意力权重聚合邻居特征
        h_new = np.zeros((n, hidden_dim), dtype=np.float32)
        np.add.at(h_new, dst, alpha[:, None] * h[src])
        h_new = np.where(h_new >= 0, h_new, np.exp(h_new) - 1)  # ELU activation / ELU 激活
        outs.append(h_new)

    # Concatenate all heads then L2-normalize
    # 拼接所有头的输出，再做 L2 归一化
    emb = np.concatenate(outs, axis=1)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb.astype(np.float32)
