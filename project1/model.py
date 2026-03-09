"""
project1/model.py
─────────────────
Prediction model layer — three options ordered by dependency weight.
预测模型层，按依赖从轻到重排列，三个选项。

1. RidgeForecaster     — NumPy ridge regression, zero deps, deterministic
                         NumPy 岭回归，零依赖，完全确定性
2. GraphAlphaNet       — PyTorch + PyG, full GAT training with Pearson loss
                         PyTorch + PyG，完整 GAT 训练，Pearson loss
3. TemporalStockMemory — Cross-time_id EMA embedding (for Ubiquant)
                         跨 time_id EMA embedding（Ubiquant 专用）
"""
from __future__ import annotations

import numpy as np


# ── 1. RidgeForecaster (OpenClaw, preserved) ────────────────────────────────
# ── 1. RidgeForecaster（OpenClaw 原版，保留）────────────────────────────────

class RidgeForecaster:
    """
    Closed-form ridge regression: w* = (X^T X + αI)^{-1} X^T y
    闭式岭回归：w* = (X^T X + αI)^{-1} X^T y

    Zero dependencies, fully deterministic — suitable for fast baselines and CI.
    零依赖，结果完全确定性，适合快速基线和持续集成测试。
    """
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha  = alpha
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        xtx = x.T @ x
        # Solve normal equations with L2 regularization / 求解加了 L2 正则的正规方程
        self.coef_ = np.linalg.solve(xtx + self.alpha * np.eye(xtx.shape[0]), x.T @ y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit() first.")
        return x @ self.coef_


# ── 2. GraphAlphaNet (our design, PyTorch optional) ─────────────────────────
# ── 2. GraphAlphaNet（我们的设计，PyTorch 可选）────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    _TORCH = True
except ImportError:
    _TORCH = False


if _TORCH:
    class _GATEncoder(nn.Module):
        """
        Two-layer GATConv encoder with BatchNorm and Dropout.
        两层 GATConv 编码器，每层后接 BatchNorm + Dropout。

        [OpenClaw] PyG GATConv framework + BatchNorm/Dropout engineering
        [OpenClaw] PyG GATConv 框架 + BatchNorm/Dropout 工程化改进
        [Ours] edge_weight passed through to GATConv so structural graph
               weights directly modulate attention scores
        [我们] edge_weight 透传给 GATConv，让结构边权重直接影响 attention 计算
        """
        def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 32,
                     heads: int = 4, dropout: float = 0.3):
            super().__init__()
            self.bn0   = nn.BatchNorm1d(in_dim)
            self.conv1 = GATConv(in_dim, hidden, heads=heads,
                                 dropout=dropout, add_self_loops=True)
            self.bn1   = nn.BatchNorm1d(hidden * heads)
            self.conv2 = GATConv(hidden * heads, out_dim, heads=1,
                                 concat=False, dropout=dropout)
            self.drop  = nn.Dropout(dropout)

        def forward(self, x, edge_index, edge_weight=None):
            x = self.bn0(x)
            x = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_weight)))
            x = self.drop(x)
            return self.conv2(x, edge_index, edge_attr=edge_weight)

    class GraphAlphaNet(nn.Module):
        """
        Merged prediction model:
        合并版预测模型：
          X(N×F) → GATEncoder → embed(N×32)
          → concat [embed || X] → MLP(128→64→1) → pred(N,)

        Loss: Pearson correlation (directly optimizes competition metric).
        Loss：Pearson 相关（直接对齐竞赛指标，比 MSE 更合适）。

        Why Pearson over MSE?
        为什么用 Pearson 而不用 MSE？
          Ubiquant evaluates per-snapshot rank correlation, not absolute error.
          Ubiquant 竞赛按截面 rank correlation 评估，而非绝对误差。
          Optimizing MSE can push predictions toward large-cap stocks with
          high signal magnitude, which biases the cross-sectional ranking.
          优化 MSE 会偏向信号绝对值大的大盘股，破坏截面排序。
        """
        def __init__(self, in_dim: int, hidden: int = 64,
                     embed_dim: int = 32, heads: int = 4, dropout: float = 0.3):
            super().__init__()
            self.encoder = _GATEncoder(in_dim, hidden, embed_dim, heads, dropout)
            head_in = embed_dim + in_dim
            self.head = nn.Sequential(
                nn.Linear(head_in, 128), nn.ReLU(),
                nn.BatchNorm1d(128), nn.Dropout(dropout),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(64, 1),
            )

        def forward(self, x, edge_index, edge_weight=None):
            emb = self.encoder(x, edge_index, edge_weight)
            # Fuse graph embedding with raw alpha factors
            # 融合图 embedding 与原始 alpha 因子
            return self.head(torch.cat([x, emb], dim=1))

        @staticmethod
        def pearson_loss(pred, target):
            """
            Minimize negative Pearson correlation: L = -corr(pred, target)
            最小化负 Pearson 相关：L = -corr(pred, target)

            This directly optimizes the per-time_id IC metric used in Ubiquant.
            直接优化 Ubiquant 使用的 per-time_id IC 指标。
            """
            p = pred - pred.mean()
            t = target - target.mean()
            return -(p * t).sum() / (p.norm() * t.norm() + 1e-8)

        def fit(self, snapshots: list, lr: float = 1e-3,
                epochs: int = 50, device: str = "cpu") -> None:
            """
            Full training loop: Adam + CosineAnnealingLR + gradient clipping.
            完整训练循环：Adam + CosineAnnealing + 梯度裁剪。
            """
            self.to(device)
            opt   = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
            # Cosine schedule decays LR smoothly to 0 — avoids late-stage oscillation
            # Cosine 调度平滑衰减 LR 到 0，避免训练末期震荡
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            for ep in range(epochs):
                self.train()
                losses = []
                for (X_np, ei_np, ew_np, y_np) in snapshots:
                    X  = torch.tensor(X_np,  dtype=torch.float32, device=device)
                    ei = torch.tensor(ei_np, dtype=torch.long,    device=device)
                    ew = torch.tensor(ew_np, dtype=torch.float32, device=device)
                    y  = torch.tensor(y_np,  dtype=torch.float32, device=device)
                    opt.zero_grad()
                    pred = self(X, ei, ew).squeeze(-1)
                    loss = self.pearson_loss(pred, y)
                    loss.backward()
                    # Clip to prevent exploding gradients in deep GAT layers
                    # 裁剪防止深层 GAT 梯度爆炸
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    opt.step()
                    losses.append(loss.item())
                sched.step()
                if (ep + 1) % 10 == 0:
                    print(f"  epoch {ep+1:3d}/{epochs} | pearson_loss={np.mean(losses):.4f}")


# ── 3. TemporalStockMemory (our design) ─────────────────────────────────────
# ── 3. TemporalStockMemory（我们的设计）────────────────────────────────────

class TemporalStockMemory:
    """
    Cross-time_id EMA stock embedding — handles Ubiquant's changing universe.
    跨 time_id 的 EMA stock embedding，解决 Ubiquant 每期股票 universe 不同的问题。

    Update rule / 更新规则：
      z_i^t = β · z_i^{t-1}  +  (1-β) · gat_embed_i^t

    First appearance: initialize directly from current embedding.
    首次出现的股票直接用当期 embedding 初始化。

    Why EMA instead of concatenation?
    为什么用 EMA 而不是拼接历史序列？
      Universe size varies per snapshot (300–3800 stocks). Concatenation
      would require padding and variable-length sequences. EMA is O(1) memory
      per stock and handles entry/exit gracefully.
      每期 universe 大小不同（300–3800 只股票），拼接历史需要 padding 和
      变长序列处理。EMA 每只股票 O(1) 内存，且能优雅处理股票上市退市。
    """
    def __init__(self, max_inv_id: int = 3775,
                 embed_dim: int = 32, beta: float = 0.9) -> None:
        self.beta   = beta
        self.memory = np.zeros((max_inv_id + 1, embed_dim), dtype=np.float32)
        self.seen   = np.zeros(max_inv_id + 1, dtype=bool)

    def update(self, inv_ids: np.ndarray, embeds: np.ndarray) -> None:
        for i, iid in enumerate(inv_ids):
            if self.seen[iid]:
                # EMA update: blend past memory with new embedding
                # EMA 更新：融合历史记忆与新 embedding
                self.memory[iid] = (self.beta * self.memory[iid]
                                    + (1 - self.beta) * embeds[i])
            else:
                # Cold start: initialize from first observation
                # 冷启动：首次出现直接用当期值初始化
                self.memory[iid] = embeds[i]
                self.seen[iid]   = True

    def get(self, inv_ids: np.ndarray) -> np.ndarray:
        return self.memory[inv_ids]
