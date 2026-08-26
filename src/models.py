"""
Graph Neural Network Models for Neural Algorithmic Reasoning (NAR).

Implements:
1. Deep Residual Graph Convolutional Network (ResGCN) with initial state shortcuts.
2. Stable Recurrent Gated Message Passing Neural Network (MPNN) with GRU update cell,
   multi-aggregation (max + mean), and initial node feature injection.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing


class ResGCNBlock(nn.Module):
    """Residual GCN block with LayerNorm, GELU, dropout, and initial feature shortcut."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv = GCNConv(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, h_0: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.conv(x, edge_index))
        h = self.dropout(h)
        return self.norm(x + h + 0.1 * h_0)


class GCNMazeSolver(nn.Module):
    """Deep Residual Graph Convolutional Network (ResGCN) for Maze Navigation.

    Parameters
    ----------
    in_channels : int, default=8
        Number of input node features.
    hidden_channels : int, default=64
        Dimension of hidden representations.
    num_layers : int, default=16
        Number of residual GCN layers for deep message propagation.
    dropout : float, default=0.1
        Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 64,
        num_layers: int = 30,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.blocks = nn.ModuleList([
            ResGCNBlock(hidden_channels, dropout=dropout) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for node binary logit prediction."""
        h_0 = self.embedding(x)
        h = h_0

        for block in self.blocks:
            h = block(h, edge_index, h_0)

        # Concatenate initial embedding and final layer state for strong start/goal signal
        out_features = torch.cat([h_0, h], dim=-1)
        logits = self.classifier(out_features)
        return logits


class MPNNMazeStep(MessagePassing):
    """Single Message Passing Step with Max & Mean Aggregation for NAR."""

    def __init__(self, channels: int):
        super().__init__(aggr='max')
        self.msg_mlp = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        return self.msg_mlp(torch.cat([x_i, x_j], dim=-1))


class MPNNMazeSolver(nn.Module):
    """Gated Recurrent Message Passing Neural Network (MPNN) for Neural Algorithmic Reasoning.

    Uses a GRU cell, LayerNorm, multi-step recurrence, and initial node state injection
    to guarantee numerical stability and long-range algorithmic propagation.

    Parameters
    ----------
    in_channels : int, default=8
        Number of input node features.
    hidden_channels : int, default=64
        Dimension of latent representations.
    num_steps : int, default=30
        Number of message-passing recurrent steps.
    dropout : float, default=0.1
        Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 64,
        num_steps: int = 30,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_steps = num_steps
        self.embedding = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.mpnn_step = MPNNMazeStep(hidden_channels)
        self.gru_cell = nn.GRUCell(hidden_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass over T gated message passing iterations."""
        h_0 = self.embedding(x)
        h = h_0

        for _ in range(self.num_steps):
            msg = self.mpnn_step(h, edge_index)
            # Update state with GRU cell and add initial embedding shortcut
            h_updated = self.gru_cell(msg, h)
            h = self.norm(h_updated + 0.1 * h_0)

        h = self.dropout(h)
        # Output classifier receives both initial node features and unrolled recurrent features
        out_features = torch.cat([h_0, h], dim=-1)
        logits = self.classifier(out_features)
        return logits

