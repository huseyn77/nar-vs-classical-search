"""
Training Engine for Graph Neural Network Maze Solvers.

Handles optimization loops with Focal BCE Loss, AdamW optimizer,
cosine annealing learning rate scheduling, gradient norm clipping, early stopping,
and model checkpointing.
"""

from typing import Dict, Any, Tuple, Optional
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from src.dataset import MazeDataset


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def calculate_pos_weight(dataset: MazeDataset) -> torch.Tensor:
    """Calculate positive class weight (negative_count / positive_count) for BCE loss."""
    total_pos = 0.0
    total_neg = 0.0
    for data in dataset:
        pos = (data.y == 1.0).sum().item()
        neg = (data.y == 0.0).sum().item()
        total_pos += pos
        total_neg += neg

    pos_weight = total_neg / max(1.0, total_pos)
    return torch.tensor([pos_weight], dtype=torch.float32)


class FocalBCEWithLogitsLoss(nn.Module):
    """Focal Binary Cross Entropy Loss for Class-Imbalanced Graph Nodes.

    Parameters
    ----------
    pos_weight : torch.Tensor
        Weight factor for positive node class.
    gamma : float, default=2.0
        Focusing parameter for modulating factor (1 - p_t)^gamma.
    """

    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        focal_factor = (1.0 - pt) ** self.gamma
        focal_loss = focal_factor * bce_loss
        return focal_loss.mean()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index).squeeze(-1)
        loss = criterion(logits, batch.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_nodes
        total_samples += batch.num_nodes

        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(batch.y.long().cpu().numpy())

    epoch_loss = total_loss / max(1, total_samples)
    epoch_f1 = f1_score(all_targets, all_preds, zero_division=0)
    return epoch_loss, epoch_f1


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)

        logits = model(batch.x, batch.edge_index).squeeze(-1)
        loss = criterion(logits, batch.y)

        total_loss += loss.item() * batch.num_nodes
        total_samples += batch.num_nodes

        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(batch.y.long().cpu().numpy())

    epoch_loss = total_loss / max(1, total_samples)
    epoch_f1 = f1_score(all_targets, all_preds, zero_division=0)
    return epoch_loss, epoch_f1


def train_model(
    model: nn.Module,
    train_dataset: MazeDataset,
    val_dataset: MazeDataset,
    model_name: str = "GCN",
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    save_dir: str = "models",
    seed: int = 42
) -> Dict[str, Any]:
    """Complete training loop with AdamW, Focal Loss, LR scheduling, early stopping, and checkpointing.

    Returns training history and metrics dictionary.
    """
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    pos_weight = calculate_pos_weight(train_dataset).to(device)
    criterion = FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=2.0)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "training_time_sec": 0.0
    }

    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_path = os.path.join(save_dir, f"best_{model_name.lower()}.pt")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_f1 = evaluate_epoch(model, val_loader, criterion, device)

        scheduler.step(v_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(v_f1)

        print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {tr_loss:.4f} | Val Loss: {v_loss:.4f} | Train F1: {tr_f1:.4f} | Val F1: {v_f1:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    history["training_time_sec"] = time.time() - start_time

    # Load best checkpoint
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    return history

