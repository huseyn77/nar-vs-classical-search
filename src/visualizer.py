"""
Visualization Module for Publication-Quality Paper Figures.

Generates and saves Matplotlib figures:
1. Training & Validation Loss / F1 curves
2. ROC & Precision-Recall (PR) Curves
3. Confusion Matrices
4. Performance Comparison (BFS vs Dijkstra vs A* vs GCN vs MPNN)
5. Generalization across maze scales
6. Maze solution visual overlays
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

# Matplotlib Academic Style Settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def plot_training_curves(
    gcn_history: Dict[str, Any],
    mpnn_history: Dict[str, Any],
    save_path: str = "figures/training_validation_curves.png"
) -> None:
    """Plot Training & Validation Loss and F1 curves for GCN and MPNN."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(gcn_history["train_loss"], label="GCN Train Loss", color="#1f77b4", linestyle="--")
    axes[0].plot(gcn_history["val_loss"], label="GCN Val Loss", color="#1f77b4")
    axes[0].plot(mpnn_history["train_loss"], label="MPNN Train Loss", color="#2ca02c", linestyle="--")
    axes[0].plot(mpnn_history["val_loss"], label="MPNN Val Loss", color="#2ca02c")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 plot
    axes[1].plot(gcn_history["train_f1"], label="GCN Train F1", color="#1f77b4", linestyle="--")
    axes[1].plot(gcn_history["val_f1"], label="GCN Val F1", color="#1f77b4")
    axes[1].plot(mpnn_history["train_f1"], label="MPNN Train F1", color="#2ca02c", linestyle="--")
    axes[1].plot(mpnn_history["val_f1"], label="MPNN Val F1", color="#2ca02c")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Training & Validation F1-Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curves(
    model_metrics: Dict[str, Dict[str, Any]],
    save_path: str = "figures/roc_curve.png"
) -> None:
    """Plot ROC curves for GCN and MPNN models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7, 6))

    colors = {"GCN": "#1f77b4", "MPNN": "#2ca02c"}

    for name, metrics in model_metrics.items():
        fpr, tpr, _ = roc_curve(metrics["y_true"], metrics["y_prob"])
        roc_auc = metrics.get("roc_auc", auc(fpr, tpr))
        plt.plot(fpr, tpr, label=f"{name} (ROC-AUC = {roc_auc:.4f})", color=colors.get(name, "black"), lw=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path)
    plt.close()


def plot_pr_curves(
    model_metrics: Dict[str, Dict[str, Any]],
    save_path: str = "figures/pr_curve.png"
) -> None:
    """Plot Precision-Recall (PR) curves for imbalanced class evaluation."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7, 6))

    colors = {"GCN": "#1f77b4", "MPNN": "#2ca02c"}

    for name, metrics in model_metrics.items():
        prec, rec, _ = precision_recall_curve(metrics["y_true"], metrics["y_prob"])
        ap_score = metrics.get("pr_auc", average_precision_score(metrics["y_true"], metrics["y_prob"]))
        plt.plot(rec, prec, label=f"{name} (AP / PR-AUC = {ap_score:.4f})", color=colors.get(name, "black"), lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curve (Class Imbalance)")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path)
    plt.close()


def plot_roc_and_pr_curves(
    model_metrics: Dict[str, Dict[str, Any]],
    save_path: str = "figures/roc_pr_combined.png"
) -> None:
    """Plot side-by-side ROC and Precision-Recall (PR) curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"GCN": "#1f77b4", "MPNN": "#2ca02c"}

    # ROC Subplot
    for name, metrics in model_metrics.items():
        fpr, tpr, _ = roc_curve(metrics["y_true"], metrics["y_prob"])
        roc_auc = metrics.get("roc_auc", auc(fpr, tpr))
        axes[0].plot(fpr, tpr, label=f"{name} (ROC-AUC = {roc_auc:.4f})", color=colors.get(name, "black"), lw=2)

    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # PR Subplot
    for name, metrics in model_metrics.items():
        prec, rec, _ = precision_recall_curve(metrics["y_true"], metrics["y_prob"])
        ap_score = metrics.get("pr_auc", average_precision_score(metrics["y_true"], metrics["y_prob"]))
        axes[1].plot(rec, prec, label=f"{name} (PR-AUC = {ap_score:.4f})", color=colors.get(name, "black"), lw=2)

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall (PR) Curve")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrices(
    model_metrics: Dict[str, Dict[str, Any]],
    save_path: str = "figures/confusion_matrix.png"
) -> None:
    """Plot confusion matrices for models side by side."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_models = len(model_metrics)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))

    if num_models == 1:
        axes = [axes]

    for idx, (name, metrics) in enumerate(model_metrics.items()):
        cm = metrics["confusion_matrix"]
        tau = metrics.get("threshold", 0.5)
        im = axes[idx].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[idx].set_title(f"Confusion Matrix ({name}, \\tau={tau:.2f})")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(["Not Path", "Path"])
        axes[idx].set_yticklabels(["Not Path", "Path"])

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[idx].text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_performance_comparison(
    classical_summary: Dict[str, Dict[str, float]],
    gcn_metrics: Dict[str, Any],
    mpnn_metrics: Dict[str, Any],
    save_path: str = "figures/inference_time_memory_comparison.png"
) -> None:
    """Plot Latency, Memory Usage, and Computational Efficiency across BFS, Dijkstra, A*, GCN, and MPNN."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    methods = ["BFS", "Dijkstra", "A*", "ResGCN", "Gated MPNN"]
    times = [
        classical_summary["BFS"]["avg_inference_time_ms"],
        classical_summary["Dijkstra"]["avg_inference_time_ms"],
        classical_summary["A*"]["avg_inference_time_ms"],
        gcn_metrics["avg_inference_time_ms"],
        mpnn_metrics["avg_inference_time_ms"]
    ]
    mems = [
        classical_summary["BFS"]["peak_memory_mb"],
        classical_summary["Dijkstra"]["peak_memory_mb"],
        classical_summary["A*"]["peak_memory_mb"],
        gcn_metrics["peak_memory_mb"],
        mpnn_metrics["peak_memory_mb"]
    ]
    efficiencies = [
        classical_summary["BFS"]["computational_efficiency_nodes_per_ms"],
        classical_summary["Dijkstra"]["computational_efficiency_nodes_per_ms"],
        classical_summary["A*"]["computational_efficiency_nodes_per_ms"],
        gcn_metrics["computational_efficiency_nodes_per_ms"],
        mpnn_metrics["computational_efficiency_nodes_per_ms"]
    ]

    colors = ["#d62728", "#ff7f0e", "#9467bd", "#1f77b4", "#2ca02c"]

    # Subplot 1: Inference Latency
    bars1 = axes[0].bar(methods, times, color=colors, width=0.5, edgecolor="black")
    axes[0].set_ylabel("Inference Time (ms)")
    axes[0].set_title("Inference Latency (Lower is Better)")
    axes[0].grid(True, axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=15)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(times), f"{yval:.2f}ms", ha='center', va='bottom', fontsize=10)

    # Subplot 2: Memory Usage
    bars2 = axes[1].bar(methods, mems, color=colors, width=0.5, edgecolor="black")
    axes[1].set_ylabel("Memory Usage (MB)")
    axes[1].set_title("Peak Memory Usage (Lower is Better)")
    axes[1].grid(True, axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=15)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(mems), f"{yval:.2f}MB", ha='center', va='bottom', fontsize=10)

    # Subplot 3: Computational Efficiency
    bars3 = axes[2].bar(methods, efficiencies, color=colors, width=0.5, edgecolor="black")
    axes[2].set_ylabel("Nodes Processed / ms")
    axes[2].set_title("Computational Efficiency (Higher is Better)")
    axes[2].grid(True, axis='y', alpha=0.3)
    axes[2].tick_params(axis='x', rotation=15)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(efficiencies), f"{yval:.1f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_generalization(
    scales: List[str],
    gcn_f1s: List[float],
    mpnn_f1s: List[float],
    save_path: str = "figures/generalization_performance.png"
) -> None:
    """Plot F1 generalization score across increasing maze scales."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))

    plt.plot(scales, [1.0] * len(scales), label="BFS (Exact Baseline)", color="#d62728", linestyle=":", lw=2)
    plt.plot(scales, gcn_f1s, label="ResGCN (Deep Residual)", marker="o", color="#1f77b4", lw=2)
    plt.plot(scales, mpnn_f1s, label="MPNN (NAR Gated)", marker="s", color="#2ca02c", lw=2)

    plt.xlabel("Maze Dimension / Difficulty Scale")
    plt.ylabel("Test F1-Score")
    plt.title("Generalization Across Unseen & Large Mazes")
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path)
    plt.close()


def plot_maze_prediction_overlay(
    meta: Dict[str, Any],
    gcn_probs: np.ndarray,
    mpnn_probs: np.ndarray,
    node_to_idx: Dict[tuple, int],
    save_path: str = "figures/maze_prediction_overlay.png"
) -> None:
    """Plot side-by-side visual comparison of maze, OpenCV entrance detection, BFS path, and GNN path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    grid = meta["binary_grid"]
    h, w = grid.shape
    start = meta["start_coords"]
    goal = meta["goal_coords"]
    shortest_path = meta["shortest_path"]

    # Panel A: Input PNG + Detected Entrances
    axes[0].imshow(grid, cmap="gray")
    axes[0].plot(start[1], start[0], 'go', markersize=10, label="Start (1st Opening)")
    axes[0].plot(goal[1], goal[0], 'ro', markersize=10, label="Goal (2nd Opening)")
    axes[0].set_title("Input Image & Entrance Detection")
    axes[0].legend(loc="upper right")
    axes[0].axis("off")

    # Panel B: BFS Ground Truth Path
    bfs_img = np.zeros((h, w, 3), dtype=np.float32)
    bfs_img[grid == 1] = [0.9, 0.9, 0.9]  # Walkable
    bfs_img[grid == 0] = [0.1, 0.1, 0.1]  # Wall
    for r, c in shortest_path:
        bfs_img[r, c] = [1.0, 0.0, 0.0]   # Red shortest path
    axes[1].imshow(bfs_img)
    axes[1].set_title(f"BFS Ground Truth Path\n(Length: {len(shortest_path)})")
    axes[1].axis("off")

    # Panel C: GCN Predicted Probabilities
    gcn_heatmap = np.zeros((h, w), dtype=np.float32)
    for coords, idx in node_to_idx.items():
        gcn_heatmap[coords] = gcn_probs[idx]
    axes[2].imshow(grid, cmap="gray", alpha=0.4)
    im2 = axes[2].imshow(gcn_heatmap, cmap="jet", alpha=0.8, vmin=0.0, vmax=1.0)
    axes[2].set_title("ResGCN Predicted Path Heatmap")
    axes[2].axis("off")

    # Panel D: MPNN Predicted Probabilities
    mpnn_heatmap = np.zeros((h, w), dtype=np.float32)
    for coords, idx in node_to_idx.items():
        mpnn_heatmap[coords] = mpnn_probs[idx]
    axes[3].imshow(grid, cmap="gray", alpha=0.4)
    im3 = axes[3].imshow(mpnn_heatmap, cmap="jet", alpha=0.8, vmin=0.0, vmax=1.0)
    axes[3].set_title("Gated MPNN Predicted Path Heatmap")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
