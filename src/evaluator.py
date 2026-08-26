"""
Evaluation and Metrics Engine for GNN Maze Solvers & Classical Baselines.

Calculates comprehensive classification, latency, memory, and computational efficiency parameters:
- Validation-based threshold optimization (Zero Test Data Leakage)
- Imbalanced Metrics: PR-AUC (Average Precision), ROC-AUC, F1-Score, Precision, Recall, Confusion Matrix
- Latency (ms), Peak Memory (MB), and Computational Throughput (nodes/ms).
"""

from typing import Dict, Any, List
import time
import tracemalloc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from src.dataset import MazeDataset


@torch.no_grad()
def find_optimal_threshold(
    model: nn.Module,
    val_dataset: MazeDataset,
    device: torch.device = torch.device("cpu")
) -> float:
    """Find the optimal classification threshold using ONLY the validation dataset.
    
    Prevents test data leakage by selecting tau that maximizes F1 score on validation predictions.
    """
    model.eval()
    model = model.to(device)

    all_probs = []
    all_targets = []

    for data in val_dataset:
        data = data.to(device)
        logits = model(data.x, data.edge_index).squeeze(-1)
        probs = torch.sigmoid(logits)

        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(data.y.long().cpu().numpy())

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)

    candidate_thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_f1 = -1.0

    for tau in candidate_thresholds:
        preds = (y_prob >= tau).astype(np.int64)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(tau)

    print(f"[Validation Threshold Tuner] Selected optimal threshold tau = {best_threshold:.3f} (Val F1 = {best_f1:.4f})")
    return best_threshold


@torch.no_grad()
def evaluate_model_performance(
    model: nn.Module,
    dataset: MazeDataset,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Evaluate a trained GNN model on a MazeDataset using a fixed validation-derived threshold."""
    model.eval()
    model = model.to(device)

    all_preds_binary = []
    all_probs = []
    all_targets = []

    total_inference_time_ms = 0.0
    total_nodes = 0

    tracemalloc.start()

    for data in dataset:
        data = data.to(device)
        total_nodes += data.num_nodes

        start_t = time.perf_counter()

        logits = model(data.x, data.edge_index).squeeze(-1)
        probs = torch.sigmoid(logits)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        total_inference_time_ms += elapsed_ms

        probs_np = probs.cpu().numpy()
        targets_np = data.y.long().cpu().numpy()

        preds_binary = (probs_np >= threshold).astype(np.int64)

        all_preds_binary.extend(preds_binary)
        all_probs.extend(probs_np)
        all_targets.extend(targets_np)

    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    num_samples = max(1, len(dataset))
    avg_inference_time_ms = total_inference_time_ms / num_samples
    peak_memory_mb = peak_mem_bytes / (1024.0 * 1024.0)
    computational_efficiency = total_nodes / max(1e-6, total_inference_time_ms)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds_binary)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc_score = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_score = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc_score,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "threshold": threshold,
        "avg_inference_time_ms": avg_inference_time_ms,
        "peak_memory_mb": peak_memory_mb,
        "computational_efficiency_nodes_per_ms": computational_efficiency,
        "y_true": y_true,
        "y_prob": y_prob
    }

    return metrics


def benchmark_all_classical_baselines(dataset: MazeDataset) -> Dict[str, Dict[str, float]]:
    """Benchmark BFS, Dijkstra, and A* classical algorithms across dataset."""
    results = {
        "BFS": {"total_time_ms": 0.0, "total_mem_mb": 0.0, "total_nodes": 0},
        "Dijkstra": {"total_time_ms": 0.0, "total_mem_mb": 0.0, "total_nodes": 0},
        "A*": {"total_time_ms": 0.0, "total_mem_mb": 0.0, "total_nodes": 0}
    }

    num_samples = max(1, len(dataset))

    for meta in dataset.metadata_list:
        for alg, key in [("BFS", "bfs_metrics"), ("Dijkstra", "dijkstra_metrics"), ("A*", "astar_metrics")]:
            alg_meta = meta[key]
            results[alg]["total_time_ms"] += alg_meta["execution_time_ms"]
            results[alg]["total_mem_mb"] += alg_meta["peak_memory_mb"]
            results[alg]["total_nodes"] += alg_meta["num_visited_nodes"]

    summary = {}
    for alg, data in results.items():
        avg_time = data["total_time_ms"] / num_samples
        avg_mem = data["total_mem_mb"] / num_samples
        comp_eff = data["total_nodes"] / max(1e-6, data["total_time_ms"])
        summary[alg] = {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "roc_auc": 1.0,
            "pr_auc": 1.0,
            "avg_inference_time_ms": avg_time,
            "peak_memory_mb": avg_mem,
            "computational_efficiency_nodes_per_ms": comp_eff
        }

    return summary


def build_graph_info_dataframe(dataset: MazeDataset) -> pd.DataFrame:
    """Construct a Pandas DataFrame containing metadata for each graph/maze sample."""
    rows = []
    for idx, (pyg_data, meta) in enumerate(zip(dataset.data_list, dataset.metadata_list)):
        rows.append({
            "Maze_ID": idx,
            "Image_Path": meta["image_path"],
            "Grid_Shape": f"{meta['grid_shape'][0]}x{meta['grid_shape'][1]}",
            "Num_Nodes": pyg_data.num_nodes,
            "Num_Edges": pyg_data.edge_index.shape[1],
            "Start_Coords": str(meta["start_coords"]),
            "Goal_Coords": str(meta["goal_coords"]),
            "Path_Length": len(meta["shortest_path"]),
            "BFS_Time_ms": round(meta["bfs_metrics"]["execution_time_ms"], 3),
            "Dijkstra_Time_ms": round(meta["dijkstra_metrics"]["execution_time_ms"], 3),
            "AStar_Time_ms": round(meta["astar_metrics"]["execution_time_ms"], 3)
        })

    return pd.DataFrame(rows)


def build_comparison_dataframe(
    classical_summary: Dict[str, Dict[str, float]],
    gcn_metrics: Dict[str, Any],
    mpnn_metrics: Dict[str, Any]
) -> pd.DataFrame:
    """Construct a structured Pandas DataFrame summarizing all comparative metrics."""
    rows = []

    for alg in ["BFS", "Dijkstra", "A*"]:
        m = classical_summary[alg]
        rows.append({
            "Method": alg,
            "Accuracy (%)": round(m["accuracy"] * 100, 2),
            "Precision (%)": round(m["precision"] * 100, 2),
            "Recall (%)": round(m["recall"] * 100, 2),
            "F1-Score": round(m["f1"], 4),
            "ROC-AUC (%)": round(m["roc_auc"] * 100, 2),
            "Average Precision (%)": round(m["pr_auc"] * 100, 2),
            "Inference Time (ms)": round(m["avg_inference_time_ms"], 2),
            "Memory Usage (MB)": round(m["peak_memory_mb"], 2),
            "Computational Efficiency (nodes/ms)": round(m["computational_efficiency_nodes_per_ms"], 1)
        })

    rows.append({
        "Method": "GCN (AdamW)",
        "Accuracy (%)": round(gcn_metrics["accuracy"] * 100, 2),
        "Precision (%)": round(gcn_metrics["precision"] * 100, 2),
        "Recall (%)": round(gcn_metrics["recall"] * 100, 2),
        "F1-Score": round(gcn_metrics["f1"], 4),
        "ROC-AUC (%)": round(gcn_metrics["roc_auc"] * 100, 2),
        "Average Precision (%)": round(gcn_metrics["pr_auc"] * 100, 2),
        "Inference Time (ms)": round(gcn_metrics["avg_inference_time_ms"], 2),
        "Memory Usage (MB)": round(gcn_metrics["peak_memory_mb"], 2),
        "Computational Efficiency (nodes/ms)": round(gcn_metrics["computational_efficiency_nodes_per_ms"], 1)
    })

    rows.append({
        "Method": "MPNN (AdamW)",
        "Accuracy (%)": round(mpnn_metrics["accuracy"] * 100, 2),
        "Precision (%)": round(mpnn_metrics["precision"] * 100, 2),
        "Recall (%)": round(mpnn_metrics["recall"] * 100, 2),
        "F1-Score": round(mpnn_metrics["f1"], 4),
        "ROC-AUC (%)": round(mpnn_metrics["roc_auc"] * 100, 2),
        "Average Precision (%)": round(mpnn_metrics["pr_auc"] * 100, 2),
        "Inference Time (ms)": round(mpnn_metrics["avg_inference_time_ms"], 2),
        "Memory Usage (MB)": round(mpnn_metrics["peak_memory_mb"], 2),
        "Computational Efficiency (nodes/ms)": round(mpnn_metrics["computational_efficiency_nodes_per_ms"], 1)
    })

    return pd.DataFrame(rows)
