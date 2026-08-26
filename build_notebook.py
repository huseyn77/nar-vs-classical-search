"""
Builder script to generate neural_algorithmic_reasoning_maze.ipynb
"""

import json
import os

def create_notebook():
    notebook_path = "neural_algorithmic_reasoning_maze.ipynb"

    def code_cell(source):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.strip().split("\n")
        }

    def markdown_cell(source):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source.strip().split("\n")
        }

    cells = []

    # Title & Markdown intro
    cells.append(markdown_cell("""
# Neural Algorithmic Reasoning for Graph-Based Maze Solving
### Comparative Evaluation of Classical Search Algorithms (BFS, Dijkstra, A*) vs Deep Graph Architectures (ResGCN, Gated MPNN)

**Project Goal:**  
Benchmark classical search algorithms (BFS, Dijkstra, A*) against deep Neural Algorithmic Reasoning (NAR) models—specifically **Deep Residual GCNs** and **Recurrent Gated Message Passing Neural Networks (MPNN)**—on graph maze navigation problems.

---
### Key Features & Methodological Rigor:
1. **Zero Test-Data Leakage**: Classification thresholds ($\tau^*$) are selected strictly on the validation dataset to maximize validation F1-score (`find_optimal_threshold`), and then applied as fixed thresholds across test/unseen datasets.
2. **Imbalanced Class Evaluation**: Evaluation emphasizes **Precision-Recall Curves (PR-AUC / Average Precision)**, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices to account for the ~75% negative class imbalance.
3. **Deep Receptive Field Propagation**: Models utilize 30 residual layers (ResGCN) and 30 recurrent message-passing iterations (MPNN) to match the graph diameter (~100–120 vertices).
4. **Randomized Boundary Entrances**: Automated detection of Start and Goal entrance openings across all 4 outer maze boundaries.
""".strip()))

    # Imports cell
    cells.append(code_cell("""
import os
import random
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import to_undirected
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
""".strip()))

    # Seed setting
    cells.append(code_cell("""
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
""".strip()))

    # Dataset Generator & Graph Builder
    cells.append(markdown_cell("""
## 1. Dataset Generation & Graph Construction
Generates 2D binary mazes with randomized entrance/exit locations and converts them into 4-connected NetworkX / PyG grid graphs.
""".strip()))

    cells.append(code_cell("""
from dataset_generator import prepare_kaggle_maze_dataset
from src.dataset import create_maze_dataset_from_dir
from src.graph_builder import build_maze_graph

project_root = os.getcwd()
kaggle_dir = os.path.join(project_root, "maze")
data_base = os.path.join(project_root, "data")

print("Generating/Preparing Clean Maze Datasets...")
prepare_kaggle_maze_dataset(kaggle_dir, data_base, num_train=30, num_val=10, num_test=10, num_unseen=10)

train_dir = os.path.join(data_base, "train")
val_dir = os.path.join(data_base, "val")
test_dir = os.path.join(data_base, "test")
unseen_dir = os.path.join(data_base, "unseen")
large_dir = os.path.join(data_base, "large")
very_large_dir = os.path.join(data_base, "very_large")

train_ds = create_maze_dataset_from_dir(train_dir)
val_ds = create_maze_dataset_from_dir(val_dir)
test_ds = create_maze_dataset_from_dir(test_dir)
unseen_ds = create_maze_dataset_from_dir(unseen_dir)
large_ds = create_maze_dataset_from_dir(large_dir)
very_large_ds = create_maze_dataset_from_dir(very_large_dir)

print(f"Dataset Sizes -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
""".strip()))

    # Graph Information DataFrame
    cells.append(code_cell("""
from src.evaluator import build_graph_info_dataframe

graph_info_df = build_graph_info_dataframe(test_ds)
display(graph_info_df.head(10))
""".strip()))

    # Models definition
    cells.append(markdown_cell("""
## 2. Model Architectures: ResGCN & Gated MPNN
- **Deep ResGCN (30 Layers)**: Residual GCN blocks with LayerNorm, GELU, and initial state shortcuts ($0.1 h_0$).
- **Recurrent Gated MPNN (30 Steps)**: Recurrent MessagePassing module using GRU cell updates and initial state injection.
""".strip()))

    cells.append(code_cell("""
from src.models import GCNMazeSolver, MPNNMazeSolver
from src.trainer import train_model

gcn_model = GCNMazeSolver(in_channels=8, hidden_channels=64, num_layers=30, dropout=0.1)
mpnn_model = MPNNMazeSolver(in_channels=8, hidden_channels=64, num_steps=30, dropout=0.1)

print("ResGCN Architecture:")
print(gcn_model)

print("\nGated MPNN Architecture:")
print(mpnn_model)
""".strip()))

    # Training
    cells.append(markdown_cell("""
## 3. Training NAR Models (AdamW & Focal BCE Loss)
Train both models using **AdamW** optimizer, **Focal BCE Loss** ($\gamma = 2.0$), and early stopping.
""".strip()))

    cells.append(code_cell("""
print("--- Training Deep ResGCN (30 Layers) ---")
gcn_history = train_model(gcn_model, train_ds, val_ds, model_name="GCN", epochs=40, lr=1e-3, patience=10, seed=42)

print("\n--- Training Recurrent Gated MPNN (30 Steps) ---")
mpnn_history = train_model(mpnn_model, train_ds, val_ds, model_name="MPNN", epochs=40, lr=1e-3, patience=10, seed=42)
""".strip()))

    # Validation Threshold Tuning & Test Evaluation
    cells.append(markdown_cell("""
## 4. Zero Test-Data Leakage: Validation Threshold Tuning & Test Evaluation
Validation threshold $\tau^*$ is selected strictly on the validation dataset to maximize Validation F1-score, and locked for test evaluation.
""".strip()))

    cells.append(code_cell("""
from src.evaluator import (
    find_optimal_threshold, evaluate_model_performance,
    benchmark_all_classical_baselines, build_comparison_dataframe
)

# Benchmark Classical Baselines
classical_summary = benchmark_all_classical_baselines(test_ds)

# Find Validation Thresholds (Zero Leakage)
gcn_tau = find_optimal_threshold(gcn_model, val_ds, device)
mpnn_tau = find_optimal_threshold(mpnn_model, val_ds, device)

print(f"Validation Selected Thresholds -> ResGCN tau*: {gcn_tau:.3f}, Gated MPNN tau*: {mpnn_tau:.3f}")

# Evaluate on Test Set
gcn_test = evaluate_model_performance(gcn_model, test_ds, device, threshold=gcn_tau)
mpnn_test = evaluate_model_performance(mpnn_model, test_ds, device, threshold=mpnn_tau)

comparison_df = build_comparison_dataframe(classical_summary, gcn_test, mpnn_test)
display(comparison_df)
""".strip()))

    # Visualization
    cells.append(markdown_cell("""
## 5. Visualizing Evaluation Metrics & Solution Overlays
Plotting ROC & PR curves, confusion matrices, latency/memory benchmarks, scale generalization, and maze solution heatmaps.
""".strip()))

    cells.append(code_cell("""
from src.visualizer import (
    plot_training_curves, plot_roc_and_pr_curves, plot_confusion_matrices,
    plot_performance_comparison, plot_generalization, plot_maze_prediction_overlay
)

# 1. Training Curves
plot_training_curves(gcn_history, mpnn_history)

# 2. ROC & PR Curves Combined
plot_roc_and_pr_curves({"GCN": gcn_test, "MPNN": mpnn_test})

# 3. Confusion Matrices
plot_confusion_matrices({"GCN": gcn_test, "MPNN": mpnn_test})

# 4. Performance Comparison
plot_performance_comparison(classical_summary, gcn_test, mpnn_test)

# 5. Scale Generalization
gcn_unseen = evaluate_model_performance(gcn_model, unseen_ds, device, threshold=gcn_tau)
gcn_large = evaluate_model_performance(gcn_model, large_ds, device, threshold=gcn_tau)
gcn_very_large = evaluate_model_performance(gcn_model, very_large_ds, device, threshold=gcn_tau)

mpnn_unseen = evaluate_model_performance(mpnn_model, unseen_ds, device, threshold=mpnn_tau)
mpnn_large = evaluate_model_performance(mpnn_model, large_ds, device, threshold=mpnn_tau)
mpnn_very_large = evaluate_model_performance(mpnn_model, very_large_ds, device, threshold=mpnn_tau)

scales = ["Standard 21x21", "Unseen 21x21", "Large 31x31", "Very Large 41x41"]
gcn_f1s = [gcn_test["f1"], gcn_unseen["f1"], gcn_large["f1"], gcn_very_large["f1"]]
mpnn_f1s = [mpnn_test["f1"], mpnn_unseen["f1"], mpnn_large["f1"], mpnn_very_large["f1"]]

plot_generalization(scales, gcn_f1s, mpnn_f1s)

# 6. Maze Prediction Heatmap Overlay
sample_meta = test_ds.get_metadata(0)
sample_data = test_ds[0].to(device)

with torch.no_grad():
    gcn_logits = gcn_model(sample_data.x, sample_data.edge_index).squeeze(-1)
    mpnn_logits = mpnn_model(sample_data.x, sample_data.edge_index).squeeze(-1)
    gcn_probs = torch.sigmoid(gcn_logits).cpu().numpy()
    mpnn_probs = torch.sigmoid(mpnn_logits).cpu().numpy()

G, node_to_idx, _ = build_maze_graph(
    sample_meta["binary_grid"], sample_meta["start_coords"], sample_meta["goal_coords"]
)

plot_maze_prediction_overlay(sample_meta, gcn_probs, mpnn_probs, node_to_idx)
""".strip()))

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    print(f"Jupyter notebook '{notebook_path}' generated successfully!")

if __name__ == "__main__":
    create_notebook()
