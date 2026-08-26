"""
Main Entry Point for End-to-End NAR Maze Solver Experiment.

Executes dataset creation with randomized entrance/exit locations, model training (GCN & MPNN),
validation-based threshold selection (Zero Test Data Leakage), and performance evaluation
across classical baselines (BFS, Dijkstra, A*) and neural models using imbalanced metrics
(PR-AUC, ROC-AUC, Precision, Recall, F1, Confusion Matrix).
"""

import os
import pandas as pd
import torch

from dataset_generator import prepare_kaggle_maze_dataset
from src.dataset import create_maze_dataset_from_dir
from src.graph_builder import build_maze_graph
from src.models import GCNMazeSolver, MPNNMazeSolver
from src.trainer import train_model, set_seed
from src.evaluator import (
    evaluate_model_performance, find_optimal_threshold, benchmark_all_classical_baselines,
    build_graph_info_dataframe, build_comparison_dataframe
)
from src.visualizer import (
    plot_training_curves, plot_roc_curves, plot_pr_curves, plot_roc_and_pr_curves,
    plot_confusion_matrices, plot_performance_comparison, plot_generalization,
    plot_maze_prediction_overlay
)


def run_experiment():
    """Run full experimental pipeline."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing experiment on device: {device}")

    # 1. Prepare clean datasets with randomized entrance/exit positions
    project_root = os.path.dirname(__file__)
    kaggle_dir = os.path.join(project_root, "maze")
    data_base = os.path.join(project_root, "data")
    print("\n--- Step 1: Loading & Preparing Maze Datasets (21x21 Standard, 31x31 Large, 41x41 Very Large) ---")
    prepare_kaggle_maze_dataset(kaggle_dir, data_base, num_train=50, num_val=15, num_test=15, num_unseen=15)

    train_dir = os.path.join(data_base, "train")
    val_dir = os.path.join(data_base, "val")
    test_dir = os.path.join(data_base, "test")
    unseen_dir = os.path.join(data_base, "unseen")
    large_dir = os.path.join(data_base, "large")
    very_large_dir = os.path.join(data_base, "very_large")

    # 2. Build PyG Maze Datasets
    print("\n--- Step 2: Processing Graph Representations & Algorithmic Labels ---")
    train_ds = create_maze_dataset_from_dir(train_dir)
    val_ds = create_maze_dataset_from_dir(val_dir)
    test_ds = create_maze_dataset_from_dir(test_dir)
    unseen_ds = create_maze_dataset_from_dir(unseen_dir)
    large_ds = create_maze_dataset_from_dir(large_dir)
    very_large_ds = create_maze_dataset_from_dir(very_large_dir)

    print(f"Loaded Datasets: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # Display Pandas DataFrame with Graph Info
    print("\n--- Pandas DataFrame: Test Graphs Information & Topological Properties ---")
    graph_info_df = build_graph_info_dataframe(test_ds)
    print(graph_info_df.head(10).to_string(index=False))

    fig_dir = os.path.join(project_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    graph_info_df.to_csv(os.path.join(fig_dir, "graph_info_table.csv"), index=False)

    # 3. Benchmark Classical Baselines (BFS, Dijkstra, A*)
    print("\n--- Step 3: Benchmarking Classical Baselines (BFS, Dijkstra, A*) ---")
    classical_summary = benchmark_all_classical_baselines(test_ds)

    # 4. Train Models with AdamW & Focal Loss (Deep Message Passing)
    print("\n--- Step 4: Training ResGCN Model (AdamW & Focal Loss, 30 Layers) ---")
    gcn_model = GCNMazeSolver(in_channels=8, hidden_channels=64, num_layers=30, dropout=0.1)
    gcn_history = train_model(
        gcn_model, train_ds, val_ds, model_name="GCN", epochs=40, lr=1e-3, patience=10, seed=42
    )

    print("\n--- Step 5: Training NAR Gated MPNN Model (AdamW & Focal Loss, 30 Steps) ---")
    mpnn_model = MPNNMazeSolver(in_channels=8, hidden_channels=64, num_steps=30, dropout=0.1)
    mpnn_history = train_model(
        mpnn_model, train_ds, val_ds, model_name="MPNN", epochs=40, lr=1e-3, patience=10, seed=42
    )

    # 5. Validation-Based Threshold Selection (Zero Test Data Leakage)
    print("\n--- Step 6: Validation Threshold Selection (Zero Test Data Leakage) ---")
    gcn_tau = find_optimal_threshold(gcn_model, val_ds, device)
    mpnn_tau = find_optimal_threshold(mpnn_model, val_ds, device)

    # 6. Evaluation on Standard Test Set
    print("\n--- Step 7: Evaluating Models on Test Set using Validation-Selected Thresholds ---")
    gcn_test_metrics = evaluate_model_performance(gcn_model, test_ds, device, threshold=gcn_tau)
    mpnn_test_metrics = evaluate_model_performance(mpnn_model, test_ds, device, threshold=mpnn_tau)

    comparison_df = build_comparison_dataframe(classical_summary, gcn_test_metrics, mpnn_test_metrics)
    comparison_df.to_csv(os.path.join(fig_dir, "algorithm_comparison_table.csv"), index=False)

    print("\n=========================================================================================================")
    print("                              PANDAS DATAFRAME: ALGORITHM COMPARISON TABLE                               ")
    print("=========================================================================================================")
    print(comparison_df.to_string(index=False))
    print("=========================================================================================================\n")

    # 7. Generalization Assessment
    print("\n--- Step 8: Assessing Generalization Across Scales ---")
    gcn_unseen = evaluate_model_performance(gcn_model, unseen_ds, device, threshold=gcn_tau)
    gcn_large = evaluate_model_performance(gcn_model, large_ds, device, threshold=gcn_tau)
    gcn_very_large = evaluate_model_performance(gcn_model, very_large_ds, device, threshold=gcn_tau)

    mpnn_unseen = evaluate_model_performance(mpnn_model, unseen_ds, device, threshold=mpnn_tau)
    mpnn_large = evaluate_model_performance(mpnn_model, large_ds, device, threshold=mpnn_tau)
    mpnn_very_large = evaluate_model_performance(mpnn_model, very_large_ds, device, threshold=mpnn_tau)

    scales = ["Standard 21x21", "Unseen 21x21", "Large 31x31", "Very Large 41x41"]
    gcn_f1s = [gcn_test_metrics["f1"], gcn_unseen["f1"], gcn_large["f1"], gcn_very_large["f1"]]
    mpnn_f1s = [mpnn_test_metrics["f1"], mpnn_unseen["f1"], mpnn_large["f1"], mpnn_very_large["f1"]]

    # 8. Generate Figures
    print("\n--- Step 9: Generating Publication Figures (ROC & PR Curves, Heatmaps) ---")
    plot_training_curves(gcn_history, mpnn_history, os.path.join(fig_dir, "training_validation_curves.png"))
    plot_roc_curves({"GCN": gcn_test_metrics, "MPNN": mpnn_test_metrics}, os.path.join(fig_dir, "roc_curve.png"))
    plot_pr_curves({"GCN": gcn_test_metrics, "MPNN": mpnn_test_metrics}, os.path.join(fig_dir, "pr_curve.png"))
    plot_roc_and_pr_curves({"GCN": gcn_test_metrics, "MPNN": mpnn_test_metrics}, os.path.join(fig_dir, "roc_pr_combined.png"))
    plot_confusion_matrices({"GCN": gcn_test_metrics, "MPNN": mpnn_test_metrics}, os.path.join(fig_dir, "confusion_matrix.png"))
    plot_performance_comparison(classical_summary, gcn_test_metrics, mpnn_test_metrics, os.path.join(fig_dir, "inference_time_memory_comparison.png"))
    plot_generalization(scales, gcn_f1s, mpnn_f1s, os.path.join(fig_dir, "generalization_performance.png"))

    # Sample visual overlay
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

    plot_maze_prediction_overlay(
        sample_meta, gcn_probs, mpnn_probs, node_to_idx, os.path.join(fig_dir, "maze_prediction_overlay.png")
    )

    print("All figures and CSV tables successfully created in 'figures/' directory!")


if __name__ == "__main__":
    run_experiment()
