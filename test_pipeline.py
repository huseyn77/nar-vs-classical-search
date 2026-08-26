"""
Unit & Integration Test Suite for NAR Maze Solver Pipeline.
"""

import os
import shutil
import tempfile
import numpy as np
import torch

from dataset_generator import generate_maze_grid, save_maze_image
from src.image_processor import load_and_preprocess_image, detect_boundary_entrances
from src.graph_builder import build_maze_graph
from src.bfs_solver import solve_bfs
from src.models import GCNMazeSolver, MPNNMazeSolver
from src.evaluator import evaluate_model_performance, find_optimal_threshold, build_comparison_dataframe
from src.dataset import MazeDataset


def test_end_to_end_pipeline():
    """Test entire image -> graph -> BFS -> GNN forward pass & threshold pipeline."""
    temp_dir = tempfile.mkdtemp()
    try:
        maze_path = os.path.join(temp_dir, "test_maze.png")
        grid = generate_maze_grid(height=21, width=21, seed=42)
        save_maze_image(grid, maze_path)
        assert os.path.exists(maze_path), "Maze PNG was not saved properly."

        # 1. Image processing & entrance detection
        binary_grid = load_and_preprocess_image(maze_path)
        assert binary_grid.shape == (21, 21), f"Unexpected grid shape {binary_grid.shape}"
        start_coords, goal_coords, openings = detect_boundary_entrances(binary_grid)
        assert len(openings) == 2, f"Expected 2 openings, got {len(openings)}"
        print(f"[PASS] OpenCV Entrance Detection: Start={start_coords}, Goal={goal_coords}")

        # 2. Graph construction
        G, node_to_idx, pyg_data = build_maze_graph(binary_grid, start_coords, goal_coords)
        assert pyg_data.x.shape[1] == 8, f"Node feature dimension should be 8, got {pyg_data.x.shape[1]}"
        print(f"[PASS] NetworkX & PyG Graph Construction: Nodes={pyg_data.num_nodes}, Edges={pyg_data.edge_index.shape[1]}")

        # 3. BFS ground truth calculation
        path, y, metrics = solve_bfs(G, start_coords, goal_coords, node_to_idx, pyg_data.num_nodes)
        assert len(path) > 0, "BFS failed to find a path"
        assert y.shape[0] == pyg_data.num_nodes, "Label shape mismatch"
        print(f"[PASS] BFS Baseline: Path length={len(path)}, Time={metrics['execution_time_ms']:.2f} ms")

        # 4. GCN Forward Pass (30 Layers)
        gcn = GCNMazeSolver(in_channels=8, hidden_channels=64, num_layers=30)
        logits_gcn = gcn(pyg_data.x, pyg_data.edge_index)
        assert logits_gcn.shape == (pyg_data.num_nodes, 1), f"GCN output shape mismatch {logits_gcn.shape}"
        print(f"[PASS] ResGCN Forward Pass (30 Layers): Output shape={logits_gcn.shape}")

        # 5. MPNN Forward Pass (30 Steps)
        mpnn = MPNNMazeSolver(in_channels=8, hidden_channels=64, num_steps=30)
        logits_mpnn = mpnn(pyg_data.x, pyg_data.edge_index)
        assert logits_mpnn.shape == (pyg_data.num_nodes, 1), f"MPNN output shape mismatch {logits_mpnn.shape}"
        print(f"[PASS] MPNN Forward Pass (30 Steps): Output shape={logits_mpnn.shape}")

        # 6. Evaluation with Dataset
        dataset = MazeDataset([maze_path])
        tau = find_optimal_threshold(gcn, dataset)
        eval_metrics = evaluate_model_performance(gcn, dataset, threshold=tau)
        assert "pr_auc" in eval_metrics, "PR-AUC missing from metrics"
        assert "roc_auc" in eval_metrics, "ROC-AUC missing from metrics"
        print(f"[PASS] Evaluator & PR-AUC: Val Tau={tau:.3f}, PR-AUC={eval_metrics['pr_auc']:.4f}, ROC-AUC={eval_metrics['roc_auc']:.4f}")

        print("\nALL PIPELINE TESTS PASSED SUCCESSFULLY!")

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_end_to_end_pipeline()
