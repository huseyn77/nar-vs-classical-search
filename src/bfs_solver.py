"""
Classical Graph Algorithms (BFS, Dijkstra, A*) Baseline & Ground Truth Generator.

Computes exact shortest paths on maze graphs, generates ground-truth binary
node labels, and measures algorithmic parameters: Inference Time (ms),
Memory Usage (MB), and Computational Efficiency (nodes/ms).
"""

from typing import Tuple, List, Dict, Any
import time
import tracemalloc
import networkx as nx
import torch


def solve_bfs(
    G: nx.Graph,
    start_coords: Tuple[int, int],
    goal_coords: Tuple[int, int],
    node_to_idx: Dict[Tuple[int, int], int],
    num_nodes: int
) -> Tuple[List[Tuple[int, int]], torch.Tensor, Dict[str, Any]]:
    """Execute BFS to find exact shortest path, ground-truth labels, and performance parameters."""
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        shortest_path = nx.shortest_path(G, source=start_coords, target=goal_coords)
    except nx.NetworkXNoPath:
        tracemalloc.stop()
        raise ValueError(f"No valid path exists between {start_coords} and {goal_coords}.")

    elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_mb = peak_mem_bytes / (1024.0 * 1024.0)
    computational_efficiency = num_nodes / max(1e-6, elapsed_time_ms)  # nodes processed per ms

    path_node_indices = {node_to_idx[coord] for coord in shortest_path}
    y_labels = [1.0 if idx in path_node_indices else 0.0 for idx in range(num_nodes)]
    y = torch.tensor(y_labels, dtype=torch.float32)

    metrics = {
        "algorithm": "BFS",
        "execution_time_ms": elapsed_time_ms,
        "peak_memory_mb": peak_memory_mb,
        "computational_efficiency_nodes_per_ms": computational_efficiency,
        "path_length": len(shortest_path),
        "num_visited_nodes": len(G)
    }

    return shortest_path, y, metrics


def solve_dijkstra(
    G: nx.Graph,
    start_coords: Tuple[int, int],
    goal_coords: Tuple[int, int],
    node_to_idx: Dict[Tuple[int, int], int],
    num_nodes: int
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """Execute Dijkstra's Algorithm for exact shortest path and profiling metrics."""
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        shortest_path = nx.dijkstra_path(G, source=start_coords, target=goal_coords)
    except nx.NetworkXNoPath:
        tracemalloc.stop()
        raise ValueError(f"No valid Dijkstra path between {start_coords} and {goal_coords}.")

    elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_mb = peak_mem_bytes / (1024.0 * 1024.0)
    computational_efficiency = num_nodes / max(1e-6, elapsed_time_ms)

    metrics = {
        "algorithm": "Dijkstra",
        "execution_time_ms": elapsed_time_ms,
        "peak_memory_mb": peak_memory_mb,
        "computational_efficiency_nodes_per_ms": computational_efficiency,
        "path_length": len(shortest_path),
        "num_visited_nodes": len(G)
    }

    return shortest_path, metrics


def manhattan_heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Manhattan distance heuristic for A* algorithm on 4-connected grid graphs."""
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def solve_astar(
    G: nx.Graph,
    start_coords: Tuple[int, int],
    goal_coords: Tuple[int, int],
    node_to_idx: Dict[Tuple[int, int], int],
    num_nodes: int
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """Execute A* Search Algorithm with Manhattan distance heuristic."""
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        shortest_path = nx.astar_path(
            G, source=start_coords, target=goal_coords, heuristic=manhattan_heuristic
        )
    except nx.NetworkXNoPath:
        tracemalloc.stop()
        raise ValueError(f"No valid A* path between {start_coords} and {goal_coords}.")

    elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_mb = peak_mem_bytes / (1024.0 * 1024.0)
    computational_efficiency = num_nodes / max(1e-6, elapsed_time_ms)

    metrics = {
        "algorithm": "A*",
        "execution_time_ms": elapsed_time_ms,
        "peak_memory_mb": peak_memory_mb,
        "computational_efficiency_nodes_per_ms": computational_efficiency,
        "path_length": len(shortest_path),
        "num_visited_nodes": len(G)
    }

    return shortest_path, metrics
