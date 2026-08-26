"""
Graph Construction Module for Maze Representation.

Converts a 2D binary maze grid into a 4-directional NetworkX grid graph
and transforms it into a PyTorch Geometric Data object with 8-dimensional node feature vectors.
"""

from typing import Tuple, Dict, Any, List
import networkx as nx
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch_geometric.data import Data


def build_maze_graph(
    binary_grid: np.ndarray,
    start_coords: Tuple[int, int],
    goal_coords: Tuple[int, int]
) -> Tuple[nx.Graph, Dict[Tuple[int, int], int], Data]:
    """Convert binary maze grid to NetworkX graph and PyG Data object.

    Parameters
    ----------
    binary_grid : np.ndarray
        2D binary array (H, W) where 1 = walkable cell, 0 = wall.
    start_coords : Tuple[int, int]
        (row, col) coordinates of the Start node.
    goal_coords : Tuple[int, int]
        (row, col) coordinates of the Goal node.

    Returns
    -------
    G : nx.Graph
        NetworkX graph representing walkable maze topology.
    node_to_idx : Dict[Tuple[int, int], int]
        Mapping from grid (row, col) to 0-based node index.
    pyg_data : Data
        PyTorch Geometric Data instance containing node feature matrix `x`,
        `edge_index`, `pos`, and start/goal metadata.
    """
    h, w = binary_grid.shape
    G = nx.Graph()

    # Calculate distance to nearest wall cell for all walkable nodes
    wall_mask = (binary_grid == 0).astype(np.uint8)
    if np.any(wall_mask):
        dist_transform = distance_transform_edt(binary_grid)
        max_dist = dist_transform.max() if dist_transform.max() > 0 else 1.0
        norm_dist_transform = dist_transform / max_dist
    else:
        norm_dist_transform = np.ones((h, w), dtype=np.float32)

    # Max Euclidean distance across grid for relative distance normalization
    max_grid_dist = np.sqrt(h**2 + w**2)

    # 1. Add nodes for walkable cells
    walkable_coords: List[Tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if binary_grid[r, c] == 1:
                walkable_coords.append((r, c))

    node_to_idx = {coords: idx for idx, coords in enumerate(walkable_coords)}

    for coords in walkable_coords:
        G.add_node(coords)

    # 2. Add 4-directional edges between adjacent walkable cells
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r, c in walkable_coords:
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (nr, nc) in node_to_idx:
                G.add_edge((r, c), (nr, nc))

    # 3. Compute Geodesic Graph Distances to Start and Goal (BFS Positional Encodings)
    num_nodes = len(walkable_coords)
    try:
        dist_start_dict = nx.single_source_shortest_path_length(G, start_coords)
        dist_goal_dict = nx.single_source_shortest_path_length(G, goal_coords)
        max_gdist = max(1.0, float(dist_start_dict.get(goal_coords, num_nodes)))
    except Exception:
        dist_start_dict = {}
        dist_goal_dict = {}
        max_gdist = float(num_nodes)

    # 4. Construct 8-dimensional Node Feature Matrix X
    # Features: [norm_x, norm_y, walkable, dist_b, is_start, is_goal, norm_gdist_start, norm_gdist_goal]
    features = np.zeros((num_nodes, 8), dtype=np.float32)

    sr, sc = start_coords
    gr, gc = goal_coords

    for idx, (r, c) in enumerate(walkable_coords):
        norm_x = c / max(1, w - 1)
        norm_y = r / max(1, h - 1)
        walkable = 1.0
        dist_b = norm_dist_transform[r, c]
        is_start = 1.0 if (r, c) == start_coords else 0.0
        is_goal = 1.0 if (r, c) == goal_coords else 0.0

        # Relative Geodesic Graph Distance Positional Encodings
        gdist_s = dist_start_dict.get((r, c), max_gdist) / max_gdist
        gdist_g = dist_goal_dict.get((r, c), max_gdist) / max_gdist

        features[idx] = [
            norm_x, norm_y, walkable, dist_b,
            is_start, is_goal, gdist_s, gdist_g
        ]

    x = torch.tensor(features, dtype=torch.float32)

    # 4. Construct Edge Index (bidirectional)
    edge_list: List[Tuple[int, int]] = []
    for u, v in G.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        edge_list.append((u_idx, v_idx))
        edge_list.append((v_idx, u_idx))

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    pos = x[:, :2].clone()
    start_idx = node_to_idx[start_coords]
    goal_idx = node_to_idx[goal_coords]

    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        pos=pos,
        start_idx=torch.tensor(start_idx, dtype=torch.long),
        goal_idx=torch.tensor(goal_idx, dtype=torch.long),
        num_nodes=num_nodes
    )

    return G, node_to_idx, pyg_data
