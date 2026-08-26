"""
PyTorch Geometric Custom Dataset for Maze Graphs.

Loads raw maze PNG images, processes them into graph representations, computes
ground-truth classical algorithmic baselines (BFS, Dijkstra, A*), and constructs
a dataset for GNN training and evaluation.
"""

from typing import List, Optional, Callable, Dict, Any
import os
import glob
import torch
from torch_geometric.data import Dataset, Data

from src.image_processor import load_and_preprocess_image, detect_boundary_entrances
from src.graph_builder import build_maze_graph
from src.bfs_solver import solve_bfs, solve_dijkstra, solve_astar


class MazeDataset(Dataset):
    """Custom PyTorch Geometric Dataset for Maze Navigation.

    Parameters
    ----------
    image_paths : List[str]
        List of paths to maze PNG files.
    transform : Optional[Callable], optional
        Optional PyG transform function.
    pre_transform : Optional[Callable], optional
        Optional pre-transform function.
    """

    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None
    ):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.image_paths = sorted(image_paths)
        self.data_list: List[Data] = []
        self.metadata_list: List[Dict[str, Any]] = []

        self._process_images()

    def _process_images(self) -> None:
        """Process all PNG images into PyG Data objects with BFS/Dijkstra/A* targets."""
        for path in self.image_paths:
            try:
                binary_grid = load_and_preprocess_image(path)
                start_coords, goal_coords, openings = detect_boundary_entrances(binary_grid)
                G, node_to_idx, pyg_data = build_maze_graph(binary_grid, start_coords, goal_coords)

                # Solve with BFS baseline & ground truth
                shortest_path, y, bfs_metrics = solve_bfs(
                    G, start_coords, goal_coords, node_to_idx, pyg_data.num_nodes
                )

                # Solve with Dijkstra
                _, dijkstra_metrics = solve_dijkstra(
                    G, start_coords, goal_coords, node_to_idx, pyg_data.num_nodes
                )

                # Solve with A*
                _, astar_metrics = solve_astar(
                    G, start_coords, goal_coords, node_to_idx, pyg_data.num_nodes
                )

                pyg_data.y = y
                pyg_data.image_path = path

                meta = {
                    "image_path": path,
                    "grid_shape": binary_grid.shape,
                    "start_coords": start_coords,
                    "goal_coords": goal_coords,
                    "shortest_path": shortest_path,
                    "bfs_metrics": bfs_metrics,
                    "dijkstra_metrics": dijkstra_metrics,
                    "astar_metrics": astar_metrics,
                    "binary_grid": binary_grid
                }

                self.data_list.append(pyg_data)
                self.metadata_list.append(meta)

            except Exception as e:
                print(f"Warning: Skipping maze image '{path}' due to processing error: {e}")

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        return self.metadata_list[idx]


def create_maze_dataset_from_dir(directory: str) -> MazeDataset:
    """Helper function to create a MazeDataset from a directory of PNG images."""
    png_files = glob.glob(os.path.join(directory, "*.png"))
    if not png_files:
        raise FileNotFoundError(f"No PNG maze images found in directory: {directory}")
    return MazeDataset(image_paths=png_files)
