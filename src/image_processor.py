"""
Image Processing Module for Automatic Maze Analysis.

This module loads maze PNG images, converts them to grayscale, applies binary
thresholding, and automatically detects entrance/exit openings along the boundary.
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np


def load_and_preprocess_image(
    image_path: str,
    threshold_value: Optional[int] = None
) -> np.ndarray:
    """Load a maze PNG image, convert to grayscale, and binarize.

    Parameters
    ----------
    image_path : str
        Path to the maze PNG image.
    threshold_value : Optional[int], optional
        Manual threshold value. If None, Otsu's thresholding is applied.

    Returns
    -------
    np.ndarray
        Binary grid where 1 represents a walkable corridor (light) and 0 represents a wall (dark).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to load image from path: {image_path}")

    # Apply binarization (walls are dark < threshold, corridors are light >= threshold)
    if threshold_value is None:
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

    # Convert to 1 (walkable corridor) and 0 (wall)
    binary_grid = (binary > 127).astype(np.uint8)
    return binary_grid


def detect_boundary_entrances(
    binary_grid: np.ndarray
) -> Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
    """Automatically detect entrance and exit openings on the maze boundary.

    Scanning order along boundary:
    Top row (left to right) -> Right col (top to bottom) -> Bottom row (right to left) -> Left col (bottom to top).

    Parameters
    ----------
    binary_grid : np.ndarray
        2D binary numpy array (H, W) where 1 is walkable corridor and 0 is wall.

    Returns
    -------
    start_coords : Tuple[int, int]
        (row, col) coordinates of the Start opening (first detected boundary opening).
    goal_coords : Tuple[int, int]
        (row, col) coordinates of the Goal opening (second detected boundary opening).
    all_openings : List[Tuple[int, int]]
        List of all detected boundary opening coordinates.

    Raises
    ------
    ValueError
        If the number of boundary openings is not exactly two.
    """
    h, w = binary_grid.shape
    boundary_cells: List[Tuple[int, int]] = []

    # 1. Top border: row = 0, col from 0 to w-1
    for c in range(w):
        if binary_grid[0, c] == 1:
            boundary_cells.append((0, c))

    # 2. Right border: col = w-1, row from 1 to h-1
    for r in range(1, h):
        if binary_grid[r, w - 1] == 1:
            boundary_cells.append((r, w - 1))

    # 3. Bottom border: row = h-1, col from w-2 down to 0
    for c in range(w - 2, -1, -1):
        if binary_grid[h - 1, c] == 1:
            boundary_cells.append((h - 1, c))

    # 4. Left border: col = 0, row from h-2 down to 1
    for r in range(h - 2, 0, -1):
        if binary_grid[r, 0] == 1:
            boundary_cells.append((r, 0))

    if not boundary_cells:
        raise ValueError("No boundary openings found in the maze image.")

    # Group contiguous boundary cells into distinct entrance openings
    openings: List[Tuple[int, int]] = []
    current_cluster: List[Tuple[int, int]] = [boundary_cells[0]]

    for i in range(1, len(boundary_cells)):
        prev_r, prev_c = boundary_cells[i - 1]
        curr_r, curr_c = boundary_cells[i]

        # Check adjacency in border traversal loop
        if abs(prev_r - curr_r) + abs(prev_c - curr_c) <= 2:
            current_cluster.append((curr_r, curr_c))
        else:
            cluster_r = int(round(np.mean([r for r, c in current_cluster])))
            cluster_c = int(round(np.mean([c for r, c in current_cluster])))
            openings.append((cluster_r, cluster_c))
            current_cluster = [(curr_r, curr_c)]

    if current_cluster:
        first_cell = boundary_cells[0]
        last_cell = current_cluster[-1]
        if openings and (abs(first_cell[0] - last_cell[0]) + abs(first_cell[1] - last_cell[1]) <= 2):
            pass
        else:
            cluster_r = int(round(np.mean([r for r, c in current_cluster])))
            cluster_c = int(round(np.mean([c for r, c in current_cluster])))
            openings.append((cluster_r, cluster_c))

    if len(openings) != 2:
        raise ValueError(
            f"Invalid maze format: Expected exactly 2 boundary openings (Start & Goal), "
            f"but found {len(openings)} openings at {openings}."
        )

    start_coords = openings[0]
    goal_coords = openings[1]

    return start_coords, goal_coords, openings
