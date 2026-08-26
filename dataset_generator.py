"""
Dataset Generator & Maze Split Manager.

Handles generating synthetic maze images with randomized entrance/exit positions,
purging stale generated image assets, and creating reproducible train/val/test/unseen splits.
"""

from typing import Tuple, List, Optional
import os
import glob
import random
import shutil
import cv2
import numpy as np


def clean_directory(directory_path: str) -> None:
    """Remove all existing PNG files in the specified directory to avoid mixing dataset runs."""
    if os.path.exists(directory_path):
        for png_file in glob.glob(os.path.join(directory_path, "*.png")):
            try:
                os.remove(png_file)
            except OSError:
                pass
    else:
        os.makedirs(directory_path, exist_ok=True)


def generate_maze_grid(height: int, width: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a binary maze grid using Depth-First Search with randomized entrance and exit."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    h = height if height % 2 == 1 else height + 1
    w = width if width % 2 == 1 else width + 1

    grid = np.zeros((h, w), dtype=np.uint8)
    stack: List[Tuple[int, int]] = []
    start_r, start_c = 1, 1
    grid[start_r, start_c] = 1
    stack.append((start_r, start_c))

    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 1 <= nr < h - 1 and 1 <= nc < w - 1 and grid[nr, nc] == 0:
                neighbors.append((nr, nc, dr, dc))

        if neighbors:
            nr, nc, dr, dc = random.choice(neighbors)
            grid[r + dr // 2, c + dc // 2] = 1
            grid[nr, nc] = 1
            stack.append((nr, nc))
        else:
            stack.pop()

    # Collect candidate boundary opening locations along all 4 outer borders
    candidates: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    # 1. Top border: (0, c) adjacent to (1, c)
    for c in range(1, w - 1):
        if grid[1, c] == 1:
            candidates.append(((0, c), (1, c)))

    # 2. Right border: (r, w-1) adjacent to (r, w-2)
    for r in range(1, h - 1):
        if grid[r, w - 2] == 1:
            candidates.append(((r, w - 1), (r, w - 2)))

    # 3. Bottom border: (h-1, c) adjacent to (h-2, c)
    for c in range(1, w - 1):
        if grid[h - 2, c] == 1:
            candidates.append(((h - 1, c), (h - 2, c)))

    # 4. Left border: (r, 0) adjacent to (r, 1)
    for r in range(1, h - 1):
        if grid[r, 1] == 1:
            candidates.append(((r, 0), (r, 1)))

    if len(candidates) >= 2:
        # Pick 2 distant boundary openings randomly
        chosen_indices = random.sample(range(len(candidates)), 2)
        # Ensure they are somewhat separated
        if len(candidates) >= 4:
            attempts = 0
            while attempts < 10:
                idx1, idx2 = random.sample(range(len(candidates)), 2)
                b1, _ = candidates[idx1]
                b2, _ = candidates[idx2]
                dist = abs(b1[0] - b2[0]) + abs(b1[1] - b2[1])
                if dist >= max(h, w) // 2:
                    chosen_indices = [idx1, idx2]
                    break
                attempts += 1

        open1, _ = candidates[chosen_indices[0]]
        open2, _ = candidates[chosen_indices[1]]
        grid[open1[0], open1[1]] = 1
        grid[open2[0], open2[1]] = 1
    else:
        # Fallback to top and bottom openings
        grid[0, 1] = 1
        grid[h - 1, w - 2] = 1

    return grid


def save_maze_image(grid: np.ndarray, output_path: str) -> None:
    """Save binary maze grid as PNG image (walls = dark 0, corridors = light 255)."""
    img = (grid * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)


def generate_dataset_split(
    output_dir: str,
    num_samples: int,
    height: int = 21,
    width: int = 21,
    start_seed: int = 42
) -> List[str]:
    """Generate a synthetic split of maze PNG images after clearing existing images."""
    clean_directory(output_dir)
    file_paths = []
    for i in range(num_samples):
        seed = start_seed + i
        grid = generate_maze_grid(height, width, seed=seed)
        filename = f"maze_{height}x{width}_{i:04d}.png"
        filepath = os.path.join(output_dir, filename)
        save_maze_image(grid, filepath)
        file_paths.append(filepath)
    return file_paths


def prepare_kaggle_maze_dataset(
    kaggle_dir: str,
    base_data_dir: str,
    num_train: int = 50,
    num_val: int = 15,
    num_test: int = 15,
    num_unseen: int = 15
) -> None:
    """Prepare dataset splits from Kaggle maze directory `maze/` or synthetic 21x21 mazes."""
    # Clean output directories to avoid mixing old & new dataset images
    for split_sub in ["train", "val", "test", "unseen", "large", "very_large"]:
        clean_directory(os.path.join(base_data_dir, split_sub))

    png_files = sorted(glob.glob(os.path.join(kaggle_dir, "*.png")))
    if not png_files:
        print(f"No PNG files found in '{kaggle_dir}'. Generating clean synthetic 21x21 dataset...")
        generate_dataset_split(os.path.join(base_data_dir, "train"), num_samples=30, height=21, width=21, start_seed=100)
        generate_dataset_split(os.path.join(base_data_dir, "val"), num_samples=10, height=21, width=21, start_seed=200)
        generate_dataset_split(os.path.join(base_data_dir, "test"), num_samples=10, height=21, width=21, start_seed=300)
        generate_dataset_split(os.path.join(base_data_dir, "unseen"), num_samples=10, height=21, width=21, start_seed=400)
    else:
        print(f"Found {len(png_files)} Kaggle maze PNG files in '{kaggle_dir}'. Splitting into train/val/test/unseen...")
        splits = {
            "train": png_files[:num_train],
            "val": png_files[num_train : num_train + num_val],
            "test": png_files[num_train + num_val : num_train + num_val + num_test],
            "unseen": png_files[num_train + num_val + num_test : num_train + num_val + num_test + num_unseen]
        }

        for split_name, files in splits.items():
            split_dir = os.path.join(base_data_dir, split_name)
            for fpath in files:
                dst = os.path.join(split_dir, os.path.basename(fpath))
                shutil.copy2(fpath, dst)

    # Large & Very Large generalization mazes (31x31 and 41x41)
    generate_dataset_split(os.path.join(base_data_dir, "large"), num_samples=5, height=31, width=31, start_seed=500)
    generate_dataset_split(os.path.join(base_data_dir, "very_large"), num_samples=5, height=41, width=41, start_seed=600)
    print("Dataset preparation complete!")


if __name__ == "__main__":
    project_root = os.path.dirname(__file__)
    kaggle_dir = os.path.join(project_root, "maze")
    data_dir = os.path.join(project_root, "data")
    prepare_kaggle_maze_dataset(kaggle_dir, data_dir)
