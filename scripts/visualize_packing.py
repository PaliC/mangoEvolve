#!/usr/bin/env python3
"""
Visualize circle packing results from trial JSON files.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def _call_packing_fn(
    namespace: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, float] | None:
    run_fn = namespace.get("run_packing")
    if callable(run_fn):
        return cast(Callable[[], tuple[np.ndarray, np.ndarray, float]], run_fn)()
    construct_fn = namespace.get("construct_packing")
    if callable(construct_fn):
        return cast(Callable[[], tuple[np.ndarray, np.ndarray, float]], construct_fn)()
    return None


def visualize_trial(trial_path: str | Path, output_path: str | Path | None = None) -> Path:
    """
    Visualize a circle packing from a trial JSON file.

    Args:
        trial_path: Path to the trial JSON file (e.g., trial_0_0.json)
        output_path: Optional output path for PNG. If None, uses trial_path with .png extension

    Returns:
        Path to the generated PNG file
    """
    trial_path = Path(trial_path)

    if output_path is None:
        output_path = trial_path.with_suffix('.png')
    else:
        output_path = Path(output_path)

    # Load trial data
    with open(trial_path) as f:
        trial_data = json.load(f)

    trial_id = trial_data.get('trial_id', trial_path.stem)
    code = trial_data.get('code', '')
    metrics = trial_data.get('metrics', {})

    # Execute the code to get the packing
    centers = None
    radii = None
    sum_radii = None
    error_msg = None

    if code and metrics.get('valid', False):
        try:
            # Create execution namespace
            namespace = {'np': np, 'numpy': np}
            exec(code, namespace)

            # Call run_packing() to get results
            result = _call_packing_fn(namespace)
            if result is None:
                error_msg = "No run_packing() or construct_packing() function found"
            else:
                centers, radii, sum_radii = result
        except Exception as e:
            error_msg = str(e)
    elif not metrics.get('valid', False):
        error_msg = metrics.get('error', 'Invalid trial')
    else:
        error_msg = "No code in trial"

    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Draw the unit square boundary
    square = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='#f5f5f5')
    ax.add_patch(square)

    if centers is not None and radii is not None:
        # Draw circles with a nice color gradient
        n_circles = len(centers)
        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0.2, 0.8, n_circles))

        for i, (center, radius) in enumerate(zip(centers, radii)):
            circle = patches.Circle(
                center, radius,
                linewidth=1.5,
                edgecolor='#2d3436',
                facecolor=colors[i],
                alpha=0.7
            )
            ax.add_patch(circle)

        # Add title with metrics
        actual_sum = float(np.sum(radii))
        score = metrics.get('combined_score', metrics.get('target_ratio', 0))
        title = f"{trial_id}\nCircles: {n_circles} | Sum of radii: {actual_sum:.4f} | Score: {score:.4f}"
    else:
        # Show error state
        title = f"{trial_id}\nError: {error_msg}"
        ax.text(0.5, 0.5, f"Error:\n{error_msg}",
                ha='center', va='center', fontsize=12, color='red',
                transform=ax.transAxes, wrap=True)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Saved: {output_path}")
    return output_path


def visualize_directory(
    directory: str | Path,
    output_dir: str | Path | None = None,
    pattern: str = "trial_*.json"
) -> list[Path]:
    """
    Visualize all trial JSON files in a directory.

    Args:
        directory: Directory containing trial JSON files
        output_dir: Optional output directory for PNGs. If None, saves alongside JSON files
        pattern: Glob pattern to match trial files (default: "trial_*.json")

    Returns:
        List of paths to generated PNG files
    """
    directory = Path(directory)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find all trial files
    trial_files = sorted(directory.glob(pattern))

    if not trial_files:
        print(f"No files matching '{pattern}' found in {directory}")
        return []

    print(f"Found {len(trial_files)} trial files in {directory}")

    output_paths = []
    for trial_path in trial_files:
        if output_dir is not None:
            output_path = output_dir / trial_path.with_suffix('.png').name
        else:
            output_path = None

        try:
            png_path = visualize_trial(trial_path, output_path)
            output_paths.append(png_path)
        except Exception as e:
            print(f"Error processing {trial_path}: {e}")

    print(f"\nGenerated {len(output_paths)} visualizations")
    return output_paths


def create_summary_grid(
    directory: str | Path,
    output_path: str | Path | None = None,
    pattern: str = "trial_*.json",
    cols: int = 4
) -> Path | None:
    """
    Create a summary grid of all trial visualizations.

    Args:
        directory: Directory containing trial JSON files
        output_path: Output path for summary PNG. If None, saves as 'summary.png' in directory
        pattern: Glob pattern to match trial files
        cols: Number of columns in the grid

    Returns:
        Path to the summary PNG, or None if no trials found
    """
    directory = Path(directory)
    trial_files = sorted(directory.glob(pattern))

    if not trial_files:
        print(f"No files matching '{pattern}' found in {directory}")
        return None

    n_trials = len(trial_files)
    rows = (n_trials + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, trial_path in enumerate(trial_files):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        # Load and execute trial
        with open(trial_path) as f:
            trial_data = json.load(f)

        trial_id = trial_data.get('trial_id', trial_path.stem)
        code = trial_data.get('code', '')
        metrics = trial_data.get('metrics', {})

        # Draw unit square
        square = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='black', facecolor='#f5f5f5')
        ax.add_patch(square)

        centers = None
        radii = None

        if code and metrics.get('valid', False):
            try:
                namespace = {'np': np, 'numpy': np}
                exec(code, namespace)
                result = _call_packing_fn(namespace)
                if result is not None:
                    centers, radii, _ = result
            except Exception:
                pass

        if centers is not None and radii is not None:
            n_circles = len(centers)
            cmap = plt.get_cmap("viridis")
            colors = cmap(np.linspace(0.2, 0.8, n_circles))
            for i, (center, radius) in enumerate(zip(centers, radii)):
                circle = patches.Circle(center, radius, linewidth=0.5,
                                        edgecolor='#2d3436', facecolor=colors[i], alpha=0.7)
                ax.add_patch(circle)

        score = metrics.get('combined_score', 0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.set_title(f"{trial_id}\nScore: {score:.4f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for idx in range(n_trials, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].set_visible(False)

    plt.suptitle(f"Circle Packing Trials - {directory.name}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path is None:
        output_path = directory / "summary.png"
    else:
        output_path = Path(output_path)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Saved summary grid: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Visualize single trial:  python visualize_packing.py <trial.json>")
        print("  Visualize directory:     python visualize_packing.py <directory>")
        print("  Create summary grid:     python visualize_packing.py <directory> --summary")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_file():
        visualize_trial(path)
    elif path.is_dir():
        if "--summary" in sys.argv:
            create_summary_grid(path)
        else:
            visualize_directory(path)
    else:
        print(f"Error: {path} does not exist")
        sys.exit(1)
