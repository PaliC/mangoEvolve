#!/usr/bin/env python3
"""
Heilbronn Triangle Visualization
Plots the 11 points inside an equilateral triangle and optionally shows sub-triangles.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import itertools
from pathlib import Path

# Path to best trial
BASE_DIR = Path(__file__).parent.parent.parent
BEST_TRIAL = BASE_DIR / "experiments/circle_packing_gemini_3_flash_20260120_104304/generations/gen_12/trial_12_5.json"

def execute_heilbronn_code(code: str) -> np.ndarray:
    """Execute the Heilbronn code and return point coordinates."""
    import time
    import itertools
    from scipy.optimize import minimize

    # Reduce timeouts for faster visualization
    code = code.replace("< 12:", "< 2:")  # Reduce 12s to 2s
    code = code.replace("> 20:", "> 3:")  # Reduce 20s to 3s
    code = code.replace("< 28.0:", "< 5.0:")  # Reduce 28s to 5s

    namespace = {
        "np": np,
        "numpy": np,
        "time": time,
        "itertools": itertools,
        "minimize": minimize
    }
    exec(code, namespace, namespace)

    if "heilbronn_triangle11" in namespace:
        return namespace["heilbronn_triangle11"]()
    else:
        raise ValueError("Could not find heilbronn_triangle11 function in code")

def load_trial(path: Path) -> dict:
    """Load a trial JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def get_equilateral_triangle():
    """Returns vertices of equilateral triangle with base on x-axis."""
    # Vertices: (0, 0), (1, 0), (0.5, sqrt(3)/2)
    return np.array([
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3)/2]
    ])

def plot_heilbronn(ax, points, title, score, show_triangles=False, n_smallest=5):
    """Plot the Heilbronn configuration."""
    h = np.sqrt(3) / 2

    # Draw the equilateral triangle
    triangle = patches.Polygon(
        [[0, 0], [1, 0], [0.5, h]],
        closed=True, linewidth=2,
        edgecolor='black', facecolor='lightyellow', alpha=0.3
    )
    ax.add_patch(triangle)

    # Calculate all triangle areas for coloring
    triplets = list(itertools.combinations(range(len(points)), 3))
    areas = []
    for i, j, k in triplets:
        x0, y0 = points[i]
        x1, y1 = points[j]
        x2, y2 = points[k]
        area = 0.5 * abs(x0*(y1-y2) + x1*(y2-y0) + x2*(y0-y1))
        areas.append((area, i, j, k))

    areas.sort(key=lambda x: x[0])
    min_area = areas[0][0]

    # Optionally draw the smallest triangles
    if show_triangles:
        for area, i, j, k in areas[:n_smallest]:
            tri_points = points[[i, j, k]]
            tri_patch = patches.Polygon(
                tri_points, closed=True, linewidth=1,
                edgecolor='red', facecolor='red', alpha=0.2
            )
            ax.add_patch(tri_patch)

    # Plot points
    ax.scatter(points[:, 0], points[:, 1], s=100, c='darkblue',
               zorder=5, edgecolor='white', linewidth=1.5)

    # Label points
    for i, (x, y) in enumerate(points):
        ax.annotate(str(i), (x, y), textcoords="offset points",
                   xytext=(5, 5), fontsize=8, color='darkblue')

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, h + 0.1)
    ax.set_aspect('equal')
    ax.set_title(f"{title}Min Area: {min_area:.10f}",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

def main():
    print("Loading Heilbronn trial file...")

    trial = load_trial(BEST_TRIAL)
    print(f"Trial ID: {trial['trial_id']}")
    print(f"Score: {trial['metrics']['score']}")

    print("Executing Heilbronn solution...")
    try:
        points = execute_heilbronn_code(trial['code'])
        print(f"Got {len(points)} points")
    except Exception as e:
        print(f"Error executing code: {e}")
        print("Using placeholder data...")
        # Create placeholder points
        h = np.sqrt(3) / 2
        points = np.array([
            [0.5, h * 0.1],  # near bottom
            [0.2, h * 0.3],
            [0.8, h * 0.3],
            [0.35, h * 0.5],
            [0.65, h * 0.5],
            [0.15, h * 0.6],
            [0.85, h * 0.6],
            [0.5, h * 0.7],
            [0.3, h * 0.8],
            [0.7, h * 0.8],
            [0.5, h * 0.9]
        ])

    # Create figure with two views
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Points only
    plot_heilbronn(axes[0], points,
                  "11-Point Heilbronn Configuration",
                  trial['metrics']['score'],
                  show_triangles=False)

    # Right: Points with smallest triangles highlighted
    plot_heilbronn(axes[1], points,
                  "With 5 Smallest Triangles Highlighted",
                  trial['metrics']['score'],
                  show_triangles=True, n_smallest=5)

    # Add overall title
    fig.suptitle("Heilbronn Triangle Problem: 11 Points",
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent.parent / "heilbronn_solution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved visualization to: {output_path}")

    plt.show()

if __name__ == "__main__":
    main()
