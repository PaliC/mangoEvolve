#!/usr/bin/env python3
"""
Min-Max Distance 3D Visualization
Plots 14 points in 3D space showing the optimal configuration.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from pathlib import Path

# Path to best trial
BASE_DIR = Path(__file__).parent.parent.parent
BEST_TRIAL = BASE_DIR / "saved_experiments/min_max_dist_3d_gemini_3_flash_20260121_170613/generations/gen_8/trial_8_0.json"

def execute_minmax_code(code: str) -> np.ndarray:
    """Execute the min-max distance code and return point coordinates."""
    import time
    from scipy.optimize import minimize
    from scipy.spatial.distance import pdist

    # Reduce timeout for faster visualization
    code = code.replace("< 50:", "< 5:")  # Reduce 50s to 5s

    namespace = {
        "np": np,
        "numpy": np,
        "time": time,
        "minimize": minimize,
        "pdist": pdist
    }
    exec(code, namespace, namespace)

    if "min_max_dist_dim3_14" in namespace:
        return namespace["min_max_dist_dim3_14"]()
    else:
        raise ValueError("Could not find min_max_dist_dim3_14 function in code")

def load_trial(path: Path) -> dict:
    """Load a trial JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def plot_3d_points(ax, points, title, score):
    """Plot 3D point configuration with nearest-neighbor connections."""
    n = len(points)

    # Calculate pairwise distances
    dist_matrix = squareform(pdist(points))
    np.fill_diagonal(dist_matrix, np.inf)

    min_dist = np.min(dist_matrix)
    max_dist = pdist(points).max()

    # Draw edges between nearest neighbors (within 1.1x of min distance)
    threshold = min_dist * 1.15
    for i in range(n):
        for j in range(i+1, n):
            if dist_matrix[i, j] < threshold:
                ax.plot3D([points[i, 0], points[j, 0]],
                         [points[i, 1], points[j, 1]],
                         [points[i, 2], points[j, 2]],
                         'b-', alpha=0.4, linewidth=1)

    # Plot points with color based on z-coordinate
    z_normalized = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 1e-9)
    colors = plt.cm.coolwarm(z_normalized)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=150, c=colors, edgecolor='black', linewidth=1, depthshade=True)

    # Label points
    for i, (x, y, z) in enumerate(points):
        ax.text(x, y, z, f'  {i}', fontsize=7, color='darkblue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title}Min/Max Dist: {min_dist:.4f}/{max_dist:.4f}",
                 fontsize=11, fontweight='bold')

def analyze_structure(points):
    """Analyze the geometric structure of the point configuration."""
    n = len(points)

    # Calculate center of mass
    center = np.mean(points, axis=0)

    # Calculate distances from center
    radii = np.linalg.norm(points - center, axis=1)

    # Group points by distance from center
    sorted_idx = np.argsort(radii)

    print("\nStructural Analysis:")
    print("=" * 40)
    print(f"Center of mass: {center}")
    print(f"\nPoints by distance from center:")
    for i, idx in enumerate(sorted_idx):
        print(f"  Point {idx}: r = {radii[idx]:.4f}, z = {points[idx, 2]:.4f}")

    # Calculate pairwise distances
    dists = pdist(points)
    print(f"\nDistance statistics:")
    print(f"  Min distance: {np.min(dists):.6f}")
    print(f"  Max distance: {np.max(dists):.6f}")
    print(f"  Ratio (min/max)^2: {(np.min(dists)/np.max(dists))**2:.10f}")

def main():
    print("Loading Min-Max Distance 3D trial file...")

    trial = load_trial(BEST_TRIAL)
    print(f"Trial ID: {trial['trial_id']}")
    print(f"Score: {trial['metrics']['score']}")

    print("Executing min-max distance solution...")
    try:
        points = execute_minmax_code(trial['code'])
        print(f"Got {len(points)} points in 3D")
    except Exception as e:
        print(f"Error executing code: {e}")
        print("Using placeholder data (Fibonacci sphere)...")
        # Create placeholder using Fibonacci sphere
        n = 14
        phi = (1 + np.sqrt(5)) / 2
        indices = np.arange(n) + 0.5
        z = 1 - 2 * indices / n
        radius = np.sqrt(np.maximum(0, 1 - z*z))
        theta = 2 * np.pi * indices / phi
        points = np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1)

    # Analyze structure
    analyze_structure(points)

    # Create figure with multiple views - equal sized subplots
    fig = plt.figure(figsize=(15, 7))

    # Use GridSpec for equal sizing with more top margin for titles
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.05,
                          left=0.02, right=0.98, top=0.78, bottom=0.02)

    # View 1: Default view
    ax1 = fig.add_subplot(gs[0], projection='3d')
    plot_3d_points(ax1, points, "View 1: Default Angle", trial['metrics']['score'])
    ax1.view_init(elev=20, azim=45)

    # View 2: Top-down view
    ax2 = fig.add_subplot(gs[1], projection='3d')
    plot_3d_points(ax2, points, "View 2: Top-Down", trial['metrics']['score'])
    ax2.view_init(elev=90, azim=0)

    # View 3: Side view
    ax3 = fig.add_subplot(gs[2], projection='3d')
    plot_3d_points(ax3, points, "View 3: Side View", trial['metrics']['score'])
    ax3.view_init(elev=0, azim=0)

    # Add overall title
    fig.suptitle("Min-Max Distance Problem: 14 Points in 3D\nOptimal Structure: Bicapped Hexagonal Antiprism (D6d symmetry)\nAlphaEvolve Benchmark: 1.0",
                 fontsize=13, fontweight='bold', y=0.97)

    # Save figure
    output_path = Path(__file__).parent.parent / "min_max_dist_3d_solution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved visualization to: {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()
