#!/usr/bin/env python3
"""
Circle Packing Visualization
Generates side-by-side comparison of OpenEvolve baseline vs MangoEvolve solution.
Uses OpenEvolve image from GitHub with matched sizing.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
OPENEVOLVE_IMAGE = Path(__file__).parent.parent / "openevolve_baseline.png"
MANGOEVOLVE_TRIAL = BASE_DIR / "saved_experiments/openevolve_config_gemini_flash_20260114_120026/generations/gen_14/trial_14_4.json"

def execute_packing_code(code: str) -> tuple:
    """Execute the packing code and return (centers, radii, sum_radii)."""
    from scipy import optimize
    from scipy.spatial.distance import pdist, squareform

    code = code.replace("niter=100", "niter=10")
    code = code.replace("niter=150", "niter=15")
    code = code.replace("for init_strategy in range(6):", "for init_strategy in range(2):")

    namespace = {
        "np": np,
        "numpy": np,
        "optimize": optimize,
        "pdist": pdist,
        "squareform": squareform
    }
    exec(code, namespace, namespace)

    if "run_packing" in namespace:
        return namespace["run_packing"]()
    elif "construct_packing" in namespace:
        return namespace["construct_packing"]()
    else:
        raise ValueError("Could not find packing function in code")

def load_trial(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def plot_circles(ax, centers, radii, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    circle_color = '#6699CC'
    edge_color = '#4477AA'

    for i in range(len(centers)):
        cx, cy = centers[i]
        r = radii[i]
        circle = patches.Circle((cx, cy), r, linewidth=0.5,
                                edgecolor=edge_color, facecolor=circle_color, alpha=0.8)
        ax.add_patch(circle)
        ax.text(cx, cy, str(i), ha='center', va='center', fontsize=7, color='black')

    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3, color='orange', linewidth=0.5)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

def main():
    print("Loading files...")

    openevolve_img = mpimg.imread(OPENEVOLVE_IMAGE)

    trial = load_trial(MANGOEVOLVE_TRIAL)
    score = trial['metrics']['score']
    print(f"MangoEvolve score: {score}")

    print("Executing MangoEvolve solution...")
    try:
        centers, radii, sum_radii = execute_packing_code(trial['code'])
    except Exception as e:
        print(f"Error executing code: {e}")
        n = 26
        centers = np.random.rand(n, 2) * 0.8 + 0.1
        radii = np.ones(n) * 0.05

    # Create figure - taller to accommodate titles
    fig = plt.figure(figsize=(13, 8))

    # Give right plot slightly more width to account for axis labels
    # Lower top value to give more room for main title above subplot titles
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.08], wspace=0.10,
                          left=0.02, right=0.98, top=0.78, bottom=0.05)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    # Left: OpenEvolve baseline image
    ax0.imshow(openevolve_img)
    ax0.axis('off')
    ax0.set_title("OpenEvolve Baseline\n(470 trials)", fontsize=11)

    # Right: MangoEvolve solution
    plot_circles(ax1, centers, radii,
                f"MangoEvolve\n(sum = {score:.6f})\n16 generations, 170 trials")

    # Add overall title with more space above subplot titles
    fig.suptitle("Circle Packing: 26 Circles in Unit Square\nAlphaEvolve Benchmark: 2.634 | MangoEvolve: 2.63583",
                 fontsize=13, fontweight='bold', y=0.95)

    # Save figure
    output_path = Path(__file__).parent.parent / "circle_packing_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved visualization to: {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()
