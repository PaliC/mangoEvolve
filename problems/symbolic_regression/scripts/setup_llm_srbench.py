#!/usr/bin/env python3
"""
Setup script to download LLM-SRBench dataset from Hugging Face.

LLM-SRBench is a benchmark for scientific equation discovery with 239 problems
across four scientific domains: biology, chemistry, materials science, and physics.

Paper: https://arxiv.org/abs/2504.10415
Dataset: https://huggingface.co/datasets/nnheui/llm-srbench

Usage:
    python -m problems.symbolic_regression.scripts.setup_llm_srbench
    # or
    python problems/symbolic_regression/scripts/setup_llm_srbench.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Available dataset splits in LLM-SRBench
# Maps our short names to HuggingFace split names and HDF5 group paths
DATASETS = {
    "bio": {
        "hf_split": "lsr_synth_bio_pop_growth",
        "hdf5_group": "lsr_synth/bio_pop_growth",
        "prefix": "BPG",  # Problem name prefix in HDF5
    },
    "chem": {
        "hf_split": "lsr_synth_chem_react",
        "hdf5_group": "lsr_synth/chem_react",
        "prefix": "CR",
    },
    "matsci": {
        "hf_split": "lsr_synth_matsci",
        "hdf5_group": "lsr_synth/matsci",
        "prefix": "MS",
    },
    "phys": {
        "hf_split": "lsr_synth_phys_osc",
        "hdf5_group": "lsr_synth/phys_osc",
        "prefix": "PO",
    },
    "transform": {
        "hf_split": "lsr_transform",
        "hdf5_group": "lsr_transform",
        "prefix": None,  # Transform uses actual problem names
    },
}

REPO_ID = "nnheui/llm-srbench"
HDF5_FILENAME = "lsr_bench_data.hdf5"


def download_hdf5_data(hf_token: str | None = None) -> Path:
    """Download the HDF5 data file using snapshot_download."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: 'huggingface_hub' package not installed.")
        print("Install with: uv add huggingface_hub")
        sys.exit(1)

    print("Downloading HDF5 data file from Hugging Face...")
    cache_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        token=hf_token,
        allow_patterns=[HDF5_FILENAME],
    )
    hdf5_path = Path(cache_dir) / HDF5_FILENAME

    if not hdf5_path.exists():
        print(f"Error: HDF5 file not found at {hdf5_path}")
        print("The dataset may have a different structure than expected.")
        sys.exit(1)

    print(f"  HDF5 file: {hdf5_path}")
    return hdf5_path


def download_dataset(output_dir: Path, hf_token: str | None = None) -> None:
    """Download LLM-SRBench dataset from Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed.")
        print("Install with: uv add datasets")
        sys.exit(1)

    try:
        import h5py
    except ImportError:
        print("Error: 'h5py' package not installed.")
        print("Install with: uv add h5py")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # First download the HDF5 file containing numerical data
    hdf5_path = download_hdf5_data(hf_token)

    # Open HDF5 file
    print(f"\nOpening HDF5 file: {hdf5_path}")
    h5file = h5py.File(hdf5_path, "r")

    # Debug: print top-level groups
    print("HDF5 top-level groups:", list(h5file.keys()))

    for short_name, config in DATASETS.items():
        hf_split = config["hf_split"]
        hdf5_group = config["hdf5_group"]
        prefix = config["prefix"]

        print(f"\nProcessing {short_name} ({hf_split})...")

        try:
            dataset = load_dataset(REPO_ID, split=hf_split, token=hf_token)
        except Exception as e:
            error_str = str(e)
            if "gated dataset" in error_str.lower() or "authenticated" in error_str.lower():
                print(f"  Authentication required for {hf_split}")
                print("  See instructions above for how to authenticate.")
                continue
            print(f"  Error downloading {hf_split}: {e}")
            continue

        # Create directory for this domain
        domain_dir = output_dir / short_name
        domain_dir.mkdir(exist_ok=True)

        # Save metadata about the domain
        metadata = {
            "hf_split": hf_split,
            "hdf5_group": hdf5_group,
            "num_problems": len(dataset),
            "problems": [],
        }

        # Check if HDF5 group exists
        if hdf5_group not in h5file and "/" + hdf5_group not in h5file:
            # Try without slashes
            parts = hdf5_group.split("/")
            if len(parts) == 2 and parts[0] in h5file:
                hdf5_base = h5file[parts[0]]
                if parts[1] in hdf5_base:
                    hdf5_domain = hdf5_base[parts[1]]
                else:
                    print(f"  Warning: HDF5 group '{hdf5_group}' not found")
                    print(f"    Available in {parts[0]}: {list(hdf5_base.keys())}")
                    continue
            else:
                print(f"  Warning: HDF5 group '{hdf5_group}' not found")
                print(f"    Available groups: {list(h5file.keys())}")
                continue
        else:
            hdf5_domain = h5file[hdf5_group] if hdf5_group in h5file else h5file["/" + hdf5_group]

        print(f"  Found {len(dataset)} problems in dataset")
        print(f"  HDF5 problems available: {list(hdf5_domain.keys())[:5]}...")

        for idx, problem in enumerate(dataset):
            problem_id = f"{short_name}_{idx:03d}"
            problem_dir = domain_dir / problem_id
            problem_dir.mkdir(exist_ok=True)

            # Get the problem name from the dataset or construct it
            if "name" in problem:
                hdf5_problem_name = problem["name"]
            elif prefix:
                hdf5_problem_name = f"{prefix}{idx}"
            else:
                print(f"  Warning: Cannot determine HDF5 name for {problem_id}")
                continue

            # Extract metadata
            problem_data = {
                "problem_id": problem_id,
                "domain": short_name,
                "hf_index": idx,
                "hdf5_name": hdf5_problem_name,
            }

            # Extract equation metadata from HuggingFace dataset
            if "expression" in problem:
                problem_data["expression"] = problem["expression"]
            if "symbols" in problem:
                problem_data["symbols"] = problem["symbols"]
            if "symbol_descs" in problem:
                problem_data["symbol_descs"] = problem["symbol_descs"]
            if "symbol_properties" in problem:
                problem_data["symbol_properties"] = problem["symbol_properties"]

            # Extract numerical data from HDF5
            if hdf5_problem_name in hdf5_domain:
                h5_problem = hdf5_domain[hdf5_problem_name]

                # Save train data
                if "train" in h5_problem:
                    train_data = np.array(h5_problem["train"], dtype=np.float64)
                    # Data format: each row is [x1, x2, ..., xn, y]
                    # Last column is y, rest are X
                    X_train = train_data[:, :-1]
                    y_train = train_data[:, -1]
                    np.savez(
                        problem_dir / "train.npz",
                        X=X_train,
                        y=y_train,
                    )
                    problem_data["n_train"] = len(y_train)
                    problem_data["n_features"] = X_train.shape[1]

                # Save test data (in-distribution)
                if "id_test" in h5_problem:
                    test_data = np.array(h5_problem["id_test"], dtype=np.float64)
                    X_test = test_data[:, :-1]
                    y_test = test_data[:, -1]
                    np.savez(
                        problem_dir / "test.npz",
                        X=X_test,
                        y=y_test,
                    )
                    problem_data["n_test"] = len(y_test)
                elif "test" in h5_problem:
                    test_data = np.array(h5_problem["test"], dtype=np.float64)
                    X_test = test_data[:, :-1]
                    y_test = test_data[:, -1]
                    np.savez(
                        problem_dir / "test.npz",
                        X=X_test,
                        y=y_test,
                    )
                    problem_data["n_test"] = len(y_test)

                # Save OOD test data if available
                if "ood_test" in h5_problem:
                    ood_data = np.array(h5_problem["ood_test"], dtype=np.float64)
                    X_ood = ood_data[:, :-1]
                    y_ood = ood_data[:, -1]
                    np.savez(
                        problem_dir / "ood_test.npz",
                        X=X_ood,
                        y=y_ood,
                    )
                    problem_data["n_ood_test"] = len(y_ood)

                print(f"  Saved {problem_id} ({hdf5_problem_name})")
            else:
                print(f"  Warning: {hdf5_problem_name} not found in HDF5")
                # Still save metadata even without numerical data
                print(f"    Available: {list(hdf5_domain.keys())[:10]}...")

            # Save problem metadata
            with open(problem_dir / "metadata.json", "w") as f:
                json.dump(problem_data, f, indent=2)

            metadata["problems"].append(problem_id)

        # Save domain metadata
        with open(domain_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Total: {len(metadata['problems'])} problems")

    h5file.close()

    # Create top-level index
    index = {
        "datasets": list(DATASETS.keys()),
        "total_problems": sum(
            len(list((output_dir / d).glob("*/train.npz")))
            for d in DATASETS.keys()
            if (output_dir / d).exists()
        ),
    }
    with open(output_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n✓ Downloaded {index['total_problems']} problems with data to {output_dir}")


def main() -> None:
    """Main entry point."""
    import os

    from dotenv import load_dotenv

    # Load .env file from project root
    project_root = Path(__file__).parent.parent.parent.parent
    load_dotenv(project_root / ".env")

    # Determine output directory relative to this script
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "data" / "llm_srbench"

    # Get HF token from environment (loaded from .env or system env)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print("=" * 60)
    print("LLM-SRBench Dataset Setup")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()

    if not hf_token:
        print("NOTE: LLM-SRBench is a gated dataset requiring authentication.")
        print()
        print("To download, you need to:")
        print("  1. Create a Hugging Face account: https://huggingface.co/join")
        print("  2. Accept the dataset terms: https://huggingface.co/datasets/nnheui/llm-srbench")
        print("  3. Create an access token: https://huggingface.co/settings/tokens")
        print("  4. Add the token to your .env file: HF_TOKEN=hf_your_token_here")
        print()
        print("Then re-run this script.")
        print()

    print("Domains to download:")
    for short, config in DATASETS.items():
        print(f"  - {short}: {config['hf_split']}")
    print()

    download_dataset(output_dir, hf_token)

    print()
    print("Setup complete! You can now use LLM-SRBench problems with:")
    print()
    print("  evaluation:")
    print('    evaluator_fn: "problems.symbolic_regression.evaluator:SymbolicRegressionEvaluator"')
    print("    evaluator_kwargs:")
    print('      benchmark: "llm_srbench"')
    print('      domain: "phys"  # bio, chem, matsci, phys, or transform')
    print("      problem_index: 0  # 0 to N-1 problems in domain")


if __name__ == "__main__":
    main()
