"""
Symbolic Regression Evaluator for MangoEvolve.

Evolves mathematical expressions to fit data by minimizing MSE.
Inspired by OpenEvolve's symbolic regression example.

Supports two modes:
1. Synthetic data: Generate data from a target function (default)
2. LLM-SRBench: Load real scientific datasets from the benchmark

For LLM-SRBench, run the setup script first:
    python -m problems.symbolic_regression.scripts.setup_llm_srbench
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mango_evolve.problem import BaseProblemEvaluator, ProblemSpec

# LLM-SRBench domains
LLM_SRBENCH_DOMAINS = {
    "bio": "Biology (population growth)",
    "chem": "Chemistry (reactions)",
    "matsci": "Materials Science",
    "phys": "Physics (oscillation)",
    "transform": "Transformed physical models",
}


class LLMSRBenchNotSetupError(Exception):
    """Raised when LLM-SRBench data is not available."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        return f"""
LLM-SRBench dataset not found at: {self.data_dir}

To download the dataset, run:
    python -m problems.symbolic_regression.scripts.setup_llm_srbench

This will download ~239 problems across 5 scientific domains from Hugging Face.
See: https://huggingface.co/datasets/nnheui/llm-srbench
"""


def _run_evaluation(
    code: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_params: int,
    timeout_seconds: float,
    X_ood: np.ndarray | None = None,
    y_ood: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run evaluation with train/test split."""
    from scipy.optimize import minimize

    start_time = time.time()

    try:
        # Execute the code to get the model function
        namespace: dict[str, Any] = {"np": np, "numpy": np}
        exec(code, namespace)

        if "model" not in namespace:
            return {
                "valid": False,
                "score": 0,
                "error": "Code must define a 'model(x, params)' function",
                "eval_time": time.time() - start_time,
            }

        model_fn = namespace["model"]

        # Test that the function works with dummy data
        test_x = X_train[:1]
        test_params = np.zeros(n_params)
        try:
            test_output = model_fn(test_x, test_params)
            if not isinstance(test_output, (np.ndarray, float, int)):
                return {
                    "valid": False,
                    "score": 0,
                    "error": f"model() must return array or scalar, got {type(test_output)}",
                    "eval_time": time.time() - start_time,
                }
        except Exception as e:
            return {
                "valid": False,
                "score": 0,
                "error": f"model() failed on test input: {e}",
                "eval_time": time.time() - start_time,
            }

        # Define objective function for optimization (uses training data)
        def objective(params: np.ndarray) -> float:
            try:
                predictions = model_fn(X_train, params)
                predictions = np.asarray(predictions).flatten()
                if predictions.shape != y_train.shape:
                    predictions = np.broadcast_to(predictions, y_train.shape)
                mse = np.mean((predictions - y_train) ** 2)
                if np.isnan(mse) or np.isinf(mse):
                    return 1e10
                return float(mse)
            except Exception:
                return 1e10

        # Optimize parameters using BFGS
        initial_params = np.zeros(n_params)
        result = minimize(
            objective,
            initial_params,
            method="BFGS",
            options={"maxiter": 100, "disp": False},
        )

        optimized_params = result.x
        train_mse = objective(optimized_params)

        # Compute test MSE with optimized parameters
        try:
            test_predictions = model_fn(X_test, optimized_params)
            test_predictions = np.asarray(test_predictions).flatten()
            if test_predictions.shape != y_test.shape:
                test_predictions = np.broadcast_to(test_predictions, y_test.shape)
            test_mse = float(np.mean((test_predictions - y_test) ** 2))
            if np.isnan(test_mse) or np.isinf(test_mse):
                test_mse = 1e10
        except Exception:
            test_mse = 1e10

        # Compute OOD test MSE if available
        ood_mse = None
        if X_ood is not None and y_ood is not None:
            try:
                ood_predictions = model_fn(X_ood, optimized_params)
                ood_predictions = np.asarray(ood_predictions).flatten()
                if ood_predictions.shape != y_ood.shape:
                    ood_predictions = np.broadcast_to(ood_predictions, y_ood.shape)
                ood_mse = float(np.mean((ood_predictions - y_ood) ** 2))
                if np.isnan(ood_mse) or np.isinf(ood_mse):
                    ood_mse = 1e10
            except Exception:
                ood_mse = 1e10

        # Compute score: -log10(mse + epsilon) so higher is better
        # Use TRAIN MSE for evolution score (what optimizer sees)
        mse_clamped = max(train_mse, 1e-10)
        score = -np.log10(mse_clamped)

        # Also compute test score for comparison
        test_mse_clamped = max(test_mse, 1e-10)
        test_score = -np.log10(test_mse_clamped)

        results = {
            "valid": True,
            "score": float(score),
            # Raw MSE values (comparable to OpenEvolve)
            "mse_train": float(train_mse),
            "mse_test": float(test_mse),
            # Log-transformed scores
            "score_train": float(score),
            "score_test": float(test_score),
            # Additional info
            "optimized_params": optimized_params.tolist(),
            "optimization_success": result.success,
            "eval_time": time.time() - start_time,
            "error": None,
        }

        # Add OOD metrics if available
        if ood_mse is not None:
            results["mse_ood"] = float(ood_mse)
            ood_mse_clamped = max(ood_mse, 1e-10)
            results["score_ood"] = float(-np.log10(ood_mse_clamped))

        return results

    except SyntaxError as e:
        return {
            "valid": False,
            "score": 0,
            "error": f"Syntax error: {e}",
            "eval_time": time.time() - start_time,
        }
    except Exception as e:
        return {
            "valid": False,
            "score": 0,
            "error": f"Evaluation error: {e}",
            "eval_time": time.time() - start_time,
        }


class SymbolicRegressionEvaluator(BaseProblemEvaluator):
    """
    Evaluator for symbolic regression problems.

    Evolves mathematical expressions to minimize MSE on training data.
    Uses scipy's BFGS optimizer to fit parameters.

    Two modes of operation:
    1. Synthetic: Generate data from a target_function (default)
    2. LLM-SRBench: Load real scientific data from benchmark

    For LLM-SRBench mode, set benchmark="llm_srbench" and specify domain/problem_index.
    """

    def __init__(
        self,
        # Common parameters
        n_params: int = 5,
        timeout_seconds: float = 30.0,
        # Synthetic data parameters
        n_samples: int = 100,
        n_features: int = 1,
        target_function: str | None = None,
        noise_std: float = 0.1,
        x_range: tuple[float, float] = (-5.0, 5.0),
        test_split: float = 0.2,
        # LLM-SRBench parameters
        benchmark: str | None = None,
        domain: str | None = None,
        problem_index: int = 0,
    ):
        """
        Initialize the symbolic regression evaluator.

        Args:
            n_params: Number of parameters the model can use.
            timeout_seconds: Timeout for each evaluation.

            Synthetic data mode (default):
                n_samples: Number of total samples to generate.
                n_features: Number of input features.
                target_function: Target function as string (e.g., "np.sin(x[:, 0])").
                noise_std: Standard deviation of Gaussian noise.
                x_range: Range for generating input features.
                test_split: Fraction of data for testing.

            LLM-SRBench mode (set benchmark="llm_srbench"):
                domain: Domain name - "bio", "chem", "matsci", "phys", or "transform".
                problem_index: Problem index within domain (0-indexed).
        """
        self.n_params = n_params
        self.timeout_seconds = timeout_seconds
        self.benchmark = benchmark
        self.domain = domain
        self.problem_index = problem_index

        # Store for synthetic mode
        self.target_function = target_function
        self.noise_std = noise_std
        self.x_range = x_range
        self.test_split = test_split

        # OOD test data (only for LLM-SRBench)
        self.X_ood: np.ndarray | None = None
        self.y_ood: np.ndarray | None = None

        # Problem metadata (for LLM-SRBench)
        self.problem_metadata: dict[str, Any] | None = None

        if benchmark == "llm_srbench":
            # Load LLM-SRBench data
            self._load_llm_srbench_data()
        else:
            # Generate synthetic data
            self.n_samples = n_samples
            self.n_features = n_features
            self.X_train, self.y_train, self.X_test, self.y_test = self._generate_data()

    def _get_llm_srbench_data_dir(self) -> Path:
        """Get the LLM-SRBench data directory."""
        # Data is stored relative to this file
        return Path(__file__).parent / "data" / "llm_srbench"

    def _load_llm_srbench_data(self) -> None:
        """Load data from LLM-SRBench benchmark."""
        data_dir = self._get_llm_srbench_data_dir()

        # Check if data exists
        if not data_dir.exists() or not (data_dir / "index.json").exists():
            raise LLMSRBenchNotSetupError(data_dir)

        # Validate domain
        if self.domain not in LLM_SRBENCH_DOMAINS:
            valid_domains = ", ".join(LLM_SRBENCH_DOMAINS.keys())
            raise ValueError(
                f"Invalid domain '{self.domain}'. Must be one of: {valid_domains}"
            )

        domain_dir = data_dir / self.domain
        if not domain_dir.exists():
            raise LLMSRBenchNotSetupError(data_dir)

        # Load domain metadata
        with open(domain_dir / "metadata.json") as f:
            domain_metadata = json.load(f)

        # Validate problem index
        num_problems = domain_metadata["num_problems"]
        if self.problem_index < 0 or self.problem_index >= num_problems:
            raise ValueError(
                f"Invalid problem_index {self.problem_index}. "
                f"Domain '{self.domain}' has {num_problems} problems (0-{num_problems - 1})."
            )

        # Load problem data
        problem_id = f"{self.domain}_{self.problem_index:03d}"
        problem_dir = domain_dir / problem_id

        # Load metadata
        with open(problem_dir / "metadata.json") as f:
            self.problem_metadata = json.load(f)

        # Load train data
        train_data = np.load(problem_dir / "train.npz")
        self.X_train = train_data["X"]
        self.y_train = train_data["y"].flatten()

        # Load test data
        test_data = np.load(problem_dir / "test.npz")
        self.X_test = test_data["X"]
        self.y_test = test_data["y"].flatten()

        # Load OOD test data if available
        ood_path = problem_dir / "ood_test.npz"
        if ood_path.exists():
            ood_data = np.load(ood_path)
            self.X_ood = ood_data["X"]
            self.y_ood = ood_data["y"].flatten()

        # Set derived attributes
        self.n_samples = len(self.y_train) + len(self.y_test)
        self.n_features = self.X_train.shape[1] if len(self.X_train.shape) > 1 else 1

    def _generate_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic train and test data."""
        np.random.seed(42)  # Reproducible data
        X = np.random.uniform(
            self.x_range[0], self.x_range[1], (self.n_samples, self.n_features)
        )

        if self.target_function:
            # Use provided target function
            namespace = {"np": np, "numpy": np, "x": X}
            y = eval(self.target_function, namespace)
        else:
            # Default: simple polynomial y = x^2 + 0.5*x + noise
            if self.n_features == 1:
                y = X[:, 0] ** 2 + 0.5 * X[:, 0]
            else:
                y = np.sum(X**2, axis=1) + 0.5 * np.sum(X, axis=1)

        # Add noise
        y = y + np.random.normal(0, self.noise_std, y.shape)
        y = y.flatten()

        # Split into train/test
        n_test = int(self.n_samples * self.test_split)
        n_train = self.n_samples - n_test

        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        return X_train, y_train, X_test, y_test

    def get_problem_spec(self) -> ProblemSpec:
        """Return the problem specification for symbolic regression."""
        n_train = len(self.y_train)

        if self.benchmark == "llm_srbench":
            # LLM-SRBench problem description
            domain_name = LLM_SRBENCH_DOMAINS.get(self.domain, self.domain)
            problem_id = self.problem_metadata.get("problem_id", f"{self.domain}_{self.problem_index}")

            description = f"""Find a mathematical expression that fits scientific data.

Domain: {domain_name}
Problem: {problem_id}

Data:
- Training samples: {n_train}
- Test samples: {len(self.y_test)}
- Input features: {self.n_features}
- {self.n_params} parameters (params array) are available for optimization
- Parameters are automatically optimized using BFGS after your model is defined

Your model(x, params) function should return predictions for the given inputs.
Use the params array for learnable coefficients that will be optimized."""

            if self.X_ood is not None:
                description += f"\n- OOD test samples: {len(self.y_ood)} (out-of-distribution)"
        else:
            # Synthetic data description
            feature_desc = (
                f"x is a ({n_train}, {self.n_features}) array"
                if self.n_features > 1
                else f"x is a ({n_train}, 1) array (use x[:, 0] for the values)"
            )

            target_desc = (
                f"Target function: {self.target_function}"
                if self.target_function
                else "Target: unknown function (discover it!)"
            )

            description = f"""Find a mathematical expression that fits the training data.

{target_desc}

Data:
- {feature_desc}
- y is the target values to predict
- {self.n_params} parameters (params array) are available for optimization
- Parameters are automatically optimized using BFGS after your model is defined

Your model(x, params) function should return predictions for the given inputs.
Use the params array for learnable coefficients that will be optimized."""

        return ProblemSpec(
            name="Symbolic Regression",
            description=description,
            objective="minimize",
            metric_name="MSE (mean squared error)",
            entry_function="model",
            return_description="Array of predictions with same length as x",
            best_known_solution=0.0,  # Perfect MSE
            helper_functions=None,
            allowed_modules=["numpy", "np"],
            constraints=[
                f"Function must accept (x, params) where x has {self.n_features} feature(s)",
                f"params is a numpy array of length {self.n_params}",
                "Return predictions as array matching y's shape",
                "Avoid division by zero and numerical instabilities",
            ],
            example_code=f'''def model(x, params):
    """
    Symbolic regression model.

    Args:
        x: Input features, shape (N, {self.n_features})
        params: Learnable parameters, shape ({self.n_params},)

    Returns:
        Predictions, shape (N,)
    """
    # Example: linear model (evolve this to find better expressions!)
    # Access first feature with x[:, 0]
    return params[0] * x[:, 0] + params[1]''',
            secondary_metrics=["mse", "optimization_success"],
        )

    def evaluate(self, code: str) -> dict[str, Any]:
        """
        Evaluate a symbolic regression model.

        Args:
            code: Python code defining a model(x, params) function.

        Returns:
            Dictionary with evaluation results including:
            - score: Log-transformed train MSE (higher = better)
            - mse_train: Raw training MSE
            - mse_test: Raw test MSE
            - mse_ood: Raw OOD test MSE (if LLM-SRBench with OOD data)
            - score_train, score_test, score_ood: Log-transformed scores
        """
        return _run_evaluation(
            code,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.n_params,
            self.timeout_seconds,
            self.X_ood,
            self.y_ood,
        )
