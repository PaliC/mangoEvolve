"""
mango_evolve: LLM-driven evolutionary code generation.

This package implements an evolutionary system for generating and optimizing
algorithms using LLMs.
"""

from .config import (
    BudgetConfig,
    Config,
    EvaluationConfig,
    EvolutionConfig,
    ExperimentConfig,
    LLMConfig,
    config_from_dict,
    load_config,
    load_evaluator,
)
from .cost_tracker import (
    CostSummary,
    CostTracker,
    ExperimentSummary,
    ExperimentTracker,
    TimingRecord,
    TokenUsage,
)
from .evolution_api import EvolutionAPI, GenerationSummary, TrialResult, TrialSelection
from .exceptions import (
    BudgetExceededError,
    CodeExtractionError,
    ConfigValidationError,
    ContextOverflowError,
    EvaluationError,
    MangoEvolveError,
)
from .llm import (
    LLMClient,
    LLMResponse,
    MockLLMClient,
    get_root_system_prompt,
)
from .logger import ExperimentLogger
from .repl import REPLEnvironment, REPLResult
from .root_llm import OrchestratorResult, RootLLMOrchestrator

__all__ = [
    # Exceptions
    "MangoEvolveError",
    "BudgetExceededError",
    "ConfigValidationError",
    "CodeExtractionError",
    "ContextOverflowError",
    "EvaluationError",
    # Config
    "Config",
    "ExperimentConfig",
    "LLMConfig",
    "EvolutionConfig",
    "BudgetConfig",
    "EvaluationConfig",
    "load_config",
    "config_from_dict",
    "load_evaluator",
    # Experiment tracking (cost + timing)
    "ExperimentTracker",
    "ExperimentSummary",
    "TimingRecord",
    "TokenUsage",
    # Backward-compat aliases
    "CostTracker",
    "CostSummary",
    # Logging
    "ExperimentLogger",
    # REPL
    "REPLEnvironment",
    "REPLResult",
    # Evolution API
    "EvolutionAPI",
    "TrialResult",
    "TrialSelection",
    "GenerationSummary",
    # LLM
    "LLMClient",
    "LLMResponse",
    "MockLLMClient",
    "get_root_system_prompt",
    # Root LLM Orchestrator
    "RootLLMOrchestrator",
    "OrchestratorResult",
]

__version__ = "0.1.0"
