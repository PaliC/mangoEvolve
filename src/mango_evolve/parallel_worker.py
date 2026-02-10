"""
Parallel worker for child LLM calls.

This module contains the worker function that runs in a thread
to make LLM calls and evaluate the results.

Supports multiple providers (Anthropic, OpenRouter, and Google).
"""

import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import OpenAI
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

from .config import load_evaluator_from_string
from .utils.code_extraction import extract_python_code, extract_reasoning


@dataclass
class WorkerInput:
    """Input arguments for parallel worker functions."""

    prompt: str
    parent_id: str | None
    model: str
    evaluator_fn: str | None
    evaluator_kwargs: dict[str, Any]
    max_tokens: int
    temperature: float
    trial_id: str
    generation: int
    experiment_dir: str
    system_prompt: str | None
    provider: str
    model_alias: str | None


@dataclass
class WorkerResult:
    """Result from a parallel worker."""

    prompt: str
    parent_id: str | None
    response_text: str
    code: str
    reasoning: str
    metrics: dict[str, Any]
    success: bool
    error: str | None
    input_tokens: int
    output_tokens: int
    call_id: str
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    trial_id: str = ""
    model_alias: str | None = None
    model_config: dict[str, Any] | None = None
    # Timing data
    llm_call_duration_s: float = 0.0
    eval_duration_s: float = 0.0


def query_llm(inp: WorkerInput) -> dict[str, Any]:
    """
    Submit an LLM query with no evaluation or file I/O.
    """
    model_config: dict[str, Any] = {
        "model": inp.model,
        "temperature": inp.temperature,
    }

    call_id = str(uuid.uuid4())
    response_text = ""
    input_tokens = 0
    output_tokens = 0
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0
    error: str | None = None
    provider_debug: dict[str, Any] | None = None
    llm_call_duration_s = 0.0

    llm_start = time.monotonic()
    try:
        if inp.provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            response = _make_openrouter_call_with_retry(
                client=client,
                model=inp.model,
                prompt=inp.prompt,
                max_tokens=inp.max_tokens,
                temperature=inp.temperature,
                system_prompt=inp.system_prompt,
            )

            response_text = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
        elif inp.provider == "google":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            contents = [types.Content(role="user", parts=[types.Part(text=inp.prompt)])]
            config_kwargs: dict[str, Any] = {
                "max_output_tokens": inp.max_tokens,
                "temperature": inp.temperature,
            }
            if inp.system_prompt:
                config_kwargs["system_instruction"] = inp.system_prompt

            response = _make_google_call_with_retry(
                client=client,
                model=inp.model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )

            response_text = _extract_google_text(response)
            finish_reason = None
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
                if finish_reason is not None:
                    finish_reason = getattr(finish_reason, "name", str(finish_reason))

            usage_metadata: dict[str, Any] = {}
            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count or 0
                output_tokens = response.usage_metadata.candidates_token_count or 0
                usage_metadata = {
                    "prompt_token_count": response.usage_metadata.prompt_token_count,
                    "candidates_token_count": response.usage_metadata.candidates_token_count,
                    "thoughts_token_count": getattr(
                        response.usage_metadata, "thoughts_token_count", 0
                    )
                    or 0,
                }

            provider_debug = {
                "finish_reason": finish_reason,
                "usage_metadata": usage_metadata,
                "response_text_len": len(response_text),
            }
        else:
            anthropic_client = anthropic.Anthropic()

            response = _make_anthropic_call_with_retry(
                client=anthropic_client,
                model=inp.model,
                prompt=inp.prompt,
                max_tokens=inp.max_tokens,
                temperature=inp.temperature,
                system_prompt=inp.system_prompt,
            )

            if response.content:
                response_text = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cache_creation_input_tokens = (
                getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            )
            cache_read_input_tokens = getattr(response.usage, "cache_read_input_tokens", 0) or 0
    except Exception as e:
        error = f"LLM call failed: {str(e)}"
    finally:
        llm_call_duration_s = time.monotonic() - llm_start

    success = error is None
    reasoning = response_text if success else ""
    if provider_debug:
        model_config["provider_debug"] = provider_debug

    return {
        "trial_id": inp.trial_id,
        "prompt": inp.prompt,
        "parent_id": inp.parent_id,
        "response_text": response_text,
        "code": "",
        "reasoning": reasoning,
        "metrics": {},
        "success": success,
        "error": error,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "call_id": call_id,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "model_alias": inp.model_alias,
        "model_config": model_config,
        "llm_call_duration_s": llm_call_duration_s,
        "eval_duration_s": 0.0,
    }


def _write_trial_file(
    trial_id: str,
    generation: int,
    experiment_dir: str,
    code: str | None,
    metrics: dict[str, Any],
    prompt: str,
    response: str,
    reasoning: str | None,
    parent_id: str | None,
    model_config: dict[str, Any] | None = None,
    llm_call_duration_s: float = 0.0,
    eval_duration_s: float = 0.0,
) -> None:
    """Write trial JSON file to disk for real-time progress tracking."""
    gen_dir = Path(experiment_dir) / "generations" / f"gen_{generation}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    trial_data = {
        "trial_id": trial_id,
        "generation": generation,
        "parent_id": parent_id,
        "code": code,
        "metrics": metrics,
        "prompt": prompt,
        "response": response,
        "reasoning": reasoning,
        "timestamp": datetime.now().isoformat(),
        "cost_data": None,
        "model_config": model_config,
        "llm_call_duration_s": llm_call_duration_s,
        "eval_duration_s": eval_duration_s,
    }

    trial_path = gen_dir / f"{trial_id}.json"
    with open(trial_path, "w") as f:
        json.dump(trial_data, f, indent=2)


def _make_anthropic_call_with_retry(
    client: anthropic.Anthropic,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str | None = None,
    max_retries: int = 3,
) -> anthropic.types.Message:
    """Make an Anthropic API call with retry logic and optional caching."""

    @retry(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
            )
        ),
        reraise=True,
    )
    def _call():
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Add system prompt with cache_control if provided
        if system_prompt:
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        return client.messages.create(**kwargs)

    return _call()


def _is_empty_openrouter_response(response) -> bool:
    """Check if OpenRouter response content is empty (triggers retry)."""
    if not response.choices:
        return True
    content = response.choices[0].message.content
    return not content or not content.strip()


def _make_openrouter_call_with_retry(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str | None = None,
    max_retries: int = 3,
):
    """Make an OpenRouter API call with retry logic."""

    @retry(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=(
            retry_if_exception_type((OpenAIRateLimitError, OpenAIAPIConnectionError))
            | retry_if_result(_is_empty_openrouter_response)
        ),
        reraise=True,
    )
    def _call():
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return _call()


def _is_empty_google_response(response: Any) -> bool:
    """Check if Google response content is empty (triggers retry)."""
    if not response.candidates:
        return True
    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        return True
    text = "".join(
        part.text or ""
        for part in candidate.content.parts
        if not getattr(part, "thought", False)
    )
    return not text.strip()


def _make_google_call_with_retry(
    client: Any,
    model: str,
    contents: Any,
    config: Any,
    max_retries: int = 3,
):
    """Make a Google Gemini API call with retry logic."""
    from typing import cast

    @retry(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=(
            retry_if_exception_type((ConnectionError, TimeoutError))
            | retry_if_result(_is_empty_google_response)
        ),
        reraise=True,
    )
    def _call():
        return client.models.generate_content(
            model=model,
            contents=cast(Any, contents),
            config=config,
        )

    return _call()


def _extract_google_text(response: Any) -> str:
    """Extract non-thinking content from a Gemini response."""
    content = ""
    if response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if getattr(part, "thought", False):
                    continue
                content += part.text or ""
    return content


def spawn_child(inp: WorkerInput) -> dict[str, Any]:
    """
    Worker function for parallel child LLM calls.

    This function runs in a thread and:
    1. Makes the LLM call
    2. Extracts code from the response
    3. Evaluates the code
    4. Writes trial JSON file for real-time progress tracking
    5. Returns all results including timing data
    """
    result = query_llm(inp)

    # If LLM call failed, write trial file and return as-is
    if result["error"]:
        _write_trial_file(
            trial_id=inp.trial_id,
            generation=inp.generation,
            experiment_dir=inp.experiment_dir,
            code="",
            metrics={},
            prompt=inp.prompt,
            response=result["response_text"],
            reasoning=result["reasoning"],
            parent_id=inp.parent_id,
            model_config=result.get("model_config"),
            llm_call_duration_s=result.get("llm_call_duration_s", 0.0),
        )
        return result

    response_text = result["response_text"]
    code = extract_python_code(response_text)
    reasoning = extract_reasoning(response_text)

    if not code:
        result.update(
            {
                "code": "",
                "reasoning": reasoning,
                "metrics": {},
                "success": False,
                "error": "No Python code block found in response",
            }
        )
        _write_trial_file(
            trial_id=inp.trial_id,
            generation=inp.generation,
            experiment_dir=inp.experiment_dir,
            code="",
            metrics={},
            prompt=inp.prompt,
            response=response_text,
            reasoning=reasoning,
            parent_id=inp.parent_id,
            model_config=result.get("model_config"),
            llm_call_duration_s=result.get("llm_call_duration_s", 0.0),
        )
        return result

    # Dynamically load the evaluator based on evaluator_fn from config
    try:
        evaluator = load_evaluator_from_string(inp.evaluator_fn, inp.evaluator_kwargs)
    except Exception as e:
        metrics = {
            "valid": False,
            "error": f"Failed to load evaluator '{inp.evaluator_fn}': {str(e)}",
        }
        result.update(
            {
                "code": code,
                "reasoning": reasoning,
                "metrics": metrics,
                "success": False,
                "error": metrics["error"],
            }
        )
        _write_trial_file(
            trial_id=inp.trial_id,
            generation=inp.generation,
            experiment_dir=inp.experiment_dir,
            code=code,
            metrics=metrics,
            prompt=inp.prompt,
            response=response_text,
            reasoning=reasoning,
            parent_id=inp.parent_id,
            model_config=result.get("model_config"),
            llm_call_duration_s=result.get("llm_call_duration_s", 0.0),
        )
        return result

    eval_start = time.monotonic()
    try:
        metrics = evaluator.evaluate(code)
    except Exception as e:
        metrics = {
            "valid": False,
            "error": f"Evaluation error: {str(e)}",
        }
    eval_duration_s = time.monotonic() - eval_start

    success = bool(metrics.get("valid", False))
    error_value = metrics.get("error") if not success else None
    error = str(error_value) if error_value is not None else None

    result.update(
        {
            "code": code,
            "reasoning": reasoning,
            "metrics": metrics,
            "success": success,
            "error": error,
            "eval_duration_s": eval_duration_s,
        }
    )

    _write_trial_file(
        trial_id=inp.trial_id,
        generation=inp.generation,
        experiment_dir=inp.experiment_dir,
        code=code,
        metrics=metrics,
        prompt=inp.prompt,
        response=response_text,
        reasoning=reasoning,
        parent_id=inp.parent_id,
        model_config=result.get("model_config"),
        llm_call_duration_s=result.get("llm_call_duration_s", 0.0),
        eval_duration_s=eval_duration_s,
    )

    return result


def child_worker(inp: WorkerInput) -> dict[str, Any]:
    """Backward compatible alias for spawn_child."""
    return spawn_child(inp)
