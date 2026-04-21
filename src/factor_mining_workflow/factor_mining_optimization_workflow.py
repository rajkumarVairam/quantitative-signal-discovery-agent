# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Factor Mining Optimization Workflow.

Orchestrates the loop:
    factor_generator -> factor_code_generator -> factor_evaluator
                ^                                           |
                |__________ optimization advisor ___________|

The agents themselves live in their own modules; this file only handles
iteration, scoring, feedback, and persisting accepted/best results.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from .factor_code_generator import generate_factor_code
from .factor_evaluator import (
    compute_forward_returns,
    compute_rank_ic,
    execute_factor_code,
    extract_code_from_response,
    load_stock_data,
)
from .factor_generator import generate_factor_json, load_calculator_operators
from .llm_utils import extract_response_text

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"


# =============================================================================
# Result persistence
# =============================================================================


def save_factor_results(
    factor_json: str,
    factor_code: str,
    ic_results: dict,
    iteration: int,
) -> str:
    """Save factor results to ``output/factor_<timestamp>_iter<n>.json``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = OUTPUT_DIR / f"factor_{timestamp}_iter{iteration}.json"

    metrics = {k: v for k, v in ic_results.items() if k != "selected_factor"}
    payload = {
        "timestamp": timestamp,
        "iteration": iteration,
        "selected_factor": ic_results.get("selected_factor"),
        "factor_description": factor_json,
        "factor_code": factor_code,
        "evaluation_metrics": metrics,
    }
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved factor results to {filepath}")
    return str(filepath)


# =============================================================================
# Workflow config
# =============================================================================


class FactorOptimizerConfig(FunctionBaseConfig, name="factor_optimizer"):
    """Iteratively generate, evaluate, and refine quantitative factors."""

    factor_generator_llm: str | None = Field(default=None, description="LLM for factor generation")
    code_generator_llm: str | None = Field(default=None, description="LLM for code generation")
    optimization_advisor_llm: str | None = Field(default=None, description="LLM for optimization feedback")
    llm_name: str | None = Field(default=None, description="Fallback LLM if specific ones not set")
    ic_threshold: float = Field(default=0.03, description="Minimum |IC| to accept")
    p_value_threshold: float = Field(default=0.05, description="Maximum p-value")
    max_iterations: int = Field(default=3, description="Max optimization attempts")
    num_factors: int = Field(default=5, description="Factors per iteration")
    forward_periods: int = Field(default=5, description="Forward return periods")
    save_results: bool = Field(default=True, description="Save results to disk")


# =============================================================================
# Orchestrator
# =============================================================================


@register_function(config_type=FactorOptimizerConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def factor_optimizer_function(config: FactorOptimizerConfig, builder: Builder):
    """Compose the factor / code / evaluator agents into an optimization loop."""

    factor_llm_name = config.factor_generator_llm or config.llm_name
    code_llm_name = config.code_generator_llm or config.llm_name
    advisor_llm_name = config.optimization_advisor_llm or config.llm_name
    if not factor_llm_name:
        raise ValueError("Must specify llm_name or factor_generator_llm")

    factor_llm = await builder.get_llm(factor_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    code_llm = await builder.get_llm(code_llm_name or factor_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    advisor_llm = await builder.get_llm(advisor_llm_name or factor_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    logger.info(f"LLMs: factor={factor_llm_name}, code={code_llm_name}, advisor={advisor_llm_name}")

    operators = load_calculator_operators()
    stock_data = load_stock_data()

    # ---- per-step helpers (close over the LLMs and shared state) ----

    def evaluate_ic(factor_code: str) -> dict[str, Any]:
        """Run the factor code and compute its rank IC against forward returns."""
        if not stock_data:
            return {"error": "No stock data", "mean_ic": None}

        clean_code = extract_code_from_response(factor_code)
        exec_result = execute_factor_code(clean_code, stock_data)
        if exec_result is None:
            return {"error": "Code execution failed", "mean_ic": None}
        factor_values, selected_factor = exec_result

        close = stock_data.get("Close")
        if close is None:
            return {"error": "No Close data", "mean_ic": None}

        forward_returns = compute_forward_returns(close, periods=config.forward_periods)
        ic_results = compute_rank_ic(factor_values, forward_returns)
        ic_results["selected_factor"] = selected_factor
        return ic_results

    def is_acceptable(ic_results: dict) -> bool:
        mean_ic = ic_results.get("mean_ic")
        p_value = ic_results.get("p_value")
        if mean_ic is None or abs(mean_ic) < config.ic_threshold:
            return False
        if p_value is not None and p_value > config.p_value_threshold:
            return False
        return True

    async def generate_feedback(factor_json: str, ic_results: dict, iteration: int) -> str:
        """Ask the advisor LLM for optimization advice for the next iteration."""
        mean_ic = ic_results.get("mean_ic")
        p_value = ic_results.get("p_value")
        mean_ic_str = f"{mean_ic:.4f}" if mean_ic is not None else "N/A"
        p_value_str = f"{p_value:.4f}" if p_value is not None else "N/A"
        error = ic_results.get("error", "")

        prompt = f"""Factor evaluation (iteration {iteration}):
- Mean IC: {mean_ic_str} (need >= {config.ic_threshold})
- P-value: {p_value_str} (need <= {config.p_value_threshold})
{f"- Error: {error}" if error else ""}

FACTOR: {factor_json[:500]}

Provide 3-5 specific improvements:"""

        response = await advisor_llm.ainvoke(
            [
                SystemMessage(content="You are a senior quant providing factor optimization advice."),
                HumanMessage(content=prompt),
            ]
        )
        return extract_response_text(response)

    # ---- main optimization loop ----

    async def run_optimization(request: str) -> str:
        """Run the closed-loop factor mining optimization."""
        best_result: dict | None = None
        best_ic: float | None = None
        feedback: str | None = None

        for iteration in range(1, config.max_iterations + 1):
            logger.info(f"=== Iteration {iteration}/{config.max_iterations} ===")

            logger.info("Generating factors...")
            factor_json = await generate_factor_json(
                factor_llm, request, config.num_factors, operators, feedback
            )

            logger.info("Generating code...")
            factor_code = await generate_factor_code(code_llm, factor_json, operators)

            logger.info("Evaluating IC...")
            ic_results = evaluate_ic(factor_code)
            mean_ic = ic_results.get("mean_ic")

            if mean_ic is not None:
                logger.info(f"Mean IC: {mean_ic:.4f}, p-value: {ic_results.get('p_value', 'N/A')}")
                if best_ic is None or abs(mean_ic) > abs(best_ic):
                    best_ic = mean_ic
                    best_result = {
                        "factor_json": factor_json,
                        "factor_code": factor_code,
                        "ic_results": ic_results,
                        "iteration": iteration,
                    }

            if is_acceptable(ic_results):
                logger.info("Factor accepted!")
                saved_path = (
                    save_factor_results(factor_json, factor_code, ic_results, iteration)
                    if config.save_results
                    else None
                )
                return json.dumps(
                    {
                        "status": "accepted",
                        "iteration": iteration,
                        "mean_ic": mean_ic,
                        "p_value": ic_results.get("p_value"),
                        "saved_path": saved_path,
                    },
                    indent=2,
                )

            logger.info("Generating optimization feedback...")
            feedback = await generate_feedback(factor_json, ic_results, iteration)

        if best_result:
            saved_path = (
                save_factor_results(
                    best_result["factor_json"],
                    best_result["factor_code"],
                    best_result["ic_results"],
                    best_result["iteration"],
                )
                if config.save_results
                else None
            )
            return json.dumps(
                {
                    "status": "best_effort",
                    "iteration": best_result["iteration"],
                    "mean_ic": best_ic,
                    "saved_path": saved_path,
                },
                indent=2,
            )

        return json.dumps({"status": "failed", "iterations": config.max_iterations}, indent=2)

    yield FunctionInfo.from_fn(
        run_optimization,
        description="Run factor mining optimization loop: generate -> code -> evaluate -> feedback.",
    )
