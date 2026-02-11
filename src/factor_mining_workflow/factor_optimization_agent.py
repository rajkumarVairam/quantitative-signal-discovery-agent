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
Factor Optimization Agent for Factor Mining Workflow.

This agent orchestrates the factor mining loop:
1. Generate factor descriptions
2. Generate executable code
3. Evaluate rank IC
4. If IC is good → accept and save the factor
5. If IC is poor → provide optimization advice and retry
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

# Output directory for saved factors
OUTPUT_DIR = Path(__file__).parent / "output"


def ensure_output_dir() -> Path:
    """Ensure the output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def save_factor_results(
    factor_json: str,
    factor_code: str,
    ic_results: dict[str, Any],
    iteration: int,
) -> str:
    """
    Save successful factor results to a file.

    Returns:
        Path to the saved file.
    """
    output_dir = ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"factor_{timestamp}_iter{iteration}.json"
    filepath = output_dir / filename

    results = {
        "timestamp": timestamp,
        "iteration": iteration,
        "factor_description": factor_json,
        "factor_code": factor_code,
        "evaluation_metrics": ic_results,
    }

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved factor results to {filepath}")
    return str(filepath)


class FactorOptimizationAgentConfig(FunctionBaseConfig, name="factor_optimization_agent"):
    """
    Factor Optimization Agent: Iteratively improves factors based on rank IC feedback.

    This agent runs a loop that generates factors, evaluates them, and either
    accepts good factors or provides optimization advice for poor performers.
    """

    llm_name: str = Field(
        description="LLM to use for factor generation and optimization advice.",
    )
    ic_threshold: float = Field(
        default=0.03,
        description="Minimum mean IC required to accept a factor (absolute value).",
    )
    p_value_threshold: float = Field(
        default=0.1,
        description="Maximum p-value for statistical significance.",
    )
    max_iterations: int = Field(
        default=3,
        description="Maximum optimization iterations before accepting best result.",
    )
    num_factors: int = Field(
        default=1,
        description="Number of factors to generate per iteration.",
    )
    forward_periods: int = Field(
        default=5,
        description="Forward periods for return calculation in IC evaluation.",
    )
    save_results: bool = Field(
        default=True,
        description="Whether to save successful factor results to disk.",
    )


@register_function(config_type=FactorOptimizationAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def factor_optimization_agent_function(config: FactorOptimizationAgentConfig, builder: Builder):
    """
    Factor Optimization Agent that iteratively improves factors based on IC feedback.
    """
    from .factor_generator import (
        load_calculator_operators,
        format_operators_for_prompt,
        get_output_format_prompt,
    )
    from .rank_ic_evaluator import (
        load_stock_data,
        compute_forward_returns,
        compute_rank_ic,
        extract_code_from_response,
        execute_factor_code,
    )

    # Get LLM
    llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Load resources once
    operators = load_calculator_operators()
    operators_list = format_operators_for_prompt(operators)
    output_format = get_output_format_prompt().replace("{num_factors}", str(config.num_factors))
    stock_data = load_stock_data()

    # Build operator code map for code generation
    code_map = {op['name']: op['code'] for op in operators}

    async def generate_factors(request: str, previous_feedback: str | None = None) -> str:
        """Generate factor descriptions using LLM."""
        feedback_section = ""
        if previous_feedback:
            feedback_section = f"""

IMPORTANT - PREVIOUS ATTEMPT FEEDBACK:
{previous_feedback}

Please generate IMPROVED factors based on this feedback. Avoid the issues mentioned above.
"""

        prompt = f"""You are a senior quantitative researcher at a top hedge fund.
Generate {config.num_factors} unique stock selection factors based on the request.

REQUEST: {request}
{feedback_section}
DATA AVAILABLE:
- Open: Opening price
- Close: Closing price
- High: Highest price
- Low: Lowest price
- Volume: Trading volume

OPERATORS YOU CAN USE (combine these to create complex factors):
{operators_list}

{output_format}

IMPORTANT:
1. Use ONLY the operators listed above
2. Create factors with clear economic intuition
3. Each factor should be unique and capture different alpha signals
4. Return valid JSON that can be parsed

Generate {config.num_factors} factors now:"""

        response = await llm.ainvoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    async def generate_code(factor_json: str) -> str:
        """Generate executable Python code from factor JSON."""
        import re

        # Extract required operators
        required_ops = set()
        try:
            data = json.loads(factor_json)
            if isinstance(data, list):
                for factor in data:
                    if 'operators_used' in factor:
                        required_ops.update(factor['operators_used'])
                    if 'formula' in factor:
                        pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\('
                        required_ops.update(re.findall(pattern, factor['formula']))
            elif isinstance(data, dict):
                if 'operators_used' in data:
                    required_ops.update(data['operators_used'])
                if 'formula' in data:
                    pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\('
                    required_ops.update(re.findall(pattern, data['formula']))
        except json.JSONDecodeError:
            pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\('
            required_ops.update(re.findall(pattern, factor_json))

        valid_ops = {op for op in required_ops if op in code_map}
        operator_codes = [code_map[op] for op in sorted(valid_ops)]
        operator_code_block = "\n".join(operator_codes)

        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = """You are a senior programmer at a top global hedge fund, proficient in Python.
You write precise, executable Python code for calculating price-volume factors.

IMPORTANT RULES:
1. Output ONLY the main factor function(s) - operator functions are already provided
2. Use the exact operator function names as they are defined
3. The factor function should take pandas DataFrames as input and return a DataFrame
4. DO NOT redefine the operator functions - they will be prepended automatically

OUTPUT FORMAT (only the factor function, no imports or operator definitions):
```python
def factor_name(Open: pd.DataFrame, Close: pd.DataFrame, ...) -> pd.DataFrame:
    '''Factor description'''
    result = ...
    return result
```"""

        user_prompt = f"""Write ONLY the main factor function(s) for this JSON:

{factor_json}

AVAILABLE OPERATORS (already defined):
{', '.join(sorted(valid_ops))}

Generate ONLY the factor function(s):"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await llm.ainvoke(messages)
        factor_function_code = response.content if hasattr(response, 'content') else str(response)

        # Extract code from markdown
        code_blocks = re.findall(r'```python\n(.*?)```', factor_function_code, re.DOTALL)
        if code_blocks:
            factor_function_code = code_blocks[0]

        # Build complete code
        output_parts = [
            "import pandas as pd",
            "import numpy as np",
            "",
            "# Required operator functions",
            operator_code_block,
            "",
            "# Factor calculation function(s)",
            factor_function_code,
        ]

        return "\n".join(output_parts)

    def evaluate_ic(factor_code: str) -> dict[str, Any]:
        """Evaluate rank IC of the factor."""
        if not stock_data:
            return {"error": "No stock data available", "mean_ic": None}

        clean_code = extract_code_from_response(factor_code)
        factor_values = execute_factor_code(clean_code, stock_data)

        if factor_values is None:
            return {"error": "Failed to execute factor code", "mean_ic": None}

        close_data = stock_data.get('Close')
        if close_data is None:
            return {"error": "Close price data not available", "mean_ic": None}

        forward_returns = compute_forward_returns(close_data, periods=config.forward_periods)
        return compute_rank_ic(factor_values, forward_returns)

    def is_factor_acceptable(ic_results: dict[str, Any]) -> bool:
        """Check if factor meets acceptance criteria."""
        mean_ic = ic_results.get("mean_ic")
        p_value = ic_results.get("p_value")

        if mean_ic is None:
            return False

        # Check if IC magnitude meets threshold
        if abs(mean_ic) < config.ic_threshold:
            return False

        # Check statistical significance
        if p_value is not None and p_value > config.p_value_threshold:
            return False

        return True

    async def generate_optimization_feedback(
        factor_json: str,
        ic_results: dict[str, Any],
        iteration: int,
    ) -> str:
        """Generate optimization advice based on IC results."""
        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = """You are a senior quantitative researcher providing feedback on factor performance.
Based on the rank IC evaluation results, provide specific, actionable advice to improve the factor.

Focus on:
1. What might be wrong with the current factor design
2. Specific changes to the formula or lookback periods
3. Alternative approaches that might work better
4. Economic intuition that could improve predictive power"""

        mean_ic = ic_results.get("mean_ic")
        ic_std = ic_results.get("ic_std")
        p_value = ic_results.get("p_value")
        positive_ratio = ic_results.get("positive_ic_ratio", 0)

        # Format values safely
        mean_ic_str = f"{mean_ic:.4f}" if mean_ic is not None else "N/A"
        ic_std_str = f"{ic_std:.4f}" if ic_std is not None else "N/A"
        p_value_str = f"{p_value:.4f}" if p_value is not None else "N/A"
        positive_ratio_str = f"{positive_ratio:.2%}"

        issues_str = _identify_issues(ic_results)

        user_prompt = f"""Iteration {iteration} factor evaluation results:

FACTOR DESCRIPTION:
{factor_json}

EVALUATION METRICS:
- Mean Rank IC: {mean_ic_str}
- IC Standard Deviation: {ic_std_str}
- P-value: {p_value_str}
- Positive IC Ratio: {positive_ratio_str}

TARGET THRESHOLDS:
- Required |IC| >= {config.ic_threshold}
- Required p-value <= {config.p_value_threshold}

ISSUES:
{issues_str}

Provide specific optimization advice (3-5 bullet points):"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await llm.ainvoke(messages)
        return response.content if hasattr(response, 'content') else str(response)

    def _identify_issues(ic_results: dict[str, Any]) -> str:
        """Identify specific issues with the factor."""
        issues = []
        mean_ic = ic_results.get("mean_ic")
        p_value = ic_results.get("p_value")
        positive_ratio = ic_results.get("positive_ic_ratio", 0)

        if mean_ic is None:
            issues.append("- Factor code execution failed or returned invalid values")
        elif abs(mean_ic) < 0.01:
            issues.append("- IC is essentially zero - factor has no predictive power")
            issues.append("- Consider using different data combinations or lookback periods")
        elif abs(mean_ic) < config.ic_threshold:
            issues.append(f"- IC magnitude ({abs(mean_ic):.4f}) is below threshold ({config.ic_threshold})")
            issues.append("- Factor shows weak signal, needs stronger alpha source")

        if p_value and p_value > config.p_value_threshold:
            issues.append(f"- Results not statistically significant (p={p_value:.4f})")
            issues.append("- High variance in IC - factor behavior is inconsistent")

        if positive_ratio < 0.4:
            issues.append(f"- Low positive IC ratio ({positive_ratio:.1%}) - factor often gives wrong signals")
        elif positive_ratio > 0.6 and mean_ic and mean_ic < 0:
            issues.append("- Negative mean IC despite frequent positive periods - large losses on bad days")

        return "\n".join(issues) if issues else "- No specific issues identified"

    async def optimize_factor(request: str) -> str:
        """
        Run the factor optimization loop.

        Iteratively generates and evaluates factors until one meets the acceptance
        criteria or max iterations is reached.

        Args:
            request: Description of the type of factor to generate.
                    Examples: "momentum factors", "volatility factors",
                    "mean reversion factors", "volume-price divergence factors"

        Returns:
            JSON string containing:
            - status: "accepted" or "best_effort"
            - iterations: Number of iterations run
            - factor_description: The factor JSON
            - factor_code: The generated Python code
            - evaluation_metrics: Rank IC and related metrics
            - optimization_history: Feedback from each iteration
            - saved_path: Path to saved results (if save_results=True)
        """
        optimization_history = []
        best_result = None
        best_ic = 0

        feedback = None

        for iteration in range(1, config.max_iterations + 1):
            logger.info(f"=== Optimization Iteration {iteration}/{config.max_iterations} ===")

            # Step 1: Generate factors
            logger.info("Generating factor descriptions...")
            factor_json = await generate_factors(request, feedback)

            # Step 2: Generate code
            logger.info("Generating factor code...")
            factor_code = await generate_code(factor_json)

            # Step 3: Evaluate rank IC
            logger.info("Evaluating rank IC...")
            ic_results = evaluate_ic(factor_code)

            iteration_result = {
                "iteration": iteration,
                "factor_json": factor_json,
                "factor_code": factor_code,
                "ic_results": ic_results,
            }

            # Track best result
            mean_ic = ic_results.get("mean_ic")
            if mean_ic is not None and abs(mean_ic) > abs(best_ic):
                best_ic = mean_ic
                best_result = iteration_result

            # Step 4: Check acceptance criteria
            if is_factor_acceptable(ic_results):
                logger.info(f"Factor ACCEPTED at iteration {iteration}!")
                logger.info(f"Mean IC: {mean_ic:.4f}, p-value: {ic_results.get('p_value', 'N/A')}")

                saved_path = None
                if config.save_results:
                    saved_path = save_factor_results(
                        factor_json, factor_code, ic_results, iteration
                    )

                return json.dumps({
                    "status": "accepted",
                    "iterations": iteration,
                    "factor_description": factor_json,
                    "factor_code": factor_code,
                    "evaluation_metrics": ic_results,
                    "optimization_history": optimization_history,
                    "saved_path": saved_path,
                    "message": f"Factor accepted after {iteration} iteration(s) with IC={mean_ic:.4f}",
                }, indent=2)

            # Step 5: Generate optimization feedback
            logger.info("Factor not acceptable. Generating optimization advice...")
            feedback = await generate_optimization_feedback(factor_json, ic_results, iteration)

            optimization_history.append({
                "iteration": iteration,
                "mean_ic": mean_ic,
                "p_value": ic_results.get("p_value"),
                "feedback": feedback,
            })

            logger.info(f"Optimization feedback:\n{feedback}")

        # Max iterations reached - return best result
        logger.info("Max iterations reached. Returning best result.")

        if best_result is None:
            return json.dumps({
                "status": "failed",
                "iterations": config.max_iterations,
                "message": "Failed to generate any valid factors",
                "optimization_history": optimization_history,
            }, indent=2)

        saved_path = None
        if config.save_results:
            saved_path = save_factor_results(
                best_result["factor_json"],
                best_result["factor_code"],
                best_result["ic_results"],
                best_result["iteration"],
            )

        return json.dumps({
            "status": "best_effort",
            "iterations": config.max_iterations,
            "factor_description": best_result["factor_json"],
            "factor_code": best_result["factor_code"],
            "evaluation_metrics": best_result["ic_results"],
            "optimization_history": optimization_history,
            "saved_path": saved_path,
            "message": (
                f"Max iterations reached. Best factor from iteration {best_result['iteration']} "
                f"with IC={best_result['ic_results'].get('mean_ic', 'N/A')}"
            ),
        }, indent=2)

    yield FunctionInfo.from_fn(
        optimize_factor,
        description=(
            "Run the factor optimization loop. Generates factors, evaluates rank IC, "
            "and iteratively improves based on feedback until a good factor is found "
            "or max iterations is reached. Input: type of factor to generate "
            "(e.g., 'momentum factors', 'volatility factors')."
        ),
    )
