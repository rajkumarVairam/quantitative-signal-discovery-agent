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
Human-Readable Output Formatter for Factor Mining Workflow.

This module provides functions to format the JSON output from the factor
optimization agent into a clear, readable format for terminal display.
"""

import json
import logging
import re
from typing import Any

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def _format_status_badge(status: str) -> str:
    """Format status as a colored badge."""
    if status == "accepted":
        return f"{Colors.GREEN}{Colors.BOLD}✓ ACCEPTED{Colors.RESET}"
    elif status == "best_effort":
        return f"{Colors.YELLOW}{Colors.BOLD}◐ BEST EFFORT{Colors.RESET}"
    elif status == "failed":
        return f"{Colors.RED}{Colors.BOLD}✗ FAILED{Colors.RESET}"
    return status


def _format_metric(name: str, value: Any, threshold: float | None = None, higher_is_better: bool = True) -> str:
    """Format a single metric with optional threshold comparison."""
    if value is None:
        return f"  {Colors.DIM}{name}: N/A{Colors.RESET}"

    if isinstance(value, float):
        formatted_value = f"{value:.4f}"
    else:
        formatted_value = str(value)

    # Add threshold indicator if provided
    if threshold is not None and isinstance(value, (int, float)):
        if higher_is_better:
            meets = abs(value) >= threshold
        else:
            meets = value <= threshold
        indicator = f"{Colors.GREEN}●{Colors.RESET}" if meets else f"{Colors.RED}○{Colors.RESET}"
        return f"  {indicator} {name}: {Colors.BOLD}{formatted_value}{Colors.RESET}"

    return f"  • {name}: {Colors.BOLD}{formatted_value}{Colors.RESET}"


def _format_percentile_bar(percentiles: dict[str, float]) -> str:
    """Format IC percentiles as a visual bar."""
    if not percentiles:
        return ""

    p5 = percentiles.get("5th", 0)
    p25 = percentiles.get("25th", 0)
    p50 = percentiles.get("50th", 0)
    p75 = percentiles.get("75th", 0)
    p95 = percentiles.get("95th", 0)

    lines = [
        f"  {Colors.DIM}IC Distribution:{Colors.RESET}",
        f"    5th: {p5:+.3f}  │  25th: {p25:+.3f}  │  50th: {p50:+.3f}  │  75th: {p75:+.3f}  │  95th: {p95:+.3f}",
    ]
    return "\n".join(lines)


def _extract_factor_info(factor_description: str) -> dict[str, Any]:
    """Extract factor information from the description JSON."""
    # Try to parse JSON from the description
    try:
        # Handle markdown code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', factor_description, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = factor_description

        data = json.loads(json_str)
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data
    except json.JSONDecodeError:
        return {"raw_description": factor_description}


def _format_factor_details(factor_info: dict[str, Any]) -> str:
    """Format factor details section."""
    lines = []

    name = factor_info.get("name", "Unknown Factor")
    lines.append(f"\n{Colors.CYAN}{Colors.BOLD}╭{'─' * 60}╮{Colors.RESET}")
    lines.append(f"{Colors.CYAN}{Colors.BOLD}│ FACTOR: {name:<50}│{Colors.RESET}")
    lines.append(f"{Colors.CYAN}{Colors.BOLD}╰{'─' * 60}╯{Colors.RESET}")

    if "category" in factor_info:
        category = factor_info["category"].upper()
        lines.append(f"\n  {Colors.BLUE}Category:{Colors.RESET} {category}")

    if "meaning" in factor_info:
        meaning = factor_info["meaning"]
        # Word wrap the meaning
        wrapped = _word_wrap(meaning, 56)
        lines.append(f"\n  {Colors.BLUE}Economic Intuition:{Colors.RESET}")
        for line in wrapped:
            lines.append(f"    {line}")

    if "formula" in factor_info:
        lines.append(f"\n  {Colors.BLUE}Formula:{Colors.RESET}")
        lines.append(f"    {Colors.DIM}{factor_info['formula']}{Colors.RESET}")

    if "operators_used" in factor_info:
        ops = ", ".join(factor_info["operators_used"])
        lines.append(f"\n  {Colors.BLUE}Operators:{Colors.RESET} {ops}")

    if "lookback_periods" in factor_info:
        periods = ", ".join(str(p) for p in factor_info["lookback_periods"])
        lines.append(f"  {Colors.BLUE}Lookback Periods:{Colors.RESET} {periods} days")

    return "\n".join(lines)


def _format_metrics_section(metrics: dict[str, Any]) -> str:
    """Format the evaluation metrics section."""
    lines = [
        f"\n{Colors.HEADER}{Colors.BOLD}┌{'─' * 40}┐{Colors.RESET}",
        f"{Colors.HEADER}{Colors.BOLD}│ EVALUATION METRICS                     │{Colors.RESET}",
        f"{Colors.HEADER}{Colors.BOLD}└{'─' * 40}┘{Colors.RESET}",
    ]

    # Primary metrics with thresholds
    lines.append(_format_metric("Mean Rank IC", metrics.get("mean_ic"), threshold=0.03))
    lines.append(_format_metric("P-Value", metrics.get("p_value"), threshold=0.1, higher_is_better=False))

    # Secondary metrics
    lines.append("")
    lines.append(_format_metric("IC Std Dev", metrics.get("ic_std")))
    lines.append(_format_metric("IC IR (Info Ratio)", metrics.get("ic_ir")))
    lines.append(_format_metric("T-Statistic", metrics.get("t_stat")))

    # Positive IC ratio
    pos_ratio = metrics.get("positive_ic_ratio")
    if pos_ratio is not None:
        lines.append(_format_metric("Positive IC Ratio", f"{pos_ratio:.1%}"))

    lines.append(_format_metric("Periods Evaluated", metrics.get("num_periods")))

    # Percentile distribution
    percentiles = metrics.get("ic_percentiles")
    if percentiles:
        lines.append("")
        lines.append(_format_percentile_bar(percentiles))

    return "\n".join(lines)


def _format_optimization_history(history: list[dict[str, Any]]) -> str:
    """Format the optimization history section."""
    if not history:
        return ""

    lines = [
        f"\n{Colors.YELLOW}{Colors.BOLD}┌{'─' * 40}┐{Colors.RESET}",
        f"{Colors.YELLOW}{Colors.BOLD}│ OPTIMIZATION HISTORY                   │{Colors.RESET}",
        f"{Colors.YELLOW}{Colors.BOLD}└{'─' * 40}┘{Colors.RESET}",
    ]

    for entry in history:
        iteration = entry.get("iteration", "?")
        mean_ic = entry.get("mean_ic")
        p_value = entry.get("p_value")

        ic_str = f"{mean_ic:.4f}" if mean_ic is not None else "N/A"
        p_str = f"{p_value:.4f}" if p_value is not None else "N/A"

        lines.append(f"\n  {Colors.BOLD}Iteration {iteration}:{Colors.RESET}")
        lines.append(f"    IC: {ic_str}  |  P-value: {p_str}")

        feedback = entry.get("feedback", "")
        if feedback:
            # Extract key points from feedback
            points = [line.strip() for line in feedback.split("\n") if line.strip().startswith("-")]
            if points:
                lines.append(f"    {Colors.DIM}Feedback:{Colors.RESET}")
                for point in points[:3]:  # Limit to first 3 points
                    lines.append(f"      {Colors.DIM}{point}{Colors.RESET}")

    return "\n".join(lines)


def _word_wrap(text: str, width: int) -> list[str]:
    """Wrap text to specified width."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def format_factor_output(result_json: str) -> str:
    """
    Format factor mining result JSON into human-readable output.

    Args:
        result_json: JSON string from factor optimization agent.

    Returns:
        Formatted string for terminal display.
    """
    try:
        result = json.loads(result_json)
    except json.JSONDecodeError:
        return f"{Colors.RED}Error: Invalid JSON output{Colors.RESET}\n{result_json}"

    lines = []

    # Header
    lines.append("")
    lines.append(f"{Colors.BOLD}{'═' * 64}{Colors.RESET}")
    lines.append(f"{Colors.BOLD}  FACTOR MINING RESULTS{Colors.RESET}")
    lines.append(f"{Colors.BOLD}{'═' * 64}{Colors.RESET}")

    # Status
    status = result.get("status", "unknown")
    iterations = result.get("iterations", "?")
    lines.append(f"\n  Status: {_format_status_badge(status)}")
    lines.append(f"  Iterations: {iterations}")

    # Message
    message = result.get("message")
    if message:
        lines.append(f"\n  {Colors.DIM}{message}{Colors.RESET}")

    # Factor details
    factor_desc = result.get("factor_description", "")
    if factor_desc:
        factor_info = _extract_factor_info(factor_desc)
        lines.append(_format_factor_details(factor_info))

    # Evaluation metrics
    metrics = result.get("evaluation_metrics", {})
    if metrics and not metrics.get("error"):
        lines.append(_format_metrics_section(metrics))

    # Optimization history (for best_effort results)
    history = result.get("optimization_history", [])
    if history and status == "best_effort":
        lines.append(_format_optimization_history(history))

    # Saved path
    saved_path = result.get("saved_path")
    if saved_path:
        lines.append(f"\n  {Colors.GREEN}Saved to:{Colors.RESET} {saved_path}")

    # Footer
    lines.append("")
    lines.append(f"{Colors.BOLD}{'═' * 64}{Colors.RESET}")
    lines.append("")

    return "\n".join(lines)


class OutputFormatterConfig(FunctionBaseConfig, name="output_formatter"):
    """
    Output Formatter: Formats factor mining results into human-readable output.

    This component takes the JSON output from the factor optimization agent
    and formats it into a clear, colored terminal display.
    """

    use_colors: bool = Field(
        default=True,
        description="Whether to use ANSI colors in output.",
    )


@register_function(config_type=OutputFormatterConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def output_formatter_function(config: OutputFormatterConfig, builder: Builder):
    """
    Output Formatter that transforms JSON results into human-readable format.
    """

    async def format_output(result_json: str) -> str:
        """
        Format factor mining results for human-readable display.

        Args:
            result_json: The JSON output from factor_optimization_agent.
                        Contains status, factor details, metrics, and history.

        Returns:
            A formatted string suitable for terminal display with:
            - Status badge (accepted/best_effort/failed)
            - Factor details (name, category, formula, meaning)
            - Evaluation metrics with threshold indicators
            - Optimization history (if applicable)
            - Save location
        """
        if not config.use_colors:
            # Strip ANSI codes if colors disabled
            formatted = format_factor_output(result_json)
            return re.sub(r'\033\[[0-9;]*m', '', formatted)
        return format_factor_output(result_json)

    yield FunctionInfo.from_fn(
        format_output,
        description=(
            "Format factor mining JSON results into human-readable terminal output. "
            "Input: JSON string from factor_optimization_agent. "
            "Output: Formatted text with colored sections for status, factor details, "
            "metrics, and optimization history."
        ),
    )


class FactorOptimizerConfig(FunctionBaseConfig, name="factor_optimizer"):
    """
    Factor Optimizer: Generates and optimizes quantitative factors with formatted output.

    Iteratively generates factors, evaluates rank IC, and improves based on feedback.
    Results are displayed in a clear, formatted terminal output.
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
    use_colors: bool = Field(
        default=True,
        description="Whether to use ANSI colors in output.",
    )


@register_function(config_type=FactorOptimizerConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def factor_optimizer_function(config: FactorOptimizerConfig, builder: Builder):
    """
    Factor Optimizer that generates, evaluates, and optimizes quantitative factors.
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
    from .factor_optimization_agent import save_factor_results

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

    # Build signature map for operators
    sig_map = {op['name']: op.get('signature', op['name']) for op in operators}

    async def generate_code(factor_json: str) -> str:
        """Generate executable Python code from factor JSON."""
        import re as regex

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
                        required_ops.update(regex.findall(pattern, factor['formula']))
            elif isinstance(data, dict):
                if 'operators_used' in data:
                    required_ops.update(data['operators_used'])
                if 'formula' in data:
                    pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\('
                    required_ops.update(regex.findall(pattern, data['formula']))
        except json.JSONDecodeError:
            pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\('
            required_ops.update(regex.findall(pattern, factor_json))

        valid_ops = {op for op in required_ops if op in code_map}
        operator_codes = [code_map[op] for op in sorted(valid_ops)]
        operator_code_block = "\n".join(operator_codes)

        # Build operator signatures list for the prompt
        operator_signatures = []
        for op_name in sorted(valid_ops):
            sig = sig_map.get(op_name, op_name)
            # Clean up signature - remove 'def ' prefix if present
            if sig.startswith('def '):
                sig = sig[4:]
            operator_signatures.append(f"  - {sig}")
        operator_sig_block = "\n".join(operator_signatures)

        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = """You are a senior programmer at a top global hedge fund, proficient in Python.
You write precise, executable Python code for calculating price-volume factors.

IMPORTANT RULES:
1. Output ONLY the main factor function(s) - operator functions are already provided
2. Use the EXACT operator function signatures as defined - pass ALL required arguments
3. The factor function should take pandas DataFrames as input and return a DataFrame
4. DO NOT redefine the operator functions - they will be prepended automatically
5. Pay close attention to the number of arguments each operator requires

OUTPUT FORMAT (only the factor function, no imports or operator definitions):
```python
def factor_name(Open: pd.DataFrame, Close: pd.DataFrame, ...) -> pd.DataFrame:
    '''Factor description'''
    result = ...
    return result
```"""

        user_prompt = f"""Write ONLY the main factor function(s) for this JSON:

{factor_json}

AVAILABLE OPERATORS WITH EXACT SIGNATURES (you MUST use these exact signatures):
{operator_sig_block}

CRITICAL: Each operator must be called with ALL required arguments as shown in the signatures above.
For example:
- TS_Corr requires 3 args: TS_Corr(x, y, d) where d is the lookback period
- TS_Std requires 2 args: TS_Std(x, d) where d is the lookback period
- Rank operators take DataFrames, not raw numbers

Generate ONLY the factor function(s):"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await llm.ainvoke(messages)
        factor_function_code = response.content if hasattr(response, 'content') else str(response)

        # Extract code from markdown
        code_blocks = regex.findall(r'```python\n(.*?)```', factor_function_code, regex.DOTALL)
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

        if abs(mean_ic) < config.ic_threshold:
            return False

        if p_value is not None and p_value > config.p_value_threshold:
            return False

        return True

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

    async def generate_optimization_feedback(
        factor_json: str,
        ic_results: dict[str, Any],
        iteration: int,
    ) -> str:
        """Generate optimization advice using LLM based on IC results."""
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
        positive_ratio_str = f"{positive_ratio:.2%}" if positive_ratio else "N/A"

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

    async def run_formatted_optimization(request: str) -> str:
        """
        Run factor optimization with formatted output.

        Generates and optimizes quantitative factors based on the request,
        then formats the results into a human-readable display.

        Args:
            request: Description of the type of factor to generate.
                    Examples: "momentum factors", "volatility factors",
                    "mean reversion factors", "volume-price divergence factors"

        Returns:
            Human-readable formatted output containing:
            - Status (accepted/best_effort/failed)
            - Factor details (name, formula, economic meaning)
            - Evaluation metrics (IC, p-value, etc.)
            - Optimization history (if applicable)
            - File save location
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

                saved_path = None
                if config.save_results:
                    saved_path = save_factor_results(
                        factor_json, factor_code, ic_results, iteration
                    )

                result = {
                    "status": "accepted",
                    "iterations": iteration,
                    "factor_description": factor_json,
                    "factor_code": factor_code,
                    "evaluation_metrics": ic_results,
                    "optimization_history": optimization_history,
                    "saved_path": saved_path,
                    "message": f"Factor accepted after {iteration} iteration(s) with IC={mean_ic:.4f}",
                }
                result_json = json.dumps(result, indent=2)
                formatted = format_factor_output(result_json)
                if not config.use_colors:
                    formatted = re.sub(r'\033\[[0-9;]*m', '', formatted)
                print(formatted)
                # Return simple message (formatted output already printed above)
                return f"Factor accepted with IC={mean_ic:.4f}. Saved to: {saved_path}"

            # Step 5: Generate feedback for next iteration using LLM
            logger.info("Factor not acceptable. Generating optimization advice...")
            feedback = await generate_optimization_feedback(factor_json, ic_results, iteration)
            logger.debug(f"Optimization feedback:\n{feedback}")

            optimization_history.append({
                "iteration": iteration,
                "mean_ic": mean_ic,
                "p_value": ic_results.get("p_value"),
                "feedback": feedback,
            })

        # Max iterations reached - return best result
        logger.info("Max iterations reached. Returning best result.")

        if best_result is None:
            result = {
                "status": "failed",
                "iterations": config.max_iterations,
                "message": "Failed to generate any valid factors",
                "optimization_history": optimization_history,
            }
        else:
            saved_path = None
            if config.save_results:
                saved_path = save_factor_results(
                    best_result["factor_json"],
                    best_result["factor_code"],
                    best_result["ic_results"],
                    best_result["iteration"],
                )

            result = {
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
            }

        result_json = json.dumps(result, indent=2)
        formatted = format_factor_output(result_json)
        if not config.use_colors:
            formatted = re.sub(r'\033\[[0-9;]*m', '', formatted)
        print(formatted)
        # Return simple message (formatted output already printed above)
        if best_result:
            return f"Best effort: IC={best_ic:.4f} after {config.max_iterations} iterations. Saved to: {result.get('saved_path', 'N/A')}"
        return "Failed to generate valid factors."

    yield FunctionInfo.from_fn(
        run_formatted_optimization,
        description=(
            "Run factor optimization with human-readable formatted output. "
            "Generates, evaluates, and iteratively improves quantitative factors "
            "based on rank IC feedback. Input: type of factor to generate "
            "(e.g., 'momentum factors', 'volatility factors'). "
            "Output: Formatted text with status, factor details, metrics, and history."
        ),
    )

