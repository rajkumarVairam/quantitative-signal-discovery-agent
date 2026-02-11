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
Factor Generator Component for Factor Mining Workflow.

This component generates quantitative price-volume factors using LLM
based on the calculator operators defined in template/calculator.json.
"""

import json
import logging
from pathlib import Path

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

# Path to templates
TEMPLATE_DIR = Path(__file__).parent / "template"
CALCULATOR_JSON = TEMPLATE_DIR / "calculator.json"
OUTPUT_TEMPLATE_JSON = TEMPLATE_DIR / "factor_output_template.json"

# Data fields based on sp500 folder structure
DATA_FIELDS = """
Available price-volume data fields (from data/sp500/):
- Open: Opening price (Open.csv)
- Close: Closing price (Close.csv)
- High: Highest price (High.csv)
- Low: Lowest price (Low.csv)
- Volume: Trading volume (Volume.csv)
"""

# Valid data field names for validation
VALID_DATA_FIELDS = {'Open', 'Close', 'High', 'Low', 'Volume'}


def load_calculator_operators() -> list[dict]:
    """Load calculator operators from the template JSON file."""
    if not CALCULATOR_JSON.exists():
        logger.warning(f"Calculator template not found at {CALCULATOR_JSON}")
        return []

    with open(CALCULATOR_JSON, "r") as f:
        return json.load(f)


def load_output_template() -> dict:
    """Load the factor output template."""
    if not OUTPUT_TEMPLATE_JSON.exists():
        logger.warning(f"Output template not found at {OUTPUT_TEMPLATE_JSON}")
        return {}

    with open(OUTPUT_TEMPLATE_JSON, "r") as f:
        return json.load(f)


def get_output_format_prompt() -> str:
    """Get the output format instructions from the template."""
    template = load_output_template()
    if not template:
        return ""

    example = template.get("output_format", {}).get("example", [])
    if example:
        example_json = json.dumps(example[0], indent=2)
        return f"""
OUTPUT FORMAT:
Return each factor as a JSON object with these fields:
- name: Factor name (descriptive)
- formula: Formula using ONLY the operators listed above
- meaning: Economic intuition (what alpha it captures)
- category: One of [momentum, volatility, volume, reversal, quality, other]
- data_fields_used: List of data fields used (Open, Close, High, Low, Volume)
- operators_used: List of operators used
- lookback_periods: List of lookback days used

Example output for ONE factor:
```json
{example_json}
```

Return a JSON array containing all {'{num_factors}'} factors.
"""
    return ""


def format_operators_for_prompt(operators: list[dict], max_operators: int = 30) -> str:
    """Format calculator operators into a prompt-friendly string with full signatures."""
    # Select most useful operators to keep prompt manageable
    priority_prefixes = ['TS_', 'Rank', 'Add', 'Sub', 'Mul', 'Div', 'Decay', 'EMA', 'CS_']
    priority_ops = []
    other_ops = []

    for op in operators:
        is_priority = any(op['name'].startswith(p) or op['name'] == p for p in priority_prefixes)
        if is_priority:
            priority_ops.append(op)
        else:
            other_ops.append(op)

    selected = priority_ops[:max_operators]
    if len(selected) < max_operators:
        selected.extend(other_ops[:max_operators - len(selected)])

    formatted = []
    for op in selected:
        # Include the full signature so LLM knows exact parameter requirements
        signature = op.get('signature', op['name'])
        # Clean up signature - remove 'def ' prefix if present
        if signature.startswith('def '):
            signature = signature[4:]
        formatted.append(f"- {signature}")
        formatted.append(f"  Description: {op['meanings']}")
    return "\n".join(formatted)


# Re-export utility functions for backward compatibility
def get_operator_code_map(operators: list[dict]) -> dict[str, str]:
    """Create a mapping of operator names to their code implementations."""
    return {op['name']: op['code'] for op in operators}


class FactorGeneratorConfig(FunctionBaseConfig, name="factor_generator"):
    """
    Factor Generator: Creates quantitative factors using calculator operators.
    """

    num_factors: int = Field(
        default=3,
        description="Number of factors to generate."
    )
    llm_name: str | None = Field(
        default=None,
        description="LLM to use for generation. If None, uses the agent's LLM."
    )


@register_function(config_type=FactorGeneratorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def factor_generator_function(config: FactorGeneratorConfig, builder: Builder):
    """
    Factor generator that creates quantitative factors using predefined operators.
    """

    # Load calculator operators and output template
    operators = load_calculator_operators()
    operators_list = format_operators_for_prompt(operators)
    output_format = get_output_format_prompt().replace("{num_factors}", str(config.num_factors))

    # Get LLM if specified
    llm = None
    if config.llm_name:
        llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def generate_factors(request: str) -> str:
        """
        Generate quantitative factors for stock selection.

        Args:
            request: What kind of factors to generate. Examples:
                    - "momentum factors"
                    - "volatility factors"
                    - "volume-price divergence factors"

        Returns:
            Generated factors in JSON format with names, formulas, meanings, and metadata.
        """
        prompt = f"""You are a senior quantitative researcher at a top hedge fund.
Generate {config.num_factors} unique stock selection factors based on the request.

REQUEST: {request}

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

        if llm:
            # Use the configured LLM
            response = await llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        else:
            # Return the prompt for the agent to process
            return f"Please generate factors based on this specification:\n\n{prompt}"

    yield FunctionInfo.from_fn(
        generate_factors,
        description="Generate quantitative stock selection factors in JSON format. Input: description of factors needed (e.g., 'momentum factors', 'volatility factors')."
    )


class FactorValidatorConfig(FunctionBaseConfig, name="factor_validator"):
    """
    Validates factor formulas against calculator.json operators.
    """

    strict_mode: bool = Field(
        default=True,
        description="Reject factors with unknown operators."
    )


@register_function(config_type=FactorValidatorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def factor_validator_function(config: FactorValidatorConfig, builder: Builder):
    """
    Validates factor formulas to ensure they use valid operators.
    """
    import re

    operators = load_calculator_operators()
    valid_operators = {op['name'] for op in operators}

    async def validate_factor(formula: str) -> str:
        """
        Validate a factor formula.

        Args:
            formula: Factor formula to validate. Example: "Div(TS_Return(Close, 20), TS_Std(Close, 20))"

        Returns:
            Validation result showing if formula is valid.
        """
        # Extract function names
        pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\('
        matches = re.findall(pattern, formula)

        unknown = [m for m in matches if m not in valid_operators and m not in VALID_DATA_FIELDS]
        valid_used = [m for m in matches if m in valid_operators]

        if unknown and config.strict_mode:
            return f"INVALID: Unknown operators: {', '.join(unknown)}"
        else:
            return f"VALID: Uses operators: {', '.join(set(valid_used))}"

    yield FunctionInfo.from_fn(
        validate_factor,
        description="Validate a factor formula against available operators."
    )


class ListOperatorsConfig(FunctionBaseConfig, name="list_operators"):
    """
    Lists available calculator operators.
    """

    category: str | None = Field(
        default=None,
        description="Filter by category prefix (e.g., 'TS_', 'CS_')."
    )


@register_function(config_type=ListOperatorsConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def list_operators_function(config: ListOperatorsConfig, builder: Builder):
    """
    Lists available operators for factor construction.
    """

    operators = load_calculator_operators()

    async def list_operators(category: str | None = None) -> str:
        """
        List available operators for factor construction.

        Args:
            category: Optional filter prefix. Examples: "TS_" for time-series, "CS_" for cross-sectional.

        Returns:
            List of available operators with descriptions.
        """
        cat = category if category is not None else config.category

        filtered = operators
        if cat:
            filtered = [op for op in operators if op['name'].startswith(cat)]

        if not filtered:
            return f"No operators found for category: {cat}"

        result = [f"Available Operators ({len(filtered)} total):\n"]
        for op in filtered[:25]:  # Limit output
            result.append(f"- {op['name']}: {op['meanings'][:80]}...")

        if len(filtered) > 25:
            result.append(f"\n... and {len(filtered) - 25} more")

        return "\n".join(result)

    yield FunctionInfo.from_fn(
        list_operators,
        description="List available calculator operators for factor construction."
    )
