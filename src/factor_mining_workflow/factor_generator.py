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
Factor Generator Agent.

Calls an LLM to produce factor descriptions in the JSON format defined by
``template/factor_output_template.json``. Each factor is a dict with at least
``name``, ``formula``, ``meaning``, ``data_fields_used``, ``operators_used``.

This module also provides:
  - ``load_calculator_operators`` / ``format_operators_for_prompt`` helpers
  - ``factor_validator_function`` and ``list_operators_function`` NAT tools
"""

import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field

from .llm_utils import NO_THINK_INSTRUCTION, extract_response_text

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "template"
CALCULATOR_JSON = TEMPLATE_DIR / "calculator.json"
OUTPUT_TEMPLATE_JSON = TEMPLATE_DIR / "factor_output_template.json"

VALID_DATA_FIELDS = {"Open", "Close", "High", "Low", "Volume"}


# =============================================================================
# Operator catalogue helpers
# =============================================================================


def load_calculator_operators() -> list[dict]:
    """Load operator definitions from ``template/calculator.json``."""
    if not CALCULATOR_JSON.exists():
        logger.warning(f"Calculator template not found at {CALCULATOR_JSON}")
        return []
    with open(CALCULATOR_JSON, "r") as f:
        return json.load(f)


def load_output_template() -> dict:
    """Load the factor output schema from ``template/factor_output_template.json``."""
    if not OUTPUT_TEMPLATE_JSON.exists():
        logger.warning(f"Output template not found at {OUTPUT_TEMPLATE_JSON}")
        return {}
    with open(OUTPUT_TEMPLATE_JSON, "r") as f:
        return json.load(f)


def get_operator_code_map(operators: list[dict]) -> dict[str, str]:
    """Map operator name -> Python implementation source string."""
    return {op["name"]: op["code"] for op in operators}


def format_operators_for_prompt(operators: list[dict], max_operators: int = 30) -> str:
    """
    Format operator signatures + descriptions for inclusion in a prompt.

    Prioritizes common families (TS_*, Rank, basic arithmetic, Decay/EMA, CS_)
    so the prompt stays bounded when many operators are available.
    """
    priority_prefixes = ["TS_", "Rank", "Add", "Sub", "Mul", "Div", "Decay", "EMA", "CS_"]
    priority_ops, other_ops = [], []
    for op in operators:
        if any(op["name"].startswith(p) or op["name"] == p for p in priority_prefixes):
            priority_ops.append(op)
        else:
            other_ops.append(op)

    selected = priority_ops[:max_operators]
    if len(selected) < max_operators:
        selected.extend(other_ops[: max_operators - len(selected)])

    lines: list[str] = []
    for op in selected:
        signature = op.get("signature", op["name"])
        if signature.startswith("def "):
            signature = signature[4:]
        lines.append(f"- {signature}")
        lines.append(f"  Description: {op['meanings']}")
    return "\n".join(lines)


def build_factor_template(num_factors: int, template: dict | None = None) -> str:
    """
    Render the placeholder JSON the factor generator should fill in.

    The shape comes from ``factor_output_template.json::factor_template`` so
    the schema is the single source of truth: edit the template file to
    change every place the placeholder is shown.
    """
    template = template if template is not None else load_output_template()
    factor_template = template.get("factor_template")
    if not factor_template:
        raise RuntimeError(
            "factor_output_template.json is missing the 'factor_template' field"
        )
    item = json.dumps(factor_template, indent=2)
    items = ",\n".join([item] * num_factors)
    return "```json\n[\n" + items + "\n]\n```"


def build_factor_example(template: dict | None = None) -> str:
    """Return an illustrative few-shot example from the output template."""
    template = template if template is not None else load_output_template()
    example = template.get("output_format", {}).get("example") or []
    if not example:
        return ""
    return "```json\n" + json.dumps(example[:1], indent=2) + "\n```"


def build_factor_prompt(
    request: str,
    num_factors: int,
    operators_list: str,
    template_block: str,
    feedback: str | None = None,
    example_block: str = "",
) -> str:
    """Assemble the user prompt for the factor generator LLM."""
    feedback_section = f"\n\nPREVIOUS FEEDBACK:\n{feedback}\n" if feedback else ""
    example_section = (
        f"\nFor reference, here is an example of one valid factor:\n{example_block}\n"
        if example_block
        else ""
    )
    return f"""You are a senior quantitative researcher. Generate {num_factors} stock selection factors.

REQUEST: {request}
{feedback_section}
DATA: Open, Close, High, Low, Volume

OPERATORS:
{operators_list}
{example_section}
Fill in this exact template at the END of your reply (inside a ```json block):

{template_block}

Generate {num_factors} factors now."""


async def generate_factor_json(
    llm,
    request: str,
    num_factors: int,
    operators: list[dict],
    feedback: str | None = None,
) -> str:
    """
    Call the factor LLM and return its raw response text.

    The response is expected to contain a JSON array of factor objects (often
    inside a ```json fence at the end of a reasoning trace). Downstream code
    is responsible for extracting the JSON.
    """
    template = load_output_template()
    prompt = build_factor_prompt(
        request,
        num_factors,
        format_operators_for_prompt(operators),
        build_factor_template(num_factors, template),
        feedback,
        build_factor_example(template),
    )

    # Disable Nemotron's chain-of-thought trace so the entire token budget
    # goes toward producing the JSON answer (not reasoning prose). Without
    # this, small-context responses can be truncated mid-formula.
    response = await llm.ainvoke(
        [
            SystemMessage(content=NO_THINK_INSTRUCTION),
            HumanMessage(content=prompt),
        ]
    )
    content = extract_response_text(response)

    if not content.strip():
        extras = list((getattr(response, "additional_kwargs", {}) or {}).keys())
        logger.warning(
            "Factor generator returned empty .content. Increase max_tokens "
            f"for the factor_generator LLM. additional_kwargs={extras}"
        )

    logger.debug(f"Factor generator output: {len(content)} chars")
    return content


# =============================================================================
# NAT-registered functions
# =============================================================================


class FactorGeneratorConfig(FunctionBaseConfig, name="factor_generator"):
    """Generate quantitative factors using the calculator operators."""

    num_factors: int = Field(default=3, description="Number of factors to generate.")
    llm_name: str | None = Field(
        default=None,
        description="LLM to use for generation. If None, returns the prompt for the agent to process.",
    )


@register_function(config_type=FactorGeneratorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def factor_generator_function(config: FactorGeneratorConfig, builder: Builder):
    """NAT function wrapper around ``generate_factor_json``."""
    operators = load_calculator_operators()

    llm = None
    if config.llm_name:
        llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def generate_factors(request: str) -> str:
        """
        Generate quantitative factors for stock selection.

        Args:
            request: What kind of factors to generate. e.g. "momentum factors",
                     "volatility factors", "volume-price divergence factors".

        Returns:
            JSON array (as a string) of factor objects with name, formula,
            meaning, data_fields_used, operators_used.
        """
        if llm:
            return await generate_factor_json(llm, request, config.num_factors, operators)

        template = load_output_template()
        prompt = build_factor_prompt(
            request,
            config.num_factors,
            format_operators_for_prompt(operators),
            build_factor_template(config.num_factors, template),
            example_block=build_factor_example(template),
        )
        return f"Please generate factors based on this specification:\n\n{prompt}"

    yield FunctionInfo.from_fn(
        generate_factors,
        description=(
            "Generate quantitative stock selection factors in JSON format. "
            "Input: description of factors needed (e.g., 'momentum factors')."
        ),
    )


class FactorValidatorConfig(FunctionBaseConfig, name="factor_validator"):
    """Validate factor formulas against calculator.json operators."""

    strict_mode: bool = Field(default=True, description="Reject factors with unknown operators.")


@register_function(config_type=FactorValidatorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def factor_validator_function(config: FactorValidatorConfig, builder: Builder):
    """Validate factor formulas to ensure they use known operators."""
    operators = load_calculator_operators()
    valid_operators = {op["name"] for op in operators}

    async def validate_factor(formula: str) -> str:
        """
        Validate a factor formula.

        Example: ``Div(TS_Return(Close, 20), TS_Std(Close, 20))``
        """
        matches = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", formula)
        unknown = [m for m in matches if m not in valid_operators and m not in VALID_DATA_FIELDS]
        valid_used = [m for m in matches if m in valid_operators]
        if unknown and config.strict_mode:
            return f"INVALID: Unknown operators: {', '.join(unknown)}"
        return f"VALID: Uses operators: {', '.join(set(valid_used))}"

    yield FunctionInfo.from_fn(
        validate_factor,
        description="Validate a factor formula against available operators.",
    )


class ListOperatorsConfig(FunctionBaseConfig, name="list_operators"):
    """List available calculator operators."""

    category: str | None = Field(
        default=None,
        description="Filter by category prefix (e.g., 'TS_', 'CS_').",
    )


@register_function(config_type=ListOperatorsConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def list_operators_function(config: ListOperatorsConfig, builder: Builder):
    """List available operators for factor construction."""
    operators = load_calculator_operators()

    async def list_operators(category: str | None = None) -> str:
        """List available operators, optionally filtered by name prefix."""
        cat = category if category is not None else config.category
        filtered = operators
        if cat:
            filtered = [op for op in operators if op["name"].startswith(cat)]
        if not filtered:
            return f"No operators found for category: {cat}"

        result = [f"Available Operators ({len(filtered)} total):\n"]
        for op in filtered[:25]:
            result.append(f"- {op['name']}: {op['meanings'][:80]}...")
        if len(filtered) > 25:
            result.append(f"\n... and {len(filtered) - 25} more")
        return "\n".join(result)

    yield FunctionInfo.from_fn(
        list_operators,
        description="List available calculator operators for factor construction.",
    )
