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
Factor Code Generator Component for Factor Mining Workflow.

This component generates executable Python code from factor descriptions
using the operators defined in template/calculator.json.
"""

import json
import logging
import re
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


def load_calculator_operators() -> list[dict]:
    """Load calculator operators from the template JSON file."""
    if not CALCULATOR_JSON.exists():
        logger.warning(f"Calculator template not found at {CALCULATOR_JSON}")
        return []

    with open(CALCULATOR_JSON, "r") as f:
        return json.load(f)


def get_operator_code_map(operators: list[dict]) -> dict[str, str]:
    """Create a mapping of operator names to their code implementations."""
    return {op['name']: op['code'] for op in operators}


def get_required_operator_codes(formula: str, operators: list[dict]) -> str:
    """Extract the code for operators used in a formula."""
    pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\('
    used_ops = set(re.findall(pattern, formula))

    code_map = get_operator_code_map(operators)
    codes = []

    for op_name in used_ops:
        if op_name in code_map:
            codes.append(code_map[op_name])

    return "\n".join(codes)


def extract_operators_from_json(factor_json: str, code_map: dict[str, str]) -> tuple[set[str], str]:
    """
    Parse factor JSON and extract the required operator codes.
    
    Returns:
        Tuple of (set of operator names, concatenated operator code string)
    """
    required_ops = set()
    
    # Try to parse as JSON to get operators_used field
    try:
        # Handle both array and single object
        data = json.loads(factor_json)
        if isinstance(data, list):
            for factor in data:
                if 'operators_used' in factor:
                    required_ops.update(factor['operators_used'])
                # Also extract from formula in case operators_used is incomplete
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
        # Fallback: extract function names from the text
        pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\('
        required_ops.update(re.findall(pattern, factor_json))
    
    # Filter to only operators that exist in our code_map
    valid_ops = {op for op in required_ops if op in code_map}
    
    # Build the operator code string
    operator_codes = []
    for op_name in sorted(valid_ops):  # Sort for consistent output
        operator_codes.append(code_map[op_name])
    
    return valid_ops, "\n".join(operator_codes)


class FactorCodeGeneratorConfig(FunctionBaseConfig, name="factor_code_generator"):
    """
    Factor Code Generator: Generates executable Python code from factor descriptions.

    Takes factor formulas and generates Python functions that can be executed
    on pandas DataFrames using the operators from calculator.json.
    """

    llm_name: str | None = Field(
        default=None,
        description="LLM to use for code generation."
    )
    include_imports: bool = Field(
        default=True,
        description="Whether to include import statements in the generated code."
    )
    include_operator_functions: bool = Field(
        default=True,
        description="Whether to include the operator function definitions."
    )


@register_function(config_type=FactorCodeGeneratorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def factor_code_generator_function(config: FactorCodeGeneratorConfig, builder: Builder):
    """
    Generates executable Python code from factor descriptions using calculator operators.
    """

    # Load calculator operators with their code
    operators = load_calculator_operators()
    code_map = get_operator_code_map(operators)

    # Get LLM if specified
    llm = None
    if config.llm_name:
        llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def generate_factor_code(factor_json: str) -> str:
        """
        Generate executable Python code for factors.

        Takes the JSON output from factor_generator and generates
        executable Python functions that calculate the factors using pandas DataFrames.

        Args:
            factor_json: JSON array from factor_generator containing factors with fields:
                        - name: Factor name (e.g., "Momentum_20d")
                        - formula: Formula using operators (e.g., "Rank(TS_Return(Close, 20))")
                        - meaning: Economic intuition
                        - category: momentum|volatility|volume|reversal|quality|other
                        - data_fields_used: ["Close", "Volume", ...]
                        - operators_used: ["Rank", "TS_Return", ...]
                        - lookback_periods: [20, 60, ...]

        Returns:
            Executable Python code as a string, including:
            - Import statements (pandas, numpy)
            - Required operator functions from calculator.json
            - The main factor calculation function(s)
        """
        # Extract required operators and their code from the factor JSON
        required_ops, operator_code_block = extract_operators_from_json(factor_json, code_map)
        
        logger.info(f"Extracted {len(required_ops)} operators from factor JSON: {required_ops}")

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
    # Use the operator functions directly
    result = ...
    return result
```"""

        user_prompt = f"""Write ONLY the main factor function(s) for this JSON (operator functions are already defined):

{factor_json}

The JSON contains factors with these fields:
- name: The factor function name to use
- formula: The calculation formula using operator functions
- meaning: The economic meaning (use as docstring)
- data_fields_used: Which data inputs are needed (Open, Close, High, Low, Volume)

AVAILABLE OPERATORS (already defined, just call them):
{', '.join(sorted(required_ops))}

REQUIREMENTS:
1. Create ONLY the factor function(s) - no imports, no operator definitions
2. Function name should match the factor "name" field (use snake_case)
3. Takes only the data fields listed in "data_fields_used" as pd.DataFrame inputs
4. Implements the "formula" using the operator functions
5. Returns the result as pd.DataFrame
6. Add docstring from "meaning" field
7. You can check ./src/factor_mining_workflow/template/calculator.json 
   for the operator functions and their signatures.

Generate ONLY the factor function(s):"""

        if llm:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = await llm.ainvoke(messages)
            factor_function_code = response.content if hasattr(response, 'content') else str(response)
        else:
            factor_function_code = f"# Please generate factor function for:\n# {factor_json}"

        # Extract code from markdown code blocks if present
        code_blocks = re.findall(r'```python\n(.*?)```', factor_function_code, re.DOTALL)
        if code_blocks:
            factor_function_code = code_blocks[0]

        # Build the complete output with imports, operators, and factor function
        output_parts = []
        
        # Add imports
        if config.include_imports:
            output_parts.append("import pandas as pd\nimport numpy as np\n")
        
        # Add operator function definitions
        if config.include_operator_functions and operator_code_block:
            output_parts.append("# Required operator functions from calculator.json")
            output_parts.append(operator_code_block)
            output_parts.append("")  # Empty line separator
        
        # Add the factor function(s) from LLM
        output_parts.append("# Factor calculation function(s)")
        output_parts.append(factor_function_code)
        
        return "\n".join(output_parts)

    yield FunctionInfo.from_fn(
        generate_factor_code,
        description="Generate executable Python code from factor_generator JSON output containing factor name, formula, meaning, operators_used, and data_fields_used."
    )

