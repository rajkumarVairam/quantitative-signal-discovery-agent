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
    """Format calculator operators into a prompt-friendly string."""
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
        formatted.append(f"- {op['name']}: {op['meanings']}")
    return "\n".join(formatted)


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


def get_operator_code_map(operators: list[dict]) -> dict[str, str]:
    """Create a mapping of operator names to their code implementations."""
    return {op['name']: op['code'] for op in operators}


def get_required_operator_codes(formula: str, operators: list[dict]) -> str:
    """Extract the code for operators used in a formula."""
    import re
    pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\('
    used_ops = set(re.findall(pattern, formula))

    code_map = get_operator_code_map(operators)
    codes = []

    for op_name in used_ops:
        if op_name in code_map:
            codes.append(code_map[op_name])

    return "\n".join(codes)


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


def extract_operators_from_json(factor_json: str, code_map: dict[str, str]) -> tuple[set[str], str]:
    """
    Parse factor JSON and extract the required operator codes.
    
    Returns:
        Tuple of (set of operator names, concatenated operator code string)
    """
    import re
    
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
        import re
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
