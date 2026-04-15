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

Orchestrates the factor mining loop by composing:
- factor_generator: Factor description generation
- factor_code_generator: Code generation
- factor_evaluator: IC evaluation utilities

Supports multiple LLMs for different agents.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

# Import from agent modules (single source of truth)
from .factor_generator import (
    load_calculator_operators,
    format_operators_for_prompt,
    get_output_format_prompt,
)
from .factor_evaluator import (
    load_stock_data,
    compute_forward_returns,
    compute_rank_ic,
    extract_code_from_response,
    execute_factor_code,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"


def save_factor_results(factor_json: str, factor_code: str, ic_results: dict, iteration: int) -> str:
    """Save factor results to file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = OUTPUT_DIR / f"factor_{timestamp}_iter{iteration}.json"
    
    metrics = {k: v for k, v in ic_results.items() if k != "selected_factor"}
    
    with open(filepath, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "iteration": iteration,
            "selected_factor": ic_results.get("selected_factor"),
            "factor_description": factor_json,
            "factor_code": factor_code,
            "evaluation_metrics": metrics,
        }, f, indent=2)
    
    logger.info(f"Saved factor results to {filepath}")
    return str(filepath)


class FactorOptimizerConfig(FunctionBaseConfig, name="factor_optimizer"):
    """Factor Optimizer: Iteratively generates and optimizes quantitative factors."""

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


@register_function(config_type=FactorOptimizerConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def factor_optimizer_function(config: FactorOptimizerConfig, builder: Builder):
    """Factor Optimizer that composes generation, coding, and evaluation agents."""
    
    # Resolve LLM names
    factor_llm_name = config.factor_generator_llm or config.llm_name
    code_llm_name = config.code_generator_llm or config.llm_name
    advisor_llm_name = config.optimization_advisor_llm or config.llm_name
    
    if not factor_llm_name:
        raise ValueError("Must specify llm_name or factor_generator_llm")
    
    # Get LLMs
    factor_llm = await builder.get_llm(llm_name=factor_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    code_llm = await builder.get_llm(llm_name=code_llm_name or factor_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    advisor_llm = await builder.get_llm(llm_name=advisor_llm_name or factor_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    logger.info(f"LLMs: factor={factor_llm_name}, code={code_llm_name}, advisor={advisor_llm_name}")
    
    # Load resources
    operators = load_calculator_operators()
    operators_list = format_operators_for_prompt(operators)
    output_format = get_output_format_prompt().replace("{num_factors}", str(config.num_factors))
    stock_data = load_stock_data()
    code_map = {op['name']: op['code'] for op in operators}
    sig_map = {op['name']: op.get('signature', op['name']) for op in operators}

    async def generate_factors(request: str, feedback: str | None = None) -> str:
        """Generate factor descriptions using factor_generator LLM."""
        feedback_section = f"\n\nPREVIOUS FEEDBACK:\n{feedback}\n" if feedback else ""
        prompt = f"""You are a senior quantitative researcher. Generate {config.num_factors} stock selection factors.

REQUEST: {request}
{feedback_section}
DATA: Open, Close, High, Low, Volume

OPERATORS:
{operators_list}

{output_format}

Generate {config.num_factors} factors:"""
        
        response = await factor_llm.ainvoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    async def generate_code(factor_json: str) -> str:
        """Generate Python code using code_generator LLM."""
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Extract required operators
        required_ops = set()
        try:
            data = json.loads(factor_json)
            for factor in (data if isinstance(data, list) else [data]):
                if 'operators_used' in factor:
                    required_ops.update(factor['operators_used'])
                if 'formula' in factor:
                    required_ops.update(re.findall(r'\b([A-Za-z_]\w*)\s*\(', factor['formula']))
        except json.JSONDecodeError:
            required_ops.update(re.findall(r'\b([A-Za-z_]\w*)\s*\(', factor_json))
        
        valid_ops = {op for op in required_ops if op in code_map}
        operator_code = "\n".join(code_map[op] for op in sorted(valid_ops))
        sigs = "\n".join(f"- {sig_map.get(op, op)}" for op in sorted(valid_ops))
        
        system = """You are a Python expert generating factor calculation functions.

CRITICAL RULES:
1. Each function MUST accept DataFrames as parameters: Open, Close, High, Low, Volume
2. Each function MUST return a pd.DataFrame (rows=dates, cols=stocks)
3. Use ONLY the operator functions provided - they are already defined
4. DO NOT redefine operators - just call them

CORRECT FUNCTION FORMAT:
```python
def factor_momentum(Close: pd.DataFrame) -> pd.DataFrame:
    '''20-day momentum factor'''
    return TS_Return(Close, 20)

def factor_volume_adjusted(Close: pd.DataFrame, Volume: pd.DataFrame) -> pd.DataFrame:
    '''Volume-adjusted momentum'''
    return Div(TS_Return(Close, 20), TS_Mean(Volume, 20))
```

WRONG (DO NOT DO THIS):
- def factor(): return 0.5  # No DataFrame input/output
- return Close.mean()  # Returns scalar, not DataFrame
- def TS_Return(...):  # Don't redefine operators"""

        user = f"""FACTOR JSON:
{factor_json}

AVAILABLE OPERATORS (already defined, just call them):
{sigs}

Write factor functions. Each must:
- Accept only the data fields it needs (Close, Volume, etc.) as pd.DataFrame
- Return a pd.DataFrame with the same shape (dates x stocks)
- Use the operators above to compute the factor"""
        
        response = await code_llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
        code = response.content if hasattr(response, 'content') else str(response)
        
        # Extract from markdown
        blocks = re.findall(r'```python\n(.*?)```', code, re.DOTALL)
        if blocks:
            code = blocks[0]
        
        return f"import pandas as pd\nimport numpy as np\n\n{operator_code}\n\n{code}"

    def evaluate_ic(factor_code: str) -> dict[str, Any]:
        """Evaluate rank IC using factor_evaluator utilities."""
        if not stock_data:
            return {"error": "No stock data", "mean_ic": None}
        
        clean_code = extract_code_from_response(factor_code)
        exec_result = execute_factor_code(clean_code, stock_data)
        
        if exec_result is None:
            return {"error": "Code execution failed", "mean_ic": None}
        
        factor_values, selected_factor = exec_result
        
        close = stock_data.get('Close')
        if close is None:
            return {"error": "No Close data", "mean_ic": None}
        
        forward_returns = compute_forward_returns(close, periods=config.forward_periods)
        ic_results = compute_rank_ic(factor_values, forward_returns)
        ic_results["selected_factor"] = selected_factor
        return ic_results

    def is_acceptable(ic_results: dict) -> bool:
        """Check if factor meets thresholds."""
        mean_ic = ic_results.get("mean_ic")
        p_value = ic_results.get("p_value")
        if mean_ic is None:
            return False
        if abs(mean_ic) < config.ic_threshold:
            return False
        if p_value is not None and p_value > config.p_value_threshold:
            return False
        return True

    async def generate_feedback(factor_json: str, ic_results: dict, iteration: int) -> str:
        """Generate optimization advice using advisor LLM."""
        from langchain_core.messages import HumanMessage, SystemMessage
        
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
        
        response = await advisor_llm.ainvoke([
            SystemMessage(content="You are a senior quant providing factor optimization advice."),
            HumanMessage(content=prompt)
        ])
        return response.content if hasattr(response, 'content') else str(response)

    async def run_optimization(request: str) -> str:
        """Run the optimization loop."""
        best_result = None
        best_ic = None
        feedback = None
        
        for iteration in range(1, config.max_iterations + 1):
            logger.info(f"=== Iteration {iteration}/{config.max_iterations} ===")
            
            # Step 1: Generate factors
            logger.info("Generating factors...")
            factor_json = await generate_factors(request, feedback)
            
            # Step 2: Generate code
            logger.info("Generating code...")
            factor_code = await generate_code(factor_json)
            
            # Step 3: Evaluate
            logger.info("Evaluating IC...")
            ic_results = evaluate_ic(factor_code)
            mean_ic = ic_results.get("mean_ic")
            
            if mean_ic is not None:
                logger.info(f"Mean IC: {mean_ic:.4f}, p-value: {ic_results.get('p_value', 'N/A')}")
                if best_ic is None or abs(mean_ic) > abs(best_ic):
                    best_ic = mean_ic
                    best_result = {"factor_json": factor_json, "factor_code": factor_code, "ic_results": ic_results, "iteration": iteration}
            
            # Step 4: Accept or feedback
            if is_acceptable(ic_results):
                logger.info("Factor accepted!")
                saved_path = save_factor_results(factor_json, factor_code, ic_results, iteration) if config.save_results else None
                return json.dumps({
                    "status": "accepted",
                    "iteration": iteration,
                    "mean_ic": mean_ic,
                    "p_value": ic_results.get("p_value"),
                    "saved_path": saved_path,
                }, indent=2)
            
            # Generate feedback for next iteration
            logger.info("Generating optimization feedback...")
            feedback = await generate_feedback(factor_json, ic_results, iteration)
        
        # Return best result
        if best_result:
            saved_path = save_factor_results(
                best_result["factor_json"], best_result["factor_code"], 
                best_result["ic_results"], best_result["iteration"]
            ) if config.save_results else None
            return json.dumps({
                "status": "best_effort",
                "iteration": best_result["iteration"],
                "mean_ic": best_ic,
                "saved_path": saved_path,
            }, indent=2)
        
        return json.dumps({"status": "failed", "iterations": config.max_iterations}, indent=2)

    yield FunctionInfo.from_fn(
        run_optimization,
        description="Run factor mining optimization loop: generate → code → evaluate → feedback.",
    )
