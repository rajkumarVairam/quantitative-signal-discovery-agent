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
Rank IC Evaluator Component for Factor Mining Workflow.

This component evaluates the rank IC (Information Coefficient) of generated factors
by computing the Spearman correlation between factor values and forward stock returns.
"""

import json
import logging
import re
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field
from scipy import stats

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

# Path to data directory
DATA_DIR = Path(__file__).parent / "data" / "sp500"


def load_stock_data() -> dict[str, pd.DataFrame]:
    """Load all available stock price-volume data from CSV files."""
    data = {}
    data_files = ['Open', 'Close', 'High', 'Low', 'Volume']

    for field in data_files:
        file_path = DATA_DIR / f"{field}.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                data[field] = df
                logger.info(f"Loaded {field}.csv with shape {df.shape}")
            except Exception as e:
                logger.warning(f"Failed to load {field}.csv: {e}")
        else:
            logger.warning(f"Data file not found: {file_path}")

    return data


def compute_forward_returns(close: pd.DataFrame, periods: int = 5) -> pd.DataFrame:
    """
    Compute forward returns for the next N periods.

    Args:
        close: DataFrame of closing prices (rows=dates, cols=stocks).
        periods: Number of forward periods for return calculation.

    Returns:
        DataFrame of forward returns with the same shape as input.
    """
    forward_returns = close.shift(-periods) / close - 1
    return forward_returns


def compute_rank_ic(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute rank IC (Information Coefficient) between factor values and forward returns.

    The rank IC is the Spearman correlation between factor ranks and return ranks
    computed cross-sectionally for each date, then aggregated.

    Args:
        factor_values: DataFrame of factor values (rows=dates, cols=stocks).
        forward_returns: DataFrame of forward returns (rows=dates, cols=stocks).

    Returns:
        Dictionary containing IC statistics.
    """
    # Align the dataframes
    common_dates = factor_values.index.intersection(forward_returns.index)
    common_stocks = factor_values.columns.intersection(forward_returns.columns)

    if len(common_dates) == 0 or len(common_stocks) == 0:
        return {
            "mean_ic": None,
            "ic_std": None,
            "ic_ir": None,
            "t_stat": None,
            "p_value": None,
            "num_periods": 0,
            "error": "No common dates or stocks between factor and returns",
        }

    factor_aligned = factor_values.loc[common_dates, common_stocks]
    returns_aligned = forward_returns.loc[common_dates, common_stocks]

    # Compute rank IC for each date (cross-sectional correlation)
    ic_series = []
    for date in common_dates:
        factor_row = factor_aligned.loc[date].dropna()
        returns_row = returns_aligned.loc[date].dropna()

        # Get common stocks with valid values
        common = factor_row.index.intersection(returns_row.index)
        if len(common) < 10:  # Need minimum number of stocks
            continue

        factor_vals = factor_row[common].values
        return_vals = returns_row[common].values

        # Compute Spearman correlation (rank IC)
        try:
            correlation, _ = stats.spearmanr(factor_vals, return_vals)
            if not np.isnan(correlation):
                ic_series.append(correlation)
        except Exception:
            continue

    if len(ic_series) == 0:
        return {
            "mean_ic": None,
            "ic_std": None,
            "ic_ir": None,
            "t_stat": None,
            "p_value": None,
            "num_periods": 0,
            "error": "Could not compute IC for any period",
        }

    ic_array = np.array(ic_series)
    mean_ic = float(np.mean(ic_array))
    ic_std = float(np.std(ic_array))
    num_periods = len(ic_series)

    # IC IR (Information Ratio) = mean_ic / ic_std
    ic_ir = mean_ic / ic_std if ic_std > 0 else None

    # T-statistic for testing if mean IC is significantly different from 0
    t_stat = mean_ic / (ic_std / np.sqrt(num_periods)) if ic_std > 0 else None
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=num_periods - 1))) if t_stat else None

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "t_stat": t_stat,
        "p_value": p_value,
        "num_periods": num_periods,
        "positive_ic_ratio": float(np.mean(ic_array > 0)),
        "ic_percentiles": {
            "5th": float(np.percentile(ic_array, 5)),
            "25th": float(np.percentile(ic_array, 25)),
            "50th": float(np.percentile(ic_array, 50)),
            "75th": float(np.percentile(ic_array, 75)),
            "95th": float(np.percentile(ic_array, 95)),
        },
    }


def extract_code_from_response(code_response: str) -> str:
    """Extract Python code from markdown code blocks or raw text."""
    # Try to extract from markdown code blocks
    code_blocks = re.findall(r'```python\n(.*?)```', code_response, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks)

    # Try generic code blocks
    code_blocks = re.findall(r'```\n(.*?)```', code_response, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks)

    # Return as-is (might be raw code)
    return code_response


# Known operator function names (not factor functions)
OPERATOR_NAMES = {
    'TS_Return', 'TS_Std', 'TS_Mean', 'TS_Max', 'TS_Min', 'TS_Rank', 'TS_Sum',
    'TS_Corr', 'TS_Cov', 'TS_Delta', 'TS_Delay', 'TS_Product', 'TS_Skewness',
    'TS_Kurtosis', 'TS_Argmax', 'TS_Argmin', 'TS_WMA',
    'Decay_Linear', 'Decay_Exp', 'EMA',
    'CS_Rank', 'CS_Zscore', 'CS_Demean', 'CS_Scale',
    'Rank', 'Add', 'Sub', 'Mul', 'Div', 'Abs', 'Sign', 'Log', 'Power', 'Sqrt',
    'Max', 'Min', 'Neg', 'Sigmoid', 'Tanh', 'Clip',
}


def execute_factor_code(code: str, stock_data: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """
    Execute factor code and return the factor values.

    Args:
        code: Python code string containing factor function(s).
        stock_data: Dictionary of stock data DataFrames.

    Returns:
        DataFrame of factor values, or None if execution fails.
    """
    # Create execution namespace with required modules and data
    namespace = {
        'pd': pd,
        'np': np,
        'Open': stock_data.get('Open'),
        'Close': stock_data.get('Close'),
        'High': stock_data.get('High'),
        'Low': stock_data.get('Low'),
        'Volume': stock_data.get('Volume'),
    }

    try:
        # Execute the code to define functions
        exec(code, namespace)

        # Find factor functions (excluding operators and builtins)
        factor_functions = []
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith('_') and name not in ['pd', 'np']:
                if hasattr(obj, '__code__'):
                    # Skip known operator functions
                    if name not in OPERATOR_NAMES:
                        factor_functions.append((name, obj))

        if not factor_functions:
            logger.warning("No factor functions found in the code")
            return None

        logger.info(f"Found {len(factor_functions)} factor function(s): {[f[0] for f in factor_functions]}")

        # Execute all factor functions and find the best one by absolute IC
        best_result = None
        best_ic = None
        best_name = None
        
        for func_name, factor_func in factor_functions:
            try:
                # Determine which data fields the function needs
                import inspect
                sig = inspect.signature(factor_func)
                params = list(sig.parameters.keys())

                # Build kwargs based on function parameters
                kwargs = {}
                for param in params:
                    if param in stock_data:
                        kwargs[param] = stock_data[param]

                if kwargs:
                    result = factor_func(**kwargs)
                else:
                    # Try calling with common data fields
                    result = factor_func(
                        Open=stock_data.get('Open'),
                        Close=stock_data.get('Close'),
                        High=stock_data.get('High'),
                        Low=stock_data.get('Low'),
                        Volume=stock_data.get('Volume'),
                    )

                if isinstance(result, pd.Series):
                    result = result.to_frame()
                
                if isinstance(result, pd.DataFrame):
                    # Compute IC on ALL dates (not just a sample) for accurate selection
                    forward_ret = stock_data['Close'].shift(-5) / stock_data['Close'] - 1
                    valid_dates = result.dropna(how='all').index
                    sample_ics = []
                    
                    for date in valid_dates:
                        if date in forward_ret.index:
                            fr = result.loc[date].dropna()
                            rr = forward_ret.loc[date].dropna()
                            common = fr.index.intersection(rr.index)
                            if len(common) >= 10:
                                try:
                                    from scipy import stats as sp_stats
                                    corr, _ = sp_stats.spearmanr(fr[common], rr[common])
                                    if not np.isnan(corr):
                                        sample_ics.append(corr)
                                except:
                                    pass
                    
                    if sample_ics:
                        mean_ic = abs(np.mean(sample_ics))  # Use |mean IC| for selection
                        logger.info(f"  {func_name}: |IC| = {mean_ic:.4f}")
                        
                        if best_ic is None or mean_ic > best_ic:
                            best_ic = mean_ic
                            best_result = result
                            best_name = func_name
                    else:
                        # No IC computed, but still a valid result
                        if best_result is None:
                            best_result = result
                            best_name = func_name

            except Exception as e:
                logger.warning(f"Error executing {func_name}: {e}")
                continue

        if best_result is not None:
            ic_str = f"{best_ic:.4f}" if best_ic is not None else "N/A"
            logger.info(f"Selected best factor: {best_name} with |IC| = {ic_str}")
            return best_result
        
        return None

    except Exception as e:
        logger.error(f"Error executing factor code: {e}")
        logger.debug(traceback.format_exc())
        return None


class RankICEvaluatorConfig(FunctionBaseConfig, name="factor_evaluator"):
    """
    Rank IC Evaluator: Evaluates factor performance using rank IC.

    Computes the Spearman correlation between factor values and forward stock returns
    to measure the predictive power of generated factors.
    """

    forward_periods: int = Field(
        default=5,
        description="Number of forward periods for return calculation (e.g., 5 for weekly returns).",
    )
    min_periods: int = Field(
        default=20,
        description="Minimum number of periods required for IC calculation.",
    )


@register_function(config_type=RankICEvaluatorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def rank_ic_evaluator_function(config: RankICEvaluatorConfig, builder: Builder):
    """
    Rank IC Evaluator that measures factor predictive power.

    Takes generated factor code, executes it on stock data, and computes
    the rank IC (Spearman correlation with forward returns).
    """

    # Load stock data once at initialization
    stock_data = load_stock_data()

    if not stock_data:
        logger.error("No stock data available for evaluation")

    async def evaluate_rank_ic(factor_code: str) -> str:
        """
        Evaluate the rank IC of a factor.

        Takes the Python code output from factor_code_generator and:
        1. Executes the code to compute factor values
        2. Computes forward returns
        3. Calculates rank IC (Spearman correlation)
        4. Returns evaluation metrics

        Args:
            factor_code: Python code string from factor_code_generator containing
                        import statements, operator functions, and factor function(s).

        Returns:
            JSON string containing evaluation results:
            - mean_ic: Average rank IC across all periods
            - ic_std: Standard deviation of IC
            - ic_ir: Information Ratio (mean_ic / ic_std)
            - t_stat: T-statistic for significance testing
            - p_value: P-value for the t-test
            - num_periods: Number of periods evaluated
            - positive_ic_ratio: Proportion of periods with positive IC
            - ic_percentiles: Distribution of IC values
            - interpretation: Human-readable interpretation of results
        """
        if not stock_data:
            return json.dumps({
                "error": "No stock data available",
                "status": "failed",
            }, indent=2)

        # Extract clean code
        clean_code = extract_code_from_response(factor_code)

        # Execute factor code to get factor values
        factor_values = execute_factor_code(clean_code, stock_data)

        if factor_values is None:
            return json.dumps({
                "error": "Failed to execute factor code",
                "status": "failed",
                "code_preview": clean_code[:500] + "..." if len(clean_code) > 500 else clean_code,
            }, indent=2)

        # Compute forward returns
        close_data = stock_data.get('Close')
        if close_data is None:
            return json.dumps({
                "error": "Close price data not available",
                "status": "failed",
            }, indent=2)

        forward_returns = compute_forward_returns(close_data, periods=config.forward_periods)

        # Compute rank IC
        ic_results = compute_rank_ic(factor_values, forward_returns)

        if ic_results.get("error"):
            return json.dumps({
                "error": ic_results["error"],
                "status": "failed",
            }, indent=2)

        # Add interpretation
        mean_ic = ic_results.get("mean_ic", 0)
        p_value = ic_results.get("p_value", 1)

        if mean_ic is None:
            interpretation = "Could not compute IC"
        elif abs(mean_ic) < 0.01:
            interpretation = "Very weak predictive power (IC close to 0)"
        elif abs(mean_ic) < 0.03:
            interpretation = "Weak predictive power"
        elif abs(mean_ic) < 0.05:
            interpretation = "Moderate predictive power"
        elif abs(mean_ic) < 0.1:
            interpretation = "Good predictive power"
        else:
            interpretation = "Strong predictive power"

        if p_value and p_value < 0.05:
            interpretation += " (statistically significant at 5% level)"
        elif p_value and p_value < 0.1:
            interpretation += " (marginally significant at 10% level)"
        else:
            interpretation += " (not statistically significant)"

        if mean_ic and mean_ic < 0:
            interpretation += ". Negative IC suggests inverse relationship with returns."

        ic_results["interpretation"] = interpretation
        ic_results["status"] = "success"
        ic_results["forward_periods"] = config.forward_periods
        ic_results["factor_shape"] = list(factor_values.shape)

        return json.dumps(ic_results, indent=2)

    yield FunctionInfo.from_fn(
        evaluate_rank_ic,
        description=(
            "Evaluate the rank IC (Information Coefficient) of a factor. "
            "Takes Python code from factor_code_generator and returns IC metrics "
            "measuring the Spearman correlation between factor values and forward returns."
        ),
    )
