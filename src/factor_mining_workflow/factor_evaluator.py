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
Factor Evaluator Component for Factor Mining Workflow.

This module provides:
1. Rank IC (Information Coefficient) calculation utilities
2. Factor code execution and evaluation
3. NAT functions for factor evaluation and optimization

The rank IC measures the Spearman correlation between factor values and forward
stock returns, which indicates the predictive power of generated factors.
"""

import inspect
import json
import logging
import re
import traceback
from datetime import datetime
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

# Path to operator definitions
CALCULATOR_PATH = Path(__file__).parent / "template" / "calculator.json"


def _load_all_operators() -> dict[str, callable]:
    """Load and compile all operator functions from calculator.json into a callable dict."""
    if not CALCULATOR_PATH.exists():
        return {}
    with open(CALCULATOR_PATH, "r") as f:
        operators = json.load(f)
    op_namespace = {"pd": pd, "np": np}
    for op in operators:
        try:
            exec(op["code"], op_namespace)
        except Exception as e:
            logger.warning(f"Failed to compile operator {op.get('name')}: {e}")
    return {
        name: obj
        for name, obj in op_namespace.items()
        if callable(obj) and name not in ("pd", "np")
    }


_OPERATOR_FUNCTIONS = _load_all_operators()


# =============================================================================
# Core Evaluation Utilities (from rank_ic_evaluator)
# =============================================================================


def load_stock_data() -> dict[str, pd.DataFrame]:
    """Load all available stock price-volume data from CSV files."""
    data = {}
    data_files = ["Open", "Close", "High", "Low", "Volume"]

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

    import warnings

    ic_series = []
    for date in common_dates:
        factor_row = factor_aligned.loc[date].dropna()
        returns_row = returns_aligned.loc[date].dropna()

        common = factor_row.index.intersection(returns_row.index)
        if len(common) < 10:
            continue

        factor_vals = factor_row[common].values
        return_vals = returns_row[common].values

        # Spearman correlation requires non-constant inputs
        if np.std(factor_vals) < 1e-10 or np.std(return_vals) < 1e-10:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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

    ic_ir = mean_ic / ic_std if ic_std > 0 else None
    t_stat = mean_ic / (ic_std / np.sqrt(num_periods)) if ic_std > 0 else None
    p_value = (
        float(2 * (1 - stats.t.cdf(abs(t_stat), df=num_periods - 1)))
        if t_stat
        else None
    )

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
    """Extract Python code from markdown code blocks, falling back to raw text."""
    code_blocks = re.findall(r"```python\n(.*?)```", code_response, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks)

    code_blocks = re.findall(r"```\n(.*?)```", code_response, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks)

    return code_response


# Operator function names — derived from the actual operators loaded from
# calculator.json so this set always stays in sync with available operators.
OPERATOR_NAMES = set(_OPERATOR_FUNCTIONS.keys())

# Standard data field names. Order matters: when a factor function uses a
# generic parameter name (e.g. `x`, `data`, `prices`), parameters are filled
# positionally from this list.
STANDARD_FIELDS = ["Close", "Volume", "High", "Low", "Open"]


def _is_dataframe_param(p: inspect.Parameter) -> bool:
    """Whether this parameter expects a DataFrame to be passed in."""
    ann = p.annotation
    if ann is pd.DataFrame:
        return True
    if isinstance(ann, str) and "DataFrame" in ann:
        return True
    if ann is inspect.Parameter.empty:
        # No type hint — treat as DataFrame only if there's no scalar default.
        if p.default is inspect.Parameter.empty:
            return True
        if p.default is None or isinstance(p.default, pd.DataFrame):
            return True
        return False
    return False


def _resolve_factor_args(
    sig: inspect.Signature,
    stock_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Map factor function parameters to stock data DataFrames.

    Only DataFrame parameters are filled. Numeric/string parameters
    (lookback windows, thresholds, etc.) are left to use their defaults so
    they don't get a DataFrame mistakenly passed in.

    DataFrame parameter resolution:
      1. Exact name match (e.g. `Close` -> Close).
      2. Case-insensitive substring match in either direction
         (e.g. `closing_price` -> Close, `vol` -> Volume).
      3. Positional fallback for unmatched params, drawing from
         STANDARD_FIELDS in order, skipping fields already assigned.
    """
    available_fields = [f for f in STANDARD_FIELDS if f in stock_data]
    df_params = [name for name, p in sig.parameters.items() if _is_dataframe_param(p)]

    kwargs: dict[str, pd.DataFrame] = {}
    used_fields: set[str] = set()
    unmatched: list[str] = []

    for param in df_params:
        match: str | None = None
        p_lower = param.lower()
        for field in available_fields:
            f_lower = field.lower()
            if p_lower == f_lower or f_lower in p_lower or p_lower in f_lower:
                match = field
                break

        if match and match not in used_fields:
            kwargs[param] = stock_data[match]
            used_fields.add(match)
        else:
            unmatched.append(param)

    fallback_pool = [f for f in available_fields if f not in used_fields]
    for param in unmatched:
        if not fallback_pool:
            break
        kwargs[param] = stock_data[fallback_pool.pop(0)]

    return kwargs


def execute_factor_code(
    code: str, stock_data: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, str] | None:
    """
    Execute self-contained factor code and call its factor function(s).

    The ``code`` string is expected to be a complete, runnable Python module
    that defines its own imports, operator functions, and factor functions.
    No pre-seeded globals are injected: factor functions receive their input
    DataFrames as arguments, and operators come from definitions in the
    module itself. This makes the saved ``factor_code`` portable: it can be
    copy-pasted into any Python session and run as-is.

    Args:
        code: Python source containing imports, operator defs, and at least
              one factor function with OHLCV DataFrame parameters.
        stock_data: Dictionary of stock data DataFrames passed as the
                    arguments to each factor function.

    Returns:
        Tuple of (factor_values DataFrame, selected_factor_name), or None
        if no factor function executed successfully.
    """
    namespace: dict[str, Any] = {}

    try:
        # Normalize "smart" Unicode quotes/dashes that some LLMs emit
        # (e.g. ' ' " " - -) which would otherwise raise SyntaxError.
        code = code.translate(
            str.maketrans(
                {
                    "\u2018": "'",
                    "\u2019": "'",
                    "\u201c": '"',
                    "\u201d": '"',
                    "\u2013": "-",
                    "\u2014": "-",
                    "\u00a0": " ",
                }
            )
        )
        exec(code, namespace)

        # Factor functions are user-defined callables that aren't operators.
        # `__code__` filters out modules/builtins; OPERATOR_NAMES filters out
        # the operator definitions inlined at the top of the module.
        candidate_functions = [
            (name, obj)
            for name, obj in namespace.items()
            if not name.startswith("_")
            and callable(obj)
            and hasattr(obj, "__code__")
            and name not in OPERATOR_NAMES
        ]

        if not candidate_functions:
            logger.warning("No factor functions found in the code")
            return None

        # Detect helper functions: any function called by another candidate
        # is treated as a helper, not a factor.
        helper_names: set[str] = set()
        candidate_name_set = {name for name, _ in candidate_functions}
        for _, fn in candidate_functions:
            try:
                src = inspect.getsource(fn)
            except (OSError, TypeError):
                continue
            for name in candidate_name_set:
                if re.search(rf"\b{re.escape(name)}\s*\(", src) and name != fn.__name__:
                    helper_names.add(name)

        factor_functions = [
            (n, f) for n, f in candidate_functions if n not in helper_names
        ]
        if not factor_functions:
            factor_functions = candidate_functions

        logger.info(
            f"Found {len(factor_functions)} factor function(s): "
            f"{[f[0] for f in factor_functions]}"
            + (f" (skipping helpers: {sorted(helper_names)})" if helper_names else "")
        )

        best_result = None
        best_ic = None
        best_name = None

        import warnings
        from scipy import stats as sp_stats

        for func_name, factor_func in factor_functions:
            try:
                sig = inspect.signature(factor_func)
                kwargs = _resolve_factor_args(sig, stock_data)
                df_param_count = sum(
                    1 for p in sig.parameters.values() if _is_dataframe_param(p)
                )

                if len(kwargs) == df_param_count:
                    result = factor_func(**kwargs)
                elif len(sig.parameters) == 0:
                    result = factor_func()
                else:
                    logger.warning(
                        f"Cannot determine args for {func_name} "
                        f"(params={list(sig.parameters)}), skipping"
                    )
                    continue

                if isinstance(result, pd.Series):
                    result = result.to_frame()

                if isinstance(result, pd.DataFrame):
                    forward_ret = (
                        stock_data["Close"].shift(-5) / stock_data["Close"] - 1
                    )
                    valid_dates = result.dropna(how="all").index
                    sample_ics = []

                    for date in valid_dates:
                        if date not in forward_ret.index:
                            continue
                        fr = result.loc[date].dropna()
                        rr = forward_ret.loc[date].dropna()
                        common = fr.index.intersection(rr.index)
                        if len(common) < 10:
                            continue
                        f_vals = fr[common].values
                        r_vals = rr[common].values
                        if np.std(f_vals) < 1e-10 or np.std(r_vals) < 1e-10:
                            continue
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                corr, _ = sp_stats.spearmanr(f_vals, r_vals)
                            if not np.isnan(corr):
                                sample_ics.append(corr)
                        except Exception:
                            pass

                    if sample_ics:
                        mean_ic = abs(np.mean(sample_ics))
                        logger.info(f"  {func_name}: |IC| = {mean_ic:.4f}")
                        if best_ic is None or mean_ic > best_ic:
                            best_ic = mean_ic
                            best_result = result
                            best_name = func_name
                    elif best_result is None:
                        best_result = result
                        best_name = func_name

            except Exception as e:
                logger.warning(f"Error executing {func_name}: {e}")
                continue

        if best_result is not None:
            ic_str = f"{best_ic:.4f}" if best_ic is not None else "N/A"
            logger.info(f"Selected best factor: {best_name} with |IC| = {ic_str}")
            return best_result, best_name

        return None

    except Exception as e:
        logger.error(f"Error executing factor code: {e}")
        logger.debug(traceback.format_exc())
        return None


# =============================================================================
# Helper Functions
# =============================================================================

# Output directory for saved factors
OUTPUT_DIR = Path(__file__).parent / "output"


def ensure_output_dir() -> Path:
    """Ensure the output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def compute_factor_returns(
    factor_values: "pd.DataFrame",
    forward_returns: "pd.DataFrame",
    n_quantiles: int = 5,
    holding_period: int = 5,
) -> dict[str, Any]:
    """
    Compute backtest returns for a long-short factor strategy.

    Strategy: Long top quintile, Short bottom quintile (rebalanced every holding_period days).

    Args:
        factor_values: DataFrame of factor values (rows=dates, cols=stocks).
        forward_returns: DataFrame of forward returns.
        n_quantiles: Number of quantiles for portfolio construction (default 5 = quintiles).
        holding_period: Days between rebalancing.

    Returns:
        Dictionary with performance metrics.
    """
    import numpy as np
    import pandas as pd

    # Align the dataframes
    common_dates = factor_values.index.intersection(forward_returns.index)
    common_stocks = factor_values.columns.intersection(forward_returns.columns)

    if len(common_dates) < 252:  # Need at least 1 year of data
        return {"error": "Insufficient data for backtest", "annual_return": None}

    factor_aligned = factor_values.loc[common_dates, common_stocks]
    returns_aligned = forward_returns.loc[common_dates, common_stocks]

    # Compute daily portfolio returns
    portfolio_returns = []
    dates_used = []

    # Sample every holding_period days for rebalancing
    rebalance_dates = common_dates[::holding_period]

    for i, date in enumerate(rebalance_dates[:-1]):
        try:
            # Get factor values for this date
            factor_row = factor_aligned.loc[date].dropna()
            if len(factor_row) < n_quantiles * 2:
                continue

            # Rank stocks into quantiles
            ranks = factor_row.rank(pct=True)
            top_quintile = ranks[ranks >= (1 - 1 / n_quantiles)].index
            bottom_quintile = ranks[ranks <= 1 / n_quantiles].index

            # Get returns for next period
            next_date = rebalance_dates[i + 1]
            returns_row = returns_aligned.loc[next_date]

            # Long-short return: long top, short bottom
            long_return = returns_row[top_quintile].mean()
            short_return = returns_row[bottom_quintile].mean()

            if not np.isnan(long_return) and not np.isnan(short_return):
                ls_return = long_return - short_return
                portfolio_returns.append(ls_return)
                dates_used.append(next_date)

        except Exception:
            continue

    if len(portfolio_returns) < 10:
        return {"error": "Insufficient valid periods", "annual_return": None}

    returns_series = pd.Series(portfolio_returns, index=dates_used)

    # Calculate performance metrics
    total_return = (1 + returns_series).prod() - 1
    n_years = len(returns_series) * holding_period / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Annualized volatility
    annual_vol = returns_series.std() * np.sqrt(252 / holding_period)

    # Sharpe ratio (assuming 0 risk-free rate)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

    # Max drawdown
    cumulative = (1 + returns_series).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    max_drawdown = drawdowns.min()

    # Win rate
    win_rate = (returns_series > 0).mean()

    return {
        "annual_return": float(annual_return),
        "annual_return_pct": f"{annual_return * 100:.2f}%",
        "annual_volatility": float(annual_vol),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "max_drawdown_pct": f"{max_drawdown * 100:.2f}%",
        "total_return": float(total_return),
        "total_return_pct": f"{total_return * 100:.2f}%",
        "win_rate": float(win_rate),
        "win_rate_pct": f"{win_rate * 100:.1f}%",
        "n_periods": len(portfolio_returns),
        "n_years": round(n_years, 2),
        "strategy": f"Long top {100 // n_quantiles}% / Short bottom {100 // n_quantiles}%",
        "holding_period_days": holding_period,
    }


def save_factor_results(
    factor_code: str,
    ic_results: dict[str, Any],
    backtest_results: dict[str, Any] | None = None,
) -> str:
    """Save successful factor results to a file."""
    output_dir = ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"factor_{timestamp}.json"
    filepath = output_dir / filename

    results = {
        "timestamp": timestamp,
        "factor_code": factor_code,
        "evaluation_metrics": ic_results,
    }

    if backtest_results:
        results["backtest_performance"] = backtest_results

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved factor results to {filepath}")
    return str(filepath)


class FactorEvaluatorConfig(FunctionBaseConfig, name="factor_evaluator"):
    """
    Factor Evaluator: Evaluates factors and provides optimization advice or accepts.

    This step evaluates the rank IC of generated factor code and either:
    - Accepts the factor if it meets criteria (saves results)
    - Provides optimization advice for the next iteration
    """

    llm_name: str = Field(
        description="LLM to use for generating optimization advice.",
    )
    ic_threshold: float = Field(
        default=0.03,
        description="Minimum mean IC required to accept a factor (absolute value).",
    )
    p_value_threshold: float = Field(
        default=0.1,
        description="Maximum p-value for statistical significance.",
    )
    forward_periods: int = Field(
        default=5,
        description="Number of forward periods for return calculation.",
    )
    save_on_accept: bool = Field(
        default=True,
        description="Whether to save factor results when accepted.",
    )


@register_function(
    config_type=FactorEvaluatorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def factor_evaluator_function(config: FactorEvaluatorConfig, builder: Builder):
    """
    Factor Evaluator that provides IC metrics and optimization advice.
    """

    # Get LLM for generating advice
    llm = await builder.get_llm(
        llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )

    # Load stock data once
    stock_data = load_stock_data()

    def identify_issues(ic_results: dict[str, Any]) -> str:
        """Identify specific issues with the factor."""
        issues = []
        mean_ic = ic_results.get("mean_ic")
        p_value = ic_results.get("p_value")
        positive_ratio = ic_results.get("positive_ic_ratio", 0)

        if mean_ic is None:
            issues.append("- Factor code execution failed or returned invalid values")
        elif abs(mean_ic) < 0.01:
            issues.append("- IC is essentially zero - factor has no predictive power")
            issues.append(
                "- Consider using different data combinations or lookback periods"
            )
        elif abs(mean_ic) < config.ic_threshold:
            issues.append(
                f"- IC magnitude ({abs(mean_ic):.4f}) is below threshold ({config.ic_threshold})"
            )
            issues.append("- Factor shows weak signal, needs stronger alpha source")

        if p_value and p_value > config.p_value_threshold:
            issues.append(f"- Results not statistically significant (p={p_value:.4f})")
            issues.append("- High variance in IC - factor behavior is inconsistent")

        if positive_ratio < 0.4:
            issues.append(
                f"- Low positive IC ratio ({positive_ratio:.1%}) - factor often gives wrong signals"
            )
        elif positive_ratio > 0.6 and mean_ic and mean_ic < 0:
            issues.append(
                "- Negative mean IC despite frequent positive periods - large losses on bad days"
            )

        return "\n".join(issues) if issues else "- No specific issues identified"

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

    async def generate_optimization_advice(
        factor_code: str, ic_results: dict[str, Any]
    ) -> str:
        """Generate optimization advice based on IC results."""
        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = """You are a senior quantitative researcher providing feedback on factor performance.
Based on the rank IC evaluation results, provide specific, actionable advice to improve the factor.

Your advice should be concise and directly usable by the factor generator.
Focus on:
1. What might be wrong with the current factor design
2. Specific changes to the formula or lookback periods
3. Alternative approaches that might work better"""

        mean_ic = ic_results.get("mean_ic")
        ic_std = ic_results.get("ic_std")
        p_value = ic_results.get("p_value")
        positive_ratio = ic_results.get("positive_ic_ratio", 0)

        mean_ic_str = f"{mean_ic:.4f}" if mean_ic is not None else "N/A"
        ic_std_str = f"{ic_std:.4f}" if ic_std is not None else "N/A"
        p_value_str = f"{p_value:.4f}" if p_value is not None else "N/A"

        issues_str = identify_issues(ic_results)

        user_prompt = f"""Factor evaluation results:

FACTOR CODE (excerpt):
{factor_code[:1000]}...

EVALUATION METRICS:
- Mean Rank IC: {mean_ic_str}
- IC Standard Deviation: {ic_std_str}
- P-value: {p_value_str}
- Positive IC Ratio: {positive_ratio:.2%}

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
        return response.content if hasattr(response, "content") else str(response)

    async def evaluate_factor(factor_code: str) -> str:
        """
        Evaluate factor and provide optimization advice or accept.

        Takes the Python code from factor_code_generator and:
        1. Executes the code to compute factor values
        2. Computes rank IC
        3. If IC is good → accepts and saves the factor
        4. If IC is poor → provides optimization advice

        Args:
            factor_code: Python code string from factor_code_generator.

        Returns:
            JSON string containing:
            - status: "accepted" or "needs_improvement"
            - evaluation_metrics: IC metrics
            - optimization_advice: Advice for improvement (if needs_improvement)
            - saved_path: Path to saved results (if accepted)
        """
        if not stock_data:
            return json.dumps(
                {
                    "status": "error",
                    "error": "No stock data available",
                },
                indent=2,
            )

        # Extract and execute factor code
        clean_code = extract_code_from_response(factor_code)
        exec_result = execute_factor_code(clean_code, stock_data)

        if exec_result is None:
            advice = (
                "Factor code execution failed. Please ensure:\n"
                "1. The factor function is properly defined\n"
                "2. All operators are correctly used\n"
                "3. The function returns a pandas DataFrame"
            )
            return json.dumps(
                {
                    "status": "needs_improvement",
                    "error": "Failed to execute factor code",
                    "optimization_advice": advice,
                },
                indent=2,
            )

        factor_values, selected_factor = exec_result

        # Compute forward returns and rank IC
        close_data = stock_data.get("Close")
        forward_returns = compute_forward_returns(
            close_data, periods=config.forward_periods
        )
        ic_results = compute_rank_ic(factor_values, forward_returns)

        if ic_results.get("error"):
            return json.dumps(
                {
                    "status": "needs_improvement",
                    "error": ic_results["error"],
                    "optimization_advice": "Unable to compute IC. Check factor output validity.",
                },
                indent=2,
            )

        # Check if factor is acceptable
        if is_factor_acceptable(ic_results):
            logger.info("Factor ACCEPTED!")
            mean_ic = ic_results.get("mean_ic", 0)
            logger.info(
                f"Mean IC: {mean_ic:.4f}, p-value: {ic_results.get('p_value', 'N/A')}"
            )

            saved_path = None
            if config.save_on_accept:
                saved_path = save_factor_results(factor_code, ic_results)

            return json.dumps(
                {
                    "status": "accepted",
                    "selected_factor": selected_factor,
                    "evaluation_metrics": ic_results,
                    "saved_path": saved_path,
                    "message": f"Factor accepted with IC={mean_ic:.4f}",
                },
                indent=2,
            )

        # Factor needs improvement - generate advice
        logger.info("Factor needs improvement. Generating optimization advice...")
        advice = await generate_optimization_advice(factor_code, ic_results)

        return json.dumps(
            {
                "status": "needs_improvement",
                "evaluation_metrics": ic_results,
                "optimization_advice": advice,
            },
            indent=2,
        )

    yield FunctionInfo.from_fn(
        evaluate_factor,
        description=(
            "Evaluate factor code using rank IC. Returns 'accepted' with saved results "
            "if factor meets criteria, or 'needs_improvement' with optimization advice."
        ),
    )


class FactorLoopExecutorConfig(FunctionBaseConfig, name="factor_loop_executor"):
    """
    Factor Loop Executor: Runs the factor mining loop with feedback.

    Orchestrates the sequential steps (generate -> code -> evaluate) in a loop,
    passing optimization advice back to the generator until a factor is accepted.
    """

    llm_name: str = Field(
        description="LLM to use for factor generation and optimization.",
    )
    max_iterations: int = Field(
        default=3,
        description="Maximum number of optimization iterations.",
    )
    num_factors: int = Field(
        default=1,
        description="Number of factors to generate per iteration.",
    )
    ic_threshold: float = Field(
        default=0.03,
        description="Minimum mean IC to accept a factor.",
    )
    p_value_threshold: float = Field(
        default=0.1,
        description="Maximum p-value for significance.",
    )
    forward_periods: int = Field(
        default=5,
        description="Forward periods for return calculation.",
    )
    history_length: int = Field(
        default=3,
        description="Number of recent optimization attempts to include as context (0 = all).",
    )


@register_function(
    config_type=FactorLoopExecutorConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def factor_loop_executor_function(
    config: FactorLoopExecutorConfig, builder: Builder
):
    """
    Factor Loop Executor that runs sequential steps with feedback.
    """
    from .factor_generator import (
        load_calculator_operators,
        format_operators_for_prompt,
        get_output_format_prompt,
    )
    from .factor_code_generator import (
        get_operator_code_map,
        extract_operators_from_json,
    )

    # Get LLM
    llm = await builder.get_llm(
        llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )

    # Load resources
    operators = load_calculator_operators()
    operators_list = format_operators_for_prompt(operators)
    output_format = get_output_format_prompt().replace(
        "{num_factors}", str(config.num_factors)
    )
    stock_data = load_stock_data()
    code_map = get_operator_code_map(operators)

    # ===== Step 1: Factor Generator =====
    async def step1_generate_factors(
        request: str, optimization_history: list[dict] | None = None
    ) -> str:
        """Generate factor descriptions, incorporating recent previous advice."""
        advice_section = ""
        if optimization_history and len(optimization_history) > 0:
            # Limit to last N entries based on config.history_length (0 = all)
            if config.history_length > 0:
                recent_history = optimization_history[-config.history_length :]
            else:
                recent_history = optimization_history

            # Format recent attempts and their advice
            history_parts = []
            for entry in recent_history:
                iteration = entry.get("iteration", "?")
                ic = entry.get("mean_ic")
                ic_str = f"{ic:.4f}" if ic is not None else "N/A"
                advice = entry.get("advice", "No advice")
                history_parts.append(f"""
--- Iteration {iteration} (IC: {ic_str}) ---
{advice}
""")

            all_advice = "\n".join(history_parts)
            shown_count = len(recent_history)
            total_count = len(optimization_history)
            history_note = (
                f"Showing last {shown_count} of {total_count} attempts."
                if shown_count < total_count
                else ""
            )

            advice_section = f"""

IMPORTANT - OPTIMIZATION HISTORY FROM PREVIOUS ITERATIONS:
You have tried {total_count} factor(s) that did not meet the criteria.
{history_note}
Learn from these attempts to create a better factor.

{all_advice}

CRITICAL INSTRUCTIONS:
1. Do NOT repeat similar approaches that failed before
2. Combine insights from the feedback to design a fundamentally different factor
3. Address ALL issues mentioned in the iterations shown
4. Try a significantly different formula structure or data combination
"""

        prompt = f"""You are a senior quantitative researcher at a top hedge fund.
Generate {config.num_factors} unique stock selection factors based on the request.

REQUEST: {request}
{advice_section}
DATA AVAILABLE:
- Open: Opening price
- Close: Closing price
- High: Highest price
- Low: Lowest price
- Volume: Trading volume

OPERATORS YOU CAN USE:
{operators_list}

{output_format}

CRITICAL RULES - READ CAREFULLY:
1. Use ONLY the EXACT operator names listed above - DO NOT invent new operators
2. To combine operations, NEST them: e.g., Rank(TS_Return(Close, 20)) NOT Rank_TS_Return
3. Valid examples: Rank(TS_Mean(Volume, 10)), Div(TS_Return(Close, 5), TS_Std(Close, 20))
4. INVALID examples: Rank_TS_Return, TS_Mean_Volume, Rank_Div_Return (these don't exist!)
5. Create factors with clear economic intuition
6. Return valid JSON that can be parsed

Generate {config.num_factors} factors now:"""

        response = await llm.ainvoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    # ===== Step 2: Code Generator =====
    async def step2_generate_code(factor_json: str) -> str:
        """Generate executable Python code from factor JSON."""
        import re

        _, operator_code_block = extract_operators_from_json(factor_json, code_map)

        # Extract operators used
        required_ops = set()
        try:
            data = json.loads(factor_json)
            if isinstance(data, list):
                for factor in data:
                    if "operators_used" in factor:
                        required_ops.update(factor["operators_used"])
        except json.JSONDecodeError:
            pass

        valid_ops = {op for op in required_ops if op in code_map}

        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = """You are a senior programmer. Write ONLY the factor function.
DO NOT include imports or operator definitions - they are already provided.

OUTPUT FORMAT:
```python
def factor_name(Open: pd.DataFrame, Close: pd.DataFrame, ...) -> pd.DataFrame:
    '''Docstring'''
    result = ...
    return result
```"""

        user_prompt = f"""Write the factor function for:

{factor_json}

AVAILABLE OPERATORS: {", ".join(sorted(valid_ops))}

Generate ONLY the factor function:"""

        messages = [
            SystemMessage(content="detailed thinking off"),
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await llm.ainvoke(messages)
        factor_function_code = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Extract from markdown
        code_blocks = re.findall(
            r"```python\n(.*?)```", factor_function_code, re.DOTALL
        )
        if code_blocks:
            factor_function_code = code_blocks[0]

        # Build complete code
        return "\n".join(
            [
                "import pandas as pd",
                "import numpy as np",
                "",
                "# Operator functions",
                operator_code_block,
                "",
                "# Factor function",
                factor_function_code,
            ]
        )

    # ===== Step 3: Evaluate =====
    def step3_evaluate(factor_code: str) -> dict:
        """Evaluate factor and return results."""
        if not stock_data:
            return {"status": "error", "error": "No stock data"}

        clean_code = extract_code_from_response(factor_code)
        exec_result = execute_factor_code(clean_code, stock_data)

        if exec_result is None:
            return {
                "status": "needs_improvement",
                "error": "Code execution failed",
                "mean_ic": None,
            }

        factor_values, selected_factor = exec_result

        close_data = stock_data.get("Close")
        forward_returns = compute_forward_returns(
            close_data, periods=config.forward_periods
        )
        ic_results = compute_rank_ic(factor_values, forward_returns)

        mean_ic = ic_results.get("mean_ic")
        p_value = ic_results.get("p_value")

        # Check acceptance
        is_acceptable = (
            mean_ic is not None
            and abs(mean_ic) >= config.ic_threshold
            and (p_value is None or p_value <= config.p_value_threshold)
        )

        return {
            "status": "accepted" if is_acceptable else "needs_improvement",
            "selected_factor": selected_factor,
            "mean_ic": mean_ic,
            "p_value": p_value,
            "ic_results": ic_results,
        }

    async def generate_advice(factor_code: str, eval_results: dict) -> str:
        """Generate optimization advice."""
        mean_ic = eval_results.get("mean_ic")
        p_value = eval_results.get("p_value")
        error = eval_results.get("error", "")

        mean_ic_str = f"{mean_ic:.4f}" if mean_ic is not None else "N/A"
        p_value_str = f"{p_value:.4f}" if p_value is not None else "N/A"

        if error and "Code execution failed" in error:
            return (
                f"CODE EXECUTION ERROR: {error}\n\n"
                f"Common issues to fix:\n"
                f"1. Make sure all operators have correct number of arguments\n"
                f"   - TS_Std(x, d) needs TWO arguments: data and lookback period\n"
                f"   - TS_Return(x, d) needs TWO arguments: data and lookback period\n"
                f"2. Use ONLY valid operators (nest them, don't combine names):\n"
                f"   - WRONG: Rank_TS_Return, TS_Mean_Volume\n"
                f"   - RIGHT: Rank(TS_Return(Close, 20)), TS_Mean(Volume, 10)\n"
                f"3. Return a pandas DataFrame from the factor function"
            )

        error_section = f"\nERROR: {error}\n" if error else ""

        prompt = f"""Factor evaluation failed. Provide optimization advice.
{error_section}
METRICS:
- Mean IC: {mean_ic_str} (need >= {config.ic_threshold})
- P-value: {p_value_str} (need <= {config.p_value_threshold})

FACTOR CODE:
{factor_code[:800]}...

Provide 3-5 specific improvements:"""

        response = await llm.ainvoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    # ===== Backtest Helper =====
    def compute_backtest_for_code(factor_code: str) -> dict[str, Any] | None:
        """Execute factor code and compute backtest returns."""
        if not stock_data:
            return None

        clean_code = extract_code_from_response(factor_code)
        exec_result = execute_factor_code(clean_code, stock_data)

        if exec_result is None:
            return {"error": "Could not execute factor code for backtest"}

        factor_values, _ = exec_result

        close_data = stock_data.get("Close")
        if close_data is None:
            return {"error": "Close price data not available"}

        forward_returns = compute_forward_returns(
            close_data, periods=config.forward_periods
        )

        return compute_factor_returns(
            factor_values,
            forward_returns,
            n_quantiles=5,
            holding_period=config.forward_periods,
        )

    # ===== Main Loop =====
    async def run_factor_loop(request: str) -> str:
        """
        Run the factor mining loop with feedback.

        Executes the sequential steps (generate → code → evaluate) in a loop,
        passing ALL accumulated optimization advice back to the generator.

        Args:
            request: Type of factor to generate (e.g., "momentum factors").

        Returns:
            JSON with final results including factor code and evaluation metrics.
        """
        optimization_history: list[dict] = []  # Accumulate ALL advice
        best_result = None
        best_ic = 0

        for iteration in range(1, config.max_iterations + 1):
            logger.info(f"=== Iteration {iteration}/{config.max_iterations} ===")

            # Step 1: Generate factors (with ALL accumulated advice)
            logger.info("Step 1: Generating factor descriptions...")
            if optimization_history:
                logger.info(
                    f"Passing {len(optimization_history)} previous attempts as context"
                )
            factor_json = await step1_generate_factors(
                request, optimization_history if optimization_history else None
            )

            # Step 2: Generate code
            logger.info("Step 2: Generating factor code...")
            factor_code = await step2_generate_code(factor_json)

            # Step 3: Evaluate
            logger.info("Step 3: Evaluating rank IC...")
            eval_results = step3_evaluate(factor_code)

            # Track best result (keep any valid result, prioritize by |IC|)
            mean_ic = eval_results.get("mean_ic")
            if mean_ic is not None:
                if best_result is None or abs(mean_ic) > abs(best_ic):
                    best_ic = mean_ic
                    best_result = {
                        "iteration": iteration,
                        "factor_json": factor_json,
                        "factor_code": factor_code,
                        "eval_results": eval_results,
                    }
                    logger.info(
                        f"Updated best result: IC={mean_ic:.4f} from iteration {iteration}"
                    )

            # Check if accepted
            if eval_results["status"] == "accepted":
                logger.info(f"Factor ACCEPTED at iteration {iteration}!")

                # Compute backtest performance
                logger.info("Computing backtest performance...")
                backtest_results = compute_backtest_for_code(factor_code)

                if (
                    backtest_results
                    and backtest_results.get("annual_return") is not None
                ):
                    annual_ret = backtest_results["annual_return_pct"]
                    sharpe = backtest_results["sharpe_ratio"]
                    logger.info(
                        f"Backtest: Annual Return={annual_ret}, Sharpe={sharpe:.2f}"
                    )

                saved_path = save_factor_results(
                    factor_code, eval_results["ic_results"], backtest_results
                )

                return json.dumps(
                    {
                        "status": "accepted",
                        "iterations": iteration,
                        "selected_factor": eval_results.get("selected_factor"),
                        "factor_code": factor_code,
                        "evaluation_metrics": eval_results["ic_results"],
                        "backtest_performance": backtest_results,
                        "saved_path": saved_path,
                        "optimization_history_length": len(optimization_history),
                        "message": f"Factor accepted at iteration {iteration} with IC={mean_ic:.4f}",
                    },
                    indent=2,
                )

            # Generate advice and ADD to history (accumulate all)
            logger.info("Generating optimization advice...")
            advice = await generate_advice(factor_code, eval_results)
            optimization_history.append(
                {
                    "iteration": iteration,
                    "mean_ic": mean_ic,
                    "p_value": eval_results.get("p_value"),
                    "advice": advice,
                }
            )
            logger.info(
                f"Advice added to history. Total attempts: {len(optimization_history)}"
            )

        # Return best result after max iterations
        logger.info("Max iterations reached. Returning best result.")

        if best_result:
            # Compute backtest for best result
            logger.info("Computing backtest performance for best result...")
            backtest_results = compute_backtest_for_code(best_result["factor_code"])

            if backtest_results and backtest_results.get("annual_return") is not None:
                annual_ret = backtest_results["annual_return_pct"]
                sharpe = backtest_results["sharpe_ratio"]
                logger.info(
                    f"Backtest: Annual Return={annual_ret}, Sharpe={sharpe:.2f}"
                )

            saved_path = save_factor_results(
                best_result["factor_code"],
                best_result["eval_results"].get("ic_results", {}),
                backtest_results,
            )
            return json.dumps(
                {
                    "status": "best_effort",
                    "iterations": config.max_iterations,
                    "selected_factor": best_result["eval_results"].get(
                        "selected_factor"
                    ),
                    "factor_code": best_result["factor_code"],
                    "evaluation_metrics": best_result["eval_results"].get(
                        "ic_results", {}
                    ),
                    "backtest_performance": backtest_results,
                    "saved_path": saved_path,
                    "message": f"Best factor from iteration {best_result['iteration']} with IC={best_ic:.4f}",
                },
                indent=2,
            )

        return json.dumps(
            {
                "status": "failed",
                "iterations": config.max_iterations,
                "message": "No valid factors generated",
            },
            indent=2,
        )

    yield FunctionInfo.from_fn(
        run_factor_loop,
        description=(
            "Run the factor mining loop: generate → code → evaluate → feedback. "
            "Iterates until a good factor is found or max iterations reached."
        ),
    )
