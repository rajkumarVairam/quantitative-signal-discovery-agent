# Quant Factor Mining Agent developer example

An end-to-end factor mining workflow for quantitative finance using NVIDIA NeMo Agent Toolkit. This workflow demonstrates how to leverage LLMs to automatically generate, code, and evaluate alpha factors.

## Overview

Factor mining is the process of discovering quantitative signals (factors) that have predictive power for future stock returns. This workflow automates the traditional labor-intensive process using LLMs.

### Workflow Architecture

![Workflow Architecture](notebooks/images/workflow-architecture.png)

The workflow uses a **closed-loop optimization** approach:
1. Generate factor ideas using an LLM
2. Convert ideas to executable Python code
3. Evaluate the factor's predictive power (Rank IC)
4. If IC meets threshold → Accept and save
5. If IC is poor → Generate optimization advice and retry

## Getting Started

### Prerequisites

- **Platform:** Linux, macOS, or Windows
- **Python:** version 3.11, 3.12, or 3.13
- **Package manager:** pip or uv

### API Keys

You will need an NVIDIA API key. Get yours from [build.nvidia.com](https://build.nvidia.com/settings/api-keys).

```bash
export NVIDIA_API_KEY="your-api-key-here"
```

### Installing Dependencies

```bash
uv venv
uv pip install -e .
```

(All commands below use `uv run ...` so you don't need to activate the venv. If you prefer to activate it once and drop the `uv run` prefix, run `source .venv/bin/activate` first.)

### Download Data

The workflow requires S&P 500 price-volume data (Open, Close, High, Low, Volume). Use the included script to download fresh data via [yfinance](https://github.com/ranaroussi/yfinance):

```bash
uv run python -m factor_mining_workflow.download_data
```

You can customize the date range:

```bash
uv run python -m factor_mining_workflow.download_data --start 2015-01-01 --end 2025-12-31
```

> **Disclaimer:** Each user is responsible for checking the content of datasets and the applicable licenses and determining if suitable for the intended use.

## Deployment Options

This workflow can be deployed in two ways:

### Option 1: Interactive Notebook Deployment

Best for exploration, experimentation, and learning. The notebook provides step-by-step execution with inline documentation.

```bash
uv run jupyter notebook notebooks/factor-mining-workflow.ipynb
```

The notebook includes:
- API key setup
- Configuration exploration
- Step-by-step workflow execution
- Interactive result visualization
- Ability to modify parameters on-the-fly

### Option 2: CLI Deployment

Best for production, automation, and scripting. Run the workflow directly from the command line.

#### Basic Usage

```bash
# Run the factor mining workflow
uv run nat run --config_file configs/config-optimization.yml --input "momentum factors"
```

#### With Phoenix Tracing (Recommended)

For full observability with LLM tracing, run Phoenix in a separate terminal first:

**Terminal 1 - Start Phoenix Server:**
```bash
uv run phoenix serve
```

Phoenix will start at http://localhost:6006

**Terminal 2 - Run the Workflow:**
```bash
export NVIDIA_API_KEY="your-api-key-here"
uv run nat run --config_file configs/config-optimization.yml --input "momentum factors"
```

View traces at http://localhost:6006 to see:
- LLM calls and responses
- Token usage
- Latency metrics
- Full execution trace

#### Running Different Factor Types

```bash
# Generate volatility factors
uv run nat run --config_file configs/config-optimization.yml --input "volatility factors"

# Generate mean reversion factors
uv run nat run --config_file configs/config-optimization.yml --input "mean reversion factors"

# Generate volume-based factors
uv run nat run --config_file configs/config-optimization.yml --input "volume price divergence factors"
```

## Components

| Component | Description |
|-----------|-------------|
| **Factor Agent** | Uses an LLM to generate factor expressions based on price-volume data and operators |
| **Code Agent** | Wraps each factor formula in a Python function via an LLM, and inlines the required operator implementations from `calculator.json` to produce a self-contained executable module |
| **Eval Agent** | Performs backtesting via Rank IC and generates optimization suggestions |
| **Data Download Script** | Fetches S&P 500 price-volume data from Yahoo Finance via `yfinance` |

## Configuration

The workflow configuration is defined in `configs/config-optimization.yml`:

> **Note:** The `base_url` for the LLMs depends on your API key. Set it to either:
> - `https://integrate.api.nvidia.com/v1/` — for keys from [build.nvidia.com](https://build.nvidia.com)
> - `https://inference-api.nvidia.com/v1/` — for NVIDIA internal or enterprise API keys

| Parameter | Description |
|-----------|-------------|
| `factor_generator_llm` | Reference to the LLM block used for factor ideation (typically higher temperature for creativity) |
| `code_generator_llm` | Reference to the LLM block used for translating formulas into Python (low temperature for determinism) |
| `optimization_advisor_llm` | Reference to the LLM block used to produce iteration feedback (balanced temperature) |
| `ic_threshold` | Minimum absolute IC value to accept a factor (e.g., 0.02 = 2%) |
| `p_value_threshold` | Maximum p-value for statistical significance (e.g., 0.05 = 5%) |
| `max_iterations` | Maximum number of optimization iterations before returning the best result |
| `num_factors` | Number of factors to generate per iteration |
| `forward_periods` | Number of days for forward return calculation (e.g., 5 = weekly) |
| `save_results` | Whether to save accepted/best-effort factors to `output/` |

You can use the same model for all three agents (the default), or mix sizes: for example, assign a higher-capability reasoning model like `nvidia/llama-3.3-nemotron-super-49b-v1.5` to the Factor Agent for richer ideation while keeping the smaller `nvidia/nvidia-nemotron-nano-9b-v2` for the Code and Advisor agents — a one-line change in the YAML.

## Evaluation Metrics

The workflow uses two key metrics to decide whether to accept or reject a generated factor:

| Metric | Description | Acceptance Criteria |
|--------|-------------|---------------------|
| **Mean IC** | Average Spearman rank correlation between factor values and forward returns, computed across all time periods | \|IC\| ≥ `ic_threshold` (default: 0.02) |
| **P-value** | Statistical significance of the mean IC being different from zero | ≤ `p_value_threshold` (default: 0.05) |

A factor is accepted when both criteria are met. Otherwise, the Eval Agent generates optimization suggestions and the workflow retries.

## Workflow Result Format

Each run returns a structured JSON result containing the outcome, metrics, the factors that were tried, and (if the loop did not accept on the first iteration) the optimization advice produced for the next attempt. Example:

```json
{
  "status": "best_effort",
  "headline": "Best-effort result (IC threshold not met)",
  "request": "momentum factors",
  "iteration": 2,
  "total_iterations": 3,
  "selected_factor": "factor_volume_decayed_momentum",
  "thresholds": {
    "ic_threshold": 0.02,
    "p_value_threshold": 0.05
  },
  "metrics": {
    "mean_ic": 0.0103,
    "ic_std": 0.21,
    "ic_ir": 0.049,
    "t_stat": 2.91,
    "p_value": 0.0036,
    "num_periods": 3494,
    "positive_ic_ratio": 0.529
  },
  "factors": [
    {
      "name": "Volume-Decayed Momentum",
      "formula": "Mul(TS_Return(Close, 20), Decay_Linear(Volume, 20))",
      "category": "momentum",
      "data_fields_used": ["Close", "Volume"],
      "lookback_periods": [20]
    }
  ],
  "saved_path": "src/factor_mining_workflow/output/factor_xxx.json",
  "last_feedback": "- Try TS_Std instead of TS_Var for cleaner volatility signal\n- Use 60-day lookback instead of 20\n- Replace Volume with Close*Volume for dollar-volume weighting"
}
```

| Field | Description |
|-------|-------------|
| `status` | `"accepted"`, `"best_effort"`, or `"failed"` |
| `headline` | Human-readable summary |
| `iteration` / `total_iterations` | Which iteration produced this result, out of the max allowed |
| `selected_factor` | Python function name of the factor whose IC was reported (the evaluator picks the best when multiple factors are returned) |
| `metrics` | All non-null IC statistics |
| `factors` | Compact summary of every factor that was generated |
| `saved_path` | Where the full factor JSON + code was persisted |
| `last_feedback` | Optimization advice from the last failed iteration; pass it back to resume the loop |

### Resuming an Optimization Loop

The workflow input accepts either a plain string or a JSON object that bundles `seed_feedback` from a prior run. Pass the `last_feedback` field from a prior result to start a new loop with the previous advice already applied — useful when you want more iterations than `max_iterations` allows, or want to switch models mid-run.

The shell snippet below uses `jq` to read `last_feedback` from the prior result and pack it into the JSON input shape (install with `brew install jq` on macOS or `apt-get install jq` on Debian/Ubuntu):

```bash
# First run — best effort, did not converge
nat run --config_file configs/config-optimization.yml --input "momentum factors" > result1.json

# Extract last_feedback and resume
SEED=$(jq -r '.last_feedback' result1.json)
nat run --config_file configs/config-optimization.yml \
  --input "$(jq -nc --arg req 'momentum factors' --arg seed "$SEED" \
              '{request: $req, seed_feedback: $seed}')"
```

Or programmatically, passing the same JSON shape as the workflow's input string:

```python
import json

result1 = json.loads(await runner.ainvoke("momentum factors"))

resume_input = json.dumps({
    "request": "momentum factors",
    "seed_feedback": result1["last_feedback"],
})
result2 = json.loads(await runner.ainvoke(resume_input))
```

The `last_feedback` field can be persisted to disk and re-loaded later — there's no in-memory state required to resume.

## Development

Tests cover the agent helpers, prompt builders, JSON parsing, module assembly, and end-to-end execution:

```bash
uv pip install -e ".[test]"
uv run pytest tests/
```

The repo also ships a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs `ruff` lint and the test suite on every pull request to `main`.

## Project Structure

```
quant-factor-mining-agent/
├── .github/workflows/ci.yml          # PR-level CI: ruff lint + pytest
├── configs/
│   └── config-optimization.yml       # Workflow + LLM config (single source of truth)
├── notebooks/
│   ├── factor-mining-workflow.ipynb  # Interactive walkthrough
│   └── images/workflow-architecture.png
├── pyproject.toml                    # Dependencies, ruff/pytest config, NAT entry point
├── uv.lock                           # Pinned dependency resolution
├── README.md
├── src/factor_mining_workflow/
│   ├── __init__.py
│   ├── register.py                            # NAT function registration
│   ├── factor_generator.py                    # Factor agent: generates JSON factor descriptions
│   ├── factor_code_generator.py               # Code agent: turns JSON into executable Python
│   ├── factor_evaluator.py                    # Eval agent: runs factor code, computes Rank IC
│   ├── factor_mining_optimization_workflow.py # Orchestrator (closed-loop generate/code/eval/feedback)
│   ├── llm_utils.py                           # Shared LLM-output helpers (parse, sanitize, normalize)
│   ├── download_data.py                       # Fetches S&P 500 data via yfinance
│   ├── data/sp500/                            # OHLCV CSVs (gitignored)
│   ├── output/                                # Saved factor results (gitignored)
│   └── template/
│       ├── calculator.json                    # Operator catalogue (name, signature, code)
│       └── factor_output_template.json        # JSON schema the factor agent fills in
└── tests/                                     # pytest suite (86 tests)
```

## Additional Resources

- [NeMo Agent Toolkit Documentation](https://docs.nvidia.com/nemo-agent-toolkit/)
- [Arize Phoenix Documentation](https://arize.com/docs/phoenix)
- [NeMo Fine-tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/index.html) — to specialize Nemotron on your factor history

## License

See [LICENSE.txt](LICENSE.txt) and [LICENSE-3rd-party.txt](LICENSE-3rd-party.txt) for details.
