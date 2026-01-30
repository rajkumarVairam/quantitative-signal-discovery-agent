# Factor Mining Workflow with NeMo Agent Toolkit

An end-to-end factor mining workflow for quantitative finance using NVIDIA NeMo Agent Toolkit. This workflow demonstrates how to leverage LLMs to automatically generate, code, and evaluate alpha factors.

## Overview

Factor mining is the process of discovering quantitative signals (factors) that have predictive power for future stock returns. This workflow automates the traditional labor-intensive process using LLMs.

### Workflow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Factor Optimization Agent                      │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Factor    │    │    Code     │    │    Rank IC          │  │
│  │  Generator  │───▶│  Generator  │───▶│    Evaluator        │  │
│  │   (LLM)     │    │   (LLM)     │    │  (Statistical)      │  │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘  │
│                                                   │             │
│                         ┌─────────────────────────┘             │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │  IC Good? Accept!   │                            │
│              │  IC Poor? Optimize  │◀────────────┐              │
│              └──────────┬──────────┘             │              │
│                         │                        │              │
│                         ▼                        │              │
│              ┌─────────────────────┐             │              │
│              │   Optimization      │─────────────┘              │
│              │   Feedback (LLM)    │                            │
│              └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

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
source .venv/bin/activate
uv pip install -e .
```

## Usage

### Running via Command Line

```bash
nat run --config_file configs/config-optimization.yml --input "momentum factors"
```

### Running Different Factor Types

```bash
# Generate volatility factors
nat run --config_file configs/config-optimization.yml --input "volatility factors"

# Generate mean reversion factors
nat run --config_file configs/config-optimization.yml --input "mean reversion factors"
```

## Components

| Component | Description |
|-----------|-------------|
| **Factor Generator** | Uses an LLM to create quantitative factor descriptions based on price-volume data |
| **Code Generator** | Converts factor descriptions into executable Python code |
| **Rank IC Evaluator** | Computes Spearman correlation between factor values and forward returns |
| **Factor Optimization Agent** | Orchestrates the optimization loop with feedback |

## Configuration

The workflow configuration is defined in `configs/config-optimization.yml`:

| Parameter | Description |
|-----------|-------------|
| `llm_name` | The LLM to use for factor generation and optimization advice |
| `ic_threshold` | Minimum absolute IC value to accept a factor (e.g., 0.03 = 3%) |
| `p_value_threshold` | Maximum p-value for statistical significance (e.g., 0.1 = 10%) |
| `max_iterations` | Maximum number of optimization iterations before accepting best result |
| `num_factors` | Number of factors to generate per iteration |
| `forward_periods` | Number of days for forward return calculation (e.g., 5 = weekly) |
| `save_results` | Whether to save successful factors to disk |

## Evaluation Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Mean IC** | Average Spearman correlation between factor and forward returns | \|IC\| > 0.03 |
| **IC Std** | Standard deviation of IC values | Lower is more consistent |
| **IC IR** | Information Ratio = Mean IC / IC Std | > 0.5 is good |
| **T-statistic** | Statistical significance of mean IC | \|t\| > 2 is significant |
| **P-value** | Probability IC is different from zero | < 0.05 is significant |
| **Positive IC Ratio** | Fraction of periods with positive IC | > 0.55 is good |

## Project Structure

```
factor_mining_complete/
├── configs/
│   └── config-optimization.yml
├── factor-mining-complete.ipynb
├── pyproject.toml
├── README.md
└── src/
    └── factor_mining_workflow/
        ├── __init__.py
        ├── data/sp500/           # Sample market data
        ├── factor_evaluator.py
        ├── factor_generator.py
        ├── factor_mining_workflow.py
        ├── factor_optimization_agent.py
        ├── rank_ic_evaluator.py
        ├── register.py
        ├── output/               # Generated factors saved here
        └── template/
            ├── calculator.json
            └── factor_output_template.json
```

## Additional Resources

- [NeMo Agent Toolkit Documentation](https://docs.nvidia.com/nemo-agent-toolkit/)

## License

See LICENSE file for details.

