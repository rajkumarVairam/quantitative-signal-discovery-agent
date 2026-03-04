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

# Disclaimer:
# Each user is responsible for checking the content of datasets and the
# applicable licenses and determining if suitable for the intended use.

"""
Download S&P 500 price-volume data using yfinance.

Fetches Open, Close, High, Low, and Volume data for current S&P 500
constituents and saves each field as a separate CSV file compatible
with the factor mining workflow.

Usage:
    python -m factor_mining_workflow.download_data
    python -m factor_mining_workflow.download_data --start 2015-01-01 --end 2025-12-31
    python -m factor_mining_workflow.download_data --output /path/to/output
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data" / "sp500"
DATA_FIELDS = ["Open", "Close", "High", "Low", "Volume"]


def get_sp500_tickers() -> list[str]:
    """Fetch the current list of S&P 500 ticker symbols from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)[0]
    tickers = sorted(table["Symbol"].str.replace(".", "-", regex=False).tolist())
    logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
    return tickers


def download_sp500_data(
    start: str = "2012-01-01",
    end: str = "2025-12-31",
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Download S&P 500 price-volume data and save as CSV files.

    Args:
        start: Start date in YYYY-MM-DD format.
        end: End date in YYYY-MM-DD format.
        output_dir: Directory to save CSV files. Defaults to data/sp500/.

    Returns:
        Dictionary mapping field names to DataFrames.
    """
    output_path = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    tickers = get_sp500_tickers()

    logger.info(f"Downloading data from {start} to {end} for {len(tickers)} tickers...")
    raw = yf.download(tickers, start=start, end=end, group_by="column", auto_adjust=True)

    results = {}
    for field in DATA_FIELDS:
        if field in raw.columns.get_level_values(0):
            df = raw[field].copy()
        else:
            logger.warning(f"Field '{field}' not found in downloaded data, skipping")
            continue

        df.index.name = "Date"
        df = df.dropna(axis=1, how="all")

        csv_path = output_path / f"{field}.csv"
        df.to_csv(csv_path)
        results[field] = df
        logger.info(f"Saved {field}.csv with shape {df.shape}")

    logger.info(f"Data saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download S&P 500 price-volume data for factor mining."
    )
    parser.add_argument(
        "--start", default="2012-01-01", help="Start date (YYYY-MM-DD). Default: 2012-01-01"
    )
    parser.add_argument(
        "--end", default="2025-12-31", help="End date (YYYY-MM-DD). Default: 2025-12-31"
    )
    parser.add_argument(
        "--output", default=None, help="Output directory. Default: data/sp500/"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    download_sp500_data(start=args.start, end=args.end, output_dir=args.output)


if __name__ == "__main__":
    main()
