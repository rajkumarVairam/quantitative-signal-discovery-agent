"""
Daily data refresh script — run at 5:00 PM CST via Windows Task Scheduler.

Downloads the latest S&P 500 OHLCV prices from Yahoo Finance and overwrites
the existing CSV files so the Streamlit dashboard picks up fresh data on its
next cache expiry (TTL = 1 hour).

Usage:
    uv run python webapp/update_data.py

Logs output to webapp/update_data.log for diagnosing Task Scheduler failures.
"""

import logging
import sys
from datetime import date
from pathlib import Path

# Ensure the installed package is importable when invoked directly by Task Scheduler
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signal_discovery_workflow.download_data import DEFAULT_OUTPUT_DIR, download_sp500_data

LOG_FILE = Path(__file__).parent / "update_data.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = "2012-01-01"  # keep full history for signal lookback warmup

    logger.info("=" * 60)
    logger.info(f"S&P 500 daily data refresh starting")
    logger.info(f"Date range : {start_date} → {end_date}")
    logger.info(f"Output dir : {DEFAULT_OUTPUT_DIR}")

    try:
        results = download_sp500_data(start=start_date, end=end_date)
        for field, df in results.items():
            logger.info(f"  {field:8s}: {df.shape[0]:,} rows × {df.shape[1]:,} tickers")
        logger.info("Data refresh complete. Dashboard will pick up new data within 1 hour.")
    except Exception:
        logger.exception("Data refresh FAILED — check network and yfinance availability.")
        sys.exit(1)


if __name__ == "__main__":
    main()
