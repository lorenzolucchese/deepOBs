from data_process import multiprocess_orderbooks, aggregate_stats
from config.directories import ROOT_DIR
import os
import time
import sys
import numpy as np

if __name__ == "__main__":
    # set global parameters
    TICKERS = ["LILAK", "QRTEA", "XRAY", "CHTR", "PCAR", "EXC", "AAL", "WBA", "ATVI", "AAPL"]

    for TICKER in TICKERS:
        output_path = os.path.join(ROOT_DIR, "data", TICKER)
        os.makedirs(output_path, exist_ok=True)
        stats_path = os.path.join(output_path, "stats")
        os.makedirs(stats_path, exist_ok=True)

        aggregate_stats(TICKER, stats_path)
    
