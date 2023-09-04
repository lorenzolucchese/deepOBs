from data_process import multiprocess_orderbooks, aggregate_stats
from config.directories import ROOT_DIR
import os
import time
import sys
import numpy as np

if __name__ == "__main__":
    # set global parameters
    TICKERS = ["LILAK", "QRTEA", "XRAY", "CHTR", "PCAR", "EXC", "AAL", "WBA", "ATVI", "AAPL"]

    TICKER = TICKERS[int(sys.argv[1])]
    input_path = os.path.join(ROOT_DIR, "data_raw", TICKER + "_data_dwn")
    log_path = os.path.join(ROOT_DIR, "data", "logs", TICKER + "_processing_logs")
    horizons = np.array([10, 20, 30, 50, 100, 200, 300, 500, 1000])

    os.makedirs(log_path, exist_ok=True)

    # ============================================================================
    # LOBSTER DATA (multiprocess)

    output_path = os.path.join(ROOT_DIR, "data", TICKER)
    os.makedirs(output_path, exist_ok=True)
    stats_path = os.path.join(output_path, "stats")
    os.makedirs(stats_path, exist_ok=True)

    startTime = time.time()
    multiprocess_orderbooks(TICKER=TICKER,
                            input_path=input_path, 
                            output_path=output_path,
                            log_path=log_path, 
                            stats_path=stats_path,
                            horizons=horizons, 
                            NF_volume=40, 
                            queue_depth=10, 
                            smoothing="uniform", 
                            k=10,
                            check_for_processed_data=True)
    executionTime = (time.time() - startTime)

    print("Execution time in minutes: " + str(executionTime/60))

    aggregate_stats(TICKER, stats_path)
    
