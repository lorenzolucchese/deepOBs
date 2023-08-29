from data_process import dependence_responses
from config.directories import ROOT_DIR
import os
import time
import sys

if __name__ == "__main__":
    # set global parameters
    TICKERS = ["LILAK", "QRTEA", "XRAY", "CHTR", "PCAR", "EXC", "AAL", "WBA", "ATVI", "AAPL"]

    TICKER = TICKERS[int(sys.argv[1])]
    processed_data_path = os.path.join(ROOT_DIR, "data", TICKER)
    stats_path = os.path.join(ROOT_DIR, "data", "stats")
    results_path = os.path.join(ROOT_DIR, "results")

    os.makedirs(stats_path, exist_ok=True)

    # ============================================================================
    startTime = time.time()
    dependence_responses(TICKER = TICKER, 
                         processed_data_path = processed_data_path, 
                         results_path = results_path,
                         stats_path = stats_path)
    executionTime = (time.time() - startTime)

    print("Execution time in minutes: " + str(executionTime/60))
    
