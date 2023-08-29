from data_process import percentiles_features
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
    percentiles = [0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100]
    features = ["orderbook", "orderflow", "volume"]

    os.makedirs(stats_path, exist_ok=True)

    # ============================================================================
    startTime = time.time()
    percentiles_features(TICKER = TICKER, 
                         processed_data_path = processed_data_path, 
                         stats_path = stats_path, 
                         percentiles = percentiles, 
                         features = features)
    executionTime = (time.time() - startTime)

    print("Execution time in minutes: " + str(executionTime/60))
    
