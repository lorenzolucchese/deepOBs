from process_data.data_process import process_data, multiprocess_L3
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
    # LOBSTER DATA - ORDERBOOKS

    output_path = os.path.join(ROOT_DIR, "data", TICKER + "_orderbooks")
    os.makedirs(output_path, exist_ok=True)

    startTime = time.time()
    process_data(TICKER=TICKER, 
                 input_path=input_path, 
                 output_path=output_path,
                 log_path=log_path,
                 features="orderbooks",
                 horizons=horizons)
    executionTime = (time.time() - startTime)

    print("Orderbooks execution time in minutes: " + str(executionTime/60))

    # ============================================================================
    # LOBSTER DATA - ORDERFLOWS

    output_path = os.path.join(ROOT_DIR, "data", TICKER + "_orderflows")
    os.makedirs(output_path, exist_ok=True)

    startTime = time.time()
    process_data(TICKER=TICKER, 
                 input_path=input_path, 
                 output_path=output_path,
                 log_path=log_path,
                 features="orderflows",
                 horizons=horizons)
    executionTime = (time.time() - startTime)

    print("Orderflows execution time in minutes: " + str(executionTime/60))

    # ============================================================================
    # LOBSTER DATA - VOLUMES L3 (multiprocess)

    output_path = os.path.join(ROOT_DIR, "data", TICKER + "_volumes")
    os.makedirs(output_path, exist_ok=True)

    startTime = time.time()
    multiprocess_L3(TICKER=TICKER,
                    input_path=input_path, 
                    output_path=output_path,
                    log_path=log_path, 
                    queue_depth=10, 
                    horizons=horizons)
    executionTime = (time.time() - startTime)

    print("Volumes execution time in minutes: " + str(executionTime/60))
    
