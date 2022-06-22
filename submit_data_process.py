from data_process import process_data, multiprocess_L3
import os
import time
import sys
import numpy as np

if __name__ == "__main__":
    # set global parameters
    TICKERS = ["LILAK", "QRTEA", "XRAY", "CHTR", "PCAR", "EXC", "AAL", "WBA", "ATVI", "AAPL"]

    TICKER = TICKERS[sys.argv[1]]
    input_path = "data_raw/" + TICKER + "_data_dwn"
    log_path = "data/logs/" + TICKER + "_processing_logs"
    horizons = np.array([10, 20, 30, 50, 100, 200, 300, 500, 1000])

    os.mkdir(log_path)

    # ============================================================================
    # LOBSTER DATA - ORDERBOOKS

    output_path = "data/" + TICKER + "_orderbooks"
    os.mkdir(output_path)

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

    output_path = "data/" + TICKER + "_orderflows"
    os.mkdir(output_path)

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

    output_path = "data/" + TICKER + "_volumes"
    os.mkdir(output_path)

    startTime = time.time()
    multiprocess_L3(TICKER=TICKER,
                    input_path=input_path, 
                    output_path=output_path,
                    log_path=log_path, 
                    queue_depth=10, 
                    horizons=horizons)
    executionTime = (time.time() - startTime)

    print("Volumes execution time in minutes: " + str(executionTime/60))

    # #============================================================================
    # # SIMULATED DATA

    # # set parameters
    # input_path = r"/scratch/lucchese/deepLOBs/data/simulated"
    # output_path = r"/scratch/lucchese/deepLOBs/data/model/simulated_orderflows"

    # startTime = time.time()
    # process_simulated_data(input_path=input_path, 
    #              output_path=output_path,
    #              features="orderflows")
    # executionTime = (time.time() - startTime)

    # print("Execution time in seconds: " + str(executionTime))