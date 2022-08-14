from data_process import process_data, multiprocess_L3, process_L3_orderbook
import os
import time
import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # set global parameters 
    TICKERS = ["LILAK", "QRTEA", "XRAY", "CHTR", "PCAR", "EXC", "AAL", "WBA", "ATVI", "AAPL"] 

    TICKER = "AAPL" 
    input_path = "data_raw/" + TICKER + "_data_dwn" 
    log_path = "data/logs/" + TICKER + "_processing_logs" 
    horizons = np.array([10, 20, 30, 50, 100, 200, 300, 500, 1000]) 

    # # ============================================================================
    # # LOBSTER DATA - ORDERBOOKS

    # output_path = "data/" + TICKER + "_orderbooks_test"

    # startTime = time.time()
    # process_data(TICKER=TICKER, 
    #              input_path=input_path, 
    #              output_path=output_path,
    #              log_path=log_path,
    #              features="orderbooks",
    #              horizons=horizons)
    # executionTime = (time.time() - startTime)

    # print("Orderbooks execution time in minutes: " + str(executionTime/60))

    # # ============================================================================
    # # LOBSTER DATA - ORDERFLOWS

    # output_path = "data/" + TICKER + "_orderflows_test"

    # startTime = time.time()
    # process_data(TICKER=TICKER, 
    #              input_path=input_path, 
    #              output_path=output_path,
    #              log_path=log_path,
    #              features="orderflows",
    #              horizons=horizons)
    # executionTime = (time.time() - startTime)

    # print("Orderflows execution time in minutes: " + str(executionTime/60))

    # # ============================================================================
    # # LOBSTER DATA - VOLUMES (L2)

    # output_path = "data/" + TICKER + "_volumes_test"

    # startTime = time.time()
    # process_data(TICKER=TICKER, 
    #              input_path=input_path, 
    #              output_path=output_path,
    #              log_path=log_path,
    #              features="volumes",
    #              horizons=horizons)
    # executionTime = (time.time() - startTime)

    # print("Volumes execution time in minutes: " + str(executionTime/60))

    ## ============================================================================
    # LOBSTER DATA - VOLUMES L3 (multiprocess)

    output_path = "data/" + TICKER + "_volumes"
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

    # # ================================================================================
    # # VOLUMES - OUTPUT CHECK
    # with np.load('data/WBA_volumes/WBA_volumes_2019-11-05.npz') as data:
    #     L3_orderbook = data['features']
    #     L3_response = data['responses']
    
    # L2 = pd.read_csv('data/WBA_volumes_test/WBA_volumes_2019-11-05.csv')

    # L2_orderbook = L2.iloc[:, :-len(horizons)]
    # L2_response = L2.iloc[:, -len(horizons):]

    # print(len(L2_orderbook))
    # print(len(L3_orderbook))

    # print(L2_orderbook.iloc[210000, :].values)
    # print(np.sum(np.sum(L3_orderbook[:, :, :], axis=1)<0))
