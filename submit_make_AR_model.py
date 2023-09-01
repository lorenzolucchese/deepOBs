from AR_model_results import make_empirical_AR_model
from config.directories import ROOT_DIR
import os
import time

if __name__ == "__main__":
    # set global parameters
    tickers = ['LILAK', 'QRTEA', 'XRAY', 'CHTR', 'PCAR', 'EXC', 'AAL', 'WBA', 'ATVI', 'AAPL']
    horizons = [10, 20, 30, 50, 100, 200, 300, 500, 1000]
    periods = ['W' + str(i) for i in range(11)]
    processed_data_path = os.path.join(ROOT_DIR, "data")
    stats_path = os.path.join(ROOT_DIR, "data", "stats")
    results_path = os.path.join(ROOT_DIR, "results")

    os.makedirs(stats_path, exist_ok=True)

    # ============================================================================
    startTime = time.time()
    make_empirical_AR_model(tickers=tickers,
                            periods=periods,
                            horizons=horizons,
                            processed_data_path = processed_data_path, 
                            results_path = results_path,
                            stats_path = stats_path)
    executionTime = (time.time() - startTime)

    print("Execution time in minutes: " + str(executionTime/60))
    
