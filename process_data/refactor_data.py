import os
import glob
import pandas as pd
import re
import datetime
import numpy as np
import re
from multiprocessing import Pool
from config.directories import ROOT_DIR

def refactor_data(TICKER, delete=False):
    logs = []
    # paths
    orderbooks_dir = os.path.join(ROOT_DIR, "data", TICKER + "_orderbooks")
    orderflows_dir = os.path.join(ROOT_DIR, "data", TICKER + "_orderflows")
    volumes_dir = os.path.join(ROOT_DIR, "data", TICKER + "_volumes")
    output_path = os.path.join(ROOT_DIR, "data", TICKER)
    # files
    orderbooks_file_list = glob.glob(os.path.join(orderbooks_dir, "*.{}".format("csv")))
    orderflows_file_list = glob.glob(os.path.join(orderflows_dir, "*.{}".format("csv")))
    volumes_file_list = glob.glob(os.path.join(volumes_dir, "*.{}".format("npz")))
    dates = [re.split('_|\.', file_name)[-2] for file_name in orderbooks_file_list]
    for date in dates:
        try:
            # get filename for date
            orderbook_file = [file_name for file_name in orderbooks_file_list if date in file_name][0]
            orderflow_file = [file_name for file_name in orderflows_file_list if date in file_name][0]
            volume_file = [file_name for file_name in volumes_file_list if date in file_name][0]
            # load orderbook data
            orderbook_load = pd.read_csv(orderbook_file)
            orderbook_features = orderbook_load.loc[:, [feature for feature in orderbook_load.columns if ('ASK' in feature or 'BID' in feature)]].values
            orderbook_responses = orderbook_load.loc[:, [feature for feature in orderbook_load.columns if not ('ASK' in feature or 'BID' in feature)]].values
            # load orderflow data
            orderflow_load = pd.read_csv(orderflow_file)
            orderflow_features = orderflow_load.loc[:, [feature for feature in orderflow_load.columns if ('ASK' in feature or 'BID' in feature)]].values
            orderflow_responses = orderflow_load.loc[:, [feature for feature in orderflow_load.columns if not ('ASK' in feature or 'BID' in feature)]].values
            # load volume data
            volume_load = np.load(volume_file)
            volume_features, volume_responses = volume_load['features'], volume_load['responses']
            # note the arrays match based on index as follows: 
            # orderflow starts one index after orderbook (as it is a diff), while volume has the same index as orderbook
            # thus drop first lines in orderbooks and volumes
            orderbook_features, orderbook_responses = orderbook_features[1:], orderbook_responses[1:]
            volume_features, volume_responses = volume_features[1:], volume_responses[1:] 
            # sanity check (some numerical error is ok)
            assert(np.allclose(orderbook_responses, orderflow_responses))
            assert(np.allclose(orderbook_responses, volume_responses))
            # save as single .npz file
            output_name = os.path.join(output_path, TICKER + "_" + "data" + "_" + date)
            np.savez(output_name + ".npz", orderbook_features=orderbook_features, orderflow_features=orderflow_features, volume_features=volume_features, responses=orderbook_responses)
        except Exception as e: 
            logs.append('An exception occured, for the following TICKER ' + TICKER + ' and date ' + date + '.')
            logs.append(str(e) + '.')
        else:
            del volume_load, volume_features, volume_responses
            if delete:
                os.remove(orderbook_file)
                os.remove(orderflow_file)
                os.remove(volume_file)
            else:
                pass

    with open(os.path.join(ROOT_DIR, "data", "logs", TICKER + "_processing_logs.txt"), "w") as f:
        for log in logs:
            f.write(log + "\n")

    print("please check processing logs.")

    
if __name__ == "__main__":
    TICKERs = ["LILAK", "QRTEA", "XRAY", "CHTR", "PCAR", "EXC", "AAL", "WBA", "ATVI", "AAPL"]
    for TICKER in TICKERs:
        os.makedirs(os.path.join(ROOT_DIR, "data", TICKER), exist_ok=True)
        refactor_data(TICKER, delete=True)

