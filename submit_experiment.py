# from model import deepLOB
import datetime as dt
import sys
import numpy as np
import pandas as pd
import glob
import os
import random

if __name__ == "__main__":
    # set global parameters
    TICKERS = ["LILAK", "QRTEA", "XRAY", "CHTR", "PCAR", "EXC", "AAL", "WBA", "ATVI", "AAPL"]

    # TICKER = TICKERS[int(sys.argv[1])]
    TICKER = "AAL"
    model_list = ["deepLOB_L1", "deepOF_L1", "deepLOB_L2", "deepOF_L2", "deepVOL_L2", "deepVOL_L3"]
    features_list = ["orderbooks", "orderflows", "orderbooks", "orderflows", "volumes", "volumes"]
    model_inputs_list = ["orderbooks", "orderflows", "orderbooks", "orderflows", "volumes", "volumes_L3"]
    levels = [1, 1, 10, 10, 10, 10]

    horizons = np.array([10, 20, 30, 50, 100, 200, 300, 500, 1000])
    orderbook_updates = [10, 20, 30, 50, 100, 200, 300, 500, 1000]

    start_date = dt.date(2019, 1, 14)
    end_date = dt.date(2020, 1, 31)
    dates = [str(start_date + dt.timedelta(days=_)) for _ in range((end_date - start_date).days + 1)]
    weeks = list(zip(*[dates[i::7] for i in range(5)]))

    for i in range(0, len(weeks), 5):
        train_val_dates = [date for week in weeks[i:i+4] for date in week]
        test_dates = [date for date in weeks[i+4]]

        for i, model in enumerate(model_list):
            #################################### SETTINGS ########################################
            features = features_list[i]
            model_inputs = model_inputs_list[i]           # options: "orderbooks", "orderflows", "volumes", "volumes_L3"
            data = "LOBSTER"                              # options: "FI2010", "LOBSTER", "simulated"
            data_dir = "data/" + TICKER + "_" + features
            file_list = os.listdir(data_dir)
            val_train_list = [file for date in train_val_dates for file in file_list if date in file]
            random.shuffle(val_train_list)
            test_list = [file for date in train_val_dates for file in file_list if date in file]
            print(val_train_list)
            print(test_list)
            files = {
                "val": val_train_list[:5],
                "train": val_train_list[5:25],
                "test": test_list
            }
            # alphas, distributions = get_alphas(files["train"])
            alphas = np.array([0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 3.2942814e-05])
            distributions = pd.DataFrame(np.vstack([np.array([0.121552, 0.194825, 0.245483, 0.314996, 0.334330]), 
                                                    np.array([0.752556, 0.604704, 0.504695, 0.368647, 0.330456]),
                                                    np.array([0.125893, 0.200471, 0.249821, 0.316357, 0.335214])]), 
                                        index=["down", "stationary", "up"], 
                                        columns=["10", "20", "30", "50", "100"])
            imbalances = distributions.to_numpy()
            # imbalances = None
            task = "classification"
            multihorizon = True                         # options: True, False
            decoder = "seq2seq"                         # options: "seq2seq", "attention"

            T = 100
            levels = 1                                  # remember to change this when changing models
            queue_depth = 10                            # for L3 data only
            n_horizons = 5
            horizon = 0                                 # prediction horizon (0, 1, 2, 3, 4, 5, 6, 7) -> (10, 20, 30, 50, 100, 200, 300, 500, 1000) orderbook events
            epochs = 50
            patience = 10
            training_verbose = 2
            train_roll_window = 100
            batch_size = 256                            # note we use 256 for LOBSTER, 32 for FI2010 or simulated
            number_of_lstm = 64

            checkpoint_filepath = './model_weights/test_model'
            load_weights = False
            load_weights_filepath = './model_weights/test_model'

            break
        break

            #######################################################################################

            # model = deepLOB(T = T, 
            #                 levels = levels, 
            #                 horizon = horizon, 
            #                 number_of_lstm = number_of_lstm, 
            #                 data = data, 
            #                 data_dir = data_dir, 
            #                 files = files, 
            #                 model_inputs = model_inputs, 
            #                 queue_depth = queue_depth,
            #                 task = task, 
            #                 alphas = alphas, 
            #                 multihorizon = multihorizon, 
            #                 decoder = decoder, 
            #                 n_horizons = n_horizons,
            #                 train_roll_window = train_roll_window,
            #                 imbalances = imbalances)

            # model.create_model()

            # model.fit_model(epochs = epochs,
            #                 checkpoint_filepath = checkpoint_filepath,
            #                 load_weights = load_weights,
            #                 load_weights_filepath = load_weights_filepath,
            #                 verbose = training_verbose,
            #                 batch_size = batch_size,
            #                 patience = patience)

            # model.evaluate_model(load_weights_filepath = load_weights_filepath, eval_set = "test")
            # model.evaluate_model(load_weights_filepath = load_weights_filepath, eval_set = "train")
            # model.evaluate_model(load_weights_filepath = load_weights_filepath, eval_set = "val")