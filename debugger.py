import numpy as np
import pandas as pd
import pickle
import glob
import re
import os
import random
import tensorflow as tf
from keras.layers import Reshape, concatenate
from custom_datasets import CustomtfDataset
from data_methods import get_alphas
from model import deepOB, linearOB
from config.directories import ROOT_DIR
import datetime as dt

if __name__ == '__main__':
    # limit gpu memory
    visible_gpus = tf.config.experimental.get_visible_devices("GPU")
    physical_gpus = tf.config.experimental.list_physical_devices("GPU")
    print("This machine has", len(visible_gpus), "visible gpus.")
    print("This machine has", len(physical_gpus), "physical gpus.")
    if visible_gpus:
        try:
            # Use only one GPUs
            tf.config.set_visible_devices(visible_gpus[0], "GPU")
            logical_gpus = tf.config.list_logical_devices("GPU")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # select TICKER
    TICKER = "WBA"

    # set global parameters
    orderbook_updates = [10]             
    task = "classification"
    multihorizon = False                              
    decoder = "seq2seq"                                            
    T = 100
    queue_depth = 10                                          
    n_horizons = len(orderbook_updates)
    epochs = 50
    patience = 10
    training_verbose = 2
    testing_verbose = 1
    train_roll_window = 1000
    batch_size = 256
    number_of_lstm = 64
    model_list = ["linearLOB_L1", "linearOF_L1", "linearLOB_L2", "linearOF_L2", "linearVOL_L2", "linearVOL_L3"]
    features_list = ["orderbooks", "orderflows", "orderbooks", "orderflows", "volumes", "volumes"]
    model_inputs_list = ["orderbooks", "orderflows", "orderbooks", "orderflows", "volumes", "volumes_L3"]
    levels_list = [1, 1, 10, 10, 10, 10]

    TICKER_filepath = os.path.join(ROOT_DIR, "results", "debugger", TICKER)
    os.makedirs(TICKER_filepath, exist_ok=True)
    window_filepath = os.path.join(TICKER_filepath, "W8")

    # select specific window
    train_dates = ["2019-11-05", "2019-11-06"]
    val_dates = ["2019-11-05", "2019-11-06"]
    test_dates = ["2019-11-05", "2019-11-06"]

    # set random seeds
    random.seed(0)
    np.random.seed(1)
    tf.random.set_seed(2)
    
    alphas = pickle.load(open("results/WBA/W8/alphas.pkl", "rb"))

    # iterate through model types
    for m, model_type in enumerate(model_list):
        model_filepath = os.path.join(window_filepath, model_type)
        os.makedirs(model_filepath, exist_ok=True)
        
        # set local parameters
        features = features_list[m]
        model_inputs = model_inputs_list[m]
        levels = levels_list[m]
        
        data_dir = os.path.join(ROOT_DIR, "data", TICKER)
        file_list = [file for file in os.listdir(data_dir) if '.npz' in file]
        print(file_list)
        files = {
            "train": [os.path.join(data_dir, file) for date in test_dates for file in file_list if date in file],
            "val": [os.path.join(data_dir, file) for date in test_dates for file in file_list if date in file],
            "test": [os.path.join(data_dir, file) for date in test_dates for file in file_list if date in file]
        }
        
        distributions = pickle.load(open("results/WBA/W8/distributions.pkl", "rb"))
        imbalances = distributions.to_numpy()
        
        # iterate through horizons
        for h in range(n_horizons):
            horizon = h
            results_filepath = os.path.join(model_filepath, "h" + str(orderbook_updates[h]))
            checkpoint_filepath = os.path.join(model_filepath, "weights")
            os.makedirs(results_filepath, exist_ok=True)

            # create model
            model = linearOB(T = T, 
                            levels = levels, 
                            horizon = horizon,
                            data_dir = data_dir, 
                            files = files, 
                            model_inputs = model_inputs, 
                            queue_depth = queue_depth,
                            task = task, 
                            alphas = alphas, 
                            orderbook_updates = orderbook_updates,
                            train_roll_window = train_roll_window,
                            imbalances = imbalances,
                            batch_size = batch_size)
            
            model.fit_model(epochs=epochs,
                            checkpoint_filepath=checkpoint_filepath,
                            load_weights=False,
                            load_weights_filepath=None,
                            verbose=testing_verbose,
                            patience=10,
                            CV_l1=10.**np.arange(-8, -6))

            # evaluate model
            model.evaluate_model(load_weights_filepath = checkpoint_filepath,
                                 eval_set = "train",
                                 results_filepath = results_filepath,
                                 verbose = testing_verbose)

    # TICKER = "WBA"
    # n = 5
    # aggregated_stats = pd.read_csv("data/WBA/stats/WBA_orderbook_stats.csv", index_col=[0, 1])
    # dates = aggregated_stats.index.levels[0]
    # standardizations = pd.DataFrame(index = pd.MultiIndex.from_product([dates, ["mean", "std", "count"]], names=['Date', 'stat']), columns = aggregated_stats.columns)
    # for i, date in enumerate(dates):
    #     means = aggregated_stats.xs("mean", level=1).loc[dates[i-n:i], :].copy()
    #     stds = aggregated_stats.xs("std", level=1).loc[dates[i-n:i], :].copy()
    #     counts = aggregated_stats.xs("count", level=1).loc[dates[i-n:i], :].copy()
    #     # total count
    #     count = counts.sum(axis=0)
    #     # aggregate mean
    #     mean = (means * counts).sum(axis=0) / count
    #     # aggregate std
    #     std = np.sqrt((((counts - 1)*stds**2 + counts*means**2).sum(axis=0) - count*mean**2) / (count - 1))
    #     standardizations.loc[(date, "mean")] = mean.values
    #     standardizations.loc[(date, "std")] = std.values
    #     standardizations.loc[(date, "count")] = count.values
    
    # print(standardizations.loc["2019-11-05"])
    # print(re.search(r'\d{4}-\d{2}-\d{2}', "afwedsx_2019-11-05.csv").group(0))