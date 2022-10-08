from importlib.metadata import distribution
from model import deepLOB
from data_prepare import get_alphas, get_class_distributions, get_class_distributions_univ
import datetime as dt
import sys
import numpy as np
import pickle
import os
import random
import tensorflow as tf

if __name__ == "__main__":
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

    # set random seeds
    random.seed(0)
    np.random.seed(1)
    tf.random.set_seed(2)

    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # set global parameters
    # TICKERS = ["LILAK", "QRTEA", "XRAY", "CHTR", "PCAR", "EXC", "AAL", "WBA", "ATVI", "AAPL"]
    TICKERS = ["AAL"]
    TICKERS_insample = ["QRTEA", "CHTR", "EXC", "WBA", "AAPL"]
    TICKERS_outofsample = ["LILAK", "XRAY", "PCAR", "AAL", "ATVI"]
    Ws = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    orderbook_updates = [10, 20, 30, 50, 100, 200, 300, 500, 1000]
    n_horizons = len(orderbook_updates)
    window = 0
    data = "LOBSTER"                                                
    task = "classification"
    multihorizon = False                              
    decoder = None                                            
    T = 100
    queue_depth = 10                                          
    epochs = 50
    patience = 10
    training_verbose = 2
    train_roll_window = 10
    batch_size = 256
    number_of_lstm = 64
    model_list = ["deepLOB_L1", "deepOF_L1", "deepLOB_L2", "deepOF_L2", "deepVOL_L2", "deepVOL_L3"]
    features_list = ["orderbooks", "orderflows", "orderbooks", "orderflows", "volumes", "volumes"]
    model_inputs_list = ["orderbooks", "orderflows", "orderbooks", "orderflows", "volumes", "volumes_L3"]
    levels_list = [1, 1, 10, 10, 10, 10]

    start_date = dt.date(2019, 1, 14)
    end_date = dt.date(2020, 1, 31)
    slide_by = 5
    dates = [str(start_date + dt.timedelta(days=_)) for _ in range((end_date - start_date).days + 1)]
    weeks = list(zip(*[dates[i::7] for i in range(5)]))
    
    # set local parameters
    features = "orderflows"
    model_inputs = "orderflows"
    levels = 1

    distributions = pickle.load(open("results/universal/W0/distributions.pkl", "rb"))
    imbalances = distributions.to_numpy()
    alphas = pickle.load(open("results/universal/W0/alphas.pkl", "rb"))
    val_train_test_dates = pickle.load(open("results/universal/W0/val_train_test_dates.pkl", "rb"))
    val_dates = val_train_test_dates[0]
    train_dates = val_train_test_dates[1]
    test_dates = val_train_test_dates[2]

    print(alphas['AAL'])
    
    data_dir = "data"
    
    horizon = 0

    checkpoint_filepath = "results/universal/W0/deepOF_L1/h10/weights"
    results_filepath = "results_elixir/universal/W0/deepOF_L1/h10"
    os.makedirs(results_filepath, exist_ok=True)

    # test results on each stock

    for TICKER in TICKERS:        
        data_dir = "data/" + TICKER + "_" + features
        file_list = os.listdir(data_dir)
        val_files_dict = [os.path.join(data_dir, file) for date in val_dates[TICKER] for file in file_list if date in file]
        train_files_dict = [os.path.join(data_dir, file) for date in train_dates[TICKER] for file in file_list if date in file]
        test_files_dict = [os.path.join(data_dir, file) for date in test_dates[TICKER] for file in file_list if date in file]

        files_stock = {
            "val": val_files_dict,
            "train": train_files_dict,
            "test": test_files_dict
        }

        results_filepath_stock = results_filepath + "/" + TICKER
        os.makedirs(results_filepath_stock, exist_ok=True)

        model = deepLOB(T = T, 
                        levels = levels, 
                        horizon = horizon, 
                        number_of_lstm = number_of_lstm, 
                        data = data, 
                        data_dir = data_dir, 
                        files = files_stock, 
                        model_inputs = model_inputs, 
                        queue_depth = queue_depth,
                        task = task, 
                        alphas = alphas[TICKER], 
                        orderbook_updates = orderbook_updates,
                        multihorizon = multihorizon, 
                        decoder = decoder, 
                        n_horizons = n_horizons,
                        train_roll_window = train_roll_window,
                        imbalances = imbalances,
                        universal = False)
        
        model.create_model()

        model.model.load_weights(checkpoint_filepath).expect_partial()

        print("testing model on single stock:", results_filepath_stock)

        print("results obtained via built-in function model.evaluate:")
        print("test set:", model.model.evaluate(model.test_generator))
        print("train set:", model.model.evaluate(model.train_generator))
        print("val set:", model.model.evaluate(model.val_generator))

        model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                            eval_set = "test",
                            results_filepath = results_filepath_stock)
        model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                            eval_set = "train",
                            results_filepath = results_filepath_stock)
        model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                            eval_set = "val",
                            results_filepath = results_filepath_stock)



