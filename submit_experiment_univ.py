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
    TICKERS = ["LILAK", "QRTEA", "XRAY", "CHTR", "PCAR", "EXC", "AAL", "WBA", "ATVI", "AAPL"]
    TICKERS_insample = ["QRTEA", "CHTR", "EXC", "WBA", "AAPL"]
    TICKERS_outofsample = ["LILAK", "XRAY", "PCAR", "AAL", "ATVI"]
    Ws = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    window = Ws[int(sys.argv[1])]

    orderbook_updates = [10, 20, 30, 50, 100, 200, 300, 500, 1000]
    data = "LOBSTER"                                                
    task = "classification"
    multihorizon = False                              
    decoder = None                                            
    T = 100
    queue_depth = 10                                          
    n_horizons = len(orderbook_updates)
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

    # make universal training set
    val_dates = {}
    train_dates = {}
    test_dates = {}
    alphas = {}
    imbalances = np.array([])

    for TICKER in TICKERS:
        TICKER_filepath = "results/" + TICKER
        window_filepath = TICKER_filepath + "/W" + str(window)

        val_train_test_dates = pickle.load(open(window_filepath + "/val_train_test_dates.pkl", "rb"))
        alphas[TICKER] = pickle.load(open(window_filepath + "/alphas.pkl", "rb"))
        
        val_dates[TICKER] = val_train_test_dates[0]
        train_dates[TICKER] = val_train_test_dates[1]
        test_dates[TICKER] = val_train_test_dates[2]

    univ_filepath = "results/universal"
    os.makedirs(univ_filepath, exist_ok=True)

    for m, model_type in enumerate(model_list):
        model_filepath = univ_filepath + "/" + model_type
        os.makedirs(model_filepath, exist_ok=True)
        
        # set local parameters
        features = features_list[m]
        model_inputs = model_inputs_list[m]
        levels = levels_list[m]

        val_files_dict = {}
        train_files_dict = {}
        test_files_dict = {}

        for TICKER in TICKERS_insample:        
            data_dir = "data/" + TICKER + "_" + features
            file_list = os.listdir(data_dir)
            val_files_dict[TICKER] = [os.path.join(data_dir, file) for date in val_dates for file in file_list if date in file]
            train_files_dict[TICKER] = [os.path.join(data_dir, file) for date in val_dates for file in file_list if date in file]
            test_files_dict[TICKER] = [os.path.join(data_dir, file) for date in val_dates for file in file_list if date in file]

        files = {
            "val": val_files_dict,
            "train": train_files_dict,
            "test": test_files_dict
        }

        # TODO: now, test results on out of sample tickers // reload models and load trained weights
        val_files_dict = {}
        train_files_dict = {}
        test_files_dict = {}

        for TICKER in TICKERS_outofsample:        
            data_dir = "data/" + TICKER + "_" + features
            file_list = os.listdir(data_dir)
            val_files_dict[TICKER] = [os.path.join(data_dir, file) for date in val_dates for file in file_list if date in file]
            train_files_dict[TICKER] = [os.path.join(data_dir, file) for date in val_dates for file in file_list if date in file]
            test_files_dict[TICKER] = [os.path.join(data_dir, file) for date in val_dates for file in file_list if date in file]

        files_outofsample = {
            "val": val_files_dict,
            "train": train_files_dict,
            "test": test_files_dict
        }
        #==================

        if imbalances.size == None:
            distributions = get_class_distributions_univ(files["train"], alphas, orderbook_updates)   
            imbalances = distributions.to_numpy()
        
        for h in range(n_horizons):
            horizon = h
            results_filepath = model_filepath + "/" + "h" + str(orderbook_updates[h])
            checkpoint_filepath = results_filepath + "/" + "weights"
            os.makedirs(results_filepath, exist_ok=True)

            model = deepLOB(T = T, 
                            levels = levels, 
                            horizon = horizon, 
                            number_of_lstm = number_of_lstm, 
                            data = data, 
                            data_dir = data_dir, 
                            files = files, 
                            model_inputs = model_inputs, 
                            queue_depth = queue_depth,
                            task = task, 
                            alphas = alphas, 
                            orderbook_updates = orderbook_updates,
                            multihorizon = multihorizon, 
                            decoder = decoder, 
                            n_horizons = n_horizons,
                            train_roll_window = train_roll_window,
                            imbalances = imbalances,
                            universal = True)

            model.create_model()

            print("training model:", results_filepath)

            model.fit_model(epochs = epochs,
                            checkpoint_filepath = checkpoint_filepath,
                            verbose = training_verbose,
                            batch_size = batch_size,
                            patience = patience)
            
            print("testing model:", results_filepath)

            results_filepath = results_filepath + "/" + "TICKERS_in_sample"

            model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                eval_set = "test",
                                results_filepath = results_filepath)
            model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                eval_set = "train",
                                results_filepath = results_filepath)
            model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                eval_set = "val",
                                results_filepath = results_filepath)

            # TODO: now, test results on stock by stock

