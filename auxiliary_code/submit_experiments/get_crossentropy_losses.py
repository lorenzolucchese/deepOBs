from model import deepOB
from data_methods import get_alphas, get_class_distributions
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
    Ws = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    TICKER = TICKERS[int(sys.argv[1])]

    orderbook_updates = [10, 20, 30, 50, 100, 200, 300, 500, 1000]
    data = "LOBSTER"                                                
    task = "classification"
    multihorizon = False                              
    decoder = "seq2seq"                                            
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

    TICKER_filepath = "results/" + TICKER
    os.makedirs(TICKER_filepath, exist_ok=True)

    for d in range(0, len(weeks), slide_by):
        window = d // slide_by

        window_filepath = TICKER_filepath + "/W" + str(window)
        os.makedirs(window_filepath, exist_ok=True)
        val_train_test_dates = pickle.load(open(window_filepath + "/val_train_test_dates.pkl", "rb"))
        val_dates = val_train_test_dates[0]
        train_dates = val_train_test_dates[1]
        test_dates = val_train_test_dates[2]

        alphas = np.array([])

        for m, model_type in enumerate(model_list):
            model_filepath = window_filepath + "/" + model_type
            os.makedirs(model_filepath, exist_ok=True)
            
            # set local parameters
            features = features_list[m]
            model_inputs = model_inputs_list[m]
            levels = levels_list[m]
            
            data_dir = "data/" + TICKER + "_" + features
            file_list = os.listdir(data_dir)
            files = {
                "val": [os.path.join(data_dir, file) for date in val_dates for file in file_list if date in file],
                "train": [os.path.join(data_dir, file) for date in train_dates for file in file_list if date in file],
                "test": [os.path.join(data_dir, file) for date in test_dates for file in file_list if date in file]
            }
            
            # on first iteration compute alphas
            if alphas.size == 0:
                print("getting alphas...")
                alphas, distributions = get_alphas(files["train"], orderbook_updates)
                pickle.dump(alphas, open(window_filepath + "/alphas.pkl", "wb"))
                pickle.dump(distributions, open(window_filepath + "/distributions.pkl", "wb"))

                val_distributions = get_class_distributions(files["val"], alphas, orderbook_updates)
                test_distributions = get_class_distributions(files["test"], alphas, orderbook_updates)
                pickle.dump(val_distributions, open(window_filepath + "/val_distributions.pkl", "wb"))
                pickle.dump(test_distributions, open(window_filepath + "/test_distributions.pkl", "wb"))
            else:
                pass
            imbalances = distributions.to_numpy()
            
            for h in range(n_horizons):
                horizon = h
                results_filepath = model_filepath + "/" + "h" + str(orderbook_updates[h])
                checkpoint_filepath = results_filepath + "/" + "weights"
                os.makedirs(results_filepath, exist_ok=True)

                model = deepOB(T = T, 
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
                                imbalances = imbalances)

                model.create_model()
                
                print("testing model:", results_filepath)

                model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                    eval_set = "test",
                                    results_filepath = results_filepath)
                model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                    eval_set = "train",
                                    results_filepath = results_filepath)
                model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                    eval_set = "val",
                                    results_filepath = results_filepath)
                                    
                tf.keras.backend.clear_session()
