from model import deepOB
from data_methods import get_alphas, get_class_distributions
from config.directories import ROOT_DIR
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

    # select TICKER-period from $PBS_ARRAY_INDEX
    TICKERS = ["LILAK", "QRTEA", "XRAY", "CHTR", "PCAR", "EXC", "AAL", "WBA", "ATVI", "AAPL"]
    Ws = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    TICKER = TICKERS[int(sys.argv[1]) % 10]
    W = Ws[int(sys.argv[1]) // 10]

    # set global parameters
    orderbook_updates = [10, 20, 30, 50, 100, 200, 300, 500, 1000]             
    task = "classification"
    multihorizon = False                              
    decoder = "seq2seq"                                            
    T = 100
    queue_depth = 10                                          
    n_horizons = len(orderbook_updates)
    tot_horizons = 9
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

    # divide 55 weeks into 11 periods, 5 week each
    start_date = dt.date(2019, 1, 14)
    end_date = dt.date(2020, 1, 31)
    slide_by = 5
    dates = [str(start_date + dt.timedelta(days=_)) for _ in range((end_date - start_date).days + 1)]
    weeks = list(zip(*[dates[i::7] for i in range(5)]))

    TICKER_filepath = os.path.join(ROOT_DIR, "results", TICKER)
    os.makedirs(TICKER_filepath, exist_ok=True)

    # select specific window
    for d in range(0, len(weeks), slide_by):
        window = d // slide_by
        if window == W:
            pass
        else:
            continue
        train_val_dates = [date for week in weeks[d:d+4] for date in week]
        test_dates = [date for date in weeks[d+4]]

        # set random seeds
        random.seed(0)
        np.random.seed(1)
        tf.random.set_seed(2)
        
        # random train-val split
        random.shuffle(train_val_dates)
        val_dates = train_val_dates[:5]
        train_dates = train_val_dates[5:]

        window_filepath = os.path.join(TICKER_filepath, "W" + str(window))
        os.makedirs(window_filepath, exist_ok=True)
        pickle.dump([val_dates, train_dates, test_dates], open(os.path.join(window_filepath, "val_train_test_dates.pkl"), "wb"))

        alphas = np.array([])

        # iterate through model types
        for m, model_type in enumerate(model_list):
            model_filepath = os.path.join(window_filepath, model_type)
            os.makedirs(model_filepath, exist_ok=True)
            
            # set local parameters
            features = features_list[m]
            model_inputs = model_inputs_list[m]
            levels = levels_list[m]
            
            data_dir = os.path.join(ROOT_DIR, "data", TICKER + "_" + features)
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
                pickle.dump(alphas, open(os.path.join(window_filepath, "alphas.pkl"), "wb"))
                pickle.dump(distributions, open(os.path.join(window_filepath, "distributions.pkl"), "wb"))

                val_distributions = get_class_distributions(files["val"], alphas, orderbook_updates)
                test_distributions = get_class_distributions(files["test"], alphas, orderbook_updates)
                pickle.dump(val_distributions, open(os.path.join(window_filepath, "val_distributions.pkl"), "wb"))
                pickle.dump(test_distributions, open(os.path.join(window_filepath + "test_distributions.pkl"), "wb"))
            else:
                pass
            imbalances = distributions.to_numpy()
            
            # iterate through horizons
            for h in range(tot_horizons):
                horizon = h
                results_filepath = os.path.join(model_filepath, "h" + str(orderbook_updates[h]))
                checkpoint_filepath = os.path.join(results_filepath, "weights")
                os.makedirs(results_filepath, exist_ok=True)

                # create model
                model = deepOB(T = T, 
                               levels = levels, 
                               horizon = horizon, 
                               number_of_lstm = number_of_lstm,
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
                               tot_horizons = tot_horizons,
                               train_roll_window = train_roll_window,
                               imbalances = imbalances,
                               batch_size = batch_size)

                model.create_model()

                print("training model:", results_filepath)

                # set random seeds
                random.seed(0)
                np.random.seed(1)
                tf.random.set_seed(2)

                # train model
                model.fit_model(epochs = epochs,
                                checkpoint_filepath = checkpoint_filepath,
                                verbose = training_verbose,
                                patience = patience)
                
                print("testing model:", results_filepath)

                # evaluate model
                model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                     eval_set = "test",
                                     results_filepath = results_filepath)
                model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                     eval_set = "train",
                                     results_filepath = results_filepath)
                model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                     eval_set = "val",
                                     results_filepath = results_filepath)
