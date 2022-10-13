from model import deepOB
from data_methods import get_class_distributions_univ
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

    # select period-horizons from $PBS_ARRAY_INDEX
    TICKERS = ["LILAK", "QRTEA", "XRAY", "CHTR", "PCAR", "EXC", "AAL", "WBA", "ATVI", "AAPL"]
    TICKERS_insample = ["QRTEA", "CHTR", "EXC", "WBA", "AAPL"]
    TICKERS_outofsample = ["LILAK", "XRAY", "PCAR", "AAL", "ATVI"]
    Ws = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    window = Ws[int(sys.argv[1])//3]
    indicator_range = int(sys.argv[1])%3

    # set global parameters
    orderbook_updates = [10, 20, 30, 50, 100, 200, 300, 500, 1000]
    n_horizons = len(orderbook_updates)
    tot_horizons = 9
    range_models = range(indicator_range*3, (indicator_range+1)*3)                                              
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

    # divide 55 weeks into 11 periods, 5 week each
    start_date = dt.date(2019, 1, 14)
    end_date = dt.date(2020, 1, 31)
    slide_by = 5
    dates = [str(start_date + dt.timedelta(days=_)) for _ in range((end_date - start_date).days + 1)]
    weeks = list(zip(*[dates[i::7] for i in range(5)]))

    univ_filepath = "results/universal"
    os.makedirs(univ_filepath, exist_ok=True)
    window_filepath = univ_filepath + "/W" + str(window)
    os.makedirs(window_filepath, exist_ok=True)

    # make universal dicts from stock specific train, val and test dates, alphas and distributions
    val_dates = {}
    train_dates = {}
    test_dates = {}
    alphas = {}
    imbalances = np.array([])

    for TICKER in TICKERS:
        TICKER_filepath = "results/" + TICKER
        TICKER_window_filepath = TICKER_filepath + "/W" + str(window)

        val_train_test_dates = pickle.load(open(TICKER_window_filepath + "/val_train_test_dates.pkl", "rb"))
        alphas[TICKER] = pickle.load(open(TICKER_window_filepath + "/alphas.pkl", "rb"))
        
        val_dates[TICKER] = val_train_test_dates[0]
        train_dates[TICKER] = val_train_test_dates[1]
        test_dates[TICKER] = val_train_test_dates[2]

    pickle.dump([val_dates, train_dates, test_dates], open(window_filepath + "/val_train_test_dates.pkl", "wb"))
    pickle.dump(alphas, open(window_filepath + "/alphas.pkl", "wb"))

    # iterate through model types
    for m, model_type in enumerate(model_list):
        model_filepath = window_filepath + "/" + model_type
        os.makedirs(model_filepath, exist_ok=True)
        
        # set local parameters
        features = features_list[m]
        model_inputs = model_inputs_list[m]
        levels = levels_list[m]

        # make "in-sample" universal dicts
        val_files_dict = {}
        train_files_dict = {}
        test_files_dict = {}
        train_val_dict = {}

        for TICKER in TICKERS_insample:        
            data_dir = "data/" + TICKER + "_" + features
            file_list = os.listdir(data_dir)
            val_files_dict[TICKER] = [os.path.join(data_dir, file) for date in val_dates[TICKER] for file in file_list if date in file]
            train_files_dict[TICKER] = [os.path.join(data_dir, file) for date in train_dates[TICKER] for file in file_list if date in file]
            test_files_dict[TICKER] = [os.path.join(data_dir, file) for date in test_dates[TICKER] for file in file_list if date in file]
            train_val_dict[TICKER] = val_files_dict[TICKER] + train_files_dict[TICKER]

        files = {
            "val": val_files_dict,
            "train": train_files_dict,
            "test": test_files_dict
        }

        # make "out-of-sample" universal dicts
        val_files_dict = {}
        train_files_dict = {}
        test_files_dict = {}

        for TICKER in TICKERS_outofsample:        
            data_dir = "data/" + TICKER + "_" + features
            file_list = os.listdir(data_dir)
            val_files_dict[TICKER] = [os.path.join(data_dir, file) for date in val_dates[TICKER] for file in file_list if date in file]
            train_files_dict[TICKER] = [os.path.join(data_dir, file) for date in train_dates[TICKER] for file in file_list if date in file]
            test_files_dict[TICKER] = [os.path.join(data_dir, file) for date in test_dates[TICKER] for file in file_list if date in file]

        files_outofsample = {
            "val": val_files_dict,
            "train": train_files_dict,
            "test": test_files_dict
        }

        # on first iteration, compute universal distributions
        if imbalances.size == 0:
            distributions = get_class_distributions_univ(files["train"], alphas, orderbook_updates)   
            val_distributions = get_class_distributions_univ(files["val"], alphas, orderbook_updates)
            test_distributions = get_class_distributions_univ(files["test"], alphas, orderbook_updates)
            train_val_distributions = get_class_distributions_univ(train_val_dict, alphas, orderbook_updates)
            
            imbalances = distributions.to_numpy()

            pickle.dump(distributions, open(window_filepath + "/distributions.pkl", "wb"))
            pickle.dump(val_distributions, open(window_filepath + "/val_distributions.pkl", "wb"))
            pickle.dump(test_distributions, open(window_filepath + "/test_distributions.pkl", "wb"))
            pickle.dump(train_val_distributions, open(window_filepath + "/train_val_distributions.pkl", "wb"))
        else:
            pass
        
        data_dir = "data"

        # iterate through horizons
        for h in range_models:
            horizon = h
            results_filepath = model_filepath + "/" + "h" + str(orderbook_updates[h])
            checkpoint_filepath = results_filepath + "/" + "weights"
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
                           batch_size = batch_size,
                           universal = True)

            model.create_model()

            print("training model:", results_filepath)

            # set random seeds
            random.seed(0)
            np.random.seed(1)
            tf.random.set_seed(2)

            # fit model
            model.fit_model(epochs = epochs,
                            checkpoint_filepath = checkpoint_filepath,
                            verbose = training_verbose,
                            patience = patience)

            results_filepath_insample = results_filepath + "/" + "TICKERS_in_sample"
            os.makedirs(results_filepath_insample, exist_ok=True)

            print("testing model in sample:", results_filepath_insample)

            # evaluate "in-sample"
            model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                eval_set = "test",
                                results_filepath = results_filepath_insample)
            model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                eval_set = "train",
                                results_filepath = results_filepath_insample)
            model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                eval_set = "val",
                                results_filepath = results_filepath_insample)

            results_filepath_outofsample = results_filepath + "/" + "TICKERS_out_of_sample"
            os.makedirs(results_filepath_outofsample, exist_ok=True)

            # evaluate "out-of-sample": need to re-create model with different datasets (but do not train again!)
            model = deepOB(T = T, 
                           levels = levels, 
                           horizon = horizon, 
                           number_of_lstm = number_of_lstm,
                           data_dir = data_dir, 
                           files = files_outofsample, 
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
                           batch_size = batch_size,
                           universal = True)
            
            model.create_model()

            print("testing model out of sample:", results_filepath_outofsample)

            model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                eval_set = "test",
                                results_filepath = results_filepath_outofsample)
            model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                eval_set = "train",
                                results_filepath = results_filepath_outofsample)
            model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                eval_set = "val",
                                results_filepath = results_filepath_outofsample)

            # evaluate on each TICKER: need to re-create model with different datasets (but do not train again!)
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

                model = deepOB(T = T, 
                               levels = levels, 
                               horizon = horizon, 
                               number_of_lstm = number_of_lstm,
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
                               tot_horizons = tot_horizons,
                               train_roll_window = train_roll_window,
                               imbalances = imbalances,
                               batch_size = batch_size,
                               universal = False)
                
                model.create_model()

                print("testing model on single stock:", results_filepath_stock)

                model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                    eval_set = "test",
                                    results_filepath = results_filepath_stock)
                model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                    eval_set = "train",
                                    results_filepath = results_filepath_stock)
                model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                                    eval_set = "val",
                                    results_filepath = results_filepath_stock)



