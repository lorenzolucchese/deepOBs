from importlib.metadata import distribution
from model import deepLOB
from data_prepare import get_alphas, get_class_distributions
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
    TICKER = "CHTR"

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
    model_list = ["deepVOL_L2", "deepVOL_L3"]
    features_list = ["volumes", "volumes"]
    model_inputs_list = ["volumes", "volumes_L3"]
    levels_list = [10, 10]
    NF = 20

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

        alphas = pickle.load(open(window_filepath + "/alphas.pkl", "rb"))
        distributions = pickle.load(open(window_filepath + "/distributions.pkl", "rb"))

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
            
            imbalances = distributions.to_numpy()
            
            for h in range(n_horizons):
                horizon = h
                results_filepath = model_filepath + "/" + "h" + str(orderbook_updates[h])
                checkpoint_filepath = results_filepath + "/" + "weights"
                os.makedirs(results_filepath, exist_ok=True)

                for file in files:
                    dataset = np.load(file)

                    features = dataset['features']
                    if model_inputs == "volumes":
                        features = np.sum(features, axis = 2)
                    D = features.shape[1]
                    features = features[:, (D//2 - NF//2):(D//2 + NF//2)]
                    features = np.expand_dims(features, axis=-1)
                    features[features > 65535] = 65535
                    features = tf.convert_to_tensor(features, dtype=tf.uint16)

                    if np.isnan(features).any():
                        print(file)
                        print("number of rows:", features.shape[0])
                        print("NaNs are present at the following row indices:")
                        print(np.isnan(features).any(axis=1).nonzero())
                        print("These are the rows where NaNs are present:")
                        print(features[np.isnan(features).any(axis=1)])
                    
                    responses = dataset['responses'][(window-1):, horizon]

                # model = deepLOB(T = T, 
                #         levels = levels, 
                #         horizon = horizon, 
                #         number_of_lstm = number_of_lstm, 
                #         data = data, 
                #         data_dir = data_dir, 
                #         files = files, 
                #         model_inputs = model_inputs, 
                #         queue_depth = queue_depth,
                #         task = task, 
                #         alphas = alphas, 
                #         orderbook_updates = orderbook_updates,
                #         multihorizon = multihorizon, 
                #         decoder = decoder, 
                #         n_horizons = n_horizons,
                #         train_roll_window = train_roll_window,
                #         imbalances = imbalances)

                # model.create_model()
                
                # tot_samples = 0
                # nan_samples = 0
                # for x, y in model.test_generator:
                #     tot_samples += 1
                #     if np.isnan(x).all():
                #         nan_samples += 1
                #         print(nan_samples)
                # print('In test set there are', nan_samples, 'NaN samples out of', tot_samples, 'total_samples')
                
                # print("testing model:", results_filepath)

                # model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                #                     eval_set = "test",
                #                     results_filepath = results_filepath)
                # predY = model.predY
                # print("there are ", np.count_nonzero(np.isnan(predY)), " NaN values in test predY")
                # print("there are ", np.count_nonzero(predY == 0), " 0 values in test predY")
                # evalY = model.evalY
                # print("there are ", np.count_nonzero(np.isnan(evalY)), " NaN values in test evalY")
                # print("there are ", np.count_nonzero(evalY == 0), " 0 values in test evalY")
                
                # model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                #                     eval_set = "train",
                #                     results_filepath = results_filepath)
                # predY = model.predY
                # print("there are ", np.count_nonzero(np.isnan(predY)), " NaN values in train predY")
                # print("there are ", np.count_nonzero(predY == 0), " 0 values in train predY")
                # evalY = model.evalY
                # print("there are ", np.count_nonzero(np.isnan(evalY)), " NaN values in train evalY")
                # print("there are ", np.count_nonzero(evalY == 0), " 0 values in train evalY")

                # model.evaluate_model(load_weights_filepath = checkpoint_filepath, 
                #                     eval_set = "val",
                #                     results_filepath = results_filepath)
                # predY = model.predY
                # print("there are ", np.count_nonzero(np.isnan(predY)), " NaN values in val predY")
                # print("there are ", np.count_nonzero(predY == 0), " 0 values in val predY")
                # evalY = model.evalY
                # print("there are ", np.count_nonzero(np.isnan(evalY)), " NaN values in val evalY")
                # print("there are ", np.count_nonzero(evalY == 0), " 0 values in val evalY")
                                    
                tf.keras.backend.clear_session()
                break
            break
        break