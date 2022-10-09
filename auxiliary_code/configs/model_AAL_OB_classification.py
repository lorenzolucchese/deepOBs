from model import deepOB

import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import os

if __name__ == '__main__':
    # limit gpu memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            # Use only one GPUs
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')

            # Or use all GPUs, memory growth needs to be the same across GPUs
            # for gpu in gpus:
            #     tf.config.experimental.set_memory_growth(gpu, True)
            # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # set random seeds
    np.random.seed(1)
    tf.random.set_seed(2)
    
    orderbook_updates = [10, 20, 30, 50, 100]
    decoders = ["seq2seq", "attention"]
    distributions = pd.DataFrame(np.vstack([np.array([0.121552, 0.194825, 0.245483, 0.314996, 0.334330]), 
                                            np.array([0.752556, 0.604704, 0.504695, 0.368647, 0.330456]),
                                            np.array([0.125893, 0.200471, 0.249821, 0.316357, 0.335214])]), 
                                 index=["down", "stationary", "up"], 
                                 columns=["10", "20", "30", "50", "100"])
    data_imbalances = distributions.to_numpy()

    for imbalances_flag in [True, False]:
        if imbalances_flag:
            imbalances  = data_imbalances
        else:
            imbalances = None
        for train_roll_window in [100, 10, 1]:
            print("##################### deepLOB - seq2seq #####################")
            print("train_roll_window = ", train_roll_window)
            print("imbalances = ", imbalances_flag)
            print(imbalances)
            #################################### SETTINGS ########################################
            model_inputs = "orderbook"                 # options: "order book", "order flow", "volumes"
            data = "LOBSTER"                            # options: "FI2010", "LOBSTER", "simulated"
            data_dir = "data/AAL_orderbooks"
            csv_file_list = glob.glob(os.path.join(data_dir, "*.{}").format("csv"))
            csv_file_list.sort()
            files = {
                "val": csv_file_list[:5],
                "train": csv_file_list[5:25],
                "test": csv_file_list[25:30]
            }
            alphas = np.array([0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 3.2942814e-05])
            task = "classification"
            multihorizon = True                         # options: True, False
            decoder = "seq2seq"                         # options: "seq2seq", "attention"

            T = 100
            levels = 10                                 # remember to change this when changing features
            queue_depth = 10
            n_horizons = 5
            horizon = 0                                 # prediction horizon (0, 1, 2, 3, 4) -> (10, 20, 30, 50, 100) order book events
            epochs = 50
            patience = 5
            training_verbose = 2
            train_roll_window = train_roll_window
            batch_size = 256                            # note we use 256 for LOBSTER, 32 for FI2010 or simulated
            number_of_lstm = 64

            checkpoint_filepath = './model_weights_new/test'
            load_weights = False
            load_weights_filepath = './model_weights_new/test'

            #######################################################################################

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
                    multihorizon = multihorizon, 
                    decoder = decoder, 
                    n_horizons = n_horizons,
                    batch_size = batch_size,
                    train_roll_window = train_roll_window,
                    imbalances = imbalances)

            model.create_model()

            # model.model.summary()

            model.fit_model(epochs = epochs,
                            checkpoint_filepath = checkpoint_filepath,
                            load_weights = load_weights,
                            load_weights_filepath = load_weights_filepath,
                            verbose = training_verbose)

            model.evaluate_model(load_weights_filepath=load_weights_filepath)
                    
