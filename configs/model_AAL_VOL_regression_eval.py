from model import deepLOB

import tensorflow as tf
import numpy as np
import os

if __name__ == '__main__':

    # limit gpu memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Use only one GPUs
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')

            # Or use all GPUs, memory growth needs to be the same across GPUs
            # for gpu in gpus:
            #     tf.config.experimental.set_memory_growth(gpu, True)
            # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # set random seeds
    np.random.seed(1)
    tf.random.set_seed(2)
    
    orderbook_updates = [10, 20, 30, 50, 100]

    decoders = ["seq2seq", "attention"]

    print("##################### deepVOL #####################")

    for h in range(5):
        #################################### SETTINGS ########################################
        model_inputs = "volumes"                    # options: "order book", "order flow"
        data = "LOBSTER"                            # options: "FI2010", "LOBSTER", "simulated"
        TICKER = "AAL"
        data_dir = "data/model/AAL_volumes_W1_2"
        task = "regression"
        multihorizon = False                        # options: True, False
        decoder = None                              # options: "seq2seq", "attention"

        T = 100
        NF = 40
        n_horizons = 5
        horizon = h                                 # prediction horizon (0, 1, 2, 3, 4) -> (10, 20, 30, 50, 100) order book events
        epochs = 50
        verbose = 2
        batch_size = 256
        number_of_lstm = 64

        checkpoint_filepath = os.path.join('./model_weights', task, 'deep_' + model_inputs + '_weights_' + TICKER + '_W1', 'weights' + str(orderbook_updates[h]))
        load_weights = True
        load_weights_filepath = checkpoint_filepath

        #######################################################################################

        model = deepLOB(T, 
                NF,
                horizon = horizon, 
                number_of_lstm = number_of_lstm, 
                data = data, 
                data_dir = data_dir, 
                model_inputs = model_inputs, 
                task = task, 
                multihorizon = multihorizon, 
                decoder = decoder, 
                n_horizons = n_horizons)

        model.create_model()

        eval_sets = ["train", "val", "test"]

        for eval_set in eval_sets:

            model.evaluate_model(load_weights_filepath = checkpoint_filepath, eval_set = eval_set)

    for dec in decoders:
        
        print("##################### deepVOL", dec, "#####################")

        #################################### SETTINGS ########################################

        model_inputs = "volumes"                    # options: "order book", "order flow"
        data = "LOBSTER"                            # options: "FI2010", "LOBSTER", "simulated"
        TICKER = "AAL"
        data_dir = "data/model/AAL_volumes_W1_2"
        task = "regression"
        multihorizon = True                        # options: True, False
        decoder = dec                              # options: "seq2seq", "attention"

        T = 100
        NF = 40
        n_horizons = 5
        horizon = None                              # prediction horizon (0, 1, 2, 3, 4) -> (10, 20, 30, 50, 100) order book events
        epochs = 50
        verbose = 2
        batch_size = 256
        number_of_lstm = 64

        checkpoint_filepath = os.path.join('./model_weights', task, 'deep_' + model_inputs + '_weights_' + TICKER + '_W1', 'weights' + dec)
        load_weights = True
        load_weights_filepath = checkpoint_filepath

        #######################################################################################

        model = deepLOB(T, 
                NF,
                horizon = horizon, 
                number_of_lstm = number_of_lstm, 
                data = data, 
                data_dir = data_dir, 
                model_inputs = model_inputs, 
                task = task, 
                multihorizon = multihorizon, 
                decoder = decoder, 
                n_horizons = n_horizons)

        model.create_model()

        eval_sets = ["train", "val", "test"]

        for eval_set in eval_sets:

            model.evaluate_model(load_weights_filepath = checkpoint_filepath, eval_set = eval_set)
