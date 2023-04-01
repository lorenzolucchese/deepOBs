import numpy as np
# import pandas as pd
# import pickle
# import glob
# import os
# import random
# import tensorflow as tf
# from keras.layers import Reshape, concatenate
# from custom_datasets import CustomtfDataset
# from data_methods import get_alphas
# from model import deepOB
# from config.directories import ROOT_DIR
# import datetime as dt

if __name__ == '__main__':
    # limit gpu memory
    # visible_gpus = tf.config.experimental.get_visible_devices("GPU")
    # physical_gpus = tf.config.experimental.list_physical_devices("GPU")
    # print("This machine has", len(visible_gpus), "visible gpus.")
    # print("This machine has", len(physical_gpus), "physical gpus.")
    # if visible_gpus:
    #     try:
    #         # Use only one GPUs
    #         tf.config.set_visible_devices(visible_gpus[0], "GPU")
    #         logical_gpus = tf.config.list_logical_devices("GPU")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)

    # # select TICKER
    # TICKER = "ATVI"

    # # set global parameters
    # orderbook_updates = [10]             
    # task = "classification"
    # multihorizon = False                              
    # decoder = "seq2seq"                                            
    # T = 100
    # queue_depth = 10                                          
    # n_horizons = len(orderbook_updates)
    # epochs = 50
    # patience = 10
    # training_verbose = 2
    # testing_verbose = 1
    # train_roll_window = 1000
    # batch_size = 256
    # number_of_lstm = 64
    # model_list = ["deepLOB_L1", "deepOF_L1", "deepLOB_L2", "deepOF_L2", "deepVOL_L2", "deepVOL_L3"]
    # features_list = ["orderbooks", "orderflows", "orderbooks", "orderflows", "volumes", "volumes"]
    # model_inputs_list = ["orderbooks", "orderflows", "orderbooks", "orderflows", "volumes", "volumes_L3"]
    # levels_list = [1, 1, 10, 10, 10, 10]

    # TICKER_filepath = os.path.join(ROOT_DIR, "results", "debugger", TICKER)
    # os.makedirs(TICKER_filepath, exist_ok=True)
    # window_filepath = os.path.join(TICKER_filepath, "W0")

    # # select specific window
    # test_dates = ["2019-01-10"]

    # # set random seeds
    # random.seed(0)
    # np.random.seed(1)
    # tf.random.set_seed(2)
    
    # alphas = pickle.load(open("results/ATVI/W0/alphas.pkl", "rb"))

    # # iterate through model types
    # for m, model_type in enumerate(model_list):
    #     model_filepath = os.path.join(window_filepath, model_type)
    #     os.makedirs(model_filepath, exist_ok=True)
        
    #     # set local parameters
    #     features = features_list[m]
    #     model_inputs = model_inputs_list[m]
    #     levels = levels_list[m]
        
    #     data_dir = os.path.join(ROOT_DIR, "data", TICKER)
    #     file_list = os.listdir(data_dir)
    #     print(file_list)
    #     files = {
    #         "train": [os.path.join(data_dir, file) for date in test_dates for file in file_list if date in file],
    #         "val": [os.path.join(data_dir, file) for date in test_dates for file in file_list if date in file],
    #         "test": [os.path.join(data_dir, file) for date in test_dates for file in file_list if date in file]
    #     }
        
    #     distributions = pickle.load(open("results/ATVI/W0/distributions.pkl", "rb"))
    #     imbalances = distributions.to_numpy()
        
    #     # iterate through horizons
    #     for h in range(n_horizons):
    #         horizon = h
    #         results_filepath = os.path.join(model_filepath, "h" + str(orderbook_updates[h]))
    #         checkpoint_filepath = os.path.join(ROOT_DIR, "results", TICKER, "W0", model_type, "h" + str(orderbook_updates[h]), "weights")
    #         os.makedirs(results_filepath, exist_ok=True)

    #         # create model
    #         model = deepOB(T = T, 
    #                        levels = levels, 
    #                        horizon = horizon, 
    #                        number_of_lstm = number_of_lstm,
    #                        data_dir = data_dir, 
    #                        files = files, 
    #                        model_inputs = model_inputs, 
    #                        queue_depth = queue_depth,
    #                        task = task, 
    #                        alphas = alphas, 
    #                        orderbook_updates = orderbook_updates,
    #                        multihorizon = multihorizon, 
    #                        decoder = decoder, 
    #                        n_horizons = n_horizons,
    #                        train_roll_window = train_roll_window,
    #                        imbalances = imbalances,
    #                        batch_size = batch_size)

    #         model.create_model()

    #         # evaluate model
    #         model.evaluate_model(load_weights_filepath = checkpoint_filepath,
    #                              eval_set = "train",
    #                              results_filepath = results_filepath,
    #                              verbose = testing_verbose)
            

    TICKER = "WBA"
    old = np.load(".\data\WBA_2\WBA_data_2019-11-06.npz")
    new = np.load(".\data\WBA\WBA_data_2019-11-06.npz")

    # mus = np.zeros(40)
    # sigmas = np.zeros(40)
    
    # for i in range(40):
    #     print(i)
    #     n = 0
    #     condition = True
    #     while condition:
    #         z1 = old["orderbook_features"][0, i]
    #         z2 = old["orderbook_features"][n+1, i]
    #         x1 = new["orderbook_features"][1, i]
    #         x2 = new["orderbook_features"][n+2, i]
    
    #         if x1 == x2 or z1 == z2:
    #             n += 1
    #         else:
    #             mu = np.abs((z2*x1 - z1*x2) / (z2 - z1))
    #             sigma = np.abs((x1 - mu) / z1)
    #             if np.isnan(mu) or np.isinf(mu):
    #                 n+=1
    #             else:  
    #                 mus[i] = mu
    #                 sigmas[i] = sigma
    #                 condition = False

    # np.save("mus.np", mus)
    # np.save("sigmas.np", sigmas)

    # mus = np.load("./auxiliary_code/mus.npy")
    # mus[8] -= 100
    # sigmas = np.load("./auxiliary_code/sigmas.npy")

    # print(new["orderbook_features"][101:110, 7])
    # old_rec = (old["orderbook_features"]*sigmas + mus).astype(int)
    # print(old_rec[100:109, 7])

    # for i in range(1, 300000):
    #     if not np.allclose((old["orderbook_features"][i, :]*sigmas + mus)[:10], new["orderbook_features"][i+1, :10], atol = 10):
    #         print(i)
    #         print((old["orderbook_features"][i, :]*sigmas + mus).astype(int)[:10])
    #         print(new["orderbook_features"][i+1, :10])
    #         break

    # print(np.sum(1-np.isclose((old["orderbook_features"] * sigmas + mus)[:, 8], new["orderbook_features"][1:, 8], atol=10)))

