from model import deepLOB

import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import multiprocessing as mp
import time
import glob
import os
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Flatten, Dense, Dropout, LeakyReLU, Activation, Input, LSTM, CuDNNLSTM, Reshape, Conv2D, MaxPooling2D, concatenate, Lambda, dot, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, MeanSquaredError

from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # limit gpu memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Use only one GPUs
            tf.config.set_visible_devices(gpus[1], 'GPU')
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

    for h in range(5):
        print("##################### deepLOB #####################")
        #################################### SETTINGS ########################################
        model_inputs = "order book"                 # options: "order book", "order flow", "volumes"
        data = "LOBSTER"                            # options: "FI2010", "LOBSTER", "simulated"
        data_dir = "data/AAL_orderbooks"
        csv_file_list = glob.glob(os.path.join(data_dir, "*.{}").format("csv"))
        csv_file_list.sort()
        files = {
            "val": csv_file_list[:5],
            "train": csv_file_list[5:25],
            "test": csv_file_list[25:30]
        }
        # alphas = get_alphas(files["train"])
        alphas = np.array([0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 3.2942814e-05])
        task = "classification"
        multihorizon = False                        # options: True, False
        decoder = "seq2seq"                         # options: "seq2seq", "attention"

        T = 100
        NF = 40                                     # remember to change this when changing features
        n_horizons = 5
        horizon = h                                 # prediction horizon (0, 1, 2, 3, 4) -> (10, 20, 30, 50, 100) order book events
        epochs = 50
        training_verbose = 2
        train_roll_window = 100
        batch_size = 256                            # note we use 256 for LOBSTER, 32 for FI2010 or simulated
        number_of_lstm = 64

        checkpoint_filepath = './model_weights_new/deepOB_weights_AAL_W1/weights' + str(orderbook_updates[h])
        load_weights = False
        load_weights_filepath = './model_weights_new/deepOB_weights_AAL_W1/weights' + str(orderbook_updates[h])

        #######################################################################################

        model = deepLOB(T, 
                        NF, 
                        horizon, 
                        number_of_lstm, 
                        data, 
                        data_dir, 
                        files, 
                        model_inputs, 
                        task, 
                        alphas, 
                        multihorizon, 
                        decoder, 
                        n_horizons,
                        batch_size, 
                        train_roll_window)

        model.create_model()

        # model.model.summary()

        model.fit_model(epochs = epochs,
                        checkpoint_filepath = checkpoint_filepath,
                        load_weights = load_weights,
                        load_weights_filepath = load_weights_filepath,
                        verbose = training_verbose)

        model.evaluate_model(load_weights_filepath=load_weights_filepath)
    
    for dec in decoders:
        print("##################### deepLOB", dec, "#####################")
        #################################### SETTINGS ########################################
        model_inputs = "order book"                 # options: "order book", "order flow", "volumes"
        data = "LOBSTER"                            # options: "FI2010", "LOBSTER", "simulated"
        data_dir = "data/AAL_orderbooks"
        csv_file_list = glob.glob(os.path.join(data_dir, "*.{}").format("csv"))
        csv_file_list.sort()
        files = {
            "val": csv_file_list[:5],
            "train": csv_file_list[5:25],
            "test": csv_file_list[25:30]
        }
        # alphas = get_alphas(files["train"])
        alphas = np.array([0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 3.2942814e-05])
        task = "classification"
        multihorizon = True                         # options: True, False
        decoder = dec                               # options: "seq2seq", "attention"

        T = 100
        NF = 40                                     # remember to change this when changing features
        n_horizons = 5
        horizon = 0                                 # prediction horizon (0, 1, 2, 3, 4) -> (10, 20, 30, 50, 100) order book events
        epochs = 50
        training_verbose = 2
        train_roll_window = 100
        batch_size = 256                            # note we use 256 for LOBSTER, 32 for FI2010 or simulated
        number_of_lstm = 64

        checkpoint_filepath = './model_weights_new/deepOB_weights_AAL_W1/weights' + dec
        load_weights = False
        load_weights_filepath = './model_weights_new/deepOB_weights_AAL_W1/weights' + dec

        #######################################################################################

        model = deepLOB(T, 
                        NF, 
                        horizon, 
                        number_of_lstm, 
                        data, 
                        data_dir, 
                        files, 
                        model_inputs, 
                        task, 
                        alphas, 
                        multihorizon, 
                        decoder, 
                        n_horizons,
                        batch_size, 
                        train_roll_window)

        model.create_model()

        # model.model.summary()

        model.fit_model(epochs = epochs,
                        checkpoint_filepath = checkpoint_filepath,
                        load_weights = load_weights,
                        load_weights_filepath = load_weights_filepath,
                        verbose = training_verbose)

        model.evaluate_model(load_weights_filepath=load_weights_filepath)
                    
