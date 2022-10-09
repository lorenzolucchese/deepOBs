from model import deepOB

import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import multiprocessing as mp
import time
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
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # set random seeds
    np.random.seed(1)
    tf.random.set_seed(2)
    
    orderbook_updates = [10, 20, 30, 50, 100]

    decoders = ["seq2seq", "attention"]

    print("##################### deepOF #####################")
    
    for h in range(0, 5):
        #################################### SETTINGS ########################################

        model_inputs = "order flow"                 # options: "order book", "order flow"
        data = "LOBSTER"                            # options: "FI2010", "AAL"
        data_dir = "data/model/AAL_orderflows_W1"
        task = "classification"
        multihorizon = False                        # options: True, False
        decoder = "seq2seq"                         # options: "seq2seq", "attention"

        T = 100
        NF = 20
        n_horizons = 5
        horizon = h                                 # prediction horizon (0, 1, 2, 3, 4) -> (10, 20, 30, 50, 100) order book events
        epochs = 50
        batch_size = 256
        number_of_lstm = 64

        checkpoint_filepath = './model_weights/deepOF_weights_AAL_W1/weights' + str(orderbook_updates[h])
        load_weights = False
        load_weights_filepath = './model_weights/deepOF_weights_AAL_W1/weights' + str(orderbook_updates[h])

        #######################################################################################

        model = deepOB(T, 
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

        # model.model.summary()

        model.fit_model(epochs = epochs, 
                    batch_size = batch_size,
                    checkpoint_filepath = checkpoint_filepath,
                    load_weights = load_weights,
                    load_weights_filepath = load_weights_filepath)

        model.evaluate_model(load_weights_filepath = checkpoint_filepath)

    for dec in decoders:
        print("##################### deepOF", dec, "#####################")

        #################################### SETTINGS ########################################

        model_inputs = "order flow"                 # options: "order book", "order flow"
        data = "LOBSTER"                            # options: "FI2010", "AAL"
        data_dir = "data/model/AAL_orderflows_W1"
        task = "classification"
        multihorizon = True                         # options: True, False
        decoder = dec                               # options: "seq2seq", "attention"

        T = 100
        NF = 20
        n_horizons = 5
        horizon = "NA"                              # prediction horizon (0, 1, 2, 3, 4) -> (10, 20, 30, 50, 100) order book events
        epochs = 50
        batch_size = 256
        number_of_lstm = 64

        checkpoint_filepath = './model_weights/deepOF_weights_AAL_W1/weights' + dec
        load_weights = False
        load_weights_filepath = './model_weights/deepOF_weights_AAL_W1/weights' + dec

        #######################################################################################

        model = deepOB(T, 
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

        # model.model.summary()

        model.fit_model(epochs = epochs, 
                    batch_size = batch_size,
                    checkpoint_filepath = checkpoint_filepath,
                    load_weights = load_weights,
                    load_weights_filepath = load_weights_filepath,
                    patience = 10)

        model.evaluate_model(load_weights_filepath = checkpoint_filepath)