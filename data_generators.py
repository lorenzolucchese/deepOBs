import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import time
import os
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, LSTM, CuDNNLSTM, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def CustomtfDataset(files, 
                    NF, 
                    horizon, 
                    task = "classification", 
                    alphas = np.array([]), 
                    multihorizon = False, 
                    normalise = False, 
                    batch_size=256, 
                    shuffle=True, 
                    teacher_forcing=False, 
                    window=100, 
                    roll_window=1):
    """
    :param dir: directory of files
    :param files: list of files in directory to use
    :param NF: number of features
    :param horizon: prediction horizon, 0, 1, 2, 3, 4 or 0
    :param task: regression or classification
    :param alphas: array of alphas for class boundaries if task = classification.
    :param multihorizon: whether the predictions are multihorizon, if True overrides horizon
                         In this case trainX is [trainX, decoder]
    :param batch_size:
    :param samples_per_file: how many samples are in each file
    :param shuffle
    :param teacher_forcing: when using multihorizon, whether to use teacher forcing on the decoder
    Need batch_size to be divisible by samples_per_file
    """
    # methods to be used
    def scale_fn(x, y):
        return x / tf.keras.backend.max(x), y

    def add_decoder_input(x, y):
        if teacher_forcing:
            if task == "classification":
                first_decoder_input = to_categorical(tf.zeros(x.shape[0]), y.shape[-1])
                first_decoder_input = tf.reshape(first_decoder_input, [first_decoder_input.shape[0], 1, y.shape[-1]])
                decoder_input_data = tf.hstack((x[:, :-1, :], first_decoder_input))
            elif task == "regression":
                raise ValueError('teacher forcing with regression not yet implemented.')
            else:
                raise ValueError('task must be either classification or regression.')

        if not teacher_forcing:
            if task == "classification":
                # this sets the initial hidden state of the decoder to be y_0 = [0, 0, 0] for classification
                decoder_input_data = tf.zeros_like(y[:, 0:1, :])
            elif task == "regression":
                # this sets the initial hidden state of the decoder to be y_0 = 0 for regression
                decoder_input_data = tf.zeros_like(y[:, 0:1])
            else:
                raise ValueError('task must be either classification or regression.')

        return {'input': x, 'decoder_input': decoder_input_data}, y
    
    if multihorizon:
        horizon = slice(0, 5)

    if (task == "classification")&(alphas.size == 0):
        raise ValueError('alphas must be assigned if task is classification.')

    # create combined dataset
    dataset = np.array([]).reshape(0, NF + 5)
    for file in files:
        dataset = np.concatenate([dataset, pd.read_csv(file).to_numpy()])

    features = dataset[:, :NF]
    features = features.reshape(features.shape[0], NF, 1)
    responses = dataset[(window-1):, -5:]
    responses = responses[:, horizon]

    if task == "classification":
        if multihorizon:
            all_label = []
            for h in range(responses.shape[1]):
                one_label = (+1)*(responses[:, h]>=-alphas[h]) + (+1)*(responses[:, h]>alphas[h])
                one_label = to_categorical(one_label, 3)
                one_label = one_label.reshape(len(one_label), 1, 3)
                all_label.append(one_label)
            y = np.hstack(all_label)
        else:
            y = (+1)*(responses>=-alphas[horizon]) + (+1)*(responses>alphas[horizon])
            y = to_categorical(y, 3)

    tf_dataset = tf.keras.utils.timeseries_dataset_from_array(features, 
                                                              y, 
                                                              window, 
                                                              sequence_stride=roll_window, 
                                                              batch_size=batch_size, 
                                                              shuffle=shuffle)
    
    if normalise:
        tf_dataset = tf_dataset.map(scale_fn)

    if multihorizon:
        tf_dataset = tf_dataset.map(add_decoder_input)

    return tf_dataset