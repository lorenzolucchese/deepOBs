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


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dir, horizon, task = "classification", multihorizon = False, batch_size=256, shuffle=False, samples_per_file=256, XYsplit=False, teacher_forcing=False):
        """Initialization.
        :param dir: directory of files, contains folder "X" and "Y"
        :param horizon: prediction horizon, 0, 1, 2, 3, 4 or 0
        :param multihorizon: whether the predictions are multihorizon, if True overrides horizon
                             In this case trainX is [trainX, decoder]
        :param batch_size:
        :param samples_per_file: how many samples are in each file
        :param shuffle
        :param teacher_forcing: when using multihorizon, whether to use teacher forcing on the decoder
        Need batch_size to be divisible by samples_per_file
        """
        self.dir = dir
        self.horizon = horizon
        self.multihorizon = multihorizon
        self.teacher_forcing = teacher_forcing
        
        if self.multihorizon:
            self.horizon = slice(0, 5)

        self.task = task
        if self.task == "regression":
            self.Y = "Y_reg"
        elif self.task == "classification":
            self.Y = "Y_class"
        else:
            raise ValueError('task must be either classification or regression.')

        if XYsplit:
            self.X_files = os.listdir(os.path.join(dir, "X"))
            self.Y_files = os.listdir(os.path.join(dir, self.Y))
        else:
            self.files = os.listdir(dir)

        self.batch_size = batch_size
        self.samples_per_file = samples_per_file
        self.files_per_batch = (self.batch_size // self.samples_per_file)
        self.shuffle = shuffle

        self.XYsplit = XYsplit

        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.indices) // self.files_per_batch

    def __getitem__(self, index):
        # Generate indexes of the batch
        file_indices = self.indices[index * self.files_per_batch:(index + 1) * self.files_per_batch]

        # Generate data
        x, y = self.__data_generation(file_indices)

        return x, y

    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
        if self.XYsplit:
            assert (len(self.X_files) == len(self.Y_files))
            self.indices = np.arange(len(self.X_files))
        else:
            self.indices = np.arange(len(self.files))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def prepare_decoder_input(self, x, y):
        if self.teacher_forcing:
            if self.task == "classification":
                first_decoder_input = to_categorical(np.zeros(len(x)), y.shape[-1])
                first_decoder_input = first_decoder_input.reshape(len(first_decoder_input), 1, y.shape[-1])
                decoder_input_data = np.hstack((x[:, :-1, :], first_decoder_input))
            elif self.task == "regression":
                raise ValueError('teacher forcing with regression not yet implemented.')
            else:
                raise ValueError('task must be either classification or regression.')

        if not self.teacher_forcing:
            if self.task == "classification":
                # this sets the initial hidden state of the decoder to be y_0 = [1, 0, ..., 0] for classification
                decoder_input_data = np.zeros((len(x), 1, y.shape[-1]))
                decoder_input_data[:, 0, 0] = 1.
            elif self.task == "regression":
                # this sets the initial hidden state of the decoder to be y_0 = 0 for regression
                decoder_input_data = np.zeros((len(x), 1))
                decoder_input_data[:, 0] = 0
            else:
                raise ValueError('task must be either classification or regression.')

        return tf.convert_to_tensor(decoder_input_data)

    def __data_generation(self, file_indices):
        x_list, y_list = [], []
        for file_index in file_indices:
            if self.XYsplit:
                x_list.append(tf.convert_to_tensor(np.load(os.path.join(self.dir, "X", self.X_files[file_index]))))
                y_list.append(tf.convert_to_tensor(np.load(os.path.join(self.dir, self.Y, self.Y_files[file_index])))[:, self.horizon, ...])
            else:
                with np.load(os.path.join(self.dir, self.files[file_index])) as data:
                    x_list.append(tf.convert_to_tensor(data["X"]))
                    y_list.append(tf.convert_to_tensor(data[self.Y])[:, self.horizon, ...])
        
        if self.samples_per_file==1:
            x = tf.stack(x_list)
            y = tf.stack(y_list)
        else:
            x = tf.concat(x_list, axis=0)
            y = tf.concat(y_list, axis=0)

        if self.multihorizon:
            decoder_input = self.prepare_decoder_input(x, y)
            x = [x, decoder_input]

        return x, y