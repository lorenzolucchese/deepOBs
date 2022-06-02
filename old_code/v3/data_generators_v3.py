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
    def __init__(self, dir, file_list, NF, horizon, task = "classification", alphas = np.array([]), multihorizon = False, normalise = False, batch_size=256, shuffle=True, teacher_forcing=False, window=100):
        """Initialization.
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
        self.dir = dir
        self.file_list = file_list        
        self.NF = NF
        self.horizon = horizon
        self.task = task
        self.alphas = alphas
        self.multihorizon = multihorizon
        self.normalise = normalise
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.teacher_forcing = teacher_forcing
        self.window = window
        
        if self.multihorizon:
            self.horizon = slice(0, 5)

        if (self.task == "classification")&(self.alphas.size == 0):
            raise ValueError('alphas must be assigned if task is classification.')

        self.files = []
        self.batches_per_file = []
        for file in self.file_list:
            file = pd.read_csv(file).to_numpy()
            self.files.append(file)
            self.batches_per_file.append((file.shape[0] - self.window + 1) // self.batch_size)

        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return sum(self.batches_per_file)

    def __getitem__(self, index):
        # Generate indexes of the batch
        index = index - self.done_batches
        file_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(file_indices)

        return x, y

    def on_file_end(self):
        self.done_batches += self.batches_in_file
        self.file_index += 1
        self.file = pd.read_csv(self.files[self.file_index]).to_numpy()
        print(self.files[self.file_index])
        self.samples_in_file = (self.file.shape[0] - self.window + 1)
        self.batches_in_file = self.samples_in_file // self.batch_size
        self.indices = np.arange(self.samples_in_file)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def on_epoch_end(self):
        'Restart file list after each epoch'
        self.file_index = 0
        self.done_batches = 0        
        self.file = pd.read_csv(self.files[self.file_index]).to_numpy()
        self.samples_in_file = (self.file.shape[0] - self.window + 1)
        self.batches_in_file = self.samples_in_file // self.batch_size
        self.indices = np.arange(self.samples_in_file)
        if self.shuffle:
            np.random.shuffle(self.files)
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
                decoder_input_data = np.zeros((len(x), 1, 3))
                decoder_input_data[:, 0, 0] = 1.
            elif self.task == "regression":
                # this sets the initial hidden state of the decoder to be y_0 = 0 for regression
                decoder_input_data = np.zeros((len(x), 1))
                decoder_input_data[:, 0] = 0
            else:
                raise ValueError('task must be either classification or regression.')

        return decoder_input_data

    def __data_generation(self, file_indices):
        x_list, y_list = [], []

        for file_index in file_indices:
            x_sample = self.file[file_index:(file_index+self.window), :self.NF]
            x_sample = x_sample.reshape(x_sample.shape[0], x_sample.shape[1], 1)
            if self.normalise:
                x_sample = x_sample / np.max(x_sample)
            y_sample = self.file[(file_index+self.window-1), -5:]
            if self.task == "classification":
                y_sample = (+1)*(y_sample > -self.alphas) + (+1)*(y_sample > self.alphas)
                y_sample = to_categorical(y_sample, 3).reshape(5, 3)
            y_sample = y_sample[self.horizon, :]
            x_list.append(tf.convert_to_tensor(x_sample))
            y_list.append(tf.convert_to_tensor(y_sample))
        
        x = tf.stack(x_list)
        y = tf.stack(y_list)
        # x = tf.concat(x_list, axis = 0)
        # y = tf.concat(y_list, axis = 0)

        if self.multihorizon:
            decoder_input = self.prepare_decoder_input(x, y)
            x = [x, decoder_input]

        return x, y


class CustomDataGenerator2(tf.keras.utils.Sequence):
    def __init__(self, dir, files, NF, horizon, task = "classification", alphas = np.array([]), multihorizon = False, normalise = False, batch_size=256, shuffle=True, teacher_forcing=False, window=100):
        """Initialization.
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
        self.dir = dir
        self.files = files        
        self.NF = NF
        self.horizon = horizon
        self.task = task
        self.alphas = alphas
        self.multihorizon = multihorizon
        self.normalise = normalise
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.teacher_forcing = teacher_forcing
        self.window = window
        
        if self.multihorizon:
            self.horizon = slice(0, 5)

        if (self.task == "classification")&(self.alphas.size == 0):
            raise ValueError('alphas must be assigned if task is classification.')

        # create combined dataset
        self.dataset = np.array([]).reshape(0, self.NF + 5)
        for file in self.files:
            self.dataset = np.concatenate([self.dataset, pd.read_csv(file).to_numpy()])
        # self.dataset = tf.convert_to_tensor(self.dataset)
        
        self.samples_in_dataset = (self.dataset.shape[0] - self.window + 1)

        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.samples_in_dataset // self.batch_size

    def __getitem__(self, index):
        file_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(file_indices)

        return x, y

    def on_epoch_end(self):
        self.indices = np.arange(self.samples_in_dataset)
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
                decoder_input_data = np.zeros((len(x), 1, 3))
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
            x_sample = self.dataset[file_index:(file_index+self.window), :self.NF]
            # x_sample = tf.reshape(x_sample, [x_sample.shape[0], x_sample.shape[1], 1])
            x_sample = x_sample.reshape(x_sample.shape[0], x_sample.shape[1], 1)
            if self.normalise:
                x_sample = x_sample / np.max(x_sample)
            y_sample = self.dataset[(file_index+self.window-1), -5:]
            if self.task == "classification":
                y_sample = (+1)*(y_sample > -self.alphas) + (+1)*(y_sample > self.alphas)
                y_sample = to_categorical(y_sample, 3).reshape(5, 3)
                # y_sample = tf.cast(y_sample > -self.alphas, tf.float32) + tf.cast(y_sample > self.alphas, tf.float32)
                # y_sample = tf.reshape(to_categorical(y_sample, 3), [5, 3])
            y_sample = y_sample[self.horizon, :]
            x_list.append(tf.convert_to_tensor(x_sample))
            y_list.append(tf.convert_to_tensor(y_sample))
            # x_list.append(x_sample)
            # y_list.append(y_sample)
        
        x = tf.stack(x_list)
        y = tf.stack(y_list)
        # x = tf.concat(x_list, axis = 0)
        # y = tf.concat(y_list, axis = 0)

        if self.multihorizon:
            decoder_input = self.prepare_decoder_input(x, y)
            x = [x, decoder_input]

        return x, y