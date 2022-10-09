import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import multiprocessing as mp
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
    def __init__(self, dir, horizon, multihorizon = False, batch_size=32, shuffle=False, samples_per_file=32, XYsplit=False, multiprocess=False, teacher_forcing=False):
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

        if XYsplit:
            self.Xfiles = os.listdir(os.path.join(dir, "X"))
            self.Yfiles = os.listdir(os.path.join(dir, "Y"))
        else:
            self.files = os.listdir(dir)

        self.batch_size = batch_size
        self.samples_per_file = samples_per_file
        self.files_per_batch = (self.batch_size // self.samples_per_file)
        self.shuffle = shuffle

        self.multiprocess = multiprocess
        self.XYsplit = XYsplit
        self.n_proc = mp.cpu_count()
        self.chunksize = batch_size // self.n_proc

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
            assert (len(self.Xfiles) == len(self.Yfiles))
            self.indices = np.arange(len(self.Xfiles))
        else:
            self.indices = np.arange(len(self.files))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def prepare_decoder_input(self, data):
        if self.teacher_forcing:
            first_decoder_input = to_categorical(np.zeros(len(data)), 3)
            first_decoder_input = first_decoder_input.reshape(len(first_decoder_input), 1, 3)
            decoder_input_data = np.hstack((data[:, :-1, :], first_decoder_input))

        if not self.teacher_forcing:
            decoder_input_data = np.zeros((len(data), 1, 3))
            decoder_input_data[:, 0, 0] = 1.

        return decoder_input_data

    def load_chunk(self, file_indices):
        x_list = []
        y_list = []
        for file_index in file_indices:
            if self.XYsplit:
                x_list.append(tf.convert_to_tensor(np.load(os.path.join(self.dir, "X", self.Xfiles[file_index]))))
                y_list.append(tf.convert_to_tensor(np.load(os.path.join(self.dir, "Y", self.Yfiles[file_index])))[:, self.horizon, :])
            else:
                with np.load(os.path.join(self.dir, self.files[file_index])) as data:
                    x_list.append(tf.convert_to_tensor(data["X"]))
                    y_list.append(tf.convert_to_tensor(data["Y"])[:, self.horizon, :])
                # data = np.load(os.path.join(self.dir, self.files[file_index]))
                # x_list.append(tf.convert_to_tensor(data["X"]))
                # y_list.append(tf.convert_to_tensor(data["Y"]))
        if self.samples_per_file==1:
            x = tf.stack(x_list)
            y = tf.stack(y_list)
        else:
            x = tf.concat(x_list, axis=0)
            y = tf.concat(y_list, axis=0)
        return x, y

    def __data_generation(self, file_indices):
        if self.multiprocess:
            # parallelize
            file_indices_chunks = np.array_split(file_indices, self.chunksize)

            with mp.Pool(processes=self.n_proc) as pool:
                # starts the sub-processes without blocking
                # pass the chunk to each worker process
                proc_results = [pool.apply_async(self.load_chunk, args=(file_indices_chunk,))
                                for file_indices_chunk in file_indices_chunks]

                # blocks until all results are fetched
                results = [r.get() for r in proc_results]
                x = tf.concat(list(zip(*results))[0], axis=0)
                y = tf.concat(list(zip(*results))[1], axis=0)

        else:
            x, y = self.load_chunk(file_indices)

        if self.multihorizon:
            decoder_input = self.prepare_decoder_input(x)
            x = [x, decoder_input]

        return x, y