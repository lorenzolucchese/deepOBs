"""
This code is adapted from the Jupyter notebook by Zihao Zhang, Stefan Zohren and Stephen Roberts
'deepLOB: Deep Convolutional Neural Networks for Limit Order Books'
Oxford-Man Institute of Quantitative Finance, Department of Engineering Science, University of Oxford

The model is deepLOB from paper [1]. We apply it to LOBSTER data [2].

[1] Zhang Z, Zohren S, Roberts S. deepLOB: Deep convolutional neural networks for limit order books.
    IEEE Transactions on Signal Processing. 2019 Mar 25;67(11):3001-12.
    https://arxiv.org/abs/1808.03668
[2] Huang, R and Polak, T. LOBSTER: Limit Order Book Reconstruction System. (December 27, 2011). 
    Available at SSRN: https://ssrn.com/abstract=1977207 or http://dx.doi.org/10.2139/ssrn.1977207
"""

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
from keras.utils import np_utils

from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dir, batch_size, shuffle=True, samples_per_file=1, XYsplit=True, multiprocess=False):
        """Initialization.
        :param dir: directory of files, contains folder "X" and "Y"
        :param batch_size:
        :param samples_per_file: how many samples are in each file
        :param shuffle
        Need batch_size to be divisible by samples_per_file
        """
        self.dir = dir

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

    def load_chunk(self, file_indices):
        x_list = []
        y_list = []
        for file_index in file_indices:
            if self.XYsplit:
                x_list.append(tf.convert_to_tensor(np.load(os.path.join(self.dir, "X", self.Xfiles[file_index]))))
                y_list.append(tf.convert_to_tensor(np.load(os.path.join(self.dir, "Y", self.Yfiles[file_index]))))
            else:
                with np.load(os.path.join(self.dir, self.files[file_index])) as data:
                    x_list.append(tf.convert_to_tensor(data["X"]))
                    y_list.append(tf.convert_to_tensor(data["Y"]))
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

        return x, y

def create_deepLOB(T, NF, number_of_lstm):
    input_lmd = Input(shape=(T, NF, 1))

    # build the convolutional block
    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 10))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    # build the inception module
    convsecond_1 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
    conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)
    conv_reshape = keras.layers.Dropout(0.2, noise_shape=(None, 1, int(conv_reshape.shape[2])))(conv_reshape,
                                                                                                training=True)

    # build the last LSTM layer
    conv_lstm = LSTM(number_of_lstm, batch_input_shape=(32, T, int(conv_reshape.shape[2])))(conv_reshape)

    # build the output layer
    out = Dense(3, activation='softmax')(conv_lstm)
    model = Model(inputs=input_lmd, outputs=out)
    adam = Adam(learning_rate=0.01, epsilon=1)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # limit gpu memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

#    # use only one gpu
#    gpus = tf.config.list_physical_devices('GPU')
#    if gpus:
#        # Restrict TensorFlow to use only the first GPU
#        try:
#            tf.config.set_visible_devices(gpus[0], 'GPU')
#            logical_gpus = tf.config.list_logical_devices('GPU')
#        except RuntimeError as e:
#            # Visible devices must be set before GPUs have been initialized
#            print(e)

    # set random seeds
    np.random.seed(1)
    tf.random.set_seed(2)

    k = 4
    # which prediction horizon (k = (0, 1, 2, 3, 4) -> (10, 20, 30, 50, 100) order book events)
    T = 100
    # the length of a sample sequence. Even though this is a single long time series, LSTMs usually work with
    # input sequences of max length 200-400, we hence split the time series into sequences of length 100
    # rolling forward by one time-step each time.
    n_hiddens = 64
    # number of hidden states in LSTM
    
    checkpoint_filepath = './model_weights/deepLOB_weights_AAL_W1_100/weights'

    # data
    val_generator = DataGenerator(r"data/AAL_OB_W1_batch32/val", batch_size=32, 
                                  XYsplit = False, samples_per_file =32)
    train_generator = DataGenerator(r"data/AAL_OB_W1_batch32/train", batch_size=32, 
                                    XYsplit=False, samples_per_file=32)
    test_generator = DataGenerator(r"data/AAL_OB_W1_batch32/test", batch_size=32, 
                                   XYsplit=False, samples_per_file=32, shuffle=False)

#    # Create a MirroredStrategy.
#    strategy = tf.distribute.MirroredStrategy()
#    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
#    
#    # Open a strategy scope.
#    with strategy.scope():
#        # Everything that creates variables should be under the strategy scope.
#    
#        # build model
#        deepLOB = create_deepLOB(T, 40, n_hiddens)
    
    deepLOB = create_deepLOB(T, 40, n_hiddens)
    
    deepLOB.summary()
    
    # load weights
    deepLOB.load_weights(checkpoint_filepath)
    
#    # train model
#    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#        filepath=checkpoint_filepath,
#        save_weights_only=True,
#        monitor='val_loss',
#        mode='auto',
#        save_best_only=True)
#    
#    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20)
#    
#    deepLOB.fit(train_generator, validation_data=val_generator,
#                epochs=50, verbose=1, workers=8,
#                callbacks=[model_checkpoint_callback, early_stopping])
    
    # evaluate model performance    
    pred = deepLOB.evaluate(train_generator, workers=8)
    
    # print('accuracy_score:', accuracy_score(np.argmax(testY_CNN, axis=1), np.argmax(pred, axis=1)))
    # print(classification_report(np.argmax(testY_CNN, axis=1), np.argmax(pred, axis=1), digits=4))
    
    
    
