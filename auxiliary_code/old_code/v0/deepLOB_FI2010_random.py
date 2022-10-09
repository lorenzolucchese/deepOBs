"""
This code is adapted from the Jupyter notebook by Zihao Zhang, Stefan Zohren and Stephen Roberts
'DeepLOB: Deep Convolutional Neural Networks for Limit Order Books'
Oxford-Man Institute of Quantitative Finance, Department of Engineering Science, University of Oxford

The model is DeepLOB from paper [1]. We apply it to both the FI-2020 data set [2] and the LOBSTER data.

[1] Zhang Z, Zohren S, Roberts S. DeepLOB: Deep convolutional neural networks for limit order books.
    IEEE Transactions on Signal Processing. 2019 Mar 25;67(11):3001-12.
    https://arxiv.org/abs/1808.03668
[2] Ntakaris A, Magris M, Kanniainen J, Gabbouj M, Iosifidis A. Benchmark dataset for mid‐price forecasting of limit
    order book data with machine learning methods. Journal of Forecasting. 2018 Dec;37(8):852-66.
    https://arxiv.org/abs/1705.03233
"""

import tensorflow as tf            
import pandas as pd
import numpy as np
import random
import keras
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, LSTM, CuDNNLSTM, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils

from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


def prepare_x(data):
    df1 = data[:, :40]
    return np.array(df1)


def get_label(data):
    lob = data[:, -5:]
    return lob


def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)
    dY = np.array(Y)
    dataY = dY[T - 1:N]
    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]
    return dataX.reshape(dataX.shape + (1,)), dataY


def prepare_x_y(data, k, T):
    x = prepare_x(data)
    y = get_label(data)
    x, y = data_classification(x, y, T=T)
    y = y[:, k] - 1
    y = np_utils.to_categorical(y, 3)
    return x, y


def create_deeplob(T, NF, number_of_lstm):
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
    conv_lstm = CuDNNLSTM(number_of_lstm, batch_input_shape=(32, T, int(conv_reshape.shape[2])))(conv_reshape)

    # build the output layer
    out = Dense(3, activation='softmax')(conv_lstm)
    model = Model(inputs=input_lmd, outputs=out)
    adam = Adam(learning_rate=0.01, epsilon=1)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # limit gpu memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
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

    # prepare data
    data = np.loadtxt(r'data/FI2010/Train_Dst_NoAuction_ZScore_CF_7.txt').T
    train = data[:int(np.floor(data.shape[0] * 0.8)), :]
    val = data[int(np.floor(data.shape[0] * 0.8)):, :]

    test1 = np.loadtxt(r'data/FI2010/Test_Dst_NoAuction_ZScore_CF_7.txt').T
    test2 = np.loadtxt(r'data/FI2010/Test_Dst_NoAuction_ZScore_CF_8.txt').T
    test3 = np.loadtxt(r'data/FI2010/Test_Dst_NoAuction_ZScore_CF_9.txt').T

    test = np.vstack((test1, test2, test3))

    k = 4
    # which prediction horizon (k = (0, 1, 2, 3, 4) -> (10, 20, 30, 50, 100) order book events)
    T = 100
    # the length of a sample sequence. Even though this is a single long time series, LSTMs usually work with
    # input sequences of max length 200-400, we hence split the time series into sequences of length 100
    # rolling forward by one time-step each time.
    # How doe the FI-2010 dataset treat going from one day to the next??
    n_hiddens = 64
    # number of hidden states in LSTM
    checkpoint_filepath = './model_weights/deepLOB_weights_FI2010_100/weights_random'

    trainX_CNN, trainY_CNN = prepare_x_y(train, k, T)
    valX_CNN, valY_CNN = prepare_x_y(val, k, T)
    testX_CNN, testY_CNN = prepare_x_y(test, k, T)

    random.shuffle(trainY_CNN)
    random.shuffle(valY_CNN)

    print(train.shape)
    print(trainX_CNN.shape, trainY_CNN.shape)
    print(valX_CNN.shape, valY_CNN.shape)
    print(testX_CNN.shape, testY_CNN.shape)

    # build model
    deeplob = create_deeplob(trainX_CNN.shape[1], trainX_CNN.shape[2], n_hiddens)
    deeplob.summary()

    # train model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=True)

    # avoid RAM issues
    generator = tf.keras.preprocessing.image.ImageDataGenerator()
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20)

    deeplob.fit(generator.flow(trainX_CNN, trainY_CNN, batch_size=32, shuffle=False),
                validation_data=(valX_CNN, valY_CNN), steps_per_epoch=len(trainX_CNN) // 32,
                epochs=50, verbose=1, callbacks=[model_checkpoint_callback, early_stopping])

    # evaluate model performance
    deeplob.load_weights(checkpoint_filepath)

    # avoid RAM issues
    pred = deeplob.predict(generator.flow(testX_CNN, batch_size=32, shuffle=False), steps=len(testX_CNN) // 32)

    print('accuracy_score:', accuracy_score(np.argmax(testY_CNN, axis=1), np.argmax(pred, axis=1)))
    print(classification_report(np.argmax(testY_CNN, axis=1), np.argmax(pred, axis=1), digits=4))



