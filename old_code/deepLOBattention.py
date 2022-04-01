"""
This code is adapted from the github code by Zihao Zhang and Stefan Zohren
'Multi-Horizon-Forecasting-for-Limit-Order-Books'
Oxford-Man Institute of Quantitative Finance, Department of Engineering Science, University of Oxford

The model is DeepLOB from paper [1]. We apply it to FI-2010 data set [2].

[1] Zhang Z, Zohren S. Multi-Horizon Forecasting for Limit Order Books: Novel Deep Learning Approaches and Hardware
    Acceleration using Intelligent Processing Units. 2021 Aug.
    https://arxiv.org/pdf/2105.10430.pdf
[2] Ntakaris A, Magris M, Kanniainen J, Gabbouj M, Iosifidis A. Benchmark dataset for mid‐price forecasting of limit
    order book data with machine learning methods. Journal of Forecasting. 2018 Dec; 37(8):852-66.
    https://arxiv.org/abs/1705.03233
"""
import os
import logging
import glob
import argparse
import sys
import time
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from tensorflow import keras
from keras import backend as K


def prepare_x(data):
    df1 = data[:, :40]
    return np.array(df1)


def get_label(data):
    lob = data[:, -5:]
    all_label = []

    for i in range(lob.shape[1]):
        one_label = lob[:, i] - 1
        one_label = keras.utils.to_categorical(one_label, 3)
        one_label = one_label.reshape(len(one_label), 1, 3)
        all_label.append(one_label)

    return np.hstack(all_label)


def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)
    dY = np.array(Y)
    dataY = dY[T - 1:N]
    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]
    return dataX.reshape(dataX.shape + (1,)), dataY


def create_dataset(x_train, y_train, batch_size, method, shuffle=False):
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if shuffle:
        train_ds = train_ds.shuffle(len(x_train))
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.map(lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))

    if method == 'train':
        return train_ds.repeat()

    if method == 'val':
        return train_ds

    if method == 'test':
        return train_ds

    if method == 'prediction':
        train_ds = tf.data.Dataset.from_tensor_slices((x_train))
        train_ds = train_ds.batch(batch_size, drop_remainder=True)
        train_ds = train_ds.map(lambda d: (tf.cast(d, tf.float32)))
        return train_ds


def evaluation_metrics(real_y, pred_y):
    real_y = real_y[:len(pred_y)]
    logging.info('-------------------------------')

    for i in range(real_y.shape[1]):
        print(f'Prediction horizon = {i}')
        print(f'accuracy_score = {accuracy_score(np.argmax(real_y[:, i], axis=1), np.argmax(pred_y[:, i], axis=1))}')
        print(f'classification_report = {classification_report(np.argmax(real_y[:, i], axis=1), np.argmax(pred_y[:, i], axis=1), digits=4)}')
        print('-------------------------------')


def prepare_decoder_input(data, teacher_forcing):
    if teacher_forcing:
        first_decoder_input = keras.utils.to_categorical(np.zeros(len(data)), 3)
        first_decoder_input = first_decoder_input.reshape(len(first_decoder_input), 1, 3)
        decoder_input_data = np.hstack((data[:, :-1, :], first_decoder_input))

    if not teacher_forcing:
        decoder_input_data = np.zeros((len(data), 1, 3))
        decoder_input_data[:, 0, 0] = 1.

    return decoder_input_data


def get_model_seq(latent_dim):

    # Cho 在 2014 年提出了 Encoder–Decoder 结构，即由两个 RNN 组成，
    # https://arxiv.org/pdf/1406.1078.pdf

    input_train = keras.layers.Input(shape=(100, 40, 1))

    conv_first1 = keras.layers.Conv2D(32, (1, 2), strides=(1, 2))(input_train)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = keras.layers.Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = keras.layers.Conv2D(32, (1, 10))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    # build the inception module
    convsecond_1 = keras.layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = keras.layers.Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = keras.layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = keras.layers.Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = keras.layers.MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = keras.layers.Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
    conv_reshape = keras.layers.Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)

    ############
    # seq2seq
    encoder_inputs = conv_reshape
    encoder = keras.layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    states = [state_h, state_c]

    # Set up the decoder, which will only process one time step at a time.
    decoder_inputs = keras.layers.Input(shape=(1, 3))
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_dense = keras.layers.Dense(3, activation='softmax')

    ######################
    all_outputs = []
    encoder_outputs = keras.layers.Reshape((1, int(encoder_outputs.shape[1])))(encoder_outputs)
    inputs = keras.layers.concatenate([decoder_inputs, encoder_outputs], axis=2)
    ######################

    # start off decoder with
    # inputs: y_0 = decoder_inputs (exogenous), c = encoder_outputs (hidden state only)
    # hidden state: h'_0 = h_T (encoder output states: hidden state, state_h, and cell state, state_c)

    for _ in range(5):

        # h'_t = f(h'_{t-1}, y_{t-1}, c)
        outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)

        # y = g(h'_t[0], c)
        outputs = decoder_dense(keras.layers.concatenate([outputs, encoder_outputs], axis=2))
        all_outputs.append(outputs)

        # [y_{t-1}, c]
        inputs = keras.layers.concatenate([outputs, encoder_outputs], axis=2)

        # h'_{t-1}
        states = [state_h, state_c]

    # Concatenate all predictions
    decoder_outputs = keras.layers.Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    model = keras.models.Model([input_train, decoder_inputs], decoder_outputs)
    return model


def get_model_attention(latent_dim):
    # Luong Attention
    # https://arxiv.org/abs/1508.04025

    input_train = keras.layers.Input(shape=(100, 40, 1))

    conv_first1 = keras.layers.Conv2D(32, (1, 2), strides=(1, 2))(input_train)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = keras.layers.Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = keras.layers.Conv2D(32, (1, 10))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    # build the inception module
    convsecond_1 = keras.layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = keras.layers.Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = keras.layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = keras.layers.Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = keras.layers.MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = keras.layers.Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
    conv_reshape = keras.layers.Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(
        convsecond_output)

    # seq2seq
    encoder_inputs = conv_reshape
    encoder = keras.layers.LSTM(latent_dim, return_state=True, return_sequences=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    states = [state_h, state_c]

    # Set up the decoder, which will only process one time step at a time.
    # The attention decoder will have a different context vector at each time step, depending on attention weights.
    decoder_inputs = keras.layers.Input(shape=(1, 3))
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_dense = keras.layers.Dense(3, activation='softmax', name='output_layer')

    # start off decoder with
    # inputs: y_0 = decoder_inputs (exogenous), c = encoder_state_h (h_T[0], final hidden state only)
    # hidden state: h'_0 = h_T (encoder output states: hidden state, state_h, and cell state, state_c)

    encoder_state_h = keras.layers.Reshape((1, int(state_h.shape[1])))(state_h)
    inputs = keras.layers.concatenate([decoder_inputs, encoder_state_h], axis=2)

    all_outputs = []
    all_attention = []

    for _ in range(5):

        # h'_t = f(h'_{t-1}, y_{t-1}, c_{t-1})
        outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)

        # dot attention weights, alpha_{i,t} = exp(h_i h'_{t}) / sum_{i=1}^T exp(h_i h'_{t})
        attention = keras.layers.dot([outputs, encoder_outputs], axes=2)
        attention = keras.layers.Activation('softmax')(attention)

        # context vector, weighted average of all hidden states of encoder, weights determined by attention
        # c_{t} = sum_{i=1}^T alpha_{i, t} h_i
        context = keras.layers.dot([attention, encoder_outputs], axes=[2, 1])
        context = keras.layers.BatchNormalization(momentum=0.6)(context)

        # y_t = g(h'_t, c_t)
        decoder_combined_context = keras.layers.concatenate([context, outputs])
        outputs = decoder_dense(decoder_combined_context)
        all_outputs.append(outputs)
        all_attention.append(attention)

        inputs = keras.layers.concatenate([outputs, context], axis=2)
        states = [state_h, state_c]

    # Concatenate all predictions
    decoder_outputs = keras.layers.Lambda(lambda x: K.concatenate(x, axis=1), name='outputs')(all_outputs)
    decoder_attention = keras.layers.Lambda(lambda x: K.concatenate(x, axis=1), name='attentions')(all_attention)

    # Define and compile model as previously
    model = keras.models.Model([input_train, decoder_inputs], decoder_outputs)

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

    T = 100
    epochs = 50
    batch_size = 32
    n_hidden = 64
    checkpoint_filepath = './model_weights/deepLOBattention_weights_FI2010/weights'

    # %%
    train = np.loadtxt('data/FI2010/Train_Dst_NoAuction_ZScore_CF_7.txt').T
    test1 = np.loadtxt('data/FI2010/Test_Dst_NoAuction_ZScore_CF_7.txt').T
    test2 = np.loadtxt('data/FI2010/Test_Dst_NoAuction_ZScore_CF_8.txt').T
    test3 = np.loadtxt('data/FI2010/Test_Dst_NoAuction_ZScore_CF_9.txt').T
    test = np.vstack((test1, test2, test3))

    # extract limit order book data from the FI-2010 dataset
    train_lob = prepare_x(train)
    test_lob = prepare_x(test)

    # extract label from the FI-2010 dataset
    train_label = get_label(train)
    test_label = get_label(test)

    # prepare training data. We feed past 100 observations into our algorithms.
    train_encoder_input, train_decoder_target = data_classification(train_lob, train_label, T)
    # this sets the initial hidden state of the decoder to be y_0 = [1, 0, 0].
    train_decoder_input = prepare_decoder_input(train_encoder_input, teacher_forcing=False)

    # prepare testing data. We feed past 100 observations into our algorithms.
    test_encoder_input, test_decoder_target = data_classification(test_lob, test_label, T)
    # this sets the initial hidden state of the decoder to be y_0 = [1, 0, 0].
    test_decoder_input = prepare_decoder_input(test_encoder_input, teacher_forcing=False)

    print(f'train_encoder_input.shape = {train_encoder_input.shape}, '
          f'train_decoder_input.shape = {train_decoder_input.shape}, '
          f'train_decoder_target.shape = {train_decoder_target.shape}')
    print(f'test_encoder_input.shape = {test_encoder_input.shape}, '
          f'test_decoder_input.shape = {test_decoder_input.shape}, '
          f'test_decoder_target.shape = {test_decoder_target.shape}')

    model = get_model_attention(n_hidden)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()

    split_train_val = int(np.floor(len(train_encoder_input) * 0.8))

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=True)

    model.fit([train_encoder_input[:split_train_val], train_decoder_input[:split_train_val]],
              train_decoder_target[:split_train_val],
              validation_data=([train_encoder_input[split_train_val:], train_decoder_input[split_train_val:]],
                               train_decoder_target[split_train_val:]),
              epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[model_checkpoint_callback])

    model.load_weights(checkpoint_filepath)
    pred = model.predict([test_encoder_input, test_decoder_input])

    evaluation_metrics(test_decoder_target, pred)

