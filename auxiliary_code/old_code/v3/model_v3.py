from pyparsing import PrecededBy
from data_generators_v3 import CustomDataGenerator2
from data_prepare_v3 import get_alphas

import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import multiprocessing as mp
import time
import os
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Flatten, Dense, Dropout, LeakyReLU, Activation, Input, LSTM, CuDNNLSTM, Reshape, Conv2D, Conv3D, MaxPooling2D, concatenate, Lambda, dot, BatchNormalization, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, MeanSquaredError
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob


class CustomReshape(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CustomReshape, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(CustomReshape, self).build(input_shape)
    
    def call(self, input_data):
        batch_size = tf.shape(input_data)[0]
        T = tf.shape(input_data)[1]
        NF = tf.shape(input_data)[2]
        input_BID = tf.reshape(input_data[:, :, :NF//2, :], [batch_size, T, NF//2, 1, 1])
        input_BID = tf.reverse(input_BID, axis = [2])
        input_ASK = tf.reshape(input_data[:, :, NF//2:, :], [batch_size, T, NF//2, 1, 1])
        output = concatenate([input_BID, input_ASK], axis = 3)
        # if not tf.executing_eagerly():
        #     # Set the static shape for the result since it might lost during array_ops
        #     # reshape, eg, some `None` dim in the result could be inferred.
        #     output.set_shape(self.compute_output_shape(input_data.shape))
        return output

    def compute_output_shape(self, input_shape): 
        batch_size = input_shape[0]
        T = input_shape[1]
        NF = input_shape[2]
        return (batch_size, T, NF//2, 2, 1)


class deepLOB:
    def __init__(self, T, NF, horizon, number_of_lstm, data, data_dir, files = None, model_inputs = "order book", task = "classification", alphas=None, multihorizon=False, decoder = "seq2seq", n_horizons=5):
        """Initialization.
        :param T: time window 
        :param NF: number of features
        :param horizon: when not multihorizon, the horizon to consider
        :param number_of_lstm: number of nodes in lstm
        :param data: whether the data fits in the RAM and is thus divided in train, val, test datasets (and, if multihorizon, corresponding decoder_input) - "FI2010", "simulated"
                     or if custom data generator is required - "LOBSTER".
        :param data_dir: parent directory for data
        :param model_inputs: what type of inputs
        :param task: regression or classification
        :param multihorizon: whether the forecasts need to be multihorizon, if True this overrides horizon
        :param decoder: the decoder to use for multihorizon forecasts, seq2seq or attention
        :param n_horizons: the number of forecast horizons in multihorizon
        """
        self.T = T
        self.NF = NF
        self.horizon = horizon
        if multihorizon:
            self.horizon = slice(0, 5)
        self.number_of_lstm = number_of_lstm
        self.model_inputs = model_inputs
        self.task = task
        self.alphas = alphas
        self.multihorizon = multihorizon
        self.decoder = decoder
        self.n_horizons = n_horizons
        self.orderbook_updates = [10, 20, 30, 50, 100]
        self.data_dir = data_dir
        self.files = files
        self.data = data

        if data in ["FI2010", "simulated"]:
            train_data = np.load(os.path.join(data_dir, "train.npz"))
            trainX, trainY = train_data["X"], train_data["Y"]
            val_data = np.load(os.path.join(data_dir, "val.npz"))
            valX, valY = val_data["X"], val_data["Y"]
            test_data = np.load(os.path.join(data_dir, "test.npz"))
            testX, testY = test_data["X"], test_data["Y"]

            if not(multihorizon):
                trainY = trainY[:, self.horizon ,:]
                valY = valY[:, self.horizon , :]
                testY = testY[:, self.horizon ,:]

            if multihorizon:
                train_decoder_input = np.load(os.path.join(data_dir, "train_decoder_input.npy"))
                val_decoder_input = np.load(os.path.join(data_dir, "val_decoder_input.npy"))
                test_decoder_input = np.load(os.path.join(data_dir, "test_decoder_input.npy"))

                trainX = [trainX, train_decoder_input]
                valX = [valX, val_decoder_input]
                testX = [testX, test_decoder_input]

            generator = tf.keras.preprocessing.image.ImageDataGenerator()
            self.train_generator = generator.flow(trainX, trainY, batch_size=32, shuffle=True)
            self.val_generator = generator.flow(valX, valY, batch_size=32, shuffle=True)
            self.test_generator = generator.flow(testX, testY, batch_size=32, shuffle=False)
    
        elif data == "LOBSTER":
            self.train_generator = CustomDataGenerator2(dir = self.data_dir, files = self.files["train"], NF = self.NF, horizon = self.horizon, alphas = self.alphas, multihorizon = self.multihorizon)
            self.val_generator = CustomDataGenerator2(dir = self.data_dir, files = self.files["val"], NF = self.NF, horizon = self.horizon, alphas = self.alphas, multihorizon = self.multihorizon)
            self.test_generator = CustomDataGenerator2(dir = self.data_dir, files = self.files["test"], NF = self.NF, horizon = self.horizon, alphas = self.alphas, multihorizon = self.multihorizon, shuffle=False)

        else:
            raise ValueError('data must be either FI2010, simulated or LOBSTER.')


    def create_model(self):
        # network parameters
        if self.task == "classification":
            output_activation = "softmax"
            output_dim = 3
            loss = "categorical_crossentropy"
            if self.multihorizon:
                metrics = []
                for i in range(self.n_horizons):
                    h = str(self.orderbook_updates[i])
                    metrics.append([CategoricalAccuracy(name = "accuracy"+ h)])
            else:
                h = str(self.orderbook_updates[self.horizon])
                metrics = [CategoricalAccuracy(name = "accuracy"+ h)]
        elif self.task == "regression":
            output_activation = "linear"
            output_dim = 1
            loss = "mean_squared_error"
            metrics = ["mean_squared_error"]
            if self.multihorizon:
                metrics = []
                for i in range(self.n_horizons):
                    h = str(self.orderbook_updates[i])
                    metrics.append([MeanSquaredError(name = "mse"+ h)])
            else:
                h = str(self.orderbook_updates[self.horizon])
                metrics = [MeanSquaredError(name = "mse"+ h)]
        else:
            raise ValueError('task must be either classification or regression.')
        self.metrics = metrics

        adam = Adam(learning_rate=0.01, epsilon=1)

        input_lmd = Input(shape=(self.T, self.NF, 1))

        # build the convolutional block
        if self.model_inputs == "order book":
            conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
            conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
            conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
            conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
            conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
            conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        elif self.model_inputs == "volumes":
            input_reshaped = CustomReshape(0)(input_lmd)
            conv_first1 = Conv3D(32, (1, 2, 2), strides=(1, 1, 1))(input_reshaped)
            conv_first1 = Reshape((int(conv_first1.shape[1]), int(conv_first1.shape[2]), int(conv_first1.shape[4])))(conv_first1)
            conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
            conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
            conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
            conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
            conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        elif self.model_inputs == "order flow":
            conv_first1 = input_lmd
        else:
            raise ValueError('task must be either order book, order flow or volumes.')

        conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

        conv_first1 = Conv2D(32, (1, conv_first1.shape[2]))(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

        # build the inception module
        convsecond_1 = Conv2D(64, (1, 1), padding='same')(conv_first1)
        convsecond_1 = LeakyReLU(alpha=0.01)(convsecond_1)
        convsecond_1 = Conv2D(64, (3, 1), padding='same')(convsecond_1)
        convsecond_1 = LeakyReLU(alpha=0.01)(convsecond_1)

        convsecond_2 = Conv2D(64, (1, 1), padding='same')(conv_first1)
        convsecond_2 = LeakyReLU(alpha=0.01)(convsecond_2)
        convsecond_2 = Conv2D(64, (5, 1), padding='same')(convsecond_2)
        convsecond_2 = LeakyReLU(alpha=0.01)(convsecond_2)

        convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
        convsecond_3 = Conv2D(64, (1, 1), padding='same')(convsecond_3)
        convsecond_3 = LeakyReLU(alpha=0.01)(convsecond_3)

        convsecond_output = concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
        conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)
        conv_reshape = Dropout(0.2, noise_shape=(None, 1, int(conv_reshape.shape[2])))(conv_reshape, training=True)

        if not(self.multihorizon):
            # build the last LSTM layer
            conv_lstm = LSTM(self.number_of_lstm, batch_input_shape=(32, self.T, int(conv_reshape.shape[2])))(conv_reshape)
            out = Dense(output_dim, activation=output_activation)(conv_lstm)
            self.model = Model(inputs=input_lmd, outputs=out)
        
        else:
            if self.decoder == "seq2seq":
                # seq2seq
                encoder_inputs = conv_reshape
                encoder = LSTM(self.number_of_lstm, return_state=True)
                encoder_outputs, state_h, state_c = encoder(encoder_inputs)
                states = [state_h, state_c]

                # Set up the decoder, which will only process one time step at a time.
                decoder_inputs = Input(shape=(1, output_dim))
                decoder_lstm = LSTM(self.number_of_lstm, return_sequences=True, return_state=True)
                decoder_dense = Dense(output_dim, activation=output_activation)

                all_outputs = []
                encoder_outputs = Reshape((1, int(encoder_outputs.shape[1])))(encoder_outputs)
                inputs = concatenate([decoder_inputs, encoder_outputs], axis=2)

                # start off decoder with
                # inputs: y_0 = decoder_inputs (exogenous), c = encoder_outputs (hidden state only)
                # hidden state: h'_0 = h_T (encoder output states: hidden state, state_h, and cell state, state_c)

                for _ in range(self.n_horizons):

                    # h'_t = f(h'_{t-1}, y_{t-1}, c)
                    outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)

                    # y_t = g(h'_t[0], c)
                    outputs = decoder_dense(concatenate([outputs, encoder_outputs], axis=2))
                    all_outputs.append(outputs)

                    # [y_t, c]
                    inputs = concatenate([outputs, encoder_outputs], axis=2)

                    # h'_t
                    states = [state_h, state_c]

                # Concatenate all predictions
                decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
            
            elif self.decoder == "attention":
                # attention
                encoder_inputs = conv_reshape
                encoder = LSTM(self.number_of_lstm, return_state=True, return_sequences=True)
                encoder_outputs, state_h, state_c = encoder(encoder_inputs)
                states = [state_h, state_c]

                # Set up the decoder, which will only process one time step at a time.
                # The attention decoder will have a different context vector at each time step, depending on attention weights.
                decoder_inputs = Input(shape=(1, output_dim))
                decoder_lstm = LSTM(self.number_of_lstm, return_sequences=True, return_state=True)
                decoder_dense = Dense(output_dim, activation=output_activation, name='output_layer')

                # start off decoder with
                # inputs: y_0 = decoder_inputs (exogenous), c = encoder_state_h (h_T[0], final hidden state only)
                # hidden state: h'_0 = h_T (encoder output states: hidden state, state_h, and cell state, state_c)

                encoder_state_h = Reshape((1, int(state_h.shape[1])))(state_h)
                inputs = concatenate([decoder_inputs, encoder_state_h], axis=2)

                all_outputs = []
                all_attention = []

                for _ in range(self.n_horizons):

                    # h'_t = f(h'_{t-1}, y_{t-1}, c_{t-1})
                    outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)

                    # dot attention weights, alpha_{i,t} = exp(h_i h'_{t}) / sum_{i=1}^T exp(h_i h'_{t})
                    attention = dot([outputs, encoder_outputs], axes=2)
                    attention = Activation('softmax')(attention)

                    # context vector, weighted average of all hidden states of encoder, weights determined by attention
                    # c_{t} = sum_{i=1}^T alpha_{i, t} h_i
                    context = dot([attention, encoder_outputs], axes=[2, 1])
                    context = BatchNormalization(momentum=0.6)(context)

                    # y_t = g(h'_t, c_t)
                    decoder_combined_context = concatenate([context, outputs])
                    outputs = decoder_dense(decoder_combined_context)
                    all_outputs.append(outputs)
                    all_attention.append(attention)
                    
                    # [y_t, c_t]
                    inputs = concatenate([outputs, context], axis=2)

                    # h'_t
                    states = [state_h, state_c]

                # Concatenate all predictions
                decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1), name='outputs')(all_outputs)
                # decoder_attention = Lambda(lambda x: K.concatenate(x, axis=1), name='attentions')(all_attention)
            
            elif self.decoder == None:
                pass

            else:
                raise ValueError('multihorizon must be either seq2seq, attention or None.')

            self.model = Model([input_lmd, decoder_inputs], decoder_outputs)
        
        self.model.compile(loss=loss, metrics=metrics, optimizer=adam)

    def fit_model(self, epochs, batch_size, checkpoint_filepath, load_weights, load_weights_filepath, verbose = 1, patience=5):
        model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
        					                        save_weights_only=True,
						                            monitor='val_loss',
                                                    mode='auto',
                                                    save_best_only=True)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='auto')

        if load_weights == True:
            self.model.load_weights(load_weights_filepath)

        self.model.fit(self.train_generator, validation_data=self.val_generator,
                       epochs=epochs, batch_size=batch_size, verbose=verbose, workers=8,
                       max_queue_size=10,
                       callbacks=[model_checkpoint_callback, early_stopping])

    def evaluate_model(self, load_weights_filepath, eval_set = "test"):
        self.model.load_weights(load_weights_filepath)

        print("Evaluating performance on ", eval_set, "set...")

        if eval_set == "test":
            generator = self.test_generator
        elif eval_set == "val":
            generator = self.val_generator
        elif eval_set == "train":
            generator = self.train_generator
        else:
            raise ValueError("eval_set must be test, val or train.")
        
        # avoid RAM issues
        predY = np.squeeze(self.model.predict(generator))

        if self.data in ["FI2010", "simulated"]:
            eval_data = np.load(os.path.join(self.data_dir, eval_set + ".npz"))
            evalY = eval_data["Y"][:, self.horizon, ...]
        if self.data == "LOBSTER":
            evalY = np.zeros(predY.shape)
            eval_files = self.files[eval_set]
            index = 0
            for file in eval_files:
                file = pd.read_csv(file).to_numpy()
                true_y = files[self.T:, -5:]
                if self.task == "classification":
                    label_list = []
                    for h in range(5):
                        labels = to_categorical(true_y[:, h], 3)
                        labels = labels.reshape(len(labels), 1, 3)
                        label_list.append(labels)
                    true_y = np.hstack(label_list)
                true_y = true_y[:, self.horizon, ...]                    
                evalY[index:(index+true_y.shape[0]), ...] = true_y
                index = index + true_y.shape[0]
        if self.task == "classification":
            if not self.multihorizon:
                print("Prediction horizon:", self.orderbook_updates[self.horizon], " orderbook updates")
                print('accuracy_score:', accuracy_score(np.argmax(evalY, axis=1), np.argmax(predY, axis=1)))
                print(classification_report(np.argmax(evalY, axis=1), np.argmax(predY, axis=1), digits=4))
            else:
                for h in range(5):
                    print("Prediction horizon:", self.orderbook_updates[h], " orderbook updates")
                    print('accuracy_score:', accuracy_score(np.argmax(evalY[:, h, :], axis=1), np.argmax(predY[:, h, :], axis=1)))
                    print(classification_report(np.argmax(evalY[:, h, :], axis=1), np.argmax(predY[:, h, :], axis=1), digits=4))
        elif self.task == "regression":
            if not self.multihorizon:
                print("Prediction horizon:", self.orderbook_updates[self.horizon], " orderbook updates")
                print('MSE:', mean_squared_error(evalY, predY))
                print('MAE:', mean_absolute_error(evalY, predY))
                print('r2:', r2_score(evalY, predY))
                regression_fit_plot(evalY, predY, title = eval_set + str(self.orderbook_updates[self.horizon]), 
                                    path = os.path.join("plots", self.data, "single-horizon", eval_set + str(self.orderbook_updates[self.horizon]) + '.png'))
                
            else:
                for h in range(5):
                    print("Prediction horizon:", self.orderbook_updates[h], " orderbook updates")
                    print('MSE:', mean_squared_error(evalY[:, h], predY[:, h]))
                    print('MAE:', mean_absolute_error(evalY[:, h], predY[:, h]))
                    print('r2:', r2_score(evalY[:, h], predY[:, h]))
                    regression_fit_plot(evalY, predY, title = eval_set + str(self.orderbook_updates[h]), 
                                        path=os.path.join("plots", self.data, "multi-horizon", self.decoder, eval_set + str(self.orderbook_updates[h]) + '.png'))


def regression_fit_plot(evalY, predY, title, path):
    fig, ax = plt.subplots()
    mpl.rcParams['agg.path.chunksize'] = len(evalY)
    ax.scatter(evalY, predY, s=10, c='k', alpha=0.5)
    lims = [np.min([evalY, predY]), np.max([evalY, predY])]
    ax.plot(lims, lims, linestyle='--', color='k', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(title)
    ax.set_xlabel("True y")
    ax.set_ylabel("Pred y")
    fig.savefig(path)

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
    decoder = "seq2seq"                         # options: "seq2seq", "attention"

    T = 100
    NF = 40                                     # remember to change this when changing features
    n_horizons = 5
    horizon = 0                                 # prediction horizon (0, 1, 2, 3, 4) -> (10, 20, 30, 50, 100) order book events
    epochs = 50
    batch_size = 256                            # note we use 256 for LOBSTER, 32 for FI2010 or simulated
    number_of_lstm = 64

    checkpoint_filepath = './model_weights_new/deepOB_weights_AAL_W1/weights' + decoder
    load_weights = False
    load_weights_filepath = './model_weights_new/deepOB_weights_AAL_W1/weights' + decoder

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
                    n_horizons)

    model.create_model()

    model.model.summary()

    model.fit_model(epochs = epochs, 
                    batch_size = batch_size,
                    checkpoint_filepath = checkpoint_filepath,
                    load_weights = load_weights,
                    load_weights_filepath = load_weights_filepath)

    model.evaluate_model(load_weights_filepath=load_weights_filepath)
                
