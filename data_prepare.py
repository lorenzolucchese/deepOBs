import pandas as pd
import numpy as np
import os
import tensorflow as tf

import multiprocess as mp


def prepare_x(data, NF, T, normalise=False):
    dataX = np.array(data[:, :NF])
    N = dataX.shape[0]
    x = np.zeros((N - T + 1, T, NF))
    for i in range(T, N + 1):
        x[i - T] = dataX[i - T:i, :]
        if normalise:
            x[i - T] = x[i - T] / np.max(x[i - T])
    x = x.reshape(x.shape + (1,))
    return x


def prepare_y(data, T, n_horizons = 5):
    labels = data[(T-1):, -n_horizons:]
    all_label = []
    for i in range(labels.shape[1]):
        one_label = labels[:, i] - 1 
        one_label = tf.keras.utils.to_categorical(one_label, 3)
        one_label = one_label.reshape(len(one_label), 1, 3)
        all_label.append(one_label)
    y = np.hstack(all_label)
    return y


def get_label(y_reg, alphas):
    all_label = []
    for i in range(y_reg.shape[1]):
        one_label = (+1)*(y_reg[:, i]>=-alphas[i]) + (+1)*(y_reg[:, i]>alphas[i])
        one_label = tf.keras.utils.to_categorical(one_label, 3)
        one_label = one_label.reshape(len(one_label), 1, 3)
        all_label.append(one_label)
    y = np.hstack(all_label)
    return y


def prepare_x_y(data, T, NF, alphas, n_horizons = 5, normalise=False):
    x = prepare_x(data, NF, T, normalise=normalise)
    y_reg = data[(T - 1):, -n_horizons:]
    y_class = get_label(y_reg, alphas)
    return x, y_reg, y_class


def get_alphas(files, orderbook_updates, distribution=True):
    n_horizons = len(orderbook_updates)
    returns = []
    for file in files:
        print(file)
        df = pd.read_csv(file)
        df = df.dropna()
        df = df.to_numpy()
        returns.append(df[:, -n_horizons:])
    returns = np.vstack(returns)
    alphas = (np.abs(np.quantile(returns, 0.33, axis = 0)) + np.quantile(returns, 0.66, axis = 0))/2
    if distribution:
        n = returns.shape[0]
        class0 = np.array([sum(returns[:, i] < -alphas[i])/n for i in range(n_horizons)])
        class2 = np.array([sum(returns[:, i] > alphas[i])/n for i in range(n_horizons)])
        class1 = 1 - (class0 + class2)
        distributions = pd.DataFrame(np.vstack([class0, class1, class2]), 
                                    index=["down", "stationary", "up"], 
                                    columns=orderbook_updates)
        return alphas, distributions
    return alphas


def get_class_distributions(files, alphas, orderbook_updates):
    n_horizons = len(orderbook_updates)
    returns = []
    for file in files:
        df = pd.read_csv(file)
        df = df.dropna()
        df = df.to_numpy()
        returns.append(df[:, -n_horizons:])
    returns = np.vstack(returns)
    n = returns.shape[0]
    class0 = np.array([sum(returns[:, i] < -alphas[i])/n for i in range(n_horizons)])
    class2 = np.array([sum(returns[:, i] > alphas[i])/n for i in range(n_horizons)])
    class1 = 1 - (class0 + class2)
    distributions = pd.DataFrame(np.vstack([class0, class1, class2]), 
                                 index = ["down", "stationary", "up"], 
                                 columns = orderbook_updates)
    return distributions

def get_class_distributions_univ(dict_of_files, dict_of_alphas, orderbook_updates):
    n_horizons = len(orderbook_updates)
    n = 0
    class0 = np.array([0 for _ in range(n_horizons)])
    class2 = np.array([0 for _ in range(n_horizons)])
    for TICKER in dict_of_files.keys():
        files = dict_of_files[TICKER]
        alphas = dict_of_alphas[TICKER]
        returns = []
        for file in files:
            df = pd.read_csv(file)
            df = df.dropna()
            df = df.to_numpy()
            returns.append(df[:, -n_horizons:])
        returns = np.vstack(returns)
        n += returns.shape[0]
        class0 += np.array([sum(returns[:, i] < -alphas[i]) for i in range(n_horizons)])
        class2 += np.array([sum(returns[:, i] > alphas[i]) for i in range(n_horizons)])
    class0 = class0/n
    class2 = class2/n
    class1 = 1 - (class0 + class2)
    distributions = pd.DataFrame(np.vstack([class0, class1, class2]), 
                                 index = ["down", "stationary", "up"], 
                                 columns = orderbook_updates)
    return distributions


def prepare_decoder_input(data, teacher_forcing):
    if teacher_forcing:
        first_decoder_input = tf.keras.utils.to_categorical(np.zeros(len(data)), 3)
        first_decoder_input = first_decoder_input.reshape(len(first_decoder_input), 1, 3)
        decoder_input_data = np.hstack((data[:, :-1, :], first_decoder_input))

    if not teacher_forcing:
        decoder_input_data = np.zeros((len(data), 1, 3))
        decoder_input_data[:, 0, 0] = 1.

    return decoder_input_data


def process_data(files, T, NF, dir, type, alphas, samples_per_file=1, XYsplit=True, normalise=False):
    def process_chunk(X, Y_reg, Y_class, file_id, chunk_id):
        for j in range(len(X)//samples_per_file):
            # note that due to batch specification the last batch may have size smaller than samples_per_file
            id = str(file_id) + str(chunk_id) + str(j)
            if XYsplit:
                np.save(os.path.join(dir, type, "X", type + "X" + id), 
                        X[(samples_per_file*j):(samples_per_file*(j+1)), ])
                np.save(os.path.join(dir, type, "Y_reg", type + "Y_reg" + id), 
                        Y_reg[(samples_per_file*j):(samples_per_file*(j+1)), ])
                np.save(os.path.join(dir, type, "Y_class", type + "Y_class" + id), 
                        Y_class[(samples_per_file*j):(samples_per_file*(j+1)), ])
            else:
                np.savez(os.path.join(dir, type, type + id),
                         X=X[(samples_per_file*j):(samples_per_file*(j+1)), ],
                         Y_reg=Y_reg[(samples_per_file*j):(samples_per_file*(j+1)), ],
                         Y_class=Y_class[(samples_per_file*j):(samples_per_file*(j+1)), ])
        return len(X)

    for file_id, file in enumerate(files):
        print(file)
        df = pd.read_csv(file)
        df = df.dropna()
        df = df.to_numpy()
        X, Y_reg, Y_class = prepare_x_y(df, T, NF, alphas, normalise=normalise)

        # parallelize
        n_proc = mp.cpu_count()
        chunksize = len(X) // n_proc
        X_proc_chunks, Y_reg_proc_chunks, Y_class_proc_chunks = [], [], []

        for i_proc in range(n_proc):
            chunkstart = i_proc * chunksize
            # make sure to include the division remainder for the last process
            chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None

            X_proc_chunks.append(X[slice(chunkstart, chunkend), ])
            Y_reg_proc_chunks.append(Y_reg[slice(chunkstart, chunkend), ])
            Y_class_proc_chunks.append(Y_class[slice(chunkstart, chunkend), ])

        with mp.Pool(processes=n_proc) as pool:
            # starts the sub-processes without blocking
            # pass the chunk to each worker process
            proc_results = [pool.apply_async(process_chunk,  args=(X_chunk, Y_reg_chunk, Y_class_chunk, file_id, chunck_id))
                            for chunck_id, (X_chunk, Y_reg_chunk, Y_class_chunk) in enumerate(zip(X_proc_chunks, Y_reg_proc_chunks, Y_class_proc_chunks))]

            # blocks until all results are fetched
            [r.get() for r in proc_results]


if __name__ == '__main__':

    #################################### SETTINGS ########################################
    data = "LOBSTER"                            # "LOBSTER" or "FI2010"

    raw_data_dir = r"data/FI2010"               # r"data/FI2010"
    processed_data_dir = r"data/model/FI2010"

    NF = 40                                     # number of features
    T = 100
    #######################################################################################

    if data == "LOBSTER":
        pass

    elif data == "FI2010":
        data = np.loadtxt(os.path.join(raw_data_dir, "Train_Dst_NoAuction_ZScore_CF_7.txt")).T
        train = data[:int(np.floor(data.shape[0] * 0.8)), :]
        val = data[int(np.floor(data.shape[0] * 0.8)):, :]

        test1 = np.loadtxt(os.path.join(raw_data_dir, "Test_Dst_NoAuction_ZScore_CF_7.txt")).T
        test2 = np.loadtxt(os.path.join(raw_data_dir, "Test_Dst_NoAuction_ZScore_CF_8.txt")).T
        test3 = np.loadtxt(os.path.join(raw_data_dir, "Test_Dst_NoAuction_ZScore_CF_9.txt")).T

        test = np.vstack((test1, test2, test3))

        trainX, trainY = prepare_x(train, NF, T, n_horizons=5), prepare_y(train, T, n_horizons=5)
        valX, valY = prepare_x(val, NF, T, n_horizons=5), prepare_y(val, T, n_horizons=5)
        testX, testY = prepare_x(test, NF, T, n_horizons=5), prepare_y(test, T, n_horizons=5)

        np.savez(os.path.join(processed_data_dir, "train"), X=trainX, Y=trainY)        
        np.savez(os.path.join(processed_data_dir, "val"), X=valX, Y=valY)
        np.savez(os.path.join(processed_data_dir, "test"), X=testX, Y=testY)

        # this sets the initial hidden state of the decoder to be y_0 = [1, 0, 0].
        train_decoder_input = prepare_decoder_input(trainX, teacher_forcing=False)
        val_decoder_input = prepare_decoder_input(valX, teacher_forcing=False)
        test_decoder_input = prepare_decoder_input(testX, teacher_forcing=False)

        np.save(os.path.join(processed_data_dir, "train_decoder_input"), train_decoder_input)        
        np.save(os.path.join(processed_data_dir, "val_decoder_input"), val_decoder_input)
        np.save(os.path.join(processed_data_dir, "test_decoder_input"), test_decoder_input)


