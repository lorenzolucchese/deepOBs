import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
import os
import glob

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


def prepare_y(data, T):
    labels = data[(T-1):, -5:]
    all_label = []
    for i in range(labels.shape[1]):
        one_label = labels[:, i] - 1 
        one_label = to_categorical(one_label, 3)
        one_label = one_label.reshape(len(one_label), 1, 3)
        all_label.append(one_label)
    y = np.hstack(all_label)
    return y


def get_label(y_reg, alphas):
    all_label = []
    for i in range(y_reg.shape[1]):
        one_label = (+1)*(y_reg[:, i]>=-alphas[i]) + (+1)*(y_reg[:, i]>alphas[i])
        one_label = to_categorical(one_label, 3)
        one_label = one_label.reshape(len(one_label), 1, 3)
        all_label.append(one_label)
    y = np.hstack(all_label)
    return y


def prepare_x_y(data, T, NF, alphas, normalise=False):
    x = prepare_x(data, NF, T, normalise=normalise)
    y_reg = data[(T - 1):, -5:]
    y_class = get_label(y_reg, alphas)
    return x, y_reg, y_class


def get_alphas(files, distribution=True):
    returns = []
    for file in files:
        print(file)
        df = pd.read_csv(file)
        df = df.dropna()
        df = df.to_numpy()
        returns.append(df[:, -5:])
    returns = np.vstack(returns)
    alphas = (np.abs(np.quantile(returns, 0.33, axis = 0)) + np.quantile(returns, 0.66, axis = 0))/2
    if distribution:
        n = returns.shape[0]
        class0 = np.array([sum(returns[:, i] < -alphas[i])/n for i in range(5)])
        class2 = np.array([sum(returns[:, i] > alphas[i])/n for i in range(5)])
        class1 = 1 - (class0 + class2)
        print("train class distributions")
        distributions = pd.DataFrame(np.vstack([class0, class1, class2]), 
                                    index=["down", "stationary", "up"], 
                                    columns=["10", "20", "30", "50", "100"])
        print(distributions)
    return alphas


def get_class_distributions(files, alphas):
    returns = []
    for file in files:
        df = pd.read_csv(file)
        df = df.dropna()
        df = df.to_numpy()
        returns.append(df[:, -5:])
    returns = np.vstack(returns)
    n = returns.shape[0]
    class0 = np.array([sum(returns[:, i] < -alphas[i])/n for i in range(5)])
    class2 = np.array([sum(returns[:, i] > alphas[i])/n for i in range(5)])
    class1 = 1 - (class0 + class2)
    distributions = pd.DataFrame(np.vstack([class0, class1, class2]), 
                                 index = ["down", "stationary", "up"], 
                                 columns = ["10", "20", "30", "50", "100"])
    return distributions


def prepare_decoder_input(data, teacher_forcing):
    if teacher_forcing:
        first_decoder_input = to_categorical(np.zeros(len(data)), 3)
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

    raw_data_dir = r"data/AAL_volumes"          # r"data/AAL_orderbooks", r"data/AAL_orderflows", r"data/AAL_volumes" or r"data/FI2010"
    processed_data_dir = r"data/model/AAL_volumes_W1_2"

    NF = 40                                     # number of features
    T = 100
    
    # If data == LOBSTER
    samples_per_file = 256                      # how many samples to save per file
    split = {"val": 5, "train": 20, "test": 5}  # use (1, 4, 1) window: one week for validation, one week for training, one week for testing
    normalise = True                            # for volumes need to normalise
    #######################################################################################

    if data == "LOBSTER":
        extension = "csv"
        csv_file_list = glob.glob(os.path.join(raw_data_dir, "*.{}").format(extension))
        csv_file_list.sort()

        val_files = csv_file_list[0:split["val"]]
        train_files = csv_file_list[split["val"]:(split["val"] + split["train"])]
        test_files = csv_file_list[(split["val"] + split["train"]):(split["val"] + split["train"] + split["test"])]

        print("getting alphas...")
        alphas = get_alphas(train_files, distribution=True)

        print("alphas = ", alphas)

        print("val class distributions")
        print(get_class_distributions(val_files, alphas))
        print("test class distributions")
        print(get_class_distributions(test_files, alphas))

        print("processing files to batches...")
        process_data(val_files, T, NF, processed_data_dir, "val", alphas, samples_per_file=samples_per_file, XYsplit=False, normalise=normalise)
        process_data(train_files, T, NF, processed_data_dir, "train", alphas, samples_per_file=samples_per_file, XYsplit=False, normalise=normalise)
        process_data(test_files, T, NF, processed_data_dir, "test", alphas, samples_per_file=samples_per_file, XYsplit=False, normalise=normalise)

    elif data == "FI2010":
        data = np.loadtxt(os.path.join(raw_data_dir, "Train_Dst_NoAuction_ZScore_CF_7.txt")).T
        train = data[:int(np.floor(data.shape[0] * 0.8)), :]
        val = data[int(np.floor(data.shape[0] * 0.8)):, :]

        test1 = np.loadtxt(os.path.join(raw_data_dir, "Test_Dst_NoAuction_ZScore_CF_7.txt")).T
        test2 = np.loadtxt(os.path.join(raw_data_dir, "Test_Dst_NoAuction_ZScore_CF_8.txt")).T
        test3 = np.loadtxt(os.path.join(raw_data_dir, "Test_Dst_NoAuction_ZScore_CF_9.txt")).T

        test = np.vstack((test1, test2, test3))

        trainX, trainY = prepare_x(train, NF, T), prepare_y(train, T)
        valX, valY = prepare_x(val, NF, T), prepare_y(val, T)
        testX, testY = prepare_x(test, NF, T), prepare_y(test, T)

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


