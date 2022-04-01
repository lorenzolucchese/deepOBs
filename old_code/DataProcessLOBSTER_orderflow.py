import pandas as pd
import numpy as np
import os
import glob

import multiprocess as mp


def prepare_x(data):
    df1 = data[:, :20]
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
    y = y[:, k].astype(int)
    y = np.eye(3)[y]
    return x, y


def process_data(files, k, T, dir, type, samples_per_file=1, XYsplit=True):
    print(type)

    def process_chunk(X, Y, file_id, chunk_id):
        for j in range(len(X)//samples_per_file):
            # note that due to batch specification the last batch may have size smaller than samples_per_file
            id = str(file_id) + str(chunk_id) + str(j)
            if XYsplit:
                np.save(os.path.join(dir, type, "X", type + "X" + id), X[(samples_per_file*j):(samples_per_file*(j+1)), ])
                np.save(os.path.join(dir, type, "Y", type + "Y" + id), Y[(samples_per_file*j):(samples_per_file*(j+1)), ])
            else:
                np.savez(os.path.join(dir, type, type + id),
                         X=X[(samples_per_file*j):(samples_per_file*(j+1)), ],
                         Y=Y[(samples_per_file*j):(samples_per_file*(j+1)), ])
        return len(X)

    for file_id, file in enumerate(files):
        print(file)
        df = pd.read_csv(file).to_numpy()
        X, Y = prepare_x_y(df, k, T)

        # parallelize
        n_proc = mp.cpu_count()
        chunksize = len(X) // n_proc
        Xproc_chunks, Yproc_chunks = [], []

        for i_proc in range(n_proc):
            chunkstart = i_proc * chunksize
            # make sure to include the division remainder for the last process
            chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None

            Xproc_chunks.append(X[slice(chunkstart, chunkend), ])
            Yproc_chunks.append(Y[slice(chunkstart, chunkend), ])

        with mp.Pool(processes=n_proc) as pool:
            # starts the sub-processes without blocking
            # pass the chunk to each worker process
            proc_results = [pool.apply_async(process_chunk,  args=(Xchunk, Ychunk, file_id, chunck_id))
                            for chunck_id, (Xchunk, Ychunk) in enumerate(zip(Xproc_chunks, Yproc_chunks))]

            # blocks until all results are fetched
            [r.get() for r in proc_results]


if __name__ == '__main__':
    k = 4
    # which prediction horizon (k = (0, 1, 2, 3, 4) -> (10, 20, 30, 50, 100) order book events)
    T = 100
    # the length of a sample sequence. Even though this is a single long time series, LSTMs usually work with
    # input sequences of max length 200-400, we hence split the time series into sequences of length 100
    # rolling forward by one time-step each time.

    # prepare data: use (1, 4, 1) window: one week for validation, one week for training, one week for testing
    split = {"val": 5,
             "train": 20,
             "test": 5}

    extension = "csv"
    # csv_file_list = glob.glob("data/AAL_orderbooks/*.{}".format(extension))
    csv_file_list = glob.glob("data/AAL_orderflows/*.{}".format(extension))
    csv_file_list.sort()

    val_files = csv_file_list[0:split["val"]]
    train_files = csv_file_list[split["val"]:(split["val"] + split["train"])]
    test_files = csv_file_list[(split["val"] + split["train"]):(split["val"] + split["train"] + split["test"])]

#    process_data(val_files, k, T, r"data/AAL_OB_W1_batch32", "val", samples_per_file=32, XYsplit=False)
#    process_data(train_files, k, T, r"data/AAL_OB_W1_batch32", "train", samples_per_file=32, XYsplit=False)
#    process_data(test_files, k, T, r"data/AAL_OB_W1_batch32", "test", samples_per_file=32, XYsplit=False)

    process_data(val_files, k, T, r"data/AAL_OF_W1_batch32", "val", samples_per_file=32, XYsplit=False)
    process_data(train_files, k, T, r"data/AAL_OF_W1_batch32", "train", samples_per_file=32, XYsplit=False)
    process_data(test_files, k, T, r"data/AAL_OF_W1_batch32", "test", samples_per_file=32, XYsplit=False)
    