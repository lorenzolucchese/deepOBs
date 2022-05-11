import numpy as np
import glob
import os
import tensorflow as tf
from keras.layers import Reshape, concatenate
from data_generators import CustomDataGenerator
from data_prepare import get_alphas

if __name__ == '__main__':
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
    NF = 40
    horizon = 0
    data_gen = CustomDataGenerator(data_dir, files["val"], NF, horizon, task = "classification", alphas = alphas, multihorizon = False, normalise = False, batch_size=256, shuffle=False, teacher_forcing=False, window=100)
    n_batches = data_gen.__len__()
    print(n_batches)
    for i in range(n_batches):
        batch = data_gen.__getitem__(i)
        if i % 10000 == 0:
            print(batch)
