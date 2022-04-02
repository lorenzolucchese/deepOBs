import numpy as np
import pandas as pd
from deepLOB_FI2010 import prepare_x_y


if __name__ == '__main__':
    # prepare data
    data = np.loadtxt(r'data/FI2010/Train_Dst_NoAuction_ZScore_CF_7.txt').T
    train = data[:int(np.floor(data.shape[0] * 0.8)), :]
    val = data[int(np.floor(data.shape[0] * 0.8)):, :]

    test1 = np.loadtxt(r'data/FI2010/Test_Dst_NoAuction_ZScore_CF_7.txt').T
    test2 = np.loadtxt(r'data/FI2010/Test_Dst_NoAuction_ZScore_CF_8.txt').T
    test3 = np.loadtxt(r'data/FI2010/Test_Dst_NoAuction_ZScore_CF_9.txt').T

    test = np.vstack((test1, test2, test3))

    T = 100
    n_hiddens = 64

    horizons = [10, 20, 30, 50, 100]
    datasets = [val, train, test]

    for dataset in datasets:
        classes = np.zeros((3, 5))

        for k in range(5):
            X, y = prepare_x_y(dataset, k, T)
            classes[:, k] = np.sum(y, axis = 0) / len(y)
        
        distributions = pd.DataFrame(classes, 
                                    index = ["down", "stationary", "up"], 
                                    columns = ["10", "20", "30", "50", "100"])
        
        print(distributions)