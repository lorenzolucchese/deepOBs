import pandas as pd
import numpy as np
import tensorflow as tf


def prepare_x(data, NF, T, normalise=False):
    """
    Extract features from data.
    :param data: (:, NF' + n_horizons) dimensional array
    :param NF: number of features to keep, NF <= NF'
    :param T: lookback window of features
    :param normalise: whether to normalize each feature (by its maximum)
    :retrun: x: (:-T+1, T, NF) array containing the features
    """
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
    """
    Extract categorical responses from labelled data 0, 1, 2.
    :param data: (:, NF' + n_horizons) dimensional array
    :param n_horizons: number of horizons in data
    :param normalise: whether to normalize each feature (by its maximum)
    :retrun: y: (:-T+1, n_horizons, 3) array containing (down, no change, up) categorical responses
    """
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
    """
    Extract categorical responses from return ("regression") data.
    :param y_reg: (:, n_horizons) dimensional array containing raw returns
    :param alphas: threshold(s) for classification (down, no change, up) = ((-infty, -alpha), [-alpha, +alpha] (+alpha, +infty)), length n_horizons
    :retrun: y: (:, n_horizons, 3) array containing (down, no change, up) categorical responses
    """
    all_label = []
    for i in range(y_reg.shape[1]):
        one_label = (+1)*(y_reg[:, i]>=-alphas[i]) + (+1)*(y_reg[:, i]>alphas[i])
        one_label = tf.keras.utils.to_categorical(one_label, 3)
        one_label = one_label.reshape(len(one_label), 1, 3)
        all_label.append(one_label)
    y = np.hstack(all_label)
    return y


def prepare_x_y(data, T, NF, alphas, n_horizons = 5, normalise=False):
    """
    Extract features, regression responses and categorical responses from data.
    :param data: (:, NF' + n_horizons) dimensional array
    :param T: lookback window of features
    :param NF: number of features to keep, NF <= NF'
    :param alphas: threshold(s) for classification (down, no change, up) = ((-infty, -alpha), [-alpha, +alpha] (+alpha, +infty)), length n_horizons
    :param n_horizons: number of horizons in data
    :param normalise: whether to normalize each feature (by its maximum)
    :retrun: x: (:-T+1, T, NF) array containing the features
             y_reg: (:-T+1, n_horizons) array containing regression responses
             y_class: (:-T+1, n_horizons, 3) array containing (down, no change, up) categorical responses
    """
    x = prepare_x(data, NF, T, normalise=normalise)
    y_reg = data[(T - 1):, -n_horizons:]
    y_class = get_label(y_reg, alphas)
    return x, y_reg, y_class


def get_alphas(files, orderbook_updates, distribution=True):
    """
    Empirically estimate alphas for (down, no change, up) categorical distributions from files (either orderbook or orderflow features).
    At each horizon we set alpha = (|Q(0.33)| + Q(0.66))/2 where Q() is the edf of returns in files.
    :param files: list of processed orderbook/orderflow files with returns (:, NF' + n_horizons)
    :param orderbook_updates: (n_horizons,) array with the number of orderbook updates corresponding to each horizon
    :param distribution: whether to return the distributions of each class for the selected alphas
    :return: alphas: (n_horizons,) array containing the estimated alpha at each horizon 
             if distribution = True:
                distributions: (3, n_horizons) dataframe with (down, no change, up) distributions at each horizon
    """
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
    """
    For a given set of files and alphas return the (down, no change, up) distributions of the returns.
    :param files: list of processed orderbook/orderflow files with returns (:, NF' + n_horizons)
    :param alphas: (n_horizons,) array containing the alphas for classification (down, no change, up) = ((-infty, -alpha), [-alpha, +alpha] (+alpha, +infty))
    :param orderbook_updates: (n_horizons,) array with the number of orderbook updates corresponding to each horizon
    :return: distributions: (3, n_horizons) dataframe with (down, no change, up) distributions at each horizon
    """
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
    """
    For a set of TICKERs and corresponding files and alphas return the (down, no change, up) distributions of the returns.
    :param dict_of_files: dict of lists, each list corresponds to the processed orderbook/orderflow files of a TICKER (:, NF' + n_horizons)
    :param dict_of_alphas: dict of (n_horizons,) array containing the alphas of a given TICKER, (down, no change, up) = ((-infty, -alpha), [-alpha, +alpha] (+alpha, +infty))
    :param orderbook_updates: (n_horizons,) array with the number of orderbook updates corresponding to each horizon
    :return: distributions: (3, n_horizons) dataframe with (down, no change, up) distributions at each horizon
    """
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
    """
    For multihorizon models the dataset of (1, 3) dimensional decoder input.
    :param data: (:, T, NF) feature data
    :param teacher_forcing: whether to use teacher forcing
    :return: decoder_input_data: (:, 1, 3) decoder input data
    """
    if teacher_forcing:
        first_decoder_input = tf.keras.utils.to_categorical(np.zeros(len(data)), 3)
        first_decoder_input = first_decoder_input.reshape(len(first_decoder_input), 1, 3)
        decoder_input_data = np.hstack((data[:, :-1, :], first_decoder_input))

    if not teacher_forcing:
        decoder_input_data = np.zeros((len(data), 1, 3))

    return decoder_input_data