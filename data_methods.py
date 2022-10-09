import pandas as pd
import numpy as np
import tensorflow as tf


def get_alphas(files, orderbook_updates, distribution=True):
    """
    Empirically estimate alphas for (down, no change, up) categorical distributions from files (either orderbook or orderflow features).
    At each horizon we set alpha = (|Q(0.33)| + Q(0.66))/2 where Q() is the edf of returns in files.
    :param files: processed .csv orderbook/orderflow files of size (:, NF' + tot_horizons), list of str
    :param orderbook_updates: number of orderbook updates corresponding to each horizon, (tot_horizons,) array
    :param distribution: whether to return the distributions of each class for the selected alphas, bool
    :return: alphas: estimated alpha at each horizon, (tot_horizons,) array
             if distribution = True:
                distributions: (down, no change, up) distributions at each horizon, (3, tot_horizons) dataframe
    """
    tot_horizons = len(orderbook_updates)
    returns = []
    for file in files:
        print(file)
        df = pd.read_csv(file)
        df = df.dropna()
        df = df.to_numpy()
        returns.append(df[:, -tot_horizons:])
    returns = np.vstack(returns)
    alphas = (np.abs(np.quantile(returns, 0.33, axis = 0)) + np.quantile(returns, 0.66, axis = 0))/2
    if distribution:
        n = returns.shape[0]
        class0 = np.array([sum(returns[:, i] < -alphas[i])/n for i in range(tot_horizons)])
        class2 = np.array([sum(returns[:, i] > alphas[i])/n for i in range(tot_horizons)])
        class1 = 1 - (class0 + class2)
        distributions = pd.DataFrame(np.vstack([class0, class1, class2]), 
                                    index=["down", "stationary", "up"], 
                                    columns=orderbook_updates)
        return alphas, distributions
    return alphas


def get_class_distributions(files, alphas, orderbook_updates):
    """
    For a given set of files and alphas return the (down, no change, up) distributions of the returns.
    :param files: processed .csv orderbook/orderflow files of size (:, NF' + tot_horizons), list of str
    :param alphas: alphas for classification (down, no change, up) = ((-infty, -alpha), [-alpha, +alpha] (+alpha, +infty)), (tot_horizons,) array
    :param orderbook_updates: number of orderbook updates corresponding to each horizon, (tot_horizons,) array
    :return: distributions: (down, no change, up) distributions at each horizon, (3, tot_horizons) dataframe
    """
    tot_horizons = len(orderbook_updates)
    returns = []
    for file in files:
        df = pd.read_csv(file)
        df = df.dropna()
        df = df.to_numpy()
        returns.append(df[:, -tot_horizons:])
    returns = np.vstack(returns)
    n = returns.shape[0]
    class0 = np.array([sum(returns[:, i] < -alphas[i])/n for i in range(tot_horizons)])
    class2 = np.array([sum(returns[:, i] > alphas[i])/n for i in range(tot_horizons)])
    class1 = 1 - (class0 + class2)
    distributions = pd.DataFrame(np.vstack([class0, class1, class2]), 
                                 index = ["down", "stationary", "up"], 
                                 columns = orderbook_updates)
    return distributions


def get_class_distributions_univ(dict_of_files, dict_of_alphas, orderbook_updates):
    """
    For a set of TICKERs and corresponding files and alphas return the (down, no change, up) distributions of the returns.
    :param dict_of_files: processed .csv orderbook/orderflow files of size (:, NF' + tot_horizons), dict of lists of str
    :param dict_of_alphas: alphas for classification (down, no change, up) = ((-infty, -alpha), [-alpha, +alpha] (+alpha, +infty)), dict of (tot_horizons,) arrays
    :param orderbook_updates: number of orderbook updates corresponding to each horizon, (tot_horizons,) array
    :return: distributions: (down, no change, up) distributions at each horizon, (3, tot_horizons) dataframe
    """
    tot_horizons = len(orderbook_updates)
    n = 0
    class0 = np.array([0 for _ in range(tot_horizons)])
    class2 = np.array([0 for _ in range(tot_horizons)])
    for TICKER in dict_of_files.keys():
        files = dict_of_files[TICKER]
        alphas = dict_of_alphas[TICKER]
        returns = []
        for file in files:
            df = pd.read_csv(file)
            df = df.dropna()
            df = df.to_numpy()
            returns.append(df[:, -tot_horizons:])
        returns = np.vstack(returns)
        n += returns.shape[0]
        class0 += np.array([sum(returns[:, i] < -alphas[i]) for i in range(tot_horizons)])
        class2 += np.array([sum(returns[:, i] > alphas[i]) for i in range(tot_horizons)])
    class0 = class0/n
    class2 = class2/n
    class1 = 1 - (class0 + class2)
    distributions = pd.DataFrame(np.vstack([class0, class1, class2]), 
                                 index = ["down", "stationary", "up"], 
                                 columns = orderbook_updates)
    return distributions