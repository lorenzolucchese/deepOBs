import pickle
import os
import numpy as np
import pandas as pd
from MCS_results import classification_report_to_df
from sklearn.metrics import confusion_matrix, classification_report
import glob
from datetime import datetime
import re
from functools import *

def make_empirical_AR_model(tickers, periods, horizons, processed_data_path, results_path, stats_path, k=10):
    """
    Compute confusion matrix, classification report and categorical cross entropy loss of empirical AR model on test set,
    i.e. for each ticker-period-horizon combination, the conditional train-val distributions (given the last h-step return has been down, stationary or up). 
    Assume that prediction is the argmax of probabilities.
    Save as .pkl's for each ticker-period-horizon combination.
    :param tickers: tickers under consideration, list of str
    :param periods: periods under consideration, list of str
    :param horizons: horizons under consideration, list of str
    :param processed_data_path: the path where the processed data is stored, str
    :param results_path: the path where the results are stored, str
    :param stats_path: the path where stats are to be saved, str
    :param k: smoothing window for averaging prices in return definition, int
    """
    for ticker in tickers:
        dependence_responses = pd.read_csv(os.path.join(stats_path, ticker + "_dependence_responses.csv"), index_col=(0,1,2))
        npz_file_list = sorted(glob.glob(os.path.join(processed_data_path, ticker, "*.{}".format("npz"))))
        for period in periods:
            os.makedirs(os.path.join(results_path, ticker, period, 'empirical_AR_model'), exist_ok=True)
            with open(os.path.join(results_path, ticker, period, "alphas.pkl"), 'rb') as f:
                alphas = pickle.load(f)
            with open(os.path.join(results_path, ticker, period, 'val_train_test_dates.pkl'), 'rb') as f:
                dates = pickle.load(f)
            trainval_dates = dates[0] + dates[1]
            for h, horizon in enumerate(horizons):
                # compute conditional distributions
                dependence_matrix = np.zeros((3, 3))
                for date in trainval_dates:
                    if date in dependence_responses.index.get_level_values(0):
                        dependence_matrix += dependence_responses.loc[(date, horizon), :].values
                train_val_conditional_distributions = dependence_matrix / dependence_matrix.sum(axis=1, keepdims=True)
                npz_file_list_window = [file for file in npz_file_list if re.search(r'\d{4}-\d{2}-\d{2}', file).group() in dates[2]]
                print(horizon)
                print(npz_file_list_window)
                past_labels = np.array([])
                target_labels = np.array([])
                for file in npz_file_list_window:
                    date = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', file).group(), '%Y-%m-%d').date()
                    with np.load(file) as data:
                        responses = data['mid_returns']
                    labels = (+1)*(responses[:, h] >= -alphas[h]) + (+1)*(responses[:, h] > alphas[h])
                    past_labels = np.append(past_labels, labels[:-horizon-k//2])
                    target_labels = np.append(target_labels, labels[horizon+k//2:])
                predicted_distributions = np.zeros((len(target_labels), 3))
                for _, label in enumerate(past_labels):
                    predicted_distributions[_, :] = train_val_conditional_distributions[int(label), :]
                predicted_labels = np.argmax(predicted_distributions, axis=1)
                os.makedirs(os.path.join(results_path, ticker, period, 'empirical_AR_model', 'h' + str(horizon)), exist_ok=True)
                # confusion matrix
                confusion_matrix_ = confusion_matrix(target_labels, predicted_labels)
                # classification report
                classification_report_ = classification_report(target_labels, predicted_labels, digits=4, output_dict=True, zero_division=0)
                classification_report_ = classification_report_to_df(classification_report_)
                # benchmark categorical cross entropy loss
                one_hot_target_labels = pd.get_dummies(target_labels).values
                categorical_crossentropy = - np.sum(one_hot_target_labels * np.log(predicted_distributions)) / len(target_labels)
                pickle.dump(confusion_matrix_, open(os.path.join(results_path, ticker, period, 'empirical_AR_model', 'h' + str(horizon), 'confusion_matrix_test.pkl'), 'wb'))
                pickle.dump(classification_report_, open(os.path.join(results_path, ticker, period, 'empirical_AR_model', 'h' + str(horizon), 'classification_report_test.pkl'), 'wb'))
                pickle.dump(categorical_crossentropy, open(os.path.join(results_path, ticker, period, 'empirical_AR_model', 'h' + str(horizon), 'categorical_crossentropy_test.pkl'), 'wb'))
