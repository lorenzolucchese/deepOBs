import pickle
import os
import numpy as np
import pandas as pd
import random
from functools import *

def classification_report_str_to_df(report:str):
    report_data = []
    lines = report.split('\n')
    row_labels = [0, 1, 2, 'macro avg', 'weighted avg']
    i = 0
    for line in lines[2:]:
        row = {}
        row_data = line.split()[-4:]
        if len(row_data) < 4:
            continue
        row['class'] = row_labels[i]
        i += 1
        row['precision'] = float(row_data[0])
        row['recall'] = float(row_data[1])
        row['f1-score'] = float(row_data[2])
        row['support'] = int(row_data[3])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.set_index('class', inplace=True)
    return dataframe

def classification_report_dict_to_df(report:dict):
    if type(report) is pd.DataFrame: 
        return report
    else:
        report.pop('accuracy')
        dataframe = pd.DataFrame.from_dict(report).T
        dataframe.index = [0, 1, 2, 'macro avg', 'weighted avg']                                                        
        return dataframe

def all_classification_reports_to_df(from_type='str'):
    if from_type == 'str':
        classification_report_converter = classification_report_str_to_df
    elif from_type == 'dict':
        classification_report_converter = classification_report_dict_to_df
    else:
        raise ValueError('from_type must be str or dict.')
    tickers = ['AAL', 'AAPL', 'ATVI', 'CHTR', 'EXC', 'LILAK', 'PCAR', 'QRTEA', 'WBA', 'XRAY']
    periods = ['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10']
    models = ['deepLOB_L1', 'deepOF_L1', 'deepLOB_L2', 'deepOF_L2', 'deepVOL_L2', 'deepVOL_L3']
    horizons = ['h10', 'h20', 'h30', 'h50', 'h100', 'h200', 'h300', 'h500', 'h1000']
    for ticker in tickers:
        for period in periods: 
            for model in models:
                for horizon in horizons:
                    for set_ in ['val', 'train', 'test']:
                        classification_report = pickle.load(open('results/' + ticker + '/' + period + '/' + model + '/' + horizon + '/' + 'classification_report_' + set_ + '.pkl', 'rb'))
                        classification_report = classification_report_converter(classification_report)
                        pickle.dump(classification_report, open('results/' + ticker + '/' + period + '/' + model + '/' + horizon + '/' + 'classification_report_' + set_ + '.pkl', 'wb'))

# DEPRECATED
def make_test_val_distributions_old():
    tickers = ['AAL', 'AAPL', 'ATVI', 'CHTR', 'EXC', 'LILAK', 'PCAR', 'QRTEA', 'WBA', 'XRAY']
    periods = ['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10']
    horizons = ['h10', 'h20', 'h30', 'h50', 'h100', 'h200', 'h300', 'h500', 'h1000']
    orderbook_updates = [10, 20, 30, 50, 100, 200, 300, 500, 1000]
    for ticker in tickers:
        for period in periods:
            # note that since val and train are sub-sampled these may not correspond exactly to those given by submit_experiment.py 
            val_distributions = pd.DataFrame(np.zeros((3, len(orderbook_updates))),
                                     index = ["down", "stationary", "up"], 
                                     columns = orderbook_updates)
            test_distributions = pd.DataFrame(np.zeros((3, len(orderbook_updates))),
                                     index = ["down", "stationary", "up"], 
                                     columns = orderbook_updates)
            train_val_distributions = pd.DataFrame(np.zeros((3, len(orderbook_updates))),
                                     index = ["down", "stationary", "up"], 
                                     columns = orderbook_updates)
            for j, horizon in enumerate(horizons):
                support_train = pickle.load(open('results/' + ticker + '/' + period + '/deepLOB_L1/' + horizon + '/classification_report_train.pkl', 'rb'))['support'][:3]
                support_val = pickle.load(open('results/' + ticker + '/' + period + '/deepLOB_L1/' + horizon + '/classification_report_val.pkl', 'rb'))['support'][:3]
                support_test = pickle.load(open('results/' + ticker + '/' + period + '/deepLOB_L1/' + horizon + '/classification_report_test.pkl', 'rb'))['support'][:3]
                train_val_distributions.iloc[:, j] = (support_train.values + support_val.values) / (support_train.values + support_val.values).sum()
                test_distributions.iloc[:, j] = support_test.values / support_test.values.sum()
                val_distributions.iloc[:, j] = support_val.values / support_val.values.sum()
            pickle.dump(train_val_distributions, open('results/' + ticker + '/' + period + '/train_val_distributions.pkl', 'wb'))
            pickle.dump(test_distributions, open('results/' + ticker + '/' + period + '/test_distributions.pkl', 'wb'))
            pickle.dump(val_distributions, open('results/' + ticker + '/' + period + '/val_distributions.pkl', 'wb'))


def make_train_val_distributions():
    tickers = ['AAL', 'AAPL', 'ATVI', 'CHTR', 'EXC', 'LILAK', 'PCAR', 'QRTEA', 'WBA', 'XRAY']
    periods = ['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10']
    horizons = ['h10', 'h20', 'h30', 'h50', 'h100', 'h200', 'h300', 'h500', 'h1000']
    orderbook_updates = [10, 20, 30, 50, 100, 200, 300, 500, 1000]
    for ticker in tickers:
        for period in periods:
            # note that since val and train are sub-sampled these may not correspond exactly to those given by submit_experiment.py 
            val_distributions = pickle.load(open('results/' + ticker + '/' + period + '/val_distributions.pkl', 'rb'))
            train_distributions = pickle.load(open('results/' + ticker + '/' + period + '/distributions.pkl', 'rb'))
            train_val_distributions = pd.DataFrame(np.zeros((3, len(orderbook_updates))),
                                     index = ["down", "stationary", "up"], 
                                     columns = orderbook_updates)
            for j, horizon in enumerate(horizons):
                # note true support might be x10/x100 but since taking wavg this is the same
                support_train = pickle.load(open('results/' + ticker + '/' + period + '/deepLOB_L1/' + horizon + '/classification_report_train.pkl', 'rb')).loc['macro avg', 'support']
                support_val = pickle.load(open('results/' + ticker + '/' + period + '/deepLOB_L1/' + horizon + '/classification_report_val.pkl', 'rb')).loc['macro avg', 'support']
                train_val_distributions.iloc[:, j] = (train_distributions.iloc[:, j].values * support_train + val_distributions.iloc[:, j].values * support_val)/(support_train + support_val)
            pickle.dump(train_val_distributions, open('results/' + ticker + '/' + period + '/train_val_distributions.pkl', 'wb'))


def make_benchmark():
    tickers = ['AAL', 'AAPL', 'ATVI', 'CHTR', 'EXC', 'LILAK', 'PCAR', 'QRTEA', 'WBA', 'XRAY']
    periods = ['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10']
    horizons = ['h10', 'h20', 'h30', 'h50', 'h100', 'h200', 'h300', 'h500', 'h1000']
    for ticker in tickers:
        for period in periods:
            os.makedirs('results/' + ticker + '/' + period + '/benchmark', exist_ok=True)
            train_val_distributions = pickle.load(open('results/' + ticker + '/' + period + '/train_val_distributions.pkl', 'rb'))
            for j, horizon in enumerate(horizons):
                os.makedirs('results/' + ticker + '/' + period + '/benchmark/' + horizon, exist_ok=True)
                for set_ in ['train', 'test', 'val']:
                    pred_benchmark = np.argmax(train_val_distributions.iloc[:, j])
                    # load a sample confusion matrix and classification report
                    confusion_matrix = pickle.load(open('results/' + ticker + '/' + period + '/deepLOB_L1/' + horizon + '/confusion_matrix_' + set_ + '.pkl', 'rb'))
                    classification_report = pickle.load(open('results/' + ticker + '/' + period + '/deepLOB_L1/' + horizon + '/classification_report_' + set_ + '.pkl', 'rb'))
                    support = classification_report['support'][:3].values
                    # benchmark confusion matrix
                    confusion_matrix = np.zeros_like(confusion_matrix)
                    confusion_matrix[:, pred_benchmark] = support
                    # benchmark classification report
                    classification_report[['precision', 'recall', 'f1-score']] = 0
                    classification_report.loc[pred_benchmark, 'precision'] = support[pred_benchmark] / support.sum()
                    classification_report.loc[pred_benchmark, 'recall'] = 1.0
                    classification_report.loc[pred_benchmark, 'f1-score'] = 2 * support[pred_benchmark] / (support[pred_benchmark] + support.sum())
                    classification_report.loc['macro avg', ['precision', 'recall', 'f1-score']] = np.average(classification_report.loc[[0, 1, 2], ['precision', 'recall', 'f1-score']].values, axis=0)
                    classification_report.loc['weighted avg', ['precision', 'recall', 'f1-score']] = np.average(classification_report.loc[[0, 1, 2], ['precision', 'recall', 'f1-score']].values, axis=0, weights=support)
                    classification_report = classification_report.round(4)
                    # benchamark categorical cross entropy loss
                    cce = - np.sum(support * np.log(train_val_distributions.iloc[:, j].values)) / support.sum()
                    pickle.dump(confusion_matrix, open('results/' + ticker + '/' + period + '/benchmark/' + horizon + '/confusion_matrix_' + set_ + '.pkl', 'wb'))
                    pickle.dump(classification_report, open('results/' + ticker + '/' + period + '/benchmark/' + horizon + '/classification_report_' + set_ + '.pkl', 'wb'))
                    pickle.dump(cce, open('results/' + ticker + '/' + period + '/benchmark/' + horizon + '/categorical_crossentropy_' + set_ + '.pkl', 'wb'))


def f1_dataframe(TICKER, horizon, set_='test', avg_type = 'macro avg'):
    models = ['benchmark', 'deepLOB_L1', 'deepOF_L1', 'deepLOB_L2', 'deepOF_L2', 'deepVOL_L2', 'deepVOL_L3']
    periods = ['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10']
    dataframe = pd.DataFrame(np.zeros((len(periods), len(models))), columns = models, index = periods)
    for period in periods:
        for model in models:
            classification_report = pickle.load(open('results/' + TICKER + '/' + period + '/' + model + '/' + horizon + '/classification_report_' + set_ + '.pkl', 'rb'))
            dataframe.loc[period, model] = classification_report.loc[avg_type, 'f1-score']
    return dataframe


def accuracy_dataframe(TICKER, horizon, set_='test'):
    models = ['benchmark', 'deepLOB_L1', 'deepOF_L1', 'deepLOB_L2', 'deepOF_L2', 'deepVOL_L2', 'deepVOL_L3']
    periods = ['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10']
    dataframe = pd.DataFrame(np.zeros((len(periods), len(models))), columns = models, index = periods)
    for period in periods:
        for model in models:
            confusion_matrix = pickle.load(open('results/' + TICKER + '/' + period + '/' + model + '/' + horizon + '/confusion_matrix_' + set_ + '.pkl', 'rb'))
            dataframe.loc[period, model] = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return dataframe


def cce_dataframe(TICKER, horizon, set_='test'):
    models = ['benchmark', 'deepLOB_L1', 'deepOF_L1', 'deepLOB_L2', 'deepOF_L2', 'deepVOL_L2', 'deepVOL_L3']
    periods = ['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10']
    dataframe = pd.DataFrame(np.zeros((len(periods), len(models))), columns = models, index = periods)
    for period in periods:
        for model in models:
            cce = pickle.load(open('results/' + TICKER + '/' + period + '/' + model + '/' + horizon + '/categorical_crossentropy_' + set_ + '.pkl', 'rb'))
            dataframe.loc[period, model] = float(cce)
    return dataframe


def cost_dataframe(TICKER, horizon, cost, set_='test',):
    models = ['benchmark', 'deepLOB_L1', 'deepOF_L1', 'deepLOB_L2', 'deepOF_L2', 'deepVOL_L2', 'deepVOL_L3']
    periods = ['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10']
    dataframe = pd.DataFrame(np.zeros((len(periods), len(models))), columns = models, index = periods)
    for period in periods:
        for model in models:
            confusion_matrix = pickle.load(open('results/' + TICKER + '/' + period + '/' + model + '/' + horizon + '/confusion_matrix_' + set_ + '.pkl', 'rb'))
            dataframe.loc[period, model] = np.sum(confusion_matrix * cost) / np.sum(confusion_matrix)
    return dataframe


def class_cost_dataframe(TICKER, horizon, set_='test'):
    models = ['benchmark', 'deepLOB_L1', 'deepOF_L1', 'deepLOB_L2', 'deepOF_L2', 'deepVOL_L2', 'deepVOL_L3']
    periods = ['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10']
    dataframe = pd.DataFrame(np.zeros((len(periods), len(models))), columns = models, index = periods)
    for period in periods:
        for model in models:
            confusion_matrix = pickle.load(open('results/' + TICKER + '/' + period + '/' + model + '/' + horizon + '/confusion_matrix_' + set_ + '.pkl', 'rb'))
            class_distributions = pickle.load(open('results/' + TICKER + '/' + period + '/' + set_ + '_distributions.pkl', 'rb'))[int(horizon[1:])]
            class_cost = np.array([1/class_distributions.values]*3).T
            np.fill_diagonal(class_cost, 0)
            dataframe.loc[period, model] = np.sum(confusion_matrix * class_cost) / np.sum(confusion_matrix)
    return dataframe


def MCS(dataframe:pd.DataFrame, l=5, B=100):
    # use t-test rejection with bootstrap
    W = len(dataframe)
    full_model_set = dataframe.columns

    # bootstrap indices, could vectorize
    # these are computed once so that common random numbers for the bootstrap resamples in each iteration of the MCS sequential testing
    full_bootstrap_indices = np.zeros((B, W))
    for b in range(B):
        boot_indices = np.zeros((l*(W//l+1),))
        for j in range(W//l+1):
            init = random.sample(range(W), 1)[0]
            boot_indices[j*l:(j+1)*l] = range(init, init + l)
        full_bootstrap_indices[b, :] = np.mod(boot_indices[:W], W)
    
    # compute sample and bootstrap statistics
    bar_L = dataframe.mean(axis = 0).to_frame().T
    bar_L_bootstrap = pd.DataFrame([], columns = full_model_set)
    for boot_indices in full_bootstrap_indices:
        bootstrap_dataframe = dataframe.iloc[boot_indices, :]
        bar_L_bootstrap = pd.concat([bar_L_bootstrap, bootstrap_dataframe.mean(axis = 0).to_frame().T], ignore_index=True)
    bootstrap_indexer = bar_L_bootstrap.index
    zeta_bootstrap = pd.DataFrame([], index=bootstrap_indexer, columns = full_model_set)
    zeta_bootstrap.loc[:, :] = bar_L_bootstrap.values - bar_L.values

    # carry out MCS procedure
    model_set = list(full_model_set)
    MCS_results = pd.DataFrame([], columns = ['avg loss', 'p-value equiv. test', 'MCS p-value'])
    p_value_MCS = 0.0
    for _ in range(len(full_model_set)):
        # reduce sample and bootstrap statistics to remaining models
        bar_L = bar_L[model_set]
        zeta_bootstrap = zeta_bootstrap[model_set]

        if np.isnan(bar_L.values).any():
            eliminate_model = model_set[np.argwhere(np.isnan(bar_L.values))[0][-1]]
            MCS_results.loc[eliminate_model, :] = bar_L[eliminate_model].values[0], 0.0, p_value_MCS
            model_set.remove(eliminate_model)
            continue

        if len(model_set) > 1:
            # compute average loss and estimate var(\bar{d}_{i.}) using bootstrap samples
            bar_L_dot = bar_L.values.mean(axis = 1)
            zeta_bootstrap_dot = zeta_bootstrap.mean(axis = 1)
            var_bar_d_dot = pd.DataFrame([], index=bootstrap_indexer, columns = model_set)
            var_bar_d_dot.loc[:, :] = (zeta_bootstrap.values - zeta_bootstrap_dot.values[..., np.newaxis])**2
            var_bar_d_dot = var_bar_d_dot.mean(axis = 0)

            # compute sample T_max statistic
            bar_d_dot = pd.DataFrame([], index = [0], columns = model_set)
            bar_d_dot.loc[:, :] = bar_L.values - bar_L_dot
            t_dot = bar_d_dot / np.sqrt(var_bar_d_dot)
            T_max = np.max(t_dot.values)

            # estimate distribution of T_max statistic using bootstrap samples
            t_dot_bootstrap = pd.DataFrame([], index=bootstrap_indexer, columns = model_set)
            t_dot_bootstrap.loc[:, :] = (zeta_bootstrap.values - zeta_bootstrap_dot.values[..., np.newaxis]) / np.sqrt(var_bar_d_dot.values)
            T_max_bootstrap = t_dot_bootstrap.max(axis = 1)

            # compute p-value of equivalence test and MCS p-value
            p_value_test = np.mean(T_max < T_max_bootstrap.values)
            p_value_MCS = np.max(np.array([p_value_MCS, p_value_test]))

            # drop model based on elimination rule and uppdate MCS results
            eliminate_model = model_set[np.argmax(t_dot.values)]
            MCS_results.loc[eliminate_model, :] = bar_L[eliminate_model].values[0], p_value_test, p_value_MCS
            model_set.remove(eliminate_model)
        else:
            MCS_results.loc[model_set[0], :] = bar_L[model_set[0]].values[0], 1, 1
    return MCS_results


def summarize_MCS_results(tickers, horizons, metric):
    if metric == "accuracy":
        metric_dataframe = accuracy_dataframe
    elif metric == "weighted_f1":
        metric_dataframe = partial(f1_dataframe, avg_type="weighted avg")
    elif metric == "macro_f1":
        metric_dataframe = partial(f1_dataframe, avg_type="macro avg")
    elif metric == "cce":
        metric_dataframe = cce_dataframe
    else:
        raise ValueError("Only accuracy, weighted_f1, macro_f1 and cce metrics supported.")
    col_names = [" "]*3*len(horizons)
    col_names[::3] = horizons
    row_names = [" "]*(7*len(tickers)+1)
    row_names[1::7] = tickers
    full_df = pd.DataFrame(np.zeros((len(row_names), len(col_names))), index = row_names, columns = col_names)
    full_df.iloc[0, :] = [" ", "avg loss", "MCS p-value"] * len(horizons)
    for i, TICKER in enumerate(tickers):
        for j, horizon in enumerate(horizons):
            df = metric_dataframe(TICKER, horizon)
            if metric in ["accuracy", "weighted_f1", "macro_f1"]:
                MCS_results = MCS(1-df, l=3, B=100)[['avg loss', 'MCS p-value']]
            elif metric in ["cce"]:
                MCS_results = MCS(df, l=3, B=100)[['avg loss', 'MCS p-value']]
            MCS_results = MCS_results.reset_index()
            full_df.iloc[(1 + 7*i):(1 + 7*(i+1)), 3*j:3*(j+1)] = MCS_results.values
    full_df.to_csv("MCS_results/" + metric + "_MCS_results.csv")
    return full_df

            

if __name__ == '__main__':
    # all_classification_reports_to_df(from_type='dict')
    # make_train_val_distributions()
    # make_benchmark()
    tickers = ['LILAK', 'QRTEA', 'XRAY', 'CHTR', 'PCAR', 'EXC', 'AAL', 'WBA', 'ATVI', 'AAPL']
    horizons = ['h10', 'h20', 'h30', 'h50', 'h100', 'h200', 'h300', 'h500', 'h1000']
    # summarize_MCS_results(tickers, horizons, metric = "accuracy")
    # summarize_MCS_results(tickers, horizons, metric = "weighted_f1")
    # summarize_MCS_results(tickers, horizons, metric = "macro_f1")
    # summarize_MCS_results(tickers, horizons, metric = "cce")
    # for horizon in horizons:
    #     print('_________________________________________________________________________')
    #     print(TICKER, horizon)
    #     f1_df = f1_dataframe(TICKER, horizon, avg_type = 'weighted avg')
    #     print(f1_df)

    #     random.seed(0)
    #     MCS_results = MCS(1-f1_df, l=3, B=100)
    #     print(MCS_results[['avg loss', 'MCS p-value']])
    
    # for horizon in horizons:
    #     print('_________________________________________________________________________')
    #     print(TICKER, horizon)
    #     acc_df = accuracy_dataframe(TICKER, horizon)
    #     print(acc_df)

    #     random.seed(0)
    #     MCS_results = MCS(1-acc_df, l=3, B=100)
    #     print(MCS_results[['avg loss', 'MCS p-value']])

    # for horizon in horizons:
    #     print('_________________________________________________________________________')
    #     print(TICKER, horizon)
    #     cce_df = cce_dataframe(TICKER, horizon)
    #     print(cce_df)

    #     random.seed(0)
    #     MCS_results = MCS(cce_df, l=3, B=100)
    #     print(MCS_results[['avg loss', 'MCS p-value']])

    #TODO: these predictions are not ok, for both benchmark and models when we change cost matrix
    # cost = np.array([[0, 3, 3], [1, 0, 1], [3, 3, 0]])
    # for horizon in horizons:
    #     print('_________________________________________________________________________')
    #     print(TICKER, horizon)
    #     cost_df = cost_dataframe(TICKER, horizon, cost)
    #     print(cost_df)

    #     random.seed(0)
    #     MCS_results = MCS(cost_df, l=3, B=100)
    #     print(MCS_results[['avg loss', 'MCS p-value']])

    #TODO: these predictions are not ok, for both benchmark and models when we change cost matrix
    # for horizon in horizons:
    #     print('_________________________________________________________________________')
    #     print(TICKER, horizon)
    #     cost_df = class_cost_dataframe(TICKER, horizon)
    #     print(cost_df)

    #     random.seed(0)
    #     MCS_results = MCS(cost_df, l=3, B=100)
    #     print(MCS_results[['avg loss', 'MCS p-value']])

    ### Determining how far ahead there is predictability ###
    # alpha = 0.05
    # for horizon in horizons:
    #     print(horizon)
    #     count_benchmark = 0
    #     for TICKER in tickers:
    #         acc_df = accuracy_dataframe(TICKER, horizon)

    #         random.seed(0)
    #         MCS_results = MCS(1-acc_df, l=3, B=100)
    #         MCS_ = MCS_results.index[MCS_results['MCS p-value'] >= alpha]
    #         if 'benchmark' in MCS_:
    #             count_benchmark +=1
    #     print(count_benchmark/len(tickers))
    
    # tickers = ['AAL', 'ATVI', 'EXC', 'LILAK', 'PCAR', 'QRTEA', 'WBA', 'XRAY']
    # for horizon in horizons:
    #     print(horizon)
    #     count_benchmark = 0
    #     for TICKER in tickers:
    #         cce_df = cce_dataframe(TICKER, horizon)

    #         random.seed(0)
    #         MCS_results = MCS(cce_df, l=3, B=100)
    #         MCS_ = MCS_results.index[MCS_results['MCS p-value'] >= alpha]
    #         if 'benchmark' in MCS_:
    #             count_benchmark +=1
    #     print(count_benchmark/len(tickers))


