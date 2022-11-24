import pickle
import os
import numpy as np
import pandas as pd
import random
from functools import *

def classification_report_to_df(report):
    """
    Convert a classification report from string/dict format to dataframe.
    :param report: classification report with classes 0, 1 and 2 and metrics precision, recall, accuracy and f1-score as returned by sklearn.classification_report, string or dict
    :return: dataframe: classification report with classes 0, 1 and 2 and metrics precision, recall, accuracy and f1-score, pd.DataFrame
    """
    if type(report) is pd.DataFrame: 
        return report
    elif type(report) is str:
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
    elif type(report) is dict:
        report.pop('accuracy')
        dataframe = pd.DataFrame.from_dict(report).T
        dataframe.index = [0, 1, 2, 'macro avg', 'weighted avg'] 
    else:
        raise ValueError('type(report) must be pd.DataFrame, str or dict.')
    return dataframe

def all_classification_reports_to_df(tickers, periods, models, horizons):
    """
    Convert all .pkl classification reports from string/dict format to dataframe for each ticker-period-model-horizon combination.
    :param tickers: tickers under consideration, list of str
    :param periods: periods under consideration, list of str
    :param models: models under consideration, list of str
    :param horizons: horizons under consideration, list of str
    """
    for ticker in tickers:
        for period in periods: 
            for model in models:
                for horizon in horizons:
                    for set_ in ['val', 'train', 'test']:
                        classification_report = pickle.load(open('results/' + ticker + '/' + period + '/' + model + '/' + horizon + '/' + 'classification_report_' + set_ + '.pkl', 'rb'))
                        classification_report = classification_report_to_df(classification_report)
                        pickle.dump(classification_report, open('results/' + ticker + '/' + period + '/' + model + '/' + horizon + '/' + 'classification_report_' + set_ + '.pkl', 'wb'))


def make_train_val_distributions(tickers, periods, orderbook_updates):
    """
    Compute pd.DataFrame train-val distribution of returns from train and val distributions.
    Save as .pkl for each ticker-period-horizon combination.
    :param tickers: tickers under consideration, list of str
    :param periods: periods under consideration, list of str
    :param orderbook_updates: orderbook update horizons under consideration, list of int
    """
    for ticker in tickers:
        for period in periods:
            val_distributions = pickle.load(open('results/' + ticker + '/' + period + '/val_distributions.pkl', 'rb'))
            train_distributions = pickle.load(open('results/' + ticker + '/' + period + '/distributions.pkl', 'rb'))
            train_val_distributions = pd.DataFrame(np.zeros((3, len(orderbook_updates))),
                                     index = ["down", "stationary", "up"], 
                                     columns = orderbook_updates)
            for j, horizon in enumerate(horizons):
                support_train = pickle.load(open('results/' + ticker + '/' + period + '/deepLOB_L1/' + horizon + '/classification_report_train.pkl', 'rb')).loc['macro avg', 'support']
                support_val = pickle.load(open('results/' + ticker + '/' + period + '/deepLOB_L1/' + horizon + '/classification_report_val.pkl', 'rb')).loc['macro avg', 'support']
                train_val_distributions.iloc[:, j] = (train_distributions.iloc[:, j].values * support_train + val_distributions.iloc[:, j].values * support_val)/(support_train + support_val)
            pickle.dump(train_val_distributions, open('results/' + ticker + '/' + period + '/train_val_distributions.pkl', 'wb'))


def make_benchmark(tickers, periods, horizons):
    """
    Compute confusion matrix, classification report and categorical cross entropy loss of benchmark model on train, test and val sets,
    i.e. for each ticker-period-horizon combination, the empirical train-val distributions. 
    Assume that prediction is the argmax of probabilities.
    Save as .pkl's for each ticker-period-horizon combination.
    :param tickers: tickers under consideration, list of str
    :param periods: periods under consideration, list of str
    :param horizons: horizons under consideration, list of str
    """
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
                    # benchmark categorical cross entropy loss
                    cce = - np.sum(support * np.log(train_val_distributions.iloc[:, j].values)) / support.sum()
                    pickle.dump(confusion_matrix, open('results/' + ticker + '/' + period + '/benchmark/' + horizon + '/confusion_matrix_' + set_ + '.pkl', 'wb'))
                    pickle.dump(classification_report, open('results/' + ticker + '/' + period + '/benchmark/' + horizon + '/classification_report_' + set_ + '.pkl', 'wb'))
                    pickle.dump(cce, open('results/' + ticker + '/' + period + '/benchmark/' + horizon + '/categorical_crossentropy_' + set_ + '.pkl', 'wb'))


def metric_dataframe(TICKER, horizon, models, periods, metric='cce', set_='test', **kwargs):
    """
    For a given TICKER and horizon load the time series of evaluation metrics on set_ for all models under consideration.
    The are M models under consideration and 11 time periods.
    :param TICKER: the TICKER under consideration, str
    :param horizon: the horizon under consideration, str
    :param models: models under consideration, list of str
    :param periods: periods under consideration, list of str
    :param metric: the metric for evaluation, str
    :param set_: the metric evaluation set, 'train', 'val' or 'test'
    :return: dataframe: time series of model evaluation metrics, (11, M) pd.DataFrame
    """
    dataframe = pd.DataFrame(np.zeros((len(periods), len(models))), columns = models, index = periods)
    for period in periods:
        for model in models:
            if model[-7:] == 'seq2seq':
                classification_report_path = 'results/' + TICKER + '/' + period + '/' + model[:-8] + '/seq2seq/classification_report_' + set_ + '_' + horizon + '.pkl'
                confusion_matrix_path =  'results/' + TICKER + '/' + period + '/' + model[:-8] + '/seq2seq/confusion_matrix_' + set_ + '_' + horizon + '.pkl'
                cce_path = 'results/' + TICKER + '/' + period + '/' + model[:-8] + '/seq2seq/categorical_crossentropy_' + set_ + '_' + horizon + '.pkl'
            elif model[-9:] == 'universal':
                classification_report_path = 'results/universal/' + period + '/' + model[:-10] + '/' + horizon + '/' + TICKER + '/classification_report_' + set_ + '.pkl'
                confusion_matrix_path =  'results/universal/' + period + '/' + model[:-10] + '/' + horizon + '/' + TICKER + '/confusion_matrix_' + set_ + '.pkl'
                cce_path = 'results/universal/' + period + '/' + model[:-10] + '/' + horizon + '/' + TICKER + '/categorical_crossentropy_' + set_ + '.pkl'
            else:
                classification_report_path = 'results/' + TICKER + '/' + period + '/' + model + '/' + horizon + '/classification_report_' + set_ + '.pkl'
                confusion_matrix_path = 'results/' + TICKER + '/' + period + '/' + model + '/' + horizon + '/confusion_matrix_' + set_ + '.pkl'
                cce_path = 'results/' + TICKER + '/' + period + '/' + model + '/' + horizon + '/categorical_crossentropy_' + set_ + '.pkl'
            
            distributions_path = 'results/' + TICKER + '/' + period + '/' + set_ + '_distributions.pkl'
            
            if metric == 'weighted_f1':
                classification_report = pickle.load(open(classification_report_path, 'rb'))
                dataframe.loc[period, model] = classification_report.loc['weighted svg', 'f1-score']
            elif metric == 'macro_f1':
                classification_report = pickle.load(open(classification_report_path, 'rb'))
                dataframe.loc[period, model] = classification_report.loc['macro avg', 'f1-score']
            elif metric == 'cce':
                cce = pickle.load(open(cce_path, 'rb'))
                dataframe.loc[period, model] = float(cce)
            elif metric == 'accuracy':
                confusion_matrix = pickle.load(open(confusion_matrix_path, 'rb'))
                dataframe.loc[period, model] = np.trace(confusion_matrix) / np.sum(confusion_matrix)
            elif metric == 'cost':
                confusion_matrix = pickle.load(open(confusion_matrix_path, 'rb'))
                dataframe.loc[period, model] = np.sum(confusion_matrix * kwargs['cost']) / np.sum(confusion_matrix)
            elif metric == 'class_cost':
                confusion_matrix = pickle.load(open(confusion_matrix_path, 'rb'))
                class_distributions = pickle.load(open(distributions_path, 'rb'))[int(horizon[1:])]
                class_cost = np.array([1/class_distributions.values]*3).T
                np.fill_diagonal(class_cost, 0)
                dataframe.loc[period, model] = np.sum(confusion_matrix * class_cost) / np.sum(confusion_matrix)
            else:
                raise ValueError('metric must be one of weighted_f1, macro_f1, cce, accuracy, cost, class_cost.')
    return dataframe


def MCS(dataframe:pd.DataFrame, l=3, B=500):
    """
    Carry out the model confidence set procedure on dataframe, a time series of model losses.
    The M columns of the dataframe represent different models, and the W rows of the dataframe index the time period.
    :param dataframe: the time series of model losses, (W, M) pd.DataFrame
    :param l: size of block for block bootstrap, int
    :param B: number of bootstrap samples to use, int
    :return: MCS_results: dataframe with MCS results, i.e. 'avg loss', 'p-value equiv. test' and 'MCS p-value' for each model, pd.DataFrame
    """
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


def summarize_MCS_results(tickers, horizons, metric, models, periods, set_='test', save_path='MCS_results/results.xlsx', l=3, B=500, p_values = np.array([0.1, 0.05, 0.01])):
    """
    Carry out the model confidence set procedure on for each ticker in tickers and horizon in horizons according to a specified evaluation metric on set_.
    Save result df 
    :param tickers: tickers under consideration, list of str
    :param horizons: horizons under consideration, list of str
    :param metric: the metric for evaluation, str
    :param models: models under consideration, list of str
    :param periods: periods under consideration, list of str
    :param set_: the metric evaluation set, 'train', 'val' or 'test'
    :param save_path: target directory in which to save all MCS results, str
    :param l: size of block for block bootstrap in MCS procedure, int
    :param B: number of bootstrap samples to use in MCS procedure, int
    :param p_values: p-values at which to identify predictability, np.array
    :return: full_df: dataframe with all MCS results, i.e. for each ticker-horizon combination the 'avg loss' and 'MCS p-value' of the models, pd.DataFrame
    """
    superior_model_df = pd.DataFrame(np.zeros((len(models), len(p_values))), index = models, columns = p_values)
    total_predictable_horizons = pd.DataFrame(np.zeros((1, len(p_values))), columns = p_values)
    col_names = [" "]*3*len(horizons)
    col_names[::3] = horizons
    row_names = [" "]*(len(models)*len(tickers)+1)
    row_names[1::len(models)] = tickers
    full_df = pd.DataFrame(np.zeros((len(row_names), len(col_names))), index = row_names, columns = col_names)
    full_df.iloc[0, :] = [" ", "avg loss", "MCS p-value"] * len(horizons)
    for i, TICKER in enumerate(tickers):
        for j, horizon in enumerate(horizons):
            df = metric_dataframe(TICKER, horizon, models, periods, metric=metric, set_=set_)
            if metric in ["accuracy", "weighted_f1", "macro_f1"]:
                MCS_results = MCS(1-df, l=l, B=B)[['avg loss', 'MCS p-value']]
            elif metric in ["cce"]:
                MCS_results = MCS(df, l=l, B=B)[['avg loss', 'MCS p-value']]
            # keep track of which models are considered superior models
            p_value_benchmark = MCS_results.loc['benchmark', 'MCS p-value']
            for p_value in p_values:
                if p_value_benchmark < p_value:
                    total_predictable_horizons.loc[:, p_value] += 1
                    superior_models = MCS_results.index[MCS_results.loc[:, 'MCS p-value'] >= p_value]
                    superior_model_df.loc[superior_models, p_value] += 1
            # add results to full_df
            MCS_results = MCS_results.reset_index()
            full_df.iloc[(1 + len(models)*i):(1 + len(models)*(i+1)), 3*j:3*(j+1)] = MCS_results.values
    # save full_df and superior_model_df
    superior_model_df /= total_predictable_horizons.iloc[0, :]
    with pd.ExcelWriter(save_path) as writer:
        full_df.to_excel(writer, sheet_name=metric)
        superior_model_df.to_excel(writer, sheet_name='superior models')
    return full_df

if __name__ == '__main__':
    random.seed(123)

    # tickers sorted by liquidity score
    tickers = ['LILAK', 'QRTEA', 'XRAY', 'CHTR', 'PCAR', 'EXC', 'AAL', 'WBA', 'ATVI', 'AAPL']
    periods = ['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10']
    models = ['deepLOB_L1', 'deepOF_L1', 'deepLOB_L2', 'deepOF_L2', 'deepVOL_L2', 'deepVOL_L3']
    horizons = ['h10', 'h20', 'h30', 'h50', 'h100', 'h200', 'h300', 'h500', 'h1000']
    orderbook_updates = [10, 20, 30, 50, 100, 200, 300, 500, 1000]
    
    # all_classification_reports_to_df(tickers, periods, models, horizons)
    # make_train_val_distributions(tickers, periods, orderbook_updates)
    # make_benchmark(tickers, periods, horizons)

    # general results
    summarize_MCS_results(tickers, horizons, 'cce', ['benchmark'] + models, periods, 'test', 
                          save_path='MCS_results/MCS_results_general_experiment_cce_test.xlsx')

    # multihorizon results
    horizons = ['h10', 'h20', 'h30', 'h50']
    multihorizon_models = ['benchmark'] + models + [model + '_seq2seq' for model in models]
    summarize_MCS_results(tickers, horizons, 'cce', multihorizon_models, periods, 'test', 
                          save_path='MCS_results/MCS_results_multihorizon_experiment_cce_test.xlsx')

    # universal results
    horizons = ['h10', 'h20', 'h30', 'h50', 'h100', 'h200', 'h300', 'h500', 'h1000']
    universal_models = ['benchmark'] + [model + '_universal' for model in models]
    summarize_MCS_results(tickers, horizons, 'cce', universal_models, periods, 'test', 
                          save_path='MCS_results/MCS_results_universal_experiment_cce_test.xlsx')