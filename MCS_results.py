import pickle
import os
import numpy as np
import pandas as pd

def classification_report_str_to_df(report):
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
        row['f1_score'] = float(row_data[2])
        row['support'] = int(row_data[3])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe

def all_classification_reports_str_to_df():
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
                        classification_report = classification_report_str_to_df(classification_report)
                        classification_report.set_index('class', inplace=True)
                        pickle.dump(classification_report, open('results/' + ticker + '/' + period + '/' + model + '/' + horizon + '/' + 'classification_report_' + set_ + '.pkl', 'wb'))

def make_test_val_distributions():
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

#TODO: make train_val_distributions from submit_experiment output.
#TODO: make classification_report_dict_to_df

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
                    classification_report[['precision', 'recall', 'f1_score']] = 0
                    classification_report.loc[pred_benchmark, 'precision'] = support[pred_benchmark] / support.sum()
                    classification_report.loc[pred_benchmark, 'recall'] = 1.0
                    classification_report.loc[pred_benchmark, 'f1_score'] = 2 * support[pred_benchmark] / (support[pred_benchmark] + support.sum())
                    classification_report.loc['macro avg', ['precision', 'recall', 'f1_score']] = np.average(classification_report.loc[[0, 1, 2], ['precision', 'recall', 'f1_score']].values, axis=0)
                    classification_report.loc['weighted avg', ['precision', 'recall', 'f1_score']] = np.average(classification_report.loc[[0, 1, 2], ['precision', 'recall', 'f1_score']].values, axis=0, weights=support)
                    classification_report = classification_report.round(4)
                    pickle.dump(confusion_matrix, open('results/' + ticker + '/' + period + '/benchmark/' + horizon + '/confusion_matrix_' + set_ + '.pkl', 'wb'))
                    pickle.dump(classification_report, open('results/' + ticker + '/' + period + '/benchmark/' + horizon + '/classification_report_' + set_ + '.pkl', 'wb'))

def f1_dataframe(TICKER, horizon, set_='test', avg_type = 'macro avg'):
    models = ['benchmark', 'deepLOB_L1', 'deepOF_L1', 'deepLOB_L2', 'deepOF_L2', 'deepVOL_L2', 'deepVOL_L3']
    periods = ['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10']
    dataframe = pd.DataFrame(np.zeros((len(periods), len(models))), columns = models, index = periods)
    for period in periods:
        for model in models:
            classification_report = pickle.load(open('results/' + TICKER + '/' + period + '/' + model + '/' + horizon + '/classification_report_' + set_ + '.pkl', 'rb'))
            dataframe.loc[period, model] = classification_report.loc[avg_type, 'f1_score']
    return dataframe

if __name__ == '__main__':
    # all_classification_reports_str_to_df()
    # make_test_val_distributions()
    # make_benchmark()
    f1_df = f1_dataframe('AAL', 'h10')
    print(f1_df)