# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:23:30 2020

@author: Lorenzo
"""

import os
import glob
import pandas as pd
import re
import datetime
import numpy as np
import pickle
import time
import h5py

def process_data(TICKER, input_path, output_path, time_index="seconds", output_extension="csv",
                  horizons=np.array([10, 20, 30, 50, 100]), orderflow=False):
    """
    As in DeepLOB by Zohren and Zhang.
    Function for pre-processing LOBSTER data. The data must be stored in the input_path
    directory as daily message book and order book files. The data is treated in the following way:
    - order book states with crossed quotes are removed.
    - each state in the orderbook is time-stamped, with states occurring at the same time collapsed
      onto the last state.
    - the first and last 10 minutes of market activity are dropped.
    - standardisation (z-score) to normalise our data, but use the mean and standard deviation of the previous 5 days’
      data to normalise the current day's data (Dynamic normalisation). Hence drop first 5 days.
    - the smoothed returns at the requested horizons (in order book changes) are returned
      l = (m+ - m-)/m-, where m+ and m- denote respectively the mean of the next and the previous k mid-prices.
    Moreover supplementary files are produced for:
    - order book files with problems
    - message book files with problems
    - trading days with unusual open - close times
    :param TICKER: the TICKER to be considered
    :param input_path: the path where the order book and message book files are stores in monthly
                       directories
    :param output_path: the path where we wish to save the processed datasets
    :param time_index: the time-index to use ("seconds" or "datetime")
    :param output_extension: the extension of the saved files ("hdf5" or "csv")
    :param horizons: forecasting horizons for labels
    :return: saves the processed order books in output_path, each orderbook contains the following columns:
             "ASKp1", "ASKs1", "BIDp1",  "BIDs1", ..., "ASKpN", "ASKsN", "BIDpN",  "BIDsN", "horizons[0]", ..., "horizons[-1]"
             where N is the number of levels.
    """
    os.chdir(input_path)
    extension = "csv"
    csv_file_list = glob.glob("*.{}".format(extension))

    # # get a list of all subdirectories and search for csv files in these subdirectories
    # for dir_path in dir_paths:
    #     file_path = os.path.join(dir_path,"*.{}".format(extension))
    #     csv_file_list_to_add = [i for i in glob.glob(file_path)]
    #     csv_file_list += csv_file_list_to_add

    csv_orderbook = [name for name in csv_file_list if "orderbook" in name]
    csv_message = [name for name in csv_file_list if "message" in name]

    csv_orderbook.sort()
    csv_message.sort()

    # check if exactly half of the files are order book and exactly half are messages
    assert (len(csv_message) == len(csv_orderbook))
    assert (len(csv_file_list) == len(csv_message) + len(csv_orderbook))

    print("started loop")

    orderbook_with_problems = []
    messages_with_problems = []
    opening_closing_times = []
    df_statistics = pd.DataFrame(columns=["Updates (000)", "Trades (000)", "Price Changes (000)",
                                          "Price (USD)", "Spread (bps)", "Volume (USD MM)", "Tick Size"])

    # dataframes for dynamic z-score normalisation
    mean_df = pd.DataFrame()
    mean2_df = pd.DataFrame()
    nsamples_df = pd.DataFrame()

    for orderbook_name in csv_orderbook:

        print(orderbook_name)

        # read the orderbook. keep a record of problematic files
        try:
            df_orderbook = pd.read_csv(orderbook_name, header=None)
        except:
            orderbook_with_problems.append(orderbook_name)
            print("the following file has been skipped:  " + orderbook_name)
            continue

        levels = int(df_orderbook.shape[1] / 4)
        feature_names_raw = ["ASKp", "ASKs", "BIDp", "BIDs"]
        feature_names = []
        for i in range(1, levels + 1):
            for j in range(4):
                feature_names += [feature_names_raw[j] + str(i)]
        df_orderbook.columns = feature_names

        # compute the mid prices
        df_orderbook.insert(0, "mid price", (df_orderbook["ASKp1"] + df_orderbook["BIDp1"]) / 2)

        # get date
        match = re.findall(r"\d{4}-\d{2}-\d{2}", orderbook_name)[-1]
        date = datetime.datetime.strptime(match, "%Y-%m-%d")

        # read times from message file. keep a record of problematic files
        message_name = orderbook_name.replace("orderbook", "message")
        try:
            df_message = pd.read_csv(message_name, usecols=[0, 1, 2, 3, 4, 5], header=None)
        except:
            messages_with_problems.append(message_name)
            print("the following file has been skipped:  " + message_name)
            continue

        # check the two df have the same length
        assert (len(df_message) == len(df_orderbook))

        # add column names to message book
        df_message.columns = ["seconds", "event type", "order ID", "size", "price", "direction"]

        # remove crossed quotes
        df_orderbook.drop(df_orderbook[(df_orderbook["BIDp1"] > df_orderbook["ASKp1"])].index)
        df_message.drop(df_message[(df_orderbook["BIDp1"] > df_orderbook["ASKp1"])].index)

        # add the seconds since midnight column to the order book from the message book
        df_orderbook.insert(0, "seconds", df_message["seconds"])

        # one conceptual event (e.g. limit order modification which is implemented as a cancellation followed
        # by an immediate new arrival, single market order executing against multiple resting limit orders) may
        # appear as multiple rows in the message file, all with the same timestamp.
        # We hence group the order book data by unique timestamps and take the last entry.
        df_orderbook = df_orderbook.groupby(["seconds"]).tail(1)
        df_message = df_message.groupby(["seconds"]).tail(1)

        # check market opening times for strange values
        market_open = int(df_orderbook["seconds"].iloc[0] / 60) / 60  # open at minute before first transaction
        market_close = (int(df_orderbook["seconds"].iloc[-1] / 60) + 1) / 60  # close at minute after last transaction

        if not (market_open == 9.5 and market_close == 16):
            opening_closing_times.append(str(market_open) + " - " + str(market_close))
            print("the following date has been skipped due to strange opening times: ", date)
            continue

        if time_index == "seconds":
            # drop values outside of market hours using seconds

            df_orderbook = df_orderbook.loc[(df_orderbook["seconds"] >= 34200) &
                                            (df_orderbook["seconds"] <= 57600)]
            df_message = df_message.loc[(df_message["seconds"] >= 34200) &
                                        (df_message["seconds"] <= 57600)]

            # drop first and last 10 minutes of trading using seconds
            market_open_seconds = market_open * 60 * 60 + 10 * 60
            market_close_seconds = market_close * 60 * 60 - 10 * 60
            df_orderbook = df_orderbook.loc[(df_orderbook["seconds"] >= market_open_seconds) &
                                            (df_orderbook["seconds"] <= market_close_seconds)]
            df_message = df_message.loc[(df_message["seconds"] >= market_open_seconds) &
                                        (df_message["seconds"] <= market_close_seconds)]

        elif time_index == "datetime":
            # index using datetime index
            seconds_since_midnight = pd.to_timedelta(df_orderbook["seconds"], unit="S", errors="coerce")
            timeindex_ = seconds_since_midnight.values + pd.Series(date).repeat(repeats=len(seconds_since_midnight))
            df_orderbook.index = timeindex_
            df_message.index = timeindex_

            # drop values outside of market hours using datetime index
            df_orderbook = df_orderbook.between_time(datetime.time(9, 30), datetime.time(16))
            df_message = df_message.between_time(datetime.time(9, 30), datetime.time(16))

            # drop first and last 10 minutes of trading using datetime index
            market_open = market_open + 1 / 6
            market_close = market_close - 1 / 6
            market_open = datetime.time(int(market_open), int(np.mod(60 * market_open, 60)))
            market_close = datetime.time(int(market_close), int(np.mod(60 * market_close, 60)) - 10)
            df_orderbook = df_orderbook.between_time(market_open, market_close)
            df_message = df_message.between_time(market_open, market_close)
        else:
            raise Exception("time_index must be seconds or datetime")

        # We keep track of the following (daily) statistics:
        # Updates (000): the total number of changes in the orderbook file
        # Trades (000): the total number of trades, computed by counting the number of message book events
        #               corresponding to the execution of (possibly hidden) limit orders (Event Type 4 or 5
        #               in LOBSTER)
        # Price Changes (000): the total number of price changes per day
        # Price (USD): average price on the day, weighted average by time
        # Spread (bps): average spread on the day, weighted average by time
        # Volume (USD MM): total volume traded on the day, computed as the sum of the volumes of
        #                  all the executed trades (Event Type 4 or 5). The volume of a single trade
        #                  is given by Size*Price
        # Tick size: the fraction of time that the bid-ask spread is equal to one tick for each stock
        #            (Curato et al., 2015).

        if time_index == "seconds":
            updates = df_orderbook.shape[0] / 1000
            trades = np.sum((df_message["event type"] == 4) | (df_message["event type"] == 5)) / 1000
            price_changes = np.sum(~(np.diff(df_orderbook["mid price"]) == 0.0)) / 1000
            time_deltas = np.append(np.diff(df_orderbook["seconds"]),
                                    market_close_seconds - df_orderbook["seconds"].iloc[-1])
            price = np.average(df_orderbook["mid price"] / 10 ** 4, weights=time_deltas)
            spread = np.average((df_orderbook["ASKp1"] - df_orderbook["BIDp1"]) / df_orderbook["mid price"] * 10000,
                                weights=time_deltas)
            volume = np.sum(
                df_message.loc[(df_message["event type"] == 4) | (df_message["event type"] == 5)]["size"] *
                df_message.loc[(df_message["event type"] == 4) | (df_message["event type"] == 5)][
                    "price"] / 10 ** 4) / 10 ** 6
            tick_size = np.average((df_orderbook["ASKp1"] - df_orderbook["BIDp1"]) == 100.0,
                                   weights=time_deltas)

            df_statistics.loc[date] = [updates, trades, price_changes, price, spread, volume, tick_size]

        if orderflow:
            # compute bid and ask multilevel orderflow
            ASK_prices = df_orderbook.loc[:, df_orderbook.columns.str.contains('ASKp')]
            BID_prices = df_orderbook.loc[:, df_orderbook.columns.str.contains('BIDp')]
            ASK_sizes = df_orderbook.loc[:, df_orderbook.columns.str.contains('ASKs')]
            BID_sizes = df_orderbook.loc[:, df_orderbook.columns.str.contains('BIDs')]

            ASK_price_changes = ASK_prices.diff().dropna().to_numpy()
            BID_price_changes = BID_prices.diff().dropna().to_numpy()
            ASK_size_changes = ASK_sizes.diff().dropna().to_numpy()
            BID_size_changes = BID_sizes.diff().dropna().to_numpy()

            ASK_sizes = ASK_sizes.to_numpy()
            BID_sizes = BID_sizes.to_numpy()

            ASK_OF = (ASK_price_changes > 0.0) * (-ASK_sizes[:-1, :]) + (ASK_price_changes == 0.0) * ASK_size_changes + (ASK_price_changes < 0) * ASK_sizes[1:, :]
            BID_OF = (BID_price_changes < 0.0) * (-BID_sizes[:-1, :]) + (BID_price_changes == 0.0) * BID_size_changes + (BID_price_changes > 0) * BID_sizes[1:, :]

            # remove all price-volume features and add in orderflow
            df_orderbook = df_orderbook.drop(feature_names, axis=1).iloc[1:, :]
            feature_names_raw = ["ASK_OF", "BID_OF"]
            feature_names = []
            for feature_name in feature_names_raw:
                for i in range(1, levels + 1):
                    feature_names += [feature_name + str(i)]
            df_orderbook[feature_names] = np.concatenate([ASK_OF, BID_OF], axis=1)

        # dynamic z-score normalisation
        orderbook_mean_df = df_orderbook[feature_names].mean()
        orderbook_mean2_df = (df_orderbook[feature_names] ** 2).mean()
        orderbook_nsamples_df = pd.DataFrame(np.array([[len(df_orderbook)]] * len(feature_names)).T,
                                             columns=feature_names)

        if len(mean_df) < 5:
            # don't save the first five days as we don't have enough days to normalise
            mean_df = mean_df.append(orderbook_mean_df, ignore_index=True)
            mean2_df = mean2_df.append(orderbook_mean2_df, ignore_index=True)
            nsamples_df = nsamples_df.append(orderbook_nsamples_df, ignore_index=True)
            continue
        else:
            # z-score normalisation
            z_mean_df = pd.DataFrame((nsamples_df * mean_df).sum(axis=0) / nsamples_df.sum(axis=0)).T
            z_stdev_df = pd.DataFrame(
                np.sqrt((nsamples_df * mean2_df).sum(axis=0) / nsamples_df.sum(axis=0) - z_mean_df ** 2))
            # broadcast to df_orderbook size
            z_mean_df = z_mean_df.loc[z_mean_df.index.repeat(len(df_orderbook))]
            z_stdev_df = z_stdev_df.loc[z_stdev_df.index.repeat(len(df_orderbook))]
            z_mean_df.index = df_orderbook.index
            z_stdev_df.index = df_orderbook.index
            df_orderbook[feature_names] = (df_orderbook[feature_names] - z_mean_df) / z_stdev_df

            # roll forward by dropping first rows and adding most recent mean and mean2
            mean_df = mean_df.iloc[1:, :]
            mean2_df = mean2_df.iloc[1:, :]
            nsamples_df = nsamples_df.iloc[1:, :]

            mean_df = mean_df.append(orderbook_mean_df, ignore_index=True)
            mean2_df = mean2_df.append(orderbook_mean2_df, ignore_index=True)
            nsamples_df = nsamples_df.append(orderbook_nsamples_df, ignore_index=True)

        # create labels for returns with smoothing labelling method
        for h in horizons:
            rolling_mid = df_orderbook["mid price"].rolling(h).mean().dropna()[1:]
            rolling_mid = rolling_mid.to_numpy().reshape(len(rolling_mid),)
            smooth_pct_change = rolling_mid/df_orderbook["mid price"][0:-h] - 1
            df_orderbook[str(h)] = np.concatenate((smooth_pct_change,
                                                   np.repeat(np.NaN, int(h))))

        # drop seconds and mid price columns
        # df_orderbook = df_orderbook.drop(["seconds", "mid price"], axis=1)

        # drop elements with na predictions at the end which cannot be used for training
        df_orderbook = df_orderbook.iloc[:-max(horizons), :]

        # save
        output_name = os.path.join(output_path, TICKER + "_orderbook_" + str(date.date()) + "." + output_extension)
        if orderflow:
            output_name = os.path.join(output_path, TICKER + "_orderflow_" + str(date.date()) + "." + output_extension)
        if output_extension == "hdf5":
            with h5py.File(output_name, "w") as f:
                f.create_dataset("default", data=df_orderbook)
        elif output_extension == "csv":
            df_orderbook.to_csv(output_name, header=True, index=False)
        else:
            raise Exception("output_extension must be hdf5 or csv")

    print("finished loop")

    supplementary_path = os.path.join(output_path, TICKER + "_supplementary_files")
    skipped_files_path = os.path.join(supplementary_path, "skipped")
    open_close_files_path = os.path.join(supplementary_path, "opening_closing_times")
    statistics_files_path = os.path.join(supplementary_path, TICKER + "statistics.csv")

    os.mkdir(supplementary_path)

    with open(skipped_files_path + "_orderbook.txt", "wb") as fp:
        pickle.dump(orderbook_with_problems, fp)

    with open(skipped_files_path + "_messages.txt", "wb") as fp:
        pickle.dump(messages_with_problems, fp)

    with open(open_close_files_path + ".txt", "wb") as fp:
        pickle.dump(opening_closing_times, fp)

    df_statistics.to_csv(statistics_files_path, header=True, index=True)

    print("please check supplementary files before performing analysis")

def reconstruct_orderbook_from_hdf5(file):
    """
    Reconstruct a pandas dataframe from saved hdf5 file.
    :param file: the hdf5 orderbook file to be converted to pandas dataframe
    :return: orderbook, a pandas dataframe corresponding to the orderbook
    """

    # read file as numpy array
    with h5py.File(file, "r") as f:
        orderbook = f["default"][()]

    # set orderbook column names
    col_names = ["seconds", "mid price"]
    levels = int((orderbook.shape[1] - 2) / 4)
    col_names_raw = ["ASKp", "ASKs", "BIDp", "BIDs"]
    for i in range(1, levels + 1):
        for j in range(4):
            col_names += [col_names_raw[j] + str(i)]

    # write orderbook to pandas dataframe
    orderbook = pd.DataFrame(data=orderbook, columns=col_names)
    return orderbook


if __name__ == "__main__":
    # set parameters
    TICKER = "AAL"
    input_path = r"E:\_data_dwn_16_85__AAL_2019-01-01_2020-01-31_10"
    output_path = r"E:\AAL_OB"
    index = "seconds"

    startTime = time.time()
    process_data(TICKER=TICKER, 
                 input_path=input_path, 
                 output_path=output_path, 
                 output_extension="csv", 
                 orderflow=False)
    executionTime = (time.time() - startTime)

    print("Execution time in seconds: " + str(executionTime))

