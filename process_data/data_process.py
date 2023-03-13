import os
import glob
import pandas as pd
import re
import datetime
import numpy as np
from multiprocessing import Pool

def process_data(TICKER, input_path, output_path, log_path, time_index="seconds", horizons=np.array([10, 20, 30, 50, 100]), features = "orderbooks", smoothing="uniform", k=10):
    """
    Function for pre-processing LOBSTER data. The data must be stored in the input_path
    directory as daily message book and order book files. The data is treated in the following way:
    - order book states with crossed quotes are removed.
    - each state in the orderbook is time-stamped, with states occurring at the same time collapsed
      onto the last state.
    - the first and last 10 minutes of market activity (inside usual opening times) are dropped.
    - rolling z-score normalization is applied to the data, i.e. the mean and standard deviation of the previous 5 days
      is used to normalize current day's data. Hence drop first 5 days.
    - the smoothed returns at the requested horizons (in order book changes) are returned
      if smoothing = "horizon": l = (m+ - m)/m, where m+ denotes the mean of the next h mid-prices, m(.) is current mid price.
      if smoothing = "uniform": l = (m+ - m)/m, where m+ denotes the mean of the k+1 mid-prices centered at m(. + h), m(.) is current mid price.
    A log file is produced tracking:
    - order book files with problems
    - message book files with problems
    - trading days with unusual open - close times
    - trading days with crossed quotes
    A statistics.csv file summarizes the following (daily) statistics:
        # Updates (000): the total number of changes in the orderbook file
        # Trades (000): the total number of trades, computed by counting the number of message book events
                        corresponding to the execution of (possibly hidden) limit orders (Event Type 4 or 5 in LOBSTER)
        # Price Changes (000): the total number of price changes per day
        # Price (USD): average price on the day, weighted average by time
        # Spread (bps): average spread on the day, weighted average by time
        # Volume (USD MM): total volume traded on the day, computed as the sum of the volumes of
                           all the executed trades (Event Type 4 or 5). The volume of a single trade
                           is given by Size*Price
        # Tick size: the fraction of time that the bid-ask spread is equal to one tick for each stock
                     (Curato et al., 2015).
    :param TICKER: the TICKER to be considered, str
    :param input_path: the path where the order book and message book files are stored, order book files have shape (:, 4*levels):
                       "ASKp1", "ASKs1", "BIDp1",  "BIDs1", ..., "ASKplevels", "ASKslevels", "BIDplevels",  "BIDslevels"
                       str
    :param output_path: the path where we wish to save the processed datasets, str
    :param time_index: the time-index to use ("seconds" or "datetime"), str
    :param horizons: forecasting horizons for labels, (h,) array
    :param features: whether to return "orderbooks" or "orderflows", str
    :param smoothing: whether to use "uniform" or "horizon" smoothing, bool
    :param k: smoothing window for returns when smoothing = "uniform", int
    :return: saves the processed features as .csv files in output_path, each file consists of:
             if orderbook, shape (:, 4*levels + h)
             "ASKp1", "ASKs1", "BIDp1",  "BIDs1", ..., "ASKplevels", "ASKslevels", "BIDplevels",  "BIDslevels", "horizons[0]", ..., "horizons[-1]"
             if orderflow, shape (:, 2*levels + h)
             "aOF1", "bOF1", "aOF2",  "bOF2", ..., "aOFlevels", "bOFlevels", "horizons[0]", ..., "horizons[-1]"
             where N is the number of levels.
             if volumes, , shape (:, 4*levels + h)
             "BIDvol2*levels", ...,  "BIDvol1", "ASKvol1", ..., "ASKvol2*levels", "horizons[0]", ..., "horizons[-1]"
    """
    csv_file_list = glob.glob(os.path.join(input_path, "*.{}".format("csv")))

    csv_orderbook = [name for name in csv_file_list if "orderbook" in name]
    csv_message = [name for name in csv_file_list if "message" in name]

    csv_orderbook.sort()
    csv_message.sort()

    # check if exactly half of the files are order book and exactly half are messages
    assert (len(csv_message) == len(csv_orderbook))
    assert (len(csv_file_list) == len(csv_message) + len(csv_orderbook))

    print("started loop")

    logs = []
    df_statistics = pd.DataFrame([], columns=["Updates (000)", "Trades (000)", "Price Changes (000)",
                                              "Price (USD)", "Spread (bps)", "Volume (USD MM)", "Tick Size"], dtype=float)

    # dataframes for dynamic z-score normalization
    mean_df = pd.DataFrame()
    mean2_df = pd.DataFrame()
    nsamples_df = pd.DataFrame()

    for orderbook_name in csv_orderbook:
        print(orderbook_name)

        # read the orderbook. keep a record of problematic files
        try:
            df_orderbook = pd.read_csv(orderbook_name, header=None)
        except:
            logs.append(orderbook_name + ' skipped. Error: failed to read orderbook.')

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
            logs.append(orderbook_name + ' skipped. Error: failed to read messagebook.')

        # check the two df have the same length
        assert (len(df_message) == len(df_orderbook))

        # add column names to message book
        df_message.columns = ["seconds", "event type", "order ID", "volume", "price", "direction"]

        # remove trading halts
        trading_halts_start = df_message[(df_message["event type"] == 7)&(df_message["price"] == -1)].index
        trading_halts_end = df_message[(df_message["event type"] == 7)&(df_message["price"] == 1)].index
        trading_halts_index = np.array([])
        for halt_start, halt_end in zip(trading_halts_start, trading_halts_end):
            trading_halts_index = np.append(trading_halts_index, df_message.index[(df_message.index >= halt_start)&(df_message.index < halt_end)])
        if len(trading_halts_index) > 0:
            for halt_start, halt_end in zip(trading_halts_start, trading_halts_end):
                logs.append('Warning: trading halt between ' + str(df_message.loc[halt_start, "seconds"]) + ' and ' + str(df_message.loc[halt_end, "seconds"]) + ' in ' + orderbook_name + '.')
        df_orderbook = df_orderbook.drop(trading_halts_index)
        df_message = df_message.drop(trading_halts_index)


        # remove crossed quotes
        crossed_quotes_index = df_orderbook[(df_orderbook["BIDp1"] > df_orderbook["ASKp1"])].index
        if len(crossed_quotes_index) > 0:
            logs.append('Warning: ' + str(len(crossed_quotes_index)) + ' crossed quotes removed in ' + orderbook_name + '.')
        df_orderbook = df_orderbook.drop(crossed_quotes_index)
        df_message = df_message.drop(crossed_quotes_index)

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
            logs.append('Warning: unusual opening times in ' + orderbook_name + ': ' + str(market_open) + ' - ' + str(market_close) + '.')

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

        if time_index == "seconds":
            if len(df_orderbook) > 0:
                updates = df_orderbook.shape[0] / 1000
                trades = np.sum((df_message["event type"] == 4) | (df_message["event type"] == 5)) / 1000
                price_changes = np.sum(~(np.diff(df_orderbook["mid price"]) == 0.0)) / 1000
                time_deltas = np.append(np.diff(df_orderbook["seconds"]), market_close_seconds - df_orderbook["seconds"].iloc[-1])
                price = np.average(df_orderbook["mid price"] / 10 ** 4, weights=time_deltas)
                spread = np.average((df_orderbook["ASKp1"] - df_orderbook["BIDp1"]) / df_orderbook["mid price"] * 10000,
                                    weights=time_deltas)
                volume = np.sum(
                    df_message.loc[(df_message["event type"] == 4) | (df_message["event type"] == 5)]["volume"] *
                    df_message.loc[(df_message["event type"] == 4) | (df_message["event type"] == 5)]["price"] / 10 ** 4) / 10 ** 6
                tick_size = np.average((df_orderbook["ASKp1"] - df_orderbook["BIDp1"]) == 100.0,
                                    weights=time_deltas)

                df_statistics.loc[date] = [updates, trades, price_changes, price, spread, volume, tick_size]
            
            else:
                df_statistics.loc[date] = [None]*7
        
        if features == "orderbooks":
            pass
        
        elif features == "orderflows":
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
            mid_seconds_columns = list(df_orderbook.columns)
            feature_names_raw = ["ASK_OF", "BID_OF"]
            feature_names = []
            for feature_name in feature_names_raw:
                for i in range(1, levels + 1):
                    feature_names += [feature_name + str(i)]
            df_orderbook[feature_names] = np.concatenate([ASK_OF, BID_OF], axis=1)
            
            # re-order columns
            feature_names_reordered = [[]]*len(feature_names)
            feature_names_reordered[::2] = feature_names[:levels]
            feature_names_reordered[1::2] = feature_names[levels:]
            feature_names = feature_names_reordered

            df_orderbook = df_orderbook[mid_seconds_columns + feature_names]

        else:
            raise ValueError('features must be "orderbooks" or "orderflows".')
    
        # dynamic z-score normalization
        orderbook_mean_df = pd.DataFrame(df_orderbook[feature_names].mean().values.reshape(-1, len(feature_names)), columns=feature_names)
        orderbook_mean2_df = pd.DataFrame((df_orderbook[feature_names] ** 2).mean().values.reshape(-1, len(feature_names)), columns=feature_names)
        orderbook_nsamples_df = pd.DataFrame(np.array([[len(df_orderbook)]] * len(feature_names)).T, columns=feature_names)

        if len(mean_df) < 5:
            logs.append(orderbook_name + ' skipped. Initializing rolling z-score normalization.')
            # don't save the first five days as we don't have enough days to normalize
            mean_df = pd.concat([mean_df, orderbook_mean_df], ignore_index=True)
            mean2_df = pd.concat([mean2_df, orderbook_mean2_df], ignore_index=True)
            nsamples_df = pd.concat([nsamples_df, orderbook_nsamples_df], ignore_index=True)
            continue
        else:
            # z-score normalization
            z_mean_df = pd.DataFrame((nsamples_df * mean_df).sum(axis=0) / nsamples_df.sum(axis=0)).T
            z_stdev_df = pd.DataFrame(np.sqrt((nsamples_df * mean2_df).sum(axis=0) / nsamples_df.sum(axis=0) - z_mean_df ** 2))
            
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

            mean_df = pd.concat([mean_df, orderbook_mean_df], ignore_index=True)
            mean2_df = pd.concat([mean2_df, orderbook_mean2_df], ignore_index=True)
            nsamples_df = pd.concat([nsamples_df, orderbook_nsamples_df], ignore_index=True)

        if smoothing == "horizon":
            # create labels for returns with smoothing labelling method
            for h in horizons:
                rolling_mid = df_orderbook["mid price"].rolling(h).mean().dropna()[1:]
                rolling_mid = rolling_mid.to_numpy().flatten()
                smooth_pct_change = rolling_mid/df_orderbook["mid price"][:-h] - 1
                df_orderbook[str(h)] = np.concatenate((smooth_pct_change, np.repeat(np.NaN, int(h))))
        elif smoothing == "uniform":
            # create labels for returns with smoothing labelling method
            rolling_mid = df_orderbook["mid price"].rolling(k+1, center=True).mean()
            rolling_mid = rolling_mid.to_numpy().flatten()
            for h in horizons:
                smooth_pct_change = rolling_mid[h:]/df_orderbook["mid price"][:-h] - 1
                df_orderbook[str(h)] = smooth_pct_change

        # drop seconds and mid price columns
        df_orderbook = df_orderbook.drop(["seconds", "mid price"], axis=1)

        # drop elements with na predictions at the end which cannot be used for training
        df_orderbook = df_orderbook.dropna()

        # save
        output_name = os.path.join(output_path, TICKER + "_" + features + "_" + str(date.date()))
        df_orderbook.to_csv(output_name + ".csv", header=True, index=False)

        logs.append(orderbook_name + ' completed.')

    print("finished loop")

    with open(log_path + "/" + features + "_processing_logs.txt", "w") as f:
        for log in logs:
            f.write(log + "\n")

    print("please check processing logs.")

    df_statistics.to_csv(log_path + "/" + features + "_statistics.csv", header=True, index=True)


def multiprocess_L3(TICKER, input_path, output_path, log_path, horizons=np.array([10, 20, 30, 50, 100]), NF=40, queue_depth=10, smoothing="uniform", k=10):
    """
    External call for pre-processing LOBSTER data into L3 volumes parallely. The data must be stored in the input_path
    directory as daily message book and order book files. The data is treated in the following way:
    - order book states with crossed quotes are removed.
    - each state in the orderbook is time-stamped, with states occurring at the same time collapsed
      onto the last state.
    - the first and last 10 minutes of market activity (inside usual opening times) are dropped.
    - the smoothed returns at the requested horizons (in order book changes) are returned
      if smoothing = "horizon": l = (m+ - m)/m, where m+ denotes the mean of the next h mid-prices, m(.) is current mid price.
      if smoothing = "uniform": l = (m+ - m)/m, where m+ denotes the mean of the k+1 mid-prices centered at m(. + h), m(.) is current mid price.
    A log file is produced tracking:
    - order book files with problems
    - message book files with problems
    - trading days with unusual open - close times
    - trading days with crossed quotes
    :param TICKER: the TICKER to be considered, str
    :param input_path: the path where the order book and message book files are stored, order book files have shape (:, 4*levels):
                       "ASKp1", "ASKs1", "BIDp1",  "BIDs1", ..., "ASKplevels", "ASKslevels", "BIDplevels",  "BIDslevels",
                       str
    :param output_path: the path where we wish to save the processed datasets (as .npz files), str
    :param log_path: the path where processing logs are saved, str
    :param NF: number of volume features, NF = 2W where W is number of ticks on each side of mid, int
    :param horizons: forecasting horizons for labels, (h,) array
    :param queue_depth: the depth beyond which to aggregate the queue, int
    :param smoothing: whether to use "uniform" or "horizon" smoothing, bool
    :param k: smoothing window for returns when smoothing = "uniform", int
    :return: saves the processed features in output_path, as .npz files with numpy attributes "features" (:, NF, queue_depth) and "responses" (:, h).
    """
    csv_file_list = glob.glob(os.path.join(input_path, "*.{}".format("csv")))

    csv_orderbook = [name for name in csv_file_list if "orderbook" in name]
    csv_message = [name for name in csv_file_list if "message" in name]

    csv_orderbook.sort()
    csv_message.sort()

    # check if exactly half of the files are order book and exactly half are messages
    assert (len(csv_message) == len(csv_orderbook))
    assert (len(csv_file_list) == len(csv_message) + len(csv_orderbook))

    print("started multiprocessing")
    
    try:
        pool = Pool(os.cpu_count() - 2)  # leave 2 cpus free
        engine = ProcessL3(TICKER, output_path, NF, queue_depth, horizons, smoothing, k)
        logs = pool.map(engine, csv_orderbook)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    print("finished multiprocessing")

    with open(log_path + "/volumes_processing_logs.txt", "w") as f:
        for log in logs:
            f.write(log + "\n")

    print("please check processing logs.")


class ProcessL3(object):
    """
    Multiprocessing engine for pre-processing LOBSTER data into L3 volumes.
    """
    def __init__(self, TICKER, output_path, NF, queue_depth, horizons, smoothing, k):
        self.TICKER = TICKER
        self.output_path = output_path
        self.NF = NF
        self.queue_depth = queue_depth
        self.horizons = horizons
        self.smoothing = smoothing
        self.k = k

    def __call__(self, orderbook_name):
        output = process_L3_orderbook(orderbook_name, self.TICKER, self.output_path, self.NF, self.queue_depth, self.horizons, self.smoothing, self.k)
        return output


def process_L3_orderbook(orderbook_name, TICKER, output_path, NF, queue_depth, horizons, smoothing, k):
    """
    Function carrying out L3 volume processing of single order book orderbook_name. The data is treated in the following way:
    - order book states with crossed quotes are removed.
    - each state in the orderbook is time-stamped, with states occurring at the same time collapsed
      onto the last state.
    - the first and last 10 minutes of market activity (inside usual opening times) are dropped.
    - the smoothed returns at the requested horizons (in order book changes) are returned
      if smoothing = "horizon": l = (m+ - m)/m, where m+ denotes the mean of the next h mid-prices, m(.) is current mid price.
      if smoothing = "uniform": l = (m+ - m)/m, where m+ denotes the mean of the k+1 mid-prices centered at m(. + h), m(.) is current mid price.
    :param orderbook_name: order book to process, of shape (T, 4*levels):
                          "ASKp1", "ASKs1", "BIDp1",  "BIDs1", ..., "ASKplevels", "ASKslevels", "BIDplevels",  "BIDslevels",
                          str
    :param TICKER: the TICKER to be considered, str
    :param output_path: the path where we wish to save the processed datasets (as .npz files), str
    :param NF: number of volume features, NF = 2W where W is number of ticks on each side of mid, int
    :param queue_depth: the depth beyond which to aggregate the queue, int
    :param horizons: forecasting horizons for labels, (h,) array
    :param smoothing: whether to use "uniform" or "horizon" smoothing, bool
    :param k: smoothing window for returns when smoothing = "uniform", int
    :return: save the processed features in output_path, as .npz files with numpy attributes "features" (:, NF, queue_depth) and "responses" (:, h).
    """
    print(orderbook_name)
    log = ''
    try:
        df_orderbook = pd.read_csv(orderbook_name, header=None)
    except:
        return orderbook_name + ' skipped. Error: failed to read orderbook.'

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
        return orderbook_name + ' skipped. Error: failed to read messagebook.'

    # check the two df have the same length
    assert (len(df_message) == len(df_orderbook))

    # add column names to message book
    df_message.columns = ["seconds", "event type", "order ID", "volume", "price", "direction"]

    # remove trading halts
    trading_halts_start = df_message[(df_message["event type"] == 7)&(df_message["price"] == -1)].index
    trading_halts_end = df_message[(df_message["event type"] == 7)&(df_message["price"] == 1)].index
    trading_halts_index = np.array([])
    for halt_start, halt_end in zip(trading_halts_start, trading_halts_end):
        trading_halts_index = np.append(trading_halts_index, df_message.index[(df_message.index >= halt_start)&(df_message.index < halt_end)])
    if len(trading_halts_index) > 0:
        for halt_start, halt_end in zip(trading_halts_start, trading_halts_end):
            log = log + 'Warning: trading halt between ' + str(df_message.loc[halt_start, "seconds"]) + ' and ' + str(df_message.loc[halt_end, "seconds"]) + ' in ' + orderbook_name + '.\n'
    df_orderbook = df_orderbook.drop(trading_halts_index)
    df_message = df_message.drop(trading_halts_index)

    # remove crossed quotes
    crossed_quotes_index = df_orderbook[(df_orderbook["BIDp1"] > df_orderbook["ASKp1"])].index
    if len(crossed_quotes_index) > 0:
        log = log + 'Warning: ' + str(len(crossed_quotes_index)) + ' crossed quotes removed in ' + orderbook_name + '.\n'
    df_orderbook = df_orderbook.drop(crossed_quotes_index)
    df_message = df_message.drop(crossed_quotes_index)

    # add the seconds since midnight column to the order book from the message book
    df_orderbook.insert(0, "seconds", df_message["seconds"])

    df_orderbook_full = df_orderbook
    df_message_full = df_message

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
        log = log + 'Warning: unusual opening times in ' + orderbook_name + ': ' + str(market_open) + ' - ' + str(market_close) + '.\n'
    
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

    # Assumes tick_size = 0.01 $, as per LOBSTER data
    ticks = np.hstack((np.outer(np.round((df_orderbook_full["mid price"] - 25) / 100) * 100, np.ones(NF//2)) + 100 * np.outer(np.ones(len(df_orderbook_full)), np.arange(-NF//2+1, 1)),
                        np.outer(np.round((df_orderbook_full["mid price"] + 25) / 100) * 100, np.ones(NF//2)) + 100 * np.outer(np.ones(len(df_orderbook_full)), np.arange(NF//2))))
    ticks = ticks.astype(int)
    
    volumes = np.zeros((len(df_orderbook_full), NF))

    orderbook_states = df_orderbook_full[feature_names]
    orderbook_states_prices = orderbook_states.values[:, ::2]
    orderbook_states_volumes = orderbook_states.values[:, 1::2]

    for i in range(NF):
        # match tick prices with prices in levels of orderbook
        flags = (orderbook_states_prices == np.repeat(ticks[:, i].reshape((len(orderbook_states_prices), 1)), orderbook_states_prices.shape[1], axis=1))
        volumes[flags.sum(axis=1) > 0, i] = orderbook_states_volumes[flags]

    orderbook_L3 = np.zeros((len(df_orderbook_full), NF, queue_depth))

    prices_dict = {}
    
    # skip first message_df row
    skip = True
    
    for i, index in enumerate(df_message_full.index):
        seconds = df_message_full.loc[index, "seconds"]
        event_type = df_message_full.loc[index, "event type"]
        order_id = df_message_full.loc[index, "order ID"]
        price = df_message_full.loc[index, "price"]
        volume = df_message_full.loc[index, "volume"]

        # if new price is entering range (re-)initialize dict
        for price_ in orderbook_states_prices[i, :]:
            if (price_ in orderbook_states_prices[i - 1, :]) and (price_ in prices_dict.keys()):
                pass
            else:
                j = np.where(orderbook_states_prices[i, :]==price_)[0][0]
                volume_ = orderbook_states_volumes[i, j]
                prices_dict[price_] = pd.DataFrame(np.array([[volume_, 0]]), index = ['id'], columns = ["volume", "seconds"])
                if price_ == price:
                    skip = True
        
        price_df = prices_dict.get(price, pd.DataFrame([], columns = ["volume", "seconds"], dtype=float))
        
        # if new price also corresponds to message_df price skip to avoid double counting
        if skip:
            skip = False
            pass
        
        # if the price from message_df is not in df_orderbook prices, i.e. a failure in LOBSTER reconstruction system, treat as hidden market order
        elif (not price in orderbook_states_prices[(i-1):(i+1), :])&(event_type in [1, 2, 3, 4, 5]):
            event_type = 5
            pass

        # new limit order
        elif event_type == 1:
            price_df.loc[order_id] = [volume, seconds]
        
        # cancellation (partial deletion)
        elif event_type == 2:
            # if order_id is not in price dataframe then this is part of initial order
            if order_id in price_df.index:
                price_df.loc[order_id, "volume"] -= volume
            else:
                price_df.loc["id", "volume"] -= volume
        
        # deletion
        elif event_type == 3:
            # if id is not present (i.e. it is at the start of order book), delete from "id" and check if "id" is empty
            if order_id in price_df.index:
                price_df = price_df.drop(order_id)
            else:
                price_df.loc["id", "volume"] -= volume
                if price_df.loc["id", "volume"] == 0:
                    price_df = price_df.drop("id")
        
        # execution of visible limit order
        elif event_type == 4:
            if order_id in price_df.index:
                price_df.loc[order_id, "volume"] -= volume
                if price_df.loc[order_id, "volume"] == 0:
                    price_df = price_df.drop(order_id)
            else:
                price_df.loc["id", "volume"] -= volume
                if price_df.loc["id", "volume"] == 0:
                    price_df = price_df.drop("id")
        
        # execution of hidden limit order
        elif event_type == 5:
            pass
        
        # auction trade
        elif event_type == 6:
            pass
        
        # trading halt
        elif event_type == 7:
            # re-initialize prices_dict
            prices_dict = {}
            for price_ in orderbook_states_prices[i, :]:
                j = np.where(orderbook_states_prices[i, :]==price_)[0][0]
                volume_ = orderbook_states_volumes[i, j]
                prices_dict[price_] = pd.DataFrame(np.array([[volume_, 0]]), index = ['id'], columns = ["volume", "seconds"])
            pass
        
        else:
            raise ValueError("LOBSTER event type must be 1, 2, 3, 4, 5, 6 or 7")
        
        price_df = price_df.sort_values(by="seconds")
        prices_dict[price] = price_df

        # update orderbooks_L3
        if event_type in [5, 6]:
            orderbook_L3[i, :, :] = orderbook_L3[i - 1, :, :] 
        elif (ticks[i, :] == ticks[i - 1, :]).all() & (not event_type == 7):
            if price in ticks[i, :]:
                j = np.where(ticks[i, :] == price)[0][0]
                orderbook_L3[i, :, :] = orderbook_L3[i - 1, :, :] 
                if len(price_df) == 0:
                    orderbook_L3[i, j, :] = np.zeros(queue_depth)
                elif len(price_df) < queue_depth:
                    orderbook_L3[i, j, :len(price_df)] = price_df["volume"].values
                    orderbook_L3[i, j, len(price_df):] = 0
                else:
                    orderbook_L3[i, j, :(queue_depth-1)] = price_df["volume"].values[:(queue_depth-1)]
                    orderbook_L3[i, j, queue_depth-1] = np.sum(price_df["volume"].values[(queue_depth-1):])
            else:
                orderbook_L3[i, :, :] = orderbook_L3[i - 1, :, :]
        else:
            for j, price_ in enumerate(ticks[i, :]):
                price_df_ = prices_dict.get(price_, [])
                if len(price_df_) == 0:
                    orderbook_L3[i, j, :] = np.zeros(queue_depth)
                elif len(price_df_) < queue_depth:
                    orderbook_L3[i, j, :len(price_df_)] = price_df_["volume"].values
                else:
                    orderbook_L3[i, j, :(queue_depth-1)] = price_df_["volume"].values[:(queue_depth-1)]
                    orderbook_L3[i, j, queue_depth-1] = np.sum(price_df_["volume"].values[(queue_depth-1):])
        
    # need then to remove all same timestamps and first/last 10 minutes (collapse to last)
    orderbook_L3 = orderbook_L3[df_orderbook_full.index.isin(df_orderbook.index), :, :]

    if smoothing == "horizon":
        # create labels for returns with smoothing labelling method
        for h in horizons:
            rolling_mid = df_orderbook["mid price"].rolling(h).mean().dropna()[1:]
            rolling_mid = rolling_mid.to_numpy().flatten()
            smooth_pct_change = rolling_mid/df_orderbook["mid price"][:-h] - 1
            df_orderbook[str(h)] = np.concatenate((smooth_pct_change, np.repeat(np.NaN, int(h))))
    elif smoothing == "uniform":
        # create labels for returns with smoothing labelling method
        rolling_mid = df_orderbook["mid price"].rolling(k+1, center=True).mean()
        rolling_mid = rolling_mid.to_numpy().flatten()
        for h in horizons:
            smooth_pct_change = rolling_mid[h:]/df_orderbook["mid price"][:-h] - 1
            df_orderbook[str(h)] = smooth_pct_change

    # drop seconds and mid price columns
    df_orderbook = df_orderbook.drop(["seconds", "mid price"], axis=1)

    # drop elements with na predictions at the end which cannot be used for training
    if smoothing == "horizon":
        k = 0
    orderbook_L3 = orderbook_L3[:-(max(horizons)+k//2), :, :]
    returns = df_orderbook.iloc[:-(max(horizons)+k//2), -len(horizons):].values

    # save
    output_name = os.path.join(output_path, TICKER + "_" + "volumes" + "_" + str(date.date()))
    np.savez(output_name + ".npz", features=orderbook_L3, responses=returns)

    return log + orderbook_name + ' completed.'