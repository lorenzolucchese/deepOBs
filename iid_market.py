import pandas as pd
import re
import datetime
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def action_edf(csv_file_list, order_replacement=True):
    """
    Function for mapping a set of LOBSTER data to the corresponding unconditional 
    distribution of actions. NOTE this might assume volumes are stationary.
    """

    csv_orderbook = [name for name in csv_file_list if "orderbook" in name]
    csv_message = [name for name in csv_file_list if "message" in name]

    csv_orderbook.sort()
    csv_message.sort()

    # check if exactly half of the files are order book and exactly half are messages
    assert (len(csv_message) == len(csv_orderbook))
    assert (len(csv_file_list) == len(csv_message) + len(csv_orderbook))

    print("started loop")

    logs = []
    counter = Counter({})

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
        df_message = df_message.drop("order ID", axis=1)

        # add the seconds since midnight column to the order book from the message book
        df_orderbook.insert(0, "seconds", df_message["seconds"])
        df_orderbook.insert(0, "event type", df_message["event type"])

        # check market opening times for strange values
        market_open = int(df_orderbook["seconds"].iloc[0] / 60) / 60  # open at minute before first transaction
        market_close = (int(df_orderbook["seconds"].iloc[-1] / 60) + 1) / 60  # close at minute after last transaction

        if not (market_open == 9.5 and market_close == 16):
            logs.append('Warning: unusual opening times in ' + orderbook_name + ': ' + str(market_open) + ' - ' + str(market_close) + '.')

        # drop values outside of market hours
        df_orderbook = df_orderbook.loc[(df_orderbook["seconds"] >= 34200) &
                                        (df_orderbook["seconds"] <= 57600)]
        df_message = df_message.loc[(df_message["seconds"] >= 34200) &
                                    (df_message["seconds"] <= 57600)]

        # drop first and last 10 minutes of trading
        market_open_seconds = market_open * 60 * 60 + 10 * 60
        market_close_seconds = market_close * 60 * 60 - 10 * 60
        df_orderbook = df_orderbook.loc[(df_orderbook["seconds"] >= market_open_seconds) &
                                        (df_orderbook["seconds"] <= market_close_seconds)]

        df_message = df_message.loc[(df_message["seconds"] >= market_open_seconds) &
                                    (df_message["seconds"] <= market_close_seconds)]

        # one conceptual event may appear as multiple rows in the message file, all with the same timestamp:
        # - a limit order modification which is implemented as a cancellation followed by an immediate new arrival, (we keep separate)
        # - a single market order executing against multiple resting limit orders, (we group into a market order)
        # - an aggressive limit order exceuting against multiple resting limit orders (we group into a market order)
        # need to handle these: we aggregate aggressive limit orders which leave a residual into limit orders, everything else into market orders.

        # first aggregate events happening at the same time with the same event type 4 (i.e. one order getting executed against multiple resting orders)
        # note that event types will be returned in increasing order at the same timestamp after grouping
        df_message = df_message.groupby(["seconds", "event type"]).agg({"volume": "sum", "direction": "mean", "price": "mean"}).reset_index()
        df_orderbook = df_orderbook.groupby(["seconds", "event type"]).tail(1).reset_index()

        # drop data errors (?) i.e. when multiple limit orders are exceuted at the same time on two different sides of the order book
        df_orderbook = df_orderbook.drop(df_message.index[(df_message.loc[:, "direction"] == 0.0)], axis=0)
        df_message = df_message.drop(df_message.index[(df_message.loc[:, "direction"] == 0.0)], axis=0)

        # treat case by case the remaining message rows with the same timestamp
        repeated_timestamp_index = df_message.iloc[1:-1, :].iloc[(df_message.iloc[1:-1, 0].values == df_message.iloc[2:, 0].values)|(df_message.iloc[1:-1, 0].values == df_message.iloc[:-2, 0].values), :].index
        unique_repeated_timestamps = set(df_message.loc[repeated_timestamp_index, "seconds"])
        same_timestamps_indices = []
        lengths = []
        for timestamp in unique_repeated_timestamps:
            same_timestamps_indices.append(df_message.index[df_message.loc[:, "seconds"].values == timestamp])
            lengths.append(len(df_message.index[df_message.loc[:, "seconds"].values == timestamp]))

        if order_replacement:
            # add a cancel price and volume column for replacement
            df_message.insert(len(df_message.columns), "cancel price", 0)
            df_message.insert(len(df_message.columns), "cancel volume", 0)

        remove_indices = []
        for indices in same_timestamps_indices:
            if len(indices) == 2:
                previous_row = df_message.loc[indices[0], :]
                current_row = df_message.loc[indices[1], :]
                # order replacement: treat as separate cancellation + new limit order
                if (previous_row["event type"] == 1) & (current_row["event type"] == 3):
                    if order_replacement:
                        # "seconds", "event type", "volume", "direction", "price", "cancel price", "cancel volume"
                        df_message.loc[indices[1], :] = [current_row["seconds"], 8, previous_row["volume"], previous_row["direction"], previous_row["price"], current_row["price"], current_row["volume"]]
                        remove_indices.append(indices[0])
                    else:
                        pass
                # aggressive limit order which leaves residual after executing against visible limit order: treat as a (unique) limit order
                elif (previous_row["event type"] == 1) & (current_row["event type"] == 4):
                    if order_replacement:
                        df_message.loc[indices[1], :] = [current_row["seconds"], 1, previous_row["volume"] + current_row["volume"], previous_row["direction"], previous_row["price"], 0, 0]
                    else:
                        df_message.loc[indices[1], :] = [current_row["seconds"], 1, previous_row["volume"] + current_row["volume"], previous_row["direction"], previous_row["price"]]
                    remove_indices.append(indices[0])
                # aggressive limit order which leaves residual after executing against hidden limit order: treat as a (unique) limit order
                elif (previous_row["event type"] == 1) & (current_row["event type"] == 5):
                    if order_replacement:
                        df_message.loc[indices[1], :] = [current_row["seconds"], 1, previous_row["volume"] + current_row["volume"], previous_row["direction"], previous_row["price"], 0, 0]
                    else:
                        df_message.loc[indices[1], :] = [current_row["seconds"], 1, previous_row["volume"] + current_row["volume"], previous_row["direction"], previous_row["price"]]
                    remove_indices.append(indices[0])
                # order replacement with new limit order executed
                elif (previous_row["event type"] == 3) & (current_row["event type"] == 4):
                    if order_replacement:
                        # "seconds", "event type", "volume", "direction", "price", "cancel price", "cancel volume"
                        df_message.loc[indices[1], :] = [current_row["seconds"], 8, current_row["volume"], previous_row["direction"], current_row["price"], previous_row["price"], previous_row["volume"]]
                        remove_indices.append(indices[0])
                    else:
                        # treat as separate cancellation + execution limit order
                        df_message.loc[indices[1], :] = [current_row["seconds"], 1, current_row["volume"], -1*current_row["direction"], current_row["price"]]
                # order replacement with hidden limit order executed
                elif (previous_row["event type"] == 3) & (current_row["event type"] == 5):
                    if order_replacement:
                        # "seconds", "event type", "volume", "direction", "price", "cancel price", "cancel volume"
                        df_message.loc[indices[1], :] = [current_row["seconds"], 8, current_row["volume"], -1*current_row["direction"], current_row["price"], previous_row["price"], previous_row["volume"]]
                        remove_indices.append(indices[0])
                    else:
                        # treat as separate cancellation + execution limit order
                        df_message.loc[indices[1], :] = [current_row["seconds"], 1, current_row["volume"], -1*current_row["direction"], current_row["price"]]
                # market order executing against both visible and limit orders, treat as a single market order
                elif (previous_row["event type"] == 4) & (current_row["event type"] == 5):
                    if order_replacement:
                        df_message.loc[indices[1], :] = [current_row["seconds"], 4, previous_row["volume"] + current_row["volume"], previous_row["direction"], previous_row["price"], 0, 0]
                    else:
                        df_message.loc[indices[1], :] = [current_row["seconds"], 4, previous_row["volume"] + current_row["volume"], previous_row["direction"], previous_row["price"]]
                    remove_indices.append(indices[0])
                else:
                    print(df_message.loc[indices, :])
                    raise ValueError("Error: Unkown sequence of events")
            if len(indices) == 3:
                row0 = df_message.loc[indices[0], :]
                row1 = df_message.loc[indices[1], :]
                row2 = df_message.loc[indices[2], :]
                # order replacement with execution and residual
                if (row0["event type"] == 1) & (row1["event type"] == 3) & (row2["event type"] == 4):
                    if order_replacement:
                        # "seconds", "event type", "volume", "direction", "price", "cancel price", "cancel volume"
                        df_message.loc[indices[2], :] = [row2["seconds"], 8, row0["volume"] + row2["volume"], row0["direction"], row0["price"], row1["price"], row1["volume"]]
                        remove_indices.append(indices[0])
                        remove_indices.append(indices[1])
                    else:
                        # treat as separate cancellation + new aggressive limit order
                        # leave indices[1], i.e. event type 3, unchanged but aggregate indices[0] and indices[2]
                        df_message.loc[indices[2], :] = [row2["seconds"], 1, row0["volume"] + row2["volume"], row0["direction"], row0["price"]]
                        remove_indices.append(indices[0])
                # order replacement with execution against hidden limit order and residual
                elif (row0["event type"] == 1) & (row1["event type"] == 3) & (row2["event type"] == 5):
                    if order_replacement:
                        # "seconds", "event type", "volume", "direction", "price", "cancel price", "cancel volume"
                        df_message.loc[indices[2], :] = [row2["seconds"], 8, row0["volume"] + row2["volume"], row0["direction"], row0["price"], row1["price"], row1["volume"]]
                        remove_indices.append(indices[0])
                        remove_indices.append(indices[1])
                    else:
                        # treat as separate cancellation + new aggressive limit order
                        # leave indices[1], i.e. event type 3, unchanged but aggregate indices[0] and indices[2]
                        df_message.loc[indices[2], :] = [row2["seconds"], 1, row0["volume"] + row2["volume"], row0["direction"], row0["price"]]
                        remove_indices.append(indices[0])
                # aggressive limit order executing against visible and hidden orders and leaving residual:  treat as a single aggresive limit order
                elif (row0["event type"] == 1) & (row1["event type"] == 4) & (row2["event type"] == 5):
                    # aggregate indices[0], indices[1] and indices[2]
                    if order_replacement:
                        df_message.loc[indices[2], :] = [row2["seconds"], 1, row0["volume"] + row1["volume"] + row2["volume"], row0["direction"], row0["price"], 0, 0]
                    else:
                        df_message.loc[indices[2], :] = [row2["seconds"], 1, row0["volume"] + row1["volume"] + row2["volume"], row0["direction"], row0["price"]]
                    remove_indices.append(indices[0])
                    remove_indices.append(indices[1])
                # order replacement with execution against visible and hidden orders
                elif (row0["event type"] == 3) & (row1["event type"] == 4) & (row2["event type"] == 5):
                    if order_replacement:
                        # "seconds", "event type", "volume", "direction", "price", "cancel price", "cancel volume"
                        df_message.loc[indices[2], :] = [row2["seconds"], 8, row1["volume"] + row2["volume"], -1*row1["direction"], row1["price"], row0["price"], row0["volume"]]
                        remove_indices.append(indices[0])
                        remove_indices.append(indices[1])
                    else:
                        # treat as separate cancellation + new limit order
                        # leave indices[0], i.e. event type 3, unchanged but aggregate indices[1] and indices[2]
                        df_message.loc[indices[2], :] = [row2["seconds"], 1, row1["volume"] + row2["volume"], -1*row1["direction"], row1["price"]]
                        remove_indices.append(indices[1])
                else:
                    print(df_message.loc[indices, :])
                    raise ValueError("Error: Unkown sequence of events")
        df_message = df_message.drop(remove_indices, axis=0)
        df_orderbook = df_orderbook.drop(remove_indices, axis=0)

        # change all market orders (event type = 4/5) to the other direction 
        # (since event tyoe 4/5 indicates the execution of the BUY/SELL limit order but we want to treat the action as a SELL/BUY market order)
        df_message.loc[df_message.index[(df_message.loc[:, "event type"] == 4) | (df_message.loc[:, "event type"] == 5)], "direction"] *= -1
        # drop trading halts and cross trades
        df_orderbook = df_orderbook.drop(df_message.index[(df_message.loc[:, "event type"] == 6)|(df_message.loc[:, "event type"] == 7)], axis=0)
        df_message = df_message.drop(df_message.index[(df_message.loc[:, "event type"] == 6)|(df_message.loc[:, "event type"] == 7)], axis=0)

        # change labeling of event type and direction
        df_message.loc[df_message.index[df_message.loc[:, "direction"] == +1.0], "direction"] = "buy"
        df_message.loc[df_message.index[df_message.loc[:, "direction"] == -1.0], "direction"] = "sell"
        
        df_message.loc[df_message.index[df_message.loc[:, "event type"] == 1], "event type"] = "limit order"
        df_message.loc[df_message.index[df_message.loc[:, "event type"] == 2], "event type"] = "cancellation"
        df_message.loc[df_message.index[df_message.loc[:, "event type"] == 3], "event type"] = "cancellation"
        df_message.loc[df_message.index[df_message.loc[:, "event type"] == 4], "event type"] = "market order"
        df_message.loc[df_message.index[df_message.loc[:, "event type"] == 5], "event type"] = "market order"

        if order_replacement:
            df_message.loc[df_message.index[df_message.loc[:, "event type"] == 8], "event type"] = "order replacement"

        # compute depth from best bid/ask
        # 'The k-th row in the 'message' file describes the limit order event causing the change in the limit order book from line k-1 to line k in the 'orderbook' file.'
        previous_price = (df_message.loc[df_message.index[1:], "direction"].values == "buy") * df_orderbook.loc[df_orderbook.index[:-1], "BIDp1"].values + (df_message.loc[df_message.index[1:], "direction"].values == "sell") * df_orderbook.loc[df_orderbook.index[:-1], "ASKp1"].values
        df_message = df_message.drop(df_message.index[0], axis=0)
        df_message.insert(len(df_message.columns), "depth", df_message.loc[:, "price"].values - previous_price)
        if order_replacement:
            df_message.insert(len(df_message.columns), "cancel depth", df_message.loc[:, "cancel price"].values - previous_price)

        # change sign of depth for buy actions
        df_message.loc[df_message.index[df_message.loc[:, "direction"] == "buy"], "depth"] *= -1
        if order_replacement:
            df_message.loc[df_message.index[df_message.loc[:, "direction"] == "buy"], "cancel depth"] *= -1

        # set to 0 the depth for market orders (irrelevant)
        df_message.loc[df_message.index[df_message.loc[:, "event type"] == "market order"], "depth"] = 0
        if order_replacement:
            # set to 0 the cancel depth (and volume) for limit orders, market orders and cancellations (irrelevant)
            df_message.loc[df_message.index[df_message.loc[:, "event type"] == "limit order"], "cancel depth"] = 0
            df_message.loc[df_message.index[df_message.loc[:, "event type"] == "market order"], "cancel depth"] = 0
            df_message.loc[df_message.index[df_message.loc[:, "event type"] == "cancellation"], "cancel depth"] = 0
        # drop seconds and prices
        df_message = df_message.drop(["seconds", "price"], axis=1)
        if order_replacement:
            # drop cancel price
            df_message = df_message.drop(["cancel price"], axis=1)

        counter.update(Counter([tuple(_) for _ in df_message.values.tolist()]))
    
    print("finished loop")

    return counter


if __name__ == "__main__":
    csv_file_list = [
                    #  'data_raw\WBA_data_dwn\WBA_2019-11-05_34200000_57600000_message_10.csv',
                    #  'data_raw\WBA_data_dwn\WBA_2019-11-05_34200000_57600000_orderbook_10.csv',
                     'data_raw\WBA_data_dwn\WBA_2019-11-06_34200000_57600000_message_10.csv',
                     'data_raw\WBA_data_dwn\WBA_2019-11-06_34200000_57600000_orderbook_10.csv']
    
    order_replacement = True

    counter = action_edf(csv_file_list, order_replacement=order_replacement)

    counter_df = pd.DataFrame.from_dict(counter, orient="index", columns=["frequency"]).reset_index()
    counter_df = counter_df.rename(columns={"index": "action"})
    if order_replacement:
        counter_df.loc[:, ["event type", "volume", "direction", "cancel volume", "depth", "cancel depth"]] = counter_df.loc[:, "action"].tolist()
    else:
        counter_df.loc[:, ["event type", "volume", "direction", "depth"]] = counter_df.loc[:, "action"].tolist()
    counter_df = counter_df.drop("action", axis=1)

    for display_quantity in ["depth", "volume", "cancel volume", "cancel depth"]:
        display_df = counter_df.drop(["depth" if display_quantity == "volume" else "volume"], axis=1)
        display_df = display_df.groupby(["event type", "direction", display_quantity]).sum().reset_index()
        for event_type in ["limit order", "market order", "cancellation", "order replacement"]:
            if (event_type in ["limit order", "cancellation"]) and (display_quantity in ["cancel volume", "cancel depth"]):
                pass
            elif (event_type == "market order") and (display_quantity in ["depth", "cancel volume", "cancel depth"]):
                pass
            else:
                for direction in ["buy", "sell"]:
                    labels = display_df.loc[display_df.index[(display_df.loc[:, "event type"] == event_type) & (display_df.loc[:, "direction"] == direction)], display_quantity]
                    values = display_df.loc[display_df.index[(display_df.loc[:, "event type"] == event_type) & (display_df.loc[:, "direction"] == direction)], "frequency"]
                    fig, ax = plt.subplots(figsize=(10, 10))
                    if display_quantity[-5:] == "depth":
                        ax.bar(labels, values, width=50)
                    else:
                        ax.set_yscale('log')
                        ax.bar(np.log10(np.array(labels)), values, width=0.1)
                        ax.set_xticks(np.log10(np.array([1, 10, 100, 1000, 10000])))
                        ax.set_xticklabels(np.array([1, 10, 100, 1000, 10000]))
                    fig.savefig('auxiliary_code/plots/' + event_type + '_' + direction + '_' + display_quantity + '.png', format='png', bbox_inches='tight', pad_inches=0)