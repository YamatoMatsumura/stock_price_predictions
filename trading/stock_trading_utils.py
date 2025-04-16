from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import timedelta
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import os

from data_collection.stock_data_container import StockDataContainer
import data_collection.data_utils as dataUtils
import modeling.result_saving_utils as savingUtils
from config import(
    STOCK_NAMES, PREDICTION_WINDOW, SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS, EARLY_STOP_PATIENCE
)

import simulation_globals as sim

def tradeStock(tradeType, ticker, quantity, sell_date=-1):
    if tradeType == "BUY":
        tradingSide = OrderSide.BUY
    elif tradeType == "SELL":
        tradingSide = OrderSide.SELL
    else:
        print("Invalid Trade Type: Must be a buy or sell order")


    # Initialize order data
    marketOrderData = MarketOrderRequest(
        symbol=ticker,
        qty=quantity,
        side=tradingSide,
        time_in_force=TimeInForce.DAY)
    
    # Submit order to trading client
    order = sim.TRADING_CLIENT.submit_order(order_data = marketOrderData)
    logTrade(tradeType, ticker, quantity, sell_date)

def tradeStockSimulated(tradeType, ticker, quantity, historical_prices, sell_date=-1):
    if tradeType == "BUY":
        currentPrice = historical_prices[historical_prices['date'] == sim.SIMULATED_TIME.strftime('%Y-%m-%d')]['close'].values[0]
        print(f"Buying {quantity} shares of {ticker} at {currentPrice}")
        sim.BALANCE -= (currentPrice * quantity)
        logTrade(tradeType, ticker, quantity, sell_date)

    if tradeType == "SELL":
        currentPrice = historical_prices[historical_prices['date'] == sim.SIMULATED_TIME.strftime('%Y-%m-%d')]['close'].values[0]
        print(f"Selling {quantity} shares of {ticker} at {currentPrice}")
        sim.BALANCE += (currentPrice * quantity)
        logTrade(tradeType, ticker, quantity, sell_date)

def logTrade(tradeType, ticker, quantity, sell_date=-1):
    today_date = sim.SIMULATED_TIME
    logEntry = (f"[{today_date.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{tradeType} - Ticker: {ticker} - Quantity: {quantity}")
    if tradeType == "BUY":
        logEntry += f" - Sell Date: {sell_date.strftime('%Y-%m-%d')}"
    
    logEntry += "\n"
    
    with open("trade_log.txt", 'a') as file:
        file.write(logEntry)

def getNetWorthSimulated():
    netWorth = sim.BALANCE
    for position in sim.PORTFOLIO:
        historical_price = pd.read_csv(f"data/historical_prices/{position['ticker']}.csv")
        quantity = position['quantity']
        currentPrice = historical_price[historical_price['date'] == sim.SIMULATED_TIME.strftime('%Y-%m-%d')]['close'].values[0]
        netWorth += (currentPrice * quantity)
    return netWorth

def check_model_performance(testing_labels, predicted_labels):
    r_value = np.corrcoef(testing_labels, predicted_labels)[0, 1]
    rmse = np.sqrt(mean_squared_error(testing_labels, predicted_labels))

    print(f"R Value: {r_value}")
    print(f"RMSE: {rmse}")

    '''
    Arbitrary testing metrics. Adjust later
    '''
    if (rmse < 5):
        print("Model performed well enough")
        return True
    print("Model did not perform well enough")
    return False


def make_next_prediction(stock_data, feature_scaler, label_scaler):
    '''
    Assuming model is located in home dir of ticker and called model.keras
    '''
    model = tf.keras.models.load_model(f'data/{stock_data.ticker}/model.keras')

    temp = stock_data.training_data.copy()

    temp = temp.iloc[::-1].reset_index(drop=True)
    temp = temp.drop(columns=['close', 'date'])
    new_features = temp[-PREDICTION_WINDOW:]

    new_features = feature_scaler.transform(new_features)
    new_features = tf.expand_dims(new_features, axis=0)
    predictions = label_scaler.inverse_transform(model.predict(new_features))

    return list(tf.squeeze(predictions, axis=0))

def worth_buying(stock_data, predictions):
    close_df = stock_data.training_data.copy()['close']
    close_df.iloc[::-1].reset_index(drop=True)

    max_value = max(predictions)

    '''
    Arbitrary cut off point for checking if the price is going to change enough. Change later
    '''
    # if (max_value > (close_df[0] + 0)):
    #     return True
    # print(f"Max predicted value: {max_value} is less than current price: {close_df[0]}")
    # return False
    return True


def get_sell_date(stock_data, predictions):
    close_df = stock_data.training_data.copy()['close']
    close_df.iloc[::-1].reset_index(drop=True)

    max_value = max(predictions)
    max_index = predictions.index(max_value)
    return (sim.SIMULATED_TIME + timedelta(days=max_index+1))


def stock_pick_script():
    if STOCK_NAMES[0] == 'ALL':
        dataUtils.get_all_tickers()

    for stock in STOCK_NAMES:

        # Initialize container for stock data
        stock_data = StockDataContainer(stock)

        # dataUtils.backup_data(stock_data.ticker)
        # Only call API if current day is a weekday
        if sim.SIMULATED_TIME.weekday() <= 4:
            stock_data.update_all_data(end_date=sim.SIMULATED_TIME)
        else:
            print("Weekend... not pulling new data")
            continue

        prev_data_count = len(pd.read_csv(f'data/{stock_data.ticker}/trained_data.csv'))

        # Merge old and new data
        stock_data.training_data = dataUtils.merge_with_old_data(stock_data)
        num_new_data = len(stock_data.training_data) - prev_data_count

        # Double check in case no new data was pulled
        if num_new_data == 0:
            print(f"Fetched no new data for {sim.SIMULATED_TIME}... returning")
            continue


        dataUtils.save_data(stock_data.training_data, stock_data.ticker)

        '''
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression, ElasticNet
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        modified_training_data = stock_data.training_data.copy()
        # Reverse data so model trains from oldest to newest
        modified_training_data = modified_training_data.iloc[::-1].reset_index(drop=True)
        # Drop Date column since only used for debugging/data alignment purposes
        modified_training_data = modified_training_data.drop(columns=['date'])

        features = modified_training_data.drop(columns=['close'])
        labels = modified_training_data['close']

        features = features.iloc[:-PREDICTION_WINDOW]
        labels = labels.iloc[PREDICTION_WINDOW:]

        training_features, testing_features, training_labels, testing_labels = train_test_split(features, labels, test_size=PREDICTION_WINDOW, shuffle=False)
        scaler = StandardScaler()
        training_features = scaler.fit_transform(training_features)
        testing_features = scaler.transform(testing_features)

        model = LinearRegression()
        model.fit(training_features, training_labels)
        predicted_labels = model.predict(testing_features)
        '''


        # Get rid of data that the model has already trained on
        # Explanation:
        # - num_new_data+PREDICTION_WINDOW+SEQUENCE_LENGTH gets you to the start of the last sequence of features the model has trained on
        # - -1 to move the window 1 forward for new
        # - +greatest_window to account for division between train and testing data
        greatest_window = dataUtils.get_greatest_window()
        modified_training_data = stock_data.training_data.iloc[:num_new_data+PREDICTION_WINDOW+SEQUENCE_LENGTH-1+greatest_window+20]
        # Reverse data so model trains from oldest to newest
        modified_training_data = modified_training_data.iloc[::-1].reset_index(drop=True)
        # Drop Date column since only used for debugging/data alignment purposes
        modified_training_data = modified_training_data.drop(columns=['date'])


        features, labels = dataUtils.create_windowed_dataset(modified_training_data)

        training_features, training_labels, testing_features, testing_labels = dataUtils.split_dataset(features, labels)

        (
            scaled_training_features, 
            scaled_training_labels, 
            scaled_testing_features, 
            scaled_testing_labels, 
            feature_scaler, 
            label_scaler,
        ) = dataUtils.scale_dataset(training_features, training_labels, testing_features, testing_labels)

        # Grab trained model
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
        model = tf.keras.models.load_model(f'data/{stock_data.ticker}/model.keras')

        history = model.fit(scaled_training_features, scaled_training_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping])

        # Make predictions
        scaled_predictions = model.predict(scaled_testing_features)
        predicted_labels = label_scaler.inverse_transform(scaled_predictions)

        savingUtils.save_model(model, f'data/{stock_data.ticker}')



        save_dir = f'data/debug/{stock_data.ticker}'
        os.makedirs(save_dir, exist_ok=True)
        graph = savingUtils.create_results_graph(predicted_labels, stock_data)
        savingUtils.save_graph(graph, save_dir, f"{sim.SIMULATED_TIME.strftime('%Y-%m-%d')}_results.png")
        # PREDICTIONS.append(predicted_labels)

        # Check if model performed well enough
        if (check_model_performance(testing_labels, predicted_labels) == True):
            predictions = make_next_prediction(stock_data, feature_scaler, label_scaler)
            if (worth_buying(stock_data, predictions)):
                sell_date = get_sell_date(stock_data, predictions)
                sim.BUY_ORDERS[stock_data.ticker] = {'quantity': 5, 'sell_date': sell_date}
                print(f"Queueing {stock_data.ticker} for 5 shares")
                print(f"Buy Orders: {sim.BUY_ORDERS}")

