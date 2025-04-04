from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.client import TradingClient
from datetime import datetime, timedelta
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import trading.stock_trading_utils as trade_utils
from data_collection.stock_data_container import StockDataContainer
import data_collection.data_utils as dataUtils
import modeling.result_saving_utils as savingUtils
from config import(
    STOCK_NAMES, PREDICTION_WINDOW, SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS, EARLY_STOP_PATIENCE
)

from trade_script_simulated import TRADING_CLIENT, BUY_ORDERS


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
    order = TRADING_CLIENT.submit_order(order_data = marketOrderData)
    logTrade(tradeType, ticker, quantity, sell_date)

def logTrade(tradeType, ticker, quantity, sell_date=-1):
    today_date = datetime.now()
    logEntry = (f"[{today_date.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{tradeType} - Ticker: {ticker} - Quantity: {quantity} - ")
    if tradeType == "BUY":
        logEntry += f"Sell Date: {sell_date.strftime('%Y-%m-%d')} - "
    
    with open("trade_log.txt", 'a') as file:
        file.write(logEntry)

def get_trading_client():
    # Get API keys
    with open('api_keys/alpaca/key.txt') as file:
        KEY = file.read()
    with open('api_keys/alpaca/secret.txt') as file:
        SECRET = file.read()

    # Initialize trading client
    return TradingClient(KEY, SECRET, paper=True)

def check_model_performance(testing_labels, predicted_labels):
    r_value = np.corrcoef(testing_labels, predicted_labels)[0, 1]
    rmse = np.sqrt(mean_squared_error(testing_labels, predicted_labels))

    '''
    Arbitrary testing metrics. Adjust later
    '''
    if (r_value > 0.7) and (rmse < 5):
        return True
    return False


def make_next_prediction(stock_data, feature_scaler, label_scaler):
    '''
    Assuming model is located in home dir of ticker and called model.keras
    '''
    model = tf.keras.models.load_model(f'data/{stock_data.ticker}/model.keras')

    temp = stock_data.training_data.copy()

    temp = temp.iloc[::-1].reset_index(drop=True)
    temp = temp.drop(columns=['Close', 'Date'])
    new_features = temp[-PREDICTION_WINDOW:]

    new_features = feature_scaler.transform(new_features)
    new_features = tf.expand_dims(new_features, axis=0)
    predictions = label_scaler.inverse_transform(model.predict(new_features))

    return list(tf.squeeze(predictions, axis=0))

def worth_buying(stock_data, predictions):
    close_df = stock_data.training_data.copy()['Close']
    close_df.iloc[::-1].reset_index(drop=True)

    max_value = max(predictions)

    '''
    Arbitrary cut off point for checking if the price is going to change enough. Change later
    '''
    if (max_value > (close_df[0] + 5)):
        return True
    return False


def get_sell_date(stock_data, predictions):
    close_df = stock_data.training_data.copy()['Close']
    close_df.iloc[::-1].reset_index(drop=True)

    max_value = max(predictions)
    max_index = predictions.index(max_value)
    return (datetime.now() + timedelta(days=max_index))


def buy_script():
    if STOCK_NAMES[0] == 'ALL':
        dataUtils.get_all_tickers()

    for stock in STOCK_NAMES:
        
        # Initialize container for stock data
        stock_data = StockDataContainer(stock)

        # dataUtils.backup_data(stock_data.ticker)
        stock_data.update_all_data((pd.to_datetime(stock_data.last_updated) + pd.Timedelta(days=1)))

        # Check for no new data
        if stock_data.raw_data.empty:
            return

        new_data = dataUtils.clean_and_process_data(stock_data.raw_data)
        num_new_data = len(new_data)

        # Merge old and new data
        stock_data.training_data = dataUtils.merge_with_old_data(stock_data)

        dataUtils.save_data(stock_data.training_data, stock_data.ticker)

        # Get rid of data that the model has already trained on
        # Explanation:
        # - num_new_data+PREDICTION_WINDOW+SEQUENCE_LENGTH gets you to the start of the last sequence of features the model has trained on
        # - -1 to move the window 1 forward for new
        # - +greatest_window to account for division between train and testing data
        greatest_window = dataUtils.get_greatest_window()
        modified_training_data = stock_data.training_data.iloc[:num_new_data+PREDICTION_WINDOW+SEQUENCE_LENGTH-1+greatest_window]
        modified_training_data.to_csv('data/AAPL/new_data.csv')
        # Reverse data so model trains from oldest to newest
        modified_training_data = modified_training_data.iloc[::-1].reset_index(drop=True)
        # Drop Date column since only used for debugging/data alignment purposes
        modified_training_data = modified_training_data.drop(columns=['Date'])


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

        savingUtils.save_model(model, f'data/{stock.ticker}/model.keras')


        # Check if model performed well enough
        if (check_model_performance(testing_labels, predicted_labels) == True):
            predictions = make_next_prediction(stock_data, feature_scaler, label_scaler)
            if (worth_buying(stock_data, predictions)):
                print("Worth buying!!")
                sell_date = get_sell_date(stock_data, predictions)
                BUY_ORDERS[stock_data.ticker] = {'quantity': 5, 'hold_date': sell_date}


