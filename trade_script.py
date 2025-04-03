import pandas as pd
import tensorflow as tf
from alpaca.trading.client import TradingClient
from datetime import datetime


from data_collection.stock_data_container import StockDataContainer
import data_collection.data_utils as dataUtils
import modeling.result_saving_utils as savingUtils
import trading.stock_trader as trader
from config import(
    STOCK_NAMES, PREDICTION_WINDOW, SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS, EARLY_STOP_PATIENCE, SELL_DATES
)

def run_trade_script():
    # Get API keys
    with open('api_keys/alpaca/key.txt') as file:
        KEY = file.read()
    with open('api_keys/alpaca/secret.txt') as file:
        SECRET = file.read()

    # Initialize trading client
    trading_client = TradingClient(KEY, SECRET, paper=True)

    buy_script(trading_client)
    sell_script(trading_client)

def buy_script(trading_client):
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
        if (trader.check_model_performance(testing_labels, predicted_labels) == True):
            predictions = trader.make_next_prediction(stock_data, feature_scaler, label_scaler)
            if (trader.worth_buying(stock_data, predictions)):
                print("Worth buying!!")
                trader.make_trade(stock_data, predictions, trading_client)

def sell_script(trading_client):
    today_date = datetime.now().date()
    for keys in SELL_DATES:
        # Check if there are any stocks that need to be sold
        if SELL_DATES[keys][0] == today_date:
            # Sell the stock
            trader.sellStock(keys, SELL_DATES[keys][1], trading_client, -1)
            # Remove from dictionary so it doesn't get sold again
            del SELL_DATES[keys]



if __name__ == '__main__':
    run_trade_script()



