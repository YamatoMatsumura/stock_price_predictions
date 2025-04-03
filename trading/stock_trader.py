import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error

import trading.stock_trading_utils as trade_utils
from config import PREDICTION_WINDOW

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
    Arbitrary cut off point. Change later
    '''
    if (max_value > (close_df[0] + 5)):
        return True
    return False


def make_trade(stock_data, predictions, trading_client):
    close_df = stock_data.training_data.copy()['Close']
    close_df.iloc[::-1].reset_index(drop=True)

    max_value = max(predictions)
    max_index = predictions.index(max_value)



    # Make trade
    trade_utils.buyStock(stock_data.ticker, 5, trading_client, max_index)

    return