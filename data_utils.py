import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os

from config import SEQUENCE_LENGTH, BATCH_SIZE, PREDICTION_WINDOW, STOCK_NAMES


def add_percentage_changes_to_dataset(stock_data):
    """Adds percentage changes column to the dataset.

    The new column is named "% + original_column_name".
    The percentage is calculated such that the value at Day 1 represents
    the percentage change from Day 0 to Day 1.
        
    Args:
        stock_data (StockDataContainer): The stock data container

    Returns:
        DataFrame: A DataFrame containing all stock data with the new percentage
        change columns. Data is sorted from newest to oldest.
    """

    df = stock_data.raw_data.copy()

    # Reverse data since percentage change is calculated from oldest to newest
    df = df.iloc[::-1].reset_index(drop=True)

    for col in df.columns:
        if col != 'Date':  # Date is only for debugging/data alignment purposes
            df[col] = df[col].pct_change()
            df.rename(columns={col: '%' + col}, inplace=True)
    # Oldest date has no percentage change
    df = df.iloc[1:]

    # Aroon up/down can have inf/empty values so replace with 0
    df = df.replace([np.inf, -np.inf, np.NAN, ''], 0)

    # Reverse for saving purposes
    df = df.iloc[::-1].reset_index(drop=True)

    return df

def combine_datasets(left, right):
    """Merges two datasets on the date column.

    Assumes that the only common column between the datasets is the "date".

    Args:
        left (DataFrame): The first DataFrame
        right (DataFrame): The second DataFrame

    Returns:
        DataFrame: A merged DataFrame sorted from newest to oldest
    """

    combined_df = pd.merge(left, right, on = 'Date', how='outer')
    combined_df = combined_df.replace('', np.nan).dropna(axis=0, how='any').reset_index(drop=True)
    combined_df = combined_df.iloc[::-1].reset_index(drop=True)
    return combined_df

def create_windowed_dataset(training_data):
    """Transforms the dataset into overlapping windows of data

    Args:
        training_data (DataFrame): A DataFrame containing all stock data

    Returns:
        list: A lists of windowed feature sets
            - Shape: (num_windows, SEQUENCE_LENGTH, num_features)
        list: A lists of windows labels
            - Shape: (num_windows, PREDICTION_WINDOW)
    """

    df = training_data.copy()
    labels = df.pop('Close').values
    features = df.values

    # Create windowed labels
    labels = labels[SEQUENCE_LENGTH:]  # First SEQUENCE_LENGTH labels not used
    windowed_labels = []
    for i in range(len(labels) - PREDICTION_WINDOW + 1):
        windowed_labels.append(labels[i:i+PREDICTION_WINDOW])
    windowed_labels = np.array(windowed_labels)

    # Create windowed features
    features = features[:-PREDICTION_WINDOW]  # Last PREDICTION_WINDOW features not used
    windowed_features = []
    for i in range(len(features) - SEQUENCE_LENGTH + 1):
        windowed_features.append(features[i:i+SEQUENCE_LENGTH])
    windowed_features = np.array(windowed_features)
    
    return windowed_features, windowed_labels

def split_dataset(features, labels):
    """Splits the dataset into training and testing sets.

    The testing data consists of the last windowed features and labels, while
    the remaining data is used for training.

    Args:
        features (list): A list of windowed feature sets
        labels (list): A list of windowed labels

    Returns:
        list: Training feature set
        list: Training labels
        list: Testing feature set (last windowed features)
        list: Testing labels (last windowed labels)
    """

    if SEQUENCE_LENGTH > PREDICTION_WINDOW:
        greatest_window = SEQUENCE_LENGTH
    else:
        greatest_window = PREDICTION_WINDOW

    # Grab just last element since only testing one PREDICTION_WINDOW
    testing_features = features[-1:]
    testing_labels = labels[-1:]
    # Space out greatest_window amount to make sure no part of testing dataset is in training
    training_features = features[:-1 * greatest_window] 
    training_labels = labels[:-1 * greatest_window]

    return training_features, training_labels, testing_features, testing_labels

def reverse_percentages(predictions, data):
    """Converts percentage change predictions into closing price predictions

    Args:
        predictions (list): A list of predicted percentage changes
        data (DataFrame): A DataFrame containing all stock data

    Returns:
        list: A list of predicted closing prices
    """

    initial_close = data['Close'].iloc[1*PREDICTION_WINDOW]

    predicted_label_values = []
    for prediction in predictions:
        predicted_label_values.append(initial_close * (1 + prediction))
        initial_close = predicted_label_values[-1]

    return predicted_label_values

def scale_dataset(training_features, training_labels, testing_features, testing_labels):
    """Applies the StandardScaler to normalize labels and features.

    Uses sklern's StandardScaler to scale features and labels based on training data.

    Args:
        training_features (list): A list of training feature sets
        training_labels (list): A list of training labels
        testing_features (list): A list of testing feature sets
        testing_labels (list): A list of testing labels

    Returns:
        list: Scaled training features
        list: Scaled training labels
        list: Scaled testing features
        list: Scaled testing labels
        StandardScaler: Scaler fitted on training features
        StandardScaler: Scaler fitting on training labels
    """

    # Reshape features from 3D to 2D since StandardScaler needs 2D array
    training_features = training_features.reshape(-1, training_features.shape[2])
    testing_features = testing_features.reshape(-1, testing_features.shape[2])

    feature_scaler = StandardScaler()
    training_features = feature_scaler.fit_transform(training_features)
    testing_features = feature_scaler.transform(testing_features)

    # Reshape back into 3D
    training_features = training_features.reshape(-1, SEQUENCE_LENGTH, training_features.shape[1])
    testing_features = testing_features.reshape(-1, SEQUENCE_LENGTH, testing_features.shape[1])

    label_scaler = StandardScaler()
    training_labels = label_scaler.fit_transform(training_labels)
    testing_labels = label_scaler.transform(testing_labels)

    return training_features, training_labels, testing_features, testing_labels, feature_scaler, label_scaler

def combine_data(stock_data):
    """Combines pre-existing data from "trained_data.csv" with new stock data

    Args:
        stock_data (StockDataContainer): The stock data container

    Returns:
        DataFrame: A DataFrame containing both the pre-existing and new data
    """

    ticker = stock_data.ticker
    new_data = stock_data.data

    # Check if existing data in trained_data
    try:
        # Load and add onto data
        existing_data = pd.read_csv('data/' + ticker + '/trained_data.csv')
  
        # Make sure new data has same columns as existing data
        if ('Sentiment (Avg)' not in existing_data.columns) and ('Sentiment (Avg)' in new_data.columns):
            new_data = new_data.drop(columns=['Sentiment (Avg)'])

        data = pd.concat([existing_data, new_data], ignore_index=True)

        # Merge duplicate date days and fill in data by cross referencing rows
        data = data.groupby('Date').apply(lambda x: x.ffill().bfill().iloc[0]).infer_objects(copy=False).reset_index(drop=True)

        # Get rid of all rows with missing data
        data = data.replace('', np.nan).dropna(axis=0, how='any').reset_index(drop=True)

        # Make sure no duplicates
        data.drop_duplicates(inplace=True)       

        # Make sure dates are in right order
        data = data.sort_values(by='Date', ascending=False).reset_index(drop=True)

        return data

    # If no pre-existing data  
    except:
        # Merge duplicate date days and fill in data by cross referencing rows
        new_data = new_data.groupby('Date').apply(lambda x: x.ffill().bfill().iloc[0]).infer_objects(copy=False).reset_index(drop=True)

        # Get rid of all rows with missing data
        new_data = new_data.replace('', np.nan).dropna(axis=0, how='any').reset_index(drop=True)

        # Make sure dates are in right order
        new_data = new_data.sort_values(by='Date', ascending=False).reset_index(drop=True)

        return new_data


def save_data(data, ticker):
    """Saves the provided stock data to "training_data.csv"

    Args:
        data (DataFrame): A DataFrame with stock data
        ticker (string): The stock ticker associated with the data
    """

    data.to_csv('data/' + ticker + '/trained_data.csv', index=False)

def backup_data(ticker):
    """Copies everything from "training_data.csv" to "trained_data_backup.csv".

    Should always be used before fetching new data to prevent data loss in case of an error.

    Args:
        ticker (string): The stock ticker associated with the data
    """

    if not os.path.exists(f'data/{ticker}/backup'):
        os.makedirs(f'data/{ticker}/backup')

    data = pd.read_csv(f'data/{ticker}/trained_data.csv')
    data.to_csv(f'data/{ticker}/backup/trained_data_backup.csv', index=False)

def get_all_tickers():
    """Allows STOCK_NAMES to be set to ['All'] to fetch all stock tickers in the data/ directory
    """

    STOCK_NAMES.clear()
    for dirs in os.listdir('data/'):
        STOCK_NAMES.append(dirs)