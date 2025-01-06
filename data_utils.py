import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from config import SEQUENCE_LENGTH, BATCH_SIZE, PREDICTION_WINDOW


def convert_dataset_to_percentage_changes(stock_data):
    # Day 1 has percentage change of going from day 0 to day 1

    df = stock_data.data.copy()

    # Reverse data since percentage change is calculated from oldest to newest
    df = df.iloc[::-1].reset_index(drop=True)

    for col in df.columns:
        if col != 'Date':  # Date is only for debugging/data alignment purposes
            df[col] = df[col].pct_change()

    # Oldest date has no percentage change
    df = df.iloc[1:]

    # Aroon up/down can have inf/empty values so replace with 0
    df = df.replace([np.inf, -np.inf, np.NAN, ''], 0)

    # Reverse for saving purposes
    df = df.iloc[::-1].reset_index(drop=True)
    df.to_csv('data/' + stock_data.ticker + '/percentage_change_data.csv', index=False)

    return df

def create_windowed_dataset(percentage_data):
    df = percentage_data.copy()
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
    initial_close = data['Close'].iloc[1*PREDICTION_WINDOW]

    predicted_label_values = []
    for prediction in predictions:
        predicted_label_values.append(initial_close * (1 + prediction))
        initial_close = predicted_label_values[-1]

    return predicted_label_values

def scale_dataset(training_features, training_labels, testing_features, testing_labels):
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


def log_data(new_data, ticker):
    # Check if existing data in trained_data
    try:
        # Load and add onto data
        existing_data = pd.read_csv('data/' + ticker + '/trained_data.csv')
        data = pd.concat([existing_data, new_data], ignore_index=True)

        # Merge duplicate date days and fill in data by cross referencing rows
        data = data.groupby('Date').apply(lambda x: x.ffill().bfill().iloc[0]).reset_index(drop=True)

        # Get rid of all rows with missing data
        data = data.replace('', np.nan).dropna(axis=0, how='any').reset_index(drop=True)

        # Make sure no duplicates
        data.drop_duplicates(inplace=True)       

        # Make sure dates are in right order
        data = data.sort_values(by='Date', ascending=False).reset_index(drop=True)

        # save to csv
        data.to_csv('data/' + ticker + '/trained_data.csv', index=False) 

    # If no pre-existing data  
    except:
        # Merge duplicate date days and fill in data by cross referencing rows
        data = data.groupby('Date').apply(lambda x: x.ffill().bfill().iloc[0]).reset_index(drop=True)

        # Get rid of all rows with missing data
        data = data.replace('', np.nan).dropna(axis=0, how='any').reset_index(drop=True)

        # Make sure dates are in right order
        data = data.sort_values(by='Date', ascending=False).reset_index(drop=True)

        #  Save everything to csv
        data.to_csv('data/' + ticker + '/trained_data.csv', index=False)
