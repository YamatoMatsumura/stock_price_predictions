import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from config import SEQUENCE_LENGTH, BATCH_SIZE, N_DAYS


def convert_dataset_to_percentage_changes(stock_data):
    df = stock_data.data

    for col in df.columns:
        if col != 'Date':
            df[col] = df[col].pct_change()

    # Drop first row since no percent change
    df = df.iloc[1:]

    return df

def create_dataset(data):

    # Reverse data so trains from oldest to newest
    data = data.iloc[::-1]

    # Delete Date Column since can't go into model
    data = data.drop(columns=['Date'])

    # Convert labels and data to numpy array for model
    labels = data.pop('Close').values
    data = data.values

    # # Scale data to help with fitting
    # data = dataScaler.fit_transform(data)

    # # Reshape labels into 2D array since StandardScalar needs 2D array
    # labels = labels.reshape(-1, 1)
    # labels = labelScaler.fit_transform(labels)
    # # Reshape labels back into 1D array
    # labels = labels.flatten()


    # Create windowed dataset
        # ex: Data for days 1-10 predict labels for day 11-20
        # Next element is Data for days 2-11 predict labels for day 12-21
    # Remove first sequenceLength labels since predicting future closing price (i.e. day 1 of data has day N_DAYS+1 closing price)
    labels = labels[SEQUENCE_LENGTH:]
    windowed_labels = []
    for i in range(len(labels) - N_DAYS + 1):
        windowed_labels.append(labels[i:i+N_DAYS])
    
    windowed_labels = np.array(windowed_labels)

    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data,
        windowed_labels,
        SEQUENCE_LENGTH,
        batch_size = BATCH_SIZE,
    )

    return dataset

def split_dataset(dataset):
    # Convert to list to help split into training and testing
    dataset = list(dataset)

    # Grab last element for testing
    testing_dataset = dataset[-1:]

    # Convert into np array since dataset is tuples of data, labels
    testing_data = np.array([x[0].numpy() for x in testing_dataset])
    testing_labels = np.array([x[1].numpy() for x in testing_dataset])

    # Extract remaining data to use as training and split into data and labels
    training_dataset = dataset[:-1 * SEQUENCE_LENGTH]  # Space out sequence_length amount to make sure no part of testing dataset is in training
    training_data = np.array([x[0].numpy() for x in training_dataset])
    training_labels = np.array([x[1].numpy() for x in training_dataset])

    # Convert to tf.data.Dataset to feed into neural network
    training_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_labels))
    testing_dataset = tf.data.Dataset.from_tensor_slices((testing_data, testing_labels))

    return training_dataset, testing_dataset

def extract_data_and_labels(dataset):
    data = []
    labels = []

    for d, l in dataset:
        data.append(d.numpy())
        labels.append(l.numpy())
    
    data = np.array(data)
    labels = np.array(labels)
    
    return data, labels


def scale_training_dataset(dataset):
    data, labels = extract_data_and_labels(dataset)
    data = data.reshape(-1, data.shape[3])
    labels = labels.reshape(-1)


    data_scalers = []
    for i in range(data.shape[1]):
        scaler = MinMaxScaler()
        data[i] = scaler.fit_transform(data[i])
        data_scalers.append(scaler)

    label_scalers = []
    for i in range(labels.shape[1]):
        scaler = MinMaxScaler()
        labels[i] = scaler.fit_transform(labels[i])
        label_scalers.append(scaler)

    return data_scalers, label_scalers

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
