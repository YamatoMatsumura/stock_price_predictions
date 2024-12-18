import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from config import SEQUENCE_LENGTH, BATCH_SIZE, N_DAYS


def createDataset(data, dataScaler, labelScaler):

    # Reverse data so trains from oldest to newest
    data = data.iloc[::-1]

    # Delete Date Column since can't go into model
    data = data.drop(columns=['Date'])

    # Convert labels and data to numpy array for model
    labels = data.pop('Close').values
    data = data.values

    # Scale data to help with fitting
    data = dataScaler.fit_transform(data)

    # Reshape labels into 2D array since StandardScalar needs 2D array
    labels = labels.reshape(-1, 1)
    labels = labelScaler.fit_transform(labels)
    # Reshape labels back into 1D array
    labels = labels.flatten()


    # Create overlapping dataset
    # Remove first sequenceLength labels since predicting future closing price (i.e. day 1 of data has day N_DAYS+1 closing price)
    labels = labels[SEQUENCE_LENGTH:]
    windowedLabels = []
    for i in range(len(labels) - N_DAYS + 1):
        windowedLabels.append(labels[i:i+N_DAYS])
    
    windowedLabels = np.array(windowedLabels)

    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data,
        windowedLabels,
        SEQUENCE_LENGTH,
        batch_size = BATCH_SIZE,
    )


    return dataset

def getNumFeatures(data):
    return data.shape[1]

def getScaler():
    return MinMaxScaler()

def logData(data, ticker):
    # Check if existing data in trained_data
    try:
        # Load and add onto data
        df = pd.read_csv('data/' + ticker + '/trained_data.csv')
        data = pd.concat([df, data], ignore_index=True)

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