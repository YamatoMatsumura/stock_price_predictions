import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras_tuner as kt

import neural_network as neuralNetwork
import result_saver as saver
import data_utils as dataUtils
from stock_data_container import StockDataContainer
from config import (
    SEQUENCE_LENGTH, STOCK_NAMES, EPOCHS, EARLY_STOP_PATIENCE, 
    BATCH_SIZE, UPDATE_DATA, CREATE_NEW_MODEL, TESTING, 
    LOADING_MODEL, TESTING_CUSTOM_MODEL
)


def main():
    for stock in STOCK_NAMES:
        # Initialize stock data container
        stockData = StockDataContainer(stock)
        if UPDATE_DATA:
            stockData.updateAllData()        
            dataUtils.logData(stockData.data, stockData.ticker)
        else:
            stockData.getExistingData()

        # Create Dataset
        if stockData.data.empty:
            print("DatasetError: no data to create dataset with")
            return
        else:
            dataScaler = dataUtils.getScaler()
            labelScaler = dataUtils.getScaler()
            dataset = dataUtils.createDataset(stockData.data, dataScaler, labelScaler)

        # Split Dataset into training and testing sets
        trainingDataset, testingDataset = dataUtils.splitDataset(dataset)

        # Create new directory to house training session data
        dirPath = saver.createNewDir(stockData.ticker)

        # Initialize early stopping
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
        if CREATE_NEW_MODEL:
            
            # Initialize tuner
            tuner = kt.Hyperband(
                neuralNetwork.getModel(SEQUENCE_LENGTH, stockData.data.shape[1]),
                objective='loss',
                max_epochs=50,
                factor=3,
                directory=dirPath,
                project_name=f"Hyperparam Tuning",

                # ********************************
                # Temporary while adjusting params in tuner. Remove at very end
                overwrite=True
                # ********************************
            )

            # Preform hyperparam tuning
            tuner.search(trainingDataset, epochs=EPOCHS, callbacks=[earlyStopping])

            # Create model based on best hps
            bestHps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(bestHps) 

            # Fit model on training data
            history = model.fit(trainingDataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[earlyStopping])

        elif TESTING:
            # Choose version to load
            version = input("Select tuning session to load: ")

            # Initialize tuner from correct version
            tuner = kt.Hyperband(
                neuralNetwork.getModel(SEQUENCE_LENGTH, stockData.data.shape[1]),
                objective='loss',
                max_epochs=50,
                factor=3,
                directory=f'data/{stockData.ticker}/{str(version)}',
                project_name=f"Hyperparam Tuning",
            )

            # Reload previous tuner hyperparams
            tuner.reload()

            # Create model based on best hps
            bestHps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(bestHps)

            # Fit model on training data
            history = model.fit(trainingDataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[earlyStopping])

        elif LOADING_MODEL:
            
            # Load model
            version = input("Select version to load (0 for old): ")

            if version == 0:
                model = tf.keras.models.load_model(f'data/{stockData.ticker}/model.keras')
            model = tf.keras.models.load_model(f'data/{stockData.ticker}/{str(version)}/model.keras')

            # Train model on new data
            # ****************************************************************
            # Make sure dataTrain and labelTrain only contain new data and not all the data so it doesn't model.fit on all the data again
            #*****************************************************************
            history = model.fit(trainingDataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[earlyStopping])
        
        elif TESTING_CUSTOM_MODEL:
            model = neuralNetwork.getManualModel()

            history = model.fit(trainingDataset, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(testingDataset))

            # Extract the loss values
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']

            # Plot the learning curves
            plt.figure(figsize=(10, 6))
            plt.plot(train_loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Learning Curves')
            plt.legend()
            plt.savefig(f'{dirPath}/loss.png')
            # plt.show()
        

        # Extract data and labels from testing data set
        testingData, testingLabels = dataUtils.extractDataAndLabels(testingDataset)

        # Reduce data dimensions from 4D to 3D since indexing dataset made list versions 4D
        testingData = testingData.reshape(-1, testingData.shape[2], testingData.shape[3])
        # Grab first element in case testing size > 1 since only graphing first set
        testingData = testingData[0]
        # Expand back to 3D since indexing first element made it 2D
        testingData = np.expand_dims(testingData, axis=0)

        # Reduce label dimensions from 3D to 2D since indexing it made it 3D
        testingLabels = testingLabels.reshape(-1, testingLabels.shape[2])
        # Grab first element in case testing size > 1 since only graphing first set
        testingLabels = testingLabels[0]
        # Expand back into 2D since indexing first element made it 1D
        testingLabels = np.expand_dims(testingLabels, axis=0)

        # Make predictions
        predictions = model.predict(testingData)

        # Get best val loss from training history
        valLoss = history.history['loss']
        bestValLoss = min(valLoss)

        # Prepare metadata
        metadata = [bestValLoss]

        # Save results from training session
        saver.saveResults(metadata, labelScaler, predictions, testingLabels, dirPath, model)


if __name__ == '__main__':
    main()