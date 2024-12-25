import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.metrics import mean_squared_error
import pandas as pd


import neural_network as neuralNetwork
import result_saving_utils as savingUtils
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
        stock_data = StockDataContainer(stock)
        if UPDATE_DATA:
            stock_data.update_all_data()        
            dataUtils.log_data(stock_data.data, stock_data.ticker)
        else:
            stock_data.get_existing_data()


        # Create Dataset
        if stock_data.data.empty:
            print("DatasetError: no data to create dataset with")
            return
        else:
            dataset = dataUtils.create_dataset(stock_data.data)

        # Split Dataset into training and testing sets
        training_dataset, testing_dataset = dataUtils.split_dataset(dataset)

        data_scalers, label_scalers = dataUtils.scale_training_dataset(training_dataset)


        # Create new directory to house training session data
        dir_path = savingUtils.create_new_dir(stock_data.ticker)

        # Initialize early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)

        if CREATE_NEW_MODEL:
            
            # Initialize tuner
            tuner = kt.Hyperband(
                neuralNetwork.get_model(SEQUENCE_LENGTH, stock_data.data.shape[1]),
                objective='loss',
                max_epochs=50,
                factor=3,
                directory=dir_path,
                project_name=f"Hyperparam Tuning",

                # ********************************
                # Temporary while adjusting params in tuner. Remove at very end
                overwrite=True
                # ********************************
            )

            # Preform hyperparam tuning
            tuner.search(training_dataset, epochs=EPOCHS, callbacks=[early_stopping])

            # Create model based on best hps
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(best_hps) 

            # Fit model on training data
            history = model.fit(training_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping])

        elif TESTING:

            # Choose version to load
            version = input("Select tuning session to load: ")

            # Initialize tuner from correct version
            tuner = kt.Hyperband(
                neuralNetwork.get_model(SEQUENCE_LENGTH, stock_data.data.shape[1]),
                objective='loss',
                max_epochs=50,
                factor=3,
                directory=f'data/{stock_data.ticker}/{str(version)}',
                project_name=f"Hyperparam Tuning",
            )

            # Reload previous tuner hyperparams
            tuner.reload()

            # Create model based on best hps
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(best_hps)

            # Fit model on training data
            history = model.fit(training_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping])

        elif LOADING_MODEL:
            
            # Load model
            version = input("Select version to load (0 for old): ")

            if version == 0:
                model = tf.keras.models.load_model(f'data/{stock_data.ticker}/model.keras')
            model = tf.keras.models.load_model(f'data/{stock_data.ticker}/{str(version)}/model.keras')

            # Train model on new data
            # ****************************************************************
            # Make sure dataTrain and labelTrain only contain new data and not all the data so it doesn't model.fit on all the data again
            #*****************************************************************
            # history = model.fit(trainingDataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[earlyStopping])
        
        elif TESTING_CUSTOM_MODEL:
            model = neuralNetwork.get_manual_model()

            history = model.fit(training_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping])


        # Extract data and labels from testing data set
        testing_data, testing_labels = dataUtils.extract_data_and_labels(testing_dataset)

        # Reduce data dimensions from 4D to 3D since indexing dataset made list versions 4D
        testing_data = testing_data.reshape(-1, testing_data.shape[2], testing_data.shape[3])
        # Grab first element in case testing size > 1 since only graphing first set
        testing_data = testing_data[0]
        # Expand back to 3D since indexing first element made it 2D
        testing_data = np.expand_dims(testing_data, axis=0)

        # Reduce label dimensions from 3D to 2D since indexing it made it 3D
        testing_labels = testing_labels.reshape(-1, testing_labels.shape[2])
        # Grab first element in case testing size > 1 since only graphing first set
        testing_labels = testing_labels[0]
        # Expand back into 2D since indexing first element made it 1D
        testing_labels = np.expand_dims(testing_labels, axis=0)

        # Make predictions
        predictions = model.predict(testing_data)

        # Save graph of predicted vs actual
        results = savingUtils.create_results_graph(label_scalers, predictions, testing_labels)
        savingUtils.save_graph(results, dir_path, "results.png")

        # Save notes for training session
        val_loss = history.history['loss']
        best_val_loss = min(val_loss)
        r_value = np.corrcoef(testing_labels, predictions)[0, 1]
        rmse = np.sqrt(mean_squared_error(testing_labels[0], predictions[0]))
        savingUtils.save_training_notes(dir_path, model, best_val_loss, r_value, rmse)

        # Save model
        savingUtils.save_model(model, dir_path)


if __name__ == '__main__':
    main()