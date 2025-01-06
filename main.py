import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras_tuner as kt
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
        stock_data = StockDataContainer(stock)
        if UPDATE_DATA:
            stock_data.update_all_data()        
            dataUtils.log_data(stock_data.data, stock_data.ticker)
        else:
            stock_data.get_existing_data()

        if stock_data.data.empty:
            print("DatasetError: no data to create dataset with")
            return


        # Convert dataset to percent changes since predicting stock movement/changes
        stock_data.percentage_change_data = dataUtils.convert_dataset_to_percentage_changes(stock_data)

        # Reverse data so model trains from oldest to newest
        modified_percentage_data = stock_data.percentage_change_data.iloc[::-1].reset_index(drop=True)
        # Drop Date column since only used for debugging/data alignment purposes
        modified_percentage_data = modified_percentage_data.drop(columns=['Date'])


        features, labels = dataUtils.create_windowed_dataset(modified_percentage_data)
        training_features, training_labels, testing_features, testing_labels = dataUtils.split_dataset(features, labels)
        (   
            scaled_training_features, 
            scaled_training_labels, 
            scaled_testing_features, 
            scaled_testing_labels, 
            feature_scaler, 
            label_scaler 
        ) = dataUtils.scale_dataset(training_features, training_labels, testing_features, testing_labels)

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
            tuner.search(scaled_training_features, scaled_training_labels, epochs=EPOCHS, callbacks=[early_stopping])

            # Create model based on best hps
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(best_hps) 

            # Fit model on training data
            history = model.fit(scaled_training_features, scaled_training_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping])

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
            history = model.fit(scaled_training_features, scaled_training_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping])

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

            history = model.fit(scaled_training_features, scaled_training_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping])

        # Make predictions
        scaled_predictions = model.predict(scaled_testing_features)
        predictions = label_scaler.inverse_transform(scaled_predictions)
        predicted_labels = dataUtils.reverse_percentages(predictions, stock_data.data)

        # Save graph of predicted vs actual
        results = savingUtils.create_results_graph(predicted_labels, stock_data)
        savingUtils.save_graph(results, dir_path, "results.png")

        # Save notes for training session
        savingUtils.save_training_notes(dir_path, model, history, predicted_labels, stock_data.data)

        # Save model
        savingUtils.save_model(model, dir_path)


if __name__ == '__main__':
    main()