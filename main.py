import tensorflow as tf
import keras_tuner as kt



import trading.stock_trader as trader
import modeling.neural_network as neuralNetwork
import modeling.result_saving_utils as savingUtils
import data_collection.data_utils as dataUtils
from data_collection.stock_data_container import StockDataContainer
from config import (
    SEQUENCE_LENGTH, STOCK_NAMES, EPOCHS, EARLY_STOP_PATIENCE, 
    BATCH_SIZE, UPDATE_DATA, CREATE_NEW_MODEL, 
    LOADING_MODEL, TESTING_CUSTOM_MODEL, BACKUP_DATA, PREDICTION_WINDOW
)

def main():
    if STOCK_NAMES[0] == 'ALL':
        dataUtils.get_all_tickers()

    for stock in STOCK_NAMES:

        if BACKUP_DATA:
            dataUtils.backup_data(stock)

        stock_data = StockDataContainer(stock)
        if UPDATE_DATA:
            '''
            Turning off for now while experimenting with getting data. Turn back on when everything is set since in theory should be 
            backing up data every time before pulling new data
            '''
            # Backup pre-existing data incase something goes wrong fetching new data
            # dataUtils.backup_data(stock_data.ticker)

            stock_data.update_all_data()

            # Combine with pre-existing saved data if applicable    
            stock_data.training_data = dataUtils.merge_with_old_data(stock_data)

            dataUtils.save_data(stock_data.training_data, stock_data.ticker)
        else:
            stock_data.get_existing_data()
        
    
        # Reverse data so model trains from oldest to newest
        modified_training_data = stock_data.training_data.iloc[::-1].reset_index(drop=True)
        # Drop Date column since only used for debugging/data alignment purposes
        modified_training_data = modified_training_data.drop(columns=['date'])

        features, labels = dataUtils.create_windowed_dataset(modified_training_data)

        training_features, training_labels, testing_features, testing_labels = dataUtils.split_dataset(features, labels)


        (
            training_features, 
            training_labels, 
            testing_features, 
            testing_labels, 
            feature_scaler, 
            label_scaler,
        ) = dataUtils.scale_dataset(training_features, training_labels, testing_features, testing_labels)


        # Create new directory to house training session data
        dir_path = savingUtils.create_new_dir(stock_data.ticker)

        # Initialize early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)

        if CREATE_NEW_MODEL:

            # Initialize tuner
            tuner = kt.BayesianOptimization(
                lambda hp: neuralNetwork.get_model(hp, SEQUENCE_LENGTH, stock_data.training_data.shape[1]),
                objective='loss',
                max_trials=30,
                directory=dir_path,
                project_name=f"Hyperparam Tuning",

                # ********************************
                # Temporary while adjusting params in tuner. Remove at very end
                overwrite=True
                # ********************************
            )

            # Preform hyperparam tuning
            tuner.search(training_features, training_labels, epochs=EPOCHS, callbacks=[early_stopping])

            # Create model based on best hps
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(best_hps) 

            # Fit model on training data
            history = model.fit(training_features, training_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping])

            # Save training history
            savingUtils.save_training_history(history, stock_data.ticker)

        elif LOADING_MODEL:

            # Load model
            version = input("Select version to load (0 for old): ")

            if version == str(0):
                model = tf.keras.models.load_model(f'data/{stock_data.ticker}/model.keras')
            else:
                model = tf.keras.models.load_model(f'data/{stock_data.ticker}/{str(version)}/model.keras')

            # Train model on new data
            # ****************************************************************
            # Make sure dataTrain and labelTrain only contain new data and not all the data so it doesn't model.fit on all the data again
            #*****************************************************************
            # history = model.fit(trainingDataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[earlyStopping])
        
        elif TESTING_CUSTOM_MODEL:
            model = neuralNetwork.get_manual_model()

            history = model.fit(training_features, training_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping])

            savingUtils.save_training_history(history, stock_data.ticker)


        # Make predictions
        scaled_predictions = model.predict(testing_features)
        predicted_labels = label_scaler.inverse_transform(scaled_predictions)

        # Save graph of predicted vs actual
        results = savingUtils.create_results_graph(predicted_labels, stock_data)
        savingUtils.save_graph(results, dir_path, "results.png")

        # Save notes for training session
        savingUtils.save_training_notes(dir_path, model, history, predicted_labels, stock_data.training_data)

        # Save model
        savingUtils.save_model(model, dir_path)


if __name__ == '__main__':
    main()