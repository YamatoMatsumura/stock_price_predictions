import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import tensorflow as tf
import yaml

from config import SEQUENCE_LENGTH, N_DAYS, BATCH_SIZE


def create_new_dir(ticker):
    # Count number of graphs already existing to not overwrite previous ones
    for root, dirs, files in os.walk(f'data/{ticker}'):
        dir_count = len(dirs)
        # Break after counting dirs in root directory
        break

    # Adjust if directory already exists
    if os.path.exists(f'data/{ticker}/{dir_count}'):
        dir_count += 1
    
    # Create directory to house this training session's data
    custom_name = input("Custom Name for directory (Enter to skip): ")
    if len(custom_name) != 0:
        dir_path = f'data/{ticker}/{custom_name}'
        os.makedirs(dir_path)
    else:
        dir_path = f'data/{ticker}/{dir_count}'
        os.makedirs(dir_path)

    return dir_path

def save_model(model, dir_path):
    # Save keras model
    model.save(dir_path + '/model.keras')

def create_results_graph(label_scaler, predictions, testing_labels):

    # Undo scaling on predictions and labels
    predictions = label_scaler.inverse_transform(predictions)
    testing_labels = label_scaler.inverse_transform(testing_labels)

    for i in range(len(testing_labels)):
        # Initialize Graph
        results_graph = plt.figure(figsize=(10, 6))

        # Plot actual labels
        plt.plot(testing_labels[i], linestyle='-', linewidth = 0.7, color='b', label='Actual', marker='o', markersize=1)

        # Plot predictions
        plt.plot(predictions[i], linestyle='--', linewidth = 0.7, color='r', label='Predicted', marker='o', markersize=1)

        # Customize plot
        plt.title('Actual vs. Predicted')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        return results_graph
    
def save_graph(graph, dir_path, file_name):
        graph.savefig(f'{dir_path}/{file_name}')

def create_loss_graph(history):
    # Extract the loss values
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the learning curves
    loss_graph = plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()

    return loss_graph


def save_training_notes(dir_path, model, best_val_loss, r_value, rmse):
    with open(f'{dir_path}/notes.txt', 'w') as file:
        file.write('='*17 + ' ' + "[Model Summary]" + ' ' + '='*18 + '\n')
        file.write(f"Date: {str(date.today().strftime('%m/%d/%Y'))} \n \n \n")

        file.write('='*16 + ' ' + "[Model Parameters]" + ' ' + '='*16 + '\n')
        file.write("- Sequence Length: " + str(SEQUENCE_LENGTH) + '\n')
        file.write("- Predicting N Days: " + str(N_DAYS) + '\n')
        file.write("- Batch Size: " + str(BATCH_SIZE) + '\n')
        file.write('='*52 + '\n \n')

        file.write('='*15 + ' ' + "[Model Performance]" + ' ' + '='*16 + '\n')
        file.write("- Best Validation Loss: " + str(best_val_loss) + '\n')
        file.write("- R Value: " + str(r_value) + '\n')
        file.write("- RMSE: " + str(rmse) + '\n')
        file.write('='*52 + '\n \n')
        
        # Iterate through the layers and write to file the specific attributes
        file.write('='*15 + ' ' + "[Model Architecture]" + ' ' + '='*15 + '\n')
        for layer in model.layers:
            # Check if the layer is Bidirectional
            if isinstance(layer, tf.keras.layers.Bidirectional):
                # Extract the wrapped LSTM layer
                lstm_layer = layer._layers[0]
                
                if isinstance(lstm_layer, tf.keras.layers.LSTM):
                    file.write("- Bidirectional LSTM Layer\n")
                    file.write(' '*4 + f"* Units: {lstm_layer.units}\n")
                    file.write(' '*4 + f"* Recurrent Dropout: {lstm_layer.recurrent_dropout}\n")
            # Check if the layer is an LSTM layer
            elif isinstance(layer, tf.keras.layers.LSTM):
                file.write("- LSTM Layer\n")
                file.write(' '*4 + f"* Units: {layer.units}\n")
                file.write(' '*4 + f"* Recurrent Dropout: {layer.recurrent_dropout}\n")
            # Check if the layer is a Dense layer
            elif isinstance(layer, tf.keras.layers.Dense):
                file.write("- Output Layer\n")
                file.write(' '*4 + f"* Units: {layer.units}\n")
