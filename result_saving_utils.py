import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import tensorflow as tf
import yaml

from config import SEQUENCE_LENGTH, N_DAYS, BATCH_SIZE


def createNewDir(ticker):
    # Count number of graphs already existing to not overwrite previous ones
    for root, dirs, files in os.walk(f'data/{ticker}'):
        dirCount = len(dirs)
        # Break after counting dirs in root directory
        break

    # Adjust if directory already exists
    if os.path.exists(f'data/{ticker}/{dirCount}'):
        dirCount += 1
    
    # Create directory to house this training session's data
    customName = input("Custom Name for directory (Enter to skip): ")
    if len(customName) != 0:
        dirPath = f'data/{ticker}/{customName}'
        os.makedirs(dirPath)
    else:
        dirPath = f'data/{ticker}/{dirCount}'
        os.makedirs(dirPath)

    return dirPath

def saveModel(model, dirPath):
    # Save keras model
    model.save(dirPath + '/model.keras')

def createResultsGraph(labelScaler, predictions, testingLabels):

    # Undo scaling on predictions and labels
    predictions = labelScaler.inverse_transform(predictions)
    testingLabels = labelScaler.inverse_transform(testingLabels)

    for i in range(len(testingLabels)):
        # Initialize Graph
        resultsGraph = plt.figure(figsize=(10, 6))

        # Plot actual labels
        plt.plot(testingLabels[i], linestyle='-', linewidth = 0.7, color='b', label='Actual', marker='o', markersize=1)

        # Plot predictions
        plt.plot(predictions[i], linestyle='--', linewidth = 0.7, color='r', label='Predicted', marker='o', markersize=1)

        # Customize plot
        plt.title('Actual vs. Predicted')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        return resultsGraph
    
def saveGraph(graph, dirPath, fileName):
        graph.savefig(f'{dirPath}/{fileName}')

def createLossGraph(history):
    # Extract the loss values
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the learning curves
    lossGraph = plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()

    return lossGraph


def saveTrainingNotes(dirpath, model, bestValLoss, rValue, rmse):
    with open(f'{dirpath}/notes.txt', 'w') as file:
        file.write('='*17 + ' ' + "[Model Summary]" + ' ' + '='*18 + '\n')
        file.write(f"Date: {str(date.today().strftime('%m/%d/%Y'))} \n \n \n")

        file.write('='*16 + ' ' + "[Model Parameters]" + ' ' + '='*16 + '\n')
        file.write("- Sequence Length: " + str(SEQUENCE_LENGTH) + '\n')
        file.write("- Predicting N Days: " + str(N_DAYS) + '\n')
        file.write("- Batch Size: " + str(BATCH_SIZE) + '\n')
        file.write('='*52 + '\n \n')

        file.write('='*15 + ' ' + "[Model Performance]" + ' ' + '='*16 + '\n')
        file.write("- Best Validation Loss: " + str(bestValLoss) + '\n')
        file.write("- R Value: " + str(rValue) + '\n')
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
