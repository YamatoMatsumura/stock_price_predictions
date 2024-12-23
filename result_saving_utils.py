import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import date
import tensorflow as tf

from config import SEQUENCE_LENGTH, N_DAYS


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
    customName = input("Custom Name for directory (Enter n to skip): ")
    if customName != "n":
        dirPath = f'data/{ticker}/{dirCount}_{customName}'
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
        rmse = np.sqrt(mean_squared_error(testingLabels[i], predictions[i]))
        print(rmse)

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


def saveTrainingNotes(metadata, dirpath, model):
    bestValLoss = metadata[0]

    # Initialize metadata
    metadata = {
        'Date: ': str(date.today().strftime('%m/%d/%Y')),
        'Best Val Loss: ': str(bestValLoss),
        'Sequence Length: ': SEQUENCE_LENGTH,
        'Predicting N Days: ': N_DAYS
    }

    # Write metadata to seperate file
    with open(f'{dirpath}/notes.txt', 'w') as file:
        for key, value in metadata.items():
            file.write(f'{key}: {value} \n')
        
        file.write('\n' + '-'*40)
        file.write('Model Summary')
        file.write('-'*40 + '\n')
        
        # Iterate through the layers and write to file the specific attributes
        for layer in model.layers:
            # Check if the layer is Bidirectional
            if isinstance(layer, tf.keras.layers.Bidirectional):
                # Extract the wrapped LSTM layer
                lstm_layer = layer._layers[0]
                
                if isinstance(lstm_layer, tf.keras.layers.LSTM):
                    file.write('Bidirectional LSTM Layer\n')
                    file.write(f'Units: {lstm_layer.units}\n')
                    file.write(f'Recurrent Dropout: {lstm_layer.recurrent_dropout}\n')
                    file.write('-' * 40 + '\n')
            # Check if the layer is an LSTM layer
            elif isinstance(layer, tf.keras.layers.LSTM):
                file.write('LSTM Layer\n')
                file.write(f'Units: {layer.units}\n')
                file.write(f'Recurrent Dropout: {layer.recurrent_dropout}\n')
                file.write('-' * 40 + '\n')
            # Check if the layer is a Dense layer
            elif isinstance(layer, tf.keras.layers.Dense):
                file.write(f'Output Layer\n')
                file.write(f'Units: {layer.units}\n')
                file.write('-' * 40 + '\n')