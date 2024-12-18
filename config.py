# main
STOCK_NAMES = ['AAPL', 'AMZN', 'COIN', 'GOOG', 'META', 'MSFT', 'NVDA', 'SPOT']  # List of stock names to look at
EPOCHS = 70  # Epochs to train model for
EARLY_STOP_PATIENCE = 30  # Stops model training if reaches x amount of epochs without improvement
BATCH_SIZE = 1  # Batch size for data set

UPDATE_DATA = False  # Updates training data to pull newest data

CREATE_NEW_MODEL = False  # Does Hyperparm tuning & model.fit
TESTING = False  # Loads previous hyperparm tuning session
LOADING_MODEL = True # Loads previous model
TESTING_CUSTOM_MODEL = False  # Creates model in nn.getManualModel(). No Hyperparm tuning


# data_utils
SEQUENCE_LENGTH = 30  # Sequence length for data set
BATCH_SIZE = 1  # Batch size for data set


# neural_network
N_DAYS = 30  # Controls how many days into the future to predict

