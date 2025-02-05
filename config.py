# main
STOCK_NAMES = ['AAPL']  # List of stock names to look at
EPOCHS = 70  # Epochs to train model for
EARLY_STOP_PATIENCE = 5  # Stops model training if reaches x amount of epochs without improvement

UPDATE_DATA = False  # Updates training data to pull newest data
BACKUP_DATA = False # Copies current "trained_data.csv" to "trained_data_backup.csv"

CREATE_NEW_MODEL = False  # Does Hyperparm tuning & model.fit
TESTING = False  # Loads previous hyperparm tuning session
LOADING_MODEL = False # Loads previous model
TESTING_CUSTOM_MODEL = True  # Creates model in nn.getManualModel(). No Hyperparm tuning

PERCENTAGE_DATA = False
RAW_DATA = False
COMBINED_DATA = True


# data_utils
SEQUENCE_LENGTH = 50  # Sequence length for data set
BATCH_SIZE = 2  # Batch size for data set


# neural_network
PREDICTION_WINDOW = 20 # Controls how many days into the future to predict

