import tensorflow as tf

from config import PREDICTION_WINDOW
    
def get_model(hp, sequence_length, num_features):
    """Defines and returns the model used for hyperparameter tuning

    Args:
        hp (HyperparameterTuner): An object responsible for hyperparameter tuning
        sequence_length (int): The sequence length specified
        num_features (int): The number of features in the dataset

    Returns:
        tf.keras.Sequential: A Sequential model that can be used for hyperparameter tuning
    """

    model = tf.keras.Sequential()

    # Layer 1
    hp_units1 = hp.Choice('units: 1', values=[4,8,16])
    dropout1 = hp.Choice('Dropout: 1', values=[0.0,0.2])
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=hp_units1,
        return_sequences=True,
        input_shape=(sequence_length, num_features),
        recurrent_dropout=dropout1
    )))

    # Layer 2
    skip_bidirectional = hp.Boolean('Skip optional Bidirectional')
    if not skip_bidirectional:
        hp_units2 = hp.Choice('units: 2', values=[4,8,16])
        dropout2 = hp.Choice('Dropout: 2', values=[0.0, 0.2])
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=hp_units2,
            return_sequences=True,
            recurrent_dropout=dropout2
        )))

    # Layer 3
    skip_regular = hp.Boolean('Skip optional regular')
    hp_units3 = hp.Choice('units: 3', values=[4,8])
    dropout3 = hp.Choice('Dropout: 3', values=[0.0, 0.2])
    if not skip_regular:
        model.add(tf.keras.layers.LSTM(
            units=hp_units3,
            return_sequences=True,
            recurrent_dropout=dropout3
        ))

    # Layer 4
    hp_units4 = hp.Choice('units: 4', values=[4,8])
    dropout4 = hp.Choice('Dropout: 4', values=[0.0, 0.2])
    model.add(tf.keras.layers.LSTM(
        units=hp_units4,
        recurrent_dropout=dropout4
    ))

    # Output Layer
    model.add(tf.keras.layers.Dense(PREDICTION_WINDOW))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_absolute_error"]
    )

    return model

def get_manual_model():
    """Defines and returns a fully contructed model without any hyperparameters.

    This model is intended for testing different layer configurations and parameters
    without the influence of hyperparameter tuning.

    Returns:
        tf.keras.Sequential(): A Sequential model that can be used for training on the data
    """

    model = tf.keras.Sequential()

    # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True, recurrent_dropout=0.2)))
    # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True, recurrent_dropout=0.2)))
    model.add(tf.keras.layers.LSTM(units=16, recurrent_dropout=0.2, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=8, recurrent_dropout=0.2))
    # model.add(tf.keras.layers.LSTM(units=4, recurrent_dropout=0.2))

    model.add(tf.keras.layers.Dense(PREDICTION_WINDOW))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_absolute_error"]
    )

    return model
