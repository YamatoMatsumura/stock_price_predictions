import tensorflow as tf

from config import N_DAYS
    
def getModel(hp, sequenceLength, numFeatures):
    model = tf.keras.Sequential()

    # Layer 1
    hpUnits1 = hp.Choice('units: 1', values=[4,8,16])
    dropout1 = hp.Choice('Dropout: 1', values=[0.0,0.2])
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=hpUnits1,
        return_sequences=True,
        input_shape=(sequenceLength, numFeatures),
        recurrent_dropout=dropout1
    )))

    # Layer 2
    skipBidirectional = hp.Boolean('Skip optional Bidirectional')
    if not skipBidirectional:
        hpUnits2 = hp.Choice('units: 2', values=[4,8,16])
        dropout2 = hp.Choice('Dropout: 2', values=[0.0, 0.2])
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=hpUnits2,
            return_sequences=True,
            recurrent_dropout=dropout2
        )))

    # Layer 3
    skipRegular = hp.Boolean('Skip optional regular')
    hpUnits3 = hp.Choice('units: 3', values=[4,8])
    dropout3 = hp.Choice('Dropout: 3', values=[0.0, 0.2])
    if not skipRegular:
        model.add(tf.keras.layers.LSTM(
            units=hpUnits3,
            return_sequences=True,
            recurrent_dropout=dropout3
        ))

    # Layer 4
    hpUnits4 = hp.Choice('units: 4', values=[4,8])
    dropout4 = hp.Choice('Dropout: 4', values=[0.0, 0.2])
    model.add(tf.keras.layers.LSTM(
        units=hpUnits4,
        recurrent_dropout=dropout4
    ))

    # Output Layer
    model.add(tf.keras.layers.Dense(N_DAYS))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_absolute_error"]
    )

    return model

def getManualModel(self):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=16, return_sequences=True, recurrent_dropout=0.2)))
    model.add(tf.keras.layers.LSTM(units=8, recurrent_dropout=0.2))
    model.add(tf.keras.layers.Dense(N_DAYS))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_absolute_error"]
    )

    return model
