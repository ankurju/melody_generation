import tensorflow.keras as keras

def build_lstm_model(
    output_units,
    lstm_units=256,
    dropout_rate=0.2,
    dense_units=32
):
    """
    Build and return an LSTM model for melody generation.
    """
    inputs = keras.layers.Input(shape=(None, output_units))
    
    # First LSTM layer
    x = keras.layers.LSTM(lstm_units)(inputs)
    x = keras.layers.Dropout(dropout_rate)(x)
    
    # Dense layers
    for _ in range(2):  # Two hidden layers
        x = keras.layers.Dense(dense_units, activation='relu')(x)
    
    outputs = keras.layers.Dense(output_units, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    return model 
