import tensorflow.keras as keras

def build_bilstm_model(
    output_units,
    lstm_units=512,  
    dropout_rate=0.2,
    dense_units=32
):
    """
    Build and return a Bidirectional LSTM model for melody generation.
    """
    inputs = keras.layers.Input(shape=(None, output_units))
    
    # First Bidirectional LSTM layer
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(lstm_units//2, return_sequences=True)
    )(inputs)
    x = keras.layers.Dropout(dropout_rate)(x)
    
    # Second Bidirectional LSTM layer
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(lstm_units//2)
    )(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    
    # Dense layers -->> Two hidden layers
    for _ in range(2):  
        x = keras.layers.Dense(dense_units, activation='relu')(x)
    
    outputs = keras.layers.Dense(output_units, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    return model 