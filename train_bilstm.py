import tensorflow.keras as keras
from preprocess import generate_training_sequences
from config import OUTPUT_UNITS, LOSS, LEARNING_RATE, EPOCHS, BATCH_SIZE, SEQUENCE_LENGTH
from bilstm_model import build_bilstm_model
import time
from sklearn.model_selection import train_test_split

def train_bilstm(
    output_units=OUTPUT_UNITS,
    loss=LOSS,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    lstm_units=512,
    dropout_rate=0.2,
    dense_units=32
):
    """
    Train the Bi-LSTM model and return training history and metrics.
    """
    # Generate the training sequences
    X, y = generate_training_sequences(sequence_length)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the Bi-LSTM model
    model = build_bilstm_model(
        output_units=output_units,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        dense_units=dense_units
    )

    # Compile the model
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )

    model.summary()

    # Training with timing and metrics
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    training_time = time.time() - start_time

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_val, y_val)
    
    # Save the model
    model.save("bilstm_model.h5")

    return history, training_time, test_loss, test_accuracy

if __name__ == "__main__":
    history, training_time, test_loss, test_accuracy = train_bilstm()
    print(f"\nTraining time: {training_time:.2f} seconds")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}") 