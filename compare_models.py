import matplotlib.pyplot as plt
from train_lstm import train_lstm
from train_bilstm import train_bilstm
from melody_analysis import analyze_melody_novelty
from config import OUTPUT_UNITS, LOSS, LEARNING_RATE, EPOCHS, BATCH_SIZE, SEQUENCE_LENGTH
from melody_generator import MelodyGenerator
import numpy as np

def compare_models(training_dataset_path):
    # Train LSTM model
    print("Training LSTM model...")
    lstm_history, lstm_time, lstm_loss, lstm_accuracy = train_lstm(
        output_units=OUTPUT_UNITS,
        loss=LOSS,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # Train Bi-LSTM model
    print("\nTraining Bi-LSTM model...")
    bilstm_history, bilstm_time, bilstm_loss, bilstm_accuracy = train_bilstm(
        output_units=OUTPUT_UNITS,
        loss=LOSS,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    seed = "69 _ _ _ 69 _ _"
    
    # Generate with LSTM model
    mg_lstm = MelodyGenerator("lstm_model.h5", "lstm")
    lstm_melody = mg_lstm.generate_melody(
        seed=seed,
        num_steps=500,
        temperature=0.3
    )
    mg_lstm.save_melody(lstm_melody, file_name="lstm_output")

    # Generate with Bi-LSTM model
    mg_bilstm = MelodyGenerator("bilstm_model.h5", "bilstm")
    bilstm_melody = mg_bilstm.generate_melody(
        seed=seed,
        num_steps=500,
        temperature=0.7
    )
    mg_bilstm.save_melody(bilstm_melody, file_name="bilstm_output")

    
    # Analyze novelty
    lstm_novelty = analyze_melody_novelty("lstm_output.mid", training_dataset_path)
    bilstm_novelty = analyze_melody_novelty("bilstm_output.mid", training_dataset_path)

    print("\nComparison Results:")
    print("-" * 50)
    print(f"LSTM Training Time: {lstm_time:.2f} seconds")
    print(f"Bi-LSTM Training Time: {bilstm_time:.2f} seconds")
    
    print(f"\nLSTM Test Loss: {lstm_loss:.4f}")
    print(f"Bi-LSTM Test Loss: {bilstm_loss:.4f}")
    
    print(f"\nLSTM Test Accuracy: {lstm_accuracy:.4f}")
    print(f"Bi-LSTM Test Accuracy: {bilstm_accuracy:.4f}")
    
    print("\nNovelty Analysis:")
    print(f"LSTM Average Novelty: {lstm_novelty['average_novelty']:.4f}")
    print(f"Bi-LSTM Average Novelty: {bilstm_novelty['average_novelty']:.4f}")

    # Plot training histories
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(lstm_history.history['loss'], label='LSTM Training Loss')
    plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')
    plt.plot(bilstm_history.history['loss'], label='Bi-LSTM Training Loss')
    plt.plot(bilstm_history.history['val_loss'], label='Bi-LSTM Validation Loss')
    plt.title('Model Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(2, 2, 2)
    plt.plot(lstm_history.history['accuracy'], label='LSTM Training Accuracy')
    plt.plot(lstm_history.history['val_accuracy'], label='LSTM Validation Accuracy')
    plt.plot(bilstm_history.history['accuracy'], label='Bi-LSTM Training Accuracy')
    plt.plot(bilstm_history.history['val_accuracy'], label='Bi-LSTM Validation Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Novelty comparison
    plt.subplot(2, 2, 3)
    models = ['LSTM', 'Bi-LSTM']
    novelties = [lstm_novelty['average_novelty'], bilstm_novelty['average_novelty']]
    plt.bar(models, novelties)
    plt.title('Average Novelty Comparison')
    plt.ylabel('Novelty Score')
    
    plt.tight_layout()
    plt.savefig('lstm_bilstm_comparison.png')

if __name__ == "__main__":
    compare_models("dataset") 
