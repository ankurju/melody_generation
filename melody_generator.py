import tensorflow.keras as keras
import json
import numpy as np
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
import music21 as m21


class MelodyGenerator:
    def __init__(self, model_path, model_type="lstm"):
        """
        Constructor that sets up state of the melody generator.
        :param model_path: path to the trained model (.h5 file)
        :param model_type: type of model ("lstm" or "bilstm")
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        if self.model_type not in ["lstm", "bilstm"]:
            raise ValueError("Model type must be either 'lstm' or 'bilstm'")

        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH


    def _sample_with_temperature(self, probabilites, temperature):
        """Samples an index from a probability array reapplying softmax using temperature
           return index (int): Selected output symbol
        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index
    
    def generate_melody(self, seed, num_steps, max_sequence_length=SEQUENCE_LENGTH, temperature=1.0):
        """
        Generates a melody using either LSTM or Bi-LSTM model
        """
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # Map seed to integers
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check whether we're at the end of a melody
            if output_symbol == "/":
                break

            # update melody
            melody.append(output_symbol)
        return melody

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name=None):
        """
        Converts the melody into a MIDI file
        """
        if file_name is None:
            file_name = f"{self.model_type}_output"

        # Create a music21 stream
        stream = m21.stream.Stream()

        # Parse all the symbols in the melody and create note/rest objects
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            # Handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter

                    # Handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    # Handle note
                    else:
                        m21_event = m21.note.Note(
                            int(start_symbol),
                            quarterLength=quarter_length_duration
                        )

                    stream.append(m21_event)
                    step_counter = 1

                start_symbol = symbol
            # Handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # Write the m21 stream to a file
        if format == "midi":
            stream.write(format, f"{file_name}.mid")
        else:
            stream.write(format, f"{file_name}.musicxml")
        
        print(f"Melody saved as {file_name}.{format}")

def main():
    """Example usage of the MelodyGenerator with both models"""
    # Example seed melody
    seed = "69 _ _ _ 69 _ _ _ _ _ 68 _ 69 _ _ _ 71 _ _ _ 72 _ _ _ _ _ 72"
    
    # Generate with LSTM model
    try:
        print("\nGenerating melody with LSTM model...")
        mg_lstm = MelodyGenerator("lstm_model.h5", "lstm")
        lstm_melody = mg_lstm.generate_melody(
            seed=seed,
            num_steps=500,
            temperature=0.3
        )
        mg_lstm.save_melody(lstm_melody, file_name="lstm_output")
    except Exception as e:
        print(f"Error generating LSTM melody: {e}")

    # Generate with Bi-LSTM model
    try:
        print("\nGenerating melody with Bi-LSTM model...")
        mg_bilstm = MelodyGenerator("bilstm_model.h5", "bilstm")
        bilstm_melody = mg_bilstm.generate_melody(
            seed=seed,
            num_steps=500,
            temperature=0.3
        )
        mg_bilstm.save_melody(bilstm_melody, file_name="bilstm_output")
    except Exception as e:
        print(f"Error generating Bi-LSTM melody: {e}")

if __name__ == "__main__":
    main()
