# configuration for training
OUTPUT_UNITS = 18
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"

# configuration for preprocessing
KERN_DATASET_PATH = "deutschl/test"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64
SAVE_DIR = "dataset"
ACCEPTABLE_DURATIONS=[
    0.25 , 0.5 , 0.75 , 1.0 , 1.5 , 2 , 3 , 4
]