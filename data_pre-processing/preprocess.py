import os
import music21 as m21
import json
import tensorflow.keras as keras
import numpy as np

KERN_DATASET_PATH = "deutschl/test"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64
SAVE_DIR = "dataset"
ACCEPTABLE_DURATIONS=[
    0.25 , 0.5 , 0.75 , 1.0 , 1.5 , 2 , 3 , 4
]

def load_songs_in_kern(dataset_path):
    songs=[]
    # go through all the files in the dataset and load them with music21
    for path, subdirs,files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path,file))
                songs.append(song)
    
    return songs

def has_acceptable_durations(song , acceptable_durations):
    for note in song.flatten().notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    
    return True

def transpose(song):
    """Transposes song to C maj/A min

    :param piece (m21 stream): Piece to transpose
    :return transposed_song (m21 stream):
    """

    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # get interval for transposition. E.g., Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    tranposed_song = song.transpose(interval)
    return tranposed_song

def encode_song(song,time_step=0.25):
    encoded_song=[]
    for event in song.flatten().notesAndRests:
        #handle notes
        if isinstance(event,m21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event,m21.note.Rest):
            symbol = "r"

        steps = int(event.duration.quarterLength/time_step)
        for step in range(steps):
            if step==0:
                encoded_song.append(symbol)
            else :
                encoded_song.append("_")

    #cast encoded song to a string
    encoded_song = " ".join(map(str,encoded_song))
    return encoded_song

def load(file_path):
    with open(file_path,"r") as fp:
        song = fp.read()
    return song

def create_single_file_dataset(dataset_path,file_dataset_path,sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # load encoded songs and add delimiters
    for path,_,files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path,file)
            song = load(file_path)
            songs += song + " " + new_song_delimiter
    songs = songs[:-1]

    #save strings that contain all the dataset
    with open(file_dataset_path,"w") as fp:
        fp.write(songs)

    return songs

def create_mapping(songs,mapping_path):
    mappings = {}

    #identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    #create mappings
    for i,symbol in enumerate(vocabulary):
        mappings[symbol] = i

    #save the vocabulary to a json file
    with open(mapping_path ,"w") as fp:
        json.dump(mappings,fp,indent=4)

def preprocess(dataset_path):

    # load the folk songs
    songs = load_songs_in_kern(dataset_path)

    for i,song in enumerate(songs):
        # filter out songs that have non-acceptable durations
        if not has_acceptable_durations(song,ACCEPTABLE_DURATIONS):
            continue

        # transpose songs to Cmaj/Amin
        song = transpose(song)

        # encode songs with music time series representation
        encoded_song = encode_song(song)

        # save songs to text file
        save_path = os.path.join(SAVE_DIR,str(i))
        with open(save_path,"w") as fp:
            fp.write(encoded_song)

def convert_songs_to_int(songs):
    int_songs = []
    #load mappings
    with open(MAPPING_PATH,"r") as fp:
        mappings = json.load(fp)

    #cast songs string to a list
    songs = songs.split()

    #map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs

def generate_training_sequences(sequence_length):
    X = []
    y = []

    # load songs and convert them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    # generate the training sequences
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        X.append(int_songs[i:i+sequence_length])
        y.append(int_songs[i+sequence_length])

    # one hot encode the sequences
    vocabulary_size = len(set(int_songs))
    X = keras.utils.to_categorical(X,num_classes=vocabulary_size)
    y = np.array(y)

    return X,y



def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR,SINGLE_FILE_DATASET,SEQUENCE_LENGTH)
    create_mapping(songs,MAPPING_PATH)
    X,y = generate_training_sequences(SEQUENCE_LENGTH)

if __name__== "__main__":
    main()