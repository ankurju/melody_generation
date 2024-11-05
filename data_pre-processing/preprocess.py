import os
import music21 as m21

KERN_DATASET_PATH = "deutschl/test"
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

    print(key)

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
    for event in song.flat.notesAndRests:
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

def preprocess(dataset_path):

    # load the folk songs
    print("Loading songs....")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs")

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



if __name__== "__main__":
    songs = load_songs_in_kern(KERN_DATASET_PATH)
    print(f"Loaded {len(songs)} songs.")
    song = songs[0]
    preprocess(KERN_DATASET_PATH)
    transposed_song = transpose(song)
    # song.show()
