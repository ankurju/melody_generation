import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from music21 import converter, note, chord
import os

def extract_melody_features_midi(midi_file):
    """Extract features from MIDI file for DTW comparison"""
    score = converter.parse(midi_file)
    features = []
    
    for element in score.flatten().notesAndRests:
        if isinstance(element, note.Note):
            features.append([
                float(element.pitch.midi),
                float(element.duration.quarterLength),
                float(element.offset)
            ])
        elif isinstance(element, note.Rest):
            features.append([
                0.0,  # Rest represented as pitch 0
                float(element.duration.quarterLength),
                float(element.offset)
            ])
    
    return np.array(features)

def extract_melody_features(melody_sequence):
    """
    Extract features from melody sequence similar to MIDI features:
    - Pitch (MIDI number or 0 for rest)
    - Duration (count of prolongation symbols + 1)
    - Offset (position in sequence)
    """
    features = []
    current_offset = 0.0
    
    i = 0
    while i < len(melody_sequence):
        note = melody_sequence[i]
        duration = 1.0
        j = i + 1
        while j < len(melody_sequence) and melody_sequence[j] == "_":
            duration += 1.0
            j += 1
        
        if note == "r":
            features.append([0.0, duration, current_offset])
        elif note != "_":
            try:
                features.append([
                    float(note),  
                    duration,     
                    current_offset 
                ])
            except ValueError:
                pass
        
        current_offset += duration
        i = j if j > i else i + 1
    
    return np.array(features)

def calculate_novelty_score(generated_features, training_features):
    """Calculate novelty score using DTW"""
    if len(generated_features) == 0 or len(training_features) == 0:
        return 0.0
        
    distance, _ = fastdtw(generated_features, training_features, dist=euclidean)
    # Normalize the distance
    max_len = max(len(generated_features), len(training_features))
    normalized_distance = distance / max_len
    
    # Convert distance to novelty score (higher distance = more novel)
    novelty_score = 1 / (1 + normalized_distance)
    return novelty_score

def analyze_melody_novelty(generated_melody, training_dataset_path):
    """
    Analyze the novelty of a generated melody compared to training data
    
    Args:
        generated_melody: List of note symbols from the generated melody
        training_dataset_path: Path to the folder containing training text files
    """
    generated_features = extract_melody_features_midi(generated_melody)
    
    novelty_scores = []
    for training_file in os.listdir(training_dataset_path):
        training_features = extract_melody_features(
            os.path.join(training_dataset_path, training_file)
        )
        score = calculate_novelty_score(generated_features, training_features)
        novelty_scores.append(score)
    
    if not novelty_scores:
        return {
            'average_novelty': 0.0,
            'std_novelty': 0.0,
            'max_novelty': 0.0,
            'min_novelty': 0.0
        }
    
    return {
        'average_novelty': np.mean(novelty_scores),
        'std_novelty': np.std(novelty_scores),
        'max_novelty': np.max(novelty_scores),
        'min_novelty': np.min(novelty_scores)
    } 
