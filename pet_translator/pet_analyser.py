# pet_analyzer.py
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa

print("Loading YAMNet model (first time may take a few seconds)...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Get class names
class_map_path = yamnet_model.class_map_path().numpy()
with open(class_map_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def analyze_audio(file_path):
    # Load audio
    waveform, sr = librosa.load(file_path, sr=16000)  # YAMNet expects 16kHz
    waveform = waveform.astype(np.float32)

    # Run YAMNet
    scores, embeddings, spectrogram = yamnet_model(waveform)
    mean_scores = np.mean(scores, axis=0)
    top_class = np.argmax(mean_scores)
    label = class_names[top_class]
    confidence = float(mean_scores[top_class])

    # Simplified pet classification
    if "bark" in label.lower():
        pet_type = "dog"
    elif "meow" in label.lower():
        pet_type = "cat"
    elif any(word in label.lower() for word in ["bird", "chirp", "tweet"]):
        pet_type = "bird"
    else:
        pet_type = "dog"  # Default to dog for unknown sounds

    mood = "excited" if confidence > 0.5 else "calm"

    return {
        "type": pet_type,
        "label": label,
        "confidence": round(confidence, 2),
        "mood": mood
    }
