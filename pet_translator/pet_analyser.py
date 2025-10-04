import librosa
import numpy as np

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    rms = np.sqrt(np.mean(y**2))
    
    # Pitch estimate
    pitch = librosa.yin(y, fmin=80, fmax=2000)
    pitch_mean = float(np.mean(pitch))
    # Simple heuristics
    if pitch_mean > 400:
        pet_type = "dog"
    else:
        pet_type = "cat"
    
    # Optional: add mood based on loudness
    mood = "excited" if rms > 0.05 else "calm"

    return {
        "duration": round(duration,2),
        "rms": round(float(rms),5),
        "pitch": round(pitch_mean,1),
        "type": pet_type,
        "mood": mood
    }
