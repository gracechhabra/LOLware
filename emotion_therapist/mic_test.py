import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print("🎤 Say something (you have 3–5 seconds)...")
    r.adjust_for_ambient_noise(source, duration=0.5)
    audio = r.listen(source, timeout=5, phrase_time_limit=5)

try:
    text = r.recognize_google(audio)
    print("🗣️ You said:", text)
except sr.UnknownValueError:
    print("🤔 I couldn't understand you.")
except sr.RequestError as e:
    print(f"⚠️ Speech service error: {e}")
