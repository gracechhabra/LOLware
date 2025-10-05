import cv2
from deepface import DeepFace
import speech_recognition as sr
from gtts import gTTS
import os
import google.generativeai as genai
import os
from dotenv import load_dotenv # <-- NEW LINE

load_dotenv() # <-- NEW LINE: This finds and loads the .env file

# Your existing configuration line will now work:
# genai.configure(api_key=os.getenv("GEMINI_API_KEY")) 
# ... rest of your script
# --- Configure Gemini ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Step 1: Detect emotion from webcam ---
def detect_emotion():
    cap = cv2.VideoCapture(0)
    print("📸 Camera started — press 'q' to capture emotion.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Press 'q' when ready", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    emotion = result[0]['dominant_emotion']
    print(f"🧩 Detected emotion: {emotion}")
    return emotion

# --- Step 2: Listen to user speech ---
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print(f"🗣 You said: {text}")
        return text
    except sr.UnknownValueError:
        print("⚠️ Could not understand audio.")
        return ""
    except sr.RequestError:
        print("❌ Speech service error.")
        return ""

# --- Step 3: Get Gemini’s empathetic response ---
def ask_gemini(emotion, text):
    prompt = (
        f"You are an empathetic AI therapist. "
        f"The user currently feels {emotion}. "
        f"They said: '{text}'. "
        f"Please reply in a comforting, human-like way — short and warm."
    )

    model = genai.GenerativeModel('gemini-2.5-flash')  # ✅ correct version
    response = model.generate_content(prompt)
    
    reply = response.text
    print(f"💬 Gemini says: {reply}")
    return reply

# --- Step 4: Convert Gemini’s response to speech ---
def speak(reply):
    print("🔊 Speaking...")
    tts = gTTS(reply)
    tts.save("reply.mp3")
    os.system("start reply.mp3")  # Windows compatible


    # Add a small delay to ensure the player starts before deleting
    import time
    time.sleep(5)  # Wait 5 seconds for playback to begin/finish

    os.remove("reply.mp3") # <-- CLEANUP LINE
    print("Cleanup complete.")

# --- Main program flow ---
if __name__ == "__main__":
    print("🧠 Emotion Therapist starting...")
    emotion = detect_emotion()
    text = listen()
    if text:
        reply = ask_gemini(emotion, text)
        speak(reply)
