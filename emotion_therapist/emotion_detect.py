import cv2
from deepface import DeepFace
import time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

print("✅ Webcam running. Press 'q' to quit.")

last_check = 0
emotion = "Analyzing..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # every 5 seconds → analyze
    if time.time() - last_check >= 5:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            last_check = time.time()
            print(f"🧠 Updated emotion: {emotion}")
        except Exception as e:
            print("⚠️ Error analyzing:", e)

    # overlay text on camera feed
    cv2.putText(frame, f"Emotion: {emotion}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Emotion Detector (Every 5 s)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
