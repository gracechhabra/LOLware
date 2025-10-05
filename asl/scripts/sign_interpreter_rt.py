import cv2
import numpy as np
from collections import deque, Counter
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- optional model & TTS ---
clf = None
le  = None
synth = None
try:
    from joblib import load
    clf = load("models/gesture_clf.joblib")
    le  = load("models/label_encoder.joblib")
except Exception:
    pass

try:
    import pyttsx3
    synth = pyttsx3.init()
except Exception:
    pass

# --- mediapipe hands (classic Solutions API for simplicity) ---
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# landmark indices (MediaPipe)
WRIST = 0
FINGERTIPS = [4, 8, 12, 16, 20]
# per-finger chains: (MCP, PIP, DIP, TIP)
FINGERS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9,10,11,12],
    "ring":   [13,14,15,16],
    "pinky":  [17,18,19,20],
}

# ----------------------------- feature helpers -----------------------------
def _unit(v):
    n = np.linalg.norm(v) + 1e-9
    return v / n

def angle_abc(a, b, c):
    # angle at b (in radians) from vectors BA and BC
    ba = _unit(a - b); bc = _unit(c - b)
    dot = np.clip(np.dot(ba, bc), -1.0, 1.0)
    return np.arccos(dot)

def hand_size_scale(pts):
    # a robust scale: mean distance of fingertips to wrist
    d = np.mean([np.linalg.norm(pts[i] - pts[WRIST]) for i in FINGERTIPS])
    return d if d > 1e-6 else 1.0

def normalize_landmarks(pts):
    """Center at wrist, divide by hand size. pts: (21,3) in image-space [0..1]"""
    center = pts[WRIST].copy()
    scale  = hand_size_scale(pts)
    normed = (pts - center) / scale
    return normed

def joint_angles(pts):
    """15 angles: MCP/PIP/DIP for 5 fingers (thumb MCP/IP+TIP approximations)."""
    angs = []
    # index/middle/ring/pinky: MCP=P(WRIST,MCP,PIP), PIP=P(MCP,PIP,DIP), DIP=P(PIP,DIP,TIP)
    for name in ["index","middle","ring","pinky"]:
        mcp,pip,dip,tip = [pts[i] for i in FINGERS[name]]
        wrist = pts[WRIST]
        angs += [
            angle_abc(wrist, mcp, pip),
            angle_abc(mcp,   pip, dip),
            angle_abc(pip,   dip, tip),
        ]
    # thumb: CMC≈1, MCP=2, IP=3, TIP=4; use MCP/IP + pseudo "TIP spread"
    cmc,mcp,ip,tip = [pts[i] for i in FINGERS["thumb"]]
    angs += [
        angle_abc(cmc, mcp, ip),
        angle_abc(mcp, ip, tip),
        angle_abc(pts[WRIST], cmc, mcp)  # thumb spread vs wrist
    ]
    return np.array(angs, dtype=np.float32)  # shape (15,)

def pairwise_fingertip_dists(pts):
    """10 pairwise distances across the 5 fingertips (order: (4,8),(4,12),...)."""
    tips = FINGERTIPS
    vals = []
    for i in range(len(tips)):
        for j in range(i+1, len(tips)):
            vals.append(np.linalg.norm(pts[tips[i]] - pts[tips[j]]))
    return np.array(vals, dtype=np.float32)  # shape (10,)

def hand_feature_vector(landmarks_21x3):
    """Build per-hand feature vector: normalized coords + angles + pairwise dists."""
    pts = normalize_landmarks(landmarks_21x3)
    # flattened normalized coords (63), angles (15), fingertip dists (10)
    feats = [pts.flatten(), joint_angles(pts), pairwise_fingertip_dists(pts)]
    return np.concatenate(feats, axis=0)  # shape (88,)

def pack_two_hands(left_pts, right_pts):
    """Return a fixed-length feature vector for both hands (zeros if missing)."""
    left_vec  = hand_feature_vector(left_pts)  if left_pts  is not None else np.zeros(88, np.float32)
    right_vec = hand_feature_vector(right_pts) if right_pts is not None else np.zeros(88, np.float32)
    return np.concatenate([left_vec, right_vec], axis=0)  # shape (176,)

def aggregate_window(window_np):
    """
    Turn a (T, 176) window into a fixed vector for non-sequence models.
    We use mean, std, and deltas. Result shape: 176*3 = 528
    """
    mean = window_np.mean(axis=0)
    std  = window_np.std(axis=0)
    d1   = np.diff(window_np, axis=0).mean(axis=0)
    return np.concatenate([mean, std, d1], axis=0).astype(np.float32)

# ----------------------------- runtime config -----------------------------
WIN = 32                  # sliding window length (frames)
MIN_CONF_SAY = 0.75       # speak when smoothed conf passes this
SMOOTH_K = 7              # majority vote over last K labels
DRAW = True               # draw landmarks

# ----------------------------- main loop -----------------------------
def main():
    global clf, le, synth

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam")
        return

    # better latency on some cams
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    window = deque(maxlen=WIN)
    recent_labels = deque(maxlen=SMOOTH_K)
    last_spoken = None
    last_say_t  = 0.0

    # MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            rgb.flags.writeable = True

            frame_h, frame_w = frame.shape[:2]

            left_pts = right_pts = None
            # organize by handedness so our feature order is stable
            if res.multi_hand_landmarks and res.multi_handedness:
                # pair (handedness, landmarks)
                pairs = list(zip(res.multi_handedness, res.multi_hand_landmarks))
                # ensure "Left" always maps to left_pts
                for handed, lm in pairs:
                    label = handed.classification[0].label  # "Left" or "Right"
                    pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
                    if label == "Left":
                        left_pts = pts
                    else:
                        right_pts = pts

                if DRAW:
                    for handed, lm in pairs:
                        mp_drawing.draw_landmarks(
                            frame,
                            lm,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )

            # build per-frame features (always shape 176)
            feat = pack_two_hands(left_pts, right_pts)
            window.append(feat)

            pred_label = "(no model)"
            pred_prob  = 0.0

            if clf is not None and len(window) == WIN:
                W = np.stack(window, axis=0)  # (T,176)
                x = aggregate_window(W).reshape(1, -1)  # (1,528)
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(x)[0]
                    idx = int(np.argmax(probs))
                    pred_prob = float(probs[idx])
                    pred_label = le.inverse_transform([idx])[0] if le is not None else str(idx)
                else:
                    idx = int(clf.predict(x)[0])
                    pred_label = le.inverse_transform([idx])[0] if le is not None else str(idx)
                    pred_prob = 1.0  # unknown

                # temporal smoothing: majority vote over last K frame-level labels
                recent_labels.append(pred_label)
                if len(recent_labels) == SMOOTH_K:
                    pred_label = Counter(recent_labels).most_common(1)[0][0]
                    # light probability smoothing: EMA using window mean conf
                    pred_prob = 0.6 * pred_prob + 0.4 * (pred_prob if pred_prob else 0.0)

                # (optional) speak when label changes with high confidence
                now = time.time()
                if synth and pred_prob >= MIN_CONF_SAY and pred_label != last_spoken and (now - last_say_t) > 0.6:
                    try:
                        synth.stop()
                        synth.say(pred_label)
                        synth.runAndWait()
                        last_spoken = pred_label
                        last_say_t  = now
                    except Exception:
                        pass

            # HUD
            hud = f"{pred_label}  {pred_prob:.2f}" if clf is not None else "(no model loaded)"
            cv2.rectangle(frame, (10,10), (10+380, 10+50), (0,0,0), -1)
            cv2.putText(frame, hud, (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow("Sign Interpreter — real-time", frame)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):  # ESC or q
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()