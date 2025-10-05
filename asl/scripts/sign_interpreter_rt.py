# # scripts/sign_interpreter_rt.py
# import os
# import cv2
# import time
# import json
# import pickle
# import argparse
# import warnings
# import numpy as np
# from collections import deque, Counter

# warnings.filterwarnings("ignore", category=UserWarning)

# # ----------------------------- CLI -----------------------------
# def parse_args():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model",  required=True,
#                     help="Trained model (.h5/.keras for Keras, or .joblib for sklearn)")
#     ap.add_argument("--labels", default=None,
#                     help="Label encoder (.pkl or .joblib). Defaults to <model>.labels.pkl")
#     ap.add_argument("--label-map", default=None,
#                     help="JSON dict mapping raw labels to pretty strings, e.g. {\"120\": \"X\"}")
#     ap.add_argument("--camera", type=int, default=0, help="Camera index")
#     ap.add_argument("--backend", choices=["auto","dshow","msmf","avfoundation","v4l2","gstreamer"],
#                     default="auto", help="Force a camera backend if needed")
#     ap.add_argument("--no-draw", action="store_true", help="Disable landmark drawing")
#     ap.add_argument("--win", type=int, default=32, help="Sliding window length (frames)")
#     ap.add_argument("--debug", action="store_true", help="Verbose prints")
#     ap.add_argument("--print-classes", action="store_true",
#                     help="Print the class order and their display names at startup")
#     return ap.parse_args()

# def backend_flag(name):
#     return {
#         "auto":        None,
#         "dshow":       cv2.CAP_DSHOW,        # Windows
#         "msmf":        cv2.CAP_MSMF,         # Windows
#         "avfoundation":cv2.CAP_AVFOUNDATION, # macOS
#         "v4l2":        cv2.CAP_V4L2,         # Linux
#         "gstreamer":   cv2.CAP_GSTREAMER,    # Linux alt
#     }[name]

# # ----------------------------- Loading helpers -----------------------------
# def load_any_model(path):
#     lp = path.lower()
#     if lp.endswith(".h5") or lp.endswith(".keras"):
#         from tensorflow.keras.models import load_model
#         return load_model(path), "keras"
#     else:
#         from joblib import load
#         return load(path), "sklearn"

# def load_any_labels(path):
#     lp = path.lower()
#     if lp.endswith(".pkl"):
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     else:
#         from joblib import load
#         return load(path)

# def load_label_map(path):
#     with open(path, "r", encoding="utf-8") as f:
#         raw = json.load(f)
#     # make keys accessible as both strings and ints
#     mapping = {}
#     for k, v in raw.items():
#         mapping[str(k)] = str(v)
#         try:
#             mapping[int(k)] = str(v)
#         except Exception:
#             pass
#     return mapping

# def predict_proba_any(model, x, kind):
#     if kind == "sklearn":
#         if hasattr(model, "predict_proba"):
#             return model.predict_proba(x)
#         if hasattr(model, "decision_function"):
#             s = model.decision_function(x)
#             e = np.exp(s - np.max(s, axis=1, keepdims=True))
#             return e / np.sum(e, axis=1, keepdims=True)
#         idx = int(model.predict(x)[0])
#         n   = max(idx+1, 1)
#         one = np.zeros((1, n), dtype=np.float32); one[0, idx] = 1.0
#         return one
#     else:  # keras
#         try:
#             return model.predict(x, verbose=0)
#         except TypeError:
#             return model.predict(x)

# def expected_input_dim(model, kind):
#     if kind == "keras":
#         ish = model.input_shape
#         if isinstance(ish, list):
#             ish = ish[0]
#         if isinstance(ish, tuple) and len(ish) >= 2 and ish[1] is not None:
#             return int(ish[1])
#     else:
#         n = getattr(model, "n_features_in_", None)
#         if n is not None:
#             return int(n)
#     return None

# # --- optional TTS ---
# synth = None
# try:
#     import pyttsx3
#     synth = pyttsx3.init()
# except Exception:
#     pass

# # --- mediapipe hands ---
# import mediapipe as mp
# mp_hands   = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_styles  = mp.solutions.drawing_styles

# WRIST = 0
# FINGERTIPS = [4, 8, 12, 16, 20]
# FINGERS = {
#     "thumb":  [1, 2, 3, 4],
#     "index":  [5, 6, 7, 8],
#     "middle": [9,10,11,12],
#     "ring":   [13,14,15,16],
#     "pinky":  [17,18,19,20],
# }

# # ----------------------------- feature helpers -----------------------------
# def _unit(v):
#     n = np.linalg.norm(v) + 1e-9
#     return v / n

# def angle_abc(a, b, c):
#     ba = _unit(a - b); bc = _unit(c - b)
#     dot = np.clip(np.dot(ba, bc), -1.0, 1.0)
#     return np.arccos(dot)

# def hand_size_scale(pts):
#     d = np.mean([np.linalg.norm(pts[i] - pts[WRIST]) for i in FINGERTIPS])
#     return d if d > 1e-6 else 1.0

# def normalize_landmarks(pts):
#     center = pts[WRIST].copy()
#     scale  = hand_size_scale(pts)
#     return (pts - center) / scale

# def joint_angles(pts):
#     angs = []
#     for name in ["index","middle","ring","pinky"]:
#         mcp,pip,dip,tip = [pts[i] for i in FINGERS[name]]
#         wrist = pts[WRIST]
#         angs += [
#             angle_abc(wrist, mcp, pip),
#             angle_abc(mcp,   pip, dip),
#             angle_abc(pip,   dip, tip),
#         ]
#     cmc,mcp,ip,tip = [pts[i] for i in FINGERS["thumb"]]
#     angs += [
#         angle_abc(cmc, mcp, ip),
#         angle_abc(mcp, ip, tip),
#         angle_abc(pts[WRIST], cmc, mcp)
#     ]
#     return np.array(angs, dtype=np.float32)

# def pairwise_fingertip_dists(pts):
#     tips = FINGERTIPS
#     vals = []
#     for i in range(len(tips)):
#         for j in range(i+1, len(tips)):
#             vals.append(np.linalg.norm(pts[tips[i]] - pts[tips[j]]))
#     return np.array(vals, dtype=np.float32)

# # Per-hand feature blocks
# def hand_coords63(pts):
#     return normalize_landmarks(pts).flatten().astype(np.float32)  # (63,)

# def hand_full88(pts):
#     ptsn = normalize_landmarks(pts)
#     return np.concatenate([ptsn.flatten(), joint_angles(ptsn), pairwise_fingertip_dists(ptsn)], axis=0).astype(np.float32)  # (88,)

# # Aggregate window (mean/std/delta)
# def aggregate_window(window_np):
#     mean = window_np.mean(axis=0)
#     std  = window_np.std(axis=0)
#     d1   = np.diff(window_np, axis=0).mean(axis=0)
#     return np.concatenate([mean, std, d1], axis=0).astype(np.float32)

# # ----------------------------- label display helper -----------------------------
# def make_display_labeler(label_map):
#     """
#     Returns a function display_label(raw) -> str
#     Priority: user-supplied label_map -> ASCII decode -> str(raw)
#     """
#     def display_label(raw):
#         # exact match on int key
#         if raw in label_map:
#             return label_map[raw]
#         # match on stringified key
#         raw_str = str(raw)
#         if raw_str in label_map:
#             return label_map[raw_str]
#         # try ASCII decode (common when labels are code points)
#         try:
#             if isinstance(raw, (np.integer, int)) and 32 <= int(raw) <= 126:
#                 return chr(int(raw))
#         except Exception:
#             pass
#         return raw_str
#     return display_label

# # ----------------------------- runtime config -----------------------------
# MIN_CONF_SAY = 0.75
# SMOOTH_K = 7

# # ----------------------------- main -----------------------------
# def main():
#     args = parse_args()
#     DRAW = not args.no_draw
#     WIN  = args.win

#     labels_path = args.labels or (args.model + ".labels.pkl")

#     # Load model + labels
#     try:
#         clf, model_kind = load_any_model(args.model)
#         print(f"[OK] Loaded {model_kind} model: {args.model}")
#     except Exception as e:
#         print(f"[ERR] Failed to load model '{args.model}': {e}")
#         return

#     exp_dim = expected_input_dim(clf, model_kind)
#     if exp_dim is None:
#         print("[ERR] Could not determine model input dimension.")
#         return
#     print(f"[INFO] Model expects feature dim: {exp_dim} "
#           f"({'coords 63/126' if exp_dim in (63,126) else 'single 88 / win-agg 264' if exp_dim in (88,264) else 'two-hand 176' if exp_dim==176 else 'win-agg 528' if exp_dim==528 else 'custom'})")

#     le = None
#     try:
#         le = load_any_labels(labels_path)
#         print(f"[OK] Loaded labels: {labels_path} ({len(getattr(le, 'classes_', []))} classes)")
#     except Exception as e:
#         print(f"[WARN] Could not load labels '{labels_path}': {e} (will show numeric IDs)")

#     # Optional label map for pretty names
#     label_map = {}
#     if args.label_map:
#         try:
#             label_map = load_label_map(args.label_map)
#             print(f"[OK] Loaded label map: {args.label_map} ({len(label_map)//2} unique keys)")
#         except Exception as e:
#             print(f"[WARN] Could not load label map '{args.label_map}': {e}")
#             label_map = {}
#     display_label = make_display_labeler(label_map)

#     # Optionally print class order
#     if args.print-classes if False else False:  # placeholder to avoid syntax issue
#         pass
#     if args.print_classes and le is not None:
#         print("[INFO] Class order (probs index -> raw -> display):")
#         for i, raw in enumerate(le.classes_):
#             print(f"  {i}: {raw} -> {display_label(raw)}")

#     # Open camera
#     bflag = backend_flag(args.backend)
#     cap = cv2.VideoCapture(args.camera, bflag) if bflag is not None else cv2.VideoCapture(args.camera)
#     if not cap.isOpened():
#         print(f"[ERR] Cannot open webcam index {args.camera} (backend: {args.backend})")
#         return
#     try:
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     except Exception:
#         pass

#     print("[INFO] Camera opened. Press 'q' or ESC to quit.")
#     cv2.namedWindow("Sign Interpreter — real-time", cv2.WINDOW_NORMAL)

#     # Windows for aggregated modes
#     window_both176  = deque(maxlen=WIN)  # two-hand full features (88+88)
#     window_single88 = deque(maxlen=WIN)  # single-hand full features (88)

#     # MediaPipe
#     hands = mp_hands.Hands(
#         static_image_mode=False,
#         max_num_hands=2,
#         model_complexity=1,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )

#     # recent labels for smoothing
#     recent_labels = deque(maxlen=SMOOTH_K)
#     last_spoken = None
#     last_say_t  = 0.0
#     fail_streak = 0

#     try:
#         while True:
#             ok, frame = cap.read()
#             if not ok:
#                 fail_streak += 1
#                 if fail_streak > 30:
#                     print("[ERR] Camera stopped delivering frames (fail_streak>30). Exiting.")
#                     break
#                 time.sleep(0.01)
#                 continue
#             fail_streak = 0

#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             rgb.flags.writeable = False
#             res = hands.process(rgb)
#             rgb.flags.writeable = True

#             left_pts = right_pts = None
#             pairs = []
#             if res.multi_hand_landmarks and res.multi_handedness:
#                 pairs = list(zip(res.multi_handedness, res.multi_hand_landmarks))
#                 for handed, lm in pairs:
#                     label = handed.classification[0].label  # "Left" or "Right"
#                     pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
#                     if label == "Left":
#                         left_pts = pts
#                     else:
#                         right_pts = pts
#                 if DRAW:
#                     for _, lm in pairs:
#                         mp_drawing.draw_landmarks(
#                             frame, lm, mp_hands.HAND_CONNECTIONS,
#                             mp_styles.get_default_hand_landmarks_style(),
#                             mp_styles.get_default_hand_connections_style()
#                         )

#             # Build per-frame feature blocks
#             left88  = hand_full88(left_pts)   if left_pts  is not None else None
#             right88 = hand_full88(right_pts)  if right_pts is not None else None
#             both176 = np.concatenate([
#                 left88  if left88  is not None else np.zeros(88, np.float32),
#                 right88 if right88 is not None else np.zeros(88, np.float32)
#             ], axis=0).astype(np.float32)

#             # keep windows for aggregated modes
#             window_both176.append(both176)
#             single88 = right88 if right88 is not None else left88
#             if single88 is not None:
#                 window_single88.append(single88)

#             # Decide what to feed based on exp_dim
#             x = None
#             need_warmup = False

#             if exp_dim == 528:  # win-agg two-hand full (176*3)
#                 if len(window_both176) < WIN:
#                     need_warmup = True
#                 else:
#                     W = np.stack(window_both176, axis=0)  # (T,176)
#                     x = aggregate_window(W).reshape(1, -1)

#             elif exp_dim == 176:  # two-hand full per-frame
#                 x = both176.reshape(1, -1)

#             elif exp_dim == 264:  # win-agg single-hand full (88*3)
#                 if len(window_single88) < WIN:
#                     need_warmup = True
#                 else:
#                     W = np.stack(window_single88, axis=0)  # (T,88)
#                     x = aggregate_window(W).reshape(1, -1)

#             elif exp_dim == 126:  # coords-only for both hands (63+63)
#                 l63 = hand_coords63(left_pts)  if left_pts  is not None else np.zeros(63, np.float32)
#                 r63 = hand_coords63(right_pts) if right_pts is not None else np.zeros(63, np.float32)
#                 x = np.concatenate([l63, r63], axis=0).reshape(1, -1)

#             elif exp_dim == 88:   # single-hand full per-frame
#                 if single88 is not None:
#                     x = single88.reshape(1, -1)

#             elif exp_dim == 63:   # single-hand coords-only per-frame
#                 single_pts = right_pts if right_pts is not None else left_pts
#                 if single_pts is not None:
#                     coords63 = hand_coords63(single_pts)
#                     x = coords63.reshape(1, -1)
#                     if args.debug:
#                         print(f"[DEBUG] coords63 shape: {coords63.shape}, x shape: {x.shape}, dtype: {x.dtype}")

#             else:
#                 x = None  # unknown size

#             # Predict
#             pred_label_str = "(no hands)"
#             pred_prob = 0.0

#             if need_warmup:
#                 pred_label_str, pred_prob = "warming up…", 0.0
#             elif x is not None:
#                 try:
#                     probs = predict_proba_any(clf, x, model_kind)[0]
#                     idx = int(np.argmax(probs))
#                     pred_prob = float(probs[idx])

#                     raw_label = le.inverse_transform([idx])[0] if le is not None else idx
#                     pred_label_str = display_label(raw_label)  # <-- ALWAYS STRING

#                     if args.debug:
#                         print(f"[DEBUG] probs: {probs}")
#                         print(f"[DEBUG] raw={raw_label} display='{pred_label_str}' prob={pred_prob:.4f}")
#                 except Exception:
#                     pred_label_str, pred_prob = "(error)", 0.0
#                     if args.debug:
#                         import traceback; print("[EXCEPTION] predict():"); traceback.print_exc()
#             else:
#                 if (left_pts is None and right_pts is None):
#                     pred_label_str = "(no hands)"
#                 else:
#                     pred_label_str = f"(unsupported D={exp_dim})"

#             # Smoothing (only for real labels)
#             if pred_label_str not in ("warming up…", "(no hands)", "(error)") and not pred_label_str.startswith("("):
#                 if not hasattr(main, "_recent"):
#                     main._recent = deque(maxlen=SMOOTH_K)
#                 main._recent.append(pred_label_str)
#                 if len(main._recent) == SMOOTH_K:
#                     pred_label_str = Counter(main._recent).most_common(1)[0][0]

#                 # TTS
#                 now = time.time()
#                 if synth and pred_prob >= MIN_CONF_SAY and getattr(main, "_last_spoken", None) != pred_label_str and getattr(main, "_last_say_t", 0) + 0.6 < now:
#                     try:
#                         synth.stop(); synth.say(pred_label_str); synth.runAndWait()
#                         main._last_spoken = pred_label_str; main._last_say_t = now
#                     except Exception:
#                         pass

#             # HUD
#             hud = f"[{model_kind}] {os.path.basename(args.model)} | D={exp_dim} | {pred_label_str}  {pred_prob:.2f}"
#             cv2.rectangle(frame, (10,10), (10+max(520, 18*len(hud)), 10+50), (0,0,0), -1)
#             cv2.putText(frame, hud, (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

#             cv2.imshow("Sign Interpreter — real-time", frame)
#             key = (cv2.waitKey(1) & 0xFF)
#             if key in (27, ord('q')):  # esc or q
#                 break
#             if cv2.getWindowProperty("Sign Interpreter — real-time", cv2.WND_PROP_VISIBLE) < 1:
#                 break

#     except Exception:
#         import traceback
#         print("[EXCEPTION] Unhandled error, shutting down:")
#         traceback.print_exc()
#         time.sleep(0.5)
#     finally:
#         hands.close()
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



# scripts/sign_interpreter_rt.py
import os, cv2, time, json, pickle, argparse, warnings
import numpy as np
from collections import deque, Counter
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------- CLI -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  required=True, help=".h5/.keras Keras or .joblib sklearn")
    ap.add_argument("--labels", default=None, help=".pkl/.joblib LabelEncoder (default: <model>.labels.pkl)")
    ap.add_argument("--label-map", default=None, help="JSON mapping raw labels → display strings")
    ap.add_argument("--print-classes", action="store_true", help="Print class index → raw → display")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--backend", choices=["auto","dshow","msmf","avfoundation","v4l2","gstreamer"], default="auto")
    ap.add_argument("--no-draw", action="store_true")
    ap.add_argument("--win", type=int, default=32, help="Window size for aggregated modes")
    ap.add_argument("--debug", action="store_true")
    # feature-compat toggles for 63-D models
    ap.add_argument("--coords-mode", choices=["centered","raw"], default="centered",
                    help="63-D input: centered=relative to wrist (default), raw=absolute 0..1 coords")
    ap.add_argument("--preferred-hand", choices=["auto","right","left"], default="auto",
                    help="Which hand to feed for single-hand models")
    ap.add_argument("--flip-x", action="store_true", help="Mirror x (left/right) before feature build")
    return ap.parse_args()

def backend_flag(name):
    return {
        "auto": None, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
        "avfoundation": cv2.CAP_AVFOUNDATION, "v4l2": cv2.CAP_V4L2, "gstreamer": cv2.CAP_GSTREAMER
    }[name]

# ----------------------------- Loading helpers -----------------------------
def load_any_model(path):
    lp = path.lower()
    if lp.endswith(".h5") or lp.endswith(".keras"):
        from tensorflow.keras.models import load_model
        return load_model(path), "keras"
    else:
        from joblib import load
        return load(path), "sklearn"

def load_any_labels(path):
    if path is None: return None
    lp = path.lower()
    if lp.endswith(".pkl"):
        with open(path, "rb") as f: return pickle.load(f)
    else:
        from joblib import load; return load(path)

def load_label_map(path):
    if not path: return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    m = {}
    for k,v in raw.items():
        m[str(k)] = str(v)
        try: m[int(k)] = str(v)
        except: pass
    return m

def predict_proba_any(model, x, kind):
    if kind == "sklearn":
        if hasattr(model, "predict_proba"): return model.predict_proba(x)
        if hasattr(model, "decision_function"):
            s = model.decision_function(x)
            e = np.exp(s - np.max(s, axis=1, keepdims=True))
            return e / np.sum(e, axis=1, keepdims=True)
        idx = int(model.predict(x)[0])
        one = np.zeros((1, max(idx+1,1)), np.float32); one[0, idx] = 1.0; return one
    else:
        try:    return model.predict(x, verbose=0)
        except: return model.predict(x)

def expected_input_dim(model, kind):
    if kind == "keras":
        ish = model.input_shape
        if isinstance(ish, list): ish = ish[0]
        if isinstance(ish, tuple) and len(ish) >= 2 and ish[1] is not None:
            return int(ish[1])
    else:
        n = getattr(model, "n_features_in_", None)
        if n is not None: return int(n)
    return None

# --- optional TTS ---
synth = None
try:
    import pyttsx3; synth = pyttsx3.init()
except Exception:
    pass

# --- mediapipe hands ---
import mediapipe as mp
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

WRIST = 0
FINGERTIPS = [4, 8, 12, 16, 20]
FINGERS = {"thumb":[1,2,3,4],"index":[5,6,7,8],"middle":[9,10,11,12],"ring":[13,14,15,16],"pinky":[17,18,19,20]}

# ----------------------------- feature helpers -----------------------------
def _unit(v): n = np.linalg.norm(v) + 1e-9; return v / n
def angle_abc(a,b,c):
    ba = _unit(a-b); bc = _unit(c-b)
    return np.arccos(np.clip(np.dot(ba,bc), -1.0, 1.0))

def hand_size_scale(pts):
    d = np.mean([np.linalg.norm(pts[i]-pts[WRIST]) for i in FINGERTIPS])
    return d if d>1e-6 else 1.0

def normalize_landmarks(pts):
    center = pts[WRIST].copy(); scale = hand_size_scale(pts)
    return (pts - center) / scale

def joint_angles(pts):
    angs=[]
    for name in ["index","middle","ring","pinky"]:
        mcp,pip,dip,tip=[pts[i] for i in FINGERS[name]]; wrist=pts[WRIST]
        angs += [angle_abc(wrist,mcp,pip), angle_abc(mcp,pip,dip), angle_abc(pip,dip,tip)]
    cmc,mcp,ip,tip=[pts[i] for i in FINGERS["thumb"]]
    angs += [angle_abc(cmc,mcp,ip), angle_abc(mcp,ip,tip), angle_abc(pts[WRIST], cmc, mcp)]
    return np.array(angs, np.float32)

def pairwise_fingertip_dists(pts):
    tips=FINGERTIPS; vals=[]
    for i in range(len(tips)):
        for j in range(i+1,len(tips)):
            vals.append(np.linalg.norm(pts[tips[i]]-pts[tips[j]]))
    return np.array(vals, np.float32)

def hand_full88(pts):
    ptsn = normalize_landmarks(pts); 
    return np.concatenate([ptsn.flatten(), joint_angles(ptsn), pairwise_fingertip_dists(ptsn)],0).astype(np.float32)

def hand_coords63(pts, mode="centered", flip_x=False):
    p = pts.copy()
    if flip_x: p[:,0] = 1.0 - p[:,0]
    if mode == "raw":
        return p.flatten().astype(np.float32)       # absolute 0..1
    else:
        return normalize_landmarks(p).flatten().astype(np.float32)  # relative to wrist

def aggregate_window(W):
    mean = W.mean(0); std = W.std(0); d1 = np.diff(W,0).mean(0)
    return np.concatenate([mean,std,d1],0).astype(np.float32)

# ----------------------------- labels -> display -----------------------------
def make_display_labeler(label_map):
    def display(raw):
        if raw in label_map: return label_map[raw]
        s = str(raw)
        if s in label_map: return label_map[s]
        try:
            if isinstance(raw,(np.integer,int)) and 32 <= int(raw) <= 126:
                return chr(int(raw))
        except: pass
        return s
    return display

# ----------------------------- drawing helpers -----------------------------
def draw_prob_bars(frame, probs, labels, origin=(10,70), bar_w=240, bar_h=22, gap=8):
    if probs is None or labels is None: return
    x,y = origin
    for i,(p,lab) in enumerate(zip(probs, labels)):
        top = y + i*(bar_h+gap)
        cv2.rectangle(frame, (x,top), (x+bar_w, top+bar_h), (40,40,40), 1)
        fill = int(bar_w * float(p))
        cv2.rectangle(frame, (x,top), (x+fill, top+bar_h), (60,160,60), -1)
        txt = f"{lab}: {float(p):.2f}"
        cv2.putText(frame, txt, (x+bar_w+10, top+bar_h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

# ----------------------------- runtime -----------------------------
MIN_CONF_SAY = 0.75
SMOOTH_K = 7

def main():
    args = parse_args()
    DRAW = not args.no_draw; WIN = args.win
    labels_path = args.labels or (args.model + ".labels.pkl")

    # model + labels
    clf, kind = load_any_model(args.model)
    print(f"[OK] Loaded {kind} model: {args.model}")
    exp_dim = expected_input_dim(clf, kind)
    if exp_dim is None: print("[ERR] Can't determine input dim."); return
    print(f"[INFO] Model expects feature dim: {exp_dim} "
          f"({'coords 63/126' if exp_dim in (63,126) else 'single 88 / win-agg 264' if exp_dim in (88,264) else 'two-hand 176' if exp_dim==176 else 'win-agg 528' if exp_dim==528 else 'custom'})")
    le = load_any_labels(labels_path)
    if le is not None: print(f"[OK] Loaded labels: {labels_path} ({len(getattr(le,'classes_',[]))} classes)")
    label_map = load_label_map(args.label_map) if args.label_map else {}
    display_label = make_display_labeler(label_map)

    # optional: print class order
    if args.print_classes and le is not None:
        print("[INFO] Class order (probs index -> raw -> display):")
        for i, raw in enumerate(le.classes_):
            print(f"  {i}: {raw} -> {display_label(raw)}")

    # camera
    bflag = backend_flag(args.backend)
    cap = cv2.VideoCapture(args.camera, bflag) if bflag is not None else cv2.VideoCapture(args.camera)
    if not cap.isOpened(): print(f"[ERR] Cannot open webcam index {args.camera}"); return
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except: pass
    print("[INFO] Camera opened. Press 'q' or ESC to quit.")
    cv2.namedWindow("Sign Interpreter — real-time", cv2.WINDOW_NORMAL)

    # windows for aggregated modes
    window_both176 = deque(maxlen=WIN)
    window_single88 = deque(maxlen=WIN)
    recent = deque(maxlen=SMOOTH_K)
    last_spoken=None; last_say_t=0.0

    # mediapipe
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

    try:
        while True:
            ok, frame = cap.read()
            if not ok: time.sleep(0.01); continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable=False
            res = hands.process(rgb)
            rgb.flags.writeable=True

            left_pts = right_pts = None
            pairs=[]
            if res.multi_hand_landmarks and res.multi_handedness:
                pairs = list(zip(res.multi_handedness, res.multi_hand_landmarks))
                for handed,lm in pairs:
                    label = handed.classification[0].label
                    pts = np.array([[p.x,p.y,p.z] for p in lm.landmark], np.float32)
                    if args.flip_x: pts[:,0] = 1.0 - pts[:,0]
                    if label == "Left": left_pts = pts
                    else:               right_pts = pts
                if DRAW:
                    for _,lm in pairs:
                        mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                                                  mp_styles.get_default_hand_landmarks_style(),
                                                  mp_styles.get_default_hand_connections_style())

            # per-frame features
            left88  = hand_full88(left_pts)  if left_pts  is not None else None
            right88 = hand_full88(right_pts) if right_pts is not None else None
            both176 = np.concatenate([
                left88  if left88  is not None else np.zeros(88, np.float32),
                right88 if right88 is not None else np.zeros(88, np.float32)
            ],0).astype(np.float32)

            window_both176.append(both176)
            single88 = right88 if right88 is not None else left88
            if single88 is not None: window_single88.append(single88)

            # which hand for 63/88 single-hand modes
            single_pts = None
            if args.preferred_hand == "right":
                single_pts = right_pts if right_pts is not None else left_pts
            elif args.preferred_hand == "left":
                single_pts = left_pts if left_pts is not None else right_pts
            else:  # auto: prefer right if present
                single_pts = right_pts if right_pts is not None else left_pts

            # build input x
            need_warmup=False; x=None
            if exp_dim == 528:
                if len(window_both176) < WIN: need_warmup=True
                else: x = aggregate_window(np.stack(window_both176,0)).reshape(1,-1)
            elif exp_dim == 176:
                x = both176.reshape(1,-1)
            elif exp_dim == 264:
                if len(window_single88) < WIN: need_warmup=True
                else: x = aggregate_window(np.stack(window_single88,0)).reshape(1,-1)
            elif exp_dim == 126:
                l63 = hand_coords63(left_pts,  args.coords_mode, args.flip_x)  if left_pts  is not None else np.zeros(63, np.float32)
                r63 = hand_coords63(right_pts, args.coords_mode, args.flip_x) if right_pts is not None else np.zeros(63, np.float32)
                x = np.concatenate([l63,r63],0).reshape(1,-1)
            elif exp_dim == 88:
                if single88 is not None: x = single88.reshape(1,-1)
            elif exp_dim == 63:
                if single_pts is not None:
                    x = hand_coords63(single_pts, args.coords_mode, args.flip_x).reshape(1,-1)

            # predict
            pred_label="(no hands)"; pred_prob=0.0; probs=None; disp_labels=None
            if need_warmup:
                pred_label, pred_prob = "warming up…", 0.0
            elif x is not None:
                probs = predict_proba_any(clf, x, kind)[0]
                idx = int(np.argmax(probs))
                pred_prob = float(probs[idx])
                raw = le.inverse_transform([idx])[0] if le is not None else idx
                pred_label = make_display_labeler(label_map)(raw)
                # build display list in class order for bars
                if le is not None:
                    disp_labels = [make_display_labeler(label_map)(r) for r in le.classes_]

            # smoothing + TTS
            if pred_label not in ("warming up…","(no hands)","(error)") and not pred_label.startswith("("):
                recent.append(pred_label)
                if len(recent)==SMOOTH_K:
                    pred_label = Counter(recent).most_common(1)[0][0]
                now = time.time()
                if synth and pred_prob>=MIN_CONF_SAY and (last_spoken!=pred_label) and (now-last_say_t)>0.6:
                    try: synth.stop(); synth.say(pred_label); synth.runAndWait(); last_spoken=pred_label; last_say_t=now
                    except: pass

            # HUD
            hud = f"[{kind}] {os.path.basename(args.model)} | D={exp_dim} | {pred_label}  {pred_prob:.2f}"
            cv2.rectangle(frame, (10,10), (10+max(520,18*len(hud)), 60), (0,0,0), -1)
            cv2.putText(frame, hud, (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

            # probability bars
            if probs is not None and disp_labels is not None:
                draw_prob_bars(frame, probs, disp_labels, origin=(10,70))

            cv2.imshow("Sign Interpreter — real-time", frame)
            key = (cv2.waitKey(1) & 0xFF)
            if key in (27, ord('q')): break
            if cv2.getWindowProperty("Sign Interpreter — real-time", cv2.WND_PROP_VISIBLE) < 1: break
    finally:
        hands.close(); cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
