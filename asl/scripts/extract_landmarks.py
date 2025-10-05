# scripts/extract_landmarks_fixed.py
import cv2, os, argparse, numpy as np
import mediapipe as mp
from pathlib import Path
import csv, ast

parser = argparse.ArgumentParser()
parser.add_argument("--clips_dir", default="clips")
parser.add_argument("--manifest", default="manifest.csv")
parser.add_argument("--out_dir", default="features")
parser.add_argument("--T", type=int, default=32, help="sequence length")
parser.add_argument("--min_detected_ratio", type=float, default=0.6,
                    help="min fraction of frames where a hand must be detected")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load manifest mapping
manifest = {}
with open(args.manifest, newline="", encoding="utf8") as f:
    r = csv.DictReader(f)
    for row in r:
        # Try 'clip_id', fallback to stem of 'file'
        clip_id = row.get("clip_id") or Path(row.get("file","")).stem
        manifest[clip_id] = row

def sample_indices(n, T):
    if n >= T:
        return np.linspace(0, n-1, T, dtype=int)
    else:
        idx = list(range(n)) + [n-1]*(T - n)
        return np.array(idx, dtype=int)

# Process clips
for clip_path in Path(args.clips_dir).glob("*.mp4"):
    clip_id = clip_path.stem
    out_path = Path(args.out_dir) / f"{clip_id}.npz"
    if out_path.exists():
        print(f"✅ Already processed: {clip_id}")
        continue

    if clip_id not in manifest:
        print(f"⚠️ Skipping {clip_id}: not found in manifest")
        continue

    print(f"▶️ Processing {clip_id} ...")
    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    while True:
        ret, fimg = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(fimg, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) == 0:
        print(f"⚠️ No frames found: {clip_id}")
        continue

    indices = sample_indices(len(frames), args.T)
    seq = np.zeros((args.T, 63), dtype=np.float32)
    detected_count = 0

    for i_out, i_frame in enumerate(indices):
        frame = frames[int(i_frame)]
        res = hands.process(frame)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            pts -= pts[0]  # normalize by wrist
            s = np.max(np.abs(pts))
            if s > 0: pts /= s
            seq[i_out] = pts.flatten()
            detected_count += 1
        else:
            seq[i_out] = np.zeros(63, dtype=np.float32)

    ratio = detected_count / args.T
    if ratio < args.min_detected_ratio:
        print(f"⚠️ Skip {clip_id}: detected_ratio={ratio:.2f}")
        continue

    # Save features
    try:
        label = int(manifest[clip_id]["label"])
        text = manifest[clip_id]["text"]
    except KeyError as e:
        print(f"⚠️ Missing label/text for {clip_id}: {e}")
        continue

    np.savez_compressed(out_path, X=seq, label=label, text=text)
    print(f"✅ Saved {out_path}  (label={label}, text={text})")
