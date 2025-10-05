# scripts/download_and_cut.py
import csv, os, subprocess, argparse
from yt_dlp import YoutubeDL
from pathlib import Path
import shlex

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="manifest.csv")
parser.add_argument("--out_dir", default="clips")
parser.add_argument("--reencode", action="store_true",
                    help="re-encode when cutting (more reliable), otherwise use stream copy")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
os.makedirs("downloads", exist_ok=True)

ydl_opts = {"outtmpl": "downloads/%(id)s.%(ext)s", "noplaylist": True}
ydl = YoutubeDL(ydl_opts)

with open(args.manifest, newline="", encoding="utf8") as f:
    r = csv.DictReader(f)
    for row in r:
        clip_id = row["clip_id"]
        out_mp4 = os.path.join(args.out_dir, f"{clip_id}.mp4")
        if os.path.exists(out_mp4):
            print("exists:", out_mp4)
            continue

        url = row["url"]
        start = float(row["start_time"])
        end = float(row["end_time"])
        dur = max(0.1, end - start)
        width = int(float(row.get("width", 640)))
        height = int(float(row.get("height", 360)))

        # Handle bounding box
        if "box" in row and row["box"]:
            import ast
            box = ast.literal_eval(row["box"])
        elif all(k in row for k in ["box0","box1","box2","box3"]):
            box = [float(row["box0"]), float(row["box1"]), float(row["box2"]), float(row["box3"])]
        else:
            box = [0.0, 0.0, 1.0, 1.0]  # fallback to full frame

        # convert normalized coords to pixels
        x0 = max(0, min(int(box[1] * width), width - 1))
        y0 = max(0, min(int(box[0] * height), height - 1))
        x1 = max(0, min(int(box[3] * width), width))
        y1 = max(0, min(int(box[2] * height), height))
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)

        try:
            info = ydl.extract_info(url, download=True)
            vid_id = info.get("id")
            ext = info.get("ext", "mp4")
            downloaded = f"downloads/{vid_id}.{ext}"
            if not Path(downloaded).exists():
                print("download failed:", downloaded)
                continue

            crop_filter = f"crop={w}:{h}:{x0}:{y0},scale=224:224"
            print(f"Processing {clip_id}: start={start:.2f}s dur={dur:.2f}s crop={w}x{h}+{x0}+{y0}")

            if args.reencode:
                cmd = f'ffmpeg -y -i "{downloaded}" -ss {start} -t {dur} -vf "{crop_filter}" -c:v libx264 -preset veryfast -crf 23 -c:a aac "{out_mp4}"'
            else:
                cmd = f'ffmpeg -y -i "{downloaded}" -ss {start} -t {dur} -vf "{crop_filter}" -c:v libx264 -c:a aac "{out_mp4}"'

            print("RUN:", cmd)
            subprocess.check_call(shlex.split(cmd))

        except Exception as e:
            print("ERR", clip_id, e)
