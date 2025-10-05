# scripts/build_manifest.py
import json, csv, os, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--in_json", required=True)
parser.add_argument("--out_csv", default="manifest.csv")
parser.add_argument("--subset_max_label", type=int, default=None,
                    help="optional: keep only samples with label < N (for MS-ASL100/200 subsets)")
args = parser.parse_args()

with open(args.in_json, "r") as f:
    data = json.load(f)

rows = []
for i, item in enumerate(data):
    label = item.get("label")
    if args.subset_max_label is not None and label >= args.subset_max_label:
        continue
    clip_id = f"{i:06d}_L{label}_S{item.get('signer_id', item.get('signer',0))}"
    rows.append({
        "clip_id": clip_id,
        "url": item["url"],
        "start_time": item["start_time"],
        "end_time": item["end_time"],
        "label": label,
        "text": item.get("text", item.get("clean_text","")),
        "box0": item.get("box",[0,0,1,1])[0],
        "box1": item.get("box",[0,0,1,1])[1],
        "box2": item.get("box",[0,0,1,1])[2],
        "box3": item.get("box",[0,0,1,1])[3],
        "width": item.get("width", 640),
        "height": item.get("height", 360),
        "fps": item.get("fps", 30.0),
        "file_field": item.get("file","")
    })

os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
with open(args.out_csv, "w", newline="", encoding="utf8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("Wrote", args.out_csv, "with", len(rows), "rows")
