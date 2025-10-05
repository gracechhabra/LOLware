import numpy as np
import glob
import os
import json

files = glob.glob("features/*.npz")

X_list, y_list, text_list = [], [], []

for f in files:
    data = np.load(f, allow_pickle=True)
    X_file = data["X"]
    label = data["label"]
    text = data["text"]

    # Repeat label/text for each sample in X_file
    y_file = np.full(len(X_file), label)
    text_file = np.full(len(X_file), text)

    X_list.append(X_file)
    y_list.append(y_file)
    text_list.append(text_file)

# Stack arrays
X = np.vstack(X_list)
y = np.hstack(y_list)
texts = np.hstack(text_list)

# Build classes list (unique text labels)
unique_classes = sorted(set(texts.tolist()))

print("✅ Unique classes found:", unique_classes)

# Save merged dataset
os.makedirs("data", exist_ok=True)
np.savez("data/keypoints_asl.npz", X=X, y=y, text=texts, classes=np.array(json.dumps(unique_classes)))

print("✅ Merged dataset saved as data/keypoints_asl.npz")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("text shape:", texts.shape)
