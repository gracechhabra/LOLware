# scripts/train_model.py
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import argparse
import os

def build_mlp(input_dim, n_classes):
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/keypoints_asl.npz")
    parser.add_argument("--out", default="models/asl_model.h5")
    args = parser.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X = d["X"]
    y = d["y"]
    classes = json.loads(d["classes"].tolist())
    print("Classes:", classes)

    le = LabelEncoder()
    y_int = le.fit_transform(y)
    n_classes = len(le.classes_)

    from tensorflow.keras.utils import to_categorical
    y_cat = to_categorical(y_int, n_classes)

    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.15, random_state=42, stratify=y_int)

    model = build_mlp(X.shape[1], n_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(args.out, save_best_only=True, monitor="val_loss")
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    print("Model saved to", args.out)
    # Save label encoder mapping
    import pickle
    with open(args.out + ".labels.pkl", "wb") as f:
        pickle.dump(le, f)