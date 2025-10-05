# scripts/train_sequence_model.py
import numpy as np, glob, os, argparse
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--features_dir", default="features")
parser.add_argument("--out_model", default="models/asl_seq_model.h5")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--val_split", type=float, default=0.15)
args = parser.parse_args()

files = glob.glob(os.path.join(args.features_dir, "*.npz"))
X = []
y = []
for f in files:
    d = np.load(f, allow_pickle=True)
    X.append(d["X"])   # shape (T,63)
    y.append(int(d["label"]))
X = np.stack(X)  # (N,T,63)
y = np.array(y)
print("Loaded:", X.shape, y.shape)

le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)
y_cat = tf.keras.utils.to_categorical(y_enc, num_classes)

# shuffle and split
idx = np.arange(len(X))
np.random.shuffle(idx)
X = X[idx]; y_cat = y_cat[idx]
split = int(len(X)*(1-args.val_split))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y_cat[:split], y_cat[split:]
T = X.shape[1]; D = X.shape[2]

inp = layers.Input(shape=(T,D))
x = layers.Masking(mask_value=0.0)(inp)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inp, out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

ckp = callbacks.ModelCheckpoint(args.out_model, save_best_only=True, monitor="val_loss")
es = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=args.epochs, batch_size=args.batch_size,
                    callbacks=[ckp, es])

os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
model.save(args.out_model)
with open(args.out_model + ".labels.pkl","wb") as f: pickle.dump(le, f)
print("Saved model and label encoder")
