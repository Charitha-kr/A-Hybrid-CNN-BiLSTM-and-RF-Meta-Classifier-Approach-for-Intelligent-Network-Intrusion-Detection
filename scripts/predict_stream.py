import time
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# =========================
# Load models + tools
# =========================
with open("models/random_forest_binary.pkl", "rb") as f:
    rf_binary = pickle.load(f)
cnn_bilstm = load_model("models/cnn_bilstm_multiclass.h5")

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("✅ Models loaded for real-time prediction")

# =========================
# Stream file
# =========================
STREAM_PATH = "data/stream.csv"
last_seen = 0  # line counter

BENIGN_CLASS = label_encoder.transform(["BENIGN"])[0]

while True:
    try:
        # Read new rows only
        df = pd.read_csv(STREAM_PATH)
        if len(df) <= last_seen:
            time.sleep(1)
            continue

        new_data = df.iloc[last_seen:]
        last_seen = len(df)

        # Preprocess
        X_scaled = scaler.transform(new_data)
        X_seq = np.expand_dims(X_scaled, -1)

        # Predict
        rf_preds = rf_binary.predict(X_scaled)
        dl_preds = np.argmax(cnn_bilstm.predict(X_seq, verbose=0), axis=1)

        # Fusion: if RF says attack, trust CNN, else BENIGN
        final_preds = []
        for rf, dl in zip(rf_preds, dl_preds):
            if rf == 1:
                final_preds.append(dl)
            else:
                final_preds.append(BENIGN_CLASS)

        labels = label_encoder.inverse_transform(final_preds)

        for label in labels:
            print(f"🚨 Predicted: {label}")

    except Exception as e:
        print("⚠️ Error:", e)

    time.sleep(2)  # check for new data periodically
