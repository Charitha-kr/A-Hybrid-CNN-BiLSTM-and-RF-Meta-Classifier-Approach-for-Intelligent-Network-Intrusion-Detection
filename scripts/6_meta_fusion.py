import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# =========================
# Load test data
# =========================
df = pd.read_csv("data/encoded_data.csv")
X = df.drop(columns=["Label", "Label_encoded", "Binary_Label"])
y = df["Label"]

# Load LabelEncoder + Scaler
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X)
X_seq = np.expand_dims(X_scaled, -1)

# =========================
# Load models
# =========================
with open("models/random_forest_binary.pkl", "rb") as f:
    rf_binary = pickle.load(f)
cnn_bilstm = load_model("models/cnn_bilstm_multiclass.h5")

print("✅ Models loaded: RF (binary) + CNN-BiLSTM (multiclass)")

# =========================
# Predictions
# =========================
y_pred_rf = rf_binary.predict(X_scaled)
y_pred_dl = np.argmax(cnn_bilstm.predict(X_seq, batch_size=256), axis=1)

# Fusion rule:
# if RF says attack (1), trust CNN’s multiclass
# else label as BENIGN
BENIGN_CLASS = label_encoder.transform(["BENIGN"])[0]
y_pred_final = []
for rf_pred, dl_pred in zip(y_pred_rf, y_pred_dl):
    if rf_pred == 1:
        y_pred_final.append(dl_pred)
    else:
        y_pred_final.append(BENIGN_CLASS)
y_pred_final = np.array(y_pred_final)

import os

# =========================
# Evaluation
# =========================
y_true = label_encoder.transform(y)
print("\n✅ Meta-Fusion Classification Report:")
print(classification_report(
    y_true,
    y_pred_final,
    target_names=label_encoder.classes_,
    zero_division=0
))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred_final))

# =========================
# Save fused predictions safely
# =========================
os.makedirs("predictions", exist_ok=True)   # <-- create folder if not exists

pd.DataFrame({
    "True": y,
    "Predicted": label_encoder.inverse_transform(y_pred_final)
}).to_csv("predictions/meta_fusion_predictions.csv", index=False)

print("🎉 Saved meta_fusion_predictions.csv in predictions/")
