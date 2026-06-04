import os
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

# =========================
# 1. Load Data
# =========================
print("📂 Loading encoded dataset...")
df = pd.read_csv("data/encoded_data.csv")

X = df.drop(columns=["Label", "Label_encoded", "Binary_Label"])
y_binary = df["Binary_Label"]
y_multi = df["Label_encoded"]

# Load encoders
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
scaler = joblib.load("models/scaler.pkl")

X_scaled = scaler.transform(X)
X_seq = np.expand_dims(X_scaled, -1)

# =========================
# 2. Load Models
# =========================
print("🔧 Loading trained models...")
rf_binary = pickle.load(open("models/random_forest_binary.pkl", "rb"))
rf_multi = pickle.load(open("models/random_forest_multiclass.pkl", "rb"))
cnn_bin = load_model("models/cnn_bilstm_binary.h5")
cnn_multi = load_model("models/cnn_bilstm_multiclass.h5")

# =========================
# 3. Base Predictions
# =========================
print("🤖 Generating base model predictions...")

# Binary
rf_bin_probs = rf_binary.predict_proba(X_scaled)[:, 1].reshape(-1, 1)
cnn_bin_probs = cnn_bin.predict(X_seq, batch_size=256, verbose=0)[:, 1].reshape(-1, 1)

# Multi-class
rf_multi_probs = rf_multi.predict_proba(X_scaled)
cnn_multi_probs = cnn_multi.predict(X_seq, batch_size=256, verbose=0)

# 4. Build Fusion Features
print(" Building fusion dataset...")

fusion_binary = np.hstack([rf_bin_probs, cnn_bin_probs])        # shape: (n_samples, 2)
fusion_multi = np.hstack([rf_multi_probs, cnn_multi_probs])     # shape: (n_samples, 2*n_classes)

# 5. Train Meta-classifiers
print("Training meta-classifiers...")

meta_bin = LogisticRegression(max_iter=1000)
meta_bin.fit(fusion_binary, y_binary)
y_meta_bin = meta_bin.predict(fusion_binary)

meta_multi = LogisticRegression(max_iter=1000, multi_class="auto")
meta_multi.fit(fusion_multi, y_multi)
y_meta_multi = meta_multi.predict(fusion_multi)

# =========================
# 6. Evaluate
# =========================
print("\n✅ Meta-Fusion Evaluation")

print("\n--- Binary Fusion ---")
print("Accuracy:", accuracy_score(y_binary, y_meta_bin))
print(classification_report(y_binary, y_meta_bin, digits=4))

print("\n--- Multi-class Fusion ---")
print("Accuracy:", accuracy_score(y_multi, y_meta_multi))
print(classification_report(y_multi, y_meta_multi, digits=4))

# Confusion matrix (optional but useful)
print("\nConfusion Matrix (Multi-class):\n", confusion_matrix(y_multi, y_meta_multi))

# =========================
# 7. Save Predictions
# =========================
os.makedirs("predictions", exist_ok=True)

pd.DataFrame({
    "True_Label": label_encoder.inverse_transform(y_multi),
    "Predicted_Label": label_encoder.inverse_transform(y_meta_multi)
}).to_csv("predictions/fusion_predictions.csv", index=False, encoding="utf-8")

print("\n🎉 Saved fusion predictions to predictions/fusion_predictions.csv")
