import pandas as pd
import numpy as np
import pickle, joblib
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# =========================
# 1. Load Data
# =========================
print("📂 Loading encoded dataset...")
df = pd.read_csv("data/encoded_data.csv")

X = df.drop(columns=["Label", "Label_encoded", "Binary_Label"])
y_binary = df["Binary_Label"]
y_multi = df["Label_encoded"]

# =========================
# 2. Load Scaler + Models
# =========================
print("🔧 Loading scaler & models...")
scaler = joblib.load("models/scaler.pkl")
X_scaled = scaler.transform(X)

# Base models
rf_binary = pickle.load(open("models/random_forest_binary.pkl", "rb"))
rf_multi = pickle.load(open("models/random_forest_multiclass.pkl", "rb"))
cnn_bin = load_model("models/cnn_bilstm_binary.h5")
cnn_multi = load_model("models/cnn_bilstm_multiclass.h5")

# =========================
# 3. Get Predictions
# =========================
print("🤖 Generating base model predictions...")

# Random Forest
rf_bin_pred = rf_binary.predict_proba(X_scaled)[:, 1]   # probability of attack
rf_multi_pred = rf_multi.predict_proba(X_scaled)

# CNN-BiLSTM
cnn_bin_pred = cnn_bin.predict(X_scaled.reshape(len(X_scaled), X_scaled.shape[1], 1), verbose=0)
cnn_bin_pred = cnn_bin_pred[:, 1]   # probability of attack

cnn_multi_pred = cnn_multi.predict(X_scaled.reshape(len(X_scaled), X_scaled.shape[1], 1), verbose=0)

# =========================
# 4. Build Fusion Dataset
# =========================
print("🔗 Building fusion dataset...")
fusion_binary = np.column_stack([rf_bin_pred, cnn_bin_pred])   # shape: (n_samples, 2)
fusion_multi = np.hstack([rf_multi_pred, cnn_multi_pred])      # shape: (n_samples, n_classes*2)

# =========================
# 5. Train Meta-classifier
# =========================
print("📊 Training meta-classifiers...")

# Logistic Regression as meta-classifier
meta_bin = LogisticRegression(max_iter=1000)
meta_bin.fit(fusion_binary, y_binary)
y_meta_bin = meta_bin.predict(fusion_binary)

meta_multi = LogisticRegression(max_iter=1000)
meta_multi.fit(fusion_multi, y_multi)
y_meta_multi = meta_multi.predict(fusion_multi)

# =========================
# 6. Evaluate Fusion
# =========================
print("\n✅ Hybrid Fusion Results")

print("\nBinary Fusion IDS:")
print("Accuracy:", accuracy_score(y_binary, y_meta_bin))
print(classification_report(y_binary, y_meta_bin, digits=4))

print("\nMulti-class Fusion IDS:")
print("Accuracy:", accuracy_score(y_multi, y_meta_multi))
print(classification_report(y_multi, y_meta_multi, digits=4))
