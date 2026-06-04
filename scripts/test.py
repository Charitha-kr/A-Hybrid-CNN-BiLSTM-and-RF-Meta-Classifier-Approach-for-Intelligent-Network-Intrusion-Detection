# scripts/test.py
import pandas as pd
import pickle
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score

# =========================
# 1. Load Data
# =========================
print("📂 Loading processed test data...")
df = pd.read_csv("data/encoded_data.csv")

# Features (same as training)
X = df.drop(columns=["Label", "Label_encoded", "Binary_Label"])
y_binary = df["Binary_Label"]
y_multi = df["Label_encoded"]

# =========================
# 2. Load Scaler + Encoder
# =========================
print("🔧 Loading scaler and encoder...")
scaler = joblib.load("models/scaler.pkl")  # ensure you saved this during training
X_scaled = scaler.transform(X)

# =========================
# 3. Load Models
# =========================
print("🤖 Loading trained models...")
rf_binary = pickle.load(open("models/random_forest_binary.pkl", "rb"))
rf_multi = pickle.load(open("models/random_forest_multiclass.pkl", "rb"))
cnn_bilstm_binary = load_model("models/cnn_bilstm_binary.h5")
cnn_bilstm_multi = load_model("models/cnn_bilstm_multiclass.h5")

# =========================
# 4. Run Predictions
# =========================
print("\n🔍 Testing Binary Models...")
y_rf_bin = rf_binary.predict(X_scaled)
y_cnn_bin = np.argmax(cnn_bilstm_binary.predict(X_scaled.reshape(len(X_scaled), X_scaled.shape[1], 1)), axis=1)

print("✅ RandomForest (Binary) Accuracy:", accuracy_score(y_binary, y_rf_bin))
print(classification_report(y_binary, y_rf_bin, digits=4))

print("✅ CNN-BiLSTM (Binary) Accuracy:", accuracy_score(y_binary, y_cnn_bin))
print(classification_report(y_binary, y_cnn_bin, digits=4))

print("\n🔍 Testing Multi-class Models...")
y_rf_multi = rf_multi.predict(X_scaled)
y_cnn_multi = np.argmax(cnn_bilstm_multi.predict(X_scaled.reshape(len(X_scaled), X_scaled.shape[1], 1)), axis=1)

print("✅ RandomForest (Multi) Accuracy:", accuracy_score(y_multi, y_rf_multi))
print(classification_report(y_multi, y_rf_multi, digits=4))

print("✅ CNN-BiLSTM (Multi) Accuracy:", accuracy_score(y_multi, y_cnn_multi))
print(classification_report(y_multi, y_cnn_multi, digits=4))
