import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression

# =========================
# Setup
# =========================
os.makedirs("results/reports", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# =========================
# Load Data
# =========================
print("📂 Loading encoded dataset...")
df = pd.read_csv("data/encoded_data.csv")

X = df.drop(columns=["Label", "Label_encoded", "Binary_Label"])
y_binary = df["Binary_Label"]
y_multi = df["Label_encoded"]

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
scaler = joblib.load("models/scaler.pkl")

X_scaled = scaler.transform(X)
X_seq = np.expand_dims(X_scaled, -1)

# =========================
# Load Models
# =========================
print("🔧 Loading models...")
rf_binary = pickle.load(open("models/random_forest_binary.pkl", "rb"))
rf_multi = pickle.load(open("models/random_forest_multiclass.pkl", "rb"))
cnn_bin = load_model("models/cnn_bilstm_binary.h5")
cnn_multi = load_model("models/cnn_bilstm_multiclass.h5")

# =========================
# Base Predictions
# =========================
print("🤖 Running base model predictions...")

# RF
rf_bin_pred = rf_binary.predict(X_scaled)
rf_bin_prob = rf_binary.predict_proba(X_scaled)[:, 1]
rf_multi_pred = rf_multi.predict(X_scaled)

# CNN
cnn_bin_pred = np.argmax(cnn_bin.predict(X_seq, batch_size=256, verbose=0), axis=1)
cnn_bin_prob = cnn_bin.predict(X_seq, batch_size=256, verbose=0)[:, 1]
cnn_multi_pred = np.argmax(cnn_multi.predict(X_seq, batch_size=256, verbose=0), axis=1)

# =========================
# Meta Fusion
# =========================
print("🔗 Building fusion dataset...")
fusion_binary = np.column_stack([rf_bin_prob, cnn_bin_prob])
fusion_multi = np.hstack([
    rf_multi.predict_proba(X_scaled),
    cnn_multi.predict(X_seq, batch_size=256, verbose=0)
])

meta_bin = LogisticRegression(max_iter=1000)
meta_bin.fit(fusion_binary, y_binary)
y_meta_bin = meta_bin.predict(fusion_binary)

meta_multi = LogisticRegression(max_iter=1000, multi_class="auto")
meta_multi.fit(fusion_multi, y_multi)
y_meta_multi = meta_multi.predict(fusion_multi)

# =========================
# Utility: Save Report
# =========================
def save_report(name, y_true, y_pred, labels=None):
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0, digits=4)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Save text report
    with open(f"results/reports/{name}_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # Save confusion matrix plot with annotations
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=True)
    plt.title(f"Confusion Matrix - {name} (Acc={acc:.4f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"results/plots/{name}_cm.png")
    plt.close()

    print(f"✅ Saved {name} results (accuracy={acc:.4f})")


# Binary Models
save_report("rf_binary", y_binary, rf_bin_pred, labels=["BENIGN", "ATTACK"])
save_report("cnn_binary", y_binary, cnn_bin_pred, labels=["BENIGN", "ATTACK"])
save_report("fusion_binary", y_binary, y_meta_bin, labels=["BENIGN", "ATTACK"])

# Multi-class Models
save_report("rf_multiclass", y_multi, rf_multi_pred, labels=label_encoder.classes_)
save_report("cnn_multiclass", y_multi, cnn_multi_pred, labels=label_encoder.classes_)
save_report("fusion_multiclass", y_multi, y_meta_multi, labels=label_encoder.classes_)

# =========================
# ROC Curve for Binary
# =========================
print("\n📈 Generating ROC curve (binary models)...")
fpr_rf, tpr_rf, _ = roc_curve(y_binary, rf_bin_prob)
fpr_cnn, tpr_cnn, _ = roc_curve(y_binary, cnn_bin_prob)
fpr_meta, tpr_meta, _ = roc_curve(y_binary, meta_bin.predict_proba(fusion_binary)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f"RF Binary (AUC={auc(fpr_rf,tpr_rf):.3f})")
plt.plot(fpr_cnn, tpr_cnn, label=f"CNN Binary (AUC={auc(fpr_cnn,tpr_cnn):.3f})")
plt.plot(fpr_meta, tpr_meta, label=f"Fusion Binary (AUC={auc(fpr_meta,tpr_meta):.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Binary IDS Models")
plt.legend()
plt.tight_layout()
plt.savefig("results/plots/binary_models_roc.png")
plt.close()

print("🎉 Simulation complete! Reports and plots saved in results/")
