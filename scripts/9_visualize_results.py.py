import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# =========================
# Load Models
# =========================
print("✅ Loading models...")
rf_model = pickle.load(open("models/random_forest_binary.pkl", "rb"))
cnn_bilstm_model = load_model("models/cnn_bilstm_multiclass.h5")
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

# =========================
# Load Data
# =========================
df = pd.read_csv("data/encoded_data.csv")
X = df.drop(columns=["Label", "Label_encoded", "Binary_Label"])
y = df["Label"]  # multiclass ground truth

print("✅ Models loaded for visualization")

# =========================
# RF Predictions (Binary)
# =========================
y_rf_pred_binary = rf_model.predict(X)
rf_accuracy = accuracy_score(df["Binary_Label"], y_rf_pred_binary)

# =========================
# CNN-BiLSTM Predictions (Multiclass)
# =========================
y_cnn_pred = cnn_bilstm_model.predict(X.values.reshape(X.shape[0], X.shape[1], 1))
y_cnn_pred_classes = np.argmax(y_cnn_pred, axis=1)
y_true_classes = label_encoder.transform(y)
cnn_accuracy = accuracy_score(y_true_classes, y_cnn_pred_classes)

# =========================
# Meta-Fusion Predictions (Simple Majority Voting)
# =========================
y_meta_pred = []
for i in range(len(X)):
    rf_vote = y_rf_pred_binary[i]
    cnn_vote = y_cnn_pred_classes[i]
    # If RF says attack, trust CNN’s class. If RF says benign, force benign.
    if rf_vote == 0:  
        y_meta_pred.append(label_encoder.transform(["BENIGN"])[0])
    else:
        y_meta_pred.append(cnn_vote)

y_meta_pred = np.array(y_meta_pred)
fusion_accuracy = accuracy_score(y_true_classes, y_meta_pred)

# =========================
# Reports
# =========================
print("\n✅ Random Forest (Binary) Accuracy:", rf_accuracy)
print("✅ CNN-BiLSTM (Multiclass) Accuracy:", cnn_accuracy)
print("✅ Meta-Fusion Accuracy:", fusion_accuracy)

print("\n📊 Meta-Fusion Classification Report:")
print(classification_report(
    y_true_classes, y_meta_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))

# =========================
# Confusion Matrix (Raw + Normalized)
# =========================
cm = confusion_matrix(y_true_classes, y_meta_pred)

# Raw confusion matrix
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Meta-Fusion Confusion Matrix (Counts)", fontsize=16)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("predictions/confusion_matrix_counts.png")
plt.show()

# Normalized confusion matrix
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(14, 10))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Meta-Fusion Confusion Matrix (Normalized)", fontsize=16)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("predictions/confusion_matrix_normalized.png")
plt.show()
