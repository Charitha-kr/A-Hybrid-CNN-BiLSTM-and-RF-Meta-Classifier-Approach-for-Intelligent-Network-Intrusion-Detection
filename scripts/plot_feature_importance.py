import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Load dataset (for feature names)
# =========================
df = pd.read_csv("data/encoded_data.csv")
X = df.drop(columns=["Label", "Label_encoded", "Binary_Label"])

# =========================
# Load trained models
# =========================
with open("models/random_forest_binary.pkl", "rb") as f:
    rf_binary = pickle.load(f)

with open("models/random_forest_multiclass.pkl", "rb") as f:
    rf_multiclass = pickle.load(f)

# =========================
# Feature importance extraction
# =========================
features = X.columns.tolist()
importance_binary = rf_binary.feature_importances_
importance_multiclass = rf_multiclass.feature_importances_

# Normalize for comparison
importance_binary = importance_binary / np.sum(importance_binary)
importance_multiclass = importance_multiclass / np.sum(importance_multiclass)

# =========================
# Plot side-by-side comparison
# =========================
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(features))
width = 0.35

ax.bar(x - width/2, importance_binary, width, label="Binary RF")
ax.bar(x + width/2, importance_multiclass, width, label="Multiclass RF")

ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45, ha="right")
ax.set_ylabel("Normalized Importance")
ax.set_title("Feature Importance: Binary vs Multiclass Random Forest")
ax.legend()

plt.tight_layout()
plt.show()
