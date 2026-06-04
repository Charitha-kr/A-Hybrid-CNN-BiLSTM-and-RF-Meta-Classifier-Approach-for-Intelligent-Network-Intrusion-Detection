# precision_recall_curve_plot.py
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split

# =========================
# Load dataset
# =========================
df = pd.read_csv("data/encoded_data.csv")

X = df.drop(columns=["Label", "Label_encoded", "Binary_Label"])
y = df["Binary_Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Load trained Random Forest (Binary)
# =========================
with open("models/random_forest_binary.pkl", "rb") as f:
    rf = pickle.load(f)

# =========================
# Predict probabilities
# =========================
y_scores = rf.predict_proba(X_test)[:, 1]  # probability of class "1" (attack)

# =========================
# Precision-Recall Curve
# =========================
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
avg_precision = average_precision_score(y_test, y_scores)

# =========================
# Plot
# =========================
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, marker=".", label=f'Random Forest (AP = {avg_precision:.4f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Binary IDS)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/precision_recall_curve.png", dpi=300)
plt.show()

print("🎉 Precision-Recall curve saved as results/precision_recall_curve.png")
