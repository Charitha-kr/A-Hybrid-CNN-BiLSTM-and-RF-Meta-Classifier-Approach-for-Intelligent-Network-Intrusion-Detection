import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================
# Load encoded dataset
# =========================
df = pd.read_csv("data/encoded_data.csv")

print("✅ Loaded encoded_data.csv")
print("Shape:", df.shape)

# =========================
# Features and target
# =========================
X = df.drop(columns=["Label", "Label_encoded", "Binary_Label"])
y = df["Binary_Label"]

print("⚡ Features used:", X.columns.tolist())
print("Target distribution:\n", y.value_counts(normalize=True))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=300,          # slightly more trees for stability
    max_depth=20,              # deeper trees → better detection of rare attacks
    min_samples_leaf=2,        # prevents overfitting tiny branches
    class_weight="balanced",   # ✅ handles imbalance
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

#Evaulate
y_pred = rf.predict(X_test)

print("\n✅ Binary Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

# =========================
# Save model
# =========================
with open("models/random_forest_binary.pkl", "wb") as f:
    pickle.dump(rf, f)

print("🎉 Binary Random Forest model saved as models/random_forest_binary.pkl")
