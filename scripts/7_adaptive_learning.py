# scripts/7_adaptive_learning.py
import os
import time
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_FILE = "predictions/stream_results.csv"
MODEL_FILE = "models/random_forest_adaptive.pkl"

# Load initial model
with open("models/random_forest_multiclass.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Loaded base RandomForest model for adaptive learning")

# Watch for new data
while True:
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)

        if not df.empty:
            print(f"📂 Loaded {len(df)} streaming samples for retraining...")

            # Features/labels
            if "Label_encoded" not in df.columns:
                print("⚠️ No labels available, skipping retrain")
            else:
                X = df.drop(columns=["Label", "Predicted", "Label_encoded"], errors="ignore")
                y = df["Label_encoded"]

                # Small retrain (incremental approach)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model.fit(X_train, y_train)  # retrain
                y_pred = model.predict(X_test)

                print("📊 Adaptive Retrain Report:")
                print(classification_report(y_test, y_pred, zero_division=0))

                # Save updated model
                with open(MODEL_FILE, "wb") as f:
                    pickle.dump(model, f)
                print("🎉 Updated model saved as", MODEL_FILE)

    time.sleep(30)  # check every 30s
