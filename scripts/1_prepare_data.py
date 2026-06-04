import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================
# Load processed dataset
# =========================
df = pd.read_csv("data/processed_data.csv")

print("✅ Loaded processed_data.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# =========================
# Encode labels
# =========================
label_encoder = LabelEncoder()
df['Label_encoded'] = label_encoder.fit_transform(df['Label'])

# Binary label: 0 = Normal (BENIGN), 1 = Attack
df['Binary_Label'] = df['Label'].apply(lambda x: 0 if x == "BENIGN" else 1)

# Save encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Label encoding done. Classes:", list(label_encoder.classes_))

# =========================
# Scale numerical features
# =========================
desired_features = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean", "Flow Bytes/s",
    "Flow Packets/s", "Fwd IAT Mean", "Bwd IAT Mean", "Packet Length Mean",
    "Source Port", "Destination Port", "Protocol"
]

# Only keep features that exist in the current dataset
feature_cols = [col for col in desired_features if col in df.columns]

print(f"⚡ Scaling these features: {feature_cols}")

scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# =========================
# Save final encoded dataset
# =========================
df.to_csv("data/encoded_data.csv", index=False)
print("✅ Encoded + scaled dataset saved as data/encoded_data.csv")
