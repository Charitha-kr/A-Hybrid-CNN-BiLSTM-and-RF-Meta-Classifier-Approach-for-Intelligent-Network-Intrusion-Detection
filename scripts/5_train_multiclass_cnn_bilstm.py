import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical

# =========================
# Load data
# =========================
df = pd.read_csv("data/encoded_data.csv")
print("✅ Loaded encoded_data.csv")
print("Shape:", df.shape)

# =========================
# Features and target
# =========================
X = df.drop(columns=["Label", "Label_encoded", "Binary_Label"])
y = df["Label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save LabelEncoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("✅ Saved LabelEncoder")

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Reshape for CNN+LSTM
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# =========================
# Build CNN+BiLSTM
# =========================
model = Sequential([
    Conv1D(64, kernel_size=3, activation="relu", input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(y_categorical.shape[1], activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# =========================
# Train
# =========================
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64)

# =========================
# Save model
# =========================
model.save("models/cnn_bilstm_multiclass.h5")
print("🎉 Multiclass CNN-BiLSTM model saved")
