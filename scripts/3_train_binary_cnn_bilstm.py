import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical

# =========================
# Load dataset
# =========================
df = pd.read_csv("data/encoded_data.csv")

X = df.drop(columns=["Label", "Label_encoded", "Binary_Label"])
y = df["Binary_Label"]   # ✅ binary target (0=Benign, 1=Attack)

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for Conv1D/LSTM [samples, timesteps, features]
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# =========================
# Compute Class Weights
# =========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("⚖️ Class Weights:", class_weights)


# Build CNN+BiLSTM
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(
    X_train, y_train,
    epochs=10, batch_size=64,
    validation_data=(X_test, y_test),
    class_weight=class_weights,   # ✅ balance attacks vs benign
    verbose=1
)
