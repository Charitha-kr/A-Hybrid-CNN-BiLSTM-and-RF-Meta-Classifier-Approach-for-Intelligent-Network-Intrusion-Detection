import pandas as pd
import time
import os
import random

# Paths
DATA_PATH = "data/encoded_data.csv"
STREAM_PATH = "data/stream.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Drop target columns so simulator only outputs features
features = df.drop(columns=["Label", "Label_encoded", "Binary_Label"])

print(f"✅ Loaded dataset with {len(features)} records.")
print(f"📡 Writing simulated traffic to {STREAM_PATH} ...")

# Create/clear stream file
os.makedirs("data", exist_ok=True)
with open(STREAM_PATH, "w", encoding="utf-8") as f:
    f.write(",".join(features.columns) + "\n")  # write header

# Stream loop
while True:
    # Pick random row
    row = features.sample(1).iloc[0]

    # Append to stream.csv
    with open(STREAM_PATH, "a", encoding="utf-8") as f:
        f.write(",".join(map(str, row.values)) + "\n")

    print(f"🌐 Added new traffic record at {time.strftime('%H:%M:%S')}")

    time.sleep(random.uniform(0.5, 2))  # simulate variable arrival
