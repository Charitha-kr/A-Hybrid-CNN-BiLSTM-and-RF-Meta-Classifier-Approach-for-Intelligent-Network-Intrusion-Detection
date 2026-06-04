import pandas as pd
import os
import glob

raw_path = "data/raw/*.csv"

# Desired features (normalized names)
selected_features = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean", "Flow Bytes/s",
    "Flow Packets/s", "Fwd IAT Mean", "Bwd IAT Mean", "Packet Length Mean",
    "Source Port", "Destination Port", "Protocol", "Label"
]

csv_files = glob.glob(raw_path)
print(f"🔎 Found {len(csv_files)} CSV files.")

dfs = []
for file in csv_files:
    try:
        # Load everything
        df = pd.read_csv(file, low_memory=False)

        # Normalize column names: strip spaces, unify casing
        df.columns = df.columns.str.strip().str.replace("�", " ", regex=False)

        # Fix common typos
        df.columns = df.columns.str.replace("FFwd", "Fwd", regex=False)
        df.columns = df.columns.str.replace("Infilteration", "Infiltration", regex=False)

        # Keep only desired ones (if available)
        common = [c for c in selected_features if c in df.columns]
        df = df[common]

        if "Label" not in df.columns:
            # Sometimes label column is named differently
            label_col = [c for c in df.columns if "Label" in c]
            if label_col:
                df.rename(columns={label_col[0]: "Label"}, inplace=True)

        dfs.append(df)
        print(f"✅ Processed {file} -> {df.shape}")

    except Exception as e:
        print(f"⚠️ Skipped {file}: {e}")

# Combine
if dfs:
    final_df = pd.concat(dfs, ignore_index=True)

    # Clean Inf/NaN
    final_df = final_df.replace([float("inf"), -float("inf")], 0)
    final_df = final_df.fillna(0)

    # Save
    os.makedirs("data", exist_ok=True)
    final_df.to_csv("data/processed_data.csv", index=False)

    print("✅ Final processed dataset saved to data/processed_data.csv")
    print("🔎 Final shape:", final_df.shape)
    print("🔎 Class distribution:\n", final_df["Label"].value_counts())
else:
    print("❌ No data processed. Check column names in raw CSVs.")
