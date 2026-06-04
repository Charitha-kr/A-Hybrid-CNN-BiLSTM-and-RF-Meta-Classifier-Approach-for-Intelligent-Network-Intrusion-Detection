import pickle
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Load dataset (to get feature names)
# =========================
df = pd.read_csv("data/encoded_data.csv")
X = df.drop(columns=["Label", "Label_encoded", "Binary_Label"])
feature_names = X.columns.tolist()

# =========================
# Load trained models
# =========================
with open("models/random_forest_binary.pkl", "rb") as f:
    rf_binary = pickle.load(f)

with open("models/random_forest_multiclass.pkl", "rb") as f:
    rf_multi = pickle.load(f)

print("✅ Loaded both Random Forest models")

# =========================
# Get feature importance
# =========================
fi_binary = rf_binary.feature_importances_
fi_multi = rf_multi.feature_importances_

# Put in dataframe for easy handling
df_importance = pd.DataFrame({
    "Feature": feature_names,
    "Binary_RF": fi_binary,
    "Multiclass_RF": fi_multi
})

# Sort by average importance across both models
df_importance["Avg"] = (df_importance["Binary_RF"] + df_importance["Multiclass_RF"]) / 2
df_importance = df_importance.sort_values("Avg", ascending=False).head(20)  # top 20 features

# =========================
# Plot
# =========================
plt.figure(figsize=(12, 8))

# Bar plot (side-by-side for binary vs multiclass)
width = 0.35
x = range(len(df_importance))

plt.bar([i - width/2 for i in x], df_importance["Binary_RF"], width=width, label="Binary RF")
plt.bar([i + width/2 for i in x], df_importance["Multiclass_RF"], width=width, label="Multiclass RF")

plt.xticks(x, df_importance["Feature"], rotation=75, ha="right")
plt.ylabel("Feature Importance")
plt.title("Top 20 Feature Importances (Binary vs Multiclass Random Forest)")
plt.legend()
plt.tight_layout()

# Save & Show
plt.savefig("plots/rf_feature_importance_comparison.png", dpi=300)
plt.show()

print("📊 Saved plot to plots/rf_feature_importance_comparison.png")
