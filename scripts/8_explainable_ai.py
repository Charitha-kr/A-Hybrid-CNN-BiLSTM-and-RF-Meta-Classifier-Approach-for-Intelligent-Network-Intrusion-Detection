import os
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ======================================================
# PATH SETUP
# ======================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_multiclass.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "encoded_data.csv")

print(f"📂 Model Path: {MODEL_PATH}")
print(f"📄 Data Path: {DATA_PATH}")

# ======================================================
# LOAD MODEL + DATA
# ======================================================
print("✅ Loading model and dataset...")
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

target_cols = ["Label", "Label_encoded", "Binary_Label"]
feature_cols = [c for c in df.columns if c not in target_cols]

# Align features with model
if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    print(f"🧩 Model expects {len(model_features)} features: {model_features}")
    X = df[[col for col in model_features if col in df.columns]].copy()
else:
    X = df[feature_cols].copy()

print(f"✅ Data ready with {X.shape[1]} features.")

# ======================================================
# SAMPLE DATA
# ======================================================
sample_size = min(500, len(X))
X_sample = X.sample(sample_size, random_state=42)

# ======================================================
# SHAP EXPLAINER
# ======================================================
print(" Generating SHAP explainer...")
explainer = shap.TreeExplainer(model)
raw_shap = explainer.shap_values(X_sample)

# ======================================================
# NORMALIZE SHAP FORMAT
# ======================================================
def normalize_shap(raw, n_features):
    """
    Convert SHAP output into a consistent list of (n_samples, n_features).
    Handles both list and numpy array formats automatically.
    """
    if isinstance(raw, list):
        print(f" Detected SHAP list format → {len(raw)} classes")
        return [np.array(sv) for sv in raw]

    if isinstance(raw, np.ndarray):
        print(f" SHAP raw shape: {raw.shape}")
        if raw.ndim == 2:
            # Single-class
            return [raw]

        if raw.ndim == 3:
            n0, n1, n2 = raw.shape

            # Case: (samples, features, classes)
            if n1 == n_features:
                print(" Detected shape (samples, features, classes) → splitting per class")
                return [raw[:, :, c] for c in range(n2)]

            # Case: (classes, samples, features)
            if n2 == n_features:
                print(" Detected shape (classes, samples, features)")
                return [raw[c, :, :] for c in range(n0)]

        print(" Unexpected SHAP shape; flattening to 2D.")
        flat = raw.reshape(raw.shape[0], -1)
        return [flat]

    print(" Unknown SHAP type; treating as single matrix.")
    return [np.array(raw)]

shap_per_class = normalize_shap(raw_shap, n_features=X_sample.shape[1])
n_classes = len(shap_per_class)
print(f"✅ Normalized SHAP → {n_classes} class(es).")

# ======================================================
# FEATURE IMPORTANCE COMPUTATION
# ======================================================
class_means = []
for i, sv in enumerate(shap_per_class):
    sv = np.array(sv)
    if sv.ndim != 2:
        print(f"⚠ Skipping class {i}, SHAP shape {sv.shape}")
        continue
    class_means.append(np.abs(sv).mean(axis=0))

mean_abs_shap = np.mean(class_means, axis=0)

if len(mean_abs_shap) != len(X_sample.columns):
    print(f"⚠ Adjusting SHAP vector length ({len(mean_abs_shap)} → {len(X_sample.columns)})")
    mean_abs_shap = mean_abs_shap[:len(X_sample.columns)]

importance_df = pd.DataFrame({
    "Feature": X_sample.columns,
    "Mean |SHAP|": mean_abs_shap
}).sort_values(by="Mean |SHAP|", ascending=False)

print("✅ SHAP values computed successfully!\n")
print("🔝 Top 5 most influential features:")
print(importance_df.head(5).to_string(index=False))

# ======================================================
# CREATE OUTPUT DIRECTORY
# ======================================================
output_dir = os.path.join(BASE_DIR, "explainability_plots")
os.makedirs(output_dir, exist_ok=True)

# ======================================================
# PLOT 1: GLOBAL FEATURE IMPORTANCE
# ======================================================
print("📊 Generating global feature importance plot...")
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Mean |SHAP|"], color="teal")
plt.gca().invert_yaxis()
plt.xlabel("Mean |SHAP value| (Importance)")
plt.title("Global Feature Importance (Random Forest Multiclass)")
plt.tight_layout()
bar_path = os.path.join(output_dir, "global_feature_importance.png")
plt.savefig(bar_path, dpi=300)
plt.close()
print(f"✅ Saved → {bar_path}")

# ======================================================
# PLOT 2: CLASS vs FEATURE HEATMAP
# ======================================================
if n_classes > 1:
    print("🔥 Building class vs feature heatmap...")

    # Collect mean |SHAP| per class-feature pair
    per_class_importance = []
    valid_class_names = []

    if hasattr(model, "classes_"):
        class_names = [str(c) for c in model.classes_]
    else:
        class_names = [f"Class_{i}" for i in range(n_classes)]

    for i, sv in enumerate(shap_per_class):
        if sv.ndim == 2 and sv.shape[1] == len(X_sample.columns):
            per_class_importance.append(np.abs(sv).mean(axis=0))
            valid_class_names.append(class_names[i])

    if per_class_importance:
        heatmap_data = np.vstack(per_class_importance)
        plt.figure(figsize=(12, 6))
        plt.imshow(heatmap_data, aspect="auto", cmap="viridis")
        plt.colorbar(label="Mean |SHAP value|")
        plt.yticks(range(len(valid_class_names)), valid_class_names)
        plt.xticks(range(len(X_sample.columns)), X_sample.columns, rotation=45, ha="right")
        plt.title("Feature Importance per Attack Class (Random Forest + SHAP)")
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, "class_feature_heatmap.png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"✅ Saved → {heatmap_path}")
    else:
        print("⚠ No valid SHAP matrices found for heatmap.")
else:
    print("ℹ Single-class model detected; skipping heatmap.")

print("\n🎉 Explainability analysis complete!")
print(f"📁 All results saved in: {output_dir}")