
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path

# ==========================================
# 1. PATHS
# ==========================================

MODEL_PATH = Path("models/optimized_failure_model.pkl")
DATA_PATH = Path("data/processed/X_test.csv")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

print("\nLoading optimized XGBoost model...")

pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

# ==========================================
# 3. LOAD TEST DATA
# ==========================================

X_test = pd.read_csv(DATA_PATH)

if list(X_test.columns) != FEATURES:
    raise ValueError("Unexpected test features.")

# Use a reproducible sample for explanation
X_sample = X_test.sample(
    n=min(500, len(X_test)),
    random_state=42
)

print("Samples selected:", len(X_sample))

# ==========================================
# 4. TRANSFORM INPUT DATA
# ==========================================

X_transformed = preprocessor.transform(X_sample)

# ==========================================
# 5. CREATE SHAP EXPLAINER
# ==========================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(classifier)

raw_shap = explainer(X_transformed)

# Display original sensor values in plots
shap_values = shap.Explanation(
    values=raw_shap.values,
    base_values=raw_shap.base_values,
    data=X_sample.values,
    feature_names=FEATURES
)

print("SHAP explanation generated!")

# ==========================================
# 6. GLOBAL FEATURE IMPORTANCE
# ==========================================

importance = np.abs(
    shap_values.values
).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Mean Absolute SHAP": importance
})

importance_df = importance_df.sort_values(
    by="Mean Absolute SHAP",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance_df.to_string(index=False))

importance_df.to_csv(
    REPORT_DIR / "shap_feature_importance.csv",
    index=False
)

# ==========================================
# 7. SHAP BAR PLOT
# ==========================================

shap.plots.bar(
    shap_values,
    max_display=5,
    show=False
)

plt.title("Global Feature Importance - SHAP")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 8. SHAP BEESWARM PLOT
# ==========================================

shap.plots.beeswarm(
    shap_values,
    max_display=5,
    show=False
)

plt.title("SHAP Feature Impact Analysis")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "shap_beeswarm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 9. LOCAL PREDICTION EXPLANATION
# ==========================================

probabilities = pipeline.predict_proba(
    X_sample
)[:, 1]

best_index = int(np.argmax(probabilities))

print("\n========== SAMPLE PREDICTION ==========")

print("Failure probability:",
      round(probabilities[best_index] * 100, 2), "%")

print("\nSensor readings:")

print(X_sample.iloc[best_index])

print("\nSHAP contribution values:")

for feature, value in zip(
    FEATURES,
    shap_values.values[best_index]
):
    print(f"{feature}: {value:.5f}")

# ==========================================
# 10. WATERFALL EXPLANATION
# ==========================================

shap.plots.waterfall(
    shap_values[best_index],
    max_display=5,
    show=False
)

plt.title("Individual Machine Failure Explanation")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "shap_local_explanation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 11. COMPLETION
# ==========================================

print("\n===================================")
print("     SHAP ANALYSIS COMPLETED")
print("===================================")

print("\nGenerated files:")

print("reports/shap_feature_importance.csv")
print("reports/shap_summary.png")
print("reports/shap_beeswarm.png")
print("reports/shap_local_explanation.png")

print("\nExplainable AI is ready!")
