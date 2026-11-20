
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    average_precision_score
)

# 1. Load dataset
df = pd.read_csv("data/raw/ai4i/ai4i2020.csv")

# 2. Sensor input features
FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# 3. Failure types (prediction targets)
TARGETS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

X = df[FEATURES]
y = df[TARGETS]

# 4. Split data
# Stratify using machine failure to preserve the overall failure ratio.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=df["Machine failure"]
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 5. Create multi-output Random Forest
base_model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model = MultiOutputClassifier(base_model, n_jobs=-1)

# 6. Train model
print("\nTraining failure diagnosis model...")
model.fit(X_train, y_train)

# 7. Predict failure types
predictions = model.predict(X_test)

# 8. Evaluate each failure type
print("\n========== FAILURE DIAGNOSIS RESULTS ==========")

results = []

for i, target in enumerate(TARGETS):
    actual = y_test.iloc[:, i]
    predicted = predictions[:, i]

    precision = precision_score(
        actual, predicted, zero_division=0
    )
    recall = recall_score(
        actual, predicted, zero_division=0
    )
    f1 = f1_score(
        actual, predicted, zero_division=0
    )

    try:
        probabilities = model.predict_proba(X_test)[i][:, 1]
        pr_auc = average_precision_score(actual, probabilities)
    except ValueError:
        pr_auc = np.nan

    results.append({
        "Failure Type": target,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "PR-AUC": pr_auc,
        "Test Positives": int(actual.sum())
    })

    print(f"\n{target}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Test positive cases: {int(actual.sum())}")

# 9. Save evaluation results
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv(
    "reports/failure_diagnosis_results.csv",
    index=False
)

# 10. Save model and feature information
joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "targets": TARGETS
    },
    "models/failure_diagnosis_model.pkl"
)

print("\nDiagnosis model saved successfully!")
print("Model: models/failure_diagnosis_model.pkl")
print("Results: reports/failure_diagnosis_results.csv")
