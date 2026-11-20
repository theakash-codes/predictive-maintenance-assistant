
import os
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. Load the already prepared dataset
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_test = pd.read_csv("data/processed/y_test.csv")[
    "Machine failure"
]

# 2. Create the Isolation Forest pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("isolation_forest", IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    ))
])

# 3. Train using training sensor data only
print("Training Isolation Forest...")

model.fit(X_train)

# 4. Predict anomalies
# Isolation Forest returns:
#  1 = Normal
# -1 = Anomaly

predictions = model.predict(X_test)

anomaly_flags = (predictions == -1).astype(int)

# 5. Calculate anomaly scores
# Higher score means more anomalous
anomaly_scores = -model.decision_function(X_test)

# 6. Create results dataframe
results = X_test.copy()

results["Actual Machine Failure"] = y_test.values
results["Anomaly Prediction"] = anomaly_flags
results["Anomaly Score"] = anomaly_scores

results["Status"] = results["Anomaly Prediction"].map({
    0: "Normal",
    1: "Anomaly"
})

# 7. Display summary
total = len(results)
anomalies = int(anomaly_flags.sum())
normal = total - anomalies

print("\n========== ANOMALY DETECTION RESULTS ==========")

print("Total test records:", total)
print("Normal records:", normal)
print("Anomalies detected:", anomalies)

print(
    "Anomaly percentage:",
    round((anomalies / total) * 100, 2),
    "%"
)

# 8. Exploratory comparison with actual failure labels
print("\n========== COMPARISON WITH MACHINE FAILURE ==========")

print(pd.crosstab(
    results["Actual Machine Failure"],
    results["Anomaly Prediction"],
    rownames=["Actual Failure"],
    colnames=["Predicted Anomaly"]
))

precision = precision_score(
    y_test, anomaly_flags, zero_division=0
)

recall = recall_score(
    y_test, anomaly_flags, zero_division=0
)

f1 = f1_score(
    y_test, anomaly_flags, zero_division=0
)

print("\nExploratory metrics:")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

# 9. Save model
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(
    model,
    "models/isolation_forest_model.pkl"
)

# 10. Save predictions
results.to_csv(
    "reports/anomaly_detection_results.csv",
    index=False
)

print("\nAnomaly detection model saved!")
print("Model: models/isolation_forest_model.pkl")
print("Report: reports/anomaly_detection_results.csv")
