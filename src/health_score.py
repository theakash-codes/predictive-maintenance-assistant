
import os
import joblib
import numpy as np
import pandas as pd

# 1. Load trained models
failure_model = joblib.load(
    "models/optimized_failure_model.pkl"
)

anomaly_model = joblib.load(
    "models/isolation_forest_model.pkl"
)

# 2. Load sensor data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_test = pd.read_csv("data/processed/y_test.csv")[
    "Machine failure"
]

# Ensure feature order is consistent
FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X_train = X_train[FEATURES]
X_test = X_test[FEATURES]

# 3. Predict machine failure probability
failure_probabilities = failure_model.predict_proba(
    X_test
)[:, 1]

# Convert probability into percentage
failure_risk = failure_probabilities * 100

# 4. Calculate anomaly scores
train_anomaly_scores = -anomaly_model.decision_function(
    X_train
)

test_anomaly_scores = -anomaly_model.decision_function(
    X_test
)

# 5. Normalize anomaly scores using training reference
sorted_reference = np.sort(train_anomaly_scores)

anomaly_percentile = (
    np.searchsorted(
        sorted_reference,
        test_anomaly_scores,
        side="right"
    ) / len(sorted_reference)
) * 100

# 6. Calculate combined risk
failure_weight = 0.70
anomaly_weight = 0.30

combined_risk = (
    failure_weight * failure_risk
    + anomaly_weight * anomaly_percentile
)

# 7. Calculate health score
health_score = 100 - combined_risk

health_score = np.clip(
    health_score, 0, 100
)

# 8. Assign health status
def get_status(score):

    if score >= 80:
        return "Healthy"

    elif score >= 60:
        return "Monitor"

    elif score >= 40:
        return "Warning"

    else:
        return "Critical"


# 9. Create output dataframe
results = X_test.copy()

results["Failure Probability (%)"] = np.round(
    failure_risk, 2
)

results["Anomaly Percentile (%)"] = np.round(
    anomaly_percentile, 2
)

results["Combined Risk (%)"] = np.round(
    combined_risk, 2
)

results["Health Score"] = np.round(
    health_score, 2
)

results["Health Status"] = [
    get_status(score) for score in health_score
]

# Actual failure is included only for retrospective evaluation
results["Actual Machine Failure"] = y_test.values

# 10. Display results
print("\n========== MACHINE HEALTH SCORING ==========")

print("Total machines evaluated:", len(results))

print("\nHealth Status Distribution:")
print(results["Health Status"].value_counts())

print("\nAverage Health Score:",
      round(results["Health Score"].mean(), 2))

print("\nLowest Health Score Records:")

print(
    results.sort_values("Health Score")[
        [
            "Failure Probability (%)",
            "Anomaly Percentile (%)",
            "Combined Risk (%)",
            "Health Score",
            "Health Status",
            "Actual Machine Failure"
        ]
    ].head(10).to_string(index=False)
)

# 11. Save results
os.makedirs("reports", exist_ok=True)

results.to_csv(
    "reports/machine_health_scores.csv",
    index=False
)

# 12. Save scoring configuration
joblib.dump(
    {
        "failure_weight": failure_weight,
        "anomaly_weight": anomaly_weight,
        "anomaly_reference_scores": sorted_reference,
        "health_thresholds": {
            "healthy": 80,
            "monitor": 60,
            "warning": 40
        },
        "features": FEATURES
    },
    "models/health_scoring_config.pkl"
)

print("\nHealth scoring completed!")
print("Report: reports/machine_health_scores.csv")
print("Configuration: models/health_scoring_config.pkl")
