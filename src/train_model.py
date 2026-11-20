
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# Machine Learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Model evaluation
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# XGBoost
from xgboost import XGBClassifier


# ==========================================
# 1. PROJECT PATHS
# ==========================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading dataset...")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 3. FEATURE SELECTION
# ==========================================

categorical_features = ["Type"]

numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

FEATURES = categorical_features + numerical_features

# Validate input columns
if list(X_train.columns) != FEATURES:
    raise ValueError("Unexpected training feature columns.")

if list(X_test.columns) != FEATURES:
    raise ValueError("Unexpected testing feature columns.")


# ==========================================
# 4. DATA PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)


# ==========================================
# 5. DEFINE FOUR ML MODELS
# ==========================================

negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())

if positive == 0:
    raise ValueError("No positive failure cases in training data.")

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=negative / positive,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    ),

    "Support Vector Machine": SVC(
        kernel="rbf",
        C=1.0,
        class_weight="balanced",
        probability=True,
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
    class_weight="balanced",
    solver="liblinear",
    C=0.1,
    max_iter=5000,
    random_state=42
    )
}


# ==========================================
# 6. CROSS VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "pr_auc": "average_precision"
}

results = []
pipelines = {}

print("\n===================================")
print("       MODEL TRAINING STARTED")
print("===================================")


for name, model in models.items():

    print(f"\nTraining: {name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    result = {
        "Model": name,
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1 Score": scores["test_f1"].mean(),
        "PR-AUC": scores["test_pr_auc"].mean()
    }

    results.append(result)
    pipelines[name] = pipeline

    print("Precision:", round(result["Precision"], 4))
    print("Recall:", round(result["Recall"], 4))
    print("F1 Score:", round(result["F1 Score"], 4))
    print("PR-AUC:", round(result["PR-AUC"], 4))


# ==========================================
# 7. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="PR-AUC",
    ascending=False
).reset_index(drop=True)

print("\n===================================")
print("         MODEL COMPARISON")
print("===================================")

print(results_df.to_string(index=False))

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)


# ==========================================
# 8. SELECT BEST MODEL
# ==========================================

best_name = results_df.iloc[0]["Model"]

best_pipeline = pipelines[best_name]

print("\nSelected Model:", best_name)

# Train on complete training dataset
best_pipeline.fit(X_train, y_train)


# ==========================================
# 9. FINAL TESTING
# ==========================================

print("\n===================================")
print("          FINAL TESTING")
print("===================================")

# Predict probabilities
y_probability = best_pipeline.predict_proba(X_test)[:, 1]

# Initial classification threshold
threshold = 0.50

y_prediction = (
    y_probability >= threshold
).astype(int)

# Calculate performance
accuracy = accuracy_score(y_test, y_prediction)

precision = precision_score(
    y_test, y_prediction, zero_division=0
)

recall = recall_score(
    y_test, y_prediction, zero_division=0
)

f1 = f1_score(
    y_test, y_prediction, zero_division=0
)

pr_auc = average_precision_score(
    y_test, y_probability
)

roc_auc = roc_auc_score(
    y_test, y_probability
)

print("\nBest Model:", best_name)

print("\nAccuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("PR-AUC:", round(pr_auc, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_prediction,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

cm = confusion_matrix(y_test, y_prediction)

print(cm)


# ==========================================
# 10. SAVE TEST RESULTS
# ==========================================

test_results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_prediction,
    "Failure Probability": y_probability
})

test_results.to_csv(
    REPORT_DIR / "test_predictions.csv",
    index=False
)


# ==========================================
# 11. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_pipeline,
    MODEL_DIR / "failure_prediction_model.pkl"
)

metadata = {
    "model_name": best_name,
    "threshold": threshold,
    "features": FEATURES,
    "metrics": {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc
    }
}

joblib.dump(
    metadata,
    MODEL_DIR / "model_metadata.pkl"
)


# ==========================================
# 12. COMPLETION
# ==========================================

print("\n===================================")
print("      TRAINING COMPLETED")
print("===================================")

print("Best Model:", best_name)

print("\nSaved Files:")
print("1. models/failure_prediction_model.pkl")
print("2. models/model_metadata.pkl")
print("3. reports/model_comparison.csv")
print("4. reports/test_predictions.csv")

print("\nAll operations completed successfully!")
