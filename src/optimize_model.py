
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

from xgboost import XGBClassifier


# ==========================================
# 1. PATHS
# ==========================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLoading five-feature dataset...")

X_train_full = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train_full = pd.read_csv(
    DATA_DIR / "y_train.csv"
).squeeze()

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).squeeze()

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

if list(X_train_full.columns) != FEATURES:
    raise ValueError("Training features do not match.")

if list(X_test.columns) != FEATURES:
    raise ValueError("Testing features do not match.")

print("Training pool:", len(X_train_full))
print("Locked test set:", len(X_test))


# ==========================================
# 3. TRAIN / VALIDATION SPLIT
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train_full
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))

print("\nTraining failure cases:", int(y_train.sum()))
print("Validation failure cases:", int(y_val.sum()))


# ==========================================
# 4. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            FEATURES
        )
    ]
)


# ==========================================
# 5. MODEL DEFINITIONS
# ==========================================

negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())

if positive == 0:
    raise ValueError("No failure cases in training data.")

imbalance_ratio = negative / positive

models = {

    "Random Forest": (
        RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1
        ),
        {
            "classifier__n_estimators": [200, 300, 500],
            "classifier__max_depth": [None, 10, 15, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", 0.8]
        }
    ),

    "XGBoost": (
        XGBClassifier(
            scale_pos_weight=imbalance_ratio,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1
        ),
        {
            "classifier__n_estimators": [200, 300, 500],
            "classifier__max_depth": [3, 4, 5],
            "classifier__learning_rate": [0.02, 0.05, 0.1],
            "classifier__subsample": [0.7, 0.8, 1.0],
            "classifier__colsample_bytree": [0.7, 0.8, 1.0],
            "classifier__min_child_weight": [1, 3, 5]
        }
    ),

    "Logistic Regression": (
        LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=RANDOM_STATE
        ),
        {
            "classifier__C": [0.01, 0.1, 1, 10, 100]
        }
    )
}


# ==========================================
# 6. CROSS VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE
)

best_estimators = {}
validation_results = []

print("\n====================================")
print("    HYPERPARAMETER OPTIMIZATION")
print("====================================")


# ==========================================
# 7. RANDOMIZED SEARCH
# ==========================================

for name, (model, parameters) in models.items():

    print(f"\nOptimizing {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameters,
        n_iter=8,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        verbose=1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    best_estimators[name] = best_model

    val_probabilities = best_model.predict_proba(
        X_val
    )[:, 1]

    val_predictions = (
        val_probabilities >= 0.50
    ).astype(int)

    result = {
        "Model": name,
        "CV PR-AUC": search.best_score_,
        "Validation PR-AUC": average_precision_score(
            y_val, val_probabilities
        ),
        "Validation Precision": precision_score(
            y_val, val_predictions, zero_division=0
        ),
        "Validation Recall": recall_score(
            y_val, val_predictions, zero_division=0
        ),
        "Validation F1": f1_score(
            y_val, val_predictions, zero_division=0
        )
    }

    validation_results.append(result)

    print("\nBest parameters:")
    print(search.best_params_)

    print("Validation PR-AUC:",
          round(result["Validation PR-AUC"], 4))


# ==========================================
# 8. COMPARE MODELS
# ==========================================

results_df = pd.DataFrame(validation_results)

results_df = results_df.sort_values(
    by="Validation PR-AUC",
    ascending=False
).reset_index(drop=True)

print("\n====================================")
print("       VALIDATION COMPARISON")
print("====================================")

print(results_df.to_string(index=False))

results_df.to_csv(
    REPORT_DIR / "optimized_model_comparison.csv",
    index=False
)


# ==========================================
# 9. SELECT BEST MODEL
# ==========================================

best_name = results_df.iloc[0]["Model"]

best_model = best_estimators[best_name]

print("\nSelected model:", best_name)


# ==========================================
# 10. OPTIMIZE CLASSIFICATION THRESHOLD
# ==========================================

val_probabilities = best_model.predict_proba(
    X_val
)[:, 1]

precision_values, recall_values, thresholds = (
    precision_recall_curve(
        y_val,
        val_probabilities
    )
)

f1_values = (
    2 * precision_values[:-1] * recall_values[:-1]
    /
    (
        precision_values[:-1] +
        recall_values[:-1] + 1e-10
    )
)

best_index = int(np.argmax(f1_values))

best_threshold = float(thresholds[best_index])

print("\n====================================")
print("       THRESHOLD OPTIMIZATION")
print("====================================")

print("Selected threshold:", round(best_threshold, 4))
print("Validation precision:",
      round(precision_values[best_index], 4))
print("Validation recall:",
      round(recall_values[best_index], 4))
print("Validation F1:",
      round(f1_values[best_index], 4))


# ==========================================
# 11. RETRAIN ON COMPLETE TRAINING POOL
# ==========================================

print("\nRetraining selected model on all 8000 records...")

best_model.fit(X_train_full, y_train_full)


# ==========================================
# 12. FINAL TEST EVALUATION
# ==========================================

print("\n====================================")
print("          FINAL TEST RESULTS")
print("====================================")

test_probabilities = best_model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

accuracy = accuracy_score(y_test, test_predictions)

precision = precision_score(
    y_test, test_predictions, zero_division=0
)

recall = recall_score(
    y_test, test_predictions, zero_division=0
)

f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

pr_auc = average_precision_score(
    y_test, test_probabilities
)

roc_auc = roc_auc_score(
    y_test, test_probabilities
)

print("Best model:", best_name)
print("Threshold:", round(best_threshold, 4))

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("PR-AUC:", round(pr_auc, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nClassification report:")

print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

cm = confusion_matrix(y_test, test_predictions)

print("\nConfusion matrix:")
print(cm)


# ==========================================
# 13. SAVE RESULTS
# ==========================================

test_results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": test_predictions,
    "Failure Probability": test_probabilities
})

test_results.to_csv(
    REPORT_DIR / "optimized_test_predictions.csv",
    index=False
)

pd.DataFrame(
    cm,
    index=["Actual Normal", "Actual Failure"],
    columns=["Predicted Normal", "Predicted Failure"]
).to_csv(
    REPORT_DIR / "optimized_confusion_matrix.csv"
)


# ==========================================
# 14. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    MODEL_DIR / "optimized_failure_model.pkl"
)

metadata = {
    "model_name": best_name,
    "threshold": best_threshold,
    "features": FEATURES,
    "metrics": {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc
    },
    "validation_pr_auc": float(
        results_df.iloc[0]["Validation PR-AUC"]
    )
}

joblib.dump(
    metadata,
    MODEL_DIR / "optimized_model_metadata.pkl"
)


# ==========================================
# 15. COMPLETION
# ==========================================

print("\n====================================")
print("     OPTIMIZATION COMPLETED")
print("====================================")

print("Selected model:", best_name)

print("\nGenerated files:")
print("models/optimized_failure_model.pkl")
print("models/optimized_model_metadata.pkl")
print("reports/optimized_model_comparison.csv")
print("reports/optimized_test_predictions.csv")
print("reports/optimized_confusion_matrix.csv")

print("\nAll operations completed successfully!")
