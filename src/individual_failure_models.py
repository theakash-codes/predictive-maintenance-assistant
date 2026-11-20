
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

df = pd.read_csv("data/raw/ai4i/ai4i2020.csv")

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGETS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

X = df[FEATURES]
y = df[TARGETS]

# Preserve the original held-out test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=df["Machine failure"]
)

# Training and validation split
X_sub, X_val, y_sub, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=df.loc[X_train.index, "Machine failure"]
)

print("INDIVIDUAL FAILURE MODEL EXPERIMENT")

for target in TARGETS:

    print(f"\n========== {target} ==========")

    # Calculate class weights from training data only
    positives = int(y_sub[target].sum())
    negatives = len(y_sub) - positives

    if positives == 0:
        print("Not enough positive training examples.")
        continue

    weight = negatives / positives

    model = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight={0: 1, 1: weight},
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_sub, y_sub[target])

    probabilities = model.predict_proba(X_val)[:, 1]

    actual = y_val[target].to_numpy()

    # Evaluate at the default threshold
    predictions = (probabilities >= 0.5).astype(int)

    print("Training positives:", positives)
    print("Validation positives:", int(actual.sum()))
    print("Precision:", round(
        precision_score(actual, predictions, zero_division=0), 4
    ))
    print("Recall:", round(
        recall_score(actual, predictions, zero_division=0), 4
    ))
    print("F1:", round(
        f1_score(actual, predictions, zero_division=0), 4
    ))
    print("PR-AUC:", round(
        average_precision_score(actual, probabilities), 4
    ))
