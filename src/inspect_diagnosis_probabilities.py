
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

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

# Load the same training and test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=df["Machine failure"]
)

# Create a validation split from training data only
X_sub, X_val, y_sub, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=df.loc[X_train.index, "Machine failure"]
)

# Train a fresh diagnosis model
base_model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model = MultiOutputClassifier(base_model, n_jobs=-1)
model.fit(X_sub, y_sub)

# Get validation probabilities
probabilities = model.predict_proba(X_val)

print("\n===== VALIDATION PROBABILITY ANALYSIS =====")

for i, target in enumerate(TARGETS):
    actual = y_val.iloc[:, i].to_numpy()
    scores = probabilities[i][:, 1]

    positive_scores = scores[actual == 1]
    negative_scores = scores[actual == 0]

    print(f"\n{target}")
    print("Actual positive validation cases:", len(positive_scores))

    if len(positive_scores) > 0:
        print("Positive probability range:",
              round(positive_scores.min(), 4),
              "to",
              round(positive_scores.max(), 4))
        print("Average positive probability:",
              round(positive_scores.mean(), 4))

    print("Highest negative probability:",
          round(negative_scores.max(), 4))

    print("Positive scores:",
          sorted(positive_scores, reverse=True))
