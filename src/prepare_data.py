
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================
# 1. PATHS
# ==========================================

INPUT_PATH = Path("data/raw/ai4i/ai4i2020.csv")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(INPUT_PATH)

print("Original dataset shape:", df.shape)

# ==========================================
# 3. SELECT FIVE SENSOR FEATURES
# ==========================================

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Machine failure"

required = FEATURES + [TARGET]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

data = df[required].copy()

# ==========================================
# 4. DATA VALIDATION
# ==========================================

if data.isnull().any().any():
    raise ValueError("Missing values detected.")

if data.duplicated().any():
    print("Warning: Duplicate records detected.")

if not set(data[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Unexpected target values.")

X = data[FEATURES]
y = data[TARGET]

print("\nSelected features:")
print(FEATURES)

print("\nTarget distribution:")
print(y.value_counts())

# ==========================================
# 5. STRATIFIED 80:20 SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# 6. SAVE DATA
# ==========================================

X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n========== DATA PREPARATION COMPLETE ==========")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining distribution:")
print(y_train.value_counts())

print("\nTesting distribution:")
print(y_test.value_counts())

print("\nTraining failure percentage:",
      round(y_train.mean() * 100, 2), "%")

print("\nTesting failure percentage:",
      round(y_test.mean() * 100, 2), "%")

print("\nSaved successfully in:", OUTPUT_DIR)
