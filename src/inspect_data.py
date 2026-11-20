
import pandas as pd
from pathlib import Path

# Dataset location
DATA_PATH = Path("data/raw/ai4i/ai4i2020.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET INFORMATION ==========")

print("\nFirst 5 records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe(include="all"))

# Target distribution
if "Machine failure" in df.columns:
    print("\nMachine failure distribution:")
    print(df["Machine failure"].value_counts())

    print("\nFailure percentage:")
    print(df["Machine failure"].value_counts(normalize=True) * 100)

print("\n========== INSPECTION COMPLETED ==========")
