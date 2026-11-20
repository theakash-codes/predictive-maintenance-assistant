
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
df = pd.read_csv("data/raw/ai4i/ai4i2020.csv")

# Create reports directory
Path("reports").mkdir(exist_ok=True)

# 1. Failure distribution
failure_counts = df["Machine failure"].value_counts()

plt.figure(figsize=(7, 5))
failure_counts.plot(kind="bar")
plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/failure_distribution.png")
plt.close()

# 2. Correlation heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))
plt.imshow(numeric_df.corr(), cmap="coolwarm", aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(
    range(len(numeric_df.columns)),
    numeric_df.columns,
    rotation=90
)
plt.yticks(
    range(len(numeric_df.columns)),
    numeric_df.columns
)
plt.title("Sensor Data Correlation Heatmap")
plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png")
plt.close()

# 3. Temperature comparison
plt.figure(figsize=(8, 5))

df.boxplot(
    column="Process temperature [K]",
    by="Machine failure"
)

plt.title("Process Temperature vs Machine Failure")
plt.suptitle("")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Process Temperature (K)")
plt.tight_layout()
plt.savefig("reports/temperature_analysis.png")
plt.close()

# 4. Torque comparison
plt.figure(figsize=(8, 5))

df.boxplot(
    column="Torque [Nm]",
    by="Machine failure"
)

plt.title("Torque vs Machine Failure")
plt.suptitle("")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Torque (Nm)")
plt.tight_layout()
plt.savefig("reports/torque_analysis.png")
plt.close()

# 5. Rotational speed comparison
plt.figure(figsize=(8, 5))

df.boxplot(
    column="Rotational speed [rpm]",
    by="Machine failure"
)

plt.title("Rotational Speed vs Machine Failure")
plt.suptitle("")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Rotational Speed (RPM)")
plt.tight_layout()
plt.savefig("reports/rpm_analysis.png")
plt.close()

print("EDA completed successfully!")
print("All graphs saved inside reports/")
