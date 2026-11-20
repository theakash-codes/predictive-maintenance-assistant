# Manufacturing Predictive Maintenance Assistant

### Machine Failure Prediction and Diagnosis Using Machine Learning

## Overview

The **Manufacturing Predictive Maintenance Assistant** is an AI-powered application designed to predict potential machine failures using sensor data and machine learning techniques.

The system analyzes machine operating conditions, estimates failure probability, identifies possible failure types, detects abnormal operating patterns, and evaluates overall machine health.

It provides an interactive dashboard developed using Streamlit to support proactive maintenance decisions and reduce unexpected machine downtime.

## Key Features

- **Machine Failure Prediction:** Predicts the possibility of machine failure using XGBoost.
- **Failure Diagnosis:** Identifies possible machine failure categories using machine learning.
- **Anomaly Detection:** Detects unusual machine operating conditions using Isolation Forest.
- **Explainable AI:** Uses SHAP to explain the contribution of sensor features to predictions.
- **Machine Health Score:** Calculates a health score based on failure risk and anomaly information.
- **Interactive Dashboard:** Provides sensor input, prediction results, visualizations, and maintenance insights.
- **Historical Records:** Stores prediction history for monitoring and analysis.

## System Architecture

The project follows this workflow:

1. AI4I 2020 Predictive Maintenance Dataset
2. Data Cleaning and Preprocessing
3. Feature Selection and Data Splitting
4. Machine Learning Model Training
5. Failure Prediction and Diagnosis
6. Anomaly Detection
7. SHAP-Based Explainability
8. Machine Health Score Calculation
9. Streamlit Dashboard and Maintenance Insights

## Machine Learning Algorithms

| Algorithm / Technique | Purpose |
|---|---|
| Logistic Regression | Baseline classification |
| Random Forest | Classification and failure diagnosis |
| Support Vector Machine (SVM) | Classification comparison |
| XGBoost | Primary machine failure prediction |
| Isolation Forest | Anomaly detection |
| SHAP | Model explainability |

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**

The project uses the AI4I 2020 dataset, a synthetic dataset representing industrial machine operating conditions.

- Total records: 10,000
- Normal records: 9,661
- Failure records: 339
- Normal condition: 96.61%
- Failure condition: 3.39%

### Sensor Features

- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]

The dataset also includes a categorical machine type feature (`Type`) in the training pipeline.

## Model Performance

Previously recorded XGBoost test results:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 98.05% |
| Precision | 69.86% |
| Recall | 75.00% |
| F1-Score | 72.34% |
| PR-AUC | 79.76% |
| ROC-AUC | 97.55% |

*Note: These are results from a previous training experiment. Results may differ when the current training pipeline is executed.*

## Technology Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy
- **Explainable AI:** SHAP
- **Visualization:** Matplotlib
- **Web Application:** Streamlit
- **Model Storage:** Joblib
- **Database:** SQLite

## Project Structure

```text
predictive-maintenance/
│
├── app.py
├── app_backup.py
│
├── assets/
├── data/
├── database/
├── docs/
├── models/
├── reports/
├── tests/
│
├── src/
│   ├── prepare_data.py
│   ├── inspect_data.py
│   ├── eda.py
│   ├── train_model.py
│   ├── optimize_model.py
│   ├── diagnose_failure.py
│   ├── individual_failure_models.py
│   ├── anomaly_detection.py
│   ├── explain_model.py
│   └── health_score.py
│
├── requirements.txt
└── README.md
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/theakash-codes/predictive-maintenance-assistant.git
```

### 2. Navigate to the project directory

```bash
cd predictive-maintenance-assistant
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv
```

### 4. Activate the environment

For macOS and Linux:

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

**Note:** Ensure the required trained model files and dataset files are available in their expected project directories before running the application.

## Future Enhancements

- Integration with real-time industrial IoT sensors.
- Deployment on edge devices.
- Improved failure prediction using additional datasets.
- Real-time maintenance alerts and notifications.
- Advanced predictive analytics and reporting.

## Project Objective

The main objective is to develop an intelligent predictive maintenance system that identifies potential machine failures before they occur, supports maintenance planning, and helps minimize unexpected industrial downtime.

---

**Developed as a B.Tech Major Project**

**GitHub:** https://github.com/theakash-codes/predictive-maintenance-assistant