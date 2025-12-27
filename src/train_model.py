import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from data_pipeline import DataPipeline

pipeline = DataPipeline()

pipeline.run(
    ticker="AAPL",
    start="2015-01-01",
    interval="1d"
)


# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/processed/AAPL_processed.csv")

# -----------------------------
# BASIC CLEAN (NO FANCY STUFF)
# -----------------------------
# Drop rows where target is NaN (just in case)
df = df.dropna(subset=["target"])

# Keep only numeric columns
df = df.select_dtypes(include=["number"])

# -----------------------------
# TRAIN / TEST SPLIT (TIME-BASED)
# -----------------------------
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df  = df.iloc[split_idx:]

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

X_test  = test_df.drop(columns=["target"])
y_test  = test_df["target"]

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
preds = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds))
print("Recall   :", recall_score(y_test, preds))
print("F1-score :", f1_score(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds))

import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
sys.path.append(str(PROJECT_ROOT))

os.makedirs("model", exist_ok=True)

with open("model/rf_direction_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to model/rf_direction_model.pkl")

