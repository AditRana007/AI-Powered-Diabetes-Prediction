import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv("data/diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Balance dataset
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Powerful XGBoost model
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Threshold tuning (IMPORTANT)
threshold = 0.35
y_pred = (y_probs >= threshold).astype(int)

# F1 score
f1 = f1_score(y_test, y_pred)
print("🔥 High F1 Score:", round(f1, 3))

# Save model & scaler
with open("models/diabetes_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)
