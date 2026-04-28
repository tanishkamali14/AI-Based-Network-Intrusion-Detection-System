import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
DATA_FILE = 'data/processed/cleaned_unbalanced_noise.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Use only a fraction of data to manage memory (25%)
SAMPLE_FRAC = 0.25
# Maximum rows for hyperparameter tuning (keeps tuning inexpensive)
MAX_TUNE_ROWS = 200_000

# ---------------------------------------------------------------------
# LOAD AND SAMPLE DATA
# ---------------------------------------------------------------------
print("Loading processed data...")
df = pd.read_csv(DATA_FILE)
if SAMPLE_FRAC < 1.0:
    df = df.sample(frac=SAMPLE_FRAC, random_state=42)

X = df.drop(columns=['Label_Encoded'])
y = df['Label_Encoded']

# Stratified train/test split (20 % test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Training samples: {len(y_train)}")
print(f"Testing samples:  {len(y_test)}")

# Prepare tuning subset: either max MAX_TUNE_ROWS or 10 % of training data
if len(X_train) > MAX_TUNE_ROWS:
    X_tune, _, y_tune, _ = train_test_split(
        X_train, y_train,
        train_size=MAX_TUNE_ROWS,
        stratify=y_train,
        random_state=42
    )
else:
    frac = min(0.1, len(X_train)/MAX_TUNE_ROWS)
    X_tune = X_train.sample(frac=frac, random_state=42)
    y_tune = y_train.loc[X_tune.index]

print(f"Tuning subset size: {len(y_tune)}")

# ---------------------------------------------------------------------
# RANDOM FOREST MODEL
# ---------------------------------------------------------------------
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 25],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

rf_grid = GridSearchCV(
    rf,
    rf_params,
    scoring='f1',
    cv=3,
    n_jobs=1,  # limit memory
    verbose=1
)
rf_grid.fit(X_tune, y_tune)
best_rf = rf_grid.best_estimator_
print("Best RF params:", rf_grid.best_params_)

# Fit best RF on full training data
best_rf.fit(X_train, y_train)
# Evaluate
y_pred_rf = best_rf.predict(X_test)
print("\nRandom Forest Performance:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
try:
    print("ROC-AUC:", roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1]))
except Exception:
    print("ROC-AUC: N/A")
# Save
rf_path = os.path.join(MODEL_DIR, 'best_rf.pkl')
joblib.dump(best_rf, rf_path)
print(f"Saved Random Forest model to {rf_path}")

# ---------------------------------------------------------------------
# GRADIENT BOOSTING MODEL
# ---------------------------------------------------------------------
print("\nTraining Gradient Boosting...")
gb = GradientBoostingClassifier(random_state=42)
gb_params = {
    'n_estimators': [100, 200],     # lower to reduce overfitting
    'learning_rate': [0.05, 0.1],
    'max_depth': [3]
}

gb_grid = GridSearchCV(
    gb,
    gb_params,
    scoring='f1',
    cv=3,
    n_jobs=1,
    verbose=1
)
gb_grid.fit(X_tune, y_tune)
best_gb = gb_grid.best_estimator_
print("Best GB params:", gb_grid.best_params_)

# Fit best Gradient Boosting on full training data
best_gb.fit(X_train, y_train)
# Evaluate
y_pred_gb = best_gb.predict(X_test)
y_proba_gb = best_gb.predict_proba(X_test)[:, 1]
print("\nGradient Boosting Performance:")
print(classification_report(y_test, y_pred_gb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gb))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_gb))
# Save
gb_path = os.path.join(MODEL_DIR, 'best_gb.pkl')
joblib.dump(best_gb, gb_path)
print(f"Saved Gradient Boosting model to {gb_path}")

print("\nSupervised learning complete. Models saved to 'models/' directory.")