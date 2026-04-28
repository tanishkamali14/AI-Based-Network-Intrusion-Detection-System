# supervised_unbalanced_with_noise.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve,
                             auc)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
# Paths
RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
MODEL_DIR = Path('models')
REPORTS_DIR = Path('reports')
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Data files (raw CIC-IDS2017 CSVs)
data_files = [
    'Monday-WorkingHours.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-workingHours.pcap_ISCX.csv',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
]

# Fraction of data to load (50 %)
SAMPLE_FRAC = 0.5

# ------------------------------------------------------------------------------
# LOAD AND PREPROCESS (UNBALANCED) WITH NOISE
# ------------------------------------------------------------------------------
def load_and_preprocess_unbalanced(data_files):
    dfs = []
    for fname in data_files:
        df = pd.read_csv(RAW_DIR / fname)
        df['Day'] = fname.split('-')[0]
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)

    # Basic cleaning: drop rows with key NaNs and duplicates
    full_df = full_df.dropna(subset=[' Flow Duration', ' Total Fwd Packets', ' Label'])
    full_df = full_df.drop_duplicates()
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_df.dropna(inplace=True)

    # Map to binary attack/benign
    full_df['Binary_Label'] = full_df[' Label'].apply(lambda x: 'Attack' if x != 'BENIGN' else 'Benign')
    le = LabelEncoder()
    full_df['Label_Encoded'] = le.fit_transform(full_df['Binary_Label'])

    # Drop unused columns
    full_df.drop(columns=[' Label', 'Binary_Label', 'Day'], inplace=True)

    # Scale numeric features and add Gaussian noise
    numeric_cols = [c for c in full_df.select_dtypes(include=[np.number]).columns if c != 'Label_Encoded']
    scaler = StandardScaler()
    scaled = scaler.fit_transform(full_df[numeric_cols])
    noise = np.random.normal(loc=0.0, scale=0.05, size=scaled.shape)
    full_df[numeric_cols] = scaled + noise  # add noise to scaled features

    return full_df

df_cleaned = load_and_preprocess_unbalanced(data_files)
# Sample 50 % of the dataset to reduce memory usage
df_sampled = df_cleaned.sample(frac=SAMPLE_FRAC, random_state=42)

# Separate features/labels
X = df_sampled.drop(columns=['Label_Encoded'])
y = df_sampled['Label_Encoded']

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Training samples: {len(y_train)}")
print(f"Testing samples:  {len(y_test)}")

# Create tuning subset (at most 10 % or 200k rows)
max_tune = 200_000
if len(X_train) > max_tune:
    X_tune, _, y_tune, _ = train_test_split(
        X_train,
        y_train,
        train_size=max_tune,
        stratify=y_train,
        random_state=42
    )
else:
    frac = min(0.1, len(X_train)/max_tune)
    X_tune = X_train.sample(frac=frac, random_state=42)
    y_tune = y_train.loc[X_tune.index]

# ------------------------------------------------------------------------------
# MODEL TRAINING AND EVALUATION
# ------------------------------------------------------------------------------
def train_model(model, param_grid, model_name):
    """
    Grid-search for a given model on tuning subset, then fit best model on full train data.
    Logs performance metrics and returns the trained estimator.
    """
    search = GridSearchCV(
        model,
        param_grid,
        scoring='f1',
        cv=3,
        n_jobs=1,
        verbose=1
    )
    search.fit(X_tune, y_tune)
    best = search.best_estimator_
    best.fit(X_train, y_train)

    print(f"\n{model_name} Best Params:", search.best_params_)

    # Predict and evaluate
    y_pred = best.predict(X_test)
    try:
        y_proba = best.predict_proba(X_test)[:, 1]
    except AttributeError:
        # Some models may not have predict_proba
        y_proba = np.zeros_like(y_pred, dtype=float)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    # Save model
    joblib.dump(best, MODEL_DIR / f'best_{model_name}.pkl')

    # Save PR curve plot
    plt.figure()
    plt.plot(recall, precision, label=f'{model_name} (AUC={pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend()
    plot_path = REPORTS_DIR / f'pr_curve_{model_name}.png'
    plt.savefig(plot_path)
    plt.close()

    # Append performance summary to log file
    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    summary = {
        'date': date_str,
        'model': model_name,
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
        'roc_auc': auc_score,
        'pr_auc': pr_auc
    }
    metrics_path = REPORTS_DIR / 'supervised_unbalanced_with_noise_metrics.csv'
    metrics_df = pd.DataFrame([summary])
    if not metrics_path.exists():
        metrics_df.to_csv(metrics_path, index=False)
    else:
        metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)

    # Return model and metrics for printing
    return best, summary, cm

# Random Forest training
rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 25],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}
rf_model, rf_summary, rf_cm = train_model(rf, rf_params, 'rf')
print("\nRandom Forest Performance Summary:")
print(rf_summary)
print("Confusion Matrix:\n", rf_cm)

# Gradient Boosting training
gb = GradientBoostingClassifier(random_state=42)
gb_params = {
    'n_estimators': [200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3]
}
gb_model, gb_summary, gb_cm = train_model(gb, gb_params, 'gb')
print("\nGradient Boosting Performance Summary:")
print(gb_summary)
print("Confusion Matrix:\n", gb_cm)

print("\nSupervised training on unbalanced noisy dataset complete.")
print(f"Models saved in {MODEL_DIR} and metrics in {REPORTS_DIR}")