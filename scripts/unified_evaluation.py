# unified_evaluation.py
#
# This script evaluates multiple supervised intrusion-detection models on a hold‑out test set.
# It computes classification metrics (precision, recall, F1, ROC‑AUC, PR‑AUC), confusion
# matrices, and saves precision‑recall and ROC curves for each model in outputs/plots.
# A consolidated CSV summarising all model metrics is saved to outputs/reports.
# Adjust MODEL_FILES and DATA_FILE to match your environment.

import os
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve, roc_curve, auc)

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
# Path to the processed dataset with labels.
DATA_FILE = 'data/processed/cleaned_unbalanced_noise.csv'
# Directory where trained model pickle files reside.
MODEL_DIR = 'models'
# Models to evaluate: (friendly_name, filename_in_MODEL_DIR)
MODEL_FILES = [
    ('Random Forest', 'best_rf.pkl'),
    ('Gradient Boosting', 'best_gb.pkl'),
    ('Balanced Random Forest', 'best_balanced_rf.pkl'),
    ('LightGBM', 'best_lightgbm.pkl')
]
# Fraction of the full dataset to sample for faster evaluation; set to 1.0 to use all rows.
SAMPLE_FRAC = 1.0
# Test set proportion; stratify to preserve class distribution.
TEST_SIZE = 0.2
# Output directories
OUTPUT_PLOTS_DIR = 'outputs/plots'
OUTPUT_REPORTS_DIR = 'outputs/reports'
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)

# ─── LOAD AND PREP DATA ───────────────────────────────────────────────────────
print(f'Loading processed data from {DATA_FILE} …')
df = pd.read_csv(DATA_FILE)
if SAMPLE_FRAC < 1.0:
    df = df.sample(frac=SAMPLE_FRAC, random_state=42)
# Separate features and label
if 'Label_Encoded' not in df.columns:
    raise ValueError('The dataset must contain a Label_Encoded column (0=benign, 1=attack).')
X = df.drop(columns=['Label_Encoded'])
y = df['Label_Encoded']
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=42
)
print(f'Training rows: {len(y_train)}, testing rows: {len(y_test)}')

# ─── EVALUATE EACH SUPERVISED MODEL ───────────────────────────────────────────
metrics_list = []
for friendly_name, filename in MODEL_FILES:
    model_path = os.path.join(MODEL_DIR, filename)
    if not os.path.isfile(model_path):
        print(f'⚠️  Skipping {friendly_name} – file not found: {model_path}')
        continue

    print(f'\nEvaluating {friendly_name} …')
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    # Some tree models have predict_proba; if not, skip AUC.
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = y_pred  # fallback: use predicted label for PR/AUC (not ideal but avoids crash)

    # Classification metrics
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    precision = report_dict['weighted avg']['precision']
    recall    = report_dict['weighted avg']['recall']
    f1        = report_dict['weighted avg']['f1-score']
    # Confusion matrix returns [[TN, FP],[FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Compute ROC & PR curves + AUCs if probabilities available
    try:
        roc_auc  = roc_auc_score(y_test, y_proba)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc   = auc(rec, prec)
    except Exception:
        roc_auc  = np.nan
        pr_auc   = np.nan
        prec, rec = [0], [0]

    metrics_list.append({
        'model': friendly_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    })

    # Plot PR curve
    plt.figure()
    plt.plot(rec, prec, label=f'{friendly_name} (PR AUC={pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve – {friendly_name}')
    plt.legend()
    plt.grid(True)
    pr_plot_path = os.path.join(OUTPUT_PLOTS_DIR, f'{friendly_name.lower().replace(" ", "_")}_pr.png')
    plt.tight_layout()
    plt.savefig(pr_plot_path)
    plt.close()

    # Plot ROC curve
    if not np.isnan(roc_auc):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{friendly_name} (ROC AUC={roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve – {friendly_name}')
        plt.legend()
        plt.grid(True)
        roc_plot_path = os.path.join(OUTPUT_PLOTS_DIR, f'{friendly_name.lower().replace(" ", "_")}_roc.png')
        plt.tight_layout()
        plt.savefig(roc_plot_path)
        plt.close()

    print(f'  • Metrics: precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, '
          f'ROC AUC={roc_auc:.4f}, PR AUC={pr_auc:.4f}')
    print(f'  • Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}')
    print(f'  • Plots saved: {pr_plot_path} and {roc_plot_path if not np.isnan(roc_auc) else "N/A"}')

# ─── SAVE CONSOLIDATED METRICS ───────────────────────────────────────────────
metrics_df = pd.DataFrame(metrics_list)
metrics_csv_path = os.path.join(OUTPUT_REPORTS_DIR, 'unified_metrics.csv')
metrics_df.to_csv(metrics_csv_path, index=False)
print(f'\nSaved consolidated metrics to {metrics_csv_path}')

# ─── OPTIONAL: MERGE UNSUPERVISED RESULTS ─────────────────────────────────────
# If you have unsupervised anomaly-detection results (e.g. from anomaly_detection_pipeline),
# you can merge them here for a holistic comparison by reading the CSV produced there:
unsup_metrics_path = 'outputs/anomaly_detection/run_20260320_145448/reports/metrics_anomaly_detection.csv'
if os.path.isfile(unsup_metrics_path):
    unsup_df = pd.read_csv(unsup_metrics_path)
    unsup_df = unsup_df.rename(columns={
        'detector': 'model', 'precision': 'precision', 'recall': 'recall',
        'f1': 'f1', 'roc_auc': 'roc_auc', 'pr_auc': 'pr_auc',
        'tn': 'tn', 'fp': 'fp', 'fn': 'fn', 'tp': 'tp'
    })
    combined = pd.concat([metrics_df, unsup_df], ignore_index=True, sort=False)
    comb_csv_path = os.path.join(OUTPUT_REPORTS_DIR, 'combined_metrics.csv')
    combined.to_csv(comb_csv_path, index=False)
    print(f'Saved combined supervised + unsupervised metrics to {comb_csv_path}')
else:
    print('No unsupervised metrics CSV found to merge.')