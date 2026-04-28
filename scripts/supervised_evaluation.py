# scripts/unified_evaluation.py
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve,
                             roc_curve, confusion_matrix,
                             classification_report,
                             precision_score, recall_score,
                             f1_score, roc_auc_score, auc)
from sklearn.model_selection import train_test_split

# 0. Paths and model filenames
PROCESSED = 'data/processed/cleaned_unbalanced_noise.csv'
MODEL_DIR = 'models'
PLOTS_DIR = os.path.join('outputs', 'plots')
REPORTS_DIR = os.path.join('outputs', 'reports')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

model_files = {
    'Random Forest': os.path.join(MODEL_DIR, 'best_rf.pkl'),
    'Gradient Boosting': os.path.join(MODEL_DIR, 'best_gb.pkl'),
    'Balanced Random Forest': os.path.join(MODEL_DIR, 'best_balanced_rf.pkl'),
    # 'LightGBM': os.path.join(MODEL_DIR, 'best_lightgbm.pkl'),
}

# 1. Load the full dataset and split
print(f"Loading processed data from {PROCESSED} …")
df = pd.read_csv(PROCESSED)
X = df.drop(columns=['Label_Encoded'])
y = df['Label_Encoded']
# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training rows: {len(y_train)}, testing rows: {len(y_test)}")

# 2. Evaluate each model
results = []
for name, file in model_files.items():
    if not os.path.exists(file):
        print(f"⚠️  Skipping {name} – file not found: {file}")
        continue

    print(f"\nEvaluating {name} …")
    model = joblib.load(file)
    y_pred = model.predict(X_test)
    try:
        y_scores = model.predict_proba(X_test)[:, 1]
    except Exception:
        # Some models (e.g. SVM without probability) may not have predict_proba
        y_scores = model.decision_function(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_scores)
    pr_auc = auc(*precision_recall_curve(y_test, y_scores)[::-1])

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results.append({
        'model': name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    })

    # Plot PR curve
    prec, rec, _ = precision_recall_curve(y_test, y_scores)
    plt.figure()
    plt.plot(rec, prec, label=f'{name} PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve – {name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pr_path = os.path.join(PLOTS_DIR, f"{name.lower().replace(' ', '_')}_pr.png")
    plt.savefig(pr_path)
    plt.close()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{name} ROC AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve – {name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    roc_path = os.path.join(PLOTS_DIR, f"{name.lower().replace(' ', '_')}_roc.png")
    plt.savefig(roc_path)
    plt.close()

    print(f"  • Metrics: precision={precision:.4f}, recall={recall:.4f}, "
          f"f1={f1:.4f}, ROC AUC={roc_auc:.4f}, PR AUC={pr_auc:.4f}")
    print(f"  • Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  • Plots saved: {pr_path} and {roc_path}")

# 3. Save consolidated metrics
metrics_df = pd.DataFrame(results)
metrics_csv = os.path.join(REPORTS_DIR, 'unified_metrics.csv')
metrics_df.to_csv(metrics_csv, index=False)
print(f"\nSaved consolidated metrics to {metrics_csv}")

# 4. (Optional) Combine with anomaly detector metrics
anom_metrics = os.path.join(REPORTS_DIR, 'metrics_anomaly_detection.csv')
if os.path.exists(anom_metrics):
    df_anom = pd.read_csv(anom_metrics)
    combined = pd.concat([metrics_df, df_anom], ignore_index=True, sort=False)
    combined_csv = os.path.join(REPORTS_DIR, 'combined_metrics.csv')
    combined.to_csv(combined_csv, index=False)
    print(f"Saved combined supervised + unsupervised metrics to {combined_csv}")