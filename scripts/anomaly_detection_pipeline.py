# scripts/anomaly_detection_pipeline.py
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

# Load data
DATA_DIR = 'data/anomaly'
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
X_eval = np.load(os.path.join(DATA_DIR, 'X_eval.npy'))
y_eval = np.load(os.path.join(DATA_DIR, 'y_eval.npy'))

# ---------------------- AUTOENCODER ----------------------
print("\nTraining Autoencoder…")
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu',
                activity_regularizer=regularizers.l1(1e-5))(input_layer)
decoded = Dense(input_dim, activation='linear')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=0
)

# Reconstruction error threshold (95th percentile)
recon_errors = np.mean(np.square(X_eval - autoencoder.predict(X_eval)), axis=1)
threshold = np.percentile(recon_errors[y_eval == 0], 95)
y_pred_ae = (recon_errors > threshold).astype(int)

print("\nAutoencoder Results:")
print(classification_report(y_eval, y_pred_ae))
print("Confusion Matrix:\n", confusion_matrix(y_eval, y_pred_ae))
print("ROC-AUC:", roc_auc_score(y_eval, recon_errors))

# -------------------- ISOLATION FOREST --------------------
print("\nTraining Isolation Forest…")
iso_forest = IsolationForest(contamination=0.5, random_state=42)
iso_forest.fit(X_train)
y_pred_if = iso_forest.predict(X_eval)
y_pred_if = np.where(y_pred_if == -1, 1, 0)

print("\nIsolation Forest Results:")
print(classification_report(y_eval, y_pred_if))
print("Confusion Matrix:\n", confusion_matrix(y_eval, y_pred_if))

# -------------------- VISUALIZATION -----------------------
print("\nGenerating Precision-Recall Curve (Autoencoder)…")
prec, rec, _ = precision_recall_curve(y_eval, recon_errors)
pr_auc = auc(rec, prec)

plt.figure()
plt.plot(rec, prec, label=f'Autoencoder PR AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Autoencoder')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Correct path setup
plot_path = os.path.join("outputs", "anomaly_detection", "run_20260320_145448", "plots", "autoencoder_pr_curve.png")
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path)

print(f"Saved plot to {plot_path}")