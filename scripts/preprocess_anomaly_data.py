# scripts/preprocess_anomaly_data.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RAW_CSV = 'data/processed/cleaned_balanced_data.csv'  # Use already cleaned, unbalanced dataset
SAVE_DIR = 'data/anomaly'
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(RAW_CSV)

# Split into benign and attack samples
benign = df[df['Label_Encoded'] == 0]
attack = df[df['Label_Encoded'] == 1]

# Use 70% benign for training the anomaly detector
benign_train, benign_val = train_test_split(benign, test_size=0.3, random_state=42)

# Evaluation = rest of benign + all attack
eval_df = pd.concat([benign_val, attack]).sample(frac=1.0, random_state=42)

# Drop label column for training, but keep for evaluation
X_train = benign_train.drop(columns=['Label_Encoded'])
X_eval = eval_df.drop(columns=['Label_Encoded'])
y_eval = eval_df['Label_Encoded']

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_eval_scaled = scaler.transform(X_eval)

# Save preprocessed data
np.save(os.path.join(SAVE_DIR, 'X_train.npy'), X_train_scaled)
np.save(os.path.join(SAVE_DIR, 'X_eval.npy'), X_eval_scaled)
np.save(os.path.join(SAVE_DIR, 'y_eval.npy'), y_eval.values)