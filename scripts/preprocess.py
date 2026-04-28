import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Directories
RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# File names for each day
files = [
    'Monday-WorkingHours.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-workingHours.pcap_ISCX.csv',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
]

# ---------------------------------------------------------------------
# LOAD AND MERGE RAW DATA
# ---------------------------------------------------------------------
dfs = []
for fname in files:
    path = os.path.join(RAW_DIR, fname)
    df = pd.read_csv(path, low_memory=False)
    df['Day'] = fname.split('-')[0]  # tag day
    dfs.append(df)
# Merge all days
full_df = pd.concat(dfs, ignore_index=True)

# ---------------------------------------------------------------------
# CLEAN AND BASIC PROCESSING
# ---------------------------------------------------------------------
# Drop rows with missing key columns
full_df.dropna(subset=[' Flow Duration', ' Total Fwd Packets', ' Label'], inplace=True)
# Remove duplicates
full_df.drop_duplicates(inplace=True)
# Replace infinite values with NaN then drop
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)

# Map multi‑class labels to binary: Attack vs Benign
full_df['Binary_Label'] = full_df[' Label'].apply(lambda x: 'Attack' if x != 'BENIGN' else 'Benign')
# Encode label (0 = Benign, 1 = Attack)
label_encoder = LabelEncoder()
full_df['Label_Encoded'] = label_encoder.fit_transform(full_df['Binary_Label'])

# Drop original label & day (not used in modelling)
full_df.drop(columns=[' Label', 'Binary_Label', 'Day'], inplace=True)

# ---------------------------------------------------------------------
# SCALE NUMERIC FEATURES AND ADD NOISE
# ---------------------------------------------------------------------
# Identify numeric columns (excluding the encoded label)
num_cols = [c for c in full_df.select_dtypes(include=[np.number]).columns if c != 'Label_Encoded']

# Standardize numeric features
scaler = StandardScaler()
full_df[num_cols] = scaler.fit_transform(full_df[num_cols])

# Inject small Gaussian noise (mean 0, std 0.01) to help prevent overfitting
noise = np.random.normal(loc=0.0, scale=0.01, size=full_df[num_cols].shape)
full_df[num_cols] = full_df[num_cols] + noise

# ---------------------------------------------------------------------
# BALANCE CLASSES WITH SMOTE
# ---------------------------------------------------------------------
X = full_df.drop(columns=['Label_Encoded'])
y = full_df['Label_Encoded']

smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

# Reassemble balanced DataFrame
balanced_df = pd.DataFrame(X_bal, columns=X.columns)
balanced_df['Label_Encoded'] = y_bal

# Save processed & balanced dataset
balanced_path = os.path.join(PROCESSED_DIR, 'cleaned_balanced_data.csv')
balanced_df.to_csv(balanced_path, index=False)
print(f"Saved balanced dataset to {balanced_path}")