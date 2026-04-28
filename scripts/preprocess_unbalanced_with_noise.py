# preprocess_unbalanced_with_noise.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Paths
raw_path = "data/raw"
processed_path = "data/processed"
os.makedirs(processed_path, exist_ok=True)

# List of raw CSV files from the CIC-IDS2017 dataset
files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# Load and merge raw data
dfs = []
for fname in files:
    df = pd.read_csv(os.path.join(raw_path, fname))
    df["Day"] = fname.split("-")[0]  # preserve day name for temporal analysis if needed
    dfs.append(df)
full_df = pd.concat(dfs, ignore_index=True)

# Drop rows with critical NaNs, duplicates, and infinite values
full_df.dropna(subset=[" Flow Duration", " Total Fwd Packets", " Label"], inplace=True)
full_df.drop_duplicates(inplace=True)
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)

# Convert multi-class labels to binary string labels
full_df["Binary_Label"] = full_df[" Label"].apply(lambda x: "Attack" if x != "BENIGN" else "Benign")

# Encode binary labels to integers (0 = Benign, 1 = Attack)
le = LabelEncoder()
full_df["Label_Encoded"] = le.fit_transform(full_df["Binary_Label"])

# Drop unused columns
full_df.drop(columns=[" Label", "Binary_Label", "Day"], inplace=True)

# Identify numeric feature columns (exclude label column)
num_cols = [col for col in full_df.select_dtypes(include=[np.number]).columns if col != "Label_Encoded"]

# Scale numeric features
scaler = StandardScaler()
full_df[num_cols] = scaler.fit_transform(full_df[num_cols])

# Add small Gaussian noise to numeric features to reduce overfitting
# Noise level: 1% of each feature's standard deviation after scaling (std=1 for StandardScaler)
noise_level = 0.01
noise = np.random.normal(loc=0.0, scale=noise_level, size=full_df[num_cols].shape)
full_df[num_cols] += noise

# Save the cleaned, unbalanced dataset with noise (no SMOTE)
output_file = os.path.join(processed_path, "cleaned_unbalanced_noise.csv")
full_df.to_csv(output_file, index=False)
print(f"Saved unbalanced and noise-injected dataset to: {output_file}")