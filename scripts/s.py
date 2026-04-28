import os
import pandas as pd

# Path to your large CSV
source_path  = r"D:\NIDS\data\processed\cleaned_unbalanced_noise.csv"

# === Option 1: Take the first N rows (e.g. 200 rows) ===
n_rows = 100
df_head = pd.read_csv(source_path, nrows=n_rows)
df_head.to_csv(r"D:\NIDS\data\processed\cleaned_unbalanced_noise_demo_200rows.csv", index=False)
print(f"Saved first {n_rows} rows to demo file.")

# === Option 2: Approximate a 400 MB subset ===
# Adjust `sample_size` and `target_size_mb` to suit your needs.
sample_size = 5000           # number of rows to sample to estimate bytes per row
target_size_mb = 200          # desired size of the subset in MB

# Read a small sample to estimate memory usage per row
sample_df = pd.read_csv(source_path, nrows=sample_size)
# Calculate average bytes per row (deep=True counts object dtypes)
bytes_per_row = sample_df.memory_usage(index=True, deep=True).sum() / len(sample_df)

# Compute how many rows roughly equal the target size
target_rows = int((target_size_mb * 1024**2) / bytes_per_row)
print(f"Estimated rows for ~{target_size_mb} MB: {target_rows}")

# Read only the required number of rows
subset_df = pd.read_csv(source_path, nrows=target_rows)
subset_df.to_csv(r"D:\NIDS\data\processed\small.csv", index=False)
print(f"Saved subset of {target_rows} rows (≈ {target_size_mb} MB) to demo file.")