import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load Data
df = pd.read_csv("anomalous_dataset.csv", low_memory=False)
print("✅ Data Loaded. Shape:", df.shape)

# Convert 'timestamp' to datetime if it exists (DO NOT SCALE THIS)
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True, errors="coerce")
    df["timestamp"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()

# Convert all columns to numeric (skip errors)
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# Fill missing values (Forward Fill → Backward Fill → Replace NaN with 0)
df_numeric = df_numeric.ffill().bfill().fillna(0)

# Drop duplicate rows
df_numeric = df_numeric.drop_duplicates()
print("✅ Duplicates Removed. Shape:", df_numeric.shape)

# Select columns to **NOT SCALE** (e.g., timestamp)
cols_not_scaled = ["timestamp"] if "timestamp" in df_numeric.columns else []

# Select columns to be **scaled**
cols_to_scale = [col for col in df_numeric.columns if col not in cols_not_scaled]

# Initialize scalers
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Apply MinMaxScaler to selected columns
df_minmax_scaled = df_numeric.copy()
df_minmax_scaled[cols_to_scale] = minmax_scaler.fit_transform(df_numeric[cols_to_scale])

# Apply StandardScaler to selected columns
df_standard_scaled = df_numeric.copy()
df_standard_scaled[cols_to_scale] = standard_scaler.fit_transform(df_numeric[cols_to_scale])

# Save processed datasets
df_minmax_scaled.to_csv("processed_synth_dataset_minmax.csv", index=False)
df_standard_scaled.to_csv("processed_synth_dataset_standard.csv", index=False)

# Save .npy files for models
np.save("scaled_minmax.npy", df_minmax_scaled.values)  # For MinMax Scaled data
np.save("scaled_standard.npy", df_standard_scaled.values)  # For Standard Scaled data
np.save("scaler_minmax.npy", minmax_scaler)  # Save MinMaxScaler for inverse transformation
np.save("scaler_standard.npy", standard_scaler)  # Save StandardScaler for inverse transformation

print("✅ Data preprocessing completed! Final Shape:", df_numeric.shape)
