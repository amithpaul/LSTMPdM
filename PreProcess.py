import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data (ensure it's in a structured Pandas DataFrame)
df = pd.read_csv("anomalous_dataset.csv",low_memory=False) # 


# Convert 'timestamp' to datetime - using infer_datetime_format
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True, errors="coerce")
    
    # Convert to seconds since the first timestamp
    df["timestamp"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()

# Ensure all columns are numeric
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# Handle missing values
df_numeric = df_numeric.ffill().bfill().fillna(0)

# Drop duplicates while keeping the last occurrence
df_numeric = df_numeric.drop_duplicates(subset=df_numeric.columns.difference(["timestamp"]), keep="last")

# Convert timestamp to datetime for resampling
if "timestamp" in df_numeric.columns:
    temp_timestamp = pd.to_datetime(df_numeric["timestamp"], unit='s', 
                                   origin=pd.Timestamp('1970-01-01'), errors="coerce")
    
    # Create a copy of df_numeric with timestamp as datetime for resampling
    df_for_resampling = df_numeric.copy()
    df_for_resampling["timestamp"] = temp_timestamp
    
    # Resample - using 's' instead of 'S' as suggested by the warning
    df_downsampled = df_for_resampling.resample('100ms', on='timestamp').mean()
    
    # Reset index to make timestamp a column again
    df_downsampled = df_downsampled.reset_index()
    
    # Convert timestamp back to numeric for scaling
    df_downsampled["timestamp"] = (df_downsampled["timestamp"] - df_downsampled["timestamp"].min()).dt.total_seconds()
else:
    df_downsampled = df_numeric

# Normalize Data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_downsampled.values)

# Save processed dataset
np.save("scaled_anomaly_dataset.npy", df_scaled)  # Save scaled data for GAN training
np.save("scaler.npy", scaler)  # Save scaler for inverse transformation

columns_to_use = df_downsampled.columns
pd.DataFrame(df_scaled, columns=columns_to_use).to_csv("processed_synth_dataset.csv", index=False) #reg or synth

print("✅ Data preprocessing completed! Timestamps are now numeric and included.")
