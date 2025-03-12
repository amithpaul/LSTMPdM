import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.mixture import GaussianMixture

# Load the test data (ensure it's scaled using the same scaler as during training)
test_data = np.load("scaled_minmax.npy").astype(np.float32)

# Assuming the same sequence length (seq_length) used during training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
    return np.array(sequences)

seq_length = 100  # Same as during training
X_test = create_sequences(test_data, seq_length)

# Load the trained LSTM Autoencoder model
model = load_model("lstm_autoencoder.keras")

# Run inference to get reconstructed outputs
reconstructed = model.predict(X_test)

# Calculate reconstruction error (MSE for the last timestep)
errors = np.mean(np.power(X_test[:, -1, :] - reconstructed, 2), axis=1)

# Fit a Gaussian Mixture Model (GMM) to reconstruction errors
errors = errors.reshape(-1, 1)
gmm = GaussianMixture(n_components=1, random_state=42)
gmm.fit(errors)

# Compute health scores based on GMM probability density
pdf_normal = np.exp(gmm.score_samples(errors))
pdf_max = np.max(pdf_normal)

health_scores = []
for e in errors:
    log_prob = gmm.score_samples(e.reshape(1, -1))
    pdf_val = np.exp(log_prob)[0]
    score = pdf_val / pdf_max  # Normalize to [0, 1]
    health_scores.append(np.clip(score, 0.0, 1.0))

health_scores = np.array(health_scores)

# Plot Reconstruction Error Distribution
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=50)
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution")
plt.show()

# Plot the aggregated health score over time
plt.figure(figsize=(10, 6))
plt.plot(health_scores, label="Aggregated System Health Score", color="blue", alpha=0.8)
plt.xlabel("Sample Index")
plt.ylabel("Health Score")
plt.title("GMM-Based Aggregated Health Score Over Time")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)  # Health scores range from [0,1]
plt.show()

# Compute an anomaly threshold using 3-sigma rule
threshold = np.mean(errors) + 3 * np.std(errors)
print("Anomaly threshold:", threshold)
