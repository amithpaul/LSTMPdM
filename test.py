# Loading and processing test data
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load your test data (ensure it's scaled using the same scaler used for training)
test_data = np.load("scaled_anomaly_dataset.npy").astype(np.float32)

# Assuming you have the same sequence length (seq_length) used during training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
    return np.array(sequences)

seq_length = 100  # same as during training
X_test = create_sequences(test_data, seq_length)



# Load the trained model (adjust the file name if using .keras or SavedModel format)
model = load_model("lstm_autoencoder.keras")

# Run inference to get reconstructed outputs
reconstructed = model.predict(X_test)


# Calculate reconstruction error (using the last timestep in this example)
errors = np.mean(np.power(X_test[:, -1, :] - reconstructed, 2), axis=1)

# Plot 
plt.hist(errors, bins=50)
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution")
plt.show()

threshold = np.mean(errors) + 3 * np.std(errors)
print("Anomaly threshold:", threshold)
