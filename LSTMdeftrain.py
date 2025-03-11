import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Load preprocessed dataset
data = np.load("scaled_dataset.npy").astype(np.float32)  # Ensure float32
input_dim = data.shape[1]  # Number of features

# Create sequences
seq_length = 100  # Adjust based on your dataset
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
    return np.array(sequences)

X_train = create_sequences(data, seq_length)


# LSTM Autoencoder
timesteps, features = X_train.shape[1], X_train.shape[2]

input_layer = Input(shape=(timesteps, features))
encoded = LSTM(64, activation="relu", return_sequences=True)(input_layer)
encoded = LSTM(32, activation="relu", return_sequences=False)(encoded)

decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(features, activation="sigmoid")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse",jit_compile=True)

# Train the model
autoencoder.fit(
    X_train, X_train[:, -1, :], 
    epochs=100, batch_size=64, 
    validation_split=0.1, 
    callbacks=[early_stop]
)

# Save the model
autoencoder.save("lstm_autoencoder.keras")
