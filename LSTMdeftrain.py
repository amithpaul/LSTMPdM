import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Masking, LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# Clear the Keras session to reset the model's state
K.clear_session()

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Load preprocessed dataset
data = np.load("processed_datasets/scaled_minmax.npy").astype(np.float32)  # Ensure float32

# Print the first 10 rows
print("First 10 rows of the dataset:\n", data[:10])
input_dim = data.shape[1]  # Number of features

# Create sequences
seq_length = 180  # Adjust based on your dataset
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
    return np.array(sequences)

X_train = create_sequences(data, seq_length)


# LSTM Autoencoder
timesteps, features = X_train.shape[1], X_train.shape[2]

input_layer = Input(shape=(timesteps, features))

# Apply masking to ignore -1 values
masked_input = Masking(mask_value=-1)(input_layer)
encoded = LSTM(64, activation="relu", return_sequences=True)(masked_input)
encoded = LSTM(32, activation="relu", return_sequences=True)(encoded)

decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(features, activation="sigmoid")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# Train the model
autoencoder.fit(
    X_train, X_train, 
    epochs=200, batch_size=256, 
    validation_split=0.1, 
    callbacks=[early_stop]
)

# Save the model
autoencoder.save("lstm_autoencoderV2.keras")
