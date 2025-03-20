import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Masking, LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import keras_tuner as kt  # Import Keras Tuner

# Disable GPU (if needed)
tf.config.set_visible_devices([], 'GPU')

# Clear the Keras session
K.clear_session()

# Load preprocessed dataset
data = np.load("processed_datasets/scaled_minmax.npy").astype(np.float32)  # Ensure float32
input_dim = data.shape[1]  # Number of features

# Create sequences
seq_length = 180  # Same as during training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):  # Fix off-by-one issue
        sequences.append(data[i : i + seq_length])
    return np.array(sequences)

X_train = create_sequences(data, seq_length)

timesteps, features = X_train.shape[1], X_train.shape[2]

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# 🔹 Define the hyperparameter search function
def build_model(hp):
    input_layer = Input(shape=(timesteps, features))

    # Apply masking to ignore missing values
    masked_input = Masking(mask_value=-1)(input_layer)

    # First LSTM layer
    lstm_units1 = hp.Int('lstm_units1', min_value=32, max_value=128, step=32)
    encoded = LSTM(lstm_units1, activation="relu", return_sequences=True)(masked_input)

    # Second LSTM layer
    lstm_units2 = hp.Int('lstm_units2', min_value=16, max_value=64, step=16)
    encoded = LSTM(lstm_units2, activation="relu", return_sequences=True)(encoded)

    # Dropout for regularization
    dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    encoded = Dropout(dropout_rate)(encoded)

    # Dense layers for decoding
    decoded = Dense(64, activation="relu")(encoded)
    decoded = Dense(features, activation="sigmoid")(decoded)

    # Compile model
    model = Model(input_layer, decoded)
    
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 5e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    
    return model

# 🔹 Use Keras Tuner for Hyperparameter Optimization
tuner = kt.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=50,
    factor=3,
    directory="kt_search",
    project_name="lstm_autoencoder_tuning"
)

# 🔹 Start the hyperparameter search
tuner.search(X_train, X_train, epochs=50, batch_size=256, validation_split=0.1, callbacks=[early_stop])

# 🔹 Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
Best Hyperparameters Found:
- LSTM Units 1: {best_hps.get('lstm_units1')}
- LSTM Units 2: {best_hps.get('lstm_units2')}
- Dropout: {best_hps.get('dropout')}
- Learning Rate: {best_hps.get('learning_rate')}
""")

# 🔹 Build the model with best hyperparameters
best_model = tuner.hypermodel.build(best_hps)

# 🔹 Train with best model
history = best_model.fit(
    X_train, X_train,
    epochs=200, batch_size=256,
    validation_split=0.1,
    callbacks=[early_stop]
)

# 🔹 Save the best model
best_model.save("lstm_autoencoderV3.keras")

# 🔹 Plot Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
