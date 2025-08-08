import numpy as np
import os
import sys
import tensorflow as tf
import optuna
import sqlite3
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from faster_loop.sliding_window import SlidingWindowGenerator

# Enable GPU acceleration
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU acceleration enabled!")
else:
    print("No GPU found. Running on CPU.")

class AutoencoderSignalProcessor:
    def __init__(self, data_path, window_size=512, overlap_ratio=0.5, batch_size=32, epochs=2, encoding_dim=128):
        self.data_path = data_path
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.encoding_dim = encoding_dim
        self.autoencoder = None
        self.encoder = None
        self.history = None
        self.X_test = None
        self.decoded_test = None

        # Set seeds for reproducibility
        SEED = 42
        tf.random.set_seed(SEED)
        np.random.seed(SEED)

    def load_data(self):
        """Loads and normalizes signal data from NPZ file."""
        all_i_data = []
        all_q_data = []
        for file in self.data_path:
            npz_file = np.load(file, allow_pickle=True)
            i_data = npz_file['storages'][0]['i']
            q_data = npz_file['storages'][0]['q']
            all_i_data.append(i_data)
            all_q_data.append(q_data)
        
         # Stack all i and q data to create a large dataset
        i_data_combined = np.concatenate(all_i_data, axis=0)
        q_data_combined = np.concatenate(all_q_data, axis=0)
        data = np.vstack((i_data_combined, q_data_combined))

        # Normalize (zero mean, unit variance)
        # mean = np.mean(data, axis=1, keepdims=True)
        # std = np.std(data, axis=1, keepdims=True)
        # normalized_data = (data - mean) / (std + 1e-6)

        # Shuffle data
        #num_samples = data.shape[1]
        # np.random.shuffle(normalized_data.T)
        return data

    def build_model(self, input_dim, learning_rate):
        input_signal = Input(shape=(input_dim,))
        x = Dense(1024, kernel_regularizer=keras.regularizers.l2(1e-5))(input_signal)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(512, kernel_regularizer=keras.regularizers.l2(1e-5))(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        encoded = BatchNormalization(scale=True, center=True)(x)

        x = Dense(512)(encoded)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        decoded = Dense(input_dim, activation='linear')(x)

        autoencoder = Model(input_signal, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        encoder = Model(inputs=input_signal, outputs=encoded)

        self.autoencoder = autoencoder
        self.encoder = encoder

    def train(self, data, learning_rate):
        window_generator = SlidingWindowGenerator(data, self.window_size, self.overlap_ratio, self.batch_size)
        windows = list(window_generator.generator())
        np.random.shuffle(windows)  # Shuffle the full windows, not individual samples
        X = np.vstack(windows)
        train_size, val_size = int(0.6 * len(X)), int(0.2 * len(X))
        X_train, X_val, X_test = np.split(X, [train_size, train_size + val_size])
        self.X_test = X_test
        
        self.build_model(X.shape[1], learning_rate)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        self.history = self.autoencoder.fit(
            X_train, X_train, epochs=self.epochs, batch_size=self.batch_size,
            validation_data=(X_val, X_val), callbacks=[early_stopping]
        )

    def evaluate(self):
        """Evaluates the trained model and calculates metrics."""
        test_loss = self.autoencoder.evaluate(self.X_test, self.X_test)
        print(f"Test loss: {test_loss:.6f}")

        # Extract encoded representation
        #encoded_test = self.encoder.predict(self.X_test)
        # Extract encoder model
        
        self.decoded_test = self.autoencoder.predict(self.X_test)
        
        # Reshape to (num_test_windows, 2, window_size)
        self.X_test_reshaped = self.X_test.reshape(-1, 2, self.window_size)
        self.decoded_test_reshaped = self.decoded_test.reshape(-1, 2, self.window_size)
        # Compute Compression Ratio
        compression_ratio = self.calculate_compression_ratio(self.X_test.shape, self.encoding_dim)
        print(f"Compression Ratio: {compression_ratio:.1f}:1")

        # Calculate EVM
        # Combine all windows into a single array of I/Q samples
        X_test_evm = self.X_test_reshaped.transpose(0, 2, 1).reshape(-1, 2)  # Shape: (N*512, 2)
        decoded_test_evm = self.decoded_test_reshaped.transpose(0, 2, 1).reshape(-1, 2)  # Shape: (N*512, 2)

        evm, evm_percent, evm_db = self.calculate_evm(X_test_evm, decoded_test_evm)
        print(f"\nEVM: {evm:.4f} ({evm_percent:.2f}%) | {evm_db:.2f} dB")

    def optimize_hyperparameters(self):
        def objective(trial):
            encoding_dim = trial.suggest_categorical('encoding_dim', [32, 64, 128, 256])
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
            self.encoding_dim = encoding_dim
            data = self.load_data()
            self.train(data, learning_rate)
            loss = self.autoencoder.evaluate(self.X_test, self.X_test, verbose=0)
            return loss

        study = optuna.create_study(direction='minimize', study_name='autoencoder_opt', storage='sqlite:///optuna_autoencoder.db', load_if_exists=True)
        study.optimize(objective, n_trials=10)
        print("Best hyperparameters:", study.best_params)
    def plot_loss(self):
        """Plots the training and validation loss."""
        plt.figure()
        plt.plot(self.history.history['loss'], label="Training Loss")
        plt.plot(self.history.history['val_loss'], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_latent_space(self):
        """Plots latent space using t-SNE."""
        encoded_test = self.encoder.predict(self.X_test)  # Extract encoded representations
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        latent_2d = tsne.fit_transform(encoded_test)  # Reduce to 2D

        # Plot latent space clusters
        plt.figure(figsize=(8, 6))
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5, cmap='viridis')
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.title("Latent Space Visualization")
        plt.grid(True)
        plt.show()

    def plot_constellation(self):
        """Plots the original and reconstructed signal constellation."""
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(self.X_test_reshaped[0,0,:], self.X_test_reshaped[0,1,:], alpha=0.5, s=1, color='b')
        plt.title("Original Constellation")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(self.decoded_test_reshaped[0,0,:], self.decoded_test_reshaped[0,1,:], alpha=0.5, s=1, color='r')
        plt.title("Reconstructed Constellation")
        plt.grid(True)
        plt.show()

    def plot_signal_windows(self):
        """Plots the first two signal windows (original vs. reconstructed)."""
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        for i in range(2):  # First two windows
            axes[i].plot(self.X_test_reshaped[i,0,:], label=f"Original Window {i+1}", color='b')
            axes[i].plot(self.decoded_test_reshaped[i,0,:], label=f"Reconstructed Window {i+1}", color='r', linestyle='dashed')
            axes[i].legend()
            axes[i].set_title(f"Window {i+1}: Original vs. Reconstructed")
            axes[i].grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def calculate_evm(measured_iq, reference_iq):
        """Computes EVM between measured and reference IQ symbols."""
        error_vectors = measured_iq - reference_iq
        valid_indices = np.where(np.abs(reference_iq) > 1e-6)  # Ignore near-zero signals
        evm = np.sqrt(np.mean(np.abs(error_vectors[valid_indices]) ** 2)) / \
        np.sqrt(np.mean(np.abs(reference_iq[valid_indices]) ** 2))
        #rms_error = np.sqrt(np.mean(np.abs(error_vectors) ** 2))
        ##rms_reference = np.sqrt(np.mean(np.abs(reference_iq) ** 2))
        #evm = rms_error / rms_reference
        evm_percent = evm * 100
        evm_db = 20 * np.log10(evm) if evm != 0 else -np.inf
        return evm, evm_percent, evm_db

    @staticmethod
    def calculate_compression_ratio(original_shape, encoded_dim):
        """Calculates compression ratio."""
        original_size = original_shape[0] * original_shape[1]
        compressed_size = encoded_dim * original_shape[0]
        return original_size / compressed_size
    

# === RUN THE MODEL ===
data_path = [r"C:\Users\Aakarsh\Desktop\R&S-Hackathon\pca\faster_loop\8psk_default_1Msym_gray_differential_3mio.npz",
             r"C:\Users\Aakarsh\Desktop\R&S-Hackathon\pca\faster_loop\default_format=PSK8_sr=500000_sl=3000000_coding=GRAY.npz"]
processor = AutoencoderSignalProcessor(data_path)
processor.optimize_hyperparameters()
data = processor.load_data()
processor.train(data, learning_rate=0.001)
processor.evaluate()

import subprocess
print("\nRunning hyperparameter optimization analysis...")
subprocess.run(["python", "optuna_study_analysis.py"])
