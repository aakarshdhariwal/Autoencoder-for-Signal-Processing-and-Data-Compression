import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Lambda, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from faster_loop.sliding_window import SlidingWindowGenerator

class AutoencoderSignalProcessor:
    def __init__(self, data_path, window_size=512, overlap_ratio=0.4, batch_size=32, 
                 learning_rate=0.0001, epochs=10, encoding_dim=64):
        self.data_path = data_path
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.batch_size = batch_size
        self.learning_rate = learning_rate
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
        npz_file = np.load(self.data_path, allow_pickle=True)
        i_data = npz_file['storages'][0]['i']
        q_data = npz_file['storages'][0]['q']
        data = np.vstack((i_data, q_data))

        # Normalize (zero mean, unit variance)
        # mean = np.mean(data, axis=1, keepdims=True)
        # std = np.std(data, axis=1, keepdims=True)
        # normalized_data = (data - mean) / (std + 1e-6)

        # Shuffle data
        #num_samples = data.shape[1]
        #indices = np.random.choice(num_samples, size=num_samples, replace=False)
        return data

    def build_model(self, input_dim):
        """Builds and compiles the autoencoder model."""
        input_signal = Input(shape=(input_dim,))

        # Encoder
        x = GaussianNoise(0.01)(input_signal)
        x = Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(self.encoding_dim, activation='linear')(x)
        encoded = Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1) * np.sqrt(self.encoding_dim))(x)

        # Decoder
        x = Dense(512, activation='relu')(encoded)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        decoded = Dense(input_dim, activation='linear')(x)

        # Compile
        autoencoder = Model(input_signal, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='huber')

        # Extract encoder part
        encoder = Model(inputs=input_signal, outputs=encoded)

        self.autoencoder = autoencoder
        self.encoder = encoder

    def train(self, data):
        """Trains the autoencoder model."""
        window_generator = SlidingWindowGenerator(data, self.window_size, self.overlap_ratio, self.batch_size)
        X = np.vstack(list(window_generator.generator()))
        input_dim = X.shape[1]
        print(X.shape)

         # Train/Validation/Test Split
        train_size = int(0.6 * len(X))
        val_size = int(0.2 * len(X))
        X_train, X_val, X_test = np.split(X, [train_size, train_size + val_size])
        self.X_test = X_test
        print(self.X_test.shape)
        # Build and train model
        self.build_model(input_dim)
        self.history = self.autoencoder.fit(X_train, X_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, X_val))

    def evaluate(self):
        """Evaluates the trained model and calculates metrics."""
        test_loss = self.autoencoder.evaluate(self.X_test, self.X_test)
        print(f"Test loss: {test_loss:.6f}")

        # Extract encoded representation
        #encoded_test = self.encoder.predict(self.X_test)
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

    def plot_loss(self):
        """Plots the training and validation loss."""
        plt.plot(self.history.history['loss'], label="Training Loss")
        plt.plot(self.history.history['val_loss'], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
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
        rms_error = np.sqrt(np.mean(np.abs(error_vectors) ** 2))
        rms_reference = np.sqrt(np.mean(np.abs(reference_iq) ** 2))
        evm = rms_error / rms_reference
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
data_path = r"C:\Users\Aakarsh\Desktop\R&S-Hackathon\pca\faster_loop\8psk_default_1Msym_gray_differential_3mio.npz"
processor = AutoencoderSignalProcessor(data_path)
data = processor.load_data()
processor.train(data)
processor.evaluate()
processor.plot_loss()
processor.plot_constellation()
processor.plot_signal_windows()
