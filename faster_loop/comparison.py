import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GaussianNoise, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Add to your existing imports
from scipy import signal
from sklearn.metrics import mean_squared_error

# ======================
# Data Loading & Preprocessing
# ======================

class IQDataProcessor:
    def __init__(self, npz_path, window_size=1024, overlap=0.4):
        self.npz_path = npz_path
        self.window_size = window_size
        self.overlap = overlap
        self.step = int(window_size * (1 - overlap))
        
    def load_data(self):
        """Load and stack IQ data from NPZ file"""
        with np.load(self.npz_path, allow_pickle=True) as npz_file:
            i_data = npz_file['storages'][0]['i']
            q_data = npz_file['storages'][0]['q']
        return np.stack([i_data, q_data], axis=0)  # Shape (2, N)

    def create_windows(self, data):
        """Generate overlapping windows with padding"""
        num_samples = data.shape[1]
        num_windows = (num_samples - self.window_size) // self.step + 1
        windows = []
        
        for i in range(num_windows):
            start = i * self.step
            end = start + self.window_size
            window = data[:, start:end]
            windows.append(window)
            
        # Handle last partial window
        if num_samples % self.step != 0:
            last_window = data[:, -self.window_size:]
            windows.append(last_window)
            
        return np.array(windows)

# ======================
# Autoencoder Architecture
# ======================

class CompressionAutoencoder:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
    def build(self):
        # Encoder
        inputs = Input(shape=(self.input_dim,))
        x = Dense(512, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        encoded = Dense(self.encoding_dim, activation='linear')(x)
        
        # Normalization layer
        encoded_normalized = Lambda(
            lambda x: K.l2_normalize(x, axis=1) * np.sqrt(self.encoding_dim))(encoded)
        
        # Decoder
        x = Dense(256, activation='relu')(encoded_normalized)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        decoded = Dense(self.input_dim, activation='linear')(x)
        
        # Models
        autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded_normalized)
        
        return autoencoder, encoder

# ======================
# Training & Evaluation
# ======================

class AETrainer:
    def __init__(self, autoencoder, lr=0.001):
        self.autoencoder = autoencoder
        self.autoencoder.compile(optimizer=Adam(lr), loss='mse')
        
    def train(self, X_train, X_val, epochs=100, batch_size=128):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('best_ae.h5', save_best_only=True)
        ]
        
        history = self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=callbacks
        )
        return history

def calculate_evm(original, reconstructed):
    """Calculate Error Vector Magnitude"""
    error = original - reconstructed
    evm_rms = np.sqrt(np.mean(np.abs(error)**2))
    ref_rms = np.sqrt(np.mean(np.abs(original)**2))
    return 20 * np.log10(evm_rms / ref_rms)

# ======================
# Main Workflow
# ======================

if __name__ == "__main__":
    # Configuration
    WINDOW_SIZE = 1024
    OVERLAP = 0.4
    ENCODING_DIM = 64
    BATCH_SIZE = 256
    
    # 1. Load and Process Data
    processor = IQDataProcessor(
        "your_data.npz",
        window_size=WINDOW_SIZE,
        overlap=OVERLAP
    )
    iq_data = processor.load_data()
    windows = processor.create_windows(iq_data)
    
    # Flatten windows for AE input
    X = windows.reshape(-1, 2 * WINDOW_SIZE)
    
    # 2. Split Data
    indices = np.random.permutation(len(X))
    train, val, test = np.split(indices, [int(.7*len(X)), int(.85*len(X))])
    
    # 3. Build and Train Autoencoder
    autoencoder, encoder = CompressionAutoencoder(
        input_dim=2*WINDOW_SIZE,
        encoding_dim=ENCODING_DIM
    ).build()
    
    trainer = AETrainer(autoencoder)
    history = trainer.train(X[train], X[val], batch_size=BATCH_SIZE)
    
    # 4. Evaluate Reconstruction
    test_windows = X[test].reshape(-1, 2, WINDOW_SIZE)
    reconstructed = autoencoder.predict(X[test]).reshape(-1, 2, WINDOW_SIZE)
    
    # Calculate EVM
    evm_results = []
    for orig, rec in zip(test_windows, reconstructed):
        evm = calculate_evm(orig[0] + 1j*orig[1], rec[0] + 1j*rec[1])
        evm_results.append(evm)
    
    print(f"\nAverage EVM: {np.mean(evm_results):.2f} dB")
    print(f"95th Percentile EVM: {np.percentile(evm_results, 95):.2f} dB")
    
    # 5. Visualization
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    
    plt.subplot(122)
    plt.hist(evm_results, bins=30)
    plt.title('EVM Distribution')
    plt.tight_layout()
    plt.show()

    # ======================
# Enhanced Evaluation
# ======================

def plot_signal_comparison(original, reconstructed, window_idx=0, samples=500):
    """Plot original vs reconstructed signals"""
    plt.figure(figsize=(15, 8))
    
    # Time Domain Comparison
    plt.subplot(2, 2, 1)
    plt.plot(original[window_idx,0,:samples], 'b', label='Original I')
    plt.plot(reconstructed[window_idx,0,:samples], 'r--', label='Reconstructed I')
    plt.title(f'Window {window_idx} - I Component Comparison')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(original[window_idx,1,:samples], 'b', label='Original Q')
    plt.plot(reconstructed[window_idx,1,:samples], 'r--', label='Reconstructed Q')
    plt.title(f'Window {window_idx} - Q Component Comparison')
    plt.legend()
    plt.grid(True)

    # Frequency Domain Comparison
    plt.subplot(2, 2, 3)
    f_orig, Pxx_orig = signal.welch(original[window_idx,0,:])
    f_recon, Pxx_recon = signal.welch(reconstructed[window_idx,0,:])
    plt.semilogy(f_orig, Pxx_orig, 'b', label='Original I')
    plt.semilogy(f_recon, Pxx_recon, 'r--', label='Reconstructed I')
    plt.title('Power Spectral Density Comparison - I')
    plt.xlabel('Frequency')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    f_orig, Pxx_orig = signal.welch(original[window_idx,1,:])
    f_recon, Pxx_recon = signal.welch(reconstructed[window_idx,1,:])
    plt.semilogy(f_orig, Pxx_orig, 'b', label='Original Q')
    plt.semilogy(f_recon, Pxx_recon, 'r--', label='Reconstructed Q')
    plt.title('Power Spectral Density Comparison - Q')
    plt.xlabel('Frequency')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('signal_comparison.png')
    plt.close()

def calculate_advanced_metrics(original, reconstructed):
    """Calculate comprehensive performance metrics"""
    metrics = {}
    
    # Time Domain Metrics
    metrics['MSE'] = mean_squared_error(original.flatten(), reconstructed.flatten())
    metrics['PSNR'] = 20 * np.log10(np.max(original) / np.sqrt(metrics['MSE']))
    
    # Correlation Analysis
    metrics['Correlation_I'] = np.corrcoef(original[:,0,:].flatten(), 
                                          reconstructed[:,0,:].flatten())[0,1]
    metrics['Correlation_Q'] = np.corrcoef(original[:,1,:].flatten(), 
                                          reconstructed[:,1,:].flatten())[0,1]
    
    # EVM Statistics
    evm_values = []
    for i in range(original.shape[0]):
        orig_complex = original[i,0,:] + 1j*original[i,1,:]
        recon_complex = reconstructed[i,0,:] + 1j*reconstructed[i,1,:]
        evm = np.linalg.norm(orig_complex - recon_complex) / np.linalg.norm(orig_complex)
        evm_values.append(20 * np.log10(evm))
    
    metrics['EVM_dB_mean'] = np.mean(evm_values)
    metrics['EVM_dB_std'] = np.std(evm_values)
    metrics['EVM_dB_95th'] = np.percentile(evm_values, 95)
    
    return metrics

# After your existing evaluation code:
# ... [previous evaluation code] ...

# Perform advanced analysis
print("\n=== Advanced Performance Analysis ===")

# 1. Generate comparison plots
plot_signal_comparison(original_iq, reconstructed_iq, window_idx=0)
plot_signal_comparison(original_iq, reconstructed_iq, window_idx=1)

# 2. Calculate comprehensive metrics
metrics = calculate_advanced_metrics(original_iq, reconstructed_iq)

print(f"\nMSE: {metrics['MSE']:.4e}")
print(f"PSNR: {metrics['PSNR']:.2f} dB")
print(f"I Component Correlation: {metrics['Correlation_I']:.4f}")
print(f"Q Component Correlation: {metrics['Correlation_Q']:.4f}")
print(f"\nEVM Statistics:")
print(f"Mean: {metrics['EVM_dB_mean']:.2f} dB")
print(f"Std Dev: {metrics['EVM_dB_std']:.2f} dB")
print(f"95th Percentile: {metrics['EVM_dB_95th']:.2f} dB")

# 3. Plot EVM distribution
plt.figure(figsize=(10, 6))
plt.hist(evms, bins=30, density=True, alpha=0.7)
plt.title('EVM Distribution Across Test Windows')
plt.xlabel('EVM (dB)')
plt.ylabel('Probability Density')
plt.grid(True)
plt.savefig('evm_distribution.png')
plt.close()

# 4. Constellation diagram comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(original_iq[0,0,:500], original_iq[0,1,:500], alpha=0.6)
plt.title('Original Constellation')
plt.grid(True)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.scatter(reconstructed_iq[0,0,:500], reconstructed_iq[0,1,:500], alpha=0.6)
plt.title('Reconstructed Constellation')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig('constellation_comparison.png'))
plt.close()

print("\n=== Analysis Complete ===")