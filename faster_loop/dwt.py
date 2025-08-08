import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import decimate, resample

# Load the .npz file
npz_file = np.load(r"C:\Users\Aakarsh\Desktop\R&S-Hackathon\pca\faster_loop\8psk_default_1Msym_gray_differential_3mio.npz", allow_pickle=True)

# Extract IQ data
i_data = npz_file['storages'][0]['i']
q_data = npz_file['storages'][0]['q']

# Combine I and Q
iq_data = np.vstack((i_data, q_data))  # Shape: (2, N)

# -------------------------- 1. Downsampling and Upsampling --------------------------
downsample_factor = 4
# Apply FIR filter for zero-phase filtering
i_downsampled = decimate(i_data, downsample_factor, ftype='fir')
q_downsampled = decimate(q_data, downsample_factor, ftype='fir')

# Upsample by a factor of 4
i_upsampled = resample(i_downsampled, len(i_data))
q_upsampled = resample(q_downsampled, len(q_data))

# -------------------------- 2. Calculate EVM for Downsampling and Upsampling --------------------------
def calculate_evm(measured_iq, reference_iq):
    """
    Compute the Error Vector Magnitude (EVM).
    """
    error_vectors = measured_iq - reference_iq
    rms_error = np.sqrt(np.mean(np.abs(error_vectors)**2))
    rms_reference = np.sqrt(np.mean(np.abs(reference_iq)**2))
    evm = rms_error / rms_reference
    evm_percent = evm * 100
    evm_db = 20 * np.log10(evm) if evm != 0 else -np.inf
    return evm, evm_percent, evm_db

# Combine the upsampled I and Q data
iq_upsampled = np.vstack((i_upsampled, q_upsampled))

# Compute EVM for upsampled signal
evm_upsampled, evm_percent_upsampled, evm_db_upsampled = calculate_evm(iq_upsampled, iq_data)

# -------------------------- 3. Calculate Compression Ratio --------------------------
# Calculate the Compression Ratio for Downsampling and Upsampling
original_size = i_data.size + q_data.size
downsampled_size = i_downsampled.size + q_downsampled.size
compression_ratio_downsampling = original_size / downsampled_size

print(f"Compression Ratio (Downsampling by 4, FIR filtering): {compression_ratio_downsampling:.2f}")
print(f"EVM for Downsampled + Upsampled signal: {evm_upsampled:.4f} ({evm_percent_upsampled:.2f}%) --> {evm_db_upsampled:.2f} dB")

# -------------------------- 4. DWT Compression (As Before) --------------------------
def dwt_compress(signal, wavelet="db8", level=4):
    """
    Compress a signal using DWT by keeping only approximation coefficients.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs[0], coeffs  # Return full coefficients for later reconstruction

# Apply DWT compression to both I and Q signals
i_compressed, i_coeffs = dwt_compress(i_data)
q_compressed, q_coeffs = dwt_compress(q_data)

# Calculate Compression Ratio for DWT
compressed_size = i_compressed.size + q_compressed.size
compression_ratio_dwt = original_size / compressed_size

print(f"Compression Ratio (DWT Compression): {compression_ratio_dwt:.2f}")

# -------------------------- 5. Plot Constellation Diagram --------------------------
plt.figure(figsize=(8, 8))
plt.scatter(i_data[:100000], q_data[:100000], s=1, alpha=0.5, label="Original")
plt.scatter(i_upsampled[:100000], q_upsampled[:100000], s=1, alpha=0.5, label="Upsampled")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.title("Constellation Diagram: Original vs. Upsampled")
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.show()
