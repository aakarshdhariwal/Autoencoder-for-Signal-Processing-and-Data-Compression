import numpy as np
import matplotlib.pyplot as plt

class SlidingWindowGenerator:
    def __init__(self, data, window_size, overlap_ratio, batch_size=32):
        """
        Initialize sliding window generator
        :param data: Input IQ data (2, N) numpy array
        :param window_size: Number of samples per window
        :param overlap_ratio: Ratio of overlap between windows (0-1)
        :param batch_size: Number of windows per batch
        """
        self.data = data
        self.window_size = window_size
        self.overlap = overlap_ratio
        self.batch_size = batch_size
        self.step_size = int(window_size * (1 - overlap_ratio))
        
        # Validate input data
        if self.data.ndim != 2 or self.data.shape[0] != 2:
            raise ValueError("Input data must be 2D array with shape (2, N)")
            
        self.total_samples = self.data.shape[1]
        self.num_windows = max(1, (self.total_samples - self.window_size) // self.step_size + 1)
        
    def _create_windows(self):
        """Generate sliding windows from IQ data"""
        windows = []
        
        for i in range(self.num_windows):
            start = i * self.step_size
            end = start + self.window_size
            
            # Handle last window
            if end > self.total_samples:
                pad_size = end - self.total_samples
                window_i = np.pad(self.data[0, start:], (0, pad_size), 'constant')
                window_q = np.pad(self.data[1, start:], (0, pad_size), 'constant')
            else:
                window_i = self.data[0, start:end]
                window_q = self.data[1, start:end]
                
            windows.append(np.stack([window_i, window_q], axis=0))
            
        return np.array(windows)
    
    def generator(self):
        """Create batches of windows for training"""
        windows = self._create_windows()
        
        # Flatten windows for autoencoder input
        flattened = windows.reshape(-1, 2 * self.window_size)
        
        # Create batches
        num_batches = int(np.ceil(len(flattened) / self.batch_size))
        
        for i in range(num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            yield flattened[start:end]
    
    def visualize_windows(self, num_windows=3):
        """Visualize first few windows for verification"""
        windows = self._create_windows()
        
        plt.figure(figsize=(15, 6))
        for i in range(min(num_windows, len(windows))):
            plt.subplot(2, num_windows, i+1)
            plt.plot(windows[i][0], label=f'Window {i} I')
            plt.plot(self.data[0], alpha=0.3, label='Original I')
            plt.legend()
            
            plt.subplot(2, num_windows, i+1+num_windows)
            plt.plot(windows[i][1], label=f'Window {i} Q')
            plt.plot(self.data[1], alpha=0.3, label='Original Q')
            plt.legend()
            
        plt.tight_layout()
        plt.show()