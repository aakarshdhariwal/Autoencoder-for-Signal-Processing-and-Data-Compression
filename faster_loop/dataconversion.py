import numpy as np
import pandas as pd
import os

def npz_to_csv(npz_file, output_dir="csv_files"):
    # Load the .npz file with allow_pickle=True
    data = np.load(npz_file, allow_pickle=True)

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each array in the .npz file and save as CSV
    for key, value in data.items():
        csv_filename = f"{output_dir}/{key}.csv"
        
        # Handle complex data (IQ signals) by saving real and imaginary parts separately
        if np.iscomplexobj(value):
            pd.DataFrame({'real': value.real, 'imag': value.imag}).to_csv(csv_filename, index=False)
        else:
            pd.DataFrame(value).to_csv(csv_filename, index=False, header=False)
        
        print(f"Saved {csv_filename}")

# Example usage
npz_to_csv(r"C:\Users\Aakarsh\Desktop\R&S-Hackathon\pca\faster_loop\8psk_default_1Msym_gray_differential_3mio.npz")
