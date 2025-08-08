import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
arrays = np.load(r"C:\Users\Aakarsh\Desktop\R&S-Hackathon\pca\faster_loop\8psk_default_1Msym_gray_differential_3mio.npz", allow_pickle=True)
data_list = []
i_data = arrays['storages'][0]['i']
q_data = arrays['storages'][0]['q']
combined_data = np.vstack((i_data, q_data))
data_list.append(combined_data)
print(combined_data.shape)

# Plot I data
plt.figure(figsize=(10, 4))
plt.plot(combined_data[0, 0:20000], label="Inphase")
plt.legend()
plt.title("Raw I Signal")

# Plot Q data
plt.figure(figsize=(10, 4))
plt.plot(combined_data[1, 0:20000], label="Quadrature")
plt.legend()
plt.title("Raw Q Signal")

# Create constellation diagram
plt.figure(figsize=(8, 8))
plt.scatter(combined_data[0, 0:100000], combined_data[1, 0:100000], s=1, alpha=0.5)
plt.axis('equal')  # Equal aspect ratio
plt.grid(True)
plt.title("Constellation Diagram")
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()
print(data_list)