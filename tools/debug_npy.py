import numpy as np
import os

path = r"dataset\MP_Data\hello"

print("Checking:", path)

if not os.path.exists(path):
    print("❌ Path not found")
    exit()

for file in os.listdir(path):
    if file.endswith(".npy"):
        full_path = os.path.join(path, file)

        data = np.load(full_path, allow_pickle=True)

        print("File:", file)
        print("Shape:", data.shape)
        print("Dimensions:", data.ndim)
        print("First values:", data.flatten()[:10])
        print("-" * 50)

        break