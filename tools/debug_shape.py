import numpy as np

path = r"dataset\MP_Data\hello\0\0.npy"

data = np.load(path)

print("Shape:", data.shape)
print("Type:", type(data))
print(data)