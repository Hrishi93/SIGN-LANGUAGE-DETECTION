import numpy as np
import pandas as pd
import os

DATASET_PATH = r"..\dataset\MP_Data"

columns = ["label"] + [f"F{i}" for i in range(63)]

all_rows = []

print("📊 Converting Holistic NPY → Hand CSV...")

for label in os.listdir(DATASET_PATH):

    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    print(f"\n📂 Label: {label}")

    for root, dirs, files in os.walk(label_path):

        for file in files:

            if file.endswith(".npy"):

                npy_path = os.path.join(root, file)
                data = np.load(npy_path)
                data = np.array(data)

                try:
                    if data.size >= 63:
                        hand_data = data[-63:]  # ✅ extract hand landmarks

                        row = [label] + hand_data.tolist()
                        all_rows.append(row)
                    else:
                        raise ValueError(f"Too small: {data.size}")

                except Exception as e:
                    print(f"❌ Skipping {file} ({e})")

df = pd.DataFrame(all_rows, columns=columns)

print("\n✅ Conversion finished!")
print("📈 Shape:", df.shape)

df.to_csv("../dataset/mp_landmarks_xyz.csv", index=False)
print("💾 Saved → dataset/mp_landmarks_xyz.csv")