import pandas as pd

df_mp = pd.read_csv("dataset/mp_landmarks_dataset.csv")
df_asl = pd.read_csv("dataset/asl_landmarks_final.csv")

print("MP Dataset:", df_mp.shape)
print("ASL Dataset:", df_asl.shape)

merged = pd.concat([df_mp, df_asl], ignore_index=True)

print("Merged:", merged.shape)
print("\nClass balance:\n", merged['label'].value_counts())

merged.to_csv("dataset/final_dataset.csv", index=False)

print("\n✅ Saved → dataset/final_dataset.csv")