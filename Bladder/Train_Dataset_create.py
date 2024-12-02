# Import Bladder Cancer and Normal data
import pandas as pd

df_blad = pd.read_csv("Dataset/bladder.csv")
df_normal = pd.read_csv("Dataset/normal.csv")
# Create new dataset of Bladder Cancer + Normal data
num_cancer = df_blad.shape[0]
sampled_normal = df_normal.sample(n=num_cancer, random_state=42)
combined_df = pd.concat([df_blad, sampled_normal], ignore_index=True)
combined_df.to_csv("Dataset/bladder+normal.csv", index=False)