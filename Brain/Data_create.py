import pandas as pd

df_brain = pd.read_csv("Dataset/brain.csv")
df_nor = pd.read_csv("Dataset/normal.csv")
df_test = pd.read_csv("Dataset/test_data.csv")

num_cancer = df_brain.shape[0]
sampled_normal = df_nor.sample(n=num_cancer, random_state=42)
combined_df = pd.concat([df_brain, sampled_normal], ignore_index=True)
combined_df.to_csv("Dataset/brain+normal.csv", index=False)

test_data = pd.read_csv("Dataset/test_data.csv")
test_data = test_data[(test_data['cancer_type'] == "normal") | (test_data['cancer_type'] == "brain")]
test_data.to_csv("Dataset/brain_test_data.csv")
