import pandas as pd

# Load datasets
df1 = pd.read_csv("merged_dataset_folder1.csv")
df2 = pd.read_csv("dataset2_multilabel.csv")
df3 = pd.read_csv("dataset3_multilabel.csv")

# Define final label structure
labels = [
    "hate",
    "offensive",
    "aggression",
    "bullying",
    "toxic",
    "racism",
    "sexism",
    "religious_hate"
]

# Ensure all datasets have all columns
for df in [df1, df2, df3]:
    for label in labels:
        if label not in df.columns:
            df[label] = 0

# Keep consistent column order
df1 = df1[["text"] + labels]
df2 = df2[["text"] + labels]
df3 = df3[["text"] + labels]

# Merge
final_df = pd.concat([df1, df2, df3], axis=0)

# Remove duplicate texts
final_df = final_df.groupby("text").max().reset_index()

# Shuffle
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Total rows after merge:", len(final_df))

# Save full dataset
final_df.to_csv("final_full_dataset.csv", index=False)

print("Final dataset saved successfully!")
