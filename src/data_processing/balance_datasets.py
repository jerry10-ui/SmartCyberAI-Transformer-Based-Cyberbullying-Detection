import pandas as pd

df = pd.read_csv("final_full_dataset.csv")

print("Original size:", len(df))
labels = [
    "hate", "offensive", "aggression", "bullying", "toxic", "racism", "sexism", "religious_hate"
]

# Rows with any rare label
rare_labels = ["offensive","aggression","bullying","toxic","racism","sexism","religious_hate"]
rare_rows = df[df[rare_labels].sum(axis=1) > 0]

# Hate-only rows
hate_only = df[
    (df["hate"] == 1) &
    (df[rare_labels].sum(axis=1) == 0)
]

# Pure normal rows
normal_rows = df[df[labels].sum(axis=1) == 0]

print("Rare rows:", len(rare_rows))
print("Hate-only rows:", len(hate_only))
print("Normal rows:", len(normal_rows))

# Sample
hate_sample = hate_only.sample(n=120000, random_state=42)
normal_sample = normal_rows.sample(n=120000, random_state=42)

# Combine everything
balanced_df = pd.concat([rare_rows, hate_sample, normal_sample])

# Shuffle
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Final balanced size:", len(balanced_df))
balanced_df.to_csv("final_balanced_dataset.csv", index=False)
print("Balanced dataset saved successfully!")