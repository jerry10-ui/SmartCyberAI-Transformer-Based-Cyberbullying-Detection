import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../../data/processed/final_balanced_dataset.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))

train_df.to_csv("../../data/processed/train.csv", index=False)
val_df.to_csv("../../data/processed/val.csv", index=False)

print("Train/Validation split completed!")