import pandas as pd

df = pd.read_csv(r"C:\Coding\Projects\Minor Project\dataset\Hate Speech Detection curated Dataset🤬\HateSpeechDatasetBalanced.csv")

# Rename text column properly if needed
df.columns = ["text", "label"]

# Create new label columns
df["hate"] = df["label"]
df["offensive"] = 0
df["aggression"] = 0
df["bullying"] = 0
df["toxic"] = 0
df["racism"] = 0
df["sexism"] = 0
df["religious_hate"] = 0

# Keep required columns
df = df[[
    "text",
    "hate",
    "offensive",
    "aggression",
    "bullying",
    "toxic",
    "racism",
    "sexism",
    "religious_hate"
]]

df.to_csv("dataset3_multilabel.csv", index=False)

print("Dataset 3 converted successfully!")
