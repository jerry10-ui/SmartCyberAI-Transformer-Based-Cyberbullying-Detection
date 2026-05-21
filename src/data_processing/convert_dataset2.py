import pandas as pd

# 🔹 Replace this with the correct path to your dataset
file_path = r"C:\Coding\Projects\Minor Project\dataset\CyberBullying Detection Dataset\final_hateXplain.csv"

df = pd.read_csv(file_path)

print("Original Columns:", df.columns)

# --------------------------------------------------
# STEP 1: Detect and standardize text column
# --------------------------------------------------

possible_text_columns = ["comment", "Comment", "Text", "text", "tweet", "Tweet"]

text_column_found = None

for col in possible_text_columns:
    if col in df.columns:
        text_column_found = col
        break

if text_column_found is None:
    raise Exception("No text column found! Check column names.")

df.rename(columns={text_column_found: "text"}, inplace=True)

# --------------------------------------------------
# STEP 2: Initialize multi-label columns
# --------------------------------------------------

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

for label in labels:
    df[label] = 0

# --------------------------------------------------
# STEP 3: Map main label column
# --------------------------------------------------

if "label" in df.columns:
    df.loc[df["label"] == "hatespeech", "hate"] = 1
    df.loc[df["label"] == "offensive", "offensive"] = 1

# --------------------------------------------------
# STEP 4: Map target attributes (if present)
# --------------------------------------------------

if "Race" in df.columns:
    df.loc[df["Race"] != "No_race", "racism"] = 1

if "Gender" in df.columns:
    df.loc[df["Gender"] != "No_gender", "sexism"] = 1

if "Religion" in df.columns:
    df.loc[df["Religion"] != "Nonreligious", "religious_hate"] = 1

# --------------------------------------------------
# STEP 5: Keep only required columns
# --------------------------------------------------

final_columns = ["text"] + labels
df = df[final_columns]

# --------------------------------------------------
# STEP 6: Save converted dataset
# --------------------------------------------------

df.to_csv("dataset2_multilabel.csv", index=False)

print("Dataset 2 converted successfully!")
print("Total rows:", len(df))
