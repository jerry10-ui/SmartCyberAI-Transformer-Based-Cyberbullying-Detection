import pandas as pd
import os

folder_path = r"c:/Coding/Projects/Minor Project/dataset/Cyberbullying Dataset"

# Define mapping from filename to label
label_mapping = {
    "aggression_parsed_dataset.csv": "aggression",
    "attack_parsed_dataset.csv": "bullying",
    "toxicity_parsed_dataset.csv": "toxic",
    "twitter_racism_parsed_dataset.csv": "racism",
    "twitter_sexism_parsed_dataset.csv": "sexism",
    "youtube_parsed_dataset.csv": "toxic",
    "twitter_parsed_dataset.csv": "toxic",
    "kaggle_parsed_dataset.csv": "toxic"
}

all_data = []

for file_name, label_name in label_mapping.items():
    file_path = os.path.join(folder_path, file_name)
    
    df = pd.read_csv(file_path)
    
    # Keep only necessary columns
    df = df[["Text", "oh_label"]]
    
    # Rename text column consistently
    df.rename(columns={"Text": "text"}, inplace=True)
    
    # Create label column initialized to 0
    df[label_name] = df["oh_label"]
    
    # Keep only text and the assigned label
    df = df[["text", label_name]]
    
    all_data.append(df)

# Merge all
merged_df = pd.concat(all_data, axis=0)

# Fill missing label columns with 0
merged_df = merged_df.fillna(0)

# Group by text to combine labels (multi-label creation)
final_df = merged_df.groupby("text").max().reset_index()

# Save
final_df.to_csv("merged_dataset_folder1.csv", index=False)

print("Dataset built successfully!")
