import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import f1_score
from tqdm import tqdm

# Configuration
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 2
MODEL_NAME = "distilbert-base-uncased"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Labels
labels = [
    "hate", "offensive", "aggression", "bullying", "toxic", "racism", "sexism", "religious_hate"
]

# Load data
train_df = pd.read_csv("../../data/processed/train.csv")
val_df = pd.read_csv("../../data/processed/val.csv")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

class CyberDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = df[labels].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

train_dataset = CyberDataset(train_df)
val_dataset = CyberDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    problem_type="multi_label_classification"
)

model.to(DEVICE)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader)
    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels_tensor = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_tensor
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Loss:", total_loss / len(train_loader))

    # Validation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_tensor = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs.logits)

            all_preds.append(preds.cpu())
            all_labels.append(labels_tensor.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    preds_binary = (all_preds > 0.5).int()

    f1 = f1_score(all_labels, preds_binary, average="macro")
    print("Validation F1 Score:", f1)

# Save model
model.save_pretrained("../../models/saved_model")
tokenizer.save_pretrained("../../models/saved_model")

print("Model saved successfully!")
