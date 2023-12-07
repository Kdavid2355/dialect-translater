import os
import torch
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
#os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

tokenizer=AutoTokenizer.from_pretrained('KETI-AIR/long-ke-t5-base')
model=AutoModelForSeq2SeqLM.from_pretrained('KETI-AIR/long-ke-t5-base')

path="./data/merge.csv"
df = pd.read_csv(path)

max_input_length=256
max_target_length=256
batch_size= 90
learning_rate = 1e-4
num_epochs = 3

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_length, max_target_length):
        self.data = df
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src_text = row['표준어']
        tgt_text = row['방언']

        inputs = self.tokenizer(
            src_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        targets = self.tokenizer(
            tgt_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze()
        }

dataset = CustomDataset(df, tokenizer, max_input_length, max_target_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = torch.nn.DataParallel(model)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        labels[labels[:, :] == tokenizer.pad_token_id] = -100 
        labels = labels.to(device)


        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss.mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 50 == 49:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
    
    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Average Loss: {average_loss:.4f}")

    model_save_path = f"{epoch}_{average_loss}.bin"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
