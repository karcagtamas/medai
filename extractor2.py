import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
import random
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


SYNTHETIC_REPORTS = [
    {
        "text": """Hospital XYZ
Patient Name: Alice Smith
Patient ID: A-112233
DOB: 12/03/1980
Report Date: 09/08/2025
Blood Pressure: 118/76 mmHg
Heart Rate: 72 bpm
SpO2: 96%
Temperature: 37.2 C
Respiratory Rate: 16 /min
        Glucose: 100 mg/dL""",
        "labels": {
            "name": "Alice Smith",
            "patient_id": "A-112233",
            "date_of_birth": "12/03/1980",
            "report_date": "09/08/2025",
            "blood_pressure_systolic": "118",
            "blood_pressure_diastolic": "76",
            "spo2": "96",
            "heart_rate": "72",
            "temperature_c": "37.2",
            "respiratory_rate": "16",
            "glucose_mgdl": "100"
        }
    },
    {
        "text": """General Hospital
Patient Name: John Doe
Patient ID: P-998877
DOB: 05/12/1975
Report Date: 08/09/2025
Blood Pressure: 120/80 mmHg
Heart Rate: 70 bpm
SpO2: 97%
Temperature: 98.6 F
Respiratory Rate: 18 /min
        Glucose: 6.0 mmol/L""",
        "labels": {
            "name": "John Doe",
            "patient_id": "P-998877",
            "date_of_birth": "05/12/1975",
            "report_date": "08/09/2025",
            "blood_pressure_systolic": "120",
            "blood_pressure_diastolic": "80",
            "spo2": "97",
            "heart_rate": "70",
            "temperature_c": "37.0",  # converted from F
            "respiratory_rate": "18",
            "glucose_mgdl": "108"   # converted from mmol/L
        }
    }
]

FIELDS = list(SYNTHETIC_REPORTS[0]["labels"].keys())


class MedReportExtractor(nn.Module):

    def __init__(self, base_model="distilbert-base-uncased", num_fields=10):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(base_model)
        hidden_size = self.encoder.config.hidden_size

        self.field_heads = nn.ModuleDict({
            "name": nn.Linear(hidden_size, 256),       # predict span embedding
            "patient_id": nn.Linear(hidden_size, 128),
            "date_of_birth": nn.Linear(hidden_size, 64),
            "report_date": nn.Linear(hidden_size, 64),
            "blood_pressure_systolic": nn.Linear(hidden_size, 32),
            "blood_pressure_diastolic": nn.Linear(hidden_size, 32),
            "spo2": nn.Linear(hidden_size, 32),
            "heart_rate": nn.Linear(hidden_size, 32),
            "temperature_c": nn.Linear(hidden_size, 32),
            "glucose_mgdl": nn.Linear(hidden_size, 32),
        })

        self.pooler = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        pooled = hidden_states.mean(dim=1)

        out = {}
        for field, head in self.field_heads.items():
            out[field] = head(pooled)
        return out

class MedDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        enc = self.tokenizer(ex["text"], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        labels = ex["labels"]

        return {
            "inputs_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": labels
        }

def constrative_loss(model, pred_vecs, labels, tokenizer, device):
    total_loss = 0.0
    for field, pred in pred_vecs.items():
        label_text = labels[field]
        enc = tokenizer(label_text, return_tensors="pt").to(device)
        with torch.no_grad():
            gold_emb = model.encoder(**enc).last_hidden_state.mean(dim=1)

        cos = nn.CosineSimilarity(dim=-1)
        sim = cos(pred, gold_emb)
        loss = 1 - sim.mean()
        total_loss += loss
    return total_loss / len(pred_vecs)

def train(model, dataloader, tokenizer, device, epochs=3):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = constrative_loss(model, preds, labels, tokenizer, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")

def predict(model, tokenizer, text, device):
    model.eval()
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        preds = model(enc["input_ids"], enc["attention_mask"])

    results = {}
    for field, vec in preds.items():
        results[field] = f"[vector-norm {vec.norm().item():.2f}]"
    return results

if __name__ == "__main__":
    device = "cude" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = MedDataset(SYNTHETIC_REPORTS, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = MedReportExtractor().to(device)
    train(model, dataloader, tokenizer, device, epochs=2)

    test_text = SYNTHETIC_REPORTS[0]["text"]
    output = predict(model, tokenizer, test_text, device)
    print(json.dumps(output, indent=2))
