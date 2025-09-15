import json
import os.path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
from tqdm import tqdm


class MedDataset(Dataset):
    def __init__(self, samples, tokenizer, max_input_len=512, max_target_len=256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_enc = self.tokenizer(
            sample["input"],
            truncation=True,
            padding="max_length",
            max_length=self.max_input_len,
            return_tensors="pt"
        )

        target_enc = self.tokenizer(
            sample["output"],
            truncation=True,
            padding="max_length",
            max_length=self.max_target_len,
            return_tensors="pt"
        )

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": target_enc["input_ids"].squeeze()
        }


def load_dataset(folder_name: str, tokenizer) -> MedDataset:
    input_folder = os.path.join(folder_name, "input")
    output_folder = os.path.join(folder_name, "output")

    samples = []
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path):
            item = {}
            with open(file_path, 'r') as file:
                item["input"] = file.read()

            with open(file_path.replace(".txt", ".json")) as file:
                item["output"] = json.dumps(json.load(file))

            samples.append(item)

    return MedDataset(samples, tokenizer)


def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, device, epochs=5):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} - Training"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch + 1} - Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        print(f"Epoch {epoch + 1} | Train loss: {avg_train_loss:.4f} | Val loss {avg_val_loss:.4f}")


def predict(text, model, tokenizer, device):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.generate(**inputs, max_length=256)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        return json.loads(decoded)
    except:
        return decoded


def main():
    train_folder = "data/train"
    val_folder = "data/val"
    test_folder = "data/test"

    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_dataset = load_dataset(train_folder, tokenizer)
    val_dataset = load_dataset(val_folder, tokenizer)
    test_dataset = load_dataset(test_folder, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4)
    test_dataloader = DataLoader(test_dataset, batch_size=4)
    device = torch.device("cude" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    train(model, train_dataloader, val_dataloader, optimizer, device, epochs=5)

    test_text = """City Clinic
    Patient Name: Charlie Adams
    Patient ID: C-778899
    DOB: 03/14/1990
    BP: 122/78 mmHg
    HR: 76 bpm
    SpO2: 97%
    Temp: 36.8 C
    """

    print("\nPrediction on test report:")
    print(predict(test_text, model, tokenizer, device))


if __name__ == "__main__":
    main()
