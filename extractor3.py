import json
import os.path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
from tqdm import tqdm


def safe_json_parse(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads("{" + text.strip().strip(",") + "}")
        except Exception:
            return {"_raw": text}


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

        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels
        }


def load_dataset(folder_name: str, tokenizer) -> MedDataset:
    input_folder = os.path.join(folder_name, "input")
    output_folder = os.path.join(folder_name, "output")

    samples = []
    for file_name in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)
        if os.path.isfile(input_file_path):
            item = {}
            with open(input_file_path, 'r') as file:
                item["input"] = file.read()

            with open(output_file_path.replace(".txt", ".json")) as file:
                data = json.load(file)
                item["output"] = json.dumps(data, ensure_ascii=False)

            samples.append(item)

    return MedDataset(samples, tokenizer)


def train(model, train_dataloader, val_dataloader, optimizer, device, epochs=5):
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


def predict(model, test_loader, tokenizer, device, max_len=512):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_len,
                num_beams=4,
                early_stopping=True
            )

            for i in range(len(input_ids)):
                input_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)

                labels = batch["labels"][i]
                labels = [t.item() for t in labels if t.item() != -100]
                target_text = tokenizer.decode(labels, skip_special_tokens=True)
                pred_text = tokenizer.decode(outputs[i], skip_special_tokens=True)

                preds.append({
                    "input_text": input_text,
                    "target_text": safe_json_parse(target_text),
                    "predicted_text": safe_json_parse(pred_text)
                })

    return preds


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    train(model, train_dataloader, val_dataloader, optimizer, device, epochs=15)

    preds = predict(model, test_dataloader, tokenizer, device)

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(preds)} predictions to results.json")


if __name__ == "__main__":
    main()
