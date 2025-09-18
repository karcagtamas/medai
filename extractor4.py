import json
import os
from typing import List, Dict, Any
import re

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
from tqdm import tqdm

MODEL_NAME = "t5-small"
KEYS = [
    "name",
    "patient_id",
    "date_of_birth",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "heart_rate",
    "spo2",
    "temperature"
]
INSTRUCTION_PREFIX = "Extract the following keys as a JSON object with these exact keys (use empty string if missing): " + ", ".join(
    KEYS) + ".\n\nReport:\n"


def canonicalize_target(g: Dict[str, Any], keys: List[str]) -> Dict[str, str]:
    return {k: (str(g.get(k, "")) if g.get(k, "") is not None else "") for k in keys}


def safe_json_parse(text: str) -> Dict[str, Any]:
    if text is None:
        return {"_raw", None}
    t = text.strip()

    try:
        return json.loads(t)
    except Exception:
        pass

    try:
        candidate = t
        if not candidate.startswith("{"):
            candidate = "{" + candidate
        if not candidate.endswith("}"):
            candidate = candidate + "}"
        return json.loads(candidate)
    except Exception:
        pass

    try:
        candidate = t
        candidate = candidate.replace("'", '"')
        candidate = re.sub(r'(\b[a-zA-Z_][a-zA-Z0-9_\- ]*\b)\s*:', r'"\1"', candidate)
        candidate = re.sub(r',\s*}', '}', candidate)
        return json.loads(candidate)
    except Exception:
        pass

    return {"_raw", text}


def fill_missing_keys(pred: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    out = {}
    for k in keys:
        v = pred.get(k, "")
        if isinstance(v, dict) or isinstance(v, list):
            # convert to string if model put nested structure
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = "" if v is None else str(v)
    return out


class MedDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer, max_input_len=512, max_target_len=256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = INSTRUCTION_PREFIX + sample["input"].strip() + "\n\nReturn JSON:"
        target_text = sample["output_str"]  # already canonical JSON string

        input_enc = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_input_len,
            return_tensors="pt",
        )

        target_enc = self.tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_target_len,
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze()
        # set pad token ids to -100 for loss ignoring
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels
        }


def load_dataset(folder_name: str, tokenizer) -> MedDataset:
    """Load dataset from folder with /input/*.txt and /output/*.json. Ensures canonical target JSON string."""
    input_folder = os.path.join(folder_name, "input")
    output_folder = os.path.join(folder_name, "output")

    samples = []
    for file_name in sorted(os.listdir(input_folder)):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)
        if os.path.isfile(input_file_path):
            with open(input_file_path, "r", encoding="utf-8") as f:
                raw_in = f.read()

            # corresponding json file assumed to be same basename with .json
            out_json_path = output_file_path.replace(".txt", ".json")
            if not os.path.exists(out_json_path):
                # try same base name but .json
                base = os.path.splitext(file_name)[0]
                out_json_path = os.path.join(output_folder, base + ".json")
            if not os.path.exists(out_json_path):
                # skip if no label
                print(f"[WARN] Missing label JSON for {input_file_path}, skipping.")
                continue

            with open(out_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)  # dict

            canonical = canonicalize_target(data, KEYS)
            samples.append({
                "input": raw_in,
                # store canonical JSON string for tokenizer
                "output_str": json.dumps(canonical, ensure_ascii=False)
            })
    return MedDataset(samples, tokenizer)


def train(model, train_dataloader, val_dataloader, optimizer, tokenizer, device, epochs=5):
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch} Train"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_dataloader))

        # validation (compute loss and also sample a few predictions)
        model.eval()
        val_loss = 0.0
        sample_preds = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch} Val")):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                # capture a few samples from the first batch for inspection
                if i == 0:
                    gen = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=256,
                        num_beams=4,
                        length_penalty=1.0,
                        no_repeat_ngram_size=2,
                        early_stopping=True
                    )
                    for j in range(len(input_ids)):
                        inp_dec = tokenizer.decode(input_ids[j], skip_special_tokens=True)
                        lbl = labels[j]
                        lbl = [t.item() for t in lbl if t.item() != -100]
                        tar_dec = tokenizer.decode(lbl, skip_special_tokens=True)
                        pred_dec = tokenizer.decode(gen[j], skip_special_tokens=True)
                        sample_preds.append((inp_dec, tar_dec, pred_dec))

        avg_val_loss = val_loss / max(1, len(val_dataloader))
        print(f"Epoch {epoch} | Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")
        # show sample preds
        print("Sample validation predictions (raw):")
        for (inp, tar, pred) in sample_preds:
            print("INPUT:", inp[:200].replace("\n", " "))
            print("TARGET:", tar)
            print("PRED:", pred)
            print("---")


def predict_and_save(model, test_loader, tokenizer, device, save_path="results.json"):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256,
                num_beams=5,
                length_penalty=1.0,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )

            for i in range(len(input_ids)):
                inp_dec = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                labels = batch["labels"][i]
                labels = [t.item() for t in labels if t.item() != -100]
                tar_dec = tokenizer.decode(labels, skip_special_tokens=True)
                pred_dec = tokenizer.decode(gen[i], skip_special_tokens=True)

                tar_parsed = safe_json_parse(tar_dec)
                pred_parsed = safe_json_parse(pred_dec)

                # If parsed produced raw text, try to salvage key-values by filling keys
                if not isinstance(tar_parsed, dict):
                    tar_parsed = {"_raw": tar_dec}
                if not isinstance(pred_parsed, dict):
                    pred_parsed = {"_raw": pred_dec}

                tar_filled = fill_missing_keys(tar_parsed, KEYS)
                pred_filled = fill_missing_keys(pred_parsed, KEYS)

                preds.append({
                    "input_text": inp_dec,
                    "target": tar_filled,
                    "prediction": pred_filled
                })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(preds)} predictions to {save_path}")
    return preds


def main():
    train_folder = "data/train"
    val_folder = "data/val"
    test_folder = "data/test"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_dataset = load_dataset(train_folder, tokenizer)
    val_dataset = load_dataset(val_folder, tokenizer)
    test_dataset = load_dataset(test_folder, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    test_loader = DataLoader(test_dataset, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    train(model, train_loader, val_loader, optimizer, tokenizer, device, epochs=5)

    predict_and_save(model, test_loader, tokenizer, device, save_path="results.json")


if __name__ == "__main__":
    main()
