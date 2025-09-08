import os
import re
import json
import pdfplumber
import docx
from transformers import pipeline

ner = pipeline("ner", model="kamalkraj/bert-medical-ner", aggregation_strategy="simple")


def load_txt(path: str) -> str:
    with open(path, "r", encodings="utf-8", errors="ignore") as f:
        return f.read()


def load_docx(path: str) -> str:
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def load_pdf(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def load_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt": return load_txt(path)
    if ext == ".docx": return load_docx(path)
    if ext == ".pdf": return load_pdf(path)
    raise ValueError(f"Unsupported file type: {ext}")


def normalize_entities(entities):
    out = {
        "name": None,
        "patient_id": None,
        "date_of_birth": None,
        "report_date": None,
        "blood_pressure_systolic": None,
        "blood_pressure_diastolic": None,
        "spo2": None,
        "heart_rate": None,
        "temperature_c": None,
        "respiratory_rate": None,
        "glucose_mgdl": None,
    }

    for ent in entities:
        text = ent["word"]
        label = ent["entity_group"].lower()

        if "name" in label and out["name"] is None:
            out["name"] = text

        if "id" in label and out["patient_id"] is None:
            out["patient_id"] = text

        if "date" in label:
            if out["date_of_birth"] is None:
                out["date_of_birth"] = text
            else:
                out["report_date"] = text

        if "blood" in label or "bp" in text.lower():
            m = re.search(r"(\d{2,3})[\/\-](\d{2,3})", text)
            if m:
                out["blood_pressure_systolic"] = int(m.group(1))
                out["blood_pressure_diastolic"] = int(m.group(2))

        if "spo" in text.lower() or "oxygen" in text.lower():
            m = re.search(r"(\d{2,3})", text)
            if m: out["spo2"] = int(m.group(1))

        if "heart" in text.lower() or "pulse" in text.lower():
            m = re.search(r"(\d{2,3})", text)
            if m: out["heart_rate"] = int(m.group(1))

        if "temp" in text.lower():
            m = re.search(r"(\d{2,3}\.?\d*)", text)
            if m:
                val = float(m.group(1))
                if "f" in text.lower():
                    val = round((val - 32) * 5 / 9, 1)
                out["temperature_c"] = val

        if "resp" in text.lower():
            m = re.search(r"(\d{1,2})", text)
            if m: out["respiratory_rate"] = int(m.group(1))

        if "glucose" in text.lower() or "sugar" in text.lower():
            m = re.search(r"(\d+\.?\d*)", text)
            if m:
                val = float(m.group(1))
                if "mmol" in text.lower():
                    val = round(val * 18, 1)  # convert mmol/L to mg/dL
                out["glucose_mgdl"] = val

    return out


def extract(path: str):
    text = load_text(path)
    entities = ner(text)
    return normalize_entities(entities)


if __name__ == "__main__":
    sample_files = ["sample_report.txt", "sample_report.docx", "sample_report.pdf"]

    for f in sample_files:
        if os.path.exists(f):
            print(f"\n--- Extracting from {f} ---")
            data = extract(f)
            print(json.dumps(data, indent=2))
