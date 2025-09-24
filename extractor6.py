import os
from typing import List, Dict, Any

import evaluate
from PIL import Image
import numpy as np

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

MODEL_NAME = "nielsr/layoutlmv3-finetuned-funsd"
OUTPUT_DIR = "./layoutlmv3-kv-finetuned"

LABEL_LIST = [
    "O",
    "B-QUESTION", "I-QUESTION",
    "B-ANSWER", "I-ANSWER",
    "B-HEADER", "I-HEADER"
]
label2id = {l: i for i, l in enumerate(LABEL_LIST)}
id2label = {i: l for l, i in label2id.items()}

# EXAMPLES
toy_examples = {
    "id": ["doc1"],
    "tokens": [
        # example tokens for one page
        ["Hospital", "ABC", "Medical", "Center", "Patient", "Name:", "John", "Doe",
         "Patient", "ID:", "P-998877", "DOB:", "05/12/1975", "Report", "Date:", "08/09/2025",
         "Vitals:", "Blood", "Pressure:", "120/80", "mmHg",
         "Heart", "Rate:", "70", "bpm", "SpO2:", "97%",
         "Temperature:", "98.6", "F", "Respiratory", "Rate:", "18", "/min", "Glucose:", "6.0", "mmol/L"]
    ],
    # toy bboxes: here we just create plausible boxes (x0,y0,x1,y1) in pixel coords
    # In real life: use pdfplumber or OCR to get actual bounding boxes per token.
    "bboxes": [
        [[10 + i * 40, 10 + (i // 10) * 20, 10 + i * 40 + 30, 10 + (i // 10) * 20 + 16] for i in range(37)]
    ],
    "labels": [
        # Corresponding labels (strings) matching tokens above
        ["O", "B-QUESTION", "B-QUESTION", "O", "O", "B-QUESTION", "B-QUESTION", "I-QUESTION",
         "B-ANSWER", "B-ANSWER", "B-ANSWER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER", "I-ANSWER",
         "B-HEADER", "I-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION",
         "B-ANSWER", "I-ANSWER", "I-ANSWER", "I-ANSWER", "I-ANSWER", "I-ANSWER",
         "B-QUESTION", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER", "I-ANSWER", "I-ANSWER", "I-ANSWER",
         "I-ANSWER"]
    ],
    # One simple placeholder image path. Replace it with real images.
    # For testing, we create a blank image and save to disk (so processor has something).
    "image_path": ["./toy_doc1_page0.png"],
}


def preprocess_examples(examples: Dict[str, Any]) -> Dict[str, Any]:
    pixel_values_list = []
    input_ids_list = []
    attention_mask_list = []
    bbox_list = []
    labels_list = []

    processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)

    for i in range(len(examples["tokens"])):
        image_path = examples["image_path"][i]

        image = Image.open(image_path).convert("RGB")

        tokens = examples["tokens"][i]
        bboxes = examples["bboxes"][i]
        word_labels = examples["label_ids"][i]

        encoding = processor(
            images=image,
            text=tokens,
            boxes=bboxes,
            word_labels=word_labels,
            padding="max_length",
            trunaction=True,
            return_tensors="pt"
        )

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in encoding:
                encoding[key] = encoding[key].long()

        pixel_values_list.append(encoding["pixel_values"])
        input_ids_list.append(encoding["input_ids"])
        attention_mask_list.append(encoding["attention_mask"])
        bbox_list.append(encoding["bbox"])
        labels_list.append(encoding["labels"])

    out = {
        "pixel_values": pixel_values_list,
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "bbox": bbox_list,
        "labels": labels_list,
        "image_path": examples["image_path"]
    }

    return out


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    true_labels, pred_labels = [], []

    for label_ids, pred_ids in zip(p.label_ids, preds):
        seq_true, seq_pred = [], []
        for lid, pid in zip(label_ids, pred_ids):
            if lid == -100:
                continue
            seq_true.append(id2label[int(lid)])
            seq_pred.append(id2label[int(pid)])
        true_labels.append(seq_true)
        pred_labels.append(seq_pred)

    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=pred_labels, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "acciracy": results.get("overall_accuracy", 0.0)
    }


def convert_to_torch(example: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pixel_values": example["pixel_values"],
        "input_ids": example["input_ids"],
        "attention_mask": example["attention_mask"],
        "bbox": example["bbox"],
        "labels": example["labels"],
    }


def main():
    dataset = Dataset.from_dict({
        "id": toy_examples["id"],
        "tokens": toy_examples["tokens"],
        "bboxes": toy_examples["bboxes"],
        "label_ids": list(map(lambda x: list(map(lambda y: label2id[y], x)), toy_examples["labels"])),
        "image_path": toy_examples["image_path"],
    })

    dataset_dict = DatasetDict({"train": dataset, "validation": dataset})
    processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id,
    )
    encoded_ds = dataset_dict.map(
        preprocess_examples,
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
    )
    encoded_ds = encoded_ds.map(lambda ex: ex, batched=True)
    encoded_ds.set_format(type="torch", columns=["pixel_values", "input_ids", "attention_mask", "bbox", "labels"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_ds["train"],
        eval_dataset=encoded_ds["validation"],
        tokenizer=processor.feature_extractor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training examples: ", len(encoded_ds["train"]))
    print("Validation examples: ", len(encoded_ds["validation"]))

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print("Model save to ", OUTPUT_DIR)


if __name__ == "__main__":
    main()
