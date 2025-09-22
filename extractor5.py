import pdfplumber
from pdf2image import convert_from_path
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
import torch
from PIL import Image

model_name = "nielsr/layoutlmv3-finetuned-funsd"
processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)


def pdf_page_to_image(pdf_path, page_number):
    pages = convert_from_path(pdf_path, dpi=300)
    image = pages[page_number]

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def extract_text_and_boxes(pdf_path):
    pages_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words()
            page_words = []
            for word in words:
                page_words.append({
                    "text": word["text"],
                    "bbox": [
                        word["x0"] / page.width,
                        word["top"] / page.height,
                        word["x1"] / page.width,
                        word["bottom"] / page.height
                    ],
                })

            pages_data.append({
                "words": page_words,
                "image": pdf_page_to_image(pdf_path, page.page_number - 1),
            })

    return pages_data


def prepare_inputs(page_data):
    texts = [w["text"] for w in page_data["words"]]
    boxes = [w["bbox"] for w in page_data["words"]]
    encoding = processor(text=texts, boxes=boxes, images=[page_data["image"]], return_tensors="pt",
                         padding="max_length", truncation=True)

    #for key in ["bbox"]:
    #    if key in encoding:
    #        encoding[key] = encoding[key].long()

    return encoding, texts


def predict_kv(page_data):
    encoding, texts = prepare_inputs(page_data)

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()

    labels = [model.config.id2label[p] for p in predictions][:len(texts)]
    kv_pairs = {}
    current_key = None
    for word, label in zip(texts, labels):
        if "KEY" in label:
            current_key = word
        elif "VALUE" in label and current_key:
            kv_pairs[current_key] = word
            current_key = None
    return kv_pairs


pdf_path = "sample_report.pdf"
pages_data = extract_text_and_boxes(pdf_path)
print(pages_data)
all_kv_pairs = []
for page_data in pages_data:
    kv = predict_kv(page_data)
    all_kv_pairs.append(kv)

print(all_kv_pairs)
