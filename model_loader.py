from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

MODEL_DIR = "./local_model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# Load label map
with open(f"{MODEL_DIR}/label2text.json", "r") as f:
    label2text = json.load(f)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Decode label
    label = label2text.get(str(predicted_class), f"Unknown class {predicted_class}")
    return {
        "class_id": predicted_class,
        "prediction": label
    }
