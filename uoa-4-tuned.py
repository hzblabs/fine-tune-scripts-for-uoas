#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install transformers datasets accelerate -q')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import pandas as pd
from datasets import Dataset

# Load your JSON file
df = pd.read_json("/content/drive/MyDrive/Colab Notebooks/combined_training_data.json")

# If file is JSON Lines
# df = pd.read_json("/content/drive/MyDrive/Notebooks/ref_training_data.json", lines=True)

# Remove exact duplicates
df = df.drop_duplicates()

# Optionally, remove rows with duplicate text only (keep first occurrence)
# df = df.drop_duplicates(subset="text")

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Preview
print(f"Dataset length after duplicates removed: {len(dataset)}")
print(dataset[0])



# In[ ]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# === 1. Label mapping ===
label_map = {"4*": 0, "3*": 1, "2*": 2, "1*": 3}
dataset = dataset.map(lambda x: {"label": label_map.get(x["label"], -1)})
dataset = dataset.filter(lambda x: x["label"] != -1)

# === 2. Load BigBird tokenizer ===
model_name = "google/bigbird-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# === 3. Tokenize dataset
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=4096)

tokenized_dataset = dataset.map(tokenize, batched=True)

# === 4. Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

# === 5. TrainingArguments
training_args = TrainingArguments(
    output_dir="./bigbird_ref_model",
    per_device_train_batch_size=1,              # bigbird is memory-heavy
    gradient_accumulation_steps=4,              # simulate batch size of 4
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=5,
    save_steps=10,
    save_total_limit=2,
    report_to="none"                            # avoid wandb, etc.
)

# === 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# === 7. Train!
trainer.train()

# === 8. Save the model
trainer.save_model("/content/bigbird_ref_star_classifier")
tokenizer.save_pretrained("/content/bigbird_ref_star_classifier")

print("✅ BigBird fine-tuning complete. Model saved to: /content/bigbird_ref_star_classifier")


# **INFERENCE**

# In[ ]:


pip install pymupdf


# In[ ]:


import zipfile
import os

zip_path = "/content/new_pdfs.zip"
extract_dir = "/content/ref_pdfs"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("✅ Extracted to:", extract_dir)


# In[ ]:


import os
import fitz  # PyMuPDF
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# === 1. Paths ===
pdf_folder = "/content/ref_pdfs/new_pdfs"  # 🔁 upload or mount this folder
model_path = "/content/bigbird_ref_star_classifier"
output_csv = "/content/ref_predictions.csv"

# === 2. Load model + tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# === 3. Label mapping ===
label_map = {0: "4*", 1: "3*", 2: "2*", 3: "1*"}

# === 4. PDF text extractor ===
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = " ".join([page.get_text() for page in doc])
        return " ".join(text.strip().split())  # clean up
    except Exception as e:
        print(f"❌ Error in {pdf_path}: {e}")
        return None

# === 5. Predict from text ===
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=4096)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
        return label_map[pred], round(confidence, 4)

# === 6. Process PDFs ===
results = []
print(f"📂 Scanning folder: {pdf_folder}")

for fname in os.listdir(pdf_folder):
    if fname.endswith(".pdf"):
        fpath = os.path.join(pdf_folder, fname)
        print(f"\n🔍 Processing: {fname}")

        text = extract_text_from_pdf(fpath)
        if text:
            label, conf = predict(text)
            print(f"✅ Predicted: {label} (Confidence: {conf})")
            results.append({"filename": fname, "predicted_label": label, "confidence": conf})
        else:
            print("⚠️ Skipped due to extraction failure.")

# === 7. Save to CSV ===
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"\n📄 All predictions saved to: {output_csv}")

