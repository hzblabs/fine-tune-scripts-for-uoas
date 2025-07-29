#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('wandb disabled')


# In[ ]:


import os
os.environ["WANDB_DISABLED"] = "true"


# In[ ]:


from transformers import BigBirdTokenizer, BigBirdForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, ClassLabel, load_metric
import numpy as np
import torch


# In[ ]:


# Load the tokenizer and model
model_name = "google/bigbird-roberta-base"
tokenizer = BigBirdTokenizer.from_pretrained(model_name)
model = BigBirdForSequenceClassification.from_pretrained(model_name, num_labels=3)


# In[ ]:


# Load dataset
import pandas as pd
from datasets import Dataset

# Load the JSON file into a pandas DataFrame
df = pd.read_json("uoa-20-trainset-cleaned.json", lines=True)

# Convert the pandas DataFrame to a datasets Dataset
dataset = Dataset.from_pandas(df)


# In[ ]:


# Label adjustment: convert 2, 3, 4 to 0, 1, 2 for model
def preprocess_labels(example):
    example["label"] = int(example["label"]) - 2
    return example

dataset = dataset.map(preprocess_labels)


# In[ ]:


# Tokenization
def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=4096,
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)


# In[ ]:


# Training args
training_args = TrainingArguments(
    output_dir="./bigbird-ref",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=5,
    save_steps=10,
    save_total_limit=2,
    report_to="none",              # ensures output is shown in notebook
    disable_tqdm=False,            # enables tqdm logging bar
    logging_strategy="steps",      # explicitly logs by steps
)


# In[ ]:


# Split dataset into train and test sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
tokenized_dataset = train_test_split


# In[ ]:


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

# Train
trainer.train()


# In[ ]:


# Save model
trainer.save_model("final_model")
tokenizer.save_pretrained("final_model")

# Print final training logs
for log in trainer.state.log_history:
    print(log)


# **INFERENCE**

# In[ ]:


get_ipython().run_line_magic('pip', 'install PyPDF2')


# In[ ]:


import os
import zipfile
import glob
import pandas as pd
import torch
import PyPDF2
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Paths ----
zip_path = "./pdfs.zip"
extract_dir = "./pdfs_unzipped"
csv_path = "uoa-20-ots.csv"
model_name = "./final_model"

# ---- Unzip PDFs ----
if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        print("Extracted:", zip_ref.namelist())

# ---- Load model and tokenizer ----
tokenizer = BigBirdTokenizer.from_pretrained(model_name)
model = BigBirdForSequenceClassification.from_pretrained(model_name)
model.eval()

# ---- Extract text from PDF ----
def extract_text_from_pdf(pdf_file):
    try:
        with open(pdf_file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading {pdf_file}: {e}")
        return None

# ---- Predict score ----
def predict_pdf(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    if text is None or not text.strip():
        return "No text"

    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=4096,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return {0: "2*", 1: "3*", 2: "4*"}.get(predicted_class, "Unknown")

# ---- Load ground truth ----
df = pd.read_csv(csv_path)
ground_truth = dict(zip(df["PDF_File"], df["Assigned_Label"]))

# ---- Predict and Compare ----
pdf_files = glob.glob(os.path.join(extract_dir, "**", "*.pdf"), recursive=True)
print(f"Found {len(pdf_files)} PDF files.\n")

y_true = []
y_pred = []

for full_path in pdf_files:
    filename = os.path.basename(full_path)
    true_label = ground_truth.get(filename)

    if not true_label:
        print(f"⚠️ Skipping {filename}: No REF score found in CSV.")
        continue

    predicted_label = predict_pdf(full_path)

    if predicted_label == "No text":
        print(f"⚠️ Skipping {filename}: Could not extract text.")
        continue

    print(f"{filename}: True = {true_label}, Predicted = {predicted_label}")

    y_true.append(true_label)
    y_pred.append(predicted_label)

# ---- Evaluation ----
print("\n--- Evaluation Metrics ---")
label_order = ["2*", "3*", "4*"]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, labels=label_order))
print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred, labels=label_order)
print(cm)

# ---- Plot Confusion Matrix ----
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_order, yticklabels=label_order)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ---- Save predictions to CSV ----
results_df = pd.DataFrame({
    "PDF_File": [os.path.basename(p) for p in pdf_files if os.path.basename(p) in ground_truth and extract_text_from_pdf(p) is not None and extract_text_from_pdf(p).strip()],
    "True_Label": y_true,
    "Predicted_Label": y_pred
})
results_df.to_csv("ref_predictions.csv", index=False)
print("\n✅ Saved predictions to 'ref_predictions.csv'")

