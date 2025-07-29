#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wandb disabled')


# In[2]:


import os
os.environ["WANDB_DISABLED"] = "true"


# In[3]:


from transformers import BigBirdTokenizer, BigBirdForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, ClassLabel, load_metric
import numpy as np
import torch


# In[4]:


# Load the tokenizer and model
model_name = "google/bigbird-roberta-base"
tokenizer = BigBirdTokenizer.from_pretrained(model_name)
model = BigBirdForSequenceClassification.from_pretrained(model_name, num_labels=3)


# In[5]:


# Load dataset
import pandas as pd
from datasets import Dataset

# Load the JSON file into a pandas DataFrame
df = pd.read_json("uoa-11-trainset-cleaned.json")

# Convert the pandas DataFrame to a datasets Dataset
dataset = Dataset.from_pandas(df)


# In[6]:


# Label adjustment: convert 2, 3, 4 to 0, 1, 2 for model
def preprocess_labels(example):
    example["label"] = int(example["label"]) - 2
    return example

dataset = dataset.map(preprocess_labels)


# In[7]:


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


# In[8]:


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


# Use the full dataset for training without splitting
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train
trainer.train()


# In[ ]:


# Save model
trainer.save_model("uoa_11_model")
tokenizer.save_pretrained("uoa_11_model")

# Print final training logs
for log in trainer.state.log_history:
    print(log)

