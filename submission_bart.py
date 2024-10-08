import re
from tqdm import tqdm
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# 1. Load the data
test_df = pd.read_csv('data/test_clean.csv', engine='python', encoding='latin-1')
categories_str = "NEWS AND POLITICS|AUTOMOTIVES|ARTS AND CULTURE|HOBBIES AND INTERESTS|FOOD AND DRINKS|ACADEMIC INTERESTS|PERSONAL FINANCE|HEALTHY LIVING|HOME AND GARDEN|PETS|REAL ESTATE|TELEVISION|VIDEO GAMING|FAMILY AND RELATIONSHIPS|BUSINESS AND FINANCE|MOVIES|SHOPPING|BOOKS AND LITERATURE|CAREERS|STYLE AND FASHION|MUSIC AND AUDIO|SPORTS|TECHNOLOGY AND COMPUTING|TRAVEL|HEALTH|PHARMACEUTICALS, CONDITIONS, AND SYMPTOMS"
print(categories_str)

# clean the text
def clean_text(text):
    # Remove unwanted characters except for newlines
    text = re.sub(r'[^a-zA-Z0-9\s\.,?!\n]', '', text)
    # Collapse multiple spaces (but keep newlines intact)
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace 3 or more newlines with exactly 2 newlines
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)
    # Convert to lowercase
    text = text.lower()
    return text

# 2. Define the prompt
def add_prompt(text):
    prompt = (
        f"Classify the following text into one of the following categories: {categories_str}.\n\n"
        f"Text: {text}\nCategory:"
    )
    return prompt

test_df['text'] = test_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(add_prompt)


model_name = "bart-large-cnn/checkpoint-10000"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")


def generate_results(texts, model, tokenizer):
    with torch.no_grad():
        input_ids = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
        generated_ids = model.generate(input_ids.to("cuda"))
        generated_ids = generated_ids.cpu()
        results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return results

# 3. Generate predictions on a batch of batch size 8
batch_size = 256
results = []
for i in tqdm(range(0, len(test_df), batch_size)):
    batch = test_df['text'][i:i+batch_size].tolist()
    results.extend(generate_results(batch, model, tokenizer))


test_df["target"] = results
test_df["target1"] = test_df["target"].apply(lambda x: x.upper())

test_df[["target1", "Index"]].to_csv("submission1.csv", index=False)