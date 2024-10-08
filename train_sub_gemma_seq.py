import torch
import re
import pandas as pd
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)

import bitsandbytes as bnb

import evaluate
import numpy as np

import random


# 1. Load the data
train_df = pd.read_csv('data/train.csv').sample(frac=0.33)
val_df = pd.read_csv('data/val.csv')
train_df["target"] = train_df["target"].str.upper()
val_df["target"] = val_df["target"].str.upper()

label2idx = {label: idx for idx, label in enumerate(train_df["target"].unique())}
idx2label = {idx: label for label, idx in label2idx.items()}

from huggingface_hub import login
login('hf_zdlWVWyXnnNMbaWxxxxxxxxxtOoyUossuuhf')

model_id  = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f' Vocab size of the model {model_id}: {len(tokenizer.get_vocab())}')


# 3. Retrieve unique categories
categories = train_df['target'].unique().tolist()
# join categories by | and making everything inside capital letters
categories_str = "|".join(categories).upper()
print(categories_str)

# 4. Define the prompt
def add_prompt(text):
    prompt = (
        f"Text: {text}"
        f"Classify the following text into one of the following categories: {categories_str}.\n\n"
        f"Category:"
    )
    return prompt


train_df['text'] = train_df['text'].apply(add_prompt)
val_df['text'] = val_df['text'].apply(add_prompt)

train_df["label"] = train_df["target"].apply(lambda x: label2idx[x])
val_df["label"] = val_df["target"].apply(lambda x: label2idx[x])

# 2. Convert to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(val_df)

def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=1024, truncation=True)


# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# First define the metric without accuracy
metric = evaluate.combine(["f1", "precision", "recall"])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Convert probabilities to predicted labels
    predictions = np.argmax(predictions, axis=1)
    
    for pred, label in zip(predictions, labels):
        print(f"Predicted: {idx2label[pred]}, True: {idx2label[label]}")

    # Calculate accuracy separately without the 'average' parameter
    accuracy = (predictions == labels).mean()
    
    # Compute other metrics that need 'average' parameter
    other_metrics = metric.compute(predictions=predictions, references=labels, average="weighted")
    
    # Add accuracy to the final metrics result
    other_metrics["accuracy"] = accuracy
    
    return other_metrics


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enables 4-bit quantization
    bnb_4bit_use_double_quant=True,  # Use double quantization for potentially higher accuracy (optional)
    bnb_4bit_quant_type="nf4",  # Quantization type (specifics depend on hardware and library)
    bnb_4bit_compute_dtype=torch.bfloat16  # Compute dtype for improved efficiency (optional)
)



model = AutoModelForSequenceClassification.from_pretrained(
    model_id,  # "google/gemma-2b-it"
    num_labels=26,  # Number of output labels (2 for binary sentiment classification)
    id2label=idx2label,
    label2id=label2idx,
    trust_remote_code=True,
    quantization_config=bnb_config,  # configuration for quantization 
    device_map={"": 0}  # Optional dictionary specifying device mapping (single GPU with index 0 here)
)


model.gradient_checkpointing_enable()

model = prepare_model_for_kbit_training(model)


def find_linear_names(model):
    """
    This function identifies all linear layer names within a model that use 4-bit quantization.
    Args:
        model (torch.nn.Module): The PyTorch model to inspect.
    Returns:
        list: A list containing the names of all identified linear layers with 4-bit quantization.
    """
    cls = bnb.nn.Linear4bit  

    # Set to store identified layer names
    lora_module_names = set()

    # Iterate through named modules in the model
    for name, module in model.named_modules():
        # Check if the current module is an instance of the 4-bit linear layer class
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        # Special case: remove 'lm_head' if present
        if 'lm_head' in lora_module_names: 
            lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_linear_names(model)
print(modules)
    
    
lora_config = LoraConfig(
    r=512,  # Reduction factor (lower r means more parameters in the adapter)
    lora_alpha=384,  # Dimensionality of the adapter projection
    target_modules=modules,  # List of modules to apply the LoRA adapter
    lora_dropout=0.05,  # Dropout rate for the adapter
    bias="none",  # Bias configuration for the adapter
    task_type="SEQ_CLS"  # Task type (sequence classification in this case)
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


training_args = TrainingArguments(
    output_dir="epoch_weights",  # Output directory for checkpoints
    learning_rate=2e-5,  # Learning rate for the optimizer
    per_device_train_batch_size=6,  # Batch size per device
    per_device_eval_batch_size=6,  # Batch size per device for evaluation 
    num_train_epochs=2,  # Number of training epochs
    weight_decay=0.01,  # Weight decay for regularization
    eval_strategy='steps',  # Evaluate after each epoch
    eval_steps=200,
    save_strategy="steps",  # Save model checkpoints after each epoch
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,  # Load the best model based on the chosen metric
    push_to_hub=False,  # Disable pushing the model to the Hugging Face Hub 
    report_to="wandb",  # Disable logging to Weight&Bias
    metric_for_best_model='eval_loss' # Metric for selecting the best model 
)

trainer = Trainer(
    model=model,  # The LoRA-adapted model
    args=training_args,  # Training arguments
    train_dataset=tokenized_train,  # Training dataset
    eval_dataset=tokenized_test,  # Evaluation dataset
    tokenizer=tokenizer,  # Tokenizer for processing text
    data_collator=data_collator,  # Data collator for preparing batches
    compute_metrics=compute_metrics,  # Function to calculate evaluation metrics
)


trainer.train()

def predict(input_text):
    """
    Predicts the sentiment label for a given text input.

    Args:
        input_text (str): The text to predict the sentiment for.

    Returns:
        float: The predicted probability of the text being positive sentiment.
    """
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # Convert to PyTorch tensors and move to GPU (if available)
    with torch.no_grad():
        outputs = model(**inputs).logits  # Get the model's output logits
    y_prob = torch.sigmoid(outputs).tolist()[0]  # Apply sigmoid activation and convert to list
    return np.round(y_prob, 5)  # Round the predicted probability to 5 decimal places


test_df = pd.read_csv('data/test_clean.csv', engine='python', encoding='latin-1')

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

test_df['prediction'] = test_df['text'].map(predict)
test_df['y_pred'] = test_df['prediction'].apply(lambda x: np.argmax(x, axis=0)) 

test_df["target3"] = test_df["y_pred"].apply(lambda x: idx2label[x])

test_df[["target3", "Index"]].to_csv("submission3.csv", index=False)
