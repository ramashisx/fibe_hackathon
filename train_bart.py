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
train_df = pd.read_csv('data/train.csv').sample(frac=1)
val_df = pd.read_csv('data/val.csv').sample(frac=1)
train_df["target"] = train_df["target"].str.upper()
val_df["target"] = val_df["target"].str.upper()

# 2. Convert to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(val_df)

# 3. Retrieve unique categories
categories = train_df['target'].unique().tolist()
# join categories by | and making everything inside capital letters
categories_str = "|".join(categories).upper()
print(categories_str)

# save that string in a file
with open("categories.txt", "w") as f:
    f.write(categories_str)
    f.close()


# 4. Define the prompt
def add_prompt(example):
    prompt = (
        f"Text: {example['text']}"
        f"Classify the following text into one of the following categories: {categories_str}.\n"
        f"Category:"
    )
    example['input'] = prompt
    example['output'] = example['target']
    return example

# Apply prompt
train_dataset = train_dataset.map(add_prompt)
test_dataset = test_dataset.map(add_prompt)

# Rename columns
train_dataset = train_dataset.rename_column("input", "input_text")
train_dataset = train_dataset.rename_column("output", "target_text")
test_dataset = test_dataset.rename_column("input", "input_text")
test_dataset = test_dataset.rename_column("output", "target_text")

# 5. Initialize tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 6. Tokenization function
def tokenize_function(example):
    # Tokenize the input text
    input_encodings = tokenizer(
        example['input_text'],
        max_length=1024,
        truncation=True
    )
    
    # Tokenize the target text
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            example['target_text'],
            max_length=64,
            truncation=True
        )
    
    # Add labels
    input_encodings["labels"] = target_encodings["input_ids"]
    
    return input_encodings

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# 7. Initialize model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.generation_config.max_length = 64

# 8. Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./bart",
    evaluation_strategy="steps",
    eval_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="steps",
    report_to="wandb",
)

# 9. Define compute_metrics
def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.

    Args:
        eval_pred (EvalPrediction): Object containing predictions and label_ids.

    Returns:
        Dict[str, float]: Dictionary containing the computed metrics.
    """
        
    predictions, labels = eval_pred

    pad_token_id = tokenizer.pad_token_id  # Replace this with your actual pad token ID if different

    labels = [[pad_token_id if token == -100 else token for token in label] for label in labels]
    prediction = [[pad_token_id if token == -100 else token for token in prediction] for prediction in predictions]

    # Decode the generated predictions and the labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean up the predictions and labels
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    for decoded_label, decoded_pred in zip(decoded_labels, decoded_preds):
        print(f"Label: {decoded_label} | Prediction: {decoded_pred}")

    # Calculate accuracy
    accuracy = accuracy_score(decoded_labels, decoded_preds)

    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        decoded_labels, decoded_preds, average='weighted', zero_division=0
    )

    eval_result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    print("\nEvaluation Results: \n", eval_result)

    return eval_result

# 10. Initialize Trainer with compute_metrics

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 11. Train the model
trainer.train()
