import re
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split


# Parallel apply function to a single column with tqdm (manually)
def parallel_apply_column(series, func, n_jobs=-1):
    # Use tqdm to wrap the series for progress tracking
    results = Parallel(n_jobs=n_jobs)(
        delayed(func)(value) for value in tqdm(series, desc="Processing column")
    )
    return results


# DataFrame
df = pd.read_csv('dataset/train.csv', engine='python', encoding='latin-1')
test = pd.read_csv('dataset/test.csv', engine='python', encoding='latin-1')

# clean the text
def clean_text(text):
    # Remove unwanted HTML/CSS-like code (anything in angle brackets or complex formatting)
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    
    # Remove all numbers
    text = re.sub(r'\d+', '', text)  # Remove all digits
    
    # Remove CSS-like properties or attributes with numbers/units, or other irrelevant code
    text = re.sub(r'[\w-]*\s*:\s*[\w\s\.\(\)\-]*[;]*', '', text)  # Remove any word and its property definition
    text = re.sub(r'[a-zA-Z]+\d+[a-zA-Z]*', '', text)  # Remove any alphanumeric word
    
    # Remove words with two or more special characters
    text = ' '.join([word for word in text.split() if len(re.findall(r'[^a-zA-Z0-9\s]', word)) < 2])
    
    # Remove any remaining non-word characters except common punctuation
    text = re.sub(r'[^a-zA-Z\s\.,?!]', '', text)  # Keep only letters, punctuation, and spaces
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Replace 3 or more newlines with exactly 2 newlines
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)
    
    # Remove words with more than 10 characters
    text = ' '.join([word for word in text.split() if len(word) <= 10])
    
    # Convert to lowercase
    text = text.lower()
    
    return text


# Define features and target
X = parallel_apply_column(df['text'], clean_text)
test["text"] = parallel_apply_column(test["text"], clean_text)
y = df['target']

# Split the data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.002, stratify=y, random_state=42
)

# Combine the split data back into DataFrames (optional)
train_df = pd.DataFrame({'text': X_train, 'target': y_train}).reset_index(drop=True)
val_df = pd.DataFrame({'text': X_val, 'target': y_val}).reset_index(drop=True)

print("Training set size:", train_df.shape)
print("Validation set size:", val_df.shape)


# Calculate the number of samples per category
category_counts = train_df['target'].value_counts()
print("Original category counts in training set:")
print(category_counts)

# Calculate the mean number of samples per category
mean_count = int(category_counts.mean())
mean_count = 10000
print("\nMean number of samples per category:", mean_count)

# Function to sample up to mean_count for each category
def sample_category(group, max_samples):
    return group.sample(n=min(len(group), max_samples), random_state=42)

# Apply the function to each category
balanced_train_df = train_df.groupby('target').apply(
    lambda x: sample_category(x, mean_count)
).reset_index(drop=True)

print("\nBalanced training set category counts:")
print(balanced_train_df['target'].value_counts())

balanced_train_df = balanced_train_df.reset_index(drop=True)

print("\nMissing values in balanced training set:")
print(balanced_train_df.isna().sum())

print("\nMissing values in validation set:")
print(val_df.isna().sum())

print("\nMissing values in test set:")
print(test.isna().sum())

balanced_train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)
