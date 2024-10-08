from tqdm import tqdm
import pandas as pd
import ctranslate2
import transformers


# 1. Load the data
test_df = pd.read_csv('data/test.csv')
test_df["text"] = test_df["text"].fillna(" ")

translator = ctranslate2.Translator("bart-large-cnn", device="cuda")
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
pad_token_id = tokenizer.pad_token_id


def generate_results(texts):
    # print([text for text in texts])
    sources = [tokenizer.convert_ids_to_tokens(tokenizer.encode(text)) for text in texts]
    # print([len(source) for source in sources])
    results = translator.translate_batch(sources, max_batch_size=64, max_input_length=1024, max_decoding_length=1024, min_decoding_length=64)
    results = [result.hypotheses[0] for result in results]
    results = [tokenizer.convert_tokens_to_ids(result) for result in results]
    # print([len(result) for result in results])
    results = tokenizer.batch_decode(results, skip_special_tokens=True)
    # print([result for result in results])
    return results

# 3. Generate predictions on a batch of batch size 256
batch_size = 256//2
results = []
for i in tqdm(range(0, len(test_df), batch_size)):
    batch = test_df['text'][i:i+batch_size].tolist()
    results.extend(generate_results(batch))

test_df["clean_text"] = results
test_df.to_csv('data/test_clean.csv', index=False)
