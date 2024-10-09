"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from minbpe import BasicTokenizer, RegexTokenizer, MYTETokenizer
import datasets

# open some text and train a vocab of 512 tokens
# text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()

# Load the dataset
d_ar = datasets.load_dataset("wikimedia/wikipedia", "20231101.ar", split='train[:500]')
d_en = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split='train[:500]')
d_es = datasets.load_dataset("wikimedia/wikipedia", "20231101.es", split='train[:500]')
d_ja = datasets.load_dataset("wikimedia/wikipedia", "20231101.ja", split='train[:500]')
d_zh = datasets.load_dataset("wikimedia/wikipedia", "20231101.zh", split='train[:500]')

# Concatenate the datasets
d = datasets.concatenate_datasets([d_ar, d_en, d_es, d_ja, d_zh])

# Convert dataset to string
text = " ".join(d["text"])

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()

import concurrent.futures

tokenizers = {
    BasicTokenizer: "basic",
    RegexTokenizer: "regex",
    MYTETokenizer: "myte",
}

vocab_size = 10_000
test_text = "Hello my name is Taylor Swift"

def train_and_save_tokenizer(TokenizerClass, name):
    print(f"Training {name} tokenizer...")
    tokenizer = TokenizerClass()
    tokenizer.train(text, vocab_size, verbose=True)
    print(f"    Encoding '{test_text}'...")
    enc = tokenizer.encode(test_text)
    print(f"    Encoded: {enc}")
    dec = tokenizer.decode(enc)
    print(f"    Decoded: {dec}")
    print()
    prefix = os.path.join("models", name)
    print(f"Saving {name} tokenizer to {prefix}...")
    tokenizer.save(prefix)

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(train_and_save_tokenizer, TokenizerClass, name): name for TokenizerClass, name in tokenizers.items()}
    for future in concurrent.futures.as_completed(futures):
        future.result()  # Wait for the future to complete

t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")