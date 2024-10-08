"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from minbpe import BasicTokenizer, RegexTokenizer, MYTETokenizer

# open some text and train a vocab of 512 tokens
text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()

tokenizers = {
    BasicTokenizer: "basic",
    RegexTokenizer: "regex",
    MYTETokenizer: "myte",
}

test_text = "Hello my name is Taylor Swift"

for TokenizerClass, name in tokenizers.items():
    print(f"Training {name} tokenizer...")
    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)
    # writes two files in the models directory: name.model, and name.vocab
    print(f"    Encoding '{test_text}'...")
    enc = tokenizer.encode(test_text)
    print(f"    Encoded: {enc}")
    dec = tokenizer.decode(enc)
    print(f"    Decoded: {dec}")
    print()
    prefix = os.path.join("models", name)
    print(f"Saving {name} tokenizer to {prefix}...")
    tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")