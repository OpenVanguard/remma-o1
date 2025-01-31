from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset

# Train tokenizer on your subset
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=["data/processed/c4_subset.txt"], 
               vocab_size=50_000, 
               min_frequency=2,
               special_tokens=["<|endoftext|>"])

# Save tokenizer
tokenizer.save_model("data/tokenizers/c4_tokenizer")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer.encode_batch(examples["text"])

dataset = load_dataset("text", data_files={"train": "data/processed/c4_subset.txt"})
tokenized_dataset = dataset.map(tokenize_function, batched=True)