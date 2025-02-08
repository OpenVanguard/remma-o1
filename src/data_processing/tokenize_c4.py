from tokenizers import Tokenizer
from datasets import Dataset, load_dataset
import os
from pathlib import Path
from tqdm import tqdm

def tokenize_c4():
    # Path configuration
    tokenizer_path = "data/tokenizers/remma_unified_tokenizer.json"
    input_path = "data/processed/c4_subset.txt"
    output_dir = Path("data/processed/tokenized_c4")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Create dataset
    dataset = load_dataset("text", data_files=input_path, split="train")
    
    # Tokenization function
    def tokenize_function(examples):
        tokenized_outputs = tokenizer.encode_batch(examples["text"])
        return {"input_ids": [output.ids for output in tokenized_outputs]}

    # Process dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
        num_proc=min(8, os.cpu_count() // 2)  # Ensure safe multiprocessing
    )
    
    # Add domain labels
    tokenized_dataset = tokenized_dataset.map(lambda x: {"domain": ["c4"] * len(x["input_ids"])})

    # Save in Arrow format
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenized_dataset.save_to_disk(output_dir)
    print(f"Tokenized C4 dataset saved to {output_dir}")

if __name__ == "__main__":
    tokenize_c4()
