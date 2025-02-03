from tokenizers import Tokenizer
from datasets import load_dataset, Dataset
import os
from pathlib import Path
from tqdm import tqdm

def tokenize_and_save(tokenizer, dataset, output_dir, save_interval=5000, batch_size=1000):
    """Tokenizes dataset and saves every 'save_interval' examples to prevent memory overload."""
    tokenized_batches = []
    os.makedirs(output_dir, exist_ok=True)
    
    for i, example in enumerate(dataset):
        # Tokenize text and append to batch
        tokenized = tokenizer.encode(example["text"]).ids
        tokenized_batches.append({"input_ids": tokenized})

        # Save periodically to prevent memory overload
        if (i + 1) % save_interval == 0:
            checkpoint_path = os.path.join(output_dir, f"tokenized_checkpoint_{i+1}.arrow")
            Dataset.from_list(tokenized_batches).save_to_disk(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            tokenized_batches = []  # Clear memory to prevent overload

    # Save any remaining tokenized data
    if tokenized_batches:
        final_checkpoint = os.path.join(output_dir, "tokenized_final.arrow")
        Dataset.from_list(tokenized_batches).save_to_disk(final_checkpoint)
        print(f"Final checkpoint saved: {final_checkpoint}")

def tokenize_files(tokenizer_path: str, input_path: str, output_dir: str):
    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("text", data_files=input_path, split="train")
    
    # Process dataset in batches
    tokenize_and_save(tokenizer, dataset, output_dir)
    
if __name__ == "__main__":
    # Example usage for math data
    tokenize_files(
        tokenizer_path="data/tokenizers/remma_unified_tokenizer.json",
        input_path="data/raw/math/*.jsonl",
        output_dir="data/processed/tokenized_math"
    )
    
    # For Wikipedia data
    tokenize_files(
        tokenizer_path="data/tokenizers/remma_unified_tokenizer.json",
        input_path="data/raw/wikipedia/en_wikipedia_20220301.txt",
        output_dir="data/processed/tokenized_wikipedia"
    )
