import os
from datasets import load_dataset

def download_c4_subset(output_path, num_examples=10_000_000):
    """
    Download a subset of C4 dataset and save as text file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add trust_remote_code=True here
    dataset = load_dataset(
        "c4", 
        "en", 
        split="train", 
        streaming=True,
        trust_remote_code=True  # <-- CRITICAL FIX
    )
    
    subset = dataset.take(num_examples)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(subset):
            f.write(example["text"] + "\n")
            if (i + 1) % 10_000 == 0:
                print(f"Processed {i+1}/{num_examples} examples...")

if __name__ == "__main__":
    output_path = "data/processed/c4_subset.txt"
    download_c4_subset(output_path)