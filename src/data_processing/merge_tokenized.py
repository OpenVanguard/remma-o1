# src/data_processing/merge_tokenized.py
import os
from datasets import Dataset, concatenate_datasets, load_from_disk
from pathlib import Path
import re

def merge_tokenized_data():
    # Input paths
    math_dir = Path("data/processed/tokenized_math")
    wiki_dir = Path("data/processed/tokenized_wikipedia")
    
    # Output path
    output_dir = Path("data/processed/final_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    def load_shards(folder: Path):
        """Load all checkpoint directories from a parent folder"""
        # Get sorted list of checkpoint directories
        dirs = sorted(
            [d for d in folder.iterdir() if d.is_dir() and "tokenized_checkpoint" in d.name],
            key=lambda x: int(re.search(r'\d+', x.name).group())
        )
        
        # Add final directory if exists
        final_dir = folder / "tokenized_final.arrow"
        if final_dir.exists():
            dirs.append(final_dir)

        datasets = []
        for d in dirs:
            try:
                # Each checkpoint directory contains sharded data
                ds = load_from_disk(d)
                datasets.append(ds)
                print(f"Loaded {len(ds)} examples from {d.name}")
            except Exception as e:
                print(f"Error loading {d}: {str(e)}")
                continue

        if not datasets:
            raise FileNotFoundError(f"No valid datasets found in {folder}")
            
        return concatenate_datasets(datasets)

    # Load and label datasets
    print("Loading math data...")
    math_ds = load_shards(math_dir).map(lambda x: {"domain": "math"})
    
    print("Loading wikipedia data...")
    wiki_ds = load_shards(wiki_dir).map(lambda x: {"domain": "wikipedia"})

    # Combine and shuffle
    print("Merging datasets...")
    combined = concatenate_datasets([math_ds, wiki_ds]).shuffle(seed=42)

    # Save merged dataset
    print("Saving combined dataset...")
    combined.save_to_disk(output_dir)
    print(f"✅ Saved merged dataset to {output_dir}")
    print(f"Total examples: {len(combined)}")

if __name__ == "__main__":
    merge_tokenized_data()