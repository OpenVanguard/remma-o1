# src/data_processing/merge_tokenized.py
import os
from datasets import Dataset, concatenate_datasets, load_from_disk
from pathlib import Path
import pyarrow as pa

def merge_tokenized_data():
    # Input paths
    math_dir = Path("data/processed/tokenized_math")
    wiki_dir = Path("data/processed/tokenized_wikipedia")
    
    # Output path
    output_dir = Path("data/processed/final_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    def load_shards(folder: Path):
        """Load all .arrow shards from a directory"""
        tables = []
        # Get sorted list of checkpoint files (excluding final)
        files = sorted(
            [f for f in folder.glob("*.arrow") if "final" not in f.name],
            key=lambda x: int(x.stem.split("_")[-1])
        )
        
        # Add final file last if exists
        final_file = folder / "tokenized_final.arrow"
        if final_file.exists():
            files.append(final_file)

        for f in files:
            if f.exists():
                tables.append(pa.ipc.RecordBatchFileReader(f).read_all())
            else:
                print(f"Warning: Missing file {f}, skipping")

        if not tables:
            raise FileNotFoundError(f"No valid .arrow files found in {folder}")
            
        return Dataset(pa.concat_tables(tables))

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