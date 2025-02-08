from datasets import load_from_disk
from pathlib import Path

def split_dataset():
    # Load merged dataset
    dataset = load_from_disk("data/processed/final_dataset")

    # Split into domains
    wiki_ds = dataset.filter(lambda x: x["domain"] == "wikipedia")

    # Save separated datasets
    wiki_ds.save_to_disk("data/processed/wiki_only")

if __name__ == "__main__":
    split_dataset()