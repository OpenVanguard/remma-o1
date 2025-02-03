# src/data_processing/combine_datasets.py
from datasets import load_from_disk, concatenate_datasets

def create_final_dataset():
    math_data = load_from_disk("data/processed/retokenized_math")
    wiki_data = load_from_disk("data/processed/retokenized_wikipedia")
    
    # Create balanced mix (adjust ratios as needed)
    final_dataset = concatenate_datasets([
        wiki_data.shuffle(seed=42).select(range(1_000_000)),  # 1M general text examples
        math_data.shuffle(seed=42).select(range(200_000))     # 200K math examples
    ]).shuffle(seed=42)
    
    final_dataset.save_to_disk("data/processed/final_dataset")

if __name__ == "__main__":
    create_final_dataset()