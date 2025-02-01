# Quick check script (save as src/data_processing/verify_data.py)
import random

def verify_sample(file_path, num_samples=5):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in random.sample(range(len(lines)), num_samples):
            print(f"Sample {i+1}:\n{lines[i][:500]}...\n{'-'*50}")

if __name__ == "__main__":
    verify_sample("data/processed/c4_subset.txt")