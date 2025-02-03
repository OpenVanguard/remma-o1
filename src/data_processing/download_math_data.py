from datasets import load_dataset
import json
import os
from tqdm import tqdm
from huggingface_hub import login

def download_small_math_datasets():
    # Log in to Hugging Face to access datasets
    hf_token = ""  # Replace with your actual token
    login(token=hf_token)

    # Create output directory
    os.makedirs("data/raw/math", exist_ok=True)

    # ✅ 1. GSM8K (Subset of dataset for smaller size)
    try:
        print("Loading GSM8K dataset (subset)...")
        gsm8k = load_dataset("gsm8k", "main", split="train[:1000]", trust_remote_code=True)  # Limiting to 1000 samples
        with open("data/raw/math/gsm8k_subset.jsonl", "w") as f:
            for example in tqdm(gsm8k, desc="Saving GSM8K subset"):
                json.dump(example, f)
                f.write("\n")
        print("GSM8K subset saved successfully.")
    except Exception as e:
        print(f"Error loading GSM8K: {e}")

    # ✅ 2. MathQA (Smaller Math Question Answering Dataset)
    try:
        print("Loading MathQA dataset...")
        mathqa = load_dataset("math_qa")  # You may need to adjust this name based on the exact dataset
        with open("data/raw/math/mathqa.jsonl", "w") as f:
            for example in tqdm(mathqa["train"], desc="Saving MathQA"):
                json.dump(example, f)
                f.write("\n")
        print("MathQA dataset saved successfully.")
    except Exception as e:
        print(f"Error loading MathQA dataset: {e}")

    # ✅ 3. AQUA-RAT (Smaller QA Math Dataset)
    try:
        print("Loading AQUA-RAT dataset...")
        aqua_rat = load_dataset("aqua_rat")  # You may need to adjust this name based on the exact dataset
        with open("data/raw/math/aqua_rat.jsonl", "w") as f:
            for example in tqdm(aqua_rat["train"], desc="Saving AQUA-RAT"):
                json.dump(example, f)
                f.write("\n")
        print("AQUA-RAT dataset saved successfully.")
    except Exception as e:
        print(f"Error loading AQUA-RAT dataset: {e}")

if __name__ == "__main__":
    download_small_math_datasets()
