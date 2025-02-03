import os
from datasets import load_dataset
from tqdm import tqdm

def download_wikipedia(output_dir="data/raw/wikipedia"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "en_wikipedia_20220301.txt")
    
    # Skip if already downloaded
    if os.path.exists(output_path):
        print(f"Wikipedia dataset already exists at {output_path}")
        return

    # Load dataset (March 2022 English snapshot)
    print("Downloading Wikipedia dataset...")
    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        trust_remote_code=True,
        streaming=False  # Set to True for memory-constrained systems
    )

    # Save to text file
    with open(output_path, "w", encoding="utf-8") as f:
        for article in tqdm(dataset, desc="Processing articles"):
            f.write(f"{article['title']}\n{article['text']}\n\n")

    print(f"Saved Wikipedia dataset to {output_path}")

if __name__ == "__main__":
    download_wikipedia()