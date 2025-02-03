from tokenizers import Tokenizer
from datasets import load_from_disk

def verify_tokenization():
    tokenizer = Tokenizer.from_file("data/tokenizers/remma_unified_tokenizer.json")
    dataset = load_from_disk("data/processed/final_dataset")
    '''

    dataset = load_from_disk("data/processed/final_dataset")
    is loading a dataset that was previously saved using the save_to_disk() method from the Hugging Face Datasets library.

    Explanation:
    load_from_disk("data/processed/final_dataset") reads the dataset stored in the "data/processed/final_dataset" directory.
    This dataset is in Arrow format (a highly efficient columnar storage format) and was saved using dataset.save_to_disk(output_path).
    The dataset retains all features and structure, including splits like "train", "test", and "validation", if they were included during saving.

    When is this useful?
    When we have large datasets, and we don’t want to tokenize everything again.
    If the dataset is too big to fit into memory, we can process it in chunks, save it to disk, and reload only the required parts.
    
    '''
    
    # Check first 10 examples
    for i in range(10):
        example = dataset["train"][i]
        decoded = tokenizer.decode(example["input_ids"])
        print(f"Example {i+1}:")
        print("Original IDs:", example["input_ids"][:10], "...")
        print("Decoded Text:", decoded[:100] + "...\n")

if __name__ == "__main__":
    verify_tokenization()