'''
Dont use now for future use
remma-o1-v1.0
first train 
c4, maths, wikipedia
'''
from tokenizers import Tokenizer
from datasets import load_from_disk

def retokenize_dataset(dataset_path, tokenizer_path, output_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    dataset = load_from_disk(dataset_path)
    
    def batch_tokenize(examples):
        return tokenizer.encode_batch(examples["text"])
    
    return dataset.map(
        batch_tokenize,
        batched=True,
        batch_size=1000,
        num_proc=4,
        cache_file_name=output_path
    )

if __name__ == "__main__":
    # Retokenize math data
    retokenize_dataset(
        "data/processed/tokenized_math",
        "data/tokenizers/remma_unified_tokenizer.json",
        "data/processed/retokenized_math"
    )
    
    # Retokenize Wikipedia
    retokenize_dataset(
        "data/processed/tokenized_wikipedia",
        "data/tokenizers/remma_unified_tokenizer.json",
        "data/processed/retokenized_wikipedia"
    )