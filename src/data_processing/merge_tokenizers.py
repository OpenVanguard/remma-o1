from tokenizers import Tokenizer, models, trainers
from pathlib import Path
import json
from collections import OrderedDict

def merge_tokenizers():
    # Load tokenizers
    math_tokenizer = Tokenizer.from_file("data/tokenizers/remma_math_tokenizer.json")
    wiki_tokenizer = Tokenizer.from_file("data/tokenizers/remma_tokenizer_wikipedia.json")
    
    # Get vocabularies
    math_vocab = math_tokenizer.get_vocab()
    wiki_vocab = wiki_tokenizer.get_vocab()
    
    # Get and process merges
    try:
        with open("data/tokenizers/remma_math_tokenizer.json", encoding='utf-8') as f:
            math_data = json.load(f)
            math_merges = [(pair.split()[0], pair.split()[1]) 
                          for pair in math_data.get("model", {}).get("merges", [])]
        
        with open("data/tokenizers/remma_tokenizer_wikipedia.json", encoding='utf-8') as f:
            wiki_data = json.load(f)
            wiki_merges = [(pair.split()[0], pair.split()[1]) 
                          for pair in wiki_data.get("model", {}).get("merges", [])]
    except Exception as e:
        print(f"Error loading merges: {e}")
        math_merges = []
        wiki_merges = []
    
    # Initialize merged vocabulary
    merged_vocab = {}
    current_idx = 0
    
    # Add special tokens
    special_tokens = ["<|endoftext|>", "<|equation|>", "<|mathsep|>"]
    for token in special_tokens:
        merged_vocab[token] = current_idx
        current_idx += 1

    # Merge vocabularies
    for token in {**math_vocab, **wiki_vocab}:
        if token not in merged_vocab:
            merged_vocab[token] = current_idx
            current_idx += 1

    # Convert merges to correct format
    merged_merges = []
    seen = set()
    for merge in math_merges + wiki_merges:
        merge_str = f"{merge[0]} {merge[1]}"
        if merge_str not in seen:
            merged_merges.append(merge)
            seen.add(merge_str)

    # Create BPE model with converted merges
    merged_model = models.BPE(
        vocab=merged_vocab,
        merges=merged_merges,
        dropout=None,
        unk_token="<|endoftext|>"
    )

    # Build new tokenizer
    merged_tokenizer = Tokenizer(merged_model)
    
    # Configure normalizers and pre-tokenizers
    merged_tokenizer.normalizer = math_tokenizer.normalizer
    merged_tokenizer.pre_tokenizer = math_tokenizer.pre_tokenizer
    merged_tokenizer.post_processor = math_tokenizer.post_processor

    # Save merged tokenizer
    output_path = Path("data/tokenizers/remma_unified_tokenizer.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_tokenizer.save(str(output_path))
    
    print(f"Merged tokenizer saved to {output_path}")
    print(f"Final vocabulary size: {len(merged_vocab)}")
    print(f"Total merges: {len(merged_merges)}")

if __name__ == "__main__":
    merge_tokenizers()
    