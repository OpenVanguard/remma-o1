from tokenizers import Tokenizer

def print_vocab_size():
    tokenizer = Tokenizer.from_file("data/tokenizers/remma_tokenizer_wikipedia.json")
    vocab_size = len(tokenizer.get_vocab())
    print(f"Tokenizer vocabulary size: {vocab_size}")
    print("Update configs/model_remma.yaml with:")
    print(f"vocab_size: {vocab_size}")

if __name__ == "__main__":
    print_vocab_size()