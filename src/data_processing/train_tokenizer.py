from tokenizers import Tokenizer, models, trainers
import os

def train_tokenizer():
    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE())
    trainer = trainers.BpeTrainer(
        vocab_size=50000,
        special_tokens=["<|endoftext|>"]
    )
    
    # Train on your dataset
    tokenizer.train(
        files=["data/processed/c4_subset.txt"],
        trainer=trainer
    )
    
    # Save to tokenizers directory
    os.makedirs("data/tokenizers", exist_ok=True)
    tokenizer.save("data/tokenizers/remma_tokenizer_v1.json")  # Versioned name

if __name__ == "__main__":
    train_tokenizer()