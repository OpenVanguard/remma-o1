from tokenizers import Tokenizer, models, trainers, normalizers, pre_tokenizers
import os
from pathlib import Path

def train_tokenizer_with_checkpoints():
    # Configuration
    input_file = "data/raw/wikipedia/en_wikipedia_20220301.txt"
    checkpoint_dir = "data/tokenizers/checkpoints-wikipedia"
    final_tokenizer_path = "data/tokenizers/remma_tokenizer_wikipedia.json"
    batch_size = 50000000  # Lines per batch
    vocab_size = 50000000
    special_tokens = ["<|endoftext|>"]
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize tokenizer and trainer
    tokenizer = Tokenizer(models.BPE())
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    # Add normalization and pre-tokenization
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    
    # Check for existing checkpoints
    last_checkpoint = max(
        Path(checkpoint_dir).glob("checkpoint_*.json"),
        key=os.path.getctime,
        default=None
    )
    
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        tokenizer = Tokenizer.from_file(str(last_checkpoint))
        processed_lines = int(last_checkpoint.stem.split("_")[1])
    else:
        processed_lines = 0

    # Process file in batches
    with open(input_file, "r", encoding="utf-8") as f:
        # Skip already processed lines
        for _ in range(processed_lines):
            next(f)
        
        batch = []
        for line_number, line in enumerate(f, start=1):
            batch.append(line.strip())
            
            if len(batch) >= batch_size:
                print(f"Processing lines {processed_lines+1}-{processed_lines+len(batch)}")
                
                # Train on current batch
                tokenizer.train_from_iterator(
                    batch,
                    trainer=trainer,
                    length=len(batch)
                )
                
                # Save checkpoint
                checkpoint_path = f"{checkpoint_dir}/checkpoint_{processed_lines+len(batch)}.json"
                tokenizer.save(checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
                
                # Update counters and clear batch
                processed_lines += len(batch)
                batch = []
                
                # Memory cleanup
                del batch
                batch = []

        # Train on remaining lines
        if batch:
            print(f"Processing final batch of {len(batch)} lines")
            tokenizer.train_from_iterator(
                batch,
                trainer=trainer,
                length=len(batch)
            )
            processed_lines += len(batch)

    # Save final tokenizer
    tokenizer.save(final_tokenizer_path)
    print(f"Final tokenizer saved to: {final_tokenizer_path}")

if __name__ == "__main__":
    train_tokenizer_with_checkpoints()