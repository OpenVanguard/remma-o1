from tokenizers import Tokenizer, models, trainers, normalizers, pre_tokenizers
import os
import json
from pathlib import Path
from tqdm import tqdm

def train_math_tokenizer():
    # Configuration
    input_dir = Path("data/raw/math")
    checkpoint_dir = Path("data/tokenizers/checkpoints-math")
    final_tokenizer_path = "data/tokenizers/remma_math_tokenizer.json"
    
    # Parameters (adjusted for math datasets)
    batch_size = 5000000  # Reduced for better memory management
    vocab_size = 5000000  # More reasonable vocabulary size
    special_tokens = [
        "<|frac|>", "<|sqrt|>", "<|sum|>", "<|int|>",
        "<|equation|>", "<|proof|>", "<|matrix|>", 
        "<|theorem|>", "<|definition|>", "<|lemma|>",
        "<|endoftext|>"
    ]
    
    # Create directories
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Configure tokenizer components
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()
    ])
    
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Digits(individual_digits=True)
    ])
    
    # Initialize trainer with math-specific tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2
    )
    
    # Get list of math dataset files
    math_files = list(input_dir.glob("*.jsonl"))
    if not math_files:
        raise ValueError(f"No JSONL files found in {input_dir}")
    
    # Dataset processing generator
    def text_generator():
        for file_path in math_files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"Processing {file_path.name}"):
                    try:
                        data = json.loads(line)
                        # Handle different dataset formats
                        if "question" in data and "answer" in data:  # GSM8K format
                            yield f"{data['question']} {data['answer']}"
                        elif "text" in data:  # OpenWebMath format
                            yield data["text"]
                        elif "problem" in data:  # AQUA-RAT format
                            yield data["problem"]
                    except json.JSONDecodeError:
                        continue
    
    # Check for existing checkpoints
    last_checkpoint = max(
        checkpoint_dir.glob("checkpoint_*.json"),
        key=os.path.getctime,
        default=None
    )
    
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        tokenizer = Tokenizer.from_file(str(last_checkpoint))
        processed_items = int(last_checkpoint.stem.split("_")[1])
    else:
        processed_items = 0
    
    # Train tokenizer with checkpointing
    try:
        total_items = sum(1 for _ in text_generator())  # Calculate the total number of items
        for i, batch in enumerate(text_generator()):
            tokenizer.train_from_iterator([batch], trainer=trainer)
            
            # Save checkpoint after processing a batch
            if (i + 1) % batch_size == 0:
                checkpoint_file = checkpoint_dir / f"checkpoint_{i + 1}.json"
                tokenizer.save(str(checkpoint_file))
                print(f"Saved checkpoint at {checkpoint_file}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final checkpoint...")
    
    # Save final tokenizer
    tokenizer.save(final_tokenizer_path)
    print(f"Math tokenizer saved to: {final_tokenizer_path}")
    
    # Test tokenization
    test_text = "Solve ∫ x² dx from 0 to 1. Solution: <|equation|>∫₀¹x²dx = [x³/3]₀¹ = 1/3<|endoftext|>"
    encoded = tokenizer.encode(test_text)
    print("\nTest tokenization:")
    print(f"Text: {test_text}")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")

if __name__ == "__main__":
    train_math_tokenizer()
