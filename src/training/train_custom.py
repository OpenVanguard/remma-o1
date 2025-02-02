# src/training/train_custom.py

'''
Key improvements made:
    Added proper dataset class to handle tokenized data
    Device management for GPU/CPU compatibility
    Progress tracking with tqdm
    Wandb integration for experiment tracking (optional)
    Configurable training duration through total_steps
    Full training loop structure with proper batching
    Error handling for missing dependencies
    Batch size and workers configuration
'''

import os
import yaml
import torch
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from src.modeling import build_model
from src.training.utils.checkpoint import save_checkpoint
from torch.cuda.amp import GradScaler, autocast

class TextDataset(Dataset):
    def __init__(self, tokenized_data, block_size=1024):
        self.data = tokenized_data
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]["input_ids"][:self.block_size]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def main():
    # Load configuration
    with open("configs/model_remma.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize model and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    
    # Load tokenized dataset
    tokenized_data = load_from_disk("data/processed/tokenized_c4")
    dataset = TextDataset(tokenized_data["train"], config["block_size"])
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=True,
        num_workers=os.cpu_count() // 2
    )
    
    # Initialize optimizer and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 3e-4))
    scaler = GradScaler()
    
    # Training loop
    global_step = 0
    model.train()
    
    try:
        import wandb
        wandb.init(project="remma-training", config=config)
    except ImportError:
        print("Wandb not installed, skipping logging")
    
    progress_bar = tqdm(total=config.get("total_steps", 100_000), desc="Training")
    
    while global_step < config.get("total_steps", 100_000):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            with autocast(dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update progress
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Save checkpoint
            if global_step % config.get("checkpoint_steps", 10_000) == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    config=config
                )
            
            # Log to wandb
            if "wandb" in locals():
                wandb.log({"loss": loss.item()}, step=global_step)
                
            if global_step >= config.get("total_steps", 100_000):
                break

    progress_bar.close()
    print("Training completed!")

if __name__ == "__main__":
    main()