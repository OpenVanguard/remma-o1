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


# src/training/train_custom.py
import os
import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from src.modeling import build_model
from src.training.utils.checkpoint import save_checkpoint
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

class WikipediaDataset(Dataset):
    def __init__(self, tokenized_data, block_size=2048):
        self.data = tokenized_data
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        chunk = example["input_ids"][:self.block_size]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def main():
    # Load configuration
    with open("configs/model_remma.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"]).to(device)
    
    # Load Wikipedia dataset
    try:
        dataset = load_from_disk("data/processed/wiki_only")
    except FileNotFoundError:
        raise RuntimeError("Wikipedia dataset not found. Run data processing first.")
    
    # Create dataloader
    train_dataset = WikipediaDataset(dataset["train"], config["model"]["block_size"])
    dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        pin_memory=True
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["total_steps"],
        eta_min=config["training"]["min_lr"]
    )
    scaler = GradScaler()
    
    # Training setup
    global_step = 0
    model.train()
    
    # Initialize WandB
    wandb_available = False
    try:
        import wandb
        wandb.init(project="remma-wiki-training", config=config)
        wandb_available = True
        wandb.watch(model)
    except ImportError:
        print("Wandb not installed, skipping logging")
    
    # Progress tracking
    progress_bar = tqdm(total=config["training"]["total_steps"], desc="Training")
    
    try:
        while global_step < config["training"]["total_steps"]:
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                
                with autocast(dtype=torch.bfloat16):
                    _, loss = model(x, targets=y)
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Update tracking
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]['lr']
                })
                
                # Log metrics
                if wandb_available:
                    wandb.log({
                        "loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "step": global_step
                    })
                
                # Checkpointing
                if global_step % config["training"]["checkpoint_steps"] == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=global_step,
                        config=config
                    )
                
                if global_step >= config["training"]["total_steps"]:
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final checkpoint...")
        save_checkpoint(model, optimizer, scheduler, global_step, config)
    
    finally:
        progress_bar.close()
        print("Training completed!")

if __name__ == "__main__":
    main()