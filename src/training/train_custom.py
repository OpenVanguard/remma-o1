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
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from src.modeling import build_model
from src.training.utils.checkpoint import save_checkpoint
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

class MultiDomainDataset(Dataset):
    def __init__(self, tokenized_data, block_size=2048):
        self.data = tokenized_data
        self.block_size = block_size
        self.domain_weights = {
            'c4': 0.7,
            'wikipedia': 0.2,
            'math': 0.1
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        # Ensure proper sequence length handling
        chunk = example["input_ids"][:self.block_size].tolist()
        # Add padding if needed
        if len(chunk) < self.block_size:
            chunk += [0] * (self.block_size - len(chunk))
        
        # Domain-aware processing
        domain = example.get("domain", "c4")
        loss_weight = self.domain_weights.get(domain, 1.0)
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y, torch.tensor(loss_weight)

def main():
    # Load configuration
    with open("configs/model_remma.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"]).to(device)  # Pass model config
    
    # Load combined dataset
    try:
        dataset = load_from_disk("data/processed/final_dataset")
    except FileNotFoundError:
        raise RuntimeError("Dataset not found. Run data processing pipeline first.")
    
    # Create dataloader with domain-aware sampling
    train_dataset = MultiDomainDataset(dataset["train"], config["model"]["block_size"])
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        pin_memory=True
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 3e-4),
        weight_decay=config.get("weight_decay", 0.1)
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.get("total_steps", 100_000),
        eta_min=config.get("min_lr", 1e-5)
    )
    scaler = GradScaler()
    
    # Training setup
    global_step = 0
    model.train()
    
    # Initialize WandB
    wandb_available = False
    try:
        import wandb
        wandb.init(project="remma-training", config=config)
        wandb_available = True
        wandb.watch(model, log="all")
    except ImportError:
        print("Wandb not installed, skipping logging")
    
    # Progress tracking
    progress_bar = tqdm(total=config.get("total_steps", 100_000), desc="Training")
    
    # Mixed-precision training loop
    try:
        while global_step < config.get("total_steps", 100_000):
            for x, y, weights in dataloader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                weights = weights.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                
                with autocast(dtype=torch.bfloat16):
                    logits, loss = model(x, targets=y)
                    weighted_loss = (loss * weights).mean()
                
                scaler.scale(weighted_loss).backward()
                if (global_step + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                
                # Update tracking
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": weighted_loss.item(),
                    "lr": optimizer.param_groups[0]['lr']
                })
                
                # Log metrics
                if wandb_available:
                    wandb.log({
                        "loss": weighted_loss.item(),
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "step": global_step
                    })
                
                # Checkpointing
                if global_step % config.get("checkpoint_steps", 10_000) == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=global_step,
                        config=config
                    )
                    
                    # Validation (optional)
                    if config.get("run_validation", False):
                        validation_loss = run_validation(model, device, config)
                        if wandb_available:
                            wandb.log({"validation_loss": validation_loss})
                
                if global_step >= config.get("total_steps", 100_000):
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final checkpoint...")
        save_checkpoint(model, optimizer, scheduler, global_step, config)
    
    finally:
        progress_bar.close()
        print("Training completed!")

def run_validation(model, device, config):
    """Optional validation loop"""
    model.eval()
    valid_dataset = load_from_disk("data/processed/validation_dataset")
    valid_loader = DataLoader(
        MultiDomainDataset(valid_dataset, config["model"]["block_size"]),
        batch_size=config.get("val_batch_size", 16),
        num_workers=os.cpu_count() // 2
    )
    
    total_loss = 0.0
    with torch.no_grad():
        for x, y, _ in valid_loader:
            x, y = x.to(device), y.to(device)
            with autocast(dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(valid_loader)

if __name__ == "__main__":
    main()