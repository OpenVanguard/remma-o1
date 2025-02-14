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
from torch.optim.lr_scheduler import CosineAnnealingLR

class WikipediaDataset(Dataset):
    def __init__(self, tokenized_data, block_size=2048):
        self.data = tokenized_data
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        chunk = example["input_ids"]
        
        if len(chunk) > self.block_size + 1:
            start_idx = torch.randint(0, len(chunk) - self.block_size - 1, (1,)).item()
            chunk = chunk[start_idx:start_idx + self.block_size + 1]
        else:
            padding = [0] * (self.block_size + 1 - len(chunk))
            chunk = chunk + padding
            
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y

    @staticmethod
    def collate_fn(batch):
        x = torch.stack([item[0] for item in batch])
        y = torch.stack([item[1] for item in batch])
        return x, y

def train_batch(model, x, y, optimizer, device):
    model.train()
    optimizer.zero_grad()
    
    x, y = x.to(device), y.to(device)
    _, loss = model(x, targets=y)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

def main():
    # Load configuration
    with open("configs/model_remma.yaml") as f:
        config = yaml.safe_load(f)
    
    # Convert learning rates to float
    config["training"]["learning_rate"] = float(config["training"]["learning_rate"])
    config["training"]["min_lr"] = float(config["training"]["min_lr"])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Adjust batch size for CPU training
    if device.type == 'cpu':
        original_batch_size = config["training"]["batch_size"]
        config["training"]["batch_size"] = min(8, original_batch_size)  # Smaller batch size for CPU
        print(f"Adjusted batch size from {original_batch_size} to {config['training']['batch_size']} for CPU training")
    
    # Load dataset
    try:
        dataset = load_from_disk(config["training"]["dataset_path"].replace('.arrow', ''))
        dataset = dataset.shuffle()  # Shuffle the dataset for better training


        print(f"Dataset loaded with {len(dataset)} examples")
        print(f"Available columns: {dataset.column_names}")
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {str(e)}")

    # Initialize model
    model = build_model(config["model"]).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Create training dataset and dataloader
    train_dataset = WikipediaDataset(dataset, config["model"]["block_size"])
    train_dataset = torch.utils.data.Subset(train_dataset, range(0, 10000))  # Limit to first 10,000 samples for testing

    dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0 if device.type == 'cpu' else max(1, os.cpu_count() // 2),  # No workers for CPU
        pin_memory=device.type == 'cuda',
        collate_fn=WikipediaDataset.collate_fn
    )
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=(0.9, 0.95)
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["total_steps"],
        eta_min=config["training"]["min_lr"]
    )
    
    # Training setup
    global_step = 0
    model.train()
    
    # Initialize WandB if available
    try:
        import wandb
        wandb.init(project="remma-wiki-training", config=config)
        wandb.watch(model)
        use_wandb = True
    except ImportError:
        print("Wandb not installed, skipping logging")
        use_wandb = False
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Progress tracking
    progress_bar = tqdm(total=config["training"]["total_steps"], desc="Training")
    
    try:
        epoch = 0
        while global_step < config["training"]["total_steps"]:
            print(f"\nStarting epoch {epoch}")
            
            for batch_idx, (x, y) in enumerate(dataloader):
                # Train batch
                loss = train_batch(model, x, y, optimizer, device)
                scheduler.step()
                
                # Update tracking
                global_step += 1
                progress_bar.update(1)
                
                # Log metrics
                if global_step % 10 == 0:
                    progress_bar.set_postfix({
                        "loss": f"{loss:.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "epoch": epoch
                    })
                    
                    if use_wandb:
                        wandb.log({
                            "loss": loss,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "epoch": epoch,
                            "step": global_step
                        })
                
                # Save checkpoint
                if global_step % config["training"]["checkpoint_steps"] == 0:
                    checkpoint_path = os.path.join("checkpoints", f"step_{global_step}.pt")
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        step=global_step,
                        config=config,
                        path=checkpoint_path
                    )

                    print(f"\nCheckpoint saved at step {global_step}")
                
                if global_step >= config["training"]["total_steps"]:
                    break
            
            epoch += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final checkpoint...")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=global_step,
            config=config,
            path=os.path.join("checkpoints", "interrupted.pt")
        )

    
    finally:
        progress_bar.close()
        if use_wandb:
            wandb.finish()
        print(f"Training completed! Final step: {global_step}")

if __name__ == "__main__":
    main()
