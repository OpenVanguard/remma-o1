import torch
import os
from datetime import datetime

def save_checkpoint(model, optimizer, step, config):
    checkpoint_dir = f"models/checkpoints/{config['model']['name']}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{config['model']['name']}_step{step}_{timestamp}.pt"
    
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config
    }, f"{checkpoint_dir}/{filename}")