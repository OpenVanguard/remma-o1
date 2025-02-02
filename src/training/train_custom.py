import yaml
import torch
from src.modeling import build_model

# Load config
with open("configs/model_remma.yaml") as f:
    config = yaml.safe_load(f)

# Initialize model
model = build_model(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training loop (simplified)
for batch in dataloader:  # You'll need to implement DataLoader
    inputs, targets = batch
    logits, loss = model(inputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()