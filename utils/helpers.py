import torch
import os

def save_checkpoint(state, checkpoint_dir="checkpoints", filename="best_model.pt"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"✅ Saved checkpoint to {filepath}")

def load_checkpoint(filepath, model, optimizer=None, device="cpu"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get("loss", float("inf"))
    print(f"🔁 Loaded checkpoint from {filepath} (epoch {start_epoch}, best_loss={best_loss:.4f})")
    return start_epoch, best_loss