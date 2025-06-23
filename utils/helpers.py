import torch
import os

def save_best_model(model, optimizer, path, epoch=0, loss=0.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss
    }, path)
    print(f"✅ Saved best model to {path}")

def load_best_model(model, filepath, optimizer=None, device="cpu"):
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

def get_checkpoint_path(config):
    return os.path.join("checkpoints", config["model"]["name"])