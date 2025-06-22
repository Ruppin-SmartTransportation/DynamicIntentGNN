import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.dataset import get_snapshot_dataloader, get_vehicle_mask
from models import get_model
from utils.config import load_config
from utils.helpers import save_checkpoint, load_checkpoint
import torch.nn.functional as F
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        start = time.time()
        batch = batch.to(device)
        mask = get_vehicle_mask(batch)
        pred_y = model(batch)[mask]
        true_y = batch.y
        loss = F.mse_loss(pred_y, true_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()
        total_loss += loss.item()
        print(f"Batch time: {end - start:.3f} sec | loss {loss.item()} , len {len(loader)}")
    return total_loss / len(loader)

def run_training(config_path):
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(config).to(device)
    loader = get_snapshot_dataloader(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    best_model_path = os.path.join("checkpoints", config["model"]["name"], "best_model.pt")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    best_loss = float("inf")
    start_epoch = 0

    if os.path.exists(best_model_path):
        print(f"Resuming from checkpoint: {best_model_path}")
        start_epoch, best_loss = load_checkpoint(best_model_path, model, optimizer, device)

    exp_name = f"{config['model']['name']}_bs{config['dataset']['batch_size']}_hid{config['model']['hidden_channels']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{exp_name}")

    for epoch in range(start_epoch, config["training"]["epochs"]):
        loss = train_one_epoch(model, loader, optimizer, device)
        writer.add_scalar("Loss/train", loss, epoch)
        print(f"[Epoch {epoch+1}] Loss: {loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss
            }, checkpoint_dir=os.path.dirname(best_model_path), filename=os.path.basename(best_model_path))
            print(f"✅ Saved new best model at epoch {epoch+1} with loss {best_loss:.4f}")
    writer.close()

if __name__ == "__main__":
    run_training("configs/baseline_gcn.yaml")
