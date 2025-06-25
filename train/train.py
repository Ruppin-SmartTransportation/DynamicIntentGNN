import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
from utils.dataset import get_snapshot_dataloaders, get_vehicle_mask
from models import get_model
from utils.config import load_config
import torch.nn.functional as F
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils.helpers import save_best_model, load_best_model, get_checkpoint_path
from eval import run_evaluation

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_vehicles = 0

    for batch in tqdm(loader, desc="Training"):
        start = time.time()

        batch = batch.to(device)
        mask = get_vehicle_mask(batch)
        pred_y = model(batch)[mask]
        true_y = batch.y

        # Compute sum of squared errors
        loss = F.mse_loss(pred_y, true_y, reduction='sum')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()

        total_loss += loss.item()
        total_vehicles += mask.sum().item()

        print(
            f"Batch time: {end - start:.3f} sec | "
            f"batch.y.std: {batch.y.std():.4f}, batch.y.mean: {batch.y.mean():.4f}, "
            f"num_vehicles: {int(mask.sum())} | ave batch loss: {loss.item()/mask.sum().item():.4f}"
        )

    mean_loss = total_loss / total_vehicles if total_vehicles > 0 else float('inf')
    return mean_loss



def run_training(config_path):
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(config).to(device)
    loaders = get_snapshot_dataloaders(config)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    
    ckpt_dir = get_checkpoint_path(config)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = os.path.join(ckpt_dir, "best_model.pt")

    if os.path.exists(best_model_path):
        print("Resuming from checkpoint...")
        load_best_model(model, best_model_path, optimizer, device)
    else:
        print("No checkpoint found, starting fresh.")

    exp_name = f"{config['model']['name']}_bs{config['dataset']['batch_size']}_hid{config['model']['hidden_channels']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{exp_name}")
    best_loss = float('inf')
    for epoch in range(config["training"]["epochs"]):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        writer.add_scalar("Loss/train", loss, epoch)
        print(f"[Epoch {epoch+1}] Loss: {loss:.10f}")

        if loss < best_loss:
            best_loss = loss
            save_best_model(model, optimizer, best_model_path, epoch, loss)
            print(f"New best model saved with loss: {best_loss:.10f}")
        
        if (epoch + 1) % config["training"]["eval_interval"] == 0:
            print("Running evaluation...")
            run_evaluation(config_path, best_model_path, val_loader, writer, epoch)
            print("Evaluation complete.")

    writer.close()
if __name__ == "__main__":
    run_training("configs/baseline_gcn.yaml")
