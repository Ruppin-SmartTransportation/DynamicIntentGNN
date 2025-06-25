import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch
import torch.nn.functional as F
from utils.dataset import get_vehicle_mask
from models import get_model
from utils.config import load_config
from tqdm import tqdm

def load_label_stats(stats_path):
    df = pd.read_csv(stats_path)
    eta_row = df[df["feature"] == "eta"]
    if eta_row.empty:
        raise ValueError("ETA statistics not found in labels summary CSV.")
    mean = float(eta_row["mean"].values[0])
    std = float(eta_row["std"].values[0])
    return mean, std

def evaluate_model(model, dataloader, device, eta_mean, eta_std):
    model.eval()
    mae_total, rmse_total, mape_total, count = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            mask = get_vehicle_mask(batch)
            if mask.sum() == 0:
                continue

            pred = model(batch)[mask]
            true = batch.y

            # Inverse normalization
            pred_eta = pred * eta_std + eta_mean
            true_eta = true * eta_std + eta_mean

            mae = F.l1_loss(pred_eta, true_eta, reduction='sum').item()
            rmse = torch.sum((pred_eta - true_eta) ** 2).item()
            mape = torch.sum(torch.abs((pred_eta - true_eta) / true_eta.clamp(min=1))).item() * 100

            n = true.size(0)
            mae_total += mae
            rmse_total += rmse
            mape_total += mape
            count += n

    if count == 0:
        print("⚠️ No valid predictions for this batch.")
        return None

    mae = mae_total / count
    rmse = (rmse_total / count) ** 0.5
    mape = mape_total / count
    print(f"MAE: {mae:.2f} sec | RMSE: {rmse:.2f} sec | MAPE: {mape:.2f}%")
    print(f"🧪 Debug prediction vs true (first few):")
    print(f"  Pred ETA: {pred_eta[:5].squeeze().tolist()}")
    print(f"  True ETA: {true_eta[:5].squeeze().tolist()}")

    return mae, rmse, mape

def run_evaluation(config_path, model_ckpt_path=None, val_loader=None, writer=None, epoch=None):
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(config).to(device)
    
    if val_loader is None:
        from utils.dataset import get_snapshot_dataloaders
        loaders = get_snapshot_dataloaders(config)
        val_loader = loaders["val"]
    
    if model_ckpt_path is None:
        model_ckpt_path = os.path.join("checkpoints", config["model"]["name"], "best_model.pt")

    checkpoint = torch.load(model_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    stats_path = os.path.join(config["dataset"]["labels_path"],"labels_feature_summary.csv")
    eta_mean, eta_std = load_label_stats(stats_path)

    metrics = evaluate_model(model, val_loader, device, eta_mean, eta_std)
    if metrics:
        mae, rmse, mape = metrics
        print(f"MAE: {mae:.2f} sec | RMSE: {rmse:.2f} sec | MAPE: {mape:.2f}%")
        if writer is not None and epoch is not None:
            writer.add_scalar("MAE/val", mae, epoch)
            writer.add_scalar("RMSE/val", rmse, epoch)
            writer.add_scalar("MAPE/val", mape, epoch)
    else:
        print("No valid samples found for evaluation.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_gcn.yaml", help="Path to config YAML")
    args = parser.parse_args()

    run_evaluation(args.config)
    