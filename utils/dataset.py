import os
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

class TrafficSnapshotDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.pt_files = sorted([
            f for f in os.listdir(root)
            if f.endswith('.pt')
        ])

    def len(self):
        return len(self.pt_files)

    def get(self, idx):
        file_path = os.path.join(self.root, self.pt_files[idx])
        data = torch.load(file_path, weights_only=False)
        data.snapshot_idx = idx  # for tracking
        return data

def get_snapshot_dataloader(config):
    dataset = TrafficSnapshotDataset(config["dataset"]["path"])
    return DataLoader(
        dataset,
        batch_size=config["dataset"].get("batch_size", 1),
        shuffle=config["dataset"].get("shuffle", False),
        num_workers=config["dataset"].get("num_workers", 0)
    )

def get_vehicle_mask(batch):
    # node_type is the first feature in x
    node_type_column = batch.x[:, 0]
    return (node_type_column == 1)
