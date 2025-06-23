import os
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

class TrafficSnapshotDataset(Dataset):
    def __init__(self, pt_file_paths, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.pt_files = pt_file_paths

    def len(self):
        return len(self.pt_files)

    def get(self, idx):
        file_path = self.pt_files[idx]
        data = torch.load(file_path, weights_only=False)
        data.snapshot_idx = idx  # for tracking
        return data

def get_snapshot_dataloaders(config):
    data_dir = config["dataset"]["path"]
    all_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".pt")
    ])

    total = len(all_files)
    train_ratio = config["dataset"].get("train_ratio", 0.8)
    val_ratio = config["dataset"].get("val_ratio", 0.1)

    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)

    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    preserve_temporal = config["dataset"].get("preserve_temporal_order", False)
    shuffle_train = not preserve_temporal and config["dataset"].get("shuffle", True)

    def make_loader(file_list, shuffle):
        dataset = TrafficSnapshotDataset(file_list)
        return DataLoader(
            dataset,
            batch_size=config["dataset"].get("batch_size", 1),
            shuffle=shuffle,
            num_workers=config["dataset"].get("num_workers", 0)
        )

    return {
        "train": make_loader(train_files, shuffle_train),
        "val": make_loader(val_files, False),
        "test": make_loader(test_files, False),
    }


def get_vehicle_mask(batch):
    # node_type is the first feature in x
    node_type_column = batch.x[:, 0]
    return (node_type_column == 1)
