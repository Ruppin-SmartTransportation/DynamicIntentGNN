from utils.config import load_config
from utils.dataset import get_snapshot_dataloaders

config = load_config("configs/stgnn.yaml")
loaders = get_snapshot_dataloaders(config)


for batch in loaders["train"]:
    print(batch)
    break