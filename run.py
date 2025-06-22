from utils.config import load_config
from utils.dataset import get_snapshot_dataloader

config = load_config("configs/stgnn.yaml")
loader = get_snapshot_dataloader(config)

for batch in loader:
    print(batch)
    break