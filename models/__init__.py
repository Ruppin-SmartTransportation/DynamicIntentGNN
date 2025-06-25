from models.baseline_gcn import BaselineGCN

def get_model(config):
    name = config["model"]["name"]
    if name == "baseline_gcn":
        return BaselineGCN(
            node_in_channels=config["model"]["node_in_channels"],
            hidden_channels=config["model"]["hidden_channels"],
            out_channels=config["model"].get("out_channels", 1),
            dropout=config["model"].get("dropout", 0.2)
        )
    raise ValueError(f"Unknown model: {name}")
