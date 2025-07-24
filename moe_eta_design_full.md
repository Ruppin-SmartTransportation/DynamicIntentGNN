# Mixture of Experts (MoE) Design for ETA Prediction in Dynamic Graphs

## Overview
We propose a Mixture of Experts (MoE) architecture for vehicle-level ETA prediction using dynamic spatio-temporal traffic graphs. The model integrates a shared GNN encoder, multiple specialized expert networks, and a gating network that dynamically routes each vehicle embedding to a suitable expert (or weighted combination of experts).

---

## Model Components

### 1. Shared Graph Encoder

```python
# Pseudocode for Transformer-based Route Encoder
class RouteEncoder(nn.Module):
    def __init__(self, num_edges, edge_embed_dim=16, n_heads=2, n_layers=1):
        super().__init__()
        self.edge_embedding = nn.Embedding(num_edges, edge_embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=edge_embed_dim, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, route_edge_ids):
        # route_edge_ids: [batch_size, max_route_len] (padded edge indices)
        edge_embeds = self.edge_embedding(route_edge_ids)  # → [B, L, D]
        route_encoded = self.transformer(edge_embeds)      # → [B, L, D]
        return route_encoded.mean(dim=1)                   # → [B, D] aggregated embedding
```

- **Temporal Input Window**: When incorporating temporal aggregation, the model receives a window of graph snapshots `[G_{t-T+1}, ..., G_t]`. For each vehicle, the encoder extracts a sequence of embeddings over time. A temporal model (e.g., LSTM, GRU, Transformer, or 1D Conv) aggregates these into a single embedding `	ilde{h}_i`, which captures the vehicle's recent spatio-temporal context. This temporal embedding is then passed to the gating and expert modules to produce a single ETA prediction per vehicle.

- **Extended Vehicle Features**: In addition to the existing node features, we augment each vehicle's feature vector with the following:
  - **Current edge demand**: a normalized estimate of future traffic load on the current road segment
  - **Current edge occupancy**: number of vehicles currently on the edge, normalized or log-normalized
  - **Current edge lane count**: number of lanes (one-hot encoded)
  - **Trip progress**: computed as `1 - (route_length_left / route_length)`, indicating how far along the trip the vehicle is
  - **Route encoder output**: The aggregated route embedding, produced by a dedicated route encoder (e.g., the `RouteEncoder` Transformer block shown above), is treated as a key feature in the model. This vector summarizes the structural and contextual characteristics of the vehicle’s full path. The output is appended to the vehicle’s node feature vector, contributing an additional fixed-size embedding (e.g., 10–32 values) before graph encoding. This representation enables the GNN and gating components to reason about ETA in light of the planned route. (See `RouteEncoder` pseudocode above.)
  - **Destination coordinates**: the x/y position of the vehicle's destination, normalized or embedded, to help the model disambiguate between similar current contexts but different trip goals.

These features are appended to the vehicle node feature vector before being passed to the GNN encoder.

- **Input**: Dynamic traffic graph snapshot `G_t`
- **Structure**: A spatio-temporal GNN that incorporates both static topology and dynamic edge/node features (e.g., current edge, vehicle state, route position).
- **Recommended Architectures**:
  - GCN or GAT with edge-type encoding
  - Custom modules supporting dynamic edge construction (e.g., relative position-based edges)
  - Optional temporal aggregation (e.g., with GRU, LSTM, or 1D conv on previous snapshots)
- **Output**: Contextual node embeddings `h_i`, where each embedding captures a vehicle's state and surrounding traffic context.
- **Normalization & Input Handling**:
  - Normalize numerical features (speed, position)
  - One-hot or embedding for categorical features (edge ID, junction ID, vehicle type)
  - Ensure batching with PyG DataLoader is consistent (collation, padding)

- **Embedding Intuition**: The encoder produces an embedding `h_i` for each vehicle, which captures both the vehicle's internal state (speed, route position, current edge) and its external context (local traffic density, road conditions, surrounding junctions). This embedding serves as a comprehensive input to the expert network, enabling accurate and context-aware ETA predictions.

### 2. Expert Networks
- **Structure**: Each expert is an MLP that takes a vehicle embedding `h_i` and outputs a scalar ETA prediction. In practice, each expert receives the full `[num_vehicles, embedding_dim]` tensor — i.e., all vehicle embeddings in the current batch. The expert processes each vehicle independently and in parallel, producing a tensor `[num_vehicles, 1]` of ETA predictions. This is repeated across all `K` experts, resulting in `K` candidate predictions for each vehicle that will be weighted and aggregated by the gating network.
- **Input Interpretation**: Each expert operates on the vehicle's embedding `h_i`, which is not a final set of prediction coefficients, but a learned representation summarizing all relevant contextual factors — such as traffic density, route topology, current edge dynamics, and vehicle state. This embedding is produced by the GNN encoder, and serves as a rich input for the expert's MLP to transform into an ETA prediction. The MLP learns to interpret the embedded features in a nonlinear way, capturing complex relationships beyond what a linear model could achieve.
- **Specialization Goal**: Each expert is implicitly or explicitly encouraged to specialize in distinct traffic conditions (e.g., zone, time of day, congestion level)
- **Number of Experts**: `K`, to be tuned via validation

### 3. Gating Network
- **Input**: Vehicle embedding `h_i`, which is the output of the shared graph encoder. Optionally, this embedding can be concatenated with auxiliary features such as time of day, current zone, route length, or vehicle type to help the gating network make more informed decisions about which expert(s) to prioritize.
- **Output**: Softmax-normalized vector `[lpha_1, ..., lpha_K]`
- **Purpose**: Assigns a weight to each expert's output for the given vehicle

### 4. MoE Output
- **Learning Behavior**: Although gating weights are used after the experts compute their outputs, the entire MoE block is trained end-to-end. The gating network learns to assign different weights to experts based on vehicle context. Each expert receives gradient signals based on how much it contributed to the final prediction (proportional to `lpha_ij`). This leads each expert to specialize in the subspace of inputs it is most responsible for, without requiring explicit supervision.
- **Computation**:
  ```
ETA_i = sum_{j=1}^{K} alpha_ij * Expert_j(h_i)
```
- **Loss Function**: Mean Squared Error (MSE) between predicted ETA and ground truth
- **Regularization**:
  - Expert usage entropy (to encourage balanced expert activation)
  - Diversity regularization (optional)

---

## Workflow

### Training
1. Load traffic snapshot and labels
2. Encode graph with GNN → get embeddings `h_i`
3. For each vehicle:
   - Compute gating weights `[lpha_1, ..., lpha_K]`
   - Compute expert outputs `[y_1, ..., y_K]`
   - Compute final prediction `\hat{y} = \sum lpha_j y_j`
4. Compute total loss (MSE + regularization)
5. Backpropagate through encoder, experts, and gate
6. Log expert usage statistics and evaluate diversity

### Evaluation
- Use validation set to tune:
  - Number of experts
  - Gating network structure
  - Use of auxiliary inputs (time, zone)
- Monitor:
  - ETA RMSE/MAE
  - Expert entropy (diversity)
  - Expert load balancing

### Inference
1. Receive new snapshot `G_t`
2. Encode graph to get `h_i`
3. For each vehicle:
   - Compute `lpha` via gating network
   - Compute expert predictions
   - Output weighted ETA prediction
4. (Optional) Return top contributing experts per vehicle for interpretability

---

## Next Steps
- Explore optional architectural enhancements:
  - **Temporal Embedding Aggregation**: Stack embeddings from past N snapshots and apply LSTM, GRU, or Transformer encoder to model traffic trends.
  - **Expert Load Balancing Loss**: Add entropy or KL-divergence regularization to encourage diverse expert usage and avoid expert collapse.
  - **Expert Interpretability Logging**: Track and log expert selection per vehicle during inference to validate specialization.
  - **Monitor Feature Redundancy**: Ensure that overlapping information (e.g., edge features shared via both GNN and auxiliary inputs) improves rather than harms model performance.
- Design expert specialization heuristics (manual or learned)
- Implement gating and expert diversity diagnostics
- Build MoE model class in PyTorch based on this specification
- Integrate with DatasetCreator-generated `.pt` files
