
# Thesis Model and Research Plan

## 🎓 Thesis Title (Proposed)

**Intent-Aware Dynamic Spatio-Temporal Graph Neural Networks for Vehicle-Level Traffic Forecasting**

---

## 🔬 Research Contributions

1. **Dynamic Vehicle-Junction Graph Representation**  
   We introduce a novel graph-based representation of urban traffic in which both *vehicles* and *junctions* are modeled as graph nodes. This enables fine-grained, vehicle-level forecasting and allows the graph to dynamically evolve as vehicles move.

2. **User Intent Integration**  
   Unlike previous work that treats traffic as an aggregate process, we incorporate *user intent* in the form of GPS-based source-destination requests. This allows our model to predict not just current traffic but anticipated future congestion along users’ likely paths.

3. **Path-Aware Spatio-Temporal Modeling**  
   We propose a spatio-temporal graph architecture that conditions predictions on the inferred future trajectories of vehicles. This enables the model to capture both real-time state and projected traffic loads at future timestamps.

4. **Graph-Based Dataset for Intent-Aware Forecasting**  
   We design and release a new dataset composed of dynamic graph snapshots that capture both vehicle mobility and user destination information, enabling supervised learning of future traffic conditions along likely paths.

5. **Comprehensive Benchmarking**  
   We compare our model with several established spatio-temporal GNNs (DCRNN, ASTGCN, TGAT, etc.) and show that our intent-aware architecture significantly improves accuracy in predicting future ETA and congestion levels under realistic simulated traffic.

---

## 🧠 Custom Model: IA-STGNN

**Intent-Aware Spatio-Temporal Graph Neural Network (IA-STGNN)**

### Architecture Components

| Component | Description |
|----------|-------------|
| **Input Graph** | Nodes = {vehicles, junctions}, Edges = road segments + inferred movement paths |
| **Node Features** | Time-aware vehicle features (speed, heading, occupancy), junction state (load, signal status, etc.) |
| **Edge Features** | Distance, time delta, relative motion, path confidence (based on intent) |
| **Intent Encoder** | Encodes GPS source-destination request into a "trajectory likelihood" embedding |
| **Graph Encoder (GCN or GAT)** | Encodes each graph snapshot spatially |
| **Temporal Module (LSTM / GRU)** | Captures sequence of graph snapshots over time |
| **Trajectory Conditioning** | Conditions prediction on future path embedding (from Intent Encoder) |
| **Output** | ETA or congestion level at future time t along the path or next node(s) |

---

## 🏆 Models to Benchmark

- DCRNN
- ASTGCN
- TGAT
- GAT + LSTM
- ST-GCN

---

## 🚀 Action Plan

1. Write Architecture Section of Thesis
2. Sketch Model Block Diagram
3. Implement IA-STGNN
4. Prepare Baseline Comparison Framework
5. Write Final Contributions Section


Start writing the Proposed Model section of your thesis (in scientific style)?

Help define the model code plan (e.g. model.py layout for IA-STGNN)?

Create a block diagram showing the model pipeline?