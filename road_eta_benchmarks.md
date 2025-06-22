# Benchmarks in Road Travel Time Prediction

This table summarizes key papers and models in the domain of Estimated Time of Arrival (ETA) prediction. It includes both academic and production-scale models, with results where available in seconds.

| Name / Model                      | Link | Summary | Results |
|----------------------------------|------|---------|---------|
| **AttentionTTE (2024)**          | [DOI](https://doi.org/10.3389/frai.2024.1258086) | Deep learning model combining local/global spatial correlation, temporal dependencies, and external factors using attention + LSTM. | MAE: **227 s** |
| **Transformer ETA (2023)**       | [Paper](https://ieeexplore.ieee.org/document/9998743) | Uses Transformer architecture for urban travel time prediction. Compared with LSTM and kNN. | MAE: **27.6 s** (Transformer), LSTM: 29.3 s, kNN: 31.8 s |
| **SA-LSTM + Filter (2022)**      | [Study](https://www.mdpi.com/1424-8220/22/1/330) | Applies self-attention LSTM with Butterworth filter on 100 km highway data. | MAE: **729 s / 100 km** |
| **Google Maps GNN (2020+)**      | [Blog](https://ai.googleblog.com/2020/06/using-graph-neural-networks-to.html) | Production-scale GNN by Google Maps for ETA, improves over prior methods. | ~**40% reduction** in ETA error |
| **DiDi / DeepTravel (2020)**     | [Paper](https://arxiv.org/abs/1802.04798) | Deep learning model for large-scale ride-hailing ETA. Incorporates multi-source data. | Relative improvement; absolute seconds not specified |
