# ✅ DynamicIntentGNN: ETA Prediction Model – To-Do List

## 🔹 1. Evaluation Pipeline
- [ ] Create `evaluate.py`
  - [ ] Load trained model + `.pt` graph files
  - [ ] Load ETA normalization params from `labels_feature_summary.csv`
  - [ ] Apply **inverse normalization** to predictions
  - [ ] Compute:
    - [ ] **MAE** (Mean Absolute Error)
    - [ ] **RMSE** (Root Mean Squared Error)
    - [ ] **MAPE** (Mean Absolute Percentage Error)
  - [ ] Optionally: Save predictions + errors to CSV

## 🔹 2. Live Metrics in Training
- [ ] Load ETA `mean` and `std` from file (already done in EDA)
- [ ] In `train.py`, log:
  - [ ] Inverse-normalized **MAE / RMSE** after each epoch
  - [ ] Target is:
    - MAE ≈ 300 sec → normalized MAE ≤ 0.116
    - MSE loss target ≈ 0.013

## 🔹 3. Checkpointing
- [ ] In `train.py`, add:
  - [ ] Save best model (lowest val loss)
  - [ ] Save latest model (resume support)
- [ ] Add resume logic:
  - [ ] Detect if checkpoint exists
  - [ ] Load model, optimizer, and epoch
  - [ ] Continue training from saved point

## 🔹 4. ETA Normalization
- [x] Normalize ETA using z-score in dataset creation
- [ ] In **evaluation & logging**, convert back to seconds:
```python
def inverse_normalize(y, mean, std):
    return y * std + mean
```

## 🔹 5. Dataset Preprocessing
- [ ] Load ETA `mean` and `std` from `labels_feature_summary.csv`
- [ ] In `convert_snapshot()`, filter labels with `|z| > 3`
- [ ] Only include vehicles with ETA in normal range
- [ ] Log number of filtered vehicles per snapshot

## 🔹 6. Nice-to-Have Add-ons
- [ ] Add CLI argument to select train/eval mode
- [ ] Option to dump per-vehicle prediction error
- [ ] Visualize prediction vs. actual ETA distributions