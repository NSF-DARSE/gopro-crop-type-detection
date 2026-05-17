# Testing Strategy

This document outlines the evaluation methodology used for the Maize Crop Detection pipeline.

---

## 1. Dataset Split

The dataset was split using an **80/10/10 strategy**:

| Split | Images | Purpose |
|-------|--------|---------|
| Train | 1,104 (81%) | Model learning |
| Validation | 126 (9%) | Monitor training, early stopping |
| Test | 136 (10%) | Final unbiased evaluation |

**Why 80/10/10:**
- Training set is large enough for the model to learn diverse maize appearances
- Validation set is used during training to trigger early stopping (patience=40) and save the best checkpoint
- Test set is held out completely during training — used only once for final evaluation to prevent data leakage

**Important:** The test set images were never seen during training or validation. All detection results shown on the interactive map are from test set inference only.

---

## 2. Evaluation Metrics

### Primary Metrics

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **mAP50** | Mean Average Precision at IoU=0.50 | Standard object detection benchmark — measures both precision and recall |
| **mAP50-95** | mAP averaged over IoU 0.50–0.95 | Stricter metric — measures bounding box quality |
| **Precision** | True positives / (True + False positives) | Measures how many detections are actually maize |
| **Recall** | True positives / (True + False negatives) | Measures how many maize plants were found |
| **F1 Score** | Harmonic mean of Precision and Recall | Balanced metric |

### Why Recall Matters Most For This Project

For field survey mapping, **missing a maize plant is more costly than a false positive**. A missed detection means a real maize location is absent from the map. Therefore recall was prioritized alongside mAP50 when selecting the final model.

---

## 3. Model Comparison & Progressive Evaluation

Models were evaluated progressively — each iteration built on lessons from the previous:

| Model | imgsz | mAP50 | Precision | Recall | Key Change |
|-------|-------|-------|-----------|--------|------------|
| YOLOv8m (baseline) | 640 | 0.491 | — | — | Initial baseline |
| YOLOv8l | 640 | 0.508 | 0.502 | 0.497 | Larger model |
| YOLOv9s | 640 | 0.459 | 0.431 | 0.559 | Architecture change |
| YOLOv9m | 640 | 0.530 | 0.501 | 0.532 | Medium model |
| **YOLOv9m + CBAM** | **640** | **0.561** | **0.498** | **0.625** | CBAM attention added |

**Key finding:** YOLOv9m + CBAM gave the best mAP50 (0.561) and highest recall (0.625) — identifying 25% more maize plants than the baseline.

---

## 4. Confusion Matrix Analysis

The normalized confusion matrix from the best model revealed:

```
True Positive  (Maize → Maize)       : 57%  ✅ Correct detections
False Negative (Maize → Background)  : 43%  ⚠️ Missed maize plants
False Positive (Background → Maize)  : Some ⚠️ False alarms
```

**What this means:**
- The model correctly identifies 57% of maize plants
- 43% of maize plants are missed — primarily distant/small plants and plants with heavy background vegetation overlap
- False positives occur when background vegetation resembles maize structure

**Actions taken based on confusion matrix:**
- Added 61 negative samples (background images) to reduce false positives
- Lowered confidence threshold from 0.35 → 0.25 to improve recall
- Added CBAM attention to help distinguish maize from background

---

## 5. Inference Threshold Tuning

Two key thresholds were tuned during evaluation:

### Confidence Threshold (`conf`)
| Value | Effect |
|-------|--------|
| 0.50 | High precision, many misses |
| 0.35 | Balanced — original setting |
| **0.25** | **Higher recall — final setting** |

Lowering from 0.35 → 0.25 increased detections by ~30% with acceptable false positive rate for field mapping purposes.

### IoU Threshold (`iou`)
| Value | Effect |
|-------|--------|
| 0.45 | Standard NMS threshold |
| **0.35** | **Removes more duplicate boxes — final setting** |

Lowering IoU threshold reduced overlapping detections on the same plant.

### Test Time Augmentation (TTA)
`augment=True` was enabled at inference — applies horizontal flips and multi-scale inference, then merges results. This improved detection count by ~15% with no additional training.

---

## 6. Dataset Version Testing

Multiple dataset versions were tested to find the optimal configuration:

| Version | Images | Changes | mAP50 |
|---------|--------|---------|-------|
| v15 | 850 | Baseline dataset | 0.491 |
| v18 | 1,038 | Removed null/empty tiles | 0.508 |
| v21 | 1,100 | Added auto-annotated images | 0.530 |
| v24 | 1,366 | Fixed class name (Maize), final | **0.561** |

**Key lesson:** Data quality matters more than quantity. Fixing the class naming issue (from "item" → "Maize") and removing null tiles improved mAP more than simply adding more images.

---

## 7. Early Stopping Strategy

Early stopping was used to prevent overfitting:

| Run | Patience | Best Epoch | Total Epochs |
|-----|----------|------------|--------------|
| YOLOv8l | 30 | 18 | 48 |
| YOLOv9m | 50 | 27 | 77 |
| YOLOv9m + CBAM | 40 | 130 | 170 |

The CBAM model trained for significantly longer (130 epochs) before plateauing, indicating the attention mechanism required more training to converge — ultimately producing the best results.

---

## 8. Final Test Results

Inference on the 136 held-out test images:

| Metric | Value |
|--------|-------|
| Images processed | 136 |
| Images with detections | 128 (94%) |
| Total maize plants detected | 873 |
| States with detections | 6 of 8 |
| Average confidence | 0.457 |
| Max confidence | 0.845 |

### Detections by State

| State | Images | Plants Detected |
|-------|--------|----------------|
| Plateau | 85 | 494 |
| Ogun | 20 | 201 |
| Niger | 13 | 119 |
| FCT | 8 | 41 |
| Nasarawa | 1 | 9 |
| Oyo | 1 | 9 |
| **Total** | **128** | **873** |

---

## 9. Limitations & Future Testing

- **GPS ambiguity:** Duplicate image names across states may cause occasional coordinate mismatches
- **Test coverage:** Only 10% of images are used for evaluation — running inference on all splits would provide a more complete map
- **Annotation quality:** Auto-generated annotations (~216 images) may contain noise affecting model evaluation
- **Future:** Ensemble of YOLOv9m + YOLOv9m+CBAM showed 873 detections in preliminary testing and could be explored further
