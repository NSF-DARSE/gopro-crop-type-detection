# 🌽 Maize Crop Detection Using GoPro Imagery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv9](https://img.shields.io/badge/Model-YOLOv9m+CBAM-green.svg)](https://github.com/WongKinYiu/yolov9)

An end-to-end deep learning pipeline for detecting maize crops in GoPro field imagery, mapping GPS coordinates, and visualizing detections across Nigerian states.

🗺️ **[View Live Detection Map](https://nivaspanidapu.github.io/gopro-crop-type-detection/)**

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Setup & Installation](#setup--installation)
- [Pipeline](#pipeline)
- [GPS Coordinate Mapping](#gps-coordinate-mapping)
- [Interactive Map](#interactive-map)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [License](#license)

---

## Overview

This project detects maize crops from GoPro camera footage collected across 8 Nigerian states as part of a field survey. The pipeline:

1. Trains a YOLOv9m + CBAM attention model on annotated GoPro imagery
2. Runs inference on test images to detect maize plants
3. Matches detections to GPS coordinates from field survey CSV files
4. Exports results to Excel
5. Visualizes detections on an interactive map

**Collaborators:** Nivas Panidapu, Chinoye Agadi  
**Mentor:** Dr. Kyle Davis, University of Delaware  
**Institution:** University of Delaware — M.S. Data Science, 2025

---

## Dataset

- **Source:** GoPro field imagery from 8 Nigerian states (Plateau, Ogun, Niger, FCT, Nasarawa, Kwara, Oyo, Benue)
- **Annotations:** Roboflow — [crop-detection-tigp8](https://universe.roboflow.com/chinoye/crop-detection-tigp8)
- **Total images:** 1,366 (original resolution: 4000×3000)
- **Split:** 81% train / 9% valid / 10% test
- **Classes:** 1 (Maize)
- **Annotations:** 5,561 bounding boxes
- **GPS data:** 1,612,837 coordinate entries across all states

### Dataset Versions

| Version | Images | Notes |
|---------|--------|-------|
| v15 | 850 | Original baseline dataset |
| v18 | 1,038 | Null tiles removed |
| v21 | 1,100 | Auto-annotated images added |
| v24 | 1,366 | Final dataset — Maize class fixed, no resize |

---

## Model Architecture

### Final Model: YOLOv9m + CBAM Attention

YOLOv9m enhanced with **CBAM (Convolutional Block Attention Module)** at three backbone stages:

- **Channel Attention:** Focuses on which feature maps matter (e.g. green maize leaves vs brown soil)
- **Spatial Attention:** Focuses on where in the image to look

CBAM is applied after P3, P4, and P5 feature map stages to improve small object detection in dense field imagery.

---

## Results

### Model Comparison

| Model | imgsz | mAP50 | Precision | Recall | Detections |
|-------|-------|-------|-----------|--------|------------|
| YOLOv8l (baseline) | 640 | 0.508 | 0.502 | 0.497 | 106 |
| YOLOv9s | 640 | 0.459 | 0.431 | 0.559 | — |
| YOLOv9m | 640 | 0.530 | 0.501 | 0.532 | 106 |
| **YOLOv9m + CBAM (final)** | **640** | **0.561** | **0.498** | **0.625** | **873** |

### Final Detection Results

| State | Images Detected | Maize Plants |
|-------|----------------|--------------|
| Plateau | 85 | 494 |
| Ogun | 20 | 201 |
| Niger | 13 | 119 |
| FCT | 8 | 41 |
| Nasarawa | 1 | 9 |
| Oyo | 1 | 9 |
| **Total** | **128** | **873** |

### Training Configuration (Best Model)

```yaml
model:        YOLOv9m + CBAM
optimizer:    AdamW
lr0:          0.001
warmup_epochs: 5
patience:     40
batch:        16
imgsz:        640
epochs:       170 (stopped at best epoch 130)
augmentations: mosaic=1.0, mixup=0.15, degrees=15, flipud=0.3
```

---

## Compute Resources & Performance

### Resources Used

| Resource | Details |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| VRAM Used | ~8.3 GB (of 80 GB) |
| Platform | Google Colab Pro |
| Training Time | ~2.4 hours (170 epochs) |
| Inference Speed | 7.2ms per image |
| Model Size | 41 MB (best.pt) |

### Model Performance

| Metric | Value |
|---|---|
| mAP50 | 0.561 |
| mAP50-95 | 0.238 |
| Precision | 0.498 |
| Recall | 0.625 |
| Best Epoch | 130 / 170 |

### Detection Results

| Metric | Value |
|---|---|
| Test Images | 136 |
| Images with Detections | 128 |
| Total Maize Plants Detected | 873 |
| States Covered | 6 of 8 |
| Avg Confidence | 0.457 |

---

## Setup & Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Google Colab (Recommended)

All notebooks are designed to run on Google Colab with GPU runtime.

**Step 1 — Clone or open the notebook:**
- Open `CODE.ipynb` directly in Google Colab

**Step 2 — Enable GPU:**
- `Runtime → Change runtime type → Hardware accelerator → GPU`
- A100 recommended, T4 also works

**Step 3 — Add Roboflow API key:**
- Click the 🔑 Secrets icon in the left sidebar
- Add secret: Name = `ROBOFLOW_API_KEY`, Value = your key
- Get your key from [roboflow.com](https://roboflow.com) → Settings → API

**Step 4 — Upload GPS CSV:**
- Upload `coord2025_all.csv` to `/content/` in Colab
- This file contains 1.6M GPS coordinates from the field survey

**Step 5 — Run cells in order (Cell 1 → Cell 13):**
- Set `switch_train = True` to train the model
- After training, set `switch_train = False` and `switch_pred = True` to run inference

### Local Setup

```bash
git clone https://github.com/NSF-DARSE/gopro-crop-type-detection.git
cd gopro-crop-type-detection
pip install -r requirements.txt
```

> **Note:** GPU is required for training. CPU-only inference is possible but slow.

---

## Pipeline

The full pipeline runs in a single notebook with two switches:

```python
switch_train = True   # Train the model
switch_pred  = False  # Run inference (set True after training)
```

### Step 1: Training

```python
model = YOLO('yolov9m.pt')
model.train(
    data       = 'data.yaml',
    epochs     = 200,
    imgsz      = 640,
    batch      = 16,
    optimizer  = 'AdamW',
    lr0        = 0.001,
    single_cls = True
)
```

### Step 2: Inference (with TTA)

```python
result = model(image_path, conf=0.25, iou=0.35, augment=True)[0]
```

> **Future work:** Ensemble of YOLOv9m + YOLOv9m+CBAM could further improve detection coverage.

---

## GPS Coordinate Mapping

GPS coordinates are sourced from field survey CSV files (not image EXIF, which was stripped by Roboflow).

**Why not EXIF?**
GoPro cameras reset sequential filenames across sessions, causing duplicate image names across states (e.g., `G0013988` exists in both Nasarawa and Benue with different coordinates). Since annotated images were randomly sampled across all states without state-level metadata, EXIF-based lookup was unreliable.

**Solution:** Pre-built GPS CSV files from the field survey are used for coordinate lookup by image name.

```python
gps_lookup = gps_df.drop_duplicates('IMG').set_index('IMG')[
    ['Latitude', 'Longitude', 'State']
].to_dict('index')
```

---

## Interactive Map

The detection results are published as an interactive Folium map:

🗺️ **[https://nivaspanidapu.github.io/gopro-crop-type-detection/](https://nivaspanidapu.github.io/gopro-crop-type-detection/)**

**Features:**
- Street, Dark, and Satellite tile layers
- Density heatmap of detections
- Per-state toggleable marker layers
- Marker size proportional to detection count
- Popup with image name, state, plant count, confidence, and coordinates
- MiniMap and fullscreen support

---

## Project Structure

```
gopro-crop-type-detection/
│
├── notebooks/
│   ├── Maize_YOLOv9m_CBAM_v25_FINAL.ipynb   # Main training + inference notebook
│   └── auto_annotate_v2.py                    # Auto-annotation script
│
├── configs/
│   └── yolov9m_cbam.yaml                      # CBAM model config
│
├── index.html                                  # Interactive detection map
├── maize_detections.xlsx                       # Detection results with GPS
├── requirements.txt                            # Python dependencies
├── LICENSE                                     # MIT License
└── README.md
```

---

## Limitations

- **GPS ambiguity:** Duplicate image names across states may cause occasional coordinate mismatches
- **Test set only:** Map currently shows detections from 10% test split (~136 images)
- **Annotation quality:** Auto-generated annotations may contain false positives
- **Coverage:** 6 of 8 surveyed states have detections; Kwara and Benue have no matched test images
- **Confidence:** Average detection confidence of 0.457 — model benefits from more annotated data

---

## License

MIT License

Copyright (c) 2025 Nivas Panidapu, Chinoye Agadi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

**Dataset:** Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) via Roboflow.
