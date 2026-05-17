# Changelog

All notable changes to the Maize Crop Detection project are documented here.

---

## [v1.0.0] — 2025-04-30 (Final Release)

### Model
- Final model: YOLOv9m + CBAM attention backbone
- mAP50: 0.561 | Recall: 0.625 | 873 maize plant detections
- Training: AdamW, lr=0.001, imgsz=640, batch=16, patience=40

### Pipeline
- End-to-end detection pipeline: inference → GPS lookup → Excel → map
- TTA (Test Time Augmentation) enabled at inference (augment=True)
- GPS coordinates sourced from field survey CSV (1,612,837 entries)
- Results exported to Excel with image name, state, confidence, coordinates

### Map
- Interactive Folium map published on GitHub Pages
- Features: satellite/street/dark tiles, heatmap, per-state layers, popups
- Live map: https://nivaspanidapu.github.io/gopro-crop-type-detection/

---

## [v0.4.0] — 2025-04-25

### Added
- CBAM (Convolutional Block Attention Module) attention backbone to YOLOv9m
- Dropout (0.1) and weight decay (0.001) to reduce overfitting
- Auto-annotation script (auto_annotate_v2.py) for new images
- 216 auto-annotated images + 61 negative samples added to dataset

### Changed
- Dataset version: v21 → v24 (Maize class name fixed, was showing as "item")
- Optimizer: SGD → AdamW
- Learning rate: 0.0001 → 0.001 with 5 epoch warmup
- patience: 30 → 40

### Results
- mAP50: 0.561 | Recall: 0.625 | Detections: 873

---

## [v0.3.0] — 2025-04-20

### Added
- YOLOv9m model replacing YOLOv8l
- GPS coordinate lookup from combined field survey CSV
- Excel output with detection results and GPS coordinates
- Enhanced interactive map with heatmap and state-level filtering

### Changed
- Dataset: v15 → v18 (null tiles removed, 1,038 clean images)
- imgsz: 640 → 1280 (original GoPro resolution)
- Confidence threshold: 0.35 → 0.25 at inference

### Results
- mAP50: 0.530 | Recall: 0.599 | Detections: 289

---

## [v0.2.0] — 2025-04-10

### Added
- YOLOv8l baseline model
- Roboflow dataset with 80/10/10 split
- Basic GPS coordinate extraction from image EXIF
- Initial Folium map visualization

### Changed
- Dataset augmentation: 5x outputs, flip, brightness, exposure
- Discovered EXIF GPS stripping issue with Roboflow — switched to CSV-based lookup

### Results
- mAP50: 0.508 | Recall: 0.497 | Detections: 106

---

## [v0.1.0] — 2025-03-15 (Initial)

### Added
- Initial project setup
- GoPro image collection from 8 Nigerian states
- Manual annotation on Roboflow (850 original images)
- YOLOv8m baseline training
- Basic detection pipeline

### Results
- mAP50: 0.491 | Initial baseline established
