"""
test_model.py
-------------
Tests for YOLOv9m+CBAM model performance on the held-out test set.
Run: pytest tests/test_model.py -v
"""

import pytest
from ultralytics import YOLO
import os

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "bestcbam.pt"
TEST_IMAGES  = "data/test/images"   # path to your 128 test images

# Expected metrics from final evaluation
EXPECTED_MAP50     = 0.561
EXPECTED_RECALL    = 0.625
EXPECTED_PRECISION = 0.498
EXPECTED_DETECTIONS = 873
METRIC_TOLERANCE   = 0.02   # allow ±2% tolerance


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def model():
    assert os.path.exists(MODEL_PATH), f"Model weights not found: {MODEL_PATH}"
    return YOLO(MODEL_PATH)


@pytest.fixture(scope="module")
def val_results(model):
    """Run validation on the test split once and reuse across tests."""
    return model.val(data="data.yaml", split="test", conf=0.25, iou=0.45)


@pytest.fixture(scope="module")
def inference_results(model):
    """Run inference on all test images."""
    return model.predict(TEST_IMAGES, conf=0.25, save=False)


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestModelPerformance:

    def test_model_loads(self, model):
        """Model weights load without error."""
        assert model is not None, "Model failed to load"

    def test_map50_within_tolerance(self, val_results):
        """mAP50 should be ≥ expected value within tolerance."""
        map50 = float(val_results.box.map50)
        assert map50 >= EXPECTED_MAP50 - METRIC_TOLERANCE, (
            f"mAP50 {map50:.3f} below expected {EXPECTED_MAP50:.3f} "
            f"(tolerance ±{METRIC_TOLERANCE})"
        )

    def test_recall_within_tolerance(self, val_results):
        """Recall should be ≥ expected value within tolerance."""
        recall = float(val_results.box.r.mean())
        assert recall >= EXPECTED_RECALL - METRIC_TOLERANCE, (
            f"Recall {recall:.3f} below expected {EXPECTED_RECALL:.3f}"
        )

    def test_precision_within_tolerance(self, val_results):
        """Precision should be ≥ expected value within tolerance."""
        precision = float(val_results.box.p.mean())
        assert precision >= EXPECTED_PRECISION - METRIC_TOLERANCE, (
            f"Precision {precision:.3f} below expected {EXPECTED_PRECISION:.3f}"
        )

    def test_total_detection_count(self, inference_results):
        """Total detections across all test images should match expected count."""
        total = sum(len(r.boxes) for r in inference_results)
        assert abs(total - EXPECTED_DETECTIONS) <= 10, (
            f"Detection count {total} differs from expected {EXPECTED_DETECTIONS}"
        )

    def test_detects_only_maize_class(self, inference_results):
        """Model should only output class 0 (Maize) — single-class project."""
        for result in inference_results:
            if result.boxes is not None and len(result.boxes):
                classes = result.boxes.cls.tolist()
                assert all(int(c) == 0 for c in classes), (
                    f"Non-maize class detected in {result.path}"
                )

    def test_confidence_above_threshold(self, inference_results):
        """All detections should have confidence ≥ 0.25 (our threshold)."""
        for result in inference_results:
            if result.boxes is not None and len(result.boxes):
                confs = result.boxes.conf.tolist()
                assert all(c >= 0.25 for c in confs), (
                    f"Detection below confidence threshold in {result.path}"
                )

    def test_bbox_coordinates_valid(self, inference_results):
        """Bounding boxes should be within image bounds (normalised 0–1)."""
        for result in inference_results:
            if result.boxes is not None and len(result.boxes):
                xyxyn = result.boxes.xyxyn.tolist()
                for box in xyxyn:
                    x1, y1, x2, y2 = box
                    assert 0 <= x1 < x2 <= 1, f"Invalid x coords: {x1}, {x2}"
                    assert 0 <= y1 < y2 <= 1, f"Invalid y coords: {y1}, {y2}"
