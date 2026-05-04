"""
test_output.py
--------------
Tests for maize_detections.xlsx output schema and GPS coordinate validity.
Run: pytest tests/test_output.py -v
"""

import pytest
import pandas as pd
import os

# ── Config ────────────────────────────────────────────────────────────────────
XLSX_PATH = "maize_detections.xlsx"

REQUIRED_COLUMNS = ["image_id", "confidence", "x1", "y1", "x2", "y2", "lat", "lon", "state"]

# Nigeria bounding box (lat 4–14°N, lon 2–15°E)
NIGERIA_LAT_MIN, NIGERIA_LAT_MAX = 4.0, 14.0
NIGERIA_LON_MIN, NIGERIA_LON_MAX = 2.0, 15.0

VALID_STATES = {
    "Plateau", "Ogun", "Niger", "FCT",
    "Nasarawa", "Kwara", "Oyo", "Benue"
}

EXPECTED_TOTAL_DETECTIONS = 873
EXPECTED_TEST_IMAGES      = 128


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def detections():
    assert os.path.exists(XLSX_PATH), f"Output file not found: {XLSX_PATH}"
    return pd.read_excel(XLSX_PATH)


# ── Schema Tests ──────────────────────────────────────────────────────────────
class TestOutputSchema:

    def test_file_exists(self):
        """Output xlsx file must exist."""
        assert os.path.exists(XLSX_PATH), f"{XLSX_PATH} not found"

    def test_required_columns_present(self, detections):
        """All required columns must be present in xlsx."""
        missing = [c for c in REQUIRED_COLUMNS if c not in detections.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_no_null_lat_lon(self, detections):
        """No detection should have a null lat or lon — GPS lookup must succeed."""
        null_lat = detections["lat"].isnull().sum()
        null_lon = detections["lon"].isnull().sum()
        assert null_lat == 0, f"{null_lat} detections have null latitude"
        assert null_lon == 0, f"{null_lon} detections have null longitude"

    def test_no_null_image_id(self, detections):
        """Every detection must reference a valid image_id."""
        assert detections["image_id"].isnull().sum() == 0, "Null image_id found"

    def test_total_detection_count(self, detections):
        """Total rows should equal expected detection count."""
        assert len(detections) == EXPECTED_TOTAL_DETECTIONS, (
            f"Expected {EXPECTED_TOTAL_DETECTIONS} detections, got {len(detections)}"
        )

    def test_confidence_range(self, detections):
        """All confidence scores must be between 0.25 and 1.0."""
        below = (detections["confidence"] < 0.25).sum()
        above = (detections["confidence"] > 1.0).sum()
        assert below == 0, f"{below} detections below confidence threshold"
        assert above == 0, f"{above} detections above 1.0"

    def test_image_count(self, detections):
        """Detections should come from exactly 128 test images."""
        unique_images = detections["image_id"].nunique()
        assert unique_images == EXPECTED_TEST_IMAGES, (
            f"Expected {EXPECTED_TEST_IMAGES} unique images, got {unique_images}"
        )


# ── GPS Validity Tests ────────────────────────────────────────────────────────
class TestGPSPipeline:

    def test_lat_within_nigeria(self, detections):
        """All latitudes must fall within Nigeria's bounding box (4–14°N)."""
        out = detections[
            (detections["lat"] < NIGERIA_LAT_MIN) |
            (detections["lat"] > NIGERIA_LAT_MAX)
        ]
        assert len(out) == 0, (
            f"{len(out)} detections have latitude outside Nigeria: "
            f"{out[['image_id','lat']].head()}"
        )

    def test_lon_within_nigeria(self, detections):
        """All longitudes must fall within Nigeria's bounding box (2–15°E)."""
        out = detections[
            (detections["lon"] < NIGERIA_LON_MIN) |
            (detections["lon"] > NIGERIA_LON_MAX)
        ]
        assert len(out) == 0, (
            f"{len(out)} detections have longitude outside Nigeria: "
            f"{out[['image_id','lon']].head()}"
        )

    def test_state_values_valid(self, detections):
        """All state values must be one of the 8 surveyed Nigerian states."""
        invalid = detections[~detections["state"].isin(VALID_STATES)]
        assert len(invalid) == 0, (
            f"Invalid state values found: {invalid['state'].unique()}"
        )

    def test_known_image_gps_lookup(self, detections):
        """
        Spot-check: image G0324525 should map to Plateau State.
        This verifies the GPS lookup is correct for a known image.
        """
        row = detections[detections["image_id"] == "G0324525"]
        if len(row) > 0:
            assert row.iloc[0]["state"] == "Plateau", (
                f"G0324525 expected Plateau, got {row.iloc[0]['state']}"
            )
            # Plateau State lat/lon range
            assert 8.0 <= row.iloc[0]["lat"] <= 10.5, "Lat outside Plateau range"
            assert 8.0 <= row.iloc[0]["lon"] <= 11.0, "Lon outside Plateau range"
        else:
            pytest.skip("G0324525 not in test set — skipping spot check")

    def test_gps_lookup_coverage(self, detections):
        """At least 95% of images should have a successful GPS match."""
        matched = detections["lat"].notnull().sum()
        coverage = matched / len(detections)
        assert coverage >= 0.95, (
            f"GPS coverage {coverage:.1%} below 95% threshold"
        )
