"""
test_pipeline.py
----------------
Tests for data integrity, split correctness, and reproducibility.
Run: pytest tests/test_pipeline.py -v
"""

import pytest
import os
import yaml
import glob

# ── Config ────────────────────────────────────────────────────────────────────
DATA_YAML        = "docs/data.yaml"
MODEL_YAML       = "docs/yolov9m_cbam.yaml"
REQUIREMENTS_TXT = "requirements.txt"
COORD_CSV        = "coord2025_all.csv"
COORD_SAMPLE     = "docs/coord2025_sample.xlsx"

EXPECTED_TOTAL_IMAGES  = 1366
EXPECTED_TRAIN_RATIO   = 0.81
EXPECTED_VAL_RATIO     = 0.09
EXPECTED_TEST_RATIO    = 0.10
SPLIT_TOLERANCE        = 0.02

EXPECTED_GPS_RECORDS   = 1_612_837


# ── Data Split Integrity Tests ────────────────────────────────────────────────
class TestDataSplit:

    @pytest.fixture(scope="class")
    def data_config(self):
        assert os.path.exists(DATA_YAML), f"data.yaml not found"
        with open(DATA_YAML) as f:
            return yaml.safe_load(f)

    def _count_images(self, path):
        exts = ["*.jpg", "*.jpeg", "*.png"]
        return sum(len(glob.glob(os.path.join(path, e))) for e in exts)

    def test_data_yaml_exists(self):
        assert os.path.exists(DATA_YAML), "data.yaml missing"

    def test_train_val_test_paths_defined(self, data_config):
        for key in ["train", "val", "test"]:
            assert key in data_config, f"'{key}' split missing from data.yaml"

    def test_single_class_maize(self, data_config):
        """Must be single class (nc=1) named Maize."""
        assert data_config.get("nc") == 1, (
            f"Expected nc=1, got {data_config.get('nc')}"
        )
        names = data_config.get("names", [])
        assert "maize" in [n.lower() for n in names], (
            f"'maize' not in class names: {names}"
        )

    def test_no_train_test_image_overlap(self, data_config):
        """No image should appear in both train and test splits."""
        train_path = data_config.get("train", "")
        test_path  = data_config.get("test", "")
        if os.path.exists(train_path) and os.path.exists(test_path):
            train_files = set(os.path.basename(f) for f in glob.glob(
                os.path.join(train_path, "*.jpg")))
            test_files  = set(os.path.basename(f) for f in glob.glob(
                os.path.join(test_path, "*.jpg")))
            overlap = train_files & test_files
            assert len(overlap) == 0, (
                f"Data leakage: {len(overlap)} images in both train and test"
            )

    def test_no_train_val_image_overlap(self, data_config):
        """No image should appear in both train and val splits."""
        train_path = data_config.get("train", "")
        val_path   = data_config.get("val", "")
        if os.path.exists(train_path) and os.path.exists(val_path):
            train_files = set(os.path.basename(f) for f in glob.glob(
                os.path.join(train_path, "*.jpg")))
            val_files   = set(os.path.basename(f) for f in glob.glob(
                os.path.join(val_path, "*.jpg")))
            overlap = train_files & val_files
            assert len(overlap) == 0, (
                f"Data leakage: {len(overlap)} images in both train and val"
            )

    def test_approximate_split_ratios(self, data_config):
        """Train/val/test split should be approximately 81/9/10%."""
        train_path = data_config.get("train", "")
        val_path   = data_config.get("val", "")
        test_path  = data_config.get("test", "")
        if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
            n_train = self._count_images(train_path)
            n_val   = self._count_images(val_path)
            n_test  = self._count_images(test_path)
            total   = n_train + n_val + n_test
            assert total > 0, "No images found"
            assert abs(n_train / total - EXPECTED_TRAIN_RATIO) <= SPLIT_TOLERANCE
            assert abs(n_val   / total - EXPECTED_VAL_RATIO)   <= SPLIT_TOLERANCE
            assert abs(n_test  / total - EXPECTED_TEST_RATIO)  <= SPLIT_TOLERANCE


# ── Reproducibility Tests ─────────────────────────────────────────────────────
class TestReproducibility:

    def test_requirements_txt_exists(self):
        """requirements.txt must be present for reproducible setup."""
        assert os.path.exists(REQUIREMENTS_TXT), "requirements.txt missing"

    def test_requirements_not_empty(self):
        """requirements.txt must not be empty."""
        with open(REQUIREMENTS_TXT) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        assert len(lines) > 0, "requirements.txt is empty"

    def test_key_packages_in_requirements(self):
        """Critical packages must be pinned in requirements.txt."""
        with open(REQUIREMENTS_TXT) as f:
            content = f.read().lower()
        for pkg in ["ultralytics", "torch", "pandas", "folium", "openpyxl"]:
            assert pkg in content, f"'{pkg}' missing from requirements.txt"

    def test_model_yaml_exists(self):
        """yolov9m_cbam.yaml must exist in docs/ for reproducibility."""
        assert os.path.exists(MODEL_YAML), f"{MODEL_YAML} missing from docs/"

    def test_model_yaml_valid(self):
        """yolov9m_cbam.yaml must be valid YAML."""
        with open(MODEL_YAML) as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None, "yolov9m_cbam.yaml is empty or invalid"

    def test_coord_sample_exists(self):
        """coord2025_sample.xlsx must exist in docs/ for GPS pipeline testing."""
        assert os.path.exists(COORD_SAMPLE), f"{COORD_SAMPLE} missing"

    def test_testing_md_exists(self):
        """TESTING.md must exist and document the testing strategy."""
        assert os.path.exists("TESTING.md"), "TESTING.md missing from repo root"

    def test_changelog_md_exists(self):
        """CHANGELOG.md must exist documenting version history."""
        assert os.path.exists("CHANGELOG.md"), "CHANGELOG.md missing"

    def test_readme_md_exists(self):
        """README.md must exist with project documentation."""
        assert os.path.exists("README.md"), "README.md missing"

    def test_license_exists(self):
        """LICENSE file must exist (MIT)."""
        assert os.path.exists("LICENSE"), "LICENSE file missing"
