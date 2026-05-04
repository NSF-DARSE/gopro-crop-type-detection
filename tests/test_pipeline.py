import pytest, os, yaml

DATA_YAML = "docs/data.yaml"

class TestDataConfig:
    def test_data_yaml_exists(self):
        assert os.path.exists(DATA_YAML), "docs/data.yaml missing"
    def test_single_class(self):
        with open(DATA_YAML) as f:
            cfg = yaml.safe_load(f)
        assert cfg["nc"] == 1
    def test_class_name_is_maize(self):
        with open(DATA_YAML) as f:
            cfg = yaml.safe_load(f)
        assert "Maize" in cfg["names"]
    def test_roboflow_version(self):
        with open(DATA_YAML) as f:
            cfg = yaml.safe_load(f)
        assert cfg["roboflow"]["version"] == 24
    def test_roboflow_workspace(self):
        with open(DATA_YAML) as f:
            cfg = yaml.safe_load(f)
        assert cfg["roboflow"]["workspace"] == "chinoye"
    def test_train_val_test_splits_defined(self):
        with open(DATA_YAML) as f:
            cfg = yaml.safe_load(f)
        for split in ["train", "val", "test"]:
            assert split in cfg

class TestReproducibility:
    def test_requirements_txt_exists(self):
        assert os.path.exists("requirements.txt")
    def test_requirements_not_empty(self):
        with open("requirements.txt") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        assert len(lines) > 0
    def test_key_packages_in_requirements(self):
        with open("requirements.txt") as f:
            content = f.read().lower()
        for pkg in ["ultralytics", "torch", "pandas", "folium"]:
            assert pkg in content, f"{pkg} missing"
    def test_model_yaml_exists(self):
        assert os.path.exists("docs/yolov9m_cbam.yaml")
    def test_model_yaml_valid(self):
        with open("docs/yolov9m_cbam.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None
    def test_coord_sample_exists(self):
        found = any(os.path.exists(p) for p in [
            "docs/coord2025_sample.xlsx",
            "docs/Coordinates_Sample.xlsx",
            "docs/Coordinates_sample.xlsx"
        ])
        assert found
    def test_testing_md_exists(self):
        assert os.path.exists("TESTING.md")
    def test_changelog_md_exists(self):
        assert os.path.exists("CHANGELOG.md")
    def test_readme_md_exists(self):
        assert os.path.exists("README.md")
    def test_license_exists(self):
        assert os.path.exists("LICENSE")
    def test_notebook_exists(self):
        notebooks = [f for f in os.listdir(".") if f.endswith(".ipynb")]
        assert len(notebooks) > 0
    def test_tests_folder_exists(self):
        assert os.path.isdir("tests")
    def test_maize_detections_exists(self):
        assert os.path.exists("maize_detections.xlsx")
