"""
test_output.py — fixed column names matching actual maize_detections.xlsx
Columns: IMG, FILE, BOX, CONF, X1, Y1, X2, Y2, Latitude, Longitude, State, Class, Confidence
"""
import pytest, pandas as pd, os

XLSX_PATH = "maize_detections.xlsx"
REQUIRED_COLUMNS = ["IMG","FILE","BOX","CONF","X1","Y1","X2","Y2","Latitude","Longitude","State","Class","Confidence"]
NIGERIA_LAT_MIN, NIGERIA_LAT_MAX = 4.0, 14.0
NIGERIA_LON_MIN, NIGERIA_LON_MAX = 2.0, 15.0
VALID_STATES = {"Plateau","Ogun","Niger","FCT","Nasarawa","Kwara","Oyo","Benue"}
EXPECTED_DETECTIONS = 873
EXPECTED_IMAGES = 132
CONF_THRESHOLD = 0.25

@pytest.fixture(scope="module")
def df():
    assert os.path.exists(XLSX_PATH), f"{XLSX_PATH} not found"
    return pd.read_excel(XLSX_PATH)

class TestOutputSchema:
    def test_file_exists(self):
        assert os.path.exists(XLSX_PATH)
    def test_required_columns_present(self, df):
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        assert not missing, f"Missing columns: {missing}"
    def test_total_detection_count(self, df):
        assert len(df) == EXPECTED_DETECTIONS, f"Expected {EXPECTED_DETECTIONS}, got {len(df)}"
    def test_unique_image_count(self, df):
        unique = df["IMG"].nunique()
        assert unique == EXPECTED_IMAGES, f"Expected {EXPECTED_IMAGES} images, got {unique}"
    def test_no_null_latitude(self, df):
        assert df["Latitude"].isnull().sum() == 0, "Null Latitude values found"
    def test_no_null_longitude(self, df):
        assert df["Longitude"].isnull().sum() == 0, "Null Longitude values found"
    def test_no_null_state(self, df):
        assert df["State"].isnull().sum() == 0, "Null State values found"
    def test_confidence_above_threshold(self, df):
        below = (df["CONF"] < CONF_THRESHOLD).sum()
        assert below == 0, f"{below} detections below conf threshold {CONF_THRESHOLD}"
    def test_class_is_maize(self, df):
        non_maize = df[df["Class"] != "Maize"]
        assert len(non_maize) == 0, f"Non-Maize classes: {non_maize['Class'].unique()}"
    def test_bbox_x_valid(self, df):
        assert (df["X1"] < df["X2"]).all(), "Some boxes have X1 >= X2"
    def test_bbox_y_valid(self, df):
        assert (df["Y1"] < df["Y2"]).all(), "Some boxes have Y1 >= Y2"

class TestGPSPipeline:
    def test_latitude_within_nigeria(self, df):
        out = df[(df["Latitude"] < NIGERIA_LAT_MIN) | (df["Latitude"] > NIGERIA_LAT_MAX)]
        assert len(out) == 0, f"{len(out)} detections outside Nigeria latitude"
    def test_longitude_within_nigeria(self, df):
        out = df[(df["Longitude"] < NIGERIA_LON_MIN) | (df["Longitude"] > NIGERIA_LON_MAX)]
        assert len(out) == 0, f"{len(out)} detections outside Nigeria longitude"
    def test_state_values_valid(self, df):
        invalid = df[~df["State"].isin(VALID_STATES)]
        assert len(invalid) == 0, f"Invalid states: {invalid['State'].unique()}"
    def test_plateau_has_most_images(self, df):
        top = df.groupby("State")["IMG"].nunique().idxmax()
        assert top == "Plateau", f"Expected Plateau most images, got {top}"
    def test_known_image_spot_check(self, df):
        row = df[df["IMG"] == "G0449783"]
        if len(row) > 0:
            assert row.iloc[0]["State"] == "Plateau"
            assert 8.0 <= row.iloc[0]["Latitude"] <= 10.5
        else:
            pytest.skip("G0449783 not in dataset")
    def test_gps_full_coverage(self, df):
        coverage = df["Latitude"].notnull().sum() / len(df)
        assert coverage == 1.0, f"GPS coverage {coverage:.1%}"
