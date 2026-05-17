# conftest.py
# Shared pytest configuration for the Maize Crop Detection test suite.

import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (model inference)"
    )
