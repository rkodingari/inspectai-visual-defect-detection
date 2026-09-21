from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


@pytest.fixture()
def synthetic_data(tmp_path):
    from inspectai.data.prepare import generate_synthetic

    root = tmp_path / "data"
    generate_synthetic(root)
    return root

