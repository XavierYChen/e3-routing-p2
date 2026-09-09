import numpy as np
import pytest
from PIL import Image

from e3_p2.trained_analysis import apply_transform, comparison


def test_probability_comparison_reports_switch_and_mae():
    reference = np.asarray([[[0.7, 0.4]], [[0.3, 0.6]]], dtype=np.float32)
    candidate = np.asarray([[[0.6, 0.8]], [[0.4, 0.2]]], dtype=np.float32)
    result = comparison(reference, candidate)
    assert result["probability_mae"] == pytest.approx(0.25)
    assert result["dominant_switch_fraction"] == pytest.approx(0.5)
    assert result["dominant_expert_agreement"] == pytest.approx(0.5)


def test_appearance_transform_changes_pixels_without_geometry_drift():
    image = Image.new("RGB", (7, 5), (100, 120, 140))
    transformed = apply_transform(image, {"name": "brightness_090", "kind": "brightness", "factor": 0.9})
    assert transformed.size == image.size
    assert np.any(np.asarray(transformed) != np.asarray(image))
