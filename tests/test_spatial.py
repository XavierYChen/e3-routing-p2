from pathlib import Path

import numpy as np
import pytest

from e3_routing_p2.spatial import letterbox, normalized_entropy, project_to_original, resolve_allowed_file, validate_record


def test_letterbox_round_trip_hotspot_is_aligned():
    image = np.zeros((60, 120, 3), dtype=np.uint8)
    _, transform = letterbox(image, 160)
    grid = np.zeros((160, 160), dtype=np.float32)
    grid[79:82, 79:82] = 1
    restored = project_to_original(grid, transform)
    y, x = np.unravel_index(np.argmax(restored), restored.shape)
    assert abs(x - 60) <= 2 and abs(y - 30) <= 2


def test_entropy_boundaries():
    one_hot = np.stack([np.ones((2,2)), np.zeros((2,2))])
    uniform = np.full((4,2,2), .25)
    assert float(normalized_entropy(one_hot).max()) < 1e-5
    assert np.allclose(normalized_entropy(uniform), 1, atol=1e-6)


def test_path_whitelist(tmp_path: Path):
    allowed = tmp_path / "allowed"; allowed.mkdir(); inside = allowed / "x.jpg"; inside.write_bytes(b"x")
    outside = tmp_path / "outside.jpg"; outside.write_bytes(b"x")
    assert resolve_allowed_file(inside, [allowed]) == inside.resolve()
    with pytest.raises(ValueError): resolve_allowed_file(outside, [allowed])


def test_schema_rejects_fake_token_record():
    valid = {"schema_version":"e3.spatial_routing.v1","family":"mot","family_display":"MOT","layer":"x","source":"router_softmax[B,E,H,W]","spatial_granularity":"token","feature_grid_hw":[10,10],"num_experts":3,"top_k":2,"probability_sum_max_error":0.0,"artifacts":{}}
    validate_record(valid)
    valid["family_display"] = "mot"
    with pytest.raises(ValueError): validate_record(valid)

