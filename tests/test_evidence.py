import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "verified-p2-cpu-20260907"


def test_committed_evidence_contract_and_video_duration():
    payload = json.loads((RESULT / "spatial_records.json").read_text(encoding="utf-8"))
    by_family = {record["family"]: record for record in payload["records"]}
    assert set(by_family) == {"moe", "mot", "latent"}
    assert by_family["mot"]["spatial_granularity"] == "token"
    assert by_family["mot"]["feature_grid_hw"] == [20, 20]
    assert by_family["mot"]["representative_expert_probability"]["range"] > 0
    assert by_family["moe"]["spatial_granularity"] == by_family["latent"]["spatial_granularity"] == "image"
    serialized = json.dumps(payload)
    assert "D:\\\\AI" not in serialized and "C:\\\\Users" not in serialized
    cap = cv2.VideoCapture(str(RESULT / "e3_p2_demo_120s.mp4"))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert abs(duration - 120.0) <= 0.1
