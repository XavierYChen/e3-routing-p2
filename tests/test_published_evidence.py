import json
from pathlib import Path


RUN = Path(__file__).parents[1] / "artifacts" / "p2" / "p2-v2-five-family-final"


def test_published_demo_has_no_stale_individual_paths():
    if not RUN.is_dir():
        return
    index = json.loads((RUN / "demo-index.json").read_text(encoding="utf-8"))
    assert len(index["entries"]) == 192
    assert len({entry["sprite_path"] for entry in index["entries"]}) == 32
    assert all("path" not in entry for entry in index["entries"])
    assert all((RUN / entry["sprite_path"]).is_file() for entry in index["entries"])
    assert not (RUN / "overlays").exists()
