"""Verify the narrated demo, caption coverage, and manifest hashes."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import av

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "p2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE).casefold()


def main() -> None:
    metadata_path = ARTIFACTS / "e3-p2-narrated-subtitled-demo.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    video = ARTIFACTS / metadata["file"]
    proof = ARTIFACTS / metadata["proof_image"]
    captions = ARTIFACTS / metadata["caption_sidecar"]
    manifest = json.loads((ROOT / metadata["caption_text_source"]).read_text(encoding="utf-8"))

    assert video.stat().st_size == metadata["file_size_bytes"]
    assert sha256(video) == metadata["sha256"]
    assert proof.stat().st_size == metadata["proof_image_size_bytes"]
    assert sha256(proof) == metadata["proof_image_sha256"]

    cues = [block for block in re.split(r"\n\s*\n", captions.read_text(encoding="utf-8").strip()) if block]
    displayed = "".join(" ".join(block.splitlines()[2:]) for block in cues)
    authored = "".join(scene["vo_line"] for scene in manifest["scenes"])
    assert len(cues) == metadata["caption_count"]
    assert normalize(displayed) == normalize(authored)

    with av.open(video) as container:
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0]
        duration = float(container.duration / av.time_base)
        assert abs(duration - metadata["duration_seconds"]) < 0.2
        assert video_stream.codec_context.name == metadata["video_codec"]
        assert audio_stream.codec_context.name == metadata["audio_codec"]
        assert (video_stream.width, video_stream.height) == (metadata["width"], metadata["height"])

    print(f"PASS: {duration:.3f}s, {len(cues)} captions, complete authored text, hashes verified")


if __name__ == "__main__":
    main()
