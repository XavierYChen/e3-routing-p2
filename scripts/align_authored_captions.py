"""Align approved Chinese narration to speech windows measured by Whisper.

The Whisper transcript supplies timing only. Displayed text always comes from
``narrated_demo_manifest.json`` so model names and technical terms remain exact.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
from pathlib import Path


def timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def split_line(text: str, max_chars: int = 15) -> list[str]:
    """Create short caption phrases while preserving every authored character."""
    pieces = [piece for piece in re.split(r"(?<=[。！？；，])", text) if piece]
    chunks: list[str] = []
    for piece in pieces:
        count = max(1, math.ceil(len(piece) / max_chars))
        size = math.ceil(len(piece) / count)
        while piece:
            chunks.append(piece[:size].strip())
            piece = piece[size:].strip()
    for index in range(len(chunks) - 1):
        left = re.search(r"([A-Za-z0-9-]+)$", chunks[index])
        right = re.match(r"([A-Za-z0-9-]+)", chunks[index + 1])
        if left and right:
            word = left.group(1) + right.group(1)
            chunks[index] = chunks[index][: left.start()].rstrip()
            chunks[index + 1] = word + chunks[index + 1][right.end() :]
    chunks = [chunk for chunk in chunks if chunk]
    return chunks


def normalize(text: str) -> str:
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE).casefold()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--whisper-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    whisper = json.loads(args.whisper_json.read_text(encoding="utf-8"))
    measured = whisper["captions"]
    raw_chars: list[tuple[str, float, float]] = []
    for cue in measured:
        chars = list(normalize(cue["text"]))
        if not chars:
            continue
        start, end = float(cue["start"]), float(cue["end"])
        step = (end - start) / len(chars)
        raw_chars.extend((char, start + index * step, start + (index + 1) * step) for index, char in enumerate(chars))

    authored_text = "".join(scene["vo_line"] for scene in manifest["scenes"])
    authored_chars = list(normalize(authored_text))
    matcher = difflib.SequenceMatcher(a=[item[0] for item in raw_chars], b=authored_chars, autojunk=False)
    similarity = matcher.ratio()
    aligned: list[tuple[float, float] | None] = [None] * len(authored_chars)
    for tag, raw_start, raw_end, authored_start, authored_end in matcher.get_opcodes():
        if authored_start == authored_end:
            continue
        if tag == "equal":
            for offset in range(authored_end - authored_start):
                aligned[authored_start + offset] = raw_chars[raw_start + offset][1:]
            continue
        left = raw_chars[raw_start][1] if raw_start < raw_end else (aligned[authored_start - 1][1] if authored_start else 0.0)
        right = raw_chars[raw_end - 1][2] if raw_start < raw_end else (
            raw_chars[raw_start][1] if raw_start < len(raw_chars) else left + 0.05 * (authored_end - authored_start)
        )
        step = max((right - left) / (authored_end - authored_start), 0.01)
        for offset in range(authored_end - authored_start):
            aligned[authored_start + offset] = (left + offset * step, left + (offset + 1) * step)
    if any(item is None for item in aligned):
        raise RuntimeError("failed to align every authored character")

    cues: list[dict] = []
    char_cursor = 0
    for scene in manifest["scenes"]:
        for chunk in split_line(scene["vo_line"]):
            length = len(normalize(chunk))
            span = aligned[char_cursor : char_cursor + length]
            cues.append({"start": span[0][0], "end": span[-1][1], "text": chunk, "scene": scene["title"]})
            char_cursor += length

    authored = normalize("".join(scene["vo_line"] for scene in manifest["scenes"]))
    aligned = normalize("".join(cue["text"] for cue in cues))
    if authored != aligned:
        raise RuntimeError("aligned captions do not preserve the complete authored narration")

    srt = "\n".join(
        f"{index}\n{timestamp(cue['start'])} --> {timestamp(cue['end'])}\n{cue['text']}\n"
        for index, cue in enumerate(cues, start=1)
    )
    args.output.write_text(srt, encoding="utf-8")
    print(json.dumps({"caption_count": len(cues), "last_caption_end": round(cues[-1]["end"], 3), "whisper_similarity": round(similarity, 3)}))


if __name__ == "__main__":
    main()
