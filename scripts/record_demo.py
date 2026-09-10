"""Render a two-minute MP4 from the real P2 interactive evidence page."""

from __future__ import annotations

import argparse
import contextlib
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "artifacts" / "p2"
DEMO_PAGE = "trained-routing-analysis-20260909/trained-demo.html"


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        return


@contextlib.contextmanager
def local_server(directory: Path):
    handler = partial(QuietHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/{DEMO_PAGE}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def select_label(page, selector: str, contains: str) -> None:
    value = page.eval_on_selector(
        selector,
        "(el, needle) => [...el.options].find(x => x.textContent.includes(needle))?.value",
        contains,
    )
    if value is None:
        raise RuntimeError(f"no option containing {contains!r} in {selector}")
    page.select_option(selector, value)
    page.wait_for_timeout(120)


def set_state(page, *, family="mot", layer="19", tab="routing", top=False):
    page.select_option("#family", family)
    page.wait_for_timeout(120)
    select_label(page, "#layer", f"model.{layer}.")
    page.click(f'[data-tab="{tab}"]')
    page.evaluate("value => window.scrollTo({top: value, behavior: 'instant'})", 0 if top else 250)
    page.wait_for_timeout(160)


def screenshot(page) -> np.ndarray:
    raw = page.screenshot(type="jpeg", quality=92)
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("browser screenshot could not be decoded")
    return frame


def decorate(frame: np.ndarray, title: str, subtitle: str, progress: float, phase: float) -> np.ndarray:
    height, width = frame.shape[:2]
    zoom = 1.0 + 0.007 * np.sin(np.pi * phase)
    crop_w, crop_h = int(width / zoom), int(height / zoom)
    x0, y0 = (width - crop_w) // 2, (height - crop_h) // 2
    frame = cv2.resize(frame[y0:y0 + crop_h, x0:x0 + crop_w], (width, height), interpolation=cv2.INTER_LINEAR)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 92), (5, 12, 27), -1)
    cv2.rectangle(overlay, (0, height - 34), (width, height), (5, 12, 27), -1)
    frame = cv2.addWeighted(overlay, 0.93, frame, 0.07, 0)
    cv2.putText(frame, title, (42, 40), cv2.FONT_HERSHEY_DUPLEX, 0.90, (255, 229, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (42, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 235, 248), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, height - 10), (int(width * progress), height), (217, 200, 0), -1)
    cursor_x = int(width * (0.18 + 0.64 * phase))
    cursor_y = int(height * (0.20 + 0.04 * np.sin(phase * 2 * np.pi)))
    cv2.circle(frame, (cursor_x, cursor_y), 11, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (cursor_x, cursor_y), 4, (80, 210, 255), -1, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "p2" / "e3-p2-two-minute-demo.mp4")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scenes = [
        (12, "E3 ROUTING LENS", "Interactive evidence generated from the trained checkpoint", dict(top=True)),
        (9, "MOT / LAYER 13", "Two active experts are measured on this image", dict(family="mot", layer="13")),
        (9, "MOT / LAYER 16", "Expert activity changes with router depth", dict(family="mot", layer="16")),
        (11, "MOT / LAYER 19", "All three experts are active; colors are real argmax assignments", dict(family="mot", layer="19")),
        (9, "MOT / LAYER 22", "Two active experts here is valid Top-K behavior", dict(family="mot", layer="22")),
        (11, "MOA / LAYER 16", "Three spatial regions with continuous probability statistics", dict(family="moa", layer="16")),
        (9, "MOA / LAYER 19", "Margin and entropy prevent over-reading the color map", dict(family="moa", layer="19")),
        (8, "APPEARANCE INPUT AUDIT", "Exact brightness, contrast and blur inputs keep the same geometry", dict(tab="inputs")),
        (10, "APPEARANCE SENSITIVITY", "Mean and descriptive SD cover 16 image-by-layer units", dict(tab="appearance")),
        (10, "RELATIVE ATTRIBUTION", "Layer shares are descriptive rather than causal", dict(tab="attribution")),
        (8, "ABSOLUTE ROUTER MAE", "One shared log scale exposes the MOT/MOA magnitude gap", dict(tab="absolute")),
        (7, "SCATTER ANALYSIS", "Probability margin explains apparent routing instability", dict(tab="scatter")),
        (7, "EXPERT USAGE", "Per-layer usage exposes balance and specialization", dict(tab="usage")),
    ]
    total_seconds = sum(scene[0] for scene in scenes)
    if total_seconds != 120:
        raise RuntimeError(f"scene plan must total 120 seconds, got {total_seconds}")
    edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    with local_server(DEMO_DIR) as url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=str(edge), args=["--disable-gpu-sandbox"])
        page = browser.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1)
        page.goto(url, wait_until="networkidle")
        page.wait_for_function("document.body.dataset.ready === 'true'")
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1600, 900))
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open the MP4 writer")
        previous = None
        elapsed = 0
        try:
            for duration, title, subtitle, state in scenes:
                set_state(page, **state)
                current = screenshot(page)
                count = duration * args.fps
                for index in range(count):
                    phase = index / max(count - 1, 1)
                    frame = current
                    if previous is not None and index < args.fps:
                        alpha = index / max(args.fps - 1, 1)
                        frame = cv2.addWeighted(previous, 1 - alpha, current, alpha, 0)
                    progress = (elapsed + index / args.fps) / total_seconds
                    writer.write(decorate(frame.copy(), title, subtitle, progress, phase))
                previous = current
                elapsed += duration
        finally:
            writer.release()
            browser.close()
    print(f"wrote {output} ({total_seconds}s at {args.fps}fps)")


if __name__ == "__main__":
    main()
