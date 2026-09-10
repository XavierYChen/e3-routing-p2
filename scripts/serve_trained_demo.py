"""Serve the archived trained-checkpoint P2 demo on localhost."""

from __future__ import annotations

import contextlib
import http.server
import socketserver
import webbrowser
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "artifacts" / "p2" / "trained-routing-analysis-20260909"
PORT = 8767


class ReusableServer(socketserver.TCPServer):
    allow_reuse_address = True


def main() -> None:
    if not (DIRECTORY / "trained-demo.html").is_file():
        raise SystemExit("trained-demo.html is missing; run run_trained_analysis.cmd first")
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(DIRECTORY))
    with ReusableServer(("127.0.0.1", PORT), handler) as server:
        url = f"http://127.0.0.1:{PORT}/trained-demo.html"
        print(f"E3 P2 trained demo: {url}")
        with contextlib.suppress(Exception):
            webbrowser.open(url)
        server.serve_forever()


if __name__ == "__main__":
    main()
