"""CLI for the E3 P2 evidence generator and local interactive demo."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="E3 P2 five-family routing lens")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="generate verified evidence")
    run_parser.add_argument("--config", type=Path, default=Path("configs/p2_v2.yaml"))
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--no-latest", action="store_true")
    demo_parser = commands.add_parser("demo", help="serve the interactive evidence viewer")
    demo_parser.add_argument("--host", default="127.0.0.1")
    demo_parser.add_argument("--port", type=int, default=8766)
    demo_parser.add_argument("--run-dir", type=Path)
    demo_parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        from .runner import run
        print(run(args.config, run_id=args.run_id, update_latest=not args.no_latest))
    else:
        from .demo import serve
        serve(host=args.host, port=args.port, run_dir=args.run_dir, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
