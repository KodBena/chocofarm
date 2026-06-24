"""
throughput-lab/hp/__main__.py — `python -m hp` entry point (delegates to cli.main).

Public Domain (The Unlicense).
"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
