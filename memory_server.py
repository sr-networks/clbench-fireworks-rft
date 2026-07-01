#!/usr/bin/env python3
"""Launcher for the learn-and-recall MemoryMcp server."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_mcp import MemoryMcp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["streamable-http", "stdio"], default="streamable-http")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    if a.transport == "streamable-http":
        os.environ["PORT"] = str(a.port)
    MemoryMcp(seed=a.seed).run(transport=a.transport)


if __name__ == "__main__":
    main()
