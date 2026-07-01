#!/usr/bin/env python3
"""Launcher for the Blind-Spectrum MCP-Gym server (started as a subprocess by the rollout processor).

    python spectrum_server.py --port 9100 [--seed 0] [--transport streamable-http|stdio]
"""

import argparse
import os
import sys
from pathlib import Path

# Make sibling modules (spectrum_mcp, spectrum_adapter) importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spectrum_mcp import SpectrumMcp


def main():
    parser = argparse.ArgumentParser(description="CLBench Blind-Spectrum MCP-Gym Server")
    parser.add_argument("--transport", choices=["streamable-http", "stdio"], default="streamable-http")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.transport == "streamable-http":
        os.environ["PORT"] = str(args.port)

    server = SpectrumMcp(seed=args.seed)
    print(f"📡 Starting CLBench Blind-Spectrum MCP server on port {args.port} (seed={args.seed}, transport={args.transport})")
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
