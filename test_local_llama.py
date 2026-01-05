"""
Simple test script for the Local LLaMA MCP server.

Usage (from project root):

    npx @modelcontextprotocol/inspector python test_local_llama.py

This script:
- Loads environment variables from `.env.example` if present
- Starts the Local LLaMA MCP server in stdio mode
- Prints nothing to stdout except MCP protocol traffic
"""

import os
import sys
from pathlib import Path


def load_env_from_file(file_path: str) -> bool:
    """Load environment variables from a simple KEY=VALUE file.

    This mirrors the behavior used in the other test_*.py scripts and
    is safe for use with MCP Inspector (no extra stdout noise).
    """
    if not os.path.exists(file_path):
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                value = value.strip("'\"")
                os.environ[key] = value
    return True


def main() -> None:
    # Load environment from .env.example (current dir),
    # then fall back to parent if needed. Silent on purpose.
    if not load_env_from_file(".env.example"):
        load_env_from_file(str(Path(__file__).parent.parent / ".env.example"))

    # Ensure local_llama_server package is importable
    sys.path.append(str(Path(__file__).parent))

    from local_llama_server.server import main as server_main
    import asyncio

    # Delegate entirely to the server's main async entrypoint
    asyncio.run(server_main())


if __name__ == "__main__":
    main()
