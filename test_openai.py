"""
Simple OpenAI Server Test

Quick test script for the OpenAI MCP server.
Run with: npx @modelcontextprotocol/inspector python test_openai.py
"""

import os
import sys
from pathlib import Path

def load_env_from_file(file_path):
    """Load environment variables from a file."""
    if not os.path.exists(file_path):
        return False
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip('"\'')
                os.environ[key] = value
    return True

def main():
    print("Starting OpenAI MCP Server Test...")
    
    # Load environment from .env.example
    if load_env_from_file('.env.example'):
        print("[OK] Environment loaded")
    else:
        print("[WARN] No .env.example found")
    
    # Check API key
    if os.getenv("OPENAI_API_KEY"):
        print("[OK] OpenAI API key configured")
    else:
        print("[ERROR] OpenAI API key missing")
        return
    
    # Import and run the server
    sys.path.append(str(Path(__file__).parent))
    
    from openai_server.server import main as server_main
    import asyncio
    
    asyncio.run(server_main())

if __name__ == "__main__":
    main()
