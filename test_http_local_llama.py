"""
Local LLaMA Server HTTP Test

Starts the Local LLaMA MCP server in HTTP mode on port 8004.
Run with: python test_http_local_llama.py

Prerequisites:
    - Ollama must be running: ollama serve

Then connect via MCP Inspector:
    npx @modelcontextprotocol/inspector --url http://127.0.0.1:8004
"""

import os
import sys
from pathlib import Path


def load_env_from_file(file_path):
    """Load environment variables from a file."""
    if not os.path.exists(file_path):
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip('"\'')
                os.environ[key] = value
    return True


def main():
    # Load environment
    load_env_from_file('.env.example')
    load_env_from_file('.env')
    load_env_from_file('local_llama_server/.env')
    
    print("Starting Local LLaMA MCP Server in HTTP mode...")
    print("Port: 8004")
    print("URL: http://127.0.0.1:8004")
    print("Ollama Model:", os.getenv("OLLAMA_MODEL", "tinyllama"))
    print("Press Ctrl+C to stop")
    print("-" * 40)
    
    # Import and run the server in HTTP mode
    sys.path.append(str(Path(__file__).parent))
    
    from local_llama_server.server import main as server_main
    import asyncio
    
    asyncio.run(server_main(transport="http", host="127.0.0.1", port=8004))


if __name__ == "__main__":
    main()
