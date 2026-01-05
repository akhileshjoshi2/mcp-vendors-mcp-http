"""
Setup Script for MCP Vendors Ecosystem

This script helps set up the MCP ecosystem by:
- Installing dependencies for all servers
- Validating environment configuration
- Testing server connectivity
- Providing setup guidance
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple

def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")

def run_command(cmd: List[str], cwd: str = None) -> Tuple[bool, str, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def install_dependencies():
    """Install dependencies for all servers."""
    print_header("Installing Dependencies")
    
    servers = [
        "openai_server",
        "groq_server", 
        "local_llama_server",
        "utility_server",
        "orchestrator"
    ]
    
    for server in servers:
        print_section(f"Installing {server} dependencies")
        
        requirements_path = Path(server) / "requirements.txt"
        if requirements_path.exists():
            success, stdout, stderr = run_command([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
            ])
            
            if success:
                print(f"✅ {server} dependencies installed successfully")
            else:
                print(f"❌ Failed to install {server} dependencies:")
                print(f"   Error: {stderr}")
        else:
            print(f"⚠️  No requirements.txt found for {server}")

def check_environment():
    """Check environment variables and configuration."""
    print_header("Environment Configuration Check")
    
    # Required environment variables by server
    env_requirements = {
        "OpenAI Server": ["OPENAI_API_KEY"],
        "Groq Server": ["GROQ_API_KEY"],
        "Local LLaMA Server": ["MODEL_PATH"],
        "Utility Server": ["WEATHER_API_KEY"]
    }
    
    # Optional environment variables
    optional_env = {
        "OpenAI": ["OPENAI_MODEL", "OPENAI_EMBEDDING_MODEL"],
        "Groq": ["GROQ_MODEL", "GROQ_BASE_URL"],
        "Local LLaMA": ["MODEL_TYPE", "DEVICE", "N_THREADS"],
        "Utility": ["WEATHER_BASE_URL", "SEARCH_RESULTS_LIMIT"]
    }
    
    print_section("Required Environment Variables")
    
    all_good = True
    for server, vars_list in env_requirements.items():
        print(f"\n{server}:")
        for var in vars_list:
            value = os.getenv(var)
            if value:
                # Mask API keys for security
                if "API_KEY" in var:
                    display_value = f"{value[:8]}..." if len(value) > 8 else "***"
                else:
                    display_value = value
                print(f"  ✅ {var} = {display_value}")
            else:
                print(f"  ❌ {var} = NOT SET")
                all_good = False
    
    print_section("Optional Environment Variables")
    
    for category, vars_list in optional_env.items():
        print(f"\n{category}:")
        for var in vars_list:
            value = os.getenv(var)
            if value:
                print(f"  ✅ {var} = {value}")
            else:
                print(f"  ⚪ {var} = using default")
    
    return all_good

def create_env_template():
    """Create a .env template file."""
    print_section("Creating .env Template")
    
    env_template = """# MCP Vendors Ecosystem Environment Configuration

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# Groq Configuration  
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama2-70b-4096
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Local LLaMA Configuration
MODEL_PATH=/path/to/your/local/model.gguf
MODEL_TYPE=llama
DEVICE=cpu
MAX_TOKENS=2048
TEMPERATURE=0.7
CONTEXT_LENGTH=4096
N_THREADS=4
N_GPU_LAYERS=0

# For GPU acceleration (uncomment and adjust):
# DEVICE=cuda
# N_GPU_LAYERS=35

# Utility Services Configuration
WEATHER_API_KEY=your_weather_api_key_here
WEATHER_BASE_URL=https://api.openweathermap.org/data/2.5
WEATHER_UNITS=metric

# Search Configuration
SEARCH_ENGINE=duckduckgo
SEARCH_RESULTS_LIMIT=10
SEARCH_TIMEOUT=10

# General Configuration
TIMEOUT=30
LOG_LEVEL=INFO
"""
    
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️  .env file already exists. Creating .env.template instead.")
        env_file = Path(".env.template")
    
    with open(env_file, "w") as f:
        f.write(env_template)
    
    print(f"✅ Created {env_file}")
    print(f"   Please edit this file with your actual API keys and configuration.")

def validate_config():
    """Validate the MCP servers configuration file."""
    print_section("Validating Configuration")
    
    config_path = Path("config/mcp_servers.json")
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        
        print("✅ Configuration file is valid JSON")
        
        # Check required sections
        if "servers" not in config:
            print("❌ Missing 'servers' section in configuration")
            return False
        
        servers = config["servers"]
        print(f"✅ Found {len(servers)} server configurations")
        
        # Validate each server config
        required_fields = ["name", "command", "args", "port"]
        for server_id, server_config in servers.items():
            print(f"\nValidating {server_id}:")
            
            for field in required_fields:
                if field in server_config:
                    print(f"  ✅ {field}: {server_config[field]}")
                else:
                    print(f"  ❌ Missing required field: {field}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in configuration file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False

def check_python_version():
    """Check Python version compatibility."""
    print_section("Python Version Check")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def check_server_files():
    """Check if all required server files exist."""
    print_section("Server Files Check")
    
    servers = {
        "openai_server": ["server.py", "config.py", "tools/chat.py", "tools/embeddings.py"],
        "groq_server": ["server.py", "config.py", "tools/chat.py", "tools/embeddings.py"],
        "local_llama_server": ["server.py", "config.py", "tools/inference.py", "tools/embeddings.py"],
        "utility_server": ["server.py", "config.py", "tools/web_search.py", "tools/weather.py"],
        "orchestrator": ["orchestrator.py", "mcp_client.py"]
    }
    
    all_good = True
    for server, files in servers.items():
        print(f"\n{server}:")
        
        server_path = Path(server)
        if not server_path.exists():
            print(f"  ❌ Server directory not found")
            all_good = False
            continue
        
        for file in files:
            file_path = server_path / file
            if file_path.exists():
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} - missing")
                all_good = False
    
    return all_good

async def test_imports():
    """Test if required Python packages can be imported."""
    print_section("Package Import Test")
    
    packages = {
        "fastmcp": "FastMCP framework",
        "httpx": "HTTP client",
        "asyncio": "Async support", 
        "json": "JSON support",
        "os": "OS interface",
        "pathlib": "Path handling"
    }
    
    optional_packages = {
        "openai": "OpenAI API client",
        "beautifulsoup4": "HTML parsing",
        "numpy": "Numerical computing",
        "dotenv": "Environment file loading"
    }
    
    print("Required packages:")
    all_good = True
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package} - {description}")
        except ImportError:
            print(f"  ❌ {package} - {description} (MISSING)")
            all_good = False
    
    print("\nOptional packages:")
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package} - {description}")
        except ImportError:
            print(f"  ⚠️  {package} - {description} (optional, install if needed)")
    
    return all_good

def provide_setup_guidance():
    """Provide setup guidance based on findings."""
    print_header("Setup Guidance")
    
    print("""
Next Steps:

1. 📝 Configure Environment Variables
   - Copy .env.template to .env
   - Add your API keys:
     * OpenAI API key from https://platform.openai.com/api-keys
     * Groq API key from https://console.groq.com
     * Weather API key from https://openweathermap.org/api
   - Set MODEL_PATH for local LLaMA models

2. 📥 Download Models (for Local LLaMA)
   - Download GGUF format models from Hugging Face
   - Recommended: TheBloke quantized models
   - Example: llama-2-7b-chat.Q4_0.gguf

3. 🚀 Start the System
   - Run: python orchestrator/orchestrator.py
   - Or use individual servers: python openai_server/server.py

4. 🧪 Test the Setup
   - Run: python examples/basic_usage.py
   - Check server status in orchestrator logs

5. 📚 Read Documentation
   - Each server has detailed README.md
   - Check examples/ directory for usage patterns
""")

async def main():
    """Main setup function."""
    print_header("MCP Vendors Ecosystem Setup")
    
    # Check Python version first
    if not check_python_version():
        print("\n❌ Setup cannot continue with incompatible Python version")
        return
    
    # Check if files exist
    if not check_server_files():
        print("\n❌ Some required files are missing. Please ensure all server files are present.")
        return
    
    # Install dependencies
    install_dependencies()
    
    # Test imports
    await test_imports()
    
    # Create .env template if needed
    if not Path(".env").exists():
        create_env_template()
    
    # Check environment
    env_ok = check_environment()
    
    # Validate configuration
    config_ok = validate_config()
    
    # Provide guidance
    provide_setup_guidance()
    
    # Summary
    print_header("Setup Summary")
    
    if env_ok and config_ok:
        print("✅ Setup appears to be complete!")
        print("   You can now start the orchestrator and begin using the system.")
    else:
        print("⚠️  Setup needs attention:")
        if not env_ok:
            print("   - Configure missing environment variables")
        if not config_ok:
            print("   - Fix configuration file issues")
    
    print("\n🎉 Welcome to the MCP Vendors Ecosystem!")

if __name__ == "__main__":
    asyncio.run(main())
