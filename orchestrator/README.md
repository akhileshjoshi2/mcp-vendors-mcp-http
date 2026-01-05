# MCP Orchestrator

Central router and connector layer that manages multiple MCP servers and provides unified access to all AI and utility services through a single interface.

## Overview

The MCP Orchestrator acts as the central hub for the entire MCP ecosystem, managing multiple vendor servers and providing intelligent routing, health monitoring, and unified access patterns. It abstracts the complexity of managing multiple MCP servers and provides a simple, consistent interface for applications.

## Architecture

```
orchestrator/
├── orchestrator.py         # Core orchestrator implementation
├── mcp_client.py          # High-level client interface
├── requirements.txt       # Python dependencies
└── README.md             # This documentation
```

### Components

- **Orchestrator**: Core server management and routing engine
- **MCP Client**: Simplified client interface for applications
- **Health Monitor**: Automatic server health checking and restart
- **Request Router**: Intelligent routing based on capabilities

## Features

### 🔄 Dynamic Server Management
- Automatic server discovery from configuration
- Health monitoring with auto-restart capabilities
- Graceful startup and shutdown sequences
- Environment validation and dependency checking

### 🎯 Intelligent Request Routing
- Automatic vendor selection based on capabilities
- Load balancing and failover support
- Request parameter adaptation per vendor
- Unified response formatting

### 📊 Comprehensive Monitoring
- Real-time server status tracking
- Performance metrics and health checks
- Capability discovery and mapping
- Detailed logging and error reporting

### 🔌 Simple Client Interface
- High-level convenience methods
- Automatic connection management
- Consistent response formats
- Error handling and retry logic

## Setup Instructions

### 1. Install Dependencies

```bash
cd orchestrator
pip install -r requirements.txt
```

### 2. Configure Servers

The orchestrator uses the configuration file at `config/mcp_servers.json`. This should already be configured if you've set up the individual servers.

### 3. Set Environment Variables

Ensure all required environment variables are set for the servers you want to use:

```bash
# OpenAI
OPENAI_API_KEY=your_key_here

# Groq  
GROQ_API_KEY=your_key_here

# Local LLaMA
MODEL_PATH=/path/to/model.gguf

# Utility Services
WEATHER_API_KEY=your_key_here
```

### 4. Run the Orchestrator

```bash
python orchestrator.py
```

## Usage Examples

### Using the High-Level Client

```python
import asyncio
from mcp_client import MCPClient

async def main():
    # Initialize client
    client = MCPClient()
    await client.connect()
    
    try:
        # Chat with auto vendor selection
        response = await client.chat("Explain quantum computing")
        print(f"Response: {response['message']}")
        
        # Chat with specific vendor
        response = await client.chat(
            "Write a Python function", 
            vendor="openai",
            model="gpt-4"
        )
        
        # Generate embeddings
        embeddings = await client.embed([
            "Document 1 content",
            "Document 2 content"
        ])
        
        # Search the web
        results = await client.search_web("latest AI news", num_results=5)
        for result in results["results"]:
            print(f"- {result['title']}: {result['url']}")
            
        # Get weather
        weather = await client.get_weather("New York")
        print(f"Temperature: {weather['temperature']}°C")
        
    finally:
        await client.disconnect()

asyncio.run(main())
```

### Using Convenience Functions

```python
import asyncio
from mcp_client import chat, embed, search, weather

async def quick_examples():
    # Quick chat
    response = await chat("What is machine learning?")
    print(response["message"])
    
    # Quick search
    results = await search("Python tutorials")
    print(f"Found {len(results['results'])} results")
    
    # Quick weather
    weather_data = await weather("London")
    print(f"London: {weather_data['temperature']}°C, {weather_data['description']}")

asyncio.run(quick_examples())
```

### Direct Orchestrator Usage

```python
import asyncio
from orchestrator import MCPOrchestrator

async def orchestrator_example():
    orch = MCPOrchestrator()
    await orch.start()
    
    try:
        # Discover capabilities
        capabilities = await orch.discover_capabilities()
        print(f"Available servers: {list(capabilities['servers'].keys())}")
        
        # Route specific requests
        response = await orch.route_request(
            vendor="openai",
            tool="openai_chat", 
            params={
                "messages": [{"role": "user", "content": "Hello!"}]
            }
        )
        
        # Get system status
        status = await orch.get_server_status()
        print(f"Running servers: {status['orchestrator']['running']}")
        
    finally:
        await orch.stop()

asyncio.run(orchestrator_example())
```

## Advanced Usage

### Custom Vendor Selection

```python
async def smart_vendor_selection():
    client = MCPClient()
    await client.connect()
    
    # Get capabilities to make informed decisions
    capabilities = await client.get_capabilities()
    
    # Choose vendor based on requirements
    if "openai" in capabilities["servers"] and capabilities["servers"]["openai"]["status"] == "running":
        vendor = "openai"  # Use for high quality
    elif "groq" in capabilities["servers"] and capabilities["servers"]["groq"]["status"] == "running":
        vendor = "groq"    # Use for speed
    else:
        vendor = "local_llama"  # Use for privacy
    
    response = await client.chat("Complex reasoning task", vendor=vendor)
    await client.disconnect()
```

### Batch Processing

```python
async def batch_processing():
    client = MCPClient()
    await client.connect()
    
    # Batch chat requests
    messages = [
        "Summarize machine learning",
        "Explain neural networks", 
        "What is deep learning?"
    ]
    
    tasks = [client.chat(msg, vendor="groq") for msg in messages]
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses):
        print(f"Q{i+1}: {messages[i]}")
        print(f"A{i+1}: {response['message']}\n")
    
    await client.disconnect()
```

### Error Handling and Fallbacks

```python
async def robust_chat(message: str):
    client = MCPClient()
    await client.connect()
    
    # Try vendors in order of preference
    vendors = ["openai", "groq", "local_llama"]
    
    for vendor in vendors:
        try:
            response = await client.chat(message, vendor=vendor)
            if not response.get("error"):
                print(f"Success with {vendor}: {response['message']}")
                break
        except Exception as e:
            print(f"Failed with {vendor}: {e}")
            continue
    else:
        print("All vendors failed")
    
    await client.disconnect()
```

### Health Monitoring

```python
async def monitor_system():
    client = MCPClient()
    await client.connect()
    
    while True:
        status = await client.get_status()
        
        print("\n=== System Status ===")
        for server_id, server_info in status["servers"].items():
            status_icon = "✅" if server_info["status"] == "running" else "❌"
            print(f"{status_icon} {server_info['name']}: {server_info['status']}")
            
        await asyncio.sleep(30)  # Check every 30 seconds
```

## Configuration Management

### Server Configuration

The orchestrator reads server configurations from `config/mcp_servers.json`:

```json
{
  "servers": {
    "openai": {
      "name": "OpenAI Server",
      "description": "OpenAI GPT and embedding services",
      "command": "python",
      "args": ["openai_server/server.py"],
      "port": 8001,
      "tools": ["openai_chat", "openai_embed", "list_models"],
      "required_env": ["OPENAI_API_KEY"]
    }
  },
  "orchestrator": {
    "port": 8000,
    "default_timeout": 30,
    "retry_attempts": 3
  }
}
```

### Environment Validation

The orchestrator automatically checks for required environment variables:

```python
# Check which servers can start
status = await orchestrator.get_server_status()
for server_id, server_info in status["servers"].items():
    env_status = server_info["required_env_status"]
    missing_vars = [var for var, present in env_status.items() if not present]
    if missing_vars:
        print(f"Server {server_id} missing: {missing_vars}")
```

## API Reference

### MCPOrchestrator Class

#### Methods

- `start()`: Start orchestrator and all servers
- `stop()`: Stop orchestrator and all servers  
- `route_request(vendor, tool, params)`: Route request to specific server
- `discover_capabilities()`: Get all available tools and resources
- `get_server_status()`: Get detailed server status information

### MCPClient Class

#### Methods

- `connect()`: Connect to orchestrator
- `disconnect()`: Disconnect from orchestrator
- `chat(message, vendor, **kwargs)`: Send chat message
- `embed(text, vendor, **kwargs)`: Generate embeddings
- `search_web(query, **kwargs)`: Search the web
- `get_weather(location, **kwargs)`: Get weather data
- `list_models(vendor)`: List available models
- `get_capabilities()`: Get system capabilities
- `get_status()`: Get system status

## Performance Optimization

### Connection Pooling

```python
# Reuse client connections
class MCPService:
    def __init__(self):
        self.client = MCPClient()
        self._connected = False
    
    async def ensure_connected(self):
        if not self._connected:
            await self.client.connect()
            self._connected = True
    
    async def chat(self, message: str):
        await self.ensure_connected()
        return await self.client.chat(message)
```

### Caching Responses

```python
import time
from typing import Dict, Any

class CachedMCPClient:
    def __init__(self):
        self.client = MCPClient()
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    async def cached_embed(self, text: str):
        cache_key = f"embed_{hash(text)}"
        now = time.time()
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if now - cached_data["timestamp"] < self.cache_duration:
                return cached_data["data"]
        
        # Generate new embedding
        result = await self.client.embed(text)
        
        # Cache result
        self.cache[cache_key] = {
            "data": result,
            "timestamp": now
        }
        
        return result
```

## Troubleshooting

### Common Issues

1. **Servers Not Starting**
   ```
   Check environment variables and server logs
   Verify model paths and API keys
   Ensure ports are not in use
   ```

2. **Request Routing Failures**
   ```
   Check server status with get_status()
   Verify tool availability with get_capabilities()
   Check network connectivity
   ```

3. **Health Check Failures**
   ```
   Verify server processes are running
   Check port accessibility
   Review server logs for errors
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run orchestrator with debug logging
orchestrator = MCPOrchestrator()
await orchestrator.start()
```

### Manual Server Management

```python
# Start specific server
await orchestrator._start_server("openai", orchestrator.servers["openai"])

# Check server health
healthy = await orchestrator._check_server_health(orchestrator.servers["openai"])

# Restart server
await orchestrator._restart_server("openai", orchestrator.servers["openai"])
```

## Integration Patterns

### Web Application Integration

```python
from fastapi import FastAPI
from mcp_client import MCPClient

app = FastAPI()
mcp_client = MCPClient()

@app.on_event("startup")
async def startup():
    await mcp_client.connect()

@app.on_event("shutdown") 
async def shutdown():
    await mcp_client.disconnect()

@app.post("/chat")
async def chat_endpoint(message: str):
    response = await mcp_client.chat(message)
    return response
```

### CLI Tool Integration

```python
import click
from mcp_client import chat, search, weather

@click.group()
def cli():
    """MCP CLI Tool"""
    pass

@cli.command()
@click.argument("message")
def chat_cmd(message):
    """Send a chat message"""
    response = asyncio.run(chat(message))
    click.echo(response["message"])

@cli.command()
@click.argument("query")
def search_cmd(query):
    """Search the web"""
    results = asyncio.run(search(query))
    for result in results["results"]:
        click.echo(f"- {result['title']}: {result['url']}")

if __name__ == "__main__":
    cli()
```

## Security Considerations

- **API Key Management**: All keys stored in environment variables
- **Process Isolation**: Each server runs in separate process
- **Network Security**: Local-only communication by default
- **Input Validation**: All requests validated before routing
- **Error Sanitization**: Sensitive information filtered from responses

## License

MIT License - See the main project LICENSE file for details.
