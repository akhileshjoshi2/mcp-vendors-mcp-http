# Groq MCP Server

Provides Groq-accelerated inference for ultra-fast LLM completions through the Model Context Protocol (MCP). Optimized for high-performance, low-latency AI applications.

## Overview

This MCP server exposes Groq's lightning-fast inference capabilities for open-source language models. Groq's custom silicon delivers 10x+ faster inference speeds compared to traditional GPU-based solutions, making it ideal for real-time applications and high-throughput scenarios.

## Architecture

```
groq_server/
├── server.py               # FastMCP server entry point
├── config.py              # Configuration and environment management
├── tools/                 # MCP tool implementations
│   ├── chat.py            # High-speed chat completion
│   └── embeddings.py      # Embedding generation (compatibility layer)
├── resources/             # MCP resource endpoints
│   └── models.py          # Groq models information
├── requirements.txt       # Python dependencies
└── README.md             # This documentation
```

### Components

- **Server**: FastMCP-based server optimized for Groq's inference speed
- **Tools**: High-performance tools leveraging Groq's acceleration
- **Resources**: Model information and performance characteristics
- **Config**: Environment-based configuration for Groq API

## Setup Instructions

### 1. Install Dependencies

```bash
cd groq_server
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (with defaults)
GROQ_MODEL=llama2-70b-4096
GROQ_BASE_URL=https://api.groq.com/openai/v1
MAX_TOKENS=4096
TEMPERATURE=0.7
TIMEOUT=30
```

### 3. Get Groq API Access

1. Visit [Groq Console](https://console.groq.com)
2. Sign up for an account
3. Generate an API key
4. Add the key to your `.env` file

### 4. Run the Server

```bash
python server.py
```

The server will start on `localhost:8002` by default.

## Tool Reference

### `groq_chat`

Ultra-fast chat completions using Groq's accelerated inference.

**Parameters:**
- `messages` (List[Dict]): List of message objects with 'role' and 'content'
- `model` (Optional[str]): Groq model to use (defaults to config)
- `max_tokens` (Optional[int]): Maximum tokens in response
- `temperature` (Optional[float]): Sampling temperature 0-2
- `system_prompt` (Optional[str]): System message to prepend
- `stream` (bool): Whether to stream response (future feature)

**Returns:**
- `message`: Response message with role and content
- `usage`: Token usage statistics
- `model`: Model used for generation
- `finish_reason`: Reason completion finished
- `provider`: "groq"
- `response_time_ms`: Response time in milliseconds

**Example:**
```python
messages = [{"role": "user", "content": "Explain quantum computing briefly"}]
response = await groq_chat(messages, model="llama3-70b-8192")
print(f"Response time: {response['response_time_ms']}ms")
```

### `groq_embed`

Text embedding generation with Groq-compatible methods.

**Note**: Groq specializes in inference rather than embeddings. This tool provides a compatibility layer for embedding functionality.

**Parameters:**
- `input_text` (Union[str, List[str]]): Text(s) to embed
- `model` (Optional[str]): Model identifier
- `encoding_format` (str): Format for embeddings ("float" or "base64")

**Returns:**
- `embeddings`: Generated embeddings
- `model`: Model used
- `usage`: Token usage statistics
- `dimensions`: Embedding vector dimensions
- `note`: Implementation details

**Example:**
```python
result = await groq_embed("Fast inference with Groq")
embeddings = result["embeddings"]
```

## Supported Models

### Available Models

| Model | Context Window | Parameters | Best For |
|-------|----------------|------------|----------|
| `llama2-70b-4096` | 4,096 | 70B | Complex reasoning, analysis |
| `mixtral-8x7b-32768` | 32,768 | 8x7B MoE | Long documents, multilingual |
| `gemma-7b-it` | 8,192 | 7B | Instruction following |
| `llama3-8b-8192` | 8,192 | 8B | General conversation |
| `llama3-70b-8192` | 8,192 | 70B | Advanced reasoning |

### Model Selection Guide

- **Speed Priority**: `llama3-8b-8192` - Fastest inference
- **Quality Priority**: `llama3-70b-8192` - Best reasoning
- **Long Context**: `mixtral-8x7b-32768` - 32K context window
- **Balanced**: `llama2-70b-4096` - Good speed/quality balance

## Performance Benefits

### Groq Advantage

- **10x+ Faster**: Compared to traditional GPU inference
- **Sub-second Latency**: Ultra-low response times
- **High Throughput**: Handle many concurrent requests
- **Cost Effective**: Efficient resource utilization

### Benchmarks

Typical response times (varies by model and prompt complexity):
- Simple queries: 50-200ms
- Complex reasoning: 200-500ms
- Long-form generation: 300-800ms

## Example Usage

### High-Speed Chat

```python
import asyncio
import time
from groq_server.tools.chat import groq_chat

async def speed_test():
    messages = [{"role": "user", "content": "Write a Python function to calculate fibonacci"}]
    
    start_time = time.time()
    response = await groq_chat(messages, model="llama3-8b-8192")
    end_time = time.time()
    
    print(f"Total time: {(end_time - start_time) * 1000:.2f}ms")
    print(f"Groq response time: {response['response_time_ms']:.2f}ms")
    print(response["message"]["content"])

asyncio.run(speed_test())
```

### Batch Processing

```python
import asyncio
from groq_server.tools.chat import groq_chat

async def batch_process():
    prompts = [
        "Summarize machine learning in one sentence",
        "What is quantum computing?",
        "Explain blockchain technology briefly"
    ]
    
    tasks = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        task = groq_chat(messages, model="llama3-8b-8192")
        tasks.append(task)
    
    # Process all requests concurrently
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses):
        print(f"Query {i+1}: {response['response_time_ms']:.2f}ms")

asyncio.run(batch_process())
```

### Model Comparison

```python
import asyncio
from groq_server.tools.chat import groq_chat

async def compare_models():
    prompt = "Explain the theory of relativity"
    messages = [{"role": "user", "content": prompt}]
    
    models = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
    
    for model in models:
        response = await groq_chat(messages, model=model)
        print(f"{model}: {response['response_time_ms']:.2f}ms")
        print(f"Quality preview: {response['message']['content'][:100]}...\n")

asyncio.run(compare_models())
```

## Extending the Server

### Adding New Tools

```python
# tools/new_tool.py
from fastmcp import tool
import httpx

@tool()
async def groq_custom_tool(param: str) -> dict:
    """Custom Groq tool implementation."""
    config = Config()
    # Use Groq's fast inference for custom logic
    pass

# Register in server.py
from tools.new_tool import groq_custom_tool
app.add_tool(groq_custom_tool)
```

### Performance Optimization

1. **Batch Requests**: Group multiple requests for better throughput
2. **Model Selection**: Choose appropriate model for speed/quality trade-off
3. **Caching**: Implement response caching for repeated queries
4. **Connection Pooling**: Reuse HTTP connections

## Error Handling

### Common Issues

1. **API Key Invalid**
   ```
   Error: HTTP 401: Invalid API key
   ```
   Solution: Check your Groq API key in `.env`

2. **Rate Limiting**
   ```
   Error: HTTP 429: Rate limit exceeded
   ```
   Solution: Implement exponential backoff or upgrade plan

3. **Model Not Available**
   ```
   Error: HTTP 404: Model not found
   ```
   Solution: Use `models` resource to check available models

### Debug Mode

Enable detailed logging:
```bash
LOG_LEVEL=DEBUG python server.py
```

## Integration Patterns

### Real-time Applications

```python
# WebSocket integration example
async def websocket_handler(websocket):
    async for message in websocket:
        messages = [{"role": "user", "content": message}]
        response = await groq_chat(messages, model="llama3-8b-8192")
        await websocket.send(response["message"]["content"])
```

### API Gateway

```python
# FastAPI integration
from fastapi import FastAPI
from groq_server.tools.chat import groq_chat

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(messages: list):
    return await groq_chat(messages)
```

## Security Considerations

- **API Key Protection**: Store in environment variables only
- **Input Validation**: Validate all user inputs before processing
- **Rate Limiting**: Implement client-side rate limiting
- **Error Sanitization**: Don't expose sensitive error details

## Troubleshooting

### Performance Issues

1. **Slow Responses**: Check network latency to Groq servers
2. **Timeouts**: Increase timeout values in config
3. **Memory Usage**: Monitor for memory leaks in long-running processes

### Connection Issues

1. **Network Errors**: Verify internet connectivity
2. **SSL Errors**: Update certificates if needed
3. **Proxy Issues**: Configure proxy settings if behind firewall

## License

MIT License - See the main project LICENSE file for details.

## Resources

- [Groq Documentation](https://docs.groq.com)
- [Groq Console](https://console.groq.com)
- [Model Performance Benchmarks](https://groq.com/benchmarks)
