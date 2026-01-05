# OpenAI MCP Server

Provides OpenAI GPT chat and embedding tools for text generation and semantic search through the Model Context Protocol (MCP).

## Overview

This MCP server exposes OpenAI's powerful language models and embedding capabilities as standardized tools. It enables seamless integration with OpenAI's API while providing a consistent interface for chat completions, text embeddings, and model management.

## Architecture

```
openai_server/
├── server.py               # FastMCP server entry point
├── config.py              # Configuration and environment management
├── tools/                 # MCP tool implementations
│   ├── chat.py            # Chat completion tool
│   ├── embeddings.py      # Text embedding tool
│   └── models.py          # Model listing tool
├── resources/             # MCP resource endpoints
│   └── models.py          # Models resource
├── requirements.txt       # Python dependencies
└── README.md             # This documentation
```

### Components

- **Server**: FastMCP-based server that exposes tools and resources
- **Tools**: Individual MCP tools for specific OpenAI capabilities
- **Resources**: Static data endpoints for model information
- **Config**: Environment-based configuration management

## Setup Instructions

### 1. Install Dependencies

```bash
cd openai_server
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
OPENAI_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
MAX_TOKENS=2048
TEMPERATURE=0.7
TIMEOUT=30
```

### 3. Run the Server

```bash
python server.py
```

The server will start on `localhost:8001` by default.

## Tool Reference

### `openai_chat`

Generate chat completions using OpenAI GPT models.

**Parameters:**
- `messages` (List[Dict]): List of message objects with 'role' and 'content'
- `model` (Optional[str]): OpenAI model to use (defaults to config)
- `max_tokens` (Optional[int]): Maximum tokens in response
- `temperature` (Optional[float]): Sampling temperature 0-2
- `system_prompt` (Optional[str]): System message to prepend

**Returns:**
- `message`: Response message with role and content
- `usage`: Token usage statistics
- `model`: Model used for generation
- `finish_reason`: Reason completion finished

**Example:**
```python
messages = [{"role": "user", "content": "Explain quantum computing"}]
response = await openai_chat(messages, temperature=0.3)
```

### `openai_embed`

Generate text embeddings using OpenAI's embedding models.

**Parameters:**
- `input_text` (Union[str, List[str]]): Text(s) to embed
- `model` (Optional[str]): Embedding model to use
- `encoding_format` (str): Format for embeddings ("float" or "base64")

**Returns:**
- `embeddings`: Generated embeddings (single array or list of arrays)
- `model`: Model used for embedding
- `usage`: Token usage statistics
- `dimensions`: Embedding vector dimensions

**Example:**
```python
# Single text
result = await openai_embed("Hello world")

# Multiple texts
texts = ["Hello", "World", "AI"]
result = await openai_embed(texts)
```

### `list_models`

List available OpenAI models with capabilities and metadata.

**Parameters:**
- `model_type` (Optional[str]): Filter by type ("chat", "embedding", "completion")
- `include_deprecated` (bool): Include deprecated models

**Returns:**
- `models`: List of model objects with details
- `total_count`: Number of models returned
- `filter_applied`: Applied filters

**Example:**
```python
# List all models
models = await list_models()

# List only chat models
chat_models = await list_models(model_type="chat")
```

## Resources

### `models`

Provides comprehensive model information organized by capability.

**Structure:**
- `categories`: Models grouped by capability (chat, embeddings, etc.)
- `configuration`: Current server configuration
- `total_models`: Total number of available models

## Example Usage

### Basic Chat Completion

```python
import asyncio
from openai_server.tools.chat import openai_chat

async def example_chat():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
    
    response = await openai_chat(messages, temperature=0.7)
    print(response["message"]["content"])

asyncio.run(example_chat())
```

### Generating Embeddings

```python
import asyncio
from openai_server.tools.embeddings import openai_embed

async def example_embeddings():
    texts = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing handles text"
    ]
    
    result = await openai_embed(texts)
    embeddings = result["embeddings"]
    print(f"Generated {len(embeddings)} embeddings with {result['dimensions']} dimensions")

asyncio.run(example_embeddings())
```

### Running the Server

```python
# Start the server
python server.py

# The server exposes:
# - Tools: openai_chat, openai_embed, list_models
# - Resources: models
# - Health check: health_check
```

## Extending the Server

### Adding New Tools

1. Create a new file in `tools/` directory
2. Implement the tool using `@tool()` decorator
3. Register the tool in `server.py`

```python
# tools/new_tool.py
from fastmcp import tool

@tool()
async def new_openai_tool(param: str) -> dict:
    """Your new OpenAI tool implementation."""
    # Implementation here
    pass

# server.py
from tools.new_tool import new_openai_tool
app.add_tool(new_openai_tool)
```

### Adding New Resources

1. Create a new file in `resources/` directory
2. Implement the resource using `@resource()` decorator
3. Register the resource in `server.py`

```python
# resources/new_resource.py
from fastmcp import resource

@resource()
async def new_resource() -> dict:
    """Your new resource implementation."""
    # Implementation here
    pass

# server.py
from resources.new_resource import new_resource
app.add_resource(new_resource)
```

### Configuration Options

Add new configuration options in `config.py`:

```python
class Config:
    def __init__(self):
        # Existing config...
        self.NEW_SETTING = os.getenv("NEW_SETTING", "default_value")
```

## Error Handling

All tools include comprehensive error handling:

- **API Errors**: OpenAI API failures are caught and returned as error responses
- **Configuration Errors**: Missing API keys raise clear error messages
- **Validation Errors**: Invalid parameters are validated before API calls
- **Timeout Handling**: Configurable timeouts prevent hanging requests

## Security Considerations

- **API Keys**: Stored in environment variables, never hardcoded
- **Input Validation**: All user inputs are validated before processing
- **Error Messages**: Sensitive information is not exposed in error responses
- **Rate Limiting**: Respects OpenAI's rate limits and usage policies

## Troubleshooting

### Common Issues

1. **Missing API Key**
   ```
   Error: Required environment variable OPENAI_API_KEY is not set
   ```
   Solution: Set your OpenAI API key in the `.env` file

2. **Model Not Found**
   ```
   Error: The model 'gpt-5' does not exist
   ```
   Solution: Use `list_models` tool to see available models

3. **Rate Limit Exceeded**
   ```
   Error: Rate limit exceeded
   ```
   Solution: Implement exponential backoff or upgrade your OpenAI plan

### Debug Mode

Enable debug logging by setting:
```bash
LOG_LEVEL=DEBUG
```

## License

MIT License - See the main project LICENSE file for details.
