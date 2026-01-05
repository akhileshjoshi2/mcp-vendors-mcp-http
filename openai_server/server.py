"""
OpenAI MCP Server

Provides OpenAI GPT chat and embedding tools for text generation and semantic search.
This server exposes OpenAI's capabilities through standardized MCP tools.
"""

import asyncio
import argparse
import logging
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Also try loading from .env.example if .env doesn't exist
def load_env_from_file(file_path):
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

# Load from .env.example as fallback
if not os.getenv("OPENAI_API_KEY"):
    if not load_env_from_file('.env.example'):
        load_env_from_file('../.env.example')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP application
app = FastMCP("OpenAI MCP Server")

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is required")
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

@app.tool()
async def openai_chat(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate chat completion using OpenAI GPT models.
    
    Args:
        messages: List of message objects with 'role' and 'content' keys
        model: OpenAI model to use (defaults to gpt-4)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0-2)
        system_prompt: Optional system prompt to prepend
    
    Returns:
        Dict containing the chat response and metadata
    """
    try:
        logger.info(f"openai_chat called with messages: {messages}")
        logger.info(f"messages type: {type(messages)}")
        
        # Use provided model or default
        model_name = model or OPENAI_MODEL
        
        # Prepare messages
        chat_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        
        # Add user messages
        chat_messages.extend(messages)
        
        # Prepare request parameters
        request_params = {
            "model": model_name,
            "messages": chat_messages
        }
        
        if max_tokens:
            request_params["max_tokens"] = max_tokens
        if temperature is not None:
            request_params["temperature"] = temperature
        
        # Make API call
        response = await client.chat.completions.create(**request_params)
        
        # Extract response
        message_content = response.choices[0].message.content
        
        return {
            "message": message_content,
            "model": model_name,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "finish_reason": response.choices[0].finish_reason
        }
        
    except Exception as e:
        logger.error(f"OpenAI chat error: {e}")
        return {
            "error": str(e),
            "model": model or OPENAI_MODEL
        }

@app.tool()
async def openai_embed(
    input_text: str | List[str],
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate embeddings using OpenAI's embedding models.
    
    Args:
        input_text: Text or list of texts to embed
        model: Embedding model to use (defaults to text-embedding-ada-002)
    
    Returns:
        Dict containing embeddings and metadata
    """
    try:
        # Use provided model or default
        model_name = model or OPENAI_EMBEDDING_MODEL
        
        # Ensure input is a list
        texts = input_text if isinstance(input_text, list) else [input_text]
        
        # Make API call
        response = await client.embeddings.create(
            model=model_name,
            input=texts
        )
        
        # Extract embeddings
        embeddings = [item.embedding for item in response.data]
        
        return {
            "embeddings": embeddings,
            "model": model_name,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "dimensions": len(embeddings[0]) if embeddings else 0
        }
        
    except Exception as e:
        logger.error(f"OpenAI embedding error: {e}")
        return {
            "error": str(e),
            "model": model or OPENAI_EMBEDDING_MODEL
        }

async def _fetch_models(model_type: Optional[str] = None) -> Dict[str, Any]:
    """Internal helper to fetch models from OpenAI API."""
    try:
        response = await client.models.list()
        
        models = []
        for model in response.data:
            if model_type:
                if model_type.lower() not in model.id.lower():
                    continue
            
            models.append({
                "id": model.id,
                "object": model.object,
                "created": model.created,
                "owned_by": model.owned_by
            })
        
        return {
            "models": models,
            "total": len(models),
            "filter": model_type
        }
        
    except Exception as e:
        logger.error(f"List models error: {e}")
        return {
            "error": str(e),
            "models": []
        }

@app.tool()
async def list_models(model_type: Optional[str] = None) -> Dict[str, Any]:
    """
    List available OpenAI models.
    
    Args:
        model_type: Filter by model type (e.g., 'gpt', 'embedding')
    
    Returns:
        Dict containing available models
    """
    return await _fetch_models(model_type)

@app.resource("openai://models")
async def models_resource() -> str:
    """
    Provide comprehensive information about available OpenAI models.
    
    Returns:
        JSON string with model information and capabilities
    """
    try:
        models_info = await _fetch_models()
        
        resource_data = {
            "description": "OpenAI Models Information",
            "models": models_info.get("models", []),
            "capabilities": {
                "chat": ["gpt-4", "gpt-3.5-turbo"],
                "embeddings": ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
                "features": {
                    "streaming": True,
                    "function_calling": True,
                    "vision": True
                }
            },
            "pricing": {
                "note": "Check OpenAI pricing page for current rates",
                "url": "https://openai.com/pricing"
            }
        }
        
        import json
        return json.dumps(resource_data, indent=2)
        
    except Exception as e:
        logger.error(f"Models resource error: {e}")
        return f'{{"error": "{str(e)}"}}'

@app.tool()
def health_check() -> str:
    """Health check endpoint for the OpenAI MCP server."""
    return "OpenAI MCP Server is running"

async def main(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8001):
    """Main entry point for the OpenAI MCP server."""
    try:
        logger.info("Starting OpenAI MCP Server...")
        logger.info(f"Using OpenAI model: {OPENAI_MODEL}")
        logger.info(f"Transport: {transport}")
        
        if transport == "http":
            logger.info(f"Running HTTP server on {host}:{port}")
            config = uvicorn.Config(app.http_app(), host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()
        else:
            # Run the FastMCP server over stdio
            await app.run_stdio_async()
        
    except Exception as e:
        logger.error(f"Failed to start OpenAI MCP Server: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio", help="Transport type")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="HTTP port (default: 8001)")
    args = parser.parse_args()
    
    asyncio.run(main(transport=args.transport, host=args.host, port=args.port))
