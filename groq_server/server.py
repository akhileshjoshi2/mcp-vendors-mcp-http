"""
Groq MCP Server

Provides Groq-accelerated inference for fast LLM completions and embeddings.
This server exposes Groq's high-performance inference capabilities through MCP.
"""

import asyncio
import argparse
import logging
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP
from groq import AsyncGroq
import os
from dotenv import load_dotenv
import uvicorn
import httpx

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
if not os.getenv("GROQ_API_KEY"):
    if not load_env_from_file('.env.example'):
        load_env_from_file('../.env.example')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP application
app = FastMCP("Groq MCP Server")

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama2-70b-4096")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable is required")
    raise ValueError("GROQ_API_KEY environment variable is required")

@app.tool()
async def groq_chat(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate chat completion using Groq's accelerated inference.
    
    Args:
        messages: List of message objects with 'role' and 'content' keys
        model: Groq model to use (defaults to config model)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0-2)
        system_prompt: Optional system prompt to prepend
    
    Returns:
        Dict containing the chat response and metadata
    """
    try:
        logger.info(f"groq_chat called with messages: {messages}")
        
        # Use provided model or default
        model_name = model or GROQ_MODEL
        
        # Prepare messages
        chat_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        
        # Add user messages
        chat_messages.extend(messages)
        
        # Prepare request
        request_data = {
            "model": model_name,
            "messages": chat_messages,
            "max_tokens": max_tokens or MAX_TOKENS,
            "temperature": temperature if temperature is not None else TEMPERATURE
        }
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                json=request_data,
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "message": result["choices"][0]["message"]["content"],
                "model": model_name,
                "usage": result.get("usage", {}),
                "finish_reason": result["choices"][0].get("finish_reason"),
                "system_fingerprint": result.get("system_fingerprint")
            }
            
    except httpx.HTTPStatusError as e:
        logger.error(f"Groq API HTTP error: {e}")
        return {
            "error": f"Groq API error: {e.response.status_code}",
            "details": e.response.text if hasattr(e.response, 'text') else str(e)
        }
    except Exception as e:
        logger.error(f"Groq chat error: {e}")
        return {
            "error": str(e)
        }

@app.tool()
async def groq_embed(
    text: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate text embeddings using Groq-compatible embedding models.
    
    Args:
        text: Text to embed
        model: Embedding model to use
    
    Returns:
        Dict containing the embedding vector and metadata
    """
    try:
        # Note: Groq doesn't have native embedding models yet
        # This is a placeholder that could use alternative embedding services
        return {
            "error": "Groq embedding models not yet available",
            "text": text,
            "suggestion": "Use OpenAI embeddings or local embedding models"
        }
        
    except Exception as e:
        logger.error(f"Groq embed error: {e}")
        return {
            "error": str(e)
        }

def _get_models_info() -> Dict[str, Any]:
    """
    Helper function to get models information (not a tool).
    
    Returns:
        Dict containing available models and their capabilities
    """
    try:
        models = [
            {
                "id": "llama2-70b-4096",
                "name": "Llama 2 70B",
                "description": "Meta's Llama 2 70B model optimized for Groq",
                "context_length": 4096,
                "type": "chat"
            },
            {
                "id": "mixtral-8x7b-32768",
                "name": "Mixtral 8x7B",
                "description": "Mistral's Mixtral 8x7B model with 32K context",
                "context_length": 32768,
                "type": "chat"
            },
            {
                "id": "gemma-7b-it",
                "name": "Gemma 7B IT",
                "description": "Google's Gemma 7B instruction-tuned model",
                "context_length": 8192,
                "type": "chat"
            }
        ]
        
        return {
            "models": models,
            "total": len(models),
            "default_model": GROQ_MODEL
        }
        
    except Exception as e:
        logger.error(f"Get models info error: {e}")
        return {
            "error": str(e)
        }

@app.tool()
def list_models() -> Dict[str, Any]:
    """
    List available Groq models.
    
    Returns:
        Dict containing available models and their capabilities
    """
    return _get_models_info()

@app.resource("groq://models")
async def models_resource() -> str:
    """
    Resource providing comprehensive information about available Groq models.
    
    Returns:
        JSON string containing detailed model information
    """
    import json
    
    try:
        models_info = _get_models_info()
        
        detailed_info = {
            "provider": "Groq",
            "description": "High-performance LLM inference with Groq's custom silicon",
            "models": models_info.get("models", []),
            "capabilities": [
                "Ultra-fast inference",
                "Low latency responses", 
                "High throughput",
                "Open-source model support"
            ],
            "pricing": "Pay-per-token with competitive rates",
            "documentation": "https://console.groq.com/docs"
        }
        
        return json.dumps(detailed_info, indent=2)
        
    except Exception as e:
        logger.error(f"Models resource error: {e}")
        return json.dumps({"error": str(e)})

@app.tool()
def health_check() -> str:
    """Health check endpoint for the Groq MCP server."""
    return "Groq MCP Server is running"

async def main(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8002):
    """Main entry point for the Groq MCP server."""
    try:
        logger.info("Starting Groq MCP Server...")
        logger.info(f"Using Groq model: {GROQ_MODEL}")
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
        logger.error(f"Failed to start Groq MCP Server: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Groq MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio", help="Transport type")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8002, help="HTTP port (default: 8002)")
    args = parser.parse_args()
    
    asyncio.run(main(transport=args.transport, host=args.host, port=args.port))
