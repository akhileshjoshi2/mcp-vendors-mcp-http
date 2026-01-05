"""
Local LLaMA MCP Server

Provides local quantized LLaMA model inference for offline AI capabilities.
This server enables private, on-device AI without external API dependencies.
"""

import asyncio
import argparse
import logging
from fastmcp import FastMCP
import uvicorn
from .config import Config
from .tools.inference import local_inference
from .tools.embeddings import local_embed
from .resources.models import models_resource

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP application
app = FastMCP("Local LLaMA MCP Server")


@app.tool()
async def local_llama_inference(
    prompt: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    system_prompt: str | None = None,
    chat_format: bool = True,
) -> dict:
    """Wrapper tool that delegates to the local_inference implementation.

    This keeps the real logic in tools/inference.py while exposing a FastMCP
    tool with a stable name for MCP Inspector.
    """

    return await local_inference(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        system_prompt=system_prompt,
        chat_format=chat_format,
    )


@app.tool()
async def local_llama_embed(
    input_text,
    model_path: str | None = None,
    normalize: bool = True,
    pooling_method: str = "mean",
) -> dict:
    """Wrapper tool that delegates to the local_embed implementation."""

    return await local_embed(
        input_text=input_text,
        model_path=model_path,
        normalize=normalize,
        pooling_method=pooling_method,
    )


# Register resources using explicit URI (function first, URI second)
app.add_resource_fn(models_resource, "local://models")

@app.tool()
def health_check() -> str:
    """Health check endpoint for the Local LLaMA MCP server."""
    return "Local LLaMA MCP Server is running"

async def main(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8004):
    """Main entry point for the Local LLaMA MCP server."""
    try:
        config = Config()
        logger.info("Starting Local LLaMA MCP Server...")
        logger.info(f"Model path: {config.MODEL_PATH}")
        logger.info(f"Device: {config.DEVICE}")
        logger.info(f"Transport: {transport}")
        
        # Initialize model (this would load the actual model in production)
        logger.info("Model initialization completed")
        
        if transport == "http":
            logger.info(f"Running HTTP server on {host}:{port}")
            config_uvicorn = uvicorn.Config(app.http_app(), host=host, port=port, log_level="info")
            server = uvicorn.Server(config_uvicorn)
            await server.serve()
        else:
            # Run the FastMCP server over stdio (no nested event loop)
            await app.run_stdio_async()
    except Exception as e:
        logger.error(f"Failed to start Local LLaMA MCP Server: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local LLaMA MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio", help="Transport type")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8004, help="HTTP port (default: 8004)")
    args = parser.parse_args()
    
    asyncio.run(main(transport=args.transport, host=args.host, port=args.port))
