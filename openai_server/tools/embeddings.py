"""
OpenAI Embeddings Tool

Provides text embedding functionality using OpenAI's embedding models.
Useful for semantic search, clustering, and similarity comparisons.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from openai import AsyncOpenAI
from fastmcp import tool
from ..config import Config

logger = logging.getLogger(__name__)

@tool()
async def openai_embed(
    input_text: Union[str, List[str]],
    model: Optional[str] = None,
    encoding_format: str = "float"
) -> Dict[str, Any]:
    """
    Generate embeddings for text using OpenAI's embedding models.
    
    Args:
        input_text: Text string or list of strings to embed
        model: OpenAI embedding model to use (defaults to config model)
        encoding_format: Format for embeddings ("float" or "base64")
        
    Returns:
        Dict containing embeddings, usage stats, and metadata
        
    Example:
        # Single text
        result = await openai_embed("Hello world")
        
        # Multiple texts
        texts = ["Hello", "World", "OpenAI"]
        result = await openai_embed(texts)
    """
    try:
        config = Config()
        client = AsyncOpenAI(**config.get_openai_config())
        
        # Use provided model or fall back to config default
        embedding_config = config.get_embedding_config()
        model = model or embedding_config["model"]
        
        # Ensure input is a list
        if isinstance(input_text, str):
            input_texts = [input_text]
            single_input = True
        else:
            input_texts = input_text
            single_input = False
            
        logger.info(f"Generating embeddings for {len(input_texts)} text(s) using model: {model}")
        
        # Make API call
        response = await client.embeddings.create(
            model=model,
            input=input_texts,
            encoding_format=encoding_format
        )
        
        # Extract embeddings
        embeddings = [item.embedding for item in response.data]
        
        # If single input, return single embedding
        if single_input:
            embeddings = embeddings[0]
            
        result = {
            "embeddings": embeddings,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "encoding_format": encoding_format,
            "dimensions": len(response.data[0].embedding) if response.data else 0
        }
        
        logger.info(f"Embeddings generated successfully. Dimensions: {result['dimensions']}")
        return result
        
    except Exception as e:
        logger.error(f"OpenAI embedding generation failed: {e}")
        return {
            "error": str(e),
            "embeddings": None,
            "usage": None
        }
