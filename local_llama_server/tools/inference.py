"""
Local LLaMA Inference Tool

Provides local text generation using quantized LLaMA models.
Enables private, offline AI inference without external dependencies.
"""

import asyncio
import logging
import os
import time
from typing import List, Dict, Any, Optional

import httpx

from ..config import Config

logger = logging.getLogger(__name__)

# Global model instance / settings (resolved once)
_model_instance = None

async def local_inference(
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    system_prompt: Optional[str] = None,
    chat_format: bool = True
) -> Dict[str, Any]:
    """
    Generate text using local LLaMA model inference.
    
    Args:
        prompt: Input text prompt for generation
        max_tokens: Maximum tokens to generate (defaults to config)
        temperature: Sampling temperature 0-2 (defaults to config)
        top_p: Nucleus sampling parameter (defaults to config)
        top_k: Top-k sampling parameter (defaults to config)
        system_prompt: Optional system message for chat format
        chat_format: Whether to use chat format or raw completion
        
    Returns:
        Dict containing generated text, timing info, and metadata
        
    Example:
        response = await local_inference(
            "Explain quantum computing",
            max_tokens=500,
            temperature=0.7
        )
    """
    try:
        config = Config()
        
        # Use provided parameters or fall back to config defaults
        generation_config = config.get_generation_config()
        max_tokens = max_tokens or generation_config["max_tokens"]
        temperature = temperature or generation_config["temperature"]
        top_p = top_p or generation_config["top_p"]
        top_k = top_k or generation_config["top_k"]
        
        logger.info(f"Starting local inference for prompt length: {len(prompt)}")
        start_time = time.time()
        
        # Initialize model if not already loaded
        model = await _get_or_load_model(config)
        
        # Format prompt based on chat_format
        if chat_format:
            formatted_prompt = _format_chat_prompt(prompt, system_prompt)
        else:
            formatted_prompt = prompt
            
        logger.info(f"Formatted prompt length: {len(formatted_prompt)}")
        
        # Generate text using the model
        generated_text = await _generate_with_model(
            model,
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Calculate tokens (approximate)
        prompt_tokens = len(formatted_prompt.split())
        completion_tokens = len(generated_text.split())
        total_tokens = prompt_tokens + completion_tokens
        
        result = {
            "generated_text": generated_text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "model_info": {
                "model_path": config.MODEL_PATH,
                "model_type": config.MODEL_TYPE,
                "device": config.DEVICE
            },
            "generation_config": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            },
            "timing": {
                "inference_time_ms": inference_time,
                "tokens_per_second": completion_tokens / (inference_time / 1000) if inference_time > 0 else 0
            },
            "provider": "local_llama"
        }
        
        logger.info(f"Local inference completed in {inference_time:.2f}ms")
        logger.info(f"Generated {completion_tokens} tokens at {result['timing']['tokens_per_second']:.2f} tokens/sec")
        
        return result
        
    except Exception as e:
        logger.error(f"Local inference failed: {e}")
        return {
            "error": str(e),
            "generated_text": None,
            "usage": None
        }

async def _get_or_load_model(config: Config):
    """Get or load the model settings for Ollama-backed inference.

    We treat the "model" as a configuration pointing to an Ollama model
    running locally (e.g., tinyllama), rather than loading weights in-process.
    """
    global _model_instance

    if _model_instance is None:
        logger.info("Configuring local LLaMA inference via Ollama...")

        # Allow overriding via env vars; fall back to tinyllama
        ollama_model = os.getenv("OLLAMA_MODEL", "tinyllama")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

        _model_instance = {
            "provider": "ollama",
            "model": ollama_model,
            "base_url": base_url,
            "config": config.get_model_config(),
        }

        logger.info(
            "Using Ollama model '%s' at %s for local inference",
            ollama_model,
            base_url,
        )

    return _model_instance

async def _generate_with_model(
    model,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int
) -> str:
    """Generate text by calling a local Ollama model.

    This uses Ollama's HTTP API (default: http://localhost:11434) to
    perform actual local inference using the configured model.
    """

    base_url = model["base_url"]
    model_name = model["model"]

    # Build request payload for Ollama's /api/generate endpoint
    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        # Ollama's defaults are usually fine; we forward a few knobs
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            # Ollama uses num_predict instead of max_tokens
            "num_predict": max_tokens,
        },
    }

    logger.info("Calling Ollama model '%s' at %s", model_name, base_url)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()

            data = response.json()
            # For /api/generate with stream=false, the full response text is
            # available under the 'response' key.
            generated_text = data.get("response", "")

            if not generated_text:
                logger.warning("Ollama returned empty response; raw payload: %s", data)
                generated_text = "[No response from local model]"

            return generated_text

    except httpx.HTTPError as e:
        logger.error("HTTP error while calling Ollama: %s", e)
        return f"[Local LLaMA inference failed: HTTP error: {e}]"
    except Exception as e:
        logger.error("Unexpected error while calling Ollama: %s", e)
        return f"[Local LLaMA inference failed: {e}]"

def _format_chat_prompt(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Format prompt for chat-style interaction."""
    
    if system_prompt:
        formatted = f"System: {system_prompt}\nUser: {prompt}\nAssistant: "
    else:
        formatted = f"User: {prompt}\nAssistant: "
    
    return formatted
