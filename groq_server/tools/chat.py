"""
Groq Chat Completion Tool

Provides high-speed chat completion functionality using Groq's accelerated inference.
Optimized for fast response times with various open-source models.
"""

import logging
from typing import List, Dict, Any, Optional
import httpx
from fastmcp import tool
from ..config import Config

logger = logging.getLogger(__name__)

@tool()
async def groq_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    system_prompt: Optional[str] = None,
    stream: bool = False
) -> Dict[str, Any]:
    """
    Generate chat completion using Groq's accelerated inference.
    
    Args:
        messages: List of message objects with 'role' and 'content' keys
        model: Groq model to use (defaults to config model)
        max_tokens: Maximum tokens in response (defaults to config)
        temperature: Sampling temperature 0-2 (defaults to config)
        system_prompt: Optional system message to prepend
        stream: Whether to stream the response (not implemented yet)
        
    Returns:
        Dict containing the response message, usage stats, and metadata
        
    Example:
        messages = [{"role": "user", "content": "Explain machine learning"}]
        response = await groq_chat(messages, model="llama3-70b-8192")
    """
    try:
        config = Config()
        
        # Use provided parameters or fall back to config defaults
        chat_config = config.get_chat_config()
        model = model or chat_config["model"]
        max_tokens = max_tokens or chat_config["max_tokens"]
        temperature = temperature or chat_config["temperature"]
        
        # Validate model
        supported_models = config.get_supported_models()
        if model not in supported_models:
            logger.warning(f"Model {model} not in supported list: {supported_models}")
        
        # Prepare messages
        formatted_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
            
        # Add user messages
        formatted_messages.extend(messages)
        
        logger.info(f"Sending chat completion request to Groq model: {model}")
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        # Make API call using httpx
        groq_config = config.get_groq_config()
        headers = {
            "Authorization": f"Bearer {groq_config['api_key']}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=groq_config['timeout']) as client:
            response = await client.post(
                f"{groq_config['base_url']}/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
        
        # Extract response data
        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage", {})
        
        result = {
            "message": {
                "role": message["role"],
                "content": message["content"]
            },
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            },
            "model": data.get("model", model),
            "finish_reason": choice.get("finish_reason"),
            "provider": "groq",
            "response_time_ms": response.elapsed.total_seconds() * 1000
        }
        
        logger.info(f"Groq chat completion successful. Tokens used: {result['usage']['total_tokens']}")
        return result
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Groq API HTTP error: {e.response.status_code} - {e.response.text}")
        return {
            "error": f"HTTP {e.response.status_code}: {e.response.text}",
            "message": None,
            "usage": None
        }
    except Exception as e:
        logger.error(f"Groq chat completion failed: {e}")
        return {
            "error": str(e),
            "message": None,
            "usage": None
        }
