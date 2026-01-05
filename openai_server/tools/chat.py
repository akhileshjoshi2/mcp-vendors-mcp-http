"""
OpenAI Chat Completion Tool

Provides chat completion functionality using OpenAI's GPT models.
Supports various parameters for customizing the chat experience.
"""

import logging
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from fastmcp import tool
from ..config import Config

logger = logging.getLogger(__name__)

@tool()
async def openai_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate chat completion using OpenAI GPT models.
    
    Args:
        messages: List of message objects with 'role' and 'content' keys
        model: OpenAI model to use (defaults to config model)
        max_tokens: Maximum tokens in response (defaults to config)
        temperature: Sampling temperature 0-2 (defaults to config)
        system_prompt: Optional system message to prepend
        
    Returns:
        Dict containing the response message, usage stats, and metadata
        
    Example:
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        response = await openai_chat(messages)
    """
    try:
        config = Config()
        client = AsyncOpenAI(**config.get_openai_config())
        
        # Use provided parameters or fall back to config defaults
        chat_config = config.get_chat_config()
        model = model or chat_config["model"]
        max_tokens = max_tokens or chat_config["max_tokens"]
        temperature = temperature or chat_config["temperature"]
        
        # Prepare messages
        formatted_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
            
        # Add user messages
        formatted_messages.extend(messages)
        
        logger.info(f"Sending chat completion request to model: {model}")
        
        # Make API call
        response = await client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Extract response data
        message = response.choices[0].message
        usage = response.usage
        
        result = {
            "message": {
                "role": message.role,
                "content": message.content
            },
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            },
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason
        }
        
        logger.info(f"Chat completion successful. Tokens used: {usage.total_tokens}")
        return result
        
    except Exception as e:
        logger.error(f"OpenAI chat completion failed: {e}")
        return {
            "error": str(e),
            "message": None,
            "usage": None
        }
