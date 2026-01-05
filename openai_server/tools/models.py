"""
OpenAI Models Tool

Provides functionality to list and inspect available OpenAI models.
Useful for discovering model capabilities and selecting appropriate models.
"""

import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from fastmcp import tool
from ..config import Config

logger = logging.getLogger(__name__)

@tool()
async def list_models(
    model_type: Optional[str] = None,
    include_deprecated: bool = False
) -> Dict[str, Any]:
    """
    List available OpenAI models with their capabilities and metadata.
    
    Args:
        model_type: Filter by model type ("chat", "embedding", "completion", etc.)
        include_deprecated: Whether to include deprecated models
        
    Returns:
        Dict containing list of models with their details and capabilities
        
    Example:
        # List all models
        models = await list_models()
        
        # List only chat models
        chat_models = await list_models(model_type="chat")
    """
    try:
        config = Config()
        client = AsyncOpenAI(**config.get_openai_config())
        
        logger.info("Fetching available OpenAI models...")
        
        # Get models from OpenAI API
        response = await client.models.list()
        models = response.data
        
        # Filter models based on criteria
        filtered_models = []
        
        for model in models:
            # Skip deprecated models if not requested
            if not include_deprecated and hasattr(model, 'deprecated') and model.deprecated:
                continue
                
            model_info = {
                "id": model.id,
                "object": model.object,
                "created": model.created,
                "owned_by": model.owned_by
            }
            
            # Add additional metadata if available
            if hasattr(model, 'permission'):
                model_info["permission"] = model.permission
                
            # Categorize model type based on ID patterns
            model_id = model.id.lower()
            if model_type:
                if model_type == "chat" and not any(x in model_id for x in ["gpt", "chat"]):
                    continue
                elif model_type == "embedding" and "embedding" not in model_id:
                    continue
                elif model_type == "completion" and any(x in model_id for x in ["gpt", "chat", "embedding"]):
                    continue
                    
            # Add inferred capabilities
            capabilities = []
            if any(x in model_id for x in ["gpt", "chat"]):
                capabilities.append("chat_completion")
            if "embedding" in model_id:
                capabilities.append("embeddings")
            if any(x in model_id for x in ["davinci", "curie", "babbage", "ada"]) and "embedding" not in model_id:
                capabilities.append("text_completion")
                
            model_info["capabilities"] = capabilities
            filtered_models.append(model_info)
            
        # Sort models by creation date (newest first)
        filtered_models.sort(key=lambda x: x.get("created", 0), reverse=True)
        
        result = {
            "models": filtered_models,
            "total_count": len(filtered_models),
            "filter_applied": {
                "model_type": model_type,
                "include_deprecated": include_deprecated
            }
        }
        
        logger.info(f"Retrieved {len(filtered_models)} models")
        return result
        
    except Exception as e:
        logger.error(f"Failed to list OpenAI models: {e}")
        return {
            "error": str(e),
            "models": [],
            "total_count": 0
        }
