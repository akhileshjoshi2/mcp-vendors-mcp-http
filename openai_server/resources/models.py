"""
OpenAI Models Resource

Provides model information as an MCP resource for discovery and metadata access.
This resource can be used by clients to understand available model capabilities.
"""

import logging
from typing import Dict, Any
from openai import AsyncOpenAI
from fastmcp import resource
from ..config import Config

logger = logging.getLogger(__name__)

@resource()
async def models_resource() -> Dict[str, Any]:
    """
    Provide OpenAI models information as an MCP resource.
    
    Returns:
        Dict containing comprehensive model information and capabilities
        
    This resource provides:
    - Available models and their IDs
    - Model capabilities (chat, embeddings, etc.)
    - Recommended use cases
    - Pricing tier information (if available)
    """
    try:
        config = Config()
        client = AsyncOpenAI(**config.get_openai_config())
        
        logger.info("Loading OpenAI models resource...")
        
        # Get models from OpenAI API
        response = await client.models.list()
        models = response.data
        
        # Organize models by capability
        chat_models = []
        embedding_models = []
        completion_models = []
        other_models = []
        
        for model in models:
            model_id = model.id.lower()
            model_info = {
                "id": model.id,
                "created": model.created,
                "owned_by": model.owned_by
            }
            
            # Categorize models
            if any(x in model_id for x in ["gpt-4", "gpt-3.5"]):
                chat_models.append(model_info)
            elif "embedding" in model_id:
                embedding_models.append(model_info)
            elif any(x in model_id for x in ["davinci", "curie", "babbage", "ada"]) and "embedding" not in model_id:
                completion_models.append(model_info)
            else:
                other_models.append(model_info)
        
        # Create resource data
        resource_data = {
            "provider": "OpenAI",
            "last_updated": "2024-01-01",  # This would be dynamically set
            "categories": {
                "chat_completion": {
                    "description": "Models optimized for conversational AI and chat applications",
                    "models": chat_models,
                    "recommended_use_cases": [
                        "Conversational AI",
                        "Content generation",
                        "Code assistance",
                        "Question answering"
                    ]
                },
                "embeddings": {
                    "description": "Models for generating text embeddings for semantic search",
                    "models": embedding_models,
                    "recommended_use_cases": [
                        "Semantic search",
                        "Text similarity",
                        "Clustering",
                        "Classification"
                    ]
                },
                "text_completion": {
                    "description": "Models for text completion and generation tasks",
                    "models": completion_models,
                    "recommended_use_cases": [
                        "Text completion",
                        "Creative writing",
                        "Code completion",
                        "Text transformation"
                    ]
                },
                "other": {
                    "description": "Specialized models for specific tasks",
                    "models": other_models,
                    "recommended_use_cases": [
                        "Specialized tasks",
                        "Fine-tuned applications"
                    ]
                }
            },
            "configuration": {
                "default_chat_model": config.OPENAI_MODEL,
                "default_embedding_model": config.OPENAI_EMBEDDING_MODEL,
                "max_tokens": config.MAX_TOKENS,
                "default_temperature": config.TEMPERATURE
            },
            "total_models": len(models)
        }
        
        logger.info(f"Models resource loaded with {len(models)} models")
        return resource_data
        
    except Exception as e:
        logger.error(f"Failed to load models resource: {e}")
        return {
            "error": str(e),
            "provider": "OpenAI",
            "categories": {},
            "total_models": 0
        }
