"""
Groq Models Resource

Provides information about available Groq models and their capabilities.
This resource helps clients understand Groq's model offerings and performance characteristics.
"""

import logging
from typing import Dict, Any
from fastmcp import resource
from ..config import Config

logger = logging.getLogger(__name__)

@resource()
async def models_resource() -> Dict[str, Any]:
    """
    Provide Groq models information as an MCP resource.
    
    Returns:
        Dict containing comprehensive Groq model information and capabilities
        
    This resource provides:
    - Available Groq models and their specifications
    - Performance characteristics and use cases
    - Context window sizes and capabilities
    - Recommended applications for each model
    """
    try:
        config = Config()
        
        logger.info("Loading Groq models resource...")
        
        # Define Groq model information
        groq_models = {
            "llama2-70b-4096": {
                "id": "llama2-70b-4096",
                "name": "LLaMA 2 70B",
                "provider": "Meta",
                "context_window": 4096,
                "parameters": "70B",
                "type": "chat",
                "capabilities": ["chat_completion", "text_generation"],
                "performance": "high",
                "use_cases": [
                    "Complex reasoning",
                    "Long-form content generation",
                    "Code assistance",
                    "Analysis and summarization"
                ],
                "strengths": [
                    "High quality responses",
                    "Good reasoning capabilities",
                    "Reliable performance"
                ]
            },
            "mixtral-8x7b-32768": {
                "id": "mixtral-8x7b-32768", 
                "name": "Mixtral 8x7B",
                "provider": "Mistral AI",
                "context_window": 32768,
                "parameters": "8x7B (MoE)",
                "type": "chat",
                "capabilities": ["chat_completion", "text_generation", "multilingual"],
                "performance": "very_high",
                "use_cases": [
                    "Long document processing",
                    "Multilingual tasks",
                    "Complex analysis",
                    "Code generation"
                ],
                "strengths": [
                    "Large context window",
                    "Mixture of Experts architecture",
                    "Multilingual support",
                    "Fast inference"
                ]
            },
            "gemma-7b-it": {
                "id": "gemma-7b-it",
                "name": "Gemma 7B Instruct",
                "provider": "Google",
                "context_window": 8192,
                "parameters": "7B",
                "type": "chat",
                "capabilities": ["chat_completion", "instruction_following"],
                "performance": "high",
                "use_cases": [
                    "Instruction following",
                    "Educational content",
                    "General chat",
                    "Task completion"
                ],
                "strengths": [
                    "Instruction tuned",
                    "Efficient inference",
                    "Good safety alignment"
                ]
            },
            "llama3-8b-8192": {
                "id": "llama3-8b-8192",
                "name": "LLaMA 3 8B",
                "provider": "Meta",
                "context_window": 8192,
                "parameters": "8B",
                "type": "chat",
                "capabilities": ["chat_completion", "text_generation"],
                "performance": "high",
                "use_cases": [
                    "General conversation",
                    "Content creation",
                    "Question answering",
                    "Text analysis"
                ],
                "strengths": [
                    "Latest LLaMA architecture",
                    "Balanced performance/efficiency",
                    "Good general capabilities"
                ]
            },
            "llama3-70b-8192": {
                "id": "llama3-70b-8192",
                "name": "LLaMA 3 70B",
                "provider": "Meta", 
                "context_window": 8192,
                "parameters": "70B",
                "type": "chat",
                "capabilities": ["chat_completion", "text_generation", "reasoning"],
                "performance": "very_high",
                "use_cases": [
                    "Complex reasoning tasks",
                    "Professional content generation",
                    "Advanced analysis",
                    "Research assistance"
                ],
                "strengths": [
                    "Excellent reasoning",
                    "High quality outputs",
                    "Latest architecture",
                    "Strong performance across domains"
                ]
            }
        }
        
        # Organize models by category
        chat_models = []
        for model_id, model_info in groq_models.items():
            if "chat_completion" in model_info["capabilities"]:
                chat_models.append(model_info)
        
        # Create resource data
        resource_data = {
            "provider": "Groq",
            "description": "High-performance inference for open-source LLMs",
            "last_updated": "2024-01-01",
            "categories": {
                "chat_completion": {
                    "description": "Models optimized for conversational AI with accelerated inference",
                    "models": chat_models,
                    "recommended_use_cases": [
                        "Real-time chat applications",
                        "High-throughput text generation",
                        "Interactive AI assistants",
                        "Content creation pipelines"
                    ],
                    "performance_benefits": [
                        "Ultra-fast inference speeds",
                        "Low latency responses",
                        "High throughput capabilities",
                        "Cost-effective scaling"
                    ]
                }
            },
            "configuration": {
                "default_model": config.GROQ_MODEL,
                "max_tokens": config.MAX_TOKENS,
                "default_temperature": config.TEMPERATURE,
                "base_url": config.BASE_URL
            },
            "features": {
                "accelerated_inference": True,
                "streaming_support": True,
                "batch_processing": True,
                "custom_models": False
            },
            "performance_characteristics": {
                "inference_speed": "Ultra-fast (10x+ faster than standard)",
                "latency": "Sub-second response times",
                "throughput": "High concurrent request handling",
                "scalability": "Horizontal scaling support"
            },
            "total_models": len(groq_models)
        }
        
        logger.info(f"Groq models resource loaded with {len(groq_models)} models")
        return resource_data
        
    except Exception as e:
        logger.error(f"Failed to load Groq models resource: {e}")
        return {
            "error": str(e),
            "provider": "Groq",
            "categories": {},
            "total_models": 0
        }
