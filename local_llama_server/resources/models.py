"""
Local LLaMA Models Resource

Provides information about local model capabilities and configurations.
This resource helps clients understand available local models and their specifications.
"""

import logging
import os
from typing import Dict, Any, List

import httpx

from ..config import Config

logger = logging.getLogger(__name__)


async def _get_ollama_models() -> Dict[str, Any]:
    """Query the local Ollama daemon for available models.

    Returns a dict with a list of models and any error information, but never
    raises so the main resource can still respond.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()

            data = resp.json()
            # Ollama /api/tags returns { "models": [ {"name": ..., "model": ..., ...}, ... ] }
            models: List[Dict[str, Any]] = data.get("models", [])

            # Normalize into a simpler structure
            normalized = [
                {
                    "name": m.get("name"),
                    "model": m.get("model"),
                    "modified_at": m.get("modified_at"),
                    "size": m.get("size"),
                }
                for m in models
            ]

            return {
                "base_url": base_url,
                "models": normalized,
                "error": None,
            }

    except httpx.HTTPError as e:
        logger.error("Failed to query Ollama models: %s", e)
        return {
            "base_url": base_url,
            "models": [],
            "error": f"HTTP error talking to Ollama: {e}",
        }
    except Exception as e:
        logger.error("Unexpected error while querying Ollama models: %s", e)
        return {
            "base_url": base_url,
            "models": [],
            "error": str(e),
        }


async def models_resource() -> Dict[str, Any]:
    """
    Provide local LLaMA models information as an MCP resource.
    
    Returns:
        Dict containing comprehensive local model information and capabilities
        
    This resource provides:
    - Local model configuration and paths
    - Hardware requirements and recommendations
    - Performance characteristics
    - Supported model formats and quantization levels
    """
    try:
        config = Config()
        
        logger.info("Loading local LLaMA models resource...")
        
        # Check if model path exists
        model_exists = os.path.exists(config.MODEL_PATH) if config.MODEL_PATH else False
        embedding_model_exists = os.path.exists(config.EMBEDDING_MODEL_PATH) if config.EMBEDDING_MODEL_PATH else False
        
        # Define model categories and specifications
        model_categories = {
            "text_generation": {
                "description": "Local LLaMA models for text generation and chat",
                "current_model": {
                    "path": config.MODEL_PATH,
                    "exists": model_exists,
                    "type": config.MODEL_TYPE,
                    "context_length": config.CONTEXT_LENGTH,
                    "device": config.DEVICE,
                    "gpu_layers": config.N_GPU_LAYERS,
                    "threads": config.N_THREADS
                },
                "supported_formats": config.get_supported_formats(),
                "recommended_use_cases": [
                    "Private conversations",
                    "Offline content generation",
                    "Code assistance without internet",
                    "Document analysis and summarization",
                    "Creative writing"
                ],
                "performance_notes": [
                    "CPU inference: Slower but works on any hardware",
                    "GPU acceleration: Requires CUDA-compatible GPU",
                    "Quantized models: Reduced memory usage with slight quality trade-off",
                    "Context length affects memory requirements"
                ]
            },
            "embeddings": {
                "description": "Local embedding models for semantic search and similarity",
                "current_model": {
                    "path": config.EMBEDDING_MODEL_PATH or config.MODEL_PATH,
                    "exists": embedding_model_exists or model_exists,
                    "device": config.DEVICE
                },
                "recommended_use_cases": [
                    "Private semantic search",
                    "Document clustering",
                    "Similarity analysis",
                    "Offline RAG applications"
                ],
                "performance_notes": [
                    "Embedding generation is typically faster than text generation",
                    "Batch processing improves throughput",
                    "Normalized embeddings recommended for similarity search"
                ]
            }
        }
        
        # Hardware requirements and recommendations
        hardware_requirements = {
            "minimum": {
                "ram": "8GB",
                "storage": "5GB for 7B model",
                "cpu": "4 cores",
                "notes": "For basic 7B quantized models"
            },
            "recommended": {
                "ram": "16GB+",
                "storage": "20GB+ for multiple models",
                "cpu": "8+ cores",
                "gpu": "8GB+ VRAM for GPU acceleration",
                "notes": "For optimal performance with larger models"
            },
            "optimal": {
                "ram": "32GB+",
                "storage": "50GB+ for model collection",
                "cpu": "16+ cores",
                "gpu": "24GB+ VRAM",
                "notes": "For running 70B+ models efficiently"
            }
        }
        
        # Quantization options
        quantization_info = {
            "q4_0": {
                "description": "4-bit quantization, good balance of size and quality",
                "size_reduction": "~75%",
                "quality_impact": "Minimal for most tasks"
            },
            "q5_0": {
                "description": "5-bit quantization, better quality than q4",
                "size_reduction": "~65%",
                "quality_impact": "Very minimal"
            },
            "q8_0": {
                "description": "8-bit quantization, high quality",
                "size_reduction": "~50%",
                "quality_impact": "Nearly imperceptible"
            },
            "f16": {
                "description": "16-bit floating point, original quality",
                "size_reduction": "~50% vs f32",
                "quality_impact": "None"
            },
            "f32": {
                "description": "32-bit floating point, maximum quality",
                "size_reduction": "0%",
                "quality_impact": "None (baseline)"
            }
        }
        
        # Discover local models via Ollama (if available)
        ollama_info = await _get_ollama_models()
        active_ollama_model = os.getenv("OLLAMA_MODEL")

        # Create resource data
        resource_data = {
            "provider": "Local LLaMA",
            "description": "Private, offline AI inference using local quantized models",
            "last_updated": "2024-01-01",
            "categories": model_categories,
            "configuration": {
                "model_path": config.MODEL_PATH,
                "model_type": config.MODEL_TYPE,
                "device": config.DEVICE,
                "context_length": config.CONTEXT_LENGTH,
                "max_tokens": config.MAX_TOKENS,
                "temperature": config.TEMPERATURE,
                "gpu_acceleration": config.is_gpu_available()
            },
            "hardware_requirements": hardware_requirements,
            "quantization_options": quantization_info,
            "features": {
                "offline_inference": True,
                "privacy_focused": True,
                "customizable_models": True,
                "gpu_acceleration": config.is_gpu_available(),
                "streaming_support": False,  # Could be implemented
                "batch_processing": True
            },
            "performance_characteristics": {
                "inference_speed": "Depends on hardware and model size",
                "memory_usage": "Varies by model and quantization",
                "startup_time": "Model loading required on first use",
                "scalability": "Limited by local hardware"
            },
            "local_models": {
                "ollama": {
                    "base_url": ollama_info["base_url"],
                    "available_models": ollama_info["models"],
                    "active_model": active_ollama_model,
                    "error": ollama_info["error"],
                }
            },
            "setup_requirements": [
                "Download compatible model files (GGUF/GGML format recommended)",
                "Configure MODEL_PATH environment variable",
                "Install llama-cpp-python for production use",
                "Optional: Configure GPU acceleration"
            ],
            "model_sources": [
                "Hugging Face Hub (GGUF models)",
                "TheBloke quantized models",
                "Official LLaMA model releases",
                "Custom fine-tuned models"
            ],
            "status": {
                "model_loaded": model_exists,
                "embedding_model_loaded": embedding_model_exists,
                "gpu_available": config.is_gpu_available(),
                "ready_for_inference": model_exists
            }
        }
        
        logger.info("Local LLaMA models resource loaded successfully")
        return resource_data
        
    except Exception as e:
        logger.error(f"Failed to load local models resource: {e}")
        return {
            "error": str(e),
            "provider": "Local LLaMA",
            "categories": {},
            "status": {
                "model_loaded": False,
                "ready_for_inference": False
            }
        }
