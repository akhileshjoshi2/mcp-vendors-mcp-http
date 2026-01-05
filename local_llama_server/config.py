"""
Configuration management for Local LLaMA MCP Server.

Loads environment variables and provides configuration settings
for local model inference and management.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for Local LLaMA MCP Server."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self.MODEL_PATH: str = self._get_required_env("MODEL_PATH")
        self.MODEL_TYPE: str = os.getenv("MODEL_TYPE", "llama")
        self.DEVICE: str = os.getenv("DEVICE", "cpu")
        self.MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
        self.TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
        self.TOP_P: float = float(os.getenv("TOP_P", "0.9"))
        self.TOP_K: int = int(os.getenv("TOP_K", "40"))
        self.CONTEXT_LENGTH: int = int(os.getenv("CONTEXT_LENGTH", "4096"))
        self.N_THREADS: int = int(os.getenv("N_THREADS", "4"))
        self.N_GPU_LAYERS: int = int(os.getenv("N_GPU_LAYERS", "0"))
        self.EMBEDDING_MODEL_PATH: str = os.getenv("EMBEDDING_MODEL_PATH", "")
        
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable or raise error."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def get_model_config(self) -> dict:
        """Get model loading configuration."""
        return {
            "model_path": self.MODEL_PATH,
            "model_type": self.MODEL_TYPE,
            "device": self.DEVICE,
            "context_length": self.CONTEXT_LENGTH,
            "n_threads": self.N_THREADS,
            "n_gpu_layers": self.N_GPU_LAYERS
        }
    
    def get_generation_config(self) -> dict:
        """Get text generation configuration."""
        return {
            "max_tokens": self.MAX_TOKENS,
            "temperature": self.TEMPERATURE,
            "top_p": self.TOP_P,
            "top_k": self.TOP_K
        }
    
    def get_embedding_config(self) -> dict:
        """Get embedding configuration."""
        return {
            "model_path": self.EMBEDDING_MODEL_PATH or self.MODEL_PATH,
            "device": self.DEVICE
        }
    
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is configured."""
        return self.DEVICE.lower() in ["cuda", "gpu"] and self.N_GPU_LAYERS > 0
    
    def get_supported_formats(self) -> list:
        """Get list of supported model formats."""
        return [
            "gguf",  # llama.cpp format
            "ggml",  # Legacy llama.cpp format
            "bin",   # Hugging Face format
            "safetensors"  # Safe tensors format
        ]
