"""
Configuration management for Groq MCP Server.

Loads environment variables and provides configuration settings
for Groq API integration.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for Groq MCP Server."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self.GROQ_API_KEY: str = self._get_required_env("GROQ_API_KEY")
        self.GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama2-70b-4096")
        self.MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))
        self.TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
        self.TIMEOUT: int = int(os.getenv("TIMEOUT", "30"))
        self.BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable or raise error."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def get_groq_config(self) -> dict:
        """Get Groq client configuration."""
        return {
            "api_key": self.GROQ_API_KEY,
            "base_url": self.BASE_URL,
            "timeout": self.TIMEOUT
        }
    
    def get_chat_config(self) -> dict:
        """Get default chat completion configuration."""
        return {
            "model": self.GROQ_MODEL,
            "max_tokens": self.MAX_TOKENS,
            "temperature": self.TEMPERATURE
        }
    
    def get_supported_models(self) -> list:
        """Get list of supported Groq models."""
        return [
            "llama2-70b-4096",
            "mixtral-8x7b-32768",
            "gemma-7b-it",
            "llama3-8b-8192",
            "llama3-70b-8192"
        ]
