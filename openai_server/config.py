"""
Configuration management for OpenAI MCP Server.

Loads environment variables and provides configuration settings
for OpenAI API integration.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for OpenAI MCP Server."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self.OPENAI_API_KEY: str = self._get_required_env("OPENAI_API_KEY")
        self.OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
        self.OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        self.MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
        self.TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
        self.TIMEOUT: int = int(os.getenv("TIMEOUT", "30"))
        
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable or raise error."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def get_openai_config(self) -> dict:
        """Get OpenAI client configuration."""
        return {
            "api_key": self.OPENAI_API_KEY,
            "timeout": self.TIMEOUT
        }
    
    def get_chat_config(self) -> dict:
        """Get default chat completion configuration."""
        return {
            "model": self.OPENAI_MODEL,
            "max_tokens": self.MAX_TOKENS,
            "temperature": self.TEMPERATURE
        }
    
    def get_embedding_config(self) -> dict:
        """Get default embedding configuration."""
        return {
            "model": self.OPENAI_EMBEDDING_MODEL
        }
