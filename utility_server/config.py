"""
Configuration management for Utility MCP Server.

Loads environment variables and provides configuration settings
for various utility services and APIs.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for Utility MCP Server."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # Weather API configuration
        self.WEATHER_API_KEY: str = self._get_required_env("WEATHER_API_KEY")
        self.WEATHER_BASE_URL: str = os.getenv("WEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5")
        self.WEATHER_UNITS: str = os.getenv("WEATHER_UNITS", "metric")  # metric, imperial, kelvin
        
        # Web search configuration (using DuckDuckGo - no API key required)
        self.SEARCH_ENGINE: str = os.getenv("SEARCH_ENGINE", "duckduckgo")
        self.SEARCH_RESULTS_LIMIT: int = int(os.getenv("SEARCH_RESULTS_LIMIT", "10"))
        self.SEARCH_TIMEOUT: int = int(os.getenv("SEARCH_TIMEOUT", "10"))
        
        # General configuration
        self.TIMEOUT: int = int(os.getenv("TIMEOUT", "30"))
        self.USER_AGENT: str = os.getenv("USER_AGENT", "MCP-Utility-Server/1.0")
        
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable or raise error."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def get_weather_config(self) -> dict:
        """Get weather API configuration."""
        return {
            "api_key": self.WEATHER_API_KEY,
            "base_url": self.WEATHER_BASE_URL,
            "units": self.WEATHER_UNITS,
            "timeout": self.TIMEOUT
        }
    
    def get_search_config(self) -> dict:
        """Get web search configuration."""
        return {
            "engine": self.SEARCH_ENGINE,
            "results_limit": self.SEARCH_RESULTS_LIMIT,
            "timeout": self.SEARCH_TIMEOUT,
            "user_agent": self.USER_AGENT
        }
    
    def get_http_config(self) -> dict:
        """Get general HTTP client configuration."""
        return {
            "timeout": self.TIMEOUT,
            "user_agent": self.USER_AGENT,
            "headers": {
                "User-Agent": self.USER_AGENT,
                "Accept": "application/json, text/html, */*",
                "Accept-Language": "en-US,en;q=0.9"
            }
        }
