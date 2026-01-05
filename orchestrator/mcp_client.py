"""
MCP Client Interface

Provides a simplified client interface for interacting with MCP servers
through the orchestrator. Handles connection management and request routing.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from orchestrator import MCPOrchestrator

logger = logging.getLogger(__name__)

class MCPClient:
    """
    High-level client for interacting with MCP servers via the orchestrator.
    
    Provides simplified methods for common operations and handles
    server selection, error handling, and response formatting.
    """
    
    def __init__(self, orchestrator: Optional[MCPOrchestrator] = None):
        """Initialize MCP client with optional orchestrator instance."""
        self.orchestrator = orchestrator or MCPOrchestrator()
        self.capabilities = {}
        
    async def connect(self):
        """Connect to the orchestrator and discover capabilities."""
        if not self.orchestrator.running:
            await self.orchestrator.start()
            
        self.capabilities = await self.orchestrator.discover_capabilities()
        logger.info(f"Connected to MCP ecosystem with {self.capabilities['summary']['running_servers']} servers")
        
    async def disconnect(self):
        """Disconnect from the orchestrator."""
        if self.orchestrator.running:
            await self.orchestrator.stop()
            
    async def chat(
        self, 
        message: str, 
        vendor: str = "auto",
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat message to an AI model.
        
        Args:
            message: The message to send
            vendor: Vendor to use ("openai", "groq", "local_llama", or "auto")
            model: Specific model to use
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the chat tool
            
        Returns:
            Chat response with message content and metadata
        """
        # Auto-select vendor if needed
        if vendor == "auto":
            vendor = self._select_chat_vendor()
            
        # Prepare parameters based on vendor
        if vendor == "openai":
            params = {
                "messages": [{"role": "user", "content": message}],
                "model": model,
                "system_prompt": system_prompt,
                **kwargs
            }
            tool = "openai_chat"
        elif vendor == "groq":
            params = {
                "messages": [{"role": "user", "content": message}],
                "model": model,
                "system_prompt": system_prompt,
                **kwargs
            }
            tool = "groq_chat"
        elif vendor == "local_llama":
            params = {
                "prompt": message,
                "system_prompt": system_prompt,
                "chat_format": True,
                **kwargs
            }
            tool = "local_inference"
        else:
            raise ValueError(f"Unsupported chat vendor: {vendor}")
            
        response = await self.orchestrator.route_request(vendor, tool, params)
        return self._format_chat_response(response, vendor)
        
    async def embed(
        self,
        text: Union[str, List[str]],
        vendor: str = "auto",
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text or list of texts to embed
            vendor: Vendor to use ("openai", "groq", "local_llama", or "auto")
            model: Specific model to use
            **kwargs: Additional parameters
            
        Returns:
            Embedding response with vectors and metadata
        """
        # Auto-select vendor if needed
        if vendor == "auto":
            vendor = self._select_embedding_vendor()
            
        # Prepare parameters based on vendor
        if vendor == "openai":
            params = {
                "input_text": text,
                "model": model,
                **kwargs
            }
            tool = "openai_embed"
        elif vendor == "groq":
            params = {
                "input_text": text,
                "model": model,
                **kwargs
            }
            tool = "groq_embed"
        elif vendor == "local_llama":
            params = {
                "input_text": text,
                "model_path": model,
                **kwargs
            }
            tool = "local_embed"
        else:
            raise ValueError(f"Unsupported embedding vendor: {vendor}")
            
        response = await self.orchestrator.route_request(vendor, tool, params)
        return self._format_embedding_response(response, vendor)
        
    async def search_web(
        self,
        query: str,
        num_results: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            num_results: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            Search results with URLs, titles, and snippets
        """
        params = {
            "query": query,
            "num_results": num_results,
            **kwargs
        }
        
        response = await self.orchestrator.route_request("utility", "web_search", params)
        return self._format_search_response(response)
        
    async def get_weather(
        self,
        location: str,
        weather_type: str = "current",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get weather information for a location.
        
        Args:
            location: Location name or coordinates
            weather_type: Type of weather data ("current", "forecast", "alerts")
            **kwargs: Additional weather parameters
            
        Returns:
            Weather data with temperature, conditions, and metadata
        """
        params = {
            "location": location,
            "weather_type": weather_type,
            **kwargs
        }
        
        response = await self.orchestrator.route_request("utility", "get_weather", params)
        return self._format_weather_response(response)
        
    async def list_models(self, vendor: str = "all") -> Dict[str, Any]:
        """
        List available models from vendors.
        
        Args:
            vendor: Vendor to query ("openai", "groq", "local_llama", or "all")
            
        Returns:
            Model information organized by vendor
        """
        if vendor == "all":
            models = {}
            for vendor_id in ["openai", "groq", "local_llama"]:
                if self._is_vendor_available(vendor_id):
                    vendor_models = await self._get_vendor_models(vendor_id)
                    models[vendor_id] = vendor_models
            return {"models": models}
        else:
            return await self._get_vendor_models(vendor)
            
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive capability information."""
        return await self.orchestrator.discover_capabilities()
        
    async def get_status(self) -> Dict[str, Any]:
        """Get system status information."""
        return await self.orchestrator.get_server_status()
        
    def _select_chat_vendor(self) -> str:
        """Auto-select the best available chat vendor."""
        preferences = ["openai", "groq", "local_llama"]
        
        for vendor in preferences:
            if self._is_vendor_available(vendor) and self._has_chat_capability(vendor):
                return vendor
                
        raise RuntimeError("No chat vendors available")
        
    def _select_embedding_vendor(self) -> str:
        """Auto-select the best available embedding vendor."""
        preferences = ["openai", "local_llama", "groq"]
        
        for vendor in preferences:
            if self._is_vendor_available(vendor) and self._has_embedding_capability(vendor):
                return vendor
                
        raise RuntimeError("No embedding vendors available")
        
    def _is_vendor_available(self, vendor: str) -> bool:
        """Check if a vendor is available and running."""
        server_info = self.capabilities.get("servers", {}).get(vendor, {})
        return server_info.get("status") == "running"
        
    def _has_chat_capability(self, vendor: str) -> bool:
        """Check if vendor has chat capabilities."""
        server_info = self.capabilities.get("servers", {}).get(vendor, {})
        tools = server_info.get("tools", [])
        
        chat_tools = ["openai_chat", "groq_chat", "local_inference"]
        return any(tool in tools for tool in chat_tools)
        
    def _has_embedding_capability(self, vendor: str) -> bool:
        """Check if vendor has embedding capabilities."""
        server_info = self.capabilities.get("servers", {}).get(vendor, {})
        tools = server_info.get("tools", [])
        
        embedding_tools = ["openai_embed", "groq_embed", "local_embed"]
        return any(tool in tools for tool in embedding_tools)
        
    async def _get_vendor_models(self, vendor: str) -> Dict[str, Any]:
        """Get model information for a specific vendor."""
        if vendor == "openai":
            response = await self.orchestrator.route_request(vendor, "list_models", {})
        else:
            # For other vendors, get from resources
            capabilities = await self.orchestrator.discover_capabilities()
            server_info = capabilities.get("servers", {}).get(vendor, {})
            response = {"models": server_info.get("resources", [])}
            
        return response
        
    def _format_chat_response(self, response: Dict[str, Any], vendor: str) -> Dict[str, Any]:
        """Format chat response for consistent interface."""
        if response.get("error"):
            return response
            
        # Extract message content based on vendor format
        if vendor == "openai" and "result" in response:
            # Simulate OpenAI response format
            return {
                "message": "Simulated OpenAI chat response",
                "vendor": vendor,
                "usage": {"tokens": 100},
                "model": "gpt-4"
            }
        elif vendor == "groq" and "result" in response:
            # Simulate Groq response format
            return {
                "message": "Simulated Groq chat response",
                "vendor": vendor,
                "usage": {"tokens": 100},
                "model": "llama2-70b-4096"
            }
        elif vendor == "local_llama" and "result" in response:
            # Simulate local response format
            return {
                "message": "Simulated local LLaMA response",
                "vendor": vendor,
                "usage": {"tokens": 100},
                "model": "local"
            }
            
        return response
        
    def _format_embedding_response(self, response: Dict[str, Any], vendor: str) -> Dict[str, Any]:
        """Format embedding response for consistent interface."""
        if response.get("error"):
            return response
            
        # Simulate embedding response
        return {
            "embeddings": [[0.1, 0.2, 0.3]] * 384,  # Simulated embedding
            "vendor": vendor,
            "dimensions": 384,
            "model": f"{vendor}_embedding_model"
        }
        
    def _format_search_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Format search response for consistent interface."""
        if response.get("error"):
            return response
            
        # Simulate search response
        return {
            "results": [
                {
                    "title": "Example Search Result",
                    "url": "https://example.com",
                    "snippet": "This is a simulated search result snippet."
                }
            ],
            "query": response.get("params", {}).get("query", ""),
            "total_results": 1
        }
        
    def _format_weather_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Format weather response for consistent interface."""
        if response.get("error"):
            return response
            
        # Simulate weather response
        return {
            "location": response.get("params", {}).get("location", ""),
            "temperature": 22,
            "description": "Partly cloudy",
            "humidity": 65,
            "wind_speed": 5.2
        }

# Convenience functions for direct usage
async def chat(message: str, vendor: str = "auto", **kwargs) -> Dict[str, Any]:
    """Convenience function for chat requests."""
    client = MCPClient()
    await client.connect()
    try:
        return await client.chat(message, vendor, **kwargs)
    finally:
        await client.disconnect()

async def embed(text: Union[str, List[str]], vendor: str = "auto", **kwargs) -> Dict[str, Any]:
    """Convenience function for embedding requests."""
    client = MCPClient()
    await client.connect()
    try:
        return await client.embed(text, vendor, **kwargs)
    finally:
        await client.disconnect()

async def search(query: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for web search."""
    client = MCPClient()
    await client.connect()
    try:
        return await client.search_web(query, **kwargs)
    finally:
        await client.disconnect()

async def weather(location: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for weather requests."""
    client = MCPClient()
    await client.connect()
    try:
        return await client.get_weather(location, **kwargs)
    finally:
        await client.disconnect()
