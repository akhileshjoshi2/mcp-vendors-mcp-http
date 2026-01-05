"""
MCP Orchestrator

Central router and connector layer that manages multiple MCP servers.
Provides unified access to all MCP vendors through a single interface.
"""

import asyncio
import json
import logging
import subprocess
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPServer:
    """Represents an MCP server configuration."""
    name: str
    description: str
    command: str
    args: List[str]
    cwd: str
    port: int
    health_endpoint: str
    tools: List[str]
    resources: List[str]
    required_env: List[str]
    process: Optional[subprocess.Popen] = None
    status: str = "stopped"
    last_health_check: Optional[float] = None

class MCPOrchestrator:
    """
    Orchestrates multiple MCP servers and provides unified access.
    
    Features:
    - Dynamic server discovery and management
    - Health monitoring and auto-restart
    - Request routing and load balancing
    - Unified tool and resource discovery
    """
    
    def __init__(self, config_path: str = "config/mcp_servers.json"):
        """Initialize the orchestrator with server configuration."""
        self.config_path = Path(config_path)
        self.servers: Dict[str, MCPServer] = {}
        self.config: Dict[str, Any] = {}
        self.running = False
        
    async def start(self):
        """Start the orchestrator and all configured servers."""
        logger.info("Starting MCP Orchestrator...")
        
        # Load configuration
        await self._load_config()
        
        # Start all servers
        await self._start_all_servers()
        
        # Begin health monitoring
        self.running = True
        asyncio.create_task(self._health_monitor())
        
        logger.info("MCP Orchestrator started successfully")
        
    async def stop(self):
        """Stop the orchestrator and all servers."""
        logger.info("Stopping MCP Orchestrator...")
        
        self.running = False
        
        # Stop all servers
        await self._stop_all_servers()
        
        logger.info("MCP Orchestrator stopped")
        
    async def _load_config(self):
        """Load server configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
                
            self.config = config_data
            
            # Parse server configurations
            for server_id, server_config in config_data.get("servers", {}).items():
                server = MCPServer(
                    name=server_config["name"],
                    description=server_config["description"],
                    command=server_config["command"],
                    args=server_config["args"],
                    cwd=server_config.get("cwd", "."),
                    port=server_config["port"],
                    health_endpoint=server_config.get("health_endpoint", "/health"),
                    tools=server_config.get("tools", []),
                    resources=server_config.get("resources", []),
                    required_env=server_config.get("required_env", [])
                )
                self.servers[server_id] = server
                
            logger.info(f"Loaded configuration for {len(self.servers)} servers")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
            
    async def _start_all_servers(self):
        """Start all configured MCP servers."""
        for server_id, server in self.servers.items():
            await self._start_server(server_id, server)
            
    async def _start_server(self, server_id: str, server: MCPServer):
        """Start a single MCP server."""
        try:
            logger.info(f"Starting server: {server.name}")
            
            # Check environment variables
            missing_env = self._check_required_env(server.required_env)
            if missing_env:
                logger.warning(f"Server {server.name} missing required env vars: {missing_env}")
                server.status = "env_missing"
                return
                
            # Start the server process
            cmd = [server.command] + server.args
            server.process = subprocess.Popen(
                cmd,
                cwd=server.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for startup
            await asyncio.sleep(2)
            
            # Check if process is still running
            if server.process.poll() is None:
                server.status = "running"
                logger.info(f"Server {server.name} started successfully (PID: {server.process.pid})")
            else:
                server.status = "failed"
                stdout, stderr = server.process.communicate()
                logger.error(f"Server {server.name} failed to start: {stderr}")
                
        except Exception as e:
            logger.error(f"Failed to start server {server.name}: {e}")
            server.status = "error"
            
    async def _stop_all_servers(self):
        """Stop all running MCP servers."""
        for server_id, server in self.servers.items():
            await self._stop_server(server_id, server)
            
    async def _stop_server(self, server_id: str, server: MCPServer):
        """Stop a single MCP server."""
        if server.process and server.process.poll() is None:
            logger.info(f"Stopping server: {server.name}")
            server.process.terminate()
            
            # Wait for graceful shutdown
            try:
                server.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing server: {server.name}")
                server.process.kill()
                
            server.status = "stopped"
            server.process = None
            
    def _check_required_env(self, required_env: List[str]) -> List[str]:
        """Check for missing required environment variables."""
        import os
        missing = []
        for env_var in required_env:
            if not os.getenv(env_var):
                missing.append(env_var)
        return missing
        
    async def _health_monitor(self):
        """Monitor server health and restart if needed."""
        while self.running:
            for server_id, server in self.servers.items():
                if server.status == "running":
                    healthy = await self._check_server_health(server)
                    if not healthy:
                        logger.warning(f"Server {server.name} health check failed, restarting...")
                        await self._restart_server(server_id, server)
                        
            await asyncio.sleep(30)  # Check every 30 seconds
            
    async def _check_server_health(self, server: MCPServer) -> bool:
        """Check if a server is healthy."""
        try:
            # Check if process is still running
            if not server.process or server.process.poll() is not None:
                return False
                
            # Try HTTP health check if available
            health_url = f"http://localhost:{server.port}{server.health_endpoint}"
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(health_url)
                server.last_health_check = time.time()
                return response.status_code == 200
                
        except Exception as e:
            logger.debug(f"Health check failed for {server.name}: {e}")
            return False
            
    async def _restart_server(self, server_id: str, server: MCPServer):
        """Restart a failed server."""
        await self._stop_server(server_id, server)
        await asyncio.sleep(1)
        await self._start_server(server_id, server)
        
    async def route_request(self, vendor: str, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a request to the appropriate MCP server.
        
        Args:
            vendor: Target vendor/server (e.g., "openai", "groq")
            tool: Tool name to invoke
            params: Parameters for the tool
            
        Returns:
            Response from the target server
        """
        if vendor not in self.servers:
            return {
                "error": f"Unknown vendor: {vendor}",
                "available_vendors": list(self.servers.keys())
            }
            
        server = self.servers[vendor]
        
        if server.status != "running":
            return {
                "error": f"Server {vendor} is not running (status: {server.status})",
                "server_status": server.status
            }
            
        if tool not in server.tools:
            return {
                "error": f"Tool {tool} not available on server {vendor}",
                "available_tools": server.tools
            }
            
        try:
            # Route request to the server
            # In a real implementation, this would use MCP protocol
            # For now, we'll simulate the routing
            
            logger.info(f"Routing request: {vendor}.{tool}")
            
            # Simulate request processing
            await asyncio.sleep(0.1)
            
            return {
                "vendor": vendor,
                "tool": tool,
                "params": params,
                "result": f"Simulated response from {vendor}.{tool}",
                "status": "success",
                "server_info": {
                    "name": server.name,
                    "port": server.port,
                    "status": server.status
                }
            }
            
        except Exception as e:
            logger.error(f"Request routing failed: {e}")
            return {
                "error": str(e),
                "vendor": vendor,
                "tool": tool
            }
            
    async def discover_capabilities(self) -> Dict[str, Any]:
        """
        Discover all available tools and resources across servers.
        
        Returns:
            Comprehensive capability map of all servers
        """
        capabilities = {
            "servers": {},
            "tools": {},
            "resources": {},
            "summary": {
                "total_servers": len(self.servers),
                "running_servers": 0,
                "total_tools": 0,
                "total_resources": 0
            }
        }
        
        for server_id, server in self.servers.items():
            server_info = {
                "name": server.name,
                "description": server.description,
                "status": server.status,
                "port": server.port,
                "tools": server.tools,
                "resources": server.resources,
                "last_health_check": server.last_health_check
            }
            
            capabilities["servers"][server_id] = server_info
            
            # Count running servers
            if server.status == "running":
                capabilities["summary"]["running_servers"] += 1
                
            # Map tools to servers
            for tool in server.tools:
                if tool not in capabilities["tools"]:
                    capabilities["tools"][tool] = []
                capabilities["tools"][tool].append(server_id)
                
            # Map resources to servers
            for resource in server.resources:
                if resource not in capabilities["resources"]:
                    capabilities["resources"][resource] = []
                capabilities["resources"][resource].append(server_id)
                
        capabilities["summary"]["total_tools"] = len(capabilities["tools"])
        capabilities["summary"]["total_resources"] = len(capabilities["resources"])
        
        return capabilities
        
    async def get_server_status(self) -> Dict[str, Any]:
        """Get detailed status of all servers."""
        status = {
            "orchestrator": {
                "running": self.running,
                "config_path": str(self.config_path),
                "uptime": time.time() - getattr(self, '_start_time', time.time())
            },
            "servers": {}
        }
        
        for server_id, server in self.servers.items():
            server_status = {
                "name": server.name,
                "status": server.status,
                "port": server.port,
                "pid": server.process.pid if server.process else None,
                "last_health_check": server.last_health_check,
                "required_env_status": {
                    env_var: bool(os.getenv(env_var)) 
                    for env_var in server.required_env
                }
            }
            status["servers"][server_id] = server_status
            
        return status

async def main():
    """Main entry point for the orchestrator."""
    import os
    
    # Set start time for uptime calculation
    orchestrator = MCPOrchestrator()
    orchestrator._start_time = time.time()
    
    try:
        await orchestrator.start()
        
        # Keep running and handle requests
        logger.info("Orchestrator is ready. Press Ctrl+C to stop.")
        
        # Example of capability discovery
        capabilities = await orchestrator.discover_capabilities()
        logger.info(f"Discovered {capabilities['summary']['total_tools']} tools across {capabilities['summary']['running_servers']} servers")
        
        # Example request routing
        if capabilities['summary']['running_servers'] > 0:
            # Find first available server and tool
            for server_id, server_info in capabilities['servers'].items():
                if server_info['status'] == 'running' and server_info['tools']:
                    tool = server_info['tools'][0]
                    response = await orchestrator.route_request(
                        vendor=server_id,
                        tool=tool,
                        params={"test": "parameter"}
                    )
                    logger.info(f"Test request result: {response}")
                    break
        
        # Keep running
        while orchestrator.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
    finally:
        await orchestrator.stop()

if __name__ == "__main__":
    import os
    asyncio.run(main())
