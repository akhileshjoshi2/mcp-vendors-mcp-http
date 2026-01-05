"""
Start All Servers Script

Convenience script to start all MCP servers in the correct order
with proper dependency checking and health monitoring.
"""

import asyncio
import os
import sys
import time
import signal
from pathlib import Path
from typing import List, Dict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from orchestrator.orchestrator import MCPOrchestrator

class MCPSystemManager:
    """Manages the entire MCP system startup and shutdown."""
    
    def __init__(self):
        self.orchestrator = None
        self.running = False
        
    async def start_system(self):
        """Start the entire MCP system."""
        print("🚀 Starting MCP Vendors Ecosystem...")
        print("=" * 50)
        
        try:
            # Initialize orchestrator
            self.orchestrator = MCPOrchestrator()
            
            # Start orchestrator (this will start all servers)
            await self.orchestrator.start()
            
            # Wait for servers to stabilize
            print("\n⏳ Waiting for servers to stabilize...")
            await asyncio.sleep(5)
            
            # Check system status
            await self.check_system_status()
            
            # Set up signal handlers for graceful shutdown
            self.setup_signal_handlers()
            
            self.running = True
            print("\n✅ MCP System is running!")
            print("   Press Ctrl+C to stop all servers")
            
            # Keep running until interrupted
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested by user")
        except Exception as e:
            print(f"\n❌ System startup failed: {e}")
        finally:
            await self.stop_system()
    
    async def stop_system(self):
        """Stop the entire MCP system."""
        print("\n🛑 Stopping MCP System...")
        
        if self.orchestrator:
            await self.orchestrator.stop()
        
        self.running = False
        print("✅ MCP System stopped")
    
    async def check_system_status(self):
        """Check and display system status."""
        print("\n📊 System Status Check:")
        print("-" * 30)
        
        try:
            # Get capabilities
            capabilities = await self.orchestrator.discover_capabilities()
            
            total_servers = capabilities["summary"]["total_servers"]
            running_servers = capabilities["summary"]["running_servers"]
            total_tools = capabilities["summary"]["total_tools"]
            
            print(f"Servers: {running_servers}/{total_servers} running")
            print(f"Tools: {total_tools} available")
            
            # Show individual server status
            for server_id, server_info in capabilities["servers"].items():
                status = server_info["status"]
                status_icon = "✅" if status == "running" else "❌"
                print(f"  {status_icon} {server_info['name']}: {status}")
            
            if running_servers == 0:
                print("\n⚠️  No servers are running. Check environment variables and logs.")
            elif running_servers < total_servers:
                print(f"\n⚠️  Only {running_servers}/{total_servers} servers started successfully.")
            else:
                print(f"\n🎉 All {total_servers} servers are running successfully!")
                
        except Exception as e:
            print(f"❌ Status check failed: {e}")
    
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n📡 Received signal {signum}")
            self.running = False
        
        # Handle common shutdown signals
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)

async def run_health_monitor():
    """Run a simple health monitoring loop."""
    print("\n🏥 Starting health monitor...")
    
    orchestrator = MCPOrchestrator()
    
    try:
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            try:
                status = await orchestrator.get_server_status()
                
                print(f"\n[{time.strftime('%H:%M:%S')}] Health Check:")
                
                for server_id, server_info in status["servers"].items():
                    status_text = server_info["status"]
                    if status_text == "running":
                        print(f"  ✅ {server_info['name']}")
                    else:
                        print(f"  ❌ {server_info['name']}: {status_text}")
                        
            except Exception as e:
                print(f"  ⚠️  Health check error: {e}")
                
    except asyncio.CancelledError:
        print("🏥 Health monitor stopped")

async def quick_test():
    """Run a quick test of the system."""
    print("\n🧪 Running Quick System Test...")
    print("-" * 35)
    
    # Import the client
    from orchestrator.mcp_client import MCPClient
    
    client = MCPClient()
    
    try:
        await client.connect()
        
        # Test capabilities
        capabilities = await client.get_capabilities()
        running_servers = capabilities["summary"]["running_servers"]
        
        if running_servers == 0:
            print("❌ No servers running - cannot run tests")
            return
        
        print(f"✅ Connected to {running_servers} servers")
        
        # Test each available service
        tests_passed = 0
        total_tests = 0
        
        # Test chat if available
        if any("chat" in str(tools) for tools in [s.get("tools", []) for s in capabilities["servers"].values()]):
            total_tests += 1
            try:
                response = await client.chat("Hello, this is a test message", vendor="auto")
                if not response.get("error"):
                    print("✅ Chat test passed")
                    tests_passed += 1
                else:
                    print(f"❌ Chat test failed: {response.get('error')}")
            except Exception as e:
                print(f"❌ Chat test error: {e}")
        
        # Test search if available
        if "utility" in capabilities["servers"] and capabilities["servers"]["utility"]["status"] == "running":
            total_tests += 1
            try:
                response = await client.search_web("test query", num_results=1)
                if not response.get("error"):
                    print("✅ Search test passed")
                    tests_passed += 1
                else:
                    print(f"❌ Search test failed: {response.get('error')}")
            except Exception as e:
                print(f"❌ Search test error: {e}")
        
        # Test weather if available
        if "utility" in capabilities["servers"] and capabilities["servers"]["utility"]["status"] == "running":
            total_tests += 1
            try:
                response = await client.get_weather("London")
                if not response.get("error"):
                    print("✅ Weather test passed")
                    tests_passed += 1
                else:
                    print(f"❌ Weather test failed: {response.get('error')}")
            except Exception as e:
                print(f"❌ Weather test error: {e}")
        
        # Summary
        if total_tests > 0:
            print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
            if tests_passed == total_tests:
                print("🎉 All tests passed! System is working correctly.")
            else:
                print("⚠️  Some tests failed. Check server logs and configuration.")
        else:
            print("⚠️  No tests could be run. Check server configuration.")
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
    finally:
        await client.disconnect()

def print_usage():
    """Print usage information."""
    print("""
MCP System Manager

Usage:
    python start_all.py [command]

Commands:
    start     - Start all MCP servers (default)
    test      - Run quick system test
    monitor   - Run health monitoring
    help      - Show this help message

Examples:
    python start_all.py           # Start all servers
    python start_all.py test      # Test the system
    python start_all.py monitor   # Monitor server health
""")

async def main():
    """Main entry point."""
    # Parse command line arguments
    command = sys.argv[1] if len(sys.argv) > 1 else "start"
    
    if command == "help":
        print_usage()
        return
    elif command == "test":
        await quick_test()
        return
    elif command == "monitor":
        await run_health_monitor()
        return
    elif command == "start":
        # Start the system
        manager = MCPSystemManager()
        await manager.start_system()
    else:
        print(f"Unknown command: {command}")
        print_usage()
        return

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
