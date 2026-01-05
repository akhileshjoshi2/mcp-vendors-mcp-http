"""
MCP Vendors Client - Streamlit UI

A simple UI to test and interact with all MCP vendor servers.

Run with:
    pip install streamlit httpx
    streamlit run mcp_client_streamlit.py
"""

import streamlit as st
import httpx
import json
import asyncio
from typing import Dict, Any, Optional

# Page config
st.set_page_config(
    page_title="MCP Vendors Client",
    page_icon="🔌",
    layout="wide"
)

# Vendor configurations
VENDORS = {
    "openai": {
        "name": "OpenAI",
        "port": 8001,
        "url": "http://127.0.0.1:8001",
        "icon": "🤖",
        "description": "GPT chat and embeddings"
    },
    "groq": {
        "name": "Groq",
        "port": 8002,
        "url": "http://127.0.0.1:8002",
        "icon": "⚡",
        "description": "Fast LLM inference"
    },
    "utility": {
        "name": "Utility",
        "port": 8003,
        "url": "http://127.0.0.1:8003",
        "icon": "🔧",
        "description": "Web search & weather"
    },
    "local_llama": {
        "name": "Local LLaMA",
        "port": 8004,
        "url": "http://127.0.0.1:8004",
        "icon": "🦙",
        "description": "Ollama local models"
    }
}


class MCPClient:
    """MCP HTTP Client with session management."""
    
    def __init__(self, url: str):
        self.url = url
        self.session_id = None
        self.initialized = False
    
    async def _request(self, method: str, params: dict = None, request_id: int = 1) -> Dict:
        """Make an MCP request with session handling."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        # Add session ID if we have one
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id
        }
        if params:
            payload["params"] = params
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.url}/mcp",
                headers=headers,
                json=payload
            )
            
            # Capture session ID from response
            if "mcp-session-id" in response.headers:
                self.session_id = response.headers["mcp-session-id"]
            
            # Handle SSE response
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                text = response.text
                results = []
                for line in text.split("\n"):
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data:
                            try:
                                results.append(json.loads(data))
                            except:
                                pass
                # Return the last result (usually the actual response)
                return results[-1] if results else {"error": "No data in SSE response"}
            
            return response.json()
    
    async def _notification(self, method: str, params: dict = None):
        """Send a notification (no id, no response expected)."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        
        payload = {"jsonrpc": "2.0", "method": method}
        if params:
            payload["params"] = params
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(f"{self.url}/mcp", headers=headers, json=payload)
    
    async def initialize(self) -> bool:
        """Initialize the MCP session."""
        try:
            result = await self._request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "streamlit-client", "version": "1.0.0"}
            })
            self.initialized = "result" in result
            
            # Send initialized notification (no id field for notifications)
            if self.initialized:
                await self._notification("notifications/initialized")
            
            return self.initialized
        except:
            return False
    
    async def get_tools(self) -> list:
        """Get list of tools."""
        if not self.initialized:
            await self.initialize()
        
        result = await self._request("tools/list", {}, 3)
        return result.get("result", {}).get("tools", [])
    
    async def get_resources(self) -> list:
        """Get list of resources."""
        if not self.initialized:
            await self.initialize()
        
        result = await self._request("resources/list", {}, 4)
        return result.get("result", {}).get("resources", [])
    
    async def call_tool(self, name: str, arguments: dict) -> Dict:
        """Call a tool."""
        if not self.initialized:
            await self.initialize()
        
        return await self._request("tools/call", {
            "name": name,
            "arguments": arguments
        }, 5)
    
    async def read_resource(self, uri: str) -> Dict:
        """Read a resource."""
        if not self.initialized:
            await self.initialize()
        
        return await self._request("resources/read", {"uri": uri}, 6)


async def check_server_health(url: str) -> bool:
    """Check if server is online."""
    try:
        client = MCPClient(url)
        return await client.initialize()
    except:
        return False


async def get_tools(url: str) -> list:
    """Get list of tools from server."""
    try:
        client = MCPClient(url)
        return await client.get_tools()
    except Exception as e:
        st.error(f"Failed to get tools: {e}")
        return []


async def get_resources(url: str) -> list:
    """Get list of resources from server."""
    try:
        client = MCPClient(url)
        return await client.get_resources()
    except Exception as e:
        st.error(f"Failed to get resources: {e}")
        return []


async def execute_tool(url: str, tool_name: str, params: Dict[str, Any]) -> Dict:
    """Execute a tool on the server."""
    try:
        client = MCPClient(url)
        return await client.call_tool(tool_name, params)
    except Exception as e:
        return {"error": str(e)}


async def read_resource(url: str, uri: str) -> Dict:
    """Read a resource from the server."""
    try:
        client = MCPClient(url)
        return await client.read_resource(uri)
    except Exception as e:
        return {"error": str(e)}


def run_async(coro):
    """Run async function in Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Main UI
st.title("🔌 MCP Vendors Client")
st.markdown("Test and interact with your MCP servers")

# Sidebar - Vendor Selection
with st.sidebar:
    st.header("Select Vendor")
    
    # Check server status
    if st.button("🔄 Refresh Status"):
        st.rerun()
    
    st.divider()
    
    # Vendor selection
    selected_vendor = None
    for key, vendor in VENDORS.items():
        is_online = run_async(check_server_health(vendor["url"]))
        status = "🟢" if is_online else "🔴"
        
        if st.button(
            f"{status} {vendor['icon']} {vendor['name']}\n`Port: {vendor['port']}`",
            key=f"vendor_{key}",
            use_container_width=True
        ):
            st.session_state.selected_vendor = key
    
    st.divider()
    st.markdown("### Server URLs")
    for key, vendor in VENDORS.items():
        st.code(f"{vendor['name']}: {vendor['url']}")

# Get selected vendor
selected_vendor = st.session_state.get("selected_vendor")

if not selected_vendor:
    st.info("👈 Select a vendor from the sidebar to get started")
    st.stop()

vendor = VENDORS[selected_vendor]
st.header(f"{vendor['icon']} {vendor['name']} Server")
st.caption(vendor['description'])

# Check if server is online
is_online = run_async(check_server_health(vendor["url"]))
if not is_online:
    st.error(f"❌ {vendor['name']} server is offline. Start it with:")
    st.code(f"python test_http_{selected_vendor.replace('_', '_')}.py")
    st.stop()

st.success(f"✅ Connected to {vendor['name']} at {vendor['url']}")

# Create tabs
tab_tools, tab_resources = st.tabs(["🛠️ Tools", "📁 Resources"])

# Tools Tab
with tab_tools:
    tools = run_async(get_tools(vendor["url"]))
    
    if not tools:
        st.warning("No tools available or failed to load tools")
    else:
        # Tool selection
        tool_names = [t["name"] for t in tools]
        selected_tool_name = st.selectbox("Select Tool", tool_names)
        
        # Get selected tool details
        selected_tool = next((t for t in tools if t["name"] == selected_tool_name), None)
        
        if selected_tool:
            st.markdown(f"**Description:** {selected_tool.get('description', 'No description')}")
            
            # Build input form
            st.subheader("Parameters")
            schema = selected_tool.get("inputSchema", {})
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            params = {}
            
            if not properties:
                st.info("This tool requires no parameters")
            else:
                for prop_name, prop_info in properties.items():
                    # Handle anyOf/oneOf for union types (e.g., int | None)
                    prop_type = prop_info.get("type", "string")
                    if "anyOf" in prop_info or "oneOf" in prop_info:
                        types = prop_info.get("anyOf", prop_info.get("oneOf", []))
                        for t in types:
                            if t.get("type") and t.get("type") != "null":
                                prop_type = t.get("type")
                                break
                    
                    prop_desc = prop_info.get("description", "")
                    is_required = prop_name in required
                    default_val = prop_info.get("default")
                    
                    label = f"{prop_name} {'*' if is_required else ''}"
                    help_text = f"{prop_desc} (type: {prop_type})"
                    
                    if prop_type == "boolean":
                        default_bool = default_val if isinstance(default_val, bool) else False
                        params[prop_name] = st.checkbox(label, value=default_bool, help=help_text)
                    elif prop_type == "integer":
                        value = st.text_input(label, value=str(default_val) if default_val is not None else "", help=help_text)
                        if value:
                            try:
                                params[prop_name] = int(value)
                            except ValueError:
                                st.warning(f"{prop_name}: Invalid integer value")
                    elif prop_type == "number":
                        value = st.text_input(label, value=str(default_val) if default_val is not None else "", help=help_text)
                        if value:
                            try:
                                params[prop_name] = float(value)
                            except ValueError:
                                st.warning(f"{prop_name}: Invalid number value")
                    elif prop_type == "array":
                        value = st.text_area(label, help=f"{help_text} - Enter as JSON array")
                        if value:
                            try:
                                params[prop_name] = json.loads(value)
                            except:
                                params[prop_name] = value
                    elif prop_type == "object":
                        value = st.text_area(label, help=f"{help_text} - Enter as JSON object")
                        if value:
                            try:
                                params[prop_name] = json.loads(value)
                            except:
                                params[prop_name] = value
                    else:
                        default_str = str(default_val) if default_val is not None else ""
                        value = st.text_input(label, value=default_str, help=help_text)
                        if value:
                            params[prop_name] = value
            
            # Execute button
            if st.button("▶️ Execute Tool", type="primary"):
                with st.spinner(f"Executing {selected_tool_name}..."):
                    result = run_async(execute_tool(vendor["url"], selected_tool_name, params))
                
                st.subheader("Result")
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Try to extract content from MCP response
                    content = result.get("result", {}).get("content", [])
                    if content:
                        for item in content:
                            if item.get("type") == "text":
                                try:
                                    # Try to parse as JSON for pretty display
                                    parsed = json.loads(item.get("text", ""))
                                    st.json(parsed)
                                except:
                                    st.code(item.get("text", ""))
                    else:
                        st.json(result)

# Resources Tab
with tab_resources:
    resources = run_async(get_resources(vendor["url"]))
    
    if not resources:
        st.warning("No resources available or failed to load resources")
    else:
        for resource in resources:
            uri = resource.get("uri", "")
            name = resource.get("name", uri)
            
            with st.expander(f"📄 {name}"):
                st.code(uri)
                
                if st.button(f"Read Resource", key=f"read_{uri}"):
                    with st.spinner(f"Reading {uri}..."):
                        result = run_async(read_resource(vendor["url"], uri))
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        content = result.get("result", {}).get("contents", [])
                        if content:
                            for item in content:
                                text = item.get("text", "")
                                try:
                                    parsed = json.loads(text)
                                    st.json(parsed)
                                except:
                                    st.code(text)
                        else:
                            st.json(result)

# Footer
st.divider()
st.caption("MCP Vendors Client | Built with Streamlit")
