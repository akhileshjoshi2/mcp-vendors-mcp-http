"""
MCP Vendors Client - Gradio UI

A simple UI to test and interact with all MCP vendor servers.

Run with:
    pip install gradio httpx
    python mcp_client_gradio.py
"""

import gradio as gr
import httpx
import json
import asyncio
from typing import Dict, Any, List, Tuple

# Vendor configurations
VENDORS = {
    "OpenAI (Port 8001)": {"url": "http://127.0.0.1:8001", "key": "openai"},
    "Groq (Port 8002)": {"url": "http://127.0.0.1:8002", "key": "groq"},
    "Utility (Port 8003)": {"url": "http://127.0.0.1:8003", "key": "utility"},
    "Local LLaMA (Port 8004)": {"url": "http://127.0.0.1:8004", "key": "local_llama"},
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
        
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        
        payload = {"jsonrpc": "2.0", "method": method, "id": request_id}
        if params:
            payload["params"] = params
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{self.url}/mcp", headers=headers, json=payload)
            
            if "mcp-session-id" in response.headers:
                self.session_id = response.headers["mcp-session-id"]
            
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                results = []
                for line in response.text.split("\n"):
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data:
                            try:
                                results.append(json.loads(data))
                            except:
                                pass
                return results[-1] if results else {"error": "No data in SSE"}
            
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
        try:
            result = await self._request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "gradio-client", "version": "1.0.0"}
            })
            self.initialized = "result" in result
            if self.initialized:
                await self._notification("notifications/initialized")
            return self.initialized
        except:
            return False
    
    async def get_tools(self) -> List[Dict]:
        if not self.initialized:
            await self.initialize()
        result = await self._request("tools/list", {}, 3)
        return result.get("result", {}).get("tools", [])
    
    async def get_resources(self) -> List[Dict]:
        if not self.initialized:
            await self.initialize()
        result = await self._request("resources/list", {}, 4)
        return result.get("result", {}).get("resources", [])
    
    async def call_tool(self, name: str, arguments: dict) -> Dict:
        if not self.initialized:
            await self.initialize()
        return await self._request("tools/call", {"name": name, "arguments": arguments}, 5)
    
    async def read_resource(self, uri: str) -> Dict:
        if not self.initialized:
            await self.initialize()
        return await self._request("resources/read", {"uri": uri}, 6)


async def check_health(url: str) -> bool:
    try:
        client = MCPClient(url)
        return await client.initialize()
    except:
        return False


async def get_tools_async(url: str) -> List[Dict]:
    try:
        client = MCPClient(url)
        return await client.get_tools()
    except:
        return []


async def get_resources_async(url: str) -> List[Dict]:
    try:
        client = MCPClient(url)
        return await client.get_resources()
    except:
        return []


async def execute_tool_async(url: str, tool_name: str, params: Dict) -> Dict:
    try:
        client = MCPClient(url)
        return await client.call_tool(tool_name, params)
    except Exception as e:
        return {"error": str(e)}


async def read_resource_async(url: str, uri: str) -> Dict:
    try:
        client = MCPClient(url)
        return await client.read_resource(uri)
    except Exception as e:
        return {"error": str(e)}


def run_async(coro):
    """Run async function."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# State to store tools and resources
tools_cache = {}
resources_cache = {}


def check_server_status(vendor_name: str) -> str:
    """Check server status and return status message."""
    if vendor_name not in VENDORS:
        return "Please select a vendor"
    
    url = VENDORS[vendor_name]["url"]
    is_online = run_async(check_health(url))
    
    if is_online:
        return f"✅ {vendor_name} is ONLINE at {url}"
    else:
        key = VENDORS[vendor_name]["key"]
        return f"❌ {vendor_name} is OFFLINE\n\nStart with: python test_http_{key}.py"


def load_tools(vendor_name: str) -> Tuple[gr.Dropdown, str]:
    """Load tools for selected vendor."""
    if vendor_name not in VENDORS:
        return gr.Dropdown(choices=[], value=None), "Select a vendor first"
    
    url = VENDORS[vendor_name]["url"]
    tools = run_async(get_tools_async(url))
    tools_cache[vendor_name] = tools
    
    if not tools:
        return gr.Dropdown(choices=[], value=None), "No tools available or server offline"
    
    tool_names = [t["name"] for t in tools]
    tools_info = "\n\n".join([
        f"**{t['name']}**\n{t.get('description', 'No description')}"
        for t in tools
    ])
    
    return gr.Dropdown(choices=tool_names, value=tool_names[0] if tool_names else None), tools_info


def get_tool_schema(vendor_name: str, tool_name: str) -> str:
    """Get tool input schema."""
    if vendor_name not in tools_cache:
        return "{}"
    
    tools = tools_cache[vendor_name]
    tool = next((t for t in tools if t["name"] == tool_name), None)
    
    if not tool:
        return "{}"
    
    schema = tool.get("inputSchema", {})
    properties = schema.get("properties", {})
    
    # Generate example params
    example = {}
    for name, prop in properties.items():
        prop_type = prop.get("type", "string")
        if prop_type == "string":
            example[name] = f"<{name}>"
        elif prop_type in ["number", "integer"]:
            example[name] = 0
        elif prop_type == "boolean":
            example[name] = True
        elif prop_type == "array":
            example[name] = []
        elif prop_type == "object":
            example[name] = {}
    
    return json.dumps(example, indent=2)


def execute_tool(vendor_name: str, tool_name: str, params_json: str) -> str:
    """Execute the selected tool."""
    if vendor_name not in VENDORS:
        return "Error: Select a vendor first"
    
    if not tool_name:
        return "Error: Select a tool first"
    
    url = VENDORS[vendor_name]["url"]
    
    try:
        params = json.loads(params_json) if params_json.strip() else {}
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON parameters\n{e}"
    
    result = run_async(execute_tool_async(url, tool_name, params))
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    # Extract content from MCP response
    content = result.get("result", {}).get("content", [])
    if content:
        texts = []
        for item in content:
            if item.get("type") == "text":
                text = item.get("text", "")
                try:
                    parsed = json.loads(text)
                    texts.append(json.dumps(parsed, indent=2))
                except:
                    texts.append(text)
        return "\n\n".join(texts) if texts else json.dumps(result, indent=2)
    
    return json.dumps(result, indent=2)


def load_resources(vendor_name: str) -> Tuple[gr.Dropdown, str]:
    """Load resources for selected vendor."""
    if vendor_name not in VENDORS:
        return gr.Dropdown(choices=[], value=None), "Select a vendor first"
    
    url = VENDORS[vendor_name]["url"]
    resources = run_async(get_resources_async(url))
    resources_cache[vendor_name] = resources
    
    if not resources:
        return gr.Dropdown(choices=[], value=None), "No resources available"
    
    uris = [r["uri"] for r in resources]
    info = "\n".join([f"- {r['uri']}" for r in resources])
    
    return gr.Dropdown(choices=uris, value=uris[0] if uris else None), info


def read_resource(vendor_name: str, uri: str) -> str:
    """Read the selected resource."""
    if vendor_name not in VENDORS:
        return "Error: Select a vendor first"
    
    if not uri:
        return "Error: Select a resource first"
    
    url = VENDORS[vendor_name]["url"]
    result = run_async(read_resource_async(url, uri))
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    content = result.get("result", {}).get("contents", [])
    if content:
        texts = []
        for item in content:
            text = item.get("text", "")
            try:
                parsed = json.loads(text)
                texts.append(json.dumps(parsed, indent=2))
            except:
                texts.append(text)
        return "\n\n".join(texts)
    
    return json.dumps(result, indent=2)


# Build Gradio UI
with gr.Blocks(title="MCP Vendors Client", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🔌 MCP Vendors Client")
    gr.Markdown("Test and interact with your MCP servers")
    
    with gr.Row():
        # Left column - Vendor selection
        with gr.Column(scale=1):
            gr.Markdown("### Select Vendor")
            vendor_dropdown = gr.Dropdown(
                choices=list(VENDORS.keys()),
                label="Vendor",
                value=None
            )
            check_btn = gr.Button("🔄 Check Status", variant="secondary")
            status_output = gr.Textbox(label="Status", lines=3, interactive=False)
        
        # Right column - Main content
        with gr.Column(scale=3):
            with gr.Tabs():
                # Tools Tab
                with gr.TabItem("🛠️ Tools"):
                    with gr.Row():
                        with gr.Column():
                            load_tools_btn = gr.Button("Load Tools", variant="primary")
                            tool_dropdown = gr.Dropdown(label="Select Tool", choices=[])
                            tools_info = gr.Markdown("Select a vendor and load tools")
                        
                        with gr.Column():
                            params_input = gr.Code(
                                label="Parameters (JSON)",
                                language="json",
                                value="{}",
                                lines=8
                            )
                            execute_btn = gr.Button("▶️ Execute", variant="primary")
                    
                    tool_output = gr.Code(label="Result", language="json", lines=15)
                
                # Resources Tab
                with gr.TabItem("📁 Resources"):
                    with gr.Row():
                        load_resources_btn = gr.Button("Load Resources", variant="primary")
                        resource_dropdown = gr.Dropdown(label="Select Resource", choices=[])
                    
                    resources_info = gr.Markdown("Select a vendor and load resources")
                    read_btn = gr.Button("📖 Read Resource", variant="primary")
                    resource_output = gr.Code(label="Content", language="json", lines=15)
    
    # Event handlers
    check_btn.click(
        fn=check_server_status,
        inputs=[vendor_dropdown],
        outputs=[status_output]
    )
    
    vendor_dropdown.change(
        fn=check_server_status,
        inputs=[vendor_dropdown],
        outputs=[status_output]
    )
    
    load_tools_btn.click(
        fn=load_tools,
        inputs=[vendor_dropdown],
        outputs=[tool_dropdown, tools_info]
    )
    
    tool_dropdown.change(
        fn=get_tool_schema,
        inputs=[vendor_dropdown, tool_dropdown],
        outputs=[params_input]
    )
    
    execute_btn.click(
        fn=execute_tool,
        inputs=[vendor_dropdown, tool_dropdown, params_input],
        outputs=[tool_output]
    )
    
    load_resources_btn.click(
        fn=load_resources,
        inputs=[vendor_dropdown],
        outputs=[resource_dropdown, resources_info]
    )
    
    read_btn.click(
        fn=read_resource,
        inputs=[vendor_dropdown, resource_dropdown],
        outputs=[resource_output]
    )
    
    gr.Markdown("---")
    gr.Markdown("**Tip:** Start servers first with `python test_http_<vendor>.py`")


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860)
