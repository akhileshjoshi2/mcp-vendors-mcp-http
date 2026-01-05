# MCP Vendors Ecosystem

A modular, vendor-agnostic Model Context Protocol (MCP) ecosystem that allows seamless connection to different AI and utility services through standardized MCP servers.

## 🏗️ Architecture Overview

```
mcp-vendors/
├── openai_server/          # OpenAI GPT and embedding services
├── groq_server/            # Groq-accelerated inference
├── local_llama_server/     # Local quantized LLaMA models
├── utility_server/         # Non-AI tools (search, weather)
├── orchestrator/           # Central router and connector
└── config/                 # Global configuration files
```

## 🧩 Core Design Principles

- **Modular Servers**: Each vendor runs independently using FastMCP
- **Decoupled Configuration**: Dynamic server discovery via configuration
- **LLM Agnostic**: Easy addition of new AI providers
- **Composable Logic**: Tools can embed advanced reasoning flows
- **Lightweight Interface**: Self-describing tools via MCP decorators
- **Secure Configuration**: Environment-based credential management

## 🚀 Quick Start

### 1. Install Dependencies

**Linux/macOS (Bash):**
```bash
# Install core dependencies
pip install fastmcp python-dotenv uvicorn

# Server-specific dependencies
cd openai_server && pip install -r requirements.txt
cd groq_server && pip install -r requirements.txt
cd local_llama_server && pip install -r requirements.txt
cd utility_server && pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
# Install core dependencies
pip install fastmcp python-dotenv uvicorn

# Server-specific dependencies (run each separately)
cd c:\mcp-vendors\openai_server
pip install -r requirements.txt

cd c:\mcp-vendors\groq_server
pip install -r requirements.txt

cd c:\mcp-vendors\local_llama_server
pip install -r requirements.txt

cd c:\mcp-vendors\utility_server
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root with the following keys:

```env
# OpenAI Configuration (required for openai_server)
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# Groq Configuration (required for groq_server)
GROQ_API_KEY=your-groq-api-key-here
GROQ_MODEL=llama-3.1-8b-instant

# Local LLaMA / Ollama Configuration (required for local_llama_server)
MODEL_PATH=/path/to/your/local/model
MODEL_TYPE=llama
DEVICE=cpu
MAX_TOKENS=2048
OLLAMA_MODEL=tinyllama
OLLAMA_BASE_URL=http://localhost:11434

# Utility Services Configuration (required for utility_server)
WEATHER_API_KEY=your-openweathermap-api-key-here
WEATHER_BASE_URL=https://api.openweathermap.org/data/2.5

# Orchestrator Configuration (optional)
ORCHESTRATOR_HOST=localhost
ORCHESTRATOR_PORT=8000
LOG_LEVEL=INFO
```

**Where to get API keys:**

| Key | Provider | URL |
|-----|----------|-----|
| `OPENAI_API_KEY` | OpenAI | https://platform.openai.com/api-keys |
| `GROQ_API_KEY` | Groq | https://console.groq.com/keys |
| `WEATHER_API_KEY` | OpenWeatherMap | https://openweathermap.org/api |

**Local LLaMA / Ollama:**

- Install Ollama from https://ollama.com
- Pull a model: `ollama pull tinyllama`
- Set `OLLAMA_MODEL` to the model name you want to use
- Ollama must be running (`ollama serve`) before starting `local_llama_server`

### 3. Run Individual Servers

Each server supports two transport modes:
- **STDIO Mode**: For testing with MCP Inspector (interactive debugging)
- **HTTP Mode**: For production use (REST API on specific ports)

---

## 🖥️ OpenAI Server

**STDIO Mode (MCP Inspector):**
```bash
# From project root (c:\mcp-vendors)
npx @modelcontextprotocol/inspector python test_openai.py
```

**HTTP Mode (Port 8001):**
```bash
# From project root (c:\mcp-vendors)
python test_http_openai.py
```

---

## 🚀 Groq Server

**STDIO Mode (MCP Inspector):**
```bash
# From project root (c:\mcp-vendors)
npx @modelcontextprotocol/inspector python test_groq.py
```

**HTTP Mode (Port 8002):**
```bash
# From project root (c:\mcp-vendors)
python test_http_groq.py
```

---

## 🔧 Utility Server

**STDIO Mode (MCP Inspector):**
```bash
# From project root (c:\mcp-vendors)
npx @modelcontextprotocol/inspector python test_utility.py
```

**HTTP Mode (Port 8003):**
```bash
# From project root (c:\mcp-vendors)
python test_http_utility.py
```

---

## 🦙 Local LLaMA Server

> **Prerequisite:** Ollama must be running (`ollama serve`) before starting this server.

**STDIO Mode (MCP Inspector):**
```bash
# From project root (c:\mcp-vendors)
npx @modelcontextprotocol/inspector python test_local_llama.py
```

**HTTP Mode (Port 8004):**
```bash
# From project root (c:\mcp-vendors)
python test_http_local_llama.py
```

---

## 📊 Server Ports Summary

| Server | Default Port | URL |
|--------|--------------|-----|
| OpenAI | 8001 | http://127.0.0.1:8001 |
| Groq | 8002 | http://127.0.0.1:8002 |
| Utility | 8003 | http://127.0.0.1:8003 |
| Local LLaMA | 8004 | http://127.0.0.1:8004 |

---

## 🖥️ Web UI Clients

Test your MCP servers with a visual interface using Streamlit or Gradio.

### Streamlit Client
```bash
pip install streamlit httpx
streamlit run mcp_client_streamlit.py
```
Opens at: http://localhost:8501

### Gradio Client
```bash
pip install gradio httpx
python mcp_client_gradio.py
```
Opens at: http://localhost:7860

**Features:**
- Select vendor from dropdown
- View server status (online/offline)
- Browse and execute tools with parameter forms
- Browse and read resources
- JSON response display

---

## 🏃 Run All Servers (HTTP Mode)

Open 4 terminals from the project root (`mcp-vendors`) and run:

```bash
# Terminal 1 - OpenAI
python test_http_openai.py

# Terminal 2 - Groq
python test_http_groq.py

# Terminal 3 - Utility
python test_http_utility.py

# Terminal 4 - Local LLaMA
python test_http_local_llama.py
```

---

### 4. Use the Orchestrator

```bash
cd orchestrator
python orchestrator.py
```

## 📋 Server Responsibilities

| Server | Purpose | Example Tools |
|--------|---------|---------------|
| `openai_server` | OpenAI GPT and embeddings | `openai_chat`, `openai_embed`, `list_models` |
| `groq_server` | Groq-accelerated inference | `groq_chat`, `groq_embed` |
| `local_llama_server` | Local/offline inference | `local_inference`, `local_embed` |
| `utility_server` | Non-LLM utilities | `web_search`, `get_weather` |
| `orchestrator` | Routing and discovery | Dynamic server management |

## 🔄 Example Workflow

1. **Request**: User/system sends request to orchestrator
2. **Discovery**: Orchestrator reads `config/mcp_servers.json`
3. **Routing**: Request routed to appropriate MCP server
4. **Execution**: Tool executed via `@mcp.tool` decorator
5. **Response**: Uniform response returned to client

## 🛠️ Adding New Servers

1. Create new server directory following the template
2. Implement `server.py` with FastMCP
3. Add tools in `tools/` directory
4. Update `config/mcp_servers.json`
5. Add environment variables to `.env`

## 📚 Documentation

Each server contains detailed documentation:
- Setup and configuration instructions
- Tool reference with parameters
- Example usage and integration
- Extension guidelines

## 🔐 Security

- API keys stored in environment variables
- No hardcoded credentials in source code
- Each server runs in isolated process
- Configurable access controls

## 🤝 Contributing

1. Follow the established folder structure
2. Use FastMCP decorators for tools/resources
3. Include comprehensive docstrings
4. Add tests for new functionality
5. Update documentation

## 📄 License

MIT License - See individual server directories for specific licensing information.
