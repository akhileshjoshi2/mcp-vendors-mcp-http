# MCP Servers Testing Guide

Quick guide for testing your MCP servers with MCP Inspector.

## 🚀 Quick Start

### Test OpenAI Server
```bash
npx @modelcontextprotocol/inspector python test_openai.py
```

### Test Utility Server  
```bash
npx @modelcontextprotocol/inspector python test_utility.py
```

### Test Groq Server
```bash
npx @modelcontextprotocol/inspector python test_groq.py
```

### Test Local LLaMA Server
```bash
npx @modelcontextprotocol/inspector python test_local_llama.py
```

## 🧪 Test Parameters

### OpenAI Server Tools

#### `openai_chat`
```json
[
  {
    "role": "user",
    "content": "Hello! How are you?"
  }
]
```

#### `openai_embed`
```json
"Python programming is fun"
```

#### `list_models`
```json
{}
```

### Groq Server Tools

#### `groq_chat`

**Basic Chat:**
```json
[
  {
    "role": "user",
    "content": "Hello! What makes Groq special for AI inference?"
  }
]
```

**Code Generation:**
```json
[
  {
    "role": "user",
    "content": "Write a Python function to calculate factorial using recursion"
  }
]
```

**Conversation with History:**
```json
[
  {
    "role": "user",
    "content": "What is machine learning?"
  },
  {
    "role": "assistant",
    "content": "Machine learning is a subset of AI that enables computers to learn from data."
  },
  {
    "role": "user",
    "content": "Give me a practical example"
  }
]
```

**Expected Output:**
```json
{
  "message": "Hello! Groq is special for AI inference because it uses custom Language Processing Units (LPUs) designed specifically for sequential processing tasks like language models. This allows for extremely fast inference speeds - often 10x faster than traditional GPUs...",
  "model": "llama-3.1-8b-instant",
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 142,
    "total_tokens": 157
  },
  "finish_reason": "stop",
  "system_fingerprint": "fp_..."
}
```

#### `groq_embed`
**Parameter:**
```json
"Machine learning with Groq acceleration"
```

**Expected Output:**
```json
{
  "error": "Groq embedding models not yet available",
  "text": "Machine learning with Groq acceleration",
  "suggestion": "Use OpenAI embeddings or local embedding models"
}
```

#### `list_models`
**Parameter:**
```json
{}
```

**Expected Output:**
```json
{
  "models": [
    {
      "id": "llama2-70b-4096",
      "name": "Llama 2 70B",
      "description": "Meta's Llama 2 70B model optimized for Groq",
      "context_length": 4096,
      "type": "chat"
    },
    {
      "id": "mixtral-8x7b-32768",
      "name": "Mixtral 8x7B",
      "description": "Mistral's Mixtral 8x7B model with 32K context",
      "context_length": 32768,
      "type": "chat"
    },
    {
      "id": "gemma-7b-it",
      "name": "Gemma 7B IT",
      "description": "Google's Gemma 7B instruction-tuned model",
      "context_length": 8192,
      "type": "chat"
    }
  ],
  "total": 3,
  "default_model": "llama-3.1-8b-instant"
}
```

#### `health_check`
**Parameter:**
```json
{}
```

**Expected Output:**
```json
"Groq MCP Server is running"
```

### Groq Server Resources

#### `groq://models`
**Expected Output:**
```json
{
  "provider": "Groq",
  "description": "High-performance LLM inference with Groq's custom silicon",
  "models": [
    {
      "id": "llama2-70b-4096",
      "name": "Llama 2 70B",
      "description": "Meta's Llama 2 70B model optimized for Groq",
      "context_length": 4096,
      "type": "chat"
    },
    {
      "id": "mixtral-8x7b-32768",
      "name": "Mixtral 8x7B",
      "description": "Mistral's Mixtral 8x7B model with 32K context",
      "context_length": 32768,
      "type": "chat"
    },
    {
      "id": "gemma-7b-it",
      "name": "Gemma 7B IT",
      "description": "Google's Gemma 7B instruction-tuned model",
      "context_length": 8192,
      "type": "chat"
    }
  ],
  "capabilities": [
    "Ultra-fast inference",
    "Low latency responses",
    "High throughput",
    "Open-source model support"
  ],
  "pricing": "Pay-per-token with competitive rates",
  "documentation": "https://console.groq.com/docs"
}
```

### Utility Server Tools

#### `web_search`
```json
"Python programming tutorials"
```

#### `get_weather`
```json
{
  "location": "London",
  "weather_type": "current"
}
```

### Local LLaMA Server Tools

Local LLaMA uses **Ollama** under the hood for inference. Make sure:

- Ollama is running (e.g. `ollama serve` or the desktop app).
- You have at least one model pulled (e.g. `tinyllama`, `alibayram/medgemma:4b`).
- `local_llama_server/.env` (or your environment) sets:
  - `OLLAMA_MODEL` (e.g. `tinyllama`)
  - `OLLAMA_BASE_URL` (default `http://localhost:11434`)

#### `local_llama_inference`

This tool calls your configured Ollama model for **text generation**.

Typical MCP Inspector inputs (fill individual fields, not raw JSON):

```json
{
  "prompt": "Explain what a transformer model is in simple terms.",
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "system_prompt": null,
  "chat_format": true
}
```

Example output shape:

```json
{
  "generated_text": "...model-generated explanation...",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 80,
    "total_tokens": 92
  },
  "model_info": {
    "model_path": "/path/to/your/local/model",
    "model_type": "llama",
    "device": "cpu"
  },
  "generation_config": {
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40
  },
  "timing": {
    "inference_time_ms": 1200.0,
    "tokens_per_second": 60.0
  },
  "provider": "local_llama"
}
```

**Tips:**

- For interactive testing, prefer a small model like `tinyllama`.
- Keep `max_tokens` around `64–128` to avoid MCP Inspector timeouts.

#### `local_llama_embed`

Embeddings are currently **simulated** (hash-based) for testing. The tool
shape is still useful for wiring:

```json
{
  "input_text": "Local LLaMA embeddings are useful for offline semantic search.",
  "model_path": null,
  "normalize": true,
  "pooling_method": "mean"
}
```

Expected response shape:

```json
{
  "embeddings": [0.12, 0.34, 0.56, ...],
  "model_info": {
    "model_path": "...",
    "model_type": "local_embedding",
    "device": "cpu",
    "pooling_method": "mean"
  },
  "usage": {
    "input_texts": 1,
    "total_characters": 62
  },
  "dimensions": 384,
  "normalized": true,
  "provider": "local_llama"
}
```

#### `health_check`

```json
{}
```

Expected:

```json
"Local LLaMA MCP Server is running"
```

### Local LLaMA Server Resources

#### `local://models`

This resource now includes **live information from Ollama** plus static
config details.

Example response snippet:

```json
{
  "provider": "Local LLaMA",
  "description": "Private, offline AI inference using local quantized models",
  "configuration": {
    "model_path": "/path/to/your/local/model",
    "model_type": "llama",
    "device": "cpu",
    "max_tokens": 2048,
    "temperature": 0.7,
    "gpu_acceleration": false
  },
  "local_models": {
    "ollama": {
      "base_url": "http://localhost:11434",
      "available_models": [
        {
          "name": "alibayram/medgemma:4b",
          "model": "alibayram/medgemma:4b",
          "size": 2489895540
        },
        {
          "name": "tinyllama:latest",
          "model": "tinyllama:latest",
          "size": 637700138
        }
      ],
      "active_model": "tinyllama",
      "error": null
    }
  }
}
```

**Notes:**

- `available_models` is taken from `ollama list` via `GET /api/tags`.
- `active_model` is read from `OLLAMA_MODEL` in your environment.
- If `error` is non-null, check that Ollama is running and reachable at
  `OLLAMA_BASE_URL`.

## 🚀 Groq Testing Tips

### **Speed Comparison**
- **Groq responses** are typically much faster than OpenAI (often under 1 second)
- **Notice the speed difference** when testing similar prompts on both servers
- **Perfect for real-time applications** requiring low latency

### **Model Selection**
- **llama-3.1-8b-instant** - Your configured model, optimized for speed
- **mixtral-8x7b-32768** - Larger context window (32K tokens)
- **llama2-70b-4096** - More powerful but slower

### **Best Use Cases**
- **Code generation** - Very fast Python, JavaScript, etc.
- **Quick Q&A** - Instant responses to questions
- **Creative writing** - Fast story/poem generation
- **Technical explanations** - Clear, concise explanations

### **Limitations**
- **No embeddings** - Use OpenAI server for embedding tasks
- **Smaller models** - May be less capable than GPT-4 for complex tasks
- **Open-source focus** - Different training data than proprietary models

## 📊 Expected Results

### OpenAI Chat Response
```json
{
  "message": "Hello! I'm doing well, thank you for asking...",
  "model": "gpt-4",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 25,
    "total_tokens": 37
  }
}
```

### Web Search Response
```json
{
  "results": [
    {
      "title": "Python Tutorial - W3Schools",
      "url": "https://www.w3schools.com/python/",
      "snippet": "Learn Python programming with tutorials..."
    }
  ],
  "query": "Python programming tutorials",
  "total_results": 5
}
```

### Weather Response
```json
{
  "location": "London",
  "country": "GB", 
  "temperature": 16.05,
  "description": "overcast clouds",
  "humidity": 78
}
```

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Kill existing MCP Inspector processes
taskkill /F /IM node.exe
```

### Missing API Keys
- Check `.env.example` file has your API keys
- Ensure `OPENAI_API_KEY`, `GROQ_API_KEY`, and `WEATHER_API_KEY` are set

### Import Errors
- Make sure you're running from the `mcp-vendors` directory
- Check that all dependencies are installed

## 📁 Clean Project Structure

```
mcp-vendors/
├── test_openai.py          # OpenAI server test
├── test_utility.py         # Utility server test
├── test_groq.py            # Groq server test
├── .env.example            # API keys configuration
├── openai_server/
│   └── server.py           # Working OpenAI server
├── utility_server/
│   └── server.py           # Working utility server
└── config/
    └── mcp_servers.json    # Server configuration
```

## ✅ Success Indicators

- ✅ MCP Inspector opens in browser
- ✅ Tools appear in left panel
- ✅ Tools execute without errors
- ✅ Responses contain expected data
- ✅ No "Input validation error" messages
