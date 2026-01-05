# Local LLaMA MCP Server

Provides local quantized LLaMA model inference for private, offline AI capabilities through the Model Context Protocol (MCP). Enables AI functionality without external API dependencies or internet connectivity.

## Overview

This MCP server enables private, on-device AI inference using local LLaMA models. It's designed for scenarios requiring data privacy, offline operation, or reduced dependency on external services. The server supports various quantized model formats and can run on both CPU and GPU hardware.

## Architecture

```
local_llama_server/
├── server.py               # FastMCP server entry point
├── config.py              # Configuration and environment management
├── tools/                 # MCP tool implementations
│   ├── inference.py       # Local text generation
│   └── embeddings.py      # Local embedding generation
├── resources/             # MCP resource endpoints
│   └── models.py          # Local models information
├── requirements.txt       # Python dependencies
└── README.md             # This documentation
```

### Components

- **Server**: FastMCP-based server for local model management
- **Tools**: Local inference and embedding generation tools
- **Resources**: Model configuration and capability information
- **Config**: Environment-based configuration for local models

## Setup Instructions

### 1. Install Dependencies

```bash
cd local_llama_server
pip install -r requirements.txt

# For production use, install additional dependencies:
pip install llama-cpp-python torch transformers sentence-transformers
```

### 2. Download Models

Download compatible LLaMA models in GGUF or GGML format:

```bash
# Example: Download a quantized model
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf

# Or use huggingface-hub
pip install huggingface-hub
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_0.gguf --local-dir ./models
```

### 3. Configure Environment

Create a `.env` file in the project root:

```bash
# Required
MODEL_PATH=/path/to/your/model.gguf

# Optional (with defaults)
MODEL_TYPE=llama
DEVICE=cpu
MAX_TOKENS=2048
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=40
CONTEXT_LENGTH=4096
N_THREADS=4
N_GPU_LAYERS=0

# For GPU acceleration (optional)
DEVICE=cuda
N_GPU_LAYERS=35

# For embedding models (optional)
EMBEDDING_MODEL_PATH=/path/to/embedding/model
```

### 4. Run the Server

```bash
python server.py
```

The server will start on `localhost:8003` by default.

## Tool Reference

### `local_inference`

Generate text using local LLaMA model inference.

**Parameters:**
- `prompt` (str): Input text prompt for generation
- `max_tokens` (Optional[int]): Maximum tokens to generate
- `temperature` (Optional[float]): Sampling temperature 0-2
- `top_p` (Optional[float]): Nucleus sampling parameter
- `top_k` (Optional[int]): Top-k sampling parameter
- `system_prompt` (Optional[str]): System message for chat format
- `chat_format` (bool): Whether to use chat format or raw completion

**Returns:**
- `generated_text`: Generated response text
- `usage`: Token usage statistics
- `model_info`: Model configuration details
- `timing`: Performance metrics including tokens/second

**Example:**
```python
response = await local_inference(
    "Explain quantum computing in simple terms",
    max_tokens=500,
    temperature=0.7,
    system_prompt="You are a helpful science teacher."
)
print(f"Response: {response['generated_text']}")
print(f"Speed: {response['timing']['tokens_per_second']:.2f} tokens/sec")
```

### `local_embed`

Generate text embeddings using local models for semantic search.

**Parameters:**
- `input_text` (Union[str, List[str]]): Text(s) to embed
- `model_path` (Optional[str]): Path to embedding model
- `normalize` (bool): Whether to normalize embeddings to unit vectors
- `pooling_method` (str): Method for pooling ("mean", "max", "cls")

**Returns:**
- `embeddings`: Generated embedding vectors
- `model_info`: Model configuration details
- `usage`: Processing statistics
- `dimensions`: Embedding vector dimensions

**Example:**
```python
# Single text embedding
result = await local_embed("Hello world")
embedding = result["embeddings"]

# Batch embedding generation
texts = ["Document 1", "Document 2", "Query text"]
result = await local_embed(texts, normalize=True)
embeddings = result["embeddings"]
```

## Model Configuration

### Supported Model Formats

- **GGUF**: Recommended format for llama.cpp compatibility
- **GGML**: Legacy format, still supported
- **Safetensors**: Hugging Face format
- **PyTorch**: Standard PyTorch model files

### Quantization Levels

| Format | Size Reduction | Quality Impact | Use Case |
|--------|----------------|----------------|----------|
| Q4_0 | ~75% | Minimal | General use, good balance |
| Q5_0 | ~65% | Very minimal | Better quality than Q4 |
| Q8_0 | ~50% | Nearly none | High quality applications |
| F16 | ~50% vs F32 | None | Maximum compatibility |
| F32 | 0% | None (baseline) | Research, fine-tuning |

### Hardware Requirements

#### Minimum (7B Model)
- **RAM**: 8GB
- **Storage**: 5GB
- **CPU**: 4 cores
- **Use case**: Basic chat, simple tasks

#### Recommended (13B Model)
- **RAM**: 16GB
- **Storage**: 20GB
- **CPU**: 8+ cores
- **GPU**: 8GB+ VRAM (optional)
- **Use case**: General productivity, coding assistance

#### Optimal (70B Model)
- **RAM**: 32GB+
- **Storage**: 50GB+
- **CPU**: 16+ cores
- **GPU**: 24GB+ VRAM
- **Use case**: Research, complex reasoning

## Example Usage

### Basic Text Generation

```python
import asyncio
from local_llama_server.tools.inference import local_inference

async def generate_text():
    response = await local_inference(
        prompt="Write a Python function to calculate fibonacci numbers",
        max_tokens=300,
        temperature=0.3
    )
    
    print("Generated code:")
    print(response["generated_text"])
    print(f"Generation time: {response['timing']['inference_time_ms']:.2f}ms")

asyncio.run(generate_text())
```

### Chat Conversation

```python
async def chat_conversation():
    system_prompt = "You are a helpful programming assistant."
    
    messages = [
        "How do I optimize Python code for performance?",
        "What about memory optimization specifically?",
        "Can you show an example of memory profiling?"
    ]
    
    for message in messages:
        response = await local_inference(
            prompt=message,
            system_prompt=system_prompt,
            max_tokens=200,
            temperature=0.7,
            chat_format=True
        )
        
        print(f"User: {message}")
        print(f"Assistant: {response['generated_text']}\n")

asyncio.run(chat_conversation())
```

### Semantic Search with Embeddings

```python
import asyncio
from local_llama_server.tools.embeddings import local_embed
import numpy as np

async def semantic_search():
    # Document corpus
    documents = [
        "Python is a high-level programming language",
        "Machine learning uses algorithms to find patterns",
        "Quantum computing leverages quantum mechanics",
        "Web development involves creating websites"
    ]
    
    # Generate embeddings for documents
    doc_result = await local_embed(documents, normalize=True)
    doc_embeddings = doc_result["embeddings"]
    
    # Query
    query = "What is artificial intelligence?"
    query_result = await local_embed(query, normalize=True)
    query_embedding = query_result["embeddings"]
    
    # Calculate similarities
    similarities = []
    for doc_emb in doc_embeddings:
        similarity = np.dot(query_embedding, doc_emb)
        similarities.append(similarity)
    
    # Find most similar document
    best_match_idx = np.argmax(similarities)
    print(f"Query: {query}")
    print(f"Best match: {documents[best_match_idx]}")
    print(f"Similarity: {similarities[best_match_idx]:.3f}")

asyncio.run(semantic_search())
```

## Performance Optimization

### CPU Optimization

```bash
# Set optimal thread count (usually number of physical cores)
N_THREADS=8

# Use appropriate model size for your hardware
# 7B models for 8-16GB RAM
# 13B models for 16-32GB RAM
# 70B models for 32GB+ RAM
```

### GPU Acceleration

```bash
# Enable GPU acceleration
DEVICE=cuda
N_GPU_LAYERS=35  # Adjust based on VRAM

# For Apple Silicon
DEVICE=metal
```

### Memory Management

```bash
# Reduce context length if memory is limited
CONTEXT_LENGTH=2048

# Use more aggressive quantization
# Q4_0 for maximum memory savings
# Q8_0 for quality/memory balance
```

## Extending the Server

### Adding Custom Models

1. Download or train your model
2. Convert to GGUF format if needed
3. Update MODEL_PATH in configuration
4. Restart the server

```python
# Example: Loading custom model
config = Config()
config.MODEL_PATH = "/path/to/custom/model.gguf"
```

### Custom Inference Parameters

```python
# tools/custom_inference.py
@tool()
async def custom_inference(prompt: str, **kwargs) -> dict:
    """Custom inference with specialized parameters."""
    # Implement custom logic
    pass
```

### Integration with RAG Systems

```python
async def rag_pipeline(query: str, documents: List[str]):
    # Generate embeddings for documents
    doc_embeddings = await local_embed(documents)
    
    # Find relevant documents
    query_embedding = await local_embed(query)
    relevant_docs = find_similar_documents(query_embedding, doc_embeddings)
    
    # Generate response with context
    context = "\n".join(relevant_docs)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    
    response = await local_inference(prompt, max_tokens=300)
    return response["generated_text"]
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```
   Error: Failed to load model from path
   ```
   Solution: Verify MODEL_PATH exists and model format is supported

2. **Out of Memory**
   ```
   Error: Not enough memory to load model
   ```
   Solution: Use smaller model or more aggressive quantization

3. **Slow Inference**
   ```
   Very slow token generation
   ```
   Solution: Increase N_THREADS, enable GPU acceleration, or use smaller model

4. **GPU Not Detected**
   ```
   Warning: GPU acceleration not available
   ```
   Solution: Install CUDA toolkit, verify GPU compatibility

### Performance Monitoring

```python
# Monitor inference performance
response = await local_inference("Test prompt")
print(f"Tokens/sec: {response['timing']['tokens_per_second']}")
print(f"Memory usage: {response['model_info']['device']}")
```

### Debug Mode

```bash
LOG_LEVEL=DEBUG python server.py
```

## Security and Privacy

### Privacy Benefits

- **No External Calls**: All processing happens locally
- **Data Isolation**: No data leaves your system
- **Offline Operation**: Works without internet connection
- **Custom Models**: Use domain-specific or fine-tuned models

### Security Considerations

- **Model Integrity**: Verify model checksums before use
- **Resource Limits**: Monitor CPU/memory usage
- **Access Control**: Implement authentication if needed
- **Model Updates**: Keep models updated for security patches

## License

MIT License - See the main project LICENSE file for details.

## Resources

- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [TheBloke Quantized Models](https://huggingface.co/TheBloke)
- [LLaMA Model Cards](https://github.com/facebookresearch/llama)
