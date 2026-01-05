"""
Local LLaMA Embeddings Tool

Provides local text embedding functionality using local models.
Enables private, offline embedding generation without external dependencies.
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from ..config import Config

logger = logging.getLogger(__name__)

# Global embedding model instance
_embedding_model = None

async def local_embed(
    input_text: Union[str, List[str]],
    model_path: Optional[str] = None,
    normalize: bool = True,
    pooling_method: str = "mean"
) -> Dict[str, Any]:
    """
    Generate embeddings using local models for offline semantic search.
    
    Args:
        input_text: Text string or list of strings to embed
        model_path: Path to embedding model (defaults to config)
        normalize: Whether to normalize embeddings to unit vectors
        pooling_method: Method for pooling token embeddings ("mean", "max", "cls")
        
    Returns:
        Dict containing embeddings, model info, and metadata
        
    Example:
        # Single text
        result = await local_embed("Hello world")
        
        # Multiple texts
        texts = ["Hello", "World", "Local AI"]
        result = await local_embed(texts)
    """
    try:
        config = Config()
        
        # Ensure input is a list
        if isinstance(input_text, str):
            input_texts = [input_text]
            single_input = True
        else:
            input_texts = input_text
            single_input = False
            
        logger.info(f"Generating local embeddings for {len(input_texts)} text(s)")
        
        # Initialize embedding model if not already loaded
        embedding_model = await _get_or_load_embedding_model(config, model_path)
        
        # Generate embeddings
        embeddings = []
        for text in input_texts:
            embedding = await _generate_embedding(embedding_model, text, pooling_method)
            
            if normalize:
                embedding = _normalize_embedding(embedding)
                
            embeddings.append(embedding)
        
        # If single input, return single embedding
        if single_input:
            embeddings = embeddings[0]
            
        result = {
            "embeddings": embeddings,
            "model_info": {
                "model_path": embedding_model["model_path"],
                "model_type": "local_embedding",
                "device": config.DEVICE,
                "pooling_method": pooling_method
            },
            "usage": {
                "input_texts": len(input_texts),
                "total_characters": sum(len(text) for text in input_texts)
            },
            "dimensions": len(embeddings[0]) if isinstance(embeddings[0], list) else len(embeddings),
            "normalized": normalize,
            "provider": "local_llama"
        }
        
        logger.info(f"Local embeddings generated successfully. Dimensions: {result['dimensions']}")
        return result
        
    except Exception as e:
        logger.error(f"Local embedding generation failed: {e}")
        return {
            "error": str(e),
            "embeddings": None,
            "usage": None
        }

async def _get_or_load_embedding_model(config: Config, model_path: Optional[str] = None):
    """Get or load the embedding model instance."""
    global _embedding_model
    
    if _embedding_model is None:
        logger.info("Loading local embedding model...")
        
        embedding_config = config.get_embedding_config()
        model_path = model_path or embedding_config["model_path"]
        
        # In a real implementation, this would load an actual embedding model:
        # from sentence_transformers import SentenceTransformer
        # _embedding_model = SentenceTransformer(model_path, device=config.DEVICE)
        
        # Or using transformers library:
        # from transformers import AutoTokenizer, AutoModel
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # model = AutoModel.from_pretrained(model_path)
        
        # For this example, we'll simulate model loading
        _embedding_model = {
            "model_path": model_path,
            "loaded": True,
            "device": config.DEVICE,
            "dimensions": 384  # Typical embedding dimension
        }
        
        logger.info("Embedding model loaded successfully")
    
    return _embedding_model

async def _generate_embedding(
    model,
    text: str,
    pooling_method: str = "mean"
) -> List[float]:
    """Generate embedding for a single text."""
    
    # In a real implementation, this would use the actual model:
    # if using sentence-transformers:
    # embedding = model.encode(text, convert_to_tensor=False)
    # return embedding.tolist()
    
    # if using transformers directly:
    # inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     embeddings = outputs.last_hidden_state
    #     if pooling_method == "mean":
    #         embedding = embeddings.mean(dim=1).squeeze()
    #     elif pooling_method == "max":
    #         embedding = embeddings.max(dim=1).values.squeeze()
    #     elif pooling_method == "cls":
    #         embedding = embeddings[:, 0, :].squeeze()
    # return embedding.tolist()
    
    # Simulate embedding generation with deterministic results
    await asyncio.sleep(0.1)  # Simulate processing time
    
    # Create a deterministic embedding based on text hash
    import hashlib
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Convert hash to embedding vector
    embedding = []
    for i in range(0, len(text_hash), 2):
        hex_pair = text_hash[i:i+2]
        value = int(hex_pair, 16) / 255.0  # Normalize to 0-1
        embedding.append(value)
    
    # Pad or truncate to target dimensions
    target_dims = model["dimensions"]
    if len(embedding) < target_dims:
        # Pad with text-based values
        text_values = [ord(c) / 255.0 for c in text[:target_dims - len(embedding)]]
        embedding.extend(text_values)
        
    # Ensure exact dimension count
    embedding = embedding[:target_dims]
    
    # Add some text-length based variation
    text_factor = len(text) / 1000.0
    embedding = [x + text_factor * 0.1 for x in embedding]
    
    return embedding

def _normalize_embedding(embedding: List[float]) -> List[float]:
    """Normalize embedding to unit vector."""
    embedding_array = np.array(embedding)
    norm = np.linalg.norm(embedding_array)
    
    if norm > 0:
        normalized = embedding_array / norm
        return normalized.tolist()
    else:
        return embedding

def _calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 > 0 and norm2 > 0:
        return dot_product / (norm1 * norm2)
    else:
        return 0.0
