"""
Basic Usage Examples for MCP Vendors Ecosystem

This file demonstrates common usage patterns and provides
getting-started examples for the MCP ecosystem.
"""

import asyncio
import sys
import os

# Add the parent directory to the path so we can import from orchestrator
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from orchestrator.mcp_client import MCPClient, chat, embed, search, weather

async def basic_chat_example():
    """Demonstrate basic chat functionality with different vendors."""
    print("=== Basic Chat Examples ===")
    
    # Using convenience function (auto vendor selection)
    print("\n1. Auto vendor selection:")
    response = await chat("What is artificial intelligence?")
    print(f"Response: {response.get('message', response.get('error', 'No response'))}")
    
    # Using specific vendors
    vendors = ["openai", "groq", "local_llama"]
    
    for vendor in vendors:
        print(f"\n2. Using {vendor} vendor:")
        try:
            response = await chat(
                "Explain machine learning in one sentence",
                vendor=vendor
            )
            print(f"Response: {response.get('message', response.get('error', 'No response'))}")
        except Exception as e:
            print(f"Error with {vendor}: {e}")

async def embedding_example():
    """Demonstrate text embedding functionality."""
    print("\n=== Embedding Examples ===")
    
    # Single text embedding
    print("\n1. Single text embedding:")
    text = "Machine learning is a subset of artificial intelligence"
    result = await embed(text)
    
    if result.get("embeddings"):
        print(f"Generated embedding with {len(result['embeddings'])} dimensions")
        print(f"First 5 values: {result['embeddings'][:5]}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Multiple text embeddings
    print("\n2. Batch text embeddings:")
    texts = [
        "Natural language processing",
        "Computer vision applications", 
        "Deep learning networks"
    ]
    
    result = await embed(texts)
    if result.get("embeddings"):
        print(f"Generated {len(result['embeddings'])} embeddings")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

async def search_example():
    """Demonstrate web search functionality."""
    print("\n=== Web Search Examples ===")
    
    # Basic search
    print("\n1. Basic web search:")
    results = await search("Python programming tutorials")
    
    if results.get("results"):
        print(f"Found {len(results['results'])} results:")
        for i, result in enumerate(results["results"][:3], 1):
            print(f"  {i}. {result.get('title', 'No title')}")
            print(f"     URL: {result.get('url', 'No URL')}")
            print(f"     Snippet: {result.get('snippet', 'No snippet')[:100]}...")
    else:
        print(f"Error: {results.get('error', 'No results found')}")
    
    # Search with parameters
    print("\n2. Search with parameters:")
    results = await search(
        "latest AI developments",
        num_results=5,
        region="us-en",
        time_range="m"  # Last month
    )
    
    if results.get("results"):
        print(f"Found {len(results['results'])} recent results")
    else:
        print(f"Error: {results.get('error', 'No results found')}")

async def weather_example():
    """Demonstrate weather functionality."""
    print("\n=== Weather Examples ===")
    
    # Current weather
    print("\n1. Current weather:")
    locations = ["New York", "London", "Tokyo"]
    
    for location in locations:
        result = await weather(location, weather_type="current")
        
        if result.get("temperature"):
            print(f"{location}: {result['temperature']}°C, {result.get('description', 'N/A')}")
        else:
            print(f"{location}: Error - {result.get('error', 'Unknown error')}")
    
    # Weather forecast
    print("\n2. Weather forecast:")
    result = await weather("San Francisco", weather_type="forecast")
    
    if result.get("temperature"):
        print(f"San Francisco forecast: {result.get('description', 'N/A')}")
    else:
        print(f"Forecast error: {result.get('error', 'Unknown error')}")

async def advanced_client_example():
    """Demonstrate advanced client usage patterns."""
    print("\n=== Advanced Client Usage ===")
    
    # Using persistent client connection
    client = MCPClient()
    await client.connect()
    
    try:
        # Get system capabilities
        print("\n1. System capabilities:")
        capabilities = await client.get_capabilities()
        
        print(f"Available servers: {list(capabilities.get('servers', {}).keys())}")
        print(f"Total tools: {capabilities.get('summary', {}).get('total_tools', 0)}")
        print(f"Running servers: {capabilities.get('summary', {}).get('running_servers', 0)}")
        
        # Get system status
        print("\n2. System status:")
        status = await client.get_status()
        
        for server_id, server_info in status.get("servers", {}).items():
            status_icon = "✅" if server_info.get("status") == "running" else "❌"
            print(f"  {status_icon} {server_info.get('name', server_id)}: {server_info.get('status', 'unknown')}")
        
        # Multiple operations with same connection
        print("\n3. Multiple operations:")
        
        # Chat
        chat_response = await client.chat("What's the weather like?")
        print(f"Chat: {chat_response.get('message', chat_response.get('error', 'No response'))[:50]}...")
        
        # Search
        search_response = await client.search_web("weather forecast", num_results=3)
        print(f"Search: Found {len(search_response.get('results', []))} results")
        
        # Weather
        weather_response = await client.get_weather("New York")
        print(f"Weather: {weather_response.get('temperature', 'N/A')}°C in New York")
        
    finally:
        await client.disconnect()

async def error_handling_example():
    """Demonstrate error handling patterns."""
    print("\n=== Error Handling Examples ===")
    
    # Handle missing API keys
    print("\n1. Handling missing API keys:")
    try:
        # This might fail if API keys are not configured
        response = await chat("Test message", vendor="openai")
        print("OpenAI chat successful")
    except Exception as e:
        print(f"OpenAI error (expected if no API key): {e}")
    
    # Handle invalid locations
    print("\n2. Handling invalid inputs:")
    result = await weather("InvalidLocationName12345")
    if result.get("error"):
        print(f"Weather error (expected): {result['error']}")
    else:
        print("Weather request unexpectedly succeeded")
    
    # Handle network issues
    print("\n3. Handling network issues:")
    try:
        # This might timeout or fail
        results = await search("test query")
        print(f"Search successful: {len(results.get('results', []))} results")
    except Exception as e:
        print(f"Search error: {e}")

async def integration_example():
    """Demonstrate integration patterns."""
    print("\n=== Integration Examples ===")
    
    # RAG-like pattern: Search + Chat
    print("\n1. Search + Chat integration:")
    
    # Search for information
    search_results = await search("latest developments in quantum computing", num_results=3)
    
    if search_results.get("results"):
        # Prepare context from search results
        context = "\n".join([
            f"- {result.get('title', '')}: {result.get('snippet', '')}"
            for result in search_results["results"][:2]
        ])
        
        # Use context in chat
        prompt = f"""Based on this recent information about quantum computing:
        
{context}

Please provide a brief summary of the latest developments."""
        
        response = await chat(prompt)
        print("Summary based on search results:")
        print(response.get('message', response.get('error', 'No response'))[:200] + "...")
    
    # Weather + Chat integration
    print("\n2. Weather + Chat integration:")
    
    weather_data = await weather("Paris")
    if weather_data.get("temperature"):
        prompt = f"""The current weather in Paris is {weather_data['temperature']}°C with {weather_data.get('description', 'unknown conditions')}. 
        
What clothing would you recommend for someone going out in this weather?"""
        
        response = await chat(prompt)
        print("Weather-based recommendation:")
        print(response.get('message', response.get('error', 'No response'))[:150] + "...")

async def main():
    """Run all examples."""
    print("MCP Vendors Ecosystem - Basic Usage Examples")
    print("=" * 50)
    
    examples = [
        basic_chat_example,
        embedding_example,
        search_example,
        weather_example,
        advanced_client_example,
        error_handling_example,
        integration_example
    ]
    
    for example in examples:
        try:
            await example()
            print("\n" + "-" * 50)
        except Exception as e:
            print(f"Example failed: {e}")
            print("-" * 50)
        
        # Small delay between examples
        await asyncio.sleep(1)
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    asyncio.run(main())
