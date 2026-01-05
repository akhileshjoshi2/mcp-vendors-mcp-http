# Utility MCP Server

Provides essential non-AI utility tools such as web search and weather data through the Model Context Protocol (MCP). Designed to complement AI model capabilities with real-world data access.

## Overview

This MCP server offers commonly needed utility functions that enhance AI applications with real-world data. It provides web search capabilities via DuckDuckGo and weather information through OpenWeatherMap, enabling AI systems to access current information and environmental data.

## Architecture

```
utility_server/
├── server.py               # FastMCP server entry point
├── config.py              # Configuration and environment management
├── tools/                 # MCP tool implementations
│   ├── web_search.py      # Web search via DuckDuckGo
│   └── weather.py         # Weather data via OpenWeatherMap
├── requirements.txt       # Python dependencies
└── README.md             # This documentation
```

### Components

- **Server**: FastMCP-based server for utility tool management
- **Tools**: Web search and weather data retrieval tools
- **Config**: Environment-based configuration for external APIs

## Setup Instructions

### 1. Install Dependencies

```bash
cd utility_server
pip install -r requirements.txt
```

### 2. Get API Keys

#### OpenWeatherMap API (Required for Weather)
1. Visit [OpenWeatherMap](https://openweathermap.org/api)
2. Sign up for a free account
3. Generate an API key
4. Note: Free tier includes current weather and 5-day forecast

#### DuckDuckGo Search (No API Key Required)
- Web search functionality uses DuckDuckGo's public interface
- No registration or API key needed

### 3. Configure Environment

Create a `.env` file in the project root:

```bash
# Required for weather functionality
WEATHER_API_KEY=your_openweathermap_api_key_here

# Optional (with defaults)
WEATHER_BASE_URL=https://api.openweathermap.org/data/2.5
WEATHER_UNITS=metric

# Search configuration (optional)
SEARCH_ENGINE=duckduckgo
SEARCH_RESULTS_LIMIT=10
SEARCH_TIMEOUT=10

# General configuration
TIMEOUT=30
USER_AGENT=MCP-Utility-Server/1.0
```

### 4. Run the Server

```bash
python server.py
```

The server will start on `localhost:8004` by default.

## Tool Reference

### `web_search`

Search the web using DuckDuckGo search engine.

**Parameters:**
- `query` (str): Search query string
- `num_results` (Optional[int]): Number of results to return (default: 10)
- `region` (str): Search region (default: "us-en")
- `safe_search` (str): Safe search setting ("strict", "moderate", "off")
- `time_range` (Optional[str]): Time filter ("d", "w", "m", "y")

**Returns:**
- `query`: Original search query
- `results`: List of search results with title, URL, snippet, domain
- `total_results`: Number of results returned
- `search_engine`: "DuckDuckGo"
- `parameters`: Search parameters used

**Example:**
```python
results = await web_search(
    "Python machine learning tutorials",
    num_results=5,
    region="us-en",
    safe_search="moderate"
)

for result in results["results"]:
    print(f"{result['title']}")
    print(f"URL: {result['url']}")
    print(f"Snippet: {result['snippet']}")
    print(f"Domain: {result['domain']}")
    print("---")
```

### `get_weather`

Get weather information using OpenWeatherMap API.

**Parameters:**
- `location` (str): Location name, city, or coordinates (lat,lon)
- `weather_type` (str): Type of data ("current", "forecast", "alerts")
- `units` (Optional[str]): Temperature units ("metric", "imperial", "kelvin")
- `include_forecast_days` (int): Forecast days for forecast type (1-16)

**Returns:**
- `location`: Resolved location information
- `weather_type`: Type of weather data retrieved
- `units`: Temperature units used
- `data`: Weather data (structure varies by type)
- `timestamp`: Request timestamp
- `provider`: "OpenWeatherMap"

**Example:**
```python
# Current weather
current = await get_weather("New York", "current", units="metric")
print(f"Temperature: {current['data']['temperature']}°C")
print(f"Description: {current['data']['description']}")
print(f"Humidity: {current['data']['humidity']}%")

# 5-day forecast
forecast = await get_weather("London", "forecast", units="imperial")
for day in forecast["data"]["forecasts"]:
    print(f"{day['date']}: {day['temperature']['min']}-{day['temperature']['max']}°F")
    print(f"  {day['description']}")

# Using coordinates
weather = await get_weather("40.7128,-74.0060", "current")
```

## Detailed Usage Examples

### Web Search Integration

```python
import asyncio
from utility_server.tools.web_search import web_search

async def search_and_summarize():
    # Search for recent information
    results = await web_search(
        "latest AI developments 2024",
        num_results=5,
        time_range="m"  # Last month
    )
    
    print(f"Found {results['total_results']} results for: {results['query']}")
    
    for i, result in enumerate(results["results"], 1):
        print(f"\n{i}. {result['title']}")
        print(f"   Source: {result['domain']}")
        print(f"   URL: {result['url']}")
        print(f"   Summary: {result['snippet']}")

asyncio.run(search_and_summarize())
```

### Weather-Based Decision Making

```python
import asyncio
from utility_server.tools.weather import get_weather

async def weather_advisory():
    cities = ["New York", "London", "Tokyo", "Sydney"]
    
    for city in cities:
        weather = await get_weather(city, "current", units="metric")
        
        if weather.get("error"):
            print(f"Error getting weather for {city}: {weather['error']}")
            continue
            
        data = weather["data"]
        temp = data["temperature"]
        desc = data["description"]
        humidity = data["humidity"]
        
        print(f"\n{weather['location']['name']}, {weather['location']['country']}")
        print(f"Temperature: {temp}°C ({desc})")
        print(f"Humidity: {humidity}%")
        
        # Weather-based recommendations
        if temp < 0:
            print("❄️  Bundle up! It's freezing.")
        elif temp < 10:
            print("🧥  Wear a warm coat.")
        elif temp > 30:
            print("☀️  Stay hydrated! It's hot.")
        elif humidity > 80:
            print("💧  High humidity - might feel muggy.")

asyncio.run(weather_advisory())
```

### Combined Search and Weather

```python
async def travel_research(destination: str):
    # Get current weather
    weather = await get_weather(destination, "current")
    
    # Search for travel information
    search_query = f"{destination} travel guide weather what to pack"
    search_results = await web_search(search_query, num_results=3)
    
    print(f"Travel Research for {destination}")
    print("=" * 40)
    
    if not weather.get("error"):
        data = weather["data"]
        print(f"Current Weather:")
        print(f"  Temperature: {data['temperature']}°C")
        print(f"  Conditions: {data['description']}")
        print(f"  Humidity: {data['humidity']}%")
        print(f"  Wind: {data['wind']['speed']} m/s")
    
    print(f"\nTravel Information:")
    for result in search_results["results"]:
        print(f"  • {result['title']}")
        print(f"    {result['url']}")

# Example usage
asyncio.run(travel_research("Barcelona"))
```

## Advanced Features

### Search Result Filtering

```python
async def filtered_search(query: str, domain_filter: str = None):
    results = await web_search(query, num_results=20)
    
    if domain_filter:
        filtered_results = [
            r for r in results["results"] 
            if domain_filter.lower() in r["domain"].lower()
        ]
        results["results"] = filtered_results
        results["total_results"] = len(filtered_results)
    
    return results

# Search only on specific domains
github_results = await filtered_search("Python libraries", "github.com")
```

### Weather Trend Analysis

```python
async def weather_forecast_analysis(location: str):
    forecast = await get_weather(location, "forecast", include_forecast_days=5)
    
    if forecast.get("error"):
        return forecast
    
    forecasts = forecast["data"]["forecasts"]
    
    # Analyze temperature trends
    temps = [day["temperature"]["avg"] for day in forecasts]
    trend = "rising" if temps[-1] > temps[0] else "falling"
    
    # Find extreme days
    hottest_day = max(forecasts, key=lambda x: x["temperature"]["max"])
    coldest_day = min(forecasts, key=lambda x: x["temperature"]["min"])
    
    analysis = {
        "location": forecast["location"]["name"],
        "trend": trend,
        "temperature_range": {
            "min": min(temps),
            "max": max(temps),
            "average": sum(temps) / len(temps)
        },
        "hottest_day": {
            "date": hottest_day["date"],
            "temperature": hottest_day["temperature"]["max"]
        },
        "coldest_day": {
            "date": coldest_day["date"],
            "temperature": coldest_day["temperature"]["min"]
        },
        "rainy_days": len([d for d in forecasts if d.get("rain")])
    }
    
    return analysis
```

## Error Handling

### Common Issues

1. **Weather API Key Missing**
   ```
   Error: Required environment variable WEATHER_API_KEY is not set
   ```
   Solution: Add your OpenWeatherMap API key to `.env`

2. **Location Not Found**
   ```
   Error: Location 'XYZ' not found
   ```
   Solution: Use more specific location names or coordinates

3. **Search Timeout**
   ```
   Error: Request timeout during search
   ```
   Solution: Increase SEARCH_TIMEOUT or check internet connection

4. **Rate Limiting**
   ```
   Error: HTTP 429 Too Many Requests
   ```
   Solution: Implement request throttling or upgrade API plan

### Robust Error Handling

```python
async def safe_weather_request(location: str):
    try:
        weather = await get_weather(location, "current")
        
        if weather.get("error"):
            print(f"Weather error: {weather['error']}")
            return None
            
        return weather["data"]
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

async def safe_search_request(query: str):
    try:
        results = await web_search(query, num_results=5)
        
        if results.get("error"):
            print(f"Search error: {results['error']}")
            return []
            
        return results["results"]
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []
```

## Extending the Server

### Adding New Tools

```python
# tools/new_tool.py
from fastmcp import tool
import httpx

@tool()
async def new_utility_tool(param: str) -> dict:
    """New utility tool implementation."""
    # Implementation here
    pass

# Register in server.py
from tools.new_tool import new_utility_tool
app.add_tool(new_utility_tool)
```

### Custom Search Engines

```python
async def custom_search_engine(query: str, engine: str):
    if engine == "bing":
        return await _bing_search(query)
    elif engine == "google":
        return await _google_search(query)
    else:
        return await web_search(query)  # Default to DuckDuckGo
```

### Weather Data Enrichment

```python
async def enhanced_weather(location: str):
    # Get basic weather
    weather = await get_weather(location, "current")
    
    # Add air quality data (requires additional API)
    # Add UV index data
    # Add weather history comparison
    
    return enhanced_data
```

## Performance Optimization

### Caching Results

```python
import time
from typing import Dict, Any

# Simple in-memory cache
_cache: Dict[str, Dict[str, Any]] = {}
CACHE_DURATION = 300  # 5 minutes

async def cached_weather(location: str):
    cache_key = f"weather_{location}"
    now = time.time()
    
    if cache_key in _cache:
        cached_data = _cache[cache_key]
        if now - cached_data["timestamp"] < CACHE_DURATION:
            return cached_data["data"]
    
    # Fetch new data
    weather = await get_weather(location, "current")
    
    # Cache the result
    _cache[cache_key] = {
        "data": weather,
        "timestamp": now
    }
    
    return weather
```

### Batch Processing

```python
async def batch_weather_requests(locations: List[str]):
    tasks = [get_weather(loc, "current") for loc in locations]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    weather_data = {}
    for location, result in zip(locations, results):
        if isinstance(result, Exception):
            weather_data[location] = {"error": str(result)}
        else:
            weather_data[location] = result
    
    return weather_data
```

## Security Considerations

- **API Key Protection**: Store keys in environment variables only
- **Rate Limiting**: Implement client-side rate limiting for APIs
- **Input Validation**: Sanitize search queries and location inputs
- **Error Sanitization**: Don't expose sensitive API details in errors

## Troubleshooting

### Debug Mode

```bash
LOG_LEVEL=DEBUG python server.py
```

### Testing API Connectivity

```python
async def test_apis():
    # Test weather API
    try:
        weather = await get_weather("London", "current")
        print("Weather API: ✓ Working")
    except Exception as e:
        print(f"Weather API: ✗ Error - {e}")
    
    # Test search
    try:
        results = await web_search("test query", num_results=1)
        print("Search API: ✓ Working")
    except Exception as e:
        print(f"Search API: ✗ Error - {e}")

asyncio.run(test_apis())
```

## License

MIT License - See the main project LICENSE file for details.

## Resources

- [OpenWeatherMap API Documentation](https://openweathermap.org/api)
- [DuckDuckGo Search](https://duckduckgo.com)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [HTTPX Documentation](https://www.python-httpx.org)
