"""
Improved Utility MCP Server

Provides utility tools for web search and weather information.
This server exposes non-AI utility capabilities through standardized MCP tools.
"""

import asyncio
import argparse
import logging
from typing import Dict, Any, Optional, List
from fastmcp import FastMCP
import httpx
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import json
import urllib.parse
import uvicorn

# Load environment from .env.example
def load_env_from_file(file_path):
    if not os.path.exists(file_path):
        return False
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip('"\'')
                os.environ[key] = value
    return True

# Load environment - try both current directory and parent
if not load_env_from_file('.env.example'):
    load_env_from_file('../.env.example')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP application
app = FastMCP("Improved Utility MCP Server")

# Configuration
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_BASE_URL = os.getenv("WEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5")

if not WEATHER_API_KEY:
    logger.warning("WEATHER_API_KEY not set - weather functionality will be limited")

@app.tool()
async def web_search(
    query: str,
    num_results: int = 10,
    region: Optional[str] = None,
    time_range: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        num_results: Number of results to return (max 20)
        region: Search region (e.g., 'us-en', 'uk-en')
        time_range: Time range filter ('d', 'w', 'm', 'y')
    
    Returns:
        Dict containing search results with titles, URLs, and snippets
    """
    try:
        # Clean up query - remove extra quotes if present
        clean_query = query.strip('"\'') if isinstance(query, str) else str(query)
        logger.info(f"Searching for: {clean_query}")
        
        # Limit results to reasonable number
        num_results = min(num_results, 20)
        
        # Use DuckDuckGo Instant Answer API first, then fallback to HTML scraping
        instant_url = "https://api.duckduckgo.com/"
        instant_params = {
            "q": clean_query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        results = []
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Try instant answers first
            try:
                instant_response = await client.get(instant_url, params=instant_params)
                if instant_response.status_code == 200:
                    instant_data = instant_response.json()
                    
                    # Check for instant answer
                    if instant_data.get("Answer"):
                        results.append({
                            "title": f"Instant Answer: {clean_query}",
                            "url": instant_data.get("AnswerURL", "https://duckduckgo.com"),
                            "snippet": instant_data["Answer"]
                        })
                    
                    # Add related topics
                    for topic in instant_data.get("RelatedTopics", [])[:3]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append({
                                "title": topic.get("Text", "")[:100] + "...",
                                "url": topic.get("FirstURL", ""),
                                "snippet": topic.get("Text", "")
                            })
            except Exception as e:
                logger.debug(f"Instant answer failed: {e}")
            
            # If we don't have enough results, try HTML scraping
            if len(results) < num_results:
                try:
                    # Build search URL for HTML scraping
                    html_url = "https://html.duckduckgo.com/html/"
                    html_params = {
                        "q": clean_query,
                        "kl": region or "wt-wt"
                    }
                    
                    if time_range:
                        html_params["df"] = time_range
                    
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    }
                    
                    html_response = await client.get(html_url, params=html_params, headers=headers)
                    html_response.raise_for_status()
                    
                    # Parse HTML response
                    soup = BeautifulSoup(html_response.text, 'html.parser')
                    
                    # Look for organic results (not ads)
                    result_selectors = [
                        'div.result',
                        'div.web-result',
                        'div.result__body',
                        '.result'
                    ]
                    
                    found_results = []
                    for selector in result_selectors:
                        found_results = soup.select(selector)
                        if found_results:
                            break
                    
                    for div in found_results[:num_results - len(results)]:
                        try:
                            # Try different selectors for title and URL
                            title_link = (
                                div.find('a', class_='result__a') or
                                div.find('h2', class_='result__title') or
                                div.find('a', href=True) or
                                div.find('h3')
                            )
                            
                            if not title_link:
                                continue
                            
                            # Get title
                            if title_link.name == 'a':
                                title = title_link.get_text(strip=True)
                                url = title_link.get('href', '')
                            else:
                                title = title_link.get_text(strip=True)
                                url_link = title_link.find('a', href=True)
                                url = url_link.get('href', '') if url_link else ''
                            
                            # Skip if it's an ad (contains ad indicators)
                            if any(indicator in url.lower() for indicator in ['ad_domain', 'click_metadata', '/aclick']):
                                continue
                            
                            # Extract snippet - try multiple approaches
                            snippet = ""
                            
                            # Try various snippet selectors
                            snippet_selectors = [
                                ('div', 'result__snippet'),
                                ('div', 'snippet'),
                                ('p', 'result-snippet'),
                                ('span', 'result__snippet'),
                                ('div', 'result-snippet')
                            ]
                            
                            for tag, class_name in snippet_selectors:
                                snippet_div = div.find(tag, class_=class_name)
                                if snippet_div:
                                    snippet = snippet_div.get_text(strip=True)
                                    break
                            
                            # If no snippet found, try to get any text content from the result
                            if not snippet:
                                # Look for any text content in the result div
                                text_elements = div.find_all(text=True)
                                text_content = ' '.join([t.strip() for t in text_elements if t.strip() and t.strip() not in [title, url]])
                                if text_content:
                                    snippet = text_content[:200] + "..." if len(text_content) > 200 else text_content
                            
                            # Clean up snippet - remove common prefixes
                            if snippet:
                                snippet = snippet.replace("This is the visible part", "").strip()
                                snippet = snippet.replace("www.", "").strip()
                                # Remove URL duplicates from snippet
                                if url:
                                    domain = url.replace("https://", "").replace("http://", "").split("/")[0]
                                    snippet = snippet.replace(domain, "").strip()
                                # Clean up extra whitespace
                                snippet = ' '.join(snippet.split())
                            
                            # Clean up URL if it's a redirect
                            if url.startswith('/'):
                                url = f"https://duckduckgo.com{url}"
                            elif url.startswith('//'):
                                url = f"https:{url}"
                            
                            # Extract actual URL from DuckDuckGo redirect
                            if 'duckduckgo.com/l/?uddg=' in url:
                                try:
                                    import urllib.parse
                                    # Extract the actual URL from the redirect
                                    parsed = urllib.parse.urlparse(url)
                                    query_params = urllib.parse.parse_qs(parsed.query)
                                    if 'uddg' in query_params:
                                        actual_url = urllib.parse.unquote(query_params['uddg'][0])
                                        url = actual_url
                                except Exception as e:
                                    logger.debug(f"Failed to extract redirect URL: {e}")
                                    pass
                            
                            if title and url and not any(indicator in url for indicator in ['duckduckgo.com/y.js', 'ad_domain']):
                                results.append({
                                    "title": title[:200],  # Limit title length
                                    "url": url,
                                    "snippet": snippet[:300]  # Limit snippet length
                                })
                                
                        except Exception as e:
                            logger.debug(f"Error parsing search result: {e}")
                            continue
                
                except Exception as e:
                    logger.debug(f"HTML scraping failed: {e}")
            
            # If still no results, provide a fallback
            if not results:
                results = [{
                    "title": f"Search for '{clean_query}' on DuckDuckGo",
                    "url": f"https://duckduckgo.com/?q={urllib.parse.quote(clean_query)}",
                    "snippet": f"Click to search for '{clean_query}' directly on DuckDuckGo"
                }]
            
            return {
                "results": results[:num_results],
                "query": clean_query,  # Return the cleaned query string
                "total_results": len(results),
                "region": region,
                "time_range": time_range,
                "search_method": "duckduckgo_api_and_html"
            }
            
    except httpx.TimeoutException:
        logger.error("Search request timed out")
        return {
            "error": "Search request timed out",
            "query": query,
            "results": []
        }
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return {
            "error": str(e),
            "query": query,
            "results": []
        }

@app.tool()
async def get_weather(
    location: str,
    weather_type: str = "current",
    units: str = "metric"
) -> Dict[str, Any]:
    """
    Get weather information for a location using OpenWeatherMap.
    
    Args:
        location: Location name or coordinates (lat,lon)
        weather_type: Type of weather data ('current', 'forecast', 'alerts')
        units: Temperature units ('metric', 'imperial', 'kelvin')
    
    Returns:
        Dict containing weather data
    """
    if not WEATHER_API_KEY:
        return {
            "error": "Weather API key not configured",
            "location": location
        }
    
    try:
        logger.info(f"Getting weather for: {location}")
        
        # Determine API endpoint based on weather type
        if weather_type == "current":
            endpoint = f"{WEATHER_BASE_URL}/weather"
        elif weather_type == "forecast":
            endpoint = f"{WEATHER_BASE_URL}/forecast"
        else:
            return {
                "error": f"Invalid weather_type: {weather_type}. Use 'current' or 'forecast'",
                "location": location
            }
        
        # Build request parameters
        params = {
            "appid": WEATHER_API_KEY,
            "units": units
        }
        
        # Handle location (name or coordinates)
        if "," in location and all(part.replace(".", "").replace("-", "").isdigit() for part in location.split(",")):
            # Coordinates format: lat,lon
            lat, lon = location.split(",")
            params["lat"] = lat.strip()
            params["lon"] = lon.strip()
        else:
            # Location name
            params["q"] = location
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Format response based on weather type
            if weather_type == "current":
                return {
                    "location": data.get("name", location),
                    "country": data.get("sys", {}).get("country", ""),
                    "temperature": data.get("main", {}).get("temp"),
                    "feels_like": data.get("main", {}).get("feels_like"),
                    "humidity": data.get("main", {}).get("humidity"),
                    "pressure": data.get("main", {}).get("pressure"),
                    "description": data.get("weather", [{}])[0].get("description", ""),
                    "wind_speed": data.get("wind", {}).get("speed"),
                    "wind_direction": data.get("wind", {}).get("deg"),
                    "visibility": data.get("visibility"),
                    "units": units,
                    "weather_type": weather_type
                }
            elif weather_type == "forecast":
                forecasts = []
                for item in data.get("list", [])[:5]:  # Next 5 forecasts
                    forecasts.append({
                        "datetime": item.get("dt_txt"),
                        "temperature": item.get("main", {}).get("temp"),
                        "description": item.get("weather", [{}])[0].get("description", ""),
                        "humidity": item.get("main", {}).get("humidity"),
                        "wind_speed": item.get("wind", {}).get("speed")
                    })
                
                return {
                    "location": data.get("city", {}).get("name", location),
                    "country": data.get("city", {}).get("country", ""),
                    "forecasts": forecasts,
                    "units": units,
                    "weather_type": weather_type
                }
                
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            error_msg = "Invalid weather API key"
        elif e.response.status_code == 404:
            error_msg = f"Location '{location}' not found"
        else:
            error_msg = f"Weather API error: {e.response.status_code}"
            
        logger.error(error_msg)
        return {
            "error": error_msg,
            "location": location
        }
    except Exception as e:
        logger.error(f"Weather error: {e}")
        return {
            "error": str(e),
            "location": location
        }

@app.tool()
def health_check() -> str:
    """Health check endpoint for the Utility MCP server."""
    return "Improved Utility MCP Server is running"

async def main(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8003):
    """Main entry point for the Utility MCP server."""
    try:
        logger.info("Starting Improved Utility MCP Server...")
        logger.info(f"Weather API configured: {bool(WEATHER_API_KEY)}")
        logger.info(f"Transport: {transport}")
        
        if transport == "http":
            logger.info(f"Running HTTP server on {host}:{port}")
            config = uvicorn.Config(app.http_app(), host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()
        else:
            # Run the FastMCP server over stdio
            await app.run_stdio_async()
        
    except Exception as e:
        logger.error(f"Failed to start Utility MCP Server: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio", help="Transport type")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8003, help="HTTP port (default: 8003)")
    args = parser.parse_args()
    
    asyncio.run(main(transport=args.transport, host=args.host, port=args.port))
