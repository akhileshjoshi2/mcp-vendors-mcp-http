"""
Web Search Tool

Provides web search functionality using DuckDuckGo search engine.
Enables searching the web without requiring API keys or external dependencies.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import httpx
from bs4 import BeautifulSoup
import urllib.parse
from fastmcp import tool
from ..config import Config

logger = logging.getLogger(__name__)

@tool()
async def web_search(
    query: str,
    num_results: Optional[int] = None,
    region: str = "us-en",
    safe_search: str = "moderate",
    time_range: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo search engine.
    
    Args:
        query: Search query string
        num_results: Number of results to return (defaults to config limit)
        region: Search region (e.g., "us-en", "uk-en", "de-de")
        safe_search: Safe search setting ("strict", "moderate", "off")
        time_range: Time range filter ("d" for day, "w" for week, "m" for month, "y" for year)
        
    Returns:
        Dict containing search results with titles, URLs, snippets, and metadata
        
    Example:
        results = await web_search("Python programming tutorials", num_results=5)
        for result in results["results"]:
            print(f"{result['title']}: {result['url']}")
    """
    try:
        config = Config()
        search_config = config.get_search_config()
        http_config = config.get_http_config()
        
        # Use provided parameters or fall back to config defaults
        num_results = num_results or search_config["results_limit"]
        
        logger.info(f"Searching web for: '{query}' (limit: {num_results})")
        
        # Perform DuckDuckGo search
        search_results = await _duckduckgo_search(
            query=query,
            num_results=num_results,
            region=region,
            safe_search=safe_search,
            time_range=time_range,
            config=search_config,
            http_config=http_config
        )
        
        result = {
            "query": query,
            "results": search_results,
            "total_results": len(search_results),
            "search_engine": "DuckDuckGo",
            "parameters": {
                "num_results": num_results,
                "region": region,
                "safe_search": safe_search,
                "time_range": time_range
            },
            "provider": "utility_server"
        }
        
        logger.info(f"Web search completed. Found {len(search_results)} results")
        return result
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {
            "error": str(e),
            "query": query,
            "results": [],
            "total_results": 0
        }

async def _duckduckgo_search(
    query: str,
    num_results: int,
    region: str,
    safe_search: str,
    time_range: Optional[str],
    config: dict,
    http_config: dict
) -> List[Dict[str, Any]]:
    """Perform DuckDuckGo search and parse results."""
    
    # Build search URL
    base_url = "https://html.duckduckgo.com/html/"
    params = {
        "q": query,
        "kl": region,
        "s": "0",  # Start index
        "dc": str(num_results),
        "v": "l",  # Layout
        "o": "json",
        "api": "/d.js"
    }
    
    # Add safe search parameter
    if safe_search == "strict":
        params["safe"] = "strict"
    elif safe_search == "off":
        params["safe"] = "off"
    # moderate is default, no parameter needed
    
    # Add time range filter
    if time_range:
        params["df"] = time_range
    
    try:
        async with httpx.AsyncClient(
            timeout=config["timeout"],
            headers=http_config["headers"]
        ) as client:
            
            # First, get the search page
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            
            # Parse HTML results
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Find search result containers
            result_containers = soup.find_all('div', class_='result')
            
            for container in result_containers[:num_results]:
                try:
                    # Extract title and URL
                    title_link = container.find('a', class_='result__a')
                    if not title_link:
                        continue
                        
                    title = title_link.get_text(strip=True)
                    url = title_link.get('href', '')
                    
                    # Extract snippet
                    snippet_elem = container.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    # Extract domain
                    domain_elem = container.find('span', class_='result__url')
                    domain = domain_elem.get_text(strip=True) if domain_elem else ""
                    
                    if title and url:
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "domain": domain,
                            "rank": len(results) + 1
                        })
                        
                except Exception as e:
                    logger.warning(f"Error parsing search result: {e}")
                    continue
            
            # If HTML parsing didn't work well, try alternative approach
            if not results:
                results = await _fallback_search(query, num_results, client)
            
            return results
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during search: {e.response.status_code}")
        return []
    except Exception as e:
        logger.error(f"Error during DuckDuckGo search: {e}")
        return []

async def _fallback_search(query: str, num_results: int, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """Fallback search method using DuckDuckGo instant answer API."""
    
    try:
        # Use DuckDuckGo instant answer API as fallback
        api_url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        response = await client.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        
        # Extract results from various sections
        if data.get("RelatedTopics"):
            for topic in data["RelatedTopics"][:num_results]:
                if isinstance(topic, dict) and "Text" in topic and "FirstURL" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:100] + "..." if len(topic.get("Text", "")) > 100 else topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                        "domain": urllib.parse.urlparse(topic.get("FirstURL", "")).netloc,
                        "rank": len(results) + 1
                    })
        
        # Add abstract if available
        if data.get("Abstract") and data.get("AbstractURL"):
            results.insert(0, {
                "title": data.get("AbstractSource", "DuckDuckGo"),
                "url": data.get("AbstractURL"),
                "snippet": data.get("Abstract"),
                "domain": urllib.parse.urlparse(data.get("AbstractURL", "")).netloc,
                "rank": 1
            })
            
        return results[:num_results]
        
    except Exception as e:
        logger.error(f"Fallback search failed: {e}")
        return []
