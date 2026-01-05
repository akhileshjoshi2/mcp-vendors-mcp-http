"""
Weather Tool

Provides weather information using OpenWeatherMap API.
Enables retrieval of current weather, forecasts, and weather alerts.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
import httpx
from datetime import datetime
from fastmcp import tool
from ..config import Config

logger = logging.getLogger(__name__)

@tool()
async def get_weather(
    location: str,
    weather_type: str = "current",
    units: Optional[str] = None,
    include_forecast_days: int = 5
) -> Dict[str, Any]:
    """
    Get weather information for a specified location.
    
    Args:
        location: Location name, city, or coordinates (lat,lon)
        weather_type: Type of weather data ("current", "forecast", "alerts")
        units: Temperature units ("metric", "imperial", "kelvin")
        include_forecast_days: Number of forecast days (1-16, only for forecast type)
        
    Returns:
        Dict containing weather data, location info, and metadata
        
    Example:
        # Current weather
        weather = await get_weather("New York", "current")
        
        # 5-day forecast
        forecast = await get_weather("London", "forecast", units="metric")
        
        # Weather with coordinates
        weather = await get_weather("40.7128,-74.0060", "current")
    """
    try:
        config = Config()
        weather_config = config.get_weather_config()
        
        # Use provided units or fall back to config default
        units = units or weather_config["units"]
        
        logger.info(f"Getting {weather_type} weather for: {location}")
        
        # Parse location (could be city name or coordinates)
        location_data = await _resolve_location(location, weather_config)
        
        if weather_type == "current":
            weather_data = await _get_current_weather(location_data, units, weather_config)
        elif weather_type == "forecast":
            weather_data = await _get_forecast_weather(location_data, units, include_forecast_days, weather_config)
        elif weather_type == "alerts":
            weather_data = await _get_weather_alerts(location_data, weather_config)
        else:
            raise ValueError(f"Unsupported weather type: {weather_type}")
        
        result = {
            "location": location_data,
            "weather_type": weather_type,
            "units": units,
            "data": weather_data,
            "timestamp": datetime.utcnow().isoformat(),
            "provider": "OpenWeatherMap"
        }
        
        logger.info(f"Weather data retrieved successfully for {location}")
        return result
        
    except Exception as e:
        logger.error(f"Weather request failed: {e}")
        return {
            "error": str(e),
            "location": location,
            "weather_type": weather_type,
            "data": None
        }

async def _resolve_location(location: str, config: dict) -> Dict[str, Any]:
    """Resolve location string to coordinates and location info."""
    
    # Check if location is already coordinates (lat,lon)
    if "," in location and len(location.split(",")) == 2:
        try:
            lat, lon = map(float, location.split(","))
            return {
                "name": f"Coordinates ({lat}, {lon})",
                "lat": lat,
                "lon": lon,
                "country": "Unknown"
            }
        except ValueError:
            pass  # Not valid coordinates, treat as city name
    
    # Use geocoding API to resolve city name to coordinates
    geocoding_url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {
        "q": location,
        "limit": 1,
        "appid": config["api_key"]
    }
    
    async with httpx.AsyncClient(timeout=config["timeout"]) as client:
        response = await client.get(geocoding_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            raise ValueError(f"Location '{location}' not found")
        
        location_info = data[0]
        return {
            "name": location_info.get("name", location),
            "lat": location_info["lat"],
            "lon": location_info["lon"],
            "country": location_info.get("country", "Unknown"),
            "state": location_info.get("state", "")
        }

async def _get_current_weather(location_data: dict, units: str, config: dict) -> Dict[str, Any]:
    """Get current weather data."""
    
    url = f"{config['base_url']}/weather"
    params = {
        "lat": location_data["lat"],
        "lon": location_data["lon"],
        "units": units,
        "appid": config["api_key"]
    }
    
    async with httpx.AsyncClient(timeout=config["timeout"]) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract and format relevant weather data
        weather_info = {
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "description": data["weather"][0]["description"].title(),
            "main": data["weather"][0]["main"],
            "icon": data["weather"][0]["icon"],
            "visibility": data.get("visibility", 0) / 1000,  # Convert to km
            "uv_index": None,  # Not available in current weather endpoint
            "wind": {
                "speed": data["wind"]["speed"],
                "direction": data["wind"].get("deg", 0),
                "gust": data["wind"].get("gust", 0)
            },
            "clouds": data["clouds"]["all"],
            "sunrise": datetime.fromtimestamp(data["sys"]["sunrise"]).isoformat(),
            "sunset": datetime.fromtimestamp(data["sys"]["sunset"]).isoformat()
        }
        
        # Add rain/snow data if available
        if "rain" in data:
            weather_info["rain"] = data["rain"]
        if "snow" in data:
            weather_info["snow"] = data["snow"]
            
        return weather_info

async def _get_forecast_weather(location_data: dict, units: str, days: int, config: dict) -> Dict[str, Any]:
    """Get weather forecast data."""
    
    # Use 5-day forecast endpoint (free tier) or One Call API (requires subscription)
    url = f"{config['base_url']}/forecast"
    params = {
        "lat": location_data["lat"],
        "lon": location_data["lon"],
        "units": units,
        "appid": config["api_key"]
    }
    
    async with httpx.AsyncClient(timeout=config["timeout"]) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Process forecast data
        forecasts = []
        current_date = None
        daily_data = {}
        
        for item in data["list"]:
            forecast_date = datetime.fromtimestamp(item["dt"]).date()
            
            # Group by day and take representative values
            if forecast_date != current_date:
                if current_date and daily_data:
                    forecasts.append(daily_data)
                
                current_date = forecast_date
                daily_data = {
                    "date": forecast_date.isoformat(),
                    "temperature": {
                        "min": item["main"]["temp_min"],
                        "max": item["main"]["temp_max"],
                        "avg": item["main"]["temp"]
                    },
                    "description": item["weather"][0]["description"].title(),
                    "main": item["weather"][0]["main"],
                    "icon": item["weather"][0]["icon"],
                    "humidity": item["main"]["humidity"],
                    "pressure": item["main"]["pressure"],
                    "wind_speed": item["wind"]["speed"],
                    "clouds": item["clouds"]["all"],
                    "precipitation_probability": item.get("pop", 0) * 100
                }
                
                # Add rain/snow data if available
                if "rain" in item:
                    daily_data["rain"] = item["rain"]
                if "snow" in item:
                    daily_data["snow"] = item["snow"]
            else:
                # Update min/max temperatures for the same day
                if daily_data:
                    daily_data["temperature"]["min"] = min(daily_data["temperature"]["min"], item["main"]["temp_min"])
                    daily_data["temperature"]["max"] = max(daily_data["temperature"]["max"], item["main"]["temp_max"])
        
        # Add the last day if it exists
        if daily_data:
            forecasts.append(daily_data)
        
        return {
            "forecast_days": len(forecasts),
            "forecasts": forecasts[:days]
        }

async def _get_weather_alerts(location_data: dict, config: dict) -> Dict[str, Any]:
    """Get weather alerts and warnings."""
    
    # Weather alerts are available through One Call API (requires subscription)
    # For free tier, we'll return a message about availability
    
    return {
        "alerts": [],
        "message": "Weather alerts require OpenWeatherMap One Call API subscription",
        "alternative": "Check local weather services for current alerts and warnings"
    }
