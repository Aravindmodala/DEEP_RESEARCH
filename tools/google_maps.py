"""Google Maps Places API integration for restaurant research.

This module uses ONLY the Places API - no Geocoding or Distance Matrix APIs required.
Distances are calculated using the Haversine formula from lat/lng coordinates.
"""

from __future__ import annotations

import math
import os
from typing import Any

import googlemaps
from langchain_core.tools import tool
from loguru import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from models.schemas import CompetitorInfo


class PlaceSearchResult(BaseModel):
    """Structured result from place search."""

    name: str
    address: str
    place_id: str
    rating: float | None = None
    user_ratings_total: int | None = None
    price_level: int | None = None
    types: list[str] = Field(default_factory=list)
    lat: float | None = None
    lng: float | None = None


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate distance between two points using Haversine formula.
    
    Args:
        lat1, lng1: First point coordinates
        lat2, lng2: Second point coordinates
        
    Returns:
        Distance in miles
    """
    R = 3959  # Earth's radius in miles
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)
    
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return round(R * c, 2)


class GoogleMapsTools:
    """
    Google Maps API wrapper using ONLY Places API.
    
    No Geocoding API or Distance Matrix API required.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY not found in environment")
        self.client = googlemaps.Client(key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def find_restaurant(
        self, restaurant_name: str, location: str
    ) -> PlaceSearchResult | None:
        """
        Find a restaurant using Places API Text Search.
        
        Args:
            restaurant_name: Name of the restaurant
            location: City, State (e.g., "Austin, TX")
            
        Returns:
            PlaceSearchResult with restaurant details including lat/lng
        """
        try:
            # Use text search - location is included in query
            query = f"{restaurant_name} restaurant {location}"
            logger.info(f"[GOOGLE MAPS] Searching for: {query}")
            
            result = self.client.places(query=query)

            if result.get("results"):
                place = result["results"][0]
                geometry = place.get("geometry", {})
                location_data = geometry.get("location", {})
                
                return PlaceSearchResult(
                    name=place.get("name", ""),
                    address=place.get("formatted_address", ""),
                    place_id=place.get("place_id", ""),
                    rating=place.get("rating"),
                    user_ratings_total=place.get("user_ratings_total"),
                    price_level=place.get("price_level"),
                    types=place.get("types", []),
                    lat=location_data.get("lat"),
                    lng=location_data.get("lng"),
                )
            return None
        except Exception as e:
            logger.error(f"[GOOGLE MAPS] Error finding restaurant: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def find_competitors(
        self,
        cuisine_type: str,
        location: str,
        center_lat: float | None = None,
        center_lng: float | None = None,
        radius_meters: int = 3000,
        max_results: int = 20,
    ) -> list[PlaceSearchResult]:
        """
        Find competitor restaurants using Places API Text Search.
        
        Args:
            cuisine_type: Type of cuisine (e.g., "Mexican", "Italian")
            location: City, State for text search
            center_lat: Optional center latitude for radius search
            center_lng: Optional center longitude for radius search
            radius_meters: Search radius in meters (default 3000 = ~2 miles)
            max_results: Maximum results to return
            
        Returns:
            List of PlaceSearchResult for competitors
        """
        try:
            query = f"{cuisine_type} restaurants {location}"
            logger.info(f"[GOOGLE MAPS] Finding competitors: {query}")
            
            # If we have center coordinates, use location bias
            if center_lat and center_lng:
                result = self.client.places(
                    query=query,
                    location=(center_lat, center_lng),
                    radius=radius_meters,
                )
            else:
                # Just use text search with location in query
                result = self.client.places(query=query)

            competitors = []
            for place in result.get("results", [])[:max_results]:
                geometry = place.get("geometry", {})
                location_data = geometry.get("location", {})
                
                competitors.append(
                    PlaceSearchResult(
                        name=place.get("name", ""),
                        address=place.get("vicinity", place.get("formatted_address", "")),
                        place_id=place.get("place_id", ""),
                        rating=place.get("rating"),
                        user_ratings_total=place.get("user_ratings_total"),
                        price_level=place.get("price_level"),
                        types=place.get("types", []),
                        lat=location_data.get("lat"),
                        lng=location_data.get("lng"),
                    )
                )

            logger.info(f"[GOOGLE MAPS] Found {len(competitors)} competitors")
            return competitors
        except Exception as e:
            logger.error(f"[GOOGLE MAPS] Error finding competitors: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def get_place_details(self, place_id: str) -> dict[str, Any]:
        """Get detailed information about a place."""
        try:
            result = self.client.place(
                place_id=place_id,
                fields=[
                    "name",
                    "formatted_address",
                    "formatted_phone_number",
                    "website",
                    "rating",
                    "user_ratings_total",
                    "price_level",
                    "opening_hours",
                    "reviews",
                    "url",
                ],
            )
            return result.get("result", {})
        except Exception as e:
            logger.error(f"[GOOGLE MAPS] Error getting place details: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def get_place_reviews(
        self, place_id: str, max_reviews: int = 5
    ) -> list[dict[str, Any]]:
        """Get reviews for a specific place."""
        try:
            details = self.get_place_details(place_id)
            reviews = details.get("reviews", [])[:max_reviews]
            return [
                {
                    "author": r.get("author_name", "Anonymous"),
                    "rating": r.get("rating"),
                    "text": r.get("text", ""),
                    "time": r.get("relative_time_description", ""),
                }
                for r in reviews
            ]
        except Exception as e:
            logger.error(f"[GOOGLE MAPS] Error getting reviews: {e}")
            return []

    def _price_level_to_string(self, level: int | None) -> str | None:
        """Convert numeric price level to $ symbols."""
        if level is None:
            return None
        mapping = {0: "$", 1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}
        return mapping.get(level, "$$")

    def build_competitor_list(
        self,
        target_restaurant: str,
        location: str,
        cuisine_type: str = "restaurant",
        radius_meters: int = 3000,
    ) -> tuple[PlaceSearchResult | None, list[CompetitorInfo]]:
        """
        Full workflow: find target, find competitors, calculate distances.
        Uses ONLY Places API + Haversine formula.
        
        Returns (target, list of competitors with distances).
        """
        # Find target restaurant
        target = self.find_restaurant(target_restaurant, location)
        if not target:
            logger.warning(f"[GOOGLE MAPS] Could not find: {target_restaurant}")
            return None, []

        logger.info(f"[GOOGLE MAPS] Found target: {target.name} at ({target.lat}, {target.lng})")

        # Infer cuisine type from target's types
        inferred_cuisine = cuisine_type
        if target.types:
            cuisine_keywords = [
                "italian", "mexican", "chinese", "japanese", "indian", "thai",
                "korean", "french", "american", "mediterranean", "vietnamese",
                "greek", "spanish", "fast_food", "pizza", "burger", "seafood",
            ]
            for t in target.types:
                t_lower = t.lower().replace("_", " ")
                for kw in cuisine_keywords:
                    if kw in t_lower:
                        inferred_cuisine = kw
                        break

        # Find nearby competitors using text search with location bias
        nearby = self.find_competitors(
            cuisine_type=inferred_cuisine,
            location=location,
            center_lat=target.lat,
            center_lng=target.lng,
            radius_meters=radius_meters,
        )

        # Build competitor list with distances calculated via Haversine
        competitors: list[CompetitorInfo] = []
        for place in nearby:
            # Skip the target restaurant itself
            if place.place_id == target.place_id:
                continue

            # Calculate distance using Haversine formula
            distance = 0.0
            if target.lat and target.lng and place.lat and place.lng:
                distance = haversine_distance(
                    target.lat, target.lng,
                    place.lat, place.lng
                )

            # Get website from details if available
            website = None
            try:
                details = self.get_place_details(place.place_id)
                website = details.get("website")
            except Exception:
                pass

            competitors.append(
                CompetitorInfo(
                    name=place.name,
                    address=place.address,
                    distance_miles=distance,
                    rating=place.rating,
                    review_count=place.user_ratings_total,
                    price_level=self._price_level_to_string(place.price_level),
                    cuisine_type=inferred_cuisine,
                    website=website,
                    place_id=place.place_id,
                )
            )

        # Sort by distance
        competitors.sort(key=lambda c: c.distance_miles)

        logger.info(f"[GOOGLE MAPS] Built list of {len(competitors)} competitors")
        return target, competitors


# =============================================================================
# LangChain Tool Wrappers
# =============================================================================


def create_google_maps_tools(api_key: str | None = None) -> list:
    """Create LangChain-compatible tools from GoogleMapsTools."""
    maps = GoogleMapsTools(api_key)

    @tool
    def find_restaurant(restaurant_name: str, location: str) -> str:
        """
        Find a specific restaurant by name and location using Google Places API.
        Returns restaurant details including address, rating, and coordinates.
        
        Args:
            restaurant_name: Name of the restaurant to find
            location: City, State location (e.g., "Austin, TX")
        """
        result = maps.find_restaurant(restaurant_name, location)
        if result:
            return result.model_dump_json(indent=2)
        return "Restaurant not found"

    @tool
    def find_competitors(
        restaurant_name: str,
        location: str,
        cuisine_type: str = "restaurant",
        radius_miles: float = 2.0,
    ) -> str:
        """
        Find competitor restaurants near a target restaurant.
        Returns a list of competitors with ratings, distances, and price levels.
        
        Args:
            restaurant_name: Name of the target restaurant
            location: City, State location
            cuisine_type: Type of cuisine to search for (e.g., "Mexican", "Italian")
            radius_miles: Search radius in miles (default 2.0)
        """
        radius_meters = int(radius_miles * 1609.34)
        target, competitors = maps.build_competitor_list(
            target_restaurant=restaurant_name,
            location=location,
            cuisine_type=cuisine_type,
            radius_meters=radius_meters,
        )

        output = {
            "target": target.model_dump() if target else None,
            "competitors": [c.model_dump() for c in competitors[:15]],
            "total_found": len(competitors),
        }
        return str(output)

    @tool
    def get_restaurant_reviews(place_id: str, max_reviews: int = 5) -> str:
        """
        Get customer reviews for a restaurant using its Google Place ID.
        
        Args:
            place_id: Google Place ID of the restaurant
            max_reviews: Maximum number of reviews to return (default 5)
        """
        reviews = maps.get_place_reviews(place_id, max_reviews)
        return str(reviews)

    @tool
    def get_restaurant_details(place_id: str) -> str:
        """
        Get detailed information about a restaurant including hours, website, phone.
        
        Args:
            place_id: Google Place ID of the restaurant
        """
        details = maps.get_place_details(place_id)
        return str(details)

    return [find_restaurant, find_competitors, get_restaurant_reviews, get_restaurant_details]
