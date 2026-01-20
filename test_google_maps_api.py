import os
import json
import googlemaps
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def test_google_maps_data(restaurant_name, location):
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("Error: GOOGLE_MAPS_API_KEY not found in environment.")
        return

    client = googlemaps.Client(key=api_key)
    query = f"{restaurant_name} restaurant {location}"
    
    print(f"\n" + "="*80)
    print(f"SEARCHING FOR: {query}")
    print("="*80)
    
    # This is what find_restaurant uses
    search_results = client.places(query=query)
    
    if not search_results.get("results"):
        print("No results found.")
        return

    # Let's look at the first result
    first_place = search_results["results"][0]
    place_id = first_place["place_id"]
    
    print("\n[RAW SEARCH RESULT FROM .places()]")
    print(json.dumps(first_place, indent=2))

    print(f"\n--- FETCHING FULL DETAILS FOR: {first_place.get('name')} (ID: {place_id}) ---")
    
    # These are all the fields we can get. 
    # Note: 'editorial_summary' is critical for avoiding hardcoding!
    details = client.place(
        place_id=place_id,
        fields=[
            "name",
            "formatted_address",
            "editorial_summary",  
            "type",              
            "price_level",
            "rating",
            "user_ratings_total",
            "reviews",            
            "website",
            "opening_hours",
            "geometry",
            "serves_wine",
            "serves_beer",
            "serves_vegetarian_food",
            "serves_dinner",
            "serves_lunch",
        ]
    )

    print("\n[RAW PLACE DETAILS FROM .place()]")
    result = details.get("result", {})
    print(json.dumps(result, indent=2))
    
    if "editorial_summary" in result:
        print(f"\n[!] DYNAMIC CUISINE HINT: {result['editorial_summary'].get('overview')}")
    else:
        print("\n[!] No editorial summary found for this place.")

if __name__ == "__main__":
    # Test 1: South Indian specificity
    test_google_maps_data("godavari", "woburn")
    


