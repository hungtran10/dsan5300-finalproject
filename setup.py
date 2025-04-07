import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NPS_API_KEY")
BASE_URL = "https://developer.nps.gov/api/v1"

endpoints = {
    "parks": "/parks",
    "events": "/events",
    "alerts": "/alerts"
}

os.makedirs("data", exist_ok=True)

def fetch_all_data(endpoint, max_results=1000):
    all_data = []
    start = 0
    while True:
        params = {
            "api_key": API_KEY,
            "start": start,
            "limit": 50
        }
        response = requests.get(BASE_URL + endpoint, params=params)
        response.raise_for_status()
        result = response.json()
        data = result.get("data", [])
        all_data.extend(data)
        if len(data) < 50 or len(all_data) >= max_results:
            break
        start += 50
    return all_data

# Fetch and save each dataset
for name, endpoint in endpoints.items():
    print(f"Fetching {name} data...")
    data = fetch_all_data(endpoint)
    df = pd.json_normalize(data)
    df.to_csv(f"data/{name}.csv", index=False)
    print(f"Saved {name}.csv with {len(df)} records.")

print("All data pulled and saved to /data.")
