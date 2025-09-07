# MAKE SURE TO PIP INSTALL SODAPY
from sodapy import Socrata
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = "Datasets"

# Initialize client
client = Socrata("data.cityofchicago.org", None)

# Census Socioeconomics
# results = client.get("kn9c-c2s2", limit=1000)
# df_census = pd.DataFrame.from_records(results)

offset = 0
limit = 10000
two_years_ago = (datetime.now() - timedelta(days=2*365)).strftime("%Y-%m-%dT%H:%M:%S")

crimes = []

print("Fetching crime data...")
while True:
    batch = client.get("ijzp-q8t2", where=f"date >= '{two_years_ago}'", limit=limit, offset=offset)
    if not batch:
        break
    crimes.extend(batch)
    offset += limit
    print(f"Fetched {offset} records so far")

df_crimes = pd.DataFrame.from_records(crimes)
df_crimes.to_csv(f"{DATA_DIR}/crimes_last2years.csv", index=False)

try:
    print("Fetching community boundaries...")
    boundaries = client.get("igwz-8jzy")
    df_boundaries = pd.DataFrame.from_records(boundaries)
    df_boundaries.to_csv(f"{DATA_DIR}/community_boundaries.csv", index=False)
    print(f"Fetched {len(boundaries)} community boundaries")
except Exception as e:
    print(f"Error fetching community boundaries: {e}")

acs_data = []
offset = 0 # Reset offset for ACS data

print("Fetching ACS data...")
while True:
    batch = client.get("t68z-cikk", limit=limit, offset=offset)
    if not batch:
        break
    acs_data.extend(batch)
    offset += limit
    print(f"Fetched {offset} records so far")

df_acs = pd.DataFrame.from_records(acs_data)
df_acs.to_csv(f"{DATA_DIR}/acs_by_community.csv", index=False)
