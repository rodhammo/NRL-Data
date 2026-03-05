"""Quick check of what data the NRL draw API provides for team lists."""
import requests, json
from bs4 import BeautifulSoup

url = "https://www.nrl.com/draw/?competition=111&round=1&season=2026"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")
div = soup.find("div", {"id": "vue-draw"})
raw = div["q-data"].replace("&quot;", '"')
data = json.loads(raw)

fixtures = data.get("fixtures", [])
f = fixtures[2]  # Storm v Eels
print("Fixture keys:", list(f.keys()))
print()

ht = f.get("homeTeam", {})
print("Home team keys:", list(ht.keys()))
print("Home:", ht.get("nickName"))
print()

at = f.get("awayTeam", {})
print("Away:", at.get("nickName"))
print()

# Check match centre URL
print("Match URL:", f.get("matchCentreUrl"))

# Look for any player/squad data in the entire fixture
def find_keys(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(word in k.lower() for word in ["player", "squad", "lineup", "team_list", "list"]):
                print(f"  Found key: {path}.{k} = {repr(v)[:200]}")
            find_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj[:2]):
            find_keys(item, f"{path}[{i}]")

print("\nSearching for player/squad keys in fixture data:")
find_keys(f)
