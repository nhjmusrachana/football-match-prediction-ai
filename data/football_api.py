import requests

API_KEY = "e0377f4de9bf43c6a763ca4d8e97ae42"

headers = {
    "X-Auth-Token": API_KEY
}

url = "https://api.football-data.org/v4/competitions/PL/standings"

response = requests.get(url, headers=headers)

data = response.json()

# PRINT TEAM STANDINGS
for team in data["standings"][0]["table"]:
    print(
        team["position"],
        team["team"]["name"],
        "- Points:",
        team["points"]
    )