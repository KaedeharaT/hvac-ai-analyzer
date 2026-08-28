from __future__ import annotations

import requests


def fetch_open_meteo(latitude: float, longitude: float, start: str, end: str) -> dict | None:
    try:
        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={"latitude": latitude, "longitude": longitude,
                    "start_date": start, "end_date": end,
                    "hourly": "temperature_2m,relative_humidity_2m"},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None
