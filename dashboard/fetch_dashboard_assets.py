"""
One-off fetch for the dashboard.

Pulls two things:
1. London LSOA 2011 boundaries: 33 borough files from martinjc/UK-GeoJSON,
   merged and simplified.
2. Police stations in Greater London from OpenStreetMap, via Overpass.

Both end up in dashboard_assets/.

Run once:  uv run python fetch_dashboard_assets.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape

ASSETS = Path(__file__).parent / "dashboard_assets"
ASSETS.mkdir(exist_ok=True)

LONDON_LADS = [
    "E09000001", "E09000002", "E09000003", "E09000004", "E09000005",
    "E09000006", "E09000007", "E09000008", "E09000009", "E09000010",
    "E09000011", "E09000012", "E09000013", "E09000014", "E09000015",
    "E09000016", "E09000017", "E09000018", "E09000019", "E09000020",
    "E09000021", "E09000022", "E09000023", "E09000024", "E09000025",
    "E09000026", "E09000027", "E09000028", "E09000029", "E09000030",
    "E09000031", "E09000032", "E09000033",
]

LSOA_BASE = (
    "https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/"
    "json/statistical/eng/lsoa_by_lad/{lad}.json"
)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_QUERY = """
[out:json][timeout:60];
(
  node["amenity"="police"](51.28,-0.51,51.69,0.34);
  way["amenity"="police"](51.28,-0.51,51.69,0.34);
  relation["amenity"="police"](51.28,-0.51,51.69,0.34);
);
out center tags;
"""


def fetch_lsoa_boundaries() -> Path:
    out_path = ASSETS / "london_lsoa.geojson"
    if out_path.exists():
        size = out_path.stat().st_size / 1_000_000
        print(f"[lsoa] already exists at {out_path} ({size:.1f} MB), skipping")
        return out_path

    print(f"[lsoa] fetching {len(LONDON_LADS)} London LAD files...")
    frames = []
    for i, lad in enumerate(LONDON_LADS, 1):
        url = LSOA_BASE.format(lad=lad)
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        gj = r.json()
        gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
        frames.append(gdf)
        print(f"  [{i:2d}/33] {lad}: {len(gdf)} LSOAs")
        time.sleep(0.05)

    london = pd.concat(frames, ignore_index=True)
    london = gpd.GeoDataFrame(london, crs="EPSG:4326")
    print(f"[lsoa] merged: {len(london)} polygons")

    london["geometry"] = london.geometry.simplify(
        tolerance=0.0005, preserve_topology=True
    )

    code_col = next(
        (c for c in ["LSOA11CD", "lsoa11cd", "code", "LSOA_CODE"] if c in london.columns),
        None,
    )
    name_col = next(
        (c for c in ["LSOA11NM", "lsoa11nm", "name", "LSOA_NAME"] if c in london.columns),
        None,
    )
    if code_col is None:
        print(f"[lsoa] WARNING: no code column found, columns={list(london.columns)}")
        sys.exit(1)
    keep = ["geometry", code_col]
    if name_col:
        keep.append(name_col)
    london = london[keep].rename(columns={code_col: "LSOA11CD", name_col or "": "LSOA11NM"})

    london.to_file(out_path, driver="GeoJSON")
    size = out_path.stat().st_size / 1_000_000
    print(f"[lsoa] saved {out_path} ({size:.1f} MB)")
    return out_path


def fetch_police_stations() -> Path:
    out_path = ASSETS / "london_police_stations.geojson"
    if out_path.exists():
        size = out_path.stat().st_size / 1_000
        print(f"[police] already exists at {out_path} ({size:.0f} KB), skipping")
        return out_path

    print("[police] querying Overpass for amenity=police in Greater London...")
    headers = {
        "User-Agent": "tu-e-cbl-group-16-dashboard/0.1 (academic project)",
        "Accept": "application/json",
    }
    r = requests.post(
        OVERPASS_URL,
        data={"data": OVERPASS_QUERY},
        headers=headers,
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()

    features = []
    for el in data.get("elements", []):
        if el["type"] == "node":
            lon, lat = el["lon"], el["lat"]
        elif "center" in el:
            lon, lat = el["center"]["lon"], el["center"]["lat"]
        else:
            continue
        tags = el.get("tags", {})
        name = tags.get("name") or tags.get("operator") or "Police station"
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "name": name,
                "operator": tags.get("operator", ""),
                "osm_id": el.get("id"),
                "osm_type": el["type"],
            },
        })

    fc = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(fc))
    print(f"[police] saved {len(features)} stations to {out_path}")
    return out_path


if __name__ == "__main__":
    lsoa_path = fetch_lsoa_boundaries()
    police_path = fetch_police_stations()
    print()
    print("Done. Files:")
    print(f"  {lsoa_path}")
    print(f"  {police_path}")
