"""
One-off fetch for the dashboard.

Pulls two things:
1. London LSOA 2021 boundaries from the ONS Open Geography Portal
   (ArcGIS REST). Spatial-filtered to a Greater London bbox, then
   restricted to the LSOAs in the team's Phase 5 output so the codes
   match the model 1:1.
2. Police stations in Greater London from OpenStreetMap, via Overpass.

Both end up in dashboard_assets/.

Run once:  uv run python fetch_dashboard_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

ROOT = Path(__file__).parent
ASSETS = ROOT / "dashboard_assets"
ASSETS.mkdir(exist_ok=True)

ONS_LSOA21_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4/"
    "FeatureServer/0/query"
)
LONDON_BBOX = {"xmin": -0.51, "ymin": 51.28, "xmax": 0.34, "ymax": 51.69,
               "spatialReference": {"wkid": 4326}}

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


def _team_lsoa_set() -> set[str] | None:
    """Load the team's Phase 5 LSOA list if available, to restrict the
    boundary set to exactly what the model covers."""
    for p in [ROOT / "team_model" / "phase5_clusters.parquet",
              ROOT.parent / "phase5" / "phase5_clusters.parquet",
              ROOT / "phase5" / "phase5_clusters.parquet"]:
        if p.exists():
            df = pd.read_parquet(p, columns=["lsoa21cd"])
            return set(df["lsoa21cd"].unique())
    return None


def fetch_lsoa_boundaries() -> Path:
    out_path = ASSETS / "london_lsoa.geojson"
    if out_path.exists():
        size = out_path.stat().st_size / 1_000_000
        print(f"[lsoa] already exists at {out_path} ({size:.1f} MB), skipping")
        return out_path

    print("[lsoa] querying ONS for London LSOA 2021 boundaries...")
    features: list[dict] = []
    offset = 0
    page_size = 2000
    while True:
        params = {
            "where": "1=1",
            "geometry": json.dumps(LONDON_BBOX),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "LSOA21CD,LSOA21NM",
            "outSR": "4326",
            "f": "geojson",
            "resultRecordCount": page_size,
            "resultOffset": offset,
        }
        r = requests.get(ONS_LSOA21_URL, params=params, timeout=90)
        r.raise_for_status()
        gj = r.json()
        page = gj.get("features", [])
        features.extend(page)
        print(f"  fetched {len(features)} so far...")
        if len(page) < page_size:
            break
        offset += page_size

    print(f"[lsoa] bbox returned {len(features)} polygons")

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    if "LSOA21CD" not in gdf.columns:
        raise RuntimeError(f"missing LSOA21CD, got columns: {list(gdf.columns)}")

    team_set = _team_lsoa_set()
    if team_set:
        before = len(gdf)
        gdf = gdf[gdf["LSOA21CD"].isin(team_set)].reset_index(drop=True)
        print(f"[lsoa] filtered to team's Phase 5 set: "
              f"{before} -> {len(gdf)} polygons")

    gdf["geometry"] = gdf.geometry.simplify(
        tolerance=0.0005, preserve_topology=True
    )
    gdf = gdf[["geometry", "LSOA21CD", "LSOA21NM"]]
    gdf.to_file(out_path, driver="GeoJSON")
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
