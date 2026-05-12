"""
CBL Group 16 demand dashboard.

Reads the Phase 5 k=6 cluster output and renders an interactive Greater
London map: choropleth by risk score, top-10 hotspot leaderboard,
click-to-detail with the six model drivers, tier distribution chart,
and a coverage-gaps table built from Met Police station locations.

Run from the repo root:
    pip install -r dashboard/requirements.txt
    cd dashboard
    python fetch_dashboard_assets.py   # one-off: LSOA boundaries + stations
    streamlit run 10_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

ROOT = Path(__file__).parent
ASSETS = ROOT / "dashboard_assets"

LSOA_GEOJSON = ASSETS / "london_lsoa.geojson"
POLICE_GEOJSON = ASSETS / "london_police_stations.geojson"
STYLE_CSS = ASSETS / "style.css"


def _find_parquet(name: str) -> Path:
    """Look for a team-model parquet in either layout: in-repo (../phase5/X)
    or the dev sandbox (team_model/X)."""
    candidates = [
        ROOT.parent / "phase5" / name,
        ROOT / "team_model" / name,
        ROOT / "phase5" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


PHASE5_PARQUET = _find_parquet("phase5_clusters.parquet")
PROFILE_CSV = _find_parquet("phase5_cluster_profiles.csv")

# k=6 tier labels, ordered highest demand to lowest.
# Names picked to read clearly on a briefing slide.
TIER_LABEL = {
    1: "Hotspot",
    2: "High demand",
    3: "Elevated",
    4: "Steady",
    5: "Light",
    6: "Quiet",
}
TIER_COLOR = {
    1: "#7F1D1D",
    2: "#DC2626",
    3: "#F97316",
    4: "#FBBF24",
    5: "#A7F3D0",
    6: "#34D399",
}
ACTION_TIERS = {1, 2}  # tiers that get coverage-gap flagging

# How the team's six final features map to readable labels.
FEATURES = {
    "severity_weighted_count": "Severity-weighted crime",
    "seasonal_volatility": "Seasonal volatility",
    "stop_search_rate": "Stop-and-search rate",
    "employment_deprivation": "Employment deprivation",
    "resolution_rate": "Resolution rate",
    "total_footfall": "Footfall (TfL)",
}

COVERAGE_THRESHOLD_KM = 1.5


# ----- loaders -----

@st.cache_data(show_spinner="Loading LSOA boundaries…")
def load_lsoa_geojson() -> dict:
    return json.loads(LSOA_GEOJSON.read_text())


@st.cache_data(show_spinner="Loading police stations…")
def load_police_geojson() -> dict:
    return json.loads(POLICE_GEOJSON.read_text())


@st.cache_data(show_spinner="Loading team Phase 5 model…")
def load_phase5() -> pd.DataFrame:
    df = pd.read_parquet(PHASE5_PARQUET)
    df = df.rename(columns={"lsoa21cd": "lsoa", "lsoa21nm": "lsoa_name",
                            "lad22nm": "borough"})
    df["tier_label"] = df["tier"].map(TIER_LABEL)
    return df


@st.cache_data
def load_tier_profile() -> pd.DataFrame:
    if not PROFILE_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(PROFILE_CSV)


def _polygon_centroid(geom: dict) -> tuple[float | None, float | None]:
    """Mean of the outer ring vertices. Good enough at LSOA scale."""
    g_type = geom.get("type")
    if g_type == "Polygon":
        ring = geom["coordinates"][0]
    elif g_type == "MultiPolygon":
        # Pick the largest sub-polygon's outer ring.
        ring = max(geom["coordinates"], key=lambda p: len(p[0]))[0]
    else:
        return None, None
    arr = np.asarray(ring, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


@st.cache_data(show_spinner="Computing station distances…")
def load_station_distances() -> pd.DataFrame:
    """LSOA to nearest police station, in km, with station name.

    Pure-numpy haversine, no geopandas: keeps the Streamlit Cloud
    memory footprint well under the 1 GB free-tier limit.
    """
    feats = load_lsoa_geojson()["features"]
    codes: list[str] = []
    cent_ll: list[tuple[float, float]] = []  # lat, lon
    for f in feats:
        code = f["properties"].get("LSOA21CD")
        lon, lat = _polygon_centroid(f["geometry"])
        if code is None or lon is None:
            continue
        codes.append(code)
        cent_ll.append((lat, lon))
    cent = np.array(cent_ll)  # [N, 2]

    stations_gj = load_police_geojson()
    sta_records = []
    sta_names: list[str] = []
    for f in stations_gj["features"]:
        lon, lat = f["geometry"]["coordinates"][:2]
        sta_records.append((lat, lon))
        sta_names.append(f["properties"].get("name", "Police"))
    sta = np.array(sta_records)  # [M, 2]

    # Vectorised haversine: N centroids x M stations.
    r_km = 6371.0
    lat1 = np.radians(cent[:, 0])[:, None]
    lat2 = np.radians(sta[:, 0])[None, :]
    dlat = lat2 - lat1
    dlon = np.radians(sta[:, 1][None, :] - cent[:, 1][:, None])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dists = 2 * r_km * np.arcsin(np.sqrt(a))

    nearest_idx = np.argmin(dists, axis=1)
    nearest_km = dists[np.arange(len(codes)), nearest_idx]
    nearest_names = [sta_names[i] for i in nearest_idx]
    return pd.DataFrame({
        "lsoa": codes,
        "dist_km": nearest_km,
        "station_name": nearest_names,
    })


# ----- allocation -----

def allocate_officers(df: pd.DataFrame, total_officers: int) -> pd.DataFrame:
    """Officers per LSOA proportional to risk_score_scaled, with a tier floor."""
    df = df.copy()
    weights = df["risk_score_scaled"].clip(lower=0)
    if weights.sum() <= 0:
        weights = pd.Series(np.ones(len(df)), index=df.index)
    df["officers_proposed"] = total_officers * weights / weights.sum()
    df["officers_equal"] = total_officers / len(df)
    df["officers_delta"] = df["officers_proposed"] - df["officers_equal"]
    return df


def make_recommendation(row: pd.Series, dist_km: float | None) -> str:
    bits: list[str] = []
    delta = row["officers_delta"]
    if row["tier"] == 1:
        bits.append(f"**Hotspot.** Add about {abs(delta):.0f} officers above an even split.")
    elif row["tier"] == 2:
        bits.append(f"**High demand.** Add about {abs(delta):.0f} officers above an even split.")
    elif row["tier"] == 3:
        bits.append(f"**Elevated.** Slightly above the London baseline ({delta:+.1f} officers).")
    elif row["tier"] == 4:
        bits.append("**Steady.** Around the London baseline.")
    elif row["tier"] == 5:
        bits.append(f"**Light.** Could free up about {abs(delta):.0f} officers for elsewhere.")
    else:
        bits.append(f"**Quiet.** Could free up about {abs(delta):.0f} officers for elsewhere.")

    has_dist = dist_km is not None and not pd.isna(dist_km)
    if has_dist:
        if dist_km > COVERAGE_THRESHOLD_KM and row["tier"] in ACTION_TIERS:
            bits.append(f"Coverage gap. Closest station is **{dist_km:.1f} km** away.")
        else:
            bits.append(f"Closest station {dist_km:.1f} km away.")
    else:
        bits.append("No boundary geometry found for this neighbourhood.")

    # Dominant feature relative to the London average, as a ratio rather
    # than a z-score so it reads cleanly in a briefing.
    panel = load_phase5()
    best_ratio, best_name = 1.0, None
    for col in FEATURES:
        mu = panel[col].mean()
        if mu and not np.isnan(mu) and mu > 0:
            ratio = row[col] / mu
            if ratio > best_ratio:
                best_ratio, best_name = ratio, FEATURES[col]
    if best_name and best_ratio >= 1.5:
        bits.append(f"Main driver: **{best_name}** ({best_ratio:.0f}× London average).")

    return "  \n".join(bits)


# ----- session state -----

if "selected_lsoa" not in st.session_state:
    st.session_state.selected_lsoa = None
if "last_consumed_click_sig" not in st.session_state:
    st.session_state.last_consumed_click_sig = None


# ----- UI -----

st.set_page_config(
    page_title="London policing demand dashboard",
    layout="wide",
)

if STYLE_CSS.exists():
    st.markdown(f"<style>{STYLE_CSS.read_text()}</style>", unsafe_allow_html=True)

st.markdown(
    '<div class="poc-banner">'
    "<b>Proof of concept by CBL Group 16.</b> A planning aid, not a "
    "deployment tool. Risk score per neighbourhood, built from 36 months "
    "of crime records, outcomes, stop-and-search, TfL footfall, deprivation, "
    "and weather. All 4,994 London LSOAs (2021 boundaries) included."
    "</div>",
    unsafe_allow_html=True,
)

st.title("London policing demand dashboard")
st.markdown(
    "<p class='muted'>"
    "Where the risk sits across London. 4,994 neighbourhoods scored and "
    "grouped into 6 tiers, busiest to quietest."
    "</p>",
    unsafe_allow_html=True,
)

# ----- load -----

phase5 = load_phase5()
dist_df = load_station_distances()
boroughs_all = sorted(phase5["borough"].dropna().unique())

# ----- sidebar -----

with st.sidebar:
    st.header("Controls")

    st.markdown("**Borough**")
    if st.button("All London", use_container_width=True,
                  type="primary" if not st.session_state.get("borough_sel") else "secondary"):
        st.session_state.borough_sel = []
        st.rerun()
    borough_sel = st.multiselect(
        "Filter to boroughs",
        boroughs_all,
        default=st.session_state.get("borough_sel", []),
        key="borough_sel",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Tiers visible**")
    tier_sel_default = list(TIER_LABEL.keys())
    visible_tiers = st.multiselect(
        "Show tiers",
        options=tier_sel_default,
        default=tier_sel_default,
        format_func=lambda t: f"{t}. {TIER_LABEL[t]}",
        label_visibility="collapsed",
    )
    qa, qb = st.columns(2)
    if qa.button("Hotspots only", use_container_width=True):
        st.session_state.visible_tiers_override = [1]
        st.rerun()
    if qb.button("Top 2 tiers", use_container_width=True):
        st.session_state.visible_tiers_override = [1, 2]
        st.rerun()
    if "visible_tiers_override" in st.session_state:
        visible_tiers = st.session_state.pop("visible_tiers_override")

    st.markdown("---")
    total_officers = st.slider(
        "Total officers (Met)",
        25_000, 40_000, 34_500, step=500,
        help="HMICFRS 2024 baseline is around 34,500.",
    )

    st.markdown("---")
    show_stations = st.checkbox("Show police stations", value=True)
    show_top_hotspots = st.checkbox("Highlight top 10 hotspots", value=True)
    confident_only = st.checkbox(
        "Hide tier-borderline cases",
        value=False,
        help="Hides neighbourhoods that sit between two tiers. "
             "Use this if you only want decisive cases.",
    )

# ----- compute -----

df = phase5.copy()
if borough_sel:
    df = df[df["borough"].isin(borough_sel)]
if visible_tiers:
    df = df[df["tier"].isin(visible_tiers)]
if confident_only:
    df = df[df["silhouette"] >= 0]

df = allocate_officers(df, total_officers)
df = df.merge(dist_df, on="lsoa", how="left")

# ----- KPI strip -----

n_total = len(df)
n_hotspot = int((df["tier"] == 1).sum())
n_action = int(df["tier"].isin(ACTION_TIERS).sum())
top_row = df.nlargest(1, "risk_score_scaled").iloc[0] if n_total else None
mean_risk = float(df["risk_score_scaled"].mean()) if n_total else 0.0
gap_count = int(
    (df["tier"].isin(ACTION_TIERS) & (df["dist_km"] > COVERAGE_THRESHOLD_KM)).sum()
)
officers_reallocated = int(df.loc[df["officers_delta"] > 0, "officers_delta"].sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Neighbourhoods in view", f"{n_total:,}")
k2.metric("Hotspots", f"{n_hotspot:,}",
          help="The busiest neighbourhoods in London (top tier).")
k3.metric("Average score", f"{mean_risk:.1f}",
          delta=f"top: {top_row['risk_score_scaled']:.1f}" if top_row is not None else None,
          delta_color="off",
          help="Risk score averaged across what's currently in view.")
k4.metric("Officers reallocated", f"{officers_reallocated:,}",
          help="Officers we'd move toward higher-demand neighbourhoods "
               "instead of spreading them evenly.")
k5.metric("Coverage gaps", f"{gap_count}",
          delta=f"more than {COVERAGE_THRESHOLD_KM} km from a station",
          delta_color="off",
          help="Hotspot or High-demand neighbourhoods with no station "
               "within walking distance.")

# ----- main layout -----

col_map, col_detail = st.columns([2, 1], gap="medium")

with col_map:
    risk_lookup = df.set_index("lsoa")["risk_score_scaled"].to_dict()
    tier_lookup = df.set_index("lsoa")["tier"].to_dict()
    label_lookup = df.set_index("lsoa")["tier_label"].to_dict()
    visible_set = set(df["lsoa"])
    top10 = df.nlargest(10, "risk_score_scaled")
    top10_set = set(top10["lsoa"])

    vmin = float(phase5["risk_score_scaled"].min())
    vmax = float(phase5["risk_score_scaled"].max())
    cmap = LinearColormap(
        [TIER_COLOR[6], TIER_COLOR[5], TIER_COLOR[4],
         TIER_COLOR[3], TIER_COLOR[2], TIER_COLOR[1]],
        vmin=vmin, vmax=vmax,
        caption="Risk score (low to high)",
    )

    m = folium.Map(
        location=[51.509, -0.118],
        zoom_start=10,
        tiles="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_nolabels/{z}/{x}/{y}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    )
    Fullscreen(
        position="topright",
        title="Full screen",
        title_cancel="Exit full screen",
        force_separate_button=True,
    ).add_to(m)

    geojson = load_lsoa_geojson()
    rendered_features = []
    halo_features = []
    for feat in geojson["features"]:
        code = feat["properties"].get("LSOA21CD")
        if code not in visible_set:
            continue
        feat["properties"]["risk"] = float(risk_lookup.get(code, 0))
        feat["properties"]["tier"] = int(tier_lookup.get(code, 6))
        feat["properties"]["tier_label"] = label_lookup.get(code, "n/a")
        rendered_features.append(feat)
        if show_top_hotspots and code in top10_set:
            halo_features.append(feat)
    rendered_geo = {"type": "FeatureCollection", "features": rendered_features}
    halo_geo = {"type": "FeatureCollection", "features": halo_features}

    # Soft yellow halo drawn first, so it sits under the choropleth and
    # only the outer glow shows through.
    if halo_features:
        folium.GeoJson(
            halo_geo,
            name="Top hotspots halo",
            style_function=lambda f: {
                "fillOpacity": 0,
                "color": "#FBBF24",
                "weight": 7,
                "opacity": 0.55,
            },
            interactive=False,
        ).add_to(m)

    def style_fn(feat):
        tier = feat["properties"].get("tier", 6)
        return {
            "fillColor": TIER_COLOR.get(tier, "#CCCCCC"),
            "color": "#FFFFFF",
            "weight": 0,
            "fillOpacity": 0.92,
        }

    folium.GeoJson(
        rendered_geo,
        name="LSOAs",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["LSOA21CD", "LSOA21NM", "risk", "tier_label"],
            aliases=["Code", "Neighbourhood", "Score", "Tier"],
            localize=True,
            labels=True,
        ),
        highlight_function=lambda f: {"weight": 2, "color": "#1F2937"},
    ).add_to(m)

    if show_stations:
        stations = load_police_geojson()
        for sf in stations["features"]:
            lon, lat = sf["geometry"]["coordinates"]
            name = sf["properties"].get("name", "Police")
            folium.CircleMarker(
                location=[lat, lon], radius=2.5,
                color="#1E3A8A", weight=1.2,
                fillColor="#FFFFFF", fillOpacity=0.85,
                tooltip=name,
            ).add_to(m)

    cmap.add_to(m)
    map_event = st_folium(
        m, width=None, height=560,
        returned_objects=["last_active_drawing"],
        key="risk_map",
    )

    if map_event and map_event.get("last_active_drawing"):
        drawing = map_event["last_active_drawing"]
        clicked = drawing.get("properties", {}).get("LSOA21CD")
        # Signature uniquely identifies this click event. Folium replays
        # the same drawing on every rerun, so we only honour it once.
        click_sig = json.dumps(drawing, sort_keys=True, default=str)
        if (
            clicked
            and click_sig != st.session_state.last_consumed_click_sig
        ):
            st.session_state.last_consumed_click_sig = click_sig
            st.session_state.selected_lsoa = clicked

with col_detail:
    selected = st.session_state.selected_lsoa
    if selected and selected in set(df["lsoa"]):
        row = df[df["lsoa"] == selected].iloc[0]
        head_col, clear_col = st.columns([3, 1])
        head_col.subheader(f"{row['lsoa_name']}")
        head_col.markdown(
            f"<span class='muted'>{row['borough']} · {row['lsoa']}</span>",
            unsafe_allow_html=True,
        )
        if clear_col.button("Clear", use_container_width=True):
            st.session_state.selected_lsoa = None
            st.rerun()

        m1, m2 = st.columns(2)
        m1.metric("Risk score", f"{row['risk_score_scaled']:.1f}")
        m2.metric("Tier", f"{row['tier']}. {row['tier_label']}")
        m3, m4 = st.columns(2)
        m3.metric("Proposed officers", f"{row['officers_proposed']:.1f}",
                  delta=f"{row['officers_delta']:+.1f} vs even split")
        if pd.notna(row["dist_km"]):
            m4.metric("Nearest station",
                      f"{row['dist_km']:.1f} km",
                      delta=row["station_name"] if pd.notna(row["station_name"]) else None,
                      delta_color="off")
        else:
            m4.metric("Nearest station", "n/a",
                      delta="No map polygon", delta_color="off")
        st.metric("Tier confidence",
                  f"{row['silhouette']:+.2f}",
                  help="How clearly this neighbourhood fits its tier. "
                       "Negative means the model could place it in a "
                       "neighbouring tier instead.")

        st.markdown("**Recommended action**")
        st.info(make_recommendation(
            row, row["dist_km"] if pd.notna(row["dist_km"]) else None
        ))

        st.markdown("**What's behind the score**")
        feat_df = pd.DataFrame({
            "Driver": [FEATURES[c] for c in FEATURES],
            "Value": [float(row[c]) for c in FEATURES],
            "London average": [float(phase5[c].mean()) for c in FEATURES],
        })
        feat_df["× London average"] = (
            feat_df["Value"] / feat_df["London average"].replace(0, np.nan)
        ).round(2)
        st.dataframe(
            feat_df[["Driver", "Value", "× London average"]].assign(
                Value=feat_df["Value"].round(1)
            ),
            hide_index=True, use_container_width=True,
        )
    else:
        st.subheader("Top 10 hotspots")
        st.markdown(
            "<span class='muted'>Tap a row to inspect, "
            "or tap any neighbourhood on the map.</span>",
            unsafe_allow_html=True,
        )
        for i, (_, hr) in enumerate(top10.iterrows(), 1):
            label = (
                f"**{i}.** {hr['lsoa_name']} · {hr['borough']} · "
                f"risk {hr['risk_score_scaled']:.1f} · {hr['tier_label']}"
            )
            if st.button(label, key=f"hot_{hr['lsoa']}", use_container_width=True):
                st.session_state.selected_lsoa = hr["lsoa"]
                st.rerun()

# ----- bottom: distribution + coverage gaps -----

st.markdown("### How the tiers break down")
chart_col, gap_col = st.columns([2, 1], gap="medium")

with chart_col:
    tier_counts = (
        df.groupby("tier")
        .agg(Neighbourhoods=("lsoa", "size"),
             mean_risk=("risk_score_scaled", "mean"))
        .reset_index()
    )
    tier_counts["Tier"] = tier_counts["tier"].map(
        lambda t: f"{t}. {TIER_LABEL[t]}"
    )
    chart = (
        alt.Chart(tier_counts)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("Tier:N", sort=list(tier_counts["Tier"]), title=None),
            y=alt.Y("Neighbourhoods:Q", title="Neighbourhoods in view"),
            color=alt.Color(
                "tier:N",
                scale=alt.Scale(
                    domain=list(TIER_COLOR.keys()),
                    range=list(TIER_COLOR.values()),
                ),
                legend=None,
            ),
            tooltip=["Tier", "Neighbourhoods",
                     alt.Tooltip("mean_risk:Q", title="Avg score", format=".1f")],
        )
        .properties(height=200)
    )
    st.altair_chart(chart, use_container_width=True)

with gap_col:
    st.markdown("**Coverage gaps**")
    st.markdown(
        f"<span class='muted'>The 5 top-tier neighbourhoods furthest "
        f"from a police station (over {COVERAGE_THRESHOLD_KM} km).</span>",
        unsafe_allow_html=True,
    )
    gaps = (
        df[df["tier"].isin(ACTION_TIERS) & (df["dist_km"] > COVERAGE_THRESHOLD_KM)]
        .nlargest(5, "dist_km")
        [["lsoa_name", "borough", "tier_label", "dist_km"]]
        .rename(columns={"dist_km": "km"})
    )
    if len(gaps):
        st.dataframe(
            gaps.assign(km=gaps["km"].round(1)),
            hide_index=True, use_container_width=True,
        )
    else:
        st.success(
            f"No top-tier neighbourhood is more than "
            f"{COVERAGE_THRESHOLD_KM} km from a station in this view."
        )

# ----- footer -----

st.markdown("---")
st.caption(
    "Built on data.police.uk crime and outcomes, Met stop-and-search, "
    "TfL footfall, Met Office HadUK-Grid weather, and IMD 2025. "
    f"{n_total:,} of {len(phase5):,} neighbourhoods in view. CBL Group 16."
)
