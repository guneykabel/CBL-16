"""
CBL-16 — Crime Resolution Analysis
====================================
Analyses how different crime types are resolved across all forces and months.
Produces:
  - Resolution rates per crime type
  - Outcome category breakdown per crime type
  - Temporal trends in resolution rates
  - Force-level resolution comparison
  - Heatmap of resolution rates (crime type × month)

Usage:
    python crime_resolution_analysis.py

Set BASE_PATH below to your CBL-16_data folder.
Outputs are saved to BASE_PATH/outputs/resolution/
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE_PATH   = r"C:\Users\mbeck\OneDrive\Documents\CBL-16_data"   # <-- change this
OUTPUT_DIR  = Path(BASE_PATH) / "outputs" / "resolution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Outcome categories grouped into resolved / unresolved / pending
RESOLVED = {
    "Suspect charged",
    "Offender given a caution",
    "Offender given a penalty notice",
    "Offender fined",
    "Offender deported",
    "Offender otherwise dealt with",
    "Suspect charged as part of another case",
    "Local resolution",
    "Offender given a drugs possession warning",
    "Offender given conditional discharge",
    "Offender given absolute discharge",
    "Offender sent to prison",
    "Offender given suspended prison sentence",
    "Offender given community sentence",
}

UNRESOLVED = {
    "Investigation complete; no suspect identified",
    "Unable to prosecute suspect",
    "Further investigation is not in the public interest",
    "Further action is not in the public interest",
    "Formal action is not in the public interest",
    "Action to be taken by another organisation",
}

PENDING = {
    "Awaiting court outcome",
    "Court result unavailable",
    "Status update unavailable",
}

def classify_outcome(outcome):
    if pd.isna(outcome):
        return "Pending / unknown"
    for r in RESOLVED:
        if r.lower() in str(outcome).lower():
            return "Resolved"
    for u in UNRESOLVED:
        if u.lower() in str(outcome).lower():
            return "Unresolved"
    for p in PENDING:
        if p.lower() in str(outcome).lower():
            return "Pending / unknown"
    return "Pending / unknown"


# ── STEP 1: LOAD ALL STREET CSVs ───────────────────────────────────────────────
print("Loading street CSVs...")

street_files = glob.glob(str(Path(BASE_PATH) / "**" / "*-street.csv"), recursive=True)
print(f"  Found {len(street_files)} street files")

chunks = []
for f in street_files:
    try:
        df = pd.read_csv(f, usecols=[
            "Month", "Falls within", "Crime type", "Last outcome category"
        ], dtype=str, low_memory=False)
        chunks.append(df)
    except Exception as e:
        print(f"  [SKIP] {Path(f).name}: {e}")

data = pd.concat(chunks, ignore_index=True)
data.columns = data.columns.str.strip()
data["Month"] = pd.to_datetime(data["Month"], format="%Y-%m")

# Classify each record
print("Classifying outcomes...")
data["Resolution"] = data["Last outcome category"].apply(classify_outcome)

print(f"  Total records loaded: {len(data):,}")
print(f"  Date range: {data['Month'].min().strftime('%b %Y')} → {data['Month'].max().strftime('%b %Y')}")
print(f"  Crime types: {data['Crime type'].nunique()}")
print(f"  Forces: {data['Falls within'].nunique()}")


# ── STEP 2: OVERALL RESOLUTION RATE PER CRIME TYPE ────────────────────────────
print("\nCalculating resolution rates per crime type...")

ct_res = (
    data.groupby(["Crime type", "Resolution"])
        .size()
        .reset_index(name="count")
)
ct_total = data.groupby("Crime type").size().reset_index(name="total")
ct_res   = ct_res.merge(ct_total, on="Crime type")
ct_res["rate"] = (ct_res["count"] / ct_res["total"] * 100).round(1)

# Pivot for the bar chart
ct_pivot = ct_res.pivot_table(
    index="Crime type", columns="Resolution", values="rate", fill_value=0
).reset_index()

# Sort by resolved rate descending
if "Resolved" in ct_pivot.columns:
    ct_pivot = ct_pivot.sort_values("Resolved", ascending=True)

# ── PLOT 1: Stacked horizontal bar — resolution breakdown per crime type ───────
fig, ax = plt.subplots(figsize=(13, 7))
colours = {"Resolved": "#1D9E75", "Unresolved": "#E24B4A", "Pending / unknown": "#B4B2A9"}
bottom  = np.zeros(len(ct_pivot))

for col, colour in colours.items():
    if col in ct_pivot.columns:
        vals = ct_pivot[col].values
        bars = ax.barh(ct_pivot["Crime type"], vals, left=bottom,
                       color=colour, label=col, edgecolor="white", linewidth=0.5)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 4:
                ax.text(b + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += vals

ax.set_xlabel("Percentage of cases (%)", fontsize=11)
ax.set_xlim(0, 100)
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax.set_title("Resolution rate breakdown by crime type (all forces, all months)",
             fontsize=13, pad=14)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_resolution_by_crime_type.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 01_resolution_by_crime_type.png")


# ── STEP 3: RESOLUTION RATE TREND OVER TIME (monthly) ─────────────────────────
print("Calculating monthly resolution trends...")

monthly_res = (
    data[data["Crime type"] != "Anti-social behaviour"]  # ASB never has outcome
        .groupby(["Month", "Crime type", "Resolution"])
        .size()
        .reset_index(name="count")
)
monthly_total = (
    data[data["Crime type"] != "Anti-social behaviour"]
        .groupby(["Month", "Crime type"])
        .size()
        .reset_index(name="total")
)
monthly_res = monthly_res.merge(monthly_total, on=["Month", "Crime type"])
monthly_res["rate"] = monthly_res["count"] / monthly_res["total"] * 100

resolved_trend = (
    monthly_res[monthly_res["Resolution"] == "Resolved"]
        .pivot_table(index="Month", columns="Crime type", values="rate", fill_value=0)
)

# Pick 8 most interesting crime types (highest variance in resolution)
top_types = resolved_trend.std().sort_values(ascending=False).head(8).index.tolist()

colours_trend = [
    "#D85A30", "#1D9E75", "#378ADD", "#534AB7",
    "#E24B4A", "#BA7517", "#D4537E", "#639922"
]

fig, ax = plt.subplots(figsize=(14, 6))
for i, ct in enumerate(top_types):
    if ct in resolved_trend.columns:
        ax.plot(resolved_trend.index, resolved_trend[ct],
                label=ct, color=colours_trend[i % len(colours_trend)],
                linewidth=1.8, marker="o", markersize=3.5, alpha=0.85)

ax.set_ylabel("% cases resolved", fontsize=11)
ax.set_xlabel("Month", fontsize=11)
ax.set_title("Resolution rate over time — top 8 most variable crime types",
             fontsize=13, pad=14)
ax.legend(fontsize=8.5, loc="upper left", ncol=2, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.grid(axis="y", alpha=0.25, linewidth=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_resolution_trend_over_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 02_resolution_trend_over_time.png")


# ── STEP 4: HEATMAP — resolution rate (crime type × month) ────────────────────
print("Building resolution heatmap...")

heatmap_data = (
    monthly_res[monthly_res["Resolution"] == "Resolved"]
        .pivot_table(index="Crime type", columns="Month", values="rate", fill_value=0)
)
heatmap_data.columns = [m.strftime("%b %y") for m in heatmap_data.columns]

fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(
    heatmap_data, ax=ax,
    cmap="RdYlGn", annot=True, fmt=".0f", annot_kws={"size": 8},
    linewidths=0.4, linecolor="white",
    cbar_kws={"label": "% resolved", "shrink": 0.7},
    vmin=0, vmax=30
)
ax.set_title("Resolution rate (%) by crime type and month", fontsize=13, pad=14)
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="x", labelsize=9, rotation=45)
ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_resolution_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 03_resolution_heatmap.png")


# ── STEP 5: FORCE-LEVEL RESOLUTION COMPARISON ─────────────────────────────────
print("Calculating force-level resolution rates...")

force_res = (
    data[data["Crime type"] != "Anti-social behaviour"]
        .groupby(["Falls within", "Resolution"])
        .size()
        .reset_index(name="count")
)
force_total = (
    data[data["Crime type"] != "Anti-social behaviour"]
        .groupby("Falls within")
        .size()
        .reset_index(name="total")
)
force_res = force_res.merge(force_total, on="Falls within")
force_res["rate"] = force_res["count"] / force_res["total"] * 100

force_resolved = (
    force_res[force_res["Resolution"] == "Resolved"]
        .sort_values("rate", ascending=True)
)

# Shorten force names
force_resolved["force_short"] = (
    force_resolved["Falls within"]
        .str.replace(" Constabulary", "").str.replace(" Police Service", "")
        .str.replace(" Police", "").str.replace(" Service", "")
)

fig, ax = plt.subplots(figsize=(12, 10))
bars = ax.barh(force_resolved["force_short"], force_resolved["rate"],
               color="#378ADD", edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, force_resolved["rate"]):
    ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=8.5, color="#333")

nat_avg = force_resolved["rate"].mean()
ax.axvline(nat_avg, color="#E24B4A", linewidth=1.5, linestyle="--", alpha=0.8,
           label=f"National avg: {nat_avg:.1f}%")

ax.set_xlabel("Resolution rate (%)", fontsize=11)
ax.set_title("Resolution rate by police force (excluding ASB)", fontsize=13, pad=14)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_resolution_by_force.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 04_resolution_by_force.png")


# ── STEP 6: OUTCOME DETAIL — what exactly happens to each crime type ───────────
print("Building outcome detail breakdown...")

outcome_detail = (
    data[data["Crime type"] != "Anti-social behaviour"]
        .groupby(["Crime type", "Last outcome category"])
        .size()
        .reset_index(name="count")
)
outcome_total = (
    data[data["Crime type"] != "Anti-social behaviour"]
        .groupby("Crime type")
        .size()
        .reset_index(name="total")
)
outcome_detail = outcome_detail.merge(outcome_total, on="Crime type")
outcome_detail["rate"] = (outcome_detail["count"] / outcome_detail["total"] * 100).round(1)

# Keep top 6 outcomes per crime type for readability
top_outcomes = (
    outcome_detail.groupby("Last outcome category")["count"]
        .sum().sort_values(ascending=False).head(8).index.tolist()
)
outcome_filtered = outcome_detail[outcome_detail["Last outcome category"].isin(top_outcomes)]

pivot_detail = outcome_filtered.pivot_table(
    index="Crime type", columns="Last outcome category", values="rate", fill_value=0
)

# Sort by "Unable to prosecute" descending
sort_col = [c for c in pivot_detail.columns if "Unable" in c]
if sort_col:
    pivot_detail = pivot_detail.sort_values(sort_col[0], ascending=False)

fig, ax = plt.subplots(figsize=(14, 8))
cmap = plt.cm.get_cmap("tab10", len(pivot_detail.columns))
bottom = np.zeros(len(pivot_detail))

for i, col in enumerate(pivot_detail.columns):
    vals = pivot_detail[col].values
    short = col.replace("Investigation complete; ", "").replace("Unable to prosecute", "Unable to prosecute")[:40]
    ax.bar(pivot_detail.index, vals, bottom=bottom,
           label=short, color=cmap(i), edgecolor="white", linewidth=0.5)
    bottom += vals

ax.set_ylabel("% of cases", fontsize=11)
ax.set_title("Outcome breakdown by crime type (top 8 outcome categories)",
             fontsize=13, pad=14)
ax.legend(fontsize=8, loc="upper right", ncol=1,
          bbox_to_anchor=(1.38, 1), framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", rotation=35, labelsize=9)
ax.tick_params(axis="y", labelsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_outcome_detail_by_crime.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 05_outcome_detail_by_crime.png")


# ── STEP 7: SUMMARY TABLE ──────────────────────────────────────────────────────
print("\nGenerating summary table...")

summary = (
    data[data["Crime type"] != "Anti-social behaviour"]
        .groupby("Crime type")
        .agg(
            total_crimes=("Month", "count"),
            resolved=("Resolution", lambda x: (x == "Resolved").sum()),
            unresolved=("Resolution", lambda x: (x == "Unresolved").sum()),
            pending=("Resolution", lambda x: (x == "Pending / unknown").sum()),
        )
        .reset_index()
)
summary["resolution_rate_%"] = (summary["resolved"] / summary["total_crimes"] * 100).round(1)
summary["unresolved_rate_%"]  = (summary["unresolved"] / summary["total_crimes"] * 100).round(1)
summary["pending_rate_%"]     = (summary["pending"] / summary["total_crimes"] * 100).round(1)
summary = summary.sort_values("resolution_rate_%", ascending=False)

summary.to_csv(OUTPUT_DIR / "resolution_summary.csv", index=False)

print("\n" + "=" * 65)
print(f"{'Crime type':<35} {'Total':>8} {'Resolved%':>10} {'Unresolved%':>12}")
print("=" * 65)
for _, row in summary.iterrows():
    print(f"{row['Crime type']:<35} {row['total_crimes']:>8,} "
          f"{row['resolution_rate_%']:>9.1f}%  {row['unresolved_rate_%']:>10.1f}%")
print("=" * 65)

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("Files generated:")
for f in sorted(OUTPUT_DIR.glob("*")):
    print(f"  {f.name}")