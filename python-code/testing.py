import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE_PATH = r"C:\Users\mbeck\OneDrive\Documents\CBL-16_data"   # <-- change this

archives = {
    "0423-0424": ("2023-04", "2024-04"),
    "0524-0525": ("2024-05", "2025-05"),
    "0625-0326": ("2025-06", "2026-03"),
}

file_types = ["street", "outcomes", "stop-and-search"]

print("=" * 60)
print("CBL-16 DATA AUDIT")
print("=" * 60)

total_files   = 0
total_rows    = 0
total_missing = 0
results       = []

for archive, (start, end) in archives.items():
    archive_path = Path(BASE_PATH) / archive
    if not archive_path.exists():
        print(f"\n[MISSING] Archive folder not found: {archive}")
        continue

    month_folders = sorted([d for d in archive_path.iterdir() if d.is_dir()])
    print(f"\n{'─'*60}")
    print(f"Archive: {archive}  ({len(month_folders)} month folders found)")
    print(f"{'─'*60}")

    archive_rows  = 0
    archive_files = 0

    for month_dir in month_folders:
        month = month_dir.name
        street_files = list(month_dir.glob("*-street.csv"))
        forces_found = len(street_files)
        month_rows   = 0
        month_files  = 0

        for ftype in file_types:
            files = list(month_dir.glob(f"*-{ftype}.csv"))
            month_files += len(files)
            for f in files:
                try:
                    df = pd.read_csv(f, usecols=[0], dtype=str)
                    month_rows += len(df)
                except Exception as e:
                    print(f"  [ERROR] {f.name}: {e}")

        archive_rows  += month_rows
        archive_files += month_files
        total_files   += month_files
        total_rows    += month_rows

        status = "OK" if forces_found >= 40 else f"WARNING: only {forces_found} forces"
        print(f"  {month}  |  {forces_found} forces  |  "
              f"{month_files:4d} files  |  {month_rows:7,} rows  |  {status}")

        results.append({
            "archive": archive, "month": month,
            "forces": forces_found, "files": month_files, "rows": month_rows
        })

    print(f"\n  ARCHIVE TOTAL: {archive_files} files, {archive_rows:,} rows")

print(f"\n{'='*60}")
print(f"GRAND TOTAL: {total_files} files | {total_rows:,} rows")
print(f"{'='*60}")

df_results = pd.DataFrame(results)
df_results.to_csv(Path(BASE_PATH) / "audit_log.csv", index=False)
print(f"\nAudit log saved to: {BASE_PATH}\\audit_log.csv")
