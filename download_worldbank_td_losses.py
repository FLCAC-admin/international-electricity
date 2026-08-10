"""Download World Bank EG.ELC.LOSS.ZS to data/eg_elc_loss_zs.csv."""
from datetime import date
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
BASE = "https://api.worldbank.org/v2"
INDICATOR = "EG.ELC.LOSS.ZS"


def fetch_all(path):
    page, rows = 1, []
    while True:
        r = requests.get(
            f"{BASE}/{path}",
            params={"format": "json", "per_page": 20000, "page": page},
        )
        r.raise_for_status()
        meta, data = r.json()
        rows.extend(data or [])
        if page >= meta["pages"]:
            break
        page += 1
    return rows


def main():
    print(f"Access date: {date.today()}")
    real, geo = {}, {}
    for c in fetch_all("country/all"):
        rid = (c.get("region") or {}).get("id") or ""
        rname = (c.get("region") or {}).get("value") or ""
        if rid and rid != "NA":
            real[c["id"]] = {
                "name": c["name"].strip(),
                "region_code": rid.strip(),
                "region_name": rname.strip(),
            }
            geo[rid] = rname.strip()

    out = []
    for iso3, m in real.items():
        out.append({
            "iso3": iso3, "name": m["name"], "year": "", "value_pct": "",
            "series_type": "iso3_region",
            "region_code": m["region_code"], "region_name": m["region_name"],
        })
    for row in fetch_all(f"country/all/indicator/{INDICATOR}"):
        if row.get("value") is None:
            continue
        iso3 = row.get("countryiso3code") or ""
        year, val = int(row["date"]), float(row["value"])
        if iso3 in real:
            m = real[iso3]
            out.append({
                "iso3": iso3, "name": m["name"], "year": year, "value_pct": val,
                "series_type": "country",
                "region_code": m["region_code"], "region_name": m["region_name"],
            })
        elif iso3 in geo:
            out.append({
                "iso3": iso3, "name": geo[iso3], "year": year, "value_pct": val,
                "series_type": "geographic_region",
                "region_code": "", "region_name": "",
            })

    df = pd.DataFrame(out).sort_values(["series_type", "iso3", "year"])
    path = DATA / "eg_elc_loss_zs.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)")
    print(df.groupby("series_type").size().to_string())
    yrs = pd.to_numeric(df.year, errors="coerce")
    print("Year range:", int(yrs.min()), "-", int(yrs.max()))


if __name__ == "__main__":
    main()
