"""
Test script: NCS free-text flight-number → route mapping feasibility

This script answers:
1) What schema does the current routes dictionary provide?
   - It is ROUTE → (country_name, gr_region, haul_aggr)
   - It does NOT include flight numbers, so it cannot map IB0713 → MAD-LHR directly.
2) How many NCS incidents include an explicit route (XXX-YYY) in the text?
3) How many include flight numbers without routes (e.g., IB1020) and therefore are NOT mappable with the current dictionary?

Usage:
  python scripts/test_ncs_flight_to_route_mapping.py --start 2025-12-06 --end 2025-12-12

Notes:
  - Requires valid AWS creds in /app/temp_aws_credentials.env to read NCS from S3.
  - Requires access to PBI for routes dictionary (same credentials/env as your app runtime).
"""

import argparse
import asyncio
import re
from datetime import datetime


ROUTE_RE = re.compile(r"\b([A-Z]{3})\s*[-–]\s*([A-Z]{3})\b")
FLIGHT_RE = re.compile(r"\b(IB|YW)\s*0*(\d{2,4})\b", re.IGNORECASE)


def _extract_routes(text: str) -> list[str]:
    if not text:
        return []
    return [f"{a.upper()}-{b.upper()}" for a, b in ROUTE_RE.findall(text.upper())]


def _extract_flights(text: str) -> list[str]:
    if not text:
        return []
    flights = []
    for airline, num in FLIGHT_RE.findall(text.upper()):
        flights.append(f"{airline.upper()}{int(num)}")
    # de-dupe preserving order
    seen = set()
    out = []
    for f in flights:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


async def _run(start: str, end: str, sample: int) -> int:
    from dashboard_analyzer.data_collection.ncs_collector import NCSDataCollector

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    print(f"Loading NCS from {start} to {end} ...")
    ncs = NCSDataCollector(environment="local").collect_ncs_data_for_date_range(start_dt, end_dt)
    print(f"NCS rows: {len(ncs)} | columns: {list(ncs.columns)}")
    if ncs.empty:
        return 0

    incident_col = ncs.columns[0]
    texts = ncs[incident_col].astype(str).fillna("")

    has_route = texts.apply(lambda t: bool(ROUTE_RE.search(t.upper())))
    has_flight = texts.apply(lambda t: bool(FLIGHT_RE.search(t.upper())))
    has_flight_no_route = has_flight & (~has_route)

    print(f"Rows with explicit route (XXX-YYY) in text: {int(has_route.sum())}")
    print(f"Rows with flight number (IB/YW####) in text: {int(has_flight.sum())}")
    print(f"Rows with flight number but NO route (not mappable with current route dictionary): {int(has_flight_no_route.sum())}")

    # Routes dictionary schema (route -> haul_aggr, etc.)
    print("\nRoutes dictionary used today (for haul filtering) is ROUTE-based (not flight-based):")
    print("Expected columns: route, country_name, gr_region, haul_aggr")
    print("Query template: /app/dashboard_analyzer/data_collection/queries/Rutas Diccionario.txt")

    try:
        from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector
        print("\nLoading routes dictionary (PBI) ...")
        pbi = PBIDataCollector()
        routes_df = await pbi.collect_routes_dictionary()
        print(f"Routes dictionary rows: {len(routes_df)} | columns: {list(routes_df.columns)}")
        if not routes_df.empty:
            print("Sample routes dictionary rows:")
            print(routes_df.head(5).to_string(index=False))
    except Exception as e:
        # Common in local/dev: missing PBI env vars
        print(f"\nSkipping PBI routes dictionary download (not available in this environment): {e}")

    # Show a few examples for manual inspection
    print("\nExamples (route-present):")
    route_rows = ncs[has_route].head(sample)
    for i, row in enumerate(route_rows[incident_col].astype(str).tolist(), 1):
        print(f"{i}. routes={_extract_routes(row)[:3]} | flights={_extract_flights(row)[:3]} | text={row[:160]}...")

    print("\nExamples (flight-present but route-missing):")
    flight_only_rows = ncs[has_flight_no_route].head(sample)
    for i, row in enumerate(flight_only_rows[incident_col].astype(str).tolist(), 1):
        print(f"{i}. flights={_extract_flights(row)[:5]} | text={row[:160]}...")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--sample", type=int, default=5, help="How many sample rows to print per category")
    args = ap.parse_args()
    return asyncio.run(_run(args.start, args.end, args.sample))


if __name__ == "__main__":
    raise SystemExit(main())


