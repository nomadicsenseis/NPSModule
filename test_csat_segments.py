import asyncio
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

from datetime import datetime
from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector


async def test_csat_segments():
    collector = PBIDataCollector(environment="dev")

    start_date = datetime(2026, 3, 6)
    end_date = datetime(2026, 3, 12)
    touchpoint = "Cabin Crew"

    scenarios = [
        {"name": "Global/SH", "node_path": "Global/SH"},
        {"name": "Global/LH", "node_path": "Global/LH"},
    ]

    print("=" * 60)
    print(f"TEST: CSAT vs Target for '{touchpoint}'")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("=" * 60)

    for sc in scenarios:
        print(f"\n--- {sc['name']} ---")

        cabins, companies, hauls = collector._get_node_filters(sc["node_path"])
        print(f"  Filters -> cabins={cabins}, companies={companies}, hauls={hauls}")

        query = collector._get_focus_touchpoint_csat_query(
            cabins=cabins,
            companies=companies,
            hauls=hauls,
            start_date=start_date,
            end_date=end_date,
            touchpoint_name=touchpoint,
        )
        print(f"  DAX Query:\n{query}")

        result = await collector.collect_focus_touchpoint_csat_vs_target(
            node_path=sc["node_path"],
            start_date=start_date,
            end_date=end_date,
            touchpoint_name=touchpoint,
        )

        if result:
            print(f"  CSAT   = {result['csat']:.2f}")
            print(f"  Target = {result['target']:.2f}")
            print(f"  Gap    = {result['gap']:+.2f}")
        else:
            print("  (sin datos)")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")


if __name__ == "__main__":
    asyncio.run(test_csat_segments())
