#!/usr/bin/env python3
"""
Script temporal para probar qué fecha tiene datos disponibles en PBI
usando la lógica de find_latest_available_date de weekly_deep_research.py
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector

# Constants
MIN_PBI_LAG_DAYS = 4
MAX_PBI_LAG_DAYS = 10


async def probe_data_availability(simulated_today: datetime, environment: str = "local"):
    """
    Probe for the latest date with available data.
    """
    print(f"\n{'='*60}")
    print(f"🔍 PROBING PBI DATA AVAILABILITY")
    print(f"{'='*60}")
    print(f"   Simulated today: {simulated_today.strftime('%Y-%m-%d')}")
    print(f"   Min lag to try: {MIN_PBI_LAG_DAYS} days")
    print(f"   Max lag to try: {MAX_PBI_LAG_DAYS} days")
    print(f"   Environment: {environment}")
    print(f"{'='*60}\n")
    
    collector = PBIDataCollector(environment=environment)
    
    results = []
    
    for lag in range(MIN_PBI_LAG_DAYS, MAX_PBI_LAG_DAYS + 1):
        candidate_date = simulated_today - timedelta(days=lag)
        print(f"🔍 Checking {candidate_date.strftime('%Y-%m-%d')} (lag: -{lag} days)...")
        
        try:
            with tempfile.TemporaryDirectory() as tmp_folder:
                # Probe: 1 day aggregation, Global node only
                query_results = await collector.collect_flexible_data_for_node(
                    "Global",
                    aggregation_days=1,
                    target_folder=tmp_folder,
                    analysis_date=candidate_date
                )
                
                nps_file = Path(tmp_folder) / "Global" / "flexible_NPS_1d.csv"
                
                if query_results.get('flexible_NPS', False) and nps_file.exists():
                    df = pd.read_csv(nps_file)
                    
                    # Create Unified_NPS
                    year_cols = ['NPS_2026', 'NPS_2025', 'NPS_2024', 'NPS_2019']
                    df['Unified_NPS'] = pd.NA
                    for col in year_cols:
                        if col in df.columns:
                            df['Unified_NPS'] = df['Unified_NPS'].fillna(df[col])
                    
                    # Check Period 1
                    if 'Period_Group' in df.columns:
                        period_1_data = df[df['Period_Group'] == 1]
                        
                        if not period_1_data.empty and period_1_data['Unified_NPS'].notna().any():
                            nps_value = period_1_data['Unified_NPS'].dropna().iloc[0]
                            print(f"   ✅ DATA FOUND! NPS: {nps_value:.1f}")
                            results.append({
                                'date': candidate_date,
                                'lag': lag,
                                'nps': nps_value,
                                'status': 'OK'
                            })
                        else:
                            print(f"   ⚠️ No data for Period 1")
                            results.append({
                                'date': candidate_date,
                                'lag': lag,
                                'nps': None,
                                'status': 'NO_PERIOD_1'
                            })
                    else:
                        print(f"   ⚠️ Period_Group column not found")
                        results.append({
                            'date': candidate_date,
                            'lag': lag,
                            'nps': None,
                            'status': 'NO_PERIOD_COL'
                        })
                else:
                    print(f"   ⚠️ No NPS data collected")
                    results.append({
                        'date': candidate_date,
                        'lag': lag,
                        'nps': None,
                        'status': 'NO_DATA'
                    })
                    
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:80]}")
            results.append({
                'date': candidate_date,
                'lag': lag,
                'nps': None,
                'status': f'ERROR: {str(e)[:50]}'
            })
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📋 SUMMARY")
    print(f"{'='*60}")
    
    ok_results = [r for r in results if r['status'] == 'OK']
    
    if ok_results:
        latest = ok_results[0]  # First OK is the most recent
        print(f"\n✅ LATEST AVAILABLE DATE: {latest['date'].strftime('%Y-%m-%d')}")
        print(f"   Lag from simulated today: {latest['lag']} days")
        print(f"   NPS value: {latest['nps']:.1f}")
        
        print(f"\n📊 All dates with data:")
        for r in ok_results:
            print(f"   • {r['date'].strftime('%Y-%m-%d')} (lag -{r['lag']}d) - NPS: {r['nps']:.1f}")
    else:
        print(f"\n❌ NO DATA FOUND in range {MIN_PBI_LAG_DAYS}-{MAX_PBI_LAG_DAYS} days lag")
    
    print(f"\n📊 All probe results:")
    for r in results:
        status_icon = "✅" if r['status'] == 'OK' else "❌"
        nps_str = f"NPS: {r['nps']:.1f}" if r['nps'] else r['status']
        print(f"   {status_icon} {r['date'].strftime('%Y-%m-%d')} (lag -{r['lag']}d) - {nps_str}")
    
    return results


async def main():
    # Simulated today = 2026-02-09
    simulated_today = datetime(2026, 2, 9)
    
    # Run probe with local environment (reads .env)
    await probe_data_availability(simulated_today, environment="local")


if __name__ == "__main__":
    asyncio.run(main())
