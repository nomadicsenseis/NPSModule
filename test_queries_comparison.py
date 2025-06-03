#!/usr/bin/env python3
"""
Test script to compare data collection between NPS_flex_agg.txt and Monthly NPS.txt queries
to verify they collect the same data when configured appropriately.
"""

import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime
from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector

async def test_queries_comparison():
    """Compare data collection between NPS_flex_agg and Monthly NPS queries"""
    
    print("🔍 TESTING QUERIES COMPARISON")
    print("="*60)
    
    # Initialize PBI collector
    collector = PBIDataCollector()
    
    # Load the original queries
    queries_path = Path("dashboard_analyzer/data_collection/queries")
    
    # Read NPS_flex_agg query
    nps_flex_path = queries_path / "NPS_flex_agg.txt"
    with open(nps_flex_path, 'r', encoding='utf-8') as f:
        nps_flex_query = f.read()
    
    # Read Monthly NPS query  
    monthly_nps_path = queries_path / "Monthly NPS.txt"
    with open(monthly_nps_path, 'r', encoding='utf-8') as f:
        monthly_nps_query = f.read()
    
    print("📋 Original Queries Loaded")
    print(f"  - NPS_flex_agg: {len(nps_flex_query)} characters")
    print(f"  - Monthly NPS: {len(monthly_nps_query)} characters")
    
    # Modify NPS_flex_agg for 7 days (change the grouping interval from 5 to 7)
    nps_flex_7days = nps_flex_query.replace(
        'INT(DATEDIFF( \'Date_Master\'[Date],max(\'Date_Master\'[Date]), DAY) / 5) + 1',
        'INT(DATEDIFF( \'Date_Master\'[Date],max(\'Date_Master\'[Date]), DAY) / 7) + 1'
    )
    
    # Modify Monthly NPS to use weekly setup (uncomment Week line, comment Month line)
    monthly_nps_weekly = monthly_nps_query.replace(
        "'Date_Master'[Month], -- Mes",
        "-- 'Date_Master'[Month], -- Mes"
    ).replace(
        "--'Date_Master'[Week], - Número semana",
        "'Date_Master'[Week], -- Número semana"
    )
    
    print("\n🔧 Queries Modified:")
    print("  - NPS_flex_agg: Changed grouping from 5 to 7 days")
    print("  - Monthly NPS: Changed from Month to Week grouping")
    
    # Execute both queries
    print("\n📡 Executing Queries...")
    
    try:
        print("  🔄 Executing NPS_flex_agg (7 days)...")
        df_flex = await collector._execute_query_async(nps_flex_7days)
        
        print("  🔄 Executing Monthly NPS (weekly)...")
        df_monthly = await collector._execute_query_async(monthly_nps_weekly)
        
        # Clean column names
        if not df_flex.empty:
            df_flex.columns = [col.strip('[]') for col in df_flex.columns]
        if not df_monthly.empty:
            df_monthly.columns = [col.strip('[]') for col in df_monthly.columns]
        
        # Display results
        print(f"\n📊 RESULTS COMPARISON:")
        print("="*50)
        
        print(f"NPS_flex_agg (7 days):")
        print(f"  - Rows: {len(df_flex)}")
        print(f"  - Columns: {list(df_flex.columns) if not df_flex.empty else 'No data'}")
        if not df_flex.empty:
            print(f"  - Sample data:")
            print(df_flex.head().to_string(index=False))
        
        print(f"\nMonthly NPS (weekly):")
        print(f"  - Rows: {len(df_monthly)}")
        print(f"  - Columns: {list(df_monthly.columns) if not df_monthly.empty else 'No data'}")
        if not df_monthly.empty:
            print(f"  - Sample data:")
            print(df_monthly.head().to_string(index=False))
        
        # Compare data if both have results
        if not df_flex.empty and not df_monthly.empty:
            print(f"\n🔍 DATA COMPARISON:")
            print("-" * 30)
            
            # Compare common columns
            common_cols = set(df_flex.columns) & set(df_monthly.columns)
            print(f"  - Common columns: {list(common_cols)}")
            
            if common_cols:
                # Look for NPS columns to compare
                nps_cols = [col for col in common_cols if 'NPS' in col]
                print(f"  - NPS columns found: {nps_cols}")
                
                for nps_col in nps_cols:
                    if nps_col in df_flex.columns and nps_col in df_monthly.columns:
                        flex_values = df_flex[nps_col].dropna()
                        monthly_values = df_monthly[nps_col].dropna()
                        
                        print(f"\n  📈 {nps_col} Comparison:")
                        print(f"    - Flex values: {flex_values.tolist()}")
                        print(f"    - Monthly values: {monthly_values.tolist()}")
                        
                        # Check if values are similar
                        if len(flex_values) > 0 and len(monthly_values) > 0:
                            flex_avg = flex_values.mean()
                            monthly_avg = monthly_values.mean()
                            diff = abs(flex_avg - monthly_avg)
                            
                            print(f"    - Average difference: {diff:.3f}")
                            
                            if diff < 0.01:  # Less than 0.01 difference
                                print(f"    ✅ Values are very similar!")
                            elif diff < 1.0:  # Less than 1 point difference
                                print(f"    ⚠️ Values are close but not identical")
                            else:
                                print(f"    ❌ Values are significantly different")
            
            # Save results for further analysis
            if not df_flex.empty:
                df_flex.to_csv("test_nps_flex_7days.csv", index=False)
                print(f"\n💾 Saved NPS_flex_agg results to: test_nps_flex_7days.csv")
            
            if not df_monthly.empty:
                df_monthly.to_csv("test_monthly_nps_weekly.csv", index=False)
                print(f"💾 Saved Monthly NPS results to: test_monthly_nps_weekly.csv")
        
        else:
            print("\n❌ Cannot compare: One or both queries returned no data")
            
    except Exception as e:
        print(f"❌ Error executing queries: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("🏁 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_queries_comparison()) 