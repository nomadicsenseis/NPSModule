#!/usr/bin/env python3

import asyncio
from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector
from dashboard_analyzer.anomaly_detection.flexible_anomaly_interpreter import FlexibleAnomalyInterpreter
from datetime import datetime
import pandas as pd

async def test_explanatory_drivers():
    print("🔍 Testing Explanatory Drivers Collection & Analysis")
    print("=" * 60)
    
    collector = PBIDataCollector()
    
    # Test with a known anomaly period
    start_date = datetime(2025, 5, 16)
    end_date = datetime(2025, 5, 22) 
    node_path = 'Global/LH/Premium'
    
    print(f"Node: {node_path}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print()
    
    # Test the explanatory drivers collection
    print("🚀 Collecting explanatory drivers data...")
    df = await collector.collect_explanatory_drivers_for_date_range(node_path, start_date, end_date)
    
    print(f"📊 Collection Results:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    if not df.empty:
        print(f"\n📋 Sample data:")
        print(df[['TouchPoint_Master[filtered_name', 'Satisfaction diff', 'Shapdiff', 'NPS diff']].head(5).to_string())
        
        # Test the analysis function
        print(f"\n🔍 Testing Analysis Function:")
        interpreter = FlexibleAnomalyInterpreter("dummy_folder", collector)
        
        # Test for negative anomaly (this was a negative anomaly in the data)
        print(f"\n   For NEGATIVE anomaly:")
        negative_analysis = interpreter._analyze_drivers_performance(df, anomaly_type="negative")
        print(f"   Result: {negative_analysis}")
        
        # Test for positive anomaly
        print(f"\n   For POSITIVE anomaly:")
        positive_analysis = interpreter._analyze_drivers_performance(df, anomaly_type="positive")
        print(f"   Result: {positive_analysis}")
        
        # Show SHAP value details
        print(f"\n📈 SHAP Value Analysis:")
        df['Shapdiff_numeric'] = pd.to_numeric(df['Shapdiff'], errors='coerce')
        df_clean = df.dropna(subset=['Shapdiff_numeric'])
        df_clean = df_clean.sort_values('Shapdiff_numeric', key=abs, ascending=False)
        
        print(f"   Top SHAP contributors:")
        for i, (_, row) in enumerate(df_clean.head(5).iterrows()):
            touchpoint = str(row['TouchPoint_Master[filtered_name'])[:20]
            shap_val = row['Shapdiff_numeric']
            sat_diff = row['Satisfaction diff']
            print(f"     {i+1}. {touchpoint}: SHAP={shap_val:.2f}, SatDiff={sat_diff:+.1f}")
    else:
        print("   ❌ No data returned")

if __name__ == "__main__":
    asyncio.run(test_explanatory_drivers()) 