#!/usr/bin/env python3

import asyncio
import sys
sys.path.append('.')
from dashboard_analyzer.enhanced_flexible_main import run_flexible_analysis_silent
from dashboard_analyzer.anomaly_detection.flexible_anomaly_interpreter import FlexibleAnomalyInterpreter
from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector
from datetime import datetime

async def debug_explanations():
    # Use the data folder from our analysis
    data_folder = 'tables/flight_local_daily_2025_01_20_flexible_1d_Global_LH_1413'
    
    # Initialize the interpreter
    pbi_collector = PBIDataCollector()
    interpreter = FlexibleAnomalyInterpreter(data_folder, pbi_collector=pbi_collector)
    
    print(f"Testing explanation generation for: {data_folder}")
    print("="*60)
    
    # Test 1: Check date range resolution
    print("\n1. Testing date range resolution...")
    date_range = interpreter._get_period_date_range(1, 1)
    print(f"   Date range for period 1: {date_range}")
    
    # Test 2: Try explaining an anomaly directly
    print("\n2. Testing direct explanation...")
    try:
        explanation = await interpreter.explain_anomaly(
            node_path='Global/LH/Premium',
            target_period=1,
            aggregation_days=1,
            anomaly_state='-'
        )
        print(f"   Explanation length: {len(explanation)} chars")
        print(f"   Explanation content: '{explanation}'")
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    # Test 3: Try with manual date range
    print("\n3. Testing with manual date range...")
    try:
        start_date = datetime(2025, 1, 20)
        end_date = datetime(2025, 1, 20)
        
        # Test each component
        verbatims = await interpreter._analyze_verbatims_data('Global/LH/Premium', start_date, end_date)
        print(f"   Verbatims: {verbatims[:100]}...")
        
        routes = await interpreter._analyze_routes_data('Global/LH/Premium', start_date, end_date, 'negative')
        print(f"   Routes: {routes[:100]}...")
        
        drivers = await interpreter._analyze_explanatory_drivers_data('Global/LH/Premium', start_date, end_date, 'negative')
        print(f"   Drivers: {drivers[:100]}...")
        
    except Exception as e:
        print(f"   Error in manual test: {str(e)}")

if __name__ == "__main__":
    asyncio.run(debug_explanations()) 