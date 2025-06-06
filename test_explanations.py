#!/usr/bin/env python3

import sys
sys.path.append('.')
from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector
from dashboard_analyzer.anomaly_detection.flexible_anomaly_interpreter import FlexibleAnomalyInterpreter
from datetime import datetime, timedelta
import asyncio

async def test_explanation_methods():
    collector = PBIDataCollector()
    
    # Use an existing data folder that should have the analysis results
    data_folder = "tables/flight_local_daily_2025_01_20_flexible_1d_Global_LH_1406"
    interpreter = FlexibleAnomalyInterpreter(data_folder, collector)
    
    # Test parameters - these should match what worked in our earlier tests
    node_path = 'Global/LH/Premium'
    start_date = datetime(2025, 1, 20)
    end_date = datetime(2025, 1, 20)  # Single day
    anomaly_type = "negative"
    
    print(f'Testing explanation methods for {node_path} on {start_date.strftime("%Y-%m-%d")}')
    print('='*70)
    
    # Test 1: Verbatims analysis
    print('\n1. Testing verbatims analysis...')
    try:
        verbatims_result = await interpreter._analyze_verbatims_data(node_path, start_date, end_date)
        print(f'   Verbatims result: "{verbatims_result}"')
        print(f'   Length: {len(verbatims_result)} chars')
        print(f'   Contains "No ": {"No " in verbatims_result}')
        print(f'   Contains "Error": {"Error" in verbatims_result}')
    except Exception as e:
        print(f'   Verbatims error: {str(e)}')
    
    # Test 2: Routes analysis
    print('\n2. Testing routes analysis...')
    try:
        routes_result = await interpreter._analyze_routes_data(node_path, start_date, end_date, anomaly_type)
        print(f'   Routes result: "{routes_result}"')
        print(f'   Length: {len(routes_result)} chars')
        print(f'   Contains "No ": {"No " in routes_result}')
        print(f'   Contains "Error": {"Error" in routes_result}')
        print(f'   Contains "not yet implemented": {"not yet implemented" in routes_result}')
    except Exception as e:
        print(f'   Routes error: {str(e)}')
    
    # Test 3: Drivers analysis
    print('\n3. Testing drivers analysis...')
    try:
        drivers_result = await interpreter._analyze_explanatory_drivers_data(node_path, start_date, end_date, anomaly_type)
        print(f'   Drivers result: "{drivers_result}"')
        print(f'   Length: {len(drivers_result)} chars')
        print(f'   Contains "No ": {"No " in drivers_result}')
        print(f'   Contains "Error": {"Error" in drivers_result}')
        print(f'   Contains "not yet implemented": {"not yet implemented" in drivers_result}')
    except Exception as e:
        print(f'   Drivers error: {str(e)}')
    
    # Test 4: Combined explanation
    print('\n4. Testing combined explanation...')
    try:
        # Simulate calling the analysis methods again to get their results
        operational = "No operational data changes detected"
        verbatims = verbatims_result if 'verbatims_result' in locals() else ""
        routes = routes_result if 'routes_result' in locals() else ""
        drivers = drivers_result if 'drivers_result' in locals() else ""
        
        combined = interpreter._combine_explanations(
            node_path, 1, 1, operational, verbatims, routes, drivers
        )
        print(f'   Combined result: "{combined}"')
        print(f'   Length: {len(combined)} chars')
    except Exception as e:
        print(f'   Combined error: {str(e)}')

if __name__ == "__main__":
    asyncio.run(test_explanation_methods()) 