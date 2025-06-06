#!/usr/bin/env python3

import sys
sys.path.append('.')
from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector
from dashboard_analyzer.anomaly_explanation.routes_analyzer import RoutesAnalyzer
from datetime import datetime, timedelta
import asyncio

async def test_routes():
    collector = PBIDataCollector()
    routes_analyzer = RoutesAnalyzer(collector)
    
    # Test date and node
    date = "2025-01-20"
    node_path = 'Global/LH/Premium'
    anomaly_state = "+"
    
    print(f'Testing routes for {node_path} on {date} (anomaly: {anomaly_state})')
    print('='*70)
    
    # Test 1: Load routes data
    print('\n1. Testing routes data loading...')
    try:
        await routes_analyzer.load_routes_data(date, [node_path])
        print(f'   Routes data loaded successfully')
    except Exception as e:
        print(f'   Routes loading error: {str(e)[:200]}')
    
    # Test 2: Get affected routes
    print('\n2. Testing affected routes analysis...')
    try:
        top_routes = routes_analyzer.get_most_affected_routes(date, node_path, anomaly_state, top_n=3)
        print(f'   Found {len(top_routes)} affected routes')
        if top_routes:
            for i, route in enumerate(top_routes, 1):
                print(f'   Route {i}: {route}')
    except Exception as e:
        print(f'   Routes analysis error: {str(e)[:200]}')
    
    # Test 3: Format explanation
    print('\n3. Testing routes explanation formatting...')
    try:
        if 'top_routes' in locals() and top_routes:
            explanation = routes_analyzer.format_routes_explanation(top_routes, anomaly_state)
            print(f'   Explanation: {explanation}')
        else:
            print('   No routes to format explanation for')
    except Exception as e:
        print(f'   Explanation formatting error: {str(e)[:200]}')

if __name__ == "__main__":
    asyncio.run(test_routes()) 