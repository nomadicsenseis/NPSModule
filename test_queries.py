#!/usr/bin/env python3

import sys
sys.path.append('.')
from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector
from datetime import datetime, timedelta
import asyncio

async def test_queries():
    collector = PBIDataCollector()
    
    # Test date range (recent dates)
    end_date = datetime(2025, 1, 20)
    start_date = end_date - timedelta(days=7)
    node_path = 'Global/LH/Premium'
    
    print(f'Testing queries for {node_path} from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
    print('='*70)
    
    # Test 1: Verbatims query
    print('\n1. Testing verbatims collection...')
    try:
        verbatims_df = collector.collect_verbatims_for_date_range(node_path, start_date, end_date)
        print(f'   Verbatims result: {verbatims_df.shape[0]} rows, {verbatims_df.shape[1]} columns')
        if not verbatims_df.empty:
            print(f'   Columns: {list(verbatims_df.columns)[:5]}...')
    except Exception as e:
        print(f'   Verbatims error: {str(e)[:200]}')
    
    # Test 2: Explanatory drivers query
    print('\n2. Testing explanatory drivers collection...')
    try:
        drivers_df = await collector.collect_explanatory_drivers_for_date_range(node_path, start_date, end_date)
        print(f'   Drivers result: {drivers_df.shape[0]} rows, {drivers_df.shape[1]} columns')
        if not drivers_df.empty:
            print(f'   Columns: {list(drivers_df.columns)[:5]}...')
    except Exception as e:
        print(f'   Drivers error: {str(e)[:200]}')
    
    # Test 3: Show the actual queries being generated
    print('\n3. Query generation test...')
    try:
        cabins, companies, hauls = collector._get_node_filters(node_path)
        print(f'   Filters - Cabins: {cabins}, Companies: {companies}, Hauls: {hauls}')
        
        verbatims_query = collector._get_verbatims_range_query(cabins, companies, hauls, start_date, end_date)
        print(f'   Verbatims query length: {len(verbatims_query)} chars')
        print(f'   Query preview: {verbatims_query[:200]}...')
        
        drivers_query = collector._get_explanatory_drivers_range_query(cabins, companies, hauls, start_date, end_date)
        print(f'   Drivers query length: {len(drivers_query)} chars')
        print(f'   Query preview: {drivers_query[:200]}...')
        
    except Exception as e:
        print(f'   Query generation error: {str(e)[:200]}')

if __name__ == "__main__":
    asyncio.run(test_queries()) 