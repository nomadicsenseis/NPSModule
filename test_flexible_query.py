#!/usr/bin/env python3
"""
Test flexible NPS query with real Power BI data
"""

import asyncio
from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector

async def test_flexible_query():
    """Test flexible NPS query with real PBI data"""
    print('🧪 Testing flexible NPS query with real PBI data...')
    
    try:
        collector = PBIDataCollector()
        
        # Test with Global node (all segments)
        cabins = ['Business', 'Economy', 'Premium EC']
        companies = ['IB', 'YW'] 
        hauls = ['SH', 'LH']
        
        query = collector._get_flexible_nps_query(7, cabins, companies, hauls)
        print('📡 Generated query:')
        print(query[:300] + '...')
        
        df = await collector._execute_query_async(query)
        if not df.empty:
            df.columns = [col.strip('[]') for col in df.columns]
            print(f'✅ Got {len(df)} periods of data')
            print(f'📋 Columns: {list(df.columns)}')
            print('\n📊 Sample data:')
            print(df.head().to_string())
            
            # Save for testing
            df.to_csv('test_flexible_output.csv', index=False)
            print(f'\n💾 Saved to test_flexible_output.csv')
        else:
            print('❌ No data returned')
    
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_flexible_query()) 