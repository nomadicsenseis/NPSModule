#!/usr/bin/env python3
"""
Test script to verify routes classification by company (IB vs YW) 
and minimum threshold of 3 responses
"""

import asyncio
from dashboard_analyzer.anomaly_explanation.routes_analyzer import RoutesAnalyzer
from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector

async def test_routes_classification():
    """Test routes classification and threshold"""
    print("🧪 TESTING ROUTES CLASSIFICATION & THRESHOLD")
    print("="*60)
    
    # Initialize components
    collector = PBIDataCollector()
    analyzer = RoutesAnalyzer(collector)
    
    # Load routes data
    date = '2025-05-29'
    print(f"📅 Loading routes data for {date}...")
    await analyzer.load_routes_data(date, ['Global/SH/Economy/YW', 'Global/SH/Economy/IB'])
    
    # Test YW routes
    print(f"\n🔍 Testing YW Economy routes:")
    yw_routes = analyzer.get_most_affected_routes(date, 'Global/SH/Economy/YW', '-', 3)
    print(f"  📊 YW Routes found: {len(yw_routes)}")
    for route in yw_routes:
        print(f"    - {route['route']}: NPS={route['nps']:.1f}, Pax={route['pax']}")
    
    # Test IB routes
    print(f"\n🔍 Testing IB Economy routes:")
    ib_routes = analyzer.get_most_affected_routes(date, 'Global/SH/Economy/IB', '-', 3)
    print(f"  📊 IB Routes found: {len(ib_routes)}")
    for route in ib_routes:
        print(f"    - {route['route']}: NPS={route['nps']:.1f}, Pax={route['pax']}")
    
    # Verify no cross-contamination
    print(f"\n✅ VERIFICATION:")
    yw_route_names = [r['route'] for r in yw_routes]
    ib_route_names = [r['route'] for r in ib_routes]
    
    yw_has_ib = any(name.startswith('IB') for name in yw_route_names)
    ib_has_yw = any(name.startswith('YW') for name in ib_route_names)
    
    print(f"  - YW routes contain IB routes: {'❌ YES (PROBLEM!)' if yw_has_ib else '✅ NO (CORRECT)'}")
    print(f"  - IB routes contain YW routes: {'❌ YES (PROBLEM!)' if ib_has_yw else '✅ NO (CORRECT)'}")
    
    # Check threshold
    all_routes = yw_routes + ib_routes
    low_pax_routes = [r for r in all_routes if r['pax'] < 3]
    print(f"  - Routes with <3 responses: {len(low_pax_routes)} {'❌ (SHOULD BE 0)' if low_pax_routes else '✅ (CORRECT)'}")
    
    print(f"\n🏁 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_routes_classification()) 