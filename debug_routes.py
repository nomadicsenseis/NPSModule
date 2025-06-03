#!/usr/bin/env python3
"""
Debug script to investigate available routes and how to identify YW vs IB routes
"""

import asyncio
from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector

async def debug_routes():
    """Debug routes data to understand YW vs IB classification"""
    print("🔍 DEBUGGING ROUTES DATA")
    print("="*50)
    
    collector = PBIDataCollector()
    
    # Execute the routes query to see what routes we get
    with open('dashboard_analyzer/data_collection/queries/Rutas.txt', 'r') as f:
        query = f.read().replace('date(2025,05,12)', 'date(2025,05,29)')
    
    print("📡 Executing routes query...")
    df = await collector._execute_query_async(query)
    
    if df is not None and not df.empty:
        df.columns = [col.strip('[]') for col in df.columns]
        
        print(f"✅ Total routes found: {len(df)}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Find the route column (handle different possible names)
        route_col = None
        for col in df.columns:
            if 'route' in col.lower():
                route_col = col
                break
        
        # Find the company column (handle different possible names)
        company_col = None
        for col in df.columns:
            if 'company' in col.lower():
                company_col = col
                break
        
        print(f"🎯 Using route column: '{route_col}'")
        print(f"🎯 Using company column: '{company_col}'")
        
        if route_col and company_col:
            # Sample routes with companies
            print(f"\n📝 Sample routes with companies:")
            for i, (route, company) in enumerate(zip(df[route_col].head(10), df[company_col].head(10))):
                print(f"  {i+1}. {route} ({company})")
            
            # Count by company
            company_counts = df[company_col].value_counts()
            print(f"\n🏢 Routes by company:")
            for company, count in company_counts.items():
                print(f"  {company}: {count} routes")
            
            # Show YW routes
            yw_routes = df[df[company_col] == 'YW'][route_col].tolist()
            print(f"\n🔍 YW routes ({len(yw_routes)}):")
            for route in yw_routes[:10]:
                print(f"  - {route}")
            
            # Show IB routes  
            ib_routes = df[df[company_col] == 'IB'][route_col].tolist()
            print(f"\n🔍 IB routes ({len(ib_routes)}):")
            for route in ib_routes[:10]:
                print(f"  - {route}")
        
        elif route_col:
            # Fallback to old analysis if no company column
            print(f"\n📝 Sample routes:")
            for i, route in enumerate(df[route_col].head(10)):
                print(f"  {i+1}. {route}")
        else:
            print("❌ No route column found!")
    else:
        print("❌ No routes data found")

if __name__ == "__main__":
    asyncio.run(debug_routes()) 