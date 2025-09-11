#!/usr/bin/env python3
"""
Debug script específico para operative_data_tool en modo single
"""

import sys
import asyncio
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, '/app')

from dashboard_analyzer.anomaly_explanation.genai_core.agents.causal_explanation_agent import CausalExplanationAgent

async def debug_operative_single():
    """
    Debug específico de operative_data_tool en modo single
    """
    print("🐛 DEBUG OPERATIVE_DATA_TOOL SINGLE MODE")
    print("=" * 60)
    
    try:
        # Create agent
        print("📝 Creating CausalExplanationAgent...")
        agent = CausalExplanationAgent()
        
        # Test parameters
        node_path = "Global"
        start_date = "2025-09-07"  # String format
        end_date = "2025-09-07"    # String format
        baseline_start_date = "2025-09-03"  # 4 days before for baseline
        comparison_context = "mean baseline comparison"
        baseline_periods = 4
        anomaly_detection_mode = "mean"
        aggregation_days = 1
        
        print(f"🎯 Parameters:")
        print(f"   • node_path: {node_path}")
        print(f"   • start_date: {start_date} (type: {type(start_date)})")
        print(f"   • end_date: {end_date} (type: {type(end_date)})")
        print(f"   • baseline_start_date: {baseline_start_date} (type: {type(baseline_start_date)})")
        print(f"   • baseline_periods: {baseline_periods}")
        print(f"   • anomaly_detection_mode: {anomaly_detection_mode}")
        print(f"   • aggregation_days: {aggregation_days}")
        print()
        
        # Call the method directly
        print("🔧 Calling _operative_data_tool_single_period directly...")
        result = await agent._operative_data_tool_single_period(
            node_path=node_path,
            start_date=start_date,
            end_date=end_date,
            baseline_start_date=baseline_start_date,
            comparison_context=comparison_context,
            baseline_periods=baseline_periods,
            anomaly_detection_mode=anomaly_detection_mode,
            aggregation_days=aggregation_days
        )
        
        print("✅ SUCCESS!")
        print("-" * 40)
        print(result)
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        print("\n📍 Full traceback:")
        traceback.print_exc()
        
        # Let's also try to debug the data collection part separately
        print("\n🔍 Let's debug the data collection part...")
        try:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            print(f"✅ Date conversion works: {end_dt} (type: {type(end_dt)})")
            
            # Try to call the data collection method
            print("🔧 Testing _collect_operative_data_with_query_tracking...")
            operative_data = await agent._collect_operative_data_with_query_tracking(
                node_path=node_path,
                target_date=end_dt,
                comparison_days=14,
                use_flexible=True
            )
            print(f"✅ Data collection works: {len(operative_data)} rows")
            if not operative_data.empty:
                print(f"📊 Columns: {list(operative_data.columns)}")
                print(f"📅 Date range: {operative_data.get('Date_Master', ['N/A']).min()} to {operative_data.get('Date_Master', ['N/A']).max()}")
            
        except Exception as e2:
            print(f"❌ Data collection error: {type(e2).__name__}: {str(e2)}")
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_operative_single())
