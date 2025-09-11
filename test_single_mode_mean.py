#!/usr/bin/env python3
"""
Test script para operative_data_tool en modo single con mean anomaly detection
Configuración: 3 períodos objetivo, 4 períodos baseline
"""

import sys
import asyncio
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, '/app')

from dashboard_analyzer.main import execute_analysis_flow

async def test_single_mode_mean():
    """
    Test operative_data_tool en modo single con:
    - anomaly_detection_mode: "mean"
    - aggregation_days: 1 (días individuales)
    - baseline_periods: 4 (4 períodos de baseline)
    - período de 3 días: 2025-09-05 a 2025-09-07
    """
    print("🧪 TESTING SINGLE MODE WITH MEAN ANOMALY DETECTION")
    print("=" * 60)
    print("📊 Configuración:")
    print("   • Modo: single")
    print("   • Anomaly Detection: mean")
    print("   • Período objetivo: 3 días (2025-09-05 a 2025-09-07)")
    print("   • Aggregation days: 1 (días individuales)")
    print("   • Baseline periods: 4 (4 períodos de baseline)")
    print("   • Target periods: 3 (3 períodos objetivo)")
    print("   • Segmento: Global")
    print()
    
    try:
        # Test parameters
        segment = "Global"
        analysis_date = datetime(2025, 9, 7)  # Analysis date (end date)
        date_parameter = "2025-09-07"
        explanation_mode = "agent"
        anomaly_detection_mode = "mean"
        aggregation_days = 1  # Individual days
        baseline_periods = 4  # 4 baseline periods
        periods = 3  # 3 target periods
        study_mode = "single"  # CRITICAL: Single mode
        
        print(f"🎯 Test Parameters:")
        print(f"   • Segment: {segment}")
        print(f"   • Analysis Date: {analysis_date.strftime('%Y-%m-%d')}")
        print(f"   • Explanation Mode: {explanation_mode}")
        print(f"   • Anomaly Detection Mode: {anomaly_detection_mode}")
        print(f"   • Aggregation Days: {aggregation_days} (días individuales)")
        print(f"   • Baseline Periods: {baseline_periods}")
        print(f"   • Target Periods: {periods}")
        print(f"   • Study Mode: {study_mode}")
        print()
        
        # Call the execute_analysis_flow function
        print("🔧 Ejecutando execute_analysis_flow en modo single...")
        
        result = await execute_analysis_flow(
            analysis_date=analysis_date,
            date_parameter=date_parameter,
            segment=segment,
            explanation_mode=explanation_mode,
            anomaly_detection_mode=anomaly_detection_mode,
            baseline_periods=baseline_periods,
            aggregation_days=aggregation_days,
            periods=periods,
            causal_filter=None,  # No causal filter in single mode
            comparison_start_date=None,  # No specific comparison dates in single mode
            comparison_end_date=None,
            date_flight_local=None,
            study_mode=study_mode  # CRITICAL: Single mode
        )
        
        print("✅ RESULTADO COMPLETO:")
        print("=" * 60)
        print(result if result else "Analysis completed successfully")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single_mode_mean())
