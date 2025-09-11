#!/usr/bin/env python3
"""
Script para debuggear los 3 escenarios principales:
1. Modo single con anomaly_detection_mode="mean"
2. Modo comparative con causal_filter="vs LM" (Last Month)
3. Modo comparative con causal_filter="vs Sel. Period"
"""

import sys
import os
from datetime import datetime, timedelta
import asyncio

# Add the project root to the Python path
sys.path.insert(0, '/app')

from dashboard_analyzer.anomaly_explanation.genai_core.agents.causal_explanation_agent import CausalExplanationAgent

async def test_scenario_1_single_mean():
    """
    ESCENARIO 1: Modo single con anomaly_detection_mode="mean"
    - Compara operativa del período estudiado vs media de períodos baseline
    - Similar a como funciona NPS en modo single
    """
    print("🧪 ESCENARIO 1: SINGLE MODE + MEAN")
    print("=" * 60)
    
    try:
        # Create agent for single period analysis
        agent = CausalExplanationAgent(
            causal_filter=None,  # No causal filter in single mode
            comparison_start_date=None,
            comparison_end_date=None,
            study_mode="single"
        )
        
        print("✅ Agent created for single mode")
        
        # Test operative_data_tool_single_period
        node_path = "Global/SH/Economy"
        start_date = "2025-08-31"
        end_date = "2025-08-31" 
        baseline_start_date = "2025-08-17"  # 14 days prior
        
        print(f"📋 PARAMETERS:")
        print(f"  node_path: {node_path}")
        print(f"  start_date: {start_date}")
        print(f"  end_date: {end_date}")
        print(f"  baseline_start_date: {baseline_start_date}")
        print(f"  anomaly_detection_mode: mean")
        
        print(f"\n🚀 Testing operative_data_tool in single mode...")
        
        result = await agent._operative_data_tool_single_period(
            node_path=node_path,
            start_date=start_date,
            end_date=end_date,
            baseline_start_date=baseline_start_date,
            comparison_context="vs media de los últimos 7 días",
            baseline_periods=7,
            anomaly_detection_mode="mean"
        )
        
        print(f"✅ Call completed!")
        print(f"📏 Result length: {len(result)} characters")
        print(f"\n📋 RESULT:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_scenario_2_comparative_lm():
    """
    ESCENARIO 2: Modo comparative con causal_filter="vs LM" (Last Month)
    - Calcula dinámicamente las fechas de comparación (30 días atrás)
    - Usa query simplificada con fechas calculadas
    """
    print("\n🧪 ESCENARIO 2: COMPARATIVE MODE + vs LM (Last Month)")
    print("=" * 60)
    
    try:
        # Create agent for comparative analysis with Last Month filter
        agent = CausalExplanationAgent(
            causal_filter="vs LM",  # Last Month filter
            comparison_start_date=None,  # Will be calculated dynamically
            comparison_end_date=None,
            study_mode="comparative"
        )
        
        print("✅ Agent created for comparative mode with 'vs LM' filter")
        
        # Test operative_data_tool
        node_path = "Global/SH/Economy"
        start_date = "2025-05-01"
        end_date = "2025-05-31"  # May 2025
        
        print(f"📋 PARAMETERS:")
        print(f"  node_path: {node_path}")
        print(f"  start_date: {start_date}")
        print(f"  end_date: {end_date}")
        print(f"  causal_filter: vs LM")
        print(f"  Expected comparison period: April 2025 (30 days before)")
        
        print(f"\n🚀 Testing operative_data_tool in comparative mode...")
        
        result = await agent._operative_data_tool(
            node_path=node_path,
            start_date=start_date,
            end_date=end_date,
            comparison_days=31,  # May has 31 days
            comparison_mode="vslast_dynamic",  # Use dynamic mode
            baseline_periods=7
        )
        
        print(f"✅ Call completed!")
        print(f"📏 Result length: {len(result)} characters")
        print(f"\n📋 RESULT:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_scenario_3_comparative_sel_period():
    """
    ESCENARIO 3: Modo comparative con causal_filter="vs Sel. Period"
    - Usa fechas específicas proporcionadas explícitamente
    - Usa query simplificada con fechas exactas
    """
    print("\n🧪 ESCENARIO 3: COMPARATIVE MODE + vs Sel. Period")
    print("=" * 60)
    
    try:
        # Create agent for comparative analysis with specific selected period
        agent = CausalExplanationAgent(
            causal_filter="vs Sel. Period",
            comparison_start_date=datetime(2025, 4, 1),  # Explicit dates
            comparison_end_date=datetime(2025, 4, 30),
            study_mode="comparative"
        )
        
        print("✅ Agent created for comparative mode with 'vs Sel. Period' filter")
        
        # Test operative_data_tool
        node_path = "Global/SH/Economy"
        start_date = "2025-05-01"
        end_date = "2025-05-31"  # May 2025
        
        print(f"📋 PARAMETERS:")
        print(f"  node_path: {node_path}")
        print(f"  start_date: {start_date}")
        print(f"  end_date: {end_date}")
        print(f"  causal_filter: vs Sel. Period")
        print(f"  comparison_start_date: 2025-04-01")
        print(f"  comparison_end_date: 2025-04-30")
        
        print(f"\n🚀 Testing operative_data_tool in comparative mode...")
        
        result = await agent._operative_data_tool(
            node_path=node_path,
            start_date=start_date,
            end_date=end_date,
            comparison_days=31,  # May has 31 days
            comparison_mode="vslast_dynamic",  # Use dynamic mode
            baseline_periods=7
        )
        
        print(f"✅ Call completed!")
        print(f"📏 Result length: {len(result)} characters")
        print(f"\n📋 RESULT:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_dynamic_date_calculation():
    """
    Test adicional: Verificar que el cálculo dinámico de fechas funciona correctamente
    """
    print("\n🧪 BONUS TEST: DYNAMIC DATE CALCULATION")
    print("=" * 60)
    
    try:
        test_cases = [
            ("vs LM", datetime(2025, 5, 1), datetime(2025, 5, 31)),
            ("vs L7d", datetime(2025, 5, 1), datetime(2025, 5, 7)),
            ("vs Last Week", datetime(2025, 5, 1), datetime(2025, 5, 7)),
            ("vs LY", datetime(2025, 5, 1), datetime(2025, 5, 31)),
            ("vs L14d", datetime(2025, 5, 1), datetime(2025, 5, 14)),
        ]
        
        for causal_filter, start_date, end_date in test_cases:
            agent = CausalExplanationAgent(causal_filter=causal_filter)
            
            comp_start, comp_end = agent.calculate_dynamic_comparison_dates(start_date, end_date)
            
            print(f"🔄 {causal_filter}:")
            print(f"  Analysis: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print(f"  Comparison: {comp_start.strftime('%Y-%m-%d')} to {comp_end.strftime('%Y-%m-%d')}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main async function to run all tests"""
    print("🧪 DEBUG SCRIPT FOR THREE MAIN SCENARIOS")
    print("=" * 60)
    
    results = {}
    
    # Test Scenario 1: Single + Mean
    results["scenario_1"] = await test_scenario_1_single_mean()
    
    # Test Scenario 2: Comparative + LM
    results["scenario_2"] = await test_scenario_2_comparative_lm()
    
    # Test Scenario 3: Comparative + Sel. Period
    results["scenario_3"] = await test_scenario_3_comparative_sel_period()
    
    # Bonus test: Dynamic date calculation
    results["dynamic_dates"] = await test_dynamic_date_calculation()
    
    print("\n🏁 SUMMARY OF RESULTS")
    print("=" * 60)
    for scenario, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{scenario}: {status}")
    
    total_passed = sum(results.values())
    print(f"\nTotal: {total_passed}/{len(results)} scenarios passed")
    
    print("\n🏁 DEBUG SCRIPT COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
