#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_interpreter_agent import AnomalyInterpreterAgent
import asyncio
from datetime import datetime

async def test_interpreter_parsing():
    """Test the interpreter agent with the new tree format"""
    
    # Create a sample tree in the format that build_ai_input_string generates
    sample_tree_data = """🔍 DEBUG build_ai_input_string FILTERING:
   segment_filter: Economy/SH
   relevant_nodes: ['Global/SH/Economy', 'Global/SH/Economy/IB', 'Global/SH/Economy/YW']
   original anomalies: ['Global/SH/Economy', 'Global/SH/Economy/IB', 'Global/SH/Economy/YW']
   filtered_anomalies: ['Global/SH/Economy', 'Global/SH/Economy/IB', 'Global/SH/Economy/YW']
   original explanations: ['Global/SH/Economy', 'Global/SH/Economy/IB', 'Global/SH/Economy/YW']
   nps_values available: ['Global/SH/Economy', 'Global/SH/Economy/IB', 'Global/SH/Economy/YW']

SH Economy: NEGATIVE ANOMALY (-17.6 pts) (NPS: 22.4 vs baseline: 40.0)
  └─ ANALYSIS:
     • CAUSAL AGENT INVESTIGATION:
       1. Resumen de la anomalía
          • NPS Actual: 22.4
          • NPS Baseline: 40.0  
          • Variación vs baseline: –17.6 pts

       2. Causa/s encontradas y herramientas que la validan
          2.1. Factores operativos
            – Puntualidad
              • explanatory_drivers_tool: SHAP = –3.488, Sat_diff = –6.228 (20 encuestas)
              • ncs_tool: 450 incidentes de retraso
              • verbatims_tool: quejas recurrentes por "retrasos prolongados"

├─ IB: NEGATIVE ANOMALY (-17.5 pts) (NPS: 22.5 vs baseline: 40.0)
  └─ ANALYSIS:
     • CAUSAL AGENT INVESTIGATION:
       1. Resumen de la anomalía específica para IB
          • NPS Actual: 22.5
          • NPS Baseline: 40.0
          • Variación vs baseline: –17.5 pts

       2. Principales factores identificados
          – Problemas de puntualidad específicos de IB
          – Incidentes operativos en rutas IB

└─ YW: NEGATIVE ANOMALY (-17.7 pts) (NPS: 22.3 vs baseline: 40.0)  
  └─ ANALYSIS:
     • CAUSAL AGENT INVESTIGATION:
       1. Resumen de la anomalía específica para YW
          • NPS Actual: 22.3
          • NPS Baseline: 40.0
          • Variación vs baseline: –17.7 pts

       2. Principales factores identificados
          – Deterioro operativo en puntualidad YW
          – Mayor impacto en conexiones YW

INTERPRETATION INSTRUCTIONS:
• Focus ONLY on segments marked as 'POSITIVE ANOMALY' or 'NEGATIVE ANOMALY'
• 'Normal' segments (even with deviations) are NOT anomalies - they are expected variations
• Explain the root causes using the analysis data provided for anomalous segments
• Analysis scope limited to Economy/SH segment and its children"""

    print("🧪 Testing Interpreter Agent with New Tree Format")
    print("=" * 60)
    
    try:
        # Initialize the interpreter agent
        interpreter = AnomalyInterpreterAgent()
        print("✅ Interpreter agent initialized")
        
        # Test the parsing method directly
        hierarchy = interpreter._parse_hierarchy_from_explanations(sample_tree_data)
        
        print(f"📊 Parsed hierarchy: {len(hierarchy)} nodes found")
        for node_path, node_data in hierarchy.items():
            print(f"   • {node_path}: level {node_data['level']}, parent: {node_data.get('parent', 'None')}")
            print(f"     Content preview: {node_data['content'][:100]}...")
        
        # Test the full interpretation
        print("\n🤖 Testing Full Interpretation:")
        print("-" * 40)
        
        result = await interpreter.interpret_anomaly_tree_hierarchical(
            tree_data=sample_tree_data,
            date=datetime(2025, 6, 30)
        )
        
        print("✅ Interpretation completed successfully!")
        print(f"📝 Result length: {len(result)} characters")
        print(f"📋 Result preview:\n{result[:500]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_interpreter_parsing())
    if success:
        print("\n🎉 TEST PASSED: Interpreter can parse new tree format!")
    else:
        print("\n💥 TEST FAILED: Issues with parsing")
