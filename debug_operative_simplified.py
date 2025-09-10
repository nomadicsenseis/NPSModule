#!/usr/bin/env python3
"""
Script de debugging para la nueva query simplificada de operative vs Sel. Period
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import asyncio

# Add the project root to the Python path
sys.path.insert(0, '/app')

from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector
from dashboard_analyzer.anomaly_explanation.data_analyzer import OperationalDataAnalyzer

def test_simplified_query_generation():
    """Test 1: Generar y mostrar la query simplificada"""
    print("🔍 TEST 1: Query Generation")
    print("="*50)
    
    try:
        pbi_collector = PBIDataCollector()
        
        # Parámetros de prueba
        cabins = ['Economy']
        companies = ['IB', 'YW'] 
        hauls = ['SH']
        current_start_date = datetime(2025, 5, 1)
        current_end_date = datetime(2025, 5, 31)
        comparison_start_date = datetime(2025, 4, 1)
        comparison_end_date = datetime(2025, 4, 30)
        
        print(f"📋 PARAMETERS:")
        print(f"  Node filters: cabins={cabins}, companies={companies}, hauls={hauls}")
        print(f"  Current period: {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}")
        print(f"  Comparison period: {comparison_start_date.strftime('%Y-%m-%d')} to {comparison_end_date.strftime('%Y-%m-%d')}")
        
        # Generar query
        query = pbi_collector._get_operative_vs_sel_period_query(
            cabins, companies, hauls,
            current_start_date, current_end_date,
            comparison_start_date, comparison_end_date
        )
        
        print(f"\n✅ Query generated successfully!")
        print(f"📏 Query length: {len(query)} characters")
        print(f"\n📋 FULL QUERY:")
        print("="*80)
        print(query)
        print("="*80)
        
        return query
        
    except Exception as e:
        print(f"❌ ERROR generating query: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_query_execution(query):
    """Test 2: Ejecutar la query y mostrar resultados"""
    print("\n🔍 TEST 2: Query Execution")
    print("="*50)
    
    if not query:
        print("❌ No query provided")
        return None
        
    try:
        pbi_collector = PBIDataCollector()
        
        print("🚀 Executing query against Power BI...")
        result_df = pbi_collector._execute_query(query)
        
        if result_df.empty:
            print("❌ Query returned empty result")
            return None
            
        print(f"✅ Query executed successfully!")
        print(f"📊 Result shape: {result_df.shape}")
        print(f"📋 Columns: {list(result_df.columns)}")
        
        print(f"\n📋 FULL RESULT:")
        print("="*80)
        print(result_df.to_string(index=False))
        print("="*80)
        
        # Show detailed analysis
        print(f"\n🔍 DETAILED ANALYSIS:")
        for _, row in result_df.iterrows():
            metric = row.get('Metric', 'Unknown')
            current = pd.to_numeric(row.get('Current_Value'), errors='coerce')
            comparison = pd.to_numeric(row.get('Comparison_Value'), errors='coerce')
            diff = pd.to_numeric(row.get('Difference'), errors='coerce')
            change_pct = pd.to_numeric(row.get('Change_Pct'), errors='coerce')
            
            if pd.notna(current) and pd.notna(comparison):
                direction = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                significance = "🔥 SIGNIFICANT" if abs(change_pct) > 5 else "ℹ️ minor"
                
                print(f"  {direction} {metric}:")
                print(f"    Current: {current:.2f}")
                print(f"    Comparison: {comparison:.2f}")
                print(f"    Difference: {diff:+.2f}")
                print(f"    Change: {change_pct:+.1f}% {significance}")
            else:
                print(f"  ❌ {metric}: Missing data (current={current}, comparison={comparison})")
        
        return result_df
        
    except Exception as e:
        print(f"❌ ERROR executing query: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_analyzer_processing(result_df):
    """Test 3: Procesar datos con OperationalDataAnalyzer"""
    print("\n🔍 TEST 3: Analyzer Processing")
    print("="*50)
    
    if result_df is None or result_df.empty:
        print("❌ No data to process")
        return
        
    try:
        # Crear analyzer con fechas específicas
        analyzer = OperationalDataAnalyzer(
            comparison_mode="vslast_dynamic",
            comparison_start_date=datetime(2025, 4, 1),
            comparison_end_date=datetime(2025, 4, 30)
        )
        
        print(f"📋 ANALYZER CONFIG:")
        print(f"  comparison_mode: {analyzer.comparison_mode}")
        print(f"  comparison_start_date: {analyzer.comparison_start_date}")
        print(f"  comparison_end_date: {analyzer.comparison_end_date}")
        
        # Simular carga de datos (como hace causal_explanation_agent)
        node_path = "Global/SH/Economy"
        analyzer.operative_data[node_path] = result_df
        
        print(f"\n🚀 Processing data with analyzer...")
        print(f"  Data loaded for node: {node_path}")
        print(f"  Data shape: {result_df.shape}")
        
        # Llamar analyze_operative_metrics
        target_date = "2025-05-31"
        analysis_result = analyzer.analyze_operative_metrics(node_path, target_date)
        
        print(f"✅ Analysis completed!")
        print(f"📋 Result keys: {list(analysis_result.keys())}")
        
        if 'error' in analysis_result:
            print(f"❌ Analysis error: {analysis_result['error']}")
        else:
            print(f"\n📊 ANALYSIS RESULT:")
            print("="*80)
            
            # Show metrics
            if 'metrics' in analysis_result:
                print("🔢 METRICS:")
                for metric_name, metric_data in analysis_result['metrics'].items():
                    print(f"  {metric_name}:")
                    for key, value in metric_data.items():
                        print(f"    {key}: {value}")
                    print()
            
            # Show summary
            if 'summary' in analysis_result:
                print(f"📝 SUMMARY:")
                print(f"  {analysis_result['summary']}")
                print()
            
            # Show comparison info
            if 'comparison_info' in analysis_result:
                print(f"ℹ️ COMPARISON INFO:")
                for key, value in analysis_result['comparison_info'].items():
                    print(f"  {key}: {value}")
                
            print("="*80)
        
    except Exception as e:
        print(f"❌ ERROR in analyzer processing: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

def test_full_integration():
    """Test 4: Integración completa simulando causal_explanation_agent"""
    print("\n🔍 TEST 4: Full Integration Test")
    print("="*50)
    
    try:
        from dashboard_analyzer.anomaly_explanation.genai_core.agents.causal_explanation_agent import CausalExplanationAgent
        
        # Crear agente con fechas específicas (como en el comando real)
        agent = CausalExplanationAgent(
            causal_filter="vs Sel. Period",
            comparison_start_date=datetime(2025, 4, 1),
            comparison_end_date=datetime(2025, 4, 30)
        )
        
        print(f"📋 AGENT CONFIG:")
        print(f"  causal_filter: {agent.causal_filter}")
        print(f"  comparison_start_date: {agent.comparison_start_date}")
        print(f"  comparison_end_date: {agent.comparison_end_date}")
        
        # Simular llamada a _collect_operative_data_with_query_tracking
        print(f"\n🚀 Testing _collect_operative_data_with_query_tracking...")
        
        async def run_integration_test():
            result_df = await agent._collect_operative_data_with_query_tracking(
                node_path="Global/SH/Economy",
                target_date=datetime(2025, 5, 31),
                comparison_days=31,  # Para que coincida con --aggregation-days 31
                use_flexible=True
            )
            return result_df
        
        result_df = asyncio.run(run_integration_test())
        
        if result_df.empty:
            print("❌ Integration test returned empty DataFrame")
        else:
            print(f"✅ Integration test successful!")
            print(f"📊 Result shape: {result_df.shape}")
            print(f"📋 Columns: {list(result_df.columns)}")
            
            # Check if it's the simplified format
            if ('Metric' in result_df.columns and 'Current_Value' in result_df.columns):
                print("🎯 SUCCESS: Using simplified vs Sel. Period format!")
                print("\n📋 INTEGRATION RESULT:")
                print(result_df.to_string(index=False))
            else:
                print("⚠️ Using flexible format (not simplified)")
                print(result_df.head().to_string())
        
    except Exception as e:
        print(f"❌ ERROR in integration test: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 OPERATIVE SIMPLIFIED DEBUG SCRIPT")
    print("=" * 60)
    print("Testing the new simplified vs Sel. Period operative query")
    print("=" * 60)
    
    # Test 1: Query generation
    query = test_simplified_query_generation()
    
    # Test 2: Query execution
    result_df = test_query_execution(query)
    
    # Test 3: Analyzer processing
    test_analyzer_processing(result_df)
    
    # Test 4: Full integration
    test_full_integration()
    
    print("\n🏁 DEBUG SCRIPT COMPLETED")
    print("=" * 60)
