#!/usr/bin/env python3
import asyncio
from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector
from dashboard_analyzer.anomaly_detection.anomaly_tree import AnomalyTree
from dashboard_analyzer.anomaly_detection.anomaly_interpreter import AnomalyInterpreter
from dashboard_analyzer.anomaly_explanation.anomaly_interpreter_agent import AnomalyInterpreterAgent
from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import LLMType
import logging

async def test_routes():
    # Use the folder that has data
    target_folder = '02_06_2025'
    
    tree = AnomalyTree()
    tree.build_tree_structure()
    tree.load_data(target_folder)
    tree.calculate_moving_averages()
    tree.detect_daily_anomalies(min_sample_size=5)
    
    available_dates = tree.dates
    if not available_dates:
        print('❌ No data available')
        return
        
    # Use the last available date
    date = available_dates[-1]
    print(f'🎯 Testing routes for date: {date}')
    
    # Initialize components
    collector = PBIDataCollector()
    interpreter = AnomalyInterpreter(pbi_collector=collector)
    
    # Test routes collection for one date
    print("\n📍 TESTING ROUTES COLLECTION:")
    print("="*50)
    routes_data = await interpreter.collect_routes_for_explanation_needed(tree, date)
    print(f'✅ Routes collection result: {len(routes_data)} segments with routes data')
    
    # Show some details
    for node_path, routes in routes_data.items():
        print(f'  📍 {node_path}: {len(routes)} routes')
        for route in routes[:2]:  # Show first 2 routes
            print(f'    - {route["route"]} ({route["country"]}) NPS: {route["nps"]:.1f}')
    
    # Test AI interpretation with O4_MINI
    if routes_data:
        print(f"\n🤖 TESTING AI INTERPRETATION WITH O4_MINI:")
        print("="*50)
        
        try:
            # Create AI agent with O4_MINI
            ai_agent = AnomalyInterpreterAgent(
                llm_type=LLMType.O4_MINI,
                logger=logging.getLogger("ai_interpreter")
            )
            
            # Build simple AI input string
            ai_input = f"Anomaly Tree for {date}:\n"
            if date in tree.daily_anomalies:
                anomalies = tree.daily_anomalies[date]
                for node_path, state in anomalies.items():
                    if state in ["+", "-"]:
                        ai_input += f"{node_path} [{state}]\n"
                        if node_path in routes_data:
                            routes = routes_data[node_path]
                            if routes:
                                route_info = f"Routes: {routes[0]['route']} NPS:{routes[0]['nps']:.1f}"
                                ai_input += f"  {route_info}\n"
            
            # Generate AI interpretation
            ai_interpretation = await ai_agent.interpret_anomaly_tree(
                tree_data=ai_input,
                date=date
            )
            
            print(ai_interpretation)
            
            # Show performance metrics
            metrics = ai_agent.get_performance_metrics()
            print(f"\n📊 AI Performance: {metrics['last_execution_time']:.2f}s, "
                  f"Tokens: {metrics['input_tokens']}+{metrics['output_tokens']}, "
                  f"Cost: ${metrics['money_spent']:.4f}")
            
        except Exception as e:
            print(f"❌ AI Interpretation with O4_MINI failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_routes()) 