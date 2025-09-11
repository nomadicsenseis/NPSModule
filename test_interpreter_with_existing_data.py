#!/usr/bin/env python3
"""
Script para probar el intérprete usando explicaciones causales ya generadas
"""
import json
import asyncio
import sys
import os

# Add the dashboard_analyzer to the path
sys.path.append('/app')

from dashboard_analyzer.main import build_ai_input_string

def load_causal_explanation(json_file_path):
    """Load causal explanation from JSON file"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract the final explanation from the conversation_log
        if 'conversation_log' in data:
            messages = data['conversation_log']
            # Look for the final synthesis message (last AI message)
            for msg in reversed(messages):
                if msg.get('type') == 'AI' and 'FINAL_SYNTHESIS' in msg.get('content', ''):
                    content = msg['content']
                    # Remove the FINAL_SYNTHESIS prefix
                    if content.startswith('FINAL_SYNTHESIS: '):
                        content = content[17:]  # Remove "FINAL_SYNTHESIS: "
                    return f"🤖 **AGENT CAUSAL ANALYSIS**\n{content}"
        
        # Fallback: look for synthesis in the data
        if 'synthesis' in data:
            return f"🤖 **AGENT CAUSAL ANALYSIS**\n{data['synthesis']}"
            
        return None
    except Exception as e:
        print(f"Error loading {json_file_path}: {e}")
        return None

def create_test_data():
    """Create test data using existing causal explanations"""
    
    # Load explanations from the latest files
    explanations = {}
    
    # Map of segments to their latest JSON files
    segment_files = {
        'Global/SH/Economy': '/app/dashboard_analyzer/agent_conversations/causal_explanation/causal_2025-06-01_2025-06-30_Global_SH_Economy_20250911_165402.json',
        'Global/SH/Economy/IB': '/app/dashboard_analyzer/agent_conversations/causal_explanation/causal_2025-06-01_2025-06-30_Global_SH_Economy_IB_20250911_165736.json',
        'Global/SH/Economy/YW': '/app/dashboard_analyzer/agent_conversations/causal_explanation/causal_2025-06-01_2025-06-30_Global_SH_Economy_YW_20250911_170256.json'
    }
    
    print("🔍 Loading causal explanations from existing files...")
    for segment, file_path in segment_files.items():
        if os.path.exists(file_path):
            explanation = load_causal_explanation(file_path)
            if explanation:
                explanations[segment] = explanation
                print(f"   ✅ Loaded explanation for {segment}: {len(explanation)} chars")
            else:
                print(f"   ❌ Failed to load explanation for {segment}")
        else:
            print(f"   ⚠️ File not found: {file_path}")
    
    # Create mock anomalies data (based on what we saw in the terminal)
    period_anomalies = {
        'Global/SH/Economy': '-',
        'Global/SH/Economy/IB': '-', 
        'Global/SH/Economy/YW': '-'
    }
    
    # Create mock deviations data
    period_deviations = {
        'Global/SH/Economy': -17.55,
        'Global/SH/Economy/IB': -17.48,
        'Global/SH/Economy/YW': -17.67
    }
    
    # Create mock NPS values
    period_nps_values = {
        'Global/SH/Economy': {
            'current': 22.45,
            'baseline': 40.0,
            'deviation': -17.55,
            'baseline_description': 'período seleccionado (2025-05-01 a 2025-05-31)'
        },
        'Global/SH/Economy/IB': {
            'current': 22.52,
            'baseline': 40.0,
            'deviation': -17.48,
            'baseline_description': 'período seleccionado (2025-05-01 a 2025-05-31)'
        },
        'Global/SH/Economy/YW': {
            'current': 22.33,
            'baseline': 40.0,
            'deviation': -17.67,
            'baseline_description': 'período seleccionado (2025-05-01 a 2025-05-31)'
        }
    }
    
    # Create mock parent interpretations (empty for now)
    parent_interpretations = {}
    
    # Create mock date range
    from datetime import datetime
    date_range = (datetime(2025, 6, 1), datetime(2025, 6, 30))
    
    return {
        'period': 1,
        'period_anomalies': period_anomalies,
        'period_deviations': period_deviations,
        'parent_interpretations': parent_interpretations,
        'explanations': explanations,
        'date_range': date_range,
        'segment': 'Economy/SH',
        'period_nps_values': period_nps_values
    }

async def test_interpreter():
    """Test the interpreter with existing data"""
    
    print("🚀 Testing interpreter with existing causal explanations...")
    print("=" * 60)
    
    # Create test data
    test_data = create_test_data()
    
    print(f"\n📊 Test data summary:")
    print(f"   Period: {test_data['period']}")
    print(f"   Segment: {test_data['segment']}")
    print(f"   Anomalies: {list(test_data['period_anomalies'].keys())}")
    print(f"   Explanations: {list(test_data['explanations'].keys())}")
    print(f"   Date range: {test_data['date_range']}")
    
    # Test build_ai_input_string
    print(f"\n🔍 Testing build_ai_input_string...")
    
    try:
        ai_input = build_ai_input_string(
            test_data['period'],
            test_data['period_anomalies'],
            test_data['period_deviations'],
            test_data['parent_interpretations'],
            test_data['explanations'],
            test_data['date_range'],
            test_data['segment'],
            test_data['period_nps_values']
        )
        
        print(f"✅ build_ai_input_string completed successfully!")
        print(f"📏 AI input length: {len(ai_input)} characters")
        
        # Show the first part of the AI input
        print(f"\n📄 AI Input Preview (first 1000 chars):")
        print("-" * 50)
        print(ai_input[:1000])
        print("-" * 50)
        
        # Show if explanations are included
        if any(explanation in ai_input for explanation in test_data['explanations'].values()):
            print("✅ Causal explanations are included in the AI input!")
        else:
            print("❌ Causal explanations are NOT included in the AI input!")
        
        # Save AI input to file for inspection
        with open('/app/test_ai_input.txt', 'w', encoding='utf-8') as f:
            f.write(ai_input)
        print(f"💾 AI input saved to test_ai_input.txt for inspection")
        
        # Test with AI interpreter (optional)
        print(f"\n🤖 Testing AI Interpreter...")
        
        try:
            print("🔄 Importing AnomalyInterpreterAgent...")
            from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_interpreter_agent import AnomalyInterpreterAgent
            from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import get_default_llm_type
            import logging
            
            print("🔄 Creating AI Agent...")
            ai_agent = AnomalyInterpreterAgent(
                llm_type=get_default_llm_type(),
                config_path="/app/dashboard_analyzer/anomaly_explanation/config/prompts/anomaly_interpreter.yaml",
                logger=logging.getLogger("ai_interpreter"),
                study_mode="comparative"
            )
            
            print("✅ AI Agent initialized successfully!")
            
            # Call the interpreter
            start_date = test_data['date_range'][0].strftime('%Y-%m-%d')
            segment = test_data['segment']
            
            print(f"🔄 Calling interpreter with date={start_date}, segment={segment}...")
            print(f"🔄 AI input length: {len(ai_input)} characters")
            
            # Add timeout and more detailed error handling
            try:
                print("🔄 DEBUG TEST: About to call interpret_anomaly_tree", file=sys.stderr)
                ai_interpretation = await asyncio.wait_for(
                    ai_agent.interpret_anomaly_tree(ai_input, start_date, segment),
                    timeout=180.0  # 3 minutes timeout
                )
                print("🔄 DEBUG TEST: interpret_anomaly_tree returned", file=sys.stderr)
                print(f"🔄 DEBUG TEST: Received interpretation type: {type(ai_interpretation)}", file=sys.stderr)
                print(f"🔄 DEBUG TEST: Received interpretation length: {len(ai_interpretation) if ai_interpretation else 0}", file=sys.stderr)
                
                print("✅ AI Interpreter completed successfully!")
                print(f"📏 Interpretation length: {len(ai_interpretation)} characters")
                print(f"\n📄 SÍNTESIS FINAL DEL INTÉRPRETE:")
                print("=" * 80)
                print(ai_interpretation)
                print("=" * 80)
                
            except asyncio.TimeoutError:
                print("❌ AI Interpreter timed out after 3 minutes")
            except Exception as inner_e:
                print(f"❌ AI Interpreter execution failed: {str(inner_e)}")
                import traceback
                traceback.print_exc()
                
        except ImportError as import_e:
            print(f"❌ Import failed: {str(import_e)}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"❌ AI Interpreter initialization failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ build_ai_input_string failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_interpreter())
