#!/usr/bin/env python3
"""
Script simple para mostrar solo la interpretación final del intérprete
"""
import asyncio
import sys
import os

# Add the dashboard_analyzer to the path
sys.path.append('/app')

from dashboard_analyzer.main import build_ai_input_string

def load_causal_explanation(json_file_path):
    """Load causal explanation from JSON file"""
    try:
        import json
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
        
        return None
    except Exception as e:
        return None

async def show_interpretation():
    """Show only the final interpretation"""
    
    # Load explanations from existing files
    explanations = {}
    segment_files = {
        'Global/SH/Economy': '/app/dashboard_analyzer/agent_conversations/causal_explanation/causal_2025-06-01_2025-06-30_Global_SH_Economy_20250911_165402.json',
        'Global/SH/Economy/IB': '/app/dashboard_analyzer/agent_conversations/causal_explanation/causal_2025-06-01_2025-06-30_Global_SH_Economy_IB_20250911_165736.json',
        'Global/SH/Economy/YW': '/app/dashboard_analyzer/agent_conversations/causal_explanation/causal_2025-06-01_2025-06-30_Global_SH_Economy_YW_20250911_170256.json'
    }
    
    for segment, file_path in segment_files.items():
        if os.path.exists(file_path):
            explanation = load_causal_explanation(file_path)
            if explanation:
                explanations[segment] = explanation
    
    # Create test data
    period_anomalies = {
        'Global/SH/Economy': '-',
        'Global/SH/Economy/IB': '-', 
        'Global/SH/Economy/YW': '-'
    }
    
    period_deviations = {
        'Global/SH/Economy': -17.55,
        'Global/SH/Economy/IB': -17.48,
        'Global/SH/Economy/YW': -17.67
    }
    
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
    
    from datetime import datetime
    date_range = (datetime(2025, 6, 1), datetime(2025, 6, 30))
    
    # Build AI input
    ai_input = build_ai_input_string(
        1,  # period
        period_anomalies,
        period_deviations,
        {},  # parent_interpretations
        explanations,
        date_range,
        'Economy/SH',  # segment
        period_nps_values
    )
    
    # Initialize and run interpreter
    from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_interpreter_agent import AnomalyInterpreterAgent
    from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import get_default_llm_type
    import logging
    
    ai_agent = AnomalyInterpreterAgent(
        llm_type=get_default_llm_type(),
        config_path="/app/dashboard_analyzer/anomaly_explanation/config/prompts/anomaly_interpreter.yaml",
        logger=logging.getLogger("ai_interpreter"),
        study_mode="comparative"
    )
    
    print("🚀 Ejecutando intérprete de anomalías...")
    
    ai_interpretation = await ai_agent.interpret_anomaly_tree(ai_input, '2025-06-01', 'Economy/SH')
    
    print("\n" + "="*80)
    print("📊 INTERPRETACIÓN FINAL DEL ANÁLISIS DE ANOMALÍAS NPS")
    print("="*80)
    
    # Extract only the executive synthesis section
    if "📋 SÍNTESIS EJECUTIVA FINAL" in ai_interpretation:
        # Find the synthesis section
        synthesis_start = ai_interpretation.find("📋 SÍNTESIS EJECUTIVA FINAL")
        if synthesis_start != -1:
            # Find the end (next major section or end of text)
            synthesis_section = ai_interpretation[synthesis_start:]
            end_markers = ["---", "✅ **ANÁLISIS COMPLETADO**", "*Este análisis utiliza"]
            
            synthesis_end = len(synthesis_section)
            for marker in end_markers:
                marker_pos = synthesis_section.find(marker)
                if marker_pos != -1:
                    synthesis_end = min(synthesis_end, marker_pos)
            
            final_synthesis = synthesis_section[:synthesis_end].strip()
            print(final_synthesis)
        else:
            print("⚠️ No se encontró la sección de síntesis ejecutiva")
            print(ai_interpretation[:1000] + "..." if len(ai_interpretation) > 1000 else ai_interpretation)
    else:
        print("⚠️ Formato de interpretación inesperado - mostrando completo:")
        print(ai_interpretation)
    
    print("="*80)

if __name__ == "__main__":
    asyncio.run(show_interpretation())
