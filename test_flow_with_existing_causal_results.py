#!/usr/bin/env python3
"""
Test script que ejecuta el flujo completo pero usa las explicaciones causales
ya generadas en lugar de ejecutar el causal agent desde cero.
"""

import asyncio
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

# Add the dashboard_analyzer to the path
sys.path.insert(0, '/app/dashboard_analyzer')

from anomaly_explanation.genai_core.agents.anomaly_interpreter_agent import AnomalyInterpreterAgent
from anomaly_explanation.genai_core.utils.enums import get_default_llm_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_flow")

def print_section(title, content=""):
    """Helper para imprimir secciones claramente."""
    logger.info(f"\n{'='*80}")
    logger.info(f"🔍 {title}")
    logger.info(f"{'='*80}")
    if content:
        logger.info(content)

def load_causal_explanation(file_path):
    """Carga una explicación causal desde un archivo JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)
            
        # Extract relevant information
        final_explanation = conversation_data.get('final_explanation', '')
        node_path = conversation_data.get('node_path', '')
        
        if not final_explanation or not node_path:
            logger.warning(f"⚠️ Incomplete data in {file_path}")
            return None
            
        logger.info(f"✅ Loaded explanation for {node_path}")
        logger.info(f"📏 Explanation length: {len(final_explanation)} characters")
        
        return {
            'node_path': node_path,
            'explanation': final_explanation,
            'conversation_data': conversation_data
        }
        
    except Exception as e:
        logger.error(f"❌ Error loading {file_path}: {e}")
        return None

def simulate_anomaly_tree_data():
    """Simula los datos del árbol de anomalías basándose en la última ejecución."""
    # Basado en los logs de la última ejecución
    tree_data = {
        'Global/SH/Economy': {
            'nps': 22.51751630828686,
            'baseline_nps': 40.0,
            'deviation': -17.48248369171314,
            'children': {
                'Global/SH/Economy/IB': {
                    'nps': 22.51751630828686,
                    'baseline_nps': 40.0,
                    'deviation': -17.48248369171314,
                    'has_causal_explanation': True
                },
                'Global/SH/Economy/YW': {
                    'nps': 22.327790973871736,
                    'baseline_nps': 40.0,
                    'deviation': -17.672209026128264,
                    'has_causal_explanation': True
                }
            }
        }
    }
    return tree_data

async def test_flow_with_existing_causal_results():
    """Test del flujo completo usando explicaciones causales existentes."""
    print_section("TESTING COMPLETE FLOW WITH EXISTING CAUSAL RESULTS")
    
    try:
        # 1. Simular el árbol de anomalías
        print_section("STEP 1: SIMULATING ANOMALY TREE")
        tree_data = simulate_anomaly_tree_data()
        logger.info(f"📊 Simulated tree with {len(tree_data)} root nodes")
        
        # 2. Cargar las explicaciones causales existentes
        print_section("STEP 2: LOADING EXISTING CAUSAL EXPLANATIONS")
        
        # Paths to the most recent causal explanation files
        causal_files = [
            "/app/dashboard_analyzer/agent_conversations/causal_explanation/causal_2025-06-01_2025-06-30_Global_SH_Economy_IB_20250911_120427.json",
            "/app/dashboard_analyzer/agent_conversations/causal_explanation/causal_2025-06-01_2025-06-30_Global_SH_Economy_YW_20250911_120948.json"
        ]
        
        causal_explanations = {}
        for file_path in causal_files:
            if Path(file_path).exists():
                explanation_data = load_causal_explanation(file_path)
                if explanation_data:
                    causal_explanations[explanation_data['node_path']] = explanation_data
            else:
                logger.warning(f"⚠️ File not found: {file_path}")
        
        if not causal_explanations:
            logger.error("❌ No causal explanations loaded. Cannot proceed.")
            return
            
        logger.info(f"✅ Loaded {len(causal_explanations)} causal explanations")
        
        # 3. Construir el input para el interpreter
        print_section("STEP 3: BUILDING INTERPRETER INPUT")
        
        # Simulate building the tree summary with the loaded explanations
        tree_summary_parts = []
        
        for node_path, explanation_data in causal_explanations.items():
            explanation = explanation_data['explanation']
            
            # Extract NPS values from the tree data or explanation
            node_info = f"NODO: {node_path}\n{explanation}\n"
            tree_summary_parts.append(node_info)
            
            logger.info(f"📝 Added explanation for {node_path} ({len(explanation)} chars)")
        
        # Combine all explanations
        combined_tree_summary = "\n".join(tree_summary_parts)
        
        logger.info(f"📏 Combined tree summary length: {len(combined_tree_summary)} characters")
        logger.info(f"🎯 Ready to send to interpreter")
        
        # 4. Inicializar el interpreter
        print_section("STEP 4: INITIALIZING INTERPRETER AGENT")
        
        ai_agent = AnomalyInterpreterAgent(
            llm_type=get_default_llm_type(),
            config_path="dashboard_analyzer/anomaly_explanation/config/prompts/anomaly_interpreter.yaml",
            logger=logger,
            study_mode="comparative"
        )
        
        logger.info(f"✅ Interpreter agent initialized")
        logger.info(f"📊 LLM Type: {get_default_llm_type()}")
        
        # 5. Ejecutar el interpreter
        print_section("STEP 5: RUNNING INTERPRETER")
        
        # Use the segment from one of the explanations
        main_segment = "Global/SH/Economy"
        analysis_date = "2025-06-30"
        
        logger.info(f"🎯 Interpreting tree for segment: {main_segment}")
        logger.info(f"📅 Analysis date: {analysis_date}")
        logger.info(f"📊 Input preview (first 500 chars):\n{combined_tree_summary[:500]}...")
        
        # Call the interpreter
        ai_interpretation = await asyncio.wait_for(
            ai_agent.interpret_anomaly_tree(
                combined_tree_summary,
                analysis_date,
                main_segment
            ),
            timeout=600.0
        )
        
        # 6. Mostrar resultados
        print_section("STEP 6: RESULTS", "✅ INTERPRETER COMPLETED SUCCESSFULLY!")
        
        logger.info(f"📝 INTERPRETATION LENGTH: {len(ai_interpretation)} characters")
        logger.info(f"📊 INTERPRETATION PREVIEW:\n{'-'*60}\n{ai_interpretation[:1000]}...\n{'-'*60}")
        
        # 7. Guardar el resultado (opcional)
        print_section("STEP 7: SAVING RESULTS")
        
        output_file = f"/app/test_interpreter_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("INTERPRETER RESULT - FLOW TEST WITH EXISTING CAUSAL EXPLANATIONS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Segment: {main_segment}\n")
            f.write(f"Analysis Date: {analysis_date}\n")
            f.write(f"LLM Type: {get_default_llm_type()}\n")
            f.write(f"Timestamp: {datetime.now()}\n\n")
            f.write("INPUT SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Causal explanations used: {len(causal_explanations)}\n")
            for node_path in causal_explanations.keys():
                f.write(f"  - {node_path}\n")
            f.write(f"\nCombined input length: {len(combined_tree_summary)} characters\n\n")
            f.write("INTERPRETER OUTPUT:\n")
            f.write("-" * 40 + "\n")
            f.write(ai_interpretation)
        
        logger.info(f"💾 Results saved to: {output_file}")
        
        print_section("FLOW TEST COMPLETED SUCCESSFULLY! 🎉")
        logger.info("✅ All steps completed without errors")
        logger.info(f"📊 Final result length: {len(ai_interpretation)} characters")
        
    except asyncio.TimeoutError:
        logger.error("❌ Interpreter call timed out.")
    except Exception as e:
        logger.error(f"❌ Error during flow test: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_flow_with_existing_causal_results())
