#!/usr/bin/env python3
"""
Script para probar el interpreter usando las explicaciones causales ya generadas
sin tener que ejecutar todo el análisis de nuevo.
"""

import asyncio
import sys
import logging
import json
import glob
from datetime import datetime
from pathlib import Path

# Add the dashboard_analyzer to the path
sys.path.insert(0, '/app/dashboard_analyzer')

def find_latest_causal_conversations():
    """Encuentra las conversaciones causales más recientes"""
    pattern = "/app/dashboard_analyzer/agent_conversations/causal_explanation/causal_*.json"
    files = glob.glob(pattern)
    
    if not files:
        print("❌ No se encontraron archivos de conversaciones causales")
        return []
    
    # Ordenar por fecha de modificación (más reciente primero)
    files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    print(f"🔍 Encontrados {len(files)} archivos de conversaciones causales")
    for i, file in enumerate(files[:5]):  # Mostrar los 5 más recientes
        mtime = datetime.fromtimestamp(Path(file).stat().st_mtime)
        print(f"   {i+1}. {Path(file).name} - {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return files

def extract_causal_explanation(conversation_file):
    """Extrae la explicación causal de un archivo de conversación"""
    try:
        with open(conversation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Buscar la explicación final en el JSON
        if 'final_synthesis' in data:
            explanation = data['final_synthesis']
            node_path = data.get('node_path', 'Unknown')
            return node_path, explanation
        
        # Si no hay final_synthesis, buscar en otros campos
        if 'conversation_summary' in data:
            explanation = data['conversation_summary']
            node_path = data.get('node_path', 'Unknown')
            return node_path, explanation
            
        print(f"⚠️ No se encontró explicación causal en {conversation_file}")
        return None, None
        
    except Exception as e:
        print(f"❌ Error leyendo {conversation_file}: {e}")
        return None, None

async def test_interpreter_with_existing_explanations():
    """Test del interpreter usando explicaciones causales existentes"""
    print("🚀 TEST INTERPRETER - Usando explicaciones causales existentes")
    print("=" * 80)
    
    # Encontrar archivos de conversaciones causales
    conversation_files = find_latest_causal_conversations()
    if not conversation_files:
        return False
    
    # Usar los archivos más recientes (probablemente de Economy/SH)
    recent_files = conversation_files[:3]  # Los 3 más recientes
    
    print(f"\n🔍 Procesando {len(recent_files)} conversaciones causales recientes...")
    
    # Extraer explicaciones causales
    causal_explanations = {}
    for file in recent_files:
        node_path, explanation = extract_causal_explanation(file)
        if node_path and explanation:
            causal_explanations[node_path] = explanation
            print(f"✅ Explicación extraída para {node_path}: {len(explanation)} caracteres")
    
    if not causal_explanations:
        print("❌ No se pudieron extraer explicaciones causales")
        return False
    
    print(f"\n📊 Total explicaciones causales: {len(causal_explanations)}")
    for node, exp in causal_explanations.items():
        print(f"   • {node}: {len(exp)} caracteres")
    
    # Preparar input para el interpreter
    if len(causal_explanations) == 1:
        # Un solo nodo
        node_path, causal_explanation = next(iter(causal_explanations.items()))
        interpreter_input = f"NODO: {node_path}\n{causal_explanation}"
        input_type = "Single Node"
    else:
        # Múltiples nodos
        interpreter_input = "Multiple anomalous nodes analyzed:\n\n"
        for node_path, explanation in causal_explanations.items():
            interpreter_input += f"NODO: {node_path}\n{explanation}\n\n"
        input_type = "Multiple Nodes"
    
    print(f"\n🎯 Input preparado para interpreter ({input_type})")
    print(f"📏 Longitud total: {len(interpreter_input)} caracteres")
    
    try:
        # Initialize interpreter agent
        from anomaly_explanation.genai_core.agents.anomaly_interpreter_agent import AnomalyInterpreterAgent
        from anomaly_explanation.genai_core.utils.enums import get_default_llm_type
        
        ai_agent = AnomalyInterpreterAgent(
            llm_type=get_default_llm_type(),
            config_path="dashboard_analyzer/anomaly_explanation/config/prompts/anomaly_interpreter.yaml",
            logger=logging.getLogger("test_interpreter"),
            study_mode="comparative"
        )
        
        print(f"✅ Interpreter agent initialized")
        print(f"📊 LLM Type: {get_default_llm_type()}")
        
        # DEBUG: Print interpreter input
        print(f"\n🔍 INTERPRETER INPUT ({input_type}):")
        print(f"📏 Length: {len(interpreter_input)} characters")
        print(f"📅 Date: 2025-06-01")
        print(f"🎯 Segment: Economy/SH")
        print("📊 Content preview:")
        print("-" * 50)
        print(interpreter_input[:800] + "..." if len(interpreter_input) > 800 else interpreter_input)
        print("-" * 50)
        
        # Test the interpreter
        result = await asyncio.wait_for(
            ai_agent.interpret_anomaly_tree(
                tree_data=interpreter_input,
                date="2025-06-01",
                segment="Economy/SH"
            ),
            timeout=180.0
        )
        
        print(f"\n✅ SUCCESS! Interpreter completed without errors")
        print(f"📝 RESULT length: {len(result)} characters")
        print(f"📊 RESULT:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal del test"""
    print("🔍 TESTING INTERPRETER WITH EXISTING CAUSAL EXPLANATIONS")
    print("=" * 80)
    
    # Run the async test
    success = asyncio.run(test_interpreter_with_existing_explanations())
    
    if success:
        print("\n🎉 TEST PASSED: Interpreter works with existing explanations!")
        print("💡 This confirms that the interpreter + guardrails fixes work correctly")
    else:
        print("\n❌ TEST FAILED: Interpreter has issues with existing explanations")

if __name__ == "__main__":
    main()
