#!/usr/bin/env python3
"""
Script de diagnóstico para el problema del summarizer que no encuentra secciones de análisis diarios.
"""

import os
import sys
import logging
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_summary_agent import AnomalySummaryAgent
from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import LLMType

def setup_logging():
    """Configurar logging detallado"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('diagnose_summary.log')
        ]
    )
    return logging.getLogger(__name__)

def test_section_patterns():
    """Probar los patrones de sección"""
    logger = setup_logging()
    
    # Crear agente de prueba
    agent = AnomalySummaryAgent(
        llm_type=LLMType.GPT4o_MINI,
        environment="local",
        logger=logger
    )
    
    # Probar diferentes segmentos
    test_segments = ['Global', 'SH', 'LH', 'Economy SH', 'Business SH']
    
    for segment in test_segments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Probando segmento: {segment}")
        logger.info(f"{'='*60}")
        
        patterns = agent._get_section_patterns_for_segment(segment)
        logger.info(f"Patrones encontrados para '{segment}': {len(patterns)}")
        
        for i, (section_name, pattern) in enumerate(patterns):
            logger.info(f"  {i+1}. {section_name}: {pattern[:80]}...")
    
    return agent

def test_daily_analysis_parsing():
    """Probar el parsing de análisis diarios"""
    logger = setup_logging()
    
    # Crear análisis diarios de ejemplo
    example_daily_analyses = """
📅 2025-01-20:
**SÍNTESIS EJECUTIVA**
Durante el día 20 de enero, el NPS global mostró una mejora de +5.2 pts.

**SHORT HAUL: Mejora moderada**
El radio corto registró un NPS de 45.3 con +3.1 pts.

**LONG HAUL: Estabilidad**
El radio largo mantuvo un NPS de 38.7 sin cambios significativos.

**ECONOMY SH: Desempeño positivo**
La cabina Economy de SH obtuvo 42.1 con +2.8 pts.

**BUSINESS SH: Ligera mejora**
La cabina Business de SH alcanzó 48.9 con +1.5 pts.

📅 2025-01-21:
**SÍNTESIS EJECUTIVA**
El día 21 de enero mostró una ligera caída de -2.1 pts en el NPS global.

**SHORT HAUL: Deterioro**
SH registró 41.8 con -3.5 pts, afectado por problemas de puntualidad.

**LONG HAUL: Mejora**
LH mejoró a 40.2 con +1.5 pts, gracias a mejor servicio a bordo.

**ECONOMY SH: Caída**
Economy SH cayó a 39.7 con -2.4 pts.

**BUSINESS SH: Estable**
Business SH se mantuvo en 48.5 con -0.4 pts.
"""
    
    logger.info(f"\n{'='*60}")
    logger.info("Probando parsing de análisis diarios")
    logger.info(f"{'='*60}")
    
    agent = AnomalySummaryAgent(
        llm_type=LLMType.GPT4o_MINI,
        environment="local",
        logger=logger
    )
    
    # Probar extracción de diferentes secciones
    test_sections = ['GLOBAL', 'SH', 'LH', 'ECONOMY SH', 'BUSINESS SH']
    
    for section in test_sections:
        logger.info(f"\n--- Probando sección: {section} ---")
        result = agent._filter_daily_for_section(example_daily_analyses, section)
        
        if "No se encontró" in result:
            logger.warning(f"❌ No se encontró la sección '{section}'")
        else:
            logger.info(f"✅ Sección '{section}' encontrada")
            logger.info(f"📄 Longitud del resultado: {len(result)} caracteres")
            logger.info(f"📝 Primeros 200 caracteres:\n{result[:200]}...")

def test_weekly_section_parsing():
    """Probar el parsing de secciones semanales"""
    logger = setup_logging()
    
    # Crear análisis semanal de ejemplo
    example_weekly_analysis = """
**SÍNTESIS EJECUTIVA**
Durante la semana del 20 al 26 de enero, el NPS global mostró una mejora de +3.8 pts.

A nivel diario, el fuerte repunte del 24-ene (+6.1 pts) explica gran parte de la mejora semanal.

**DETALLE POR AGREGACIÓN**

**SHORT HAUL: Mejora significativa**
El radio corto registró un NPS de 43.5 con +5.2 pts respecto a la semana anterior.

**LONG HAUL: Ligera mejora**
El radio largo alcanzó 39.8 con +1.1 pts.

**ECONOMY SH: Desempeño positivo**
La cabina Economy de SH obtuvo 40.9 con +4.3 pts.

**BUSINESS SH: Estabilidad**
La cabina Business de SH se mantuvo en 48.7 con +0.2 pts.

**ECONOMY SH IB: Mejora notable**
IB en Economy SH registró 39.8 con +5.1 pts.

**ECONOMY SH YW: Desempeño sólido**
YW en Economy SH alcanzó 42.5 con +3.8 pts.
"""
    
    logger.info(f"\n{'='*60}")
    logger.info("Probando parsing de secciones semanales")
    logger.info(f"{'='*60}")
    
    agent = AnomalySummaryAgent(
        llm_type=LLMType.GPT4o_MINI,
        environment="local",
        logger=logger
    )
    
    # Probar diferentes segmentos
    test_segments = ['Global', 'SH', 'Economy SH']
    
    for segment in test_segments:
        logger.info(f"\n--- Probando segmento: {segment} ---")
        sections = agent._parse_weekly_sections(example_weekly_analysis, segment)
        
        logger.info(f"📊 Secciones encontradas para '{segment}': {len(sections)}")
        for section_name, content in sections.items():
            logger.info(f"  • {section_name}: {len(content)} caracteres")
            if len(content) < 200:
                logger.info(f"    Contenido: {content}")
            else:
                logger.info(f"    Primeros 100 caracteres: {content[:100]}...")

def create_test_data_file():
    """Crear archivo de datos de prueba"""
    test_data = {
        "weekly_analysis": """
**SÍNTESIS EJECUTIVA**
Durante la semana del 20 al 26 de enero 2025, hemos identificado 3 factores clave que explican las variaciones de NPS. El NPS global subió **+3.2 pts** con respecto a la semana anterior.

A nivel diario, el fuerte repunte del **06-dic** (+6.1 pts) explica gran parte de la mejora semanal.

**DETALLE POR AGREGACIÓN**

**SHORT HAUL: Desempeño estable**
El radio corto registró un NPS de **36.6** con **+3.4 pts** con respecto a la semana anterior.

**LONG HAUL: Mejora moderada**
El radio largo alcanzó **42.1** con **+1.8 pts**.

**ECONOMY SH: Impacto positivo**
La cabina Economy de SH obtuvo **35.4** con **+3.5 pts**. Desglose por compañía: **IB** 34.2 con +3.8 pts, **YW** 37.1 con +3.1 pts.

**BUSINESS SH: Presión operativa**
En Business de SH, el NPS fue **29.3** con **–10.0 pts** con respecto a la semana anterior, afectado por cancelaciones en rutas domésticas.
""",
        "daily_analyses": [
            {
                "date": "2025-01-20",
                "analysis": """
📅 2025-01-20:
**SÍNTESIS EJECUTIVA**
Durante el día 20 de enero, el NPS global mostró una mejora de +5.2 pts.

**SHORT HAUL: Mejora moderada**
El radio corto registró un NPS de 45.3 con +3.1 pts.

**LONG HAUL: Estabilidad**
El radio largo mantuvo un NPS de 38.7 sin cambios significativos.

**ECONOMY SH: Desempeño positivo**
La cabina Economy de SH obtuvo 42.1 con +2.8 pts.

**BUSINESS SH: Ligera mejora**
La cabina Business de SH alcanzó 48.9 con +1.5 pts.
""",
                "anomalies": []
            },
            {
                "date": "2025-01-21",
                "analysis": """
📅 2025-01-21:
**SÍNTESIS EJECUTIVA**
El día 21 de enero mostró una ligera caída de -2.1 pts en el NPS global.

**SHORT HAUL: Deterioro**
SH registró 41.8 con -3.5 pts, afectado por problemas de puntualidad.

**LONG HAUL: Mejora**
LH mejoró a 40.2 con +1.5 pts, gracias a mejor servicio a bordo.

**ECONOMY SH: Caída**
Economy SH cayó a 39.7 con -2.4 pts.

**BUSINESS SH: Estable**
Business SH se mantuvo en 48.5 con -0.4 pts.
""",
                "anomalies": ["SH", "ECONOMY SH"]
            }
        ]
    }
    
    import json
    with open('test_summary_data.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Archivo de datos de prueba creado: test_summary_data.json")

def main():
    """Función principal"""
    print("🔍 Iniciando diagnóstico del problema del summarizer...")
    
    # Crear archivo de datos de prueba
    create_test_data_file()
    
    # Ejecutar pruebas
    print("\n1. Probando patrones de sección...")
    test_section_patterns()
    
    print("\n2. Probando parsing de análisis diarios...")
    test_daily_analysis_parsing()
    
    print("\n3. Probando parsing de secciones semanales...")
    test_weekly_section_parsing()
    
    print("\n✅ Diagnóstico completado.")
    print("📄 Revisa el archivo 'diagnose_summary.log' para ver los resultados detallados.")

if __name__ == "__main__":
    main()