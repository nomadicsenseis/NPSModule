# Guía de Integración con DynamoDB - Sistema de Agentes CATIA

Documentación completa para desarrolladores que necesiten crear agentes que accedan a las tablas DynamoDB del sistema CATIA Proactiva.

---

## 📋 Índice

1. [Arquitectura de Agentes](#1-arquitectura-de-agentes)
2. [Tablas DynamoDB](#2-tablas-dynamodb)
3. [Estructura Jerárquica de Segmentos](#3-estructura-jerárquica-de-segmentos)
4. [Guía de Uso para Desarrolladores](#4-guía-de-uso-para-desarrolladores)
5. [Ejemplos de Consultas](#5-ejemplos-de-consultas)
6. [Relaciones entre Tablas](#6-relaciones-entre-tablas)

---

## 1. Arquitectura de Agentes

### Flujo de Datos Jerárquico

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PIPELINE DE EJECUCIÓN                               │
│                         (execution_id = UUID único)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│    CAUSAL     │           │  INTERPRETER  │           │  SUMMARIZER   │
│    AGENT      │──────────▶│    AGENT      │──────────▶│    AGENT      │
│  (1 por nodo) │           │ (1 por fecha) │           │ (1 por fecha) │
└───────────────┘           └───────────────┘           └───────────────┘
        │                             │                             │
        │  • Global                   │  • Resume causales          │  • Resume interpreters
        │  • Global/LH                │    de mismo período         │    de múltiples días
        │  • Global/SH                │  • Genera Adaptive Card     │  • Genera reporte semanal
        │  • Global/SH/Economy        │  • Identifica NMAs          │  • Optimiza tamaño
        │  • Global/SH/Economy/IB     │                             │
        │  • ...                      │                             │
        ▼                             ▼                             ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│ CUSTOMER_CATIA_       │   │ CUSTOMER_CATIA_       │   │ CUSTOMER_CATIA_       │
│ CAUSAL_REPORTS        │   │ SYNTHESIS_REPORT      │   │ SYNTHESIS_REPORT      │
│ (múltiples items)     │   │ (agent_type=interpreter)│  │ (agent_type=summarizer)│
└───────────────────────┘   └───────────────────────┘   └───────────────────────┘
```

### Descripción de Responsabilidades

| Agente | Nivel | Función Principal | Input | Output |
|--------|-------|-------------------|-------|--------|
| **Causal** | Individual (por segmento) | Investiga causas de anomalías usando tools (NCS, operativos, verbatims, etc.) | Datos operativos, SHAP values | Explicación causal detallada |
| **Interpreter** | Agregado (por fecha) | Resume múltiples causales del mismo período en una narrativa coherente | Outputs de causales del mismo rango de fechas | Adaptive Card + Síntesis ejecutiva |
| **Summarizer** | Estratégico (multi-fecha) | Resume múltiples interpreters (diarios + semanal) en reporte ejecutivo | Outputs de interpreters de distintos días | Reporte consolidado + Adaptive Card optimizado |

---

## 2. Tablas DynamoDB

### 2.1 Tabla: `CUSTOMER_CATIA_CAUSAL_REPORTS`

**Propósito:** Almacena los resultados del agente causal (`CausalExplanationAgent`). Se genera **un registro por `node_path`** por ejecución.

#### Esquema de Claves

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `REPORT_ID` | `S` (PK) | Identificador compuesto: `causal_explanation#{node_path}#{start_date}#{end_date}` |
| `EXPORT_TIMESTAMP` | `S` (SK) | Timestamp ISO 8601 de cuándo se guardó el registro |

#### Dimensiones del Segmento

| Columna | Tipo | Descripción | Ejemplo |
|---------|------|-------------|---------|
| `node_path` | `S` | Ruta jerárquica completa del segmento analizado | `Global/SH/Economy/IB` |
| `segment` | `S` | Segmento raíz | `Global`, `Global/LH`, `Global/SH` |
| `haul` | `S` | Tipo de haul (nullable) | `LH`, `SH`, o `null` |
| `cabin` | `S` | Clase de cabina (nullable) | `Economy`, `Business`, `Premium EC`, o `null` |
| `company` | `S` | Compañía (nullable) | `IB`, `YW`, o `null` |

#### Contexto de Ejecución

| Columna | Tipo | Descripción | Valores Típicos |
|---------|------|-------------|-----------------|
| `execution_id` | `S` | UUID compartido entre todos los agentes de una misma ejecución | `abc-123-def-456` |
| `execution_timestamp` | `S` | ISO 8601 del momento de ejecución | `2026-03-24T14:13:04Z` |
| `analysis_start_date` | `S` | Fecha inicio del período analizado | `2026-03-17` |
| `analysis_end_date` | `S` | Fecha fin del período analizado | `2026-03-24` |
| `comparison_start_date` | `S` | Fecha inicio del período de comparación | `2026-03-10` |
| `comparison_end_date` | `S` | Fecha fin del período de comparación | `2026-03-17` |
| `causal_filter` | `S` | Filtro de comparación usado | `vs L7d`, `vs LM`, `vs LY`, `vs Sel. Period` |
| `anomaly_type` | `S` | Tipo de anomalía detectada | `positive`, `negative` |
| `anomaly_detection_mode` | `S` | Modo de detección | `vslast`, `vslast_dynamic`, `target`, `mean` |
| `study_mode` | `S` | Modo de estudio | `comparative`, `single` |
| `focus_touchpoint` | `S` | Touchpoint focalizado (opcional) | `Cabin-Crew`, `Check-in`, etc. |
| `report_group` | `S` | Grupo del reporte | `general`, `cabin-crew`, etc. |
| `llm_type` | `S` | Modelo LLM utilizado | `GPT4o`, `CLAUDE_3_5_SONNET`, etc. |
| `environment` | `S` | Entorno de ejecución | `prod`, `sbx`, `local` |

#### Resultados por Tool (Output Crudo)

Cada columna contiene el resultado JSON serializado de la herramienta correspondiente:

| Columna | Origen en el Código | Descripción |
|---------|---------------------|-------------|
| `explanatory_drivers_result` | `collected_data['explanatory_drivers']` | Resultado del análisis de drivers SHAP |
| `operative_data_result` | `collected_data['operative_data']` | Datos operativos (OTP, delays, etc.) |
| `ncs_result` | `collected_data['ncs_data']` | Datos de NCS (Net Content Score) |
| `routes_result` | `collected_data['routes_data']` | Análisis por rutas |
| `verbatims_result` | `collected_data['verbatims']` o `verbatims_conversation` | Verbatims del chatbot |
| `customer_profile_result` | `collected_data['customer_profile']` | Perfil de cliente |
| `focus_touchpoint_result` | `collected_data['focus_touchpoint_enrichment']` | Enriquecimiento del touchpoint focalizado |

#### Resultado Final

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `final_synthesis` | `S` | Explicación causal final generada por el LLM (texto) |
| `identified_routes` | `L` | Lista de rutas extraídas durante la investigación (e.g., `["MAD-BCN", "MAD-JFK"]` ) |
| `survey_count` | `N` | Número de encuestas analizadas |

#### Métricas Operativas

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `tools_used` | `L` | Lista de herramientas invocadas (e.g., `["explanatory_drivers", "operative_data", "verbatims"]` ) |
| `iteration_count` | `N` | Número de iteraciones del agente |
| `total_messages` | `N` | Mensajes en el conversation log |
| `investigation_success` | `BOOL` | Si se generaron explicaciones exitosamente |
| `dax_queries_count` | `N` | Número de queries DAX ejecutadas |
| `execution_duration_ms` | `N` | Tiempo total de ejecución en milisegundos |
| `status` | `S` | Estado de la ejecución: `completed`, `failed`, `timeout` |
| `error_message` | `S` | Mensaje de error si `status != completed` |
| `s3_report_key` | `S` | Ruta S3 del reporte completo (si se guardó en S3) |
| `conversation_log` | `S` | JSON string del log completo de conversación |
| `ttl` | `N` | Epoch timestamp para auto-eliminación (recomendado: 90 días) |

---

### 2.2 Tabla: `CUSTOMER_CATIA_SYNTHESIS_REPORT`

**Propósito:** Almacena los resultados del **interpreter** (`AnomalyInterpreterAgent`) y **summarizer** (`AnomalySummaryAgent`). Se genera **un registro por agente** por ejecución.

#### Esquema de Claves

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `REPORT_ID` | `S` (PK) | `interpreter#{segment}#{date}` o `summarizer#{segment}#{date}` |
| `EXPORT_TIMESTAMP` | `S` (SK) | Timestamp ISO 8601 de cuándo se guardó el registro |

#### Contexto de Ejecución (Común)

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `execution_id` | `S` | UUID del pipeline (mismo que en tabla causal) - **CLAVE DE VINCULACIÓN** |
| `agent_type` | `S` | Tipo de agente: `interpreter` o `summarizer` |
| `execution_timestamp` | `S` | ISO 8601 |
| `analysis_date` | `S` | Fecha o rango analizado (e.g., `2026-03-24` o `2026-03-17_to_2026-03-24`) |
| `period_type` | `S` | `weekly` o `daily` |
| `period_range` | `S` | Rango estándar: `2026-03-17_2026-03-24` |
| `analysis_start_date` | `S` | Fecha inicio período |
| `analysis_end_date` | `S` | Fecha fin período |
| `comparison_start_date` | `S` | Fecha inicio comparación |
| `comparison_end_date` | `S` | Fecha fin comparación |
| `causal_filter` | `S` | Filtro de comparación usado |
| `anomaly_detection_mode` | `S` | Modo de detección |
| `study_mode` | `S` | `comparative` o `single` |
| `segment` | `S` | Segmento raíz (normalmente `Global`) |
| `focus_touchpoint` | `S` | Touchpoint de foco si aplica |
| `report_group` | `S` | `general`, `cabin-crew`, etc. |
| `llm_type` | `S` | Modelo LLM utilizado |
| `aggregation_days` | `N` | Días de agregación |
| `baseline_periods` | `N` | Períodos de baseline |
| `environment` | `S` | `prod`, `sbx`, `local` |

#### Campos Específicos del INTERPRETER

Pasos de la conversación jerárquica (cada uno contiene el output del LLM para ese paso):

| Columna | Step Interno | Descripción |
|---------|--------------|-------------|
| `step1_company_level_diagnosis` | `COMPANY_LEVEL_DIAGNOSIS` | Diagnóstico a nivel compañía (IB/YW) |
| `step2_cabin_level_diagnosis` | `CABIN_LEVEL_DIAGNOSIS` | Diagnóstico a nivel de cabina (Economy/Business/Premium) |
| `step3_radio_global_diagnosis` | `RADIO_GLOBAL_DIAGNOSIS` | Diagnóstico Radio (LH/SH) vs Global |
| `step4_nma_identification` | `NMA_IDENTIFICATION` | Identificación de Nodos Máximo Afectados |
| `step4b_evidence_extraction` | `EVIDENCE_EXTRACTION` | Extracción de evidencias para NMAs |
| `step4c_cabin_radio_reflection` | `CABIN_RADIO_REFLECTION` | Reflexión sobre cabin-radios (solo si aplica) |
| `step5_executive_synthesis` | `EXECUTIVE_SYNTHESIS` | Síntesis ejecutiva |
| `step6_adaptive_card_json` | `ADAPTIVE_CARD_GENERATION` | Adaptive Card generado (JSON) |
| `step7_tone_modernization` | `TONE_MODERNIZATION` | Modernización del tono del texto |

**Métricas del Interpreter:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `hierarchy_total_nodes` | `N` | Total de nodos en la jerarquía |
| `hierarchy_generations` | `N` | Generaciones analizadas |
| `hierarchy_summary` | `S` | JSON con resumen de la jerarquía |
| `nma_list` | `L` | Lista de Nodos Máximo Afectados |
| `generation_reflections` | `S` | JSON con reflexiones por generación |

#### Campos Específicos del SUMMARIZER

Pasos principales de generación de contenido:

| Columna | Step Interno | Descripción |
|---------|--------------|-------------|
| `summ_step1_section_connections` | - | Conexiones diario-semanal por sección |
| `summ_step2_full_report` | - | Reporte final integrado |
| `summ_step3_executive_synthesis` | - | Síntesis ejecutiva extraída |
| `summ_step4_adaptive_card` | - | Adaptive card inicial generado |

Pasos de optimización del adaptive card:

| Columna | Step Interno | Descripción |
|---------|--------------|-------------|
| `summ_step5_modernize_tone` | `step5_modernize_tone` | Modernización de tono |
| `summ_step6a_trim_low_impact_days` | `step6a_trim_low_impact_days` | Recorte de días de bajo impacto |
| `summ_step6b_remove_subsegment_daily` | `step6b_remove_subsegment_daily_context` | Eliminación de contexto diario de subsegmentos |
| `summ_step6c_summarize_cabin_haul` | `step6c_summarize_cabin_haul_company_weekly` | Resumen cabin/haul/company weekly |
| `summ_step6d_shorten_cabin_haul` | `step6d_shorten_cabin_haul_weekly` | Acortamiento cabin/haul weekly |
| `summ_step6e_shorten_overall_global` | `step6e_shorten_overall_global` | Acortamiento overall global |
| `summ_step7_validate_and_fix_json` | `step7_validate_and_fix_json` | Validación y corrección JSON final |

**Métricas del Summarizer:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `adaptive_card_size_kb` | `N` | Tamaño del adaptive card en KB |
| `num_daily_analyses_included` | `N` | Número de análisis diarios incluidos |
| `daily_analysis_dates` | `L` | Fechas de los días incluidos (e.g., `["2026-03-17", "2026-03-18", ...]` ) |

#### Campos Comunes Finales (Ambos Agentes)

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `final_adaptive_card_json` | `S` | Adaptive card final (resultado de todo el proceso) - **Campo más importante** |
| `final_executive_synthesis` | `S` | Texto de síntesis ejecutiva final |
| `llm_calls_count` | `N` | Número de llamadas al LLM |
| `execution_duration_ms` | `N` | Tiempo total de ejecución |
| `conversation_log` | `S` | JSON string del log completo (null para summarizer) |
| `status` | `S` | `completed`, `failed`, `partial` |
| `error_message` | `S` | Error si aplica |
| `s3_report_key` | `S` | Ruta S3 del reporte |
| `source_causal_report_ids` | `L` | IDs de reportes causales fuente - **VINCULA CON TABLA 1** |
| `ttl` | `N` | TTL epoch para auto-eliminación |

---

## 3. Estructura Jerárquica de Segmentos

### Árbol de Segmentos

```
Global (Nivel 0 - Raíz)
│
├── Global/LH (Nivel 1 - Long Haul)
│   ├── Global/LH/Economy (Nivel 2)
│   │   ├── Global/LH/Economy/IB (Nivel 3)
│   │   └── Global/LH/Economy/YW (Nivel 3)
│   ├── Global/LH/Business (Nivel 2)
│   │   ├── Global/LH/Business/IB (Nivel 3)
│   │   └── Global/LH/Business/YW (Nivel 3)
│   └── Global/LH/Premium EC (Nivel 2)
│       ├── Global/LH/Premium EC/IB (Nivel 3)
│       └── Global/LH/Premium EC/YW (Nivel 3)
│
└── Global/SH (Nivel 1 - Short Haul)
    ├── Global/SH/Economy (Nivel 2)
    │   ├── Global/SH/Economy/IB (Nivel 3)
    │   └── Global/SH/Economy/YW (Nivel 3)
    ├── Global/SH/Business (Nivel 2)
    │   ├── Global/SH/Business/IB (Nivel 3)
    │   └── Global/SH/Business/YW (Nivel 3)
    └── Global/SH/Premium EC (Nivel 2)
        ├── Global/SH/Premium EC/IB (Nivel 3)
        └── Global/SH/Premium EC/YW (Nivel 3)
```

### Componentes del `node_path`

| Partes | Formato | Significado |
|--------|---------|-------------|
| Parte 0 | `Global` | Siempre presente, raíz del análisis |
| Parte 1 | `LH` / `SH` | Haul (nullable para Global puro) |
| Parte 2 | `Economy` / `Business` / `Premium EC` | Cabina (nullable) |
| Parte 3 | `IB` / `YW` | Compañía (nullable) |

### Función de Parseo (Python)

```python
def _parse_node_path(node_path: str) -> dict:
    """Derive haul / cabin / company from a node_path like 'Global/SH/Economy/IB'."""
    parts = node_path.split("/") if node_path else []
    haul = parts[1] if len(parts) >= 2 and parts[1] in ("LH", "SH") else None
    cabin_raw = parts[2] if len(parts) >= 3 else None
    cabin = "Premium EC" if cabin_raw == "Premium" else cabin_raw
    company = parts[3] if len(parts) >= 4 and parts[3] in ("IB", "YW") else None
    segment = "/".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "Global")
    return {"segment": segment, "haul": haul, "cabin": cabin, "company": company}
```

---

## 4. Guía de Uso para Desarrolladores

### 4.1 Configuración Inicial

```python
import boto3
from decimal import Decimal
import json
from datetime import datetime
import time

# Configuración de tablas
CAUSAL_TABLE = "CUSTOMER_CATIA_CAUSAL_REPORTS"
SYNTHESIS_TABLE = "CUSTOMER_CATIA_SYNTHESIS_REPORT"
TTL_SECONDS = 7776000  # 90 días

def get_dynamodb_resource(environment="prod", region="eu-west-1"):
    """Obtener recurso DynamoDB configurado."""
    # Usar credenciales de AWS según tu configuración
    session = boto3.Session(region_name=region)
    return session.resource("dynamodb", region_name=region)

def calculate_ttl(days=90):
    """Calcular timestamp TTL."""
    return int(time.time()) + (days * 24 * 60 * 60)
```

### 4.2 Guardar un Reporte Causal

```python
def save_causal_report(
    dynamodb_resource,
    execution_id: str,
    node_path: str,
    analysis_start_date: str,
    analysis_end_date: str,
    final_synthesis: str,
    tools_used: list,
    environment: str = "prod"
):
    """
    Guarda un reporte causal en DynamoDB.
    
    Args:
        execution_id: UUID compartido entre todos los agentes del pipeline
        node_path: Ruta jerárquica (e.g., "Global/SH/Economy/IB")
        analysis_start_date: Fecha inicio (YYYY-MM-DD)
        analysis_end_date: Fecha fin (YYYY-MM-DD)
        final_synthesis: Explicación causal generada
        tools_used: Lista de tools invocados
        environment: prod/sbx/local
    """
    table = dynamodb_resource.Table(CAUSAL_TABLE)
    
    # Construir REPORT_ID
    report_id = f"causal_explanation#{node_path}#{analysis_start_date}#{analysis_end_date}"
    export_timestamp = datetime.utcnow().isoformat()
    
    # Parsear node_path
    parts = node_path.split("/")
    haul = parts[1] if len(parts) >= 2 and parts[1] in ("LH", "SH") else None
    cabin = parts[2] if len(parts) >= 3 else None
    company = parts[3] if len(parts) >= 4 else None
    segment = "/".join(parts[:2]) if len(parts) >= 2 else "Global"
    
    item = {
        # Claves
        "REPORT_ID": report_id,
        "EXPORT_TIMESTAMP": export_timestamp,
        
        # Correlación
        "execution_id": execution_id,
        
        # Dimensiones
        "node_path": node_path,
        "segment": segment,
        "haul": haul,
        "cabin": cabin,
        "company": company,
        
        # Contexto
        "execution_timestamp": export_timestamp,
        "analysis_start_date": analysis_start_date,
        "analysis_end_date": analysis_end_date,
        "environment": environment,
        
        # Resultado
        "final_synthesis": final_synthesis,
        "tools_used": tools_used,
        "status": "completed",
        "ttl": calculate_ttl()
    }
    
    # Eliminar valores None
    item = {k: v for k, v in item.items() if v is not None}
    
    table.put_item(Item=item)
    return report_id
```

### 4.3 Guardar un Reporte de Interpreter

```python
def save_interpreter_report(
    dynamodb_resource,
    execution_id: str,
    segment: str,
    analysis_date: str,
    period_type: str,  # "weekly" o "daily"
    step_responses: dict,  # {step_name: content}
    final_adaptive_card_json: str,
    final_executive_synthesis: str,
    source_causal_ids: list,  # IDs de causales que alimentaron este interpreter
    environment: str = "prod"
):
    """
    Guarda un reporte de interpreter en DynamoDB.
    
    Args:
        execution_id: UUID compartido con causales
        segment: Segmento analizado (normalmente "Global")
        analysis_date: Fecha del análisis
        period_type: "weekly" o "daily"
        step_responses: Dict con respuestas de cada paso
        final_adaptive_card_json: JSON string del Adaptive Card final
        final_executive_synthesis: Síntesis ejecutiva
        source_causal_ids: Lista de REPORT_IDs de causales fuente
        environment: prod/sbx/local
    """
    table = dynamodb_resource.Table(SYNTHESIS_TABLE)
    
    report_id = f"interpreter#{segment}#{analysis_date}"
    export_timestamp = datetime.utcnow().isoformat()
    
    item = {
        # Claves
        "REPORT_ID": report_id,
        "EXPORT_TIMESTAMP": export_timestamp,
        
        # Tipo y correlación
        "agent_type": "interpreter",
        "execution_id": execution_id,
        
        # Contexto
        "execution_timestamp": export_timestamp,
        "analysis_date": analysis_date,
        "period_type": period_type,
        "segment": segment,
        "environment": environment,
        
        # Steps del interpreter
        "step1_company_level_diagnosis": step_responses.get("COMPANY_LEVEL_DIAGNOSIS"),
        "step2_cabin_level_diagnosis": step_responses.get("CABIN_LEVEL_DIAGNOSIS"),
        "step3_radio_global_diagnosis": step_responses.get("RADIO_GLOBAL_DIAGNOSIS"),
        "step4_nma_identification": step_responses.get("NMA_IDENTIFICATION"),
        "step4b_evidence_extraction": step_responses.get("EVIDENCE_EXTRACTION"),
        "step4c_cabin_radio_reflection": step_responses.get("CABIN_RADIO_REFLECTION"),
        "step5_executive_synthesis": step_responses.get("EXECUTIVE_SYNTHESIS"),
        "step6_adaptive_card_json": step_responses.get("ADAPTIVE_CARD_GENERATION"),
        "step7_tone_modernization": step_responses.get("TONE_MODERNIZATION"),
        
        # Resultados finales
        "final_adaptive_card_json": final_adaptive_card_json,
        "final_executive_synthesis": final_executive_synthesis,
        "source_causal_report_ids": source_causal_ids,
        "status": "completed",
        "ttl": calculate_ttl()
    }
    
    # Eliminar valores None
    item = {k: v for k, v in item.items() if v is not None}
    
    table.put_item(Item=item)
    return report_id
```

### 4.4 Guardar un Reporte de Summarizer

```python
def save_summarizer_report(
    dynamodb_resource,
    execution_id: str,
    segment: str,
    analysis_date: str,
    final_adaptive_card_json: str,
    final_executive_synthesis: str,
    daily_analysis_dates: list,  # Fechas diarias incluidas
    optimization_steps: dict,  # Resultados de cada paso de optimización
    environment: str = "prod"
):
    """
    Guarda un reporte de summarizer en DynamoDB.
    
    Args:
        execution_id: UUID compartido
        segment: Segmento analizado
        analysis_date: Fecha del reporte resumido
        final_adaptive_card_json: JSON final optimizado
        final_executive_synthesis: Síntesis ejecutiva
        daily_analysis_dates: Lista de fechas diarias procesadas
        optimization_steps: Dict con resultados de optimización
        environment: prod/sbx/local
    """
    table = dynamodb_resource.Table(SYNTHESIS_TABLE)
    
    report_id = f"summarizer#{segment}#{analysis_date}"
    export_timestamp = datetime.utcnow().isoformat()
    
    # Calcular tamaño en KB
    adaptive_card_size_kb = len(final_adaptive_card_json.encode("utf-8")) / 1024
    
    item = {
        # Claves
        "REPORT_ID": report_id,
        "EXPORT_TIMESTAMP": export_timestamp,
        
        # Tipo y correlación
        "agent_type": "summarizer",
        "execution_id": execution_id,
        
        # Contexto
        "execution_timestamp": export_timestamp,
        "analysis_date": analysis_date,
        "period_type": "weekly",  # El summarizer siempre es weekly
        "segment": segment,
        "environment": environment,
        
        # Steps del summarizer
        "summ_step1_section_connections": optimization_steps.get("step1_section_connections"),
        "summ_step2_full_report": optimization_steps.get("step2_full_report"),
        "summ_step3_executive_synthesis": optimization_steps.get("step3_executive_synthesis"),
        "summ_step4_adaptive_card": optimization_steps.get("step4_adaptive_card"),
        "summ_step5_modernize_tone": optimization_steps.get("step5_modernize_tone"),
        "summ_step6a_trim_low_impact_days": optimization_steps.get("step6a_trim_low_impact_days"),
        "summ_step6b_remove_subsegment_daily": optimization_steps.get("step6b_remove_subsegment_daily_context"),
        "summ_step6c_summarize_cabin_haul": optimization_steps.get("step6c_summarize_cabin_haul_company_weekly"),
        "summ_step6d_shorten_cabin_haul": optimization_steps.get("step6d_shorten_cabin_haul_weekly"),
        "summ_step6e_shorten_overall_global": optimization_steps.get("step6e_shorten_overall_global"),
        "summ_step7_validate_and_fix_json": optimization_steps.get("step7_validate_and_fix_json"),
        
        # Resultados finales
        "final_adaptive_card_json": final_adaptive_card_json,
        "final_executive_synthesis": final_executive_synthesis,
        "adaptive_card_size_kb": Decimal(str(round(adaptive_card_size_kb, 2))),
        "num_daily_analyses_included": len(daily_analysis_dates),
        "daily_analysis_dates": daily_analysis_dates,
        "status": "completed",
        "ttl": calculate_ttl()
    }
    
    # Eliminar valores None
    item = {k: v for k, v in item.items() if v is not None}
    
    table.put_item(Item=item)
    return report_id
```

---

## 5. Ejemplos de Consultas

### 5.1 Obtener Todos los Causales de una Ejecución

```python
def get_causals_by_execution(dynamodb_resource, execution_id: str):
    """
    Obtiene todos los reportes causales de una ejecución específica.
    Útil para el interpreter que necesita procesar todos los causales.
    """
    table = dynamodb_resource.Table(CAUSAL_TABLE)
    
    # Requiere GSI: execution_id-index
    response = table.query(
        IndexName="execution_id-index",
        KeyConditionExpression="execution_id = :eid",
        ExpressionAttributeValues={":eid": execution_id}
    )
    
    return response.get("Items", [])

# Uso
causals = get_causals_by_execution(dynamodb, "abc-123-def-456")
for causal in causals:
    print(f"  - {causal['node_path']}: {causal['status']}")
```

### 5.2 Obtener Causales por Rango de Fechas

```python
def get_causals_by_date_range(
    dynamodb_resource, 
    start_date: str, 
    end_date: str,
    node_path: str = None
):
    """
    Obtiene causales en un rango de fechas.
    Útil para análisis históricos.
    """
    table = dynamodb_resource.Table(CAUSAL_TABLE)
    
    # Requiere GSI: analysis_end_date-index
    filter_expr = "analysis_end_date BETWEEN :start AND :end"
    expr_values = {
        ":start": start_date,
        ":end": end_date
    }
    
    if node_path:
        filter_expr += " AND node_path = :np"
        expr_values[":np"] = node_path
    
    response = table.scan(
        FilterExpression=filter_expr,
        ExpressionAttributeValues=expr_values
    )
    
    return response.get("Items", [])
```

### 5.3 Obtener Reports de Synthesis por Tipo

```python
def get_synthesis_reports(
    dynamodb_resource,
    agent_type: str = None,  # "interpreter" o "summarizer"
    start_date: str = None,
    end_date: str = None
):
    """
    Obtiene reportes de synthesis filtrados por tipo y/o fecha.
    """
    table = dynamodb_resource.Table(SYNTHESIS_TABLE)
    
    if agent_type:
        # Requiere GSI: agent_type-index
        response = table.query(
            IndexName="agent_type-index",
            KeyConditionExpression="agent_type = :at",
            ExpressionAttributeValues={":at": agent_type}
        )
    else:
        response = table.scan()
    
    items = response.get("Items", [])
    
    # Filtrar por fecha si se especifica
    if start_date and end_date:
        items = [
            item for item in items
            if start_date <= item.get("analysis_date", "") <= end_date
        ]
    
    return items
```

### 5.4 Obtener Adaptive Card de una Fecha Específica

```python
def get_adaptive_card(
    dynamodb_resource,
    segment: str,
    analysis_date: str,
    agent_type: str = "summarizer"  # o "interpreter"
):
    """
    Obtiene el Adaptive Card final para una fecha y segmento específicos.
    """
    table = dynamodb_resource.Table(SYNTHESIS_TABLE)
    
    report_id = f"{agent_type}#{segment}#{analysis_date}"
    
    # Como EXPORT_TIMESTAMP es SK, necesitamos scan o conocer el timestamp exacto
    response = table.scan(
        FilterExpression="REPORT_ID = :rid",
        ExpressionAttributeValues={":rid": report_id}
    )
    
    items = response.get("Items", [])
    if items:
        # Devolver el más reciente si hay múltiples
        items.sort(key=lambda x: x.get("EXPORT_TIMESTAMP", ""), reverse=True)
        return items[0].get("final_adaptive_card_json")
    
    return None
```

### 5.5 Obtener Datos Operativos de un Causal

```python
def get_operative_data_from_causal(
    dynamodb_resource,
    node_path: str,
    analysis_date: str
):
    """
    Extrae los datos operativos de un reporte causal específico.
    Útil para análisis de correlación entre anomalías y datos operativos.
    """
    table = dynamodb_resource.Table(CAUSAL_TABLE)
    
    # Construir REPORT_ID
    report_id = f"causal_explanation#{node_path}#{analysis_date}#{analysis_date}"
    
    response = table.scan(
        FilterExpression="REPORT_ID = :rid",
        ExpressionAttributeValues={":rid": report_id}
    )
    
    items = response.get("Items", [])
    if items:
        item = items[0]  # Tomar el más reciente si hay múltiples
        operative_data = item.get("operative_data_result")
        if operative_data:
            return json.loads(operative_data)
    
    return None
```

---

## 6. Relaciones entre Tablas

### Diagrama de Relaciones

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              EJECUCIÓN DE PIPELINE                               │
│                           execution_id = "abc-123"                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
           ┌─────────────────────────────┼─────────────────────────────┐
           │                             │                             │
           ▼                             ▼                             ▼
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│ CUSTOMER_CATIA_      │     │ CUSTOMER_CATIA_      │     │ CUSTOMER_CATIA_      │
│ CAUSAL_REPORTS       │     │ SYNTHESIS_REPORT     │     │ SYNTHESIS_REPORT     │
├──────────────────────┤     ├──────────────────────┤     ├──────────────────────┤
│ REPORT_ID            │     │ REPORT_ID            │     │ REPORT_ID            │
│  causal_explanation# │     │  interpreter#Global# │     │  summarizer#Global#  │
│  Global/LH/Economy/  │     │  2026-03-24          │     │  2026-03-24          │
│  IB#2026-03-17#      │     ├──────────────────────┤     ├──────────────────────┤
│  2026-03-24          │     │ agent_type:          │     │ agent_type:          │
├──────────────────────┤     │   interpreter        │     │   summarizer         │
│ execution_id ─────────┼────▶│ execution_id ────────┼────▶│ execution_id ────┐   │
├──────────────────────┤     ├──────────────────────┤     ├──────────────────┼───┤
│ node_path:           │     │ source_causal_report │     │ source_causal_   │   │
│   Global/LH/Economy/ │◄────│   _ids: [            │     │   report_ids: [  │   │
│   IB                 │     │     "causal_exp#..." │     │     "causal_     │   │
├──────────────────────┤     │   ]                  │     │       exp#..."   │   │
│ final_synthesis      │     ├──────────────────────┤     │   ]              │   │
│   (texto explicativo)│     │ final_adaptive_card_ │     ├──────────────────┘   │
├──────────────────────┤     │   json               │     │                      │
│ tools_used: [        │     │   (JSON consumible   │     │                      │
│   "explanatory_      │     │    por Teams)        │     │                      │
│    drivers",         │     │                      │     │                      │
│   "operative_data",  │     │                      │     │                      │
│   "verbatims"        │     │                      │     │                      │
│ ]                    │     │                      │     │                      │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

### Cómo Navegar entre Tablas

```python
def trace_execution_pipeline(dynamodb_resource, execution_id: str):
    """
    Traza toda la pipeline de ejecución desde causales hasta summarizer.
    Útil para debugging y auditoría.
    """
    results = {
        "execution_id": execution_id,
        "causals": [],
        "interpreter": None,
        "summarizer": None
    }
    
    # 1. Obtener todos los causales
    causals = get_causals_by_execution(dynamodb_resource, execution_id)
    results["causals"] = [
        {
            "report_id": c["REPORT_ID"],
            "node_path": c["node_path"],
            "status": c["status"],
            "tools_used": c.get("tools_used", [])
        }
        for c in causals
    ]
    
    # 2. Obtener synthesis reports
    synthesis_table = dynamodb_resource.Table(SYNTHESIS_TABLE)
    syn_response = synthesis_table.scan(
        FilterExpression="execution_id = :eid",
        ExpressionAttributeValues={":eid": execution_id}
    )
    
    for item in syn_response.get("Items", []):
        agent_type = item.get("agent_type")
        
        # Vincular source_causal_ids con los causales obtenidos
        source_ids = item.get("source_causal_report_ids", [])
        
        report_summary = {
            "report_id": item["REPORT_ID"],
            "agent_type": agent_type,
            "analysis_date": item.get("analysis_date"),
            "status": item.get("status"),
            "source_causal_count": len(source_ids),
            "source_causal_ids": source_ids
        }
        
        if agent_type == "interpreter":
            results["interpreter"] = report_summary
        elif agent_type == "summarizer":
            results["summarizer"] = report_summary
    
    return results

# Ejemplo de uso
trace = trace_execution_pipeline(dynamodb, "abc-123-def-456")
print(json.dumps(trace, indent=2))
```

---

## 7. Consideraciones Importantes

### Límite de 400 KB por Item

DynamoDB tiene un límite de 400 KB por item. Campos como `conversation_log` o los step results pueden excederlo.

**Estrategia recomendada:**
```python
MAX_ITEM_BYTES = 350_000  # Margen de seguridad

def save_with_size_check(table, item: dict, label: str):
    """Guardar item con verificación de tamaño."""
    raw_size = len(json.dumps(item, default=str).encode("utf-8"))
    
    if raw_size > MAX_ITEM_BYTES:
        # Guardar en S3 y almacenar solo la referencia
        s3_key = upload_to_s3(item, label)
        
        # Reemplazar campos grandes con referencia S3
        trimmed_item = item.copy()
        for field in ["conversation_log", "generation_reflections", "final_adaptive_card_json"]:
            if field in trimmed_item and len(str(trimmed_item[field])) > 5000:
                trimmed_item[field] = f"S3://{BUCKET}/{s3_key}#{field}"
        
        item = trimmed_item
    
    table.put_item(Item=item)
```

### TTL (Time To Live)

Se recomienda un TTL de **90 días** (`7776000` segundos) para mantener un histórico razonable sin acumular datos indefinidamente.

```python
def calculate_ttl(days=90):
    return int(time.time()) + (days * 24 * 60 * 60)
```

### GSIs Recomendados

#### Para `CUSTOMER_CATIA_CAUSAL_REPORTS`:

| GSI | PK | SK | Caso de Uso |
|-----|----|----|-------------|
| `execution_id-index` | `execution_id` | `EXPORT_TIMESTAMP` | Todos los nodos de una ejecución |
| `analysis_end_date-index` | `analysis_end_date` | `EXPORT_TIMESTAMP` | Análisis por fecha |
| `node_path-index` | `node_path` | `EXPORT_TIMESTAMP` | Histórico de un segmento |

#### Para `CUSTOMER_CATIA_SYNTHESIS_REPORT`:

| GSI | PK | SK | Caso de Uso |
|-----|----|----|-------------|
| `execution_id-index` | `execution_id` | `EXPORT_TIMESTAMP` | Ambos agentes de un pipeline |
| `agent_type-index` | `agent_type` | `EXPORT_TIMESTAMP` | Histórico por tipo de agente |
| `analysis_end_date-index` | `analysis_end_date` | `agent_type` | Reportes por fecha |

---

## 8. Patrones de Uso Comunes

### Patrón 1: Agregar Nuevo Agente al Pipeline

Si quieres crear un nuevo agente que se integre con el sistema existente:

1. **Genera el mismo `execution_id`** que los demás agentes del pipeline
2. **Guarda en `CUSTOMER_CATIA_SYNTHESIS_REPORT`** con un nuevo `agent_type`
3. **Vincula con `source_causal_report_ids`** o `source_interpreter_ids` según tu caso
4. **Sigue el esquema de `REPORT_ID`**: `{agent_type}#{segment}#{date}`

### Patrón 2: Crear un Dashboard que Consume los Datos

```python
def get_recent_anomalies(dynamodb_resource, days=7):
    """Obtiene anomalías recientes para un dashboard."""
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Obtener causales del período
    causals = get_causals_by_date_range(
        dynamodb_resource,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )
    
    # Enriquecer con datos de synthesis
    for causal in causals:
        execution_id = causal.get("execution_id")
        if execution_id:
            # Buscar interpreter/summarizer relacionado
            synthesis = get_synthesis_reports(
                dynamodb_resource,
                execution_id=execution_id
            )
            causal["synthesis"] = synthesis
    
    return causals
```

### Patrón 3: Comparar Anomalías Históricas por Segmento

```python
def get_segment_anomaly_history(dynamodb_resource, node_path: str, months=6):
    """Obtiene historial de anomalías para un segmento específico."""
    table = dynamodb_resource.Table(CAUSAL_TABLE)
    
    # Requiere GSI: node_path-index
    response = table.query(
        IndexName="node_path-index",
        KeyConditionExpression="node_path = :np",
        ExpressionAttributeValues={":np": node_path},
        ScanIndexForward=False  # Más reciente primero
    )
    
    items = response.get("Items", [])
    
    # Agrupar por mes
    history = {}
    for item in items:
        date = item.get("analysis_end_date", "")
        if date:
            month = date[:7]  # YYYY-MM
            if month not in history:
                history[month] = []
            history[month].append({
                "date": date,
                "anomaly_type": item.get("anomaly_type"),
                "status": item.get("status"),
                "has_explanation": item.get("investigation_success", False)
            })
    
    return history
```

---

## 9. Resumen de Campos Clave

### Para extracción de Adaptive Cards (mostrar en UI):
- `CUSTOMER_CATIA_SYNTHESIS_REPORT.final_adaptive_card_json`
- `CUSTOMER_CATIA_SYNTHESIS_REPORT.agent_type` (saber si es interpreter o summarizer)
- `CUSTOMER_CATIA_SYNTHESIS_REPORT.analysis_date`

### Para análisis de causas:
- `CUSTOMER_CATIA_CAUSAL_REPORTS.final_synthesis`
- `CUSTOMER_CATIA_CAUSAL_REPORTS.tools_used`
- `CUSTOMER_CATIA_CAUSAL_REPORTS.node_path`
- `CUSTOMER_CATIA_CAUSAL_REPORTS.explanatory_drivers_result`
- `CUSTOMER_CATIA_CAUSAL_REPORTS.operative_data_result`

### para debugging:
- `CUSTOMER_CATIA_CAUSAL_REPORTS.conversation_log`
- `CUSTOMER_CATIA_SYNTHESIS_REPORT.step*_...` (pasos intermedios)
- Ambas tablas: `execution_duration_ms`, `status`, `error_message`

### Para correlación entre agentes:
- `execution_id` (vincula causales con synthesis)
- `source_causal_report_ids` (vincula synthesis con causales)
- `node_path` (agrupa causales por segmento)

---

*Documentación generada para el proyecto CATIA Proactiva - Sistema de Agentes de Análisis NPS*
