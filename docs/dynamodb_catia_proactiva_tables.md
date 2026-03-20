# DynamoDB Tables — CATIA Proactiva (Dashboard Analyzer)

Estructura de las tablas de DynamoDB para persistir el logging de los agentes de CATIA Proactiva.

---

## 1. `CUSTOMER_CATIA_CAUSAL_REPORTS`

Almacena los resultados del **agente causal** (`CausalExplanationAgent`).
Se genera **un registro por `node_path`** (segmento) por ejecución.

### Claves

| Columna | Tipo | Clave |
|---------|------|-------|
| `report_id` | `S` | **PK** — UUID único por registro |
| `execution_id` | `S` | UUID compartido entre todos los nodos de una misma ejecución del pipeline |

> `execution_id` permite correlacionar todos los nodos causales de un mismo run
> y vincularlos con la tabla `CUSTOMER_CATIA_SYNTHESIS_REPORT`.

### Dimensiones del segmento

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `node_path` | `S` | Ruta jerárquica completa: `Global/SH/Economy/IB` |
| `segment` | `S` | Segmento raíz: `Global`, `Global/LH`, `Global/SH` |
| `haul` | `S` | `LH`, `SH`, o `null` (cuando es Global) |
| `cabin` | `S` | `Economy`, `Business`, `Premium EC`, o `null` |
| `company` | `S` | `IB`, `YW`, o `null` |

### Contexto de ejecución

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `execution_timestamp` | `S` | ISO 8601 — momento de ejecución |
| `analysis_start_date` | `S` | Fecha inicio del período analizado |
| `analysis_end_date` | `S` | Fecha fin del período analizado |
| `comparison_start_date` | `S` | Fecha inicio del período de comparación (si aplica) |
| `comparison_end_date` | `S` | Fecha fin del período de comparación (si aplica) |
| `causal_filter` | `S` | `vs L7d`, `vs LM`, `vs LY`, `vs Sel. Period` |
| `anomaly_type` | `S` | Tipo de anomalía detectada (positiva/negativa) |
| `anomaly_detection_mode` | `S` | `vslast`, `vslast_dynamic`, `target`, `mean` |
| `study_mode` | `S` | `comparative` o `single` |
| `focus_touchpoint` | `S` | Touchpoint focalizado o `null` |
| `report_group` | `S` | `general`, `cabin-crew`, etc. |
| `llm_type` | `S` | Modelo LLM utilizado |
| `environment` | `S` | `prod`, `sbx`, `local` |

### Resultado por herramienta (output crudo)

Cada columna contiene el resultado JSON serializado de la herramienta correspondiente,
tal como el agente lo almacena en `self.collected_data`.

| Columna | Tipo | Origen en el código |
|---------|------|---------------------|
| `explanatory_drivers_result` | `S` | `self.collected_data['explanatory_drivers']` |
| `operative_data_result` | `S` | `self.collected_data['operative_data']` |
| `ncs_result` | `S` | `self.collected_data['ncs_data']` |
| `routes_result` | `S` | `self.collected_data['routes_data']` |
| `verbatims_result` | `S` | `self.collected_data['verbatims']` o `['verbatims_conversation']` |
| `customer_profile_result` | `S` | `self.collected_data['customer_profile']` |
| `focus_touchpoint_result` | `S` | `self.collected_data['focus_touchpoint_enrichment']` (cuando hay foco) |

### Resultado final

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `final_synthesis` | `S` | Explicación causal final generada por el LLM |
| `identified_routes` | `L` | Rutas extraídas durante la investigación |
| `survey_count` | `N` | Número de encuestas (del tool explanatory_drivers) |

### Métricas operativas

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `tools_used` | `L` | Lista de herramientas invocadas |
| `iteration_count` | `N` | Número de iteraciones del agente |
| `total_messages` | `N` | Mensajes en el conversation log |
| `investigation_success` | `BOOL` | Si se generaron explicaciones |
| `dax_queries_count` | `N` | Número de queries DAX ejecutadas |
| `execution_duration_ms` | `N` | Tiempo total de ejecución |
| `status` | `S` | `completed`, `failed`, `timeout` |
| `error_message` | `S` | Error si `status != completed` |
| `s3_report_key` | `S` | Ruta S3 del reporte completo |
| `conversation_log` | `S` | JSON string del log completo |
| `ttl` | `N` | TTL epoch para auto-eliminación |

### GSIs recomendados

| GSI | PK | SK | Caso de uso |
|-----|----|----|-------------|
| GSI1 | `execution_id` | `node_path` | Todos los nodos de una ejecución |
| GSI2 | `analysis_end_date` | `node_path` | Análisis por fecha |
| GSI3 | `node_path` | `execution_timestamp` | Histórico de un segmento |

---

## 2. `CUSTOMER_CATIA_SYNTHESIS_REPORT`

Almacena los resultados del **interpreter** (`AnomalyInterpreterAgent`) y
**summarizer** (`AnomalySummaryAgent`). Se genera **un registro por agente** por ejecución.

Las columnas específicas de un agente quedan a `null` cuando el registro
pertenece al otro agente.

### Claves

| Columna | Tipo | Clave |
|---------|------|-------|
| `synthesis_id` | `S` | **PK** — UUID único por registro |
| `agent_type` | `S` | `interpreter` o `summarizer` |

### Contexto de ejecución (común)

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `execution_id` | `S` | UUID del pipeline (mismo que en tabla causal) |
| `execution_timestamp` | `S` | ISO 8601 |
| `analysis_date` | `S` | Fecha o rango analizado |
| `period_type` | `S` | `weekly` o `daily` |
| `period_range` | `S` | Rango estándar: `2025-03-10_2025-03-17` |
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

### Campos específicos del INTERPRETER

Pasos de la conversación jerárquica (cada uno contiene el output del LLM para ese paso):

| Columna | Tipo | Step interno |
|---------|------|--------------|
| `step1_company_level_diagnosis` | `S` | `COMPANY_LEVEL_DIAGNOSIS` |
| `step2_cabin_level_diagnosis` | `S` | `CABIN_LEVEL_DIAGNOSIS` |
| `step3_radio_global_diagnosis` | `S` | `RADIO_GLOBAL_DIAGNOSIS` |
| `step4_nma_identification` | `S` | `NMA_IDENTIFICATION` |
| `step4b_evidence_extraction` | `S` | `EVIDENCE_EXTRACTION` |
| `step4c_cabin_radio_reflection` | `S` | `CABIN_RADIO_REFLECTION` (solo si hay cabin-radios) |
| `step5_executive_synthesis` | `S` | `EXECUTIVE_SYNTHESIS` |
| `step6_adaptive_card_json` | `S` | `ADAPTIVE_CARD_GENERATION` |
| `step7_tone_modernization` | `S` | `TONE_MODERNIZATION` |

Métricas del interpreter:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `hierarchy_total_nodes` | `N` | Total de nodos en la jerarquía |
| `hierarchy_generations` | `N` | Generaciones analizadas |
| `hierarchy_summary` | `S` | JSON con resumen de la jerarquía |
| `nma_list` | `L` | Lista de Nodos Máximo Afectados |
| `generation_reflections` | `S` | JSON con reflexiones por generación |

### Campos específicos del SUMMARIZER

Pasos principales de generación de contenido:

| Columna | Tipo | Step interno |
|---------|------|--------------|
| `summ_step1_section_connections` | `S` | Conexiones diario-semanal por sección |
| `summ_step2_full_report` | `S` | Reporte final integrado |
| `summ_step3_executive_synthesis` | `S` | Síntesis ejecutiva extraída |
| `summ_step4_adaptive_card` | `S` | Adaptive card inicial generado |

Pasos de optimización del adaptive card:

| Columna | Tipo | Step interno |
|---------|------|--------------|
| `summ_step5_modernize_tone` | `S` | Modernización de tono |
| `summ_step6a_trim_low_impact_days` | `S` | Recorte de días de bajo impacto |
| `summ_step6b_remove_subsegment_daily` | `S` | Eliminación de contexto diario subsegmento |
| `summ_step6c_summarize_cabin_haul` | `S` | Resumen cabin/haul/company weekly |
| `summ_step6d_shorten_cabin_haul` | `S` | Acortamiento cabin/haul weekly |
| `summ_step6e_shorten_overall_global` | `S` | Acortamiento overall global |
| `summ_step7_validate_and_fix_json` | `S` | Validación y corrección JSON final |

Métricas del summarizer:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `adaptive_card_size_kb` | `N` | Tamaño del adaptive card en KB |
| `num_daily_analyses_included` | `N` | Número de análisis diarios incluidos |
| `daily_analysis_dates` | `L` | Fechas de los días incluidos |

### Campos comunes finales

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `final_adaptive_card_json` | `S` | Adaptive card final (resultado de todo el proceso) |
| `final_executive_synthesis` | `S` | Texto de síntesis ejecutiva final |
| `llm_calls_count` | `N` | Número de llamadas al LLM |
| `execution_duration_ms` | `N` | Tiempo total de ejecución |
| `conversation_log` | `S` | JSON string del log completo |
| `status` | `S` | `completed`, `failed`, `partial` |
| `error_message` | `S` | Error si aplica |
| `s3_report_key` | `S` | Ruta S3 del reporte |
| `source_causal_report_ids` | `L` | IDs de reportes causales fuente (vincula con tabla 1) |
| `ttl` | `N` | TTL epoch para auto-eliminación |

### GSIs recomendados

| GSI | PK | SK | Caso de uso |
|-----|----|----|-------------|
| GSI1 | `execution_id` | `agent_type` | Ambos agentes de un pipeline |
| GSI2 | `analysis_end_date` | `agent_type` | Reportes por fecha |
| GSI3 | `agent_type` | `execution_timestamp` | Histórico por tipo de agente |

---

## Notas de implementación

### Límite de 400 KB por item

DynamoDB tiene un límite de 400 KB por item. Campos como `conversation_log` o
los step results pueden excederlo. Estrategia:

- Si el item serializado supera 350 KB, guardar el body completo en S3
  y almacenar solo el `s3_key` en la columna correspondiente.
- Campos candidatos: `conversation_log`, `final_adaptive_card_json`,
  `generation_reflections`.

### TTL

Se recomienda un TTL de 90 días (`7776000` segundos) para mantener un histórico
razonable sin acumular datos indefinidamente.

### Vinculación entre tablas

El campo `execution_id` actúa como clave de correlación entre ambas tablas:

```
Pipeline Execution (execution_id = "abc-123")
├── CUSTOMER_CATIA_CAUSAL_REPORTS
│   ├── report_id: "r1" | node_path: "Global"
│   ├── report_id: "r2" | node_path: "Global/LH"
│   ├── report_id: "r3" | node_path: "Global/SH"
│   ├── report_id: "r4" | node_path: "Global/SH/Economy/IB"
│   └── ...
└── CUSTOMER_CATIA_SYNTHESIS_REPORT
    ├── synthesis_id: "s1" | agent_type: "interpreter"
    └── synthesis_id: "s2" | agent_type: "summarizer"
```
