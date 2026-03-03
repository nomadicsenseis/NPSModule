# Plan de Implementación: focus-touchpoint

## Visión General

Implementación incremental del parámetro `--focus-touchpoint` siguiendo la cadena de propagación de abajo hacia arriba: primero la capa de datos (query DAX + PBIDataCollector), luego el agente causal, luego el intérprete, y finalmente los puntos de entrada CLI.

## Tareas

- [x] 1. Implementar la query DAX y `collect_focus_touchpoint_csat_vs_target` en `PBIDataCollector`
  - Añadir el método `collect_focus_touchpoint_csat_vs_target(node_path, start_date, end_date, touchpoint_name, comparison_filter, comparison_start_date, comparison_end_date)` en `dashboard_analyzer/data_collection/pbi_collector.py`
  - El método construye la query DAX basada en `Exp. Drivers.txt` eliminando `__DS0FilterTable5` y añadiendo filtro por `filtered_name`
  - En modo `single` (comparison_filter=None), omitir también `__DS0FilterTable7`
  - Devolver dict con keys `csat`, `target`, `gap`, `satisfaction_diff`, `shapdiff` o `None` si no hay datos
  - Manejar excepciones internamente y devolver `None` en caso de error
  - _Requisitos: 2.5, 7.1, 7.2, 7.3, 7.4_

  - [ ]* 1.1 Escribir tests unitarios para `collect_focus_touchpoint_csat_vs_target`
    - Verificar que la query generada para modo `comparative` incluye `__DS0FilterTable7`
    - Verificar que la query generada para modo `single` no incluye `__DS0FilterTable7`
    - Verificar que la query no incluye `__DS0FilterTable5` (filtro `explanatory_drivers`)
    - _Requisitos: 7.1, 7.3_

  - [ ]* 1.2 Escribir test unitario: cálculo del gap vs target
    - Verificar que `gap = csat - target` para valores positivos, negativos y cero
    - _Requisitos: 2.2_

- [x] 2. Añadir `focus_touchpoint` a `CausalExplanationAgent`
  - Añadir `focus_touchpoint: Optional[str] = None` en `__init__` de `CausalExplanationAgent` (`dashboard_analyzer/anomaly_explanation/genai_core/agents/causal_explanation_agent.py`)
  - Añadir `focus_touchpoint: Optional[str] = None` en `investigate_anomaly`
  - Implementar método privado `_collect_focus_touchpoint_data(node_path, start_dt, end_dt)` que llama a `self.pbi_collector.collect_focus_touchpoint_csat_vs_target(...)` con los parámetros del agente
  - _Requisitos: 4.3_

  - [ ]* 2.1 Escribir test de ejemplo: firma del CausalExplanationAgent
    - Verificar que `CausalExplanationAgent.__init__` acepta `focus_touchpoint`
    - Verificar que `investigate_anomaly` acepta `focus_touchpoint`
    - _Requisitos: 4.3_

- [x] 3. Implementar lógica de forzado en `_explanatory_drivers_tool`
  - En `_explanatory_drivers_tool`, después de obtener el DataFrame de drivers normales, añadir bloque condicional `if self.focus_touchpoint:`
  - Si el touchpoint aparece en el DataFrame: renombrar la fila añadiendo prefijo `🎯 FOCUS: ` al `filtered_name`
  - Si el touchpoint NO aparece: llamar a `_collect_focus_touchpoint_data(...)` y añadir fila al DataFrame con prefijo `🎯 FOCUS: `
  - Si la query adicional falla o devuelve vacío: añadir fila `🎯 FOCUS: {touchpoint} (sin datos)`
  - En modo `single`: inyectar el touchpoint directamente como touchpoint de interés especial en el contexto de investigación
  - _Requisitos: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2_

  - [ ]* 3.1 Escribir tests unitarios: lógica de forzado en `_explanatory_drivers_tool`
    - Verificar que si el touchpoint ya está en el DataFrame, se marca con `🎯 FOCUS: `
    - Verificar que si el touchpoint no está, se añade una fila con `🎯 FOCUS: `
    - Verificar que si la query adicional falla, se añade fila `🎯 FOCUS: {touchpoint} (sin datos)`
    - _Requisitos: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ]* 3.2 Escribir test unitario: robustez ante fallos de query
    - Verificar que si `_collect_focus_touchpoint_data` lanza excepción, el DataFrame resultante contiene la fila `(sin datos)` y no se propaga la excepción
    - _Requisitos: 2.4, 3.5_

- [ ] 4. Añadir `focus_touchpoint` a `FlexibleAnomalyInterpreter`
  - Añadir `focus_touchpoint: Optional[str] = None` en `__init__` de `FlexibleAnomalyInterpreter` (`dashboard_analyzer/anomaly_detection/flexible_anomaly_interpreter.py`)
  - Añadir `focus_touchpoint: Optional[str] = None` en `explain_anomaly`
  - Al inicializar/reinicializar el `CausalExplanationAgent` dentro de `_initialize_causal_agent`, pasar `focus_touchpoint=self.focus_touchpoint`
  - Al llamar a `investigate_anomaly`, pasar `focus_touchpoint=focus_touchpoint or self.focus_touchpoint`
  - _Requisitos: 3.4, 4.4_

  - [ ]* 4.1 Escribir test unitario: propagación del focus_touchpoint
    - Mockear `CausalExplanationAgent` y verificar que se instancia con el `focus_touchpoint` correcto cuando se inicializa `FlexibleAnomalyInterpreter` con ese valor
    - _Requisitos: 1.3, 4.4_

- [ ] 5. Checkpoint — Verificar que los tests pasan
  - Asegurarse de que todos los tests pasan, preguntar al usuario si surgen dudas.

- [ ] 6. Implementar lógica en `process_node_anomaly` para el `nps_context`
  - En la función interna `process_node_anomaly` dentro de `process_single_period` (`dashboard_analyzer/deep_research_period.py`), añadir bloque condicional `if focus_touchpoint:` después de construir el `nps_context` normal
  - Llamar a `pbi_collector.collect_focus_touchpoint_csat_vs_target(...)` con los parámetros del nodo
  - Si devuelve datos: construir la línea `🎯 FOCUS TOUCHPOINT '{touchpoint}': CSAT={csat:.1f}, vs Target={gap:+.1f}pts` y añadirla al `nps_context`
  - Envolver en try/except para no interrumpir el análisis si falla
  - Pasar `focus_touchpoint` al constructor de `FlexibleAnomalyInterpreter` dentro de `process_node_anomaly`
  - _Requisitos: 2.1, 2.3, 2.4_

  - [ ]* 6.1 Escribir test unitario: formato de la línea en nps_context
    - Verificar que dado `{csat: 72.3, target: 76.5, gap: -4.2}`, la línea generada es `"🎯 FOCUS TOUCHPOINT 'ifl_100_cabin_crew': CSAT=72.3, vs Target=-4.2pts"`
    - Verificar que un gap positivo se formatea con `+`
    - _Requisitos: 2.3_

- [ ] 7. Propagar `focus_touchpoint` por las funciones de orquestación en `deep_research_period.py`
  - Añadir `focus_touchpoint: Optional[str] = None` en `process_single_period`
  - Añadir `focus_touchpoint: Optional[str] = None` en `show_all_anomaly_periods_with_explanations`
  - Añadir `focus_touchpoint: Optional[str] = None` en `execute_analysis_flow`
  - Propagar el parámetro en cada llamada interna: `execute_analysis_flow` → `show_all_anomaly_periods_with_explanations` → `process_single_period` → `process_node_anomaly`
  - _Requisitos: 1.3_

- [ ] 8. Añadir argumento CLI `--focus-touchpoint` en `deep_research_period.py`
  - En `main()`, añadir `parser.add_argument('--focus-touchpoint', type=str, default=None, help='...')`
  - Normalizar el valor: si es string vacío o solo espacios, convertir a `None`
  - Pasar `focus_touchpoint=args.focus_touchpoint` a `execute_analysis_flow`
  - _Requisitos: 1.1, 1.5_

  - [ ]* 8.1 Escribir test de ejemplo: parsing del argumento CLI
    - Verificar que `--focus-touchpoint ifl_100_cabin_crew` se parsea correctamente
    - Verificar que sin `--focus-touchpoint` el valor es `None`
    - _Requisitos: 1.1, 1.2_

  - [ ]* 8.2 Escribir test unitario: idempotencia sin focus_touchpoint
    - Verificar que llamar con `focus_touchpoint=None` produce el mismo resultado que llamar sin el parámetro (usando mocks)
    - _Requisitos: 1.2_

- [ ] 9. Propagar `focus_touchpoint` en `weekly_deep_research.py`
  - Añadir `focus_touchpoint: Optional[str] = None` en `run_weekly_comprehensive_analysis`
  - Propagar a ambas llamadas a `execute_analysis_flow` (weekly y daily)
  - Añadir `parser.add_argument('--focus-touchpoint', ...)` en `main()` de `weekly_deep_research.py`
  - Normalizar el valor (vacío → None) igual que en `deep_research_period.py`
  - _Requisitos: 1.4_

- [ ] 10. Actualizar prompts YAML del `AnomalyInterpreterAgent`
  - En `dashboard_analyzer/anomaly_explanation/config/prompts/anomaly_interpreter.yaml`, añadir instrucciones al `system_prompt` de `comparative_prompts` y `single_prompts`
  - Las instrucciones deben indicar: si aparece `🎯 FOCUS TOUCHPOINT` en los datos de entrada, mencionar ese touchpoint en la síntesis con sus métricas (CSAT, gap vs target, variación vs comparación si disponible), aunque su SHAP sea bajo o neutro
  - _Requisitos: 5.1, 5.2, 5.3_

  - [ ]* 10.1 Escribir test de ejemplo: instrucciones en el YAML del intérprete
    - Verificar que el YAML cargado contiene las instrucciones de focus_touchpoint en ambos modos
    - _Requisitos: 5.2_

- [ ] 11. Actualizar prompts YAML del `AnomalySummaryAgent`
  - En `dashboard_analyzer/anomaly_explanation/config/prompts/anomaly_summary.yaml`, añadir instrucciones en `step3_extract_synthesis` y `step4_generate_adaptive_card`
  - Las instrucciones deben indicar: si hay información de `🎯 FOCUS TOUCHPOINT` en los datos, incluir sección dedicada con CSAT actual, gap vs target, variación vs comparación y resumen de hallazgos
  - Si no hay `focus_touchpoint`, generar el resumen normal sin cambios
  - _Requisitos: 6.1, 6.2, 6.3, 6.4_

  - [ ]* 11.1 Escribir test de ejemplo: instrucciones en el YAML del summary
    - Verificar que el YAML cargado contiene las instrucciones de focus_touchpoint en los pasos relevantes
    - _Requisitos: 6.3_

- [ ] 12. Checkpoint final — Verificar integración completa
  - Asegurarse de que todos los tests pasan, preguntar al usuario si surgen dudas.
  - Verificar que `python -m dashboard_analyzer.deep_research_period --help` muestra `--focus-touchpoint`
  - Verificar que `python -m dashboard_analyzer.weekly_deep_research --help` muestra `--focus-touchpoint`

- [ ] 13. Implementar mapeo `filtered_name → display_name` y `collect_focus_touchpoint_issues_pct` en `PBIDataCollector`
  - Añadir el diccionario `TOUCHPOINT_DISPLAY_NAME_MAP` en `dashboard_analyzer/data_collection/pbi_collector.py` con al menos la entrada `{"ifl_100_cabin_crew_satisfaction": "Cabin Crew"}`, extensible sin cambios de código
  - Añadir la función `get_touchpoint_display_name(filtered_name: str) -> Optional[str]` que consulta el diccionario
  - Añadir el método `collect_focus_touchpoint_issues_pct(node_path, start_date, end_date, comparison_start_date, comparison_end_date, touchpoint_display_name)` en `pbi_collector.py`
  - La query DAX filtra por `Issue_touchpoint_Dict[issue_type_3] = "{touchpoint_display_name}"` y `explanatory_drivers = 1`, parametrizada por fechas y segmento
  - La query devuelve dos filas: `L7D` (periodo actual) y `L7D_prev` (periodo comparativo) con `Pct_Issues`
  - El método devuelve dict con keys `pct_issues_current`, `pct_issues_prev`, `diff` o `None` si falla
  - Manejar excepciones internamente y devolver `None` en caso de error, registrando aviso en log
  - En `CausalExplanationAgent.investigate_anomaly`, añadir step que: (1) resuelve `display_name` via `get_touchpoint_display_name`, (2) si existe, llama a `collect_focus_touchpoint_issues_pct` para cada nodo relevante según segmento CLI, (3) si no existe, registra aviso y omite la query
  - Implementar función auxiliar `_get_relevant_nodes_for_segment(segment)` que devuelve `[Global, LH, SH]` para Global, `[SH]` para SH, `[LH]` para LH
  - Almacenar resultados en `self._focus_issues_pct_by_node` para su uso posterior en la síntesis
  - Ejecutar también para nodos donde el focus touchpoint tenga SHAP significativo según exp drivers
  - _Requisitos: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.9, 11.1, 11.2, 11.3, 11.4_

  - [ ]* 13.1 Escribir tests unitarios para `collect_focus_touchpoint_issues_pct` y el mapeo
    - Verificar que `get_touchpoint_display_name("ifl_100_cabin_crew_satisfaction")` devuelve `"Cabin Crew"`
    - Verificar que `get_touchpoint_display_name("touchpoint_sin_mapeo")` devuelve `None`
    - Verificar que la query generada incluye el filtro `issue_type_3 = "{display_name}"`
    - Verificar que la query incluye `explanatory_drivers = 1`
    - Verificar que la query se parametriza correctamente por fechas y segmento
    - Verificar que devuelve `None` ante fallo sin propagar excepción
    - _Requisitos: 10.1, 10.3, 10.4, 10.9, 11.1, 11.2_

  - [ ]* 13.2 Escribir test de propiedad: cálculo del diff de % issues
    - **Propiedad 9: Cálculo correcto del diff de % issues**
    - **Validates: Requisito 10.2**
    - Para cualquier par `(pct_issues_current, pct_issues_prev)`, `diff == pct_issues_current - pct_issues_prev`
    - _Requisitos: 10.2_

  - [ ]* 13.3 Escribir test de propiedad: robustez ante fallo de `collect_focus_touchpoint_issues_pct`
    - **Propiedad 10: Robustez ante fallo de collect_focus_touchpoint_issues_pct**
    - **Validates: Requisito 10.9**
    - Para cualquier excepción lanzada por `collect_focus_touchpoint_issues_pct`, el análisis no se interrumpe
    - _Requisitos: 10.9_

  - [ ]* 13.4 Escribir test de propiedad: cobertura de segmentos
    - **Propiedad 7: Cobertura de segmentos para verbatims y % issues del focus touchpoint**
    - **Validates: Requisitos 8.3, 8.4, 8.5, 10.6, 10.10**
    - Para cualquier segmento CLI (Global, SH, LH), los nodos cubiertos son exactamente los esperados
    - _Requisitos: 10.6, 10.10_

  - [ ]* 13.5 Escribir test de propiedad: consistencia del mapeo filtered_name → display_name
    - **Propiedad 11: Mapeo filtered_name → display_name es consistente**
    - **Validates: Requisito 11.2**
    - Para cualquier filtered_name en el mapa, get_touchpoint_display_name devuelve siempre el mismo display_name no vacío
    - _Requisitos: 11.2_

- [ ] 14. Implementar verbatims del focus touchpoint por segmento en `CausalExplanationAgent`
  - En `CausalExplanationAgent.investigate_anomaly`, cuando `focus_touchpoint` está activo, añadir step que llama a `_verbatims_tool` para cada nodo relevante según segmento CLI
  - Resolver el `display_name` via `get_touchpoint_display_name(effective_focus)` y usarlo como query de búsqueda; si no existe mapeo, usar el `filtered_name` directamente
  - La búsqueda de verbatims debe ejecutarse independientemente del valor SHAP del focus touchpoint
  - Usar `_get_relevant_nodes_for_segment(segment)` (implementada en tarea 13) para determinar los nodos a cubrir
  - Almacenar resultados en `self._focus_verbatims_by_node` (dict nodo → verbatims) para su uso en la síntesis
  - Incluir en el contexto pasado al `AnomalySummaryAgent`: resumen de verbatims del focus touchpoint, top positivos y top negativos por segmento
  - _Requisitos: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [ ]* 14.1 Escribir test unitario: verbatims del focus touchpoint se ejecutan siempre
    - Mockear `_verbatims_tool` y verificar que se llama para cada nodo relevante cuando `focus_touchpoint` está activo, independientemente del SHAP
    - Verificar que la query usa el `display_name` si existe mapeo, o el `filtered_name` si no
    - _Requisitos: 8.1, 8.2_

  - [ ]* 14.2 Escribir test unitario: ausencia de verbatims se indica en síntesis
    - Verificar que si `_focus_verbatims_by_node` está vacío para un segmento, el contexto pasado al `AnomalySummaryAgent` indica explícitamente la ausencia
    - _Requisitos: 8.7_

- [ ] 15. Actualizar prompts YAML del `AnomalySummaryAgent` para las tres nuevas secciones
  - En `dashboard_analyzer/anomaly_explanation/config/prompts/anomaly_summary.yaml`, actualizar instrucciones en `step3_extract_synthesis` y `step4_generate_adaptive_card` para incluir las tres nuevas secciones cuando `focus_touchpoint` está activo:
    1. **CSAT por segmento**: CSAT actual, satisfaction diff vs comparativo y gap vs target para cada segmento relevante (Global, LH, SH según aplique)
    2. **Verbatims del focus touchpoint por segmento**: resumen general, top verbatims positivos y top verbatims negativos para cada segmento relevante
    3. **% Issues del focus touchpoint por segmento**: `Pct_Issues` del periodo actual y variación vs comparativo para cada segmento relevante (solo si existe `display_name` para el touchpoint)
  - Las instrucciones deben indicar que si no hay datos para un segmento, se indique explícitamente
  - En modo `single`, omitir la comparación vs periodo comparativo en la sección de CSAT
  - _Requisitos: 8.6, 8.7, 9.1, 9.2, 9.3, 9.5, 10.8_

  - [ ]* 15.1 Escribir test de ejemplo: instrucciones de las tres nuevas secciones en el YAML del summary
    - Verificar que el YAML cargado contiene instrucciones para las secciones de CSAT por segmento, verbatims del focus touchpoint y % issues
    - _Requisitos: 9.1, 8.6, 10.8_

- [ ] 16. Checkpoint final — Verificar integración de las nuevas funcionalidades
  - Asegurarse de que todos los tests pasan, preguntar al usuario si surgen dudas.

## Notas

- Las tareas marcadas con `*` son opcionales y pueden omitirse para un MVP más rápido
- Cada tarea referencia requisitos específicos para trazabilidad
- El parámetro `focus_touchpoint` es siempre `Optional[str]` con valor por defecto `None` para no romper interfaces existentes
- La propagación sigue el orden: CLI → `execute_analysis_flow` → `show_all_anomaly_periods_with_explanations` → `process_single_period` → `process_node_anomaly` → `FlexibleAnomalyInterpreter` → `CausalExplanationAgent`
- Las tareas 13-15 implementan las nuevas funcionalidades: mapeo de nombres + % issues del focus touchpoint (DAX), verbatims del focus touchpoint obligatorios y las tres nuevas secciones en la síntesis final
- La función auxiliar `_get_relevant_nodes_for_segment` (tarea 13) es compartida por las tareas 13 y 14
- La cobertura de segmentos (Global/LH/SH) sigue la misma lógica en todas las nuevas funcionalidades
- El mapeo `filtered_name → display_name` (tarea 13) es necesario porque los nombres en `TouchPoint_Master` (p.ej. `ifl_100_cabin_crew_satisfaction`) difieren de los nombres en `Issue_touchpoint_Dict` (p.ej. `Cabin Crew`)
- Si un touchpoint no tiene mapeo, los verbatims usan el `filtered_name` como query y la query de % issues se omite con aviso en log
