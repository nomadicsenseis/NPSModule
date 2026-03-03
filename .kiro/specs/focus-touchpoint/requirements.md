# Documento de Requisitos: focus-touchpoint

## Introducción

Esta feature añade un parámetro opcional `--focus-touchpoint` al sistema de análisis de NPS de Iberia. Cuando se especifica, el sistema investiga ese touchpoint en profundidad —con sus métricas de CSAT actual, gap vs target y comparación vs periodo anterior— independientemente de si aparece como driver explicativo importante en los datos normales. El objetivo es permitir al analista forzar la investigación de un touchpoint concreto (p.ej. `ifl_100_cabin_crew`) en cualquier análisis, tanto en modo `comparative` como `single`.

## Glosario

- **Touchpoint**: Punto de contacto del cliente con la aerolínea (p.ej. `ifl_100_cabin_crew_satisfaction`, `ifl_200_food`). Identificado por su `filtered_name` en `TouchPoint_Master`.
- **Focus Touchpoint**: El touchpoint especificado mediante `--focus-touchpoint` que se investiga obligatoriamente. Se especifica como `filtered_name` (p.ej. `ifl_100_cabin_crew_satisfaction`).
- **filtered_name**: Nombre interno del touchpoint en `TouchPoint_Master` y en los resultados de Exp. Drivers (p.ej. `ifl_100_cabin_crew_satisfaction`). Es el identificador que se usa en el parámetro CLI.
- **display_name**: Nombre "libre" del touchpoint tal como aparece en `Issue_touchpoint_Dict` y en queries DAX de issues (p.ej. `Cabin Crew`). Puede diferir del `filtered_name`.
- **Mapeo filtered_name → display_name**: Traducción necesaria para usar el touchpoint en queries de issues. Ejemplo: `ifl_100_cabin_crew_satisfaction` → `Cabin Crew`. El sistema mantiene un diccionario de mapeo para los touchpoints conocidos; si no existe mapeo, se usa el `filtered_name` directamente.
- **CSAT**: Customer Satisfaction Score del touchpoint (campo `Satisfaction` en la query de Exp. Drivers).
- **Target Satisfaction**: Objetivo de satisfacción del touchpoint (`[Target_Satisfaction_filtered]`).
- **Gap vs Target**: Diferencia entre CSAT actual y Target Satisfaction del touchpoint.
- **Satisfaction diff**: Diferencia de CSAT del touchpoint vs el periodo de comparación (campo `Satisfaction diff` en Exp. Drivers).
- **SHAP / Shapdiff**: Valor SHAP del touchpoint, que indica su impacto en el NPS (campo `Shapdiff` en Exp. Drivers).
- **Exp. Drivers**: Query DAX `Exp. Drivers.txt` que devuelve touchpoints con `explanatory_drivers = 1` y sus métricas.
- **explanatory_drivers**: Flag en `TouchPoint_Master` que indica si un touchpoint es driver explicativo (valor 1).
- **nps_context**: Cadena de texto con el contexto NPS del nodo que se pasa al `CausalExplanationAgent`.
- **Modo comparative**: Análisis comparativo 7d vs periodo anterior, con `causal_filter` activo (p.ej. `"vs L7d"`).
- **Modo single**: Análisis de días individuales sin comparación, `causal_filter = None`.
- **CausalExplanationAgent**: Agente que investiga la causa de una anomalía usando tools (drivers, rutas, verbatims, NCS).
- **FlexibleAnomalyInterpreter**: Clase que orquesta la explicación de anomalías por nodo/periodo.
- **AnomalyInterpreterAgent**: Agente que genera la interpretación ejecutiva por periodo.
- **AnomalySummaryAgent**: Agente que genera el resumen ejecutivo final consolidado.
- **PBIDataCollector**: Clase que ejecuta queries DAX contra Power BI para recoger datos.
- **System**: El sistema de análisis de NPS de Iberia (`dashboard_analyzer`).

## Requisitos

### Requisito 1: Parámetro CLI `--focus-touchpoint`

**User Story:** Como analista de NPS, quiero especificar un touchpoint de foco desde la línea de comandos, para que el sistema lo investigue en profundidad en cualquier análisis.

#### Criterios de Aceptación

1. THE System SHALL aceptar un parámetro CLI opcional `--focus-touchpoint` con el `filtered_name` del touchpoint (p.ej. `ifl_100_cabin_crew`).
2. WHEN `--focus-touchpoint` no se especifica, THE System SHALL ejecutar el análisis normal sin ningún cambio de comportamiento.
3. THE System SHALL propagar el valor de `focus_touchpoint` desde `main()` hasta `execute_analysis_flow()`, `show_all_anomaly_periods_with_explanations()`, `process_single_period()`, `FlexibleAnomalyInterpreter` y `CausalExplanationAgent`.
4. THE System SHALL propagar el valor de `focus_touchpoint` desde `run_weekly_comprehensive_analysis()` hasta las llamadas a `execute_analysis_flow()` en `weekly_deep_research.py`.
5. IF `--focus-touchpoint` se especifica con un valor vacío o solo espacios, THEN THE System SHALL ignorarlo y ejecutar el análisis normal.

---

### Requisito 2: Contexto CSAT vs Target en el nps_context

**User Story:** Como analista, quiero que el agente arranque con el CSAT actual del touchpoint de foco y su gap vs target, para que tenga ese contexto desde el inicio de la investigación.

#### Criterios de Aceptación

1. WHEN `focus_touchpoint` está activo y se procesa un nodo en `process_node_anomaly`, THE System SHALL ejecutar una query adicional para obtener el CSAT actual y el `Target_Satisfaction_filtered` del touchpoint.
2. THE System SHALL calcular el gap vs target como `CSAT - Target_Satisfaction_filtered`.
3. THE System SHALL añadir al `nps_context` una línea con el formato: `"🎯 FOCUS TOUCHPOINT '{touchpoint}': CSAT={csat:.1f}, vs Target={gap:+.1f}pts"`.
4. IF la query adicional falla o no devuelve datos para el touchpoint, THEN THE System SHALL continuar el análisis sin añadir la línea al `nps_context` y registrar un aviso en el log.
5. THE PBIDataCollector SHALL exponer una función `collect_focus_touchpoint_csat_vs_target(node_path, start_date, end_date, touchpoint_name)` que ejecute la query de CSAT vs target.

---

### Requisito 3: Forzado del touchpoint en `_explanatory_drivers_tool` (modo comparative)

**User Story:** Como analista, quiero que en modo comparative el touchpoint de foco aparezca siempre en los drivers con sus datos de CSAT diff y SHAP, aunque no sea un driver explicativo marcado.

#### Criterios de Aceptación

1. WHEN `focus_touchpoint` está activo y el modo es `comparative`, THE `_explanatory_drivers_tool` SHALL verificar si el touchpoint aparece en los resultados normales de Exp. Drivers.
2. IF el touchpoint aparece en los resultados normales, THEN THE `_explanatory_drivers_tool` SHALL marcarlo con el prefijo `🎯 FOCUS` en la presentación al agente.
3. IF el touchpoint NO aparece en los resultados normales (filtrado por `explanatory_drivers = 1`), THEN THE `_explanatory_drivers_tool` SHALL ejecutar una query adicional sin el filtro `__DS0FilterTable5`, filtrando solo por `filtered_name = '{touchpoint}'`, para obtener su `Satisfaction diff`, `Satisfaction` y `Shapdiff`.
4. WHEN se obtienen los datos del touchpoint por query adicional, THE `_explanatory_drivers_tool` SHALL añadirlo a los resultados con el prefijo `🎯 FOCUS` y sus métricas (`Satisfaction diff`, `Shapdiff`).
5. IF la query adicional del touchpoint falla o no devuelve datos, THEN THE `_explanatory_drivers_tool` SHALL añadir una entrada indicando que el touchpoint no tiene datos disponibles para el periodo, con el prefijo `🎯 FOCUS`.

---

### Requisito 4: Investigación obligatoria del touchpoint con routes, verbatims y NCS

**User Story:** Como analista, quiero que el agente investigue el touchpoint de foco con todas las herramientas disponibles, independientemente de su SHAP.

#### Criterios de Aceptación

1. WHEN `focus_touchpoint` está activo, THE `CausalExplanationAgent` SHALL incluir el touchpoint en la lista de touchpoints a investigar con `_routes_tool`, `_verbatims_tool` y `_ncs_tool`, independientemente de su valor SHAP.
2. WHEN `focus_touchpoint` está activo en modo `single`, THE `CausalExplanationAgent` SHALL inyectar el touchpoint directamente como touchpoint de interés especial para la investigación con `_routes_tool`, `_verbatims_tool` y `_ncs_tool`.
3. THE `CausalExplanationAgent` SHALL recibir `focus_touchpoint` como parámetro en `__init__` y en `investigate_anomaly`.
4. THE `FlexibleAnomalyInterpreter` SHALL recibir `focus_touchpoint` como parámetro en `__init__` y en `explain_anomaly`, y propagarlo al `CausalExplanationAgent`.

---

### Requisito 5: Enriquecimiento del prompt del `AnomalyInterpreterAgent`

**User Story:** Como analista, quiero que la interpretación por periodo mencione explícitamente el touchpoint de foco y sus datos, para que el informe refleje el análisis solicitado.

#### Criterios de Aceptación

1. WHEN `focus_touchpoint` está activo, THE `AnomalyInterpreterAgent` SHALL recibir en su input los datos del touchpoint de foco (CSAT actual, gap vs target, CSAT diff vs comparación, SHAP).
2. THE prompt del `AnomalyInterpreterAgent` SHALL incluir instrucciones para mencionar el touchpoint de foco y sus métricas en la síntesis ejecutiva del periodo.
3. THE prompt del `AnomalyInterpreterAgent` SHALL indicar que el touchpoint de foco debe analizarse aunque su SHAP sea bajo o neutro.

---

### Requisito 6: Sección dedicada en el resumen del `AnomalySummaryAgent`

**User Story:** Como analista, quiero que el resumen ejecutivo final incluya una sección dedicada al touchpoint de foco con sus métricas consolidadas, para tener una visión rápida de su estado.

#### Criterios de Aceptación

1. WHEN `focus_touchpoint` está activo, THE `AnomalySummaryAgent` SHALL incluir en el resumen ejecutivo final una sección dedicada al touchpoint de foco.
2. THE sección dedicada SHALL incluir: CSAT actual, gap vs target, CSAT diff vs periodo de comparación (en modo comparative) y un resumen de los hallazgos de rutas/verbatims/NCS relacionados.
3. THE prompt del `AnomalySummaryAgent` SHALL incluir instrucciones para generar esta sección cuando `focus_touchpoint` está presente en los datos de entrada.
4. IF `focus_touchpoint` no está activo, THEN THE `AnomalySummaryAgent` SHALL generar el resumen normal sin sección adicional.

---

### Requisito 7: Query DAX para CSAT vs Target del touchpoint

**User Story:** Como desarrollador, quiero una query DAX reutilizable que obtenga el CSAT y el target de un touchpoint específico sin el filtro de `explanatory_drivers`, para poder calcular el gap vs target.

#### Criterios de Aceptación

1. THE System SHALL implementar una query DAX basada en `Exp. Drivers.txt` que elimine el filtro `__DS0FilterTable5` (`explanatory_drivers = 1`) y añada un filtro por `filtered_name = '{touchpoint}'`.
2. THE query SHALL devolver los campos: `Satisfaction` (CSAT actual), `Target_Satisfaction_filtered` (target), `Satisfaction diff` (vs periodo comparación) y `Shapdiff` (SHAP).
3. WHEN el modo es `single` (sin `causal_filter`), THE query SHALL omitir el filtro `__DS0FilterTable7` (filtro de comparativa) para evitar resultados vacíos.
4. THE query SHALL ser parametrizable con: `node_path` (para los filtros de cabina/haul/compañía), `start_date`, `end_date` y `touchpoint_name`.

---

### Requisito 8: Verbatims del focus touchpoint en la síntesis final (por segmento)

**User Story:** Como analista de NPS, quiero que la síntesis final incluya siempre los verbatims relacionados con el focus touchpoint para cada segmento relevante, para poder evaluar la percepción del cliente sobre ese touchpoint independientemente de su impacto en el NPS.

#### Criterios de Aceptación

1. WHEN `focus_touchpoint` está activo, THE `CausalExplanationAgent` SHALL ejecutar `_verbatims_tool` buscando verbatims relacionados con el focus touchpoint para cada nodo relevante (Global, LH y/o SH según el segmento definido por CLI), independientemente del valor SHAP del focus touchpoint.
2. THE query de verbatims SHALL usar el `display_name` del touchpoint (obtenido del mapeo `filtered_name → display_name`) como término de búsqueda. Si no existe mapeo, usar el `filtered_name` directamente.
3. WHEN el segmento es Global, THE `CausalExplanationAgent` SHALL recoger verbatims del focus touchpoint para los nodos Global, LH y SH (y sus sub-segmentos donde aplique).
4. WHEN el segmento es solo SH, THE `CausalExplanationAgent` SHALL recoger verbatims únicamente para los nodos SH (no LH ni Global).
5. WHEN el segmento es solo LH, THE `CausalExplanationAgent` SHALL recoger verbatims únicamente para los nodos LH (no SH ni Global).
6. THE `AnomalySummaryAgent` SHALL incluir en la síntesis final, para cada segmento relevante, un resumen de verbatims del focus touchpoint con: resumen general, top verbatims de sentimiento positivo y top verbatims de sentimiento negativo.
7. IF no se encuentran verbatims del focus touchpoint para un segmento, THEN THE `AnomalySummaryAgent` SHALL indicarlo explícitamente en la sección correspondiente de la síntesis.

---

### Requisito 9: CSAT del focus touchpoint en la síntesis final por segmento

**User Story:** Como analista de NPS, quiero que la síntesis final muestre el CSAT del focus touchpoint desglosado por segmento (Global, LH, SH), para tener una visión consolidada de su estado en cada nivel de análisis.

#### Criterios de Aceptación

1. WHEN `focus_touchpoint` está activo, THE `AnomalySummaryAgent` SHALL incluir en la síntesis final, para cada segmento relevante (Global, LH y/o SH según el segmento CLI), el CSAT actual del focus touchpoint.
2. THE síntesis final SHALL incluir, para cada segmento relevante, la comparación del CSAT vs el periodo comparativo (`satisfaction diff`).
3. THE síntesis final SHALL incluir, para cada segmento relevante, el gap del CSAT vs el target (`gap vs target`).
4. THE System SHALL obtener estos datos del `nps_context` ya calculado por nodo en `process_node_anomaly`, sin necesidad de queries adicionales.
5. WHEN el modo es `single` (sin periodo comparativo), THE síntesis final SHALL omitir la comparación vs periodo comparativo e incluir solo CSAT actual y gap vs target.

---

### Requisito 10: Query DAX para % Issues del focus touchpoint vs L7D

**User Story:** Como analista de NPS, quiero obtener el porcentaje de issues relacionados con el focus touchpoint para el periodo actual y el comparativo, para cuantificar la magnitud del problema en cada segmento.

#### Criterios de Aceptación

1. THE `PBIDataCollector` SHALL exponer un método `collect_focus_touchpoint_issues_pct(node_path, start_date, end_date, comparison_start_date, comparison_end_date, touchpoint_display_name)` que ejecute la query DAX de % issues del touchpoint.
2. THE query SHALL devolver dos filas: una para el periodo actual (`L7D`) y otra para el periodo comparativo (`L7D_prev`), cada una con el campo `Pct_Issues`.
3. THE query SHALL ser parametrizable por: periodo actual (start/end), periodo comparativo (start_prev/end_prev), segmento (cabina/haul/compañía según el nodo) y `touchpoint_display_name` (nombre del touchpoint en `Issue_touchpoint_Dict`).
4. THE query SHALL filtrar por `Issue_touchpoint_Dict[issue_type_3] = "{touchpoint_display_name}"` y por `explanatory_drivers = 1`.
5. THE System SHALL resolver el `touchpoint_display_name` a partir del `filtered_name` usando el mapeo `filtered_name → display_name` antes de llamar a `collect_focus_touchpoint_issues_pct`. Si no existe mapeo para el touchpoint, THE System SHALL omitir la query de % issues y registrar un aviso en el log.
6. WHEN `focus_touchpoint` está activo y existe `display_name` para el touchpoint, THE `CausalExplanationAgent` SHALL ejecutar `collect_focus_touchpoint_issues_pct` para los nodos Global, LH y/o SH relevantes según el segmento CLI.
7. WHEN el focus touchpoint tiene importancia según exp drivers (SHAP significativo) en un nodo, THE `CausalExplanationAgent` SHALL ejecutar también `collect_focus_touchpoint_issues_pct` para ese nodo aunque no sea Global/LH/SH de nivel superior.
8. THE `AnomalySummaryAgent` SHALL incluir en la síntesis final, para cada segmento relevante, el resultado de `collect_focus_touchpoint_issues_pct` mostrando el `Pct_Issues` del periodo actual y la variación vs el periodo comparativo.
9. IF la query `collect_focus_touchpoint_issues_pct` falla o no devuelve datos para un nodo, THEN THE System SHALL continuar el análisis sin ese dato y registrar un aviso en el log.
10. THE cobertura de segmentos para `collect_focus_touchpoint_issues_pct` SHALL seguir la misma lógica que el resto del análisis: Global implica Global+LH+SH, SH implica solo SH, LH implica solo LH.

---

### Requisito 11: Mapeo filtered_name → display_name para queries de issues

**User Story:** Como desarrollador, quiero un mecanismo de mapeo entre el `filtered_name` de un touchpoint (usado en exp drivers) y su `display_name` (usado en `Issue_touchpoint_Dict`), para poder parametrizar correctamente las queries de % issues.

#### Criterios de Aceptación

1. THE System SHALL mantener un diccionario de mapeo `TOUCHPOINT_DISPLAY_NAME_MAP` que traduzca `filtered_name` a `display_name` para los touchpoints conocidos. Ejemplo: `{"ifl_100_cabin_crew_satisfaction": "Cabin Crew", "ifl_200_food": "Food & Beverage"}`.
2. THE System SHALL exponer una función `get_touchpoint_display_name(filtered_name: str) -> Optional[str]` que devuelva el `display_name` o `None` si no existe mapeo.
3. IF no existe mapeo para un `filtered_name`, THE System SHALL registrar un aviso en el log indicando que no se puede ejecutar la query de % issues para ese touchpoint.
4. THE diccionario de mapeo SHALL ser extensible sin cambios de código (p.ej. configurable vía constante en `pbi_collector.py` o fichero de configuración).
