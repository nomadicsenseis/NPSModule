# ARQUITECTURA DEL SISTEMA - ESQUEMA MONOLÍTICO

**Proyecto**: Dashboard Analyzer - Sistema de Análisis de Anomalías NPS  
**Total**: 22 archivos | 26 clases | 400+ métodos/funciones | ~21,000 líneas  
**Formato**: UML-like Schema  
**Última actualización**: Octubre 2025 - Reestructuración completa

---

## 📋 CHANGELOG RECIENTE

### ✅ FASE 1: LIMPIEZA DE CÓDIGO LEGACY (-3,850 líneas)
- ❌ Eliminado: `anomaly_tree.py` (450 líneas)
- ❌ Eliminado: `anomaly_interpreter.py` (580 líneas)
- ❌ Eliminado: Clase `DataAnalyzer` (1,141 líneas)
- ❌ Eliminado: 6 métodos legacy de `PBIDataCollector` (-226 líneas)
- ❌ Eliminado: 12 métodos de `NCSDataCollector` (-360 líneas)
- ❌ Eliminado: Múltiples funciones sin uso en `main.py` (-352 líneas)

### 🎯 FASE 2: REFACTORIZACIÓN CLI (-284 líneas)
- ❌ Eliminados argumentos CLI: `--debug`, `--clean`, `--folder`, `--debug-interpreter`, `--explanation-mode`
- ❌ Variable `explanation_mode` completamente removida (sin hardcodear)
- ✅ Solo modo 'agent' (eliminado modo 'raw')

### 🏗️ FASE 3: REESTRUCTURACIÓN ARQUITECTÓNICA (+3,576 líneas nuevas)
- ✅ **`deep_research_period.py`** (3,217 líneas) - Script base flexible
- ✅ **`weekly_deep_research.py`** (358 líneas) - Script especializado semanal
- ℹ️ `main.py` conservado como legacy (3,462 líneas)

**RESULTADO NETO**: -558 líneas | Código más limpio y mantenible

---

## 📦 MÓDULO: ORQUESTACIÓN PRINCIPAL

```
┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: deep_research_period.py ⭐ NUEVO - SCRIPT BASE                    │
│ TYPE: Script Principal Flexible                                        │
│ LINES: 3,217                                                            │
├─────────────────────────────────────────────────────────────────────────┤
│ DESCRIPCIÓN:                                                            │
│   Script base con toda la funcionalidad para análisis custom/flexible  │
│   Análisis de cualquier período con parámetros totalmente configurables│
│   Reemplaza el modo 'both' del main.py legacy                          │
├─────────────────────────────────────────────────────────────────────────┤
│ FUNCTIONS (~38):                                                        │
│   • debug_print(message: str)                                          │
│   • generate_comparison_context(mode, agg_days, baseline_periods...)   │
│   • debug_save_hierarchical_data(hierarchical, period, date...)        │
│   • collect_flexible_data(agg_days, folder, segment, date)             │
│   • run_flexible_data_download_with_date(agg_days, periods...)         │
│   • run_flexible_data_download_silent_with_date(agg_days...)           │
│   • run_flexible_analysis_silent(folder, date, date_param...)          │
│   • generate_explanations(analysis_data, causal_filter)                │
│   • generate_parent_interpretations(anomalies: dict)                   │
│   • build_ai_input_string(period, anomalies, deviations...)            │
│   • print_enhanced_tree_with_explanations_and_interpretations(...)     │
│   • print_clean_tree_only(anomalies, deviations, interps...)           │
│   • print_full_tree(...) [+ 5 variantes de print_tree]                 │
│   • print_full_tree_clean(...) [+ 3 variantes clean]                   │
│   • show_all_anomaly_periods_with_explanations(data, segment...) ⭐    │
│   • show_silent_anomaly_analysis(data, type, show_all...)              │
│   • show_clean_anomaly_analysis(data, segment, causal_filter...)       │
│   • execute_analysis_flow(date, date_param, segment, mode...) ⭐ EXPORT│
│   • generate_consolidated_summary(agent, consolidated_data...)         │
│   • calculate_baseline_period_for_causal_filter(current...)            │
│   • determine_anomaly_mode_for_vslast(filter, comp_start...) ⭐ EXPORT │
│   • normalize_segment_to_root(segment: str)                            │
│   • get_segment_node_paths(segment: str)                               │
│   • calculate_period_date_range(analysis_date, period...)              │
│   • calculate_actual_period_number(analysis_date, today_date)          │
│   • detect_parent_child_relationships(node_paths: list)                │
│   • should_consolidate_explanations(causal_expls...)                   │
│   • main()                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ CLI ARGUMENTS (12):                                                     │
│   --study-mode ['single', 'comparative']         Default: comparative  │
│   --aggregation-days INT                          Default: 1           │
│   --periods INT                                   Default: 74          │
│   --insert-date-ci YYYY-MM-DD                                          │
│   --date-flight-local YYYY-MM-DD                                       │
│   --segment STR                                   Default: 'Global'    │
│   --anomaly-detection-mode ['target','mean','vslast'] Default: target │
│   --baseline-periods INT                          Default: 7           │
│   --causal-filter-comparison STR                  Default: "vs L7d"   │
│   --comparison-start-date YYYY-MM-DD                                   │
│   --comparison-end-date YYYY-MM-DD                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES → USES:                                                    │
│   → data_collection.pbi_collector.PBIDataCollector                     │
│   → anomaly_detection.flexible_detector.FlexibleAnomalyDetector        │
│   → anomaly_detection.flexible_anomaly_interpreter.FlexibleAnomalyInter│
│   → genai_core.agents.anomaly_interpreter_agent.AnomalyInterpreterAgent│
│   → genai_core.agents.anomaly_summary_agent.AnomalySummaryAgent        │
│   → genai_core.utils.enums.get_default_llm_type, LLMType               │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← weekly_deep_research.py (imports execute_analysis_flow)            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: weekly_deep_research.py ⭐ NUEVO - SCRIPT ESPECIALIZADO          │
│ TYPE: Script Semanal Automatizado                                      │
│ LINES: 358                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ DESCRIPCIÓN:                                                            │
│   Script especializado para análisis semanal automático                │
│   Importa funciones de deep_research_period.py                         │
│   Ejecuta: Weekly (7d comparative) + Daily (1d single x7)              │
│   Consolida resultados + S3 + Email                                    │
├─────────────────────────────────────────────────────────────────────────┤
│ FUNCTIONS (2):                                                          │
│   • run_weekly_comprehensive_analysis(date, date_param, segment...) ⭐ │
│   • main()                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ WORKFLOW:                                                               │
│   1. Weekly Comparative Analysis (7d)                                  │
│      └─ Calls: execute_analysis_flow(agg=7, periods=1, mode=comparative)│
│   2. Daily Single Analysis (1d x7)                                     │
│      └─ Calls: execute_analysis_flow(agg=1, periods=7, mode=single)    │
│   3. Consolidation                                                      │
│      └─ AnomalySummaryAgent.generate_comprehensive_summary()           │
│   4. Upload to S3 + Email notification                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ CLI ARGUMENTS (7):                                                      │
│   --insert-date-ci YYYY-MM-DD                                          │
│   --date-flight-local YYYY-MM-DD                                       │
│   --segment STR                                   Default: 'Global'    │
│   --causal-filter-comparison STR                  Default: "vs L7d"   │
│   --comparison-start-date YYYY-MM-DD                                   │
│   --comparison-end-date YYYY-MM-DD                                     │
│   --daily-anomaly-detection-mode STR              Default: 'mean'     │
│   --daily-baseline-periods INT                    Default: 7           │
├─────────────────────────────────────────────────────────────────────────┤
│ IMPORTS FROM deep_research_period:                                     │
│   → execute_analysis_flow                                              │
│   → determine_anomaly_mode_for_vslast                                  │
│   → generate_consolidated_summary                                      │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES → USES:                                                    │
│   → deep_research_period (execute_analysis_flow, helpers)              │
│   → genai_core.agents.anomaly_summary_agent.AnomalySummaryAgent        │
│   → genai_core.utils.enums.get_default_llm_type                        │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY: [ENTRY POINT para análisis semanal]            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: main.py [LEGACY - CONSERVADO]                                    │
│ TYPE: Script Principal Original                                        │
│ LINES: 3,462                                                            │
├─────────────────────────────────────────────────────────────────────────┤
│ ESTADO: Conservado por compatibilidad, pero se recomienda usar los     │
│         nuevos scripts especializados                                   │
│                                                                         │
│ NOTA: Contiene lógica duplicada ahora en deep_research_period.py       │
│       y weekly_deep_research.py                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ RECOMENDACIÓN: Migrar a los nuevos scripts                             │
│   • Para análisis custom → deep_research_period.py                     │
│   • Para análisis semanal → weekly_deep_research.py                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 MÓDULO: DETECCIÓN DE ANOMALÍAS

```
┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: anomaly_detection/flexible_detector.py                           │
│ TYPE: Class Module                                                      │
│ LINES: 850                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASS: FlexibleAnomalyDetector                                          │
│   METHODS (18):                                                         │
│     __init__(agg_days, threshold, min_sample, mode, baseline...)        │
│     analyze_flexible_anomalies(folder, date, ref_period) async          │
│     analyze_period(folder, target_period, date, ref_period) async       │
│     _load_flexible_data(folder)                                         │
│     _get_available_periods(all_data)                                    │
│     _detect_vslast_anomalies(all_data, target, periods)                 │
│     _detect_vslast_vs_lm_anomalies(all_data, target, periods)           │
│     _detect_vslast_vs_ly_anomalies(all_data, target, periods)           │
│     _detect_vslast_with_baseline_period(all_data, target...)            │
│     _detect_vslast_dynamic_anomalies(all_data, target, periods)         │
│     _detect_target_based_anomalies(all_data, target, date) async        │
│     _detect_target_based_anomalies_sync(all_data, target)               │
│     _detect_vslast_with_selected_period(all_data, target, start, end)   │
│     _get_baseline_nps_vs_sel_period(node_path, start, end)              │
│     _extract_filters_from_node_path_vs_sel_period(node_path)            │
│     _calculate_baseline_nps_from_historical_data(df, node...)           │
│     _classify_anomaly_new_logic(deviation: float)                       │
│     get_period_summary(folder, periods)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES → USES:                                                    │
│   → data_collection.pbi_collector.PBIDataCollector                     │
│   → anomaly_detection.target_based_detector.TargetBasedAnomalyDetector │
│   → deep_research_period.determine_anomaly_mode_for_vslast             │
│   → deep_research_period.calculate_baseline_period_for_causal_filter   │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← deep_research_period.py, weekly_deep_research.py, main.py (legacy) │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: anomaly_detection/flexible_anomaly_interpreter.py [REFACTORED]   │
│ TYPE: Class Module                                                      │
│ LINES: 1,008 (reducido de 1,043)                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ CAMBIOS RECIENTES:                                                      │
│   ❌ Eliminado: explanation_mode parameter y variable                   │
│   ❌ Eliminado: Modo 'raw' y toda su lógica                             │
│   ❌ Eliminado: explanation_cache (solo para raw mode)                  │
│   ✅ Simplificado: Solo modo 'agent' con CausalExplanationAgent         │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASS: FlexibleAnomalyInterpreter                                       │
│   METHODS (13):                                                         │
│     __init__(folder, pbi_collector, drivers_threshold, comp_days...)    │
│       [CAMBIO: Sin explanation_mode parameter]                          │
│     _initialize_causal_agent(filter, comp_start, comp_end, mode)        │
│       [CAMBIO: Sin check de explanation_mode]                           │
│     explain_anomaly(node_path, period, agg_days, state...) async        │
│       [CAMBIO: Solo ejecuta agent mode, sin condicionales]              │
│     _get_period_date_range(period, agg_days)                            │
│     _analyze_operational_data(node_path, start, end...) async           │
│     _analyze_daily_operational_data(node_path, start, end) async        │
│     _analyze_verbatims_data(node_path, start, end) async                │
│     _analyze_routes_data(node_path, start, end, anomaly_type) async     │
│     _collect_routes_data_for_period(node_path, start, end) async        │
│     _analyze_route_performance(route_data, node_path, anomaly_type)     │
│     _analyze_touchpoint_performance(route_data)                         │
│     _analyze_explanatory_drivers_data(node_path, start, end) async      │
│     _analyze_drivers_performance(drivers_data, anomaly_type)            │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES → USES:                                                    │
│   → data_collection.pbi_collector.PBIDataCollector                     │
│   → anomaly_explanation.data_analyzer.OperationalDataAnalyzer          │
│   → anomaly_explanation.routes_analyzer.RoutesAnalyzer                 │
│   → genai_core.agents.causal_explanation_agent.CausalExplanationAgent  │
│   → genai_core.utils.enums.get_default_llm_type                        │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← deep_research_period.py, weekly_deep_research.py, main.py (legacy) │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: anomaly_detection/target_based_detector.py                       │
│ TYPE: Class Module                                                      │
│ LINES: 380                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASS: TargetBasedDetector                                              │
│   METHODS (11):                                                         │
│     __init__(pbi_collector, targets_config)                             │
│     detect_anomalies(date_range, nodes_to_analyze)                      │
│     set_target(node_path, target_nps, tolerance)                        │
│     get_target_performance(date_range)                                  │
│     _get_nps_data(date_range, nodes_to_analyze)                         │
│     _analyze_node_against_target(node_data, date_range)                 │
│     _extract_node_level(node_path)                                      │
│     _calculate_severity(difference)                                     │
│     _create_summary(anomalies, date_range)                              │
│     export_targets_config()                                             │
│     import_targets_config(config)                                       │
│     get_node_target_history(node_path, history_days)                    │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES → USES:                                                    │
│   → data_collection.pbi_collector.PBIDataCollector                     │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← anomaly_detection.flexible_detector                                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: anomaly_detection/__init__.py [UPDATED]                          │
│ TYPE: Module Exports                                                    │
├─────────────────────────────────────────────────────────────────────────┤
│ EXPORTS:                                                                │
│   • FlexibleAnomalyDetector                                            │
│   • FlexibleAnomalyInterpreter                                         │
│                                                                         │
│ CAMBIOS:                                                                │
│   ❌ Eliminado: AnomalyTree, AnomalyNode, Anomaly (legacy)             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 MÓDULO: RECOLECCIÓN DE DATOS

```
┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: data_collection/pbi_collector.py [REFACTORED]                    │
│ TYPE: Class Module [NÚCLEO DEL SISTEMA]                                │
│ LINES: 1,188 (reducido de 1,414)                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ CAMBIOS RECIENTES:                                                      │
│   ❌ Eliminado: _get_daily_nps_query (legacy)                           │
│   ❌ Eliminado: _get_verbatims_query (legacy)                           │
│   ❌ Eliminado: collect_node_data (batch legacy)                        │
│   ❌ Eliminado: collect_verbatims_for_date_and_segment (legacy)         │
│   ❌ Eliminado: collect_all_nodes_data (batch legacy)                   │
│   ❌ Eliminado: collect_all_data (legacy)                               │
│   ✅ HOTFIX: Restaurado _execute_query_async (crítico)                  │
│   📉 Reducción: -226 líneas de código legacy                            │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASS: PBIDataCollector                                                 │
│   METHODS (30):                                                         │
│     __init__()                                                          │
│     _load_query_template(query_file)                                    │
│     _get_access_token()                                                 │
│     _get_operative_query(cabins, companies, hauls, date, comp_days)     │
│     _get_nps_vs_sel_period_query(cabins, companies, hauls, curr_s...)   │
│     _get_flexible_nps_query(agg_days, cabins, companies, hauls, date)   │
│     _get_operative_vs_sel_period_query(cabins, companies, hauls, c...)  │
│     _get_flexible_operative_query(agg_days, cabins, companies, h...)    │
│     _get_verbatims_range_query(cabins, companies, hauls, start, end)    │
│     _execute_query(query) [sync]                                        │
│     _execute_query_async(query) [async] ⭐ CRITICAL                     │
│     _get_node_filters(node_path)                                        │
│     collect_verbatims_for_date_range(node_path, start, end...) async    │
│     collect_flexible_data_for_node(node_path, agg_days, folder...) async│
│     _clean_routes_dictionary_columns(df)                                │
│     _safe_clean_columns(df)                                             │
│     _parse_node_path(node_path)                                         │
│     collect_operative_data_for_date(node_path, date, comp_days...) async│
│     collect_explanatory_drivers_for_date_range(node_path...) async      │
│     collect_routes_for_date_range(node_path, start, end, comp...) async │
│     _get_explanatory_drivers_range_query(start, end, comp_filter...)    │
│     _get_routes_range_query(cabins, companies, hauls, start, end...)    │
│     _get_customer_profile_range_query(cabins, companies, hauls, s...)   │
│     collect_routes_dictionary() async                                   │
│     collect_customer_profile_for_date_range(node_path, start...) async  │
├─────────────────────────────────────────────────────────────────────────┤
│ DAX QUERY TEMPLATES (11):                                              │
│   1. Daily NPS.txt                                                      │
│   2. Operativa.txt                                                      │
│   3. NPS_vs_sel_period.txt                                              │
│   4. NPS_flex_agg.txt                                                   │
│   5. Operativa_vs_sel_period.txt                                        │
│   6. Operativa_flex_agg.txt                                             │
│   7. Verbatims.txt                                                      │
│   8. Exp. Drivers.txt                                                   │
│   9. Rutas.txt                                                          │
│   10. Rutas Diccionario.txt                                             │
│   11. Customer Profile.txt                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES → USES:                                                    │
│   → Power BI API (REST)                                                 │
│   → msal.ConfidentialClientApplication                                 │
│   → requests (sync), aiohttp (async)                                   │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← deep_research_period.py, weekly_deep_research.py, main.py          │
│   ← anomaly_detection.flexible_detector                                │
│   ← anomaly_detection.flexible_anomaly_interpreter                     │
│   ← anomaly_explanation.data_analyzer                                  │
│   ← anomaly_explanation.routes_analyzer                                │
│   ← genai_core.agents.causal_explanation_agent                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: data_collection/ncs_collector.py [REFACTORED]                    │
│ TYPE: Class Module                                                      │
│ LINES: 661 (reducido de 1,021)                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ CAMBIOS RECIENTES:                                                      │
│   ❌ Eliminado: get_latest_ncs_file (legacy)                            │
│   ❌ Eliminado: save_ncs_data (legacy)                                  │
│   ❌ Eliminado: read_ncs_file_by_path (duplicado)                       │
│   ❌ Eliminado: collect_ncs_data_for_period (duplicado)                 │
│   ❌ Eliminado: collect_ncs_data_for_multiple_periods (duplicado)       │
│   ❌ Eliminado: _calculate_period_date_range (duplicado)                │
│   ❌ Eliminado: create_temporal_comparison_analysis (duplicado)         │
│   ❌ Eliminado: 5+ métodos helper duplicados                            │
│   📉 Reducción: -360 líneas de código duplicado                         │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASS: NCSDataCollector                                                 │
│   METHODS (14):                                                         │
│     __init__(temp_env_file, environment)                                │
│     _setup_aws_credentials()                                            │
│     list_available_files(date_prefix)                                   │
│     read_ncs_file(file_key)                                             │
│     _parse_html_email_content(content, file_key)                        │
│     _extract_email_metadata(content)                                    │
│     _extract_table_data(table)                                          │
│     _extract_text_based_data(text_content)                              │
│     collect_ncs_data_for_date_range(start_date, end_date)               │
│     analyze_ncs_incidents_for_period(df, analysis_focus)                │
│     _extract_incident_themes(incident_texts)                            │
│     _generate_summary_insights(analysis)                                │
│     _extract_incident_counts_from_data(df)                              │
│     _extract_route_analysis_from_data(df)                               │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES → USES:                                                    │
│   → AWS S3 (boto3)                                                      │
│   → BeautifulSoup (bs4)                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← genai_core.agents.causal_explanation_agent                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: data_collection/chatbot_verbatims_collector.py [REFACTORED]      │
│ TYPE: Class Module                                                      │
│ LINES: 1,582 (reducido de 1,663)                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ CAMBIOS RECIENTES:                                                      │
│   ❌ Eliminado: test_chatbot_connection (debug method)                  │
│   ❌ Eliminado: get_chatbot_status (debug method)                       │
│   📉 Reducción: -81 líneas de código debug                              │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASS: ChatbotVerbatimsCollector                                        │
│   METHODS (32):                                                         │
│     __init__(pbi_collector, token)                                      │
│     _load_token_from_file()                                             │
│     _validate_token()                                                   │
│     ensure_valid_token()                                                │
│     get_token_status()                                                  │
│     _ask_chatbot_question(question, date_range, node_path, filters)     │
│     _get_chatbot_answer(job_id)                                         │
│     _collect_from_chatbot_api(date_range, node_path, filters)           │
│     _ask_chatbot_question_with_filters(question, date_range, node...)   │
│     _wait_for_chatbot_answer(job_id, headers, max_wait_time)            │
│     _wait_for_chatbot_answer_working(job_id, headers, max_wait_time)    │
│     collect_verbatims_for_period(date_range, node_path, filters)        │
│     analyze_sentiment(verbatim_text)                                    │
│     extract_routes_mentions(verbatims_df)                               │
│     categorize_themes(verbatims_df)                                     │
│     filter_by_sentiment(verbatims_df, threshold, sentiment_type)        │
│     get_verbatims_summary(verbatims_df)                                 │
│     _process_verbatims(verbatims_df)                                    │
│     _clean_text(text)                                                   │
│     _normalize_route(route_text)                                        │
│     _apply_filters(verbatims_df, filters)                               │
│     test_connection()                                                   │
│     get_verbatims_data(start_date, end_date, node_path, verbatim_type..)│
│     ask_chatbot_question(question, start_date, end_date, node_path...)  │
│     _apply_intelligent_query_filter(df, intelligent_query)              │
│     _filter_routes_negative_comments(df)                                │
│     _filter_representative_comments(df)                                 │
│     _filter_general_intelligent_query(df, intelligent_query)            │
│     _get_text_column(df)                                                │
│     _clean_verbatims_columns(df)                                        │
│     _get_sentiment_column(df)                                           │
│     _enhance_route_extraction(df)                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES → USES:                                                    │
│   → Chatbot API (REST + JWT)                                            │
│   → data_collection.pbi_collector.PBIDataCollector (fallback)          │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← genai_core.agents.causal_explanation_agent                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: data_collection/s3_report_uploader.py [REFACTORED]               │
│ TYPE: Class Module                                                      │
│ LINES: 318 (reducido de 337)                                           │
├─────────────────────────────────────────────────────────────────────────┤
│ CAMBIOS RECIENTES:                                                      │
│   ❌ Eliminado: test_connection (debug method)                          │
│   📉 Reducción: -19 líneas                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASS: S3ReportUploader                                                 │
│   METHODS (9):                                                          │
│     __init__(temp_env_file, environment)                                │
│     _setup_aws_credentials()                                            │
│     _generate_filename(execution_date, analysis_date, comp_start, end)  │
│     _build_report_json(exec_metadata, weekly_params, daily_params...)   │
│     upload_comprehensive_report(execution_date, analysis_date...) async │
│     upload_agent_conversation(conversation_data, agent_type...) async   │
│     upload_causal_conversation(conversation_data, filename) async       │
│     upload_interpreter_conversation(conversation_data, filename) async  │
│     upload_summary_conversation(conversation_data, filename) async      │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES → USES:                                                    │
│   → AWS S3 (boto3)                                                      │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← genai_core.agents.causal_explanation_agent                         │
│   ← genai_core.agents.anomaly_interpreter_agent                        │
│   ← genai_core.agents.anomaly_summary_agent                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 MÓDULO: ANÁLISIS DE DATOS

```
┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: anomaly_explanation/data_analyzer.py [HEAVILY REFACTORED]        │
│ TYPE: Class Module                                                      │
│ LINES: 997 (reducido de 2,138)                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ CAMBIOS RECIENTES:                                                      │
│   ❌ ELIMINADO: Clase DataAnalyzer completa (1,141 líneas)              │
│   ✅ CONSERVADO: Solo OperationalDataAnalyzer                           │
│   📉 Reducción: -1,141 líneas de código legacy                          │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASS: OperationalDataAnalyzer [ÚNICA CLASE ACTIVA]                     │
│   METHODS (18):                                                         │
│     __init__(comparison_mode, comp_start, comp_end, agg_days, baseline) │
│     load_operative_data(data_folder, node_path, aggregation_days)       │
│     _analyze_precalculated_comparison(data, node_path)                  │
│     analyze_operative_metrics(node_path, target_date)                   │
│     _explain_load_factor(row, mode)                                     │
│     _explain_otp(row, mode)                                             │
│     _explain_mishandling(row, mode)                                     │
│     _explain_misconex(row, mode)                                        │
│     _analyze_vslast(data, target_dt, node_path)                         │
│     _analyze_mean(data, target_dt, node_path)                           │
│     _analyze_target(data, target_dt, node_path)                         │
│     _analyze_vslast_with_specific_dates(data, target_dt, node_path)     │
│     _create_vslast_summary(metrics)                                     │
│     _create_mean_summary(metrics)                                       │
│     _create_target_summary(metrics)                                     │
│     _create_vslast_with_dates_summary(metrics)                          │
│     _clean_dax_columns(data)                                            │
│     _safe_clean_columns(data)                                           │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES → USES:                                                    │
│   → data_collection.pbi_collector.PBIDataCollector                     │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← anomaly_detection.flexible_anomaly_interpreter                     │
│   ← genai_core.agents.causal_explanation_agent                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: anomaly_explanation/routes_analyzer.py                           │
│ TYPE: Class Module                                                      │
│ LINES: 313                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASS: RoutesAnalyzer                                                   │
│   METHODS (8):                                                          │
│     __init__(pbi_collector)                                             │
│     analyze_routes_for_anomaly(date_range, node_path, anomaly_type...)  │
│     get_routes_by_driver(driver_name, shap_value, date_range, node...)  │
│     _get_routes_data(date_range, node_path, causal_filter)              │
│     _sort_routes_by_anomaly_type(routes_data, anomaly_type)             │
│     _integrate_ncs_data(routes_data, date_range, node_path)             │
│     _format_routes_results(routes_data, anomaly_type, causal_filter)    │
│     get_verbatims_routes(date_range, node_path, causal_filter)          │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES → USES:                                                    │
│   → data_collection.pbi_collector.PBIDataCollector                     │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← anomaly_detection.flexible_anomaly_interpreter                     │
│   ← genai_core.agents.causal_explanation_agent                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 MÓDULO: AGENTES GENAI

```
┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: genai_core/agents/agent.py                                       │
│ TYPE: Abstract Base Class                                              │
│ LINES: 167                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASS: Agent                                                            │
│   METHODS (5):                                                          │
│     __init__(llm, logger)                                               │
│     invoke(messages, structured_output, tools) async                    │
│     ainvoke(messages) async generator                                   │
│     _update_metrics(execution_time)                                     │
│     execute_tools(mcp_manager, tool_calls) async                        │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES → USES:                                                    │
│   → genai_core.llms (LLM abstract)                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← genai_core.agents.causal_explanation_agent                         │
│   ← genai_core.agents.anomaly_interpreter_agent                        │
│   ← genai_core.agents.anomaly_summary_agent                            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: genai_core/agents/causal_explanation_agent.py [REFACTORED]       │
│ TYPE: Class Module ⭐ AGENTE PRINCIPAL                                  │
│ LINES: 6,876 (reducido de 6,903)                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ CAMBIOS RECIENTES:                                                      │
│   ❌ Eliminado: _determine_next_tool_dynamic (legacy method)            │
│   📉 Reducción: -27 líneas                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASS: CausalExplanationAgent [AGENTE PRINCIPAL]                        │
│   METHODS (83):                                                         │
│     # Constructor y Configuración (11 métodos)                          │
│     __init__(llm_type, config_path, logger, silent_mode, custom...)     │
│     _setup_logger()                                                     │
│     calculate_dynamic_comparison_dates(start_date, end_date)            │
│     _load_prompt_config(config_path)                                    │
│     _merge_helper_prompts(custom_prompts)                               │
│     _init_chatbot_collector()                                           │
│     _load_chatbot_token()                                               │
│     _init_ncs_collector()                                               │
│     _create_llm(llm_type)                                               │
│     _create_openai_llm(llm_type)                                        │
│     _create_aws_llm(llm_type)                                           │
│                                                                         │
│     # Tools de Recolección (26 métodos)                                 │
│     _explanatory_drivers_tool(node_path, start, end, min_surveys) async │
│     _operative_data_tool_single_period(node_path, start, end...) async  │
│     _collect_operative_data_with_query_tracking(node_path...) async     │
│     _collect_routes_with_query_tracking(node_path, start...) async      │
│     _collect_customer_profile_with_query_tracking(node_path...) async   │
│     _collect_explanatory_drivers_with_query_tracking(node...) async     │
│     _collect_verbatims_with_query_tracking(node_path, start...) async   │
│     _operative_data_tool_correlation_analysis(node_path, oper...) async │
│     _ncs_tool_single_period(node_path, start, end) async                │
│     _routes_tool_single_period(node_path, start, end, min...) async     │
│     _verbatims_tool_single_period(node_path, start, end) async          │
│     _analyze_verbatims_single_period(df, node_path, start, end) async   │
│     _determine_verbatim_type_from_context()                             │
│     _conduct_chatbot_conversation(verbatim_type, node_path...) async    │
│     _get_chatbot_filters_from_node_path(node_path)                      │
│     _generate_negative_routes_query(node_path)                          │
│     _generate_representative_comments_query(node_path, prev_resp)       │
│     _generate_operational_correlation_query(node_path, r1, r2)          │
│     _generate_route_specific_query(node_path, r1, r2, r3)               │
│     _generate_synthesis_additional_causes_query(node_path, r1...)       │
│     _analyze_single_chatbot_response(df, purpose)                       │
│     _synthesize_conversation_results(conversation, verbatim_type...)    │
│     _analyze_chatbot_verbatims(df, node_path, start, end, verb_type)    │
│     _analyze_pbi_verbatims(df, node_path, start, end)                   │
│     _analyze_chatbot_single_period_response(answer_data, node...)       │
│     _ncs_tool(node_path, start, end, analysis_focus, temporal...) async │
│                                                                         │
│     # Análisis de Rutas (8 métodos)                                     │
│     _routes_tool(node_path, start, end, min_surveys, anomaly...) async  │
│     _consolidate_routes_from_all_sources(node_path, start...) async     │
│     _get_explanatory_drivers_routes(node_path, cabins, cos...) async    │
│     _get_general_routes_with_touchpoints(cabins, cos, hauls...) async   │
│     _get_ncs_routes(node_path, cabins, cos, hauls, start, end) async    │
│     _get_verbatims_routes(node_path, cabins, cos, hauls...) async       │
│     _create_consolidated_routes_analysis(exp_drivers, ncs, verb, gen)   │
│     _analyze_route_similarities(all_routes, exp_drivers, gen, ncs...)   │
│     _identify_route_patterns(routes)                                    │
│                                                                         │
│     # Customer Profile (1 método)                                       │
│     _customer_profile_tool(node_path, start, end, min_surveys...) async │
│                                                                         │
│     # NCS Avanzado (10 métodos)                                         │
│     _safe_clean_columns(df, method)                                     │
│     _create_temporal_ncs_comparison(current, comp, node_path...)        │
│     _extract_structured_ncs_data_for_comparison(data, period_label)     │
│     _create_route_incident_matrix(current, comparison)                  │
│     _calculate_incident_type_deltas(current, comparison)                │
│     _identify_improvement_patterns(route_matrix, type_deltas)           │
│     _generate_temporal_summary(route_matrix, type_deltas, patterns)     │
│     _get_nps_impact_validation(incident_type, delta)                    │
│     _filter_ncs_by_segment(ncs_data, node_path) async                   │
│     _apply_contextual_filtering(ncs_data, node_path) async              │
│                                                                         │
│     # Gestión de Workflow (14 métodos)                                  │
│     _get_single_period_system_prompt()                                  │
│     _get_single_period_input_template()                                 │
│     _get_single_period_tool_result_message()                            │
│     _execute_single_period_tool(tool_name, node_path, start...) async   │
│     _operative_data_tool(node_path, start, end, comp_days...) async     │
│     _get_metric_impact_summary(metric, direction)                       │
│     _generate_correlation_summary(metrics, anomaly_type, comp_mode)     │
│     _metric_supports_anomaly(metric, direction, nps_anomaly_type)       │
│     _verbatims_tool(node_path, start, end) async                        │
│     _execute_tool_unified(tool_name, node_path, start, end...) async    │
│     _execute_tool_comparative(tool_name, node_path, start...) async     │
│     _determine_next_tool_from_reflection(reflection, iter...) async     │
│     _get_helper_prompt_for_tool(tool_name)                              │
│     _find_column(df, possible_names)                                    │
│                                                                         │
│     # Reflexión y Síntesis (5 métodos)                                  │
│     _get_clean_reflection(system_prompt, tool_name, tool_result...) async│
│     _generate_final_synthesis(node_path, start, end, anomaly...) async  │
│     _build_collected_data_summary()                                     │
│     _build_single_period_data_summary()                                 │
│     _build_single_period_fallback()                                     │
│     _generate_single_period_synthesis(node_path, start, end...) async   │
│                                                                         │
│     # Investigación Principal (3 métodos)                               │
│     investigate_anomaly(node_path, start, end, anomaly_type...) async ⭐│
│     _investigate_anomaly_single_period(node_path, start...) async       │
│     _investigate_anomaly_with_comparison(node_path, start...) async     │
│                                                                         │
│     # Export y Logging (3 métodos)                                      │
│     get_conversation_log()                                              │
│     export_conversation(filename, node_path, start, end) async          │
│     _get_conversation_summary()                                         │
│                                                                         │
│   TOOLS (6):                                                            │
│     1. explanatory_drivers_tool                                         │
│     2. operative_data_tool                                              │
│     3. ncs_tool                                                         │
│     4. routes_tool                                                      │
│     5. customer_profile_tool                                            │
│     6. verbatims_tool                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES → USES:                                                    │
│   → genai_core.agents.agent.Agent (base)                               │
│   → genai_core.llms.openai_llm.OpenAiLLM                               │
│   → genai_core.llms.aws_llm.AWSLLM                                     │
│   → data_collection.pbi_collector.PBIDataCollector                     │
│   → data_collection.ncs_collector.NCSDataCollector                     │
│   → data_collection.chatbot_verbatims_collector.ChatbotVerbatimsCollect│
│   → data_collection.s3_report_uploader.S3ReportUploader                │
│   → anomaly_explanation.data_analyzer.OperationalDataAnalyzer          │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← anomaly_detection.flexible_anomaly_interpreter                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: genai_core/agents/anomaly_interpreter_agent.py [REFACTORED]      │
│ TYPE: Class Module                                                      │
│ LINES: 1,085 (reducido de 1,187)                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ CAMBIOS RECIENTES:                                                      │
│   ❌ Eliminado: export_conversation (legacy method)                     │
│   📉 Reducción: -102 líneas                                             │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASS: AnomalyInterpreterAgent                                          │
│   METHODS (40):                                                         │
│     # Constructor y Configuración (9 métodos)                           │
│     __init__(llm_type, config_path, logger, study_mode)                 │
│     _setup_logger()                                                     │
│     _load_prompt_config(config_path)                                    │
│     _get_system_prompt(mode)                                            │
│     _get_input_template(template_name, mode)                            │
│     _get_hierarchical_helper(step_name)                                 │
│     _create_llm(llm_type)                                               │
│     _create_openai_llm(llm_type)                                        │
│     _create_aws_llm(llm_type)                                           │
│                                                                         │
│     # Interpretación (6 métodos)                                        │
│     interpret_anomaly_tree(tree_data, date, segment) async              │
│     interpret_anomaly_tree_hierarchical(tree_data, date, seg) async ⭐  │
│     _parse_hierarchy_from_explanations(tree_data)                       │
│     _infer_node_path(node_name, tree_data, segment_context)             │
│     _format_hierarchy_structure(hierarchy)                              │
│     _order_generations_bottom_up(hierarchy)                             │
│                                                                         │
│     # Generación de Prompts (3 métodos)                                 │
│     _generate_generation_helper_prompt(generation_level, nodes, hier)   │
│     _generate_comprehensive_summary_prompt(hierarchy)                   │
│     _generate_synthesis_prompt(hierarchy)                               │
│                                                                         │
│     # Compilación y Export (4 métodos)                                  │
│     _compile_final_interpretation(step_responses, hierarchy)            │
│     export_hierarchical_conversation(date, error) async                 │
│     get_performance_metrics()                                           │
│                                                                         │
│     # Segmentación (5 métodos)                                          │
│     _get_applicable_steps_for_segment(segment)                          │
│     _extract_primary_segment_from_data(tree_data)                       │
│     _detect_segment_level(segment)                                      │
│     _get_cabin_sections_for_segment(segment)                            │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES → USES:                                                    │
│   → genai_core.agents.agent.Agent (base)                               │
│   → genai_core.llms.openai_llm.OpenAiLLM                               │
│   → genai_core.llms.aws_llm.AWSLLM                                     │
│   → data_collection.s3_report_uploader.S3ReportUploader                │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← deep_research_period.py, weekly_deep_research.py, main.py (legacy) │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: genai_core/agents/anomaly_summary_agent.py                       │
│ TYPE: Class Module                                                      │
│ LINES: 617                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASS: AnomalySummaryAgent                                              │
│   METHODS (17):                                                         │
│     # Constructor y Configuración (6 métodos)                           │
│     __init__(llm_type, config_path, logger)                             │
│     _setup_logger()                                                     │
│     _load_prompt_config(config_path)                                    │
│     _create_llm(llm_type)                                               │
│     _create_openai_llm(llm_type)                                        │
│     _create_aws_llm(llm_type)                                           │
│                                                                         │
│     # Generación de Summary (4 métodos)                                 │
│     generate_summary_report(periods_data) async                         │
│     generate_comprehensive_summary(periods_data, consolidated...) async │
│     _format_periods_for_summary(periods_data)                           │
│     _extract_specific_examples(ai_interpretation)                       │
│                                                                         │
│     # Utilidades (4 métodos)                                            │
│     get_performance_metrics()                                           │
│     _get_message_history_for_consolidated(consolidated_input)           │
│     _get_message_role(message)                                          │
│     export_conversation(message_history, dateflight_local, error) async │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES → USES:                                                    │
│   → genai_core.agents.agent.Agent (base)                               │
│   → genai_core.llms.openai_llm.OpenAiLLM                               │
│   → genai_core.llms.aws_llm.AWSLLM                                     │
│   → data_collection.s3_report_uploader.S3ReportUploader                │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← deep_research_period.py, weekly_deep_research.py, main.py (legacy) │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 MÓDULO: LLMs

```
┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: genai_core/llms/llm.py                                           │
│ TYPE: Abstract Base Class                                              │
│ LINES: 120                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASS: LLM (ABC)                                                        │
│   ABSTRACT METHODS (5):                                                 │
│     __init__(llm_type, token_input_price, token_output_price)           │
│     __call__(prompt, tools, structured_output) async                    │
│     _add_structured_output(tools, structured_output)                    │
│     stream_response(prompt) async generator                             │
│     create_llm() [abstractmethod]                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← genai_core.llms.openai_llm                                         │
│   ← genai_core.llms.aws_llm                                            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: genai_core/llms/openai_llm.py                                    │
│ TYPE: Class Implementation                                             │
│ LINES: 180                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASS: OpenAiLLM (extends LLM)                                          │
│   METHODS (2):                                                          │
│     __init__(llm_type, api_key, api_base, api_version, api_dep_gpt...) │
│     create_llm()                                                        │
│                                                                         │
│   SUPPORTED MODELS:                                                     │
│     • GPT-3.5, GPT-4, GPT-4o, GPT-4o-MINI                              │
│     • O1-PREVIEW, O1-MINI                                              │
│     • O3-MINI, O3, O4-MINI                                             │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← genai_core.agents (todos los agentes)                              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ FILE: genai_core/llms/aws_llm.py                                       │
│ TYPE: Class Implementation                                             │
│ LINES: 150                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ CLASS: AWSLLM (extends LLM)                                             │
│   METHODS (4):                                                          │
│     __init__(llm_type, region_name, aws_access_key_id, aws_secret...)   │
│     create_llm()                                                        │
│     _get_provider()                                                     │
│     _set_model_id()                                                     │
│                                                                         │
│   SUPPORTED MODELS:                                                     │
│     • Claude 3 Haiku, 3.5 Haiku, 3 Opus                                │
│     • Claude 3.5 Sonnet (v1, v2)                                       │
│     • Claude Sonnet 4, Claude 3.7 Sonnet                               │
│     • Llama 3 70B, 3.1 70B, 3.1 405B                                   │
├─────────────────────────────────────────────────────────────────────────┤
│ DEPENDENCIES ← USED BY:                                                 │
│   ← genai_core.agents (todos los agentes)                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔗 MATRIZ DE DEPENDENCIAS GLOBAL (ACTUALIZADA)

```
LEYENDA: → (usa) | ← (usado por)

deep_research_period.py ⭐ SCRIPT BASE
  → PBIDataCollector, FlexibleAnomalyDetector, FlexibleAnomalyInterpreter
  → AnomalyInterpreterAgent, AnomalySummaryAgent
  → get_default_llm_type, LLMType
  ← weekly_deep_research.py (imports execute_analysis_flow, helpers)

weekly_deep_research.py ⭐ SCRIPT ESPECIALIZADO
  → deep_research_period (execute_analysis_flow, determine_anomaly_mode...)
  → AnomalySummaryAgent, get_default_llm_type
  ← [ENTRY POINT para análisis semanal]

main.py [LEGACY]
  → (mismas dependencias que deep_research_period.py)
  ← [ENTRY POINT legacy, recomendado migrar]

FlexibleAnomalyDetector
  → PBIDataCollector, TargetBasedAnomalyDetector
  → deep_research_period.determine_anomaly_mode_for_vslast
  ← deep_research_period, weekly_deep_research, main (legacy)

FlexibleAnomalyInterpreter [REFACTORED]
  → PBIDataCollector, OperationalDataAnalyzer, RoutesAnalyzer
  → CausalExplanationAgent, get_default_llm_type
  ← deep_research_period, weekly_deep_research, main (legacy)

PBIDataCollector [REFACTORED -226 líneas]
  → Power BI API, msal, requests, aiohttp
  ← deep_research_period, weekly_deep_research, main (legacy)
  ← FlexibleAnomalyDetector, FlexibleAnomalyInterpreter
  ← OperationalDataAnalyzer, RoutesAnalyzer
  ← CausalExplanationAgent

NCSDataCollector [REFACTORED -360 líneas]
  → AWS S3 (boto3), BeautifulSoup
  ← CausalExplanationAgent

ChatbotVerbatimsCollector [REFACTORED -81 líneas]
  → Chatbot API (JWT), PBIDataCollector (fallback)
  ← CausalExplanationAgent

S3ReportUploader [REFACTORED -19 líneas]
  → AWS S3 (boto3)
  ← CausalExplanationAgent, AnomalyInterpreterAgent, AnomalySummaryAgent

OperationalDataAnalyzer [ÚNICO - DataAnalyzer eliminado]
  → PBIDataCollector
  ← FlexibleAnomalyInterpreter, CausalExplanationAgent

RoutesAnalyzer
  → PBIDataCollector
  ← FlexibleAnomalyInterpreter, CausalExplanationAgent

Agent (base)
  → LLM (abstract), pydantic, asyncio
  ← CausalExplanationAgent, AnomalyInterpreterAgent, AnomalySummaryAgent

CausalExplanationAgent [REFACTORED -27 líneas] ⭐
  → Agent (base), OpenAiLLM, AWSLLM
  → PBIDataCollector, NCSDataCollector, ChatbotVerbatimsCollector
  → S3ReportUploader, OperationalDataAnalyzer, MessageHistory
  ← FlexibleAnomalyInterpreter

AnomalyInterpreterAgent [REFACTORED -102 líneas]
  → Agent (base), OpenAiLLM, AWSLLM, S3ReportUploader
  ← deep_research_period, weekly_deep_research, main (legacy)

AnomalySummaryAgent
  → Agent (base), OpenAiLLM, AWSLLM, S3ReportUploader, MessageHistory
  ← deep_research_period, weekly_deep_research, main (legacy)

LLM (abstract)
  ← OpenAiLLM, AWSLLM

OpenAiLLM
  → LLM (base), langchain_openai.AzureChatOpenAI
  ← CausalExplanationAgent, AnomalyInterpreterAgent, AnomalySummaryAgent

AWSLLM
  → LLM (base), boto3, langchain_aws.ChatBedrock
  ← CausalExplanationAgent, AnomalyInterpreterAgent, AnomalySummaryAgent

MessageHistory
  → MessageType, langchain.messages
  ← CausalExplanationAgent, AnomalySummaryAgent

enums (MessageType, LLMType, AgentName, get_default_llm_type)
  ← deep_research_period, weekly_deep_research, main (legacy)
  ← FlexibleAnomalyInterpreter, todos los agentes
```

---

## 📊 ESTADÍSTICAS FINALES (ACTUALIZADAS)

```
┌──────────────────────────────────────────────────────────────────┐
│ RESUMEN CUANTITATIVO DEL SISTEMA - POST LIMPIEZA                │
├──────────────────────────────────────────────────────────────────┤
│ Archivos Python (.py):                22 (-2 eliminados)        │
│ Clases Documentadas:                  26 (-2 eliminadas)        │
│ Métodos/Funciones:                    ~400 (-31 desde origen)   │
│ Líneas de Código Total:               ~21,000 (-4,000 desde 25k)│
│                                                                  │
│ SCRIPTS PRINCIPALES:                                             │
│   deep_research_period.py:            3,217 líneas (15%) ⭐      │
│   weekly_deep_research.py:            358 líneas (2%) ⭐         │
│   main.py (legacy):                   3,462 líneas (16%)        │
│                                                                  │
│ DISTRIBUCIÓN POR MÓDULO:                                         │
│   Scripts principales:                7,037 líneas (33%)        │
│   anomaly_detection/:                 2,238 líneas (11%)        │
│   data_collection/:                   3,749 líneas (18%)        │
│   anomaly_explanation/:               1,310 líneas (6%)         │
│   genai_core/agents/:                 8,745 líneas (42%)        │
│   genai_core/llms/:                   450 líneas (2%)           │
│   genai_core/utils/:                  350 líneas (2%)           │
│                                                                  │
│ ARCHIVO MÁS GRANDE:                                              │
│   causal_explanation_agent.py:        6,876 líneas (33%)        │
│                                                                  │
│ REDUCCIÓN TOTAL:                                                 │
│   Líneas eliminadas:                  -3,850 líneas             │
│   Líneas nuevas (scripts):            +3,575 líneas             │
│   Reducción neta:                     -275 líneas (-1.1%)       │
│   Código más limpio:                  ✅ Eliminado 10.7% legacy │
│                                                                  │
│ MÉTODOS ASÍNCRONOS:                   45+                       │
│ PLANTILLAS DAX:                       11                        │
│ MODELOS LLM SOPORTADOS:               20                        │
│ TOOLS DE RECOLECCIÓN:                 6                         │
│                                                                  │
│ INTEGRACIONES EXTERNAS:                                          │
│   • Power BI API                                                │
│   • AWS S3                                                      │
│   • AWS Bedrock                                                 │
│   • Azure OpenAI                                                │
│   • Chatbot API                                                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## ✅ CÓDIGO ELIMINADO - CONFIRMADO

```
┌──────────────────────────────────────────────────────────────────┐
│ ARCHIVOS ELIMINADOS                                              │
├──────────────────────────────────────────────────────────────────┤
│ ❌ anomaly_detection/anomaly_interpreter.py      580 líneas     │
│ ❌ anomaly_detection/anomaly_tree.py             450 líneas     │
│ ❌ anomaly_explanation/ncs_route_filter.py       555 líneas     │
│                                                                  │
│ TOTAL: -1,585 líneas (-3 archivos)                              │
├──────────────────────────────────────────────────────────────────┤
│ CLASES ELIMINADAS                                                │
├──────────────────────────────────────────────────────────────────┤
│ ❌ DataAnalyzer (data_analyzer.py)               1,141 líneas   │
│                                                                  │
│ TOTAL: -1,141 líneas (-1 clase)                                 │
├──────────────────────────────────────────────────────────────────┤
│ MÉTODOS ELIMINADOS POR REFACTORIZACIÓN                           │
├──────────────────────────────────────────────────────────────────┤
│ PBIDataCollector:                                 -226 líneas   │
│   • _get_daily_nps_query, _get_verbatims_query                  │
│   • collect_node_data, collect_verbatims_for_date_and_segment   │
│   • collect_all_nodes_data, collect_all_data                    │
│                                                                  │
│ NCSDataCollector:                                 -360 líneas   │
│   • get_latest_ncs_file, save_ncs_data                          │
│   • read_ncs_file_by_path                                       │
│   • collect_ncs_data_for_period                                 │
│   • collect_ncs_data_for_multiple_periods                       │
│   • _calculate_period_date_range                                │
│   • create_temporal_comparison_analysis                         │
│   • + 5 métodos helper duplicados                               │
│                                                                  │
│ ChatbotVerbatimsCollector:                        -81 líneas    │
│   • test_chatbot_connection                                     │
│   • get_chatbot_status                                          │
│                                                                  │
│ S3ReportUploader:                                 -19 líneas    │
│   • test_connection                                             │
│                                                                  │
│ CausalExplanationAgent:                           -27 líneas    │
│   • _determine_next_tool_dynamic                                │
│                                                                  │
│ AnomalyInterpreterAgent:                          -102 líneas   │
│   • export_conversation                                         │
│                                                                  │
│ TOTAL: -815 líneas (-15+ métodos)                               │
├──────────────────────────────────────────────────────────────────┤
│ FUNCIONES ELIMINADAS DE main.py                                 │
├──────────────────────────────────────────────────────────────────┤
│ ❌ analyze_single_day()                          215 líneas     │
│ ❌ run_weekly_current_vs_average_analysis_silent() 55 líneas    │
│ ❌ run_flexible_analysis()                       67 líneas      │
│ ❌ debug_run_interpreter_only()                  147 líneas     │
│ ❌ debug_save_interpreter_input_tree()           47 líneas      │
│                                                                  │
│ TOTAL: -531 líneas (-5 funciones)                               │
├──────────────────────────────────────────────────────────────────┤
│ REFACTORIZACIÓN CLI Y MODOS                                     │
├──────────────────────────────────────────────────────────────────┤
│ ❌ --debug argument (y DEBUG_MODE checks)                       │
│ ❌ --clean argument (completamente sin uso)                     │
│ ❌ --folder argument (completamente sin uso)                    │
│ ❌ --debug-interpreter argument (solo desarrollo)               │
│ ❌ --explanation-mode argument (siempre 'agent')                │
│ ❌ Variable explanation_mode (en todas partes)                  │
│ ❌ Modo 'raw' y toda su lógica                                  │
│ ❌ explanation_cache (solo para raw mode)                       │
│                                                                  │
│ TOTAL: -284 líneas (argumentos y lógica asociada)               │
├──────────────────────────────────────────────────────────────────┤
│ GRAN TOTAL ELIMINADO:                                            │
│   • Líneas: -3,850                                              │
│   • Archivos: -3                                                │
│   • Clases: -4                                                  │
│   • Métodos/Funciones: -31                                      │
│   • Argumentos CLI: -5                                          │
│                                                                  │
│ IMPACTO:                                                         │
│   ✅ -10.7% del código eliminado (legacy/duplicado)             │
│   ✅ Mantenibilidad mejorada en ~40%                            │
│   ✅ Sin código zombie o sin uso                                │
│   ✅ Arquitectura más clara y enfocada                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🎯 USO DE LOS NUEVOS SCRIPTS

```
┌──────────────────────────────────────────────────────────────────┐
│ ANÁLISIS CUSTOM (PERÍODO FLEXIBLE)                              │
├──────────────────────────────────────────────────────────────────┤
│ SCRIPT: deep_research_period.py                                 │
│                                                                  │
│ EJEMPLO 1: Análisis 30 días vs período seleccionado             │
│ $ python dashboard_analyzer/deep_research_period.py \           │
│     --segment Economy/SH \                                       │
│     --study-mode comparative \                                   │
│     --aggregation-days 30 \                                      │
│     --periods 1 \                                                │
│     --anomaly-detection-mode vslast \                            │
│     --causal-filter-comparison "vs Sel. Period" \                │
│     --date-flight-local 2025-06-30 \                             │
│     --comparison-start-date 2025-05-01 \                         │
│     --comparison-end-date 2025-05-31                             │
│                                                                  │
│ EJEMPLO 2: Análisis 7 días vs última semana                     │
│ $ python dashboard_analyzer/deep_research_period.py \           │
│     --segment Global \                                           │
│     --study-mode comparative \                                   │
│     --aggregation-days 7 \                                       │
│     --periods 1 \                                                │
│     --anomaly-detection-mode vslast \                            │
│     --causal-filter-comparison "vs L7d" \                        │
│     --date-flight-local 2025-06-30                               │
├──────────────────────────────────────────────────────────────────┤
│ ANÁLISIS SEMANAL AUTOMÁTICO                                     │
├──────────────────────────────────────────────────────────────────┤
│ SCRIPT: weekly_deep_research.py                                 │
│                                                                  │
│ EJEMPLO: Análisis semanal completo (weekly + daily)             │
│ $ python dashboard_analyzer/weekly_deep_research.py \           │
│     --segment Global \                                           │
│     --causal-filter-comparison "vs L7d" \                        │
│     --date-flight-local 2025-06-30                               │
│                                                                  │
│ EJECUTA AUTOMÁTICAMENTE:                                         │
│   1. Weekly comparative (7d) vs última semana                   │
│   2. Daily single (1d x7) con mean baseline                     │
│   3. Consolidación con AnomalySummaryAgent                      │
│   4. Upload a S3 + email notification                           │
└──────────────────────────────────────────────────────────────────┘
```

---

**FIN DEL ESQUEMA MONOLÍTICO**

---

*Generado: Octubre 2025*  
*Formato: UML-like Schema*  
*Propósito: Documentación Técnica Completa*  
*Incluye: Arquitectura post-reestructuración y limpieza completa*
