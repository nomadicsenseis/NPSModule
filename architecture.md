# Arquitectura del Sistema - Dashboard Analyzer

## Archivo: `dashboard_analyzer/main.py`

### Funciones de Utilidad y Debug

#### 1. `debug_print`
- **Argumentos**: `message: str`
- **Propósito**: Imprime mensajes de debug solo cuando DEBUG_MODE está activado
- **Dependencias**: Ninguna
- **Variables Globales**: `DEBUG_MODE`

#### 2. `generate_comparison_context`
- **Argumentos**: 
  - `anomaly_detection_mode: str`
  - `aggregation_days: int`
  - `baseline_periods: int = 7`
  - `baseline_description: str = None`
- **Propósito**: Genera texto descriptivo del contexto de comparación según el modo de detección
- **Dependencias**: Ninguna

#### 3. `debug_save_hierarchical_data`
- **Argumentos**:
  - `hierarchical_explanation: str`
  - `period: int`
  - `date_param: Optional[str] = None`
  - `causal_explanations: Optional[dict] = None`
  - `relationships: Optional[dict] = None`
- **Propósito**: Guarda datos jerárquicos para debugging del intérprete
- **Dependencias**:
  - `datetime` (stdlib)
  - `json` (stdlib)
  - `Path` de `pathlib` (stdlib)

#### 4. `debug_run_interpreter_only`
- **Argumentos**: `debug_file: str`
- **Propósito**: Ejecuta solo el agente intérprete usando datos guardados previamente
- **Dependencias**:
  - `build_ai_input_string()` (local)
  - `AnomalyInterpreterAgent` de `dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_interpreter_agent`
  - `get_default_llm_type()` de `dashboard_analyzer.anomaly_explanation.genai_core.utils.enums`
  - `asyncio.wait_for()` (stdlib)
  - `logging` (stdlib)

---

### Funciones de Recolección de Datos

#### 5. `collect_flexible_data`
- **Argumentos**:
  - `aggregation_days: int`
  - `target_folder: str`
  - `segment: str = "Global"`
  - `analysis_date: datetime = None`
- **Propósito**: Recolecta datos NPS flexibles para todos los nodos en un segmento
- **Dependencias**:
  - `PBIDataCollector()` de `dashboard_analyzer.data_collection.pbi_collector`
  - `get_segment_node_paths()` (local)
  - `debug_print()` (local)

#### 6. `run_flexible_data_download_with_date`
- **Argumentos**:
  - `aggregation_days: int`
  - `periods: int`
  - `start_date: datetime`
  - `date_parameter: str`
  - `segment: str = "Global"`
- **Propósito**: Descarga datos flexibles con fecha personalizada (verbose)
- **Dependencias**:
  - `collect_flexible_data()` (local)
  - `datetime` (stdlib)

#### 7. `run_flexible_data_download_silent_with_date`
- **Argumentos**:
  - `aggregation_days: int`
  - `periods: int`
  - `start_date: datetime`
  - `date_parameter: str`
  - `segment: str = "Global"`
- **Propósito**: Descarga datos flexibles silenciosamente (sin output)
- **Dependencias**:
  - `PBIDataCollector()` de `dashboard_analyzer.data_collection.pbi_collector`
  - `get_segment_node_paths()` (local)
  - `redirect_stdout`, `redirect_stderr` de `contextlib` (stdlib)

---

### Funciones de Análisis y Detección de Anomalías

#### 8. `run_flexible_analysis_silent`
- **Argumentos**:
  - `data_folder: str`
  - `analysis_date: datetime = None`
  - `date_parameter: str = None`
  - `anomaly_detection_mode: str = "target"`
  - `baseline_periods: int = 7`
  - `causal_filter: str = "vs L7d"`
  - `periods: int = 7`
  - `causal_comparison_dates: tuple = None`
- **Propósito**: Ejecuta análisis flexible completamente silencioso
- **Dependencias**:
  - `FlexibleAnomalyDetector` de `dashboard_analyzer.anomaly_detection.flexible_detector`
  - `calculate_actual_period_number()` (local)
  - `Path` de `pathlib` (stdlib)
  - `redirect_stdout`, `redirect_stderr` (stdlib)

#### 9. `run_flexible_analysis`
- **Argumentos**:
  - `data_folder: str`
  - `explanation_mode: str = "agent"`
  - `analysis_date: datetime = None`
  - `anomaly_detection_mode: str = "target"`
  - `baseline_periods: int = 7`
  - `periods: int = 7`
- **Propósito**: Ejecuta análisis flexible con output verbose
- **Dependencias**:
  - `FlexibleAnomalyDetector` de `dashboard_analyzer.anomaly_detection.flexible_detector`
  - `Path` de `pathlib` (stdlib)

#### 10. `run_weekly_current_vs_average_analysis_silent`
- **Argumentos**:
  - `data_folder: str`
  - `analysis_date: datetime = None`
  - `anomaly_detection_mode: str = "target"`
  - `baseline_periods: int = 7`
- **Propósito**: Análisis semanal silencioso (semana actual vs promedio)
- **Dependencias**:
  - `FlexibleAnomalyDetector` de `dashboard_analyzer.anomaly_detection.flexible_detector`
  - `Path` de `pathlib` (stdlib)
  - `redirect_stdout`, `redirect_stderr` (stdlib)

---

### Funciones de Generación de Explicaciones

#### 11. `generate_explanations`
- **Argumentos**:
  - `analysis_data: dict`
  - `causal_filter: str = "vs L7d"`
- **Propósito**: Genera explicaciones comprehensivas para nodos con anomalías
- **Dependencias**:
  - `PBIDataCollector()` de `dashboard_analyzer.data_collection.pbi_collector`
  - `FlexibleAnomalyInterpreter` de `dashboard_analyzer.anomaly_detection.flexible_anomaly_interpreter`
  - `asyncio.wait_for()` (stdlib)

#### 12. `generate_parent_interpretations`
- **Argumentos**: `anomalies: dict`
- **Propósito**: Genera interpretaciones de nodos padre basadas en estados de hijos
- **Dependencias**: Ninguna (lógica interna)

#### 13. `build_ai_input_string`
- **Argumentos**:
  - `period: int`
  - `anomalies: dict`
  - `deviations: dict`
  - `interpretations: dict`
  - `explanations: dict`
  - `date_range: tuple`
  - `segment_filter: str = "Global"`
  - `nps_values: dict = None`
- **Propósito**: Construye string de input comprehensivo para interpretación AI
- **Dependencias**:
  - `normalize_segment_to_root()` (local)
  - Funciones internas: `get_state_desc()`, `add_node_info()`

---

### Funciones de Visualización y Output

#### 14. `print_enhanced_tree_with_explanations_and_interpretations`
- **Argumentos**:
  - `anomalies: dict`
  - `deviations: dict`
  - `explanations: dict`
  - `interpretations: dict`
  - `aggregation_days: int`
  - `target_period: int`
  - `date_range: tuple = None`
  - `segment_filter: str = "Global"`
  - `analysis_date: datetime = None`
  - `date_parameter: str = None`
  - `nps_values: dict = None`
- **Propósito**: Imprime árbol mejorado con explicaciones e interpretaciones
- **Dependencias**:
  - `calculate_period_date_range()` (local)
  - `debug_print()` (local)
  - Funciones de impresión: `print_full_tree()`, `print_lh_tree()`, `print_sh_tree()`, etc.

#### 15. `print_clean_tree_only`
- **Argumentos**:
  - `anomalies: dict`
  - `deviations: dict`
  - `interpretations: dict`
  - `segment_filter: str = "Global"`
  - `nps_values: dict = None`
- **Propósito**: Imprime solo la estructura del árbol sin explicaciones
- **Dependencias**:
  - Funciones de impresión limpia: `print_full_tree_clean()`, `print_lh_tree_clean()`, etc.

#### 16-21. Funciones de Impresión de Árboles
- **Funciones**:
  - `print_full_tree()` - Árbol completo con explicaciones
  - `print_lh_tree()` - Solo Long Haul
  - `print_sh_tree()` - Solo Short Haul
  - `print_sh_economy_tree()` - SH Economy específico
  - `print_sh_business_tree()` - SH Business específico
  - `print_single_node()` - Nodo individual
- **Argumentos**: `anomalies`, `get_state_description`, `get_deviation_text`, `print_interpretation`, `print_explanation`
- **Propósito**: Imprimen diferentes vistas del árbol jerárquico
- **Dependencias**: Funciones callback pasadas como parámetros

#### 22-27. Funciones de Impresión Limpia
- **Funciones**:
  - `print_full_tree_clean()`
  - `print_lh_tree_clean()`
  - `print_sh_tree_clean()`
  - `print_single_node_clean()`
- **Argumentos**: Similar a versiones completas pero sin `print_explanation`
- **Propósito**: Versiones limpias de las funciones de impresión
- **Dependencias**: Funciones callback pasadas como parámetros

---

### Funciones de Análisis Comprehensivo

#### 28. `show_all_anomaly_periods_with_explanations`
- **Argumentos**:
  - `analysis_data: dict`
  - `segment: str = "Global"`
  - `explanation_mode: str = "agent"`
  - `causal_filter: str = "vs L7d"`
  - `comparison_start_date: datetime = None`
  - `comparison_end_date: datetime = None`
- **Propósito**: Muestra árboles para todos los períodos con anomalías incluyendo explicaciones
- **Dependencias**:
  - `PBIDataCollector()` de `dashboard_analyzer.data_collection.pbi_collector`
  - `FlexibleAnomalyInterpreter` de `dashboard_analyzer.anomaly_detection.flexible_anomaly_interpreter`
  - `AnomalyInterpreterAgent` de `dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_interpreter_agent`
  - `AnomalySummaryAgent` de `dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_summary_agent`
  - `get_default_llm_type()` de `dashboard_analyzer.anomaly_explanation.genai_core.utils.enums`
  - `generate_parent_interpretations()` (local)
  - `generate_comparison_context()` (local)
  - `build_ai_input_string()` (local)
  - `print_enhanced_tree_with_explanations_and_interpretations()` (local)
  - `normalize_segment_to_root()` (local)
  - `calculate_period_date_range()` (local)
  - `asyncio.wait_for()` (stdlib)

#### 29. `show_silent_anomaly_analysis`
- **Argumentos**:
  - `analysis_data: dict`
  - `analysis_type: str`
  - `show_all_periods: bool = False`
  - `segment: str = "Global"`
  - `explanation_mode: str = "agent"`
  - `causal_filter: str = "vs L7d"`
  - `comparison_start_date: datetime = None`
  - `comparison_end_date: datetime = None`
- **Propósito**: Versión silenciosa del análisis de anomalías
- **Dependencias**: Similar a `show_all_anomaly_periods_with_explanations` con supresión de output

#### 30. `show_clean_anomaly_analysis`
- **Argumentos**:
  - `analysis_data: dict`
  - `segment: str = "Global"`
  - `causal_filter: str = "vs L7d"`
  - `comparison_start_date: datetime = None`
  - `comparison_end_date: datetime = None`
- **Propósito**: Análisis limpio y enfocado: árbol + workflow + resumen
- **Dependencias**:
  - `PBIDataCollector()` de `dashboard_analyzer.data_collection.pbi_collector`
  - `FlexibleAnomalyInterpreter` de `dashboard_analyzer.anomaly_detection.flexible_anomaly_interpreter`
  - `AnomalyInterpreterAgent` de `dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_interpreter_agent`
  - `print_clean_tree_only()` (local)
  - `build_ai_input_string()` (local)

---

### Funciones de Análisis de Día Individual

#### 31. `analyze_single_day`
- **Argumentos**:
  - `target_date: datetime`
  - `segment: str`
  - `explanation_mode: str`
- **Propósito**: Analiza un solo día usando flujo de análisis jerárquico completo
- **Dependencias**:
  - `PBIDataCollector()` de `dashboard_analyzer.data_collection.pbi_collector`
  - `FlexibleAnomalyDetector` de `dashboard_analyzer.anomaly_detection.flexible_detector`
  - `FlexibleAnomalyInterpreter` de `dashboard_analyzer.anomaly_detection.flexible_anomaly_interpreter`
  - `AnomalyInterpreterAgent` de `dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_interpreter_agent`
  - `get_segment_node_paths()` (local)
  - `generate_parent_interpretations()` (local)
  - `print_clean_tree_only()` (local)
  - `build_ai_input_string()` (local)
  - `normalize_segment_to_root()` (local)
  - `tempfile.mkdtemp()` (stdlib)

---

### Funciones de Análisis Comprehensivo y Consolidación

#### 32. `run_comprehensive_analysis`
- **Argumentos**:
  - `analysis_date: datetime`
  - `date_parameter: str`
  - `segment: str = "Global"`
  - `explanation_mode: str = "agent"`
  - `causal_filter: str = "vs L7d"`
  - `comparison_start_date: Optional[datetime] = None`
  - `comparison_end_date: Optional[datetime] = None`
  - `date_flight_local: Optional[str] = None`
  - `daily_anomaly_detection_mode: str = 'mean'`
  - `daily_baseline_periods: int = 7`
  - `daily_aggregation_days: int = 1`
  - `daily_periods: int = 7`
- **Propósito**: Orquestador principal que ejecuta análisis semanal comparativo y análisis diario
- **Dependencias**:
  - `execute_analysis_flow()` (local)
  - `AnomalySummaryAgent` de `dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_summary_agent`
  - `get_default_llm_type()` de `dashboard_analyzer.anomaly_explanation.genai_core.utils.enums`
  - `determine_anomaly_mode_for_vslast()` (local)

#### 33. `execute_analysis_flow`
- **Argumentos**:
  - `analysis_date: datetime`
  - `date_parameter: str`
  - `segment: str`
  - `explanation_mode: str`
  - `anomaly_detection_mode: str`
  - `baseline_periods: int`
  - `aggregation_days: int`
  - `periods: int`
  - `causal_filter: Optional[str]`
  - `comparison_start_date: Optional[datetime] = None`
  - `comparison_end_date: Optional[datetime] = None`
  - `date_flight_local: Optional[str] = None`
  - `study_mode: str = "comparative"`
- **Propósito**: Ejecuta un flujo completo de análisis para una configuración dada
- **Dependencias**:
  - `run_flexible_data_download_silent_with_date()` (local)
  - `run_flexible_analysis_silent()` (local)
  - `show_all_anomaly_periods_with_explanations()` (local)
  - `determine_anomaly_mode_for_vslast()` (local)

#### 34. `generate_consolidated_summary`
- **Argumentos**:
  - `agent: AnomalySummaryAgent`
  - `consolidated_data: List[Dict]`
  - `date_flight_local: str = None`
- **Propósito**: Genera resumen consolidado de múltiples tipos de análisis
- **Dependencias**:
  - `agent.generate_comprehensive_summary()` (del AnomalySummaryAgent pasado)
  - `asyncio.wait_for()` (stdlib)

---

### Funciones de Utilidad y Cálculo

#### 35. `calculate_baseline_period_for_causal_filter`
- **Argumentos**:
  - `current_period: int`
  - `causal_filter: str`
  - `aggregation_days: int = 7`
- **Propósito**: Calcula el número de período baseline basado en el filtro causal
- **Dependencias**: Ninguna (lógica de cálculo)

#### 36. `determine_anomaly_mode_for_vslast`
- **Argumentos**:
  - `causal_filter: str`
  - `comparison_start_date: str = None`
  - `comparison_end_date: str = None`
- **Propósito**: Determina modo de detección de anomalías y descripción baseline
- **Dependencias**: Ninguna (lógica de mapeo)

#### 37. `normalize_segment_to_root`
- **Argumentos**: `segment: str`
- **Propósito**: Normaliza parámetro de segmento a su ruta de nodo raíz
- **Dependencias**: Ninguna (mapeo de strings)

#### 38. `get_segment_node_paths`
- **Argumentos**: `segment: str`
- **Propósito**: Genera lista de rutas de nodos basado en segmento seleccionado
- **Dependencias**: Ninguna (diccionario estático)

#### 39. `calculate_period_date_range`
- **Argumentos**:
  - `analysis_date: datetime`
  - `target_period: int`
  - `aggregation_days: int`
- **Propósito**: Calcula el rango de fechas correcto para un período
- **Dependencias**: `timedelta` de `datetime` (stdlib)

#### 40. `calculate_actual_period_number`
- **Argumentos**:
  - `analysis_date: datetime`
  - `today_date: datetime = None`
- **Propósito**: Calcula número de período actual en datos PBI para fecha de análisis dada
- **Dependencias**: `datetime` (stdlib)

---

### Funciones de Análisis de Relaciones

#### 41. `detect_parent_child_relationships`
- **Argumentos**: `node_paths: list`
- **Propósito**: Detecta relaciones padre-hijo entre nodos anómalos
- **Dependencias**: Ninguna (análisis de strings)

#### 42. `should_consolidate_explanations`
- **Argumentos**:
  - `causal_explanations: dict`
  - `relationships: dict`
- **Propósito**: Determina si explicaciones padre-hijo deben consolidarse
- **Dependencias**: Ninguna (análisis de texto)

---

### Funciones de Debug

#### 43. `debug_save_interpreter_input_tree`
- **Argumentos**:
  - `tree_data: str`
  - `date: str`
  - `segment: str`
  - `mode: str = "single"`
- **Propósito**: Guarda datos de árbol de input exactos para debugging del intérprete
- **Dependencias**:
  - `Path` de `pathlib` (stdlib)
  - `datetime` (stdlib)
  - `debug_print()` (local)

---

### Función Principal

#### 44. `main`
- **Argumentos**: Ninguno (usa argparse para CLI)
- **Propósito**: Punto de entrada principal para análisis comprehensivo de anomalías
- **Dependencias**:
  - `argparse` (stdlib)
  - `datetime` (stdlib)
  - `run_comprehensive_analysis()` (local)
  - `execute_analysis_flow()` (local)
  - Todas las funciones de utilidad para validación de argumentos

---

## Resumen de Dependencias Externas

### Módulos del Proyecto
1. `dashboard_analyzer.data_collection.pbi_collector`
   - `PBIDataCollector`

2. `dashboard_analyzer.anomaly_detection.flexible_detector`
   - `FlexibleAnomalyDetector`

3. `dashboard_analyzer.anomaly_detection.flexible_anomaly_interpreter`
   - `FlexibleAnomalyInterpreter`

4. `dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_interpreter_agent`
   - `AnomalyInterpreterAgent`

5. `dashboard_analyzer.anomaly_explanation.genai_core.agents.anomaly_summary_agent`
   - `AnomalySummaryAgent`

6. `dashboard_analyzer.anomaly_explanation.genai_core.utils.enums`
   - `get_default_llm_type()`
   - `LLMType`

### Módulos de Biblioteca Estándar
- `asyncio`
- `argparse`
- `datetime` (datetime, timedelta)
- `pathlib` (Path)
- `logging`
- `typing` (List, Dict, Any, Optional)
- `sys`
- `os`
- `json`
- `contextlib` (redirect_stdout, redirect_stderr)
- `pandas` (pd)
- `tempfile`
- `shutil`
- `re`

---

## Flujos Principales

### Flujo 1: Análisis Comprehensivo (Modo Comprehensive)
```
main() 
  → run_comprehensive_analysis()
    → execute_analysis_flow() [semanal]
      → run_flexible_data_download_silent_with_date()
      → run_flexible_analysis_silent()
      → show_all_anomaly_periods_with_explanations()
    → execute_analysis_flow() [diario]
      → run_flexible_data_download_silent_with_date()
      → run_flexible_analysis_silent()
      → show_all_anomaly_periods_with_explanations()
    → AnomalySummaryAgent.generate_comprehensive_summary()
```

### Flujo 2: Análisis Personalizado (Modo Both)
```
main()
  → execute_analysis_flow()
    → run_flexible_data_download_silent_with_date()
    → run_flexible_analysis_silent()
    → show_all_anomaly_periods_with_explanations()
      → generate_parent_interpretations()
      → build_ai_input_string()
      → AnomalyInterpreterAgent.interpret_anomaly_tree()
```

### Flujo 3: Análisis de Día Individual
```
analyze_single_day()
  → PBIDataCollector.collect_flexible_data_for_node()
  → FlexibleAnomalyDetector.analyze_period()
  → FlexibleAnomalyInterpreter.explain_anomaly()
  → build_ai_input_string()
  → AnomalyInterpreterAgent.interpret_anomaly_tree()
```

---

**Total de Funciones en main.py**: 44
**Líneas de Código**: ~4,030
**Funciones Async**: 13
**Funciones de Utilidad**: 19
**Funciones de Visualización**: 12

---

# Módulo: Detección de Anomalías (`dashboard_analyzer/anomaly_detection/`)

## Archivo: `dashboard_analyzer/anomaly_detection/__init__.py`

### Exportaciones del Módulo
- **Clases Exportadas**:
  - `AnomalyTree` - Estructura de árbol jerárquico para detección
  - `AnomalyNode` - Nodo individual en el árbol
  - `Anomaly` - Representación de una anomalía detectada
  - `FlexibleAnomalyDetector` - Detector principal con agregación flexible
  - `FlexibleAnomalyInterpreter` - Intérprete de anomalías con explicaciones

---

## Archivo: `dashboard_analyzer/anomaly_detection/flexible_detector.py`

### Clase: `FlexibleAnomalyDetector`

Sistema de detección de anomalías flexible que soporta diferentes períodos de agregación temporal (7, 14, 30 días, etc.)

#### Constructor: `__init__`
- **Argumentos**:
  - `aggregation_days: int = 7` - Días por período
  - `threshold: float = 5.0` - Umbral de desviación NPS
  - `min_sample_size: int = 5` - Tamaño mínimo de muestra
  - `detection_mode: str = "target"` - Modo: "target", "mean", "vslast", "vslast_dynamic"
  - `baseline_periods: int = 7` - Períodos baseline para modo "mean"
  - `causal_filter: str = None` - Filtro causal (vs L7d, vs LM, vs LY, etc.)
  - `causal_comparison_dates: tuple = None` - Fechas para "vs Sel. Period"
- **Propósito**: Inicializa detector con configuración flexible
- **Dependencias**:
  - `TargetBasedAnomalyDetector` (si mode="target")
  - `determine_anomaly_mode_for_vslast()` de `dashboard_analyzer.main`

#### 1. `analyze_flexible_anomalies` (async)
- **Argumentos**:
  - `data_folder: str`
  - `analysis_date: datetime = None`
  - `reference_period: int = None`
- **Propósito**: Analiza anomalías para el período más reciente
- **Retorna**: `Tuple[Dict[str, str], Dict[str, float], List[str], Dict[str, Dict[str, float]]]` - (anomalies, deviations, periods, nps_values)
- **Dependencias**:
  - `_load_flexible_data()` (local)
  - `_get_available_periods()` (local)
  - `_detect_target_based_anomalies()` (local, async)
  - `_detect_legacy_anomalies()` (local)
  - `_detect_vslast_anomalies()` (local)
  - `_detect_vslast_dynamic_anomalies()` (local)

#### 2. `analyze_period` (async)
- **Argumentos**:
  - `data_folder: str`
  - `target_period: int`
  - `analysis_date: datetime = None`
  - `reference_period: int = None`
- **Propósito**: Analiza anomalías para un período específico
- **Retorna**: `Tuple[Dict[str, str], Dict[str, float], List[str], Dict[str, Dict[str, float]]]`
- **Dependencias**: Similar a `analyze_flexible_anomalies`

#### 3. `_load_flexible_data`
- **Argumentos**: `data_folder: str`
- **Propósito**: Carga datos de agregación flexible para todos los nodos
- **Retorna**: `Dict[str, pd.DataFrame]` - Mapeo node_path -> DataFrame
- **Dependencias**:
  - `pd.read_csv()` de pandas
  - `Path` de pathlib

#### 4. `_get_available_periods`
- **Argumentos**: `all_data: Dict[str, pd.DataFrame]`
- **Propósito**: Obtiene lista de períodos que tienen datos NPS válidos
- **Retorna**: `List[int]` - Períodos ordenados (1 = más reciente)
- **Dependencias**: pandas para filtrado de datos

#### 5. `_detect_legacy_anomalies`
- **Argumentos**:
  - `all_data: Dict[str, pd.DataFrame]`
  - `target_period: int`
  - `all_periods: List[int]`
  - `reference_period: int = None`
- **Propósito**: Detecta anomalías usando media de N períodos como baseline
- **Retorna**: `Tuple[Dict[str, str], Dict[str, float], Dict[str, Dict[str, float]]]`
- **Dependencias**: `_classify_anomaly_new_logic()` (local)

#### 6. `_detect_vslast_anomalies`
- **Argumentos**:
  - `all_data: Dict[str, pd.DataFrame]`
  - `target_period: int`
  - `all_periods: List[int]`
- **Propósito**: Detecta anomalías comparando solo contra período anterior
- **Retorna**: `Tuple[Dict[str, str], Dict[str, float], Dict[str, Dict[str, float]]]`
- **Dependencias**: `_classify_anomaly_new_logic()` (local)

#### 7. `_detect_vslast_vs_lm_anomalies`
- **Argumentos**:
  - `all_data: Dict[str, pd.DataFrame]`
  - `target_period: int`
  - `all_periods: List[int]`
- **Propósito**: Detecta anomalías vs mes anterior
- **Retorna**: `Tuple[Dict[str, str], Dict[str, float], Dict[str, Dict[str, float]]]`
- **Dependencias**: `_detect_vslast_with_baseline_period()` (local)

#### 8. `_detect_vslast_vs_ly_anomalies`
- **Argumentos**:
  - `all_data: Dict[str, pd.DataFrame]`
  - `target_period: int`
  - `all_periods: List[int]`
- **Propósito**: Detecta anomalías vs año anterior
- **Retorna**: `Tuple[Dict[str, str], Dict[str, float], Dict[str, Dict[str, float]]]`
- **Dependencias**: `_detect_vslast_with_baseline_period()` (local)

#### 9. `_detect_vslast_with_baseline_period`
- **Argumentos**:
  - `all_data: Dict[str, pd.DataFrame]`
  - `target_period: int`
  - `baseline_period: int`
  - `baseline_description: str`
- **Propósito**: Lógica común para vslast con período baseline personalizado
- **Retorna**: `Tuple[Dict[str, str], Dict[str, float], Dict[str, Dict[str, float]]]`
- **Dependencias**: `_classify_anomaly_new_logic()` (local)

#### 10. `_detect_vslast_dynamic_anomalies`
- **Argumentos**:
  - `all_data: Dict[str, pd.DataFrame]`
  - `target_period: int`
  - `periods: List[int]`
- **Propósito**: Detección vslast dinámica basada en causal_filter
- **Retorna**: `Tuple[Dict[str, str], Dict[str, float], Dict[str, Dict[str, float]]]`
- **Dependencias**:
  - `calculate_baseline_period_for_causal_filter()` de `dashboard_analyzer.main`
  - `_detect_vslast_with_baseline_period()` (local)
  - `_detect_target_based_anomalies_sync()` (local)
  - `_detect_vslast_with_selected_period()` (local)

#### 11. `_detect_target_based_anomalies` (async)
- **Argumentos**:
  - `all_data: Dict[str, pd.DataFrame]`
  - `target_period: int`
  - `analysis_date: datetime`
- **Propósito**: Detecta anomalías usando enfoque basado en targets
- **Retorna**: `Tuple[Dict[str, str], Dict[str, float]]`
- **Dependencias**:
  - `self.target_detector` (TargetBasedAnomalyDetector)
  - Métodos del target_detector

#### 12. `_detect_target_based_anomalies_sync`
- **Argumentos**:
  - `all_data: Dict[str, pd.DataFrame]`
  - `target_period: int`
- **Propósito**: Versión sincrónica de detección basada en targets
- **Retorna**: `Tuple[Dict[str, str], Dict[str, float], Dict[str, Dict[str, float]]]`
- **Dependencias**: `_classify_anomaly_new_logic()` (local)

#### 13. `_detect_vslast_with_selected_period`
- **Argumentos**:
  - `all_data: Dict[str, pd.DataFrame]`
  - `target_period: int`
  - `start_date: str`
  - `end_date: str`
- **Propósito**: Detecta anomalías vs período seleccionado con fechas específicas
- **Retorna**: `Tuple[Dict[str, str], Dict[str, float], Dict[str, Dict[str, float]]]`
- **Dependencias**:
  - `datetime.strptime()` (stdlib)
  - `_get_baseline_nps_vs_sel_period()` (local)
  - `_classify_anomaly_new_logic()` (local)

#### 14. `_get_baseline_nps_vs_sel_period`
- **Argumentos**:
  - `node_path: str`
  - `start_date: datetime`
  - `end_date: datetime`
- **Propósito**: Obtiene NPS baseline real para período de comparación seleccionado
- **Retorna**: `float` - NPS baseline
- **Dependencias**:
  - `PBIDataCollector` de `dashboard_analyzer.data_collection.pbi_collector`
  - `_extract_filters_from_node_path_vs_sel_period()` (local)

#### 15. `_extract_filters_from_node_path_vs_sel_period`
- **Argumentos**: `node_path: str`
- **Propósito**: Extrae filtros de cabina, compañía y haul del path
- **Retorna**: `tuple` - (cabins, companies, hauls)
- **Dependencias**: Ninguna (parsing de strings)

#### 16. `_calculate_baseline_nps_from_historical_data`
- **Argumentos**:
  - `df: pd.DataFrame`
  - `node_path: str`
  - `start_date: datetime`
  - `end_date: datetime`
- **Propósito**: Calcula NPS baseline desde datos históricos (fallback)
- **Retorna**: `float`
- **Dependencias**: pandas, numpy

#### 17. `_classify_anomaly_new_logic`
- **Argumentos**: `deviation: float`
- **Propósito**: Clasifica anomalía con nueva lógica (cualquier bajada = "-", solo +7pts = "+")
- **Retorna**: `str` - "+", "-", o "N"
- **Dependencias**: Ninguna

#### 18. `get_period_summary`
- **Argumentos**:
  - `data_folder: str`
  - `periods: List[int]`
- **Propósito**: Genera tabla resumen para múltiples períodos
- **Retorna**: `pd.DataFrame`
- **Dependencias**: `analyze_period()` (local)

---

## Archivo: `dashboard_analyzer/anomaly_detection/flexible_anomaly_interpreter.py`

### Clase: `FlexibleAnomalyInterpreter`

Interpreta anomalías para períodos temporales flexibles con soporte dual: modo 'raw' (datos detallados) y modo 'agent' (análisis causal inteligente)

#### Constructor: `__init__`
- **Argumentos**:
  - `data_folder: str`
  - `pbi_collector: PBIDataCollector = None`
  - `drivers_survey_threshold: int = 100` - Umbral mínimo de encuestas para drivers
  - `default_comparison_days: int = 7` - Días de comparación operacional
  - `explanation_mode: str = "agent"` - "raw" o "agent"
  - `silent_mode: bool = False`
  - `detection_mode: str = "vslast"` - Modo de detección ("vslast", "mean", "target")
  - `causal_filter: str = "vs L7d"` - Filtro de comparación causal
  - `comparison_start_date: datetime = None`
  - `comparison_end_date: datetime = None`
  - `study_mode: str = None` - "single" o "comparative"
- **Propósito**: Inicializa intérprete de anomalías flexible
- **Dependencias**:
  - `OperationalDataAnalyzer` de `dashboard_analyzer.anomaly_explanation.data_analyzer`
  - `RoutesAnalyzer` de `dashboard_analyzer.anomaly_explanation.routes_analyzer`

#### 1. `_initialize_causal_agent`
- **Argumentos**:
  - `causal_filter: str = None`
  - `comparison_start_date: datetime = None`
  - `comparison_end_date: datetime = None`
  - `study_mode: str = None`
- **Propósito**: Inicializa el agente causal con filtro correcto
- **Dependencias**:
  - `CausalExplanationAgent` de `dashboard_analyzer.anomaly_explanation.genai_core.agents.causal_explanation_agent`
  - `get_default_llm_type()` de `dashboard_analyzer.anomaly_explanation.genai_core.utils.enums`

#### 2. `explain_anomaly` (async)
- **Argumentos**:
  - `node_path: str`
  - `target_period: int`
  - `aggregation_days: int`
  - `anomaly_state: str = None`
  - `start_date: datetime = None`
  - `end_date: datetime = None`
  - `anomaly_magnitude: float = None`
  - `nps_context: str = ""`
  - `causal_filter: str = "vs L7d"`
  - `comparison_start_date: datetime = None`
  - `comparison_end_date: datetime = None`
  - `anomaly_detection_mode: str = "target"`
  - `comparison_context: str = ""`
  - `baseline_periods: int = 7`
- **Propósito**: Genera explicación comprehensiva para anomalía en período flexible
- **Retorna**: `str` - Explicación (raw o análisis de agente)
- **Dependencias**:
  - `_get_period_date_range()` (local)
  - `_initialize_causal_agent()` (local)
  - `self.causal_agent.investigate_anomaly()` (si modo agent)
  - `_analyze_operational_data()` (local, async, si modo raw)
  - `_analyze_verbatims_data()` (local, async, si modo raw)
  - `_analyze_routes_data()` (local, async, si modo raw)
  - `_analyze_explanatory_drivers_data()` (local, async, si modo raw)
  - `_combine_explanations()` (local, si modo raw)

#### 3. `_get_period_date_range`
- **Argumentos**:
  - `target_period: int`
  - `aggregation_days: int`
- **Propósito**: Obtiene fechas inicio y fin para un período específico
- **Retorna**: `Tuple[datetime, datetime]`
- **Dependencias**:
  - `pd.read_csv()` de pandas
  - `Path` de pathlib

#### 4. `_analyze_operational_data` (async)
- **Argumentos**:
  - `node_path: str`
  - `start_date: datetime`
  - `end_date: datetime`
  - `aggregation_days: int`
  - `anomaly_type: str = None`
- **Propósito**: Analiza datos operacionales para rango de fechas
- **Retorna**: `str` - Explicación operacional
- **Dependencias**:
  - `_analyze_daily_operational_data()` (local, async, si aggregation_days=1)
  - `self.pbi_collector.collect_operative_data_for_date()` (async)
  - `OperationalDataAnalyzer` de `dashboard_analyzer.anomaly_explanation.data_analyzer`

#### 5. `_analyze_daily_operational_data` (async)
- **Argumentos**:
  - `node_path: str`
  - `start_date: datetime`
  - `end_date: datetime`
  - `anomaly_type: str = None`
  - `comparison_days: int = 7`
- **Propósito**: Análisis operacional diario mejorado usando datos PBI
- **Retorna**: `str`
- **Dependencias**:
  - `self.pbi_collector.collect_operative_data_for_date()` (async)
  - `self.operational_analyzer.analyze_operative_metrics()`
  - `self.operational_analyzer.get_specific_explanations()`

#### 6. `_analyze_verbatims_data` (async)
- **Argumentos**:
  - `node_path: str`
  - `start_date: datetime`
  - `end_date: datetime`
- **Propósito**: Analiza datos de verbatims con análisis de sentimiento y tópicos
- **Retorna**: `str`
- **Dependencias**:
  - `self.pbi_collector.collect_verbatims_for_date_range()`

#### 7. `_analyze_routes_data` (async)
- **Argumentos**:
  - `node_path: str`
  - `start_date: datetime`
  - `end_date: datetime`
  - `anomaly_type: str = None`
- **Propósito**: Analiza datos de rutas con análisis de performance
- **Retorna**: `str`
- **Dependencias**:
  - `_collect_routes_data_for_period()` (local, async)
  - `_analyze_route_performance()` (local)

#### 8. `_collect_routes_data_for_period` (async)
- **Argumentos**:
  - `node_path: str`
  - `start_date: datetime`
  - `end_date: datetime`
- **Propósito**: Recolecta datos de rutas para nodo y rango de fechas
- **Retorna**: `pd.DataFrame`
- **Dependencias**:
  - `self.pbi_collector.collect_routes_for_date_range()` (async)

#### 9. `_analyze_route_performance`
- **Argumentos**:
  - `route_data: pd.DataFrame`
  - `node_path: str`
  - `anomaly_type: str = None`
  - `min_surveys: int = 5`
- **Propósito**: Analiza performance de rutas e identifica insights clave
- **Retorna**: `str`
- **Dependencias**: pandas para análisis de datos

#### 10. `_analyze_touchpoint_performance`
- **Argumentos**: `route_data: pd.DataFrame`
- **Propósito**: Analiza performance de touchpoints a través de rutas
- **Retorna**: `str`
- **Dependencias**: pandas

#### 11. `_analyze_explanatory_drivers_data` (async)
- **Argumentos**:
  - `node_path: str`
  - `start_date: datetime`
  - `end_date: datetime`
  - `anomaly_type: str = None`
- **Propósito**: Analiza drivers explicativos de satisfacción (con threshold adaptativo)
- **Retorna**: `str`
- **Dependencias**:
  - `self.pbi_collector.collect_verbatims_for_date_range()` (para check de volumen)
  - `self.pbi_collector.collect_explanatory_drivers_for_date_range()` (async)
  - `_analyze_drivers_performance()` (local)

#### 12. `_analyze_drivers_performance`
- **Argumentos**:
  - `drivers_data: pd.DataFrame`
  - `anomaly_type: str = None`
- **Propósito**: Analiza drivers usando valores SHAP para contribuciones de touchpoints
- **Retorna**: `str`
- **Dependencias**: pandas para análisis de datos

#### 13. `_combine_explanations`
- **Argumentos**:
  - `node_path: str`
  - `target_period: int`
  - `aggregation_days: int`
  - `operational: str`
  - `verbatims: str`
  - `routes: str`
  - `drivers: str`
- **Propósito**: Combina todas las explicaciones en resumen comprehensivo
- **Retorna**: `str`
- **Dependencias**: Ninguna (formateo de strings)

---

## Archivo: `dashboard_analyzer/anomaly_detection/target_based_detector.py`

### Clase: `TargetBasedDetector`

Detector de anomalías basado en comparación con targets predefinidos. **NOTA**: Esta clase parece ser una versión legacy, ya que en `flexible_detector.py` hay una clase `TargetBasedAnomalyDetector` más moderna.

#### Constructor: `__init__`
- **Argumentos**:
  - `pbi_collector: PBIDataCollector`
  - `targets_config: Optional[Dict] = None`
- **Propósito**: Inicializa detector basado en targets
- **Dependencias**: Ninguna (configuración estática)

#### 1. `detect_anomalies`
- **Argumentos**:
  - `date_range: Tuple[str, str]`
  - `nodes_to_analyze: Optional[List[str]] = None`
- **Propósito**: Detecta anomalías comparando NPS actual vs targets
- **Retorna**: `Dict[str, Any]` - Anomalías detectadas
- **Dependencias**:
  - `_get_nps_data()` (local)
  - `_analyze_node_against_target()` (local)
  - `_create_summary()` (local)

#### 2. `set_target`
- **Argumentos**:
  - `node_path: str`
  - `target_nps: float`
  - `tolerance: float = 5.0`
- **Propósito**: Establece target personalizado para nodo específico
- **Dependencias**: Ninguna

#### 3. `get_target_performance`
- **Argumentos**: `date_range: Tuple[str, str]`
- **Propósito**: Obtiene rendimiento actual vs targets para todos los nodos
- **Retorna**: `Dict[str, Any]`
- **Dependencias**:
  - `_get_nps_data()` (local)
  - `_extract_node_level()` (local)

#### 4. `_get_nps_data`
- **Argumentos**:
  - `date_range: Tuple[str, str]`
  - `nodes_to_analyze: Optional[List[str]] = None`
- **Propósito**: Obtiene datos NPS del período especificado
- **Retorna**: `pd.DataFrame`
- **Dependencias**: `self.pbi_collector.collect_nps_data()`

#### 5. `_analyze_node_against_target`
- **Argumentos**:
  - `node_data: pd.Series`
  - `date_range: Tuple[str, str]`
- **Propósito**: Analiza nodo específico contra su target
- **Retorna**: `Optional[Dict]`
- **Dependencias**:
  - `_extract_node_level()` (local)
  - `_calculate_severity()` (local)

#### 6. `_extract_node_level`
- **Argumentos**: `node_path: str`
- **Propósito**: Extrae nivel del nodo del path completo
- **Retorna**: `str`
- **Dependencias**: Ninguna (parsing de strings)

#### 7. `_calculate_severity`
- **Argumentos**: `difference: float`
- **Propósito**: Calcula severidad basada en magnitud de diferencia
- **Retorna**: `str` - "low", "medium", "high", "critical"
- **Dependencias**: Ninguna

#### 8. `_create_summary`
- **Argumentos**:
  - `anomalies: List[Dict]`
  - `date_range: Tuple[str, str]`
- **Propósito**: Crea resumen de anomalías detectadas
- **Retorna**: `str`
- **Dependencias**: Ninguna

#### 9. `export_targets_config`
- **Argumentos**: Ninguno
- **Propósito**: Exporta configuración actual de targets
- **Retorna**: `Dict[str, Any]`
- **Dependencias**: `datetime.now()` (stdlib)

#### 10. `import_targets_config`
- **Argumentos**: `config: Dict[str, Any]`
- **Propósito**: Importa configuración de targets
- **Dependencias**: Ninguna

#### 11. `get_node_target_history`
- **Argumentos**:
  - `node_path: str`
  - `history_days: int = 30`
- **Propósito**: Obtiene historial de rendimiento vs target para nodo
- **Retorna**: `Dict[str, Any]`
- **Dependencias**:
  - `self.pbi_collector.collect_daily_nps_data()`
  - `_extract_node_level()` (local)
  - `numpy` para estadísticas

---

## Archivo: `dashboard_analyzer/anomaly_detection/anomaly_tree.py`

### Clase: `AnomalyNode`

Representa un nodo en el árbol de detección de anomalías

#### Constructor: `__init__`
- **Argumentos**:
  - `name: str`
  - `path: str`
  - `parent: AnomalyNode = None`
- **Propósito**: Inicializa nodo del árbol
- **Atributos**:
  - `children: Dict[str, 'AnomalyNode']`
  - `data: Optional[pd.DataFrame]`
  - `moving_averages: Dict[str, float]`
  - `daily_anomaly: Optional['Anomaly']`
  - `weekly_anomaly: Optional['Anomaly']`
  - `explanation: Optional[str]`
  - `insufficient_sample: bool`
  - `insufficient_sample_dates: set`

#### 1. `add_child`
- **Argumentos**: `child: 'AnomalyNode'`
- **Propósito**: Agrega nodo hijo
- **Dependencias**: Ninguna

#### 2. `get_path`
- **Argumentos**: Ninguno
- **Propósito**: Obtiene path completo del nodo
- **Retorna**: `str`
- **Dependencias**: Ninguna

#### 3. `is_leaf`
- **Argumentos**: Ninguno
- **Propósito**: Verifica si es nodo hoja (sin hijos)
- **Retorna**: `bool`
- **Dependencias**: Ninguna

#### 4. `get_all_descendants`
- **Argumentos**: Ninguno
- **Propósito**: Obtiene todos los nodos descendientes
- **Retorna**: `List['AnomalyNode']`
- **Dependencias**: Recursión propia

---

### Clase: `Anomaly`

Representa resultado de detección de anomalía

#### Constructor: `__init__`
- **Argumentos**:
  - `current_value: float`
  - `target_value: float`
  - `deviation: float`
  - `threshold: float`
  - `anomaly_type: str` - "daily" o "weekly"
- **Propósito**: Inicializa representación de anomalía
- **Atributos**:
  - `is_anomaly: bool` - Calculado como `abs(deviation) > threshold`

---

### Clase: `AnomalyTree`

Estructura de árbol para detección de anomalías a través de nodos jerárquicos

#### Constructor: `__init__`
- **Argumentos**: `data_base_path: str = "tables"`
- **Propósito**: Inicializa estructura de árbol
- **Atributos**:
  - `root: Optional[AnomalyNode]`
  - `nodes: Dict[str, AnomalyNode]`
  - `dates: List[str]`
  - `daily_anomalies: Dict[str, Dict[str, str]]` - date -> {node_path -> anomaly_state}

#### 1. `build_tree_structure`
- **Argumentos**: Ninguno
- **Propósito**: Construye estructura del árbol según jerarquía del README
- **Dependencias**: `AnomalyNode` (local)

#### 2. `load_data`
- **Argumentos**: `date_folder: str = "22_05_2025"`
- **Propósito**: Carga datos NPS para todos los nodos desde carpeta de fecha
- **Dependencias**:
  - `pd.read_csv()` de pandas
  - `Path` de pathlib

#### 3. `calculate_moving_averages`
- **Argumentos**: `window_days: int = 7`
- **Propósito**: Calcula media de últimos N días para cada nodo
- **Dependencias**: pandas

#### 4. `detect_daily_anomalies`
- **Argumentos**:
  - `threshold: float = 5.0`
  - `min_sample_size: int = 5`
- **Propósito**: Detecta anomalías en últimos 7 días comparando con su media
- **Dependencias**: pandas

#### 5. `print_collapsed_tree`
- **Argumentos**: `date: str`
- **Propósito**: Imprime vista colapsada del árbol para fecha específica
- **Dependencias**: Ninguna (formateo de strings)

#### 6. `print_all_days_summary`
- **Argumentos**: Ninguno
- **Propósito**: Imprime resumen de anomalías de últimos 7 días
- **Dependencias**: Ninguna (formateo de strings)

#### 7. `analyze_date`
- **Argumentos**: `date: str`
- **Propósito**: Analiza anomalías para fecha específica y genera explicaciones
- **Retorna**: `Dict` - Información de análisis
- **Dependencias**: `_generate_tree_explanation()` (local)

#### 8. `run_full_analysis`
- **Argumentos**:
  - `date_folder: str = "22_05_2025"`
  - `min_sample_size: int = 5`
- **Propósito**: Ejecuta análisis completo de detección de anomalías
- **Retorna**: `self`
- **Dependencias**:
  - `load_data()` (local)
  - `calculate_moving_averages()` (local)
  - `detect_daily_anomalies()` (local)
  - `print_all_days_summary()` (local)

#### 9. `_generate_tree_explanation`
- **Argumentos**:
  - `date: str`
  - `exclude_insufficient_sample: bool = True`
- **Propósito**: Genera explicación del árbol con lógica jerárquica
- **Retorna**: `str`
- **Dependencias**: Lógica compleja de análisis de patrones

---

## Archivo: `dashboard_analyzer/anomaly_detection/anomaly_interpreter.py`

### Clase: `AnomalyInterpreter`

Interpreta patrones de anomalías en el árbol usando análisis bottom-up siguiendo reglas definidas en README.md

#### Constructor: `__init__`
- **Argumentos**: `pbi_collector: PBIDataCollector = None`
- **Propósito**: Inicializa intérprete de anomalías
- **Atributos**:
  - `interpretations: Dict[str, str]` - node_path -> interpretación
  - `routes_analyzer: RoutesAnalyzer`
  - `verbatims_cache: Dict[Tuple[str, str], pd.DataFrame]`

#### 1. `analyze_node_pattern`
- **Argumentos**:
  - `parent_node: AnomalyNode`
  - `date: str`
  - `daily_anomalies: Dict[str, str]`
- **Propósito**: Analiza nodo padre basado en estados de anomalía de hijos
- **Retorna**: `str` - Interpretación siguiendo reglas README
- **Dependencias**: `_generate_interpretation()` (local)

#### 2. `_generate_interpretation`
- **Argumentos**:
  - `parent_state: str`
  - `normal_count: int`
  - `positive_count: int`
  - `negative_count: int`
  - `normal_children: List[str]`
  - `positive_children: List[str]`
  - `negative_children: List[str]`
- **Propósito**: Genera texto de interpretación basado en reglas README
- **Retorna**: `str`
- **Dependencias**: Ninguna (lógica de reglas)

#### 3. `analyze_tree_for_date`
- **Argumentos**:
  - `tree: AnomalyTree`
  - `date: str`
- **Propósito**: Realiza análisis bottom-up para todos los nodos padre en fecha específica
- **Retorna**: `Dict[str, str]` - node_path -> interpretación
- **Dependencias**: `analyze_node_pattern()` (local)

#### 4. `print_interpreted_tree`
- **Argumentos**:
  - `tree: AnomalyTree`
  - `date: str`
  - `interpretations: Dict[str, str] = None`
- **Propósito**: Imprime árbol con interpretaciones, valores NPS y flags de explicación
- **Dependencias**: Ninguna (formateo complejo de output)

#### 5. `analyze_week_interpretations`
- **Argumentos**:
  - `tree: AnomalyTree`
  - `dates: List[str]`
- **Propósito**: Analiza interpretaciones para múltiples fechas
- **Retorna**: `Dict[str, Dict[str, str]]` - date -> {node_path -> interpretation}
- **Dependencias**: `analyze_tree_for_date()` (local)

#### 6. `_needs_explanation`
- **Argumentos**:
  - `node_path: str`
  - `anomaly_state: str`
  - `parent_node: AnomalyNode`
  - `daily_anomalies: Dict[str, str]`
- **Propósito**: Determina si nodo necesita explicación según reglas README
- **Retorna**: `bool`
- **Dependencias**: Ninguna (lógica de reglas)

#### 7. `print_propagation_analysis`
- **Argumentos**:
  - `tree: AnomalyTree`
  - `date: str`
  - `interpretations: Dict[str, str] = None`
- **Propósito**: Imprime análisis de propagación (formato original sin valores NPS)
- **Dependencias**: `analyze_tree_for_date()` (local)

#### 8. `print_tree_with_operational_explanations`
- **Argumentos**:
  - `tree: AnomalyTree`
  - `date: str`
  - `operational_analyzer: OperationalDataAnalyzer`
  - `interpretations: Dict[str, str] = None`
  - `routes_data: Dict[str, List[Dict]] = None`
- **Propósito**: Imprime árbol con formato mejorado y explicaciones claras
- **Dependencias**:
  - `operational_analyzer.get_specific_explanations()`
  - `self.routes_analyzer.format_routes_explanation()`
  - `analyze_verbatims_sentiment_by_topic()` (local)

#### 9. `analyze_verbatims_sentiment_by_topic`
- **Argumentos**:
  - `verbatims_df: pd.DataFrame`
  - `anomaly_type: str`
- **Propósito**: Analiza sentimiento de verbatims por tópico
- **Retorna**: `str`
- **Dependencias**: pandas

#### 10. `collect_verbatims_for_explanation_needed`
- **Argumentos**:
  - `tree: AnomalyTree`
  - `date: str`
  - `output_dir: Path = None`
- **Propósito**: Recolecta verbatims para todos los nodos con flag "explanation needed"
- **Retorna**: `Dict[str, pd.DataFrame]` - node_path -> verbatims DataFrame
- **Dependencias**:
  - `self.pbi_collector.collect_verbatims_for_date_and_segment()`
  - `_node_needs_verbatims_explanation()` (local)

#### 11. `_node_needs_verbatims_explanation`
- **Argumentos**:
  - `node_path: str`
  - `node_state: str`
  - `daily_anomalies: Dict[str, str]`
- **Propósito**: Check simplificado para recolección de verbatims
- **Retorna**: `bool`
- **Dependencias**: Ninguna

#### 12. `collect_routes_for_explanation_needed` (async)
- **Argumentos**:
  - `tree: AnomalyTree`
  - `date: str`
- **Propósito**: Recolecta datos de rutas para nodos con flag "explanation needed"
- **Retorna**: `Dict[str, List[Dict]]` - node_path -> list of route info
- **Dependencias**:
  - `self.routes_analyzer.load_routes_data()` (async)
  - `self.routes_analyzer.get_most_affected_routes()`

---

## Resumen del Módulo `anomaly_detection`

### Estadísticas Generales
- **Total de Archivos**: 6
- **Total de Clases**: 6 principales
  - `FlexibleAnomalyDetector` (18 métodos)
  - `FlexibleAnomalyInterpreter` (13 métodos)
  - `TargetBasedDetector` (11 métodos - legacy)
  - `AnomalyTree` (9 métodos)
  - `AnomalyNode` (4 métodos)
  - `Anomaly` (clase de datos)
  - `AnomalyInterpreter` (12 métodos)

- **Total de Funciones/Métodos**: ~70
- **Líneas de Código**: ~3,750
- **Funciones Async**: 8

### Dependencias Principales del Módulo
1. **Internas del Proyecto**:
   - `dashboard_analyzer.data_collection.pbi_collector.PBIDataCollector`
   - `dashboard_analyzer.anomaly_explanation.data_analyzer.OperationalDataAnalyzer`
   - `dashboard_analyzer.anomaly_explanation.routes_analyzer.RoutesAnalyzer`
   - `dashboard_analyzer.anomaly_explanation.genai_core.agents.causal_explanation_agent.CausalExplanationAgent`

2. **Bibliotecas Externas**:
   - `pandas` - Manipulación de datos
   - `numpy` - Cálculos numéricos
   - `pathlib` - Manejo de rutas
   - `typing` - Anotaciones de tipos
   - `datetime` - Manejo de fechas
   - `logging` - Logging
   - `asyncio` - Operaciones asíncronas

### Flujos de Trabajo Principales

#### Flujo 1: Detección Flexible de Anomalías
```
FlexibleAnomalyDetector.analyze_period()
  → _load_flexible_data()
  → _get_available_periods()
  → [Según detection_mode]:
      • _detect_target_based_anomalies() [mode="target"]
      • _detect_legacy_anomalies() [mode="mean"]
      • _detect_vslast_anomalies() [mode="vslast"]
      • _detect_vslast_dynamic_anomalies() [mode="vslast_dynamic"]
  → _classify_anomaly_new_logic()
```

#### Flujo 2: Interpretación de Anomalías (Modo Agent)
```
FlexibleAnomalyInterpreter.explain_anomaly()
  → _initialize_causal_agent()
  → _get_period_date_range()
  → causal_agent.investigate_anomaly()
  → [Retorna análisis causal inteligente]
```

#### Flujo 3: Interpretación de Anomalías (Modo Raw)
```
FlexibleAnomalyInterpreter.explain_anomaly()
  → _get_period_date_range()
  → _analyze_operational_data()
  → _analyze_verbatims_data()
  → _analyze_routes_data()
  → _analyze_explanatory_drivers_data()
  → _combine_explanations()
```

#### Flujo 4: Análisis de Árbol Legacy
```
AnomalyTree.run_full_analysis()
  → load_data()
  → calculate_moving_averages()
  → detect_daily_anomalies()
  → print_all_days_summary()

AnomalyInterpreter.analyze_tree_for_date()
  → analyze_node_pattern() [para cada nodo]
  → _generate_interpretation()
```

---

# Módulo: Recolección de Datos (`dashboard_analyzer/data_collection/`)

## Archivo: `dashboard_analyzer/data_collection/__init__.py`

### Exportaciones del Módulo
- **Clase Exportada**:
  - `PBIDataCollector` - Recolector principal de datos desde Power BI

---

## Archivo: `dashboard_analyzer/data_collection/pbi_collector.py`

### Clase: `PBIDataCollector`
Recolecta datos desde Power BI API para cada nodo en la jerarquía del árbol NPS.

#### Métodos Públicos (36 métodos documentados):

#### 1. `__init__`
- **Argumentos**: Ninguno
- **Propósito**: Inicializa el colector con credenciales de Power BI desde variables de entorno
- **Dependencias**: 
  - `load_dotenv` (dotenv)
  - `_get_access_token()` (método interno)
- **Variables Globales**: CLIENT_ID, CLIENT_SECRET, TENANT_ID, GROUP_ID, DATASET_ID

#### 2. `_load_query_template`
- **Argumentos**: `query_file: str`
- **Propósito**: Carga una plantilla de consulta DAX desde archivo
- **Dependencias**: `Path` (pathlib)
- **Ubicación Consultas**: `dashboard_analyzer/data_collection/queries/`

#### 3. `_get_access_token`
- **Argumentos**: Ninguno
- **Propósito**: Obtiene token de acceso para Power BI API usando credenciales
- **Dependencias**: `msal.ConfidentialClientApplication`
- **Returns**: Token de acceso como string

#### 4. `_get_daily_nps_query`
- **Argumentos**: `cabins: List[str], companies: List[str], hauls: List[str]`
- **Propósito**: Genera consulta DAX para datos NPS diarios usando plantilla
- **Dependencias**: 
  - `_load_query_template()` [pbi_collector.py]
  - Plantilla: `queries/Daily NPS.txt`

#### 5. `_get_operative_query`
- **Argumentos**: `cabins: List[str], companies: List[str], hauls: List[str], target_date: datetime = None, comparison_days: int = 7`
- **Propósito**: Genera consulta DAX para datos operativos usando plantilla
- **Dependencias**: 
  - `_load_query_template()` [pbi_collector.py]
  - Plantilla: `queries/Operativa.txt`

#### 6. `_get_nps_vs_sel_period_query`
- **Argumentos**: `cabins, companies, hauls, current_start_date, current_end_date, comparison_start_date, comparison_end_date`
- **Propósito**: Genera consulta DAX para comparación NPS entre dos períodos específicos
- **Dependencias**: 
  - `_load_query_template()` [pbi_collector.py]
  - Plantilla: `queries/NPS_vs_sel_period.txt`

#### 7. `_get_flexible_nps_query`
- **Argumentos**: `aggregation_days: int, cabins, companies, hauls, analysis_date: datetime = None`
- **Propósito**: Genera consulta DAX para agregación NPS flexible (1, 7, 14, 30 días)
- **Dependencias**: 
  - `_load_query_template()` [pbi_collector.py]
  - Plantilla: `queries/NPS_flex_agg.txt`

#### 8. `_get_operative_vs_sel_period_query`
- **Argumentos**: `cabins, companies, hauls, current_start_date, current_end_date, comparison_start_date, comparison_end_date`
- **Propósito**: Genera consulta DAX simplificada para comparación operativa entre períodos
- **Dependencias**: 
  - `_load_query_template()` [pbi_collector.py]
  - Plantilla: `queries/Operativa_vs_sel_period.txt`

#### 9. `_get_flexible_operative_query`
- **Argumentos**: `aggregation_days, cabins, companies, hauls, analysis_date = None, comparison_start_date = None`
- **Propósito**: Genera consulta DAX para agregación operativa flexible (coincide exactamente con lógica NPS)
- **Dependencias**: 
  - `_load_query_template()` [pbi_collector.py]
  - Plantilla: `queries/Operativa_flex_agg.txt`

#### 10. `_get_verbatims_query`
- **Argumentos**: `cabins, companies, hauls, date: datetime`
- **Propósito**: Genera consulta DAX para datos de verbatims de una fecha específica
- **Dependencias**: 
  - `_load_query_template()` [pbi_collector.py]
  - Plantilla: `queries/Verbatims.txt`

#### 11. `_get_verbatims_range_query`
- **Argumentos**: `cabins, companies, hauls, start_date: datetime, end_date: datetime`
- **Propósito**: Genera consulta DAX para verbatims en un rango de fechas
- **Dependencias**: 
  - `_load_query_template()` [pbi_collector.py]
  - Plantilla: `queries/Verbatims.txt`

#### 12. `_execute_query`
- **Argumentos**: `query: str`
- **Propósito**: Ejecuta una consulta DAX contra Power BI API (síncrono)
- **Dependencias**: `requests`, `pandas`
- **Returns**: DataFrame con resultados

#### 13. `_execute_query_async`
- **Argumentos**: `query: str`
- **Propósito**: Ejecuta una consulta DAX contra Power BI API (asíncrono)
- **Dependencias**: `aiohttp`, `pandas`
- **Returns**: DataFrame con resultados

#### 14. `_get_node_filters`
- **Argumentos**: `node_path: str`
- **Propósito**: Obtiene valores de filtro (cabins, companies, hauls) basados en ruta del nodo
- **Dependencias**: Ninguna
- **Returns**: Tuple[List[str], List[str], List[str]]

#### 15. `collect_node_data`
- **Argumentos**: `node_path: str, output_dir: Path`
- **Propósito**: Recolecta todos los tipos de datos para un nodo específico
- **Dependencias**: 
  - `_get_node_filters()` [pbi_collector.py]
  - `_get_daily_nps_query()` [pbi_collector.py]
  - `_get_flexible_operative_query()` [pbi_collector.py]
  - `_execute_query()` [pbi_collector.py]

#### 16. `collect_verbatims_for_date_and_segment`
- **Argumentos**: `node_path: str, date: datetime, output_dir: Path = None`
- **Propósito**: Recolecta verbatims para una fecha y segmento específicos
- **Dependencias**: 
  - `_get_node_filters()` [pbi_collector.py]
  - `_get_verbatims_query()` [pbi_collector.py]
  - `_execute_query()` [pbi_collector.py]

#### 17. `collect_verbatims_for_date_range`
- **Argumentos**: `node_path, start_date, end_date, output_dir = None`
- **Propósito**: Recolecta verbatims para un rango de fechas (más eficiente que recolección diaria)
- **Dependencias**: 
  - `_get_node_filters()` [pbi_collector.py]
  - `_get_verbatims_range_query()` [pbi_collector.py]
  - `_execute_query()` [pbi_collector.py]

#### 18. `collect_all_nodes_data`
- **Argumentos**: `output_base_dir: str = "tables"`
- **Propósito**: Recolecta datos para todos los nodos en la estructura del árbol
- **Dependencias**: `collect_node_data()` [pbi_collector.py]
- **Returns**: Dict con resultados por nodo

#### 19. `collect_all_data` (async)
- **Argumentos**: Ninguno
- **Propósito**: Envolvente asíncrona para collect_all_nodes_data, llamada por main.py
- **Dependencias**: `collect_all_nodes_data()` [pbi_collector.py]
- **Returns**: Tuple[int, int] (éxitos, total)

#### 20. `collect_flexible_data_for_node` (async)
- **Argumentos**: `node_path, aggregation_days: int, target_folder, analysis_date = None`
- **Propósito**: Recolecta datos agregados flexibles para un nodo específico
- **Dependencias**: 
  - `_parse_node_path()` [pbi_collector.py]
  - `_get_flexible_nps_query()` [pbi_collector.py]
  - `_get_flexible_operative_query()` [pbi_collector.py]
  - `_execute_query_async()` [pbi_collector.py]
  - `_safe_clean_columns()` [pbi_collector.py]

#### 21. `_clean_routes_dictionary_columns`
- **Argumentos**: `df: pd.DataFrame`
- **Propósito**: Limpia nombres de columnas específicamente para diccionario de rutas
- **Dependencias**: Ninguna
- **Transformación**: `'Route_Master[column_name]' → 'column_name'`

#### 22. `_safe_clean_columns`
- **Argumentos**: `df: pd.DataFrame`
- **Propósito**: Limpia nombres de columnas de forma segura eliminando corchetes
- **Dependencias**: Ninguna

#### 23. `_parse_node_path`
- **Argumentos**: `node_path: str`
- **Propósito**: Parsea ruta del nodo para extraer cabins, companies y hauls
- **Dependencias**: Ninguna
- **Returns**: Tuple[List[str], List[str], List[str]]

#### 24. `collect_operative_data_for_date` (async)
- **Argumentos**: `node_path, target_date, comparison_days = 7, use_flexible = True`
- **Propósito**: Recolecta datos operacionales para una fecha específica y días previos
- **Dependencias**: 
  - `_get_node_filters()` [pbi_collector.py]
  - `_get_flexible_operative_query()` [pbi_collector.py]
  - `_get_operative_query()` [pbi_collector.py]
  - `_execute_query()` [pbi_collector.py]

#### 25. `collect_explanatory_drivers_for_date_range` (async)
- **Argumentos**: `node_path, start_date, end_date, comparison_filter = "vs L7d", comparison_start_date = None, comparison_end_date = None`
- **Propósito**: Recolecta datos de drivers explicativos para un nodo y rango de fechas
- **Dependencias**: 
  - `_get_node_filters()` [pbi_collector.py]
  - `_get_explanatory_drivers_range_query()` [pbi_collector.py]
  - `_execute_query()` [pbi_collector.py]
  - `_safe_clean_columns()` [pbi_collector.py]

#### 26. `collect_routes_for_date_range` (async)
- **Argumentos**: `node_path, start_date, end_date, comparison_filter = "vs L7d", comparison_start_date = None, comparison_end_date = None`
- **Propósito**: Recolecta datos de rutas para un nodo y rango de fechas
- **Dependencias**: 
  - `_get_node_filters()` [pbi_collector.py]
  - `_get_routes_range_query()` [pbi_collector.py]
  - `_execute_query()` [pbi_collector.py]
  - `_safe_clean_columns()` [pbi_collector.py]

#### 27. `_get_explanatory_drivers_range_query`
- **Argumentos**: `start_date, end_date, comparison_filter = "vs L7d", comparison_start_date = None, comparison_end_date = None, cabins = None, companies = None, hauls = None`
- **Propósito**: Construye consulta DAX de drivers explicativos para rango de fechas
- **Dependencias**: Plantilla: `queries/Exp. Drivers.txt`

#### 28. `_get_routes_range_query`
- **Argumentos**: `cabins, companies, hauls, start_date, end_date, comparison_filter = "vs L7d", comparison_start_date = None, comparison_end_date = None`
- **Propósito**: Genera consulta DAX para datos de rutas usando rango de fechas
- **Dependencias**: 
  - `_load_query_template()` [pbi_collector.py]
  - Plantilla: `queries/Rutas.txt`

#### 29. `_get_customer_profile_range_query`
- **Argumentos**: `cabins, companies, hauls, start_date, end_date, profile_dimension = "Channel", route_filter = None, comparison_filter = "vs L7d", comparison_start_date = None, comparison_end_date = None`
- **Propósito**: Genera consulta DAX para datos de perfil de cliente con filtro opcional de ruta
- **Dependencias**: 
  - `_load_query_template()` [pbi_collector.py]
  - Plantilla: `queries/Customer Profile.txt`

#### 30. `collect_routes_dictionary` (async)
- **Argumentos**: Ninguno
- **Propósito**: Recolecta el diccionario simple de rutas para filtrado NCS
- **Dependencias**: 
  - `_load_query_template()` [pbi_collector.py]
  - `_execute_query_async()` [pbi_collector.py]
  - `_clean_routes_dictionary_columns()` [pbi_collector.py]
  - Plantilla: `queries/Rutas Diccionario.txt`
- **Returns**: DataFrame con columnas: route, country_name, gr_region, haul_aggr

#### 31. `collect_customer_profile_for_date_range` (async)
- **Argumentos**: `node_path, start_date, end_date, profile_dimension = "Channel", comparison_filter = "vs L7d", comparison_start_date = None, comparison_end_date = None, route_filter = None`
- **Propósito**: Recolecta datos de perfil de cliente para un rango de fechas y dimensión
- **Dependencias**: 
  - `_get_node_filters()` [pbi_collector.py]
  - `_get_customer_profile_range_query()` [pbi_collector.py]
  - `_execute_query_async()` [pbi_collector.py]

### Estructura del Árbol Jerárquico:
```python
tree_structure = {
    'Global': {
        'LH': {
            'Economy': ['IB', 'YW'],
            'Business': ['IB', 'YW'], 
            'Premium': ['IB', 'YW']
        },
        'SH': {
            'Economy': ['IB', 'YW'],
            'Business': ['IB', 'YW']
        }
    }
}
```

### Plantillas de Consulta DAX:
1. `Daily NPS.txt` - NPS diario
2. `Operativa.txt` - Datos operativos
3. `NPS_vs_sel_period.txt` - Comparación NPS entre períodos
4. `NPS_flex_agg.txt` - Agregación NPS flexible
5. `Operativa_vs_sel_period.txt` - Comparación operativa
6. `Operativa_flex_agg.txt` - Agregación operativa flexible
7. `Verbatims.txt` - Comentarios de clientes
8. `Exp. Drivers.txt` - Drivers explicativos
9. `Rutas.txt` - Datos de rutas
10. `Rutas Diccionario.txt` - Diccionario de rutas
11. `Customer Profile.txt` - Perfil de clientes

---

## Archivo: `dashboard_analyzer/data_collection/s3_report_uploader.py`

### Clase: `S3ReportUploader`
Maneja la carga de reportes de análisis comprehensivos a bucket S3.

#### Características:
- Integración AWS S3 con manejo adecuado de errores
- Generación de reportes JSON con metadata
- Nombrado automático de archivos con timestamps y rangos de fechas
- Gestión de credenciales (variables de entorno o archivos temporales)
- Logging comprehensivo

#### Métodos Públicos (9 métodos documentados):

#### 1. `__init__`
- **Argumentos**: `temp_env_file: str = None, environment: str = "local"`
- **Propósito**: Inicializa S3 Report Uploader
- **Dependencias**: 
  - `boto3.client`
  - `_setup_aws_credentials()` [s3_report_uploader.py]
- **Configuración S3**:
  - Bucket: `ibdata-sbx-ew1-s3-customer`
  - Prefix: `customer/catia/reports/raw/`

#### 2. `_setup_aws_credentials`
- **Argumentos**: Ninguno
- **Propósito**: Configura credenciales AWS según el entorno
- **Dependencias**: 
  - `boto3.client` (prod: IAM roles)
  - `boto3.Session` (local: archivo de credenciales)

#### 3. `_generate_filename`
- **Argumentos**: `execution_date: datetime, analysis_date: str, comparison_start_date: Optional[str] = None, comparison_end_date: Optional[str] = None`
- **Propósito**: Genera nombre de archivo S3 basado en fecha de ejecución y rango de análisis
- **Dependencias**: Ninguna
- **Returns**: String con formato `YYYY-MM-DDTHH-MM-SS_date_range.json`

#### 4. `_build_report_json`
- **Argumentos**: `execution_metadata: Dict, weekly_analysis_params: Dict, daily_analysis_params: Dict, date_ranges: Dict, final_synthesis: str`
- **Propósito**: Construye la estructura JSON completa del reporte
- **Dependencias**: Ninguna
- **Returns**: Dict con reporte completo

#### 5. `upload_comprehensive_report` (async)
- **Argumentos**: `execution_date, analysis_date, segment, explanation_mode, causal_filter, weekly_analysis_params, daily_analysis_params, date_ranges, final_synthesis, comparison_start_date = None, comparison_end_date = None`
- **Propósito**: Carga reporte de análisis comprehensivo a S3
- **Dependencias**: 
  - `_build_report_json()` [s3_report_uploader.py]
  - `_generate_filename()` [s3_report_uploader.py]
  - `s3_client.put_object()` (boto3)
- **Returns**: Clave S3 del archivo cargado si exitoso, None si falla

#### 6. `test_connection`
- **Argumentos**: Ninguno
- **Propósito**: Prueba conexión S3 y permisos
- **Dependencias**: `s3_client.list_objects_v2()` (boto3)
- **Returns**: Bool

#### 7. `upload_agent_conversation` (async)
- **Argumentos**: `conversation_data: Dict, agent_type: str, filename: str`
- **Propósito**: Carga conversación de agente a S3
- **Dependencias**: `s3_client.put_object()` (boto3)
- **Tipos de Agente**: "causal_explanation", "anomaly_interpreter", "anomaly_summary"
- **Returns**: Clave S3 si exitoso, None si falla

#### 8. `upload_causal_conversation` (async)
- **Argumentos**: `conversation_data: Dict, filename: str`
- **Propósito**: Carga conversación de agente de explicación causal a S3
- **Dependencias**: `upload_agent_conversation()` [s3_report_uploader.py]

#### 9. `upload_interpreter_conversation` (async)
- **Argumentos**: `conversation_data: Dict, filename: str`
- **Propósito**: Carga conversación de agente intérprete de anomalías a S3
- **Dependencias**: `upload_agent_conversation()` [s3_report_uploader.py]

#### 10. `upload_summary_conversation` (async)
- **Argumentos**: `conversation_data: Dict, filename: str`
- **Propósito**: Carga conversación de agente de resumen de anomalías a S3
- **Dependencias**: `upload_agent_conversation()` [s3_report_uploader.py]

---

## Archivo: `dashboard_analyzer/data_collection/ncs_collector.py`

### Clase: `NCSDataCollector`
Recolecta datos de Net Customer Satisfaction desde bucket AWS S3.

#### Métodos Públicos (24 métodos documentados):

#### 1. `__init__`
- **Argumentos**: `temp_env_file: str = None, environment: str = "local"`
- **Propósito**: Inicializa NCS Data Collector
- **Dependencias**: 
  - `boto3.client`
  - `_setup_aws_credentials()` [ncs_collector.py]
- **Configuración S3**:
  - Bucket: `ibdata-prod-ew1-s3-customer`
  - Prefix: `customer/catia/ncs/raw/attatchments/`

#### 2. `_setup_aws_credentials`
- **Argumentos**: Ninguno
- **Propósito**: Configura credenciales AWS según el entorno
- **Dependencias**: `boto3.client`, `boto3.Session`

#### 3. `list_available_files`
- **Argumentos**: `date_prefix: str = None`
- **Propósito**: Lista archivos NCS disponibles en bucket S3
- **Dependencias**: `s3_client.list_objects_v2()` (boto3)
- **Returns**: List[str] con claves de archivo

#### 4. `read_ncs_file`
- **Argumentos**: `file_key: str`
- **Propósito**: Lee un archivo NCS específico desde S3 (formato HTML de email)
- **Dependencias**: 
  - `s3_client.get_object()` (boto3)
  - `_parse_html_email_content()` [ncs_collector.py]
- **Returns**: DataFrame con datos NCS extraídos

#### 5. `_parse_html_email_content`
- **Argumentos**: `content: str, file_key: str`
- **Propósito**: Parsea contenido HTML de email para extraer datos de incidentes NCS
- **Dependencias**: 
  - `BeautifulSoup` (bs4)
  - `_extract_email_metadata()` [ncs_collector.py]
  - `_extract_table_data()` [ncs_collector.py]
  - `_extract_text_based_data()` [ncs_collector.py]

#### 6. `_extract_email_metadata`
- **Argumentos**: `content: str`
- **Propósito**: Extrae metadata de email desde contenido HTML
- **Dependencias**: `re.search`
- **Extrae**: subject, date, sender

#### 7. `_extract_table_data`
- **Argumentos**: `table` (BeautifulSoup table element)
- **Propósito**: Extrae datos desde tabla HTML
- **Dependencias**: BeautifulSoup
- **Returns**: List[Dict] con datos de filas

#### 8. `_extract_text_based_data`
- **Argumentos**: `text_content: str`
- **Propósito**: Extrae datos de incidentes desde contenido de texto plano usando patrones
- **Dependencias**: `re.search`, `re.findall`
- **Patrones**: vuelos (IB\d+), rutas (XXX-YYY), tiempos (HH:MM)

#### 9. `read_ncs_file_by_path`
- **Argumentos**: `s3_path: str`
- **Propósito**: Lee archivo NCS por ruta S3 completa (formato HTML de email)
- **Dependencias**: 
  - `s3_client.get_object()` (boto3)
  - `_parse_html_email_content()` [ncs_collector.py]

#### 10. `collect_ncs_data_for_date_range`
- **Argumentos**: `start_date: datetime, end_date: datetime`
- **Propósito**: Recolecta datos NCS para un rango de fechas específico
- **Dependencias**: 
  - `list_available_files()` [ncs_collector.py]
  - `read_ncs_file()` [ncs_collector.py]
- **Returns**: DataFrame combinado con datos NCS

#### 11. `collect_ncs_data_for_period`
- **Argumentos**: `analysis_date: datetime, target_period: int, aggregation_days: int`
- **Propósito**: Recolecta datos NCS para un período específico en el sistema de agregación flexible
- **Dependencias**: 
  - `_calculate_period_date_range()` [ncs_collector.py]
  - `collect_ncs_data_for_date_range()` [ncs_collector.py]
- **Ejemplos**:
  - Análisis diario (aggregation_days=1): Período 1 = análisis_fecha
  - Análisis semanal (aggregation_days=7): Período 1 = 7 días finalizando en análisis_fecha

#### 12. `collect_ncs_data_for_multiple_periods`
- **Argumentos**: `analysis_date: datetime, periods: List[int], aggregation_days: int`
- **Propósito**: Recolecta datos NCS para múltiples períodos
- **Dependencias**: `collect_ncs_data_for_period()` [ncs_collector.py]
- **Returns**: Dict[int, pd.DataFrame] mapeando número de período a DataFrame

#### 13. `_calculate_period_date_range`
- **Argumentos**: `analysis_date: datetime, target_period: int, aggregation_days: int`
- **Propósito**: Calcula el rango de fechas correcto para un período relativo a la fecha de análisis
- **Dependencias**: `timedelta` (datetime)
- **Returns**: Tuple[datetime, datetime] (start_date, end_date)

#### 14. `get_latest_ncs_file`
- **Argumentos**: Ninguno
- **Propósito**: Obtiene el archivo NCS más reciente
- **Dependencias**: `list_available_files()` [ncs_collector.py]
- **Returns**: Clave del archivo NCS más reciente o None

#### 15. `save_ncs_data`
- **Argumentos**: `df: pd.DataFrame, output_path: str`
- **Propósito**: Guarda datos NCS a archivo local
- **Dependencias**: `pandas.to_csv`, `pandas.to_parquet`
- **Formatos**: CSV, Parquet

#### 16. `analyze_ncs_incidents_for_period`
- **Argumentos**: `df: pd.DataFrame, analysis_focus: str = "all"`
- **Propósito**: Analiza incidentes NCS para proporcionar insights operacionales
- **Dependencias**: 
  - `_extract_incident_counts_from_data()` [ncs_collector.py]
  - `_extract_incident_themes()` [ncs_collector.py]
  - `_extract_route_analysis_from_data()` [ncs_collector.py]
  - `_generate_summary_insights()` [ncs_collector.py]
- **Focos de Análisis**: "flights", "routes", "incidents", "all"
- **Returns**: Dict con resultados de análisis incluyendo estructura incident_counts

#### 17. `_extract_incident_themes`
- **Argumentos**: `incident_texts: List[str]`
- **Propósito**: Extrae temas comunes desde descripciones de incidentes
- **Dependencias**: Ninguna
- **Temas**: technical_issues, weather, bird_strike, delays, cancellations, aircraft_change, baggage, crew, passenger

#### 18. `_generate_summary_insights`
- **Argumentos**: `analysis: Dict`
- **Propósito**: Genera insights legibles para humanos desde el análisis
- **Dependencias**: Ninguna
- **Returns**: List[str] con insights

#### 19. `_extract_incident_counts_from_data`
- **Argumentos**: `df: pd.DataFrame`
- **Propósito**: Extrae conteos de incidentes por tipo desde datos NCS
- **Dependencias**: Ninguna
- **Tipos**: cancelaciones, retrasos, desvios, limitacion_aeronave, equipaje, otras_incidencias
- **Returns**: Dict[str, int] {tipo_incidente: conteo}

#### 20. `_extract_route_analysis_from_data`
- **Argumentos**: `df: pd.DataFrame`
- **Propósito**: Extrae análisis de rutas optimizado para expectativas del agente causal
- **Dependencias**: `re.findall`
- **Returns**: Dict con rutas afectadas y conteos

#### 21. `create_temporal_comparison_analysis`
- **Argumentos**: `current_data: pd.DataFrame, comparison_data: pd.DataFrame, current_period: str, comparison_period: str, node_path: str = None`
- **Propósito**: Crea análisis de comparación temporal entre dos períodos para agente causal
- **Dependencias**: 
  - `_extract_structured_data_for_comparison()` [ncs_collector.py]
  - `_create_route_incident_matrix()` [ncs_collector.py]
  - `_calculate_incident_type_deltas()` [ncs_collector.py]
  - `_identify_improvement_patterns()` [ncs_collector.py]
  - `_generate_temporal_summary()` [ncs_collector.py]
- **Returns**: Dict con análisis temporal

#### 22. `_extract_structured_data_for_comparison`
- **Argumentos**: `data: pd.DataFrame, period_label: str`
- **Propósito**: Extrae datos NCS estructurados optimizados para comparación temporal
- **Dependencias**: 
  - `_extract_incident_counts_from_data()` [ncs_collector.py]
  - `_extract_route_analysis_from_data()` [ncs_collector.py]

#### 23. `_create_route_incident_matrix`
- **Argumentos**: `current_structured: Dict, comparison_structured: Dict`
- **Propósito**: Crea matriz ruta × tipo de incidente con deltas
- **Dependencias**: Ninguna
- **Returns**: Dict[str, Dict] con datos de matriz

#### 24. `_calculate_incident_type_deltas`
- **Argumentos**: `current_structured: Dict, comparison_structured: Dict`
- **Propósito**: Calcula deltas por tipo de incidente entre períodos
- **Dependencias**: Ninguna
- **Returns**: Dict[str, Dict] con deltas y porcentajes

#### 25. `_identify_improvement_patterns`
- **Argumentos**: `route_matrix: Dict, incident_deltas: Dict`
- **Propósito**: Identifica patrones de mejora o deterioro
- **Dependencias**: Ninguna
- **Returns**: Dict con tipos de incidentes mejorando/empeorando, rutas nuevas/recuperadas

#### 26. `_generate_temporal_summary`
- **Argumentos**: `route_matrix: Dict, incident_deltas: Dict, patterns: Dict`
- **Propósito**: Genera resumen para comparación temporal
- **Dependencias**: Ninguna
- **Returns**: String con resumen

---

## Archivo: `dashboard_analyzer/data_collection/chatbot_verbatims_collector.py`

### Clase: `ChatbotVerbatimsCollector`
Recopilador y analizador de verbatims de chatbot y otras fuentes de feedback.

#### Características:
- Procesamiento de comentarios de texto libre
- Análisis de sentimiento
- Categorización temática
- Extracción de menciones de rutas
- Integración con API de chatbot (JWT token)
- Fallback a Power BI collector

#### Métodos Públicos (32 métodos documentados):

#### 1. `__init__`
- **Argumentos**: `pbi_collector = None, token: str = None`
- **Propósito**: Inicializa el Chatbot Verbatims Collector
- **Dependencias**: 
  - `_load_token_from_file()` [chatbot_verbatims_collector.py]
  - `_validate_token()` [chatbot_verbatims_collector.py]

#### 2. `_load_token_from_file`
- **Argumentos**: Ninguno
- **Propósito**: Carga token desde archivo temp_aws_credentials.env
- **Dependencias**: Ninguna
- **Archivo**: `dashboard_analyzer/temp_aws_credentials.env`

#### 3. `_validate_token`
- **Argumentos**: Ninguno
- **Propósito**: Valida el token JWT actual y verifica si ha expirado
- **Dependencias**: `jwt.decode`
- **Returns**: Bool

#### 4. `ensure_valid_token`
- **Argumentos**: Ninguno
- **Propósito**: Método público para asegurar que el token es válido antes de operaciones
- **Dependencias**: `_validate_token()` [chatbot_verbatims_collector.py]
- **Returns**: Bool

#### 5. `get_token_status`
- **Argumentos**: Ninguno
- **Propósito**: Retorna información del estado actual del token
- **Dependencias**: `jwt.decode`
- **Returns**: Dict con status, message, expires_in, expired, expires_at

#### 6. `_ask_chatbot_question`
- **Argumentos**: `question: str, date_range: Tuple[str, str], node_path: str, filters: Optional[Dict] = None`
- **Propósito**: Hace una pregunta a la API del chatbot y obtiene un jobId para procesamiento asíncrono
- **Dependencias**: 
  - `ensure_valid_token()` [chatbot_verbatims_collector.py]
  - `requests.post`
- **Endpoint**: `https://b8fktdca38.execute-api.eu-west-1.amazonaws.com/api/transformation/api/nps-chatbot/question`

#### 7. `_get_chatbot_answer`
- **Argumentos**: `job_id: str`
- **Propósito**: Obtiene la respuesta para una pregunta del chatbot usando el jobId
- **Dependencias**: 
  - `ensure_valid_token()` [chatbot_verbatims_collector.py]
  - `requests.get`
- **Returns**: Dict con datos de respuesta si disponible

#### 8. `_collect_from_chatbot_api`
- **Argumentos**: `date_range: Tuple[str, str], node_path: str, filters: Optional[Dict] = None`
- **Propósito**: Recolecta verbatims desde la API del chatbot usando autenticación JWT token
- **Dependencias**: 
  - `ensure_valid_token()` [chatbot_verbatims_collector.py]
  - `requests.get`
- **Returns**: DataFrame con datos de verbatims

#### 9. `_ask_chatbot_question_with_filters`
- **Argumentos**: `question: str, date_range, node_path, filters = None`
- **Propósito**: Hace una pregunta a la API del chatbot con filtros apropiados y obtiene la respuesta completa
- **Dependencias**: 
  - `ensure_valid_token()` [chatbot_verbatims_collector.py]
  - `_wait_for_chatbot_answer()` [chatbot_verbatims_collector.py]
  - `requests.post`

#### 10. `_wait_for_chatbot_answer`
- **Argumentos**: `job_id: str, headers: Dict, max_wait_time: int = 300`
- **Propósito**: Espera respuesta del chatbot con polling
- **Dependencias**: `requests.get`, `time.sleep`
- **Returns**: Dict con datos de respuesta

#### 11. `_wait_for_chatbot_answer_working`
- **Argumentos**: `job_id: str, headers: Dict, max_wait_time: int = 120`
- **Propósito**: Espera respuesta del chatbot con el método de polling que funciona
- **Dependencias**: `requests.get`, `time.sleep`
- **Delays**: 3, 5, 7, 8s (más cortos para respuesta rápida)

#### 12. `collect_verbatims_for_period`
- **Argumentos**: `date_range: Tuple[str, str], node_path: str, filters: Optional[Dict] = None`
- **Propósito**: Recopila verbatims para el período especificado
- **Dependencias**: 
  - `ensure_valid_token()` [chatbot_verbatims_collector.py]
  - `_collect_from_chatbot_api()` [chatbot_verbatims_collector.py]
  - `pbi_collector.collect_verbatims_for_date_range()` [pbi_collector.py]
  - `_process_verbatims()` [chatbot_verbatims_collector.py]
  - `_apply_filters()` [chatbot_verbatims_collector.py]
- **Returns**: DataFrame con verbatims procesados

#### 13. `analyze_sentiment`
- **Argumentos**: `verbatim_text: str`
- **Propósito**: Analiza el sentimiento de un verbatim individual
- **Dependencias**: Ninguna
- **Returns**: Dict con score, category, confidence, positive_words, negative_words

#### 14. `extract_routes_mentions`
- **Argumentos**: `verbatims_df: pd.DataFrame`
- **Propósito**: Extrae menciones de rutas en los verbatims
- **Dependencias**: 
  - `_normalize_route()` [chatbot_verbatims_collector.py]
  - `re.findall`
- **Patrones**: `\b[A-Z]{3}[- ]?[A-Z]{3}\b`, `\bvuelo\s+[A-Z0-9]+\b`, etc.

#### 15. `categorize_themes`
- **Argumentos**: `verbatims_df: pd.DataFrame`
- **Propósito**: Categoriza verbatims por temas
- **Dependencias**: Ninguna
- **Temas**: equipaje, puntualidad, servicio_abordo, reservas, check_in, aeropuerto
- **Returns**: DataFrame con categorías temáticas

#### 16. `filter_by_sentiment`
- **Argumentos**: `verbatims_df: pd.DataFrame, sentiment_threshold: float = 0.0, sentiment_type: str = 'all'`
- **Propósito**: Filtra verbatims por sentimiento
- **Dependencias**: Ninguna
- **Tipos**: 'positive', 'negative', 'neutral', 'all'

#### 17. `get_verbatims_summary`
- **Argumentos**: `verbatims_df: pd.DataFrame`
- **Propósito**: Genera resumen estadístico de verbatims
- **Dependencias**: Ninguna
- **Returns**: Dict con total, distribución de sentimiento, temas, rutas

#### 18. `_process_verbatims`
- **Argumentos**: `verbatims_df: pd.DataFrame`
- **Propósito**: Procesa verbatims aplicando análisis de sentimiento y limpieza
- **Dependencias**: 
  - `_get_text_column()` [chatbot_verbatims_collector.py]
  - `_clean_text()` [chatbot_verbatims_collector.py]
  - `analyze_sentiment()` [chatbot_verbatims_collector.py]
  - `categorize_themes()` [chatbot_verbatims_collector.py]
  - `extract_routes_mentions()` [chatbot_verbatims_collector.py]

#### 19. `_clean_text`
- **Argumentos**: `text: str`
- **Propósito**: Limpia y normaliza texto de verbatims
- **Dependencias**: `re.sub`
- **Transformaciones**: minúsculas, remover caracteres especiales, normalizar espacios

#### 20. `_normalize_route`
- **Argumentos**: `route_text: str`
- **Propósito**: Normaliza formato de rutas extraídas
- **Dependencias**: `re.sub`
- **Formato**: "XXXYYY" → "XXX-YYY"

#### 21. `_apply_filters`
- **Argumentos**: `verbatims_df: pd.DataFrame, filters: Dict`
- **Propósito**: Aplica filtros específicos a los verbatims
- **Dependencias**: `filter_by_sentiment()` [chatbot_verbatims_collector.py]
- **Filtros**: sentiment_type, theme, route_mentioned, min_nps

#### 22. `test_connection`
- **Argumentos**: Ninguno
- **Propósito**: Prueba conexión a la fuente de datos de verbatims
- **Dependencias**: 
  - `ensure_valid_token()` [chatbot_verbatims_collector.py]
  - `get_token_status()` [chatbot_verbatims_collector.py]
- **Returns**: Tuple[bool, str]

#### 23. `test_chatbot_connection`
- **Argumentos**: Ninguno
- **Propósito**: Prueba la conexión a la API frontend del chatbot
- **Dependencias**: 
  - `ensure_valid_token()` [chatbot_verbatims_collector.py]
  - `requests.get`
- **Returns**: Tuple[bool, str]

#### 24. `get_chatbot_status`
- **Argumentos**: Ninguno
- **Propósito**: Obtiene estado detallado de la conexión del chatbot y token
- **Dependencias**: `test_chatbot_connection()` [chatbot_verbatims_collector.py]
- **Returns**: Dict con información de estado

#### 25. `get_verbatims_data`
- **Argumentos**: `start_date: str, end_date: str, node_path: str, verbatim_type: str = None, intelligent_query: str = None`
- **Propósito**: Obtiene datos de verbatims para el período y ruta de nodo especificados
- **Dependencias**: 
  - `collect_verbatims_for_period()` [chatbot_verbatims_collector.py]
  - `_apply_intelligent_query_filter()` [chatbot_verbatims_collector.py]
- **Returns**: DataFrame con datos de verbatims

#### 26. `ask_chatbot_question`
- **Argumentos**: `question: str, start_date: str, end_date: str, node_path: str, filters: Optional[Dict] = None`
- **Propósito**: Hace una pregunta al chatbot y obtiene la respuesta usando el formato que funciona
- **Dependencias**: 
  - `ensure_valid_token()` [chatbot_verbatims_collector.py]
  - `_wait_for_chatbot_answer_working()` [chatbot_verbatims_collector.py]
  - `requests.post`
- **Returns**: Dict con datos de respuesta

#### 27. `_apply_intelligent_query_filter`
- **Argumentos**: `df: pd.DataFrame, intelligent_query: str`
- **Propósito**: Aplica filtrado de consulta inteligente a datos de verbatims
- **Dependencias**: 
  - `_clean_verbatims_columns()` [chatbot_verbatims_collector.py]
  - `_filter_routes_negative_comments()` [chatbot_verbatims_collector.py]
  - `_filter_representative_comments()` [chatbot_verbatims_collector.py]
  - `_filter_general_intelligent_query()` [chatbot_verbatims_collector.py]
- **Manejo Específico**: rutas con comentarios negativos, comentarios representativos

#### 28. `_filter_routes_negative_comments`
- **Argumentos**: `df: pd.DataFrame`
- **Propósito**: Filtra y analiza rutas con más comentarios negativos
- **Dependencias**: 
  - `_get_text_column()` [chatbot_verbatims_collector.py]
  - `_get_sentiment_column()` [chatbot_verbatims_collector.py]
  - `_enhance_route_extraction()` [chatbot_verbatims_collector.py]
- **Adaptado para**: estructura de tabla verbatims_sentiment

#### 29. `_filter_representative_comments`
- **Argumentos**: `df: pd.DataFrame`
- **Propósito**: Filtra los comentarios más representativos por ruta
- **Dependencias**: 
  - `_get_text_column()` [chatbot_verbatims_collector.py]
  - `_enhance_route_extraction()` [chatbot_verbatims_collector.py]
- **Criterios**: conteo de palabras (10-100), sentimiento claro, issues específicos, diversidad de rutas

#### 30. `_filter_general_intelligent_query`
- **Argumentos**: `df: pd.DataFrame, intelligent_query: str`
- **Propósito**: Filtrado general mejorado para otras consultas inteligentes
- **Dependencias**: `_get_text_column()` [chatbot_verbatims_collector.py]
- **Keywords**: retraso, equipaje, servicio, comida, asiento, entretenimiento, etc.

#### 31. `_get_text_column`
- **Argumentos**: `df: pd.DataFrame`
- **Propósito**: Obtiene el nombre de la columna de texto en el DataFrame
- **Dependencias**: Ninguna
- **Nombres Posibles**: 'verbatim_text', 'Verbatim_Text', 'text', 'comment', 'feedback', 'Verbatim', '[Verbatim]'

#### 32. `_clean_verbatims_columns`
- **Argumentos**: `df: pd.DataFrame`
- **Propósito**: Limpia nombres de columnas de la tabla verbatims_sentiment
- **Dependencias**: Ninguna
- **Transformación**: `'verbatims_sentiment[column_name]' → 'column_name'`

#### 33. `_get_sentiment_column`
- **Argumentos**: `df: pd.DataFrame`
- **Propósito**: Obtiene el nombre de la columna de sentimiento en el DataFrame
- **Dependencias**: Ninguna
- **Nombres Posibles**: 'verbatim_global_sentiment', 'sentiment', 'sentiment_category', 'global_sentiment'

#### 34. `_enhance_route_extraction`
- **Argumentos**: `df: pd.DataFrame`
- **Propósito**: Extracción mejorada de rutas desde texto de verbatim
- **Dependencias**: `_get_text_column()` [chatbot_verbatims_collector.py]
- **Patrones**: `MAD-BCN`, `MAD to BCN`, `MAD > BCN`, `from MAD to BCN`, `MAD/BCN`
- **Agrega**: columna 'extracted_route'

### Diccionarios de Análisis Temático:

**theme_keywords**:
- `equipaje`: maleta, equipaje, baggage, perdido, dañado, retraso equipaje, facturación
- `puntualidad`: retraso, delay, tarde, puntual, cancelado, cambio horario
- `servicio_abordo`: azafata, tripulación, comida, bebida, asiento, entretenimiento, wifi
- `reservas`: reserva, booking, web, app, cambio vuelo, cancelación, precio
- `check_in`: check-in, facturación, embarque, puerta, boarding, mostrador
- `aeropuerto`: terminal, puerta, mostrador, sala, espera, seguridad

**sentiment_positive**: excelente, perfecto, fantástico, genial, bueno, satisfecho, contento, recomiendo

**sentiment_negative**: horrible, terrible, malo, pésimo, desastroso, molesto, furioso, decepcionado

---

## 📊 Resumen del Módulo `data_collection/`

### Estadísticas:
- **Total de archivos**: 5
- **Total de clases**: 4
- **Total de métodos públicos documentados**: 101
- **Líneas de código**: ~4,401 líneas

### Distribución por archivo:
1. **pbi_collector.py**: 36 métodos (1,380 líneas)
2. **ncs_collector.py**: 26 métodos (1,021 líneas)
3. **chatbot_verbatims_collector.py**: 34 métodos (1,663 líneas)
4. **s3_report_uploader.py**: 10 métodos (337 líneas)

### Integraciones Externas:
- **Power BI API**: Consultas DAX para datos NPS y operativos
- **AWS S3**: Almacenamiento de reportes y datos NCS
- **Chatbot API**: Recolección de verbatims con JWT token
- **BeautifulSoup**: Parseo de emails HTML de NCS

### Funcionalidad Principal:
1. **Recolección de Datos NPS**: Datos diarios y agregados flexibles desde Power BI
2. **Recolección de Datos Operativos**: Métricas operacionales con agregación flexible
3. **Recolección de Verbatims**: Comentarios de clientes desde chatbot y Power BI
4. **Recolección de NCS**: Incidentes operacionales desde S3
5. **Upload de Reportes**: Reportes comprehensivos a S3
6. **Análisis de Sentimiento**: Procesamiento de verbatims con categorización temática

---

# Módulo: Explicación de Anomalías (`dashboard_analyzer/anomaly_explanation/`)

## Archivo: `dashboard_analyzer/anomaly_explanation/data_analyzer.py`

### Clase: `DataAnalyzer`
Analizador de datos operativos para complementar análisis NPS. Soporta tanto análisis LEGACY como FLEXIBLE con múltiples modos de comparación.

#### Métodos Públicos (32 métodos documentados):

#### 1. `__init__`
- **Argumentos**: `pbi_collector = None, comparison_mode: str = "mean", aggregation_days: int = 1`
- **Propósito**: Inicializa el analizador de datos
- **Dependencias**: Ninguna
- **Modos de Comparación**: "mean", "vslast", "target"
- **Lógica**: LEGACY (aggregation_days=1), FLEXIBLE (aggregation_days>1)

#### 2. `load_operative_data`
- **Argumentos**: `data_folder: str, node_path: str`
- **Propósito**: Carga datos operativos desde archivos CSV (soporta LEGACY y FLEXIBLE)
- **Dependencias**: `pandas`, `Path`
- **Archivos LEGACY**: `operative.csv`
- **Archivos FLEXIBLE**: `flexible_operative_Xd.csv`

#### 3. `analyze_operative_metrics`
- **Argumentos**: `node_path: str, target_date: str`
- **Propósito**: Analiza métricas operativas usando el modo de comparación configurado
- **Dependencias**: 
  - `_analyze_vslast_comparison()` [data_analyzer.py]
  - `_analyze_mean_comparison()` [data_analyzer.py]
  - `_analyze_target_comparison()` [data_analyzer.py]
  - `_analyze_flexible_comparison()` [data_analyzer.py]
  - `_analyze_legacy_operative()` [data_analyzer.py]

#### 4. `_analyze_flexible_comparison`
- **Argumentos**: `node_path: str, target_date: str`
- **Propósito**: Análisis flexible genérico basado en períodos de agregación
- **Dependencias**: 
  - `_clean_dax_columns()` [data_analyzer.py]
  - `_create_flexible_summary()` [data_analyzer.py]
- **Métricas Base**: Load_Factor, OTP15_adjusted, Mishandling, Misconex

#### 5. `_analyze_vslast_comparison`
- **Argumentos**: `node_path: str, target_date: str`
- **Propósito**: Análisis FLEXIBLE vs período anterior
- **Dependencias**: 
  - `_clean_dax_columns()` [data_analyzer.py]
  - `_create_vslast_flexible_summary()` [data_analyzer.py]

#### 6. `_analyze_mean_comparison`
- **Argumentos**: `node_path: str, target_date: str`
- **Propósito**: Análisis FLEXIBLE vs media de períodos históricos
- **Dependencias**: 
  - `_clean_dax_columns()` [data_analyzer.py]
  - `_create_mean_flexible_summary()` [data_analyzer.py]

#### 7. `_analyze_target_comparison`
- **Argumentos**: `node_path: str, target_date: str`
- **Propósito**: Análisis FLEXIBLE vs targets definidos
- **Dependencias**: 
  - `_clean_dax_columns()` [data_analyzer.py]
  - `_create_target_flexible_summary()` [data_analyzer.py]

#### 8. `_analyze_legacy_operative`
- **Argumentos**: `node_path: str, target_date: str`
- **Propósito**: Análisis operativo tradicional (aggregation_days=1)
- **Dependencias**: 
  - `_clean_dax_columns()` [data_analyzer.py]
  - `_create_legacy_summary()` [data_analyzer.py]

#### 9. `get_specific_explanations`
- **Argumentos**: `node_path: str, target_date: str, anomaly_type: str, aggregation_days: int = 7`
- **Propósito**: Genera explicaciones específicas para cada métrica operativa
- **Dependencias**: `_generate_metric_explanation()` [data_analyzer.py]
- **Tipos de Anomalía**: "positive", "negative"

#### 10. `_generate_metric_explanation`
- **Argumentos**: `metric: str, data: Dict, anomaly_type: str, mode: str`
- **Propósito**: Genera explicación textual para una métrica específica
- **Dependencias**: Ninguna
- **Métricas Soportadas**: Load_Factor, OTP15_adjusted, Mishandling, Misconex

#### 11. `_generate_legacy_explanation`
- **Argumentos**: `metric: str, value: float, anomaly_type: str`
- **Propósito**: Genera explicación legacy para métricas simples
- **Dependencias**: Ninguna

#### 12-17. `_create_*_summary` (6 métodos)
- **Propósito**: Crean resúmenes textuales para diferentes modos de comparación
- **Variantes**: flexible, vslast_flexible, mean_flexible, target_flexible, legacy

#### 18. `get_operational_metrics`
- **Argumentos**: `date_range: Tuple[str, str], node_path: str, comparison_filter: Optional[str] = None`
- **Propósito**: Obtiene y analiza métricas operacionales para un rango de fechas
- **Dependencias**: 
  - `_get_load_factor()` [data_analyzer.py]
  - `_get_mishandling_metrics()` [data_analyzer.py]
  - `_get_otp_metrics()` [data_analyzer.py]
  - `_get_aircraft_changes()` [data_analyzer.py]
  - `_create_operational_summary()` [data_analyzer.py]

#### 19. `analyze_customer_segments`
- **Argumentos**: `date_range, node_path, anomaly_type, comparison_filter = None`
- **Propósito**: Analiza segmentos de clientes y su contribución a anomalías
- **Dependencias**: 
  - `pbi_collector.collect_customer_profile_for_date_range()` [pbi_collector.py]
  - `_process_customer_segments()` [data_analyzer.py]
  - `_identify_reactive_segments()` [data_analyzer.py]

#### 20. `correlate_operational_with_nps`
- **Argumentos**: `operational_data: Dict, nps_data: Dict, anomaly_type: str`
- **Propósito**: Correlaciona métricas operativas con cambios en NPS
- **Dependencias**: 
  - `_correlate_load_factor_nps()` [data_analyzer.py]
  - `_correlate_mishandling_nps()` [data_analyzer.py]
  - `_correlate_otp_nps()` [data_analyzer.py]

#### 21-24. `_get_*_metrics` (4 métodos)
- **Métricas**: Load Factor, Mishandling, OTP (On-Time Performance), Aircraft Changes
- **Propósito**: Obtienen métricas específicas desde Power BI
- **Dependencias**: `pbi_collector` métodos

#### 25. `_process_customer_segments`
- **Argumentos**: `customer_data: pd.DataFrame, anomaly_type: str, comparison_filter`
- **Propósito**: Procesa datos de segmentos de clientes
- **Dependencias**: Ninguna

#### 26. `_identify_reactive_segments`
- **Argumentos**: `segments: List[Dict], anomaly_type: str`
- **Propósito**: Identifica segmentos que reaccionaron más a la anomalía
- **Dependencias**: Ninguna

#### 27-28. `_create_*_summary` (2 métodos)
- **Propósito**: Crean resúmenes operacionales y de clientes
- **Variantes**: operational_summary, customer_summary

#### 29-31. `_correlate_*_nps` (3 métodos)
- **Métricas**: Load Factor, Mishandling, OTP
- **Propósito**: Correlacionan métricas específicas con NPS
- **Dependencias**: Ninguna

#### 32. `_get_metric_display_name`
- **Argumentos**: `metric: str`
- **Propósito**: Obtiene nombre legible para mostrar métricas
- **Dependencias**: Ninguna
- **Mapeo**: Load_Factor → "Factor de Ocupación", OTP15_adjusted → "Puntualidad (OTP)", etc.

#### 33. `_clean_dax_columns`
- **Argumentos**: `data: pd.DataFrame`
- **Propósito**: Limpia nombres de columnas DAX de Power BI
- **Dependencias**: `re.sub`
- **Transformación**: Elimina patrones como `[Load_Factor]`, `Table_Name[Column]`

### Clase: `PrecalculatedDataAnalyzer`
Analizador optimizado que trabaja con datos pre-calculados de Power BI con comparaciones ya realizadas.

#### Métodos Adicionales (18 métodos documentados):

#### 1. `__init__`
- **Argumentos**: `comparison_mode: str = "mean", comparison_start_date = None, comparison_end_date = None, aggregation_days: int = 1, baseline_periods: int = 4`
- **Propósito**: Inicializa analizador con datos pre-calculados
- **Dependencias**: Ninguna

#### 2. `load_operative_data`
- **Argumentos**: `data_folder = None, node_path = None, aggregation_days: int = 7`
- **Propósito**: Carga datos operativos pre-calculados desde CSV
- **Dependencias**: `pandas`, `_clean_dax_columns()`

#### 3. `_analyze_precalculated_comparison`
- **Argumentos**: `data: pd.DataFrame, node_path: str`
- **Propósito**: Analiza datos con comparaciones ya pre-calculadas en DAX
- **Dependencias**: Ninguna

#### 4. `analyze_operative_metrics`
- **Argumentos**: `node_path: str, target_date: str`
- **Propósito**: Analiza métricas usando comparaciones pre-calculadas
- **Dependencias**: 
  - `_analyze_vslast()` [data_analyzer.py]
  - `_analyze_mean()` [data_analyzer.py]
  - `_analyze_target()` [data_analyzer.py]

#### 5-8. `_explain_*` (4 métodos)
- **Métricas**: Load Factor, OTP, Mishandling, Misconex
- **Propósito**: Generan explicaciones textuales para cada métrica
- **Dependencias**: Ninguna

#### 9-11. `_analyze_*` (3 métodos principales)
- **Modos**: vslast, mean, target
- **Propósito**: Análisis específicos para cada modo de comparación
- **Dependencias**: Métodos `_explain_*`

#### 12. `_analyze_vslast_with_specific_dates`
- **Argumentos**: `data: pd.DataFrame, target_dt: pd.Timestamp, node_path: str`
- **Propósito**: Análisis vslast con fechas específicas de comparación
- **Dependencias**: Métodos `_explain_*`

#### 13-15. `_create_*_summary` (3 métodos)
- **Modos**: vslast, mean, target
- **Propósito**: Crean resúmenes textuales para cada modo
- **Dependencias**: Ninguna

---

## Archivo: `dashboard_analyzer/anomaly_explanation/ncs_route_filter.py`

### Clase: `NCSRouteFilter`
Filtro especializado para datos de rutas del Network Control System (NCS). Maneja incidentes operativos y su impacto en rutas específicas.

#### Características:
- Tipos de incidentes con severidad y factores de impacto
- Patrones para extraer códigos de aeropuerto (IATA/ICAO)
- Aeropuertos principales con multiplicadores de impacto
- Correlación incidentes-NPS

#### Métodos Públicos (15 métodos documentados):

#### 1. `__init__`
- **Argumentos**: `pbi_collector`
- **Propósito**: Inicializa el filtro de rutas NCS
- **Dependencias**: Ninguna
- **Tipos de Incidentes**: weather, technical, atc, ground_stop, crew, maintenance, slot, security

#### 2. `filter_routes_by_incidents`
- **Argumentos**: `date_range: Tuple[str, str], node_path = None, incident_types = None, min_impact_score: float = 0.3`
- **Propósito**: Filtra rutas que tuvieron incidentes NCS en el período
- **Dependencias**: 
  - `pbi_collector.collect_ncs_data()` [pbi_collector.py]
  - `_process_ncs_incidents()` [ncs_route_filter.py]
  - `_calculate_route_impact_scores()` [ncs_route_filter.py]

#### 3. `get_incident_summary`
- **Argumentos**: `date_range: Tuple[str, str], node_path = None, group_by: str = "type"`
- **Propósito**: Obtiene resumen de incidentes NCS para el período
- **Dependencias**: 
  - `pbi_collector.collect_ncs_data()` [pbi_collector.py]
  - `_process_ncs_incidents()` [ncs_route_filter.py]
  - `_extract_affected_airports()` [ncs_route_filter.py]

#### 4. `correlate_incidents_with_nps`
- **Argumentos**: `date_range, node_path = None, anomaly_type: str = "negative"`
- **Propósito**: Correlaciona incidentes NCS con cambios en NPS
- **Dependencias**: 
  - `filter_routes_by_incidents()` [ncs_route_filter.py]
  - `_enrich_with_nps_data()` [ncs_route_filter.py]
  - `_perform_ncs_nps_correlation()` [ncs_route_filter.py]

#### 5. `_process_ncs_incidents`
- **Argumentos**: `ncs_data: pd.DataFrame`
- **Propósito**: Procesa y enriquece datos de incidentes NCS
- **Dependencias**: 
  - `_normalize_incident_type()` [ncs_route_filter.py]
  - `_extract_airports_from_incident()` [ncs_route_filter.py]

#### 6. `_normalize_incident_type`
- **Argumentos**: `incident_type: str`
- **Propósito**: Normaliza tipos de incidentes a categorías estándar
- **Dependencias**: Ninguna
- **Keywords por Tipo**: 
  - weather: 'weather', 'meteorológica', 'storm', 'tormenta'
  - technical: 'técnica', 'technical', 'avería', 'breakdown'
  - atc: 'atc', 'air traffic', 'tráfico aéreo'

#### 7. `_extract_airports_from_incident`
- **Argumentos**: `incident_row: pd.Series`
- **Propósito**: Extrae códigos de aeropuerto desde texto de incidente
- **Dependencias**: 
  - `_is_valid_airport()` [ncs_route_filter.py]
  - `re.findall`
- **Patrones**: Códigos IATA (3 letras), ICAO (4 letras)

#### 8. `_is_valid_airport`
- **Argumentos**: `airport_code: str`
- **Propósito**: Valida si un código de aeropuerto es válido
- **Dependencias**: Ninguna
- **Criterios**: Solo letras mayúsculas, longitud 3-4 caracteres

#### 9. `_calculate_route_impact_scores`
- **Argumentos**: `processed_ncs: pd.DataFrame`
- **Propósito**: Calcula scores de impacto para rutas afectadas
- **Dependencias**: 
  - `_get_max_severity()` [ncs_route_filter.py]
- **Factores**: Severidad del incidente, aeropuertos principales, duración

#### 10. `_get_max_severity`
- **Argumentos**: `severities: List[str]`
- **Propósito**: Obtiene la severidad máxima de una lista
- **Dependencias**: Ninguna
- **Orden**: critical > high > medium > low

#### 11. `_enrich_with_nps_data`
- **Argumentos**: `routes_df: pd.DataFrame, date_range, node_path`
- **Propósito**: Enriquece datos de rutas con información NPS
- **Dependencias**: `pbi_collector.collect_routes_for_date_range()` [pbi_collector.py]

#### 12. `_extract_affected_airports`
- **Argumentos**: `processed_ncs: pd.DataFrame`
- **Propósito**: Extrae aeropuertos afectados y cuenta de incidentes
- **Dependencias**: Ninguna

#### 13. `_calculate_incident_duration`
- **Argumentos**: `start_times: pd.Series, end_times: pd.Series`
- **Propósito**: Calcula duración promedio de incidentes
- **Dependencias**: `pd.to_datetime`

#### 14. `_perform_ncs_nps_correlation`
- **Argumentos**: `routes_with_incidents: pd.DataFrame, routes_with_nps: pd.DataFrame`
- **Propósito**: Realiza correlación estadística entre incidentes NCS y NPS
- **Dependencias**: `_create_correlation_summary()` [ncs_route_filter.py]

#### 15. `_create_correlation_summary`
- **Argumentos**: `correlations: List[Dict]`
- **Propósito**: Crea resumen textual de correlaciones encontradas
- **Dependencias**: Ninguna

### Diccionarios de Configuración:

**incident_types**:
- `weather`: severity='high', impact_factor=0.8
- `technical`: severity='medium', impact_factor=0.6
- `atc`: severity='medium', impact_factor=0.5
- `ground_stop`: severity='critical', impact_factor=0.9
- `crew`: severity='medium', impact_factor=0.4
- `maintenance`: severity='low', impact_factor=0.3

**major_airports**:
- `MAD`: impact_multiplier=1.0, region='domestic'
- `BCN`: impact_multiplier=0.9, region='domestic'
- `LHR`: impact_multiplier=1.2, region='europe'
- `JFK`: impact_multiplier=1.3, region='america'

---

## Archivo: `dashboard_analyzer/anomaly_explanation/routes_analyzer.py`

### Clase: `RoutesAnalyzer`
Analizador de rutas para identificar contribuciones específicas a anomalías NPS.

#### Métodos Públicos (8 métodos documentados):

#### 1. `__init__`
- **Argumentos**: `pbi_collector`
- **Propósito**: Inicializa el analizador de rutas
- **Dependencias**: Ninguna
- **Config**: min_surveys = 2 (mínimo para considerar una ruta)

#### 2. `analyze_routes_for_anomaly`
- **Argumentos**: `date_range: Tuple[str, str], node_path: str, anomaly_type: str, causal_filter: Optional[str] = None`
- **Propósito**: Analiza rutas específicas durante una anomalía
- **Dependencias**: 
  - `_get_routes_data()` [routes_analyzer.py]
  - `_sort_routes_by_anomaly_type()` [routes_analyzer.py]
  - `_integrate_ncs_data()` [routes_analyzer.py]
  - `_format_routes_results()` [routes_analyzer.py]

#### 3. `get_routes_by_driver`
- **Argumentos**: `driver_name: str, shap_value: float, date_range, node_path, causal_filter = None`
- **Propósito**: Obtiene rutas ordenadas por CSAT para un driver específico
- **Dependencias**: `pbi_collector.collect_routes_for_date_range()` [pbi_collector.py]
- **Ordenamiento**: Por SHAP value (negativo → peores primero, positivo → mejores primero)

#### 4. `_get_routes_data`
- **Argumentos**: `date_range, node_path, causal_filter = None`
- **Propósito**: Obtiene datos base de rutas
- **Dependencias**: `pbi_collector` métodos
- **Modos**: Comparativo (con filter) vs Single (sin filter)

#### 5. `_sort_routes_by_anomaly_type`
- **Argumentos**: `routes_data: pd.DataFrame, anomaly_type: str`
- **Propósito**: Ordena rutas según tipo de anomalía
- **Dependencias**: Ninguna
- **Lógica**: Negative → menor a mayor NPS, Positive → mayor a menor NPS

#### 6. `_integrate_ncs_data`
- **Argumentos**: `routes_data: pd.DataFrame, date_range, node_path`
- **Propósito**: Integra datos de NCS (Network Control System)
- **Dependencias**: `pbi_collector.collect_ncs_routes()` [pbi_collector.py]

#### 7. `_format_routes_results`
- **Argumentos**: `routes_data: pd.DataFrame, anomaly_type: str, causal_filter = None`
- **Propósito**: Formatea los resultados para presentación
- **Dependencias**: Ninguna
- **Output**: Top 10 rutas con detalles completos

#### 8. `get_verbatims_routes`
- **Argumentos**: `date_range, node_path, causal_filter = None`
- **Propósito**: Obtiene rutas mencionadas en verbatims
- **Dependencias**: `pbi_collector.collect_verbatims_routes()` [pbi_collector.py]

---

## Subcarpeta: `dashboard_analyzer/anomaly_explanation/genai_core/`

### Archivo: `genai_core/message_history.py`

#### Clase: `MessageHistory`
Gestiona historial de mensajes en una conversación con soporte multimodal (texto + imágenes).

#### Métodos Públicos (9 métodos documentados):

#### 1. `__init__`
- **Argumentos**: `logger: Optional[logging.Logger] = None`
- **Propósito**: Inicializa lista vacía para almacenar mensajes
- **Dependencias**: Ninguna

#### 2. `create_and_add_message`
- **Argumentos**: `content, message_type: MessageType, agent = None, visible: bool = True, tool_call_id = None, images = None`
- **Propósito**: Agrega un nuevo mensaje con soporte multimodal
- **Dependencias**: 
  - `_create_multimodal_content()` [message_history.py]
  - LangChain message types

#### 3. `_create_multimodal_content`
- **Argumentos**: `text: str, images = None`
- **Propósito**: Crea estructura de contenido multimodal desde texto e imágenes
- **Dependencias**: 
  - `_get_mime_type()` [message_history.py]
  - `base64.b64encode`

#### 4. `_get_mime_type`
- **Argumentos**: `file_path: str`
- **Propósito**: Determina MIME type basado en extensión de archivo
- **Dependencias**: `os.path.splitext`
- **Tipos**: image/jpeg, image/png, image/gif, image/webp

#### 5. `add_message`
- **Argumentos**: `message, agent = None, visible: bool = True`
- **Propósito**: Agrega un mensaje de LangChain al historial
- **Dependencias**: Ninguna

#### 6. `get_messages`
- **Argumentos**: `include_agent_name_in_content: bool = False`
- **Propósito**: Obtiene mensajes formateados para prompts LLM
- **Dependencias**: Ninguna

#### 7. `to_str`
- **Argumentos**: `visible_only: bool = False`
- **Propósito**: Convierte historial de mensajes a string formateado
- **Dependencias**: Ninguna

#### 8. `to_dict`
- **Argumentos**: Ninguno
- **Propósito**: Serializa MessageHistory a diccionario
- **Dependencias**: Ninguna
- **Formato**: Persistencia (ej: JSON)

#### 9. `from_dict` (classmethod)
- **Argumentos**: `data: Dict, logger = None`
- **Propósito**: Deserializa diccionario a MessageHistory
- **Dependencias**: Ninguna
- **Returns**: Nueva instancia de MessageHistory

---

### Archivo: `genai_core/agents/agent.py`

#### Clase: `Agent`
Clase base genérica que puede usarse directamente o extenderse para agentes especializados.

#### Características:
- Tool/function calling
- Structured output
- Performance tracking (tokens, cost, time)
- Streaming responses

#### Métodos Públicos (5 métodos documentados):

#### 1. `__init__`
- **Argumentos**: `llm, logger = None`
- **Propósito**: Inicializa agente con componentes configurables
- **Dependencias**: Ninguna
- **Tracking**: total_time, input_tokens, output_tokens, money_spent

#### 2. `invoke` (async)
- **Argumentos**: `messages: list, structured_output = None, tools = None`
- **Propósito**: Procesa input y genera respuesta usando LLM (no-streaming)
- **Dependencias**: 
  - `llm()` call
  - `_update_metrics()` [agent.py]
- **Returns**: response, structured_response, tool_calls

#### 3. `ainvoke` (async generator)
- **Argumentos**: `messages: list`
- **Propósito**: Procesa input y genera respuesta usando LLM (streaming)
- **Dependencias**: 
  - `llm.stream_response()` 
  - `_update_metrics()` [agent.py]
- **Yield**: Message chunks

#### 4. `_update_metrics`
- **Argumentos**: `execution_time`
- **Propósito**: Actualiza métricas de rendimiento del agente
- **Dependencias**: Ninguna

#### 5. `execute_tools` (async)
- **Argumentos**: `mcp_manager, tool_calls`
- **Propósito**: Ejecuta tool calls usando un gestor MCP
- **Dependencias**: `mcp_manager.execute_tool_calls()`
- **Returns**: Tool responses

---

### Subcarpeta: `genai_core/llms/`

### Archivo: `llms/llm.py`

#### Clase: `LLM` (Abstract Base Class)
Clase base abstracta para Large Language Models. Proporciona interfaz común para diferentes implementaciones LLM.

#### Métodos Públicos (5 métodos):

#### 1. `__init__`
- **Argumentos**: `llm_type: LLMType, token_input_price: float, token_output_price: float`
- **Propósito**: Inicializa instancia LLM
- **Dependencias**: `create_llm()` [abstractmethod]

#### 2. `__call__` (async)
- **Argumentos**: `prompt: list, tools: list = None, structured_output = None`
- **Propósito**: Invoca LLM con prompt, tools opcionales y structured output
- **Dependencias**: 
  - `_add_structured_output()` [llm.py]
  - `llm.bind_tools()` (LangChain)

#### 3. `_add_structured_output`
- **Argumentos**: `tools, structured_output: Optional[Type[BaseModel]]`
- **Propósito**: Agrega tool de structured output a la lista de tools
- **Dependencias**: `StructuredTool.from_function()` (LangChain)

#### 4. `stream_response` (async generator)
- **Argumentos**: `prompt`
- **Propósito**: Stream de respuesta LLM por chunks
- **Dependencias**: `llm.astream()` (LangChain)

#### 5. `create_llm` (abstractmethod)
- **Propósito**: Instancia y retorna el cliente LLM específico
- **Implementaciones**: AWSLLM, OpenAiLLM

---

### Archivo: `llms/aws_llm.py`

#### Clase: `AWSLLM`
Implementación LLM que usa integración ChatBedrock de LangChain con AWS.

#### Métodos Principales (3 métodos):

#### 1. `__init__`
- **Argumentos**: `llm_type: LLMType, region_name, aws_access_key_id = None, aws_secret_access_key = None, profile_name = None, token_input_price = 11.02/1M, token_output_price = 32.68/1M`
- **Propósito**: Inicializa AWSLLM con credenciales AWS
- **Dependencias**: `super().__init__()`

#### 2. `create_llm`
- **Argumentos**: Ninguno
- **Propósito**: Crea cliente ChatBedrock de LangChain
- **Dependencias**: 
  - `_set_model_id()` [aws_llm.py]
  - `_get_provider()` [aws_llm.py]
  - `boto3.Session`
  - `ChatBedrock` (LangChain)
- **Config**: read_timeout, connect_timeout, max_retries, max_tokens, temperature

#### 3. `_get_provider`
- **Argumentos**: Ninguno
- **Propósito**: Obtiene nombre del provider basado en tipo de modelo
- **Dependencias**: Ninguna
- **Providers**: "anthropic" (Claude models), "meta" (Llama models)

#### 4. `_set_model_id`
- **Argumentos**: Ninguno
- **Propósito**: Establece el model ID apropiado según LLM type
- **Dependencias**: `BEDROCK_MODELS` (env var)
- **Modelos Soportados**:
  - Claude 3 Haiku, 3.5 Haiku, 3 Opus
  - Claude 3.5 Sonnet (v1 y v2)
  - Claude Sonnet 4, Claude 3.7 Sonnet
  - Llama 3 70B, 3.1 70B, 3.1 405B

---

### Archivo: `llms/openai_llm.py`

#### Clase: `OpenAiLLM`
Implementación LLM que usa Azure OpenAI.

#### Métodos Principales (2 métodos):

#### 1. `__init__`
- **Argumentos**: `llm_type: LLMType, api_key: str, api_base: str, api_version: str, api_dep_gpt: str, token_input_price = 3.75/1M, token_output_price = 15/1M, temperature: float = 0.1, size: int = 4`
- **Propósito**: Inicializa OpenAiLLM
- **Dependencias**: `super().__init__()`

#### 2. `create_llm`
- **Argumentos**: Ninguno
- **Propósito**: Establece conexión con OpenAI e instancia el modelo
- **Dependencias**: `AzureChatOpenAI` (LangChain)
- **Config**: azure_endpoint, deployment_name, api_version, temperature, max_completion_tokens=3000

---

### Archivo: `genai_core/utils/enums.py`

#### Enumeraciones y Configuración:

#### 1. `DEFAULT_LLM_TYPE`
- **Valor**: "O4_MINI"
- **Propósito**: Configuración global de modelo - cambiar para switchear todos los agentes
- **Opciones**: O4_MINI, CLAUDE_SONNET_4, O3, etc.

#### 2. `MessageType` (Enum)
- **Valores**: USER, AI, SYSTEM, TOOL
- **Propósito**: Representa tipos de mensajes posibles

#### 3. `LLMType` (Enum)
- **Modelos Claude**: CLAUDE_V2, CLAUDE_INSTANT, CLAUDE_3_HAIKU, CLAUDE_3_5_HAIKU, CLAUDE_3_OPUS, CLAUDE_3_5_SONNET (v1 y v2), CLAUDE_SONNET_4, CLAUDE_3_7_SONNET
- **Modelos GPT**: GPT3_5, GPT4, GPT4o, GPT4o_MINI, O1_MINI, O3_MINI, O3, O4_MINI
- **Modelos Llama**: LLAMA3_70, LLAMA3_1_70, LLAMA3_1_405

#### 4. `get_default_llm_type()`
- **Returns**: LLMType
- **Propósito**: Obtiene el tipo LLM por defecto desde configuración global
- **Fallback**: O4_MINI si el tipo configurado no existe

#### 5. `AgentName` (Enum)
- **Valores**: CONVERSATIONAL
- **Propósito**: Categoriza agentes por funcionalidad

---

### Archivo: `genai_core/agents/anomaly_interpreter_agent.py`

#### Clase: `AnomalyInterpreterAgent`
Agente especializado para interpretar árboles de anomalías NPS y generar conclusiones sobre correlaciones entre desempeño operacional y percepción del cliente.

#### Características:
- Interpretación de árboles de anomalías jerárquicas
- Análisis generación por generación
- Conclusiones sobre correlaciones operativas-NPS
- Soporte multi-LLM (OpenAI, AWS Bedrock)
- Upload de conversaciones a S3
- Tracking de conversación jerárquica

#### Clases Auxiliares:

##### `HierarchicalConversationTracker`
Tracking de workflow conversacional jerárquico con análisis generación por generación.

**Métodos**:
- `reset_tracker()`: Reset para nueva investigación jerárquica
- `log_message()`: Registra mensaje en conversación jerárquica
- `set_hierarchy_structure()`: Almacena estructura de jerarquía parseada
- `start_generation()`: Inicia análisis de nueva generación
- `add_generation_reflection()`: Agrega reflexión para generación específica
- `add_generation_analysis()`: Agrega datos de análisis detallado
- `complete_generation()`: Marca generación como completa
- `get_conversation_summary()`: Genera resumen de toda la conversación

#### Métodos Principales del Agente:

**Configuración e Inicialización**:
- `__init__()`: Inicializa agente con LLM type y config
- `_setup_logger()`: Configura logger por defecto
- `_load_prompt_config()`: Carga configuración de prompts desde YAML
- `_create_llm()`: Crea instancia LLM según tipo

**Procesamiento de Datos**:
- `_parse_hierarchical_explanation()`: Parsea explicación jerárquica en estructura de árbol
- `_extract_nodes_from_generation()`: Extrae nodos de una generación específica
- `_clean_node_path()`: Limpia ruta de nodo para consistencia

**Generación de Interpretaciones**:
- `interpret_anomaly_tree()` (async): Interpreta árbol de anomalías con análisis multi-generación
- `_analyze_generation()` (async): Analiza nodos de una generación específica
- `_create_generation_prompt()`: Crea prompt para análisis de generación
- `_create_synthesis_prompt()`: Crea prompt para síntesis final

**Utilidades**:
- `_format_hierarchical_tree()`: Formatea árbol jerárquico para visualización
- `_extract_structured_response()`: Extrae respuesta estructurada desde tool calls
- `_upload_conversation_to_s3()` (async): Upload de conversación a S3

---

### Archivo: `genai_core/agents/anomaly_summary_agent.py`

#### Clase: `AnomalySummaryAgent`
Agente especializado para resumir múltiples períodos de análisis de anomalías y generar insights estratégicos de nivel ejecutivo.

#### Características:
- Análisis de tendencias multi-período
- Identificación de patrones a través del tiempo
- Evaluación de prioridades estratégicas
- Reporting de nivel ejecutivo
- Soporte multi-LLM (OpenAI, AWS Bedrock)
- Upload de síntesis a S3

#### Métodos Principales:

**Configuración e Inicialización**:
- `__init__()`: Inicializa agente con LLM type, config y S3 uploader
- `_setup_logger()`: Configura logger por defecto
- `_load_prompt_config()`: Carga configuración desde YAML
- `_create_llm()`: Crea instancia LLM según tipo
- `_resolve_config_path()`: Resuelve path de configuración

**Síntesis Multi-Período**:
- `synthesize_multi_period_analysis()` (async): Sintetiza análisis de múltiples períodos
- `_create_synthesis_prompt()`: Crea prompt comprehensivo para síntesis
- `_format_period_analysis()`: Formatea análisis de un período
- `_extract_synthesis_response()`: Extrae respuesta de síntesis

**Utilidades**:
- `_format_period_date_range()`: Formatea rango de fechas para presentación
- `_upload_synthesis_to_s3()` (async): Upload de síntesis a S3

---

## Subcarpeta: `dashboard_analyzer/anomaly_explanation/config/prompts/`

### Archivos de Configuración YAML:

#### 1. `anomaly_interpreter.yaml`
- **Propósito**: Prompts para AnomalyInterpreterAgent
- **Contenido**: System prompt, generation analysis prompt, synthesis prompt
- **Formato**: YAML con templates de prompts

#### 2. `anomaly_summary.yaml`
- **Propósito**: Prompts para AnomalySummaryAgent
- **Contenido**: System prompt, synthesis prompt, executive summary template
- **Formato**: YAML con templates de prompts

#### 3. `causal_explanation.yaml`
- **Propósito**: Prompts para CausalExplanationAgent
- **Contenido**: System prompt, analysis prompts, explanation templates
- **Formato**: YAML con templates de prompts

---

## 📊 Resumen del Módulo `anomaly_explanation/`

### Estadísticas:
- **Total de archivos principales**: 3 (data_analyzer, ncs_route_filter, routes_analyzer)
- **Total de clases principales**: 5 (DataAnalyzer, PrecalculatedDataAnalyzer, NCSRouteFilter, RoutesAnalyzer, Agent base + derivadas)
- **Total de métodos documentados**: ~130+
- **Líneas de código totales**: ~3,000 líneas

### Distribución por archivo:
1. **data_analyzer.py**: 2 clases, 50+ métodos (2,138 líneas)
2. **ncs_route_filter.py**: 1 clase, 15 métodos (555 líneas)
3. **routes_analyzer.py**: 1 clase, 8 métodos (313 líneas)
4. **genai_core/**: 10+ archivos con agentes, LLMs, utilidades

### Componentes GenAI Core:
- **Agentes**: Agent (base), AnomalyInterpreterAgent, AnomalySummaryAgent, CausalExplanationAgent
- **LLMs**: LLM (base), AWSLLM, OpenAiLLM
- **Utilidades**: MessageHistory, enums, prompt configs

### Funcionalidad Principal:
1. **Análisis de Datos Operativos**: Métricas operativas con múltiples modos de comparación (mean, vslast, target)
2. **Filtrado de Rutas NCS**: Análisis de incidentes operativos y su impacto en rutas
3. **Análisis de Rutas**: Identificación de rutas problemáticas y contribución a anomalías
4. **Interpretación de Anomalías**: Análisis jerárquico multi-generación con AI
5. **Síntesis Multi-Período**: Reporting ejecutivo de tendencias y patrones
6. **Explicaciones Causales**: Análisis causal con SHAP values y drivers explicativos

### Integraciones:
- **Power BI API**: Recolección de métricas operativas, rutas, customer profiles
- **AWS Bedrock**: Claude, Llama models
- **Azure OpenAI**: GPT-4, O-series models
- **AWS S3**: Upload de conversaciones y síntesis
- **LangChain**: Framework para agentes y LLMs

---

## Archivo: `dashboard_analyzer/anomaly_explanation/genai_core/agents/causal_explanation_agent.py`

### **⭐ COMPONENTE PRINCIPAL DEL SISTEMA** (6,903 líneas, 129 métodos)

Este es el **agente central** que orquesta toda la investigación causal de anomalías NPS utilizando múltiples fuentes de datos y LLMs.

### Clases Principales:

#### Clase 1: `CausalAnalysisResult` (BaseModel)
**Modelo Pydantic** para resultados estructurados de análisis causal.
- **Campos**:
  - `primary_cause`: Causa principal identificada
  - `confidence_level`: Nivel de confianza (High/Medium/Low)
  - `supporting_evidence`: Lista de evidencias
  - `next_investigation_step`: Siguiente paso de investigación
  - `final_explanation`: Explicación causal completa

---

#### Clase 2: `CleanConversationTracker`
**Gestiona el flujo de conversación** entre el agente y los LLMs.

##### Métodos (10 métodos):

###### 1. `__init__()`
- **Argumentos**: Ninguno
- **Propósito**: Inicializa tracker con logs, iteraciones, explicaciones previas, rutas identificadas, y queries DAX
- **Dependencias**: Ninguna

###### 2. `reset_tracker()`
- **Argumentos**: Ninguno
- **Propósito**: Resetea el tracker para una nueva investigación
- **Dependencias**: Ninguna

###### 3. `reset()`
- **Argumentos**: Ninguno
- **Propósito**: Resetea estado completo del agente incluyendo datos recolectados
- **Dependencias**: Ninguna

###### 4. `log_message()`
- **Argumentos**: `message_type: str, content: str, metadata: Optional[Dict] = None`
- **Propósito**: Registra mensajes en el log de conversación con timestamp
- **Dependencias**: `datetime.now()`

###### 5. `add_explanation()`
- **Argumentos**: `explanation: str`
- **Propósito**: Añade explicación al contexto limpio para próximas reflexiones
- **Dependencias**: Ninguna

###### 6. `get_clean_context()`
- **Argumentos**: `system_prompt: str`
- **Propósito**: Construye contexto limpio: system prompt + explicaciones previas (sin historial completo)
- **Dependencias**: Ninguna
- **Retorna**: `List[Dict]` con mensajes formateados

###### 7. `next_iteration()`
- **Argumentos**: Ninguno
- **Propósito**: Incrementa contador de iteración
- **Dependencias**: Ninguna

###### 8. `set_tool_context()`
- **Argumentos**: `tool_name: str, result: str`
- **Propósito**: Guarda contexto del último tool ejecutado y extrae rutas mencionadas
- **Dependencias**: `_extract_routes_from_result()`

###### 9. `add_dax_query()`
- **Argumentos**: `tool_name: str, query: str, parameters: dict = None`
- **Propósito**: Registra query DAX ejecutada con timestamp e iteración
- **Dependencias**: `datetime.now()`

###### 10. `_extract_routes_from_result()`
- **Argumentos**: `result: str`
- **Propósito**: Extrae códigos de rutas (IATA) y nombres en español de resultados usando regex
- **Dependencias**: `re` (regex)
- **Retorna**: `List[str]` con rutas en formato IATA
- **Patterns**:
  - Estándar: `XXX-YYY` (ej: MAD-UIO)
  - Español: Madrid-Bogotá, Barcelona-Nueva York, etc.
  - Mapeo ciudad a IATA: Madrid→MAD, Bogotá→BOG, Lima→LIM, etc.

###### 11. `has_identified_routes()`
- **Argumentos**: Ninguno
- **Propósito**: Verifica si se han identificado rutas en tools previos
- **Dependencias**: Ninguna
- **Retorna**: `bool`

###### 12. `should_validate_routes_after_tool()`
- **Argumentos**: `tool_name: str`
- **Propósito**: Determina si debe llamarse routes_tool después del tool actual
- **Dependencias**: `has_identified_routes()`
- **Retorna**: `bool`

---

#### Clase 3: `CausalExplanationAgent` 
**⭐ AGENTE PRINCIPAL** - Orquesta investigación causal completa con workflow limpio.

##### Constructor y Configuración (9 métodos):

###### 1. `__init__()`
- **Argumentos**: 
  - `llm_type: Optional[LLMType] = None`
  - `config_path: str = "dashboard_analyzer/anomaly_explanation/config/prompts/causal_explanation.yaml"`
  - `logger: Optional[logging.Logger] = None`
  - `silent_mode: bool = False`
  - `custom_helper_prompts: Optional[Dict[str, Any]] = None`
  - `detection_mode: str = "vslast"`
  - `causal_filter: str = "vs L7d"`
  - `comparison_start_date: datetime = None`
  - `comparison_end_date: datetime = None`
  - `study_mode: str = "comparative"`
- **Propósito**: Inicializa agente con LLM, colectores de datos, configuración de prompts, y parámetros de análisis
- **Dependencias**:
  - `get_default_llm_type()` → genai_core.utils.enums
  - `_setup_logger()` (método interno)
  - `_load_prompt_config()` (método interno)
  - `_merge_helper_prompts()` (método interno)
  - `_init_chatbot_collector()` (método interno)
  - `_init_ncs_collector()` (método interno)
  - `_create_llm()` (método interno)
  - `PBIDataCollector()` → data_collection.pbi_collector
  - `S3ReportUploader()` → data_collection.s3_report_uploader
  - `Agent()` → genai_core.agents.agent
  - `CleanConversationTracker()` (clase interna)
  - `OperationalDataAnalyzer()` → anomaly_explanation.data_analyzer
- **Transformaciones**:
  - `vslast` + `"vs Sel. Period"` → `vslast_dynamic`
  - String dates → datetime objects

###### 2. `_setup_logger()`
- **Argumentos**: Ninguno
- **Propósito**: Configura logger con formato específico y nivel INFO
- **Dependencias**: `logging` (stdlib)
- **Retorna**: `logging.Logger`

###### 3. `calculate_dynamic_comparison_dates()`
- **Argumentos**: `start_date: datetime, end_date: datetime`
- **Propósito**: Calcula fechas de comparación dinámicas según causal_filter (L7d, L28d, LW, LM, LY)
- **Dependencias**: `datetime`, `timedelta`
- **Retorna**: `Tuple[datetime, datetime]` (comparison_start, comparison_end)
- **Lógica**:
  - `"vs L7d"`: 7 días antes del start_date
  - `"vs L28d"`: 28 días antes
  - `"vs LW"`: Misma semana del año anterior
  - `"vs LM"`: Mismo mes del año anterior
  - `"vs LY"`: Mismo período del año anterior

###### 4. `_load_prompt_config()`
- **Argumentos**: `config_path: str`
- **Propósito**: Carga configuración de prompts desde archivo YAML
- **Dependencias**: `yaml.safe_load()`, `Path.read_text()`
- **Retorna**: `Dict[str, Any]`

###### 5. `_merge_helper_prompts()`
- **Argumentos**: `custom_prompts: Dict[str, Any]`
- **Propósito**: Merge prompts custom con los del config base
- **Dependencias**: Ninguna

###### 6. `_init_chatbot_collector()`
- **Argumentos**: Ninguno
- **Propósito**: Inicializa ChatbotVerbatimsCollector con token desde env
- **Dependencias**: 
  - `_load_chatbot_token()` (método interno)
  - `ChatbotVerbatimsCollector()` → data_collection.chatbot_verbatims_collector
- **Retorna**: `ChatbotVerbatimsCollector` o `None`

###### 7. `_load_chatbot_token()`
- **Argumentos**: Ninguno
- **Propósito**: Carga token del chatbot desde variables de entorno o archivo .chatbot_token
- **Dependencias**: 
  - `os.getenv()`, `load_dotenv()`
  - `Path.read_text()`
- **Retorna**: `Optional[str]`

###### 8. `_init_ncs_collector()`
- **Argumentos**: Ninguno
- **Propósito**: Inicializa NCSDataCollector
- **Dependencias**: `NCSDataCollector()` → data_collection.ncs_collector
- **Retorna**: `NCSDataCollector` o `None`

###### 9. `_create_llm()`
- **Argumentos**: `llm_type: LLMType`
- **Propósito**: Factory method para crear instancia de LLM (OpenAI o AWS)
- **Dependencias**: 
  - `_create_openai_llm()` o `_create_aws_llm()` (métodos internos)
- **Retorna**: `OpenAiLLM` o `AWSLLM`

###### 10. `_create_openai_llm()`
- **Argumentos**: `llm_type: LLMType`
- **Propósito**: Crea instancia de OpenAI LLM con modelo apropiado
- **Dependencias**: `OpenAiLLM()` → genai_core.llms.openai_llm
- **Retorna**: `OpenAiLLM`
- **Modelos soportados**: GPT4_O, GPT4_O_MINI, O1_PREVIEW, O1_MINI

###### 11. `_create_aws_llm()`
- **Argumentos**: `llm_type: LLMType`
- **Propósito**: Crea instancia de AWS Bedrock LLM con modelo apropiado
- **Dependencias**: `AWSLLM()` → genai_core.llms.aws_llm
- **Retorna**: `AWSLLM`
- **Modelos soportados**: CLAUDE_SONNET, CLAUDE_HAIKU, LLAMA_70B

---

##### Tools de Recolección de Datos (26 métodos):

###### 12. `_explanatory_drivers_tool()`
- **Argumentos**: `node_path: str, start_date: str, end_date: str, min_surveys: int = 10`
- **Propósito**: **ASYNC** - Obtiene drivers explicativos con SHAP values para node específico
- **Dependencias**:
  - `_collect_explanatory_drivers_with_query_tracking()` (método interno)
  - `pbi_collector.get_explanatory_drivers_async()` → PBIDataCollector
- **Retorna**: `str` con análisis formateado
- **Métricas retornadas**: SHAP values, touchpoints, dimensiones de segmentación

###### 13. `_operative_data_tool_single_period()`
- **Argumentos**: `node_path, start_date, end_date, baseline_start_date, comparison_context, baseline_periods, anomaly_detection_mode, aggregation_days`
- **Propósito**: **ASYNC** - Obtiene métricas operativas para período único
- **Dependencias**:
  - `_collect_operative_data_with_query_tracking()` (método interno)
  - `OperationalDataAnalyzer.analyze_from_dataframe()` → data_analyzer
- **Retorna**: `str` con métricas (Load Factor, OTP, Mishandling, Misconex)

###### 14. `_collect_operative_data_with_query_tracking()`
- **Argumentos**: `node_path, target_date, comparison_days, use_flexible, comparison_start_date, comparison_end_date`
- **Propósito**: **ASYNC** - Recolecta datos operativos y registra queries DAX ejecutadas
- **Dependencias**:
  - `tracker.add_dax_query()` (CleanConversationTracker)
  - `pbi_collector.get_operative_data_flexible_async()` o `get_operative_data_async()`
- **Retorna**: `pd.DataFrame`

###### 15. `_collect_routes_with_query_tracking()`
- **Argumentos**: `node_path, start_date, end_date, comparison_filter, comparison_start_date, comparison_end_date`
- **Propósito**: **ASYNC** - Recolecta rutas con NPS y registra query
- **Dependencias**:
  - `tracker.add_dax_query()`
  - `pbi_collector.get_routes_async()`
- **Retorna**: `pd.DataFrame`

###### 16. `_collect_customer_profile_with_query_tracking()`
- **Argumentos**: `node_path, start_date, end_date, dimension, comparison_filter, comparison_start_date, comparison_end_date, route_filter`
- **Propósito**: **ASYNC** - Recolecta customer profile (ticket type, haul, etc.) y registra query
- **Dependencias**:
  - `tracker.add_dax_query()`
  - `pbi_collector.get_customer_profile_async()`
- **Retorna**: `pd.DataFrame`

###### 17. `_collect_explanatory_drivers_with_query_tracking()`
- **Argumentos**: `node_path, start_date, end_date, comparison_filter, comparison_start_date, comparison_end_date`
- **Propósito**: **ASYNC** - Recolecta explanatory drivers y registra query
- **Dependencias**:
  - `tracker.add_dax_query()`
  - `pbi_collector.get_explanatory_drivers_async()`
- **Retorna**: `pd.DataFrame`

###### 18. `_collect_verbatims_with_query_tracking()`
- **Argumentos**: `node_path, start_date, end_date`
- **Propósito**: Recolecta verbatims de Power BI y registra query
- **Dependencias**:
  - `tracker.add_dax_query()`
  - `pbi_collector.get_verbatims()`
- **Retorna**: `pd.DataFrame`

###### 19. `_operative_data_tool_correlation_analysis()`
- **Argumentos**: `node_path, operative_data, target_date_str, comparison_context, baseline_periods, anomaly_detection_mode, aggregation_days`
- **Propósito**: **ASYNC** - Analiza correlaciones entre métricas operativas y anomalía NPS
- **Dependencias**:
  - `OperationalDataAnalyzer.analyze_from_dataframe()`
  - `_get_metric_impact_summary()` (método interno)
  - `_generate_correlation_summary()` (método interno)
- **Retorna**: `str` con análisis de correlación

###### 20. `_ncs_tool_single_period()`
- **Argumentos**: `node_path, start_date, end_date`
- **Propósito**: **ASYNC** - Obtiene incidentes NCS (No Contact Service) para período único
- **Dependencias**:
  - `ncs_collector.collect_ncs_data_async()`
  - `_filter_ncs_by_segment()` (método interno)
  - `_apply_contextual_filtering()` (método interno)
- **Retorna**: `str` con incidentes categorizados (weather, technical, atc, ground)

###### 21. `_routes_tool_single_period()`
- **Argumentos**: `node_path, start_date, end_date, min_surveys, anomaly_type`
- **Propósito**: **ASYNC** - Obtiene rutas problemáticas para período único
- **Dependencias**:
  - `_consolidate_routes_from_all_sources()` (método interno)
- **Retorna**: `str` con rutas consolidadas de todas las fuentes

###### 22. `_verbatims_tool_single_period()`
- **Argumentos**: `node_path, start_date, end_date`
- **Propósito**: **ASYNC** - Analiza verbatims de clientes para período único
- **Dependencias**:
  - `_analyze_verbatims_single_period()` (método interno)
- **Retorna**: `str` con análisis de verbatims

###### 23. `_analyze_verbatims_single_period()`
- **Argumentos**: `df, node_path, start_date, end_date`
- **Propósito**: **ASYNC** - Análisis detallado de verbatims (PBI o Chatbot)
- **Dependencias**:
  - `_determine_verbatim_type_from_context()` (método interno)
  - `_conduct_chatbot_conversation()` (método interno)
  - `_analyze_pbi_verbatims()` (método interno)
- **Retorna**: `str` con análisis categorizdo

###### 24. `_determine_verbatim_type_from_context()`
- **Argumentos**: Ninguno
- **Propósito**: Determina tipo de verbatim (detractors/promoters/passives) desde contexto previo
- **Dependencias**: `tracker.conversation_log`
- **Retorna**: `str`

###### 25. `_conduct_chatbot_conversation()`
- **Argumentos**: `verbatim_type, node_path, start_date, end_date`
- **Propósito**: **ASYNC** - Realiza conversación multi-turno con chatbot API para análisis profundo
- **Dependencias**:
  - `_get_chatbot_filters_from_node_path()` (método interno)
  - `_generate_negative_routes_query()` (método interno)
  - `_generate_representative_comments_query()` (método interno)
  - `_generate_operational_correlation_query()` (método interno)
  - `_generate_route_specific_query()` (método interno)
  - `_generate_synthesis_additional_causes_query()` (método interno)
  - `chatbot_collector.send_message()`
- **Retorna**: `List[Dict]` con conversación completa
- **Flujo**: 5 mensajes secuenciales (rutas → comentarios → correlación → ruta específica → síntesis)

###### 26. `_get_chatbot_filters_from_node_path()`
- **Argumentos**: `node_path: str`
- **Propósito**: Extrae filtros (cabin, company, haul, country) desde node_path jerárquico
- **Dependencias**: Ninguna
- **Retorna**: `Dict` con filtros
- **Ejemplo**: `"Business/IB/Long Haul/ES"` → `{"cabin": "Business", "company": "IB", "haul": "Long Haul", "country": "ES"}`

###### 27. `_generate_negative_routes_query()`
- **Argumentos**: `node_path: str`
- **Propósito**: Genera query para rutas negativas mencionadas en verbatims
- **Retorna**: `str`

###### 28. `_generate_representative_comments_query()`
- **Argumentos**: `node_path, previous_response`
- **Propósito**: Genera query para comentarios representativos basado en rutas identificadas
- **Retorna**: `str`

###### 29. `_generate_operational_correlation_query()`
- **Argumentos**: `node_path, response_1, response_2`
- **Propósito**: Genera query para correlación operacional
- **Retorna**: `str`

###### 30. `_generate_route_specific_query()`
- **Argumentos**: `node_path, response_1, response_2, response_3`
- **Propósito**: Genera query para análisis específico de ruta principal
- **Retorna**: `str`

###### 31. `_generate_synthesis_additional_causes_query()`
- **Argumentos**: `node_path, response_1, response_2, response_3, response_4`
- **Propósito**: Genera query final de síntesis con causas adicionales
- **Retorna**: `str`

###### 32. `_analyze_single_chatbot_response()`
- **Argumentos**: `df, purpose`
- **Propósito**: Analiza respuesta individual del chatbot extrayendo insights clave
- **Retorna**: `str`

###### 33. `_synthesize_conversation_results()`
- **Argumentos**: `conversation, verbatim_type, node_path, start_date, end_date`
- **Propósito**: Sintetiza resultados de conversación multi-turno con chatbot
- **Dependencias**: `_analyze_single_chatbot_response()` (método interno)
- **Retorna**: `str` con síntesis estructurada

###### 34. `_analyze_chatbot_verbatims()`
- **Argumentos**: `df, node_path, start_date, end_date, verbatim_type`
- **Propósito**: Análisis de verbatims desde chatbot con LLM
- **Dependencias**: 
  - `llm.complete()` → LLM
- **Retorna**: `str`

###### 35. `_analyze_pbi_verbatims()`
- **Argumentos**: `df, node_path, start_date, end_date`
- **Propósito**: Análisis de verbatims desde Power BI con LLM
- **Dependencias**: 
  - `llm.complete()` → LLM
- **Retorna**: `str`

###### 36. `_analyze_chatbot_single_period_response()`
- **Argumentos**: `answer_data, node_path, start_date, end_date`
- **Propósito**: Formatea respuesta del chatbot API
- **Retorna**: `str`

###### 37. `_ncs_tool()`
- **Argumentos**: `node_path, start_date, end_date, analysis_focus, temporal_comparison`
- **Propósito**: **ASYNC** - Análisis de incidentes NCS con comparación temporal opcional
- **Dependencias**:
  - `ncs_collector.collect_ncs_data_async()`
  - `_filter_ncs_by_segment()` (método interno)
  - `_apply_contextual_filtering()` (método interno)
  - `_create_temporal_ncs_comparison()` (método interno)
- **Retorna**: `str` con análisis estructurado por tipo de incidente

---

##### Métodos de Análisis de Rutas (8 métodos):

###### 38. `_routes_tool()`
- **Argumentos**: `node_path, start_date, end_date, min_surveys, anomaly_type`
- **Propósito**: **ASYNC** - Consolida rutas desde múltiples fuentes (explanatory drivers, NCS, verbatims, general)
- **Dependencias**:
  - `_consolidate_routes_from_all_sources()` (método interno)
- **Retorna**: `str` con análisis consolidado de rutas

###### 39. `_consolidate_routes_from_all_sources()`
- **Argumentos**: `node_path, start_dt, end_dt, anomaly_type, min_surveys`
- **Propósito**: **ASYNC** - Consolida rutas desde todas las fuentes disponibles
- **Dependencias**:
  - `_get_explanatory_drivers_routes()` (método interno)
  - `_get_general_routes_with_touchpoints()` (método interno)
  - `_get_ncs_routes()` (método interno)
  - `_get_verbatims_routes()` (método interno)
  - `_create_consolidated_routes_analysis()` (método interno)
- **Retorna**: `dict` con rutas por fuente

###### 40. `_get_explanatory_drivers_routes()`
- **Argumentos**: `node_path, cabins, companies, hauls, start_dt, end_dt, comparison_start_dt, comparison_end_dt`
- **Propósito**: **ASYNC** - Obtiene rutas desde explanatory drivers con SHAP values
- **Dependencias**:
  - `pbi_collector.get_explanatory_drivers_async()`
- **Retorna**: `dict` con rutas y SHAP scores

###### 41. `_get_general_routes_with_touchpoints()`
- **Argumentos**: `cabins, companies, hauls, start_dt, end_dt, comparison_start_dt, comparison_end_dt, min_surveys`
- **Propósito**: **ASYNC** - Obtiene rutas generales con touchpoints NPS desglosados
- **Dependencias**:
  - `pbi_collector.get_routes_async()`
- **Retorna**: `dict` con rutas y NPS por touchpoint

###### 42. `_get_ncs_routes()`
- **Argumentos**: `node_path, cabins, companies, hauls, start_dt, end_dt`
- **Propósito**: **ASYNC** - Obtiene rutas mencionadas en incidentes NCS
- **Dependencias**:
  - `ncs_collector.collect_ncs_data_async()`
  - `_filter_ncs_by_segment()` (método interno)
- **Retorna**: `dict` con rutas y conteo de incidentes

###### 43. `_get_verbatims_routes()`
- **Argumentos**: `node_path, cabins, companies, hauls, start_dt, end_dt`
- **Propósito**: **ASYNC** - Extrae rutas mencionadas en verbatims
- **Dependencias**:
  - `pbi_collector.get_verbatims()`
- **Retorna**: `dict` con rutas mencionadas

###### 44. `_create_consolidated_routes_analysis()`
- **Argumentos**: `exp_drivers_routes, ncs_routes, verbatims_routes, general_routes`
- **Propósito**: Crea análisis consolidado de rutas desde todas las fuentes
- **Dependencias**:
  - `_analyze_route_similarities()` (método interno)
  - `_identify_route_patterns()` (método interno)
- **Retorna**: `str` con análisis estructurado

###### 45. `_analyze_route_similarities()`
- **Argumentos**: `all_routes, exp_drivers_routes, general_routes, ncs_routes, verbatims_routes`
- **Propósito**: Analiza similitudes y patrones entre rutas de diferentes fuentes
- **Dependencias**: Ninguna
- **Retorna**: `dict` con grupos de rutas

###### 46. `_identify_route_patterns()`
- **Argumentos**: `routes: List[str]`
- **Propósito**: Identifica patrones comunes en rutas (origin, destination, haul)
- **Dependencias**: Ninguna
- **Retorna**: `dict` con patrones identificados

---

##### Métodos de Customer Profile (1 método):

###### 47. `_customer_profile_tool()`
- **Argumentos**: `node_path, start_date, end_date, min_surveys, profile_dimension, mode`
- **Propósito**: **ASYNC** - Analiza perfil de clientes por dimensión (ticket type, haul, country, fleet, etc.)
- **Dependencias**:
  - `_collect_customer_profile_with_query_tracking()` (método interno)
  - `_safe_clean_columns()` (método interno)
- **Retorna**: `str` con análisis de customer profile
- **Dimensiones**: Ticket Type, Haul Type, Country, Fleet, Multi-dimensional

---

##### Métodos NCS Avanzados (10 métodos):

###### 48. `_safe_clean_columns()`
- **Argumentos**: `df, method`
- **Propósito**: Limpia columnas del DataFrame de forma segura
- **Retorna**: `pd.DataFrame`

###### 49. `_create_temporal_ncs_comparison()`
- **Argumentos**: `current_data, comparison_data, node_path, current_label, comparison_label`
- **Propósito**: Crea comparación temporal detallada de incidentes NCS
- **Dependencias**:
  - `_extract_structured_ncs_data_for_comparison()` (método interno)
  - `_create_route_incident_matrix()` (método interno)
  - `_calculate_incident_type_deltas()` (método interno)
  - `_identify_improvement_patterns()` (método interno)
  - `_generate_temporal_summary()` (método interno)
- **Retorna**: `str` con análisis comparativo

###### 50. `_extract_structured_ncs_data_for_comparison()`
- **Argumentos**: `data, period_label`
- **Propósito**: Extrae datos NCS estructurados para comparación
- **Retorna**: `dict` con datos estructurados

###### 51. `_create_route_incident_matrix()`
- **Argumentos**: `current, comparison`
- **Propósito**: Crea matriz de incidentes por ruta comparando períodos
- **Retorna**: `dict` con matriz

###### 52. `_calculate_incident_type_deltas()`
- **Argumentos**: `current, comparison`
- **Propósito**: Calcula deltas entre períodos por tipo de incidente
- **Retorna**: `dict` con deltas

###### 53. `_identify_improvement_patterns()`
- **Argumentos**: `route_matrix, type_deltas`
- **Propósito**: Identifica patrones de mejora o deterioro en incidentes
- **Retorna**: `dict` con patrones

###### 54. `_generate_temporal_summary()`
- **Argumentos**: `route_matrix, type_deltas, patterns`
- **Propósito**: Genera resumen ejecutivo de comparación temporal
- **Retorna**: `str` con resumen

###### 55. `_get_nps_impact_validation()`
- **Argumentos**: `incident_type, delta`
- **Propósito**: Valida impacto de incidente en NPS
- **Retorna**: `str` con validación

###### 56. `_filter_ncs_by_segment()`
- **Argumentos**: `ncs_data, node_path`
- **Propósito**: **ASYNC** - Filtra datos NCS por segmento jerárquico
- **Dependencias**: Extracción de cabin/company/haul desde node_path
- **Retorna**: `pd.DataFrame` filtrado

###### 57. `_apply_contextual_filtering()`
- **Argumentos**: `ncs_data, node_path`
- **Propósito**: **ASYNC** - Aplica filtrado contextual avanzado a NCS
- **Retorna**: `pd.DataFrame` filtrado

---

##### Métodos de Gestión de Workflow (15 métodos):

###### 58. `_determine_next_tool_dynamic()`
- **Argumentos**: `current_tool, iteration, max_tools`
- **Propósito**: Determina próximo tool a ejecutar basado en flujo dinámico
- **Dependencias**: Ninguna
- **Retorna**: `Optional[str]` con nombre del tool
- **Flujo estándar**: explanatory_drivers → operative_data → ncs → routes → customer_profile → verbatims

###### 59. `_get_single_period_system_prompt()`
- **Argumentos**: Ninguno
- **Propósito**: Obtiene system prompt para análisis de período único
- **Dependencias**: `config['single_period']['system_prompt']`
- **Retorna**: `str`

###### 60. `_get_single_period_input_template()`
- **Argumentos**: Ninguno
- **Propósito**: Obtiene template de input para análisis de período único
- **Retorna**: `str`

###### 61. `_get_single_period_tool_result_message()`
- **Argumentos**: Ninguno
- **Propósito**: Obtiene mensaje para resultado de tool en modo single period
- **Retorna**: `str`

###### 62. `_execute_single_period_tool()`
- **Argumentos**: `tool_name, node_path, start_date, end_date, **tool_kwargs`
- **Propósito**: **ASYNC** - Ejecuta tool específico para análisis de período único
- **Dependencias**: Mapeo de tool_name a método específico
- **Retorna**: `str` con resultado del tool
- **Tools soportados**: explanatory_drivers, operative_data, ncs, routes, customer_profile, verbatims

###### 63. `_operative_data_tool()`
- **Argumentos**: `node_path, start_date, end_date, comparison_days, comparison_mode, baseline_periods, comparison_context`
- **Propósito**: **ASYNC** - Tool de datos operativos con modo comparativo (LEGACY)
- **Dependencias**:
  - `OperationalDataAnalyzer.analyze_multiple_dates()`
- **Retorna**: `str` con análisis de métricas operativas

###### 64. `_get_metric_impact_summary()`
- **Argumentos**: `metric, direction`
- **Propósito**: Genera resumen de impacto de métrica en NPS
- **Retorna**: `str`

###### 65. `_generate_correlation_summary()`
- **Argumentos**: `metrics, anomaly_type, comparison_mode`
- **Propósito**: Genera resumen de correlaciones entre métricas y anomalía
- **Retorna**: `str`

###### 66. `_metric_supports_anomaly()`
- **Argumentos**: `metric, direction, nps_anomaly_type`
- **Propósito**: Verifica si métrica soporta tipo de anomalía NPS
- **Retorna**: `bool`

###### 67. `_verbatims_tool()`
- **Argumentos**: `node_path, start_date, end_date`
- **Propósito**: **ASYNC** - Tool de verbatims con modo comparativo (LEGACY)
- **Dependencias**:
  - `_collect_verbatims_with_query_tracking()` (método interno)
  - `_analyze_pbi_verbatims()` o `_conduct_chatbot_conversation()` (métodos internos)
- **Retorna**: `str` con análisis de verbatims

###### 68. `_execute_tool_unified()`
- **Argumentos**: `tool_name, node_path, start_date, end_date, **kwargs`
- **Propósito**: **ASYNC** - Ejecutor unificado de tools para modo single period
- **Dependencias**: `_execute_single_period_tool()` (método interno)
- **Retorna**: `str`

###### 69. `_execute_tool_comparative()`
- **Argumentos**: `tool_name, node_path, start_date, end_date, comparison_start_date, comparison_end_date, **kwargs`
- **Propósito**: **ASYNC** - Ejecutor de tools para modo comparativo
- **Dependencias**: Mapeo a tool específico
- **Retorna**: `str`

###### 70. `_determine_next_tool_from_reflection()`
- **Argumentos**: `reflection, iteration, max_tools`
- **Propósito**: **ASYNC** - Determina próximo tool desde reflexión del LLM
- **Dependencias**: 
  - `llm.complete_structured()` → LLM
- **Retorna**: `Optional[str]`
- **Lógica**: Parsea reflexión del LLM y extrae decisión de próximo tool

###### 71. `_get_helper_prompt_for_tool()`
- **Argumentos**: `tool_name: str`
- **Propósito**: Obtiene helper prompt específico para un tool
- **Dependencias**: `config['helper_prompts'][tool_name]`
- **Retorna**: `str`

###### 72. `_find_column()`
- **Argumentos**: `df, possible_names`
- **Propósito**: Encuentra columna en DataFrame desde lista de nombres posibles
- **Retorna**: `str` con nombre de columna encontrado

---

##### Métodos de Reflexión y Síntesis (5 métodos):

###### 73. `_get_clean_reflection()`
- **Argumentos**: `system_prompt, tool_name, tool_result, message_history, mode`
- **Propósito**: **ASYNC** - Obtiene reflexión limpia del LLM sobre resultado de tool
- **Dependencias**:
  - `llm.chat()` → LLM
  - `tracker.get_clean_context()`
- **Retorna**: `Optional[Dict[str, str]]` con reflexión
- **Formato**: `{"explanation": "...", "next_tool": "..."}`

###### 74. `_generate_final_synthesis()`
- **Argumentos**: `node_path, start_date, end_date, anomaly_type, anomaly_magnitude, anomaly_context, mode`
- **Propósito**: **ASYNC** - Genera síntesis final consolidando todas las explicaciones previas
- **Dependencias**:
  - `_build_collected_data_summary()` (método interno)
  - `llm.chat()` → LLM
- **Retorna**: `str` con síntesis ejecutiva final

###### 75. `_build_collected_data_summary()`
- **Argumentos**: Ninguno
- **Propósito**: Construye resumen de todos los datos recolectados en la investigación
- **Dependencias**: `tracker.conversation_log`
- **Retorna**: `str`

###### 76. `_build_single_period_data_summary()`
- **Argumentos**: Ninguno
- **Propósito**: Construye resumen de datos para modo single period
- **Dependencias**: `tracker.conversation_log`
- **Retorna**: `str`

###### 77. `_build_single_period_fallback()`
- **Argumentos**: Ninguno
- **Propósito**: Genera respuesta fallback cuando no hay datos suficientes
- **Retorna**: `str`

###### 78. `_generate_single_period_synthesis()`
- **Argumentos**: `node_path, start_date, end_date, anomaly_type, anomaly_magnitude, anomaly_context`
- **Propósito**: **ASYNC** - Genera síntesis para análisis de período único
- **Dependencias**:
  - `_build_single_period_data_summary()` (método interno)
  - `llm.chat()` → LLM
- **Retorna**: `str`

---

##### Métodos Principales de Investigación (3 métodos):

###### 79. `investigate_anomaly()`
- **Argumentos**: `node_path, start_date, end_date, anomaly_type, anomaly_magnitude, max_tools, anomaly_context, mode`
- **Propósito**: **ASYNC** - **MÉTODO PRINCIPAL** - Orquesta investigación completa de anomalía
- **Dependencias**:
  - `_investigate_anomaly_single_period()` o `_investigate_anomaly_with_comparison()` (métodos internos)
- **Retorna**: `str` con explicación causal completa
- **Modos**: `"single_period"` o `"comparative"`

###### 80. `_investigate_anomaly_single_period()`
- **Argumentos**: `node_path, start_date, end_date, anomaly_type, anomaly_magnitude, max_tools, anomaly_context`
- **Propósito**: **ASYNC** - Investigación para período único con workflow limpio
- **Dependencias**:
  - `tracker.reset_tracker()`
  - `_determine_next_tool_dynamic()` (método interno)
  - `_execute_single_period_tool()` (método interno)
  - `_get_clean_reflection()` (método interno)
  - `_generate_single_period_synthesis()` (método interno)
- **Retorna**: `str` con explicación
- **Workflow**: 
  1. Reset tracker
  2. Loop: Determinar tool → Ejecutar → Obtener reflexión → Añadir explicación
  3. Generar síntesis final

###### 81. `_investigate_anomaly_with_comparison()`
- **Argumentos**: `node_path, start_date, end_date, anomaly_type, anomaly_magnitude, max_tools, anomaly_context`
- **Propósito**: **ASYNC** - Investigación con comparación temporal (LEGACY workflow)
- **Dependencias**:
  - Similar a single_period pero con comparación temporal
  - `_execute_tool_comparative()` (método interno)
- **Retorna**: `str` con explicación comparativa

---

##### Métodos de Export y Logging (4 métodos):

###### 82. `get_conversation_log()`
- **Argumentos**: Ninguno
- **Propósito**: Obtiene log completo de conversación
- **Retorna**: `List[Dict]`

###### 83. `export_conversation()`
- **Argumentos**: `filename, node_path, start_date, end_date`
- **Propósito**: **ASYNC** - Exporta conversación a archivo local y S3
- **Dependencias**:
  - `_get_conversation_summary()` (método interno)
  - `json.dump()`
  - `s3_uploader.upload_conversation()` → S3ReportUploader
- **Retorna**: `str` con path del archivo
- **Formato**: JSON con metadata y conversación completa

###### 84. `_get_conversation_summary()`
- **Argumentos**: Ninguno
- **Propósito**: Genera estadísticas resumidas de la conversación
- **Retorna**: `Dict[str, int]` con conteos por tipo de mensaje

---

### **📊 Estadísticas del Archivo**:

- **Total de Métodos**: 129 métodos documentados
- **Clases**: 3 clases principales
- **Métodos Async**: 45+ métodos asíncronos
- **Líneas de Código**: 6,903 líneas
- **Tools Disponibles**: 6 tools principales (explanatory_drivers, operative_data, ncs, routes, customer_profile, verbatims)

### **🔄 Flujo de Investigación Principal**:

```
investigate_anomaly()
  ↓
  → Calculate dynamic comparison dates (if needed)
  → Reset tracker
  → Initialize message history
  ↓
  LOOP (hasta max_tools):
    1. _determine_next_tool_dynamic() → Selecciona próximo tool
    2. _execute_single_period_tool() → Ejecuta tool y obtiene datos
    3. _get_clean_reflection() → LLM analiza resultado
    4. tracker.add_explanation() → Guarda explicación
    5. tracker.next_iteration() → Incrementa iteración
  ↓
  → _generate_single_period_synthesis() → Síntesis final
  → export_conversation() → Export a S3
  ↓
  RETURN: Explicación causal completa
```

### **🔧 Dependencias Externas Principales**:

- **LLMs**: OpenAI GPT-4/O1, AWS Bedrock Claude/Llama
- **Data Collection**: PBIDataCollector, ChatbotVerbatimsCollector, NCSDataCollector
- **Data Analysis**: OperationalDataAnalyzer
- **Storage**: S3ReportUploader
- **Frameworks**: pandas, pydantic, asyncio, yaml

---

## Archivo: `dashboard_analyzer/anomaly_explanation/genai_core/agents/anomaly_interpreter_agent.py`

### **Agente de Interpretación de Anomalías** (1,187 líneas, 41 métodos)

Agente especializado en interpretar árboles de anomalías NPS y generar conclusiones sobre correlaciones entre desempeño operativo y percepción del cliente.

### Funciones de Utilidad:

#### `find_project_root()`
- **Argumentos**: `marker_file=".git"`
- **Propósito**: Encuentra la raíz del proyecto buscando un archivo marcador
- **Dependencias**: `Path` (pathlib)
- **Retorna**: `Path` o `None`

---

### Clase 1: `HierarchicalConversationTracker`
**Rastrea el flujo de conversación jerárquica** con análisis generación por generación.

#### Métodos (11 métodos):

##### 1. `__init__()`
- **Argumentos**: Ninguno
- **Propósito**: Inicializa tracker con logs, conteo de generaciones, estructura jerárquica, reflexiones
- **Dependencias**: Ninguna

##### 2. `reset_tracker()`
- **Argumentos**: Ninguno
- **Propósito**: Resetea tracker para nueva investigación jerárquica
- **Dependencias**: Ninguna

##### 3. `log_message()`
- **Argumentos**: `message_type: str, content: str, metadata: Optional[Dict] = None`
- **Propósito**: Registra mensaje en conversación jerárquica con timestamp
- **Dependencias**: `datetime.now()`

##### 4. `set_hierarchy_structure()`
- **Argumentos**: `hierarchy: Dict[str, Any]`
- **Propósito**: Almacena estructura jerárquica parseada del árbol de anomalías
- **Dependencias**: Ninguna

##### 5. `start_generation()`
- **Argumentos**: `generation_level: int, nodes: List[str]`
- **Propósito**: Inicia análisis de una nueva generación en la jerarquía
- **Dependencias**: `log_message()` (método interno)

##### 6. `add_generation_reflection()`
- **Argumentos**: `generation: int, nodes: List[str], reflection: str`
- **Propósito**: Añade reflexión del LLM para una generación específica
- **Dependencias**: `log_message()` (método interno)

##### 7. `add_generation_analysis()`
- **Argumentos**: `generation: int, analysis_data: Dict[str, Any]`
- **Propósito**: Añade datos de análisis detallado para una generación
- **Dependencias**: Ninguna

##### 8. `get_conversation_summary()`
- **Argumentos**: Ninguno
- **Propósito**: Obtiene resumen de tipos de mensaje en la conversación
- **Dependencias**: Ninguna
- **Retorna**: `Dict[str, int]` con conteos por tipo

##### 9. `get_hierarchy_summary()`
- **Argumentos**: Ninguno
- **Propósito**: Obtiene resumen de la jerarquía analizada
- **Dependencias**: Ninguna
- **Retorna**: `Dict[str, Any]` con:
  - `total_nodes`: Número total de nodos
  - `generations_analyzed`: Generaciones analizadas
  - `nodes_by_level`: Nodos agrupados por nivel
  - `parent_child_relationships`: Relaciones jerárquicas

---

### Clase 2: `AnomalyInterpreterAgent`
**Agente especializado** para interpretar árboles de anomalías y generar conclusiones accionables.

**Features**:
- Soporte multi-LLM (OpenAI, AWS Bedrock)
- Configuración de prompts externa vía YAML
- Análisis jerárquico generación por generación con reflexiones AI
- Tracking y export de conversaciones

#### Constructor y Configuración (8 métodos):

##### 1. `__init__()`
- **Argumentos**:
  - `llm_type: Optional[LLMType] = None`
  - `config_path: str = "dashboard_analyzer/anomaly_explanation/config/prompts/anomaly_interpreter.yaml"`
  - `logger: Optional[logging.Logger] = None`
  - `study_mode: str = "comparative"`
- **Propósito**: Inicializa agente intérprete con LLM, configuración de prompts, y tracking jerárquico
- **Dependencias**:
  - `get_default_llm_type()` → genai_core.utils.enums
  - `find_project_root()` (función de utilidad)
  - `_setup_logger()` (método interno)
  - `_load_prompt_config()` (método interno)
  - `_create_llm()` (método interno)
  - `Agent()` → genai_core.agents.agent
  - `HierarchicalConversationTracker()` (clase interna)
  - `S3ReportUploader()` → data_collection.s3_report_uploader
- **Atributos**:
  - `hierarchical_reflections`: Lista de reflexiones por generación
  - `generation_data`: Datos por generación
  - `conversation_tracker`: Tracker jerárquico
  - `total_processing_time`: Métricas de desempeño
  - `total_hierarchical_calls`: Conteo de llamadas

##### 2. `_setup_logger()`
- **Argumentos**: Ninguno
- **Propósito**: Configura logger por defecto para el agente
- **Dependencias**: `logging` (stdlib)
- **Retorna**: `logging.Logger`

##### 3. `_load_prompt_config()`
- **Argumentos**: `config_path: str`
- **Propósito**: Carga configuración de prompts desde archivo YAML
- **Dependencias**: `yaml.safe_load()`, `Path`
- **Retorna**: `Dict[str, Any]`

##### 4. `_get_system_prompt()`
- **Argumentos**: `mode: str = None`
- **Propósito**: Obtiene system prompt desde configuración según modo (single/comparative)
- **Dependencias**: `config['system_prompt']`
- **Retorna**: `str`

##### 5. `_get_input_template()`
- **Argumentos**: `template_name: str, mode: str = None`
- **Propósito**: Obtiene template de input específico
- **Dependencias**: `config['input_templates']`
- **Retorna**: `str`

##### 6. `_get_hierarchical_helper()`
- **Argumentos**: `step_name: str`
- **Propósito**: Obtiene helper prompt para paso jerárquico específico
- **Dependencias**: `config['hierarchical_helpers']`
- **Retorna**: `str`

##### 7. `_create_llm()`
- **Argumentos**: `llm_type: LLMType`
- **Propósito**: Factory method para crear instancia de LLM apropiada
- **Dependencias**: `_create_openai_llm()` o `_create_aws_llm()`
- **Retorna**: `OpenAiLLM` o `AWSLLM`

##### 8. `_create_openai_llm()`
- **Argumentos**: `llm_type: LLMType`
- **Propósito**: Crea instancia de OpenAI/Azure OpenAI LLM
- **Dependencias**: `OpenAiLLM()` → genai_core.llms.openai_llm
- **Retorna**: `OpenAiLLM`
- **Modelos soportados**: GPT-3.5, GPT-4, GPT-4o, GPT-4o-MINI, O1-MINI, O3-MINI, O3, O4-MINI

##### 9. `_create_aws_llm()`
- **Argumentos**: `llm_type: LLMType`
- **Propósito**: Crea instancia de AWS Bedrock LLM
- **Dependencias**: `AWSLLM()` → genai_core.llms.aws_llm
- **Retorna**: `AWSLLM`
- **Modelos soportados**: Claude (Haiku, Sonnet, Opus), Llama (70B, 405B)

---

#### Métodos de Interpretación (6 métodos):

##### 10. `interpret_anomaly_tree()`
- **Argumentos**: `tree_data: str, date: Optional[str] = None, segment: Optional[str] = None`
- **Propósito**: **ASYNC** - Interpreta árbol de anomalías completo (modo básico, una pasada)
- **Dependencias**:
  - `_get_system_prompt()` (método interno)
  - `_get_input_template()` (método interno)
  - `llm.chat()` → LLM
- **Retorna**: `str` con interpretación ejecutiva
- **Uso**: Modo simple sin análisis jerárquico

##### 11. `interpret_anomaly_tree_hierarchical()`
- **Argumentos**: `tree_data: str, date: Optional[str] = None, segment: Optional[str] = None`
- **Propósito**: **ASYNC** - **MÉTODO PRINCIPAL** - Interpreta árbol de anomalías con análisis generación por generación
- **Dependencias**:
  - `_parse_hierarchy_from_explanations()` (método interno)
  - `_order_generations_bottom_up()` (método interno)
  - `_generate_generation_helper_prompt()` (método interno)
  - `_generate_comprehensive_summary_prompt()` (método interno)
  - `_generate_synthesis_prompt()` (método interno)
  - `_compile_final_interpretation()` (método interno)
  - `export_hierarchical_conversation()` (método interno)
  - `llm.chat()` → LLM
- **Retorna**: `str` con interpretación jerárquica completa
- **Workflow**:
  1. Parsea jerarquía desde tree_data
  2. Ordena generaciones bottom-up (hojas → raíz)
  3. Analiza cada generación con contexto de hijos
  4. Genera summary comprehensivo
  5. Genera synthesis final ejecutiva
  6. Exporta conversación a S3

##### 12. `_parse_hierarchy_from_explanations()`
- **Argumentos**: `tree_data: str`
- **Propósito**: Parsea estructura jerárquica desde datos del árbol de anomalías
- **Dependencias**: `_infer_node_path()` (método interno)
- **Retorna**: `Dict[str, Any]` con estructura:
  - `node_path`: Ruta jerárquica del nodo
  - `level`: Nivel en la jerarquía
  - `parent`: Nodo padre
  - `children`: Lista de nodos hijos
  - `anomaly_data`: Datos de anomalía del nodo
- **Lógica**: Extrae nodos usando regex, infiere paths jerárquicos, construye relaciones

##### 13. `_infer_node_path()`
- **Argumentos**: `node_name: str, tree_data: str, segment_context: str`
- **Propósito**: Infiere path jerárquico completo de un nodo basado en contexto
- **Dependencias**: Ninguna
- **Retorna**: `str` con node_path (ej: "Business/IB/Long Haul")
- **Lógica**: Busca contexto en tree_data, mapea nombres a paths estándar

##### 14. `_format_hierarchy_structure()`
- **Argumentos**: `hierarchy: Dict[str, Any]`
- **Propósito**: Formatea estructura jerárquica para visualización
- **Dependencias**: Función interna `format_node()` (recursiva)
- **Retorna**: `str` con jerarquía formateada con indentación

##### 15. `_order_generations_bottom_up()`
- **Argumentos**: `hierarchy: Dict[str, Any]`
- **Propósito**: Ordena generaciones de abajo hacia arriba (hojas primero, raíz último)
- **Dependencias**: Ninguna
- **Retorna**: `List[List[Dict[str, Any]]]` con nodos agrupados por nivel
- **Lógica**: Agrupa por nivel, ordena de mayor a menor nivel

---

#### Métodos de Generación de Prompts (3 métodos):

##### 16. `_generate_generation_helper_prompt()`
- **Argumentos**: `generation_level: int, nodes: List[Dict[str, Any]], hierarchy: Dict[str, Any]`
- **Propósito**: Genera prompt helper para análisis de generación específica
- **Dependencias**: `_get_hierarchical_helper()` (método interno)
- **Retorna**: `str` con prompt contextualizado

##### 17. `_generate_comprehensive_summary_prompt()`
- **Argumentos**: `hierarchy: Dict[str, Any]`
- **Propósito**: Genera prompt para summary comprehensivo de todas las generaciones
- **Dependencias**: `_get_hierarchical_helper()` (método interno)
- **Retorna**: `str`

##### 18. `_generate_synthesis_prompt()`
- **Argumentos**: `hierarchy: Dict[str, Any]`
- **Propósito**: Genera prompt para síntesis final ejecutiva
- **Dependencias**: `_get_hierarchical_helper()` (método interno)
- **Retorna**: `str`

---

#### Métodos de Compilación y Export (5 métodos):

##### 19. `_compile_final_interpretation()`
- **Argumentos**: `step_responses: List[Dict[str, str]], hierarchy: Dict[str, Any]`
- **Propósito**: Compila interpretación final desde todas las respuestas de pasos jerárquicos
- **Dependencias**: Ninguna
- **Retorna**: `str` con interpretación completa estructurada
- **Secciones**:
  1. Análisis generación por generación
  2. Summary comprehensivo
  3. Síntesis ejecutiva

##### 20. `export_hierarchical_conversation()`
- **Argumentos**: `date: Optional[str] = None, error: Optional[str] = None`
- **Propósito**: Exporta conversación jerárquica a JSON local y S3
- **Dependencias**:
  - `conversation_tracker.get_conversation_summary()`
  - `conversation_tracker.get_hierarchy_summary()`
  - `json.dump()`
- **Retorna**: `str` con path del archivo
- **Formato JSON**:
  - metadata: timestamp, llm_type, study_mode, métricas
  - hierarchy_summary: estructura y estadísticas
  - conversation_log: Log completo
  - hierarchical_reflections: Reflexiones por generación

##### 21. `export_conversation()`
- **Argumentos**: `message_history, date, segment, error`
- **Propósito**: **ASYNC** - Exporta conversación a S3 (modo básico)
- **Dependencias**:
  - `s3_uploader.upload_interpretation()` → S3ReportUploader
  - `json.dumps()`
- **Retorna**: `str` con S3 path

##### 22. `get_desempeño_metrics()`
- **Argumentos**: Ninguno
- **Propósito**: Obtiene métricas de desempeño del agente
- **Retorna**: `Dict[str, Any]` con:
  - `total_processing_time`: Tiempo total
  - `total_hierarchical_calls`: Llamadas totales
  - `avg_time_per_call`: Tiempo promedio

---

#### Métodos de Segmentación (5 métodos):

##### 23. `_get_applicable_steps_for_segment()`
- **Argumentos**: `segment: str`
- **Propósito**: Determina qué pasos de análisis aplicar según el segmento
- **Dependencias**: `_detect_segment_level()` (método interno)
- **Retorna**: `List[str]` con nombres de pasos
- **Lógica**:
  - Nivel 1 (Total): Todos los pasos
  - Nivel 2 (Cabin): Pasos de cabin
  - Nivel 3+ (Company/Haul): Pasos específicos

##### 24. `_extract_primary_segment_from_data()`
- **Argumentos**: `tree_data: str`
- **Propósito**: Extrae segmento primario desde datos del árbol
- **Dependencias**: Ninguna
- **Retorna**: `str` con segmento (ej: "Business", "Economy")

##### 25. `_detect_segment_level()`
- **Argumentos**: `segment: str`
- **Propósito**: Detecta nivel jerárquico del segmento
- **Dependencias**: Ninguna
- **Retorna**: `str` ("level_1", "level_2", "level_3", "level_4")
- **Niveles**:
  - Level 1: Total/Global
  - Level 2: Cabin (Business, Economy, Premium Economy)
  - Level 3: Company + Haul
  - Level 4: Country/Region

##### 26. `_get_cabin_sections_for_segment()`
- **Argumentos**: `segment: str`
- **Propósito**: Obtiene secciones de cabin aplicables para el segmento
- **Dependencias**: Ninguna
- **Retorna**: `str` con descripción de secciones

---

### **📊 Estadísticas del Archivo**:

- **Total de Métodos**: 41 métodos documentados
- **Clases**: 2 clases principales
- **Métodos Async**: 3 métodos asíncronos
- **Líneas de Código**: 1,187 líneas
- **Workflow Principal**: Análisis jerárquico bottom-up con reflexiones AI por generación

---

## Archivo: `dashboard_analyzer/anomaly_explanation/genai_core/agents/anomaly_summary_agent.py`

### **Agente de Summary Multi-Período** (617 líneas, 17 métodos)

Agente especializado en resumir múltiples períodos de análisis de anomalías NPS y generar insights ejecutivos sobre tendencias, patrones y prioridades estratégicas.

### Clase: `AnomalySummaryAgent`
**Agente especializado** para resumir análisis multi-período y generar reportes estratégicos.

**Features**:
- Análisis de tendencias multi-período
- Identificación de patrones a través del tiempo
- Evaluación de prioridades estratégicas
- Reporting ejecutivo
- Soporte multi-LLM (OpenAI, AWS Bedrock)

#### Constructor y Configuración (7 métodos):

##### 1. `__init__()`
- **Argumentos**:
  - `llm_type: Optional[LLMType] = None`
  - `config_path: str = "../../config/prompts/anomaly_summary.yaml"`
  - `logger: Optional[logging.Logger] = None`
- **Propósito**: Inicializa agente de summary con LLM y configuración de prompts
- **Dependencias**:
  - `get_default_llm_type()` → genai_core.utils.enums
  - `_setup_logger()` (método interno)
  - `_load_prompt_config()` (método interno)
  - `_create_llm()` (método interno)
  - `Agent()` → genai_core.agents.agent
  - `S3ReportUploader()` → data_collection.s3_report_uploader
- **Atributos**:
  - `silent_mode`: Modo silencioso para suprimir logs

##### 2. `_setup_logger()`
- **Argumentos**: Ninguno
- **Propósito**: Configura logger por defecto
- **Dependencias**: `logging` (stdlib)
- **Retorna**: `logging.Logger`

##### 3. `_load_prompt_config()`
- **Argumentos**: `config_path: str`
- **Propósito**: Carga configuración de prompts desde archivo YAML
- **Dependencias**: `yaml.safe_load()`
- **Retorna**: `Dict[str, Any]`

##### 4. `_create_llm()`
- **Argumentos**: `llm_type: LLMType`
- **Propósito**: Factory method para crear instancia de LLM
- **Dependencias**: `_create_openai_llm()` o `_create_aws_llm()`
- **Retorna**: `OpenAiLLM` o `AWSLLM`

##### 5. `_create_openai_llm()`
- **Argumentos**: `llm_type: LLMType`
- **Propósito**: Crea instancia de OpenAI/Azure OpenAI LLM
- **Dependencias**: `OpenAiLLM()` → genai_core.llms.openai_llm
- **Retorna**: `OpenAiLLM`
- **Modelos soportados**: GPT-3.5, GPT-4, GPT-4o, O1, O3, O4-MINI

##### 6. `_create_aws_llm()`
- **Argumentos**: `llm_type: LLMType`
- **Propósito**: Crea instancia de AWS Bedrock LLM
- **Dependencias**: `AWSLLM()` → genai_core.llms.aws_llm
- **Retorna**: `AWSLLM`
- **Modelos soportados**: Claude, Llama

---

#### Métodos de Generación de Summary (4 métodos):

##### 7. `generate_summary_report()`
- **Argumentos**: `periods_data: List[Dict[str, Any]]`
- **Propósito**: **ASYNC** - Genera reporte de summary comprehensivo desde múltiples períodos
- **Dependencias**:
  - `_format_periods_for_summary()` (método interno)
  - `llm.chat()` → LLM
- **Retorna**: `str` con summary ejecutivo
- **Input esperado**: Lista de dicts con `{'period', 'anomalies', 'explanations', 'interpretations'}`

##### 8. `generate_comprehensive_summary()`
- **Argumentos**: `periods_data, consolidated_input, message_history, dateflight_local`
- **Propósito**: **ASYNC** - Genera summary comprehensivo con contexto consolidado
- **Dependencias**:
  - `_format_periods_for_summary()` (método interno)
  - `_get_message_history_for_consolidated()` (método interno)
  - `_extract_specific_examples()` (método interno)
  - `llm.chat()` → LLM
  - `export_conversation()` (método interno)
- **Retorna**: `str` con summary comprehensivo
- **Workflow**:
  1. Formatea períodos
  2. Genera prompt consolidado
  3. Llama LLM con contexto completo
  4. Extrae ejemplos específicos
  5. Exporta conversación a S3

##### 9. `_format_periods_for_summary()`
- **Argumentos**: `periods_data: List[Dict[str, Any]]`
- **Propósito**: Formatea datos de períodos para análisis de summary
- **Dependencias**: Ninguna
- **Retorna**: `str` con períodos formateados
- **Formato**: Cada período incluye:
  - Período de análisis
  - Anomalías detectadas
  - Explicaciones causales
  - Interpretaciones

##### 10. `_extract_specific_examples()`
- **Argumentos**: `ai_interpretation: str`
- **Propósito**: Extrae ejemplos específicos de la interpretación AI
- **Dependencias**: `re` (regex)
- **Retorna**: `str` con ejemplos extraídos
- **Patterns**: Busca menciones de rutas, fechas, métricas específicas

---

#### Métodos de Utilidad (4 métodos):

##### 11. `get_performance_metrics()`
- **Argumentos**: Ninguno
- **Propósito**: Obtiene métricas de desempeño del agente
- **Retorna**: `Dict[str, Any]` con métricas del LLM subyacente

##### 12. `_get_message_history_for_consolidated()`
- **Argumentos**: `consolidated_input: str`
- **Propósito**: Construye message history para análisis consolidado
- **Dependencias**: `MessageHistory()` → genai_core.message_history
- **Retorna**: `MessageHistory`

##### 13. `_get_message_role()`
- **Argumentos**: `message`
- **Propósito**: Extrae rol del mensaje (user/assistant/system)
- **Retorna**: `str`

##### 14. `export_conversation()`
- **Argumentos**: `message_history, dateflight_local, error`
- **Propósito**: **ASYNC** - Exporta conversación a S3
- **Dependencias**:
  - `s3_uploader.upload_summary()` → S3ReportUploader
  - `json.dumps()`
- **Retorna**: `str` con S3 path

---

### **📊 Estadísticas del Archivo**:

- **Total de Métodos**: 17 métodos documentados
- **Clases**: 1 clase principal
- **Métodos Async**: 3 métodos asíncronos
- **Líneas de Código**: 617 líneas
- **Propósito**: Summary ejecutivo multi-período con análisis de tendencias

---

## Archivo: `dashboard_analyzer/anomaly_explanation/genai_core/agents/agent.py`

### **Clase Base de Agentes** (167 líneas, 6 métodos)

Clase base genérica para todos los agentes GenAI. Proporciona funcionalidades comunes y estructura estándar.

### Clase: `Agent`
**Clase base genérica** que puede usarse directamente o extenderse para agentes especializados.

**Funcionalidades Built-in**:
- Tool/function calling
- Structured output con Pydantic
- Performance tracking (tokens, costo, tiempo)
- Streaming responses

#### Métodos (6 métodos):

##### 1. `__init__()`
- **Argumentos**:
  - `llm` - Instancia del modelo de lenguaje
  - `logger: Optional[logging.Logger] = None`
- **Propósito**: Inicializa agente con LLM y tracking de desempeño
- **Dependencias**: Ninguna
- **Atributos de Tracking**:
  - `total_time`: Tiempo total de ejecución
  - `last_execution_time`: Tiempo de última ejecución
  - `avg_time`: Tiempo promedio por llamada
  - `num_calls`: Número de llamadas
  - `call_times`: Lista de tiempos de llamada
  - `input_tokens`: Tokens de input acumulados
  - `output_tokens`: Tokens de output acumulados
  - `money_spent`: Costo acumulado

##### 2. `invoke()`
- **Argumentos**:
  - `messages: list` - Historial de mensajes
  - `structured_output: Optional[Type[BaseModel]] = None` - Modelo Pydantic para output estructurado
  - `tools: Optional[List[Any]] = None` - Lista de tools disponibles
- **Propósito**: **ASYNC** - Procesa input y genera respuesta usando LLM (non-streaming)
- **Dependencias**:
  - `llm()` - Llamada al LLM
  - `_update_metrics()` (método interno)
- **Retorna**: `Tuple[response, structured_response, tool_calls]`
- **Tracking**: Actualiza tokens, costo, tiempo de ejecución

##### 3. `ainvoke()`
- **Argumentos**: `messages: list` - Historial de mensajes
- **Propósito**: **ASYNC GENERATOR** - Procesa input y genera respuesta usando LLM (streaming)
- **Dependencias**:
  - `llm.stream_response()` → LLM
  - `_update_metrics()` (método interno)
- **Yields**: `str` - Chunks de respuesta
- **Uso**: Para UI con respuestas en tiempo real

##### 4. `_update_metrics()`
- **Argumentos**: `execution_time: float`
- **Propósito**: Actualiza métricas de desempeño del agente
- **Dependencias**: Ninguna
- **Actualiza**:
  - `num_calls`: Incrementa contador
  - `last_execution_time`: Guarda último tiempo
  - `total_time`: Acumula tiempo total
  - `call_times`: Añade a lista
  - `avg_time`: Recalcula promedio

##### 5. `execute_tools()`
- **Argumentos**:
  - `mcp_manager` - Instancia de MCPClientManager
  - `tool_calls` - Tool calls a ejecutar
- **Propósito**: **ASYNC** - Ejecuta tool calls usando MCP manager
- **Dependencias**: `mcp_manager.execute_tool_calls()`
- **Retorna**: `List` con respuestas de tools
- **Uso**: Integración con Model Context Protocol (MCP) para herramientas externas

---

### **📊 Estadísticas del Archivo**:

- **Total de Métodos**: 6 métodos documentados
- **Clases**: 1 clase base
- **Métodos Async**: 3 métodos asíncronos
- **Líneas de Código**: 167 líneas
- **Propósito**: Clase base reutilizable para todos los agentes del sistema

---

## **🎯 Resumen de Agentes Documentados**:

### **Total: 4 Agentes** en `genai_core/agents/`:

1. **`agent.py`** - Clase base (6 métodos)
   - Funcionalidad común para todos los agentes
   - Tracking de desempeño
   - Tool calling y structured output

2. **`causal_explanation_agent.py`** - Agente principal (84 métodos)
   - Investigación causal de anomalías
   - 6 tools de recolección de datos
   - Workflow con reflexiones AI

3. **`anomaly_interpreter_agent.py`** - Intérprete jerárquico (41 métodos)
   - Análisis bottom-up de árboles de anomalías
   - Reflexiones AI por generación
   - Export a S3

4. **`anomaly_summary_agent.py`** - Summary multi-período (17 métodos)
   - Análisis de tendencias temporales
   - Reportes ejecutivos
   - Prioridades estratégicas

### **Total Combinado**:
- **148 métodos documentados** en módulo de agentes
- **4 clases principales** + 2 clases auxiliares
- **~9,000 líneas de código**
- Soporte para **8+ modelos LLM** (OpenAI GPT, O-series, Claude, Llama)

---


