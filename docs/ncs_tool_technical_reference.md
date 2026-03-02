# NCS Tool - Referencia Técnica Detallada

## 1. Descripción General

La `ncs_tool` es una herramienta del `CausalExplanationAgent` que recopila y analiza incidentes operacionales (NCS - Net Customer Satisfaction) desde archivos HTML almacenados en S3. Su objetivo es identificar causas operativas (cancelaciones, retrasos, desvíos, etc.) que puedan explicar anomalías en el NPS de un segmento.

Existen dos variantes:
- `_ncs_tool` — modo comparativo (compara período actual vs período de referencia)
- `_ncs_tool_single_period` — modo single (solo analiza el período actual, sin comparación)

---

## 2. Arquitectura de Componentes

```
CausalExplanationAgent
│
├── _ncs_tool()                    # Modo comparativo
├── _ncs_tool_single_period()      # Modo single
│
├── NCSDataCollector               # Recopilación de datos desde S3
│   ├── collect_ncs_data_for_date_range()
│   ├── read_ncs_file()
│   ├── _parse_html_email_content()
│   └── analyze_ncs_incidents_for_period()
│
├── _filter_ncs_by_segment()       # Filtrado por segmento NPS
│   └── _apply_contextual_filtering()
│       └── _filter_using_routes_dictionary()
│
├── _create_temporal_ncs_comparison()  # Comparación temporal
│   ├── _extract_structured_ncs_data_for_comparison()
│   ├── _create_route_incident_matrix()
│   └── _calculate_incident_type_deltas()
│
└── _ncs_reflection_with_agent()   # Análisis causal con LLM
```

---

## 3. Fuente de Datos: S3

### Bucket y Prefijo

| Parámetro | Valor |
|-----------|-------|
| Bucket | `ibdata-prod-ew1-s3-customer` |
| Prefijo base | `customer/catia/ncs/raw/attatchments/` |
| Formato de archivo | `.txt` (contenido HTML de emails) |
| Naming pattern | `ndc-{YYYY-MM-DD}...` |

### Credenciales AWS

El `NCSDataCollector` usa `get_aws_session()` con `use_sandbox=False` (credenciales estándar, no sandbox):

```python
self.session = get_aws_session(environment=self.environment, use_sandbox=False)
self.s3_client = self.session.client('s3')
```

Resolución de credenciales:
- `environment="local"` → carga desde `temp_aws_credentials.env`
- `environment="prod"` → variables de entorno del sistema (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) o IAM roles

---

## 4. Recopilación de Datos

### `collect_ncs_data_for_date_range(start_date, end_date)`

Itera día a día desde `start_date` hasta `end_date`:

1. Para cada día, llama a `list_available_files(date_str)` que hace `list_objects_v2` en S3 con prefijo `customer/catia/ncs/raw/attatchments/ndc-{YYYY-MM-DD}`
2. Para cada archivo `.txt` encontrado, llama a `read_ncs_file(file_key)`
3. Concatena todos los DataFrames resultantes

### `read_ncs_file(file_key)`

1. Descarga el archivo de S3 con `get_object`
2. Decodifica como UTF-8
3. Parsea el HTML con BeautifulSoup

### Parsing del HTML

El contenido son emails HTML con datos de incidentes operacionales. El parsing sigue este orden:

1. **Metadata del email**: extrae `Asunto`, `Enviados` (fecha), `De` (remitente) con regex sobre el HTML
2. **Tablas HTML**: busca `<table>` y extrae headers + filas como diccionarios
3. **Fallback texto**: si no hay tablas, busca patrones en texto plano:
   - Números de vuelo: `IB\d+`
   - Rutas: `[A-Z]{3}-[A-Z]{3}`
   - Horarios: `\d{1,2}:\d{2}`
   - Keywords de incidentes: `delay`, `cancel`, `incident`, `problem`, `issue`, `technical`

---

## 5. Flujo de la NCS Tool (Modo Comparativo)

### Firma

```python
async def _ncs_tool(
    self,
    node_path: str,          # Segmento NPS (ej: "Global/SH/Economy")
    start_date: str,          # Fecha inicio YYYY-MM-DD
    end_date: str,            # Fecha fin YYYY-MM-DD
    analysis_focus: str = "flights",
    temporal_comparison: bool = True
) -> str
```

### Flujo paso a paso

```
1. Convertir fechas string → datetime

2. Calcular período de comparación:
   ├── Si hay comparison_start_date/comparison_end_date explícitos → usarlos
   └── Si no → período anterior de misma duración
       Ej: actual 13-19 ene → comparación 06-12 ene

3. Recopilar datos NCS período actual:
   └── NCSDataCollector.collect_ncs_data_for_date_range(start_dt, end_dt)

4. Filtrar por segmento (período actual):
   └── _filter_ncs_by_segment(ncs_data, node_path)

5. Recopilar datos NCS período comparación (si temporal_comparison=True):
   └── NCSDataCollector.collect_ncs_data_for_date_range(comp_start, comp_end)

6. Filtrar por segmento (período comparación):
   └── _filter_ncs_by_segment(comparison_data, node_path)

7. Comparación temporal:
   └── _create_temporal_ncs_comparison(current, comparison, ...)
       ├── Extrae datos estructurados de ambos períodos
       ├── Crea matriz Ruta × Tipo de Incidente con deltas
       └── Identifica patrones de mejora/empeoramiento

8. Extraer datos estructurados del período actual:
   └── _extract_structured_ncs_data(all_incidents_text)

9. Análisis causal con LLM:
   └── _ncs_reflection_with_agent(filtered_data, ...)
       ├── Si falla → fallback a _extract_ncs_causal_insights_workflow_aware()
       └── Retorna: identified_causes, affected_routes, confidence_level

10. Construir resultado formateado (string)

11. Almacenar en self.collected_data['ncs_data']
```

### Casos de resultado

| Caso | Condición | Resultado |
|------|-----------|-----------|
| Error AWS | Credenciales inválidas/expiradas | `"❌ NCS data source unavailable..."` |
| Sin datos | `ncs_data.empty` tras recopilación | `"📅 No NCS operational incidents found..."` (puede indicar buena operación) |
| Sin datos segmento | Hay incidentes globales pero no en el segmento | `"📊 Se encontraron X incidentes pero ninguno afectó {node_path}"` |
| Datos encontrados | Incidentes en el segmento | Análisis completo con causas, rutas, comparación temporal |

---

## 6. Flujo de la NCS Tool (Modo Single)

### Firma

```python
async def _ncs_tool_single_period(
    self,
    node_path: str,
    start_date: str,
    end_date: str
) -> str
```

### Diferencias con modo comparativo

- No calcula período de comparación
- No hace `_create_temporal_ncs_comparison`
- No usa `_ncs_reflection_with_agent`
- Usa directamente `analyze_ncs_incidents_for_period()` del `NCSDataCollector`
- Resultado más simple: incidentes absolutos sin deltas

---

## 7. Filtrado por Segmento

### `_filter_ncs_by_segment(ncs_data, node_path)`

Filtra los incidentes NCS para que solo queden los relevantes al segmento analizado.

### Lógica de filtrado (3 pasos)

**Paso 1: Filtro por haul (siempre)**

Si el segmento es LH o SH (no Global), filtra usando el diccionario de rutas para determinar qué rutas son Long Haul vs Short Haul. Usa `_filter_using_routes_dictionary()`.

Para nodos no-Global, separa los incidentes en:
- **Atribuibles**: incidentes cuya ruta pertenece al haul del segmento
- **No atribuibles**: incidentes sin ruta identificable (se guardan aparte en `_ncs_unattributed` para que el LLM los evalúe)

Para Global: mantiene todos los incidentes.

**Paso 2: Filtro por cabina (solo a nivel cabina)**

Solo excluye incidentes que mencionan explícitamente que afectan SOLO a otra cabina (con lenguaje exclusivo como "solo", "únicamente", "exclusivamente"). Los incidentes que no mencionan cabina se mantienen.

**Paso 3: Fallback**

Si no quedan incidentes tras filtrar y el nodo es Global, devuelve todos los datos originales.

---

## 8. Comparación Temporal

### `_create_temporal_ncs_comparison(current, comparison, ...)`

Genera una comparación estructurada entre dos períodos:

1. **Extracción estructurada** de ambos períodos (`_extract_structured_ncs_data_for_comparison`):
   - Total de incidentes
   - Incidentes por ruta
   - Incidentes por tipo (cancelaciones, retrasos, desvíos, limitación aeronave, otras)

2. **Matriz Ruta × Tipo de Incidente** (`_create_route_incident_matrix`):
   - Para cada ruta presente en cualquiera de los dos períodos
   - Para cada tipo de incidente: valor actual, valor previo, delta absoluto, % cambio

3. **Deltas globales por tipo** (`_calculate_incident_type_deltas`):
   - Suma total de cada tipo de incidente en ambos períodos
   - Delta y porcentaje de cambio

4. **Patrones de mejora/empeoramiento** (`_identify_improvement_patterns`)

### Tipos de incidentes rastreados

| Tipo | Columnas NCS mapeadas |
|------|----------------------|
| `cancelaciones` | `Cancelaciones`, `Cancel` |
| `retrasos` | `Retrasos`, `Delay` |
| `desvios` | `Desvíos`, `Desvios`, `Divert` |
| `limitacion_aeronave` | `Limitación de la aeronave`, `Aircraft` |
| `otras_incidencias` | `Otras incidencias`, `Other`, `Otros` |

---

## 9. Análisis de Incidentes

### `analyze_ncs_incidents_for_period(df, analysis_focus)`

Método del `NCSDataCollector` que analiza un DataFrame de incidentes y produce:

```python
{
    "total_incidents": int,
    "incident_counts": {          # Conteo por tipo
        "cancelaciones": int,
        "retrasos": int,
        ...
    },
    "flight_analysis": {          # Vuelos afectados
        "total_flights_affected": int,
        "most_affected_flights": {"IB1234": 3, ...}
    },
    "detailed_incidents": {       # Incidentes con descripción larga
        "count": int,
        "sample_incidents": [...],
        "incident_themes": [...]   # Temas: technical_issues, weather, delays, etc.
    },
    "route_analysis": {           # Rutas afectadas
        "total_routes_affected": int,
        "most_affected_routes": {"MAD-JFK": 5, ...},
        "all_routes": [...]
    },
    "summary_insights": [...]     # Resumen legible
}
```

### Extracción de conteos por tipo (`_extract_incident_counts_from_data`)

Dos estrategias:
1. **Por columnas**: busca columnas del DataFrame que coincidan con nombres de tipos de incidentes
2. **Fallback por texto**: si no hay columnas específicas, cuenta keywords en el texto de la primera columna

### Temas de incidentes (`_extract_incident_themes`)

Clasifica incidentes por keywords:

| Tema | Keywords |
|------|----------|
| `technical_issues` | técnica, technical, avería, breakdown, maintenance |
| `weather` | weather, tiempo, meteorológica, tormenta, storm |
| `bird_strike` | aves, bird, impacto, strike |
| `delays` | retraso, delay, delayed, tardanza |
| `cancellations` | cancel, anulado, cancelled |
| `aircraft_change` | cambio, change, aircraft, avión |
| `baggage` | equipaje, baggage, maleta, luggage |
| `crew` | tripulación, crew, piloto, pilot |
| `passenger` | pasajero, passenger, cliente, customer |

---

## 10. Integración con el Agente Causal

### Cuándo se invoca

La `ncs_tool` es una de las herramientas disponibles en el flujo de investigación del `CausalExplanationAgent`. El LLM decide qué herramienta usar en cada iteración:

```python
valid_tools = [
    'explanatory_drivers_tool',
    'operative_data_tool',
    'ncs_tool',              # ← Esta
    'verbatims_tool',
    'routes_tool',
    'customer_profile_tool'
]
```

### Dispatch

En el método `_execute_tool_for_period`, el dispatch se hace así:

```python
if tool_name == "ncs_tool":
    # Modo single
    return await self._ncs_tool_single_period(node_path, start_date, end_date)

# En el flujo comparativo:
if tool_name == "ncs_tool":
    return await self._ncs_tool(node_path, start_date, end_date)
```

### Datos almacenados

Tras la ejecución, los resultados se guardan en `self.collected_data['ncs_data']` con esta estructura:

```python
{
    'analysis_summary': str,
    'identified_causes': [str, ...],        # Causas raíz identificadas
    'affected_routes': [str, ...],          # Rutas afectadas (IATA)
    'route_impact_details': {...},
    'route_impact_summary': {...},
    'touchpoint_correlations': {...},
    'causal_confidence': str,               # 'high_agent_analysis', 'file_found_no_incidents', etc.
    'structured_ncs_data': {...},            # Datos estructurados extraídos
    'temporal_comparison': {...},            # Comparación temporal (si aplica)
    'temporal_comparison_successful': bool,
    'data_source_status': str,              # 'available', 'available_but_empty', 'unavailable'
    'days_analyzed': int,
    'date_range': str
}
```

### Interacción con routes_tool

El `CleanConversationTracker` detecta automáticamente rutas mencionadas en los resultados de `ncs_tool`:

```python
if tool_name in ['ncs_tool', 'verbatims_tool']:
    routes = self._extract_routes_from_result(result)
    if routes:
        self.identified_routes.extend(routes)
```

Estas rutas se usan después en `_get_ncs_routes()` para cruzar con datos NPS de Power BI y obtener el impacto en NPS de las rutas con incidentes.

---

## 11. Variables de Entorno Relevantes

| Variable | Uso | Requerida |
|----------|-----|-----------|
| `AWS_ACCESS_KEY_ID` | Acceso a S3 (prod) | Sí (prod) |
| `AWS_SECRET_ACCESS_KEY` | Acceso a S3 (prod) | Sí (prod) |
| `AWS_SESSION_TOKEN` | Token de sesión temporal | No |
| `AWS_REGION` | Región AWS (default: `eu-west-1`) | No |

En modo `local`, las credenciales se cargan desde `temp_aws_credentials.env`.

---

## 12. Manejo de Errores

| Error | Causa | Comportamiento |
|-------|-------|----------------|
| `InvalidAccessKeyId` | Credenciales AWS expiradas | Retorna mensaje de error, `causal_confidence='no_data_source'` |
| `Access Denied` | Sin permisos al bucket S3 | Igual que arriba |
| S3 vacío para el rango | No hay archivos NCS para esas fechas | Retorna "No incidents found", `causal_confidence='file_found_no_incidents'` |
| Parsing HTML falla | Formato de email inesperado | DataFrame vacío para ese archivo, continúa con los demás |
| Filtrado elimina todo | Incidentes existen pero no en el segmento | Mensaje indicando incidentes en otros segmentos |
| LLM reflection falla | Error en el análisis con IA | Fallback a `_extract_ncs_causal_insights_workflow_aware()` (pattern matching) |


---

## Apéndice A: Código Fuente Completo — `NCSDataCollector`

Archivo: `dashboard_analyzer/data_collection/ncs_collector.py`

```python
import os
import boto3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tempfile
import logging
from io import StringIO
import re
from bs4 import BeautifulSoup

from dashboard_analyzer.anomaly_explanation.genai_core.utils.aws_session import get_aws_session

class NCSDataCollector:
    """Collects Net Customer Satisfaction data from AWS S3 bucket"""
    
    def __init__(self, environment: str = "prod"):
        self.logger = logging.getLogger(__name__)
        self.bucket_name = "ibdata-prod-ew1-s3-customer"
        self.base_prefix = "customer/catia/ncs/raw/attatchments/"
        self.environment = environment
        self.session = get_aws_session(environment=self.environment, use_sandbox=False)
        self.s3_client = self.session.client('s3')
        
    def list_available_files(self, date_prefix: str = None) -> List[str]:
        try:
            prefix = self.base_prefix
            if date_prefix:
                prefix += f"ndc-{date_prefix}"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.txt'):
                        files.append(obj['Key'])
            
            self.logger.info(f"Found {len(files)} NCS files with prefix '{prefix}'")
            return files
        except Exception as e:
            self.logger.error(f"Error listing S3 files: {str(e)}")
            return []
    
    def read_ncs_file(self, file_key: str) -> pd.DataFrame:
        try:
            self.logger.info(f"Reading NCS file: s3://{self.bucket_name}/{file_key}")
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            content = response['Body'].read().decode('utf-8')
            df = self._parse_html_email_content(content, file_key)
            if not df.empty:
                self.logger.info(f"✅ Successfully parsed NCS email: {len(df)} rows, {len(df.columns)} columns")
            else:
                self.logger.warning(f"⚠️ No data extracted from NCS email")
            return df
        except Exception as e:
            self.logger.error(f"❌ Error reading NCS file {file_key}: {str(e)}")
            return pd.DataFrame()
    
    def _parse_html_email_content(self, content: str, file_key: str) -> pd.DataFrame:
        try:
            email_metadata = self._extract_email_metadata(content)
            soup = BeautifulSoup(content, 'html.parser')
            tables = soup.find_all('table')
            all_data = []
            
            for table in tables:
                table_data = self._extract_table_data(table)
                if table_data:
                    all_data.extend(table_data)
            
            if not all_data:
                all_data = self._extract_text_based_data(soup.get_text())
            
            if all_data:
                df = pd.DataFrame(all_data)
                df['source_file'] = file_key
                df['email_date'] = email_metadata.get('date')
                df['email_subject'] = email_metadata.get('subject')
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error parsing HTML email content: {str(e)}")
            return pd.DataFrame()
    
    def _extract_email_metadata(self, content: str) -> Dict[str, str]:
        metadata = {}
        try:
            subject_match = re.search(r'<b>Asunto:</b>\s*([^<]+)', content)
            if subject_match:
                metadata['subject'] = subject_match.group(1).strip()
            date_match = re.search(r'<b>Enviados:</b>\s*([^<]+)', content)
            if date_match:
                metadata['date'] = date_match.group(1).strip()
            sender_match = re.search(r'<b>De:</b>\s*([^<]+)', content)
            if sender_match:
                metadata['sender'] = sender_match.group(1).strip()
        except Exception as e:
            self.logger.warning(f"Error extracting email metadata: {str(e)}")
        return metadata
    
    def _extract_table_data(self, table) -> List[Dict]:
        try:
            rows = table.find_all('tr')
            if not rows:
                return []
            headers = []
            data_rows = []
            for i, row in enumerate(rows):
                cells = row.find_all(['th', 'td'])
                if not cells:
                    continue
                cell_texts = [cell.get_text(strip=True) for cell in cells]
                if i == 0 or any('incident' in cell.lower() or 'flight' in cell.lower() or 'route' in cell.lower() for cell in cell_texts):
                    if not headers:
                        headers = cell_texts
                else:
                    data_rows.append(cell_texts)
            if headers and data_rows:
                result = []
                for row in data_rows:
                    while len(row) < len(headers):
                        row.append('')
                    row_dict = {}
                    for j, header in enumerate(headers):
                        if j < len(row):
                            row_dict[header] = row[j]
                    if any(value.strip() for value in row_dict.values()):
                        result.append(row_dict)
                return result
        except Exception as e:
            self.logger.warning(f"Error extracting table data: {str(e)}")
        return []
    
    def _extract_text_based_data(self, text_content: str) -> List[Dict]:
        try:
            incidents = []
            lines = text_content.split('\n')
            current_incident = {}
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                flight_match = re.search(r'(IB\d+)', line, re.IGNORECASE)
                if flight_match:
                    if current_incident:
                        incidents.append(current_incident)
                    current_incident = {'flight': flight_match.group(1)}
                route_match = re.search(r'([A-Z]{3})-([A-Z]{3})', line)
                if route_match and current_incident:
                    current_incident['route'] = f"{route_match.group(1)}-{route_match.group(2)}"
                time_match = re.search(r'(\d{1,2}:\d{2})', line)
                if time_match and current_incident:
                    if 'time' not in current_incident:
                        current_incident['time'] = time_match.group(1)
                incident_keywords = ['delay', 'cancel', 'incident', 'problem', 'issue', 'technical']
                if any(keyword in line.lower() for keyword in incident_keywords):
                    if current_incident:
                        current_incident['description'] = line
            if current_incident:
                incidents.append(current_incident)
            return incidents
        except Exception as e:
            self.logger.warning(f"Error extracting text-based data: {str(e)}")
            return []
    
    def collect_ncs_data_for_date_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            self.logger.info(f"Collecting NCS data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            all_data = []
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                files = self.list_available_files(date_str)
                for file_key in files:
                    df = self.read_ncs_file(file_key)
                    if not df.empty:
                        df['source_file'] = file_key
                        df['collection_date'] = current_date
                        all_data.append(df)
                current_date += timedelta(days=1)
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                self.logger.info(f"✅ Collected {len(combined_df)} total NCS records")
                return combined_df
            else:
                self.logger.warning("No NCS data found for the specified date range")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"❌ Error collecting NCS data: {str(e)}")
            return pd.DataFrame()
    
    def analyze_ncs_incidents_for_period(self, df: pd.DataFrame, analysis_focus: str = "all") -> Dict[str, Any]:
        if df.empty:
            return {
                "total_incidents": 0, "analysis": "No incidents found",
                "incident_counts": {}, "detailed_incidents": {"count": 0, "sample_incidents": []},
                "route_analysis": {}, "flight_analysis": {}
            }
        analysis = {
            "total_incidents": len(df),
            "date_range": {
                "start": df['period_start_date'].iloc[0] if 'period_start_date' in df.columns else None,
                "end": df['period_end_date'].iloc[0] if 'period_end_date' in df.columns else None
            },
            "period_info": {
                "period_number": df['period_number'].iloc[0] if 'period_number' in df.columns else None,
                "aggregation_days": df['aggregation_days'].iloc[0] if 'aggregation_days' in df.columns else None
            }
        }
        try:
            incident_counts = self._extract_incident_counts_from_data(df)
            analysis["incident_counts"] = incident_counts
            
            if analysis_focus in ["flights", "all"] and not df.empty:
                flight_incidents = df[df.iloc[:, 0].str.contains('IB\\d+', na=False, regex=True)]
                if not flight_incidents.empty:
                    flight_counts = flight_incidents.iloc[:, 0].value_counts()
                    analysis["flight_analysis"] = {
                        "total_flights_affected": len(flight_counts),
                        "most_affected_flights": flight_counts.head(5).to_dict(),
                        "flight_impact_summary": f"{len(flight_counts)} flights affected"
                    }
                else:
                    analysis["flight_analysis"] = {"total_flights_affected": 0, "most_affected_flights": {}}
            
            if analysis_focus in ["incidents", "all"]:
                incident_categories = []
                for col in df.columns:
                    if any(kw in col.lower() for kw in ['cancel', 'retraso', 'incident', 'equipaje', 'tecnic', 'desvio']):
                        if col not in ['source_file', 'email_date', 'email_subject']:
                            incident_categories.append(col)
                if incident_categories:
                    analysis["incident_categories"] = {"categories_found": incident_categories}
                
                detailed_incidents = df[df.iloc[:, 0].str.len() > 50]
                if not detailed_incidents.empty:
                    incident_texts = detailed_incidents.iloc[:, 0].tolist()
                    analysis["detailed_incidents"] = {
                        "count": len(detailed_incidents),
                        "sample_incidents": incident_texts[:3],
                        "incident_themes": self._extract_incident_themes(incident_texts)
                    }
                else:
                    analysis["detailed_incidents"] = {"count": 0, "sample_incidents": [], "incident_themes": []}
            
            if analysis_focus in ["routes", "all"]:
                analysis["route_analysis"] = self._extract_route_analysis_from_data(df)
            
            analysis["summary_insights"] = self._generate_summary_insights(analysis)
        except Exception as e:
            analysis["error"] = f"Error during analysis: {str(e)}"
        return analysis
    
    def _extract_incident_counts_from_data(self, df: pd.DataFrame) -> Dict[str, int]:
        incident_counts = {}
        incident_type_mapping = {
            'cancelaciones': ['Cancelaciones', 'Cancel'],
            'retrasos': ['Retrasos', 'Delay'],
            'desvios': ['Desvíos', 'Divert'],
            'limitacion_aeronave': ['Limitación', 'Aircraft'],
            'equipaje': ['Equipaje', 'Baggage'],
            'otras_incidencias': ['Otras incidencias', 'Other', 'Otros'],
            'incidencias_sistemas': ['Incidencias con sistemas', 'System']
        }
        for standard_type, column_variants in incident_type_mapping.items():
            count = 0
            for variant in column_variants:
                matching_cols = [col for col in df.columns if variant.lower() in col.lower()]
                for col in matching_cols:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        non_empty = col_data[col_data.astype(str).str.strip() != '']
                        if len(non_empty) > 0:
                            try:
                                numeric_data = pd.to_numeric(non_empty, errors='ignore')
                                if numeric_data.dtype in ['int64', 'float64']:
                                    count += int(numeric_data.sum())
                                else:
                                    count += len(non_empty)
                            except:
                                count += len(non_empty)
            if count > 0:
                incident_counts[standard_type] = count
        
        # Fallback: text-based counting
        if not incident_counts and not df.empty:
            main_col = df.columns[0]
            all_text = " ".join(df[main_col].astype(str).tolist()).lower()
            text_mapping = {
                'cancelaciones': ['cancel', 'anulad', 'cancelled'],
                'retrasos': ['retraso', 'delay', 'delayed', 'tarde'],
                'desvios': ['desvio', 'divert', 'reroute'],
                'limitacion_aeronave': ['aircraft', 'avión', 'aeronave', 'technical'],
                'equipaje': ['equipaje', 'baggage', 'maleta', 'luggage'],
                'otras_incidencias': ['other', 'otro', 'misc', 'various']
            }
            for incident_type, keywords in text_mapping.items():
                count = sum(all_text.count(keyword) for keyword in keywords)
                if count > 0:
                    incident_counts[incident_type] = count
        return incident_counts
    
    def _extract_incident_themes(self, incident_texts: List[str]) -> List[str]:
        themes = []
        theme_keywords = {
            "technical_issues": ["técnica", "technical", "avería", "breakdown", "maintenance"],
            "weather": ["weather", "tiempo", "meteorológica", "tormenta", "storm"],
            "bird_strike": ["aves", "bird", "impacto", "strike"],
            "delays": ["retraso", "delay", "delayed", "tardanza"],
            "cancellations": ["cancel", "anulado", "cancelled"],
            "aircraft_change": ["cambio", "change", "aircraft", "avión"],
            "baggage": ["equipaje", "baggage", "maleta", "luggage"],
            "crew": ["tripulación", "crew", "piloto", "pilot"],
            "passenger": ["pasajero", "passenger", "cliente", "customer"]
        }
        theme_counts = {}
        for theme, keywords in theme_keywords.items():
            count = 0
            for text in incident_texts:
                if any(keyword.lower() in text.lower() for keyword in keywords):
                    count += 1
            if count > 0:
                theme_counts[theme] = count
        for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
            themes.append(f"{theme}: {count} incidents")
        return themes[:5]
    
    def _generate_summary_insights(self, analysis: Dict) -> List[str]:
        insights = []
        total = analysis.get("total_incidents", 0)
        insights.append(f"Total of {total} operational incidents detected")
        if "flight_analysis" in analysis:
            flight_info = analysis["flight_analysis"]
            insights.append(f"Flight impact: {flight_info['total_flights_affected']} flights affected")
        if "route_analysis" in analysis:
            route_count = analysis["route_analysis"]["total_routes_affected"]
            insights.append(f"Route impact: {route_count} routes experienced incidents")
        return insights
    
    def _extract_route_analysis_from_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"total_routes_affected": 0, "most_affected_routes": {}, "route_impact_summary": "No route data available"}
        route_pattern = r'[A-Z]{3}-[A-Z]{3}'
        routes = []
        for col in df.columns:
            if col not in ['source_file', 'email_date', 'email_subject', 'collection_date']:
                route_incidents = df[df[col].astype(str).str.contains(route_pattern, na=False, regex=True)]
                for text in route_incidents[col]:
                    route_matches = re.findall(route_pattern, str(text))
                    routes.extend(route_matches)
        if routes:
            route_counts = pd.Series(routes).value_counts()
            return {
                "total_routes_affected": len(route_counts),
                "most_affected_routes": route_counts.head(5).to_dict(),
                "route_impact_summary": f"{len(route_counts)} routes affected",
                "all_routes": list(set(routes))
            }
        else:
            return {"total_routes_affected": 0, "most_affected_routes": {}, "all_routes": []}
```


---

## Apéndice B: Código Fuente — `_ncs_tool_single_period` (Agente Causal)

Archivo: `dashboard_analyzer/anomaly_explanation/genai_core/agents/causal_explanation_agent.py`

```python
async def _ncs_tool_single_period(self, node_path: str, start_date: str, end_date: str) -> str:
    """NCS tool for single period analysis (absolute incidents only)"""
    try:
        self.logger.info(f"Collecting NCS data for {node_path} from {start_date} to {end_date} (single period)")
        
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_dt = start_date
        if isinstance(end_date, str):
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = end_date
        
        from ....data_collection.ncs_collector import NCSDataCollector
        ncs_collector = NCSDataCollector(environment=self.environment)
        
        # Get NCS data for the specific period only
        ncs_data = ncs_collector.collect_ncs_data_for_date_range(
            start_date=start_dt,
            end_date=end_dt
        )
        
        if ncs_data.empty:
            return f"No NCS data found for {node_path} from {start_date} to {end_date}"

        # Apply segment filtering BEFORE analysis
        filtered_ncs_data = await self._filter_ncs_by_segment(ncs_data, node_path)

        if filtered_ncs_data.empty:
            return f"No NCS incidents found for segment {node_path} from {start_date} to {end_date} after filtering"

        # Analyze the FILTERED data
        incident_analysis = self.ncs_collector.analyze_ncs_incidents_for_period(filtered_ncs_data, analysis_focus="all")

        # Build the result string
        result_parts = []
        result_parts.append(f"📊 **INCIDENTES NCS - PERIODO ÚNICO**")
        result_parts.append(f"📅 Período: {start_date} a {end_date}")
        result_parts.append(f"🎯 Segmento: {node_path}")
        result_parts.append(f"Total Incidentes (Segmento): {len(filtered_ncs_data)}")
        result_parts.append("")
        result_parts.append("**INCIDENTES DEL PERIODO:**")

        incident_counts = incident_analysis.get("incident_counts", {})
        for incident_type, count in incident_counts.items():
            result_parts.append(f"• {incident_type}: {count} incidentes")
        
        if "detailed_incidents" in incident_analysis:
            detailed = incident_analysis["detailed_incidents"]
            result_parts.append(f"• Incidentes con descripción detallada: {detailed['count']}")
            if detailed.get("sample_incidents"):
                result_parts.append("• Ejemplos de incidentes:")
                for i, incident in enumerate(detailed["sample_incidents"][:2], 1):
                    truncated = incident[:150] + "..." if len(incident) > 150 else incident
                    result_parts.append(f"  {i}. {truncated}")
            if detailed.get("incident_themes"):
                result_parts.append("• Temas principales:")
                for theme in detailed["incident_themes"][:3]:
                    result_parts.append(f"  - {theme}")
        
        if "route_analysis" in incident_analysis:
            route_info = incident_analysis["route_analysis"]
            result_parts.append(f"• Rutas afectadas: {route_info['total_routes_affected']}")
            if route_info.get("most_affected_routes"):
                result_parts.append("• Rutas más impactadas:")
                for route, count in list(route_info["most_affected_routes"].items())[:3]:
                    result_parts.append(f"  - {route}: {count} incidentes")
        
        if "summary_insights" in incident_analysis:
            result_parts.append("• Resumen de impacto:")
            for insight in incident_analysis["summary_insights"][:3]:
                result_parts.append(f"  - {insight}")
        
        result_parts.append("")
        result_parts.append("**NOTA:** Estos son incidentes absolutos del período específico, sin comparación temporal.")
        
        return "\n".join(result_parts)
        
    except Exception as e:
        self.logger.error(f"❌ Error in single period NCS tool: {type(e).__name__}: {str(e)}")
        return f"ERROR in NCS tool: {type(e).__name__}: {str(e)}"
```


---

## Apéndice C: Código Fuente — `_ncs_tool` Modo Comparativo (Agente Causal)

Archivo: `dashboard_analyzer/anomaly_explanation/genai_core/agents/causal_explanation_agent.py`

```python
async def _ncs_tool(self, node_path: str, start_date: str, end_date: str, analysis_focus: str = "flights", temporal_comparison: bool = True) -> str:
    """
    Enhanced NCS tool for causal analysis that extracts:
    1. DISRUPTION CAUSES: Identifies root causes from incident patterns
    2. AFFECTED ROUTES: Maps specific routes impacted by incidents
    3. TEMPORAL COMPARISON: Compares current period vs previous period
    """
    try:
        self.logger.info(f"Collecting NCS operational incidents for {node_path} from {start_date} to {end_date}")
        
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_dt = start_date
        if isinstance(end_date, str):
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = end_date
        
        total_days = (end_dt - start_dt).days + 1
        
        from ....data_collection.ncs_collector import NCSDataCollector
        ncs_collector = NCSDataCollector(environment=self.environment)
        
        # Calculate comparison period dates
        comparison_data = None
        comparison_start_dt = None
        comparison_end_dt = None
        
        if temporal_comparison:
            if hasattr(self, 'comparison_start_date') and hasattr(self, 'comparison_end_date') and self.comparison_start_date and self.comparison_end_date:
                # Use specified comparison period
                if isinstance(self.comparison_start_date, datetime):
                    comparison_start_dt = self.comparison_start_date
                    comparison_end_dt = self.comparison_end_date
                else:
                    comparison_start_dt = datetime.strptime(self.comparison_start_date, '%Y-%m-%d')
                    comparison_end_dt = datetime.strptime(self.comparison_end_date, '%Y-%m-%d')
            else:
                # Calculate previous period of same length
                comparison_end_dt = start_dt - timedelta(days=1)
                comparison_start_dt = comparison_end_dt - timedelta(days=total_days - 1)
        
        # ── Collect current period ──
        try:
            ncs_data = ncs_collector.collect_ncs_data_for_date_range(start_dt, end_dt)
            ncs_data = await self._filter_ncs_by_segment(ncs_data, node_path, collect_unattributed_key="current")
            access_error = False
            error_msg = None
        except Exception as e:
            error_msg = str(e)
            if any(aws_err in error_msg.lower() for aws_err in ['invalidaccesskeyid', 'access denied', 'credentials', 'token']):
                access_error = True
            else:
                access_error = True
            ncs_data = pd.DataFrame()
        
        # ── Collect comparison period ──
        if temporal_comparison and not access_error and comparison_start_dt is not None:
            try:
                comparison_data = ncs_collector.collect_ncs_data_for_date_range(comparison_start_dt, comparison_end_dt)
                comparison_data = await self._filter_ncs_by_segment(comparison_data, node_path, collect_unattributed_key="comparison")
            except Exception:
                comparison_data = pd.DataFrame()
        else:
            comparison_data = pd.DataFrame()
        
        # ── CASE: Access error ──
        if access_error:
            self.collected_data['ncs_data'] = {
                'analysis_summary': f"❌ NCS data source unavailable for {start_date} to {end_date} ({total_days} days)",
                'identified_causes': [], 'affected_routes': [],
                'causal_confidence': 'no_data_source',
                'data_source_status': 'unavailable',
                'error_details': error_msg[:200] if error_msg else 'Unknown error',
            }
            return f"❌ NCS data source unavailable for {start_date} to {end_date}. Details: {error_msg[:100]}"
        
        # ── CASE: No data found ──
        if ncs_data.empty:
            self.collected_data['ncs_data'] = {
                'analysis_summary': f"📅 No NCS incidents found for {start_date} to {end_date} ({total_days} days)",
                'identified_causes': [], 'affected_routes': [],
                'causal_confidence': 'file_found_no_incidents',
                'data_source_status': 'available_but_empty',
            }
            return f"📅 No NCS incidents found for {start_date} to {end_date}. Could indicate good operational performance."
        
        # ── Temporal comparison ──
        temporal_analysis = None
        comparison_successful = False
        
        if temporal_comparison and comparison_data is not None and not comparison_data.empty and comparison_start_dt is not None:
            filtered_current = await self._filter_ncs_by_segment(ncs_data, node_path)
            filtered_comparison = await self._filter_ncs_by_segment(comparison_data, node_path)
            temporal_analysis = self._create_temporal_ncs_comparison(
                filtered_current, filtered_comparison, node_path,
                start_date, end_date,
                comparison_start_dt.strftime('%Y-%m-%d'), comparison_end_dt.strftime('%Y-%m-%d')
            )
            comparison_successful = True
        
        # ── Filter by segment ──
        filtered_ncs_data = await self._filter_ncs_by_segment(ncs_data, node_path)
        
        if filtered_ncs_data.empty:
            total_global_incidents = len(ncs_data)
            self.collected_data['ncs_data'] = {
                'analysis_summary': f"📊 Incidents in other routes - {node_path} not affected",
                'identified_causes': [], 'affected_routes': [],
                'causal_confidence': 'file_found_no_segment_incidents',
                'non_correlated_incidents': total_global_incidents,
            }
            return f"📊 {total_global_incidents} incidents found but none affected {node_path}."
        
        # ── Extract structured data ──
        incident_col = self._find_column(filtered_ncs_data, ['incident', 'incidents', ''])
        if incident_col is None:
            incident_col = filtered_ncs_data.columns[0] if not filtered_ncs_data.empty else ''
        all_incidents_text = "\n".join(filtered_ncs_data[incident_col].astype(str).tolist())
        structured_ncs_data = self._extract_structured_ncs_data(all_incidents_text)
        
        # ── Causal analysis (LLM reflection with pattern-matching fallback) ──
        current_anomaly_type = getattr(self, 'current_anomaly_type', 'unknown')
        nps_variation = getattr(self, 'nps_difference_value', None)
        temporal_incident_changes = {}
        if temporal_analysis:
            temporal_incident_changes = temporal_analysis.get('incident_type_deltas', {})
        
        comparison_data_for_reflection = None
        comparison_start_str = None
        comparison_end_str = None
        if temporal_comparison and comparison_data is not None and not comparison_data.empty:
            comparison_data_for_reflection = await self._filter_ncs_by_segment(comparison_data, node_path)
            if comparison_start_dt:
                comparison_start_str = comparison_start_dt.strftime('%Y-%m-%d')
            if comparison_end_dt:
                comparison_end_str = comparison_end_dt.strftime('%Y-%m-%d')
        
        try:
            causal_analysis = await self._ncs_reflection_with_agent(
                filtered_ncs_data=filtered_ncs_data,
                node_path=node_path,
                anomaly_type=current_anomaly_type,
                total_days=total_days,
                comparison_data=comparison_data_for_reflection,
                current_start_date=start_date,
                current_end_date=end_date,
                comparison_start_date=comparison_start_str,
                comparison_end_date=comparison_end_str,
                nps_variation=nps_variation,
                temporal_incident_changes=temporal_incident_changes
            )
            causal_analysis['structured_ncs_data'] = structured_ncs_data
            
            if (not causal_analysis.get('identified_causes') and 
                not causal_analysis.get('affected_routes') and
                causal_analysis.get('confidence_level') != 'high_agent_analysis'):
                causal_analysis = await self._extract_ncs_causal_insights_workflow_aware(
                    filtered_ncs_data, node_path, current_anomaly_type
                )
        except Exception:
            causal_analysis = await self._extract_ncs_causal_insights_workflow_aware(
                filtered_ncs_data, node_path, current_anomaly_type
            )
        
        # ── Format results ──
        analysis_result = []
        analysis_result.append(f"🎯 NCS SEGMENT ANALYSIS: {node_path}")
        analysis_result.append(f"📅 Period: {start_date} to {end_date}")
        
        if causal_analysis['identified_causes']:
            analysis_result.append(f"🚨 DISRUPTION CAUSES IDENTIFICADAS:")
            for i, cause in enumerate(causal_analysis['identified_causes'][:3], 1):
                analysis_result.append(f"  {i}. {cause}")
        
        if causal_analysis['affected_routes']:
            analysis_result.append(f"🛤️ AFFECTED ROUTES ({len(causal_analysis['affected_routes'])} routes):")
            route_impacts = causal_analysis['route_impact_summary']
            for route, impact in list(route_impacts.items())[:4]:
                analysis_result.append(f"  • {route}: {impact}")
        
        # Structured NCS breakdown
        structured_breakdown = causal_analysis.get('structured_ncs_data', {})
        if structured_breakdown and structured_breakdown.get('summary', {}).get('total_incidents', 0) > 0:
            summary = structured_breakdown['summary']
            categories = structured_breakdown.get('categories', {})
            motives = structured_breakdown.get('motives_breakdown', {})
            passengers = structured_breakdown.get('passenger_impact', {})
            delays = structured_breakdown.get('delay_statistics', {})
            
            analysis_result.append(f"📊 NCS STRUCTURED BREAKDOWN:")
            if categories:
                cat_summary = [f"{cat}: {count}" for cat, count in categories.items() if count > 0]
                if cat_summary:
                    analysis_result.append(f"   📋 Categorías: {', '.join(cat_summary)}")
            if motives:
                top_motives = sorted(motives.items(), key=lambda x: x[1], reverse=True)[:3]
                analysis_result.append(f"   🔧 Motivos: {', '.join([f'{m}: {c}' for m, c in top_motives])}")
            if passengers.get('total', 0) > 0:
                analysis_result.append(f"   👥 Pasajeros: J:{passengers.get('j_class',0)}, W:{passengers.get('w_class',0)}, Y:{passengers.get('y_class',0)} = {passengers['total']} pax")
            if delays.get('count', 0) > 0:
                analysis_result.append(f"   ⏱️ Retrasos: {delays['count']} vuelos, avg {delays.get('avg_delay',0):.1f} min")
            
            # Route incident breakdown
            route_incident_breakdown = structured_breakdown.get('route_incident_breakdown', {})
            if route_incident_breakdown:
                def route_sort_key(item):
                    route, bd = item
                    return (bd['total'], bd['cancelaciones']*5 + bd['desvios']*4 + bd['limitacion_aeronave']*3 + bd['retrasos']*2 + bd['otras_incidencias'])
                sorted_routes = sorted(route_incident_breakdown.items(), key=route_sort_key, reverse=True)
                routes_details = []
                for route, bd in sorted_routes:
                    parts = []
                    if bd['cancelaciones'] > 0: parts.append(f"C:{bd['cancelaciones']}")
                    if bd['desvios'] > 0: parts.append(f"D:{bd['desvios']}")
                    if bd['retrasos'] > 0: parts.append(f"R:{bd['retrasos']}")
                    if bd['otras_incidencias'] > 0: parts.append(f"O:{bd['otras_incidencias']}")
                    if bd['limitacion_aeronave'] > 0: parts.append(f"L:{bd['limitacion_aeronave']}")
                    routes_details.append(f"{route}({'|'.join(parts) if parts else bd['total']})")
                analysis_result.append(f"   🛫 RUTAS: {', '.join(routes_details)}")
                analysis_result.append(f"   📝 C=Cancel, D=Desvío, R=Retraso, O=Otras, L=Limitación")
        
        segment_incident_count = len(filtered_ncs_data)
        self._ncs_total_incidents = segment_incident_count
        analysis_result.append(f"📊 SEGMENT INCIDENTS: {segment_incident_count} on {node_path} across {total_days} days")
        
        # Temporal comparison in output
        if temporal_analysis is not None:
            causal_analysis['temporal_comparison'] = temporal_analysis
            causal_analysis['temporal_comparison_successful'] = comparison_successful
            if temporal_analysis.get('analysis_summary'):
                analysis_result.append(f"\n{temporal_analysis['analysis_summary'].replace(' | ', chr(10))}")
        
        self.collected_data['ncs_data'] = causal_analysis
        return " | ".join(analysis_result)
        
    except Exception as e:
        import traceback
        self.logger.error(f"💥 Error in NCS analysis: {str(e)}")
        return f"Error in NCS analysis: {str(e)}"
```


---

## Apéndice D: Código Fuente — `_filter_ncs_by_segment` (Agente Causal)

Archivo: `dashboard_analyzer/anomaly_explanation/genai_core/agents/causal_explanation_agent.py`

```python
async def _filter_ncs_by_segment(
    self,
    ncs_data: pd.DataFrame,
    node_path: str,
    collect_unattributed_key: str | None = None
) -> pd.DataFrame:
    """
    Simple NCS filtering by segment characteristics without external dependencies.
    Focuses on the specific incidents for the analyzed segment.
    """
    try:
        if ncs_data.empty:
            return ncs_data
        
        incident_col = ncs_data.columns[0] if len(ncs_data.columns) > 0 else None
        if incident_col is None:
            return ncs_data
        
        self.logger.info(f"Starting NCS filtering for segment: {node_path}")
        self.logger.info(f"Total NCS incidents to filter: {len(ncs_data)}")
        
        cabins, companies, hauls = self.pbi_collector._get_node_filters(node_path)
        
        filtered_ncs = await self._apply_contextual_filtering(
            ncs_data, node_path,
            collect_unattributed_key=collect_unattributed_key
        )
        
        if len(filtered_ncs) < len(ncs_data):
            reduction_rate = (1 - len(filtered_ncs) / len(ncs_data)) * 100
            self.logger.info(f"Filtered NCS data for {node_path}: {len(filtered_ncs)}/{len(ncs_data)} ({reduction_rate:.1f}% reduction)")
        
        return filtered_ncs
        
    except Exception as e:
        self.logger.error(f"Error in NCS segment filtering: {str(e)}")
        return ncs_data


async def _apply_contextual_filtering(
    self,
    ncs_data: pd.DataFrame,
    node_path: str,
    collect_unattributed_key: str | None = None
) -> pd.DataFrame:
    """
    Apply contextual filtering with the correct NCS logic:
    1. ALWAYS filter by haul (Global, LH, SH) based on routes/airports
    2. ONLY filter by cabin when at cabin level AND incident specifies ONLY another cabin
    3. NEVER discard incidents just because they don't mention cabin
    """
    try:
        cabins, companies, hauls = self.pbi_collector._get_node_filters(node_path)
        
        if ncs_data.empty or len(ncs_data.columns) == 0:
            return ncs_data
        
        incident_col = ncs_data.columns[0]
        filtered_ncs = ncs_data.copy()
        
        # STEP 1: ALWAYS apply haul-based filtering using Routes Dictionary
        if hauls and len(hauls) == 1:
            haul_type = hauls[0]
            is_global_root = node_path.strip() == "Global"
            
            if is_global_root:
                filtered_ncs = await self._filter_using_routes_dictionary(
                    filtered_ncs, haul_type, incident_col,
                    allow_unknown_route_incidents=True
                )
            else:
                # Strictly attributable to this haul
                target_only = await self._filter_using_routes_dictionary(
                    filtered_ncs, haul_type, incident_col,
                    allow_unknown_route_incidents=False
                )
                # Collect unattributed incidents separately for LLM evaluation
                if collect_unattributed_key:
                    target_plus_unknown = await self._filter_using_routes_dictionary(
                        filtered_ncs, haul_type, incident_col,
                        allow_unknown_route_incidents=True
                    )
                    unknown_only = target_plus_unknown[~target_plus_unknown.index.isin(target_only.index)]
                    if not hasattr(self, "_ncs_unattributed"):
                        self._ncs_unattributed = {}
                    self._ncs_unattributed[collect_unattributed_key] = unknown_only
                filtered_ncs = target_only
        
        # STEP 2: Apply cabin filtering ONLY at cabin level with exclusive language
        if cabins and len(cabins) == 1:
            cabin_type = cabins[0].lower()
            cabin_keywords = {
                'business': ['business', 'ejecutiva', 'premium', 'preferente'],
                'economy': ['economy', 'turista', 'económica'],
                'premium': ['premium', 'preferente', 'plus']
            }
            if cabin_type in cabin_keywords:
                other_cabin_patterns = []
                for other_cabin, keywords in cabin_keywords.items():
                    if other_cabin != cabin_type:
                        other_cabin_patterns.extend(keywords)
                if other_cabin_patterns:
                    exclusive_keywords = ['solo', 'solamente', 'únicamente', 'exclusivamente']
                    exclusive_pattern = '|'.join([f'\\b{kw}\\b' for kw in exclusive_keywords])
                    other_cabin_pattern = '|'.join([f'\\b{kw}\\b' for kw in other_cabin_patterns])
                    combined_pattern = f"({exclusive_pattern}).*({other_cabin_pattern})|({other_cabin_pattern}).*({exclusive_pattern})"
                    exclusive_mask = filtered_ncs[incident_col].str.contains(combined_pattern, na=False, regex=True, case=False)
                    filtered_ncs = filtered_ncs[~exclusive_mask]
        
        # STEP 3: Fallback for Global
        if len(filtered_ncs) == 0 and node_path.strip() == "Global":
            return ncs_data
        
        return filtered_ncs
        
    except Exception as e:
        self.logger.error(f"Error in contextual filtering: {str(e)}")
        return ncs_data
```
