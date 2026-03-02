# NCS Tool — Blueprint Completo para Recreación en Nuevo Proyecto

## Índice

1. [Visión General](#1-visión-general)
2. [Arquitectura de Componentes](#2-arquitectura-de-componentes)
3. [Parámetros de la Tool](#3-parámetros-de-la-tool)
4. [Fuente de Datos: S3](#4-fuente-de-datos-s3)
5. [Componente 1: NCSDataCollector](#5-componente-1-ncsdatacollector)
6. [Componente 2: Filtrado por Segmento](#6-componente-2-filtrado-por-segmento)
7. [Componente 3: Comparación Temporal](#7-componente-3-comparación-temporal)
8. [Componente 4: Reflexión con LLM](#8-componente-4-reflexión-con-llm)
9. [Submodo SINGLE](#9-submodo-single)
10. [Submodo COMPARATIVE](#10-submodo-comparative)
11. [Prompts del Agente Causal](#11-prompts-del-agente-causal)
12. [Integración con el Agente](#12-integración-con-el-agente)
13. [Estructuras de Datos de Salida](#13-estructuras-de-datos-de-salida)
14. [Manejo de Errores](#14-manejo-de-errores)
15. [Dependencias y Stack](#15-dependencias-y-stack)

---

## 1. Visión General

La `ncs_tool` es una herramienta del `CausalExplanationAgent` que recopila y analiza incidentes
operacionales (NCS - Net Customer Satisfaction) desde archivos HTML almacenados en AWS S3.
Su objetivo es identificar causas operativas (cancelaciones, retrasos, desvíos, etc.) que
puedan explicar anomalías en el NPS de un segmento.

### Dos submodos

| Submodo | Método | Propósito |
|---------|--------|-----------|
| **single** | `_ncs_tool_single_period()` | Analiza incidentes absolutos de un período sin comparación |
| **comparative** | `_ncs_tool()` | Compara período actual vs período de referencia con deltas |

### Parámetros comunes
- `node_path`: Segmento NPS (ej: `"Global/SH/Economy"`)
- `start_date`: Fecha inicio `YYYY-MM-DD`
- `end_date`: Fecha fin `YYYY-MM-DD`

---

## 2. Arquitectura de Componentes

```
CausalExplanationAgent
│
├── _ncs_tool()                         # Submodo COMPARATIVE
│   ├── NCSDataCollector.collect_ncs_data_for_date_range()  # Período actual
│   ├── _filter_ncs_by_segment()        # Filtrar por segmento
│   ├── NCSDataCollector.collect_ncs_data_for_date_range()  # Período comparación
│   ├── _filter_ncs_by_segment()        # Filtrar comparación
│   ├── _create_temporal_ncs_comparison()  # Generar deltas
│   │   ├── _extract_structured_ncs_data_for_comparison()
│   │   ├── _create_route_incident_matrix()
│   │   └── _calculate_incident_type_deltas()
│   ├── _extract_structured_ncs_data()  # Datos estructurados
│   └── _ncs_reflection_with_agent()    # Análisis causal con LLM
│
├── _ncs_tool_single_period()           # Submodo SINGLE
│   ├── NCSDataCollector.collect_ncs_data_for_date_range()
│   ├── _filter_ncs_by_segment()
│   └── NCSDataCollector.analyze_ncs_incidents_for_period()
│
├── NCSDataCollector                    # Recopilación de datos desde S3
│   ├── collect_ncs_data_for_date_range()
│   ├── list_available_files()
│   ├── read_ncs_file()
│   ├── _parse_html_email_content()
│   ├── _extract_email_metadata()
│   ├── _extract_table_data()
│   ├── _extract_text_based_data()
│   ├── analyze_ncs_incidents_for_period()
│   ├── _extract_incident_counts_from_data()
│   ├── _extract_incident_themes()
│   ├── _extract_route_analysis_from_data()
│   └── _generate_summary_insights()
│
├── _filter_ncs_by_segment()            # Filtrado por segmento NPS
│   └── _apply_contextual_filtering()
│       └── _filter_using_routes_dictionary()
│
└── _ncs_reflection_with_agent()        # Análisis causal con LLM (solo comparative)
```

---

## 3. Parámetros de la Tool

### Submodo COMPARATIVE (`_ncs_tool`)

```python
async def _ncs_tool(
    self,
    node_path: str,              # Segmento NPS: "Global/SH/Economy"
    start_date: str,             # Fecha inicio YYYY-MM-DD
    end_date: str,               # Fecha fin YYYY-MM-DD
    analysis_focus: str = "flights",  # "flights", "routes", "incidents", "all"
    temporal_comparison: bool = True   # Habilitar comparación temporal
) -> str
```

### Submodo SINGLE (`_ncs_tool_single_period`)

```python
async def _ncs_tool_single_period(
    self,
    node_path: str,              # Segmento NPS: "Global/SH/Economy"
    start_date: str,             # Fecha inicio YYYY-MM-DD
    end_date: str                # Fecha fin YYYY-MM-DD
) -> str
```

### Jerarquía de Segmentos (node_path)

```
Global
├── LH (Long Haul)
│   ├── Economy
│   ├── Business
│   └── Premium
└── SH (Short Haul)
    ├── Economy
    │   ├── IB
    │   └── YW
    └── Business
        ├── IB
        └── YW
```

Valores válidos: `"Global"`, `"Global/LH"`, `"Global/SH"`, `"Global/LH/Economy"`,
`"Global/LH/Business"`, `"Global/LH/Premium"`, `"Global/SH/Economy"`,
`"Global/SH/Business"`, `"Global/SH/Economy/IB"`, `"Global/SH/Economy/YW"`,
`"Global/SH/Business/IB"`, `"Global/SH/Business/YW"`

---

## 4. Fuente de Datos: S3

### Configuración del Bucket

| Parámetro | Valor |
|-----------|-------|
| Bucket | `ibdata-prod-ew1-s3-customer` |
| Prefijo base | `customer/catia/ncs/raw/attatchments/` |
| Formato de archivo | `.txt` (contenido HTML de emails) |
| Naming pattern | `ndc-{YYYY-MM-DD}...` |

### Credenciales AWS

```python
# El NCSDataCollector usa get_aws_session() con use_sandbox=False
self.session = get_aws_session(environment=self.environment, use_sandbox=False)
self.s3_client = self.session.client('s3')
```

Resolución de credenciales:
- `environment="local"` → carga desde `temp_aws_credentials.env`
- `environment="prod"` → variables de entorno del sistema (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) o IAM roles

### Variables de Entorno Requeridas

| Variable | Uso | Requerida |
|----------|-----|-----------|
| `AWS_ACCESS_KEY_ID` | Acceso a S3 (prod) | Sí (prod) |
| `AWS_SECRET_ACCESS_KEY` | Acceso a S3 (prod) | Sí (prod) |
| `AWS_SESSION_TOKEN` | Token de sesión temporal | No |
| `AWS_REGION` | Región AWS (default: `eu-west-1`) | No |

---

## 5. Componente 1: NCSDataCollector

### Código completo

```python
import os
import boto3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from io import StringIO
import re
from bs4 import BeautifulSoup

# Importar tu función de sesión AWS (adaptar al nuevo proyecto)
# from tu_proyecto.utils.aws_session import get_aws_session


class NCSDataCollector:
    """Recopila datos NCS (Net Customer Satisfaction) desde AWS S3"""

    def __init__(self, environment: str = "prod"):
        self.logger = logging.getLogger(__name__)
        self.bucket_name = "ibdata-prod-ew1-s3-customer"
        self.base_prefix = "customer/catia/ncs/raw/attatchments/"
        self.environment = environment

        # Inicializar sesión AWS
        self.session = get_aws_session(environment=self.environment, use_sandbox=False)
        self.s3_client = self.session.client('s3')

    def list_available_files(self, date_prefix: str = None) -> List[str]:
        """Lista archivos NCS disponibles en S3 para una fecha"""
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
        """Lee un archivo NCS específico de S3 (formato HTML email)"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            content = response['Body'].read().decode('utf-8')
            df = self._parse_html_email_content(content, file_key)
            if not df.empty:
                self.logger.info(f"✅ Parsed NCS email: {len(df)} rows")
            return df
        except Exception as e:
            self.logger.error(f"❌ Error reading NCS file {file_key}: {str(e)}")
            return pd.DataFrame()

    def _parse_html_email_content(self, content: str, file_key: str) -> pd.DataFrame:
        """Parsea contenido HTML de email para extraer datos de incidentes"""
        try:
            email_metadata = self._extract_email_metadata(content)
            soup = BeautifulSoup(content, 'html.parser')
            tables = soup.find_all('table')
            all_data = []

            for table in tables:
                table_data = self._extract_table_data(table)
                if table_data:
                    all_data.extend(table_data)

            # Fallback: si no hay tablas, extraer de texto plano
            if not all_data:
                all_data = self._extract_text_based_data(soup.get_text())

            if all_data:
                df = pd.DataFrame(all_data)
                df['source_file'] = file_key
                df['email_date'] = email_metadata.get('date')
                df['email_subject'] = email_metadata.get('subject')
                return df
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error parsing HTML: {str(e)}")
            return pd.DataFrame()

    def _extract_email_metadata(self, content: str) -> Dict[str, str]:
        """Extrae metadata del email (asunto, fecha, remitente)"""
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
            self.logger.warning(f"Error extracting metadata: {str(e)}")
        return metadata

    def _extract_table_data(self, table) -> List[Dict]:
        """Extrae datos de una tabla HTML"""
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
                if i == 0 or any(
                    kw in cell.lower()
                    for cell in cell_texts
                    for kw in ['incident', 'flight', 'route']
                ):
                    if not headers:
                        headers = cell_texts
                else:
                    data_rows.append(cell_texts)
            if headers and data_rows:
                result = []
                for row in data_rows:
                    while len(row) < len(headers):
                        row.append('')
                    row_dict = {headers[j]: row[j] for j in range(min(len(headers), len(row)))}
                    if any(v.strip() for v in row_dict.values()):
                        result.append(row_dict)
                return result
        except Exception as e:
            self.logger.warning(f"Error extracting table: {str(e)}")
        return []

    def _extract_text_based_data(self, text_content: str) -> List[Dict]:
        """Extrae datos de incidentes desde texto plano usando patrones regex"""
        try:
            incidents = []
            lines = text_content.split('\n')
            current_incident = {}
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Números de vuelo: IB seguido de dígitos
                flight_match = re.search(r'(IB\d+)', line, re.IGNORECASE)
                if flight_match:
                    if current_incident:
                        incidents.append(current_incident)
                    current_incident = {'flight': flight_match.group(1)}
                # Rutas: XXX-YYY (códigos IATA)
                route_match = re.search(r'([A-Z]{3})-([A-Z]{3})', line)
                if route_match and current_incident:
                    current_incident['route'] = f"{route_match.group(1)}-{route_match.group(2)}"
                # Horarios: HH:MM
                time_match = re.search(r'(\d{1,2}:\d{2})', line)
                if time_match and current_incident:
                    if 'time' not in current_incident:
                        current_incident['time'] = time_match.group(1)
                # Keywords de incidentes
                incident_keywords = ['delay', 'cancel', 'incident', 'problem', 'issue', 'technical']
                if any(kw in line.lower() for kw in incident_keywords):
                    if current_incident:
                        current_incident['description'] = line
            if current_incident:
                incidents.append(current_incident)
            return incidents
        except Exception as e:
            self.logger.warning(f"Error extracting text data: {str(e)}")
            return []

    def collect_ncs_data_for_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Recopila datos NCS para un rango de fechas.
        Itera día a día, lista archivos en S3, los descarga y parsea.
        """
        try:
            self.logger.info(
                f"Collecting NCS data from {start_date.strftime('%Y-%m-%d')} "
                f"to {end_date.strftime('%Y-%m-%d')}"
            )
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
```

### Análisis de Incidentes (`analyze_ncs_incidents_for_period`)

```python
    def analyze_ncs_incidents_for_period(
        self, df: pd.DataFrame, analysis_focus: str = "all"
    ) -> Dict[str, Any]:
        """
        Analiza incidentes NCS y produce un diccionario estructurado.
        Usado principalmente por el submodo SINGLE.
        """
        if df.empty:
            return {
                "total_incidents": 0,
                "analysis": "No incidents found",
                "incident_counts": {},
                "detailed_incidents": {"count": 0, "sample_incidents": []},
                "route_analysis": {},
                "flight_analysis": {}
            }

        analysis = {"total_incidents": len(df)}

        try:
            # 1. Conteo por tipo de incidente
            analysis["incident_counts"] = self._extract_incident_counts_from_data(df)

            # 2. Análisis de vuelos afectados
            if analysis_focus in ["flights", "all"] and not df.empty:
                flight_incidents = df[
                    df.iloc[:, 0].str.contains('IB\\d+', na=False, regex=True)
                ]
                if not flight_incidents.empty:
                    flight_counts = flight_incidents.iloc[:, 0].value_counts()
                    analysis["flight_analysis"] = {
                        "total_flights_affected": len(flight_counts),
                        "most_affected_flights": flight_counts.head(5).to_dict()
                    }
                else:
                    analysis["flight_analysis"] = {
                        "total_flights_affected": 0,
                        "most_affected_flights": {}
                    }

            # 3. Incidentes detallados con temas
            if analysis_focus in ["incidents", "all"]:
                detailed_incidents = df[df.iloc[:, 0].str.len() > 50]
                if not detailed_incidents.empty:
                    incident_texts = detailed_incidents.iloc[:, 0].tolist()
                    analysis["detailed_incidents"] = {
                        "count": len(detailed_incidents),
                        "sample_incidents": incident_texts[:3],
                        "incident_themes": self._extract_incident_themes(incident_texts)
                    }
                else:
                    analysis["detailed_incidents"] = {
                        "count": 0, "sample_incidents": [], "incident_themes": []
                    }

            # 4. Análisis de rutas
            if analysis_focus in ["routes", "all"]:
                analysis["route_analysis"] = self._extract_route_analysis_from_data(df)

            # 5. Resumen legible
            analysis["summary_insights"] = self._generate_summary_insights(analysis)

        except Exception as e:
            analysis["error"] = f"Error during analysis: {str(e)}"

        return analysis

    def _extract_incident_counts_from_data(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Extrae conteos de incidentes por tipo.
        Estrategia 1: busca columnas del DataFrame que coincidan con tipos.
        Estrategia 2 (fallback): cuenta keywords en texto.
        """
        incident_counts = {}

        # Mapeo de tipos estándar a variantes de columna
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
                matching_cols = [
                    col for col in df.columns if variant.lower() in col.lower()
                ]
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

        # Fallback: conteo por texto si no hay columnas específicas
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
                count = sum(all_text.count(kw) for kw in keywords)
                if count > 0:
                    incident_counts[incident_type] = count

        return incident_counts

    def _extract_incident_themes(self, incident_texts: List[str]) -> List[str]:
        """Clasifica incidentes por temas usando keywords"""
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
            count = sum(
                1 for text in incident_texts
                if any(kw.lower() in text.lower() for kw in keywords)
            )
            if count > 0:
                theme_counts[theme] = count

        themes = [
            f"{theme}: {count} incidents"
            for theme, count in sorted(
                theme_counts.items(), key=lambda x: x[1], reverse=True
            )
        ]
        return themes[:5]

    def _extract_route_analysis_from_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extrae análisis de rutas afectadas"""
        if df.empty:
            return {"total_routes_affected": 0, "most_affected_routes": {}, "all_routes": []}

        route_pattern = r'[A-Z]{3}-[A-Z]{3}'
        routes = []
        for col in df.columns:
            if col not in ['source_file', 'email_date', 'email_subject', 'collection_date']:
                for text in df[col].astype(str):
                    routes.extend(re.findall(route_pattern, str(text)))

        if routes:
            route_counts = pd.Series(routes).value_counts()
            return {
                "total_routes_affected": len(route_counts),
                "most_affected_routes": route_counts.head(5).to_dict(),
                "all_routes": list(set(routes))
            }
        return {"total_routes_affected": 0, "most_affected_routes": {}, "all_routes": []}

    def _generate_summary_insights(self, analysis: Dict) -> List[str]:
        """Genera resumen legible del análisis"""
        insights = [f"Total of {analysis.get('total_incidents', 0)} operational incidents detected"]
        if "flight_analysis" in analysis:
            fa = analysis["flight_analysis"]
            insights.append(f"Flight impact: {fa['total_flights_affected']} flights affected")
        if "route_analysis" in analysis:
            ra = analysis["route_analysis"]
            insights.append(f"Route impact: {ra['total_routes_affected']} routes affected")
        return insights
```

---

## 6. Componente 2: Filtrado por Segmento

El filtrado por segmento es crítico para que los incidentes NCS sean relevantes al segmento
NPS que se está analizando. Tiene 3 pasos:

### Lógica de filtrado

**Paso 1: Filtro por haul (siempre)**
- Si el segmento es LH o SH (no Global), filtra usando el diccionario de rutas
- Separa incidentes en "atribuibles" (ruta pertenece al haul) y "no atribuibles" (sin ruta)
- Los no atribuibles se guardan aparte para que el LLM los evalúe
- Para Global: mantiene todos los incidentes

**Paso 2: Filtro por cabina (solo a nivel cabina)**
- Solo excluye incidentes que mencionan EXPLÍCITAMENTE que afectan SOLO a otra cabina
- Keywords exclusivas: "solo", "únicamente", "exclusivamente"
- Incidentes sin mención de cabina se MANTIENEN

**Paso 3: Fallback**
- Si no quedan incidentes tras filtrar y el nodo es Global, devuelve todos los datos originales

### Código

```python
async def _filter_ncs_by_segment(
    self,
    ncs_data: pd.DataFrame,
    node_path: str,
    collect_unattributed_key: str | None = None
) -> pd.DataFrame:
    """Filtra incidentes NCS para que solo queden los relevantes al segmento"""
    try:
        if ncs_data.empty:
            return ncs_data

        incident_col = ncs_data.columns[0] if len(ncs_data.columns) > 0 else None
        if incident_col is None:
            return ncs_data

        # Obtener filtros del segmento (adaptar a tu PBI collector)
        cabins, companies, hauls = self.pbi_collector._get_node_filters(node_path)

        filtered_ncs = await self._apply_contextual_filtering(
            ncs_data, node_path, collect_unattributed_key=collect_unattributed_key
        )

        return filtered_ncs
    except Exception as e:
        self.logger.error(f"Error in NCS segment filtering: {str(e)}")
        return ncs_data  # Devolver datos originales en caso de error


async def _apply_contextual_filtering(
    self,
    ncs_data: pd.DataFrame,
    node_path: str,
    collect_unattributed_key: str | None = None
) -> pd.DataFrame:
    """
    Filtrado contextual con la lógica correcta:
    1. SIEMPRE filtrar por haul (Global, LH, SH) basado en rutas
    2. SOLO filtrar por cabina cuando estamos a nivel cabina Y el incidente
       especifica que afecta SOLO a otra cabina
    3. NUNCA descartar incidentes solo porque no mencionan cabina
    """
    try:
        cabins, companies, hauls = self.pbi_collector._get_node_filters(node_path)

        if ncs_data.empty or len(ncs_data.columns) == 0:
            return ncs_data

        incident_col = ncs_data.columns[0]
        filtered_ncs = ncs_data.copy()

        # PASO 1: Filtro por haul usando diccionario de rutas
        if hauls and len(hauls) == 1:
            haul_type = hauls[0]
            is_global_root = node_path.strip() == "Global"

            if is_global_root:
                filtered_ncs = await self._filter_using_routes_dictionary(
                    filtered_ncs, haul_type, incident_col,
                    allow_unknown_route_incidents=True
                )
            else:
                # Solo incidentes atribuibles a este haul
                target_only = await self._filter_using_routes_dictionary(
                    filtered_ncs, haul_type, incident_col,
                    allow_unknown_route_incidents=False
                )
                # Guardar no atribuibles para el LLM
                if collect_unattributed_key:
                    target_plus_unknown = await self._filter_using_routes_dictionary(
                        filtered_ncs, haul_type, incident_col,
                        allow_unknown_route_incidents=True
                    )
                    unknown_only = target_plus_unknown[
                        ~target_plus_unknown.index.isin(target_only.index)
                    ]
                    if not hasattr(self, "_ncs_unattributed"):
                        self._ncs_unattributed = {}
                    self._ncs_unattributed[collect_unattributed_key] = unknown_only
                filtered_ncs = target_only

        # PASO 2: Filtro por cabina (solo excluir si dice SOLO otra cabina)
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
                    exclusive_kws = ['solo', 'solamente', 'únicamente', 'exclusivamente']
                    exclusive_pattern = '|'.join([f'\\b{kw}\\b' for kw in exclusive_kws])
                    other_pattern = '|'.join([f'\\b{kw}\\b' for kw in other_cabin_patterns])
                    combined = (
                        f"({exclusive_pattern}).*({other_pattern})"
                        f"|({other_pattern}).*({exclusive_pattern})"
                    )
                    exclusive_mask = filtered_ncs[incident_col].str.contains(
                        combined, na=False, regex=True, case=False
                    )
                    filtered_ncs = filtered_ncs[~exclusive_mask]

        # PASO 3: Fallback para Global
        if len(filtered_ncs) == 0 and node_path.strip() == "Global":
            return ncs_data

        return filtered_ncs
    except Exception as e:
        self.logger.error(f"Error in contextual filtering: {str(e)}")
        return ncs_data
```

### Nota sobre `_filter_using_routes_dictionary`

Este método usa un diccionario de rutas (descargado de Power BI con la query `Rutas Diccionario.txt`)
que mapea cada ruta IATA (ej: `MAD-JFK`) a su haul (LH o SH). La implementación depende de cómo
tu nuevo proyecto acceda a este diccionario. La lógica es:

1. Extraer rutas IATA del texto de cada incidente (`[A-Z]{3}-[A-Z]{3}`)
2. Buscar cada ruta en el diccionario para determinar si es LH o SH
3. Si `allow_unknown_route_incidents=True`, mantener incidentes sin ruta identificable
4. Si `allow_unknown_route_incidents=False`, solo mantener incidentes con ruta del haul correcto

---

## 7. Componente 3: Comparación Temporal

Solo se usa en el submodo COMPARATIVE. Genera una comparación estructurada entre dos períodos.

### Tipos de incidentes rastreados

| Tipo | Columnas NCS mapeadas |
|------|----------------------|
| `cancelaciones` | `Cancelaciones`, `Cancel` |
| `retrasos` | `Retrasos`, `Delay` |
| `desvios` | `Desvíos`, `Desvios`, `Divert` |
| `limitacion_aeronave` | `Limitación de la aeronave`, `Aircraft` |
| `otras_incidencias` | `Otras incidencias`, `Other`, `Otros` |

### Código

```python
def _create_temporal_ncs_comparison(
    self,
    current_data: pd.DataFrame,
    comparison_data: pd.DataFrame,
    node_path: str,
    current_start: str,
    current_end: str,
    comparison_start: str,
    comparison_end: str
) -> Dict[str, Any]:
    """
    Genera comparación estructurada entre dos períodos NCS.
    Produce:
    - Conteos totales por período
    - Matriz Ruta × Tipo de Incidente con deltas
    - Deltas globales por tipo de incidente
    - Patrones de mejora/empeoramiento
    """
    try:
        # 1. Extraer datos estructurados de ambos períodos
        current_structured = self._extract_structured_ncs_data_for_comparison(current_data)
        comparison_structured = self._extract_structured_ncs_data_for_comparison(comparison_data)

        # 2. Crear matriz Ruta × Tipo de Incidente
        route_matrix = self._create_route_incident_matrix(
            current_structured, comparison_structured
        )

        # 3. Calcular deltas globales por tipo
        incident_type_deltas = self._calculate_incident_type_deltas(
            current_structured, comparison_structured
        )

        return {
            'current_period': {
                'start': current_start, 'end': current_end,
                'total_incidents': len(current_data),
                'structured': current_structured
            },
            'comparison_period': {
                'start': comparison_start, 'end': comparison_end,
                'total_incidents': len(comparison_data),
                'structured': comparison_structured
            },
            'route_incident_matrix': route_matrix,
            'incident_type_deltas': incident_type_deltas,
            'delta_total': len(current_data) - len(comparison_data)
        }
    except Exception as e:
        self.logger.error(f"Error in temporal comparison: {str(e)}")
        return {}


def _extract_structured_ncs_data_for_comparison(
    self, data: pd.DataFrame
) -> Dict[str, Any]:
    """Extrae datos estructurados de un período para comparación"""
    if data.empty:
        return {'total': 0, 'by_route': {}, 'by_type': {}}

    result = {'total': len(data), 'by_route': {}, 'by_type': {}}

    # Conteo por tipo de incidente
    incident_counts = self._extract_incident_counts_from_data(data)
    result['by_type'] = incident_counts

    # Conteo por ruta
    route_analysis = self._extract_route_analysis_from_data(data)
    result['by_route'] = route_analysis.get('most_affected_routes', {})

    return result


def _create_route_incident_matrix(
    self,
    current: Dict[str, Any],
    comparison: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Crea matriz Ruta × Tipo de Incidente con deltas.
    Para cada ruta presente en cualquiera de los dos períodos,
    calcula: valor actual, valor previo, delta absoluto, % cambio.
    """
    all_routes = set(
        list(current.get('by_route', {}).keys()) +
        list(comparison.get('by_route', {}).keys())
    )

    matrix = {}
    for route in all_routes:
        current_count = current.get('by_route', {}).get(route, 0)
        comparison_count = comparison.get('by_route', {}).get(route, 0)
        delta = current_count - comparison_count
        pct_change = (
            round((delta / comparison_count) * 100, 1)
            if comparison_count > 0 else 0
        )
        matrix[route] = {
            'current': current_count,
            'previous': comparison_count,
            'delta': delta,
            'pct_change': pct_change
        }
    return matrix


def _calculate_incident_type_deltas(
    self,
    current: Dict[str, Any],
    comparison: Dict[str, Any]
) -> Dict[str, Dict]:
    """Calcula deltas globales por tipo de incidente"""
    all_types = set(
        list(current.get('by_type', {}).keys()) +
        list(comparison.get('by_type', {}).keys())
    )

    deltas = {}
    for incident_type in all_types:
        current_count = current.get('by_type', {}).get(incident_type, 0)
        comparison_count = comparison.get('by_type', {}).get(incident_type, 0)
        delta = current_count - comparison_count
        pct_change = (
            round((delta / comparison_count) * 100, 1)
            if comparison_count > 0 else 0
        )
        deltas[incident_type] = {
            'current': current_count,
            'previous': comparison_count,
            'delta': delta,
            'pct_change': pct_change
        }
    return deltas
```

---

## 8. Componente 4: Reflexión con LLM

Solo se usa en el submodo COMPARATIVE. El LLM analiza los incidentes filtrados y produce
un análisis causal estructurado.

### Flujo de reflexión

1. Se prepara un prompt con los datos NCS filtrados del período actual
2. Se incluye la comparación temporal (si disponible)
3. El LLM identifica causas raíz, rutas afectadas y nivel de confianza
4. Si el LLM falla, hay un fallback a pattern matching (`_extract_ncs_causal_insights_workflow_aware`)

### Código de reflexión

```python
async def _ncs_reflection_with_agent(
    self,
    filtered_ncs_data: pd.DataFrame,
    node_path: str,
    anomaly_type: str,
    total_days: int,
    comparison_data: pd.DataFrame = None,
    comparison_start: str = None,
    comparison_end: str = None
) -> Dict[str, Any]:
    """
    Análisis causal de incidentes NCS usando LLM.
    Retorna: identified_causes, affected_routes, confidence_level
    """
    try:
        # Preparar texto de incidentes
        incident_col = filtered_ncs_data.columns[0]
        incidents_text = "\n".join(
            filtered_ncs_data[incident_col].astype(str).tolist()
        )

        # Preparar contexto de comparación si disponible
        comparison_context = ""
        if comparison_data is not None and not comparison_data.empty:
            comp_incident_col = comparison_data.columns[0]
            comp_text = "\n".join(
                comparison_data[comp_incident_col].astype(str).tolist()
            )
            comparison_context = f"""
PERÍODO DE COMPARACIÓN ({comparison_start} a {comparison_end}):
{comp_text[:3000]}
"""

        # Construir prompt para el LLM
        prompt = f"""Analiza los siguientes incidentes operacionales NCS para el segmento {node_path}.
Tipo de anomalía NPS: {anomaly_type}
Días analizados: {total_days}

INCIDENTES DEL PERÍODO ACTUAL:
{incidents_text[:5000]}

{comparison_context}

INSTRUCCIONES:
1. Identifica las CAUSAS RAÍZ principales de los incidentes
2. Lista las RUTAS AFECTADAS (formato IATA: XXX-YYY)
3. Determina el NIVEL DE CONFIANZA (high/medium/low)
4. Relaciona los incidentes con el tipo de anomalía NPS ({anomaly_type})

Responde en formato estructurado:
CAUSAS: [lista de causas]
RUTAS: [lista de rutas IATA]
CONFIANZA: [high/medium/low]
RESUMEN: [explicación causal]
"""

        messages = [
            {"role": "system", "content": "Eres un analista de operaciones aéreas."},
            {"role": "user", "content": prompt}
        ]

        response, _, _ = await self.agent.invoke(messages)

        # Parsear respuesta del LLM
        return self._parse_ncs_reflection_response(response.content)

    except Exception as e:
        self.logger.error(f"Error in NCS reflection: {str(e)}")
        # Fallback a pattern matching
        return await self._extract_ncs_causal_insights_workflow_aware(
            filtered_ncs_data, node_path, anomaly_type
        )
```

### Fallback: Pattern Matching sin LLM

```python
async def _extract_ncs_causal_insights_workflow_aware(
    self,
    filtered_ncs_data: pd.DataFrame,
    node_path: str,
    anomaly_type: str
) -> Dict[str, Any]:
    """
    Extrae insights causales usando pattern matching cuando el LLM falla.
    Clasifica incidentes como operativos vs producto.
    """
    if filtered_ncs_data.empty:
        return {
            'identified_causes': [],
            'affected_routes': [],
            'route_impact_summary': {},
            'touchpoint_correlations': {},
            'confidence_level': 'low'
        }

    incident_col = filtered_ncs_data.columns[0]
    all_text = "\n".join(filtered_ncs_data[incident_col].astype(str).tolist())

    # Extraer datos estructurados
    structured = self._extract_structured_ncs_data(all_text)

    # Extraer rutas
    routes = re.findall(r'[A-Z]{3}-[A-Z]{3}', all_text)
    unique_routes = list(set(routes))

    # Identificar causas por keywords
    causes = []
    cause_keywords = {
        'Retrasos operativos': ['retraso', 'delay', 'delayed'],
        'Cancelaciones': ['cancel', 'anulad', 'cancelled'],
        'Problemas técnicos': ['técnic', 'technical', 'avería', 'maintenance'],
        'Condiciones meteorológicas': ['weather', 'meteorológ', 'tormenta'],
        'Problemas de equipaje': ['equipaje', 'baggage', 'maleta'],
        'Desvíos': ['desvío', 'divert', 'reroute']
    }
    text_lower = all_text.lower()
    for cause, keywords in cause_keywords.items():
        if any(kw in text_lower for kw in keywords):
            causes.append(cause)

    return {
        'identified_causes': causes,
        'affected_routes': unique_routes[:10],
        'route_impact_summary': structured.get('summary', {}),
        'touchpoint_correlations': {},
        'confidence_level': 'medium' if len(causes) > 0 else 'low'
    }
```


---

## 9. Submodo SINGLE

El submodo single analiza incidentes absolutos de un período sin comparación temporal.
Es más simple que el comparative: no calcula deltas, no usa reflexión LLM, y se apoya
directamente en `analyze_ncs_incidents_for_period()` del `NCSDataCollector`.

### Flujo paso a paso

```
1. Convertir fechas string → datetime
2. Crear NCSDataCollector
3. Recopilar datos NCS del período:
   └── NCSDataCollector.collect_ncs_data_for_date_range(start_dt, end_dt)
4. Si no hay datos → retornar "No NCS data found"
5. Filtrar por segmento:
   └── _filter_ncs_by_segment(ncs_data, node_path)
6. Si no hay datos tras filtrar → retornar "No incidents for segment"
7. Analizar incidentes filtrados:
   └── NCSDataCollector.analyze_ncs_incidents_for_period(filtered_data, "all")
8. Construir resultado formateado (string)
```

### Código completo

```python
async def _ncs_tool_single_period(self, node_path: str, start_date: str, end_date: str) -> str:
    """NCS tool for single period analysis (absolute incidents only)"""
    try:
        self.logger.info(
            f"Collecting NCS data for {node_path} from {start_date} to {end_date} (single period)"
        )

        # Convertir fechas
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_dt = start_date

        if isinstance(end_date, str):
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = end_date

        # Crear collector y recopilar datos
        from ....data_collection.ncs_collector import NCSDataCollector
        ncs_collector = NCSDataCollector(environment=self.environment)

        ncs_data = ncs_collector.collect_ncs_data_for_date_range(
            start_date=start_dt, end_date=end_dt
        )

        if ncs_data.empty:
            return f"No NCS data found for {node_path} from {start_date} to {end_date}"

        # Filtrar por segmento ANTES del análisis
        filtered_ncs_data = await self._filter_ncs_by_segment(ncs_data, node_path)

        if filtered_ncs_data.empty:
            return (
                f"No NCS incidents found for segment {node_path} "
                f"from {start_date} to {end_date} after filtering"
            )

        # Analizar los datos FILTRADOS
        incident_analysis = ncs_collector.analyze_ncs_incidents_for_period(
            filtered_ncs_data, analysis_focus="all"
        )

        # --- Construir resultado ---
        result_parts = []
        result_parts.append(f"📊 **INCIDENTES NCS - PERIODO ÚNICO**")
        result_parts.append(f"📅 Período: {start_date} a {end_date}")
        result_parts.append(f"🎯 Segmento: {node_path}")
        result_parts.append(f"Total Incidentes (Segmento): {len(filtered_ncs_data)}")
        result_parts.append("")
        result_parts.append("**INCIDENTES DEL PERIODO:**")

        # Conteo por tipo
        incident_counts = incident_analysis.get("incident_counts", {})
        for incident_type, count in incident_counts.items():
            result_parts.append(f"• {incident_type}: {count} incidentes")

        # Incidentes detallados
        if "detailed_incidents" in incident_analysis:
            detailed = incident_analysis["detailed_incidents"]
            result_parts.append(
                f"• Incidentes con descripción detallada: {detailed['count']}"
            )
            if detailed.get("sample_incidents"):
                result_parts.append("• Ejemplos de incidentes:")
                for i, incident in enumerate(detailed["sample_incidents"][:2], 1):
                    truncated = incident[:150] + "..." if len(incident) > 150 else incident
                    result_parts.append(f"  {i}. {truncated}")
            if detailed.get("incident_themes"):
                result_parts.append("• Temas principales:")
                for theme in detailed["incident_themes"][:3]:
                    result_parts.append(f"  - {theme}")

        # Análisis de rutas
        if "route_analysis" in incident_analysis:
            route_info = incident_analysis["route_analysis"]
            result_parts.append(f"• Rutas afectadas: {route_info['total_routes_affected']}")
            if route_info.get("most_affected_routes"):
                result_parts.append("• Rutas más impactadas:")
                for route, count in list(route_info["most_affected_routes"].items())[:3]:
                    result_parts.append(f"  - {route}: {count} incidentes")

        # Resumen
        if "summary_insights" in incident_analysis:
            result_parts.append("• Resumen de impacto:")
            for insight in incident_analysis["summary_insights"][:3]:
                result_parts.append(f"  - {insight}")

        result_parts.append("")
        result_parts.append(
            "**NOTA:** Estos son incidentes absolutos del período específico, "
            "sin comparación temporal."
        )

        return "\n".join(result_parts)

    except Exception as e:
        self.logger.error(
            f"❌ Error in single period NCS tool: {type(e).__name__}: {str(e)}"
        )
        return f"ERROR in NCS tool: {type(e).__name__}: {str(e)}"
```

### Diferencias clave con COMPARATIVE

| Aspecto | SINGLE | COMPARATIVE |
|---------|--------|-------------|
| Período de comparación | No | Sí (automático o explícito) |
| `_create_temporal_ncs_comparison` | No | Sí |
| `_ncs_reflection_with_agent` (LLM) | No | Sí |
| `_extract_structured_ncs_data` | No | Sí |
| Análisis de incidentes | `analyze_ncs_incidents_for_period()` | Reflexión LLM + fallback pattern matching |
| Almacena en `collected_data` | No (retorna string directo) | Sí (`self.collected_data['ncs_data']`) |
| Complejidad del output | Simple (conteos + rutas) | Complejo (causas, deltas, confianza, narrativa) |



---

## 10. Submodo COMPARATIVE

El submodo comparative es el más completo. Compara el período actual contra un período
de referencia, calcula deltas, y usa reflexión LLM para análisis causal.

### Flujo paso a paso

```
1.  Convertir fechas string → datetime
2.  Calcular período de comparación:
    ├── Si hay self.comparison_start_date/end_date → usarlos (modo "vs Sel. Period")
    └── Si no → período anterior de misma duración
        Ej: actual 13-19 ene → comparación 06-12 ene
3.  Recopilar datos NCS período actual:
    └── NCSDataCollector.collect_ncs_data_for_date_range(start_dt, end_dt)
4.  Filtrar por segmento (período actual):
    └── _filter_ncs_by_segment(ncs_data, node_path, collect_unattributed_key="current")
5.  Si error AWS → retornar error con causal_confidence='no_data_source'
6.  Si no hay datos → retornar "No incidents found" con causal_confidence='file_found_no_incidents'
7.  Recopilar datos NCS período comparación:
    └── NCSDataCollector.collect_ncs_data_for_date_range(comp_start, comp_end)
8.  Filtrar por segmento (período comparación):
    └── _filter_ncs_by_segment(comparison_data, node_path, collect_unattributed_key="comparison")
9.  Comparación temporal:
    └── _create_temporal_ncs_comparison(current, comparison, ...)
10. Filtrar datos actuales por segmento (final):
    └── _filter_ncs_by_segment(ncs_data, node_path)
11. Si no hay datos tras filtrar → retornar "incidentes en otros segmentos"
12. Extraer datos estructurados:
    └── _extract_structured_ncs_data(all_incidents_text)
13. Análisis causal con LLM:
    └── _ncs_reflection_with_agent(filtered_data, ...)
    └── Si falla → fallback a _extract_ncs_causal_insights_workflow_aware()
14. Construir resultado formateado (string con separadores " | ")
15. Almacenar en self.collected_data['ncs_data']
```

### Código completo (simplificado para recreación)

```python
async def _ncs_tool(
    self,
    node_path: str,
    start_date: str,
    end_date: str,
    analysis_focus: str = "flights",
    temporal_comparison: bool = True
) -> str:
    """
    NCS tool para análisis causal comparativo.
    Compara período actual vs período de referencia con deltas.
    """
    try:
        self.logger.info(
            f"Collecting NCS operational incidents for {node_path} "
            f"from {start_date} to {end_date}"
        )

        # 1. Convertir fechas
        start_dt = (datetime.strptime(start_date, '%Y-%m-%d')
                    if isinstance(start_date, str) else start_date)
        end_dt = (datetime.strptime(end_date, '%Y-%m-%d')
                  if isinstance(end_date, str) else end_date)

        total_days = (end_dt - start_dt).days + 1

        # 2. Crear collector
        from ....data_collection.ncs_collector import NCSDataCollector
        ncs_collector = NCSDataCollector(environment=self.environment)

        # 3. Calcular período de comparación
        comparison_data = None
        comparison_start_dt = None
        comparison_end_dt = None

        if temporal_comparison:
            if (hasattr(self, 'comparison_start_date') and
                hasattr(self, 'comparison_end_date') and
                self.comparison_start_date and self.comparison_end_date):
                # Usar período de comparación explícito ("vs Sel. Period")
                comparison_start_dt = (
                    self.comparison_start_date
                    if isinstance(self.comparison_start_date, datetime)
                    else datetime.strptime(self.comparison_start_date, '%Y-%m-%d')
                )
                comparison_end_dt = (
                    self.comparison_end_date
                    if isinstance(self.comparison_end_date, datetime)
                    else datetime.strptime(self.comparison_end_date, '%Y-%m-%d')
                )
            else:
                # Calcular período anterior de misma duración
                comparison_end_dt = start_dt - timedelta(days=1)
                comparison_start_dt = comparison_end_dt - timedelta(days=total_days - 1)

        # 4. Recopilar datos período actual
        try:
            ncs_data = ncs_collector.collect_ncs_data_for_date_range(start_dt, end_dt)
            # Filtrar por segmento inmediatamente
            ncs_data = await self._filter_ncs_by_segment(
                ncs_data, node_path, collect_unattributed_key="current"
            )
            access_error = False
            error_msg = None
        except Exception as e:
            error_msg = str(e)
            aws_errors = ['invalidaccesskeyid', 'access denied', 'credentials', 'token']
            access_error = True
            ncs_data = pd.DataFrame()

        # 5. Recopilar datos período comparación
        if temporal_comparison and not access_error and comparison_start_dt and comparison_end_dt:
            try:
                comparison_data = ncs_collector.collect_ncs_data_for_date_range(
                    comparison_start_dt, comparison_end_dt
                )
                comparison_data = await self._filter_ncs_by_segment(
                    comparison_data, node_path, collect_unattributed_key="comparison"
                )
            except Exception:
                comparison_data = pd.DataFrame()
        else:
            comparison_data = pd.DataFrame()

        # CASO ERROR AWS
        if access_error:
            self.collected_data['ncs_data'] = {
                'analysis_summary': f"❌ NCS data source unavailable for {start_date} to {end_date}",
                'identified_causes': [],
                'affected_routes': [],
                'causal_confidence': 'no_data_source',
                'data_source_status': 'unavailable',
                'error_details': error_msg[:200] if error_msg else 'Unknown error',
                'days_analyzed': total_days,
                'temporal_comparison_successful': False
            }
            return (
                f"❌ NCS data source unavailable for {start_date} to {end_date}. "
                f"Error: {error_msg[:100] if error_msg else 'Unknown'}"
            )

        # CASO SIN DATOS
        if ncs_data.empty:
            self.collected_data['ncs_data'] = {
                'analysis_summary': f"📅 No NCS incidents found for {start_date} to {end_date}",
                'identified_causes': [],
                'affected_routes': [],
                'causal_confidence': 'file_found_no_incidents',
                'data_source_status': 'available_but_empty',
                'days_analyzed': total_days
            }
            return (
                f"📅 No NCS operational incidents found for {total_days}-day period "
                f"{start_date} to {end_date}. Could indicate good operational performance."
            )

        # 6. COMPARACIÓN TEMPORAL
        temporal_analysis = None
        comparison_successful = False

        if (temporal_comparison and comparison_data is not None and
            not comparison_data.empty and comparison_start_dt and comparison_end_dt):
            filtered_current = await self._filter_ncs_by_segment(ncs_data, node_path)
            filtered_comparison = await self._filter_ncs_by_segment(comparison_data, node_path)

            temporal_analysis = self._create_temporal_ncs_comparison(
                filtered_current, filtered_comparison, node_path,
                start_date, end_date,
                comparison_start_dt.strftime('%Y-%m-%d'),
                comparison_end_dt.strftime('%Y-%m-%d')
            )
            comparison_successful = True

        # 7. Filtrar datos actuales por segmento
        filtered_ncs_data = await self._filter_ncs_by_segment(ncs_data, node_path)

        if filtered_ncs_data.empty:
            total_global = len(ncs_data)
            self.collected_data['ncs_data'] = {
                'analysis_summary': f"📊 Incidentes en otras rutas - {node_path} no afectado",
                'identified_causes': [],
                'affected_routes': [],
                'causal_confidence': 'file_found_no_segment_incidents',
                'incidents_in_other_segments': True,
                'non_correlated_incidents': total_global,
                'days_analyzed': total_days
            }
            return (
                f"📊 Se encontraron {total_global} incidentes pero ninguno "
                f"afectó las rutas del segmento {node_path}."
            )

        # 8. Extraer datos estructurados
        incident_col = filtered_ncs_data.columns[0]
        all_incidents_text = "\n".join(
            filtered_ncs_data[incident_col].astype(str).tolist()
        )
        structured_ncs_data = self._extract_structured_ncs_data(all_incidents_text)

        # 9. Análisis causal con LLM
        current_anomaly_type = getattr(self, 'current_anomaly_type', 'unknown')
        nps_variation = getattr(self, 'nps_difference_value', None)

        # Extraer deltas de incidentes de la comparación temporal
        temporal_incident_changes = {}
        if temporal_analysis:
            temporal_incident_changes = temporal_analysis.get('incident_type_deltas', {})

        # Preparar datos de comparación para reflexión
        comparison_data_for_reflection = None
        comparison_start_str = None
        comparison_end_str = None
        if temporal_comparison and comparison_data is not None and not comparison_data.empty:
            comparison_data_for_reflection = await self._filter_ncs_by_segment(
                comparison_data, node_path
            )
            comparison_start_str = comparison_start_dt.strftime('%Y-%m-%d') if comparison_start_dt else None
            comparison_end_str = comparison_end_dt.strftime('%Y-%m-%d') if comparison_end_dt else None

        try:
            # Intentar reflexión con LLM
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

            # Si reflexión vacía → fallback
            if (not causal_analysis.get('identified_causes') and
                not causal_analysis.get('affected_routes')):
                causal_analysis = await self._extract_ncs_causal_insights_workflow_aware(
                    filtered_ncs_data, node_path, current_anomaly_type
                )
        except Exception:
            # Fallback a pattern matching
            causal_analysis = await self._extract_ncs_causal_insights_workflow_aware(
                filtered_ncs_data, node_path, current_anomaly_type
            )

        # 10. Construir resultado formateado
        analysis_result = []
        analysis_result.append(f"🎯 NCS SEGMENT ANALYSIS: {node_path}")
        analysis_result.append(f"📅 Period: {start_date} to {end_date}")

        # Causas identificadas
        if causal_analysis.get('identified_causes'):
            analysis_result.append("🚨 DISRUPTION CAUSES IDENTIFICADAS:")
            for i, cause in enumerate(causal_analysis['identified_causes'][:3], 1):
                analysis_result.append(f"  {i}. {cause}")

        # Rutas afectadas
        if causal_analysis.get('affected_routes'):
            analysis_result.append(
                f"🛤️ AFFECTED ROUTES ({len(causal_analysis['affected_routes'])} routes):"
            )
            route_impacts = causal_analysis.get('route_impact_summary', {})
            for route, impact in list(route_impacts.items())[:4]:
                analysis_result.append(f"  • {route}: {impact}")

        # Datos estructurados (breakdown por categoría, motivos, pasajeros, retrasos)
        structured = causal_analysis.get('structured_ncs_data', {})
        if structured and structured.get('summary', {}).get('total_incidents', 0) > 0:
            analysis_result.append("📊 NCS STRUCTURED BREAKDOWN:")
            categories = structured.get('categories', {})
            if categories:
                cat_summary = [f"{cat}: {c}" for cat, c in categories.items() if c > 0]
                analysis_result.append(f"   📋 Categorías: {', '.join(cat_summary)}")
            motives = structured.get('motives_breakdown', {})
            if motives:
                top = sorted(motives.items(), key=lambda x: x[1], reverse=True)[:3]
                analysis_result.append(
                    f"   🔧 Motivos principales: {', '.join(f'{m}: {c}' for m, c in top)}"
                )
            passengers = structured.get('passenger_impact', {})
            if passengers.get('total', 0) > 0:
                analysis_result.append(
                    f"   👥 Pasajeros afectados: J:{passengers.get('j_class',0)}, "
                    f"W:{passengers.get('w_class',0)}, Y:{passengers.get('y_class',0)} "
                    f"= {passengers['total']} pax"
                )
            delays = structured.get('delay_statistics', {})
            if delays.get('count', 0) > 0:
                analysis_result.append(
                    f"   ⏱️ Retrasos: {delays['count']} vuelos, "
                    f"promedio {delays.get('avg_delay',0):.1f} min"
                )

            # Rutas con breakdown detallado por tipo de incidente
            route_breakdown = structured.get('route_incident_breakdown', {})
            if route_breakdown:
                def sort_key(item):
                    _, b = item
                    return (b['total'],
                            b['cancelaciones']*5 + b['desvios']*4 +
                            b['limitacion_aeronave']*3 + b['retrasos']*2 +
                            b['otras_incidencias'])
                sorted_routes = sorted(route_breakdown.items(), key=sort_key, reverse=True)
                details = []
                for route, b in sorted_routes:
                    parts = []
                    if b['cancelaciones'] > 0: parts.append(f"C:{b['cancelaciones']}")
                    if b['desvios'] > 0: parts.append(f"D:{b['desvios']}")
                    if b['retrasos'] > 0: parts.append(f"R:{b['retrasos']}")
                    if b['otras_incidencias'] > 0: parts.append(f"O:{b['otras_incidencias']}")
                    if b['limitacion_aeronave'] > 0: parts.append(f"L:{b['limitacion_aeronave']}")
                    details.append(f"{route}({'|'.join(parts) if parts else b['total']})")
                analysis_result.append(f"   🛫 RUTAS AFECTADAS: {', '.join(details)}")
                analysis_result.append(
                    "   📝 Leyenda: C=Cancelaciones, D=Desvíos, R=Retrasos, "
                    "O=Otras, L=Limitación aeronave"
                )

        # Conteo de incidentes del segmento
        segment_count = len(filtered_ncs_data)
        if total_days > 1:
            analysis_result.append(
                f"📊 SEGMENT INCIDENTS: {segment_count} incidents on {node_path} "
                f"routes across {total_days} days"
            )
        else:
            analysis_result.append(
                f"📊 SEGMENT INCIDENTS: {segment_count} incidents on {node_path} routes"
            )

        # Comparación temporal en el resultado
        if temporal_analysis is not None:
            causal_analysis['temporal_comparison'] = temporal_analysis
            causal_analysis['temporal_comparison_successful'] = comparison_successful
            if temporal_analysis.get('analysis_summary'):
                analysis_result.append(
                    temporal_analysis['analysis_summary'].replace(' | ', '\n')
                )

        # Almacenar datos completos
        self.collected_data['ncs_data'] = causal_analysis

        return " | ".join(analysis_result)

    except Exception as e:
        self.logger.error(f"💥 Error in NCS analysis: {str(e)}")
        return f"Error in NCS analysis: {str(e)}"
```

### Casos de resultado

| Caso | Condición | `causal_confidence` | Resultado |
|------|-----------|---------------------|-----------|
| Error AWS | Credenciales inválidas/expiradas | `no_data_source` | `"❌ NCS data source unavailable..."` |
| Sin datos | `ncs_data.empty` tras recopilación | `file_found_no_incidents` | `"📅 No NCS incidents found..."` |
| Sin datos segmento | Incidentes globales pero no en el segmento | `file_found_no_segment_incidents` | `"📊 Se encontraron X incidentes pero ninguno afectó {node_path}"` |
| Datos encontrados | Incidentes en el segmento | `high_agent_analysis` o `medium` | Análisis completo con causas, rutas, deltas, narrativa |



---

## 11. Prompts del Agente Causal

Los prompts se definen en un archivo YAML (`causal_explanation.yaml`) y se cargan al inicializar
el agente. Aquí se extraen los prompts relevantes para la NCS tool en ambos modos.

### Estructura del YAML

```yaml
# causal_explanation.yaml
comparative_prompts:
  system_prompt: |
    ...
  input_template: |
    ...
  reflection_prompt: |
    ...

single_prompts:
  system_prompt: |
    ...
  input_template: |
    ...
  reflection_prompt: |
    ...

tools_prompts:
  ncs_tool:
    comparative:
      operative: |
        ...
      product: |
        ...
      mixed: |
        ...
    single: |
      ...
```

### System Prompt — Modo COMPARATIVE (extracto relevante a NCS)

```yaml
comparative_prompts:
  system_prompt: |
    Eres un analista sénior de NPS especializado en investigación de anomalías.

    ⚠️ NOMENCLATURA DE COMPAÑÍAS (OBLIGATORIO):
    - Las compañías SIEMPRE se escriben como IB e YW
    - NUNCA uses: "Iberia", "Young Wings", ni ninguna otra variación

    ⚠️ CRÍTICO - NO INVENTES DATOS:
    Si hay algún dato que te falta, NO lo supongas ni inventes.
    Para las subidas o bajadas de cualquier variable, menciona el valor exacto, NUNCA el %.

    TRES OBJETIVOS —en este orden—:
    1. CAUSA TANGIBLE · ¿Qué factor concreto explica la anomalía?
    2. RUTAS AFECTADAS · ¿Qué rutas específicas vivieron esa causa?
    3. REACTIVIDAD DE CLIENTE · ¿Qué perfiles reaccionaron con mayor intensidad?

    Gestión de Hipótesis:
    ▸ TRIANGULACIÓN DE RUTAS (CRÍTICO): Si una misma ruta aparece en múltiples
      fuentes (NCS + Verbatims + Drivers), considérala una "Causa Confirmada".
    ▸ Nivel de Confianza:
      * ALTA: Intersección de 3 fuentes en las mismas rutas/causas.
      * MEDIA: Intersección de 2 fuentes.
      * BAJA: Evidencia de una sola fuente.

    Validación Temporal NCS:
    ▸ SHAP positivo → Buscar REDUCCIÓN de incidentes vs período de referencia
    ▸ SHAP negativo → Buscar AUMENTO de incidentes vs período de referencia
    ▸ Coherencia: Drivers SHAP deben correlacionar con tendencias NCS temporales

    Mapa de Flujo de Trabajo:
    ▸ explanatory_drivers_tool → Determina flujo (operative/product/mixed)
    ▸ operative_data_tool → Valida métricas operativas vs drivers SHAP
    ▸ ncs_tool → Confirma tendencias temporales de incidentes
    ▸ verbatims_tool → Obtiene feedback cualitativo de clientes
    ▸ routes_tool → Identifica rutas específicas afectadas
    ▸ customer_profile_tool → Analiza impacto por perfil de cliente
```

### System Prompt — Modo SINGLE (extracto relevante a NCS)

```yaml
single_prompts:
  system_prompt: |
    Eres un analista sénior de NPS especializado en investigación de períodos únicos.

    TRES OBJETIVOS:
    1. CAUSA TANGIBLE · ¿Qué métricas operativas se desviaron de su media histórica?
    2. HIPÓTESIS ALTERNATIVAS · Si operative_data NO encuentra causa → buscar en NCS/verbatims
    3. TRIANGULACIÓN DE RUTAS · Busca rutas que aparezcan en NCS (incidentes) y Verbatims (quejas).
       Si coinciden, la confianza es ALTA.

    Flujo de Investigación:
    ▸ CASO A (ALTA CONFIANZA): NCS y Verbatims coinciden en las mismas rutas y causas.
    ▸ CASO B (MEDIA CONFIANZA): Solo una fuente identifica la causa pero con fuerte evidencia.
    ▸ CASO C (BAJA CONFIANZA): Sin coincidencias, evidencia fragmentada.
```

### Reflection Prompt (común a ambos modos)

```yaml
reflection_prompt: |
  🔍 REFLEXIÓN SOBRE RESULTADOS DE {tool_name}

  📊 RESULTADOS DE LA HERRAMIENTA:
  {tool_result}

  INSTRUCCIONES:
  1. Analiza los resultados de la herramienta
  2. Evalúa si se han cumplido los objetivos de esta iteración
  3. Decide si continuar con la siguiente herramienta o terminar
  4. Justifica tu decisión basándote en la evidencia obtenida

  RESPONDE EXACTAMENTE EN ESTE FORMATO MARKDOWN:

  ```reflection
  [Tu reflexión sobre los resultados]
  ```

  ```next_tool
  [Nombre de la siguiente herramienta, o "TERMINAR"]
  ```
```

### Tool-Specific Prompts para NCS

Estos prompts se inyectan después de que la NCS tool retorna resultados, para guiar
al LLM sobre qué hacer a continuación.

#### Modo COMPARATIVE — Flujo Operativo

```yaml
tools_prompts:
  ncs_tool:
    comparative:
      operative: |
        FLUJO OPERATIVO - Después de ncs_tool

        🎯 ANÁLISIS COMPLETADO: Se confirmaron las tendencias temporales de incidentes operativos.

        PRÓXIMO PASO: Ejecuta verbatims_tool para obtener feedback cualitativo
        sobre la experiencia operativa.

        OBJETIVO: Entender las opiniones específicas de los clientes sobre los
        aspectos operativos identificados.
```

#### Modo COMPARATIVE — Flujo Producto

```yaml
      product: |
        FLUJO PRODUCTO - Después de ncs_tool

        🎯 ANÁLISIS COMPLETADO: Se revisaron los incidentes NCS
        (aunque no son el foco principal).

        PRÓXIMO PASO: Ejecuta verbatims_tool para obtener feedback cualitativo
        sobre la experiencia del producto.
```

#### Modo COMPARATIVE — Flujo Mixto

```yaml
      mixed: |
        FLUJO MIXTO - Después de ncs_tool

        🎯 ANÁLISIS COMPLETADO: Se confirmaron las tendencias temporales de incidentes operativos.

        PRÓXIMO PASO: Ejecuta verbatims_tool para obtener feedback cualitativo
        sobre la experiencia operativa.
```

#### Modo SINGLE

```yaml
    single: |
      ANÁLISIS COMPLETADO: Se revisaron los incidentes NCS del día.

      HIPÓTESIS GENERADA (si aplica):
      - Si hay incidentes en rutas específicas → ANOTA esas rutas para verificar en routes_tool
      - Si no hay incidentes → buscar pistas en verbatims

      PRÓXIMO PASO: Ejecuta verbatims_tool para obtener feedback cualitativo de los clientes.

      OBJETIVO: Buscar menciones de problemas específicos (retrasos, equipaje, tripulación)
      que puedan generar hipótesis alternativas.
```

### Cómo se cargan los prompts

```python
import yaml

class CausalExplanationAgent:
    def __init__(self, config_path: str = "config/prompts/causal_explanation.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def _get_system_prompt(self, mode: str = "comparative") -> str:
        if mode == "single":
            return self.config['single_prompts']['system_prompt']
        return self.config['comparative_prompts']['system_prompt']

    def _get_helper_prompt_for_tool(self, tool_name: str, workflow_type: str = "operative") -> str:
        """Obtiene el prompt helper post-tool para guiar la siguiente decisión"""
        tool_prompts = self.config.get('tools_prompts', {}).get(tool_name, {})
        if isinstance(tool_prompts, dict):
            mode_prompts = tool_prompts.get('comparative', {})
            if isinstance(mode_prompts, dict):
                return mode_prompts.get(workflow_type, "")
            return mode_prompts
        return tool_prompts
```



---

## 12. Integración con el Agente

### Dispatch de herramientas

El agente causal tiene un sistema de dispatch unificado que enruta la ejecución de cada
herramienta según el modo (single o comparative).

```python
async def _execute_tool_unified(
    self,
    tool_name: str,
    node_path: str,
    start_date: str,
    end_date: str,
    iteration: int,
    mode: str = "comparative",
    baseline_periods: int = 7,
    comparison_context: str = ""
) -> str:
    """Punto de entrada unificado para ejecutar cualquier herramienta"""
    if mode == "single":
        return await self._execute_single_period_tool(
            tool_name, node_path, start_date, end_date, iteration
        )
    else:
        return await self._execute_tool_comparative(
            tool_name, node_path, start_date, end_date,
            iteration, baseline_periods, comparison_context
        )
```

#### Dispatch COMPARATIVE

```python
async def _execute_tool_comparative(self, tool_name, node_path, start_date, end_date, ...):
    if tool_name == "ncs_tool":
        return await self._ncs_tool(
            node_path=node_path,
            start_date=start_date,
            end_date=end_date
        )
    # ... otros tools
```

#### Dispatch SINGLE

```python
async def _execute_single_period_tool(self, tool_name, node_path, start_date, end_date, ...):
    if tool_name == "ncs_tool":
        return await self._ncs_tool_single_period(
            node_path=node_path,
            start_date=start_date,
            end_date=end_date
        )
    # ... otros tools
```

### Herramientas disponibles

```python
valid_tools = [
    'explanatory_drivers_tool',   # Drivers SHAP
    'operative_data_tool',        # Métricas operativas (OTP, mishandling, etc.)
    'ncs_tool',                   # ← Incidentes operacionales
    'verbatims_tool',             # Feedback cualitativo de clientes
    'routes_tool',                # Rendimiento NPS por ruta
    'customer_profile_tool'       # Impacto por perfil de cliente
]
```

### Flujo de iteración del agente

El agente ejecuta herramientas en un bucle iterativo (máximo 5 iteraciones):

```
1. LLM decide qué herramienta ejecutar
2. _execute_tool_unified() → dispatch al método correcto
3. Resultado de la herramienta se devuelve al LLM
4. Se inyecta el helper prompt post-tool (tools_prompts.ncs_tool.*)
5. LLM reflexiona sobre los resultados (reflection_prompt)
6. LLM decide: siguiente herramienta o TERMINAR
7. Si no TERMINAR → volver a paso 2
```

### CleanConversationTracker — Extracción de rutas

El `CleanConversationTracker` detecta automáticamente rutas IATA mencionadas en los
resultados de `ncs_tool` y las almacena para uso posterior en `routes_tool`:

```python
class CleanConversationTracker:
    def __init__(self):
        self.identified_routes = []

    def track_tool_result(self, tool_name: str, result: str):
        if tool_name in ['ncs_tool', 'verbatims_tool']:
            routes = self._extract_routes_from_result(result)
            if routes:
                self.identified_routes.extend(routes)

    def _extract_routes_from_result(self, result: str) -> List[str]:
        """Extrae rutas IATA (XXX-YYY) del resultado de una herramienta"""
        return list(set(re.findall(r'[A-Z]{3}-[A-Z]{3}', result)))
```

Estas rutas se cruzan después con datos NPS de Power BI para obtener el impacto
en NPS de las rutas con incidentes.



---

## 13. Estructuras de Datos de Salida

### `self.collected_data['ncs_data']` — Modo COMPARATIVE (caso exitoso)

```python
{
    # Resumen del análisis
    'analysis_summary': str,                    # "🎯 NCS SEGMENT ANALYSIS: Global/SH/Economy..."

    # Causas raíz identificadas (por LLM o pattern matching)
    'identified_causes': [                      # Lista de causas
        "Retrasos operativos en rutas MAD-BCN y MAD-PMI",
        "Cancelaciones por condiciones meteorológicas",
        ...
    ],

    # Rutas afectadas (códigos IATA)
    'affected_routes': [
        "MAD-BCN", "MAD-PMI", "MAD-LIS", ...
    ],

    # Detalle de impacto por ruta
    'route_impact_details': {
        "MAD-BCN": {"incidents": 5, "type": "retrasos"},
        ...
    },
    'route_impact_summary': {
        "MAD-BCN": "5 incidents (3 delays, 2 cancellations)",
        ...
    },

    # Correlaciones con touchpoints NPS
    'touchpoint_correlations': {
        "puntualidad": "high",
        "equipaje": "medium",
        ...
    },

    # Nivel de confianza del análisis
    'causal_confidence': str,
    # Valores posibles:
    #   'high_agent_analysis'           → LLM identificó causas con alta confianza
    #   'medium'                        → Pattern matching encontró causas
    #   'low'                           → Poca evidencia
    #   'file_found_no_incidents'       → Archivos NCS revisados pero sin incidentes
    #   'file_found_no_segment_incidents' → Incidentes existen pero no en el segmento
    #   'no_data_source'                → Error de acceso a S3

    # Datos estructurados extraídos del texto de incidentes
    'structured_ncs_data': {
        'summary': {
            'total_incidents': int,
            'most_affected_route': ("MAD-BCN", 5),  # Tupla (ruta, conteo)
        },
        'categories': {
            'cancelaciones': int,
            'retrasos': int,
            'desvios': int,
            'limitacion_aeronave': int,
            'otras_incidencias': int
        },
        'motives_breakdown': {
            "Meteorología adversa": 3,
            "Problema técnico": 2,
            ...
        },
        'passenger_impact': {
            'total': int,
            'j_class': int,     # Business
            'w_class': int,     # Premium Economy
            'y_class': int      # Economy
        },
        'delay_statistics': {
            'count': int,
            'avg_delay': float,     # Minutos promedio
            'total_minutes': int
        },
        'route_disruptions': {
            "MAD-BCN": 5,
            "MAD-PMI": 3,
            ...
        },
        'route_incident_breakdown': {
            "MAD-BCN": {
                'total': 5,
                'cancelaciones': 2,
                'retrasos': 2,
                'desvios': 0,
                'limitacion_aeronave': 1,
                'otras_incidencias': 0
            },
            ...
        }
    },

    # Comparación temporal (solo si comparison_successful=True)
    'temporal_comparison': {
        'current_period': {
            'start': str, 'end': str,
            'total_incidents': int,
            'structured': {...}
        },
        'comparison_period': {
            'start': str, 'end': str,
            'total_incidents': int,
            'structured': {...}
        },
        'route_incident_matrix': {
            "MAD-BCN": {
                'current': 5, 'previous': 2,
                'delta': 3, 'pct_change': 150.0
            },
            ...
        },
        'incident_type_deltas': {
            'cancelaciones': {
                'current': 10, 'previous': 5,
                'delta': 5, 'pct_change': 100.0
            },
            ...
        },
        'delta_total': int,
        'analysis_summary': str
    },
    'temporal_comparison_successful': bool,

    # Metadata
    'data_source_status': str,      # 'available', 'available_but_empty', 'unavailable'
    'days_analyzed': int,
    'date_range': str,              # "2026-01-13 to 2026-01-19"

    # Workflow match (si aplica)
    'workflow_match': bool,
    'incident_nature': str,         # 'operative', 'product', 'mixed'
    'confidence_level': str         # 'high', 'medium', 'low'
}
```

### Modo SINGLE — Retorno directo (string)

El modo single no almacena en `collected_data`. Retorna un string formateado directamente:

```
📊 **INCIDENTES NCS - PERIODO ÚNICO**
📅 Período: 2026-01-15 a 2026-01-15
🎯 Segmento: Global/SH/Economy
Total Incidentes (Segmento): 12

**INCIDENTES DEL PERIODO:**
• cancelaciones: 3 incidentes
• retrasos: 7 incidentes
• desvios: 2 incidentes
• Incidentes con descripción detallada: 5
• Ejemplos de incidentes:
  1. Retraso de 45 min en vuelo IB1234 MAD-BCN por problema técnico...
  2. Cancelación vuelo IB5678 MAD-PMI por meteorología adversa...
• Temas principales:
  - delays: 7 incidents
  - cancellations: 3 incidents
  - technical_issues: 2 incidents
• Rutas afectadas: 8
• Rutas más impactadas:
  - MAD-BCN: 4 incidentes
  - MAD-PMI: 3 incidentes
  - MAD-LIS: 2 incidentes
• Resumen de impacto:
  - Total of 12 operational incidents detected
  - Route impact: 8 routes affected

**NOTA:** Estos son incidentes absolutos del período específico, sin comparación temporal.
```



---

## 14. Manejo de Errores

### Tabla de errores y comportamiento

| Error | Causa | Comportamiento | `causal_confidence` |
|-------|-------|----------------|---------------------|
| `InvalidAccessKeyId` | Credenciales AWS expiradas | Retorna mensaje de error, no intenta comparación | `no_data_source` |
| `Access Denied` | Sin permisos al bucket S3 | Igual que arriba | `no_data_source` |
| `Token expired` | Token de sesión temporal expirado | Igual que arriba | `no_data_source` |
| S3 vacío para el rango | No hay archivos NCS para esas fechas | "No incidents found" — puede indicar buena operación | `file_found_no_incidents` |
| Parsing HTML falla | Formato de email inesperado | DataFrame vacío para ese archivo, continúa con los demás | N/A (parcial) |
| Filtrado elimina todo | Incidentes existen pero no en el segmento | Mensaje indicando incidentes en otros segmentos | `file_found_no_segment_incidents` |
| LLM reflection falla | Error en el análisis con IA | Fallback a pattern matching (`_extract_ncs_causal_insights_workflow_aware`) | `medium` o `low` |
| Comparación falla | Error recopilando datos del período de comparación | Continúa solo con período actual (sin deltas) | Según análisis actual |

### Patrón de fallback LLM → Pattern Matching

```python
try:
    # Intentar reflexión con LLM
    causal_analysis = await self._ncs_reflection_with_agent(...)

    # Si reflexión vacía → fallback
    if not causal_analysis.get('identified_causes') and not causal_analysis.get('affected_routes'):
        causal_analysis = await self._extract_ncs_causal_insights_workflow_aware(...)
except Exception:
    # Error en LLM → fallback a pattern matching
    causal_analysis = await self._extract_ncs_causal_insights_workflow_aware(...)
```

### Detección de errores AWS

```python
aws_error_keywords = [
    'invalidaccesskeyid',
    'access denied',
    'credentials',
    'token'
]

if any(kw in error_msg.lower() for kw in aws_error_keywords):
    access_error = True
    # No intentar comparación ni análisis
```

### Resiliencia del parsing

El `NCSDataCollector` es resiliente a archivos individuales que fallan:
- Si un archivo no se puede parsear, se registra el error y se continúa con los demás
- Si no hay tablas HTML, se intenta extracción por texto plano (fallback)
- Si no se extrae nada, se retorna un DataFrame vacío para ese archivo

---

## 15. Dependencias y Stack

### Paquetes Python requeridos

```
# requirements.txt

# Core
pandas>=1.5.0
numpy>=1.21.0
python-dateutil>=2.8.0

# HTTP y async
aiohttp>=3.8.0

# AWS
boto3>=1.26.0
botocore>=1.29.0

# LangChain y LLMs
langchain==0.3.27
langchain-openai==0.3.28
langchain-core==0.3.72

# Validación de datos
pydantic>=2.0.0

# Configuración
pyyaml>=6.0
python-dotenv>=1.0.0

# Parsing HTML
beautifulsoup4>=4.12.0
```

### Configuración AWS

| Parámetro | Valor |
|-----------|-------|
| Región | `eu-west-1` |
| Bucket | `ibdata-prod-ew1-s3-customer` |
| Prefijo | `customer/catia/ncs/raw/attatchments/` |
| Autenticación | IAM roles (prod) o credenciales temporales (local) |

### Configuración LLM

```python
# Ejemplo de configuración OpenAI vía LangChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,          # Baja temperatura para análisis factual
    max_tokens=4096,
    api_key=os.getenv("OPENAI_API_KEY")
)
```

Variables de entorno LLM:

| Variable | Uso |
|----------|-----|
| `OPENAI_API_KEY` | API key de OpenAI |
| `OPENAI_MODEL` | Modelo a usar (default: `gpt-4o`) |
| `LLM_TEMPERATURE` | Temperatura del modelo (default: `0.1`) |

### Estructura mínima del nuevo proyecto

```
mi_proyecto/
├── config/
│   └── prompts/
│       └── causal_explanation.yaml     # Prompts del agente
├── data_collection/
│   └── ncs_collector.py                # NCSDataCollector (Sección 5)
├── agents/
│   └── causal_agent.py                 # Agente con _ncs_tool y _ncs_tool_single_period
├── utils/
│   ├── aws_session.py                  # Función get_aws_session()
│   └── segment_filters.py             # _get_node_filters(), diccionario de rutas
├── requirements.txt
├── .env                                # Variables de entorno
└── README.md
```

### Checklist de recreación

- [ ] Configurar acceso AWS (credenciales, bucket, prefijo)
- [ ] Implementar `NCSDataCollector` con parsing HTML (Sección 5)
- [ ] Implementar filtrado por segmento con diccionario de rutas (Sección 6)
- [ ] Implementar comparación temporal con deltas (Sección 7)
- [ ] Implementar reflexión LLM con fallback a pattern matching (Sección 8)
- [ ] Implementar `_ncs_tool_single_period` (Sección 9)
- [ ] Implementar `_ncs_tool` comparative (Sección 10)
- [ ] Configurar prompts YAML para ambos modos (Sección 11)
- [ ] Implementar dispatch unificado y tracker de rutas (Sección 12)
- [ ] Configurar LLM (OpenAI vía LangChain)
- [ ] Probar con datos reales de S3
