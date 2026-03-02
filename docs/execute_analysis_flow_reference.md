# execute_analysis_flow() - Referencia Completa para Agentes

## Descripción General

`execute_analysis_flow()` es la función principal del sistema de análisis de anomalías NPS. Ejecuta un flujo completo que incluye:

1. **Descarga de datos** desde Power BI
2. **Detección de anomalías** en métricas NPS
3. **Generación de explicaciones causales** usando AI

Esta función es el punto de entrada único para cualquier tipo de análisis NPS, ya sea diario, semanal, o con cualquier configuración personalizada.

---

## Firma de la Función

```python
async def execute_analysis_flow(
    analysis_date: datetime,           # REQUERIDO
    date_parameter: str,               # REQUERIDO
    segment: str,                      # REQUERIDO
    anomaly_detection_mode: str,       # REQUERIDO
    baseline_periods: int,             # REQUERIDO
    aggregation_days: int,             # REQUERIDO
    periods: int,                      # REQUERIDO
    causal_filter: Optional[str],      # REQUERIDO (puede ser None)
    comparison_start_date: Optional[datetime] = None,
    comparison_end_date: Optional[datetime] = None,
    date_flight_local: Optional[str] = None,
    study_mode: str = "comparative",
    environment: str = "prod",
) -> List[Dict] | None
```

---

## Argumentos Detallados

### 1. `analysis_date` (datetime) - REQUERIDO

**Propósito:** Fecha de referencia para el análisis. Representa el "punto final" desde donde se calculan los períodos hacia atrás.

**Comportamiento:**
- Período 1 = datos que terminan en `analysis_date`
- Período 2 = datos del período inmediatamente anterior
- Período N = datos de N-1 períodos antes de `analysis_date`

**Formato:** Objeto `datetime` de Python

**Ejemplos:**
```python
from datetime import datetime

# Analizar datos hasta el 20 de enero 2025
analysis_date = datetime(2025, 1, 20)

# Analizar datos hasta hoy menos 4 días (lag típico de PBI)
analysis_date = datetime.now() - timedelta(days=4)
```

**Notas importantes:**
- PBI tiene un lag típico de 4 días, por lo que `analysis_date` suele ser `hoy - 4 días`
- Si se usa `--insert-date-ci`, el sistema calcula automáticamente `fecha_simulada - 4 días`

---

### 2. `date_parameter` (str) - REQUERIDO

**Propósito:** Indica el origen/tipo de la fecha de análisis. Afecta el naming de carpetas y algunos cálculos internos.

**Valores válidos:**

| Valor | Origen | Descripción |
|-------|--------|-------------|
| `'insert_ci'` | `--insert-date-ci` | Fecha simulada. El sistema aplicó lag de 4 días |
| `'flight_local'` | `--date-flight-local` | Fecha directa del dashboard. Sin transformación |
| `'available'` | Default | Fecha calculada automáticamente (hoy - 4 días) |

**Uso interno:**
- Se concatena al nombre de carpeta de datos: `{date_parameter}_{study_mode}_{aggregation_days}d`
- Afecta cálculos de `reference_period` para baselines

**Ejemplo:**
```python
# Si el usuario especificó --insert-date-ci 2025-01-25
date_parameter = 'insert_ci'

# Si el usuario especificó --date-flight-local 2025-01-20
date_parameter = 'flight_local'

# Si no se especificó fecha (usa default)
date_parameter = 'available'
```

---

### 3. `segment` (str) - REQUERIDO

**Propósito:** Define el segmento raíz del árbol jerárquico NPS a analizar.

**Jerarquía completa del árbol NPS:**
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

**Valores válidos y nodos incluidos:**

| Valor | Nodos analizados | Total nodos |
|-------|------------------|-------------|
| `"Global"` | Árbol completo | 12 |
| `"Global/LH"` | LH + Economy + Business + Premium | 4 |
| `"Global/SH"` | SH + Economy + Business + IB/YW | 7 |
| `"Global/LH/Economy"` | Solo LH Economy | 1 |
| `"Global/LH/Business"` | Solo LH Business | 1 |
| `"Global/LH/Premium"` | Solo LH Premium | 1 |
| `"Global/SH/Economy"` | SH Economy + IB + YW | 3 |
| `"Global/SH/Business"` | SH Business + IB + YW | 3 |
| `"Global/SH/Economy/IB"` | Solo SH Economy IB | 1 |
| `"Global/SH/Economy/YW"` | Solo SH Economy YW | 1 |
| `"Global/SH/Business/IB"` | Solo SH Business IB | 1 |
| `"Global/SH/Business/YW"` | Solo SH Business YW | 1 |

**Shortcuts aceptados:**
```python
"SH"          → "Global/SH"
"LH"          → "Global/LH"
"Economy/SH"  → "Global/SH/Economy"
"Business/SH" → "Global/SH/Business"
"Economy/LH"  → "Global/LH/Economy"
"Business/LH" → "Global/LH/Business"
"Premium/LH"  → "Global/LH/Premium"
```

**Recomendaciones:**
- Usar `"Global"` para análisis completo de la organización
- Usar segmentos específicos para análisis focalizados (más rápido, menos datos)

---

### 4. `anomaly_detection_mode` (str) - REQUERIDO

**Propósito:** Define el algoritmo para determinar si un valor NPS es anómalo.

**Valores válidos:**

| Modo | Fórmula de detección | Caso de uso |
|------|---------------------|-------------|
| `'target'` | `\|NPS_actual - target_mensual\| > threshold` | Comparar vs objetivos establecidos |
| `'mean'` | `\|NPS_actual - media_N_períodos\| > threshold` | Detectar desviaciones de tendencia |
| `'vslast'` | `\|NPS_actual - NPS_período_anterior\| > threshold` | Comparar vs período inmediato |
| `'vslast_dynamic'` | Igual que vslast pero período según `causal_filter` | Comparaciones flexibles |

**Threshold por defecto:** 5.0 puntos NPS

**Interacción con `study_mode`:**
```python
# IMPORTANTE: Si study_mode="single", el modo se comporta como 'mean'
# independientemente del valor especificado
if study_mode == "single":
    # Internamente usa baseline = media de últimos N períodos
    # causal_filter se fuerza a None
```

**Ejemplos de uso:**
```python
# Análisis vs targets mensuales
anomaly_detection_mode = 'target'

# Análisis de tendencia (detectar cambios vs histórico)
anomaly_detection_mode = 'mean'

# Análisis comparativo semana vs semana anterior
anomaly_detection_mode = 'vslast'
```

---

### 5. `baseline_periods` (int) - REQUERIDO

**Propósito:** Número de períodos históricos para calcular el baseline en modo `'mean'`.

**Aplica solo cuando:** `anomaly_detection_mode = 'mean'` o `study_mode = 'single'`

**Cálculo del baseline:**
```
baseline = promedio(NPS de los últimos baseline_periods períodos)
```

**Ejemplos:**
```python
# Baseline = media de últimos 7 días (análisis diario)
baseline_periods = 7
aggregation_days = 1

# Baseline = media de últimas 4 semanas (análisis semanal)
baseline_periods = 4
aggregation_days = 7

# Baseline = media de últimos 30 días
baseline_periods = 30
aggregation_days = 1
```

**Valores recomendados:**
- Análisis diario: 7-14 períodos
- Análisis semanal: 4-8 períodos
- Análisis mensual: 3-6 períodos

---

### 6. `aggregation_days` (int) - REQUERIDO

**Propósito:** Define la granularidad temporal de cada período de análisis.

**Valores comunes:**

| Valor | Tipo de análisis | Descripción |
|-------|------------------|-------------|
| `1` | Diario | Cada período = 1 día |
| `7` | Semanal | Cada período = 7 días |
| `14` | Quincenal | Cada período = 14 días |
| `30` | Mensual | Cada período = 30 días |

**Impacto en el análisis:**
```python
# Con analysis_date = 2025-01-20 y aggregation_days = 7:
# Período 1: 14-20 enero 2025
# Período 2: 07-13 enero 2025
# Período 3: 31 dic 2024 - 06 enero 2025

# Con analysis_date = 2025-01-20 y aggregation_days = 1:
# Período 1: 20 enero 2025
# Período 2: 19 enero 2025
# Período 3: 18 enero 2025
```

**Recomendaciones:**
- `aggregation_days=7` para análisis semanales (más estable, menos ruido)
- `aggregation_days=1` para detectar anomalías puntuales

---

### 7. `periods` (int) - REQUERIDO

**Propósito:** Número de períodos a analizar hacia atrás desde `analysis_date`.

**Comportamiento:**
- Se analizan períodos del 1 al N (donde N = `periods`)
- Período 1 = más reciente (termina en `analysis_date`)
- Período N = más antiguo

**Ejemplos:**
```python
# Analizar última semana solamente
periods = 1
aggregation_days = 7

# Analizar últimos 7 días individualmente
periods = 7
aggregation_days = 1

# Analizar últimas 4 semanas
periods = 4
aggregation_days = 7

# Analizar últimos 30 días individualmente
periods = 30
aggregation_days = 1
```

**Consideraciones de rendimiento:**
- Más períodos = más tiempo de ejecución
- Más períodos = más llamadas a PBI y LLM
- Recomendado: 1-7 períodos para análisis rutinarios

---

### 8. `causal_filter` (Optional[str]) - REQUERIDO (puede ser None)

**Propósito:** Define el período de referencia para comparaciones causales (operativos, verbatims, rutas).

**Valores válidos:**

| Valor | Comparación | Descripción |
|-------|-------------|-------------|
| `"vs L7d"` | Últimos 7 días | Compara métricas operativas vs semana anterior |
| `"vs L14d"` | Últimos 14 días | Compara vs últimas 2 semanas |
| `"vs LM"` | Mes anterior | Compara vs mes natural anterior |
| `"vs LY"` | Año anterior | Compara vs mismo período del año pasado |
| `"vs Target"` | Target establecido | Compara vs objetivos definidos |
| `"vs Sel. Period"` | Período personalizado | Requiere `comparison_start_date` y `comparison_end_date` |
| `None` | Sin comparación | Modo single, solo analiza período actual |

**Interacción con `study_mode`:**
```python
# IMPORTANTE: Si study_mode="single", causal_filter se fuerza a None
if study_mode == "single":
    causal_filter = None  # Automático
```

**Ejemplos:**
```python
# Análisis comparativo vs semana anterior
causal_filter = "vs L7d"
study_mode = "comparative"

# Análisis comparativo vs mismo período año pasado
causal_filter = "vs LY"
study_mode = "comparative"

# Análisis sin comparación (solo período actual)
causal_filter = None
study_mode = "single"
```

---

### 9. `comparison_start_date` (Optional[datetime]) - OPCIONAL

**Propósito:** Fecha de inicio del período de comparación personalizado.

**Requerido cuando:** `causal_filter = "vs Sel. Period"`

**Formato:** Objeto `datetime` de Python

**Ejemplo:**
```python
# Comparar período actual vs diciembre 2024
causal_filter = "vs Sel. Period"
comparison_start_date = datetime(2024, 12, 1)
comparison_end_date = datetime(2024, 12, 31)
```

---

### 10. `comparison_end_date` (Optional[datetime]) - OPCIONAL

**Propósito:** Fecha de fin del período de comparación personalizado.

**Requerido cuando:** `causal_filter = "vs Sel. Period"`

**Formato:** Objeto `datetime` de Python

**Validación:**
- `comparison_end_date` debe ser >= `comparison_start_date`
- Ambas fechas deben proporcionarse juntas

---

### 11. `date_flight_local` (Optional[str]) - OPCIONAL

**Propósito:** String original de la fecha pasada por CLI. Solo para logging y metadata.

**Uso:** Informativo, no afecta la lógica de análisis.

**Ejemplo:**
```python
# Si el usuario pasó --date-flight-local 2025-01-20
date_flight_local = "2025-01-20"
```

---

### 12. `study_mode` (str) - OPCIONAL (default: "comparative")

**Propósito:** Define el modo de estudio que determina el tipo de análisis.

**Valores válidos:**

| Modo | Descripción | Efectos |
|------|-------------|---------|
| `"comparative"` | Análisis comparativo | Usa `causal_filter`, compara vs período de referencia |
| `"single"` | Análisis individual | Fuerza `causal_filter=None`, usa media como baseline |

**Comportamiento interno:**
```python
if study_mode == "single":
    causal_filter = None  # Se fuerza automáticamente
    # Baseline = media de últimos baseline_periods
    # No hay comparación causal
```

**Casos de uso:**
```python
# Análisis semanal comparativo (vs semana anterior)
study_mode = "comparative"
causal_filter = "vs L7d"
aggregation_days = 7

# Análisis diario individual (detectar anomalías vs tendencia)
study_mode = "single"
aggregation_days = 1
# causal_filter se ignora
```

---

### 13. `environment` (str) - OPCIONAL (default: "prod")

**Propósito:** Define el entorno de ejecución para configuración de credenciales.

**Valores válidos:**

| Valor | Comportamiento |
|-------|----------------|
| `"prod"` | Lee credenciales de variables de entorno del sistema |
| `"local"` | Lee credenciales del archivo `.env` en el directorio raíz |

**Credenciales afectadas:**
- AWS (para S3, Bedrock)
- Power BI (token de acceso)
- OpenAI (si se usa como LLM alternativo)

**Ejemplo:**
```python
# Ejecución en producción (CI/CD, contenedor)
environment = "prod"

# Desarrollo local
environment = "local"
```

---

## Configuraciones Típicas

### Análisis Semanal Comparativo (Weekly)

```python
await execute_analysis_flow(
    analysis_date=datetime(2025, 1, 20),
    date_parameter='flight_local',
    segment='Global',
    anomaly_detection_mode='vslast_dynamic',
    baseline_periods=7,
    aggregation_days=7,          # Semanal
    periods=1,                   # Solo última semana
    causal_filter='vs L7d',      # Comparar vs semana anterior
    study_mode='comparative',
    environment='prod'
)
```

### Análisis Diario Individual (Daily Single)

```python
await execute_analysis_flow(
    analysis_date=datetime(2025, 1, 20),
    date_parameter='flight_local',
    segment='Global',
    anomaly_detection_mode='mean',
    baseline_periods=7,          # Media de últimos 7 días
    aggregation_days=1,          # Diario
    periods=7,                   # Últimos 7 días
    causal_filter=None,          # Sin comparación
    study_mode='single',
    environment='prod'
)
```

### Análisis de Segmento Específico

```python
await execute_analysis_flow(
    analysis_date=datetime(2025, 1, 20),
    date_parameter='flight_local',
    segment='Global/SH/Economy',  # Solo Short Haul Economy
    anomaly_detection_mode='vslast',
    baseline_periods=7,
    aggregation_days=7,
    periods=4,                    # Últimas 4 semanas
    causal_filter='vs L7d',
    study_mode='comparative',
    environment='prod'
)
```

### Análisis vs Período Personalizado

```python
await execute_analysis_flow(
    analysis_date=datetime(2025, 1, 20),
    date_parameter='flight_local',
    segment='Global',
    anomaly_detection_mode='vslast_dynamic',
    baseline_periods=7,
    aggregation_days=7,
    periods=1,
    causal_filter='vs Sel. Period',
    comparison_start_date=datetime(2024, 12, 1),
    comparison_end_date=datetime(2024, 12, 31),
    study_mode='comparative',
    environment='prod'
)
```

### Análisis vs Target Mensual

```python
await execute_analysis_flow(
    analysis_date=datetime(2025, 1, 20),
    date_parameter='flight_local',
    segment='Global',
    anomaly_detection_mode='target',  # Usa targets mensuales
    baseline_periods=7,
    aggregation_days=7,
    periods=4,
    causal_filter='vs Target',
    study_mode='comparative',
    environment='prod'
)
```

---

## Valor de Retorno

**Tipo:** `List[Dict] | None`

**Estructura cuando hay anomalías:**
```python
[
    {
        'period': 1,
        'date_range': 'YYYY-MM-DD to YYYY-MM-DD',
        'ai_interpretation': '... interpretación del agente AI ...',
        'anomalies': ['Global/SH/Economy', 'Global/LH/Business']
    },
    # ... más períodos si periods > 1
]
```

**Retorna `None` cuando:**
- Falla la descarga de datos de PBI
- No se detectan anomalías en ningún período

---

## Flujo Interno de Ejecución

```
execute_analysis_flow()
│
├─► 1. DESCARGA DE DATOS
│   │   run_flexible_data_download_silent_with_date()
│   │   └─► PBIDataCollector.collect_flexible_data_for_node()
│   │       - Descarga CSVs de NPS por nodo
│   │       - Guarda en: tables/{date_parameter}_{study_mode}_{aggregation_days}d/
│   │
├─► 2. DETECCIÓN DE ANOMALÍAS
│   │   run_flexible_analysis_silent()
│   │   └─► FlexibleAnomalyDetector.analyze_period()
│   │       - Calcula baseline según anomaly_detection_mode
│   │       - Detecta anomalías (+/-/N) por nodo
│   │       - Retorna: anomaly_periods, deviations, nps_values
│   │
└─► 3. INTERPRETACIÓN AI
    │   show_all_anomaly_periods_with_explanations()
    │   └─► Para cada período con anomalías (PARALELO, max 3):
    │       ├─► FlexibleAnomalyInterpreter.explain_anomaly()
    │       │   - Recolecta datos operativos
    │       │   - Recolecta verbatims de clientes
    │       │   - Analiza rutas afectadas
    │       │
    │       └─► AnomalyInterpreterAgent.interpret_anomaly_tree()
    │           - Genera interpretación narrativa con LLM
    │           - Identifica causas raíz
    │           - Produce síntesis ejecutiva
```

---

## Consideraciones de Rendimiento

| Factor | Impacto | Recomendación |
|--------|---------|---------------|
| `segment="Global"` | Alto (12 nodos) | Usar segmentos específicos si es posible |
| `periods` alto | Alto (más llamadas PBI/LLM) | Limitar a 1-7 períodos |
| `aggregation_days=1` | Medio (más granularidad) | Usar 7 para análisis estables |
| `environment="local"` | Bajo | Solo para desarrollo |

**Timeouts internos:**
- Explicación por nodo: 3000 segundos (50 min)
- Interpretación AI por período: 3000 segundos (50 min)

---

## Errores Comunes

| Error | Causa | Solución |
|-------|-------|----------|
| `Data collection failed` | PBI sin datos para la fecha | Verificar `analysis_date` y lag de PBI |
| `No anomalies found` | Todos los valores dentro de threshold | Normal, no es error |
| `Timeout` en explicación | Nodo muy complejo (ej: Global) | Reducir `periods` o usar segmento específico |
| `comparison dates required` | `causal_filter="vs Sel. Period"` sin fechas | Proporcionar ambas fechas |

---

## Notas para Agentes

1. **Siempre calcular `analysis_date` considerando el lag de PBI (4 días)**
2. **Usar `study_mode="single"` para análisis diarios sin comparación**
3. **Usar `study_mode="comparative"` para análisis semanales con contexto**
4. **El segmento `"Global"` es el más completo pero más lento**
5. **`causal_filter=None` es equivalente a `study_mode="single"`**
6. **Los valores de retorno son `None` si no hay anomalías - esto es normal**
