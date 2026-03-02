# Especificación Técnica: Conexión a la API del Chatbot de Verbatims

## 1. Descripción General

El módulo `ChatbotVerbatimsCollector` (`dashboard_analyzer/data_collection/chatbot_verbatims_collector.py`) es el componente encargado de conectarse a la API del chatbot NPS de Iberia para realizar consultas sobre verbatims de clientes. Funciona como un cliente HTTP que envía preguntas en lenguaje natural y recibe respuestas procesadas por el chatbot.

El collector opera con un mecanismo de fallback: si la API del chatbot no está disponible, delega la obtención de datos crudos al `PBIDataCollector` (Power BI).

---

## 2. Evolución de la Autenticación

El sistema ha pasado por dos modelos de autenticación:

| Versión | Autenticación | Estado |
|---------|---------------|--------|
| **v1 (backup)** | JWT Token (`Authorization: Bearer <token>`) | Deprecada |
| **v2 (actual)** | API Key (`X-API-Key: <key>`) | En uso |

La versión anterior (`.py.backup`) requería un JWT token que se cargaba desde `temp_aws_credentials.env` o se recibía del frontend, con validación de expiración vía `pyjwt`. La versión actual simplifica esto usando una API Key estática configurada por variable de entorno.

---

## 3. Configuración y Variables de Entorno

### Variables requeridas

| Variable | Descripción | Obligatoria |
|----------|-------------|-------------|
| `CHATBOT_API_KEY` | API Key para autenticación con el chatbot | Sí |
| `CHATBOT_API_ENDPOINT` | URL del endpoint (override del default) | No |
| `HTTP_PROXY` / `HTTPS_PROXY` | Proxy corporativo (necesario en PRE) | Según entorno |

### Endpoints por entorno

| Entorno | URL Base |
|---------|----------|
| **PROD** | `https://nps.chatbot.iberia.es/ibdp/api/question` |
| **PRE** | `https://nps.chatbot.pre.iberia.es/ibdp/api/question` |

> Nota: El entorno `local` se trata internamente como `prod` para la resolución de endpoints. Solo afecta la carga de credenciales (`.env` vs variables de sistema).

---

## 4. Autenticación y Headers HTTP

Cada request a la API incluye los siguientes headers:

```http
X-API-Key: <valor de CHATBOT_API_KEY>
User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0
Content-Type: application/json
```

**Notas importantes:**
- El header de autenticación es `X-API-Key`, no `ApiKey`. Esto fue confirmado en pruebas.
- El `User-Agent` simula un navegador Firefox para evitar bloqueos del WAF (Web Application Firewall).

---

## 5. Flujo de Comunicación con la API

### 5.1 Diagrama de Secuencia

```
Cliente                          API Chatbot
  │                                  │
  │─── POST /ibdp/api/question ─────►│  (envía pregunta + filtros)
  │                                  │
  │◄── 200/201 { jobId: "xxx" } ────│  (respuesta asíncrona)
  │                                  │
  │─── GET /ibdp/api/question/{id} ─►│  (polling por respuesta)
  │◄── 200 { answer: null } ────────│  (aún procesando)
  │                                  │
  │─── GET /ibdp/api/question/{id} ─►│  (retry con backoff)
  │◄── 200 { answer: "..." } ───────│  (respuesta lista)
  │                                  │
```

### 5.2 Paso 1: Envío de Pregunta (POST)

**Endpoint:** `POST {base_url}/ibdp/api/question`

**Payload de ejemplo:**

```json
{
  "value": "¿Cuáles son los principales temas negativos en los verbatims?",
  "verbatimCol": "nps_all_t",
  "filters": {
    "date_flight_local": ["2025-01-01", "2025-01-31"],
    "cabin": "Business",
    "haul": "LH",
    "route": "MAD-JFK",
    "company": "IB",
    "nps_category": "Detractor"
  }
}
```

**Campos del payload:**

| Campo | Tipo | Obligatorio | Descripción |
|-------|------|-------------|-------------|
| `value` | string | Sí | Pregunta en lenguaje natural |
| `verbatimCol` | string | Sí | Columna de verbatims a consultar. Valor actual: `"nps_all_t"` |
| `filters.date_flight_local` | array[string] | Sí | Rango de fechas `[start, end]` en formato `YYYY-MM-DD` |
| `filters.cabin` | string | No | Clase de cabina (ej: `"Business"`, `"Economy"`) |
| `filters.haul` | string | No | Tipo de vuelo: `"LH"` (Long Haul) o `"SH"` (Short Haul) |
| `filters.route` | string | No | Ruta del vuelo (ej: `"MAD-JFK"`) |
| `filters.company` | string | No | Compañía: `"IB"` (Iberia) o `"YW"` (Air Nostrum) |
| `filters.nps_category` | string | No | Categoría NPS (ej: `"Detractor"`, `"Promoter"`) |

> Nota: En la versión anterior (backup), `verbatimCol` era `"iag_mod_501_t_scrubbed"`. La versión actual usa `"nps_all_t"`.

**Respuesta esperada (200/201):**

```json
{
  "jobId": "abc-123-def-456"
}
```

En casos raros, la API puede devolver una respuesta inmediata con el campo `answer` ya poblado.

### 5.3 Paso 2: Polling por Respuesta (GET)

**Endpoint:** `GET {base_url}/ibdp/api/question/{jobId}`

**Estrategia de polling:**
- Delay inicial: 5 segundos
- Incremento: +2 segundos por intento
- Delay máximo entre intentos: 15 segundos
- Timeout total: 120 segundos (configurable vía `max_wait_time`)

**Respuesta en proceso:**

```json
{
  "answer": null,
  "error": null
}
```

**Respuesta exitosa:**

```json
{
  "answer": "Los principales temas negativos son...",
  "toolOutput": { ... },
  "jobId": "abc-123-def-456",
  "sessionId": "session-xxx"
}
```

**Respuesta con error:**

```json
{
  "answer": null,
  "error": "Descripción del error"
}
```

**Lógica de evaluación de respuesta:**
- Se considera "respuesta válida" si `answer` no es `null`, no es el string `"null"`, no es `None`, y no está vacío tras `strip()`.
- Se considera "error" si `error` no es `null`/`None`.
- Cualquier otro caso se interpreta como "aún procesando".

---

## 6. Configuración de Proxy

El collector soporta proxy de tres formas (en orden de prioridad):

1. **Parámetro directo:** `proxy="http://proxy.iberia.es:8080"` en el constructor
2. **Variables de entorno:** `HTTP_PROXY` / `HTTPS_PROXY` (o sus variantes en minúsculas)
3. **Sin proxy:** Si no se configura ninguno, las requests van directas

El entorno PRE requiere el proxy corporativo de Iberia para conectarse.

---

## 7. Test de Conexión

El método `test_connection()` verifica la conectividad enviando una pregunta de prueba:

```json
{
  "value": "¿Qué es NPS?",
  "filters": {
    "date_flight_local": ["2024-01-01", "2024-01-31"]
  }
}
```

**Resultados posibles:**

| Resultado | Condición |
|-----------|-----------|
| `(True, "✅ Connected")` | Status 200/201 y respuesta con `jobId` o `answer` |
| `(True, "✅ PBI fallback")` | ConnectionError pero hay `pbi_collector` disponible |
| `(False, "❌ ...")` | API Key faltante, timeout, error HTTP, o sin fallback |

---

## 8. Mecanismo de Fallback

El collector implementa un fallback a Power BI para la obtención de datos crudos de verbatims:

```
ask_chatbot_question()  →  API Chatbot (Q&A inteligente)
get_verbatims_data()    →  PBI Collector (datos crudos, siempre)
test_connection()       →  API Chatbot, fallback a PBI si falla conexión
```

El método `get_verbatims_data()` actualmente siempre delega al `PBIDataCollector` para datos en bulk. El chatbot se usa exclusivamente para consultas Q&A inteligentes a través de `ask_chatbot_question()`.

---

## 9. Integración con el Sistema

El `ChatbotVerbatimsCollector` es instanciado por el `CausalExplanationAgent`:

```python
# En causal_explanation_agent.py
def _init_chatbot_collector(self):
    return ChatbotVerbatimsCollector(
        pbi_collector=self.pbi_collector,
        environment=self.environment  # "prod" o "local"
    )
```

El agente causal lo utiliza para hacer preguntas contextuales sobre verbatims durante el análisis de anomalías NPS, como parte del flujo de explicación causal.

---

## 10. Diferencias entre API v1 (Backup) y v2 (Actual)

| Aspecto | v1 (Backup) | v2 (Actual) |
|---------|-------------|-------------|
| Autenticación | JWT Bearer Token | X-API-Key header |
| Endpoint base | `b8fktdca38.execute-api.eu-west-1.amazonaws.com/api/transformation/api/nps-chatbot/question` | `nps.chatbot.iberia.es/ibdp/api/question` |
| `verbatimCol` | `iag_mod_501_t_scrubbed` | `nps_all_t` |
| Gestión de token | Validación JWT, expiración, carga desde archivo | API Key estática desde env var |
| Proxy | No configurado | Soporte completo (param, env vars) |
| User-Agent | `CausalExplanationAgent/1.0` | Firefox spoofing (anti-WAF) |
| Polling delay | 10s inicial, +5s/intento, max 30s | 5s inicial, +2s/intento, max 15s |
| Timeout polling | 300s (v1) / 120s (v1 working) | 120s |
| Filtros de cabina | `cabin_in_surveyed_flight` | `cabin` |

---

## 11. Manejo de Errores

| Error | Causa Probable | Comportamiento |
|-------|----------------|----------------|
| `ConnectionError` | Red/proxy mal configurado, API caída | Retorna `None`, log de error |
| `Timeout` (30s por request) | API lenta o sobrecargada | Retorna `None`, log de error |
| Status != 200/201 en POST | API Key inválida, payload incorrecto | Retorna `None`, log con status y body |
| JSON decode error | Respuesta no-JSON (HTML de WAF, etc.) | Retorna `False` en test, log del contenido |
| Timeout de polling (120s) | Pregunta compleja, API sobrecargada | Retorna `None` tras agotar intentos |

---

## 12. Consideraciones de Seguridad

- La `CHATBOT_API_KEY` no está presente en el archivo `.env` del repositorio actualmente. Debe configurarse como variable de entorno del sistema o del contenedor.
- El User-Agent spoofing es necesario para evitar bloqueos del WAF corporativo.
- Los logs de debug imprimen headers completos (incluyendo la API Key) tanto a stdout como al logger. Esto debería revisarse para entornos productivos.
