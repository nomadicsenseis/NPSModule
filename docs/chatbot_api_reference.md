# API del Chatbot de Verbatims NPS - Referencia Técnica

## Descripción

API REST asíncrona para consultas en lenguaje natural sobre verbatims de clientes NPS de Iberia. Autenticación por API Key.

---

## Endpoints

| Entorno | URL Base |
|---------|----------|
| PROD | `https://nps.chatbot.iberia.es/ibdp/api/question` |
| PRE | `https://nps.chatbot.pre.iberia.es/ibdp/api/question` |

Se puede sobreescribir con la variable de entorno `CHATBOT_API_ENDPOINT`.

---

## Autenticación

Header `X-API-Key` con el valor de la variable de entorno `CHATBOT_API_KEY`.

```http
X-API-Key: <CHATBOT_API_KEY>
Content-Type: application/json
User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0
```

> El User-Agent simula un navegador para evitar bloqueos del WAF corporativo.

---

## Flujo de Comunicación

La API es asíncrona: se envía una pregunta, se recibe un `jobId`, y se hace polling hasta obtener la respuesta.

```
POST /ibdp/api/question          →  { jobId: "abc-123" }
GET  /ibdp/api/question/abc-123  →  { answer: null }        (procesando)
GET  /ibdp/api/question/abc-123  →  { answer: "..." }       (listo)
```

---

## 1. Enviar Pregunta

**`POST /ibdp/api/question`**

### Request Body

```json
{
  "value": "¿Cuáles son los principales temas negativos?",
  "verbatimCol": "nps_all_t",
  "filters": {
    "date_flight_local": ["2025-01-01", "2025-01-31"]
  }
}
```

### Campos

| Campo | Tipo | Obligatorio | Descripción |
|-------|------|:-----------:|-------------|
| `value` | string | Sí | Pregunta en lenguaje natural |
| `verbatimCol` | string | Sí | Columna de verbatims. Usar `"nps_all_t"` |
| `filters.date_flight_local` | [string, string] | Sí | Rango de fechas `[inicio, fin]` formato `YYYY-MM-DD` |
| `filters.cabin` | string | No | Clase de cabina (ej: `"Business"`, `"Economy"`) |
| `filters.haul` | string | No | `"LH"` (Long Haul) o `"SH"` (Short Haul) |
| `filters.route` | string | No | Ruta (ej: `"MAD-JFK"`) |
| `filters.company` | string | No | `"IB"` (Iberia) o `"YW"` (Air Nostrum) |
| `filters.nps_category` | string | No | `"Detractor"`, `"Passive"`, `"Promoter"` |

### Response (200/201)

```json
{
  "jobId": "abc-123-def-456"
}
```

En casos raros puede devolver directamente un campo `answer` con la respuesta completa.

---

## 2. Obtener Respuesta (Polling)

**`GET /ibdp/api/question/{jobId}`**

### Procesando

```json
{
  "answer": null,
  "error": null
}
```

### Respuesta lista

```json
{
  "answer": "Los principales temas negativos son...",
  "toolOutput": { ... },
  "jobId": "abc-123-def-456",
  "sessionId": "session-xxx"
}
```

### Error

```json
{
  "answer": null,
  "error": "Descripción del error"
}
```

### Lógica de evaluación

- **Respuesta válida:** `answer` no es `null`, no es el string `"null"`, y no está vacío.
- **Error:** `error` no es `null`/`None`.
- **Procesando:** cualquier otro caso.

### Estrategia de polling

| Parámetro | Valor |
|-----------|-------|
| Delay inicial | 5 segundos |
| Incremento por intento | +2 segundos |
| Delay máximo entre intentos | 15 segundos |
| Timeout total | 120 segundos |

Fórmula del delay: `min(5 + intento * 2, 15)` segundos.

---

## Proxy

Necesario en el entorno PRE. Se configura de dos formas:

1. Parámetro directo al instanciar: `proxy="http://proxy.iberia.es:8080"`
2. Variables de entorno: `HTTP_PROXY` / `HTTPS_PROXY`

---

## Ejemplo Completo (Python)

```python
import requests
import time
import os

API_KEY = os.getenv("CHATBOT_API_KEY")
ENDPOINT = "https://nps.chatbot.iberia.es/ibdp/api/question"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0"
}

# 1. Enviar pregunta
payload = {
    "value": "¿Qué quejas principales tienen los detractores en Long Haul Business?",
    "verbatimCol": "nps_all_t",
    "filters": {
        "date_flight_local": ["2025-01-01", "2025-01-31"],
        "haul": "LH",
        "cabin": "Business",
        "nps_category": "Detractor"
    }
}

response = requests.post(ENDPOINT, json=payload, headers=headers, timeout=30)
data = response.json()
job_id = data.get("jobId")

# 2. Polling por respuesta
answer_url = f"{ENDPOINT}/{job_id}"
for attempt in range(1, 20):
    resp = requests.get(answer_url, headers=headers, timeout=30)
    result = resp.json()

    if result.get("error"):
        raise Exception(f"API error: {result['error']}")

    answer = result.get("answer")
    if answer and answer != "null" and str(answer).strip():
        print(answer)
        break

    time.sleep(min(5 + attempt * 2, 15))
```

---

## Errores Comunes

| Síntoma | Causa Probable | Solución |
|---------|----------------|----------|
| `CHATBOT_API_KEY is missing` | Variable de entorno no configurada | Configurar `CHATBOT_API_KEY` |
| Status 403 | API Key inválida o WAF bloqueando | Verificar key y User-Agent |
| Respuesta HTML en vez de JSON | WAF interceptando la request | Verificar User-Agent header |
| `ConnectionError` | Proxy no configurado (PRE) o API caída | Configurar proxy o verificar disponibilidad |
| Timeout de polling (120s) | Pregunta compleja o API sobrecargada | Aumentar `max_wait_time` o simplificar la pregunta |
