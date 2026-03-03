# Documento de Requisitos: Corrección de Fechas en verbatims_tool

## Introducción

El agente `CausalExplanationAgent` realiza análisis causales de anomalías NPS para períodos históricos específicos (por ejemplo, "diciembre 2024 vs noviembre 2024"). Sin embargo, cuando se invoca la herramienta `verbatims_tool`, las fechas que llegan a la API del chatbot son incorrectas: en lugar de consultar el período histórico de análisis, se consultan fechas recientes (por ejemplo, febrero 2026). Este fallo provoca que el análisis de verbatims no corresponda al período investigado, comprometiendo la calidad del diagnóstico causal.

Esta especificación define los requisitos para corregir el flujo de fechas desde el contexto de análisis del agente hasta la llamada a la API del chatbot, añadir validaciones y mejorar el logging para detectar y prevenir este tipo de desvíos.

## Glosario

- **Agent**: `CausalExplanationAgent` — agente principal que orquesta el análisis causal.
- **verbatims_tool**: Herramienta del agente que consulta comentarios de clientes para un período dado.
- **ChatbotCollector**: `ChatbotVerbatimsCollector` — clase que realiza las llamadas HTTP a la API del chatbot.
- **Período de análisis (target)**: Rango de fechas principal que se está investigando (e.g., diciembre 2024).
- **Período de comparación**: Rango de fechas contra el que se compara el período de análisis (e.g., noviembre 2024).
- **causal_filter**: Parámetro del agente que indica el tipo de comparación ("vs Sel. Period", "vs L7d", "vs LM", etc.).
- **comparison_start_date / comparison_end_date**: Fechas de comparación pre-configuradas en el agente cuando `causal_filter` es "vs Sel. Period".
- **calculate_dynamic_comparison_dates**: Método del agente que calcula las fechas de comparación a partir del `causal_filter`.
- **DateValidationError**: Excepción personalizada que se lanzará cuando las fechas no superen la validación.

---

## Requisitos

### Requisito 1: Propagación correcta de fechas desde el agente hasta el chatbot

**User Story:** Como analista de NPS, quiero que cuando el agente investigue un período histórico, la API del chatbot reciba exactamente ese período, para que los verbatims analizados correspondan al momento de la anomalía.

#### Criterios de Aceptación

1. WHEN `verbatims_tool` es invocado con `start_date` y `end_date`, THE Agent SHALL pasar esas mismas fechas como `target_start` y `target_end` a `_analyze_verbatims_comparative_chatbot` sin modificarlas.
2. WHEN `_analyze_verbatims_comparative_chatbot` llama a `ChatbotCollector.ask_chatbot_question`, THE ChatbotCollector SHALL incluir en el payload de la API exactamente las fechas `start_date` y `end_date` recibidas como parámetros.
3. THE Agent SHALL calcular las fechas de comparación únicamente a partir de las fechas del período de análisis recibidas por `verbatims_tool`, no a partir de la fecha actual del sistema.
4. WHEN `causal_filter` es "vs Sel. Period" y `comparison_start_date`/`comparison_end_date` están configurados en el agente, THE Agent SHALL usar esos valores pre-configurados como fechas de comparación sin recalcularlos.
5. WHEN `causal_filter` NO es "vs Sel. Period" (e.g., "vs L7d"), THE Agent SHALL calcular las fechas de comparación aplicando el desplazamiento correspondiente sobre las fechas del período de análisis recibidas por `verbatims_tool`.

---

### Requisito 2: Validación de fechas antes de llamar a la API del chatbot

**User Story:** Como desarrollador, quiero que el sistema valide las fechas antes de enviarlas al chatbot, para detectar inmediatamente cualquier desvío entre las fechas de análisis y las fechas que se enviarían a la API.

#### Criterios de Aceptación

1. WHEN `ask_chatbot_question` recibe `start_date` y `end_date`, THE ChatbotCollector SHALL verificar que ambas fechas son anteriores a la fecha actual del sistema.
2. IF `start_date` o `end_date` son posteriores a la fecha actual del sistema, THEN THE ChatbotCollector SHALL lanzar una `DateValidationError` con un mensaje que indique las fechas recibidas y la fecha actual.
3. WHEN `ask_chatbot_question` recibe `start_date` y `end_date`, THE ChatbotCollector SHALL verificar que `start_date` es anterior o igual a `end_date`.
4. IF `start_date` es posterior a `end_date`, THEN THE ChatbotCollector SHALL lanzar una `DateValidationError` con un mensaje descriptivo.
5. WHEN `_verbatims_tool` calcula las fechas de comparación, THE Agent SHALL verificar que las fechas de comparación resultantes son anteriores a las fechas del período de análisis (o iguales en el caso de "vs Sel. Period").

---

### Requisito 3: Logging detallado del flujo de fechas

**User Story:** Como desarrollador, quiero que el sistema registre con claridad qué fechas se están usando en cada paso del flujo, para poder diagnosticar rápidamente cualquier desvío de fechas en producción.

#### Criterios de Aceptación

1. WHEN `verbatims_tool` es invocado, THE Agent SHALL registrar en el log el valor exacto de `start_date` y `end_date` recibidos como parámetros.
2. WHEN `calculate_dynamic_comparison_dates` es llamado, THE Agent SHALL registrar en el log las fechas de entrada, el `causal_filter` aplicado y las fechas de comparación resultantes.
3. WHEN `ask_chatbot_question` es llamado, THE ChatbotCollector SHALL registrar en el log las fechas `start_date` y `end_date` que se incluirán en el payload de la API.
4. WHEN la validación de fechas detecta un problema, THE ChatbotCollector SHALL registrar en el log un mensaje de nivel ERROR que incluya las fechas recibidas, la fecha actual del sistema y la descripción del problema.
5. THE Agent SHALL registrar en el log un resumen del flujo de fechas completo (fechas de análisis → fechas de comparación → fechas enviadas al chatbot) antes de cada llamada al chatbot.

---

### Requisito 4: Protección contra recálculo de fechas cuando están pre-configuradas

**User Story:** Como analista, quiero que cuando el agente tenga fechas de comparación explícitamente configuradas, éstas no sean sobreescritas por ningún cálculo dinámico, para garantizar que el análisis compare exactamente los períodos que yo especifiqué.

#### Criterios de Aceptación

1. WHEN el agente es inicializado con `comparison_start_date` y `comparison_end_date` no nulos y `causal_filter` es "vs Sel. Period", THE Agent SHALL preservar esos valores durante toda la ejecución del análisis sin modificarlos.
2. WHEN `calculate_dynamic_comparison_dates` es llamado con `causal_filter` igual a "vs Sel. Period", THE Agent SHALL retornar `self.comparison_start_date` y `self.comparison_end_date` sin ejecutar ningún cálculo adicional.
3. IF `causal_filter` es "vs Sel. Period" pero `comparison_start_date` o `comparison_end_date` son None, THEN THE Agent SHALL registrar un WARNING en el log indicando que faltan las fechas de comparación para el modo "vs Sel. Period".
4. WHILE el agente ejecuta múltiples llamadas a `verbatims_tool` en una misma sesión de análisis, THE Agent SHALL usar las mismas fechas de comparación en todas las llamadas, sin recalcularlas entre llamadas.

---

### Requisito 5: Trazabilidad de fechas en los resultados del análisis

**User Story:** Como analista, quiero que el resultado del análisis de verbatims incluya las fechas exactas que se consultaron, para poder verificar que el análisis corresponde al período correcto.

#### Criterios de Aceptación

1. WHEN `_analyze_verbatims_comparative_chatbot` genera el resultado textual, THE Agent SHALL incluir en el encabezado del resultado las fechas exactas del período de análisis y del período de comparación que se enviaron al chatbot.
2. WHEN `_analyze_verbatims_comparative_chatbot` almacena los datos en `self.collected_data['verbatims_conversation']`, THE Agent SHALL incluir los campos `target_start`, `target_end`, `comparison_start` y `comparison_end` con los valores exactos enviados a la API.
3. THE Agent SHALL incluir en el resultado de `verbatims_tool` una línea de auditoría que indique la fuente de las fechas (pre-configuradas o calculadas dinámicamente) y el `causal_filter` utilizado.
