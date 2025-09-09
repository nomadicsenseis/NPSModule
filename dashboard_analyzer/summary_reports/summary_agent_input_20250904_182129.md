===== SYSTEM =====

Eres un experto analista ejecutivo especializado en completar análisis de NPS comprehensivos.

⚠️ **CRÍTICO - NO INVENTES DATOS:**
Si hay algún dato que te falta, NO lo supongas ni inventes. En su lugar, indica claramente que ese dato específico no está disponible. Por ejemplo: "El análisis diario para Economy LH no está disponible" o "Los datos de rutas para el día 25 no están incluidos en el análisis". 

⚠️ **CRÍTICO - NO USES %:** Para las subidas o bajadas de cualquier variable, menciona el valor exacto de la variación, NUNCA el %.

⚠️ **CRÍTICO - NO CALCULES COSAS QUE NO SE TE PIDEN:**

⚠️ **IMPORTANTE - SI HAY DATOS DIARIOS, ÚSALOS:**
Si se te proporciona análisis diario en la sección "ANÁLISIS DIARIO SINGLE", DEBES usarlo e integrarlo en el resumen. NO digas que "no está disponible" si los datos están presentes en el input.

⚠️ **FORMATO DE NÚMEROS - UN DECIMAL:**
Todos los números, porcentajes, métricas y valores NPS deben mostrarse con exactamente UN decimal. Por ejemplo: 19.8 (no 19.75), -4.4 (no -4.39), 93.5% (no 93.53%), etc.

TU FUNCIÓN:
- Tomar la síntesis ejecutiva del interpreter semanal TAL COMO ESTÁ. ES VITAL QUE NO CAMBIES NI UNA LETRA.
- INTEGRAR el detalle diario del interpreter DENTRO de cada sección correspondiente de la síntesis semanal
- Identificar días especialmente reseñables en el detalle diario
- Crear un resumen fluido y ejecutivo

IMPORTANTE:
- NO incluyas recomendaciones adicionales
- NO uses títulos como "Integración del análisis diario" o similares
- El análisis diario debe fluir naturalmente como un párrafo adicional
- Identifica días especialmente reseñables en el detalle diario
- Haz el texto fluido y ejecutivo, no técnico
- Solo incluye días que tengan análisis relevantes (con caídas/subidas o datos significativos)
- Para cabinas/radio sin incidencias: comenta solo su NPS del período y el del período de comparación
- NO hables de "anomalias". Habla de "caídas" o "subidas" de NPS.
- SI hay datos diarios en el input, ÚSALOS. NO digas que "no están disponibles" si están presentes.

ANCLAJE DE SECCIONES (case-insensitive):
- Global (los 2 primeros párrafos de la síntesis)
- Economy SH
- Business SH
- Economy LH
- Business LH
- Premium LH

INTEGRACIÓN POR SECCIÓN:
- Tras cada bloque semanal anterior, añade exactamente un párrafo narrativo con los días reseñables en orden cronológico (28-jul → 03-ago si existen).
- Incluye cuando estén disponibles: NPS actual, baseline y diferencia; métricas clave (p.ej., % mishandling, nº cambios de aeronave, reprogramaciones, reubicaciones) y rutas/destinos citados.
- Si el bloque semanal indica "sin datos", REDACTA como: "Se mantiene estable a nivel semanal; pueden existir oscilaciones diarias que se detallan a continuación" y añade igualmente el párrafo diario si hay datos.
- No modifiques, no reordenes ni resumas el texto semanal. No alteres sus cifras ni redondeos.

ESTILO Y LÉXICO:
- Estilo ejecutivo, fluido y conciso. No técnico.
- Usa "subidas/bajadas", "mejoras/deterioros". Evita "anomalía/s".
- Máximo 1-2 frases por día; prioriza 28, 29, 30, 31 de julio; 1, 2, 3 de agosto.
- No inventes cifras. Si no hay NPS exacto en el diario, describe el evento y su dirección (subida/bajada) sin números.
- Para las subidas o bajadas de cualquier variable, menciona el valor exacto de la variación, NUNCA el %.


===== USER =====

Completa el análisis comprehensivo:

**ANÁLISIS SEMANAL COMPARATIVO:**
[{'period': 1, 'date_range': '2025-08-23 to 2025-08-29', 'ai_interpretation': '📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**\n\n**Nodos analizados:** 4 (Global, Global/LH/Premium, Global/SH/Business, Global/SH/Business/IB)\n\n---\n\n## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA\n\n# 🏢 DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)\n\n## 📊 ANÁLISIS SISTEMÁTICO\n\n### **A. Economy Short Haul:**\n**Estado de nodos**: No existen nodos separados Economy SH/IB ni Economy SH/YW en los datos proporcionados.\n\n**Razón**: Los datos analizados no incluyen segmentos de Economy Short Haul, por lo que no es posible realizar la comparación IB vs YW para esta cabina.\n\n### **B. Business Short Haul:**\n**Estado de nodos**: \n- ✅ **Existe**: Global/SH/Business/IB (+14.15 puntos NPS)\n- ❌ **No existe**: Global/SH/Business/YW\n\n**Razón del nodo faltante**: YW (Air Europa) no opera significativamente en el segmento Business Short Haul o no generó suficiente volumen de encuestas para crear un nodo separado durante el período analizado.\n\n## 🎯 DIAGNÓSTICO FINAL\n\n### **Business Short Haul:**\n**La causa es específica de la operación IB**, ya que solo existe evidencia para Iberia en este segmento. El incremento de +14.15 puntos NPS está impulsado por mejoras operativas en puntualidad (+9.434 SHAP) contrarrestadas por deterioro en boarding (-3.952 SHAP), patrón que no puede ser validado como común a la cabina por ausencia de datos comparativos de YW.\n\n**Conclusión**: Sin nodo YW para comparar, no es posible determinar si las causas son específicas de compañía o generales a la cabina Business Short Haul.\n\n---\n\n## 💺 DIAGNÓSTICO A NIVEL DE CABINA\n\n# ✈️ DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)\n\n## 📊 ANÁLISIS SISTEMÁTICO\n\n### **A. Short Haul - Análisis de Cabinas:**\n\n**Estado de nodos disponibles:**\n- ❌ **Economy SH**: No existe nodo\n- ✅ **Business SH**: Existe (+11.46 puntos NPS)\n- ❌ **Premium SH**: No existe nodo\n\n**Análisis de convergencia/divergencia:**\nNo es posible realizar comparación entre cabinas Short Haul ya que solo existe evidencia para Business SH. El segmento Business SH/IB muestra mejoras operativas en puntualidad (+10.279 SHAP) contrarrestadas por problemas de boarding (-2.104 SHAP).\n\n### **B. Long Haul - Análisis de Cabinas:**\n\n**Estado de nodos disponibles:**\n- ❌ **Economy LH**: No existe nodo\n- ❌ **Business LH**: No existe nodo  \n- ✅ **Premium LH**: Existe (-1.48 puntos NPS)\n\n**Análisis de convergencia/divergencia:**\nNo es posible realizar comparación entre cabinas Long Haul ya que solo existe evidencia para Premium LH. El segmento Premium LH muestra deterioro operativo multifactorial con problemas en arrivals experience (-0.903 SHAP) y cabin crew (-0.842 SHAP), parcialmente compensado por mejoras en aircraft interior (+2.720 SHAP).\n\n## 🎯 DIAGNÓSTICO FINAL\n\n### **Short Haul:**\n**Análisis inconcluyente por falta de datos comparativos**. Solo Business SH presenta evidencia, impidiendo determinar si las causas operativas (mejora en puntualidad vs deterioro en boarding) son comunes al radio o específicas de esta cabina.\n\n### **Long Haul:**\n**Análisis inconcluyente por falta de datos comparativos**. Solo Premium LH presenta evidencia, impidiendo determinar si las causas operativas (deterioro en arrivals/crew vs mejora en aircraft interior) son comunes al radio o específicas de esta cabina.\n\n**Limitación crítica**: La ausencia de múltiples cabinas por radio impide el diagnóstico de patrones convergentes o divergentes entre clases de servicio.\n\n---\n\n## 🌎 DIAGNÓSTICO GLOBAL POR RADIO\n\n# 🌍 DIAGNÓSTICO A NIVEL RADIO Y GLOBAL\n\n## 📊 ANÁLISIS SISTEMÁTICO\n\n### **A. Comparación Entre Radios:**\n\n**Estado de afectación por radio:**\n- ✅ **Short Haul**: Afectado (+11.46 puntos NPS en Business SH)\n- ✅ **Long Haul**: Afectado (-1.48 puntos NPS en Premium LH)\n- ✅ **Ambos radios están afectados** con **patrones divergentes**\n\n**Convergencia de drivers principales:**\n- **Short Haul**: Driver dominante **Punctuality** (+10.279 SHAP) con mejora operativa\n- **Long Haul**: Driver dominante **Aircraft Interior** (+2.720 SHAP) con deterioro en arrivals (-0.903 SHAP)\n- **❌ NO convergen**: Drivers principales completamente diferentes entre radios\n\n**Consistencia de evidencia operativa:**\n- **Short Haul**: Reducción retrasos (-20, -18.2%) y cancelaciones (-8, -25.0%)\n- **Long Haul**: Aumento otras incidencias (+13, +39.4%) y problemas de equipajes\n- **❌ Evidencia operativa divergente**: SH mejora, LH deteriora\n\n### **B. Coherencia con Nodo Global:**\n\n**Impacto agregado del Global (+3.85 puntos NPS):**\n- ✅ **Coherente**: Refleja el balance entre mejora SH (+11.46) y deterioro LH (-1.48)\n- ✅ **Driver Global dominante**: Punctuality (+3.548 SHAP) coincide con patrón SH\n- ✅ **Evidencia operativa Global**: Reducción total incidentes (-286, -19.7%) refleja mejora neta\n\n**Patrón de agregación:**\nEl Global **amplifica selectivamente** el patrón Short Haul debido a mayor volumen/peso estadístico, mientras que el deterioro Long Haul queda diluido en el agregado.\n\n## 🎯 DIAGNÓSTICO FINAL\n\n**Las causas son MIXTAS/COMPENSATORIAS con predominio Short Haul**. Cada radio presenta patrones causales completamente divergentes (SH mejora operativa vs LH deterioro multifactorial) que se combinan en el Global, donde la mejora Short Haul domina el resultado agregado debido a su mayor peso estadístico, resultando en una mejora neta de +3.85 puntos NPS que refleja principalmente las mejoras operativas de puntualidad del segmento Short Haul.\n\n---\n\n## 📋 ANÁLISIS DE CAUSAS DETALLADO\n\n# 🔍 ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS\n\n## 🎯 CAUSA 1: MEJORA OPERATIVA MASIVA EN PUNTUALIDAD\n\n### **A. NATURALEZA DE LA CAUSA:**\n**Hipótesis**: Implementación exitosa de estrategias operativas que redujeron significativamente los incidentes de retrasos y cancelaciones, mejorando la percepción de puntualidad y confiabilidad del servicio. Esta mejora operativa tangible se tradujo directamente en mayor satisfacción del cliente.\n\n### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**\n\n**Segmento más grande afectado**: **Global** (nivel compañía completo)\n\n**Output causal detallado**:\n- **NPS Impact**: +3.85 puntos (25.34 vs 21.50 baseline)\n- **Driver SHAP**: Punctuality +3.548 (altamente significativo)\n- **Satisfacción diferencial**: +3.27 puntos en Punctuality\n\n**Alcance de afectación**: Esta causa afecta a **todos los subsegmentos** bajo el nodo Global, incluyendo Short Haul Business/IB donde se amplifica (+10.279 SHAP) y constituye el driver dominante.\n\n**Métricas clave operativas**:\n- **Retrasos**: -219 incidentes (-23.9% vs período previo)\n- **Cancelaciones**: -64 incidentes (-35.0% vs período previo)\n- **Total incidentes operativos**: 1,455 → 1,169 (-286 incidentes, -19.7%)\n- **Conexiones perdidas**: De 99 a 0 pérdidas definitivas (SH Business)\n\n**Rutas más impactadas**:\n- MAD-VGO: +105.1 NPS (14 pax)\n- PMI-RJL: +83.3 NPS (6 pax)\n- JTR-MAD: +68.9 NPS (15 pax)\n- MAD-SFO: +51.8 NPS (13 pax)\n\n---\n\n## 🎯 CAUSA 2: DETERIORO EN EXPERIENCIA DE BOARDING\n\n### **A. NATURALEZA DE LA CAUSA:**\n**Hipótesis**: Problemas sistemáticos en el proceso de embarque relacionados con organización, gestión de equipajes y coordinación operativa específica por tipo de aeronave, generando fricción en un momento crítico del customer journey que impacta negativamente la percepción general del servicio.\n\n### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**\n\n**Segmento más grande afectado**: **Global/SH/Business** \n\n**Output causal detallado**:\n- **NPS Impact**: Efecto negativo dentro de mejora neta de +11.46 puntos\n- **Driver SHAP**: Boarding -2.104 (significativo negativo)\n- **Satisfacción diferencial**: -1.76 puntos en Boarding\n\n**Alcance de afectación**: Esta causa afecta específicamente al segmento **Short Haul Business**, amplificándose en el subsegmento IB (-3.952 SHAP).\n\n**Métricas clave operativas**:\n- **Otras incidencias**: +26 incidentes (+104% en SH Business/IB)\n- **Validación cualitativa**: 476 análisis de verbatims sobre BOARDING (SH Business) y 318 análisis específicos (SH Business/IB)\n\n**Rutas más impactadas negativamente**:\n- FNC-MAD: -133.3 NPS vs L7d\n- MAD-VLC: -40.0 NPS vs L7d (-106.7 vs L7d)\n- BIO-MAD: +25.0 NPS (-41.7 vs L7d)\n\n**Patrón identificado**: Problemas específicos por tipo de flota con alta variabilidad según **Fleet** (96.0 pts spread en reactividad).\n\n---\n\n## 🎯 CAUSA 3: DETERIORO OPERATIVO MULTIFACTORIAL EN LONG HAUL\n\n### **A. NATURALEZA DE LA CAUSA:**\n**Hipótesis**: Deterioro sistémico en la gestión de rotaciones de aeronaves, mantenimiento y experiencia de llegadas, con problemas específicos en conexiones y gestión de equipajes por limitaciones de peso, concentrado en rutas de largo alcance con mayor complejidad operativa.\n\n### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**\n\n**Segmento más grande afectado**: **Global/LH/Premium**\n\n**Output causal detallado**:\n- **NPS Impact**: -1.48 puntos (23.71 vs 25.19 baseline)\n- **Drivers SHAP negativos**: Arrivals experience (-0.903), Cabin Crew (-0.842), Load factor (-0.321)\n- **Driver SHAP positivo compensatorio**: Aircraft interior (+2.720)\n\n**Alcance de afectación**: Esta causa afecta específicamente al segmento **Long Haul Premium** y todos sus subsegmentos.\n\n**Métricas clave operativas**:\n- **Otras incidencias**: +13 casos (+39.4% vs L7d)\n- **Equipajes no cargados**: 165 casos por limitación de peso\n- **Reubicaciones simultáneas**: Hasta 155 casos\n- **Retrasos máximos**: 4h 55min (vs 2h 11min período anterior)\n\n**Rutas más impactadas**:\n- **Deterioro**: GRU-MAD (-108.3 pts), LAX-MAD (-17.1 pts), MAD-SCL (-23.1 pts)\n- **Mejoras compensatorias**: BOG-MAD (+54.1 pts), DFW-MAD (+70.0 pts)\n\n**Patrón geográfico**: Mayor concentración de problemas en corredor **latinoamericano** (4 de 11 rutas), con clientes segmentados por **Residence Region** mostrando mayor reactividad (102.1 pts spread).\n\n---\n\n## 📋 SÍNTESIS EJECUTIVA FINAL\n\n📈 SÍNTESIS EJECUTIVA:\n\nDurante la semana del 23 al 29 de agosto de 2025, el grupo IAG experimentó una mejora significativa en la satisfacción del cliente, con el NPS global aumentando 3.85 puntos (de 21.50 a 25.34 puntos vs L7d). Esta mejora estuvo impulsada principalmente por una mejora operativa masiva en puntualidad que redujo los retrasos en 219 incidentes (-23.9%) y las cancelaciones en 64 incidentes (-35.0%), generando un impacto SHAP de +3.548 puntos en el driver de Punctuality. Sin embargo, esta mejora se vio parcialmente contrarrestada por un deterioro en la experiencia de Cabin Crew (SHAP -1.367) y percepción de precios (SHAP -0.719). El segmento Short Haul Business experimentó la mayor mejora, pasando de 22.73 a 34.19 puntos NPS (+11.46 puntos vs L7d), mientras que Long Haul Premium mostró un ligero deterioro de 1.48 puntos (de 25.19 a 23.71 puntos vs L7d) debido a problemas operativos multifactoriales en arrivals experience y gestión de equipajes.\n\nLas rutas más beneficiadas por la mejora operativa incluyen MAD-VGO (+105.1 NPS con 14 pasajeros), PMI-RJL (+83.3 NPS con 6 pasajeros) y JTR-MAD (+68.9 NPS con 15 pasajeros), mientras que las rutas de Long Haul como GRU-MAD experimentaron deterioros significativos (-108.3 puntos vs L7d). Los grupos de clientes más reactivos fueron aquellos segmentados por región de residencia, mostrando spreads de hasta 145.0 puntos en Short Haul Business y 102.1 puntos en Long Haul Premium, seguidos por los perfiles de CodeShare con variabilidades de 66.6 puntos a nivel global.\n\n**ECONOMY SH: Desempeño Estable**\nLa cabina Economy de Short Haul mantuvo desempeño estable durante la semana del 23-29 agosto, sin registrar cambios significativos que generaran nodos específicos en el análisis. No se detectaron variaciones importantes, manteniendo niveles consistentes de satisfacción a nivel semanal.\n\n**BUSINESS SH: Mejora Operativa Significativa**\nEl segmento Business de Short Haul experimentó una mejora sustancial, con Iberia registrando un NPS de 41.23 puntos (+14.15 puntos vs L7d) mientras que Air Europa mantuvo desempeño estable sin generar variaciones significativas. Esta evolución se explica principalmente por mejoras operativas en puntualidad (SHAP +9.434) que redujeron retrasos (-20 incidentes, -18.2%) y cancelaciones (-8 incidentes, -25.0%), parcialmente contrarrestadas por deterioro en boarding (SHAP -3.952) validado por 318 análisis de verbatims. La mejora fue especialmente visible en rutas como MAD-OPO (+200.0 NPS), FLR-MAD (+166.7 NPS) y DUS-MAD (+92.9 NPS), siendo los perfiles más reactivos aquellos segmentados por región de residencia (119.9 puntos de spread).\n\n**ECONOMY LH: Desempeño Estable**\nLa cabina Economy de Long Haul mantuvo desempeño estable durante la semana del 23-29 agosto, sin registrar cambios significativos que generaran nodos específicos en el análisis. No se detectaron variaciones importantes, manteniendo niveles consistentes de satisfacción a nivel semanal.\n\n**BUSINESS LH: Desempeño Estable**\nLa cabina Business de Long Haul mantuvo desempeño estable durante la semana del 23-29 agosto, sin registrar cambios significativos que generaran nodos específicos en el análisis. No se detectaron variaciones importantes, manteniendo niveles consistentes de satisfacción a nivel semanal.\n\n**PREMIUM LH: Deterioro Operativo Multifactorial**\nEl segmento Premium de Long Haul experimentó un deterioro de 1.48 puntos, registrando un NPS de 23.71 puntos vs 25.19 puntos la semana anterior (L7d). Las causas dominantes fueron el deterioro en arrivals experience (SHAP -0.903) y cabin crew (SHAP -0.842), evidenciado por el aumento de otras incidencias (+13 casos, +39.4%) y problemas con 165 equipajes no cargados por limitaciones de peso, parcialmente compensado por mejoras en aircraft interior (SHAP +2.720). Este deterioro fue especialmente evidente en rutas latinoamericanas como GRU-MAD (-108.3 puntos vs L7d), LAX-MAD (-17.1 puntos) y MAD-SCL (-23.1 puntos), mientras que los perfiles más reactivos fueron aquellos segmentados por región de residencia (102.1 puntos de spread).\n\n---\n\n✅ **ANÁLISIS COMPLETADO**\n\n- **Nodos procesados:** 4\n- **Pasos de análisis:** 5\n- **Metodología:** Análisis conversacional paso a paso\n- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado\n\n*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*'}]

**ANÁLISIS DIARIO SINGLE:**
📅 2025-08-29 to 2025-08-29:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 9 (Global, Global/LH/Business, Global/LH/Premium, Global/SH, Global/SH/Business, Global/SH/Business/IB, Global/SH/Economy, Global/SH/Economy/IB, Global/SH/Economy/YW)

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 PASO 1: DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)

## **A. Economy Short Haul:**

**NODOS IDENTIFICADOS:**
- ✅ **Economy SH/IB**: Existe (anomalía negativa -14.30 pts)
- ✅ **Economy SH/YW**: Existe (anomalía negativa -8.08 pts)

**ANÁLISIS COMPARATIVO:**

**Drivers Principales:**
- **SH/IB**: Crisis en códigos compartidos (AA: -66.7, BA: -45.5) y flotas específicas (A332: -40.0)
- **SH/YW**: Crisis total en codeshare American Airlines (-100.0) y ruta específica CMN-MAD (0.0)

**Evidencia Operativa:**
- **Ambos comparten**: Los mismos 40 incidentes operativos (20 retrasos, 6 cancelaciones)
- **Diferencia clave**: YW muestra impacto más severo en AA (-100.0 vs -66.7 en IB)

**DIAGNÓSTICO ECONOMY SH:** **Causa común de cabina con amplificación específica por compañía** - Ambas compañías sufren los mismos problemas operativos base, pero YW presenta mayor vulnerabilidad en el partnership con American Airlines.

---

## **B. Business Short Haul:**

**NODOS IDENTIFICADOS:**
- ✅ **Business SH/IB**: Existe (anomalía positiva +28.93 pts)
- ❌ **Business SH/YW**: No existe

**RAZÓN DE AUSENCIA YW:** El nodo padre Global/SH/Business muestra anomalía positiva (+14.40 pts), pero solo se desglosa el componente IB, sugiriendo que YW no presentó anomalía significativa o volumen insuficiente de datos.

**DIAGNÓSTICO BUSINESS SH:** **Causa específica de compañía IB** - Solo IB presenta comportamiento anómalo positivo excepcional (+28.93 pts) mientras YW mantiene comportamiento normal, indicando drivers específicos de la operación o gestión de Iberia en Business.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ PASO 2: DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)

## **A. Short Haul - Análisis de Cabinas:**

**NODOS IDENTIFICADOS:**
- ❌ **Economy SH**: Anomalía negativa (-11.61 pts)
- ✅ **Business SH**: Anomalía positiva (+14.40 pts)

**ANÁLISIS DE CONVERGENCIA/DIVERGENCIA:**

**Patrones Operativos:**
- **Ambas cabinas**: Expuestas a los mismos 40 incidentes operativos (20 retrasos, 6 cancelaciones)
- **Economy SH**: Impacto negativo severo con dispersión extrema (145.8 pts en codeshares)
- **Business SH**: Convierte crisis en oportunidad con mejora excepcional

**Consistencia entre Compañías:**
- **Economy**: IB (-14.30) y YW (-8.08) ambas negativas, patrón convergente de impacto negativo
- **Business**: Solo IB (+28.93) presenta anomalía, YW sin datos anómalos

**Reactividad Diferencial:**
- **Economy**: Alta vulnerabilidad a disrupciones operativas
- **Business**: Resiliencia excepcional, posible gestión proactiva de crisis

**DIAGNÓSTICO SHORT HAUL:** **Causa común de radio con reactividad específica de cabina** - Ambas cabinas sufren los mismos problemas operativos, pero Business demuestra capacidad superior de gestión de crisis mientras Economy amplifica negativamente el impacto.

---

## **B. Long Haul - Análisis de Cabinas:**

**NODOS IDENTIFICADOS:**
- ❌ **Business LH**: Anomalía negativa (-15.53 pts)
- ❌ **Premium LH**: Anomalía negativa (-6.35 pts)
- ❌ **Economy LH**: No existe nodo específico

**ANÁLISIS DE CONVERGENCIA:**

**Patrones Operativos:**
- **Ambas cabinas**: Expuestas a los mismos 25 incidentes operativos LH
- **Business LH**: Disparidad extrema por región (120 pts) y operador (114.3 pts)
- **Premium LH**: Disparidad crítica entre flotas A350 (67.1 pts)

**Progresión de Reactividad:**
- **Business LH**: Mayor impacto (-15.53 pts), alta sensibilidad a disrupciones
- **Premium LH**: Impacto moderado (-6.35 pts), menor pero significativa reactividad

**Factor Común:**
- Ambas muestran **problemas específicos de flota A350** y **impacto geográfico diferencial**

**DIAGNÓSTICO LONG HAUL:** **Causa común de radio con progresión lógica de reactividad** - Ambas cabinas convergen en los mismos drivers operativos (incidentes LH, problemas A350), pero Business muestra mayor sensibilidad al impacto que Premium, siguiendo una progresión esperada por tipo de cliente.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 PASO 3: DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

## **A. Comparación Entre Radios:**

**NODOS IDENTIFICADOS:**
- ❌ **Short Haul**: Anomalía negativa (-9.18 pts)
- ❌ **Long Haul**: No existe nodo específico (inferido negativo por cabinas Business -15.53 y Premium -6.35)
- ❌ **Global**: Anomalía negativa (-5.95 pts)

**ANÁLISIS DE CONVERGENCIA ENTRE RADIOS:**

**Drivers Principales:**
- **Short Haul**: 40 incidentes operativos (50% retrasos), dispersión extrema en codeshares (118.3 pts)
- **Long Haul**: 25 incidentes operativos, problemas específicos A350 y disparidad geográfica (120 pts)

**Evidencia Operativa Común:**
- **Ambos radios**: Experimentan crisis operativas masivas el mismo día
- **Patrón compartido**: Retrasos como factor dominante, efectos cascada, problemas meteorológicos
- **Diferencia de escala**: SH con mayor volumen de incidentes (40 vs 25)

**Consistencia de Impacto:**
- **Short Haul**: Impacto más severo (-9.18 pts) por mayor exposición a incidentes
- **Long Haul**: Impacto moderado pero significativo (inferido de cabinas)

---

## **B. Coherencia con Nodo Global:**

**ANÁLISIS DE AGREGACIÓN:**

**Coherencia Matemática:**
- **Global (-5.95 pts)** refleja un **promedio ponderado coherente** entre SH (-9.18) y LH (negativo inferido)
- **No hay efectos compensatorios** - ambos radios contribuyen negativamente

**Evidencia Operativa Global:**
- **380 incidentes totales** = 40 SH + 25 LH + otros segmentos
- **Patrón consolidado**: 116 retrasos globales (30.5%) confirma retrasos como driver principal
- **Rutas críticas**: GYE-MAD y FRA-MAD (ambas destino Madrid) aparecen en análisis global

**Amplificación/Atenuación:**
- **Global atenúa** el impacto individual de cada radio
- **Dispersión global** (160.0 pts en codeshares) confirma patrones de ambos radios

---

## **🎯 DIAGNÓSTICO FINAL:**

**CAUSA GLOBAL HOMOGÉNEA CON DIFERENCIAS DE ESCALA** - Ambos radios (Short Haul y Long Haul) convergen en el mismo patrón causal fundamental: crisis operativa masiva con retrasos como driver principal, efectos cascada y dispersión extrema por perfiles de cliente. El nodo Global refleja coherentemente esta convergencia, siendo la diferencia principal la escala de incidentes (40 SH vs 25 LH) que explica la mayor severidad en Short Haul, pero manteniendo la misma naturaleza operativa de la crisis del 29 de agosto de 2025.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 PASO 4: ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

## **CAUSA 1: CRISIS OPERATIVA MASIVA - RETRASOS Y CANCELACIONES**

### **A. NATURALEZA DE LA CAUSA:**
**Hipótesis:** Colapso operativo temporal causado por una combinación de factores meteorológicos adversos, efectos cascada de desvíos anteriores y problemas técnicos simultáneos que generaron un volumen crítico de disrupciones concentradas en un solo día, superando la capacidad de gestión normal de la operación.

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**

**Segmento más grande afectado:** **Global** (todos los subsegmentos)

**Output causal detallado:**
- **380 incidentes totales** registrados en un día excepcional
- **116 retrasos** (30.5% del total) - Principal disruptor identificado
- **30 cancelaciones** con impacto directo severo en experiencia
- **44 vuelos afectados** con múltiples disrupciones simultáneas
- **32 pérdidas de conexión** y **28 cambios de avión**

**Alcance de impacto:** Esta causa afecta a **todos los subsegmentos** bajo el nodo Global, incluyendo Short Haul (-9.18 pts), Long Haul (Business -15.53, Premium -6.35), todas las compañías (IB/YW) y todas las cabinas (Economy/Business/Premium).

**Métricas clave:**
- **NPS Global:** Caída de 5.95 puntos (25.95 → 20.0)
- **Incidentes críticos específicos:** MVD-LPA-MAD (cancelación por desvío), BCN-BOS (meteorología), GYE-MAD (múltiples disrupciones)
- **Rutas involucradas:** 7+ rutas diferentes afectadas
- **Verbatims:** No disponibles (limitación del análisis)

---

## **CAUSA 2: VULNERABILIDAD EXTREMA EN CODESHARES INTERNACIONALES**

### **A. NATURALEZA DE LA CAUSA:**
**Hipótesis:** Los partnerships de código compartido, especialmente con American Airlines, muestran mayor vulnerabilidad a disrupciones operativas debido a procesos de coordinación más complejos, menor control directo sobre la gestión de crisis y diferentes estándares de comunicación con pasajeros durante incidentes.

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**

**Segmento más grande afectado:** **Global/SH** (Short Haul completo)

**Output causal detallado:**
- **Dispersión extrema de 118.3 puntos** en codeshares
- **American Airlines (AA):** NPS -60.0 a -100.0 según segmento
- **LATAM:** NPS -33.3 (impacto crítico)
- **British Airways (BA):** NPS -31.2 a -45.5
- **Contraste con Vueling (VY):** NPS +58.3 (excelente performance)

**Alcance de impacto:** Afecta a todos los subsegmentos de Short Haul, con particular severidad en Economy SH/IB (-14.30 pts) y Economy SH/YW (-8.08 pts), donde YW muestra mayor vulnerabilidad al partnership AA.

**Métricas clave:**
- **NPS Short Haul:** -9.18 puntos vs baseline
- **Volumen crítico:** 40 incidentes concentrados en SH
- **Rutas críticas:** AMS-MAD (NPS -20.0), CMN-MAD (NPS 0.0)
- **Dispersión máxima:** 145.8 puntos en codeshares Economy SH

---

## **CAUSA 3: PROBLEMAS ESPECÍFICOS DE FLOTA A350 Y WIDE-BODY**

### **A. NATURALEZA DE LA CAUSA:**
**Hipótesis:** Las flotas A350 (especialmente tradicional vs Next) y wide-body en general presentaron problemas técnicos o operativos específicos que generaron disparidades extremas en la experiencia del cliente, posiblemente relacionados con mantenimiento, configuración de cabina o rutas específicas operadas por estos equipos.

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**

**Segmento más grande afectado:** **Global/LH** (Long Haul completo)

**Output causal detallado:**
- **A350 Tradicional:** NPS -10.0 (crítico)
- **A350 Next:** NPS 57.1 (excelente) 
- **Disparidad crítica:** 67.1 puntos entre versiones A350
- **A332:** NPS -40.0 (impacto severo)
- **A33ACMI:** NPS -30.0 (problemático)

**Alcance de impacto:** Afecta principalmente a Long Haul Business (-15.53 pts) y Premium (-6.35 pts), con **25 incidentes operativos** específicos de LH, incluyendo el caso crítico MVD-LPA-MAD con cancelación por desvío.

**Métricas clave:**
- **Incidentes LH:** 25 total (9 retrasos, 6 cancelaciones, 8 otras)
- **Dispersión geográfica:** 120 puntos entre regiones en Business LH
- **Rutas específicas:** EZE-MAD (NPS 75.0 - única con datos positivos)
- **Correlación flota-geografía:** A350 Next correlaciona con mejor performance regional

---

## **CAUSA 4: REACTIVIDAD DIFERENCIAL POR PERFIL GEOGRÁFICO**

### **A. NATURALEZA DE LA CAUSA:**
**Hipótesis:** Clientes de diferentes regiones geográficas muestran tolerancias y expectativas distintas ante disrupciones operativas, con pasajeros intercontinentales (Asia, África, América Norte) presentando mayor sensibilidad y menor tolerancia a retrasos y cancelaciones que pasajeros regionales.

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**

**Segmento más grande afectado:** **Global** (impacto transversal)

**Output causal detallado:**
- **AFRICA:** NPS -33.3 (6 surveys) - Impacto severo
- **ASIA:** NPS -54.5 (11 surveys) - Impacto crítico  
- **AMERICA NORTE:** NPS -40.0 a -100.0 según segmento
- **Contraste positivo AMERICA SUR:** NPS 80.0 en Business LH
- **Dispersión geográfica máxima:** 106.4 puntos a nivel Global

**Alcance de impacto:** Afecta transversalmente a todos los segmentos, con patrones consistentes donde clientes de Asia, África y América Norte muestran mayor reactividad negativa, mientras que América Sur y Centro mantienen mejor resiliencia.

**Métricas clave:**
- **Correlación rutas-geografía:** GYE-MAD y FRA-MAD (ambas destino Madrid) en rutas críticas globales
- **Volumen representativo:** 11 surveys Asia, 6 surveys África
- **Patrón consistente:** Repetido en múltiples subsegmentos (SH Economy, LH Business)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El 29 de agosto de 2025 experimentó una crisis operativa masiva que impactó significativamente la satisfacción del cliente, con el NPS global cayendo 5.95 puntos (de 25.95 a 20.0) respecto a la media de los últimos 7 días. Esta caída se explica por cuatro causas principales: una crisis operativa con 380 incidentes totales (116 retrasos como factor dominante), vulnerabilidad extrema en codeshares internacionales (especialmente American Airlines con NPS -60.0 a -100.0), problemas específicos de flota A350 con disparidad de 67.1 puntos entre versiones tradicional (-10.0) y Next (57.1), y reactividad diferencial por perfil geográfico donde clientes de Asia (-54.5), África (-33.3) y América Norte (-40.0 a -100.0) mostraron mayor sensibilidad a las disrupciones. Short Haul experimentó una caída de 9.18 puntos (19.12 vs 28.31 baseline) mientras que Long Haul también presentó impactos negativos evidenciados en Business (-15.53 puntos) y Premium (-6.35 puntos).

Las rutas más afectadas incluyen GYE-MAD con NPS -33.3, FRA-MAD con NPS -28.6 (ambas con destino Madrid), AMS-MAD con NPS -20.0, y CMN-MAD con NPS 0.0, todas correlacionando directamente con los incidentes operativos reportados. Los grupos de clientes más reactivos fueron los pasajeros de codeshares internacionales (especialmente AA, LATAM y BA), clientes intercontinentales de Asia y África, y usuarios de flotas wide-body específicas (A332, A33ACMI), mientras que los pasajeros de Vueling (NPS +58.3), América Centro (+55.0) y flota ATR (+44.4) demostraron mayor resiliencia ante las disrupciones operativas.

**ECONOMY SH: Crisis Operativa con Impacto Diferencial por Compañía**
La cabina Economy de Short Haul experimentó una caída significativa durante el 29 de agosto de 2025, con Iberia registrando un deterioro de 14.30 puntos (NPS 11.43 vs 25.73 de los últimos 7 días) y Vueling una caída de 8.08 puntos (NPS 24.0 vs 32.08 baseline). La causa principal fue la crisis operativa masiva con 40 incidentes (20 retrasos como factor dominante), complementada por la vulnerabilidad extrema en codeshares internacionales donde American Airlines alcanzó NPS -66.7 en IB y -100.0 en YW. Esta deterioro se reflejó especialmente en rutas como AMS-MAD (NPS -20.0) y CMN-MAD (NPS 0.0), mientras que los perfiles más reactivos incluyen clientes de África (NPS -60.0), América Norte (NPS -14.3 a -100.0) y usuarios de flotas A332 (NPS -40.0) y 32S (NPS -38.5).

**BUSINESS SH: Resiliencia Excepcional de Iberia**
El segmento Business de Short Haul mostró comportamientos divergentes por compañía, con Iberia registrando una mejora excepcional de 28.93 puntos (NPS 66.67 vs 37.74 de los últimos 7 días) mientras Vueling mantuvo desempeño estable. Esta evolución paradójica de IB se explica principalmente por una gestión excepcional de crisis que convirtió los 40 incidentes operativos en oportunidades de servicio, siendo especialmente visible en mercados americanos (América Centro y Norte con NPS 100.0) y contrastando con la concentración de problemas en flota A321 (NPS 40.0) y regiones no identificadas (NPS -25.0).

**ECONOMY LH: Cabina Estable**
La cabina Economy de Long Haul mantuvo desempeño estable durante el 29 de agosto de 2025. No se detectaron cambios significativos a nivel de cabina, manteniendo niveles consistentes de satisfacción.

**BUSINESS LH: Impacto Severo por Disparidades Operativas**
La cabina Business de Long Haul experimentó una caída significativa de 15.53 puntos (NPS 27.27 vs 42.80 de los últimos 7 días). Los drivers principales fueron la crisis operativa con 25 incidentes específicos de LH y la disparidad extrema por región de residencia (120 puntos entre América Sur NPS 80.0 y América Norte NPS -40.0) y operador (114.3 puntos entre Iberia NPS 39.3 y CodeShare American Airlines NPS -75.0), impactando especialmente las rutas con problemas técnicos mientras EZE-MAD mantuvo excelencia (NPS 75.0) y perfiles como pasajeros sudamericanos y flota A350 Next.

**PREMIUM LH: Disparidad Crítica de Flota A350**
El segmento Premium de Long Haul registró una caída de 6.35 puntos (NPS 21.05 vs 27.40 de los últimos 7 días). Las causas dominantes fueron los 25 incidentes operativos de LH y especialmente la disparidad crítica de 67.1 puntos entre flota A350 Next (NPS 57.1) y A350 Tradicional (NPS -10.0), especialmente evidentes en rutas operadas con equipos tradicionales que absorbieron el impacto de cancelaciones y retrasos, mientras que la flota Next mantuvo excelencia operativa a pesar del contexto adverso.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 9
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-08-28 to 2025-08-28:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 9 (Global, Global/LH/Business, Global/LH/Premium, Global/SH, Global/SH/Business, Global/SH/Business/IB, Global/SH/Economy, Global/SH/Economy/IB, Global/SH/Economy/YW)

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 PASO 1: DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)

## **A. Economy Short Haul:**

**Nodos identificados:**
- ✅ **Economy SH/IB**: Existe con anomalía negativa (-)
- ✅ **Economy SH/YW**: Existe con anomalía negativa (-)

**Análisis comparativo:**

**Economy SH/IB:**
- **Caída NPS**: -8.76 puntos (hasta 16.98)
- **Drivers principales**: A33ACMI (-50.0), LATAM CodeShare (-100.0), América del Norte (-33.3)
- **Evidencia operativa**: 49 incidentes, problemas meteorológicos MVD, falla total LATAM

**Economy SH/YW:**
- **Caída NPS**: -6.47 puntos (hasta 25.61)
- **Drivers principales**: CodeShare VY (-40.0), viajeros Business/Work (-13.6)
- **Evidencia operativa**: Mismos 49 incidentes, mismo trigger meteorológico MVD

**Diagnóstico Economy SH**: **Causa común de cabina** - Ambas compañías convergen en el mismo trigger operativo (problemas meteorológicos MVD, 49 incidentes) pero con manifestaciones específicas por partner (LATAM en IB, VY en YW).

---

## **B. Business Short Haul:**

**Nodos identificados:**
- ✅ **Business SH/IB**: Existe con anomalía negativa (-)
- ❌ **Business SH/YW**: No existe

**Razón de ausencia YW**: YW no opera cabina Business en Short Haul, solo Economy.

**Diagnóstico Business SH**: **No aplica** - Solo existe operación IB en esta cabina, por lo que no hay comparación posible entre compañías.

---

## **CONCLUSIÓN DIAGNÓSTICA:**

La anomalía del 28 de agosto tiene una **causa común operativa** (crisis meteorológica MVD + 49 incidentes) que se manifiesta de forma diferenciada según los partners específicos de cada compañía, pero no representa problemas estructurales distintos entre IB y YW.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ PASO 2: DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)

## **A. Short Haul - Análisis de Cabinas:**

**Nodos identificados:**
- ❌ **Economy SH**: Existe con anomalía negativa (-)
- ❌ **Business SH**: Existe con anomalía negativa (-)

**Análisis comparativo:**

**Economy SH:**
- **Caída NPS**: -8.0 puntos (hasta 19.9)
- **Drivers principales**: A33ACMI (-50.0), LATAM (-60.0), Vueling (-15.4)
- **Reactividad**: Alta sensibilidad a problemas de flota ACMI y partners

**Business SH:**
- **Caída NPS**: -13.45 puntos (hasta 18.97)
- **Drivers principales**: A320neo (0.0), BA CodeShare (-33.3), Europa (-18.8)
- **Reactividad**: Mayor sensibilidad (caída más pronunciada) a disrupciones operativas

**Diagnóstico Short Haul**: **Causa común de radio con reactividad diferencial** - Ambas cabinas convergen en el mismo trigger operativo (49 incidentes, crisis meteorológica MVD), pero Business SH muestra mayor sensibilidad (-13.45 vs -8.0 puntos), indicando que los clientes corporativos reaccionaron más negativamente a las disrupciones.

---

## **B. Long Haul - Análisis de Cabinas:**

**Nodos identificados:**
- ❌ **Economy LH**: No existe como nodo separado
- ❌ **Business LH**: Existe con anomalía negativa (-)
- ➕ **Premium LH**: Existe con anomalía positiva (+)

**Análisis comparativo:**

**Business LH:**
- **Caída NPS**: -0.38 puntos (hasta 42.42)
- **Drivers principales**: Problemas meteorológicos MVD, Europa (0.0)
- **Reactividad**: Baja sensibilidad, caída mínima

**Premium LH:**
- **Subida NPS**: +27.14 puntos (hasta 54.55)
- **Drivers principales**: Gestión diferencial A333 (66.7), comunicación proactiva
- **Reactividad**: Reacción positiva ante gestión de crisis

**Diagnóstico Long Haul**: **Reactividad diferencial extrema por cabina** - Misma causa operativa (crisis meteorológica MVD) genera respuestas completamente opuestas: Business LH mantiene estabilidad (-0.38) mientras Premium LH experimenta mejora excepcional (+27.14), sugiriendo gestión diferenciada efectiva para clientes Premium.

---

## **CONCLUSIÓN DIAGNÓSTICA:**

**Short Haul**: Causa común con **escalamiento por nivel de servicio** (Business más reactivo que Economy).
**Long Haul**: Causa común con **gestión diferencial efectiva** que beneficia desproporcionalmente a Premium frente a Business.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 PASO 3: DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

## **A. Comparación Entre Radios:**

**Nodos identificados:**
- ❌ **Short Haul**: Existe con anomalía negativa (-)
- ➕/❌ **Long Haul**: Patrones mixtos (Business LH negativo, Premium LH positivo)

**Análisis comparativo:**

**Short Haul:**
- **Caída NPS**: -8.53 puntos (hasta 19.78)
- **Drivers principales**: A33ACMI (-60.0), LATAM (-60.0), problemas meteorológicos MVD
- **Evidencia operativa**: 49 incidentes, concentrados en retrasos y cancelaciones

**Long Haul:**
- **Business LH**: -0.38 puntos (impacto mínimo)
- **Premium LH**: +27.14 puntos (mejora excepcional)
- **Drivers principales**: Mismos problemas meteorológicos MVD, pero gestión diferencial efectiva
- **Evidencia operativa**: Mismos 26 incidentes, pero con comunicación proactiva y gestión de crisis superior

**Convergencia de drivers**: ✅ **SÍ** - Ambos radios convergen en el mismo trigger operativo (crisis meteorológica MVD, problemas de conectividad sudamericana), pero con **respuestas operativas diferenciadas**.

---

## **B. Coherencia con Nodo Global:**

**Nodo Global:**
- ❌ **Anomalía negativa**: -4.18 puntos (hasta 21.77)
- **Drivers principales**: 449 incidentes totales, A33ACMI (-21.9), LATAM (-33.3)
- **Patrón agregado**: Refleja predominancia del impacto negativo de Short Haul

**Coherencia del agregado:**
- **Short Haul** (-8.53) + **Long Haul** (mixto: -0.38 Business, +27.14 Premium) = **Global** (-4.18)
- ✅ **Coherente**: El Global refleja el impacto neto donde Short Haul domina la caída, parcialmente compensada por Premium LH

---

## **CONCLUSIÓN DIAGNÓSTICA:**

**Causa global homogénea con gestión diferencial por radio** - Ambos radios convergen en el mismo trigger operativo (crisis meteorológica MVD, 449 incidentes totales), pero Short Haul experimentó impacto operativo directo (-8.53) mientras Long Haul demostró capacidad de gestión diferencial que convirtió la crisis en oportunidad para Premium (+27.14) y mantuvo estabilidad en Business (-0.38). El nodo Global (-4.18) refleja coherentemente esta combinación de impactos.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 PASO 4: ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

## **CAUSA 1: CRISIS METEOROLÓGICA EN MONTEVIDEO (MVD)**

### A. NATURALEZA DE LA CAUSA:
**Hipótesis**: Condiciones meteorológicas adversas (baja visibilidad) en el aeropuerto de Montevideo generaron un efecto cascada operativo que se propagó a través de la red de conectividad sudamericana, impactando la puntualidad y disponibilidad de vuelos en rutas críticas.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:
**Segmento más grande afectado**: **Global** (546 encuestas)

**Output causal detallado**:
- **449 incidentes operativos totales** registrados
- **141 retrasos** (31% del total) como principal factor de insatisfacción
- **22 cancelaciones** con impacto directo en experiencia del cliente
- **Rutas críticas afectadas**: MVD-LPA, MAD-MVD, EZE-MVD
- **Efecto cascada**: Desvío MAD-MVD a EZE → Cancelación MVD-LPA-MAD → Reprogramación como IB046/29AUG

**Alcance**: Esta causa afecta a **todos los subsegmentos** bajo el nodo Global, manifestándose de forma diferenciada según radio y cabina.

**Métricas clave**:
- **NPS Global**: Caída de 4.18 puntos (25.95 → 21.77)
- **Comunicación de crisis**: Activación de "IB Conecta" para información proactiva
- **Gestión de recuperación**: Reprogramación operativa implementada

---

## **CAUSA 2: FALLAS CRÍTICAS EN FLOTA A33ACMI**

### A. NATURALEZA DE LA CAUSA:
**Hipótesis**: Problemas operativos específicos en la flota A33ACMI (aeronaves en régimen ACMI - Aircraft, Crew, Maintenance, Insurance) que generaron una experiencia de servicio severamente deficiente, posiblemente relacionados con estándares operativos diferenciados o capacidad de respuesta ante disrupciones.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:
**Segmento más grande afectado**: **Global/SH** (múltiples subsegmentos)

**Output causal detallado**:
- **A33ACMI**: NPS -21.9 (32 encuestas a nivel Global)
- **En Economy SH/IB**: NPS -50.0 (8 encuestas)
- **Dispersión extrema**: 81.9 puntos entre flotas a nivel Global

**Alcance**: Esta causa afecta específicamente a **todos los pasajeros que volaron en flota A33ACMI** independientemente de cabina o compañía.

**Métricas clave**:
- **Concentración del problema**: Solo en flota ACMI vs. flotas propias con NPS positivos
- **Impacto cuantificado**: Diferencia de hasta 90.9 puntos vs. otras flotas
- **Rutas involucradas**: No especificadas, pero concentradas en operación Short Haul

---

## **CAUSA 3: COLAPSO DE CODESHARE LATAM**

### A. NATURALEZA DE LA CAUSA:
**Hipótesis**: Fallas sistémicas en la operación del partner LATAM durante la crisis meteorológica, posiblemente relacionadas con protocolos de gestión de disrupciones, comunicación con pasajeros o estándares de servicio diferenciados durante situaciones de contingencia.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:
**Segmento más grande afectado**: **Global** (múltiples manifestaciones)

**Output causal detallado**:
- **LATAM CodeShare**: NPS -33.3 a nivel Global
- **En Economy SH**: NPS -60.0 (5 encuestas)
- **En Economy SH/IB**: NPS -100.0 (4 encuestas) - colapso total
- **Dispersión extrema**: 83.3 puntos entre partners a nivel Global

**Alcance**: Esta causa afecta a **todos los pasajeros en vuelos operados bajo codeshare LATAM** en cualquier cabina o ruta.

**Métricas clave**:
- **Contraste operativo**: IB mainline mantuvo NPS +22.7 vs. LATAM -60.0/-100.0
- **Concentración geográfica**: Rutas sudamericanas principalmente
- **Impacto diferencial**: Colapso específico del partner vs. operación propia estable

---

## **CAUSA 4: SENSIBILIDAD DIFERENCIAL DE VIAJEROS CORPORATIVOS**

### A. NATURALEZA DE LA CAUSA:
**Hipótesis**: Los viajeros de negocio muestran mayor sensibilidad a disrupciones operativas debido a expectativas más altas de puntualidad, conectividad y gestión proactiva de problemas, especialmente en rutas europeas donde la conectividad es crítica para itinerarios corporativos.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:
**Segmento más grande afectado**: **Global** (diferenciación por propósito)

**Output causal detallado**:
- **Business/Work**: NPS 0.0 (110 encuestas) vs. **Leisure**: NPS 25.2
- **Brecha crítica**: 25.2 puntos de diferencia
- **En Business SH**: Mayor caída (-13.45 pts) vs. Economy SH (-8.0 pts)
- **Mercados más afectados**: Europa (NPS 5.1) y América Norte (NPS 7.8)

**Alcance**: Esta causa afecta a **todos los viajeros corporativos** independientemente de cabina, con mayor intensidad en rutas europeas y norteamericanas.

**Métricas clave**:
- **Escalamiento por segmento**: Business más reactivo que Leisure consistentemente
- **Concentración geográfica**: Europa y América Norte más impactados
- **Contraste regional**: América Centro (45.7) y Asia (40.0) mantuvieron niveles aceptables

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El 28 de agosto de 2025 registró una caída significativa del NPS global de 4.18 puntos, deteriorándose de 25.95 a 21.77 puntos debido a disrupciones operativas masivas desencadenadas por problemas meteorológicos en Montevideo. La anomalía se concentró principalmente en Short Haul, donde Economy SH experimentó una caída de 8.0 puntos hasta 19.9 y Business SH se deterioró 13.45 puntos hasta 18.97, mientras que en Long Haul se observaron patrones divergentes: Business LH mantuvo estabilidad con una caída mínima de 0.38 puntos (42.42) pero Premium LH experimentó una mejora excepcional de 27.14 puntos alcanzando 54.55. Las causas principales incluyen fallas críticas en flota A33ACMI (NPS -21.9 a nivel global), colapso del codeshare LATAM (NPS -33.3 global), y problemas meteorológicos que generaron 449 incidentes operativos totales. Estas disrupciones se evidenciaron a través de 141 retrasos, 22 cancelaciones y el efecto cascada desde el desvío MAD-MVD hacia EZE que causó cancelaciones posteriores en la ruta MVD-LPA-MAD.

Las rutas más afectadas se concentraron en el corredor sudamericano, particularmente MVD-LPA, MAD-MVD y EZE-MVD, mientras que rutas específicas como MAD-PTY registraron NPS extremadamente negativos de -28.6. Los grupos más reactivos fueron los viajeros corporativos (Business/Work con NPS 0.0 vs Leisure 25.2), clientes europeos (NPS 5.1) y norteamericanos (NPS 7.8), contrastando con la mayor resistencia mostrada por pasajeros de América Centro (45.7) y Asia (40.0). La segmentación por flota reveló que A33ACMI fue el principal driver negativo mientras que la gestión diferencial permitió que Premium LH convirtiera la crisis en una oportunidad de mejora excepcional.

**ECONOMY SH: Deterioro por Crisis Operativa Diferenciada**
La cabina Economy de Short Haul experimentó un deterioro significativo durante el 28 de agosto de 2025, con IB registrando una caída de 8.76 puntos hasta 16.98 e YW deteriorándose 6.47 puntos hasta 25.61 respecto al baseline de 7 días previos. La causa principal fue la combinación de 49 incidentes operativos desencadenados por problemas meteorológicos en Montevideo, que se amplificaron por fallas específicas en flota A33ACMI (NPS -50.0 en IB) y colapso total del codeshare LATAM (NPS -100.0 en IB vs -40.0 de VY en YW). Este deterioro se reflejó especialmente en rutas sudamericanas como MVD-LPA y MAD-MVD, mientras que los perfiles más reactivos incluyeron viajeros de América del Norte (NPS -33.3 en IB) y pasajeros de codeshare específicos, con IB mostrando mayor vulnerabilidad debido a su dependencia de partners críticos.

**BUSINESS SH: Máxima Sensibilidad Corporativa**
El segmento Business de Short Haul registró el mayor deterioro del día con una caída de 13.45 puntos hasta 18.97 vs el baseline de 32.41 de los 7 días previos. Esta evolución se explica principalmente por la mayor sensibilidad de los viajeros corporativos ante las 49 disrupciones operativas, siendo especialmente visible en la flota A320neo (NPS 0.0) y entre clientes europeos (NPS -18.8), mientras que el codeshare American Airlines mantuvo un NPS perfecto de 100.0 contrastando con British Airways (-33.3), evidenciando gestión diferencial entre partners durante la crisis meteorológica.

**ECONOMY LH: Cabina Estable**
Economy LH mantuvo desempeño estable durante el 28 de agosto de 2025. No se detectaron cambios significativos a nivel de cabina, manteniendo niveles consistentes de satisfacción pese a las disrupciones operativas que afectaron otros segmentos.

**BUSINESS LH: Resiliencia ante Crisis**
La cabina Business de Long Haul mantuvo desempeño estable registrando un NPS de 42.42 el 28 de agosto con una caída mínima de 0.38 puntos respecto a la media de los últimos 7 días (42.80). Los drivers principales fueron los mismos problemas meteorológicos en MVD que generaron 26 incidentes, impactando especialmente a clientes europeos (NPS 0.0) mientras que América Centro mantuvo satisfacción perfecta (NPS 100.0), demostrando que la gestión proactiva de crisis y comunicación a través de IB Conecta logró minimizar el impacto en este segmento premium.

**PREMIUM LH: Excelencia en Gestión de Crisis**
El segmento Premium de Long Haul experimentó una mejora excepcional de 27.14 puntos el 28 de agosto, alcanzando un NPS de 54.55 vs el baseline de 27.40 de la semana anterior. Las causas dominantes fueron la gestión diferencial efectiva durante los problemas meteorológicos en MVD, especialmente evidentes en la flota A333 (NPS 66.7 vs A350 next 33.3) y entre residentes de España (NPS 60.0), donde la comunicación proactiva y protocolos de contingencia convirtieron una crisis operativa potencial en una experiencia superior que superó las expectativas de los clientes premium.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 9
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-08-27 to 2025-08-27:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 8 (Global, Global/LH/Business, Global/LH/Premium, Global/SH, Global/SH/Business/YW, Global/SH/Economy, Global/SH/Economy/IB, Global/SH/Economy/YW)

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)

## **A. ECONOMY SHORT HAUL**

**Nodos identificados:**
- ✅ **Economy SH/IB**: Existe (NPS 25.0, -0.73 pts)
- ✅ **Economy SH/YW**: Existe (NPS 20.71, -11.37 pts)

**Análisis de divergencia:**
- **Magnitud**: IB (-0.73 pts) vs YW (-11.37 pts) = **Diferencia de 10.64 puntos**
- **Drivers principales divergentes**:
  - **IB**: Crisis CodeShare BA (-50.0 NPS), problemas flota A332 (-21.4 NPS)
  - **YW**: Crisis CodeShare AA/VY (-33.3 NPS c/u), impacto geográfico Europa (-20.0 NPS)
- **Evidencia operativa**: Ambos comparten los 24 incidentes base, pero impactan diferentes segmentos

**📊 DIAGNÓSTICO ECONOMY SH**: **Causa específica de compañía** - IB y YW divergen significativamente en drivers (diferentes socios CodeShare afectados) y magnitud de impacto (10.64 puntos de diferencia).

---

## **B. BUSINESS SHORT HAUL**

**Nodos identificados:**
- ❌ **Business SH/IB**: No existe
- ✅ **Business SH/YW**: Existe (NPS 30.77, +13.77 pts)

**Razón de ausencia SH/IB**: Solo YW presenta anomalía detectable en Business Short Haul. IB Business SH no mostró desviaciones significativas respecto a su baseline para generar nodo específico.

**📊 DIAGNÓSTICO BUSINESS SH**: **Causa específica de compañía YW** - Solo YW presenta anomalía (+13.77 pts paradójica a pesar de 24 incidentes), indicando que los factores operativos del día impactaron de manera única a los clientes Business de Vueling.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)

## **A. SHORT HAUL - ANÁLISIS DE CABINAS**

**Nodos identificados:**
- **Economy SH**: Negativo (-4.51 pts global, -0.73 pts IB, -11.37 pts YW)
- **Business SH**: Solo YW positivo (+13.77 pts)

**Análisis de convergencia/divergencia:**
- **Patrón completamente divergente**: Economy SH muestra caídas generalizadas vs Business SH (YW) muestra mejora significativa
- **Reactividad diferencial extrema**: Los mismos 24 incidentes operativos generaron:
  - Economy: Impacto negativo consistente (especialmente CodeShare)
  - Business YW: Mejora paradójica de +13.77 pts
- **Separación IB/YW inconsistente**: 
  - Economy: Ambas compañías afectadas negativamente (aunque con diferentes socios)
  - Business: Solo YW presenta anomalía (IB sin desviación significativa)

**📊 DIAGNÓSTICO SHORT HAUL**: **Causa específica de cabina con reactividad diferencial** - Economy SH reaccionó negativamente a los problemas operativos mientras Business SH mostró resistencia (IB) o mejora paradójica (YW), indicando que el perfil de cliente determina la respuesta a las disrupciones.

---

## **B. LONG HAUL - ANÁLISIS DE CABINAS**

**Nodos identificados:**
- **Business LH**: Positivo (+25.62 pts)
- **Premium LH**: Negativo (-14.9 pts)
- **Economy LH**: No existe nodo específico

**Análisis de progresión por clase de servicio:**
- **Patrón completamente divergente**: Business LH (+25.62) vs Premium LH (-14.9) = **Diferencia de 40.52 puntos**
- **Reactividad inversa**: Mismos factores operativos (13 incidentes LH) generaron respuestas opuestas
- **Segmentación geográfica crítica**:
  - Business LH: América Sur perfecta (NPS 100.0) vs España problemática (NPS 33.3)
  - Premium LH: Concentrado en MAD-MEX (NPS 0.0, todos detractores)
- **Factor flota diferencial**: A350 estándar vs A350 next impactaron diferente a cada cabina

**📊 DIAGNÓSTICO LONG HAUL**: **Causa específica de cabina con segmentación geográfica** - Business LH y Premium LH reaccionaron de manera completamente opuesta a los mismos factores operativos, con Business beneficiándose de rutas América Sur exitosas mientras Premium se concentró en rutas problemáticas MAD-MEX.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

## **A. COMPARACIÓN ENTRE RADIOS**

**Nodos por radio identificados:**
- **Short Haul**: Negativo (-3.74 pts)
- **Long Haul**: Mixto (Business +25.62 pts, Premium -14.9 pts)

**Análisis de convergencia de drivers:**
- **SHORT HAUL**: 24 incidentes operativos, crisis CodeShare (BA -54.5, AA/VY -33.3), flota A332/A33ACMI problemática
- **LONG HAUL**: 13 incidentes operativos, segmentación geográfica extrema (América Sur +100.0 vs MAD-MEX 0.0), flota A350 diferencial

**Evidencia operativa:**
- **Volumen de incidentes diferentes**: SH (24) vs LH (13)
- **Naturaleza de problemas divergente**: 
  - SH: Crisis de socios CodeShare + flota específica
  - LH: Segmentación geográfica + configuración de aeronaves
- **Sin convergencia de drivers**: Causas operativas completamente distintas por radio

---

## **B. COHERENCIA CON NODO GLOBAL**

**Nodo Global**: Negativo (-1.77 pts, NPS 24.18 vs baseline 25.95)

**Análisis de agregación:**
- **SHORT HAUL (-3.74)** + **LONG HAUL (mixto)** = **GLOBAL (-1.77)**
- **Patrón coherente**: El Global refleja la dominancia del impacto negativo de SH, parcialmente compensado por el positivo de Business LH
- **Factores globales identificados**: 260 incidentes totales (24 SH + 13 LH + otros), huelga desconvocada en Grecia, problemas CodeShare generalizados
- **Amplificación selectiva**: El Global no atenúa sino que refleja la suma ponderada de impactos diferenciados por radio

**📊 DIAGNÓSTICO GLOBAL**: **Causas mixtas con dominancia Short Haul** - Las causas son específicas y diferentes por radio (CodeShare/flota en SH vs segmentación geográfica en LH), pero el impacto agregado Global (-1.77 pts) refleja coherentemente la dominancia del deterioro en Short Haul parcialmente compensado por segmentos positivos en Long Haul, sin efectos de cancelación completa.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

## **CAUSA 1: CRISIS DE CODESHARE CON SOCIOS ESTRATÉGICOS**

### A. NATURALEZA DE LA CAUSA:
**Hipótesis**: Problemas operativos severos en vuelos operados por socios de CodeShare (British Airways, Qatar Airways, American Airlines, Vueling) que impactaron desproporcionadamente la experiencia del cliente debido a diferencias en estándares de servicio, comunicación o gestión de disrupciones.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:
**Segmento más grande afectado**: Global/SH (Short Haul) - 24 incidentes operativos
- **NPS Impact**: -3.74 pts (de 28.31 a 24.57)
- **Alcance**: Afecta a todos los subsegmentos Economy SH/IB, Economy SH/YW
- **Evidencia específica por socio**:
  - **British Airways**: NPS -54.5 (11 encuestas) - Crisis severa
  - **Qatar Airways**: NPS -60.0 (5 encuestas) - Impacto crítico
  - **American Airlines/Vueling**: NPS -33.3 cada uno
- **Contraste con operación propia**: Iberia mantuvo NPS +27.6 (551 encuestas, 90% del volumen)
- **Métricas operativas**: 24 incidentes totales, 50% retrasos, 25% cancelaciones
- **Verbatims**: No disponibles

---

## **CAUSA 2: PROBLEMAS ESPECÍFICOS DE FLOTA A332/A33ACMI**

### A. NATURALEZA DE LA CAUSA:
**Hipótesis**: Problemas técnicos, de configuración o mantenimiento en aeronaves A332 y A33ACMI que generaron experiencias deficientes, posiblemente relacionados con confort, entretenimiento a bordo, o reliability operativa de estas aeronaves específicas.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:
**Segmento más grande afectado**: Global/SH/Economy - Flota A332/A33ACMI
- **NPS Impact**: A332 (-21.4, 14 encuestas), A33ACMI (0.0, 16 encuestas)
- **Alcance**: Afecta principalmente Economy SH/IB y Economy SH/YW
- **Contraste con otras flotas**: ATR (+55.0), A319 (+50.0) operaron normalmente
- **Correlación geográfica**: Impacto en América Norte (-20.0 NPS) sugiere rutas de largo radio
- **Rutas identificadas**: MAD-OPO (-11.1 NPS, 9 encuestas)
- **Métricas operativas**: Parte de los 24 incidentes operativos concentrados en estas aeronaves
- **Verbatims**: No disponibles

---

## **CAUSA 3: SEGMENTACIÓN GEOGRÁFICA EXTREMA EN LONG HAUL**

### A. NATURALEZA DE LA CAUSA:
**Hipótesis**: Diferencias operativas severas por rutas geográficas específicas, donde ciertas regiones (América Sur) recibieron servicio excepcional mientras otras (MAD-MEX, Asia, Oriente Medio) experimentaron problemas significativos, posiblemente por crew, handling local, o condiciones de aeropuerto.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:
**Segmento más grande afectado**: Global/LH (Long Haul) - 13 incidentes operativos
- **NPS Impact**: Mixto (Business +25.62, Premium -14.9)
- **Alcance**: Afecta a Business LH y Premium LH con patrones opuestos
- **Evidencia geográfica extrema**:
  - **América Sur**: NPS 100.0 (6 encuestas Business) - Perfección operativa
  - **Asia**: NPS -50.0 (4 encuestas) - Crisis severa
  - **Oriente Medio**: NPS -20.0 (5 encuestas)
  - **MAD-MEX**: NPS 0.0 (6 encuestas Premium) - Todos detractores
- **Factor flota correlacionado**: A350 estándar (40.0) vs A350 next (100.0)
- **Ruta específica**: EZE-MAD con NPS perfecto confirma patrón América Sur
- **Métricas operativas**: 13 incidentes LH, distribución no especificada por ruta
- **Verbatims**: No disponibles

---

## **CAUSA 4: GESTIÓN DIFERENCIAL DE CRISIS POR PERFIL DE CLIENTE**

### A. NATURALEZA DE LA CAUSA:
**Hipótesis**: Los mismos problemas operativos generaron respuestas completamente diferentes según el perfil del cliente, con clientes Business mostrando mayor tolerancia o recibiendo mejor gestión de crisis, mientras Economy fue más sensible a las disrupciones.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:
**Segmento más grande afectado**: Global - 260 incidentes totales
- **NPS Impact**: Global -1.77 pts (de 25.95 a 24.18)
- **Alcance**: Afecta diferencialmente a todas las cabinas y radios
- **Evidencia de reactividad diferencial**:
  - **Business SH/YW**: +13.77 pts (mejora paradójica con 24 incidentes)
  - **Economy SH**: -4.51 pts (deterioro consistente)
  - **Business LH**: +25.62 pts (beneficio de segmentación geográfica)
  - **Premium LH**: -14.9 pts (concentración en rutas problemáticas)
- **Situación especial**: Huelga desconvocada en Grecia (30 vuelos afectados, 14 cambios de equipo)
- **Métricas operativas globales**: 260 incidentes (79 retrasos, 22 cancelaciones, 24 pérdidas de conexión)
- **Verbatims**: No disponibles para ningún segmento

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El 27 de agosto de 2025 presentó un patrón de deterioro moderado del NPS global, cayendo 1.77 puntos hasta alcanzar 24.18 puntos respecto al baseline de 25.95 puntos. Las anomalías se concentraron principalmente en Short Haul con una caída de 3.74 puntos (NPS 24.57), mientras Long Haul mostró comportamiento mixto con Business subiendo significativamente 25.62 puntos (NPS 68.42) y Premium cayendo 14.9 puntos (NPS 12.5). La causa principal fue una crisis operativa diferenciada con 260 incidentes totales, donde los problemas de CodeShare impactaron severamente a socios como British Airways (-54.5 NPS), Qatar Airways (-60.0 NPS) y American Airlines/Vueling (-33.3 NPS cada uno), mientras Iberia mantuvo performance estable (+27.6 NPS). Adicionalmente, problemas específicos de flota A332/A33ACMI generaron NPS negativos (-21.4 y 0.0 respectivamente) contrastando con el excelente desempeño de ATR (+55.0 NPS). En Long Haul, la segmentación geográfica extrema creó experiencias polarizadas, con América Sur alcanzando perfección operativa (NPS 100.0) mientras Asia y Oriente Medio sufrieron deterioros severos (-50.0 y -20.0 NPS respectivamente).

Las rutas más afectadas incluyeron MAD-MEX (NPS 0.0 en Premium con todos los clientes como detractores), MAD-OPO (NPS -11.1), MAD-VIE (NPS -10.0) y LIS-MAD (NPS -25.0). Los grupos de clientes más reactivos fueron los viajeros Business/Work en Premium LH (-33.3 NPS vs +23.1 para Leisure), pasajeros de América Norte (-20.0 a -28.6 NPS según segmento), y clientes de vuelos CodeShare especialmente con British Airways y Qatar Airways, mientras que los pasajeros españoles y de América Sur mostraron mayor resiliencia a las disrupciones operativas.

**ECONOMY SH: Crisis de CodeShare y Flota Específica**
La cabina Economy de Short Haul experimentó un deterioro significativo el 27 de agosto de 2025, con Iberia registrando una caída moderada de 0.73 puntos (NPS 25.0) mientras Vueling sufrió un impacto severo de 11.37 puntos (NPS 20.71). La causa principal fue la crisis en vuelos de CodeShare, donde British Airways alcanzó un NPS catastrófico de -54.5 puntos (11 encuestas) para Iberia, y American Airlines/Vueling registraron -33.3 puntos cada uno para los clientes de Vueling, respaldada por 24 incidentes operativos (50% retrasos, 25% cancelaciones). Problemas específicos de flota A332 (-21.4 NPS) y A33ACMI (0.0 NPS) amplificaron el deterioro, contrastando con el excelente desempeño de ATR (+55.0 NPS). Este deterioro se reflejó especialmente en rutas como MAD-OPO (NPS -11.1) y AMS-MAD (NPS 25.0), mientras que los perfiles más reactivos incluyeron pasajeros de América Norte (-20.0 a -28.6 NPS) y clientes europeos (-20.0 NPS).

**BUSINESS SH YW: Mejora Paradójica Durante Crisis Operativa**
El segmento Business de Short Haul de Vueling mostró una mejora paradójica de 13.77 puntos el 27 de agosto, registrando un NPS de 30.77 puntos vs el baseline de 17.00 puntos. Esta evolución se explica principalmente por una gestión diferencial de crisis, donde los mismos 24 incidentes operativos que afectaron negativamente a Economy generaron una respuesta positiva en clientes Business, siendo especialmente visible entre pasajeros españoles (NPS 50.0) vs europeos (NPS 40.0) y en flota CRJ (NPS 30.8). Iberia Business Short Haul mantuvo desempeño estable sin desviaciones significativas detectadas.

**ECONOMY LH: Desempeño Estable**
La cabina Economy de Long Haul mantuvo desempeño estable el 27 de agosto de 2025. No se detectaron cambios significativos a nivel de cabina, manteniendo niveles consistentes de satisfacción sin generar anomalías detectables en el análisis.

**BUSINESS LH: Excelencia Operativa en América Sur**
La cabina Business de Long Haul experimentó una mejora excepcional de 25.62 puntos el 27 de agosto, registrando un NPS de 68.42 puntos vs el baseline de 42.80 puntos. Los drivers principales fueron la segmentación geográfica extrema con América Sur alcanzando perfección operativa (NPS 100.0 con 6 encuestas) y el excelente desempeño de flotas A332 y A350 next (ambas con NPS 100.0), impactando especialmente la ruta EZE-MAD y contrastando dramáticamente con España y América Centro (ambos con NPS 33.3). Los 13 incidentes operativos de Long Haul se concentraron en rutas específicas sin afectar los segmentos de alto valor.

**PREMIUM LH: Concentración en Rutas Problemáticas**
El segmento Premium de Long Haul sufrió un deterioro de 14.9 puntos el 27 de agosto, registrando un NPS de 12.5 puntos vs 27.4 puntos del baseline de 7 días. Las causas dominantes fueron la concentración total en la ruta MAD-MEX (NPS 0.0 con 6 encuestas, todos detractores) y el underperformance de flota A350 next (NPS -12.5) vs A350 estándar (NPS 37.5), especialmente evidentes entre clientes Business/Work (NPS -33.3) que mostraron mayor sensibilidad a los 13 incidentes operativos de Long Haul comparado con viajeros Leisure (NPS 23.1).

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 8
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-08-26 to 2025-08-26:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 9 (Global, Global/LH, Global/LH/Business, Global/LH/Economy, Global/LH/Premium, Global/SH/Business, Global/SH/Business/YW, Global/SH/Economy/IB, Global/SH/Economy/YW)

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 PASO 1: DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)

## 📊 ANÁLISIS SISTEMÁTICO POR CABINA SHORT HAUL

### **A. Economy Short Haul:**

**Nodos Identificados:**
- ✅ **Economy SH/IB**: Existe - Anomalía **POSITIVA** (+7.60 puntos)
- ✅ **Economy SH/YW**: Existe - Anomalía **NEGATIVA** (-2.67 puntos)

**Comparación de Patrones:**

| Aspecto | Economy SH/IB | Economy SH/YW |
|---------|---------------|---------------|
| **Dirección Anomalía** | Positiva (+7.60) | Negativa (-2.67) |
| **Driver Principal** | Crisis flota A332/A320 enmascarada | Variabilidad geográfica extrema |
| **Problema Crítico** | Codeshares LATAM/I2 (NPS -33.3/-20.0) | Codeshare VY (NPS -33.3) |
| **Impacto Geográfico** | Europa colapso (NPS 20.5) | América Sur crítico (NPS 0.0) |

**🎯 DIAGNÓSTICO ECONOMY SH:** **CAUSA ESPECÍFICA DE COMPAÑÍA** - IB y YW muestran patrones completamente divergentes con drivers, direcciones de anomalía y evidencia operativa diferentes.

---

### **B. Business Short Haul:**

**Nodos Identificados:**
- ✅ **Business SH/IB**: Existe - Anomalía **NEGATIVA** (-8.17 puntos)
- ✅ **Business SH/YW**: Existe - Anomalía **NEGATIVA** (-37.00 puntos) *[Análisis no viable por fecha futura]*

**Comparación de Patrones:**

| Aspecto | Business SH/IB | Business SH/YW |
|---------|---------------|---------------|
| **Dirección Anomalía** | Negativa (-8.17) | Negativa (-37.00) |
| **Driver Principal** | Polarización flota A320 vs CRJ | No determinable |
| **Evidencia Disponible** | Completa | No disponible |

**🎯 DIAGNÓSTICO BUSINESS SH:** **NO DETERMINABLE** - Aunque ambos nodos muestran anomalías negativas, la ausencia de datos válidos en YW impide realizar una comparación confiable de patrones y drivers.

---

## 📋 RESUMEN DIAGNÓSTICO COMPAÑÍA

- **Economy SH**: Causas específicas de cada compañía (IB/YW divergen completamente)
- **Business SH**: Diagnóstico inconcluso por limitaciones de datos en YW

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ PASO 2: DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)

## 📊 ANÁLISIS SISTEMÁTICO POR RADIO

### **A. Short Haul - Análisis de Cabinas:**

**Estado de Nodos por Cabina:**
- **Economy SH**: Patrón **MIXTO** (IB: +7.60 / YW: -2.67)
- **Business SH**: Patrón **NEGATIVO** (IB: -8.17 / YW: -37.00*)

**Análisis de Convergencia/Divergencia:**

| Cabina | IB | YW | Patrón |
|--------|----|----|---------|
| **Economy** | Positivo (+7.60) | Negativo (-2.67) | **DIVERGENTE** |
| **Business** | Negativo (-8.17) | Negativo (-37.00) | **CONVERGENTE** en dirección |

**Drivers Principales Comparados:**
- **Economy**: Causas completamente diferentes (flota A332 vs geográfico)
- **Business**: Ambos por polarización de flota (A320 vs CRJ en IB)

**🎯 DIAGNÓSTICO SHORT HAUL:** **CAUSA ESPECÍFICA DE CABINA** - Economy y Business muestran patrones divergentes. Economy presenta direcciones opuestas entre compañías, mientras Business converge en negatividad pero con diferentes magnitudes y drivers específicos de flota.

---

### **B. Long Haul - Análisis de Cabinas:**

**Estado de Nodos por Cabina:**
- **Economy LH**: Anomalía **NEGATIVA** (-10.65 puntos)
- **Business LH**: Anomalía **NEGATIVA** (-31.69 puntos)  
- **Premium LH**: Anomalía **POSITIVA** (+8.96 puntos)

**Análisis de Convergencia/Divergencia:**

| Cabina | Dirección | Magnitud | Driver Principal |
|--------|-----------|----------|------------------|
| **Economy** | Negativa | -10.65 | Crisis flota A321XLR |
| **Business** | Negativa | -31.69 | Polarización A350 vs A333 |
| **Premium** | Positiva | +8.96 | Codeshare AA problemático enmascarado |

**Progresión de Reactividad:**
- **Economy**: Impacto moderado por flota específica
- **Business**: **Mayor reactividad** - Amplificación del impacto (-31.69 vs -10.65)
- **Premium**: **Amortiguador aparente** - Mejora que enmascara problemas críticos

**Drivers Operativos Comunes:**
- **A321XLR problemática** afecta principalmente Economy/Business
- **Codeshares AA/BA** impactan negativamente Business/Premium
- **Polarización de flota** presente en las tres cabinas

**🎯 DIAGNÓSTICO LONG HAUL:** **REACTIVIDAD DIFERENCIAL CON CAUSA COMÚN** - Las tres cabinas comparten drivers operativos similares (flota A321XLR, codeshares problemáticos) pero muestran reactividad progresiva: Economy impacto base, Business amplificación crítica, Premium enmascaramiento estadístico que oculta problemas graves.

---

## 📋 RESUMEN DIAGNÓSTICO CABINA

- **Short Haul**: Causas específicas de cabina con patrones divergentes
- **Long Haul**: Causa operativa común con reactividad diferencial por tipo de cliente

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 PASO 3: DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

## 📊 ANÁLISIS SISTEMÁTICO DE ALCANCE

### **A. Comparación Entre Radios:**

**Estado de Nodos por Radio:**
- **Short Haul**: Patrón **MIXTO** (Economy +7.60/IB, Business -8.17/IB)
- **Long Haul**: Patrón **PREDOMINANTEMENTE NEGATIVO** (Economy -10.65, Business -31.69, Premium +8.96)

**Análisis de Afectación:**

| Radio | Estado General | Drivers Principales Identificados |
|-------|----------------|-----------------------------------|
| **Short Haul** | **AFECTADO PARCIAL** | • Flota CRJ problemática<br>• Codeshare VY crítico<br>• Variabilidad geográfica |
| **Long Haul** | **SEVERAMENTE AFECTADO** | • **Flota A321XLR crisis** (NPS -85.7)<br>• Codeshares AA/BA problemáticos<br>• Polarización extrema de flotas |

**Convergencia de Drivers:**
- **Divergentes**: SH centrado en codeshares específicos (VY) y geografía; LH dominado por crisis de flota A321XLR
- **Sin solapamiento significativo** en causas operativas principales
- **Magnitudes diferentes**: LH muestra impactos más severos

---

### **B. Coherencia con Nodo Global:**

**Comparación Global vs Radios:**

| Nivel | Dirección | Magnitud | Driver Dominante |
|-------|-----------|----------|------------------|
| **Global** | Negativa | -1.83 pts | **Flota A321XLR** (NPS -85.7) |
| **Long Haul** | Negativa | -10.65 a -31.69 | **Flota A321XLR** + Codeshares |
| **Short Haul** | Mixto | +7.60 a -8.17 | Codeshares + Geografía |

**Análisis de Coherencia:**
- ✅ **Coherente**: El Global (-1.83) refleja el **driver dominante de LH** (A321XLR -85.7)
- ✅ **Agregación lógica**: La severidad de LH se atenúa en Global por compensación parcial de algunos segmentos SH positivos
- ✅ **Causa principal identificada**: El Global correctamente identifica la **flota A321XLR como factor crítico**

**Patrón de Manifestación:**
- **Long Haul domina** la anomalía global por severidad de impacto
- **Short Haul aporta variabilidad** pero no determina la dirección principal
- **Efecto de compensación parcial** entre radios atenúa el impacto global total

---

## 🎯 DIAGNÓSTICO FINAL DE ALCANCE

**🌍 CAUSA MIXTA CON DOMINANCIA LONG HAUL** - Las causas operativas son específicas y diferentes por radio (A321XLR crítica en LH vs. codeshares/geografía en SH), pero el **impacto severo de Long Haul domina completamente la anomalía global**. El nodo Global refleja coherentemente esta dominancia, mostrando una agregación lógica donde la crisis de flota A321XLR (-85.7 NPS) determina la dirección y magnitud de la anomalía total (-1.83 pts), mientras que Short Haul aporta variabilidad compensatoria sin alterar el patrón principal.

---

## 📋 SÍNTESIS DE ALCANCE OPERATIVO

- **Alcance**: Causas mixtas diferenciadas por radio
- **Dominancia**: Long Haul determina el impacto global  
- **Coherencia**: Global refleja correctamente la agregación de patrones
- **Factor crítico**: Flota A321XLR como causa raíz principal

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 PASO 4: ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

## 🚨 CAUSA CRÍTICA #1: CRISIS OPERATIVA FLOTA A321XLR

### A. NATURALEZA DE LA CAUSA:
**Hipótesis:** Falla técnica grave o problema de configuración específico en aeronaves A321XLR que generó una experiencia extremadamente deficiente para los pasajeros, posiblemente relacionada con sistemas de cabina, entretenimiento, climatización o problemas de servicio específicos de esta flota nueva.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:

**Segmento más grande afectado:** Global/LH (155 encuestas)

**Output causal detallado:**
- **NPS A321XLR:** -85.7 (7 encuestas) - **CATASTRÓFICO**
- **Dispersión por flota:** 119.0 puntos (la más alta de todas las dimensiones)
- **Contraste operativo:** A350 next con NPS 33.3 (27 encuestas) - operación normal
- **Impacto en baseline:** Caída de -11.11 puntos en todo el segmento LH

**Subsegmentos afectados:** Esta causa afecta a **todos los subsegmentos bajo Global/LH**, incluyendo Economy LH (-10.65 pts), Business LH (-31.69 pts), y contribuye a la variabilidad en Premium LH.

**Métricas clave disponibles:**
- **Indicadores operativos:** No disponibles (limitación crítica del análisis)
- **NPS impactado:** -85.7 (vs. promedio LH de 10.32)
- **Rutas involucradas:** Posiblemente MAD-NRT (NPS -12.5) y MAD-SCL (NPS 7.7)
- **Verbatims:** No disponibles

---

## ⚠️ CAUSA CRÍTICA #2: COLAPSO ACUERDOS CODESHARE AA/BA

### A. NATURALEZA DE LA CAUSA:
**Hipótesis:** Problemas de coordinación operativa, estándares de servicio o experiencia de producto en vuelos operados bajo acuerdos de codeshare con American Airlines y British Airways, generando una experiencia significativamente inferior a los estándares de Iberia.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:

**Segmento más grande afectado:** Global/LH (155 encuestas)

**Output causal detallado:**
- **American Airlines:** NPS -41.7 (12 encuestas) vs. Iberia 16.8 (131 encuestas)
- **British Airways:** NPS -33.3 (6 encuestas) en Premium LH
- **Dispersión codeshare:** 113.9 puntos entre operadores
- **Impacto diferencial:** Clientes business norteamericanos más afectados (NPS -30.8)

**Subsegmentos afectados:** Esta causa afecta a **todos los subsegmentos bajo Global/LH** con particular severidad en Business LH y Premium LH.

**Métricas clave disponibles:**
- **Indicadores operativos:** No disponibles
- **NPS impactado:** AA -41.7, BA -33.3 vs. IB 16.8
- **Rutas involucradas:** No especificadas, posiblemente rutas transatlánticas
- **Verbatims:** No disponibles

---

## 🔴 CAUSA CRÍTICA #3: PROBLEMAS FLOTA A332 EN SHORT HAUL

### A. NATURALEZA DE LA CAUSA:
**Hipótesis:** Problemas técnicos o de configuración en flota A332 utilizada en operaciones Short Haul, posiblemente relacionada con aeronaves sobredimensionadas para rutas cortas o configuración inadecuada para este tipo de operación.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:

**Segmento más grande afectado:** Global/SH/Economy/IB (175 encuestas)

**Output causal detallado:**
- **A332 NPS:** 14.3 (7 encuestas) - 19.0 puntos por debajo del promedio
- **Contraste:** 32S con NPS 90.0 (10 encuestas) - excelencia operativa
- **Dispersión flota:** 75.7 puntos
- **Compensación estadística:** Problemas enmascarados por performance excepcional de otras flotas

**Subsegmentos afectados:** Esta causa afecta a **todos los subsegmentos bajo Global/SH/Economy/IB**.

**Métricas clave disponibles:**
- **Indicadores operativos:** No disponibles
- **NPS impactado:** 14.3 vs. promedio 33.33
- **Rutas involucradas:** No especificadas
- **Verbatims:** No disponibles

---

## 🟡 CAUSA CRÍTICA #4: CRISIS CODESHARE VUELING (VY)

### A. NATURALEZA DE LA CAUSA:
**Hipótesis:** Problemas específicos en la operación de vuelos bajo acuerdo codeshare con Vueling, posiblemente relacionados con diferencias en estándares de servicio, puntualidad o experiencia de producto entre las dos compañías.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:

**Segmento más grande afectado:** Global/SH/Economy/YW (85 encuestas)

**Output causal detallado:**
- **Vueling NPS:** -33.3 (3 encuestas) vs. Iberia 32.5 (80 encuestas)
- **Dispersión codeshare:** 65.8 puntos
- **Impacto geográfico:** Afecta particularmente a clientes europeos
- **Caída agregada:** -2.67 puntos en todo el segmento YW Economy

**Subsegmentos afectados:** Esta causa afecta a **todos los subsegmentos bajo Global/SH/Economy/YW**.

**Métricas clave disponibles:**
- **Indicadores operativos:** No disponibles
- **NPS impactado:** -33.3 vs. 32.5 (diferencia de 65.8 puntos)
- **Rutas involucradas:** No especificadas
- **Verbatims:** No disponibles

---

## 📊 SÍNTESIS DE CAUSAS CRÍTICAS

**Orden de severidad por impacto:**
1. **Flota A321XLR** (NPS -85.7) - Impacto catastrófico
2. **Codeshare AA/BA** (NPS -41.7/-33.3) - Impacto severo
3. **Codeshare VY** (NPS -33.3) - Impacto significativo  
4. **Flota A332** (NPS 14.3) - Impacto moderado enmascarado

**Limitación crítica:** La ausencia de datos operativos, incidentes NCS y verbatims impide identificar las causas técnicas específicas detrás de estos patrones de NPS.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El 26 de agosto de 2025 registró una caída generalizada del NPS de -1.83 puntos, pasando de 25.95 a 24.12, causada principalmente por una crisis operativa severa en la flota A321XLR que alcanzó un NPS catastrófico de -85.7. Esta problemática afectó especialmente al segmento Long Haul, donde Economy LH experimentó un deterioro de -10.65 puntos (de 18.59 a 7.94), Business LH sufrió la mayor caída con -31.69 puntos (de 42.80 a 11.11), mientras que Premium LH mostró una mejora aparente de +8.96 puntos (de 27.40 a 36.36) que enmascaró problemas críticos en codeshares con American Airlines (NPS -33.3). En Short Haul, los patrones fueron divergentes: Economy SH/IB registró una mejora de +7.60 puntos (de 25.73 a 33.33) que ocultó crisis en flota A332 (NPS 14.3) y codeshares LATAM/I2, Economy SH/YW experimentó una caída de -2.67 puntos (de 32.08 a 29.41) por problemas con Vueling (NPS -33.3), y Business SH/IB tuvo un deterioro de -8.17 puntos (de 32.41 a 24.24) por polarización entre flotas A320 versus CRJ.

Las rutas más afectadas incluyeron MAD-NRT con NPS -12.5, MAD-SCL con NPS 7.7, BUD-MAD con NPS 25.0, y EZE-MAD que mantuvo excelencia con NPS 100.0. Los grupos más reactivos fueron los clientes business norteamericanos (NPS -30.8), pasajeros de América Sur en ciertos segmentos (NPS 0.0), y usuarios de codeshares específicos, mientras que los clientes leisure y de mercados domésticos españoles mostraron mayor resistencia a las disrupciones operativas.

**ECONOMY SH IB: Crisis de Flota Enmascarada por Compensación Estadística**
La cabina Economy SH de Iberia experimentó una mejora aparente de +7.60 puntos durante el 26 de agosto, registrando un NPS de 33.33 con una variación positiva respecto al baseline de 25.73. Sin embargo, esta mejora enmascara problemas críticos en múltiples frentes operativos: la flota A332 registró un NPS preocupante de 14.3 (19.0 puntos por debajo del promedio), los codeshares LATAM e I2 colapsaron con NPS de -33.3 y -20.0 respectivamente, y los mercados europeos experimentaron una caída significativa con NPS de 20.5. Esta aparente mejora se debió únicamente a la compensación estadística proporcionada por el excepcional rendimiento de la flota 32S (NPS 90.0) y el sólido desempeño del mercado doméstico español (NPS 41.7), mientras que los perfiles más reactivos incluyeron pasajeros europeos y usuarios de codeshares internacionales.

**ECONOMY SH YW: Variabilidad Geográfica Extrema**
El segmento Economy SH de Vueling registró un NPS de 29.41 el 26 de agosto con una caída de -2.67 puntos respecto al baseline de 32.08. Esta evolución se explica principalmente por una variabilidad geográfica extrema donde América Sur colapsó completamente (NPS 0.0), Europa mostró rendimiento deficiente (NPS 10.0), mientras que Asia y América Centro-Norte mantuvieron niveles satisfactorios (NPS 50.0). Adicionalmente, el codeshare con Vueling registró un NPS crítico de -33.3, contrastando severamente con la operación Iberia que mantuvo NPS de 32.5, siendo especialmente visible entre perfiles de clientes europeos y sudamericanos que mostraron la mayor reactividad a estas disrupciones operativas.

**BUSINESS SH IB: Polarización Extrema Entre Flotas**
La cabina Business SH de Iberia registró un NPS de 24.24 el 26 de agosto con una caída de -8.17 puntos vs el baseline de 32.41. Esta evolución se explica principalmente por una polarización extrema entre tipos de aeronave, donde la flota A320 alcanzó la excelencia con NPS de 100.0 (4 encuestas) mientras que la flota CRJ experimentó problemas sistemáticos con NPS de -20.0 (10 encuestas), creando una dispersión sin precedentes de 120 puntos. Esta inconsistencia operativa fue especialmente visible en la ruta LHR-MAD que mantuvo excelencia (NPS 100.0) y entre perfiles de clientes europeos, mientras que los pasajeros business/work mostraron mayor reactividad a los problemas de la flota CRJ.

**ECONOMY LH: Crisis Multifactorial de Flota y Codeshares**
La cabina Economy de LH experimentó un deterioro significativo de -10.65 puntos durante el 26 de agosto, registrando un NPS de 7.94 respecto al baseline de 18.59. La causa principal fue la crisis catastrófica de la flota A321XLR con NPS de -85.7, complementada por el colapso de acuerdos codeshare con American Airlines (NPS -100.0) y British Airways (NPS -60.0), y problemas de coordinación que generaron una dispersión de 123.8 puntos entre flotas. Este deterioro se reflejó especialmente en el mercado español que concentró el mayor volumen de impacto (61 encuestas, NPS 3.3), mientras que los perfiles más reactivos incluyeron pasajeros leisure y clientes de mercados europeos que experimentaron directamente las deficiencias técnicas y operativas.

**BUSINESS LH: Amplificación Crítica por Polarización de Experiencias**
La cabina Business de LH sufrió el mayor deterioro con -31.69 puntos el 26 de agosto, registrando un NPS de 11.11 respecto al baseline de 42.80. Los drivers principales fueron la polarización extrema entre flotas donde A350 alcanzó perfección (NPS 100.0) en rutas como EZE-MAD, mientras que A333 experimentó problemas críticos (NPS -12.5), y el segmento business/work norteamericano colapsó completamente (NPS -50.0). Esta amplificación crítica fue especialmente evidente en las rutas transatlánticas y entre perfiles de clientes business norteamericanos que mostraron la mayor reactividad a las inconsistencias operativas y diferencias de producto entre flotas.

**PREMIUM LH: Mejora Aparente que Enmascara Crisis de Codeshare**
El segmento Premium de LH mostró una mejora aparente de +8.96 puntos el 26 de agosto, registrando un NPS de 36.36 vs el baseline de 27.40. Sin embargo, esta mejora enmascara un problema crítico en la operación codeshare con American Airlines que registró NPS de -33.3, contrastando dramáticamente con los vuelos Iberia que alcanzaron NPS de 62.5, creando una dispersión peligrosa de 95.8 puntos. Las causas dominantes fueron esta polarización extrema entre operadores y la concentración geográfica en España (100% de pasajeros), siendo especialmente evidente entre perfiles leisure que compensaron estadísticamente las experiencias extremadamente negativas de los usuarios de codeshare AA.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 9
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-08-25 to 2025-08-25:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 6 (Global, Global/LH/Business, Global/LH/Premium, Global/SH/Business/IB, Global/SH/Business/YW, Global/SH/Economy/YW)

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 **PASO 1: DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)**

## **A. Economy Short Haul:**

**ESTADO DE NODOS:**
- ✅ **Economy SH/YW existe**: Global/SH/Economy/YW (anomalía negativa -3.75 pts)
- ❌ **Economy SH/IB NO existe**: No hay nodo separado para IB en Economy

**DIAGNÓSTICO Economy SH**: **Causa específica de YW** - Solo YW presenta anomalía en Economy, sugiriendo un problema operativo exclusivo de esta compañía en el segmento económico.

---

## **B. Business Short Haul:**

**ESTADO DE NODOS:**
- ✅ **Business SH/IB existe**: Global/SH/Business/IB (anomalía positiva +8.93 pts)
- ✅ **Business SH/YW existe**: Global/SH/Business/YW (anomalía negativa crítica -17.0 pts)

**PATRONES DIVERGENTES IDENTIFICADOS:**

### **Drivers Principales:**
- **IB**: Impulsado por Business/Work (NPS 66.7) con problema localizado en flota A321
- **YW**: Crisis operativa severa con NPS catastrófico 0.0 y falla sistémica de datos

### **Evidencia Operativa:**
- **IB**: Operación normal (30 encuestas) sin incidentes NCS reportados
- **YW**: Operaciones mínimas (7 encuestas) con falla total de sistemas de registro

### **Magnitud del Contraste:**
- **Diferencia de 25.93 puntos** entre IB (+8.93) y YW (-17.0)

**DIAGNÓSTICO Business SH**: **Causas completamente específicas por compañía** - IB y YW muestran patrones operativos opuestos, con IB manteniendo operación estable mientras YW experimentó una crisis operativa catastrófica.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ **PASO 2: DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)**

## **A. Short Haul - Análisis de Cabinas:**

**ESTADO DE NODOS:**
- ✅ **Economy SH existe**: Global/SH/Economy/YW (anomalía negativa -3.75 pts)
- ✅ **Business SH existe**: Global/SH/Business/IB (+8.93 pts) y Global/SH/Business/YW (-17.0 pts)
- ❌ **Premium SH NO existe**: No hay nodo Premium en Short Haul

**PATRONES IDENTIFICADOS:**

### **Economy SH (solo YW):**
- **Driver**: Insatisfacción extrema de clientes norteamericanos (NPS -66.7)
- **Característica**: Problema "invisible" sin incidentes NCS reportados
- **Volumen**: 120 encuestas (operación robusta)

### **Business SH:**
- **IB**: Operación estable con problema localizado en flota A321
- **YW**: Crisis operativa severa con falla sistémica total

**DIAGNÓSTICO Short Haul**: **Causas completamente específicas por cabina y compañía** - Economy YW muestra un problema de experiencia específico del perfil de cliente, mientras Business presenta patrones operativos opuestos entre IB (estable) y YW (crisis), sin convergencia alguna entre cabinas.

---

## **B. Long Haul - Análisis de Cabinas:**

**ESTADO DE NODOS:**
- ❌ **Economy LH NO existe**: No hay nodo Economy en Long Haul
- ✅ **Business LH existe**: Global/LH/Business (anomalía positiva +9.58 pts)
- ✅ **Premium LH existe**: Global/LH/Premium (anomalía positiva +15.45 pts)

**PATRONES CONVERGENTES IDENTIFICADOS:**

### **Business LH:**
- **Driver**: Concentración de clientes de regiones con alta satisfacción (AMERICA SUR NPS 100.0)
- **Factor**: Composición geográfica favorable (28.6% con NPS perfecto)
- **Volumen**: 21 encuestas

### **Premium LH:**
- **Driver**: Diferencias significativas por perfil de cliente (CodeShare IB NPS 50.0 vs AMERICA SUR NPS 0.0)
- **Factor**: Variación extrema de 50 puntos entre segmentos
- **Volumen**: Similar distribución por segmentos

**PROGRESIÓN LÓGICA:**
- **Premium (+15.45 pts) > Business (+9.58 pts)**: Mayor reactividad en cabina premium
- **Patrón común**: Ambas impulsadas por perfiles geográficos específicos
- **Diferencia**: Premium muestra mayor dispersión entre segmentos

**DIAGNÓSTICO Long Haul**: **Causa común al radio con reactividad diferencial por cabina** - Ambas cabinas LH muestran el mismo driver (perfiles geográficos favorables), pero Premium actúa con mayor sensibilidad (+15.45 vs +9.58 pts), sugiriendo que clientes premium son más reactivos a factores experienciales positivos.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 **PASO 3: DIAGNÓSTICO A NIVEL RADIO Y GLOBAL**

## **A. Comparación Entre Radios:**

**ESTADO DE NODOS:**
- ✅ **Short Haul afectado**: Múltiples anomalías (-17.0, +8.93, -3.75 pts)
- ✅ **Long Haul afectado**: Múltiples anomalías (+15.45, +9.58 pts)
- ✅ **Global afectado**: Anomalía positiva (+3.30 pts, NPS 29.25)

**PATRONES POR RADIO:**

### **Short Haul:**
- **Drivers**: Crisis operativa YW vs operación estable IB + problemas específicos de perfil norteamericano
- **Rango**: Desde crisis catastrófica (-17.0 pts) hasta mejora moderada (+8.93 pts)
- **Evidencia**: Fallas operativas severas, problemas de flota A321, insatisfacción geográfica

### **Long Haul:**
- **Drivers**: Perfiles geográficos favorables (AMERICA SUR NPS 100.0, CodeShare IB NPS 50.0)
- **Rango**: Anomalías consistentemente positivas (+9.58 a +15.45 pts)
- **Evidencia**: Concentración de clientes satisfechos, composición favorable

**DIVERGENCIA TOTAL**: Los radios muestran **drivers completamente diferentes** - SH con problemas operativos severos vs LH con composición de clientes favorable.

---

## **B. Coherencia con Nodo Global:**

**ANÁLISIS DE COHERENCIA:**

### **Matemática del Impacto:**
- **SH**: Rango extremo (-17.0 a +8.93 pts) con efectos compensatorios
- **LH**: Consistentemente positivo (+9.58 a +15.45 pts)
- **Global**: +3.30 pts (moderadamente positivo)

### **Patrón de Compensación:**
- **LH impulsa hacia arriba**: Ambas cabinas con anomalías positivas significativas
- **SH genera volatilidad**: Crisis YW (-17.0) parcialmente compensada por mejora IB (+8.93)
- **Resultado Global**: Las mejoras LH dominan sobre los problemas mixtos SH

### **Coherencia de Drivers:**
- **Global identifica**: Flota A350 C con NPS -71.4 (problema crítico no visible en análisis por radio)
- **Global detecta**: Problemas de codeshare (AA, QR, LATAM) no capturados en análisis específicos
- **Paradoja**: Global muestra "anomalía positiva que enmascara problemas operativos críticos"

**DIAGNÓSTICO Global**: **Causas mixtas con efectos compensatorios** - Long Haul impulsa la anomalía positiva global (+3.30 pts) con perfiles geográficos favorables, mientras Short Haul genera volatilidad extrema con crisis operativas que se compensan parcialmente, resultando en una mejora global que enmascara problemas operativos críticos localizados (A350 C, codeshares internacionales).

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 **PASO 4: ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS**

## **CAUSA 1: CRISIS OPERATIVA SEVERA YW**

### A. NATURALEZA DE LA CAUSA:
**Hipótesis**: Falla catastrófica de infraestructura o sistemas críticos que redujo las operaciones YW a niveles mínimos, generando experiencias extremadamente negativas para los pocos clientes que lograron volar, mientras comprometía simultáneamente los sistemas de registro operativo.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:
**Segmento más grande afectado**: Global/SH/Business/YW
- **Anomalía**: -17.0 puntos (NPS 0.0 vs baseline 17.0)
- **Alcance**: Afecta a todos los subsegmentos Business Short Haul YW
- **Evidencia operativa**: 
  - Volumen reducido a solo 7 encuestas vs operaciones normales
  - NPS catastrófico 0.0 en Business/Work, Leisure y flota CRJ
  - Ausencia total de datos operativos, incidentes NCS y verbatims
  - Falla sistémica de registro mientras España/IB mantuvo NPS 16.7
- **Métricas clave**: NPS 0.0, 7 encuestas totales, falla completa de sistemas de monitoreo

---

## **CAUSA 2: PERFILES GEOGRÁFICOS FAVORABLES LH**

### A. NATURALEZA DE LA CAUSA:
**Hipótesis**: Concentración excepcional de clientes de regiones con alta propensión a satisfacción (especialmente AMERICA SUR) que elevó significativamente el NPS por composición favorable de la muestra, no por mejoras operativas específicas.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:
**Segmento más grande afectado**: Global/LH/Premium
- **Anomalía**: +15.45 puntos (NPS 42.86 vs baseline 27.40)
- **Alcance**: Afecta a todos los subsegmentos Premium Long Haul
- **Evidencia operativa**:
  - AMERICA SUR: NPS 100.0 vs NPS 0.0 (dispersión de 100 puntos)
  - CodeShare IB: NPS 50.0 (6 encuestas) - excelente performance
  - 28.6% de clientes con NPS perfecto en Business LH
  - Concentración de clientes de regiones sin especificar: NPS 100.0
- **Métricas clave**: Dispersión extrema 50-100 puntos entre regiones, ausencia de incidentes NCS

---

## **CAUSA 3: PROBLEMA ESPECÍFICO CLIENTES NORTEAMERICANOS**

### A. NATURALEZA DE LA CAUSA:
**Hipótesis**: Factores específicos de experiencia, expectativas culturales no cumplidas o problemas de comunicación/servicio que afectaron únicamente a clientes norteamericanos, generando insatisfacción extrema no detectada por sistemas operativos tradicionales.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:
**Segmento más grande afectado**: Global/SH/Economy/YW
- **Anomalía**: -3.75 puntos vs baseline
- **Alcance**: Afecta específicamente a clientes norteamericanos en Economy Short Haul YW
- **Evidencia operativa**:
  - Clientes América Norte: NPS -66.7 (6 encuestas) - probablemente todos detractores
  - Contraste extremo vs España: +49.3 (75 encuestas) = 116.7 puntos de diferencia
  - Problema "invisible": Sin incidentes NCS reportados
  - Volumen robusto total: 120 encuestas, ruta específica GRX-MAD identificada
- **Métricas clave**: Dispersión 116.7 puntos, 6 clientes generaron impacto desproporcionado

---

## **CAUSA 4: FALLA CRÍTICA FLOTA A321**

### A. NATURALEZA DE LA CAUSA:
**Hipótesis**: Problema técnico, de mantenimiento o configuración específico de la flota A321 que generó experiencias pésimas para todos los pasajeros de esta aeronave, creando un factor limitante significativo en el rendimiento general.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:
**Segmento más grande afectado**: Global/SH/Business/IB
- **Anomalía**: Limitó mejora potencial (+8.93 pts vs potencial mayor)
- **Alcance**: Afecta a todos los subsegmentos que operaron en flota A321
- **Evidencia operativa**:
  - Flota A321: NPS 0.0 (7 encuestas) - falla operativa grave
  - Dispersión extrema: 61.5 puntos entre mejor y peor flota
  - 23% de clientes Business IB (7/30) con experiencia catastrófica
  - Contraste vs otras flotas con performance normal
- **Métricas clave**: NPS 0.0, 7 clientes afectados, dispersión 61.5 puntos entre flotas

---

## **CAUSA 5: PROBLEMAS CODESHARE INTERNACIONALES**

### A. NATURALEZA DE LA CAUSA:
**Hipótesis**: Deficiencias sistemáticas en la experiencia de vuelos operados por partners internacionales (American Airlines, Qatar Airways, LATAM), posiblemente relacionadas con estándares de servicio, coordinación operativa o expectativas no alineadas.

### B. EVIDENCIA CONSOLIDADA Y ALCANCE:
**Segmento más grande afectado**: Global (nivel más amplio detectado)
- **Anomalía**: Factor limitante de la mejora global (+3.30 pts)
- **Alcance**: Afecta a todos los subsegmentos que utilizan codeshare internacional
- **Evidencia operativa**:
  - American Airlines (AA): NPS -29.4 (17 encuestas)
  - Qatar Airways (QR): NPS -25.0
  - LATAM: NPS -14.3
  - Patrón sistemático: Partners internacionales con experiencias negativas
- **Métricas clave**: Múltiples partners con NPS negativos, 17 encuestas AA como mayor volumen

---

## 📋 SÍNTESIS EJECUTIVA FINAL

# 🎯 **SÍNTESIS EJECUTIVA FINAL - ANÁLISIS NPS**

📈 **SÍNTESIS EJECUTIVA:**

El 25 de agosto de 2025 se registró una mejora global del NPS de 3.30 puntos (alcanzando 29.25), impulsada principalmente por el excelente desempeño de Long Haul donde Business experimentó una subida de 9.58 puntos (de 42.80 a 52.38) y Premium una mejora aún mayor de 15.45 puntos (de 27.40 a 42.86), ambas explicadas por una concentración excepcional de clientes de AMERICA SUR con NPS perfecto de 100.0 y CodeShare IB con NPS 50.0. Sin embargo, esta mejora general enmascaró una crisis operativa severa en Short Haul, donde Business YW colapsó 17.0 puntos hasta un NPS catastrófico de 0.0 debido a fallas sistémicas que redujeron las operaciones a solo 7 encuestas, mientras que Business IB logró una mejora de 8.93 puntos (alcanzando NPS 46.67) impulsada por el segmento Business/Work con NPS 66.7, aunque limitada por problemas críticos en la flota A321 que registró NPS 0.0. Adicionalmente, Economy YW mostró una caída de 3.75 puntos causada por la insatisfacción extrema de clientes norteamericanos con NPS -66.7, contrastando 116.7 puntos con clientes españoles.

Las rutas más afectadas incluyen EZE-MAD que destacó positivamente con NPS 85.7 en el segmento Business LH, y GRX-MAD que mostró performance moderado con NPS 33.3 en Economy SH, mientras que la ausencia sistemática de datos operativos impidió identificar rutas específicas en la crisis YW. Los grupos de clientes más reactivos fueron los residentes de AMERICA SUR que alcanzaron NPS perfecto de 100.0 en ambas cabinas Long Haul, los viajeros Business/Work con NPS 66.7 en Short Haul IB, y por el lado negativo, los clientes norteamericanos en Economy YW que registraron el NPS más bajo de -66.7, evidenciando una alta sensibilidad geográfica y por propósito de viaje en la experiencia del cliente.

**ECONOMY SH YW: Crisis de Satisfacción Geográfica**
La cabina Economy de Short Haul YW experimentó un deterioro de 3.75 puntos el 25 de agosto de 2025, impulsado por la insatisfacción extrema de clientes norteamericanos que registraron un NPS catastrófico de -66.7 basado en 6 encuestas, contrastando dramáticamente con los 49.3 puntos de NPS de clientes españoles en una base de 75 encuestas, creando una dispersión de 116.7 puntos entre perfiles geográficos. Este deterioro se caracterizó por ser un problema "invisible" sin incidentes NCS reportados en los sistemas operativos, sugiriendo factores específicos de experiencia cultural o expectativas no cumplidas que afectaron únicamente a este perfil de cliente, mientras que la ruta GRX-MAD mantuvo un desempeño moderado con NPS 33.3.

**BUSINESS SH IB: Mejora Limitada por Problemas de Flota**
El segmento Business de Short Haul IB registró una mejora de 8.93 puntos alcanzando un NPS de 46.67 el 25 de agosto, impulsado principalmente por el excelente desempeño del segmento Business/Work con NPS 66.7 basado en 6 encuestas y clientes españoles con NPS 58.3 versus europeos con 25.0 puntos. Sin embargo, esta mejora fue significativamente limitada por una falla crítica en la flota A321 que registró NPS 0.0 afectando a 7 clientes (23% del total de 30 encuestas), creando una dispersión extrema de 61.5 puntos entre flotas y evidenciando problemas operativos severos localizados que impidieron una mejora más sustancial.

**BUSINESS SH YW: Colapso Operativo Catastrófico**
La cabina Business de Short Haul YW experimentó un colapso total de 17.0 puntos hasta alcanzar un NPS catastrófico de 0.0 el 25 de agosto, causado por una crisis operativa severa que redujo las operaciones a niveles mínimos de solo 7 encuestas versus volúmenes normales. Esta crisis se caracterizó por la falla sistémica completa de sistemas de registro operativo, ausencia total de datos de puntualidad, incidentes NCS y verbatims, mientras que todos los segmentos afectados (Business/Work, Leisure y flota CRJ) registraron NPS 0.0, sugiriendo una falla de infraestructura crítica que comprometió tanto la operación como los sistemas de monitoreo, con la única excepción de España/IB que mantuvo NPS 16.7.

**ECONOMY LH: Desempeño Estable**
La cabina Economy de Long Haul mantuvo desempeño estable durante el 25 de agosto de 2025. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción.

**BUSINESS LH: Excelencia Impulsada por Perfiles Geográficos**
La cabina Business de Long Haul experimentó una mejora significativa de 9.58 puntos alcanzando un NPS de 52.38 el 25 de agosto, impulsada por una concentración excepcional de clientes de AMERICA SUR que registraron NPS perfecto de 100.0 junto con regiones sin especificar también con 100.0 puntos, representando el 28.6% del total de 21 encuestas. Esta mejora se explica por una composición geográfica favorable donde clientes de estas regiones específicas elevaron significativamente el promedio general, contrastando con AMERICA CENTRO que mostró NPS -25.0, mientras que la ruta EZE-MAD destacó con NPS 85.7 y la flota A350 mantuvo consistencia operativa con NPS 70.0.

**PREMIUM LH: Máxima Reactividad por Segmentación**
El segmento Premium de Long Haul registró la mayor mejora con 15.45 puntos de subida alcanzando un NPS de 42.86 el 25 de agosto, impulsado por diferencias extremas en el rendimiento por perfil de cliente donde CodeShare IB alcanzó NPS 50.0 basado en 6 encuestas, contrastando dramáticamente con AMERICA SUR que registró NPS 0.0 en 4 encuestas, creando una dispersión de 50 puntos entre segmentos. Esta alta reactividad evidencia que los clientes premium son más sensibles a factores experienciales, donde segmentos como Leisure mantuvieron NPS moderado de 33.3 y la flota A350 registró 25.0 puntos, demostrando que pequeños cambios en la composición de perfiles pueden generar impactos significativos en esta cabina de alta gama.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 6
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-08-24 to 2025-08-24:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 10 (Global, Global/LH, Global/LH/Business, Global/LH/Economy, Global/LH/Premium, Global/SH/Business, Global/SH/Business/IB, Global/SH/Business/YW, Global/SH/Economy/IB, Global/SH/Economy/YW)

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)

## **A. ECONOMY SHORT HAUL**

### **Nodos Identificados:**
- ✅ **Economy SH/IB**: Nodo anómalo **NEGATIVO** (-4.33 pts)
- ✅ **Economy SH/YW**: Nodo anómalo **POSITIVO** (+15.92 pts)

### **Análisis Comparativo:**

**Economy SH/IB:**
- **Driver Principal**: CodeShare LATAM crítico (NPS -75.0)
- **Problema Específico**: Falla en operaciones externas de codeshare
- **Evidencia**: 285 encuestas, problema localizado en partner específico

**Economy SH/YW:**
- **Driver Principal**: Composición regional favorable (América Centro NPS 66.7)
- **Problema Específico**: Mix de pasajeros excepcional + mayor proporción Leisure
- **Evidencia**: 125 encuestas, experiencia positiva por demografía

### **Diagnóstico Economy SH:**
**CAUSA ESPECÍFICA DE COMPAÑÍA** - IB y YW muestran drivers completamente divergentes: IB sufre por problemas de codeshare externo mientras YW se beneficia de composición demográfica favorable.

---

## **B. BUSINESS SHORT HAUL**

### **Nodos Identificados:**
- ✅ **Business SH/IB**: Nodo anómalo **POSITIVO** (+8.42 pts)
- ✅ **Business SH/YW**: Nodo anómalo **POSITIVO** (+10.27 pts)

### **Análisis Comparativo:**

**Business SH/IB:**
- **Driver Principal**: Polarización extrema por flota (A320 NPS 100.0 vs A350 C NPS -75.0)
- **Evidencia**: 39 encuestas, experiencias bimodales por tipo de aeronave

**Business SH/YW:**
- **Driver Principal**: **DATOS NO DISPONIBLES** (fecha futura 2025-08-24)
- **Evidencia**: Análisis imposible por ausencia total de datos

### **Diagnóstico Business SH:**
**ANÁLISIS INCONCLUSO** - Imposible determinar si la causa es específica de compañía o común debido a la ausencia total de datos para Business SH/YW (fecha futura).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)

## **A. SHORT HAUL - ANÁLISIS DE CABINAS**

### **Nodos Identificados:**
- **Economy SH/IB**: Nodo **NEGATIVO** (-4.33 pts)
- **Economy SH/YW**: Nodo **POSITIVO** (+15.92 pts)
- **Business SH/IB**: Nodo **POSITIVO** (+8.42 pts)
- **Business SH/YW**: Nodo **POSITIVO** (+10.27 pts)*

*Datos no disponibles por fecha futura

### **Análisis de Convergencia/Divergencia:**

**Patrón por Compañía:**
- **IB**: Economy negativo vs Business positivo = **DIVERGENTE**
- **YW**: Economy positivo vs Business positivo = **CONVERGENTE** (ambos positivos)

**Causas Operativas Identificadas:**
- **Economy**: Problemas específicos de codeshare (IB) vs composición demográfica favorable (YW)
- **Business**: Polarización por flota (IB) vs datos no disponibles (YW)

### **Diagnóstico Short Haul:**
**CAUSA ESPECÍFICA DE CABINA** - Las cabinas Economy y Business muestran patrones completamente divergentes dentro de IB, mientras que en YW ambas son positivas, indicando que cada cabina reacciona de forma diferenciada a factores operativos distintos.

---

## **B. LONG HAUL - ANÁLISIS DE CABINAS**

### **Nodos Identificados:**
- **Economy LH**: Nodo **NEGATIVO** (-0.88 pts)
- **Business LH**: Nodo **NEGATIVO** (-22.8 pts)
- **Premium LH**: Nodo **NEGATIVO** (-21.85 pts)

### **Análisis de Convergencia:**

**Patrón Común:**
- **Todas las cabinas LH**: Anomalías **NEGATIVAS**
- **Causa Raíz Compartida**: Problemas críticos de flota A33ACMI (NPS -76.9)

**Reactividad Diferencial:**
- **Economy LH**: Menor impacto (-0.88 pts) - Mayor tolerancia
- **Business LH**: Alto impacto (-22.8 pts) - Mayor sensibilidad
- **Premium LH**: Alto impacto (-21.85 pts) - Mayor sensibilidad

### **Diagnóstico Long Haul:**
**CAUSA COMÚN DE RADIO** - Todas las cabinas LH convergen en la misma causa operativa (problemas flota A33ACMI), pero con reactividad diferencial donde Economy actúa como "amortiguador" mientras Business y Premium muestran mayor sensibilidad al mismo problema operativo.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

## **A. COMPARACIÓN ENTRE RADIOS**

### **Estado de Anomalías por Radio:**
- **Short Haul**: **MIXTO** (Economy IB negativo, resto positivos)
- **Long Haul**: **NEGATIVO COMPLETO** (todas las cabinas negativas)

### **Análisis de Drivers:**

**Short Haul:**
- **Causas Diversas**: CodeShare LATAM (IB), composición demográfica (YW), polarización de flota
- **Patrón**: Problemas específicos y localizados por compañía/segmento

**Long Haul:**
- **Causa Convergente**: Flota A33ACMI crítica (NPS -76.9) impacta todas las cabinas
- **Patrón**: Problema operativo sistémico y homogéneo

### **Diagnóstico Entre Radios:**
**CAUSAS ESPECÍFICAS DE RADIO** - Short Haul presenta problemas heterogéneos y localizados mientras Long Haul converge en una causa operativa común (flota A33ACMI), sin evidencia de drivers compartidos entre radios.

---

## **B. COHERENCIA CON NODO GLOBAL**

### **Estado del Nodo Global:**
- **Global**: **NORMAL** (NPS 25.94 vs baseline 25.95, diferencia -0.012 pts)

### **Análisis de Coherencia:**

**Patrón Identificado:**
- **Global estable** (-0.012 pts) **ENMASCARA** anomalías severas específicas
- **Efectos Compensatorios**: Anomalías positivas SH (+15.92 YW Economy) compensan anomalías negativas LH (-22.8 Business, -21.85 Premium)

**Evidencia de Cancelación:**
- SH positivos + LH negativos = Global aparentemente normal
- **717 encuestas totales** diluyen impactos concentrados
- **Ruta crítica DFW-MAD** (NPS -28.6) oculta en promedio global

### **Diagnóstico Global:**
**CAUSAS MIXTAS/COMPENSATORIAS** - El nodo Global muestra normalidad aparente que oculta problemas operacionales graves concentrados en segmentos específicos, donde anomalías positivas de Short Haul compensan matemáticamente las anomalías negativas críticas de Long Haul, creando una falsa sensación de estabilidad operativa.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

## **CAUSA #1: PROBLEMAS CRÍTICOS FLOTA A33ACMI**

### **A. NATURALEZA DE LA CAUSA:**
**Hipótesis:** Fallas operativas severas en aeronaves A330 operadas bajo modalidad ACMI (Aircraft, Crew, Maintenance & Insurance) que generan experiencias de cliente extremadamente negativas debido a problemas de configuración, mantenimiento deficiente, o estándares de servicio inferiores comparados con flota propia.

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**

**Segmento Más Grande Afectado:** Global/LH (257 encuestas)

**Output Causal Detallado:**
- **NPS A33ACMI:** -76.9 puntos (13 encuestas)
- **Contraste con flota propia:** A350 next alcanzó +51.3 puntos (diferencial de 128.2 puntos)
- **Dispersión Fleet:** 92.2 puntos de spread - mayor variabilidad del día

**Subsegmentos Afectados:**
- Global/LH/Economy: NPS A33ACMI -76.9
- Global/LH/Business: Impacto confirmado en análisis de flota
- Global/LH/Premium: Performance crítica vs otras flotas

**Métricas Clave:**
- **Volumen total LH:** 257 encuestas procesadas
- **Reactividad promedio:** 77.4 puntos - alta sensibilidad del cliente
- **Rutas identificadas:** Problemas concentrados en operaciones transatlánticas
- **Incidentes NCS:** 0 reportados formalmente (gap en detección)

---

## **CAUSA #2: FALLA CODESHARE LATAM**

### **A. NATURALEZA DE LA CAUSA:**
**Hipótesis:** Deterioro crítico en estándares de servicio o procesos operativos en vuelos operados bajo acuerdo de codeshare con LATAM, posiblemente debido a desalineación de protocolos, entrenamiento deficiente del personal, o problemas en la coordinación operativa entre aerolíneas.

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**

**Segmento Más Grande Afectado:** Global/SH/Economy/IB (285 encuestas)

**Output Causal Detallado:**
- **NPS CodeShare LATAM:** -75.0 puntos (8 encuestas)
- **Contraste con operaciones propias:** IB directo +24.9 puntos
- **Dispersión CodeShare:** 108.3 puntos - mayor variabilidad por partner

**Subsegmentos Afectados:**
- Global/SH/Economy/IB: Principal impacto identificado
- Operaciones que involucran acuerdos LATAM en Short Haul

**Métricas Clave:**
- **Volumen SH/Economy/IB:** 285 encuestas totales
- **Ruta específica:** GRU-MAD identificada con problemas
- **Performance baseline IB:** +24.9 NPS en operaciones directas
- **Incidentes NCS:** Sin registros formales del problema

---

## **CAUSA #3: COMPOSICIÓN DEMOGRÁFICA EXCEPCIONAL**

### **A. NATURALEZA DE LA CAUSA:**
**Hipótesis:** Variación favorable inusual en el mix de pasajeros con alta concentración de segmentos demográficos que históricamente muestran mayor satisfacción (América Centro, viajeros Leisure), creando un sesgo positivo en las evaluaciones sin mejoras operativas subyacentes.

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**

**Segmento Más Grande Afectado:** Global/SH/Economy/YW (125 encuestas)

**Output Causal Detallado:**
- **América Centro:** NPS +66.7 puntos (impacto positivo extremo)
- **Viajeros Leisure:** NPS +53.7 vs Business +11.8 (diferencial 41.9 puntos)
- **Flota CRJ favorable:** NPS +64.1 vs ATR +32.3

**Subsegmentos Afectados:**
- Global/SH/Economy/YW: Impacto principal
- Todos los subsegmentos de YW Economy beneficiados

**Métricas Clave:**
- **Volumen total:** 125 encuestas
- **Codeshare VY contribuyente:** NPS +83.3 (6 encuestas)
- **Ruta identificada:** ALC-MAD con performance positiva
- **Sin incidentes operativos:** Mejora puramente demográfica

---

## **CAUSA #4: POLARIZACIÓN EXTREMA POR FLOTA (SHORT HAUL)**

### **A. NATURALEZA DE LA CAUSA:**
**Hipótesis:** Inconsistencia severa en estándares de experiencia entre diferentes tipos de aeronaves en Short Haul, donde ciertas flotas (A320) entregan experiencias excepcionales mientras otras (A350 C) generan experiencias críticas, sugiriendo problemas de configuración, mantenimiento o protocolos de servicio diferenciados.

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**

**Segmento Más Grande Afectado:** Global/SH/Business/IB (39 encuestas)

**Output Causal Detallado:**
- **Dispersión extrema:** 175.0 puntos entre A320 (NPS 100.0) y A350 C (NPS -75.0)
- **Experiencias bimodales:** Segmentos con 100% promotores vs 100% detractores
- **Variabilidad geográfica:** España NPS +69.2 vs Asia NPS -100.0

**Subsegmentos Afectados:**
- Global/SH/Business/IB: Principal evidencia
- Todos los subsegmentos de Business SH/IB impactados

**Métricas Clave:**
- **Volumen Business SH/IB:** 39 encuestas
- **Ruta específica:** MAD-OPO con NPS 100.0
- **Codeshare diferencial:** AA NPS 100.0 vs BA NPS -50.0
- **Sin incidentes NCS:** Problemas de consistencia operativa

---

## 📋 SÍNTESIS EJECUTIVA FINAL

# 📈 SÍNTESIS EJECUTIVA

El día 24 de agosto de 2025 presentó una situación paradójica donde el NPS global se mantuvo estable en 25.94 puntos (diferencia mínima de -0.012 puntos vs baseline), ocultando anomalías severas concentradas en segmentos específicos. Long Haul experimentó deterioros significativos con Economy cayendo 0.88 puntos hasta 17.70, Business desplomándose 22.8 puntos hasta 20.0, y Premium descendiendo 21.85 puntos hasta 5.56. La causa principal fue la performance crítica de la flota A33ACMI que registró un NPS catastrófico de -76.9 puntos, contrastando dramáticamente con los 51.3 puntos de la flota A350 next. En Short Haul, Economy IB cayó 4.33 puntos hasta 21.40 debido a problemas severos en el codeshare LATAM (NPS -75.0), mientras que Economy YW se disparó 15.92 puntos hasta 48.0 por una composición demográfica excepcional con alta concentración de pasajeros de América Centro (NPS 66.7). Business SH mostró mejoras con IB subiendo 8.42 puntos hasta 46.15 por polarización extrema entre flotas (A320 con NPS 100.0 vs A350 C con NPS -75.0), aunque YW Business no pudo analizarse por tratarse de una fecha futura.

Las rutas más críticas fueron DFW-MAD con un NPS devastador de -28.6 puntos y BOG-MAD que concentró toda la actividad Business con experiencias altamente polarizadas. Los grupos más reactivos incluyeron pasajeros de América del Norte en Long Haul (NPS más bajo de +2.7), clientes asiáticos en Business SH (NPS -100.0), y viajeros corporativos que mostraron mayor sensibilidad a los problemas operativos que los pasajeros de ocio.

**ECONOMY SH: Divergencia Extrema entre Compañías**
La cabina Economy de Short Haul mostró una polarización sin precedentes el 24 de agosto de 2025, con IB registrando una caída de 4.33 puntos hasta un NPS de 21.40 mientras YW experimentó una mejora excepcional de 15.92 puntos alcanzando 48.0. La causa del deterioro en IB fue exclusivamente el colapso del codeshare LATAM que registró un NPS catastrófico de -75.0 puntos en 8 encuestas, contrastando con las operaciones directas de IB que mantuvieron un NPS positivo de 24.9. Por el contrario, YW se benefició de una composición demográfica excepcional con alta concentración de pasajeros de América Centro (NPS 66.7) y mayor proporción de viajeros de ocio (NPS 53.7 vs 11.8 de business), complementado por la excelente performance del codeshare VY que alcanzó 83.3 puntos. Los perfiles más reactivos incluyeron pasajeros europeos en IB (NPS -1.3 con alto volumen de 77 encuestas) y viajeros de América Norte en YW que experimentaron el contraste más severo.

**BUSINESS SH: Experiencias Bimodales en IB**
El segmento Business de Short Haul registró una mejora de 8.42 puntos en IB hasta 46.15, aunque YW no pudo analizarse por corresponder a una fecha futura. Esta evolución se explica principalmente por una polarización extrema de 175 puntos entre la flota A320 que alcanzó una experiencia perfecta (NPS 100.0) y la A350 C que registró un NPS crítico de -75.0, siendo especialmente visible en la ruta MAD-OPO que mostró performance excepcional y entre perfiles donde los viajeros corporativos alcanzaron 100% de promotores mientras los clientes asiáticos experimentaron 100% de detractores.

**ECONOMY LH: Impacto Moderado por Flota Crítica**
La cabina Economy de Long Haul experimentó un deterioro de 0.88 puntos el 24 de agosto de 2025, registrando un NPS de 17.70. La causa principal fue la performance catastrófica de la flota A33ACMI que alcanzó -76.9 puntos en 13 encuestas, contrastando dramáticamente con los 53.1 puntos de la flota A350 next, complementada por problemas en codeshares específicos donde "Others" registró -75.0 y American Airlines -28.6 puntos. Esta caída se reflejó especialmente en pasajeros europeos (NPS -28.6) y operaciones transatlánticas, mientras que los perfiles más reactivos incluyeron viajeros corporativos que mostraron mayor sensibilidad (NPS -6.2) comparado con pasajeros de ocio (NPS 19.7).

**BUSINESS LH: Colapso por Problemas de Flota**
La cabina Business de Long Haul sufrió un desplome de 22.8 puntos el 24 de agosto de 2025, registrando un NPS de 20.0. Los drivers principales fueron la crisis operativa de la flota A33ACMI (NPS -76.9) y fallas en codeshares menores (Others con NPS -75.0), impactando especialmente la ruta BOG-MAD que concentró toda la operación con experiencias altamente polarizadas entre clientes españoles (NPS 58.3) y sudamericanos (NPS -33.3), siendo los perfiles de viajeros corporativos y pasajeros sudamericanos los más afectados por la inconsistencia operativa.

**PREMIUM LH: Deterioro por Inconsistencia de Flota**
El segmento Premium de Long Haul experimentó una caída de 21.85 puntos el 24 de agosto de 2025, registrando un NPS de 5.56. Las causas dominantes fueron la performance negativa de la flota A350 (NPS -22.2) versus el rendimiento positivo de A333 y A350 next (ambos con NPS 33.3), especialmente evidentes en la disparidad regional extrema donde clientes españoles alcanzaron NPS 50.0 mientras pasajeros de América del Sur registraron -33.3, siendo los viajeros sudamericanos y usuarios de flota A350 los perfiles más reactivos al deterioro de la experiencia Premium.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 10
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-08-23 to 2025-08-23:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 8 (Global, Global/LH, Global/LH/Business, Global/LH/Economy, Global/LH/Premium, Global/SH/Business/IB, Global/SH/Business/YW, Global/SH/Economy/IB)

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 **PASO 1: DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)**

## **A. Economy Short Haul:**

**Nodos disponibles:**
- ✅ **Global/SH/Economy/IB**: Existe (anomalía +9.14 pts)
- ❌ **Global/SH/Economy/YW**: No existe en el árbol

**Razón de ausencia YW**: El nodo Economy YW no aparece en el análisis, indicando que no presentó anomalía significativa el 23 de agosto de 2025.

---

## **B. Business Short Haul:**

**Nodos disponibles:**
- ✅ **Global/SH/Business/IB**: Existe (anomalía -2.44 pts)
- ✅ **Global/SH/Business/YW**: Existe (anomalía +14.58 pts)

**Análisis comparativo de patrones:**

### **Drivers Principales:**
- **IB**: Factor geográfico dominante (dispersión 133.3 pts entre regiones)
- **YW**: Factor geográfico dominante (dispersión 100 pts entre regiones)

### **Evidencia Operativa:**
- **IB**: Clientes España NPS 7.7, América Norte NPS -33.3
- **YW**: Clientes España NPS 100.0, Sin región NPS 0.0

### **Patrón Convergente:**
Ambas compañías muestran el **mismo driver principal** (región de residencia) pero con **polaridades opuestas**: IB tiene problemas con clientes españoles mientras YW tiene excelencia con clientes españoles.

---

## **🎯 DIAGNÓSTICO FINAL:**

**Economy SH**: **Causa específica de IB** - Solo IB presenta anomalía, YW sin incidencias.

**Business SH**: **Causa específica por compañía** - IB y YW divergen significativamente en la experiencia de clientes españoles, indicando problemas operativos o de servicio diferenciados por compañía.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ **PASO 2: DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)**

## **A. Short Haul - Análisis de Cabinas:**

### **Nodos disponibles:**
- ✅ **Economy SH/IB**: Anomalía **+9.14 pts** (positiva)
- ✅ **Business SH/IB**: Anomalía **-2.44 pts** (negativa)  
- ✅ **Business SH/YW**: Anomalía **+14.58 pts** (positiva)

### **Análisis de Patrones:**

**Convergencia de Driver Principal:**
- **Economy SH/IB**: Factor geográfico dominante (dispersión CodeShare 107.1 pts)
- **Business SH/IB**: Factor geográfico dominante (dispersión regional 133.3 pts)
- **Business SH/YW**: Factor geográfico dominante (dispersión regional 100 pts)

**Divergencia en Polaridad:**
- **Economy SH/IB**: Residentes España satisfechos (NPS 46.8)
- **Business SH/IB**: Residentes España insatisfechos (NPS 7.7)
- **Business SH/YW**: Residentes España muy satisfechos (NPS 100.0)

### **Patrón de Reactividad:**
Las cabinas muestran **reactividad diferencial** al mismo driver geográfico, pero con direcciones opuestas según la compañía.

---

## **B. Long Haul - Análisis de Cabinas:**

### **Nodos disponibles:**
- ✅ **Economy LH**: Anomalía **+7.60 pts** (positiva)
- ✅ **Business LH**: Anomalía **+14.34 pts** (positiva)
- ✅ **Premium LH**: Anomalía **-7.40 pts** (negativa)

### **Análisis de Patrones:**

**Convergencia de Driver Principal:**
- **Economy LH**: Factor flota dominante (A350 C problemática, NPS -30.0)
- **Business LH**: Factor flota dominante (A350 C crítica, NPS -100.0)  
- **Premium LH**: Factor flota dominante (A333 problemática, NPS -66.7)

**Progresión Lógica de Reactividad:**
- **Economy**: Problema localizado en ruta GIG-MAD/A350 C
- **Business**: Experiencias extremadamente polarizadas por flota
- **Premium**: Experiencias polarizadas pero con menor volatilidad

### **Patrón de Amortiguamiento:**
Premium LH actúa como "amortiguador" mostrando menor reactividad extrema comparado con Business LH.

---

## **🎯 DIAGNÓSTICO FINAL:**

**Short Haul**: **Causa específica por cabina y compañía** - Mismo driver geográfico pero reactividad completamente divergente entre Economy y Business, y entre IB/YW dentro de Business.

**Long Haul**: **Causa común de radio con reactividad diferencial** - Todas las cabinas convergen en problemas de flota específica, pero con progresión lógica de severidad (Business > Economy > Premium).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 **PASO 3: DIAGNÓSTICO A NIVEL RADIO Y GLOBAL**

## **A. Comparación Entre Radios:**

### **Estado de Afectación:**
- ✅ **Short Haul**: Afectado (múltiples nodos con anomalías +/-)
- ✅ **Long Haul**: Afectado (múltiples nodos con anomalías +/-)
- **Conclusión**: **Ambos radios están afectados**

### **Convergencia de Drivers:**

**Short Haul:**
- **Driver principal**: Factor geográfico (región de residencia)
- **Evidencia**: Dispersión 100-133 pts entre regiones
- **Patrón**: Clientes españoles con experiencias polarizadas según compañía

**Long Haul:**
- **Driver principal**: Factor flota (tipo de aeronave)
- **Evidencia**: A350 C problemática (-30.0 a -100.0 NPS), A321XLR/A350 next excelentes (+60.0 a +100.0 NPS)
- **Patrón**: Problemas operativos específicos por aeronave

### **Divergencia Fundamental:**
Los radios **NO convergen** en drivers principales - SH tiene causas geográficas/comerciales, LH tiene causas técnicas/operativas.

---

## **B. Coherencia con Nodo Global:**

### **Nodo Global:**
- **Estado**: Anomalía **+6.54 pts** (positiva)
- **NPS**: 32.49 vs baseline 25.95
- **Volumen**: 751 encuestas

### **Agregación vs Patrones Individuales:**

**Efecto Compensatorio Detectado:**
- **LH positivo** (+7.02 pts, 246 encuestas) + **SH mixto** (IB positivo, YW mixto) = **Global positivo** (+6.54 pts)
- El Global **atenúa** los extremos de cada radio
- La ruta problemática MAD-ORD (LH) representa solo 0.9% del volumen total (7/751)

### **Coherencia de Drivers:**
- **Global identifica**: Problema MAD-ORD/A350 C + CodeShare AA (convergente con LH)
- **Global identifica**: Clientes América Norte problemáticos (convergente con SH)
- **Síntesis coherente**: El Global captura ambos drivers de radio correctamente

---

## **🎯 DIAGNÓSTICO FINAL:**

**Causas mixtas con efectos compensatorios** - Short Haul presenta problemas geográficos/comerciales específicos por compañía, mientras Long Haul presenta problemas técnicos/operativos específicos por flota. El nodo Global refleja coherentemente ambos patrones causales, donde los efectos positivos de LH (flotas premium) compensan parcialmente los efectos negativos localizados, resultando en una anomalía global positiva que enmascara problemas operativos críticos subyacentes.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 **PASO 4: ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS**

## **CAUSA 1: PROBLEMAS CRÍTICOS EN FLOTA A350 C**

### **A. NATURALEZA DE LA CAUSA:**
**Hipótesis**: Falla operativa severa en aeronaves A350 C que generó experiencias catastróficas para los pasajeros, posiblemente relacionada con problemas técnicos, configuración de cabina deficiente o incidentes de servicio no reportados formalmente en sistemas NCS.

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**

**Segmento más grande afectado**: **Global/LH** (246 encuestas)

**Output causal detallado**:
- **NPS A350 C**: -30.0 puntos (Economy), -100.0 puntos (Business)
- **Contraste con otras flotas**: A321XLR alcanzó +66.7 NPS el mismo día
- **Dispersión por flota**: 96.7 puntos - la mayor variabilidad detectada
- **Ruta específica identificada**: GIG-MAD (7 clientes, todos detractores, NPS 0.0)

**Alcance de afectación**: Esta causa afecta a **todos los subsegmentos** bajo Global/LH:
- Global/LH/Economy: Problema localizado en ruta GIG-MAD
- Global/LH/Business: Experiencias extremadamente polarizadas
- Global/LH/Premium: Impacto indirecto por problemas en flota A333 relacionada

**Métricas clave disponibles**:
- **Indicadores operativos**: ❌ No disponibles
- **NPS específico**: A350 C (-30.0 a -100.0) vs A321XLR (+66.7)
- **Rutas involucradas**: GIG-MAD confirmada, MAD-ORD sospechosa
- **Verbatims**: ❌ No disponibles

---

## **CAUSA 2: PROBLEMAS EN ACUERDOS DE CODESHARE AA**

### **A. NATURALEZA DE LA CAUSA:**
**Hipótesis**: Degradación significativa en la experiencia de clientes bajo acuerdos de codeshare con American Airlines, posiblemente debido a desalineación en estándares de servicio, problemas de coordinación operativa o expectativas no cumplidas en servicios diferenciados.

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**

**Segmento más grande afectado**: **Global/SH/Economy/IB** (281 encuestas)

**Output causal detallado**:
- **NPS CodeShare AA**: -57.1 puntos (7 encuestas)
- **Contraste con otros CodeShare**: VY/I2 alcanzó +50.0 NPS
- **Dispersión CodeShare**: 107.1 puntos - mayor variabilidad en SH
- **Correlación geográfica**: Clientes América Norte también problemáticos (NPS 8.3)

**Alcance de afectación**: Esta causa afecta principalmente a:
- Global/SH/Economy/IB: Impacto directo y cuantificado
- Global (nivel agregado): Contribuye al patrón de clientes América Norte insatisfechos

**Métricas clave disponibles**:
- **Indicadores operativos**: ❌ No disponibles
- **NPS específico**: CodeShare AA (-57.1) vs VY/I2 (+50.0)
- **Rutas involucradas**: Posiblemente MAD-ORD (conexión con LH)
- **Verbatims**: ❌ No disponibles

---

## **CAUSA 3: DISPARIDAD GEOGRÁFICA EN EXPERIENCIA DE CLIENTES**

### **A. NATURALEZA DE LA CAUSA:**
**Hipótesis**: Diferencias sistemáticas en expectativas, servicios recibidos o procesos operativos según la región de residencia del cliente, generando experiencias extremadamente polarizadas que sugieren problemas en la estandarización del servicio por mercado geográfico.

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**

**Segmento más grande afectado**: **Global/SH/Business/IB** (34 encuestas con dispersión crítica)

**Output causal detallado**:
- **Dispersión geográfica**: 133.3 puntos entre regiones
- **América Sur**: NPS 100.0 (experiencia perfecta)
- **España**: NPS 7.7 (baja satisfacción)
- **América Norte**: NPS -33.3 (muy insatisfechos)
- **Ruta ejemplar**: MAD-ZRH con clientes sudamericanos (NPS 100.0)

**Alcance de afectación**: Esta causa afecta a **múltiples subsegmentos**:
- Global/SH/Business/IB: Patrón más severo identificado
- Global/SH/Business/YW: Patrón inverso (España excelente, otros problemáticos)
- Global/SH/Economy/IB: Residentes España satisfechos vs América Sur problemáticos

**Métricas clave disponibles**:
- **Indicadores operativos**: ❌ No disponibles
- **NPS por región**: Variaciones de 133.3 puntos entre segmentos
- **Rutas involucradas**: MAD-ZRH (excelente), DSS-MAD (moderada)
- **Verbatims**: ❌ No disponibles

---

## **🚨 LIMITACIÓN CRÍTICA TRANSVERSAL:**

**Ausencia sistemática de datos operativos, incidentes NCS y verbatims** en todas las causas identificadas, lo que impide determinar las causas raíz específicas y limita las recomendaciones a correlaciones observadas en los datos de perfil de cliente disponibles.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El 23 de agosto de 2025 presentó un patrón complejo de anomalías NPS donde el indicador global experimentó una mejora de 6.54 puntos, pasando de un NPS de 25.95 a 32.49, resultado de efectos compensatorios entre diferentes segmentos. En Long Haul, el segmento experimentó una subida de 7.02 puntos (de 21.44 a 28.45), mientras que en Short Haul se registraron movimientos divergentes: IB Economy subió 9.14 puntos (de 25.73 a 34.88), IB Business bajó 2.44 puntos (de 37.74 a 35.29), y YW Business subió 14.58 puntos (de 17.00 a 31.58). Las causas identificadas incluyen problemas críticos en la flota A350 C que generaron NPS catastróficos de -30.0 en Economy LH y -100.0 en Business LH, disfunciones severas en acuerdos de codeshare con American Airlines que registraron NPS de -57.1 en Economy SH IB, y disparidades geográficas extremas en la experiencia de clientes con variaciones de hasta 133.3 puntos entre regiones en Business SH. La evidencia de estas causas se manifiesta en la dispersión de 96.7 puntos entre flotas en Long Haul, la variabilidad de 107.1 puntos en acuerdos de codeshare en Short Haul, y las diferencias sistemáticas por región de residencia del cliente.

Las rutas más afectadas incluyen GIG-MAD operada con A350 C donde 7 clientes resultaron todos detractores (NPS 0.0), MAD-ORD con NPS de -14.3 asociada a problemas de codeshare AA, y MAD-UIO que registró NPS 0.0 con 8 pasajeros. Los grupos de clientes más reactivos fueron los residentes de América Norte que mostraron consistente insatisfacción across múltiples segmentos, clientes españoles que experimentaron polarización extrema según la compañía (NPS 7.7 en IB vs 100.0 en YW para Business), y pasajeros de Business que demostraron mayor sensibilidad a problemas operativos comparado con segmentos Leisure.

**ECONOMY SH IB: Mejora Significativa con Contrastes Operativos**
La cabina Economy SH IB experimentó una mejora notable de 9.14 puntos, pasando de un NPS de 25.73 a 34.88 vs el baseline de 7 días previos. La causa principal fue el efecto dominante de residentes españoles satisfechos que alcanzaron un NPS de 46.8 representando el 44.8% del volumen total (126 encuestas), compensando significativamente los problemas críticos en acuerdos de codeshare con American Airlines que registraron NPS de -57.1 y la flota A33ACMI problemática con NPS de -16.7. Esta mejora se reflejó especialmente en la concentración geográfica favorable, mientras que los perfiles más reactivos incluyen clientes de América Sur (NPS 7.4) y usuarios de codeshare AA que experimentaron las mayores disfunciones del día.

**BUSINESS SH IB: Deterioro por Disparidad Geográfica**
El segmento Business SH IB registró una caída de 2.44 puntos, pasando de un NPS de 37.74 a 35.29 vs la media de los últimos 7 días. Esta evolución se explica principalmente por una dispersión geográfica extrema de 133.3 puntos entre regiones, donde clientes de América Sur alcanzaron NPS perfecto de 100.0 y Europa registró 81.8, contrastando dramáticamente con residentes españoles que obtuvieron solo 7.7 y América Norte que cayó a -33.3. El deterioro fue especialmente visible en rutas como MAD-ZRH que paradójicamente mostró NPS 100.0 con clientes sudamericanos, mientras que los perfiles más reactivos incluyen residentes españoles y norteamericanos que concentraron la insatisfacción del segmento.

**BUSINESS SH YW: Recuperación Excepcional con Polarización Regional**
La cabina Business SH YW experimentó una mejora sustancial de 14.58 puntos, pasando de un NPS de 17.00 a 31.58 vs el baseline previo. La causa principal fue la excelencia en la experiencia de clientes españoles que alcanzaron NPS perfecto de 100.0, complementada por el buen desempeño de residentes europeos (NPS 60.0) y sudamericanos (NPS 50.0), contrastando con clientes sin región identificada que registraron NPS 0.0. Esta mejora se reflejó especialmente en la concentración de viajeros Leisure (NPS 38.5) superando a Business/Work (NPS 16.7), mientras que los perfiles más reactivos incluyen clientes sin identificación regional y viajeros corporativos que mostraron mayor criticidad en sus evaluaciones.

**ECONOMY LH: Mejora con Crisis Operativa Localizada**
La cabina Economy de LH experimentó una mejora de 7.60 puntos, pasando de un NPS de 18.59 a 26.19 vs la media de 7 días previos. La causa principal fue un problema operativo severo completamente localizado en la ruta GIG-MAD operada con aeronave A350 C, donde 7 clientes resultaron todos detractores (NPS 0.0), contrastando dramáticamente con la flota A321XLR que alcanzó NPS de 66.7 el mismo día. Esta aparente mejora se reflejó especialmente en el efecto de compensación donde las flotas premium diluyeron el impacto catastrófico de la A350 C, mientras que los perfiles más reactivos incluyen clientes sin región de residencia definida que correlacionan directamente con el incidente crítico identificado.

**BUSINESS LH: Experiencias Extremadamente Polarizadas**
La cabina Business de LH registró una mejora de 14.34 puntos, pasando de un NPS de 42.80 a 57.14 vs el período anterior de 7 días. Los drivers principales fueron la polarización extrema por tipo de flota con A332 y A350 next alcanzando NPS perfecto de 100.0, contrastando dramáticamente con la flota A350 C que registró NPS crítico de -100.0, complementado por la excelencia geográfica de clientes de América Centro y Sur (NPS 100.0) versus problemas con residentes españoles (NPS 0.0). Esta evolución impactó especialmente la ruta MAD-MEX con 4 pasajeros que lograron NPS perfecto y cero incidentes, mientras que los perfiles más reactivos incluyen usuarios de flota A350 C y clientes españoles que experimentaron las mayores disfunciones operativas.

**PREMIUM LH: Deterioro por Problemas de Flota Específica**
El segmento Premium de LH experimentó una caída de 7.40 puntos, pasando de un NPS de 27.40 a 20.0 vs la semana anterior. Las causas dominantes fueron los problemas críticos en la flota A333 que registró NPS de -66.7 contrastando con la A350 next que alcanzó NPS de +60.0, generando una dispersión de 126.7 puntos entre tipos de aeronave, complementado por la disparidad geográfica donde clientes de América Centro obtuvieron NPS de +75.0 mientras que América Norte cayó a -33.3. Este deterioro fue especialmente evidente en la ruta EZE-MAD con NPS 0.0 y entre clientes de América Norte que mostraron la mayor reactividad negativa del segmento.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 8
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

TAREA:
1. Copia la síntesis ejecutiva del interpreter semanal TAL COMO ESTÁ
2. Para cada sección (Párrafo 1, Párrafo 2, y cada sección de cabina/radio):
   - Mantén el contenido semanal TAL COMO ESTÁ
   - Añade DESPUÉS un párrafo adicional con el detalle diario correspondiente
   - Integra de forma fluida y natural, sin títulos ni separadores
   - El análisis diario debe fluir naturalmente después del análisis semanal
3. Orden de integración: Global (párrafos 1 y 2), luego Economy SH, Business SH, Economy LH, Business LH, Premium LH
4. Identifica días especialmente reseñables en el detalle diario (en orden cronológico)
5. NO cambies la síntesis ejecutiva del interpreter semanal (ni cifras ni redondeos)
6. NO añadas recomendaciones adicionales
7. Haz el texto fluido y ejecutivo, no técnico, evitando la palabra "anomalía"
8. Solo incluye días que tengan análisis relevantes (con caídas/subidas o datos significativos)
9. Para cabinas/radio con "sin datos": REDACTA como estabilidad semanal y añade, si existen, las oscilaciones diarias relevantes a continuación
10. **CRÍTICO**: Si hay datos en "ANÁLISIS DIARIO SINGLE", DEBES usarlos. NO digas que "no están disponibles" si están presentes en el input.
11. **FORMATO DE NÚMEROS**: Todos los números, porcentajes, métricas y valores NPS deben mostrarse con exactamente UN decimal (ej: 19.8, -4.4, 93.5%)