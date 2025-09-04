===== SYSTEM =====

Eres un experto analista ejecutivo especializado en completar análisis de NPS comprehensivos.

⚠️ **CRÍTICO - NO INVENTES DATOS:**
Si hay algún dato que te falta, NO lo supongas ni inventes. En su lugar, indica claramente que ese dato específico no está disponible. Por ejemplo: "El análisis diario para Economy LH no está disponible" o "Los datos de rutas para el día 25 no están incluidos en el análisis".

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
[{'period': 1, 'date_range': '2025-08-22 to 2025-08-28', 'ai_interpretation': '📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**\n\n**Nodos analizados:** 7 (Global, Global/LH/Business, Global/LH/Premium, Global/SH/Business, Global/SH/Business/IB, Global/SH/Business/YW, Global/SH/Economy/YW)\n\n---\n\n## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA\n\n# 🏢 DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)\n\n## **A. ECONOMY SHORT HAUL:**\n\n**Nodos disponibles:** Solo existe **Global/SH/Economy/YW** (+12.88 pts)\n**Nodo faltante:** Global/SH/Economy/IB (no reportado)\n\n**Diagnóstico:** No es posible determinar si la causa es específica de compañía YW o general de cabina Economy SH, debido a la ausencia del nodo IB para comparación.\n\n---\n\n## **B. BUSINESS SHORT HAUL:**\n\n**Nodos disponibles:** \n- **Global/SH/Business/IB** (+8.93 pts)\n- **Global/SH/Business/YW** (+7.71 pts)\n\n**Análisis comparativo de patrones:**\n\n**Convergencia en drivers principales:**\n- Ambos nodos muestran **Punctuality** como driver SHAP dominante (IB: +5.306, YW: +9.212)\n- Ambos validan mejora **OTP15** (+3.6 pts IB, +3.1 pts YW)\n- Ambos confirman **0 incidentes NCS** durante el período\n\n**Convergencia en evidencia operativa:**\n- Patrones geográficos similares (concentración en hub MAD)\n- Mejoras operativas consistentes en puntualidad\n- Ausencia total de disrupciones operativas\n\n**Diferencias menores:**\n- Drivers secundarios varían (IB: food/beverage, YW: cabin crew)\n- Magnitudes SHAP diferentes pero mismo patrón direccional\n\n**Diagnóstico:** **Causa común de cabina Business SH** - Ambas compañías (IB/YW) convergen en el mismo driver principal (puntualidad mejorada) y evidencia operativa (0 incidentes NCS, mejora OTP15).\n\n---\n\n## 💺 DIAGNÓSTICO A NIVEL DE CABINA\n\n# ✈️ DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)\n\n## **A. SHORT HAUL - ANÁLISIS DE CABINAS:**\n\n**Nodos disponibles:**\n- **Global/SH/Economy/YW:** +12.88 pts\n- **Global/SH/Business (agregado):** +10.11 pts  \n- **Global/SH/Business/IB:** +8.93 pts\n- **Global/SH/Business/YW:** +7.71 pts\n\n**Análisis de convergencia:**\n\n**Convergencia completa en driver principal:**\n- **Economy SH/YW:** Punctuality SHAP +5.644, OTP15 +3.1 pts, 0 incidentes NCS\n- **Business SH (ambas compañías):** Punctuality SHAP dominante (IB: +5.306, YW: +9.212), OTP15 mejorado (+3.6/+3.1 pts), 0 incidentes NCS\n\n**Convergencia en evidencia operativa:**\n- Todas las cabinas SH validan la misma mejora operativa fundamental\n- Ausencia total de disrupciones en el mismo período temporal\n- Patrones geográficos similares (concentración hub MAD)\n\n**Diagnóstico:** **Causa común de radio Short Haul** - Convergencia completa entre Economy y Business SH en driver principal (puntualidad mejorada) y evidencia operativa (ausencia de incidentes NCS).\n\n---\n\n## **B. LONG HAUL - ANÁLISIS DE CABINAS:**\n\n**Nodos disponibles:**\n- **Global/LH/Business:** +14.59 pts\n- **Global/LH/Premium:** -1.89 pts\n\n**Análisis de divergencia:**\n\n**Divergencia crítica en drivers:**\n- **Business LH:** Boarding SHAP +4.584 (driver principal), Punctuality +2.733 (secundario)\n- **Premium LH:** Connections experience SHAP -3.340 (driver principal negativo), deterioro específico en hub MAD\n\n**Divergencia en evidencia operativa:**\n- **Business LH:** Beneficiado por mejoras en producto/servicio (boarding, arrivals)\n- **Premium LH:** Afectado por problemas específicos de connections experience en rutas intercontinentales\n\n**Reactividad diferencial confirmada:**\n- Misma base operativa (0 incidentes NCS, OTP15 mejorado) genera respuestas opuestas por tipo de cliente\n\n**Diagnóstico:** **Causa específica de cabina Long Haul** - Business LH se beneficia de mejoras en producto mientras Premium LH sufre deterioros específicos en connections experience, evidenciando reactividad diferencial ante la misma base operativa.\n\n---\n\n## 🌎 DIAGNÓSTICO GLOBAL POR RADIO\n\n# 🌍 DIAGNÓSTICO A NIVEL RADIO Y GLOBAL\n\n## **A. COMPARACIÓN ENTRE RADIOS:**\n\n**Estado de nodos por radio:**\n- **Short Haul:** Todas las cabinas positivas (+7.71 a +12.88 pts)\n- **Long Haul:** Patrón mixto (Business LH +14.59 pts, Premium LH -1.89 pts)\n\n**Análisis de convergencia entre radios:**\n\n**Convergencia parcial en driver operativo base:**\n- **Short Haul:** Punctuality como driver SHAP dominante universal\n- **Long Haul Business:** Punctuality SHAP +2.733 (secundario tras Boarding +4.584)\n- **Long Haul Premium:** Punctuality no es driver principal (foco en connections experience)\n\n**Evidencia operativa común:**\n- **Ambos radios:** 0 incidentes NCS, mejoras OTP15 (+3.1 a +5.3 pts)\n- **Diferencia crítica:** SH capitaliza completamente la mejora operativa, LH muestra reactividad diferencial por cabina\n\n---\n\n## **B. COHERENCIA CON NODO GLOBAL:**\n\n**Nodo Global:** +5.36 pts (NPS 26.49 vs 21.14 baseline)\n\n**Validación de agregación:**\n- **Global driver principal:** Punctuality SHAP +4.067 ✅ **Coherente** con dominancia SH\n- **Global evidencia operativa:** OTP15 +3.6 pts, 0 incidentes NCS ✅ **Coherente** con ambos radios\n- **Global magnitud:** +5.36 pts refleja **promedio ponderado** entre SH positivo fuerte y LH mixto\n\n**Amplificación vs atenuación:**\n- El Global **atenúa** las mejoras de SH (+10-12 pts → +5.36 pts) por el efecto mixto de LH\n\n---\n\n## **DIAGNÓSTICO FINAL:**\n\n**Causas mixtas con base operativa común** - Ambos radios comparten la misma mejora operativa fundamental (puntualidad, 0 incidentes NCS), pero Short Haul capitaliza completamente esta mejora mientras Long Haul muestra reactividad diferencial por cabina, resultando en un impacto Global moderadamente positivo que refleja la combinación de efectos fuertes en SH y mixtos en LH.\n\n---\n\n## 📋 ANÁLISIS DE CAUSAS DETALLADO\n\n# 🔍 ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS\n\n## **CAUSA 1: MEJORA OPERATIVA EN PUNTUALIDAD**\n\n### **A. NATURALEZA DE LA CAUSA:**\n**Hipótesis:** Implementación exitosa de mejoras operativas que resultaron en una operación excepcionalmente limpia durante la semana del 22-28 agosto 2025, caracterizada por ausencia total de disrupciones y mejora significativa en puntualidad, generando un efecto cascada positivo en múltiples touchpoints del customer journey.\n\n### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**\n\n**Segmento más grande afectado:** **Global** (nivel compañía completo)\n\n**Output causal detallado:**\n- **NPS Impact:** +5.36 pts (26.49 vs 21.14 baseline)\n- **Driver SHAP principal:** Punctuality +4.067 (más influyente)\n- **Mejora operativa validada:** OTP15 +3.6 puntos (85.8% vs 82.19%)\n- **Incidentes NCS:** 0 incidentes operativos en los 7 días (reducción total)\n- **Misconex:** Mejora de -0.2 puntos (0.65% vs 0.87%)\n\n**Subsegmentos afectados:** Esta causa afecta a TODOS los subsegmentos bajo Global, incluyendo:\n- Global/SH/Economy/YW (+12.88 pts)\n- Global/SH/Business (+10.11 pts)\n- Global/SH/Business/IB (+8.93 pts) \n- Global/SH/Business/YW (+7.71 pts)\n- Global/LH/Business (+14.59 pts)\n\n**Rutas involucradas:** 8 rutas únicas identificadas con performance destacado en ODB-PMI (NPS 100.0), PMI-RJL (NPS 80.0), MAD-PMO (NPS 80.0)\n\n**Verbatims representativos:** 1,159+ comentarios sobre BOARDING validando mejoras en procesos operativos\n\n---\n\n## **CAUSA 2: DETERIORO EN CONNECTIONS EXPERIENCE (HUB MADRID)**\n\n### **A. NATURALEZA DE LA CAUSA:**\n**Hipótesis:** Problemas específicos en la gestión de conexiones en el hub de Madrid para pasajeros Premium de rutas intercontinentales, particularmente de América del Norte y Latinoamérica, que no están relacionados con la operación base (que mejoró) sino con procesos específicos de connections experience y journey preparation support.\n\n### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**\n\n**Segmento más grande afectado:** **Global/LH/Premium**\n\n**Output causal detallado:**\n- **NPS Impact:** -1.89 pts (25.53 vs 27.42 baseline)\n- **Driver SHAP principal:** Connections experience -3.340 (más negativo)\n- **Deterioro específico:** Satisfacción connections -9.93 puntos vs L7d\n- **Driver secundario:** Journey preparation support SHAP -2.766 (-6.80 puntos satisfacción)\n- **Contradictorio operativo:** Mejoras en OTP15 (+5.3 pts), Load Factor (-1.3 pts), 0 incidentes NCS\n\n**Subsegmentos afectados:** Esta causa afecta específicamente al segmento Premium LH y no se replica en otros subsegmentos.\n\n**Rutas críticas involucradas:**\n- LAX-MAD: NPS -50.0 (8 pasajeros)\n- LIM-MAD: NPS 0.0 (deterioro -42.9 vs L7d)\n- MAD-SCL: NPS 0.0 (deterioro -30.8 vs L7d)\n- GRU-MAD: NPS 0.0 (deterioro -42.9 vs L7d)\n\n**Patrón geográfico:** Hub común Madrid (MAD) como punto de conexión crítico para rutas intercontinentales de América del Norte y Latinoamérica.\n\n**Verbatims representativos:** 161 comentarios sobre BOARDING confirmando problemas específicos en procesos que complementan los issues de connections experience.\n\n---\n\n## **CAUSA 3: MEJORAS EN PRODUCTO/SERVICIO (BOARDING)**\n\n### **A. NATURALEZA DE LA CAUSA:**\n**Hipótesis:** Optimizaciones específicas implementadas en los procesos de boarding que generaron mejoras significativas en la experiencia del cliente, particularmente efectivas en segmentos Business de larga distancia, actuando como driver complementario a las mejoras operativas de puntualidad.\n\n### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**\n\n**Segmento más grande afectado:** **Global/LH/Business**\n\n**Output causal detallado:**\n- **NPS Impact:** +14.59 pts (41.77 vs 27.18 baseline)\n- **Driver SHAP principal:** Boarding +4.584 (más influyente)\n- **Mejora específica:** Satisfacción boarding +2.95 puntos vs L7d\n- **Drivers complementarios:** Journey preparation support +3.870, Arrivals experience +3.409\n- **Validación operativa:** OTP15 +5.3 pts, 0 incidentes NCS\n\n**Subsegmentos afectados:** Esta causa afecta principalmente a Business LH, con efectos secundarios en otros segmentos Business.\n\n**Rutas destacadas:**\n- GUA-MAD: NPS 100.0 (boarding perfecto 100.0)\n- MAD-MEX: NPS 81.8 (boarding 90.0, mayor volumen 11 pax)\n- BOS-MAD: NPS 50.0 (boarding perfecto 100.0)\n\n**Verbatims representativos:** 236+ análisis específicos sobre boarding en Global/LH/Business confirmando mejoras percibidas directamente por clientes.\n\n---\n\n## 📋 SÍNTESIS EJECUTIVA FINAL\n\n📈 SÍNTESIS EJECUTIVA:\n\nEl análisis del árbol de anomalías NPS para la semana del 22-28 agosto 2025 revela un patrón de mejoras generalizadas impulsado por una operación excepcionalmente limpia. A nivel Global, el NPS experimentó una mejora de 5.36 puntos, pasando de 21.14 a 26.49 puntos, explicada principalmente por una mejora operativa en puntualidad (SHAP +4.067) respaldada por la ausencia total de incidentes NCS durante los 7 días analizados y una mejora de 3.6 puntos en OTP15. Esta mejora operativa se manifestó de manera diferenciada según el radio: Short Haul capitalizó completamente la mejora con subidas entre 7.71 y 12.88 puntos NPS en todos sus segmentos, mientras que Long Haul mostró un patrón mixto donde Business LH se benefició significativamente (+14.59 puntos) por mejoras en boarding (SHAP +4.584), pero Premium LH experimentó un deterioro de 1.89 puntos debido a problemas específicos en connections experience en el hub de Madrid (SHAP -3.340) para rutas intercontinentales de América del Norte y Latinoamérica.\n\nLas rutas más afectadas positivamente incluyen ODB-PMI y PMI-RJL (ambas con NPS 100.0 y 80.0 respectivamente), MAD-OPO, LEI-MAD y MAD-XRY en Short Haul, y GUA-MAD, MAD-MEX en Business LH, mientras que las rutas críticas con deterioro se concentran en LAX-MAD, LIM-MAD, MAD-SCL y GRU-MAD en Premium LH. Los grupos de clientes más reactivos varían por segmento, destacando la alta sensibilidad por región de residencia en múltiples nodos (spreads de 62.1 a 136.8 puntos), seguida por tipo de flota y acuerdos de codeshare, confirmando que clientes de diferentes mercados geográficos experimentaron realidades operativas completamente distintas durante esta semana.\n\n**ECONOMY SH YW: Excelencia Operativa Capitalizada**\nLa cabina Economy de Short Haul YW experimentó la mayor mejora semanal con 12.88 puntos, pasando de un NPS de 20.84 a 33.72 durante la semana del 22-28 agosto. La causa principal fue una performance operativa excepcional liderada por mejoras en puntualidad (SHAP +5.644) con una mejora de 3.1 puntos en OTP15 y ausencia total de incidentes operativos NCS, complementada por mejoras en arrivals experience (+1.705 SHAP), boarding (+1.656 SHAP) y journey preparation support (+1.501 SHAP). Esta mejora se reflejó especialmente en rutas como ODB-PMI (NPS 100.0), PMI-RJL (NPS 80.0) y PMI-VLC (NPS 66.7), mientras que los perfiles más reactivos incluyen clientes por región de residencia (spread 76.8 puntos) y tipo de codeshare (spread 69.5 puntos).\n\n**BUSINESS SH: Mejora Operativa Consistente**\nEl segmento Business de Short Haul registró mejoras consistentes entre ambas compañías, con Business SH IB alcanzando un NPS de 37.88 (mejora de 8.93 puntos) y Business SH YW un NPS de 18.82 (mejora de 7.71 puntos) vs la semana anterior. Esta evolución se explica principalmente por mejoras operativas en puntualidad (SHAP +5.306 en IB, +9.212 en YW) respaldadas por mejoras en OTP15 de 3.6 y 3.1 puntos respectivamente y ausencia total de incidentes NCS, siendo especialmente visible en rutas como MAD-OPO, LEI-MAD, DBV-MAD y FLR-MAD, y entre perfiles diferenciados por región de residencia (spreads de 120.8 y 111.8 puntos) y tipo de flota.\n\n**ECONOMY LH: Desempeño Estable**\nLa cabina Economy de Long Haul mantuvo desempeño estable durante esta semana. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción.\n\n**BUSINESS LH: Optimización de Producto y Servicio**\nLa cabina Business de Long Haul experimentó la mejora más significativa con 14.59 puntos, pasando de un NPS de 27.18 a 41.77 durante la semana del 22-28 agosto. Los drivers principales fueron optimizaciones en boarding (SHAP +4.584) con mejoras de 2.95 puntos en satisfacción, complementado por journey preparation support (+3.870 SHAP) y arrivals experience (+3.409 SHAP), todo respaldado por mejoras operativas en puntualidad (+2.733 SHAP) y ausencia de incidentes NCS, impactando especialmente las rutas GUA-MAD (NPS 100.0), MAD-MEX (NPS 81.8) y BOS-MAD (NPS 50.0), y perfiles diferenciados por tipo de flota (spread 127.3 puntos) y región de residencia (spread 68.6 puntos).\n\n**PREMIUM LH: Deterioro en Connections Experience**\nEl segmento Premium de Long Haul experimentó un deterioro de 1.89 puntos, pasando de un NPS de 27.42 a 25.53 vs la semana anterior. Las causas dominantes fueron problemas críticos en connections experience (SHAP -3.340) con una caída de 9.93 puntos en satisfacción y deterioro en journey preparation support (SHAP -2.766), especialmente evidentes en rutas intercontinentales como LAX-MAD (NPS -50.0), LIM-MAD, MAD-SCL y GRU-MAD (todas con NPS 0.0 y deterioros de 30.8 a 42.9 puntos), y entre perfiles de clientes por región de residencia (spread 62.1 puntos) concentrados en América del Norte y Latinoamérica que utilizan Madrid como hub de conexión.\n\n---\n\n✅ **ANÁLISIS COMPLETADO**\n\n- **Nodos procesados:** 7\n- **Pasos de análisis:** 5\n- **Metodología:** Análisis conversacional paso a paso\n- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado\n\n*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*'}]

**ANÁLISIS DIARIO SINGLE:**
📅 2025-08-28 to 2025-08-28:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 11 (Global, Global/LH, Global/LH/Business, Global/LH/Premium, Global/SH, Global/SH/Business, Global/SH/Business/IB, Global/SH/Business/YW, Global/SH/Economy, Global/SH/Economy/IB, Global/SH/Economy/YW)

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)

## **A. ECONOMY SHORT HAUL**

**Nodos identificados:**
- ✅ **Economy SH/IB**: Nodo anómalo (-) presente
- ✅ **Economy SH/YW**: Nodo anómalo (-) presente

**Análisis comparativo:**

### **Economy SH/IB**
- **Driver principal**: Mishandling crítico (28.0% vs 18.2% promedio, +9.8pts)
- **Impacto**: NPS 20.5 vs 26.8 baseline (-6.3pts)
- **Patrón**: Impacto disperso sin concentración geográfica específica

### **Economy SH/YW**  
- **Driver principal**: Mishandling crítico (24.9% vs 17.1% promedio, +7.8pts)
- **Impacto**: NPS 29.25 vs 33.74 baseline (-4.50pts)
- **Patrón**: Impacto concentrado en CodeShare VY (-50.0 NPS)

**DIAGNÓSTICO**: **Causa común de cabina** - Ambas compañías convergen en el mismo driver operativo (mishandling crítico) con magnitudes similares, aunque con patrones de impacto ligeramente diferentes.

---

## **B. BUSINESS SHORT HAUL**

**Nodos identificados:**
- ✅ **Business SH/IB**: Nodo anómalo (-) presente  
- ✅ **Business SH/YW**: Nodo anómalo (-) presente

**Análisis comparativo:**

### **Business SH/IB**
- **Driver principal**: Mishandling crítico (28.0% vs 18.2% promedio, +9.8pts)
- **Impacto**: NPS 0.0 vs 40.77 baseline (-40.77pts)
- **Patrón**: Impacto altamente segmentado por perfil geográfico (dispersión 133.3pts)

### **Business SH/YW**
- **Driver principal**: Mishandling crítico (24.9% vs 17.1% promedio, +7.8pts)  
- **Impacto**: NPS 14.29 vs 16.92 baseline (-2.64pts)
- **Patrón**: Impacto concentrado en viajeros Leisure (NPS 0.0 vs Business/Work 50.0)

**DIAGNÓSTICO**: **Causa común de cabina** - Ambas compañías presentan el mismo driver operativo fundamental (mishandling crítico) con impactos diferenciados solo por composición de perfiles de clientes, no por causas operativas distintas.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)

## **A. SHORT HAUL - ANÁLISIS DE CABINAS**

**Nodos identificados:**
- ✅ **Economy SH**: Nodo anómalo (-) presente
- ✅ **Business SH**: Nodo anómalo (-) presente

**Análisis comparativo:**

### **Economy SH**
- **Driver operativo**: Mishandling crítico (~25-28% vs ~17-18% promedio)
- **Reactividad**: Moderada (-5.68pts en promedio)
- **Patrón**: Impacto disperso geográficamente

### **Business SH**  
- **Driver operativo**: Mishandling crítico (~25-28% vs ~17-18% promedio)
- **Reactividad**: Extrema (-29.32pts en promedio, llegando hasta -40.77pts en IB)
- **Patrón**: Alta segmentación por perfil de cliente

**Consistencia entre compañías:**
- **IB**: Business más reactivo que Economy (factor 6x mayor impacto)
- **YW**: Business menos reactivo que Economy pero aún significativo

**DIAGNÓSTICO**: **Causa común de radio con reactividad diferencial crítica** - Ambas cabinas convergen en el mismo driver operativo (mishandling), pero Business muestra hipersensibilidad extrema al problema, actuando como "amplificador" del impacto operativo.

---

## **B. LONG HAUL - ANÁLISIS DE CABINAS**

**Nodos identificados:**
- ✅ **Business LH**: Nodo anómalo (+) presente - **PARADOJA POSITIVA**
- ✅ **Premium LH**: Nodo anómalo (-) presente - **ARTEFACTO ESTADÍSTICO**

**Análisis comparativo:**

### **Business LH**
- **Driver operativo**: Mishandling crítico (27.2% vs 17.9% promedio)
- **Reactividad**: Paradójica (+10.72pts a pesar del deterioro operativo)
- **Mecanismo**: Resistencia del segmento Business compensó impacto en Leisure

### **Premium LH**
- **Driver operativo**: Sin impacto real de problemas operativos
- **Reactividad**: Artificial (volumen crítico bajo: 6-8 encuestas)
- **Realidad**: NPS 100.0 (satisfacción perfecta)

**DIAGNÓSTICO**: **Patrón específico de cabina con amortiguación Premium** - Long Haul muestra un comportamiento completamente diferente: Business actúa como "amortiguador" resistente a problemas operativos, mientras Premium está completamente aislado de disrupciones operativas, sugiriendo protocolos de servicio diferenciados que protegen efectivamente las cabinas superiores.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

## **A. COMPARACIÓN ENTRE RADIOS**

**Nodos identificados:**
- ✅ **Short Haul (SH)**: Nodo anómalo (-) presente
- ✅ **Long Haul (LH)**: Nodo anómalo (-) presente

**Análisis comparativo:**

### **Short Haul**
- **Driver principal**: Mishandling crítico (27.2% vs 17.9% promedio, +9.3pts)
- **Impacto**: NPS 21.7 vs 29.8 baseline (-8.1pts)
- **Patrón**: Concentración geográfica crítica (BCN-MAD con NPS -23.8)
- **Naturaleza**: Problema operativo localizado con impacto severo

### **Long Haul**
- **Driver principal**: Mishandling crítico (27.2% vs 17.9% promedio, +9.3pts) 
- **Impacto**: NPS 22.3 vs 31.4 baseline (-6.7pts)
- **Patrón**: Concentración geográfica (EZE-MAD con NPS 30.8)
- **Naturaleza**: Mismo problema operativo pero con mayor resistencia

**Convergencia de drivers**: ✅ **PERFECTA** - Ambos radios muestran exactamente el mismo driver operativo (mishandling +9.3pts) con valores SHAP consistentes.

---

## **B. COHERENCIA CON NODO GLOBAL**

**Nodo identificado:**
- ✅ **Global**: Nodo anómalo (-) presente

### **Global**
- **Driver principal**: Mishandling crítico (27.2% vs 17.9% promedio, +9.3pts)
- **Impacto**: NPS 24.95 vs 27.23 baseline (-2.28pts)
- **Patrón**: Impacto agregado con alta dispersión entre segmentos (100pts spread)

### **Coherencia matemática:**
- **SH**: -8.1pts de impacto
- **LH**: -6.7pts de impacto  
- **Global**: -2.28pts de impacto
- **Interpretación**: El Global refleja un **efecto de atenuación** donde segmentos resilientes (Leisure con +28.8 NPS) compensan parcialmente los impactos severos de segmentos críticos.

---

## **DIAGNÓSTICO FINAL**

**CAUSA GLOBAL HOMOGÉNEA CON ATENUACIÓN COMPENSATORIA** - Ambos radios convergen perfectamente en el mismo driver operativo (mishandling crítico +9.3pts) con evidencia operativa idéntica, pero el nodo Global muestra atenuación significativa (-2.28pts vs -8.1pts SH y -6.7pts LH) debido a que segmentos resilientes como Leisure (+28.8 NPS) y ciertos CodeShares positivos compensan matemáticamente los impactos severos de segmentos críticos, creando un "efecto amortiguador" a nivel agregado que enmascara la severidad real del problema operativo subyacente.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

## **CAUSA ÚNICA IDENTIFICADA: CRISIS OPERATIVA DE MISHANDLING**

---

## **A. NATURALEZA DE LA CAUSA**

### **Hipótesis Causal:**
**Crisis masiva en el manejo de equipajes y carga** que se manifestó como un incremento del 52% en la tasa de mishandling (de 17.9% a 27.2%), generando una degradación sistémica de la experiencia del cliente que impactó diferencialmente según el perfil y expectativas de cada segmento, pero que paradójicamente no escaló a reportes formales debido a su naturaleza "rutinaria" percibida.

### **Mecanismo de Impacto:**
- **Operativo**: Falla en procesos de ground handling, baggage handling y/o cargo handling
- **Experiencial**: Deterioro directo de la experiencia básica del cliente (equipaje perdido/dañado/retrasado)
- **Psicológico**: Mayor impacto en segmentos con expectativas altas (Business) y menor tolerancia (Leisure en ciertos contextos)

---

## **B. EVIDENCIA CONSOLIDADA Y ALCANCE**

### **Segmento Más Grande Afectado: GLOBAL**
*Incluye todos los subsegmentos: SH Economy/Business (IB/YW), LH Business/Premium*

### **Output Causal Detallado:**

**ANOMALÍA DETECTADA**: -2.28 pts (NPS 24.95 vs baseline 27.23)  
**CAUSA PRINCIPAL**: Problemas operativos concentrados en segmentos específicos  
**IMPACTO**: 513 encuestas analizadas con alta dispersión entre perfiles  

#### **Factor Primario: Mishandling Crítico**
- **Métrica**: 27.2% vs 17.9% promedio (**+9.3 pts** - aumento dramático del 52%)
- **Impacto**: Principal driver de la degradación de experiencia
- **Correlación**: Directa con la caída de NPS en todos los subsegmentos

#### **Factor Secundario: Alta Ocupación**
- **Load Factor**: 88.3% vs 87.6% promedio (+0.7pts)
- **Efecto**: Amplificó los problemas de mishandling
- **Resultado**: Mayor presión operativa y peor servicio

### **Subsegmentos Afectados Confirmados:**
✅ **Global/SH/Economy/IB**: Mishandling 28.0% (+9.8pts), NPS -6.3pts  
✅ **Global/SH/Economy/YW**: Mishandling 24.9% (+7.8pts), NPS -4.50pts  
✅ **Global/SH/Business/IB**: Mishandling 28.0% (+9.8pts), NPS -40.77pts  
✅ **Global/SH/Business/YW**: Mishandling 24.9% (+7.8pts), NPS -2.64pts  
✅ **Global/LH/Business**: Mishandling 27.2% (+9.3pts), NPS +10.72pts (paradoja por resistencia)  
✅ **Global/LH/Premium**: Sin impacto real (artefacto estadístico por volumen bajo)

---

## **MÉTRICAS CLAVE CONSOLIDADAS**

### **Indicadores Operativos:**
| Métrica | Valor 28/08 | Promedio 7d | Variación | Estado |
|---------|-------------|-------------|-----------|---------|
| **Mishandling** | 27.2% | 17.9% | **+9.3pts (+52%)** | 🔴 **CRÍTICO** |
| **Load Factor** | 88.3% | 87.6% | +0.7pts | 🟡 Elevado |
| **OTP** | 85.8% | 84.4% | +1.4pts | 🟢 Mejor |
| **Misconex** | 0.6% | 0.9% | -0.3pts | 🟢 Mejor |

### **Rutas Más Impactadas:**
- **BCN-MAD (SH)**: NPS -23.8 (n=21) - Epicentro de la crisis
- **EZE-MAD (LH)**: NPS 30.8 (n=26) - Impacto moderado pero significativo
- **LHR-MAD**: Múltiples referencias en segmentos Business

### **Segmentos Críticos:**
- **LATAM CodeShare**: NPS -33.3 (n=6) - Impacto devastador
- **VY CodeShare**: NPS -20.0 (n=10) - Alto impacto  
- **A33ACMI Flota**: NPS -19.0 (n=21) - Problemas específicos de flota

### **Verbatims Representativos:**
❌ **AUSENCIA CRÍTICA**: Cero verbatims disponibles en todos los segmentos
❌ **INCIDENTES NCS**: Cero reportes formales
**Interpretación**: **"Impacto Silencioso"** - Los clientes experimentaron problemas reales que impactaron su NPS pero no escalaron a quejas formales, típico en problemas de handling de equipaje percibidos como "molestias rutinarias".

---

## **CONCLUSIÓN CAUSAL**

La anomalía del 28 de agosto de 2025 se explica por una **causa única y sistémica**: una crisis operativa de mishandling que incrementó los problemas de equipaje/carga en un 52%, afectando a **todos los subsegmentos** pero con reactividad diferencial según el perfil del cliente. La ausencia total de reportes formales confirma la naturaleza "silenciosa" de este tipo de problemas operativos que impactan significativamente la satisfacción sin generar escalaciones proporcionales.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El análisis del 28 de agosto de 2025 reveló una crisis operativa sistémica que impactó negativamente múltiples segmentos, con el NPS global cayendo de 27.23 a 24.95 (-2.28 puntos). La causa raíz fue un incremento crítico del 52% en problemas de mishandling (de 17.9% a 27.2%), que afectó diferencialmente según el perfil de cliente y radio de operación. Los segmentos más impactados fueron Business Short Haul, que experimentó caídas devastadoras especialmente en IB (de 40.77 a 0.0, -40.77 puntos) y moderadas en YW (de 16.92 a 14.29, -2.64 puntos), mientras que Economy Short Haul mostró deterioros más contenidos tanto en IB (de 26.8 a 20.5, -6.3 puntos) como en YW (de 33.74 a 29.25, -4.50 puntos). Paradójicamente, Business Long Haul registró una mejora de 40.99 a 51.72 (+10.72 puntos) debido a la resistencia del segmento Business que compensó el impacto en clientes Leisure, mientras que Premium Long Haul mostró un artefacto estadístico por volumen crítico bajo.

Las rutas más severamente afectadas fueron BCN-MAD con un NPS catastrófico de -23.8 y EZE-MAD con impacto moderado (NPS 30.8), concentrando geográficamente los problemas operativos. Los grupos más reactivos incluyeron CodeShares LATAM (NPS -33.3) y VY (NPS -20.0), la flota A33ACMI (NPS -19.0), y especialmente clientes Business que mostraron hipersensibilidad extrema a los problemas de equipaje, mientras que los viajeros Leisure en ciertos contextos también experimentaron impactos significativos, creando una dispersión de hasta 157 puntos entre perfiles en algunos segmentos.

**ECONOMY SH IB: Deterioro por Crisis de Equipaje**
La cabina Economy SH IB experimentó un deterioro significativo durante el 28 de agosto, registrando un NPS de 20.5 con una caída de 6.3 puntos respecto al baseline de 26.8. La causa principal fue el incremento dramático del 54% en problemas de mishandling (de 18.2% a 28.0%), que generó un impacto disperso sin concentración geográfica específica pero con efectos sistémicos en la experiencia del cliente. Esta degradación se manifestó de forma silenciosa, sin generar reportes formales de incidentes NCS ni verbatims de clientes, sugiriendo que los problemas fueron percibidos como "molestias rutinarias" pero suficientemente impactantes para afectar significativamente la satisfacción.

**ECONOMY SH YW: Impacto Concentrado en CodeShares**
El segmento Economy SH YW registró un NPS de 29.25 con una reducción de 4.50 puntos vs el baseline de 33.74. Esta evolución se explica principalmente por el incremento del 46% en mishandling (de 17.1% a 24.9%), siendo especialmente visible en el CodeShare VY que experimentó un NPS devastador de -50.0, mientras que el CodeShare IB mantuvo un desempeño más resiliente con NPS 31.7, creando una dispersión extrema de 81.7 puntos entre perfiles que amplificó el impacto agregado del segmento.

**BUSINESS SH IB: Colapso Crítico por Problemas Operativos**
La cabina Business SH IB sufrió el impacto más severo, registrando un NPS de 0.0 con una caída catastrófica de 40.77 puntos desde el baseline de 40.77. La causa dominante fue la crisis de mishandling que escaló al 28.0% (+9.8 puntos vs promedio), generando una dispersión extrema de 133.3 puntos entre perfiles geográficos, donde clientes con residencia no identificada experimentaron NPS -100.0 mientras que América Norte mantuvo +33.3, evidenciando problemas específicos en sistemas de identificación y rutas internacionales complejas que afectaron desproporcionalmente a ciertos perfiles de viajeros business.

**BUSINESS SH YW: Impacto Moderado con Alta Volatilidad**
El segmento Business SH YW registró un NPS de 14.29 con una disminución de 2.64 puntos respecto al baseline de 16.92. Los drivers principales fueron el incremento del 46% en mishandling (de 17.1% a 24.9%) combinado con un volumen operativo extremadamente bajo de solo 14 encuestas, impactando especialmente a viajeros Leisure que registraron NPS 0.0 mientras que Business/Work mantuvo +50.0, creando una diferencia de 50 puntos que explica la volatilidad estadística amplificada por las circunstancias específicas del día.

**BUSINESS LH: Resistencia Paradójica ante Crisis Operativa**
La cabina Business LH mostró una mejora paradójica, registrando un NPS de 51.72 con un incremento de 10.72 puntos vs el baseline de 40.99. Los drivers principales fueron la resistencia excepcional del segmento Business (NPS 80.0) que compensó completamente el impacto del mishandling crítico (+9.3 puntos) en clientes Leisure (NPS 45.8), especialmente evidente en la ruta EZE-MAD y entre perfiles de viajeros de negocio que demostraron mayor tolerancia a problemas operativos comparado con otros segmentos, creando un efecto de compensación estadística único.

**PREMIUM LH: Estabilidad Aparente por Volumen Crítico**
El segmento Premium LH mantuvo desempeño aparentemente estable con un NPS de 100.0, aunque esta cifra representa un artefacto estadístico causado por volumen crítico de apenas 6-8 encuestas totales. No se detectaron impactos reales de los problemas operativos sistémicos, confirmando que los protocolos de servicio Premium proporcionaron aislamiento efectivo contra las disrupciones de mishandling que afectaron otros segmentos, manteniendo satisfacción perfecta entre los pocos clientes que operaron en este segmento durante el día analizado.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 11
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-08-27 to 2025-08-27:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 10 (Global, Global/LH/Business, Global/LH/Economy, Global/LH/Premium, Global/SH, Global/SH/Business/IB, Global/SH/Business/YW, Global/SH/Economy, Global/SH/Economy/IB, Global/SH/Economy/YW)

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)

## **A. ECONOMY SHORT HAUL**

### **Nodos Identificados:**
- ✅ **Economy SH/IB**: Anomalía negativa (-) presente
- ✅ **Economy SH/YW**: Anomalía negativa (-) presente

### **Análisis Comparativo:**

**Economy SH/IB:**
- Driver principal: **Mishandling 28.0%** (+9.8pts vs baseline)
- Impacto segmentado: América Norte NPS -50.0, BA Codeshare NPS -44.4
- Volumen: 153 encuestas

**Economy SH/YW:**
- Driver principal: **Mishandling 24.9%** (+7.8pts vs baseline) 
- Factor concentrador: **Flota CRJ** (81% volumen, NPS 19.0)
- Volumen: 97 encuestas

### **Diagnóstico:** 
**Causa común de cabina** - Ambas compañías experimentaron el mismo problema operativo (Mishandling elevado) con manifestaciones específicas por flota y geografía, pero sin divergencia fundamental en drivers.

---

## **B. BUSINESS SHORT HAUL**

### **Nodos Identificados:**
- ✅ **Business SH/IB**: Anomalía negativa (-) presente
- ✅ **Business SH/YW**: Anomalía negativa (-) presente

### **Análisis Comparativo:**

**Business SH/IB:**
- Driver principal: **Mishandling 28.0%** (+9.8pts vs baseline)
- Patrón crítico: Dispersión interna (Business/Work NPS 80.0 vs Leisure NPS 18.2)
- Volumen: 16 encuestas

**Business SH/YW:**
- Driver principal: **Mishandling 24.9%** (+7.8pts vs baseline)
- Patrón crítico: Volumen extremadamente bajo (29 encuestas) amplificando impacto
- Concentración en CRJ y clientes españoles

### **Diagnóstico:**
**Causa común de cabina** - Ambas compañías sufrieron el mismo problema operativo (Mishandling elevado), con diferencias en manifestación por volumen y segmentación, pero convergencia en el driver fundamental.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)

## **A. SHORT HAUL - Análisis de Cabinas**

### **Nodos Identificados:**
- ✅ **Economy SH**: Anomalía negativa (-) presente  
- ✅ **Business SH**: Anomalía negativa (-) presente

### **Análisis Comparativo:**

**Economy SH (IB + YW):**
- Driver operativo: **Mishandling 27.2-28.0%** 
- Reactividad: **Alta** - NPS cayó -4.53 a -7.97 puntos
- Patrón: Impacto directo y proporcional al problema operativo

**Business SH (IB + YW):**
- Driver operativo: **Mishandling 24.9-28.0%** (mismo problema)
- Reactividad: **Moderada** - NPS cayó -3.27 a -28.5 puntos  
- Patrón: **Segmentación interna crítica** - Protección de clientes Work vs impacto en Leisure

### **Diagnóstico Short Haul:**
**Causa común de radio con reactividad diferencial** - Ambas cabinas sufrieron el mismo problema operativo (Mishandling elevado), pero Business mostró mayor capacidad de proteger segmentos premium mientras Economy absorbió el impacto completo.

---

## **B. LONG HAUL - Análisis de Cabinas**

### **Nodos Identificados:**
- ✅ **Economy LH**: Anomalía negativa (-) presente
- ✅ **Business LH**: Anomalía positiva (+) presente  
- ✅ **Premium LH**: Anomalía negativa (-) presente

### **Análisis Comparativo:**

**Economy LH:**
- Driver operativo: **Mishandling 27.2%** (+9.3pts)
- Reactividad: **Muy Alta** - NPS cayó -6.9 puntos
- Concentración: EZE-MAD, CodeShares devastados

**Business LH:**
- Driver operativo: **Mishandling 27.2%** (mismo problema)
- Reactividad: **Inversa** - NPS subió +19.0 puntos (60.0 vs 41.0)
- Patrón: **Paradoja operativa** - Problemas operativos pero NPS positivo

**Premium LH:**
- Driver operativo: **Mishandling 27.2%** (mismo problema)  
- Reactividad: **Extrema** - NPS cayó -11.27 puntos
- Concentración: MAD-MEX, A350 next, clientes América Centro

### **Diagnóstico Long Haul:**
**Causa común de radio con reactividad extremadamente divergente** - Las tres cabinas experimentaron el mismo problema operativo (Mishandling 27.2%), pero con respuestas completamente opuestas: Economy y Premium colapsaron mientras Business mejoró, sugiriendo factores de protección/compensación específicos por clase de servicio.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

## **A. Comparación Entre Radios**

### **Nodos Identificados:**
- ✅ **Short Haul**: Anomalía negativa (-) presente
- ✅ **Long Haul**: Anomalía negativa (-) presente
- ✅ **Global**: Anomalía negativa (-) presente

### **Análisis de Convergencia:**

**Short Haul:**
- Driver principal: **Mishandling 27.2%** vs 17.9% baseline
- Magnitud: **-4.54 puntos NPS**
- Patrón: Impacto distribuido en múltiples segmentos

**Long Haul:**
- Driver principal: **Mishandling 27.2%** vs 17.9% baseline (**IDÉNTICO**)
- Magnitud: **-2.97 puntos NPS** 
- Patrón: Concentrado en rutas específicas (LIS-MAD, EZE-MAD, MAD-MEX)

### **Convergencia Confirmada:**
- ✅ **Mismo driver operativo**: Mishandling 27.2% en ambos radios
- ✅ **Mismo incremento**: +9.3pts vs baseline en ambos casos
- ✅ **Load Factor elevado**: Factor agravante común (87-92%)

---

## **B. Coherencia con Nodo Global**

### **Agregación Global:**
- **NPS Global**: 24.26 (-2.97pts vs baseline)
- **Driver Global**: **Mishandling 27.2%** (+9.3pts)
- **Patrón Global**: Impacto concentrado pero severo

### **Coherencia Matemática:**
- **Short Haul impacto**: -4.54pts (mayor volumen operativo)
- **Long Haul impacto**: -2.97pts (menor volumen pero rutas críticas)
- **Resultado Global**: -2.97pts (**coherente con agregación ponderada**)

### **Manifestación Global:**
El nodo Global refleja perfectamente la **convergencia de ambos radios** en el mismo problema operativo, con el impacto final determinado por la distribución de volumen entre Short y Long Haul.

---

## **DIAGNÓSTICO FINAL:**

**Causa global homogénea** - Ambos radios experimentaron exactamente el mismo problema operativo (Mishandling 27.2% vs 17.9% baseline) con convergencia total en drivers principales. El nodo Global refleja coherentemente la agregación ponderada de ambos impactos, confirmando que se trata de un problema sistémico de la compañía que afectó uniformemente toda la operación, independientemente del radio de vuelo.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

## **CAUSA PRINCIPAL: COLAPSO CRÍTICO EN MISHANDLING**

### **A. NATURALEZA DE LA CAUSA**

**Hipótesis:** El 27 de agosto de 2025 experimentó un **colapso sistémico en los procesos de manejo de equipajes** que afectó toda la operación de Iberia. El incremento del 52% en Mishandling (27.2% vs 17.9% baseline) sugiere una falla operativa concentrada en ground handling, posiblemente relacionada con:
- Problemas en sistemas de manejo de equipajes en hubs principales
- Sobrecarga operativa por alta ocupación de vuelos (Load Factor elevado)
- Fallas en coordinación con partners de CodeShare
- Posibles disrupciones en aeropuertos clave (MAD, EZE, LIS)

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE**

#### **Segmento Más Grande Afectado: Global/SH/Economy**
- **Volumen:** 250 encuestas (mayor representatividad)
- **Impacto:** -5.61 puntos NPS (23.6 vs 29.2 baseline)

#### **Output Causal Detallado:**
```
FACTOR OPERATIVO CRÍTICO: COLAPSO EN MISHANDLING
• Mishandling crítico: 27.2% vs 17.9% promedio (+9.3 puntos)
• Correlación directa: El incremento del 52% en Mishandling explica directamente la caída del NPS
• Impacto operativo: Problemas severos en ground handling/servicios auxiliares

FACTORES AGRAVANTES:
• Load Factor elevado: 88.6% vs 87.9% promedio (sobrecarga operativa)
• Paradoja operativa: Puntualidad mejorada (86.2% vs 84.9%) no compensó los problemas de servicio
```

#### **Alcance Total de la Causa:**
Esta causa afecta **TODOS los subsegmentos** identificados en el árbol:
- ✅ **Global/SH/Economy/IB**: Mishandling 28.0% (+9.8pts)
- ✅ **Global/SH/Economy/YW**: Mishandling 24.9% (+7.8pts)  
- ✅ **Global/SH/Business/IB**: Mishandling 28.0% (+9.8pts)
- ✅ **Global/SH/Business/YW**: Mishandling 24.9% (+7.8pts)
- ✅ **Global/LH/Economy**: Mishandling 27.2% (+9.3pts)
- ✅ **Global/LH/Business**: Mishandling 27.2% (+9.3pts)
- ✅ **Global/LH/Premium**: Mishandling 27.2% (+9.3pts)

#### **Métricas Clave Consolidadas:**
- **Indicador Operativo Principal:** Mishandling 27.2% (vs 17.9% baseline)
- **NPS Impacto Agregado:** -2.97 puntos a nivel Global
- **Load Factor Agravante:** 87-92% según segmento (sobrecarga operativa)
- **Volumen Total Afectado:** 600+ encuestas procesadas

#### **Rutas Críticas Involucradas:**
- **EZE-MAD:** Epicentro en Long Haul Economy (NPS 26.9, 26 encuestas)
- **LIS-MAD:** Impacto severo (NPS -25.0, n=4)
- **MAD-MEX:** Premium devastado (NPS -12.5 en A350 next)

#### **Segmentos Más Reactivos:**
- **CodeShares Internacionales:** Qatar Airways NPS -100.0, LATAM NPS -50.0, BA NPS -53.8
- **Clientes Premium:** América Centro NPS 0.0 (6 clientes)
- **Flota Específica:** A350 next, CRJ, A332 con impactos diferenciados

#### **Patrón de "Impacto Silencioso":**
- **❌ 0 incidentes NCS reportados** en todos los segmentos
- **❌ 0 verbatims disponibles** 
- **Característica:** Los clientes reflejaron su frustración inmediatamente en NPS sin escalar formalmente, sugiriendo que percibieron los problemas de equipaje como "inconvenientes operativos" más que incidentes que ameritaran queja formal.

---

## **CONCLUSIÓN CAUSAL**

El 27 de agosto de 2025 representa un **caso perfecto de falla operativa sistémica** donde un único problema (Mishandling +52%) generó impactos diferenciados pero consistentes a través de toda la operación de Iberia, afectando desde vuelos domésticos hasta rutas intercontinentales, con reactividad extrema en segmentos premium y CodeShares internacionales.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El 27 de agosto de 2025 experimentó una degradación sistémica del NPS causada por un colapso crítico en los procesos de manejo de equipajes que afectó toda la operación de Iberia. El NPS global cayó 2.97 puntos hasta 24.26, impulsado por un incremento del 52% en Mishandling (27.2% vs 17.9% baseline) que generó anomalías negativas en prácticamente todos los segmentos operativos. Los nodos más afectados incluyen Global/SH/Economy con caídas de -4.53 a -7.97 puntos según compañía, Global/LH/Premium con una severa caída de -11.27 puntos (14.29 vs 25.56), y Global/LH/Economy con -6.9 puntos (19.26 vs 19.81). Paradójicamente, Global/LH/Business mostró una mejora de +19.0 puntos (60.0 vs 41.0), sugiriendo factores de protección específicos para este segmento premium que compensaron los problemas operativos.

Las rutas más críticas incluyeron EZE-MAD como epicentro del deterioro en Long Haul, LIS-MAD con un NPS devastador de -25.0, y MAD-MEX donde la flota A350 next registró NPS de -12.5. Los grupos más reactivos fueron los pasajeros de CodeShare internacionales (Qatar Airways con NPS -100.0, LATAM -50.0, British Airways -53.8), clientes de América Centro que otorgaron NPS 0.0 en Premium, y pasajeros europeos en rutas intercontinentales. La característica más notable fue el patrón de "impacto silencioso" donde no se reportaron incidentes NCS formales ni verbatims de clientes, indicando que los problemas operativos impactaron directamente la satisfacción sin generar escalaciones formales.

**ECONOMY SH: Deterioro Operativo por Problemas de Equipaje**
La cabina Economy de Short Haul experimentó un deterioro significativo durante el 27 de agosto, con IB registrando un NPS de 22.22 (caída de -4.53 puntos) y YW alcanzando 25.77 (deterioro de -7.97 puntos). La causa principal fue el colapso en Mishandling que alcanzó 28.0% en IB y 24.9% en YW versus baselines de 18.2% y 17.1% respectivamente, evidenciado por la correlación directa entre estos incrementos operativos y las caídas de NPS. Esta degradación se reflejó especialmente en segmentos como América Norte (NPS -50.0 en IB), BA CodeShare (NPS -44.4 en IB), y la concentración crítica en flota CRJ para YW (NPS 19.0 con 81% del volumen), mientras que los perfiles más reactivos incluyeron pasajeros internacionales y clientes Leisure dentro del segmento Economy.

**BUSINESS SH: Impacto Diferenciado con Protección Parcial**
El segmento Business de Short Haul mostró deterioros moderados pero con patrones internos críticos, registrando IB un NPS de 37.5 (caída de -3.27 puntos) y YW un colapso más severo hasta NPS no especificado (deterioro de -28.5 puntos). Esta evolución se explica principalmente por el mismo incremento en Mishandling (28.0% en IB, 24.9% en YW), siendo especialmente visible la segmentación interna donde clientes Business/Work mantuvieron NPS 80.0 mientras Leisure colapsó a 18.2 en IB, y la concentración geográfica en España vs Europa para YW, demostrando que los perfiles Business corporativos estuvieron más protegidos ante los problemas operativos que los clientes Leisure.

**ECONOMY LH: Concentración Crítica en Rutas Intercontinentales**
La cabina Economy de Long Haul experimentó un deterioro significativo, registrando un NPS de 19.26 (caída de -6.9 puntos) el 27 de agosto. La causa principal fue el incremento crítico en Mishandling hasta 27.2% versus 17.9% de baseline, agravado por Load Factor elevado de 92.1%, evidenciado por la concentración total del impacto en la ruta EZE-MAD y el colapso devastador de CodeShares internacionales. Esta degradación se reflejó especialmente en rutas como EZE-MAD (NPS 26.9 con 26 encuestas) operada por flota A350, mientras que los perfiles más reactivos incluyeron CodeShare QR (NPS -100.0), CodeShare LATAM (NPS -40.0), y residentes europeos (NPS -54.5).

**BUSINESS LH: Paradoja Operativa con Mejora Inexplicada**
La cabina Business de Long Haul mantuvo una mejora paradójica, registrando un NPS de 60.0 (mejora de +19.0 puntos vs 41.0 baseline) el 27 de agosto, a pesar del mismo problema operativo de Mishandling 27.2%. Los drivers principales muestran una contradicción aparente donde los problemas de equipaje no impactaron negativamente este segmento, impactando especialmente las operaciones con alta dispersión entre flota A350 (NPS 25) versus A332 (NPS 100), y perfiles donde Business corporativo (NPS 33.3) mostró mayor criticidad que Leisure (NPS 66.7), sugiriendo factores de compensación o protección específicos no identificados en los datos disponibles.

**PREMIUM LH: Colapso Severo en Segmento Más Sensible**
El segmento Premium de Long Haul experimentó el deterioro más severo, registrando un NPS de 14.29 (caída devastadora de -11.27 puntos vs 25.56 baseline) el 27 de agosto. Las causas dominantes fueron el mismo colapso en Mishandling 27.2% combinado con Load Factor elevado de 91.5%, especialmente evidentes en la ruta MAD-MEX operada por A350 next donde los 6 clientes de América Centro otorgaron NPS 0.0, y entre perfiles Premium que mostraron la máxima sensibilidad a los problemas operativos de equipaje, confirmando que los clientes de mayor valor son los más reactivos a disrupciones operativas.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 10
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-08-26 to 2025-08-26:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 10 (Global, Global/LH, Global/LH/Business, Global/LH/Economy, Global/SH, Global/SH/Business, Global/SH/Business/IB, Global/SH/Business/YW, Global/SH/Economy, Global/SH/Economy/YW)

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 PASO 1: DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)

## 📊 ANÁLISIS SISTEMÁTICO

### **A. Economy Short Haul:**
- ✅ **Existe nodo Economy SH/YW**: Anomalía negativa (-6.7 pts)
- ❌ **NO existe nodo Economy SH/IB**: Ausente en el árbol de análisis

**Razón de la ausencia**: El nodo Economy SH/IB no presenta anomalía significativa, por lo que no fue incluido en el análisis de nodos anómalos.

### **B. Business Short Haul:**
- ✅ **Existe nodo Business SH/IB**: Anomalía positiva (+14.2 pts)
- ✅ **Existe nodo Business SH/YW**: Anomalía negativa (-50.26 pts)

**Comparación de patrones entre IB y YW:**

| Aspecto | Business SH/IB | Business SH/YW |
|---------|----------------|----------------|
| **Driver principal** | Mishandling +9.8pts | Mishandling +7.8pts |
| **Magnitud del problema** | 28.0% vs 18.2% | 24.9% vs 17.1% |
| **Impacto diferencial** | Business devastado (NPS 0.0) vs Leisure resiliente (68.8) | Muestra pequeña extrema (9 encuestas) |
| **Evidencia operativa** | Problemas silenciosos sin escalamiento | Problemas silenciosos sin escalamiento |

## 🎯 DIAGNÓSTICO FINAL

### **Economy Short Haul:**
**Causa indeterminada por datos insuficientes** - Solo YW presenta anomalía, IB no tiene nodo anómalo para comparar.

### **Business Short Haul:**
**Causa común de cabina** - Tanto IB como YW convergen en el mismo driver principal (Mishandling elevado) con patrones operativos idénticos, diferenciándose solo en la magnitud del impacto debido a factores de volumetría y composición de clientes.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ PASO 2: DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)

## 📊 ANÁLISIS SISTEMÁTICO

### **A. Short Haul - Análisis de Cabinas:**

**Nodos anómalos identificados:**
- ❌ **Economy SH**: Solo YW presenta anomalía (-6.7 pts)
- ❌ **Business SH**: Ambas compañías anómalas pero direcciones opuestas (IB: +14.2 pts, YW: -50.26 pts)

**Comparación de patrones:**

| Aspecto | Economy SH/YW | Business SH/IB | Business SH/YW |
|---------|---------------|----------------|----------------|
| **Driver principal** | Mishandling +7.8pts | Mishandling +9.8pts | Mishandling +7.8pts |
| **Dirección anomalía** | Negativa (-6.7) | Positiva (+14.2) | Negativa (-50.26) |
| **Reactividad** | Moderada | Alta diferencial | Extrema por muestra |

**Consistencia entre compañías:** En Business, ambas IB y YW convergen en Mishandling como causa raíz, pero divergen en la manifestación del impacto.

### **B. Long Haul - Análisis de Cabinas:**

**Nodos anómalos identificados:**
- ❌ **Economy LH**: Anomalía negativa (-10.2 pts)
- ❌ **Business LH**: Anomalía negativa (-34.75 pts)
- ❌ **Premium LH**: No presenta nodo anómalo

**Comparación de patrones:**

| Aspecto | Economy LH | Business LH |
|---------|------------|-------------|
| **Driver principal** | Mishandling +9.3pts | Mishandling +9.3pts |
| **Magnitud operativa** | 27.2% vs 17.9% | 27.2% vs 17.9% |
| **Reactividad** | Moderada (-10.2) | Severa (-34.75) |
| **Concentración** | Flota A321XLR/A350 | Ruta JFK-MAD específica |

**Progresión de reactividad:** Economy < Business (Premium no afectado)

## 🎯 DIAGNÓSTICO FINAL

### **Short Haul:**
**Causa común de radio con reactividad diferencial por cabina** - Todas las cabinas anómalas convergen en Mishandling como driver principal, pero Business muestra mayor sensibilidad y variabilidad en el impacto que Economy.

### **Long Haul:**
**Causa común de radio con progresión lógica de reactividad** - Economy y Business convergen completamente en el mismo problema operativo (Mishandling +9.3pts), pero Business presenta reactividad 3.4x mayor que Economy, mientras Premium actúa como "amortiguador" sin verse afectado.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 PASO 3: DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

## 📊 ANÁLISIS SISTEMÁTICO

### **A. Comparación Entre Radios:**

**Estado de los nodos:**
- ❌ **Short Haul**: Anomalía negativa (-1.01 pts)
- ❌ **Long Haul**: Anomalía negativa (-11.52 pts)
- ❌ **Global**: Anomalía negativa (-4.60 pts)

**Convergencia de drivers principales:**

| Radio | Driver Principal | Magnitud | Evidencia Operativa |
|-------|------------------|----------|-------------------|
| **Short Haul** | Mishandling | 27.2% vs 17.9% (+9.3pts) | Ruta BUD-MAD, flota ATR/CRJ |
| **Long Haul** | Mishandling | 27.2% vs 17.9% (+9.3pts) | Flota A321XLR, ruta MAD-NRT/JFK-MAD |
| **Global** | Mishandling | 27.2% vs 17.9% (+9.3pts) | A321XLR colapso total, MAD-NRT |

**Consistencia de valores SHAP:** ✅ **PERFECTA CONVERGENCIA** - Ambos radios muestran exactamente el mismo incremento de Mishandling (+9.3pts) como factor causal principal.

### **B. Coherencia con Nodo Global:**

**Análisis de impacto agregado:**
- **Global NPS**: -4.60 pts (intermedio entre SH: -1.01 y LH: -11.52)
- **Ponderación lógica**: El impacto Global refleja una combinación ponderada donde LH tiene mayor peso relativo
- **Driver consistente**: Global confirma Mishandling como causa raíz principal
- **Concentración específica**: Global identifica A321XLR como epicentro (flota LH)

**Patrón de amplificación/atenuación:**
- ✅ **No hay cancelación de efectos**: Ambos radios negativos se suman
- ✅ **Amplificación coherente**: LH domina el impacto global por mayor severidad
- ✅ **Convergencia causal**: Todos los niveles apuntan al mismo problema operativo

## 🎯 DIAGNÓSTICO FINAL

**Causa global homogénea con concentración diferencial** - Ambos radios (SH y LH) convergen perfectamente en el mismo driver causal (Mishandling +9.3pts) y evidencia operativa idéntica, pero Long Haul presenta mayor reactividad (11.5x más severo que Short Haul). El nodo Global refleja coherentemente esta causa única distribuida, con el impacto dominado por la mayor sensibilidad de operaciones de largo radio, particularmente concentrada en la flota A321XLR.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 PASO 4: ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

## 📋 CAUSA IDENTIFICADA: CRISIS DE MANEJO DE EQUIPAJE (MISHANDLING)

### **A. NATURALEZA DE LA CAUSA:**

**Hipótesis causal:** Crisis operativa sistémica en los procesos de manejo de equipaje el 26 de agosto de 2025, donde la tasa de Mishandling se disparó un 52% por encima del promedio normal (27.2% vs 17.9%), generando una degradación masiva pero "silenciosa" de la experiencia del cliente que impactó desproporcionadamente según el perfil de viajero y tipo de operación.

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**

**Segmento más grande afectado:** **GLOBAL** (nodo padre que engloba toda la operación)

#### **Output Causal Detallado:**

**PROBLEMA PRINCIPAL:** 
- **Mishandling crítico:** 27.2% vs 17.9% promedio (+9.3 pts, +52% incremento)
- **Flota más afectada:** A321XLR con NPS -100.0 (colapso total)
- **Concentración geográfica:** Rutas MAD-NRT y JFK-MAD como epicentros

**ALCANCE TOTAL:** Esta causa afecta a **todos los subsegmentos** bajo el nodo Global, incluyendo:
- ✅ Long Haul (Economy y Business)
- ✅ Short Haul (Economy YW y Business IB/YW)
- ✅ Todas las compañías (IB/YW)
- ✅ Todos los perfiles de cliente (Business/Leisure)
- ✅ Todas las regiones operativas

#### **Métricas Clave Consolidadas:**

| **Indicador** | **Valor del Día** | **Promedio** | **Impacto** |
|---------------|-------------------|--------------|-------------|
| **Mishandling** | 27.2% | 17.9% | **+9.3 pts (+52%)** |
| **NPS Global** | 22.63 | 27.23 | **-4.60 pts** |
| **Load Factor** | 88.3% | 87.6% | +0.6 pts (presión operativa) |
| **OTP** | 85.8% | 84.4% | +1.4 pts (no correlaciona) |

#### **Rutas Críticas Involucradas:**
- **MAD-NRT (Madrid-Narita):** NPS -16.7, flota A321XLR
- **JFK-MAD (Nueva York-Madrid):** NPS -100.0, 3 clientes Business
- **BUD-MAD (Budapest-Madrid):** NPS 22.2, operación SH
- **MAD-NAP (Madrid-Nápoles):** NPS 16.7, flota A332

#### **Impacto Diferencial por Segmento:**
- **Business LH:** -34.75 pts (mayor reactividad)
- **Economy LH:** -10.2 pts (impacto moderado)
- **Business SH:** Variabilidad extrema (IB: +14.2, YW: -50.26)
- **Economy SH:** -6.7 pts (menor reactividad)

#### **Características del Problema:**
- **"Crisis Silenciosa":** 0 incidentes NCS reportados, 0 verbatims disponibles
- **Problemas operativos reales** sin escalamiento formal
- **Concentración en flotas específicas:** A321XLR, CRJ, A332
- **Mayor impacto en viajeros Business** vs Leisure
- **Problemas "tolerables"** que degradaron satisfacción sin generar quejas oficiales

#### **Patrón de Manifestación:**
La crisis de Mishandling se manifestó como una **degradación sistemática pero no catastrófica** del servicio, suficientemente significativa para impactar el NPS pero no lo suficientemente dramática para generar quejas formales masivas, evidenciando una **desconexión crítica** entre sistemas de monitoreo tradicionales y experiencia real del cliente.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

# 📈 SÍNTESIS EJECUTIVA

El 26 de agosto de 2025 se registró una degradación generalizada del NPS a nivel global, cayendo de 27.23 a 22.63 puntos (-4.60 puntos), causada por una crisis operativa de manejo de equipaje que elevó el Mishandling del 17.9% al 27.2% (+9.3 puntos porcentuales). Esta crisis impactó de manera diferencial según el tipo de operación: Long Haul experimentó el mayor deterioro, pasando de 22.29 a 10.77 puntos (-11.52 puntos), mientras que Short Haul mostró una caída más moderada de 29.8 a 28.8 puntos (-1.01 puntos). La anomalía se concentró especialmente en la flota A321XLR que colapsó completamente con un NPS de -100.0, y en operaciones específicas como las rutas MAD-NRT y JFK-MAD que registraron NPS extremadamente negativos.

Las rutas más severamente afectadas incluyen JFK-MAD con un NPS de -100.0 impactando a viajeros de negocios transatlánticos, MAD-NRT con NPS de -16.7 operada por A321XLR, y BUD-MAD con NPS de 22.2 en operaciones Short Haul. Los grupos de clientes más reactivos fueron consistentemente los viajeros de negocios, que mostraron una sensibilidad 3.4 veces mayor que los viajeros de ocio ante los problemas operativos, especialmente en rutas intercontinentales donde las expectativas de servicio son más elevadas y la tolerancia a disrupciones es menor.

**ECONOMY SH YW: Deterioro Operativo Moderado**
La cabina Economy de Short Haul YW experimentó un deterioro de 6.7 puntos, pasando de un NPS de 33.7 a 27.0 el 26 de agosto. La causa principal fue el incremento del Mishandling del 17.1% al 24.9% (+7.8 puntos porcentuales), combinado con un Load Factor elevado del 86.1% que generó presión operativa adicional. Este deterioro se concentró especialmente entre clientes europeos que registraron un NPS negativo de -5.9, mientras que la única ruta con datos disponibles, PMI-VGO, reflejó el impacto generalizado de los problemas de equipaje en operaciones de corto radio.

**BUSINESS SH IB: Mejora Aparente con Crisis Subyacente**
El segmento Business de Short Haul IB mostró una mejora aparente de 14.2 puntos, alcanzando un NPS de 55.0 vs su baseline de 40.8. Sin embargo, esta mejora enmascara una crisis operativa crítica donde el Mishandling se disparó del 18.2% al 28.0% (+9.8 puntos), devastando completamente a los viajeros Business/Work con un NPS de 0.0, mientras que los viajeros Leisure mantuvieron un NPS resiliente de 68.8. La concentración en la ruta LHR-MAD y la diferencia de 68.8 puntos entre perfiles de cliente evidencia la alta sensibilidad de los viajeros corporativos a problemas operativos.

**BUSINESS SH YW: Colapso Severo por Muestra Crítica**
La cabina Business de Short Haul YW sufrió un colapso severo de 50.26 puntos, cayendo de 16.9 a -33.3 puntos debido al incremento del 45% en Mishandling (del 17.1% al 24.9%) combinado con una muestra extremadamente pequeña de solo 9 encuestas. Esta caída se concentró geográficamente en España con un NPS de -20.0 y operativamente en la flota CRJ, donde cada experiencia negativa tuvo un peso desproporcionado del 11% en el NPS total, amplificando dramáticamente el impacto de los problemas operativos de equipaje.

**ECONOMY LH: Degradación Sistémica Concentrada**
La cabina Economy de Long Haul experimentó un deterioro significativo de 10.2 puntos, pasando de 19.8 a 9.6 puntos el 26 de agosto. La causa principal fue el mismo incremento crítico de Mishandling del 17.9% al 27.2% (+9.3 puntos) que afectó toda la operación, con concentración específica en la flota A321XLR que registró un NPS de -100.0 y en la flota A350 con NPS de -2.9. Esta degradación se reflejó especialmente en rutas como MAD-MEX con NPS de 27.3 y entre clientes españoles que representaron el mayor volumen afectado con 50 encuestas.

**BUSINESS LH: Crisis Operativa Transatlántica**
La cabina Business de Long Haul sufrió el mayor deterioro con una caída de 34.75 puntos, cayendo dramáticamente de 40.99 a 6.25 puntos. Esta crisis se explica por el mismo factor de Mishandling elevado (27.2% vs 17.9%) pero con una reactividad 3.4 veces mayor que Economy, concentrándose específicamente en la ruta JFK-MAD que registró un NPS de -100.0 entre 3 clientes de negocios. Los viajeros de América del Norte mostraron la mayor sensibilidad con un NPS de -50.0, mientras que los viajeros Business/Work fueron completamente devastados comparado con los Leisure que mantuvieron un NPS de 25.0.

**PREMIUM LH: Estabilidad Resiliente**
El segmento Premium de Long Haul mantuvo desempeño estable durante el 26 de agosto, sin registrar anomalías significativas y actuando como "amortiguador" ante los problemas operativos que afectaron severamente a Economy y Business. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción a pesar de la crisis generalizada de Mishandling que impactó al resto de la operación.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 10
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-08-25 to 2025-08-25:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 7 (Global, Global/LH/Business, Global/LH/Premium, Global/SH/Business/IB, Global/SH/Business/YW, Global/SH/Economy, Global/SH/Economy/YW)

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 **DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)**

## **A. ECONOMY SHORT HAUL:**

**Nodos identificados:**
- ✅ **Economy SH/YW**: Existe (Global/SH/Economy/YW) - Anomalía negativa
- ❌ **Economy SH/IB**: No existe en el árbol

**Razón de ausencia SH/IB:** El nodo Economy SH/IB no presenta anomalía significativa el 25 de agosto, por lo que no fue incluido en el análisis de anomalías.

**Diagnóstico:** **Causa específica de compañía YW** - Solo YW experimentó problemas significativos en Economy SH, indicando que los issues operativos (Mishandling +7.8pts) impactaron específicamente las operaciones de esta compañía.

---

## **B. BUSINESS SHORT HAUL:**

**Nodos identificados:**
- ✅ **Business SH/IB**: Existe (Global/SH/Business/IB) - Anomalía negativa
- ✅ **Business SH/YW**: Existe (Global/SH/Business/YW) - Anomalía negativa

**Análisis comparativo de patrones:**

| Factor | SH/Business/IB | SH/Business/YW |
|--------|----------------|----------------|
| **Driver Principal** | Mishandling +9.8pts (+54%) | Mishandling +7.8pts (+45.6%) |
| **Magnitud NPS** | -7.6 pts | -33.59 pts |
| **Factor Específico** | Problemas en Fleet A321 | Impacto en segmento Business/Work |
| **Patrón** | "Problemas silenciosos" | Sensibilidad alta pasajeros business |

**Diagnóstico:** **Causa común de cabina con manifestaciones específicas por compañía** - Ambas compañías (IB y YW) convergieron en el mismo driver principal (Mishandling crítico), pero con diferentes intensidades y segmentos afectados, indicando un problema operativo compartido a nivel de cabina Business SH.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ **DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)**

## **A. SHORT HAUL - ANÁLISIS DE CABINAS:**

**Nodos identificados:**
- ✅ **Economy SH**: Anomalía negativa (solo YW presente)
- ✅ **Business SH**: Anomalía negativa (IB y YW presentes)

**Análisis de convergencia/divergencia:**

| Factor | Economy SH | Business SH |
|--------|------------|-------------|
| **Estado de Anomalía** | Negativa (-) | Negativa (-) |
| **Driver Principal** | Mishandling +7.8pts | Mishandling +9.8pts (IB) / +7.8pts (YW) |
| **Magnitud Relativa** | Moderada (-0.20 a -7.6 pts) | Severa (-7.6 a -33.59 pts) |
| **Patrón Operativo** | Problemas "silenciosos" | Problemas "silenciosos" + sensibilidad alta |

**Consistencia entre compañías:** En Business SH, tanto IB como YW convergieron en el mismo driver (Mishandling) con intensidades similares, confirmando patrón común.

**Diagnóstico Short Haul:** **Causa común de radio con reactividad diferencial por cabina** - Ambas cabinas experimentaron el mismo problema operativo (Mishandling crítico), pero Business SH mostró mayor sensibilidad y magnitud de impacto que Economy SH, indicando diferente tolerancia del cliente premium a problemas de servicio.

---

## **B. LONG HAUL - ANÁLISIS DE CABINAS:**

**Nodos identificados:**
- ❌ **Economy LH**: No presente (sin anomalías significativas)
- ✅ **Business LH**: Anomalía positiva (+9.0 pts)
- ✅ **Premium LH**: Anomalía positiva (+17.30 pts)

**Análisis de progresión por clase de servicio:**

| Factor | Business LH | Premium LH |
|--------|-------------|------------|
| **Estado de Anomalía** | Positiva (+) | Positiva (+) |
| **Driver Operativo** | Mishandling +9.3pts (paradoja) | Mishandling +9.3pts (paradoja) |
| **Magnitud NPS** | +9.0 pts | +17.30 pts |
| **Patrón** | Concentración EZE-MAD | Impacto América Sur devastador |

**Progresión lógica:** Se observa escalamiento de magnitud positiva de Business (+9.0) a Premium (+17.30), pero ambos con la misma paradoja operativa.

**Diagnóstico Long Haul:** **Causa común de radio con progresión lógica de reactividad** - Tanto Business como Premium LH convergieron en el mismo driver paradójico (Mishandling alto pero NPS mejorado), con Premium mostrando mayor magnitud de mejora, sugiriendo un problema operativo común que afectó de manera inversa la percepción según el nivel de servicio esperado.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 **DIAGNÓSTICO A NIVEL RADIO Y GLOBAL**

## **A. COMPARACIÓN ENTRE RADIOS:**

**Estados de anomalía por radio:**
- ✅ **Short Haul**: Afectado (anomalías negativas en Economy y Business)
- ✅ **Long Haul**: Afectado (anomalías positivas en Business y Premium)

**Análisis de convergencia de drivers:**

| Factor | Short Haul | Long Haul |
|--------|------------|-----------|
| **Estado General** | Negativo (-) | Positivo (+) |
| **Driver Principal** | Mishandling +7.8 a +9.8pts | Mishandling +9.3pts |
| **Dirección del Impacto** | NPS ↓ (empeora) | NPS ↑ (mejora paradójica) |
| **Evidencia Operativa** | Problemas "silenciosos" | Concentración geográfica específica |

**Convergencia identificada:** Ambos radios convergieron en el **mismo driver operativo** (Mishandling crítico), pero con **direcciones opuestas** de impacto en NPS.

---

## **B. COHERENCIA CON NODO GLOBAL:**

**Análisis del nodo Global:**
- **Estado**: Anomalía negativa (-1.98 pts)
- **Driver**: Mishandling +9.3pts (consistente con ambos radios)
- **Patrón**: "Crisis de servicio focalizada y silenciosa"

**Coherencia agregada:**
- El Global refleja el **impacto neto** donde las anomalías negativas de SH superaron las positivas de LH
- La magnitud (-1.98 pts) es coherente con un efecto de **compensación parcial** entre radios
- El driver operativo (Mishandling) es **completamente consistente** entre Global, SH y LH

---

## **DIAGNÓSTICO FINAL:**

**Causa global homogénea con manifestaciones diferenciadas por radio** - El problema operativo de Mishandling crítico (+9.3pts) afectó simultáneamente ambos radios con el mismo driver, pero SH experimentó deterioro del NPS (problemas silenciosos) mientras LH mostró mejora paradójica (concentración geográfica específica), resultando en un impacto global neto negativo que refleja la suma algebraica de efectos opuestos pero causalmente relacionados.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 **ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS**

## **CAUSA ÚNICA IDENTIFICADA: CRISIS CRÍTICA DE MISHANDLING**

### **A. NATURALEZA DE LA CAUSA:**

**Hipótesis Causal:** El 25 de agosto de 2025 experimentó una **falla sistémica en los procesos de manejo de equipajes y servicios ground** que se manifestó como problemas operativos "silenciosos" - suficientemente graves para devastar la experiencia del cliente pero no lo suficientemente visibles para generar alertas formales o escalamientos masivos de quejas. Esta falla tuvo un **impacto diferenciado por tipo de cliente**: deterioró la experiencia en Short Haul (donde los clientes tienen menor tolerancia a problemas de servicio básico) pero paradójicamente mejoró la percepción en Long Haul (donde problemas menores fueron compensados por otros factores de servicio premium).

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**

**Segmento más grande afectado:** **GLOBAL** (nodo padre que incluye todos los subsegmentos)

**Output causal detallado para segmento Global:**

#### **🔴 FACTOR PRINCIPAL: MISHANDLING CRÍTICO**
- **Magnitud operativa**: 27.2% vs 17.9% promedio (+9.3 puntos porcentuales)
- **Incremento relativo**: +52% sobre el baseline
- **Impacto en NPS**: -1.98 puntos vs baseline
- **Correlación**: Directa y consistente en todos los subsegmentos

#### **📊 MÉTRICAS OPERATIVAS CONSOLIDADAS:**
| Métrica | Valor 25/08 | Promedio 7d | Diferencia | Impacto |
|---------|-------------|-------------|------------|---------|
| **Mishandling** | 27.2% | 17.9% | **+9.3pts** | 🔴 Crítico |
| **Load Factor** | 88.3% | 87.6% | +0.6pts | 🟡 Moderado |
| **OTP15** | 85.8% | 84.4% | +1.4pts | 🟢 Positivo |
| **Misconex** | 0.6% | 0.9% | -0.3pts | 🟢 Mejor |

#### **✈️ RUTAS Y AERONAVES CRÍTICAS IDENTIFICADAS:**
- **Ruta más impactada**: GRX-MAD con NPS -33.3 (devastador)
- **Aeronave crítica**: Fleet A350 C con NPS -71.4 (peor desempeño absoluto)
- **Ruta Long Haul**: EZE-MAD concentró problemas en segmento premium
- **Flota Short Haul**: A321 experimentó problemas específicos (NPS 16.7 vs A320 con 60.0)

#### **👥 SEGMENTOS ESPECÍFICOS DEVASTADOS:**
- **América Centro**: NPS -25.0 (impacto devastador)
- **América Norte**: NPS -46.2 a -66.7 (múltiples segmentos)
- **Pasajeros Business/Work**: Sensibilidad extrema a problemas de equipaje
- **CodeShare operations**: BA (-60.0), AA (-35.7) severamente afectados

#### **🔇 PATRÓN "SILENCIOSO" SISTEMÁTICO:**
- **Cero incidentes NCS reportados** en todos los segmentos
- **Ausencia total de verbatims** disponibles
- **Sin alertas operativas** en tiempo real
- **Desconexión crítica** entre problemas reales y sistemas de detección

#### **ALCANCE CONFIRMADO:**
Esta causa afecta **todos los subsegmentos** bajo el nodo Global, incluyendo:
- **Short Haul**: Economy/YW, Business/IB, Business/YW
- **Long Haul**: Business, Premium  
- **Todas las compañías**: IB, YW, CodeShares (AA, BA, IB)
- **Múltiples geografías**: América Norte, Centro, Sur, España, Europa
- **Todas las flotas**: A321, A320, A350

#### **🎯 CONCLUSIÓN CAUSAL:**
La crisis de Mishandling del 25 de agosto representa una **falla sistémica global** que impactó toda la operación con manifestaciones específicas por segmento: problemas "silenciosos" pero devastadores que expusieron vulnerabilidades críticas en los sistemas de detección temprana y gestión de calidad del servicio a nivel empresarial.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

# 📈 SÍNTESIS EJECUTIVA

El 25 de agosto de 2025 experimentó una crisis sistémica de manejo de equipajes que impactó de manera diferenciada los segmentos de la compañía. A nivel global, el NPS cayó 1.98 puntos debido principalmente a problemas críticos de Mishandling que aumentaron del 17.9% al 27.2% (+9.3 puntos porcentuales). Esta crisis afectó negativamente los segmentos Short Haul, donde Economy YW registró caídas de hasta 7.6 puntos (de 33.7 a 26.1) y Business experimentó deterioros severos con IB cayendo 7.6 puntos (de 50.0 a 42.4) y YW desplomándose 33.59 puntos (de 16.92 a -16.67). Paradójicamente, Long Haul mostró mejoras significativas con Business subiendo 9.0 puntos (de 41.0 a 50.0) y Premium escalando 17.30 puntos (de 25.56 a 42.86), evidenciando que los mismos problemas operativos generaron reacciones opuestas según el tipo de servicio.

Las rutas más críticas incluyen GRX-MAD con un NPS devastador de -33.3, EZE-MAD que concentró problemas en segmentos premium, y MAD-MXP que registró -7.7 en Economy. Los grupos más reactivos fueron pasajeros de América del Norte (NPS hasta -66.7), América Centro (-25.0), y clientes Business/Work que mostraron sensibilidad extrema a los problemas de equipaje. La crisis se caracterizó por ser "silenciosa" - sin incidentes NCS reportados ni verbatims disponibles - pero con impacto devastador concentrado en aeronaves específicas como el A350 C (NPS -71.4) y A321 (NPS 16.7).

**ECONOMY SH YW: Crisis de Servicio Focalizada**
La cabina Economy YW de Short Haul experimentó un deterioro significativo durante el 25 de agosto, registrando un NPS de 26.1 con una caída de 7.6 puntos respecto al baseline de 33.7. La causa principal fue el incremento crítico del Mishandling del 17.1% al 24.9% (+7.8 puntos porcentuales), que impactó desproporcionadamente a pasajeros norteamericanos quienes mostraron un NPS de -66.7 versus +50.0 de pasajeros españoles. Esta crisis se reflejó especialmente en la ruta MAD-MXP con NPS de -7.7, mientras que los perfiles más reactivos incluyeron clientes de América Norte y operaciones CodeShare BA que registraron hasta -60.0 de NPS.

**BUSINESS SH IB: Problemas Silenciosos en Flota Específica**
El segmento Business IB de Short Haul registró un NPS de 42.4 con una caída de 7.6 puntos vs el baseline de 50.0. Esta evolución se explica principalmente por problemas críticos de Mishandling que aumentaron del 18.2% al 28.0% (+9.8 puntos, +54% incremento), siendo especialmente visible en aeronaves A321 que registraron un NPS devastador de 16.7 versus 60.0 en A320, y entre perfiles de pasajeros europeos (NPS 30.0) versus españoles (NPS 58.3).

**BUSINESS SH YW: Colapso en Segmento Premium**
La cabina Business YW de Short Haul experimentó el mayor deterioro con un NPS de -16.67, cayendo dramáticamente 33.59 puntos desde un baseline de 16.92. Los drivers principales fueron los problemas de Mishandling que escalaron del 17.1% al 24.9% (+7.8 puntos porcentuales), impactando especialmente a pasajeros Business/Work que registraron -33.3 de NPS versus 0.0 para Leisure, evidenciando la alta sensibilidad del segmento corporativo a disrupciones operativas.

**ECONOMY LH: Desempeño Estable**
La cabina Economy de Long Haul mantuvo desempeño estable durante el 25 de agosto. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción sin anomalías que requieran análisis específico.

**BUSINESS LH: Mejora Paradójica Concentrada**
La cabina Business de Long Haul registró un NPS de 50.0 con una mejora de 9.0 puntos respecto al baseline de 41.0. Los drivers principales fueron paradójicamente los mismos problemas de Mishandling (+9.3 puntos) que afectaron negativamente a Short Haul, impactando especialmente la ruta EZE-MAD con concentración de 6 respuestas y NPS de 83.3, y perfiles de América Sur que alcanzaron NPS de 100.0 versus América Centro con -25.0.

**PREMIUM LH: Escalamiento de Mejora Premium**
El segmento Premium de Long Haul experimentó la mayor mejora con un NPS de 42.86, subiendo 17.30 puntos vs el baseline de 25.56. Las causas dominantes fueron los mismos factores de Mishandling (+9.3 puntos) que generaron efectos diferenciados por geografía, especialmente evidentes entre pasajeros de América Sur (NPS 0.0 para 4 encuestas) y perfiles A350 (NPS 25.0) versus CodeShare IB (NPS 50.0).

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 7
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-08-24 to 2025-08-24:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 8 (Global, Global/LH, Global/LH/Business, Global/LH/Economy, Global/LH/Premium, Global/SH/Business/YW, Global/SH/Economy/IB, Global/SH/Economy/YW)

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 PASO 1: DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)

## 📊 ANÁLISIS SISTEMÁTICO

### **A. Economy Short Haul:**

**Nodos Identificados:**
- ✅ **Global/SH/Economy/IB**: Anomalía NEGATIVA (-5.5 pts)
- ✅ **Global/SH/Economy/YW**: Anomalía POSITIVA (+14.3 pts)

**Comparación de Patrones:**

| Aspecto | IB (Negativo) | YW (Positivo) | Convergencia |
|---------|---------------|---------------|--------------|
| **Driver Principal** | Mishandling 28.0% (+9.8pts) | Mishandling 24.9% (+7.8pts) | ✅ **MISMO DRIVER** |
| **Evidencia Operativa** | Load Factor normal, OTP mejorado | Load Factor alto (+2.3pts), OTP estable | ⚠️ Diferencias menores |
| **Segmentos Afectados** | CodeShare LATAM (-75.0), A33ACMI (-40.0) | Business/Work (11.8), ATR Fleet (21.1) | ❌ **DIFERENTES SEGMENTOS** |
| **Patrón NCS/Verbatims** | Sin reportes formales | Sin reportes formales | ✅ **MISMO PATRÓN** |

**DIAGNÓSTICO ECONOMY SH:** Causa común de cabina con **reactividad diferencial por composición de muestra** - ambas compañías sufren el mismo problema operativo (mishandling elevado) pero con impacto direccional opuesto debido a diferentes perfiles de clientes dominantes en cada muestra.

---

### **B. Business Short Haul:**

**Nodos Identificados:**
- ❌ **Global/SH/Business/IB**: NO EXISTE en el árbol
- ✅ **Global/SH/Business/YW**: Anomalía POSITIVA (+10.35 pts) - **VALIDEZ COMPROMETIDA**

**Análisis del Nodo Único:**
- **YW Business SH**: Volumen insuficiente para validez estadística
- **Ausencia de IB**: No hay nodo separado para Iberia Business SH
- **Razón probable**: Volumen operativo insuficiente en ambas compañías para generar anomalías detectables

**DIAGNÓSTICO BUSINESS SH:** **No aplicable** - Volumen insuficiente de datos impide análisis comparativo válido entre compañías. La anomalía detectada en YW carece de robustez estadística.

---

## 🎯 CONCLUSIÓN DIAGNÓSTICA

**ECONOMY SHORT HAUL:** Causa **común de cabina** (mishandling operativo) con efectos direccionales opuestos por composición diferencial de muestras entre IB y YW.

**BUSINESS SHORT HAUL:** **Análisis no concluyente** por limitaciones de volumen de datos.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ PASO 2: DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)

## 📊 ANÁLISIS SISTEMÁTICO

### **A. Short Haul - Análisis de Cabinas:**

**Nodos Identificados por Estado:**
- **Economy SH**: Anomalías MIXTAS (IB negativo -5.5pts, YW positivo +14.3pts)
- **Business SH**: Anomalía POSITIVA limitada (solo YW +10.35pts, sin validez estadística)

**Comparación de Patrones:**

| Aspecto | Economy SH | Business SH | Convergencia |
|---------|------------|-------------|--------------|
| **Driver Operativo Principal** | Mishandling crítico (24.9-28.0%) | Mishandling elevado (24.9%) | ✅ **MISMO DRIVER** |
| **Magnitud del Impacto** | Significativo (±5.5 a ±14.3pts) | Limitado/Inválido (+10.35pts) | ❌ **REACTIVIDAD DIFERENTE** |
| **Validez Estadística** | Alta (125-273 encuestas) | Comprometida (volumen insuficiente) | ❌ **DIFERENTE ROBUSTEZ** |
| **Segmentos Críticos** | CodeShare, ACMI, Business travelers | Datos insuficientes | ⚠️ **NO COMPARABLE** |

**DIAGNÓSTICO SHORT HAUL:** Causa **común de radio** (mishandling operativo) con **reactividad diferencial por cabina** - Economy muestra alta sensibilidad y volumen robusto, mientras Business presenta baja reactividad operativa por volumen limitado.

---

### **B. Long Haul - Análisis de Cabinas:**

**Nodos Identificados por Estado:**
- **Economy LH**: Anomalía NEGATIVA (-2.40pts)
- **Business LH**: Anomalía NEGATIVA (-21.0pts) 
- **Premium LH**: Anomalía NEGATIVA (-13.79pts)

**Comparación de Patrones:**

| Aspecto | Economy LH | Business LH | Premium LH | Convergencia |
|---------|------------|-------------|------------|--------------|
| **Driver Principal** | Mishandling 27.2% (+9.3pts) | Mishandling 27.2% (+9.3pts) | Mishandling 27.2% (+9.3pts) | ✅ **TOTAL** |
| **Flota Crítica** | A33ACMI (-76.9 NPS) | A33ACMI (-76.9 NPS) | A350 (-22.2 NPS) | ⚠️ **PARCIAL** |
| **Rutas Afectadas** | MAD-SCL | BOG-MAD | MAD-MIA | ❌ **DIFERENTES** |
| **Magnitud Impacto** | -2.40pts (moderado) | -21.0pts (severo) | -13.79pts (alto) | ❌ **PROGRESIÓN CLARA** |
| **Load Factor** | 92.1% (+1.8pts) | 88.2% (-3.9pp vs baseline) | 91.5% (+1.1pts) | ⚠️ **VARIADO** |

**Progresión de Reactividad Identificada:**
1. **Business LH**: Mayor reactividad (-21.0pts) - Clientes premium más sensibles
2. **Premium LH**: Reactividad alta (-13.79pts) - Segmento intermedio  
3. **Economy LH**: Menor reactividad (-2.40pts) - Mayor tolerancia relativa

**DIAGNÓSTICO LONG HAUL:** Causa **común de radio** (mishandling crítico + problemas de flota específicos) con **progresión lógica de reactividad ascendente** - a mayor nivel de servicio esperado, mayor sensibilidad al deterioro operativo, siguiendo el patrón Economy < Premium < Business en magnitud de impacto.

---

## 🎯 CONCLUSIÓN DIAGNÓSTICA

**SHORT HAUL:** Causa **común de radio** con reactividad diferencial - Economy altamente reactivo, Business con datos insuficientes para análisis válido.

**LONG HAUL:** Causa **común de radio** con progresión lógica y consistente de reactividad por expectativas de servicio (Business > Premium > Economy).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 PASO 3: DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

## 📊 ANÁLISIS SISTEMÁTICO

### **A. Comparación Entre Radios:**

**Nodos Identificados por Estado:**
- **Short Haul**: Anomalías MIXTAS (negativas en IB, positivas en YW)
- **Long Haul**: Anomalías NEGATIVAS consistentes (todas las cabinas afectadas)
- **Global**: Anomalía NEGATIVA (-1.08pts)

**Convergencia de Drivers Principales:**

| Aspecto | Short Haul | Long Haul | Convergencia |
|---------|------------|-----------|--------------|
| **Driver Operativo** | Mishandling 24.9-28.0% | Mishandling 27.2% | ✅ **IDÉNTICO** |
| **Magnitud Desviación** | +7.8 a +9.8pts vs baseline | +9.3pts vs baseline | ✅ **CONSISTENTE** |
| **Flota Crítica** | A33ACMI, ATR problemáticos | A33ACMI (-76.9 NPS) | ✅ **MISMO EQUIPO** |
| **Load Factor** | Elevado (86.1% SH) | Elevado (88.3-92.1% LH) | ✅ **PATRÓN COMÚN** |
| **Codeshare Issues** | LATAM (-75.0 NPS) | LATAM (-35.7 NPS), AA (-8.3) | ✅ **MISMO PARTNER** |
| **Patrón NCS/Verbatims** | Ausencia total | Ausencia total | ✅ **IDÉNTICO** |

**DIAGNÓSTICO DE RADIOS:** **Causa global homogénea** - Ambos radios convergen completamente en drivers principales (mishandling crítico, problemas A33ACMI, issues codeshare LATAM) con evidencia operativa idéntica.

---

### **B. Coherencia con Nodo Global:**

**Análisis de Agregación:**

| Métrica | Global | Derivación SH+LH | Coherencia |
|---------|--------|------------------|------------|
| **NPS Impacto** | -1.08pts | SH mixto + LH negativo | ✅ **COHERENTE** |
| **Driver Principal** | Mishandling 27.2% (+9.3pts) | Consistente ambos radios | ✅ **PERFECTO** |
| **Flota Crítica** | A33ACMI (-50.0 NPS) | Mismo problema ambos radios | ✅ **AGREGADO** |
| **Codeshare** | LATAM (-35.7), AA (-8.3) | Consistente con radios | ✅ **COHERENTE** |
| **Concentración Geo** | BRU-MAD (-33.3 NPS) | Ruta específica identificada | ✅ **ESPECÍFICO** |

**Patrón de Agregación Identificado:**
- **Short Haul**: Efectos mixtos (IB negativo compensado parcialmente por YW positivo por composición muestral)
- **Long Haul**: Impacto negativo consistente y robusto
- **Global**: Refleja agregación lógica con predominio del impacto LH más severo

**Característica Única Global:**
- **"Problemas Silenciosos"**: El Global identifica que la ausencia de NCS/verbatims es un patrón sistémico, no una limitación local
- **Riesgo Sistémico**: Problemas de calidad que no activan alertas tradicionales pero erosionan lealtad

**DIAGNÓSTICO DE COHERENCIA:** El nodo Global **amplifica y sintetiza** correctamente los patrones de ambos radios, revelando además el **riesgo sistémico** de problemas operativos "invisibles" que impactan NPS sin generar escalamientos formales.

---

## 🎯 CONCLUSIÓN DIAGNÓSTICA

**ALCANCE DE CAUSAS:** **Causa global homogénea** - Mishandling crítico, problemas flota A33ACMI y codeshare LATAM afectan consistentemente ambos radios con evidencia operativa idéntica.

**MANIFESTACIÓN GLOBAL:** El nodo Global refleja **agregación coherente** de impactos por radio y revela el **riesgo sistémico crítico** de deterioro de calidad operativa que evade sistemas tradicionales de monitoreo pero erosiona directamente la lealtad del cliente.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 PASO 4: ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

## 📋 CAUSAS IDENTIFICADAS Y ANÁLISIS CONSOLIDADO

### **CAUSA 1: CRISIS OPERATIVA DE MISHANDLING**

#### A. NATURALEZA DE LA CAUSA:
**Hipótesis:** Falla sistémica en procesos de manejo de equipajes y servicios que se disparó el 24 de agosto, probablemente causada por sobrecarga operativa, problemas de personal o fallas en sistemas de handling que impactaron la cadena completa de servicio al cliente.

#### B. EVIDENCIA CONSOLIDADA Y ALCANCE:

**Segmento más grande afectado:** **Global** (todos los subsegmentos)

**Output Causal Detallado:**
- **Mishandling Global:** 27.2% vs 17.9% baseline (+9.3 puntos porcentuales)
- **Incremento relativo:** +52% vs promedio histórico
- **Correlación directa:** Factor principal explicativo de la caída NPS -1.08 puntos

**Alcance de Impacto:**
✅ **Afecta a TODOS los subsegmentos:** Global/LH, Global/SH, Economy, Business, Premium, IB, YW
- **Long Haul:** 27.2% mishandling (+9.3pts vs baseline)
- **Short Haul IB:** 28.0% mishandling (+9.8pts vs baseline) 
- **Short Haul YW:** 24.9% mishandling (+7.8pts vs baseline)

**Métricas Clave:**
- **Load Factor agregado:** 88.3% vs 87.6% (+0.6pts) - Sobrecarga operativa
- **Rutas críticas:** BRU-MAD, MAD-SCL, BOG-MAD, MAD-MIA
- **Verbatims:** Ausentes (patrón "silencioso" sistémico)
- **NCS:** Sin reportes formales (gap en escalamiento)

---

### **CAUSA 2: FALLA CATASTRÓFICA FLOTA A33ACMI**

#### A. NATURALEZA DE LA CAUSA:
**Hipótesis:** Problemas operativos severos específicos de aeronaves ACMI (Aircraft, Crew, Maintenance, Insurance) que incluyen estándares de servicio inferiores, tripulación externa con menor entrenamiento en estándares de marca, y posibles deficiencias técnicas o de mantenimiento que degradaron masivamente la experiencia del cliente.

#### B. EVIDENCIA CONSOLIDADA Y ALCANCE:

**Segmento más grande afectado:** **Global/LH** (incluye Economy y Business LH)

**Output Causal Detallado:**
- **A33ACMI NPS:** -76.9 (13 surveys) vs A350 next +51.3 
- **Spread crítico:** 128.2 puntos de diferencia entre flotas
- **Impacto directo:** Principal responsable de caída -4.95pts en Global/LH

**Alcance de Impacto:**
✅ **Afecta a subsegmentos:** Global/LH/Economy (-76.9 NPS, 13 surveys), Global/SH/Economy/IB (-40.0 NPS, 5 surveys)

**Métricas Clave:**
- **Concentración geográfica:** Rutas intercontinentales (MAD-SCL, BRU-MAD)
- **Contraste operativo:** A350 configuraciones mantienen NPS positivos
- **Volumen crítico:** 13-20 surveys suficientes para impacto sistémico
- **Verbatims:** No disponibles (problema no escalado formalmente)

---

### **CAUSA 3: CRISIS EN COORDINACIÓN CODESHARE LATAM**

#### A. NATURALEZA DE LA CAUSA:
**Hipótesis:** Deficiencias sistémicas en coordinación operativa con socio LATAM que incluyen problemas de comunicación, estándares de servicio no alineados, procesos de handling descoordinados, y posibles issues específicos en conexiones o servicios compartidos que deterioraron la experiencia en vuelos operados bajo acuerdo comercial.

#### B. EVIDENCIA CONSOLIDADA Y ALCANCE:

**Segmento más grande afectado:** **Global** (impacto en múltiples subsegmentos)

**Output Causal Detallado:**
- **LATAM Codeshare NPS:** -35.7 (14 surveys) a nivel Global
- **Impacto por segmento:**
  - Global/SH/Economy/IB: -75.0 NPS (8 surveys) - Más severo
  - Contraste con Iberia: +22.4 NPS - Confirma problema específico LATAM

**Alcance de Impacto:**
✅ **Afecta a subsegmentos:** Global/SH/Economy/IB, otros partners también negativos (AA -8.3, BA -5.0)

**Métricas Clave:**
- **Rutas afectadas:** Conexiones con red LATAM (especialmente América Sur)
- **Comparación partners:** Solo LATAM extremadamente negativo vs otros
- **Patrón geográfico:** América Sur NPS -33.3 (correlación directa)
- **Verbatims:** Ausentes (problema operativo sin escalamiento formal)

---

### **CAUSA 4: SOBRECARGA OPERATIVA CON SACRIFICIO DE CALIDAD**

#### A. NATURALEZA DE LA CAUSA:
**Hipótesis:** Estrategia operativa que priorizó mantener puntualidad y conexiones a costa de calidad de servicio, resultando en alta ocupación de vuelos que redujo flexibilidad para resolver problemas, combinado con presión operativa que llevó a shortcuts en procesos de calidad (especialmente handling) manteniendo métricas "duras" pero deteriorando experiencia del cliente.

#### B. EVIDENCIA CONSOLIDADA Y ALCANCE:

**Segmento más grande afectado:** **Global** (patrón sistémico)

**Output Causal Detallado:**
- **Load Factor:** 88.3% vs 87.6% promedio (+0.6pts)
- **OTP15:** 85.8% vs 84.4% promedio (+1.4pts) - Puntualidad mejorada
- **Misconex:** 0.6% vs 0.9% promedio (-0.3pts) - Conexiones estables
- **Paradoja operativa:** Métricas "duras" positivas + calidad negativa

**Alcance de Impacto:**
✅ **Afecta a TODOS los subsegmentos:** Patrón consistente en LH (91.5-92.1% LF) y SH (86.1% LF)

**Métricas Clave:**
- **Trade-off identificado:** Puntualidad mantenida vs calidad sacrificada
- **Presión sistémica:** Alta ocupación reduce margen de maniobra operativa
- **Impacto diferencial:** Mayor en segmentos premium (Business -21.0pts vs Economy -2.40pts)
- **Resultado:** "Tormenta perfecta" operativa con impacto silencioso en NPS

---

## 📋 SÍNTESIS EJECUTIVA FINAL

# 📈 SÍNTESIS EJECUTIVA

El 24 de agosto de 2025 se registró una caída generalizada del NPS a nivel global de 1.08 puntos (26.15 vs 27.23 baseline), causada por una crisis operativa sistémica centrada en problemas críticos de mishandling que se dispararon al 27.2% (+9.3 puntos vs promedio). Esta anomalía impactó de forma diferenciada los segmentos Long Haul con caídas consistentes en todas las cabinas: Economy LH experimentó un deterioro de 2.40 puntos (NPS 17.41 vs baseline 19.81), Business LH sufrió la mayor caída con 21.0 puntos (NPS 20.0 vs baseline 41.0), y Premium LH registró una reducción de 13.79 puntos (NPS 11.76 vs baseline 25.56). En Short Haul, el patrón fue mixto con Economy SH IB cayendo 5.5 puntos (NPS 21.2 vs baseline 26.8) mientras que Economy SH YW mostró una mejora paradójica de 14.3 puntos (NPS 48.0 vs baseline 33.7), y Business SH YW registró un incremento de 10.35 puntos (NPS 27.27 vs baseline 16.92) aunque con validez estadística comprometida por bajo volumen. Las causas identificadas incluyen una falla catastrófica en la flota A33ACMI que registró NPS de -76.9, problemas severos en coordinación con codeshare LATAM (NPS -35.7 a -75.0 según segmento), y una estrategia operativa que priorizó puntualidad (OTP mejorado +1.4 puntos) sacrificando calidad de servicio, evidenciado por el incremento del 52% en mishandling sin generar reportes formales NCS ni verbatims.

Las rutas más afectadas incluyeron BRU-MAD con NPS extremadamente negativo de -33.3, MAD-SCL (NPS 12.5), BOG-MAD y MAD-MIA, concentrándose el impacto en conexiones intercontinentales donde se amplificaron los problemas operativos. Los clientes Business/Work mostraron la mayor reactividad con caídas de hasta 21.0 puntos en Long Haul y NPS negativos en múltiples segmentos, seguidos por pasajeros de América del Sur (NPS -33.3) que fueron especialmente sensibles a los problemas de coordinación con LATAM, mientras que los clientes Leisure demostraron mayor tolerancia relativa manteniendo niveles de satisfacción más estables ante el mismo deterioro operativo.

**ECONOMY SH IB: Deterioro por Crisis de Handling**
La cabina Economy de Short Haul Iberia experimentó un deterioro significativo de 5.5 puntos, pasando de un NPS baseline de 26.8 a 21.2 el 24 de agosto. La causa principal fue el incremento crítico del mishandling al 28.0% (+9.8 puntos vs promedio de 18.2%), que impactó especialmente a operaciones de codeshare LATAM (NPS -75.0) y flota A33ACMI (NPS -40.0), mientras que las métricas de puntualidad se mantuvieron estables. Esta caída se reflejó de forma concentrada en vuelos con alta complejidad operativa, siendo los segmentos más reactivos los clientes conectados con la red LATAM y aquellos en aeronaves ACMI con estándares de servicio comprometidos.

**ECONOMY SH YW: Mejora Paradójica por Composición Muestral**
La cabina Economy de Short Haul Vueling registró una mejora paradójica de 14.3 puntos, alcanzando un NPS de 48.0 vs baseline de 33.7, a pesar de experimentar los mismos problemas operativos con mishandling elevado al 24.9% (+7.8 puntos vs promedio). Esta aparente mejora se explica por una composición favorable de la muestra donde los segmentos resilientes como clientes Leisure (NPS 53.7) y flota CRJ (NPS 52.8) dominaron numéricamente, mientras que los segmentos más sensibles como Business/Work (NPS 11.8) y flota ATR (NPS 21.1) fueron minoritarios pero severamente afectados, creando un promedio ponderado artificialmente elevado que enmascara problemas operativos reales.

**BUSINESS SH YW: Datos Insuficientes para Análisis Válido**
El segmento Business de Short Haul Vueling mostró un incremento aparente de 10.35 puntos (NPS 27.27 vs baseline 16.92), pero esta variación carece de validez estadística debido al volumen extremadamente bajo de encuestas. No se detectaron rutas con volumen mínimo de 2 respuestas, comprometiendo cualquier conclusión operativa válida y clasificando esta anomalía como ruido estadístico más que una señal real de mejora en la experiencia del cliente.

**ECONOMY LH: Impacto Moderado por Problemas de Flota Específicos**
La cabina Economy de Long Haul experimentó un deterioro de 2.40 puntos, registrando un NPS de 17.41 vs baseline de 19.81 el 24 de agosto. La causa principal fue la crisis de mishandling al 27.2% (+9.3 puntos vs promedio) amplificada por problemas específicos de la flota A33ACMI que registró un NPS devastador de -76.9 con 13 encuestas, contrastando dramáticamente con otras configuraciones como A350 next (NPS +51.3). Este deterioro se manifestó especialmente en la ruta MAD-SCL (NPS 12.5) y afectó desproporcionalmente a clientes Business/Work (NPS -13.3), mientras que la alta ocupación del 92.1% (+1.8 puntos vs baseline) redujo la flexibilidad operativa para resolver incidencias.

**BUSINESS LH: Mayor Deterioro por Alta Sensibilidad Premium**
La cabina Business de Long Haul sufrió el mayor impacto con una caída de 21.0 puntos, pasando de un NPS baseline de 41.0 a 20.0, evidenciando la mayor sensibilidad de clientes premium ante problemas operativos. Los drivers principales fueron el mishandling crítico al 27.2% (+9.3 puntos) y problemas severos en la flota A33ACMI (NPS -76.9), impactando especialmente la ruta BOG-MAD con 30 evaluaciones y afectando desproporcionadamente a clientes de América del Sur (NPS -33.3) en contraste con Europa (NPS 75.0), mientras que los problemas de coordinación con codeshare American Airlines (NPS -33.3) vs Iberia (NPS 28.0) revelaron deficiencias específicas en acuerdos comerciales externos.

**PREMIUM LH: Deterioro Significativo por Expectativas Elevadas**
El segmento Premium de Long Haul registró una caída de 13.79 puntos, alcanzando un NPS de 11.76 vs baseline de 25.56, reflejando la alta reactividad de clientes premium ante el deterioro operativo. Las causas dominantes fueron el mishandling crítico al 27.2% (+9.3 puntos) y problemas específicos de flota A350 (NPS -22.2) que contrastaron con el mejor desempeño de A333 (NPS +60.0), especialmente evidentes entre clientes 100% Leisure que mostraron una dispersión extrema de 83.3 puntos entre regiones, siendo América del Sur (NPS -33.3) la más afectada vs España (NPS +50.0), mientras que la alta ocupación del 91.5% amplificó la insatisfacción en aeronaves ya comprometidas.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 8
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-08-23 to 2025-08-23:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 8 (Global, Global/LH/Business, Global/LH/Premium, Global/SH/Business, Global/SH/Business/IB, Global/SH/Business/YW, Global/SH/Economy/IB, Global/SH/Economy/YW)

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 PASO 1: DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)

## 📊 ANÁLISIS SISTEMÁTICO

### A. **ECONOMY SHORT HAUL**

**Nodos identificados:**
- ✅ **Economy SH/IB**: Presente (anomalía negativa -7.67 pts)
- ✅ **Economy SH/YW**: Presente (anomalía negativa -0.21 pts, pero identificada como falsa alarma)

**Comparación de patrones:**

| Aspecto | Economy SH/IB | Economy SH/YW |
|---------|---------------|---------------|
| **Driver Principal** | Mishandling +54% (28.0% vs 18.2%) | Sin driver real (falsa alarma por error de datos) |
| **Evidencia Operativa** | Crisis silenciosa con CodeShare AA crítico (NPS -57.1) | Operación normal (NPS real ~33.5) |
| **Volumen de Impacto** | 276 pasajeros afectados | 170 respuestas normales |

**DIAGNÓSTICO ECONOMY SH:** **Causa específica de IB** - Los problemas operativos de Mishandling se concentraron exclusivamente en IB, mientras YW operó normalmente.

---

### B. **BUSINESS SHORT HAUL**

**Nodos identificados:**
- ✅ **Business SH/IB**: Presente (anomalía negativa -5.48 pts)
- ✅ **Business SH/YW**: Presente (anomalía positiva +14.65 pts)

**Comparación de patrones:**

| Aspecto | Business SH/IB | Business SH/YW |
|---------|----------------|----------------|
| **Driver Principal** | Colapso operativo total (solo 3 pasajeros) | Mishandling +46% con segmentación demográfica extrema |
| **Evidencia Operativa** | Una sola ruta operativa (MAD-ZRH) | Dispersión geográfica de 100 puntos NPS |
| **Naturaleza del Problema** | Falta de operación | Experiencia diferencial por perfil |

**DIAGNÓSTICO BUSINESS SH:** **Causas específicas y divergentes por compañía** - IB experimentó colapso operativo mientras YW tuvo problemas de equidad en el servicio con impacto demográfico diferencial.

---

## 🎯 **CONCLUSIÓN GENERAL**

**Ambas cabinas Short Haul muestran causas específicas de compañía:**
- **Economy**: Problemas concentrados en IB (Mishandling crítico) vs YW operando normalmente
- **Business**: Problemas divergentes - IB con colapso operativo vs YW con inequidad de servicio

**No se identifican causas comunes a nivel de cabina Short Haul.**

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ PASO 2: DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)

## 📊 ANÁLISIS SISTEMÁTICO

### A. **SHORT HAUL - Análisis de Cabinas**

**Nodos identificados:**
- ❌ **Economy SH**: Anomalías negativas (IB: -7.67 pts, YW: -0.21 pts falsa)
- ❌ **Business SH**: Anomalías mixtas (IB: -5.48 pts, YW: +14.65 pts)

**Comparación de patrones:**

| Aspecto | Economy SH | Business SH |
|---------|------------|-------------|
| **Driver Operativo** | Mishandling +54% (solo IB) | Colapso operativo IB / Mishandling +46% YW |
| **Reactividad al Mishandling** | Alta sensibilidad (NPS -7.67 con crisis silenciosa) | Reactividad diferencial extrema |
| **Patrón de Impacto** | Concentrado en CodeShare AA | IB: Sin operación / YW: Segmentación demográfica |
| **Consistencia IB/YW** | Divergente (IB crisis, YW normal) | Totalmente divergente (colapso vs inequidad) |

**DIAGNÓSTICO SHORT HAUL:** **Causas específicas de cabina con patrones completamente divergentes** - No hay convergencia entre Economy y Business SH. Cada cabina experimentó crisis operativas de naturaleza diferente, sin patrón común de radio.

---

### B. **LONG HAUL - Análisis de Cabinas**

**Nodos identificados:**
- ❌ **Business LH**: Anomalía negativa (-16.14 pts)
- ❌ **Premium LH**: Anomalía negativa (-5.56 pts)
- ⚠️ **Economy LH**: No presente en el árbol

**Comparación de patrones:**

| Aspecto | Business LH | Premium LH |
|---------|-------------|------------|
| **Driver Principal** | Mishandling +52% (27.2% vs 17.9%) | Mishandling +52% (27.2% vs 17.9%) |
| **Reactividad** | **EXTREMA** (-16.14 pts con solo 19 encuestas) | **MODERADA** (-5.56 pts con 15 encuestas) |
| **Segmentos Críticos** | A350 C (-100.0) + España (0.0) | A333 (-66.7) + América Norte (-33.3) |
| **Patrón Geográfico** | España vs Américas (100 pts diferencia) | América Norte más afectada |
| **Fleet Problemático** | A350 C catastrófico | A333 crítico |

**Progresión de Reactividad:**
- **Business LH**: Máxima sensibilidad (-16.14 pts)
- **Premium LH**: Sensibilidad moderada (-5.56 pts)
- **Economy LH**: No disponible para comparación

**DIAGNÓSTICO LONG HAUL:** **Causa común de radio con reactividad diferencial por cabina** - Ambas cabinas LH convergen en el mismo driver operativo (Mishandling +52%) pero muestran diferentes niveles de sensibilidad, con Business LH actuando como el segmento más reactivo y Premium LH como amortiguador relativo.

---

## 🎯 **CONCLUSIÓN GENERAL**

**Short Haul:** Causas específicas de cabina - Patrones completamente divergentes sin convergencia operativa.

**Long Haul:** Causa común de radio - Mismo driver operativo (Mishandling crítico) con reactividad diferencial decreciente desde Business hacia Premium.

**Patrón identificado:** Long Haul muestra coherencia operativa mientras Short Haul presenta fragmentación por cabina y compañía.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 PASO 3: DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

## 📊 ANÁLISIS SISTEMÁTICO

### A. **Comparación Entre Radios**

**Estado de afectación:**
- ❌ **Short Haul**: Afectado (múltiples anomalías negativas)
- ❌ **Long Haul**: Afectado (anomalías negativas en Business y Premium)
- ❌ **Global**: Afectado (anomalía positiva paradójica +4.8 pts)

**Convergencia de drivers:**

| Aspecto | Short Haul | Long Haul |
|---------|------------|-----------|
| **Driver Principal** | Mishandling crítico (+54% IB, +46% YW) | Mishandling crítico (+52% ambas cabinas) |
| **Valor Operativo** | 28.0% vs 18.2% baseline | 27.2% vs 17.9% baseline |
| **Patrón de Impacto** | Fragmentado por compañía/cabina | Coherente con reactividad diferencial |
| **Evidencia SHAP** | Consistente con crisis operativa | Consistente con crisis operativa |

**CONVERGENCIA CONFIRMADA:** Ambos radios convergen en **Mishandling crítico** como driver principal con valores operativos prácticamente idénticos (~27-28% vs ~18% baseline).

---

### B. **Coherencia con Nodo Global**

**Análisis del Global:**
- **Estado**: Anomalía positiva +4.8 pts (32.0 vs 27.2 baseline)
- **Driver Identificado**: Mishandling +52% (27.2% vs 17.9%)
- **Paradoja Central**: Crisis operativa severa con NPS positivo

**Coherencia operativa:**

| Métrica | Global | Short Haul | Long Haul |
|---------|--------|------------|-----------|
| **Mishandling** | 27.2% (+9.3pts) | ~28% (+9.8pts) | 27.2% (+9.3pts) |
| **Manifestación NPS** | **+4.8 pts** (paradójico) | **Negativo** (crisis) | **Negativo** (crisis) |
| **Explicación** | Problemas de captura de datos | Impacto directo registrado | Impacto directo registrado |

**PATRÓN IDENTIFICADO:** El Global muestra **coherencia operativa perfecta** con ambos radios (mismo Mishandling crítico) pero **incoherencia en manifestación NPS** debido a problemas sistémicos de captura de feedback.

---

### C. **Manifestación de la Paradoja Global**

**Explicación de la anomalía positiva Global:**
1. **Crisis operativa real**: Mishandling +52% confirmado en todos los niveles
2. **Epicentro identificado**: CodeShare American Airlines y ruta MAD-ORD
3. **Subregistro sistemático**: 
   - ❌ Cero incidentes NCS reportados
   - ❌ Cero verbatims disponibles
   - ❌ Crisis "silenciosa" no capturada

**Mecanismo de compensación:**
- Los segmentos más afectados (CodeShare AA: NPS -20.0) no completaron suficientes encuestas
- Los segmentos menos afectados dominaron la muestra agregada
- El Global refleja sesgo de respuesta, no ausencia de problemas

---

## 🎯 **CONCLUSIÓN DIAGNÓSTICA**

**CAUSA GLOBAL HOMOGÉNEA CON MANIFESTACIÓN PARADÓJICA:**

La crisis del 23 de agosto de 2025 fue una **causa global homogénea** caracterizada por Mishandling crítico (+52%) que afectó coherentemente tanto Short Haul como Long Haul con el mismo driver operativo. 

Sin embargo, el nodo Global presenta una **manifestación paradójica** (NPS positivo) que no refleja efectos compensatorios reales entre radios, sino **problemas sistémicos de captura de feedback** que subestiman el impacto real de la crisis operativa.

**Alcance:** Crisis operativa global con impacto fragmentado por problemas de medición y sesgo de respuesta en el nivel agregado.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 PASO 4: ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

## 🎯 CAUSA PRINCIPAL IDENTIFICADA

### **CAUSA: CRISIS OPERATIVA DE MISHANDLING CRÍTICO**

---

## A. **NATURALEZA DE LA CAUSA**

**Hipótesis Causal:**
El 23 de agosto de 2025 experimentó una **crisis sistémica de manejo de equipajes** que incrementó el Mishandling en un 52% sobre el baseline, generando una cascada de insatisfacción del cliente que se manifestó de manera diferencial según el perfil del pasajero, tipo de operación y capacidad de captura de feedback de cada segmento.

**Mecanismo Operativo:**
- **Falla primaria**: Colapso en procesos de handling de equipajes a nivel sistémico
- **Amplificadores**: Alta ocupación (Load Factor 88.3%), operaciones CodeShare complejas, y rutas transatlánticas de mayor vulnerabilidad
- **Manifestación diferencial**: Impacto variable según sensibilidad del segmento (Business > Premium > Economy) y capacidad de escalación de problemas

---

## B. **EVIDENCIA CONSOLIDADA Y ALCANCE**

### **SEGMENTO MÁS GRANDE AFECTADO: GLOBAL**

**Output Causal Detallado del Nodo Global:**

> **ANOMALÍA DETECTADA**: NPS positivo de +4.8 puntos vs baseline (32.0 vs 27.2)
> 
> **CONCLUSIÓN PRINCIPAL**: Crisis operativa focalizada en CodeShare American Airlines que paradójicamente generó una anomalía NPS positiva debido a problemas de captura de feedback.
> 
> **CRISIS OPERATIVA CONFIRMADA**:
> - **Mishandling disparado**: 27.2% vs 17.9% baseline (**+52% de incremento**)
> - **Load Factor elevado**: 88.3% vs 87.6% baseline
> - **Combinación crítica**: Alta ocupación + problemas severos de equipaje
> 
> **EPICENTRO IDENTIFICADO: RUTA MAD-ORD**:
> - **NPS devastador**: -14.3 (7 encuestas)
> - **Ruta transatlántica**: Madrid → Chicago O'Hare
> - **Impacto focalizado**: Problemas concentrados en una ruta específica
> 
> **SEGMENTO MÁS AFECTADO: CODESHARE AMERICAN AIRLINES**:
> - **NPS crítico**: -20.0 (30 encuestas)
> - **Dispersión extrema**: 83.3 puntos de spread
> - **Fleet problemático**: A350 C con NPS -6.7

---

### **ALCANCE DE AFECTACIÓN**

**La causa afecta TODOS los subsegmentos bajo el nodo Global:**

#### **Short Haul:**
- **Economy SH/IB**: Crisis silenciosa con CodeShare AA (NPS -57.1)
- **Economy SH/YW**: Sin impacto real (falsa alarma)
- **Business SH/IB**: Colapso operativo total (solo 3 pasajeros)
- **Business SH/YW**: Inequidad de servicio por segmentación demográfica

#### **Long Haul:**
- **Business LH**: Máxima reactividad (-16.14 pts) con A350 C catastrófico
- **Premium LH**: Reactividad moderada (-5.56 pts) con A333 crítico

---

### **MÉTRICAS CLAVE CONSOLIDADAS**

#### **Indicadores Operativos:**
| Métrica | Valor 23/08 | Baseline 7d | Variación | Estado |
|---------|-------------|-------------|-----------|---------|
| **Mishandling** | 27.2% | 17.9% | **+9.3pts (+52%)** | 🔴 **CRÍTICO** |
| **Load Factor** | 88.3% | 87.6% | +0.6pts | 🟡 Elevado |
| **OTP** | 85.8% | 84.4% | +1.4pts | 🟢 Mejorado |
| **Misconex** | 0.6% | 0.9% | -0.3pts | 🟢 Mejorado |

#### **Impacto en NPS por Segmento:**
- **CodeShare American Airlines**: -20.0 (30 encuestas)
- **Ruta MAD-ORD**: -14.3 (7 encuestas)
- **Fleet A350 C**: -6.7 / -100.0 (según segmento)
- **Fleet A333**: -66.7 (Premium LH)

#### **Rutas Involucradas:**
- **MAD-ORD** (epicentro): NPS -14.3
- **EZE-MAD** (Premium LH): NPS 0.0
- **MAD-ZRH** (Business SH/IB): NPS 100.0 (única operativa)

#### **Verbatims Representativos:**
❌ **Sin verbatims disponibles** - Problema sistémico de captura de feedback cualitativo que contribuye a la naturaleza "silenciosa" de la crisis.

---

### **PATRÓN DE MANIFESTACIÓN DIFERENCIAL**

**Reactividad por Tipo de Cliente:**
1. **Business LH**: Máxima sensibilidad (-16.14 pts)
2. **Premium LH**: Sensibilidad moderada (-5.56 pts)  
3. **Economy SH**: Sensibilidad variable (IB crítica, YW normal)
4. **Business SH**: Patrones divergentes (IB colapso, YW inequidad)

**Factores Amplificadores Identificados:**
- **Operaciones CodeShare**: Mayor complejidad y vulnerabilidad
- **Rutas transatlánticas**: Mayor impacto por distancia y expectativas
- **Flotas específicas**: A350 C y A333 con mayor vulnerabilidad
- **Alta ocupación**: Load Factor elevado amplifica problemas operativos

---

## 🎯 **SÍNTESIS CAUSAL**

La **Crisis Operativa de Mishandling Crítico** del 23 de agosto de 2025 representa una falla sistémica que afectó globalmente a la operación con manifestaciones diferenciadas según el perfil del cliente, generando desde colapsos operativos totales hasta experiencias de inequidad de servicio, con un patrón común de problemas de equipaje que se amplificó en operaciones complejas y rutas de larga distancia.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El día 23 de agosto de 2025 presentó un patrón complejo de variaciones en el NPS caracterizado por una crisis operativa sistémica de manejo de equipajes que afectó diferencialmente a múltiples segmentos. A nivel global, se registró paradójicamente una mejora de 4.8 puntos (pasando de 27.2 a 32.0), enmascarando una crisis operativa real causada por un incremento del 52% en Mishandling (27.2% vs 17.9% baseline). Los segmentos más afectados negativamente fueron Business Long Haul con una caída de 16.14 puntos (de 40.99 a 57.14), Premium Long Haul con un deterioro de 5.56 puntos (de 25.56 a 20.0), Economy Short Haul IB con una caída de 7.67 puntos (de 26.75 a 34.42), y Business Short Haul IB con un deterioro de 5.48 puntos (de 40.77 a 35.29). Contrariamente, Business Short Haul YW experimentó una mejora significativa de 14.65 puntos (de 16.9 a 31.6), mientras que Economy Short Haul YW mostró una variación mínima de -0.21 puntos que posteriormente se identificó como error de datos.

Las rutas más afectadas fueron MAD-ORD con un NPS devastador de -14.3 y EZE-MAD con NPS de 0.0, siendo las operaciones CodeShare con American Airlines las más críticas (NPS -20.0 para el agregado y -57.1 específicamente en Economy SH IB). Los grupos de clientes más reactivos incluyeron pasajeros de América del Norte (NPS -60.0 en Business SH), clientes de rutas transatlánticas, y usuarios de flotas específicas como A350 C (NPS -100.0 en algunos segmentos) y A333 (NPS -66.7), evidenciando mayor sensibilidad a problemas operativos en segmentos de mayor expectativa de servicio.

**ECONOMY SH IB: Crisis Silenciosa de Equipajes**
La cabina Economy Short Haul de Iberia experimentó un deterioro significativo durante el 23 de agosto de 2025, registrando una caída de 7.67 puntos en su NPS (de 26.75 a 34.42). La causa principal fue un incremento masivo del 54% en Mishandling (28.0% vs 18.2% baseline) que impactó silenciosamente a 276 pasajeros sin generar reportes formales de incidentes. Esta crisis se manifestó especialmente en las operaciones CodeShare con American Airlines, donde el NPS alcanzó niveles críticos de -57.1, mientras que las operaciones ACMI con flota A33 también mostraron deterioro significativo con NPS de -16.7, evidenciando la vulnerabilidad particular de estas operaciones complejas ante problemas de equipaje.

**ECONOMY SH YW: Estabilidad Operativa**  
El segmento Economy Short Haul de Vueling mantuvo desempeño estable durante el 23 de agosto de 2025, registrando un NPS de 33.5 con una variación mínima de -0.21 puntos que posteriormente se confirmó como error en la agregación de datos. No se detectaron cambios significativos reales, manteniendo niveles consistentes de satisfacción con 170 respuestas válidas que confirmaron la operación normal de este segmento.

**ECONOMY LH: Estabilidad Semanal**
La cabina Economy Long Haul mantuvo desempeño estable a nivel semanal durante este período, sin presentar datos específicos de anomalías en el árbol analizado, sugiriendo niveles consistentes de satisfacción sin variaciones significativas.

**BUSINESS LH: Máxima Reactividad a Crisis Operativa**  
La cabina Business Long Haul experimentó el mayor deterioro del día, registrando una caída crítica de 16.14 puntos (de 40.99 a 57.14) causada por el mismo incremento del 52% en Mishandling que afectó globalmente. Los drivers principales se concentraron en problemas específicos de flota A350 C que alcanzó un NPS catastrófico de -100.0, complementado por una polarización geográfica extrema donde España registró NPS de 0.0 mientras Américas alcanzó 100.0, creando una dispersión de 200 puntos. Esta reactividad extrema se manifestó con solo 19 encuestas, donde cada respuesta negativa tuvo un impacto desproporcionado, siendo los perfiles más reactivos los pasajeros españoles y usuarios de la flota A350 C.

**BUSINESS SH IB: Colapso Operativo Total**
El segmento Business Short Haul de Iberia sufrió un colapso operativo casi total durante el 23 de agosto, registrando una caída de 5.48 puntos (de 40.77 a 35.29), aunque esta métrica resulta estadísticamente irrelevante dado que solo operó una ruta (MAD-ZRH) con únicamente 3 pasajeros. Las causas dominantes fueron la reducción masiva de la operación a una fracción mínima, mientras que paradójicamente los pocos pasajeros que volaron obtuvieron una experiencia perfecta con NPS de 100.0, evidenciando que el problema no fue de calidad de servicio sino de disponibilidad operativa.

**BUSINESS SH YW: Inequidad de Servicio por Perfil**
El segmento Business Short Haul de Vueling experimentó una mejora paradójica de 14.65 puntos (de 16.9 a 31.6) durante el mismo día de crisis operativa, registrando un incremento del 46% en Mishandling (de 17.1% a 24.9%). Esta aparente contradicción se explica por una segmentación demográfica extrema con 100 puntos de dispersión entre regiones: España alcanzó NPS perfecto de 100.0, Europa registró 60.0, América del Sur 50.0, y clientes sin región específica 0.0, sugiriendo tratamiento diferencial o resolución preferencial por perfil geográfico que generó experiencias altamente polarizadas.

**PREMIUM LH: Reactividad Moderada en Rutas Americanas**
El segmento Premium Long Haul mostró un deterioro moderado de 5.56 puntos (de 25.56 a 20.0) causado por el mismo incremento del 52% en Mishandling, pero con menor reactividad que Business LH. Las causas dominantes se concentraron en la flota A333 que registró NPS crítico de -66.7, especialmente evidentes en rutas hacia América del Norte (NPS -33.3) y América del Sur incluyendo EZE-MAD (NPS 0.0), mientras que los perfiles más reactivos fueron viajeros intercontinentales con mayor sensibilidad a problemas de equipaje en vuelos de larga distancia.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 8
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-08-22 to 2025-08-22:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 11 (Global, Global/LH, Global/LH/Business, Global/LH/Economy, Global/LH/Premium, Global/SH, Global/SH/Business, Global/SH/Business/IB, Global/SH/Business/YW, Global/SH/Economy, Global/SH/Economy/IB)

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 **DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)**

## **A. Economy Short Haul:**

**Nodos identificados:**
- ✅ **Economy SH/IB**: Presente (anomalía negativa)
- ❌ **Economy SH/YW**: Ausente

**Razón de ausencia YW**: No se detectó anomalía significativa en Economy YW el 22 de agosto, por lo que no se generó nodo específico en el árbol.

**Diagnóstico**: **Causa específica de compañía IB** - El problema operativo de Mishandling (+9.8pts) afectó exclusivamente a la operación de Iberia en Economy Short Haul.

---

## **B. Business Short Haul:**

**Nodos identificados:**
- ✅ **Business SH/IB**: Presente (anomalía negativa, -5.77 pts)
- ✅ **Business SH/YW**: Presente (anomalía positiva, +9.74 pts)

**Análisis comparativo de patrones:**

| Aspecto | IB | YW |
|---------|----|----|
| **Dirección** | Negativa (-5.77) | Positiva (+9.74) |
| **Driver principal** | Mishandling 28.0% (+9.8pts) | Mishandling 24.9% (+7.8pts) |
| **Impacto operativo** | Deterioro severo | Deterioro moderado |
| **Resultado NPS** | Caída significativa | Mejora paradójica |

**Diagnóstico**: **Causa específica de compañía** - Aunque ambas compañías experimentaron incremento en Mishandling, IB mostró impacto negativo severo mientras YW paradójicamente mejoró su NPS, indicando gestión operativa y/o percepción del cliente diferenciada entre compañías.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ **DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)**

## **A. Short Haul - Análisis de Cabinas:**

**Nodos identificados:**
- ✅ **Economy SH**: Presente (anomalía negativa)
- ✅ **Business SH**: Presente (anomalía negativa)

**Análisis de convergencia/divergencia:**

| Aspecto | Economy SH | Business SH |
|---------|------------|-------------|
| **Dirección anomalía** | Negativa (-2.56 pts) | Negativa (-2.31 pts) |
| **Driver operativo** | Mishandling 27.2% (+9.3pts) | Mishandling 27.2% (+9.3pts) |
| **Magnitud impacto** | Moderada | Moderada |
| **Localización** | ATH-MAD (A319) | LHR-MAD (A320) |

**Patrón por compañía:**
- **IB**: Economy (-7.5 pts) vs Business (-5.77 pts) - Ambas negativas, Economy más afectada
- **YW**: Solo Business (+9.74 pts) - Comportamiento anómalo positivo

**Diagnóstico SH**: **Causa común de radio con reactividad diferencial por compañía** - El mismo problema operativo de Mishandling afectó ambas cabinas, pero IB mostró impacto negativo consistente mientras YW en Business mostró resistencia paradójica.

---

## **B. Long Haul - Análisis de Cabinas:**

**Nodos identificados:**
- ✅ **Economy LH**: Presente (anomalía negativa, -7.6 pts)
- ✅ **Business LH**: Presente (anomalía positiva, +7.15 pts)  
- ✅ **Premium LH**: Presente (anomalía negativa, -5.56 pts)

**Análisis de progresión por cabina:**

| Aspecto | Economy LH | Business LH | Premium LH |
|---------|------------|-------------|------------|
| **Dirección** | Negativa (-7.6) | **Positiva (+7.15)** | Negativa (-5.56) |
| **Driver operativo** | Mishandling 27.2% | Mishandling 27.2% | Mishandling 27.2% |
| **Reactividad** | Alta vulnerabilidad | **Resistencia/mejora** | Vulnerabilidad moderada |
| **Perfil afectado** | Business/Work crítico | Leisure resiliente | Leisure más afectado |

**Diagnóstico LH**: **Causa común de radio con reactividad diferencial extrema por cabina** - El mismo problema operativo de Mishandling generó respuestas opuestas: Business LH actuó como "amortiguador" mejorando su NPS, mientras Economy y Premium se deterioraron, sugiriendo que la percepción del problema varía significativamente según las expectativas y perfil de cada cabina.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 **DIAGNÓSTICO A NIVEL RADIO Y GLOBAL**

## **A. Comparación Entre Radios:**

**Nodos identificados:**
- ✅ **Short Haul**: Presente (anomalía negativa, -2.52 pts)
- ✅ **Long Haul**: Presente (anomalía negativa, -5.56 pts)

**Análisis de convergencia entre radios:**

| Aspecto | Short Haul | Long Haul |
|---------|------------|-----------|
| **Dirección anomalía** | Negativa (-2.52) | Negativa (-5.56) |
| **Driver operativo principal** | Mishandling 27.2% (+9.3pts) | Mishandling 27.2% (+9.3pts) |
| **Load Factor** | 87.1% (+0.4pts) | 91.5% (+1.2pts) |
| **Localización crítica** | ATH-MAD, MAD-VGO | MAD-ORD, EZE-MAD |
| **Perfiles más afectados** | LATAM CodeShare, ASIA | Business/Work, Europa |

**Convergencia identificada**: **Completa** - Ambos radios muestran el mismo driver operativo crítico (Mishandling +9.3pts) con magnitudes de impacto proporcionales.

---

## **B. Coherencia con Nodo Global:**

**Análisis de agregación:**

| Nivel | NPS Anomalía | Driver Principal | Coherencia |
|-------|--------------|------------------|------------|
| **Global** | -3.41 pts | Mishandling 27.2% (+9.3pts) | ✅ |
| **Short Haul** | -2.52 pts | Mishandling 27.2% (+9.3pts) | ✅ |
| **Long Haul** | -5.56 pts | Mishandling 27.2% (+9.3pts) | ✅ |

**Patrón de agregación**: El nodo Global (-3.41 pts) representa un **promedio ponderado coherente** entre SH (-2.52) y LH (-5.56), sin efectos de cancelación o amplificación anómalos.

**Evidencia operativa consistente**: 
- Mismo problema de Mishandling (27.2% vs 17.9% baseline) en todos los niveles
- Localización en rutas específicas pero distribuidas entre ambos radios
- Impacto en perfiles similares (internacionales, CodeShare, Business/Work)

---

## **DIAGNÓSTICO FINAL:**

**Causa global homogénea** - El problema operativo de Mishandling crítico (+9.3pts) afectó simultáneamente ambos radios con el mismo patrón causal, manifestándose en rutas específicas de cada radio pero con coherencia operativa total. El nodo Global refleja correctamente la agregación proporcional de ambos impactos sin distorsiones.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 **ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS**

## **CAUSA ÚNICA IDENTIFICADA: CRISIS OPERATIVA DE MISHANDLING**

### **A. NATURALEZA DE LA CAUSA:**

**Hipótesis Causal**: El 22 de agosto de 2025 se produjo una **crisis sistémica de manejo de equipajes y servicios** que incrementó el Mishandling de 17.9% (baseline) a 27.2% (+9.3 puntos porcentuales, +52% de incremento). Esta crisis operativa se manifestó como múltiples problemas distribuidos geográficamente que afectaron la experiencia del cliente sin generar escalamiento formal masivo, creando un patrón de "anomalía silenciosa" donde el impacto real superó la visibilidad en sistemas tradicionales de monitoreo.

---

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**

**Segmento más grande afectado**: **GLOBAL** (781 encuestas totales)

#### **OUTPUT CAUSAL DETALLADO:**

**Impacto NPS**: -3.41 puntos (23.82 vs baseline 27.23)

**Métricas Operativas Críticas**:
- **Mishandling**: 27.2% vs 17.9% promedio (**+9.3 puntos**, +52% incremento)
- **Load Factor**: 88.3% vs 87.6% (+0.6 puntos) - Factor amplificador
- **OTP15**: 85.8% vs 84.4% (+1.4 puntos) - Mejorado, descarta problemas de puntualidad
- **Misconex**: 0.6% vs 0.9% (-0.3 puntos) - Mejorado, descarta problemas de conexiones

#### **RUTAS CRÍTICAS INVOLUCRADAS:**
- **MAD-MEX**: NPS -14.8 (n=27) - Epicentro principal
- **ATH-MAD**: NPS 0.0 (A319) - Short Haul crítico  
- **LHR-MAD**: NPS 60.0 (A320) - Business afectado
- **EZE-MAD**: Polarización extrema Premium
- **MAD-VGO**: NPS 0.0 (A319) - Localización específica

#### **PERFILES MÁS VULNERABLES:**
- **ASIA**: NPS -40.0 a -66.7 según segmento
- **ORIENTE MEDIO**: NPS -50.0
- **AMÉRICA NORTE**: NPS -27.3 a -66.7 según cabina
- **CodeShare LATAM**: NPS -33.3 a -100.0 según segmento
- **Clientes Business/Work**: Especialmente reactivos en Economy

#### **CARACTERÍSTICAS DE "ANOMALÍA SILENCIOSA":**
- **❌ Cero incidentes NCS reportados** en todos los segmentos
- **❌ Ausencia total de verbatims** representativos del día
- **✅ Impacto operativo real y medible** en todas las métricas NPS

#### **ALCANCE CONFIRMADO:**
Esta causa afecta a **todos los subsegmentos** bajo el nodo Global:
- **Short Haul y Long Haul** (ambos radios impactados)
- **Economy, Business y Premium** (todas las cabinas afectadas)
- **IB y YW** (ambas compañías, con YW Business como única excepción paradójica)
- **Todas las regiones geográficas** (con variaciones en intensidad)
- **Todos los tipos de flota** (A319, A320, A332, A350 con diferentes grados)

#### **PATRÓN DE IMPACTO DIFERENCIAL:**
Aunque la causa es única y global, genera **reactividad diferencial** según:
- **Expectativas del segmento**: Premium y Business más exigentes
- **Origen geográfico**: Clientes internacionales menos tolerantes
- **Tipo de operación**: CodeShare más vulnerable
- **Perfil de viaje**: Business/Work más críticos que Leisure

**CONCLUSIÓN**: Una sola causa operativa (crisis de Mishandling) con manifestación global pero impacto segmentado según perfil de cliente y expectativas de servicio.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El 22 de agosto de 2025 se registró una caída generalizada del NPS global de -3.41 puntos (pasando de 27.23 a 23.82), causada por una crisis operativa de manejo de equipajes que elevó el Mishandling del 17.9% al 27.2% (+9.3 puntos porcentuales). Esta problemática afectó tanto Short Haul como Long Haul, con el segmento Long Haul experimentando el mayor deterioro (-5.56 puntos, NPS final 16.7) y Short Haul mostrando una caída más moderada (-2.52 puntos, NPS final 27.29). La evidencia operativa confirma que el incremento del 52% en problemas de equipaje fue el único factor crítico, mientras que otras métricas como puntualidad y conexiones mejoraron, descartando causas alternativas.

Las rutas más impactadas incluyen MAD-MEX (NPS -14.8), ATH-MAD y MAD-VGO (ambas con NPS 0.0), y conexiones intercontinentales como EZE-MAD que mostraron polarización extrema. Los grupos de clientes más reactivos fueron los residentes de Asia (NPS entre -40.0 y -66.7), Oriente Medio (NPS -50.0), América Norte (NPS -27.3 a -66.7), y especialmente los usuarios de CodeShare LATAM (NPS desde -33.3 hasta -100.0 según segmento). Los viajeros Business/Work demostraron mayor sensibilidad que Leisure ante los problemas operativos.

**ECONOMY SH IB: Deterioro Operativo Severo**
La cabina Economy Short Haul de Iberia experimentó un deterioro significativo de -7.5 puntos, pasando de un NPS de 26.8 a 19.2 el 22 de agosto. La causa principal fue el incremento crítico del Mishandling al 28.0% vs 18.2% promedio (+9.8 puntos, +54% de incremento), que se concentró específicamente en la ruta MAD-VGO operada con flota A319, donde todos los pasajeros reportaron experiencia negativa (NPS 0.0). Esta degradación afectó desproporcionadamente a clientes de CodeShare LATAM (NPS -100.0), residentes de Asia (NPS -66.7) y América Norte (NPS -33.3), mientras que los segmentos Leisure y clientes con CodeShare VY mantuvieron niveles aceptables.

**BUSINESS SH IB: Impacto Localizado en Ruta Premium**
El segmento Business Short Haul de Iberia registró una caída de -5.77 puntos, alcanzando un NPS de 35.0 el 22 de agosto vs 40.77 de baseline. Esta evolución se explica principalmente por el mismo problema de Mishandling (28.0% vs 18.2% promedio), siendo especialmente visible en la ruta LHR-MAD donde se concentraron los problemas con flota A320 clásico. Los perfiles más reactivos incluyeron residentes de América Norte (NPS -66.7) y usuarios de CodeShare American Airlines (NPS -33.3), mientras que clientes españoles y operaciones con British Airways mantuvieron experiencias superiores (NPS 76.9 y 60.0 respectivamente).

**BUSINESS SH YW: Mejora Paradójica**
La cabina Business Short Haul de Air Europa mostró una mejora paradójica de +9.74 puntos, alcanzando un NPS de 26.67 vs 16.92 de baseline el 22 de agosto. Esta evolución contradictoria ocurrió a pesar del deterioro operativo en Mishandling (24.9% vs 17.1% promedio, +7.8 puntos), sugiriendo factores de mejora no capturados en las métricas operativas disponibles. Los clientes españoles lideraron esta mejora con un NPS de 44.4, mientras que la flota CRJ mostró desempeño moderado pero superior al baseline, aunque la ausencia de feedback cualitativo limita la comprensión completa de los factores impulsores.

**ECONOMY LH: Vulnerabilidad ante Crisis Operativa**
La cabina Economy Long Haul experimentó el mayor deterioro con -7.6 puntos, pasando de un NPS de 19.8 a 12.2 el 22 de agosto. La causa principal fue la crisis de Mishandling (27.2% vs 17.9% promedio) combinada con alta ocupación (92.1% vs 90.3%), que afectó especialmente a rutas intercontinentales y flota específica. Esta degradación se reflejó especialmente en operaciones con A33ACMI (NPS 0.0) y A332 (NPS 2.9), mientras que los perfiles más reactivos incluyen viajeros Business/Work (NPS -14.3) y clientes europeos (NPS -7.7), con CodeShare LATAM mostrando el mayor impacto negativo (NPS -16.7).

**BUSINESS LH: Resistencia Operativa Excepcional**
La cabina Business Long Haul registró una mejora de +7.15 puntos, alcanzando un NPS de 48.15 vs 40.99 de baseline el 22 de agosto. Los drivers principales muestran el mismo problema de Mishandling (27.2%) que afectó otros segmentos, pero esta cabina demostró resistencia excepcional, especialmente en flota A350 Next (NPS 87.5) versus A350 estándar (NPS 25.0). Esta divergencia impactó especialmente entre viajeros Leisure (NPS 57.1) que mostraron mayor resistencia comparado con Business/Work (NPS 16.7), evidenciando que las expectativas y perfil de cliente modulan significativamente la percepción de problemas operativos.

**PREMIUM LH: Polarización Extrema**
El segmento Premium Long Haul mostró una caída de -5.56 puntos, con un NPS final de 20.0 vs 25.56 de baseline el 22 de agosto. Las causas dominantes fueron la crisis de Mishandling (27.2%) y alta ocupación (91.5%), especialmente evidentes en la ruta EZE-MAD donde se generó una polarización extrema de 82.1 puntos entre diferentes perfiles. Los clientes de América Centro/Norte experimentaron el mayor impacto negativo (NPS -25.0) mientras que residentes de Sur América y España mantuvieron satisfacción alta (NPS 57.1), con viajeros Leisure siendo significativamente más vulnerables (NPS 9.5) que Business (NPS 75.0) en este segmento premium.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 11
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