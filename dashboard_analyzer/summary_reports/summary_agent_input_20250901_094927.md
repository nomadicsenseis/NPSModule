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
[{'period': 1, 'date_range': '2025-08-22 to 2025-08-28', 'ai_interpretation': '📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**\n\n**Nodos analizados:** 7 (Global, Global/LH/Business, Global/LH/Premium, Global/SH/Business, Global/SH/Business/IB, Global/SH/Business/YW, Global/SH/Economy/YW)\n\n---\n\n## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA\n\n# 🏢 **PASO 1: DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)**\n\n## **A. Economy Short Haul:**\n\n**ANÁLISIS DE DISPONIBILIDAD:**\n- ✅ **Economy SH/YW**: Existe (Global/SH/Economy/YW)\n- ❌ **Economy SH/IB**: No existe en los datos analizados\n\n**CONCLUSIÓN ECONOMY SH:** No es posible determinar diferencias por compañía ya que solo existe el nodo YW. La ausencia del nodo IB sugiere que no hubo suficiente volumen de encuestas o anomalías significativas en Economy IB durante el período analizado.\n\n---\n\n## **B. Business Short Haul:**\n\n**ANÁLISIS DE DISPONIBILIDAD:**\n- ✅ **Business SH/IB**: Existe (Global/SH/Business/IB)\n- ✅ **Business SH/YW**: Existe (Global/SH/Business/YW)\n\n**COMPARACIÓN DE PATRONES:**\n\n### **Drivers Principales y SHAP Values:**\n- **SH/IB**: Punctuality SHAP +5.306, In-flight F&B +1.178, Arrivals +1.090\n- **SH/YW**: Punctuality SHAP +9.212, Cabin Crew +1.522, Phone contact +1.325\n\n### **Evidencia Operativa:**\n- **SH/IB**: OTP15 +3.6 puntos (90.31% vs 86.66%)\n- **SH/YW**: OTP15 +3.1 puntos (82.99% vs 79.89%)\n\n### **Factores Negativos:**\n- **SH/IB**: Boarding SHAP -0.525 (problemas menores)\n- **SH/YW**: Check-in SHAP -2.821, In-flight F&B SHAP -2.280 (problemas significativos)\n\n---\n\n## **DIAGNÓSTICO FINAL:**\n\n**Economy SH:** Diagnóstico no aplicable - solo existe nodo YW.\n\n**Business SH:** **Causa común de cabina** - Ambas compañías (IB/YW) convergen en el driver principal (mejora en puntualidad) con valores SHAP similares y evidencia operativa consistente (OTP15 +3.6 vs +3.1). Las diferencias en drivers secundarios son menores y no alteran la causa raíz compartida.\n\n---\n\n## 💺 DIAGNÓSTICO A NIVEL DE CABINA\n\n# ✈️ **PASO 2: DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)**\n\n## **A. Short Haul - Análisis de Cabinas:**\n\n### **Disponibilidad de Nodos SH:**\n- ✅ **Economy SH**: Global/SH/Economy/YW (+12.88 pts)\n- ✅ **Business SH**: Global/SH/Business (+10.11 pts) + SH/Business/IB (+8.93 pts) + SH/Business/YW (+7.71 pts)\n- ❌ **Premium SH**: No existe\n\n### **Análisis de Convergencia/Divergencia:**\n\n**DRIVER PRINCIPAL COMÚN:**\n- **Economy SH/YW**: Punctuality SHAP +5.644\n- **Business SH**: Punctuality SHAP +7.498 (agregado), +5.306 (IB), +9.212 (YW)\n- **Evidencia Operativa Común**: OTP15 +3.1 a +3.6 puntos en todos los nodos\n\n**DRIVERS SECUNDARIOS DIVERGENTES:**\n- **Economy SH**: Arrivals experience (+1.705), Boarding (+1.656), Journey prep (+1.501)\n- **Business SH**: Cabin Crew, In-flight F&B, Phone contact (variables según IB/YW)\n\n**FACTORES NEGATIVOS DIFERENCIADOS:**\n- **Economy SH**: Lounge SHAP -0.583\n- **Business SH**: Check-in problems (especialmente YW: -2.821), Boarding issues (IB: -0.525)\n\n---\n\n## **B. Long Haul - Análisis de Cabinas:**\n\n### **Disponibilidad de Nodos LH:**\n- ❌ **Economy LH**: No existe\n- ✅ **Business LH**: Global/LH/Business (+14.6 pts)\n- ✅ **Premium LH**: Global/LH/Premium (-1.89 pts)\n\n### **Análisis de Convergencia/Divergencia:**\n\n**PATRÓN COMPLETAMENTE DIVERGENTE:**\n- **Business LH**: Anomalía POSITIVA (+14.6 pts) - Boarding SHAP +4.584, Journey prep +3.870, Punctuality +2.733\n- **Premium LH**: Anomalía NEGATIVA (-1.89 pts) - Connections experience SHAP -3.340, Journey prep -2.766\n\n**EVIDENCIA OPERATIVA CONTRADICTORIA:**\n- **Ambas cabinas**: Misma mejora OTP15 +5.3 puntos, cero incidentes NCS\n- **Reacción opuesta**: Business valoró las mejoras, Premium experimentó problemas específicos no capturados por métricas agregadas\n\n---\n\n## **DIAGNÓSTICO FINAL:**\n\n**Short Haul:** **Causa común de radio con reactividad diferencial por cabina** - Todas las cabinas SH convergen en la mejora de puntualidad como driver principal, pero Economy muestra mayor sensibilidad a experiencia de producto (arrivals, boarding) mientras Business se enfoca más en servicios de cabina y contacto.\n\n**Long Haul:** **Causa específica de cabina** - Business y Premium muestran patrones completamente divergentes ante la misma evidencia operativa. Business experimentó mejoras integrales mientras Premium sufrió problemas específicos en conexiones y journey preparation que no fueron detectados por sistemas operativos tradicionales.\n\n---\n\n## 🌎 DIAGNÓSTICO GLOBAL POR RADIO\n\n# 🌍 **PASO 3: DIAGNÓSTICO A NIVEL RADIO Y GLOBAL**\n\n## **A. Comparación Entre Radios:**\n\n### **Disponibilidad y Magnitud de Impacto:**\n- ✅ **Short Haul**: Múltiples nodos afectados (+7.71 a +12.88 pts) - **TODAS ANOMALÍAS POSITIVAS**\n- ✅ **Long Haul**: Dos nodos afectados - **PATRÓN MIXTO** (Business +14.6 pts, Premium -1.89 pts)\n\n### **Convergencia de Drivers Principales:**\n**DRIVER COMÚN DOMINANTE:**\n- **Short Haul**: Punctuality SHAP +5.644 a +9.212 en todos los nodos\n- **Long Haul**: Punctuality SHAP +2.733 (Business), ausente como driver positivo en Premium\n\n**EVIDENCIA OPERATIVA CONSISTENTE:**\n- **Short Haul**: OTP15 +3.1 a +3.6 puntos\n- **Long Haul**: OTP15 +5.3 puntos (mayor mejora)\n- **Ambos Radios**: Cero incidentes NCS operativos durante los 7 días\n\n### **Drivers Secundarios Divergentes:**\n- **Short Haul**: Enfoque en experiencia de producto (Arrivals, Boarding, Journey prep)\n- **Long Haul**: Polarización extrema (Business: Boarding +4.584; Premium: Connections -3.340)\n\n---\n\n## **B. Coherencia con Nodo Global:**\n\n### **Impacto Agregado:**\n- **Global**: +5.36 pts NPS (anomalía positiva moderada)\n- **Patrón Subyacente**: Dominancia de anomalías positivas SH + Business LH compensando parcialmente Premium LH negativo\n\n### **Drivers SHAP Global vs Radios:**\n- **Global**: Punctuality +4.067 (driver más significativo) ✅ **COHERENTE**\n- **Global**: Boarding +0.793, Arrivals +0.629 ✅ **COHERENTE con SH**\n- **Global**: Ticket Price -1.574 (negativo) ✅ **COHERENTE con tensiones LH**\n\n### **Validación Operativa Global:**\n- **Global**: OTP15 +3.6 puntos, cero incidentes NCS ✅ **COHERENTE con ambos radios**\n- **Global**: Misconex -0.2, Mishandling +0.5 ✅ **REFLEJA mix de mejoras/tensiones**\n\n---\n\n## **DIAGNÓSTICO FINAL:**\n\n**Causa global con manifestación diferencial por radio** - Ambos radios (SH y LH) convergen en una mejora operativa fundamental en puntualidad (+3.1 a +5.3 OTP15) y ausencia total de incidentes, pero Short Haul muestra reactividad homogéneamente positiva mientras Long Haul presenta polarización extrema entre cabinas. El nodo Global (+5.36 pts) refleja coherentemente esta dinámica mixta, con el driver Punctuality dominante (+4.067 SHAP) y efectos compensatorios que atenúan el impacto agregado versus las magnitudes individuales de cada radio.\n\n---\n\n## 📋 ANÁLISIS DE CAUSAS DETALLADO\n\n# 🔍 **PASO 4: ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS**\n\n## **CAUSA 1: MEJORA OPERATIVA EXCEPCIONAL EN PUNTUALIDAD**\n\n### **A. NATURALEZA DE LA CAUSA:**\n**Hipótesis:** Durante la semana del 22-28 agosto 2025, Iberia experimentó un período de performance operativa excepcional caracterizado por mejoras significativas en puntualidad y ausencia total de incidentes operativos. Esta mejora operativa integral se tradujo directamente en mayor satisfacción del cliente, especialmente en segmentos sensibles a la confiabilidad del servicio.\n\n### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**\n\n**SEGMENTO MÁS GRANDE AFECTADO:** Global/LH/Business (+14.59 puntos NPS)\n\n**OUTPUT CAUSAL DETALLADO:**\n- **Magnitud de Impacto:** +14.59 puntos NPS (53.8% vs período anterior)\n- **Driver SHAP Principal:** Punctuality +2.733, Boarding +4.584, Journey preparation +3.870\n- **Validación Operativa:** OTP15 mejoró +5.3 puntos (77.78% → 83.06%)\n- **Validación Temporal:** Cero incidentes NCS operativos durante los 7 días analizados\n- **Satisfacción Cliente:** Punctuality sat_diff +13.76 puntos\n\n**ALCANCE DE AFECTACIÓN:** Esta causa afecta a **todos los subsegmentos** bajo Global, incluyendo:\n- Global/SH/Business/IB (+8.93 pts): Punctuality SHAP +5.306\n- Global/SH/Business/YW (+7.71 pts): Punctuality SHAP +9.212  \n- Global/SH/Economy/YW (+12.88 pts): Punctuality SHAP +5.644\n- Global/SH/Business (+10.11 pts): Punctuality SHAP +7.498\n\n**MÉTRICAS CLAVE:**\n- **Indicadores Operativos:** OTP15 +3.1 a +5.3 puntos según segmento, Misconex -0.1 a -0.2 puntos\n- **Rutas Más Impactadas:** GUA-MAD (NPS 100.0), MAD-MEX (NPS 81.8), PMI-RJL (NPS 80.0), LEI-MAD (NPS 100.0)\n- **Verbatims Representativos:** 5,891 análisis sobre Boarding confirman mejora en experiencia de embarque como efecto secundario de la mejora operativa\n\n---\n\n## **CAUSA 2: PROBLEMAS ESPECÍFICOS EN EXPERIENCIA DE CONEXIONES (LONG HAUL PREMIUM)**\n\n### **A. NATURALEZA DE LA CAUSA:**\n**Hipótesis:** Durante el mismo período de mejora operativa general, el segmento Premium Long Haul experimentó problemas específicos en la coordinación y comunicación de conexiones en el hub MAD, particularmente afectando rutas latinoamericanas. Estos problemas no fueron detectados por sistemas de monitoreo operativo tradicionales pero impactaron severamente la experiencia del cliente en journey preparation y boarding.\n\n### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**\n\n**SEGMENTO AFECTADO:** Global/LH/Premium (-1.89 puntos NPS)\n\n**OUTPUT CAUSAL DETALLADO:**\n- **Magnitud de Impacto:** -1.89 puntos NPS vs semana anterior\n- **Drivers SHAP Críticos:** Connections experience -3.340, Journey preparation support -2.766\n- **Contradicción Operativa:** Punctuality SHAP +2.731 (la puntualidad general mejoró)\n- **Validación Temporal:** Sin incidentes NCS registrados (problemas no sistémicos)\n- **Feedback Cualitativo:** 161 análisis de verbatims negativos sobre Boarding\n\n**ALCANCE ESPECÍFICO:** Esta causa afecta exclusivamente al segmento Premium Long Haul.\n\n**MÉTRICAS CLAVE:**\n- **Rutas Específicas Afectadas:** LIM-MAD (NPS 0.0, deterioro -42.9 pts), MAD-SCL (NPS 0.0, deterioro -30.8 pts), GRU-MAD (NPS 0.0, deterioro -42.9 pts)\n- **Patrón Geográfico:** Concentrado en conexiones MAD hacia/desde Latinoamérica y Norteamérica\n- **Reactividad por Perfil:** Residence Region con 62.1 puntos de spread, clientes latinoamericanos más afectados\n- **Verbatims:** 161 análisis confirman problemas específicos en boarding y journey preparation no capturados por métricas operativas agregadas\n\n---\n\n## **SÍNTESIS DE CAUSAS:**\n\n**CAUSA DOMINANTE:** Mejora operativa excepcional en puntualidad (afecta positivamente a 6 de 7 nodos analizados)\n**CAUSA ESPECÍFICA:** Problemas de conexiones en LH Premium (afecta negativamente a 1 nodo específico)\n**RESULTADO NETO GLOBAL:** +5.36 puntos NPS, con la mejora operativa dominando sobre los problemas específicos de Premium.\n\n---\n\n## 📋 SÍNTESIS EJECUTIVA FINAL\n\n📈 SÍNTESIS EJECUTIVA:\n\nDurante la semana del 22-28 agosto 2025, Iberia experimentó una mejora generalizada en la satisfacción del cliente, con el NPS global aumentando 5.36 puntos hasta alcanzar 26.49 puntos. Esta evolución positiva se concentró principalmente en los segmentos Business, donde Long Haul Business registró la mayor mejora con un incremento de 14.59 puntos (pasando de 27.18 a 41.77 puntos), seguido por Short Haul Economy YW con una subida de 12.88 puntos (de 20.84 a 33.72 puntos) y Short Haul Business con incrementos entre 7.71 y 10.11 puntos según el subsegmento. La única excepción fue Premium Long Haul, que experimentó un deterioro de 1.89 puntos, bajando de 27.42 a 25.53 puntos. La causa principal de estas mejoras fue una performance operativa excepcional en puntualidad, evidenciada por incrementos en OTP15 de 3.1 a 5.3 puntos según el segmento, ausencia total de incidentes operativos durante los 7 días analizados, y drivers SHAP de puntualidad entre +2.733 y +9.212 puntos. El deterioro en Premium Long Haul se debió específicamente a problemas en la experiencia de conexiones (SHAP -3.340) y journey preparation (SHAP -2.766), que no fueron detectados por sistemas operativos tradicionales pero impactaron severamente la percepción del cliente.\n\nLas rutas más beneficiadas por la mejora operativa incluyeron GUA-MAD y LEI-MAD (ambas con NPS 100.0), MAD-MEX (NPS 81.8), y PMI-RJL (NPS 80.0), mientras que Premium Long Haul sufrió deterioros severos en rutas latinoamericanas como LIM-MAD, MAD-SCL y GRU-MAD (todas con NPS 0.0 y caídas de 30-43 puntos). Los perfiles de clientes más reactivos mostraron spreads de hasta 136.8 puntos según región de residencia y 127.3 puntos según tipo de flota, con clientes latinoamericanos y de ciertas aeronaves específicas mostrando las mayores variaciones tanto positivas como negativas.\n\n**ECONOMY SH: Mejora Sustancial por Excelencia Operativa**\nLa cabina Economy de Short Haul experimentó una mejora excepcional durante la semana del 22-28 agosto, registrando un NPS de 33.72 puntos con un incremento de 12.88 puntos respecto a la semana anterior. La causa principal fue una mejora operativa integral liderada por puntualidad excepcional (SHAP +5.644 y OTP15 +3.1 puntos), complementada por mejoras en arrivals experience (+1.705 SHAP), boarding (+1.656 SHAP) y journey preparation support (+1.501 SHAP), respaldadas por ausencia total de incidentes operativos y 1,159 análisis de verbatims positivos sobre boarding. Esta mejora se reflejó especialmente en rutas como ODB-PMI (NPS 100.0), PMI-RJL (NPS 80.0) y PMI-VLC (NPS 66.7), mientras que los perfiles más reactivos incluyen clientes de diferentes regiones de residencia (spread 76.8 puntos) y tipos de operación codeshare (spread 69.5 puntos).\n\n**BUSINESS SH: Performance Operativa Excepcional con Variaciones por Compañía**\nEl segmento Business de Short Haul registró mejoras significativas, con un NPS agregado de 32.16 puntos y un incremento de 10.11 puntos vs la semana anterior, aunque con variaciones entre compañías (IB: +8.93 puntos hasta 37.88; YW: +7.71 puntos hasta 18.82). Esta evolución se explica principalmente por mejoras sustanciales en puntualidad (SHAP entre +5.306 y +9.212 según compañía) respaldadas por incrementos reales en OTP15 de +3.1 a +3.6 puntos y ausencia total de incidentes operativos, siendo especialmente visible en rutas como MAD-OPO y DBV-MAD (ambas NPS 100.0) y entre perfiles de diferentes regiones de residencia que mostraron spreads de hasta 136.8 puntos.\n\n**ECONOMY LH: Desempeño Estable**\nLa cabina Economy de Long Haul mantuvo desempeño estable durante esta semana. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción.\n\n**BUSINESS LH: Mejora Excepcional por Optimización Integral**\nLa cabina Business de Long Haul experimentó la mejora más significativa de todos los segmentos, registrando un NPS de 41.77 puntos con un incremento de 14.59 puntos respecto al período anterior. Los drivers principales fueron mejoras en boarding (+4.584 SHAP), journey preparation support (+3.870 SHAP) y puntualidad (+2.733 SHAP), respaldados por un incremento en OTP15 de +5.3 puntos y validados por 236 análisis de verbatims sobre boarding, impactando especialmente las rutas GUA-MAD (NPS 100.0), MAD-MEX (NPS 81.8) y BOS-MAD (NPS 50.0) y perfiles con spreads de hasta 127.3 puntos según tipo de flota.\n\n**PREMIUM LH: Deterioro por Problemas Específicos de Conexiones**\nEl segmento Premium de Long Haul experimentó un deterioro, registrando un NPS de 25.53 puntos con una caída de 1.89 puntos vs la semana anterior. Las causas dominantes fueron problemas específicos en connections experience (-3.340 SHAP) y journey preparation support (-2.766 SHAP) que no fueron detectados por sistemas operativos tradicionales pero impactaron severamente la experiencia del cliente, especialmente evidentes en rutas latinoamericanas como LIM-MAD, MAD-SCL y GRU-MAD (todas con NPS 0.0 y deterioros de 30-43 puntos) y entre perfiles de diferentes regiones de residencia con spreads de hasta 62.1 puntos.\n\n---\n\n✅ **ANÁLISIS COMPLETADO**\n\n- **Nodos procesados:** 7\n- **Pasos de análisis:** 5\n- **Metodología:** Análisis conversacional paso a paso\n- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado\n\n*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*'}]

**ANÁLISIS DIARIO SINGLE:**
📅 2025-08-28 to 2025-08-28:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 11 (Global, Global/LH, Global/LH/Business, Global/LH/Premium, Global/SH, Global/SH/Business, Global/SH/Business/IB, Global/SH/Business/YW, Global/SH/Economy, Global/SH/Economy/IB, Global/SH/Economy/YW)

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)

## **A. ECONOMY SHORT HAUL**

### **NODOS IDENTIFICADOS:**
✅ **Economy SH/IB** - Presente  
✅ **Economy SH/YW** - Presente

### **ANÁLISIS COMPARATIVO:**

| Factor | Economy SH/IB | Economy SH/YW |
|--------|---------------|---------------|
| **Anomalía NPS** | -6.25 pts | -4.50 pts |
| **Driver Principal** | Mishandling 28.0% (+9.8pts) | Mishandling 24.9% (+7.8pts) |
| **Load Factor** | 80.1% vs 80.5% baseline | 86.1% vs 83.8% baseline |
| **Impacto CodeShare** | LATAM: -100.0, VY: 0.0 | VY: -50.0 vs IB: +31.7 |
| **Flota Crítica** | A350 C (0.0), A33ACMI (-40.0) | Dispersión menor (4.9pts) |

### **DIAGNÓSTICO ECONOMY SH:**
**Causa común de cabina** - Ambas compañías convergen en el mismo driver principal (Mishandling crítico) con magnitudes similares y patrones operativos consistentes.

---

## **B. BUSINESS SHORT HAUL**

### **NODOS IDENTIFICADOS:**
✅ **Business SH/IB** - Presente  
✅ **Business SH/YW** - Presente

### **ANÁLISIS COMPARATIVO:**

| Factor | Business SH/IB | Business SH/YW |
|--------|----------------|----------------|
| **Anomalía NPS** | -40.77 pts | -2.64 pts |
| **Driver Principal** | Mishandling 28.0% (+9.8pts) | Mishandling 24.9% (+7.8pts) |
| **Dispersión Regional** | 133.3 pts spread | 50 pts spread |
| **Impacto por Perfil** | Leisure más afectado | Leisure colapso total (0.0) |
| **Muestra** | 21 surveys | 14 surveys |

### **DIAGNÓSTICO BUSINESS SH:**
**Causa común de cabina con amplificación específica IB** - Ambas compañías comparten el mismo driver operativo (Mishandling), pero IB experimenta una amplificación dramática del impacto (-40.77 vs -2.64 pts) debido a mayor dispersión regional y volumen de muestra.

---

## **📊 CONCLUSIÓN DIAGNÓSTICA:**

**ECONOMY SH:** Causa común - Problema operativo sistémico de Mishandling afecta similarmente a ambas compañías.

**BUSINESS SH:** Causa común con amplificación IB - Mismo problema operativo base, pero IB sufre impacto desproporcionadamente mayor por factores de segmentación específicos.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)

## **A. SHORT HAUL - ANÁLISIS DE CABINAS**

### **PATRONES IDENTIFICADOS:**

| Cabina | Anomalía NPS | Driver Principal | Patrón de Impacto |
|--------|--------------|------------------|-------------------|
| **Economy SH** | -5.68 pts | Mishandling 27.2% (+9.3pts) | Impacto moderado, concentrado BCN-MAD |
| **Business SH** | -29.32 pts | Mishandling 27.2% (+9.3pts) | **Impacto severo**, dispersión regional extrema |

### **ANÁLISIS DE CONVERGENCIA/DIVERGENCIA:**

**✅ CONVERGENCIA EN CAUSA OPERATIVA:**
- Ambas cabinas comparten **idéntico driver principal**: Mishandling 27.2%
- **Mismo contexto operativo**: Load Factor elevado, OTP mejorado
- **Misma ausencia**: Sin incidentes NCS ni verbatims

**❌ DIVERGENCIA EN REACTIVIDAD:**
- **Business SH**: Reactividad **5x mayor** (-29.32 vs -5.68 pts)
- **Dispersión regional**: Business 150pts spread vs Economy más homogéneo
- **Sensibilidad al perfil**: Business devastado por problemas operativos

### **DIAGNÓSTICO SHORT HAUL:**
**Causa común de radio con reactividad diferencial extrema** - Mismo problema operativo (Mishandling) impacta ambas cabinas, pero Business SH muestra hipersensibilidad 5x mayor que Economy SH.

---

## **B. LONG HAUL - ANÁLISIS DE CABINAS**

### **PATRONES IDENTIFICADOS:**

| Cabina | Anomalía NPS | Driver Principal | Característica Distintiva |
|--------|--------------|------------------|---------------------------|
| **Economy LH** | No presente | - | Sin nodo específico identificado |
| **Business LH** | +10.72 pts | Mishandling mejorado (-9.3pts) | **Anomalía positiva** |
| **Premium LH** | +74.4 pts | Sesgo de supervivencia | **Paradoja estadística** |

### **ANÁLISIS DE PROGRESIÓN POR CLASE:**

**🔄 PROGRESIÓN INVERSA DETECTADA:**
- **Premium LH**: +74.4pts (sesgo estadístico, 6 surveys)
- **Business LH**: +10.72pts (mejora operativa real)
- **Economy LH**: Ausente del análisis

**⚠️ PATRÓN ANÓMALO:**
- **Premium como "falso amortiguador"**: Alta satisfacción por sesgo de muestra, no por calidad real
- **Business como "verdadero beneficiario"**: Mejora operativa tangible
- **Economy ausente**: Sin datos para validar patrón completo

### **DIAGNÓSTICO LONG HAUL:**
**Patrón incompleto con distorsión estadística** - Business LH muestra mejora operativa real, mientras Premium LH presenta sesgo de supervivencia. Imposible determinar causa común de radio por ausencia de Economy LH.

---

## **📊 CONCLUSIÓN DIAGNÓSTICA INTEGRADA:**

**SHORT HAUL:** Causa operativa común (Mishandling) con reactividad diferencial extrema - Business 5x más sensible que Economy.

**LONG HAUL:** Patrón fragmentado - Business mejora por factores operativos, Premium distorsionado por sesgo estadístico, Economy ausente del análisis.

**IMPLICACIÓN ESTRATÉGICA:** Los problemas operativos de Mishandling afectan sistémicamente, pero la reactividad del cliente escala exponencialmente con la clase de servicio.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

## **A. COMPARACIÓN ENTRE RADIOS**

### **ESTADO DE AFECTACIÓN:**

| Radio | Estado | Anomalía NPS | Driver Principal |
|-------|--------|--------------|------------------|
| **Short Haul** | ✅ **AFECTADO** | -8.1 pts | Mishandling 27.2% (+9.3pts) |
| **Long Haul** | ❌ **NO AFECTADO** | +9.1 pts | Mishandling mejorado (-9.3pts) |

### **ANÁLISIS DE CONVERGENCIA:**

**🔄 DRIVERS COMPLETAMENTE OPUESTOS:**
- **Short Haul**: Mishandling **deteriorado** (+9.3pts vs baseline)
- **Long Haul**: Mishandling **mejorado** (-9.3pts vs baseline)
- **Magnitud idéntica**: ±9.3pts pero en direcciones contrarias

**📊 EVIDENCIA OPERATIVA CONTRASTANTE:**
- **SH**: Crisis operativa real con impacto negativo masivo
- **LH**: Mejora operativa real con beneficio positivo
- **Valores SHAP**: Completamente inversos entre radios

---

## **B. COHERENCIA CON NODO GLOBAL**

### **ANÁLISIS DE AGREGACIÓN:**

| Nivel | Anomalía NPS | Interpretación |
|-------|--------------|----------------|
| **Global** | -2.28 pts | Impacto neto moderado |
| **Short Haul** | -8.1 pts | Impacto negativo severo |
| **Long Haul** | +9.1 pts | Impacto positivo significativo |

### **PATRÓN DE COMPENSACIÓN DETECTADO:**

**🔀 EFECTOS COMPENSATORIOS PARCIALES:**
- **SH negativo** (-8.1) + **LH positivo** (+9.1) = **Global moderado** (-2.28)
- **No cancelación completa**: El impacto global refleja predominancia del volumen SH
- **Coherencia matemática**: Global es resultado lógico de la combinación ponderada

### **CONCENTRACIÓN GEOGRÁFICA CONFIRMADA:**
- **Global epicentro**: FCO-MAD (NPS -25.0)
- **SH epicentro**: BCN-MAD (NPS -23.8)
- **LH epicentro**: EZE-MAD (NPS 30.8)
- **Patrón**: Problemas localizados por radio, no sistémicos globales

---

## **📊 DIAGNÓSTICO INTEGRADO**

### **CAUSA IDENTIFICADA:**
**Causas específicas de radio con efectos compensatorios** - Short Haul experimentó crisis operativa de Mishandling mientras Long Haul experimentó mejora operativa simultánea en la misma métrica, generando un impacto global neto moderado por compensación parcial.

### **CARACTERÍSTICAS DEL FENÓMENO:**
1. **Segregación operativa perfecta**: Problemas y mejoras completamente separados por radio
2. **Magnitud simétrica**: ±9.3pts en Mishandling pero direcciones opuestas
3. **Localización geográfica**: Cada radio con epicentros específicos (BCN-MAD vs EZE-MAD)
4. **Compensación incompleta**: Global refleja predominancia volumétrica de Short Haul

### **IMPLICACIÓN ESTRATÉGICA:**
El 28 de agosto representa un caso excepcional donde **operaciones simultáneas pero independientes** en cada radio generaron efectos opuestos que se compensaron parcialmente a nivel global, enmascarando la severidad real de los problemas en Short Haul.

**ALCANCE DIAGNÓSTICO:** Causas específicas de radio con compensación parcial - No es un problema global homogéneo sino dos fenómenos operativos independientes y contrastantes.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

## **CAUSA PRINCIPAL: CRISIS OPERATIVA DE MISHANDLING DE EQUIPAJES**

### **A. NATURALEZA DE LA CAUSA**

**Hipótesis Causal:**
El 28 de agosto de 2025 se produjo una **crisis operativa específica en el manejo de equipajes** que afectó selectivamente las operaciones Short Haul. Esta crisis se caracterizó por un incremento del 52% en incidencias de Mishandling (27.2% vs 17.9% baseline), concentrándose geográficamente en rutas específicas y manifestándose como un "problema silencioso" que impactó directamente el NPS sin generar escalación formal en sistemas de reporte.

**Mecanismo de Impacto:**
- **Operativo**: Fallas en procesos de carga/descarga y manejo de equipaje
- **Experiencial**: Deterioro directo de la experiencia del cliente sin oportunidad de gestión proactiva
- **Sistémico**: Ausencia de captura formal (0 NCS, 0 verbatims) indica brecha en sistemas de monitoreo

---

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE**

#### **SEGMENTO MÁS GRANDE AFECTADO: Global/SH (Short Haul Complete)**

**OUTPUT CAUSAL DETALLADO:**
- **Anomalía NPS**: -8.1 puntos (21.7 vs 29.8 baseline)
- **Driver Principal**: Mishandling 27.2% vs 17.9% promedio (+9.3pts, +52% incremento)
- **Muestra**: Impacto sistémico en todas las operaciones Short Haul
- **Alcance**: Esta causa afecta a **TODOS los subsegmentos** bajo Global/SH incluyendo:
  - Economy SH/IB (-6.25pts)
  - Economy SH/YW (-4.50pts) 
  - Business SH/IB (-40.77pts)
  - Business SH/YW (-2.64pts)

#### **MÉTRICAS CLAVE CONSOLIDADAS:**

**Indicadores Operativos:**
- **Mishandling Crítico**: 27.2% (+52% vs baseline)
- **Load Factor Agravante**: 87.1% vs 86.6% promedio
- **OTP Paradójico**: 86.2% vs 84.9% (mejoró, pero no compensó)
- **Misconex Estable**: Sin deterioro en conexiones

**Rutas Críticas Identificadas:**
- **Epicentro Principal**: Barcelona-Madrid (BCN-MAD) con NPS -23.8 (n=21)
- **Impacto Secundario**: Múltiples rutas SH afectadas por el mismo patrón operativo

**Flotas Más Impactadas:**
- **A33ACMI**: NPS -57.1 (impacto extremo)
- **A332**: NPS -33.3 (impacto severo)
- **A320neo**: Problemas específicos vs A320 estándar

**Segmentación de Impacto:**
- **Reactividad por Clase**: Business 5x más sensible que Economy
- **Reactividad por Propósito**: Leisure más afectado que Work
- **Reactividad por Región**: Clientes europeos más impactados
- **Reactividad por CodeShare**: LATAM (-100.0) y VY (-50.0) devastados

**Verbatims Representativos:**
❌ **Ausencia total de feedback cualitativo** - Característica definitoria del "problema silencioso"

#### **PATRÓN DE "CRISIS SILENCIOSA":**
- **0 incidentes NCS reportados** formalmente
- **0 verbatims capturados** de clientes afectados
- **Impacto real confirmado** por métricas operativas y NPS
- **Brecha crítica** entre experiencia real y sistemas de monitoreo

---

## **CAUSA SECUNDARIA: MEJORA OPERATIVA SIMULTÁNEA EN LONG HAUL**

### **A. NATURALEZA DE LA CAUSA**

**Hipótesis Causal:**
Simultáneamente a la crisis en Short Haul, las operaciones Long Haul experimentaron una **mejora operativa significativa** en los mismos procesos de Mishandling (-9.3pts mejora), generando un efecto compensatorio parcial que enmascaró la severidad real del problema a nivel global.

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE**

#### **SEGMENTO MÁS GRANDE AFECTADO: Global/LH (Long Haul Complete)**

**OUTPUT CAUSAL DETALLADO:**
- **Anomalía NPS**: +9.1 puntos (31.4 vs 22.3 baseline)
- **Driver Principal**: Mishandling mejorado 27.2% vs 17.9% (-9.3pts mejora)
- **Alcance**: Esta mejora afecta a **TODOS los subsegmentos** bajo Global/LH incluyendo:
  - Business LH (+10.72pts con mejora operativa real)
  - Premium LH (+74.4pts con sesgo estadístico adicional)

**Ruta Beneficiada Principal:**
- **EZE-MAD**: NPS 30.8 (n=26) - Concentración de la mejora operativa

---

## **🎯 SÍNTESIS CAUSAL INTEGRADA**

El 28 de agosto de 2025 representa un **caso excepcional de segregación operativa** donde:

1. **Short Haul** sufrió crisis de Mishandling (+52%) concentrada en BCN-MAD
2. **Long Haul** experimentó mejora de Mishandling (-34%) concentrada en EZE-MAD  
3. **Efectos compensatorios parciales** generaron impacto global moderado (-2.28pts)
4. **Patrón de invisibilidad sistémica** con 0 reportes formales pero impacto real confirmado

**Implicación Estratégica:** La aparente "normalidad" del impacto global (-2.28pts) enmascara una crisis operativa severa en Short Haul que requiere investigación urgente de los procesos de manejo de equipajes en operaciones domésticas.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El 28 de agosto de 2025 se caracterizó por una divergencia operativa excepcional entre radios que generó impactos contrastantes en la satisfacción del cliente. A nivel global, el NPS experimentó una caída moderada de 2.28 puntos (de 27.23 a 24.95), resultado de una compensación parcial entre una severa crisis en Short Haul (-8.1 puntos, NPS de 21.7) y una mejora simultánea en Long Haul (+9.1 puntos, NPS de 31.4). La causa principal fue una crisis operativa de manejo de equipajes que afectó exclusivamente las operaciones Short Haul, donde el Mishandling se disparó un 52% por encima del promedio (27.2% vs 17.9%), mientras que Long Haul experimentó una mejora del 34% en la misma métrica. Esta crisis se manifestó como un "problema silencioso" sin generar incidentes formales (0 NCS reportados) ni feedback cualitativo (0 verbatims), pero con impacto real confirmado por las métricas operativas y la caída del NPS en todos los segmentos Short Haul.

Las rutas más afectadas fueron Barcelona-Madrid (BCN-MAD) con un NPS de -23.8 como epicentro de la crisis Short Haul, contrastando con Buenos Aires-Madrid (EZE-MAD) que registró un NPS de 30.8 beneficiándose de las mejoras Long Haul. Los grupos de clientes más reactivos fueron los pasajeros Business, que mostraron una sensibilidad 5 veces mayor que Economy (Business SH cayó 29.32 puntos vs Economy SH que cayó 5.68 puntos), especialmente los viajeros de ocio europeos y aquellos utilizando códigos compartidos como LATAM (NPS -100.0) y Vueling (NPS -50.0 en algunos segmentos).

**ECONOMY SH: Crisis Operativa Contenida**
La cabina Economy de Short Haul experimentó un deterioro moderado durante el 28 de agosto, registrando un NPS de 23.53 con una caída de 5.68 puntos respecto al baseline. La causa principal fue el incremento crítico del 52% en Mishandling de equipajes (27.2% vs 17.9% promedio), que se concentró geográficamente en la ruta Barcelona-Madrid con un NPS de -17.6. Esta crisis operativa afectó desproporcionadamente a los clientes Business dentro de Economy (NPS -1.8) versus Leisure (NPS 29.1), mientras que los códigos compartidos como LATAM registraron impactos devastadores (NPS -100.0) y las flotas A33ACMI mostraron el peor desempeño (NPS -40.0).

**BUSINESS SH: Colapso por Hipersensibilidad Operativa**
El segmento Business de Short Haul sufrió un colapso severo, registrando un NPS de 5.7 con una caída dramática de 29.32 puntos vs el baseline de 35.0. Esta evolución se explica principalmente por el mismo problema de Mishandling (+9.8 puntos vs promedio) pero amplificado por una dispersión regional extrema de 150 puntos entre diferentes perfiles de clientes, siendo especialmente visible en rutas como Londres-Madrid donde los códigos compartidos BA registraron NPS de -33.3, y entre perfiles como clientes sin región definida que colapsaron completamente (NPS -100.0) versus clientes norteamericanos que mantuvieron experiencias positivas (NPS +50.0).

**ECONOMY LH: Estabilidad Semanal**
La cabina Economy de Long Haul mantuvo desempeño estable durante esta semana. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción.

**BUSINESS LH: Mejora por Optimización Operativa**
La cabina Business de Long Haul experimentó una mejora significativa, registrando un NPS de 51.72 con un incremento de 10.72 puntos respecto al baseline de 40.99. Los drivers principales fueron la mejora sustancial en Mishandling (-34% vs promedio) que benefició especialmente a los clientes Leisure (83% de la muestra con NPS 45.8), impactando especialmente la ruta Madrid-México con NPS perfecto de 100.0 y perfiles de clientes que respondieron positivamente a la optimización de los procesos de manejo de equipajes y carga.

**PREMIUM LH: Sesgo Estadístico por Baja Participación**
El segmento Premium de Long Haul mostró una anomalía estadística con un NPS de 100.0 y una mejora aparente de 74.4 puntos vs la semana anterior. Las causas dominantes fueron un fenómeno de "sesgo de supervivencia" donde solo 6 clientes extremadamente satisfechos completaron las encuestas mientras que los afectados por los mismos problemas operativos de Mishandling evitaron participar, creando una distorsión estadística que enmascara la realidad operativa donde coexistieron problemas severos (Mishandling 27.2%) con un NPS aparentemente perfecto.

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

# 🏢 PASO 1: DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)

## **A. ECONOMY SHORT HAUL**

### **Nodos Disponibles:**
✅ **Economy SH/IB existe** - Nodo: Global/SH/Economy/IB  
✅ **Economy SH/YW existe** - Nodo: Global/SH/Economy/YW

### **Análisis Comparativo de Patrones:**

**DRIVERS PRINCIPALES:**
- **SH/IB:** Mishandling crítico (28.0% vs 18.2% baseline, +9.8pts)
- **SH/YW:** Mishandling crítico (24.9% vs 17.1% baseline, +7.8pts)

**EVIDENCIA OPERATIVA:**
- **SH/IB:** Concentración en ruta AMS-MAD, flota A332, CodeShare BA crítico
- **SH/YW:** Concentración en flota CRJ (81% del impacto), problemas sistémicos

**PATRONES DIVERGENTES:** Las compañías muestran **problemáticas operativas diferentes** - IB con problemas específicos de codeshare y rutas internacionales, YW con crisis sistémica de flota CRJ.

**🎯 DIAGNÓSTICO ECONOMY SH:** **Causa específica de compañía** - IB y YW divergen significativamente en flotas afectadas, rutas problemáticas y naturaleza operativa del problema.

---

## **B. BUSINESS SHORT HAUL**

### **Nodos Disponibles:**
✅ **Business SH/IB existe** - Nodo: Global/SH/Business/IB  
✅ **Business SH/YW existe** - Nodo: Global/SH/Business/YW

### **Análisis Comparativo de Patrones:**

**DRIVERS PRINCIPALES:**
- **SH/IB:** Mishandling crítico (28.0% vs 18.2% baseline, +9.8pts)
- **SH/YW:** Mishandling crítico (24.9% vs 17.1% baseline, +7.8pts)

**EVIDENCIA OPERATIVA:**
- **SH/IB:** Gap sistémico masivo en captura de datos, ausencia total de información granular
- **SH/YW:** Impacto geográfico diferenciado (España vs Europa), dispersión crítica por región

**PATRONES CONVERGENTES:** Ambas compañías comparten el **mismo driver principal** (Mishandling crítico) con magnitudes similares, pero **divergen en manifestación operativa**.

**🎯 DIAGNÓSTICO BUSINESS SH:** **Causa común de cabina con manifestaciones específicas por compañía** - Driver común (Mishandling) pero impactos operativos diferenciados por geografía y sistemas de captura.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ PASO 2: DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)

## **A. SHORT HAUL - ANÁLISIS DE CABINAS**

### **Patrones por Cabina:**

**ECONOMY SH:**
- **Driver común:** Mishandling crítico (IB: +9.8pts, YW: +7.8pts)
- **Manifestación:** Problemas operativos específicos por compañía (AMS-MAD/A332 vs CRJ sistémico)
- **Reactividad:** Alta sensibilidad a problemas operativos

**BUSINESS SH:**
- **Driver común:** Mishandling crítico (IB: +9.8pts, YW: +7.8pts)
- **Manifestación:** Impacto geográfico diferenciado y gaps sistémicos
- **Reactividad:** Alta sensibilidad similar a Economy

### **Análisis de Convergencia:**
- **CONVERGENCIA TOTAL** en driver principal (Mishandling crítico)
- **CONVERGENCIA** en magnitud del problema operativo
- **DIVERGENCIA** solo en manifestaciones específicas por compañía
- **REACTIVIDAD EQUIVALENTE** entre cabinas - ambas muestran alta sensibilidad

**🎯 DIAGNÓSTICO SHORT HAUL:** **Causa común de radio** - Economy y Business SH convergen completamente en driver principal y reactividad, con divergencias solo a nivel de manifestación operativa específica por compañía.

---

## **B. LONG HAUL - ANÁLISIS DE CABINAS**

### **Patrones por Cabina:**

**ECONOMY LH:**
- **Driver:** Mishandling crítico (27.2% vs 17.9%, +9.3pts)
- **Manifestación:** Ruta EZE-MAD, flota A350/A332, codeshares internacionales
- **Reactividad:** Moderada (-0.55pts NPS)

**BUSINESS LH:**
- **Driver:** Mishandling crítico (27.2% vs 17.9%, +9.3pts)
- **Manifestación:** Anomalía positiva (+19.0pts) por efecto compensación
- **Reactividad:** **AMORTIGUADOR** - Segmentos satisfechos compensan problemas

**PREMIUM LH:**
- **Driver:** Mishandling crítico (27.2% vs 17.9%, +9.3pts)
- **Manifestación:** Ruta MAD-MEX, flota A350 next, América Centro
- **Reactividad:** Muy alta (-11.27pts NPS)

### **Progresión de Reactividad:**
**Premium LH** (muy alta reactividad) > **Economy LH** (reactividad moderada) > **Business LH** (efecto amortiguador)

### **Análisis de Convergencia:**
- **CONVERGENCIA TOTAL** en driver operativo (Mishandling +9.3pts)
- **DIVERGENCIA CRÍTICA** en reactividad al problema
- **Business LH actúa como AMORTIGUADOR** - mismo problema operativo genera NPS positivo por compensación de segmentos satisfechos

**🎯 DIAGNÓSTICO LONG HAUL:** **Causa común de radio con reactividad específica de cabina** - Todas las cabinas LH comparten el mismo driver operativo (Mishandling crítico), pero muestran reactividades diferenciales extremas, con Business actuando como amortiguador ante el problema común.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 PASO 3: DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

## **A. COMPARACIÓN ENTRE RADIOS**

### **Afectación por Radio:**
✅ **SHORT HAUL afectado** - Anomalías negativas en todas las cabinas  
✅ **LONG HAUL afectado** - Anomalías mixtas (negativas en Economy/Premium, positiva en Business)

### **Convergencia en Drivers Principales:**

**SHORT HAUL:**
- **Driver:** Mishandling crítico (IB: 28.0%, YW: 24.9% vs ~18% baseline)
- **Magnitud:** +7.8 a +9.8 puntos porcentuales

**LONG HAUL:**
- **Driver:** Mishandling crítico (27.2% vs 17.9% baseline)
- **Magnitud:** +9.3 puntos porcentuales

### **Consistencia de Evidencia Operativa:**
- **CONVERGENCIA TOTAL** en driver principal (Mishandling crítico)
- **CONVERGENCIA** en magnitudes similares (~+8 a +9.3 puntos)
- **CONVERGENCIA** en otros factores operativos (Load Factor elevado, OTP mejorada paradójicamente)
- **DIVERGENCIA** solo en manifestaciones específicas por radio/ruta

---

## **B. COHERENCIA CON NODO GLOBAL**

### **Impacto Agregado Global:**
- **NPS Global:** 24.26 vs 27.23 baseline (-2.97pts)
- **Driver Global:** Mishandling crítico (27.2% vs 17.9%, +9.3pts)
- **Factores secundarios:** Load Factor 88.3% vs 87.6% (+0.6pts)

### **Coherencia con Patrones de Radio:**

**AGREGACIÓN MATEMÁTICA COHERENTE:**
- Short Haul: Impactos negativos generalizados
- Long Haul: Impactos mixtos (Business positivo compensa Economy/Premium negativos)
- **Resultado Global:** Impacto neto negativo moderado (-2.97pts) que refleja la suma algebraica de ambos radios

### **Manifestación Global vs Individual:**
- **AMPLIFICA** el patrón común de Mishandling crítico
- **ATENÚA** los impactos extremos por efecto de compensación entre segmentos
- **REFLEJA** coherentemente la problemática operativa subyacente común

---

## **🎯 DIAGNÓSTICO FINAL**

**Alcance:** **Causa global homogénea** - Ambos radios (Short Haul y Long Haul) convergen completamente en el mismo driver principal (Mishandling crítico con magnitudes similares +8 a +9.3 puntos), evidencia operativa consistente, y el nodo Global refleja coherentemente un impacto agregado que resulta de la suma algebraica de anomalías específicas por radio, donde efectos compensatorios en Business LH atenúan pero no eliminan el impacto operativo sistémico del 27 de agosto de 2025.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 PASO 4: ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

## **CAUSA ÚNICA IDENTIFICADA: CRISIS OPERATIVA DE MISHANDLING**

### **A. NATURALEZA DE LA CAUSA**

**Hipótesis:** El 27 de agosto de 2025 se produjo una **crisis operativa sistémica de manejo de equipajes y servicios** que afectó a toda la red operativa de Iberia/Air Europa. Esta crisis se caracterizó por un incremento crítico del 46-53% en incidentes de Mishandling, que se manifestó de forma diferenciada según rutas específicas, tipos de flota y perfiles de clientes, pero mantuvo una naturaleza operativa común subyacente.

**Driver Operativo:** Falla sistémica en procesos de handling de equipajes, posiblemente relacionada con sobrecarga operativa, problemas técnicos en aeropuertos hub, o fallas en coordinación con partners de servicios terrestres.

---

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE**

#### **Segmento Más Grande Afectado: GLOBAL**
*Representa la totalidad de la operación y todos los subsegmentos bajo este nodo*

**📊 MÉTRICAS CLAVE:**
- **NPS Impactado:** 24.26 vs 27.23 baseline (-2.97 puntos)
- **Driver Principal:** Mishandling crítico 27.2% vs 17.9% promedio (+9.3 puntos porcentuales)
- **Incremento Relativo:** +52% en incidentes de Mishandling
- **Load Factor:** 88.3% vs 87.6% promedio (+0.6 puntos) - Factor amplificador

**🗺️ RUTAS CRÍTICAS IDENTIFICADAS:**
- **LIS-MAD:** NPS -25.0 (n=4) - Epicentro de problemas
- **EZE-MAD:** NPS 26.9 (n=26) - Ruta intercontinental afectada  
- **BOG-MAD:** NPS -20.0 (n=5) - Concentración Business
- **AMS-MAD:** NPS 0.0 (n=4) - Codeshare crítico
- **BCN-MAD:** NPS -7.7 (n=13) - Flota ACMI problemática

**✈️ FLOTAS ESPECÍFICAMENTE AFECTADAS:**
- **A350 next:** NPS -12.5 vs A350 estándar NPS 50.0 (diferencial -62.5 puntos)
- **A332:** NPS -28.6 (múltiples rutas afectadas)
- **CRJ:** NPS 19.0 (81% del impacto en YW Economy)
- **A33ACMI:** NPS 7.7 (problemas específicos BCN-MAD)

**👥 SEGMENTOS MÁS CRÍTICOS:**
- **Codeshare Qatar Airways:** NPS -100.0 (impacto total)
- **Codeshare British Airways:** NPS -53.8 (muy negativo)
- **Codeshare American Airlines:** NPS -33.3 (crisis severa)
- **Región ASIA:** NPS -100.0 (impacto total)
- **Región EUROPA:** NPS -6.7 a -54.5 (según subsegmento)

**🔍 EVIDENCIA OPERATIVA ESPECÍFICA:**
- **Paradoja de Puntualidad:** OTP mejorado (85.8% vs 84.4%) mientras Mishandling se deterioraba
- **Conexiones Estables:** Misconex 0.6% vs 0.9% - Sin problemas de disrupciones
- **Sobrecarga Operativa:** Load Factor elevado amplificó el impacto de problemas de servicio

**📝 FEEDBACK CUALITATIVO:**
- **Incidentes NCS:** 0 reportados formalmente (característica "silenciosa" del problema)
- **Verbatims:** 0 disponibles (gap sistémico en captura)
- **Interpretación:** Problemas operativos "normalizados" que no escalaron formalmente pero impactaron directamente el NPS

---

### **ALCANCE CONFIRMADO DE LA CAUSA**

**✅ AFECTA A TODOS LOS SUBSEGMENTOS:**
- **Short Haul:** Economy IB/YW, Business IB/YW
- **Long Haul:** Economy, Business, Premium  
- **Todas las regiones geográficas** con impactos diferenciados
- **Múltiples flotas** con manifestaciones específicas
- **Todos los tipos de codeshare** con severidad variable

**🎯 MANIFESTACIÓN DIFERENCIADA:**
La causa común (Mishandling crítico) se manifestó de forma específica según:
- **Radio:** SH con problemas de flota/ruta, LH con problemas de codeshare internacional
- **Cabina:** Premium más reactivo, Business con efectos compensatorios
- **Compañía:** IB con problemas de codeshare, YW con crisis de flota CRJ
- **Geografía:** Concentración en rutas hacia Madrid como hub principal

**CONCLUSIÓN:** Una sola causa operativa sistémica (Crisis de Mishandling) con múltiples manifestaciones específicas según el contexto operativo de cada subsegmento.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El 27 de agosto de 2025 se registró una caída generalizada del NPS a nivel global, pasando de 27.23 a 24.26 puntos (-2.97 puntos), causada principalmente por una crisis operativa sistémica de manejo de equipajes que elevó el Mishandling del 17.9% al 27.2% (+9.3 puntos porcentuales). Esta problemática afectó de manera diferenciada a todos los segmentos: Short Haul Economy experimentó deterioros significativos tanto en IB (de 26.75 a 22.22 puntos, -4.53 puntos) como en YW (de 33.74 a 25.77 puntos, -7.97 puntos), mientras que Short Haul Business mostró caídas moderadas en IB (de 40.77 a 37.5 puntos, -3.27 puntos) y severas en YW (de 74.45 a 45.45 puntos, -28.53 puntos). En Long Haul, Economy registró una caída leve (de 19.81 a 19.26 puntos, -0.55 puntos), Premium sufrió un deterioro considerable (de 25.56 a 14.29 puntos, -11.27 puntos), mientras que Business experimentó paradójicamente una mejora excepcional (de 41.0 a 60.0 puntos, +19.0 puntos) debido a efectos compensatorios de segmentos altamente satisfechos que contrarrestaron los problemas operativos localizados.

Las rutas más críticas se concentraron en conexiones hacia Madrid como hub principal: LIS-MAD registró un NPS devastador de -25.0, AMS-MAD alcanzó 0.0 puntos con 100% de detractores, EZE-MAD mostró 26.9 puntos, BOG-MAD registró -20.0, y BCN-MAD cayó a -7.7 puntos. Los grupos de clientes más reactivos fueron los codeshares internacionales, especialmente Qatar Airways (NPS -100.0), British Airways (NPS -53.8) y American Airlines (NPS -33.3), junto con pasajeros de regiones como Asia (NPS -100.0) y Europa (variando entre -6.7 y -54.5 según subsegmento), mientras que flotas específicas como A350 next, A332, CRJ y A33ACMI concentraron los mayores impactos negativos debido a problemas operativos localizados pero severos.

**ECONOMY SH: Crisis Operativa Diferenciada por Compañía**
La cabina Economy de Short Haul experimentó un deterioro significativo durante el 27 de agosto, con IB registrando un NPS de 22.22 puntos (-4.53 puntos vs baseline) y YW alcanzando 25.77 puntos (-7.97 puntos respecto al período anterior). La causa principal fue el incremento crítico del Mishandling (IB: 28.0% vs 18.2% promedio, YW: 24.9% vs 17.1% promedio), que se manifestó de forma diferenciada: IB concentró problemas en la ruta AMS-MAD con codeshare British Airways y flota A332, mientras YW sufrió una crisis sistémica de la flota CRJ que representó el 81% del impacto total. Este deterioro se reflejó especialmente en rutas como AMS-MAD (NPS 0.0) para IB y operaciones CRJ generalizadas para YW, mientras que los perfiles más reactivos incluyeron codeshares internacionales, clientes europeos y pasajeros de América Norte.

**BUSINESS SH: Impacto Moderado con Gaps Sistémicos**
El segmento Business de Short Haul mostró caídas diferenciadas, registrando IB un NPS de 37.5 puntos (-3.27 puntos) y YW 45.45 puntos (-28.53 puntos vs la semana anterior). Esta evolución se explica principalmente por el mismo incremento crítico de Mishandling (IB: +9.8 puntos, YW: +7.8 puntos), siendo especialmente visible la ausencia de datos granulares en IB que sugiere gaps sistémicos en la captura de feedback, mientras que YW mostró un impacto geográfico diferenciado con España (NPS 50.0) menos afectada que Europa (NPS 75.0) y perfiles Leisure más críticos que los corporativos.

**ECONOMY LH: Resistencia Relativa con Problemas Localizados**
La cabina Economy de Long Haul mantuvo una resistencia relativa al deterioro operativo, registrando un NPS de 19.26 puntos (-0.55 puntos respecto a la semana anterior). La causa principal fue el mismo Mishandling crítico (27.2% vs 17.9% promedio) combinado con sobrecarga operativa (Load Factor 92.1% vs 90.3%), pero su impacto se concentró específicamente en la ruta EZE-MAD con un NPS de 26.9 puntos. Esta caída moderada se reflejó especialmente en rutas intercontinentales sudamericanas y entre codeshares Qatar Airways (NPS -100.0) y LATAM (NPS -40.0), mientras que los perfiles más reactivos incluyeron pasajeros europeos (NPS -54.5) y de origen sudamericano en rutas específicas.

**BUSINESS LH: Anomalía Positiva por Compensación Segmentada**
La cabina Business de Long Haul experimentó paradójicamente una mejora excepcional, registrando un NPS de 60.0 puntos (+19.0 puntos vs el período anterior). Los drivers principales fueron efectos compensatorios donde segmentos altamente satisfechos (América Sur NPS 100.0, Leisure NPS 66.7, flota A332 NPS 100.0) contrarrestaron matemáticamente los problemas operativos localizados en la ruta BOG-MAD (NPS -20.0), impactando especialmente a clientes españoles (NPS 20.0) y viajeros corporativos en flota A350, mientras que los perfiles más resilientes incluyeron pasajeros sudamericanos y de propósito Leisure que mantuvieron experiencias completamente positivas.

**PREMIUM LH: Máxima Reactividad a Problemas Operativos**
El segmento Premium de Long Haul sufrió el mayor deterioro proporcional, registrando un NPS de 14.29 puntos (-11.27 puntos vs la semana anterior). Las causas dominantes fueron el Mishandling crítico (27.2% vs 17.9%) combinado con alta ocupación (Load Factor 91.5% vs 90.3%), especialmente evidentes en la ruta Madrid-México con flota A350 next que mostró un NPS de -12.5 puntos (62.5 puntos inferior al A350 estándar), y entre perfiles de América Centro (NPS 0.0) y viajeros Leisure que demostraron la máxima sensibilidad a los problemas operativos de equipaje y servicio.

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

# 🏢 **PASO 1: DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)**

## **A. Economy Short Haul:**

**Nodos identificados:**
- ✅ **Economy SH/YW existe**: Global/SH/Economy/YW
- ❌ **Economy SH/IB NO existe**: Sin nodo específico en el árbol

**Razón de ausencia IB**: El análisis del nodo Global/SH/Economy muestra que el 86% del volumen (89 de 104 encuestas) corresponde a **código IB**, mientras que YW representa una fracción menor. La anomalía se concentró en el segmento YW, por lo que solo se creó nodo específico para este último.

---

## **B. Business Short Haul:**

**Nodos identificados:**
- ✅ **Business SH/IB existe**: Global/SH/Business/IB  
- ✅ **Business SH/YW existe**: Global/SH/Business/YW

**Análisis comparativo de patrones:**

| **Factor** | **Business SH/IB** | **Business SH/YW** | **Convergencia** |
|------------|-------------------|-------------------|------------------|
| **Driver Principal** | Mishandling 28.0% (+9.8pp) | Mishandling 24.9% (+7.8pp) | ✅ **IDÉNTICO** |
| **Dirección NPS** | +14.2 (positivo) | -50.26 (negativo) | ❌ **DIVERGENTE** |
| **Evidencia Operativa** | Problemas compensados por leisure travelers | Impacto uniforme en todos los perfiles | ❌ **DIVERGENTE** |
| **Patrón de Escalación** | Sin NCS, problemas "silenciosos" | Sin NCS, problemas "silenciosos" | ✅ **IDÉNTICO** |

---

## **🎯 DIAGNÓSTICOS FINALES:**

**Economy Short Haul**: **Causa específica de compañía YW** - Solo YW mostró anomalía significativa, mientras que IB mantuvo operación estable.

**Business Short Haul**: **Causa común de cabina con manifestación diferencial** - Ambas compañías experimentaron el mismo driver operativo (mishandling crítico), pero con impactos divergentes debido a diferencias en composición de clientela y capacidad de compensación.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ **PASO 2: DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)**

## **A. Short Haul - Análisis de Cabinas:**

**Patrones identificados por cabina:**

| **Factor** | **Economy SH** | **Business SH** | **Convergencia** |
|------------|----------------|-----------------|------------------|
| **Driver Principal** | Mishandling 27.2% (+9.3pp) | Mishandling 27.2% (+9.3pp) | ✅ **IDÉNTICO** |
| **Magnitud Anomalía** | -0.25 pts (mínima) | -7.45 pts (significativa) | ❌ **DIVERGENTE** |
| **Reactividad** | Baja sensibilidad | Alta sensibilidad | ❌ **DIVERGENTE** |
| **Patrón Geográfico** | Concentrado MAD-NAP | Concentrado JFK-MAD | ❌ **DIVERGENTE** |
| **Escalación** | Sin NCS, "silencioso" | Sin NCS, "silencioso" | ✅ **IDÉNTICO** |

**Análisis de consistencia IB/YW:**
- **Economy**: Solo YW afectado (IB estable)
- **Business**: Ambos afectados pero con manifestaciones opuestas (IB compensado +14.2, YW crítico -50.26)

---

## **B. Long Haul - Análisis de Cabinas:**

**Patrones identificados por cabina:**

| **Factor** | **Economy LH** | **Business LH** | **Premium LH** |
|------------|----------------|-----------------|----------------|
| **Driver Principal** | Mishandling 27.2% (+9.3pp) | Mishandling 27.2% (+9.3pp) | *No disponible* |
| **Magnitud Anomalía** | -10.2 pts (alta) | -34.75 pts (crítica) | *No disponible* |
| **Reactividad** | Moderada | Extrema | *No disponible* |
| **Concentración Geográfica** | MAD-MEX | JFK-MAD | *No disponible* |
| **Escalación** | Sin NCS, "silencioso" | Sin NCS, "silencioso" | *No disponible* |

**Progresión de reactividad:**
- **Economy LH**: -10.2 pts (reactividad moderada)
- **Business LH**: -34.75 pts (reactividad extrema - 3.4x mayor impacto)
- **Premium LH**: Sin nodo específico (sin anomalía detectada)

---

## **🎯 DIAGNÓSTICOS FINALES:**

**Short Haul**: **Causa común de radio con reactividad diferencial por cabina** - Ambas cabinas experimentaron el mismo driver operativo (mishandling crítico), pero Business mostró 30x mayor reactividad que Economy, evidenciando que los clientes premium son significativamente más sensibles a problemas operativos.

**Long Haul**: **Causa común de radio con progresión lógica de reactividad** - El mismo driver operativo (mishandling) impactó ambas cabinas disponibles, pero con una progresión clara: Business LH mostró 3.4x mayor reactividad que Economy LH, confirmando que la sensibilidad aumenta proporcionalmente con el nivel de servicio contratado.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 **PASO 3: DIAGNÓSTICO A NIVEL RADIO Y GLOBAL**

## **A. Comparación Entre Radios:**

**Estado de afectación por radio:**

| **Radio** | **Afectación** | **Driver Principal** | **Magnitud** | **Evidencia Operativa** |
|-----------|----------------|---------------------|--------------|------------------------|
| **Short Haul** | ✅ Afectado | Mishandling 27.2% (+9.3pp) | -1.01 pts | Concentrado BUD-MAD/ATR |
| **Long Haul** | ✅ Afectado | Mishandling 27.2% (+9.3pp) | -11.52 pts | A321XLR MAD-NRT crítico |

**Convergencia de drivers:**
- ✅ **Driver idéntico**: Mishandling 27.2% vs 17.9% baseline (+9.3pp)
- ✅ **Métrica base común**: Mismo problema operativo sistémico
- ❌ **Manifestación diferente**: LH 11x más impacto que SH (-11.52 vs -1.01)
- ✅ **Patrón escalación**: Ambos "silenciosos" (sin NCS, sin verbatims)

---

## **B. Coherencia con Nodo Global:**

**Análisis de agregación:**

| **Nivel** | **NPS Impact** | **Driver** | **Patrón** |
|-----------|----------------|------------|------------|
| **Global** | -4.60 pts | Mishandling 27.2% (+9.3pp) | A321XLR MAD-NRT como epicentro |
| **Long Haul** | -11.52 pts | Mishandling 27.2% (+9.3pp) | Crisis operativa + partnerships |
| **Short Haul** | -1.01 pts | Mishandling 27.2% (+9.3pp) | Localizado BUD-MAD |

**Coherencia matemática:**
- ✅ **Agregación lógica**: Global (-4.60) refleja promedio ponderado entre LH (-11.52) y SH (-1.01)
- ✅ **Driver consistente**: Mishandling 27.2% aparece en los tres niveles
- ✅ **Causa raíz común**: A321XLR MAD-NRT identificado como epicentro en Global, correlaciona con crisis LH

**Amplificación/Atenuación:**
- **Long Haul domina el impacto Global** debido a mayor reactividad de clientes y rutas críticas
- **Short Haul atenúa** parcialmente el impacto global por menor sensibilidad
- **Concentración de riesgo**: Una aeronave específica (A321XLR) explica la mayor parte de la anomalía

---

## **🎯 DIAGNÓSTICO FINAL:**

**Causa global homogénea con manifestación diferencial por radio** - Ambos radios experimentaron el mismo evento operativo sistémico (mishandling crítico del 27.2%), pero Long Haul amplificó el impacto 11x más que Short Haul debido a mayor reactividad de clientes premium y criticidad de rutas intercontinentales. El nodo Global refleja coherentemente esta agregación, con el A321XLR MAD-NRT actuando como epicentro que explica tanto la anomalía global como la crisis específica de Long Haul.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 **PASO 4: ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS**

## **CAUSA ÚNICA IDENTIFICADA: CRISIS SISTÉMICA DE MISHANDLING**

---

### **A. NATURALEZA DE LA CAUSA:**

**Hipótesis Causal:**
El 26 de agosto de 2025 experimentó una **falla sistémica en los procesos de manejo de equipajes y servicios auxiliares**, con un incremento del 52% respecto al baseline normal (27.2% vs 17.9%). Esta crisis operativa se manifestó de forma "silenciosa" - sin generar escalamientos formales - pero impactó severamente la experiencia del cliente, especialmente en rutas intercontinentales y clientes premium que tienen mayor sensibilidad a disrupciones operativas.

**Mecanismo de Impacto:**
- **Operativo**: Fallas en handling, carga, servicios de rampa
- **Experiencial**: Degradación percibida sin queja formal
- **Diferencial**: Mayor impacto en clientes premium y rutas de larga distancia

---

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE:**

**Segmento más grande afectado:** **Global** (nivel compañía completo)

### **OUTPUT CAUSAL DETALLADO - NODO GLOBAL:**

**📊 MÉTRICAS CLAVE:**
- **NPS Impact**: -4.60 puntos (22.63 vs 27.23 baseline)
- **Driver Principal**: Mishandling crítico 27.2% vs 17.9% promedio (+9.3 puntos, +52% incremento relativo)
- **Load Factor**: 88.3% vs 87.6% promedio (presión operativa adicional)
- **Puntualidad**: 85.8% vs 84.4% (mejorada - no explica la anomalía)

**🎯 EPICENTRO IDENTIFICADO:**
- **Aeronave crítica**: A321XLR operando ruta MAD-NRT
- **NPS catastrófico**: -100.0 (6 encuestas)
- **Concentración**: 100% del impacto negativo en una sola aeronave/ruta intercontinental

**🗺️ RUTAS INVOLUCRADAS:**
- **Crítica**: MAD-NRT (Madrid → Narita) - A321XLR con NPS -100.0
- **Secundarias**: JFK-MAD (Business), BUD-MAD (Short Haul), MAD-MEX (Economy)

**👥 DISPERSIÓN DE IMPACTO:**
- **Fleet Dispersión**: 185.7 puntos (máxima del análisis)
- **Partnerships afectados**: American Airlines (NPS -62.5), British Airways (NPS -20.0)
- **Segmentos críticos**: Business travelers (mayor reactividad), rutas transatlánticas

**💬 VERBATIMS REPRESENTATIVOS:**
❌ **Ausencia total de feedback cualitativo** - Característica distintiva de esta anomalía "silenciosa"

**📈 PATRÓN DE ESCALACIÓN:**
- **NCS Reportados**: 0 incidentes formales
- **Verbatims**: 0 comentarios registrados  
- **Manifestación**: Exclusivamente vía puntuaciones NPS negativas

---

### **🌐 ALCANCE DE AFECTACIÓN:**

**Esta causa afecta TODOS los subsegmentos bajo el nodo Global:**

✅ **Long Haul** (-11.52 pts): Crisis amplificada por reactividad premium y rutas críticas
✅ **Short Haul** (-1.01 pts): Impacto atenuado pero presente  
✅ **Economy LH** (-10.2 pts): Moderada reactividad a problemas operativos
✅ **Business LH** (-34.75 pts): Máxima reactividad (3.4x vs Economy)
✅ **Economy SH/YW** (-6.7 pts): Segmentos específicos afectados
✅ **Business SH** (variable): IB compensado (+14.2), YW crítico (-50.26)

---

### **🔍 CONFIANZA Y VALIDACIÓN:**

**Nivel de Confianza**: **ALTO (95%)**
- ✅ Correlación perfecta entre datos operativos y NPS across todos los segmentos
- ✅ Identificación específica de aeronave y epicentro (A321XLR MAD-NRT)
- ✅ Consistencia en ausencia de reportes formales vs impacto operativo medible
- ✅ Patrón coherente de reactividad diferencial por tipo de cliente

**Limitación**: La ausencia de feedback cualitativo impide detalles específicos de la experiencia, pero no afecta la identificación sólida de la causa raíz operativa.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El 26 de agosto de 2025 registró una caída generalizada del NPS corporativo de 4.60 puntos (de 27.23 a 22.63), causada por una crisis sistémica de manejo de equipajes que elevó el mishandling al 27.2% (+9.3 puntos vs promedio). Esta degradación operativa impactó de forma diferencial: Long Haul experimentó el mayor deterioro con 11.52 puntos de caída (de 22.29 a 10.77), mientras Short Haul mostró una reducción menor de 1.01 puntos (de 29.8 a 28.8). La anomalía se concentró especialmente en la aeronave A321XLR operando la ruta Madrid-Narita, que registró un NPS catastrófico de -100.0 con 6 encuestas, explicando gran parte del impacto global. Los partnerships de codeshare también colapsaron, con American Airlines alcanzando un NPS de -62.5 y British Airways -20.0, evidenciando problemas de coordinación operativa.

Las rutas más afectadas incluyeron Madrid-Narita (NPS -100.0, A321XLR), JFK-Madrid en Business (-100.0), Budapest-Madrid en Short Haul (NPS 22.2), y Madrid-México en Economy (NPS 27.3). Los clientes Business demostraron ser los más reactivos a los problemas operativos, especialmente los viajeros de negocios de América del Norte, mientras que los pasajeros de ocio mostraron mayor tolerancia. La característica distintiva de esta crisis fue su naturaleza "silenciosa" - sin generar incidentes NCS formales ni verbatims de clientes, manifestándose exclusivamente a través de las puntuaciones NPS negativas.

**ECONOMY SH: Estabilidad con Focos Localizados**
La cabina Economy de Short Haul mantuvo relativa estabilidad durante el 26 de agosto, registrando un NPS de 28.96 con una ligera caída de 0.25 puntos respecto al baseline. La causa principal fue el incremento en mishandling de equipajes (+9.3 puntos porcentuales), que se concentró específicamente en la ruta Madrid-Nápoles operada con aeronave A332, donde el NPS cayó a 16.7. Los pasajeros europeos y aquellos en vuelos operados por partners de codeshare fueron los perfiles más reactivos, mientras que las operaciones directas de Iberia mantuvieron niveles satisfactorios de NPS.

**BUSINESS SH: Divergencia por Compañía**
El segmento Business de Short Haul mostró comportamientos contrastantes por compañía, con Iberia registrando una mejora de 14.2 puntos (de 40.8 a 55.0) mientras Vueling experimentó una caída crítica de 50.26 puntos (de 16.92 a -33.33). Esta divergencia se explica por el mismo problema operativo de mishandling (+9.8 puntos para IB, +7.8 para YW), pero con capacidades diferentes de compensación: Iberia logró que sus pasajeros de ocio (68.8 NPS) compensaran el impacto negativo en viajeros de negocios (0.0 NPS), mientras Vueling experimentó un impacto uniforme y severo en todos los perfiles de cliente.

**ECONOMY LH: Deterioro Moderado**
La cabina Economy de Long Haul experimentó un deterioro significativo de 10.2 puntos, pasando de un NPS de 19.8 a 9.6, causado principalmente por el mismo problema sistémico de mishandling (27.2% vs 17.9% promedio). Esta degradación se concentró en la ruta Madrid-México y afectó especialmente a la flota A321XLR (NPS -100.0) y A350 (NPS -2.9). Los residentes españoles y clientes en vuelos de codeshare con American Airlines y British Airways fueron los perfiles más reactivos, mientras que los vuelos directos de Iberia mantuvieron mejor desempeño relativo.

**BUSINESS LH: Impacto Crítico**
La cabina Business de Long Haul registró el mayor deterioro con una caída de 34.75 puntos, de 40.99 a 6.25, evidenciando la extrema sensibilidad de los clientes premium a problemas operativos. Los drivers principales fueron el mishandling crítico y la concentración del impacto en la ruta JFK-Madrid, donde el 100% de los clientes fueron detractores (NPS -100.0). Los viajeros de negocios de América del Norte demostraron ser el perfil más reactivo, contrastando con clientes de América del Sur que mantuvieron satisfacción alta, creando una dispersión de 150 puntos entre regiones de residencia.

**PREMIUM LH: Desempeño Estable**
El segmento Premium de Long Haul mantuvo desempeño estable durante el 26 de agosto, sin registrar anomalías significativas que requirieran análisis específico. No se detectaron cambios relevantes en este segmento, manteniendo niveles consistentes de satisfacción a pesar de los problemas operativos que afectaron otras cabinas.

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

# 🏢 DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW) - SHORT HAUL

## **A. ECONOMY SHORT HAUL**

### **Nodos Identificados:**
- ✅ **Economy SH/YW**: Nodo Global/SH/Economy/YW presente
- ❌ **Economy SH/IB**: Nodo NO presente en el árbol

### **Análisis de Presencia:**
Solo existe el nodo YW en Economy Short Haul. La ausencia del nodo IB sugiere que **no hubo anomalía significativa en Economy SH/IB** o que la anomalía fue absorbida por el nodo padre Economy SH general.

### **Patrón Identificado en YW:**
- **Driver principal**: Mishandling 24.9% (+7.8pp vs baseline)
- **Impacto segmentado**: Clientes internacionales más afectados
- **Flota específica**: Problemas concentrados en CRJ

**DIAGNÓSTICO ECONOMY SH**: **Causa específica de YW** - Los problemas operativos se concentraron exclusivamente en operaciones YW, sin impactar significativamente a IB.

---

## **B. BUSINESS SHORT HAUL**

### **Nodos Identificados:**
- ✅ **Business SH/IB**: Nodo Global/SH/Business/IB presente  
- ✅ **Business SH/YW**: Nodo Global/SH/Business/YW presente

### **Análisis Comparativo:**

| **Aspecto** | **Business SH/IB** | **Business SH/YW** |
|-------------|-------------------|-------------------|
| **Driver Principal** | Mishandling 28.0% (+54%) | Mishandling 24.9% (+46%) |
| **Magnitud NPS** | -9.23 puntos | -33.6 puntos |
| **Flota Afectada** | A321 (crítico) | CRJ (exclusivo) |
| **Perfil Cliente** | Europeos más afectados | Business/Work devastado |
| **Volumen** | 28 encuestas | 6 encuestas |

### **Convergencias Identificadas:**
- **Mismo driver operativo**: Mishandling como causa raíz en ambas compañías
- **Mismo día**: Problemas operativos simultáneos
- **Misma naturaleza**: Problemas de handling/equipaje

### **Divergencias Identificadas:**
- **Flotas diferentes**: A321 (IB) vs CRJ (YW)
- **Impacto diferencial**: Moderado (IB) vs severo (YW)
- **Segmentación distinta**: Por región (IB) vs por propósito de viaje (YW)

**DIAGNÓSTICO BUSINESS SH**: **Causa común de cabina con manifestaciones específicas por compañía** - Ambas compañías experimentaron el mismo problema operativo (Mishandling crítico) pero en flotas y segmentos diferentes, sugiriendo un evento operativo sistémico que afectó a toda la operación Short Haul Business.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)

## **A. SHORT HAUL - ANÁLISIS DE CABINAS**

### **Patrones Identificados por Cabina:**

| **Cabina** | **Driver Principal** | **Magnitud** | **Flota Crítica** | **Reactividad** |
|------------|---------------------|--------------|-------------------|-----------------|
| **Economy SH** | Mishandling 24.9% (+7.8pp) | -7.6 pts NPS | CRJ | Moderada |
| **Business SH/IB** | Mishandling 28.0% (+54%) | -9.23 pts NPS | A321 | Alta |
| **Business SH/YW** | Mishandling 24.9% (+46%) | -33.6 pts NPS | CRJ | Extrema |

### **Análisis de Convergencia:**

#### **CONVERGENCIAS IDENTIFICADAS:**
- ✅ **Driver operativo idéntico**: Mishandling como causa raíz universal
- ✅ **Timing simultáneo**: Problemas operativos el mismo día
- ✅ **Naturaleza del problema**: Handling/equipaje en todas las cabinas

#### **DIVERGENCIAS CRÍTICAS:**
- 🔴 **Reactividad escalada por cabina**: Economy (-7.6) < Business IB (-9.23) < Business YW (-33.6)
- 🔴 **Flotas diferentes afectadas**: CRJ (Economy/YW) vs A321 (IB Business)
- 🔴 **Volumen de impacto**: Economy (111 encuestas) vs Business (28-6 encuestas)

#### **Patrón de Consistencia entre Compañías:**
- **IB vs YW**: Ambas experimentaron Mishandling pero en flotas distintas
- **Escalamiento progresivo**: YW Business mostró mayor vulnerabilidad que IB Business

**DIAGNÓSTICO SHORT HAUL**: **Causa común de radio con reactividad diferencial escalonada por cabina** - El mismo evento operativo (Mishandling crítico) afectó a todo Short Haul, pero la reactividad se intensificó progresivamente: Economy → Business IB → Business YW, sugiriendo mayor sensibilidad de clientes Business y específicamente en operaciones YW.

---

## **B. LONG HAUL - ANÁLISIS DE CABINAS**

### **Patrones Identificados por Cabina:**

| **Cabina** | **Anomalía NPS** | **Driver Principal** | **Muestra** | **Patrón** |
|------------|------------------|---------------------|-------------|------------|
| **Business LH** | +9.0 pts | Mishandling 27.2% (+9.3pp) | 19 encuestas | Paradoja positiva |
| **Premium LH** | -17.3 pts | Mishandling 27.2% (+9.3pp) | 6 encuestas | Impacto negativo |

### **Análisis de Convergencia:**

#### **CONVERGENCIAS IDENTIFICADAS:**
- ✅ **Driver operativo idéntico**: Mishandling 27.2% en ambas cabinas
- ✅ **Mismo incremento**: +9.3pp vs baseline en ambos casos
- ✅ **Ausencia de feedback formal**: Sin NCS ni verbatims en ambas

#### **DIVERGENCIAS CRÍTICAS:**
- 🔴 **Dirección opuesta del NPS**: Business (+9.0) vs Premium (-17.3)
- 🔴 **Reactividad inversa**: Business "resistió" vs Premium "colapsó"
- 🔴 **Volumen diferencial**: Business (19) vs Premium (6) encuestas

#### **Progresión de Reactividad:**
**PATRÓN ANÓMALO DETECTADO**: Contrario a la lógica esperada, la cabina Premium (mayor precio) mostró **mayor vulnerabilidad** que Business ante el mismo problema operativo.

#### **Análisis de "Amortiguador":**
- **Business LH**: Actuó como amortiguador, manteniendo NPS positivo a pesar de Mishandling crítico
- **Premium LH**: Mostró máxima sensibilidad, sin capacidad de amortiguación

**DIAGNÓSTICO LONG HAUL**: **Causa común de radio con reactividad inversa anómala** - El mismo problema operativo (Mishandling 27.2%) generó respuestas opuestas: Business LH logró mantener satisfacción positiva mientras Premium LH experimentó deterioro severo, sugiriendo que factores de volumen de muestra y gestión específica por cabina influyeron más que la lógica tradicional de mayor exigencia en cabinas superiores.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

## **A. COMPARACIÓN ENTRE RADIOS**

### **Estado de Afectación por Radio:**

| **Radio** | **Estado** | **Driver Principal** | **Magnitud** | **Patrón** |
|-----------|------------|---------------------|--------------|------------|
| **Short Haul** | ✅ AFECTADO | Mishandling 24.9-28.0% | Negativo (-7.6 a -33.6 pts) | Deterioro consistente |
| **Long Haul** | ✅ AFECTADO | Mishandling 27.2% | Mixto (+9.0 a -17.3 pts) | Paradoja operativa |

**CONCLUSIÓN**: **Ambos radios afectados simultáneamente**

### **Análisis de Convergencia entre Radios:**

#### **CONVERGENCIAS IDENTIFICADAS:**
- ✅ **Driver operativo idéntico**: Mishandling como causa raíz universal (24.9-28.0% vs baselines 17.1-17.9%)
- ✅ **Timing simultáneo**: Evento operativo del 25 de agosto impactó ambos radios
- ✅ **Naturaleza del problema**: Handling/equipaje en toda la operación
- ✅ **Ausencia de feedback formal**: Sin NCS ni verbatims en ambos radios

#### **DIVERGENCIAS CRÍTICAS:**
- 🔴 **Dirección del impacto**: SH uniformemente negativo vs LH con resultados mixtos
- 🔴 **Magnitud diferencial**: SH con deterioro moderado-severo vs LH con extremos opuestos
- 🔴 **Flotas afectadas**: SH (CRJ, A321) vs LH (sin especificación de flota)

### **Valores SHAP y Evidencia Operativa:**
- **Consistencia**: Mishandling como factor SHAP principal en ambos radios
- **Intensidad**: Similar magnitud operativa (+7.8 a +9.3pp) pero impacto NPS divergente

---

## **B. COHERENCIA CON NODO GLOBAL**

### **Análisis del Nodo Global:**

| **Métrica** | **Global** | **Coherencia con Radios** |
|-------------|------------|---------------------------|
| **NPS Anomalía** | +1.98 pts | ✅ Refleja compensación SH negativo vs LH mixto |
| **Mishandling** | 27.2% (+9.3pp) | ✅ Consistente con ambos radios |
| **Rutas Críticas** | GRX-MAD (-33.3) | ✅ Problemas localizados identificados |
| **Flotas Críticas** | A350 C (-71.4), A333 (7.9) | ✅ Problemas específicos confirmados |

### **Patrón de Agregación Global:**

#### **EFECTOS COMPENSATORIOS IDENTIFICADOS:**
1. **SH negativo** (-7.6 a -33.6 pts) + **LH Business positivo** (+9.0 pts) = **Global ligeramente positivo** (+1.98 pts)
2. **Volumen diferencial**: Operación mayoritaria (Iberia n=473) compensó problemas focalizados
3. **Concentración vs Dispersión**: Problemas severos en muestras pequeñas no dominaron estadísticamente

#### **AMPLIFICACIÓN vs ATENUACIÓN:**
- **Atenuación estadística**: El Global atenúa los extremos negativos por efecto de volumen
- **Visibilidad de problemas**: El Global mantiene visibilidad de rutas y flotas críticas
- **Paradoja operativa preservada**: El Global refleja la contradicción de problemas operativos severos con NPS agregado positivo

---

## **🎯 DIAGNÓSTICO FINAL - ALCANCE GLOBAL**

**CAUSA GLOBAL HOMOGÉNEA CON MANIFESTACIONES DIFERENCIADAS**: El evento del 25 de agosto de 2025 representa una causa operativa global (Mishandling crítico 27.2%) que afectó simultáneamente a ambos radios con el mismo driver principal, pero generó manifestaciones diferenciadas por radio - deterioro consistente en Short Haul vs resultados paradójicos en Long Haul - cuya agregación en el nodo Global produce un efecto compensatorio que atenúa la severidad de los problemas específicos manteniendo un NPS ligeramente positivo (+1.98 pts) que enmascara vulnerabilidades operativas críticas.

**IMPLICACIÓN ESTRATÉGICA**: La anomalía Global positiva es engañosa y requiere análisis granular para identificar los problemas operativos reales que afectaron segmentos específicos de manera severa.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

## **CAUSA ÚNICA IDENTIFICADA: CRISIS OPERATIVA DE MISHANDLING**

---

### **A. NATURALEZA DE LA CAUSA**

#### **HIPÓTESIS CAUSAL:**
El 25 de agosto de 2025 experimentó un **evento operativo crítico sistémico** caracterizado por el colapso de los procesos de handling (equipaje, carga y servicios terrestres) que afectó simultáneamente a toda la operación, generando un incremento del **46-54% en incidentes de Mishandling** respecto al baseline normal. Este evento operativo actuó como un **shock exógeno** que impactó diferencialmente según el perfil del cliente, tipo de flota y radio de operación, pero mantuvo una **causa raíz única y homogénea** en toda la red.

#### **MECANISMO CAUSAL:**
1. **Trigger operativo**: Fallo sistémico en procesos de handling
2. **Amplificación por Load Factor**: Alta ocupación (86-91%) intensificó el impacto
3. **Manifestación diferencial**: Mismo problema, reactividad distinta por segmento
4. **Ausencia de escalación**: Problemas no generaron feedback formal masivo

---

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE**

#### **SEGMENTO MÁS GRANDE AFECTADO: Global/SH/Economy/YW**
*Seleccionado por mayor volumen de muestra (111 encuestas) y representatividad operativa*

### **📊 OUTPUT CAUSAL DETALLADO - SEGMENTO PRINCIPAL**

#### **MÉTRICAS OPERATIVAS CRÍTICAS:**
- **Mishandling**: 24.9% vs 17.1% baseline (**+7.8pp / +46% incremento**)
- **Load Factor**: 86.1% vs 83.8% baseline (**+2.3pp**)
- **OTP15**: 79.7% vs 79.6% baseline (**+0.1pp** - neutro)
- **Misconex**: 0.0% vs 0.0% baseline (**sin variación**)

#### **IMPACTO NPS:**
- **Anomalía**: -7.6 puntos (de 33.7 a 26.1)
- **Muestra**: 111 encuestas (estadísticamente robusta)
- **Confianza**: Alta (correlación directa Mishandling ↔ NPS)

#### **SEGMENTACIÓN DEL IMPACTO:**
| **Perfil Cliente** | **NPS** | **Reactividad** |
|-------------------|---------|-----------------|
| **España** | +50.0 | Resistente |
| **Europa** | -8.7 | Moderadamente afectada |
| **América del Norte** | -66.7 | Severamente impactada |
| **Dispersión total** | 116.7 puntos | Extrema variabilidad |

#### **FLOTAS INVOLUCRADAS:**
- **CRJ**: NPS 24.7 (problemática)
- **ATR**: NPS 35.7 (mejor performance relativa)

---

### **🌐 ALCANCE COMPLETO DE LA CAUSA**

#### **SUBSEGMENTOS AFECTADOS BAJO EL NODO GLOBAL:**
La causa de **Mishandling crítico** afecta a **TODOS los subsegmentos** identificados:

1. **Global/SH/Economy/YW** ✅ (nodo principal analizado)
2. **Global/SH/Business/IB** ✅ (Mishandling 28.0%, NPS -9.23)
3. **Global/SH/Business/YW** ✅ (Mishandling 24.9%, NPS -33.6)
4. **Global/LH/Business** ✅ (Mishandling 27.2%, NPS +9.0 paradójico)
5. **Global/LH/Premium** ✅ (Mishandling 27.2%, NPS -17.3)

#### **RUTAS CRÍTICAS IDENTIFICADAS:**
- **GRX-MAD**: NPS -33.3 (n=3) - Epicentro del problema
- **MAD-MXP**: NPS -7.7 (n=13) - Ruta secundaria afectada
- **FCO-MAD**: NPS 66.7 (n=3) - Impacto variable

#### **CODESHARES PROBLEMÁTICOS:**
- **American Airlines**: NPS -35.7 (n=14)
- **LATAM**: NPS -7.7 (n=13)
- **British Airways**: NPS -60.0 (n=5)

#### **FLOTAS CRÍTICAS GLOBALES:**
- **A350 C**: NPS -71.4 (n=7) - Crisis extrema
- **A333**: NPS 7.9 (n=38) - Rendimiento bajo
- **A321**: NPS 16.7 (problemas específicos IB)
- **CRJ**: NPS -16.7 a 24.7 (problemas YW)

---

### **📝 VERBATIMS REPRESENTATIVOS**
**AUSENCIA TOTAL** - No se registraron verbatims en ningún segmento, confirmando el patrón de **"crisis silenciosa"** donde los problemas operativos severos no generaron feedback cualitativo formal.

---

### **🎯 SÍNTESIS CAUSAL CONSOLIDADA**

**El 25 de agosto de 2025 experimentó una crisis operativa sistémica de Mishandling que incrementó los incidentes en 46-54% respecto al baseline normal, afectando a toda la red operativa con manifestaciones diferenciadas: deterioro severo en Short Haul (-7.6 a -33.6 pts NPS), resultados paradójicos en Long Haul (+9.0 a -17.3 pts), concentración crítica en rutas específicas (GRX-MAD, MAD-MXP) y flotas vulnerables (A350 C, CRJ, A321), impacto desproporcionado en clientes internacionales vs españoles, y ausencia total de feedback formal que creó una "zona ciega" operativa donde problemas severos pasaron desapercibidos para sistemas tradicionales de monitoreo.**

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El 25 de agosto de 2025 presentó un patrón complejo de anomalías NPS caracterizado por una crisis operativa sistémica de handling que afectó diferencialmente a los segmentos. A nivel global, el NPS experimentó una mejora aparente de 1.98 puntos (de 27.23 a 29.21), pero este resultado agregado enmascaró deterioros severos en segmentos específicos. El segmento Global/SH/Economy/YW registró una caída de 7.6 puntos (de 33.7 a 26.1 NPS), mientras que Global/SH/Business/IB experimentó un deterioro de 9.23 puntos (de 50.0 a 40.8 NPS) y Global/SH/Business/YW sufrió la mayor caída con 33.6 puntos negativos (de 16.9 a -16.7 NPS). En contraste, Global/LH/Business mostró una mejora paradójica de 9.0 puntos (de 40.99 a 50.0 NPS) y Global/LH/Premium cayó 17.3 puntos (de 60.2 a 42.9 NPS). La causa raíz unificada fue un incremento crítico en el Mishandling del 46-54% respecto al baseline (de 17.1-17.9% a 24.9-28.0%), evidenciado por problemas severos de equipaje y servicios terrestres que no generaron feedback formal pero impactaron directamente la satisfacción del cliente.

Las rutas más afectadas incluyeron GRX-MAD con un NPS de -33.3 (muestra de 3 clientes) como epicentro del problema, MAD-MXP con NPS de -7.7 (13 encuestas) y operaciones de codeshare críticas como American Airlines (NPS -35.7) y British Airways (NPS -60.0). Los grupos de clientes más reactivos fueron los viajeros internacionales, especialmente norteamericanos que experimentaron caídas hasta -66.7 NPS en Economy SH, clientes europeos con deterioros significativos, y viajeros corporativos Business/Work que mostraron la mayor sensibilidad con NPS de -33.3. Las flotas más problemáticas incluyeron A350 C con NPS de -71.4, CRJ con rangos de -16.7 a 24.7, y A321 con NPS de 16.7, mientras que los clientes españoles mostraron mayor resistencia manteniendo NPS positivos de hasta +50.0 ante los mismos problemas operativos.

**ECONOMY SH: Deterioro Operativo por Problemas de Handling**
La cabina Economy de SH experimentó un deterioro significativo durante el 25 de agosto de 2025, con el segmento YW registrando un NPS de 26.1 (caída de 7.6 puntos respecto al baseline de 33.7). La causa principal fue el incremento crítico del Mishandling al 24.9% (+7.8 puntos porcentuales vs el promedio de 17.1%), complementado por un Load Factor elevado del 86.1% que intensificó la presión operativa. Este deterioro se reflejó especialmente en la flota CRJ (NPS 24.7) versus ATR (NPS 35.7), mientras que los perfiles más reactivos incluyeron clientes norteamericanos (NPS -66.7) y europeos (NPS -8.7), contrastando con la resistencia de clientes españoles (NPS +50.0).

**BUSINESS SH: Crisis Diferenciada por Compañía**
El segmento Business de SH mostró deterioros severos pero diferenciados por compañía, con IB registrando un NPS de 40.8 (caída de 9.23 puntos vs baseline 50.0) y YW experimentando un colapso más dramático con NPS de -16.7 (deterioro de 33.6 puntos desde 16.9). Esta evolución se explica principalmente por problemas de Mishandling que alcanzaron 28.0% en IB (+54% vs baseline) y 24.9% en YW (+46% vs baseline), siendo especialmente visible en flotas como A321 para IB (NPS 16.7) y CRJ para YW, y entre perfiles de clientes europeos más reactivos que españoles, con viajeros corporativos Business/Work mostrando la mayor sensibilidad (NPS -33.3).

**ECONOMY LH: Cabina Estable**
La cabina Economy de LH mantuvo desempeño estable durante la semana del 25 de agosto de 2025. No se detectaron cambios significativos a nivel semanal, manteniendo niveles consistentes de satisfacción.

**BUSINESS LH: Mejora Paradójica Ante Crisis Operativa**
La cabina Business de LH experimentó una mejora paradójica, registrando un NPS de 50.0 (incremento de 9.0 puntos vs baseline 40.99) a pesar de enfrentar los mismos problemas operativos. Los drivers principales fueron el Mishandling crítico del 27.2% (+9.3 puntos porcentuales) que, contrariamente a la lógica esperada, no impactó negativamente este segmento, posiblemente por efectos de volumen de muestra reducido (19 encuestas) y gestión diferenciada del servicio, impactando especialmente a clientes de América Centro (NPS -25.0) versus América Sur (NPS +100.0) con una dispersión extrema de 125 puntos entre perfiles.

**PREMIUM LH: Máxima Sensibilidad a Problemas Operativos**
El segmento Premium de LH experimentó el mayor deterioro relativo con un NPS de 42.9 (caída de 17.3 puntos vs baseline 60.2). Las causas dominantes fueron los mismos problemas de Mishandling del 27.2% (+9.3 puntos porcentuales) y Load Factor elevado del 91.5%, especialmente evidentes en el ultra-bajo volumen de muestra (6 encuestas) que amplificó el impacto de cada experiencia negativa, y entre perfiles de alta exigencia que mostraron máxima reactividad a los fallos operativos sin generar feedback formal estructurado.

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

# 🏢 DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)

## **A. ECONOMY SHORT HAUL**

### **Nodos Identificados:**
✅ **Economy SH/IB existe**: Global/SH/Economy/IB  
✅ **Economy SH/YW existe**: Global/SH/Economy/YW

### **Análisis Comparativo:**

| **Aspecto** | **IB** | **YW** |
|-------------|--------|--------|
| **Driver Principal** | Mishandling: 28.0% (+9.8pts vs baseline) | Mishandling: 24.9% (+7.8pts vs baseline) |
| **Impacto NPS** | -5.5 puntos (21.2 vs 26.8) | -14.3 puntos (48.0 vs 33.7) |
| **Segmentos Críticos** | LATAM CodeShare (-75.0), A33ACMI (-40.0) | América Norte (0.0), ATR vs CRJ (-31.8pts) |
| **Evidencia Operativa** | Load Factor normal, OTP estable | Load Factor elevado (+2.3pts), saturación sistémica |

### **Diagnóstico Economy SH:**
**Causa común de cabina con manifestaciones operativas diferenciadas** - Ambas compañías comparten el mismo driver principal (mishandling crítico) pero YW experimenta mayor saturación sistémica que amplifica el impacto.

---

## **B. BUSINESS SHORT HAUL**

### **Nodos Identificados:**
❌ **Business SH/IB**: No existe nodo separado  
✅ **Business SH/YW existe**: Global/SH/Business/YW

### **Razón de Ausencia IB:**
El análisis muestra que Business Short Haul está representado únicamente por YW, sugiriendo que IB no opera significativamente en este segmento o no tuvo volumen suficiente de respuestas NPS para generar un nodo separado durante la anomalía del 24 de agosto.

### **Diagnóstico Business SH:**
**No aplica comparación** - Solo existe operación YW en Business Short Haul, por lo que no es posible determinar si la causa sería específica de compañía o común de cabina.

---

## **🎯 CONCLUSIÓN DIAGNÓSTICA:**

- **Economy SH**: Causa común de cabina (mishandling sistémico) con impactos operativos diferenciados por compañía
- **Business SH**: Análisis no aplicable por ausencia de operación IB significativa

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)

## **A. SHORT HAUL - ANÁLISIS DE CABINAS**

### **Patrones Identificados:**

| **Cabina** | **Driver Principal** | **Impacto NPS** | **Reactividad** |
|------------|---------------------|-----------------|-----------------|
| **Economy SH** | Mishandling crítico (24.9-28.0%) | Moderado (-5.5 a -14.3pts) | **Media** |
| **Business SH** | Mishandling crítico (+7.8pts) | Contenido (+10.35pts) | **Baja - Patrón "Silencioso"** |

### **Análisis de Convergencia/Divergencia:**
- **Driver Convergente**: Ambas cabinas comparten mishandling como causa raíz operativa
- **Reactividad Divergente**: Business SH muestra el patrón único de "anomalía silenciosa" donde absorbe el impacto sin manifestarlo externamente
- **Consistencia entre compañías**: En Economy, tanto IB como YW convergen en mishandling como driver principal

### **Diagnóstico Short Haul:**
**Causa común de radio con reactividad específica de cabina** - Mishandling sistémico afecta todo Short Haul, pero Business desarrolla un patrón de absorción silenciosa mientras Economy manifiesta el impacto directamente.

---

## **B. LONG HAUL - ANÁLISIS DE CABINAS**

### **Patrones Identificados:**

| **Cabina** | **Driver Principal** | **Impacto NPS** | **Segmentos Críticos** |
|------------|---------------------|-----------------|------------------------|
| **Economy LH** | Mishandling (+9.3pts) | Moderado (-2.4pts) | MAD-SCL, A33ACMI (-76.9) |
| **Business LH** | Mishandling (+9.3pts) | Severo (-21.0pts) | América Sur (-33.3), A333 |
| **Premium LH** | Mishandling (+9.3pts) | Significativo (-13.8pts) | América Sur (-33.3), A350 std |

### **Progresión de Reactividad:**
1. **Economy LH**: Menor reactividad (-2.4pts) - Expectativas base más tolerantes
2. **Premium LH**: Reactividad media (-13.8pts) - Sensibilidad intermedia  
3. **Business LH**: Mayor reactividad (-21.0pts) - Expectativas más altas, mayor sensibilidad

### **Convergencia Operativa:**
- **Driver 100% convergente**: Mishandling (+9.3pts) es idéntico en las tres cabinas
- **Segmentación geográfica común**: América del Sur como epicentro en Business y Premium
- **Flotas afectadas consistentes**: A33ACMI, A333, A350 std aparecen en múltiples cabinas

### **Diagnóstico Long Haul:**
**Causa común de radio con progresión lógica de reactividad por cabina** - Mishandling sistémico impacta todo Long Haul, pero la reactividad escala proporcionalmente con el nivel de servicio: Economy (menor) < Premium (medio) < Business (mayor).

---

## **🎯 CONCLUSIÓN DIAGNÓSTICA POR RADIO:**

- **Short Haul**: Causa común con reactividad específica (Business desarrolla patrón "silencioso")
- **Long Haul**: Causa común con progresión lógica de reactividad (escalamiento por nivel de servicio)

**Insight Clave**: En ambos radios, mishandling es la causa raíz universal, pero cada cabina desarrolla patrones de reactividad coherentes con su perfil de cliente.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

## **A. COMPARACIÓN ENTRE RADIOS**

### **Análisis de Afectación:**

| **Radio** | **Afectación** | **Driver Principal** | **Evidencia Operativa** | **Impacto NPS** |
|-----------|----------------|---------------------|------------------------|-----------------|
| **Short Haul** | ✅ **AFECTADO** | Mishandling (+7.8 a +9.8pts) | Load Factor elevado, saturación sistémica | -5.5 a -14.3pts |
| **Long Haul** | ✅ **AFECTADO** | Mishandling (+9.3pts) | Load Factor elevado (91.5%), A33ACMI crisis | -2.4 a -21.0pts |

### **Convergencia de Drivers:**
- **Driver Idéntico**: Mishandling crítico es el factor causal primario en ambos radios
- **Valores Consistentes**: Incrementos de mishandling entre +7.8pts y +9.8pts (rango coherente)
- **Factores Secundarios Convergentes**: Load Factor elevado presente en ambos radios como amplificador

### **Evidencia Operativa Consistente:**
- **Mishandling sistémico**: Problema generalizado en toda la operación
- **Alta ocupación**: Saturación operativa común (86.1-91.5% vs baselines menores)
- **Puntualidad preservada**: OTP15 estable o mejorado en ambos radios
- **Conexiones estables**: Misconex sin deterioro significativo

---

## **B. COHERENCIA CON NODO GLOBAL**

### **Análisis del Nodo Global:**

| **Aspecto** | **Global** | **Coherencia SH+LH** |
|-------------|------------|-----------------------|
| **Impacto NPS** | -1.08 puntos | ✅ **Coherente** - Agregación ponderada de impactos SH/LH |
| **Driver Principal** | Mishandling 27.2% (+9.3pts) | ✅ **Perfecta convergencia** - Mismo driver y magnitud |
| **Load Factor** | 88.3% (+0.7pts) | ✅ **Consistente** - Promedio ponderado de ambos radios |
| **Segmentos Críticos** | A33ACMI (-50.0), LATAM (-35.7) | ✅ **Reflejados** - Aparecen en análisis LH y SH |

### **Manifestación Global:**
- **Amplificación Selectiva**: El Global no atenúa sino que refleja fielmente los problemas más severos
- **Concentración Visible**: BRU-MAD (-33.3), A33ACMI, LATAM CodeShare emergen claramente
- **Patrón "Silencioso" Preservado**: 0 incidentes NCS y 0 verbatims mantienen el patrón de ambos radios

---

## **🎯 DIAGNÓSTICO DE ALCANCE:**

### **Evidencia de Causa Global Homogénea:**
1. **Afectación Universal**: Ambos radios impactados simultáneamente
2. **Driver Idéntico**: Mishandling (+9.3pts) como causa raíz única y consistente
3. **Coherencia Agregada**: Global refleja perfectamente la suma ponderada de ambos radios
4. **Patrones Sistémicos**: Load Factor elevado y ausencia de feedback formal en toda la operación

### **Descarte de Causas Específicas o Mixtas:**
- ❌ **No es específica de radio**: Ambos SH y LH afectados por igual
- ❌ **No son causas mixtas**: No hay compensación ni patrones divergentes
- ❌ **No hay efectos de cancelación**: Global no muestra normalidad artificial

---

## **📋 CONCLUSIÓN DIAGNÓSTICA:**

**Causa global homogénea con manifestación sistémica** - El 24 de agosto experimentó un colapso generalizado del sistema de manejo de equipajes que afectó uniformemente toda la operación (Short y Long Haul), con el mismo driver causal (mishandling +9.3pts) y patrones operativos consistentes (alta ocupación, ausencia de escalamiento formal), manifestándose coherentemente en el nodo Global como agregación fiel de los impactos individuales sin efectos de compensación.

**Alcance**: **GLOBAL SISTÉMICO** - Problema operativo de toda la compañía, no localizado por radio ni cabina.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

## **CAUSA ÚNICA IDENTIFICADA: COLAPSO SISTÉMICO DEL MANEJO DE EQUIPAJES**

---

### **A. NATURALEZA DE LA CAUSA**

**Hipótesis Causal:**
El 24 de agosto de 2025 se produjo un **colapso sistémico en los procesos de ground handling de equipajes** causado por la convergencia de alta saturación operativa (Load Factor elevado) con fallas en procedimientos de manejo, creando un efecto cascada que degradó masivamente la experiencia del cliente sin generar escalaciones formales debido al perfil de "sufrimiento silencioso" de los pasajeros afectados.

**Mecanismo de Impacto:**
- **Saturación operativa** (Load Factor 88.3-91.5%) sobrecargó sistemas de handling
- **Fallas procedimentales** en equipajes se dispararon (+54% vs baseline)
- **Efecto amplificador** en segmentos premium con mayores expectativas
- **Absorción silenciosa** del problema por parte de los clientes sin escalación formal

---

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE**

#### **SEGMENTO MÁS GRANDE AFECTADO: Global (Toda la Operación)**

**Alcance Total:** La causa afecta a **todos los subsegmentos** bajo el nodo Global, incluyendo:
- ✅ Global/LH (Long Haul completo)
- ✅ Global/SH (Short Haul completo) 
- ✅ Todas las cabinas: Economy, Business, Premium
- ✅ Todas las compañías: IB, YW
- ✅ Todas las flotas: A33ACMI, A333, A350, ATR, CRJ
- ✅ Todas las regiones: Europa, América Norte, América Sur, América Centro

#### **OUTPUT CAUSAL DETALLADO - NODO GLOBAL:**

**📊 MÉTRICAS OPERATIVAS CRÍTICAS:**
- **Mishandling**: 27.2% vs 17.9% baseline (**+9.3 puntos porcentuales, +52% relativo**)
- **Load Factor**: 88.3% vs 87.6% baseline (+0.7 puntos)
- **OTP15**: 85.8% vs 84.4% baseline (+1.4 puntos) - **Paradoja operativa**
- **Misconex**: 0.6% vs 0.9% baseline (-0.3 puntos) - **Estable**

**📉 IMPACTO EN NPS:**
- **NPS Global**: Caída de -1.08 puntos
- **Dispersión máxima por Fleet**: 102.4 puntos (problemas extremadamente concentrados)
- **Dispersión máxima por CodeShare**: 101.3 puntos

**🗺️ RUTAS MÁS AFECTADAS:**
- **BRU-MAD**: NPS -33.3 (n=3) - Epicentro europeo
- **MAD-SCL**: NPS 12.5 (n=16) - Ruta intercontinental crítica
- **MAD-MIA**: NPS 7.1 (n=14) - Corredor transatlántico

**✈️ FLOTAS MÁS IMPACTADAS:**
- **A33ACMI**: NPS -50.0 (20 surveys) - Crisis en operaciones wet-lease
- **A333**: NPS 0.0 - Problemas en flota de largo radio
- **ATR**: Diferencial negativo vs CRJ en short haul

**🤝 CODESHARES CRÍTICOS:**
- **LATAM**: NPS -35.7 (14 surveys) - Alianza estratégica afectada
- **American Airlines**: NPS -22.2 a -33.3 - Múltiples segmentos impactados

**💬 EVIDENCIA CUALITATIVA:**
- **Incidentes NCS**: 0 reportados - **Patrón "silencioso" sistémico**
- **Verbatims disponibles**: 0 - **Ausencia total de feedback cualitativo**
- **Interpretación**: Los clientes absorbieron los problemas sin escalarlos formalmente

**🌍 SEGMENTACIÓN GEOGRÁFICA:**
- **América del Sur**: Múltiples segmentos con NPS -33.3
- **Europa**: NPS -24.0 a -28.6 en varios subsegmentos
- **América del Norte**: Impactos variables desde 0.0 hasta +33.3

**👥 PERFILES MÁS SENSIBLES:**
- **Viajeros Business**: Mayor reactividad (-21.0 puntos en LH)
- **Clientes Premium**: Impacto significativo (-13.8 puntos)
- **Operaciones ACMI**: Menor control operativo, mayor vulnerabilidad

---

### **🎯 SÍNTESIS CAUSAL CONSOLIDADA**

**El 24 de agosto de 2025 experimentó un colapso sistémico único del manejo de equipajes (+52% vs baseline) que afectó uniformemente toda la operación global de la compañía, manifestándose con mayor severidad en segmentos de alto valor (Business, Premium) y operaciones de menor control directo (ACMI, CodeShares), caracterizado por un patrón de "sufrimiento silencioso" donde los clientes absorbieron los problemas sin generar escalaciones formales, creando una brecha crítica entre el impacto real en la experiencia y la visibilidad corporativa del problema.**

**Causa Raíz Única**: Saturación operativa + Fallas procedimentales en ground handling = Colapso sistémico de manejo de equipajes con absorción silenciosa del impacto por parte de los clientes.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El 24 de agosto de 2025 se registró un deterioro generalizado del NPS causado por un colapso sistémico en el manejo de equipajes que elevó la tasa de mishandling al 27.2% (+9.3 puntos vs baseline), impactando desproporcionalmente a segmentos premium y operaciones con menor control directo. La anomalía se manifestó más severamente en Long Haul Business, que experimentó una caída de 21 puntos (de 41.0 a 20.0), seguido por Premium LH con un deterioro de 13.8 puntos (de 25.6 a 11.8) y Economy LH con una reducción de 2.4 puntos (de 19.8 a 17.4). En Short Haul, Economy registró caídas variables entre 5.5 y 14.3 puntos según la compañía (IB: 21.2 vs 26.8 baseline; YW: 48.0 vs 33.7 baseline), mientras que Business SH mostró un patrón único de "anomalía silenciosa" con un NPS aparentemente positivo de 27.3 puntos que encubría problemas operativos reales no reportados formalmente.

Las rutas más afectadas incluyeron BRU-MAD con un NPS devastador de -33.3, MAD-SCL con 12.5 puntos y MAD-MIA con 7.1 puntos, concentrándose especialmente en corredores intercontinentales y conexiones europeas. Los grupos más reactivos fueron los viajeros de América del Sur (NPS -33.3 en múltiples segmentos), clientes de operaciones ACMI con la flota A33 registrando -50.0 a -76.9 puntos, y pasajeros de códigos compartidos con LATAM (-35.7 a -75.0 puntos) y American Airlines (-22.2 a -33.3 puntos), evidenciando mayor vulnerabilidad en operaciones de menor control directo y rutas de largo radio.

**ECONOMY SH: Deterioro Operativo Diferenciado**
La cabina Economy de Short Haul experimentó un deterioro significativo durante el 24 de agosto, con IB registrando un NPS de 21.2 puntos (caída de 5.5 puntos vs 26.8 baseline) y YW mostrando un impacto más severo con 48.0 puntos (deterioro de 14.3 puntos vs 33.7 baseline). La causa principal fue el incremento crítico en mishandling (28.0% en IB y 24.9% en YW vs baselines de 17-18%), complementado por saturación operativa con Load Factor elevado especialmente en YW. Este deterioro se reflejó especialmente en segmentos como LATAM CodeShare (NPS -75.0), operaciones A33ACMI (NPS -40.0) y rutas como ALC-MAD, mientras que los perfiles más reactivos incluyeron viajeros de América del Norte (NPS 0.0 en YW) y pasajeros de flota ATR vs CRJ.

**BUSINESS SH: Patrón de Absorción Silenciosa**
El segmento Business de Short Haul mostró un comportamiento único con un NPS aparente de 27.3 puntos (mejora aparente de 10.4 puntos vs 16.9 baseline), pero encubriendo una "anomalía silenciosa" donde los problemas operativos reales no se manifestaron en feedback formal. Esta evolución paradójica se explica principalmente por el incremento de mishandling (+7.8 puntos al 24.9%) que los clientes Business absorbieron sin reportar formalmente, siendo especialmente visible en la ausencia total de incidentes NCS y verbatims, caracterizando un perfil de cliente que internaliza los problemas sin escalarlos pero que sí impacta su satisfacción real.

**ECONOMY LH: Impacto Moderado en Rutas Específicas**
La cabina Economy de Long Haul experimentó un deterioro moderado durante el 24 de agosto, registrando un NPS de 17.4 puntos con una caída de 2.4 puntos respecto al baseline de 19.8. La causa principal fue el incremento crítico en mishandling (+9.3 puntos al 27.2%) combinado con alta ocupación (Load Factor 92.1%), siendo especialmente evidente en operaciones con menor control directo. Este deterioro se reflejó especialmente en rutas como MAD-SCL (NPS 12.5) y operaciones con flota A330 ACMI (NPS -76.9), mientras que los perfiles más reactivos incluyeron viajeros de negocio (NPS -13.3) y clientes con origen en Europa, contrastando con la mayor resistencia del segmento leisure (NPS 19.9).

**BUSINESS LH: Máxima Reactividad a Problemas Operativos**
La cabina Business de Long Haul experimentó el mayor deterioro del día, registrando un NPS de 20.0 puntos con una caída severa de 21 puntos respecto al baseline de 41.0. Los drivers principales fueron el incremento masivo en mishandling (+9.3 puntos al 27.2%) que afectó desproporcionalmente las expectativas elevadas de este segmento, impactando especialmente las rutas intercontinentales largas como los corredores América del Sur-Europa y perfiles de clientes residentes en América del Sur (NPS -33.3) y operaciones con códigos compartidos de American Airlines (NPS -33.3), mientras que los clientes españoles mostraron mayor resistencia (NPS 58.3).

**PREMIUM LH: Sensibilidad Intermedia con Concentración Geográfica**
El segmento Premium de Long Haul registró un deterioro significativo durante el 24 de agosto, alcanzando un NPS de 11.8 puntos con una caída de 13.8 puntos vs el baseline de 25.6. Las causas dominantes fueron el incremento crítico en mishandling (+9.3 puntos) combinado con alta ocupación operativa, especialmente evidentes en rutas hacia América del Sur donde los clientes registraron un NPS -33.3, y entre operaciones con flota A350 estándar (NPS -22.2), contrastando con el mejor desempeño de clientes españoles (NPS +50.0) y operaciones A333 (NPS +60.0), evidenciando mayor vulnerabilidad en rutas intercontinentales de largo radio.

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

## **A. ECONOMY SHORT HAUL**

**NODOS IDENTIFICADOS:**
- ✅ **Economy SH/IB**: Global/SH/Economy/IB (anomalía -7.67 pts)
- ✅ **Economy SH/YW**: Global/SH/Economy/YW (anomalía -0.21 pts)

**ANÁLISIS COMPARATIVO:**

| Aspecto | Economy SH/IB | Economy SH/YW |
|---------|---------------|---------------|
| **Driver Principal** | Mishandling +9.8pts (+54%) | Experiencias devastadoras en CodeShare |
| **Evidencia Operativa** | Correlación directa con Mishandling 28.0% | Mishandling 24.9% presente pero NO es causa principal |
| **Patrón de Impacto** | Problemas operativos generalizados | 92% clientes IB sanos (NPS 38.9), 8% CodeShare críticos |

**DIAGNÓSTICO ECONOMY SH:** **Causa específica de compañía** - IB sufre impacto operativo directo por Mishandling, mientras YW sufre por problemas de partners CodeShare no-IB.

---

## **B. BUSINESS SHORT HAUL**

**NODOS IDENTIFICADOS:**
- ✅ **Business SH/IB**: Global/SH/Business/IB (anomalía -5.48 pts)
- ✅ **Business SH/YW**: Global/SH/Business/YW (anomalía +14.65 pts)

**ANÁLISIS COMPARATIVO:**

| Aspecto | Business SH/IB | Business SH/YW |
|---------|----------------|----------------|
| **Driver Principal** | Volatilidad estadística (n=31) | Mishandling +7.8pts (+45%) |
| **Evidencia Operativa** | Sin correlación operativa confirmada | Correlación directa con Mishandling 24.9% |
| **Patrón de Impacto** | Artefacto de muestra pequeña | Concentrado en MAD-OPO, viajeros Business/Work |

**DIAGNÓSTICO BUSINESS SH:** **Causa específica de compañía** - IB experimenta volatilidad estadística sin causa operativa, mientras YW sufre impacto operativo real concentrado en rutas específicas.

---

## **CONCLUSIÓN PASO 1:**

**Ambas cabinas Short Haul muestran causas específicas de compañía**, con IB y YW experimentando drivers y patrones de impacto completamente diferentes dentro de cada segmento de cabina.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ PASO 2: DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)

## **A. SHORT HAUL - ANÁLISIS DE CABINAS**

**NODOS DISPONIBLES:**
- Economy SH (IB/YW separados)
- Business SH (IB/YW separados)

**ANÁLISIS COMPARATIVO:**

| Cabina | Driver Principal | Patrón de Impacto | Reactividad Operativa |
|--------|------------------|-------------------|----------------------|
| **Economy SH** | Mishandling + CodeShare issues | Mixto: operativo (IB) + partners (YW) | **Media-Alta** |
| **Business SH** | Mishandling + volatilidad estadística | Concentrado geográfico + artefactos muestra | **Variable** |

**CONSISTENCIA ENTRE COMPAÑÍAS:**
- **IB**: Economy (-7.67) más afectada que Business (-5.48) - patrón lógico de menor sensibilidad en clase superior
- **YW**: Business (+14.65) menos afectada que Economy (-0.21) - **patrón invertido anómalo**

**DIAGNÓSTICO SHORT HAUL:** **Causa específica de cabina con reactividad diferencial anómala** - Business YW muestra respuesta inversa al patrón esperado, sugiriendo factores específicos de segmento que anulan la lógica tradicional de sensibilidad por clase.

---

## **B. LONG HAUL - ANÁLISIS DE CABINAS**

**NODOS DISPONIBLES:**
- Business LH (anomalía +16.14 pts)
- Premium LH (anomalía -5.56 pts)
- **Economy LH**: No presente en el árbol

**ANÁLISIS COMPARATIVO:**

| Cabina | Driver Principal | Patrón de Impacto | Reactividad Operativa |
|--------|------------------|-------------------|----------------------|
| **Business LH** | Mishandling +9.3pts (flota A350 C) | Polarizado: 67% sin impacto, 33% severo | **Diferencial extrema** |
| **Premium LH** | Mishandling +9.3pts (flota A333) | Concentrado: falla específica A333 | **Alta sensibilidad** |

**PROGRESIÓN DE REACTIVIDAD:**
- **Premium LH**: Mayor sensibilidad (-5.56 pts) ante mismo nivel Mishandling
- **Business LH**: Efecto compensación (+16.14 pts) por segmentación del impacto

**DIAGNÓSTICO LONG HAUL:** **Causa común de radio con reactividad diferencial por clase** - Ambas cabinas sufren el mismo problema operativo (Mishandling +9.3pts) pero Premium muestra mayor sensibilidad mientras Business experimenta compensación por distribución polarizada del impacto.

---

## **CONCLUSIÓN PASO 2:**

**Short Haul**: Causas específicas de cabina con patrones anómalos de reactividad.
**Long Haul**: Causa común de radio (Mishandling) con reactividad diferencial esperada por clase de servicio.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 PASO 3: DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

## **A. COMPARACIÓN ENTRE RADIOS**

**AFECTACIÓN POR RADIO:**
- ✅ **Short Haul afectado**: 4 nodos con anomalías variadas
- ✅ **Long Haul afectado**: 2 nodos con anomalías significativas

**CONVERGENCIA DE DRIVERS:**

| Radio | Driver Principal | Evidencia Operativa | Magnitud Mishandling |
|-------|------------------|-------------------|---------------------|
| **Short Haul** | Mishandling + factores específicos | Múltiple: 24.9%-28.0% | **+7.8 a +9.8 pts** |
| **Long Haul** | Mishandling concentrado por flota | Único: 27.2% | **+9.3 pts** |

**CONSISTENCIA OPERATIVA:**
- **Convergencia confirmada**: Ambos radios sufren el mismo problema operativo base (Mishandling elevado)
- **Manifestación diferente**: SH con problemas adicionales (CodeShare, volatilidad), LH con concentración por flota específica
- **Valores SHAP coherentes**: Incrementos similares de Mishandling (~+9 pts) en ambos radios

---

## **B. COHERENCIA CON NODO GLOBAL**

**ANÁLISIS DE AGREGACIÓN:**

| Nivel | NPS | Anomalía | Driver Principal |
|-------|-----|----------|-----------------|
| **Global** | 32.0 | **+4.80 pts** | Mishandling +9.3pts + efecto compensación |
| **SH agregado** | Variable | Negativo neto | Mishandling + factores específicos |
| **LH agregado** | Variable | Mixto | Mishandling + polarización por flota |

**PATRÓN DE COMPENSACIÓN IDENTIFICADO:**
- **Efecto enmascaramiento**: El Global muestra anomalía **positiva** (+4.80) mientras ambos radios tienen impactos **negativos/mixtos**
- **Mecanismo**: Segmentos no afectados (67% según análisis Global) compensaron el impacto de segmentos críticos (33%)
- **Coherencia operativa**: El Global confirma Mishandling +9.3pts como factor común a ambos radios

---

## **DIAGNÓSTICO ESPECÍFICO:**

**ALCANCE**: **Causa global homogénea con manifestación compensatoria**

**EXPLICACIÓN**: Ambos radios sufren el mismo problema operativo base (Mishandling elevado ~+9pts), pero el impacto se distribuye de manera polarizada - segmentos específicos experimentan degradación severa mientras otros mantienen rendimiento superior, resultando en un efecto neto positivo a nivel Global que enmascara problemas operativos reales y localizados.

---

## **CONCLUSIÓN PASO 3:**

**Causa global homogénea con efectos compensatorios** - Un problema operativo único (Mishandling crítico) afecta ambos radios pero se manifiesta de forma segmentada, creando una anomalía Global positiva que enmascara impactos negativos reales en subsegmentos específicos.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 PASO 4: ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

## **CAUSA PRINCIPAL IDENTIFICADA**

### **🚨 MISHANDLING CRÍTICO GENERALIZADO**

---

## **A. NATURALEZA DE LA CAUSA**

**HIPÓTESIS CAUSAL:** Falla sistémica en procesos de manejo de equipaje y carga el 23 de agosto de 2025, con incremento del 52% respecto al baseline operativo. Esta falla se manifestó de manera concentrada en flotas específicas (A350 C, A333, A33ACMI) y operaciones CodeShare, sugiriendo problemas en coordinación operativa compleja y/o saturación de capacidad de handling en aeropuertos hub.

---

## **B. EVIDENCIA CONSOLIDADA Y ALCANCE**

### **SEGMENTO MÁS GRANDE AFECTADO: GLOBAL**
*Todos los subsegmentos bajo este nodo están afectados por esta causa*

**OUTPUT CAUSAL DETALLADO:**
- **Mishandling Global**: 27.2% vs 17.9% baseline (**+9.3 puntos porcentuales, +52% incremento**)
- **Impacto NPS**: Aparente anomalía positiva (+4.80 pts) que enmascara problemas severos localizados
- **Patrón de distribución**: 67% de operación no afectada compensó el 33% con degradación severa

### **MÉTRICAS CLAVE CONSOLIDADAS:**

**INDICADORES OPERATIVOS:**
- **Mishandling Range**: 24.9% - 28.0% según segmento
- **OTP15**: Mejorado 85.8% vs 84.4% promedio (factor descartado)
- **Load Factor**: 88.2% - 91.5% según segmento (elevado pero no causal)
- **Misconex**: 0.6% vs 0.9% promedio (mejorado, factor descartado)

**FLOTAS CRÍTICAS IDENTIFICADAS:**
- **A350 C**: NPS -100.0 (Long Haul Business) / NPS -6.7 (Global)
- **A333**: NPS -66.7 (Long Haul Premium)
- **A33ACMI**: NPS -16.7 (Short Haul Economy)
- **CRJ**: NPS 29.4 (Short Haul Business YW)

**RUTAS INVOLUCRADAS:**
- **MAD-ORD**: NPS -14.3 (epicentro de problemas operativos)
- **MAD-OPO**: NPS 33.3 (concentración de problemas YW Business)
- **ATH-MAD**: NPS 33.3 (impacto Economy IB)
- **FRA-MAD**: NPS 0.0 (impacto Economy YW)

**SEGMENTOS MÁS AFECTADOS:**
- **CodeShare AA**: NPS -20.0 (30 encuestas)
- **Others CodeShare**: NPS -33.3 a -100.0
- **América Norte**: Consistentemente más afectada que otros orígenes
- **Viajeros Business/Work**: Mayor sensibilidad que Leisure

**VERBATIMS REPRESENTATIVOS:**
❌ **LIMITACIÓN CRÍTICA**: Cero verbatims disponibles para el 23 de agosto de 2025
❌ **INCIDENTES NCS**: Cero reportes formales disponibles

---

## **ALCANCE TOTAL DE IMPACTO:**

**SUBSEGMENTOS CONFIRMADOS AFECTADOS:**
- ✅ Global/LH/Business (Mishandling +9.3pts, flota A350 C crítica)
- ✅ Global/LH/Premium (Mishandling +9.3pts, flota A333 crítica)
- ✅ Global/SH/Business/IB (Mishandling +9.8pts, correlación directa)
- ✅ Global/SH/Business/YW (Mishandling +7.8pts, concentrado MAD-OPO)
- ✅ Global/SH/Economy/IB (Mishandling +9.8pts, CodeShare AA crítico)
- ✅ Global/SH/Economy/YW (Mishandling +7.8pts, Others CodeShare devastador)

---

## **CONCLUSIÓN CAUSAL:**

**El 23 de agosto de 2025 experimentó una falla sistémica de Mishandling (+52% vs baseline) que afectó de manera diferencial pero consistente a todos los segmentos operativos, concentrándose en flotas específicas y operaciones CodeShare complejas, creando un patrón de "anomalía enmascarada" donde problemas operativos severos fueron compensados por el rendimiento superior de segmentos no afectados.**

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El 23 de agosto de 2025 mostró un patrón complejo de evolución del NPS donde una mejora global aparente de +4.80 puntos (alcanzando un NPS de 32.0 versus el baseline de 27.2) enmascaró problemas operativos severos y localizados. Esta paradoja se explica por un incremento crítico del 52% en Mishandling (27.2% vs 17.9% promedio) que afectó de manera diferencial a los segmentos: Long Haul Business experimentó una mejora aparente de +16.14 puntos llegando a 57.14, mientras que Long Haul Premium sufrió un deterioro de -5.56 puntos cayendo a 20.0. En Short Haul, Economy IB mejoró +7.67 puntos alcanzando un NPS no especificado, Economy YW se mantuvo prácticamente estable con una leve caída de -0.21 puntos a 33.53, Business IB cayó -5.48 puntos a 35.29, y Business YW mostró una mejora significativa de +14.65 puntos llegando a 31.58. La causa raíz fue una falla sistémica en procesos de handling que se concentró específicamente en flotas A350 C (NPS -100.0 en Business LH), A333 (NPS -66.7 en Premium LH) y operaciones CodeShare, particularmente con American Airlines (NPS -20.0).

Las rutas más afectadas fueron MAD-ORD que registró un NPS crítico de -14.3 convirtiéndose en el epicentro de los problemas operativos, seguida de MAD-OPO con NPS de 33.3 y ATH-MAD también con 33.3, mientras que FRA-MAD mostró un NPS de 0.0. Los grupos de clientes más reactivos fueron los pasajeros de CodeShare especialmente con American Airlines y "Others" (NPS entre -100.0 y -20.0), viajeros procedentes de América Norte que consistentemente mostraron mayor sensibilidad a las disrupciones operativas, clientes del segmento Business/Work que fueron más críticos que los de Leisure, y pasajeros españoles que demostraron menor resiliencia ante problemas operativos comparado con clientes de América Centro y Sur.

**ECONOMY SH: Impacto Mixto por Compañía**
La cabina Economy de Short Haul mostró un comportamiento divergente entre compañías durante el 23 de agosto, donde IB experimentó una mejora de +7.67 puntos debido principalmente al incremento del 54% en Mishandling (28.0% vs 18.2% promedio) que paradójicamente correlacionó con mejor NPS, especialmente visible en rutas internacionales como ATH-MAD, mientras que YW se mantuvo prácticamente estable con una leve caída de -0.21 puntos a 33.53. Esta estabilidad en YW se explica porque el 92% de sus clientes operados directamente por Iberia mantuvieron un NPS saludable de 38.9, mientras que solo el 8% de clientes en operaciones CodeShare (especialmente "Others" con NPS -100.0 y American Airlines con NPS -16.7) arrastraron el promedio, siendo los perfiles más reactivos los pasajeros sin región de residencia definida (NPS -27.3) y procedentes de América Norte.

**BUSINESS SH: Patrones Contrastantes entre Compañías**
El segmento Business de Short Haul registró comportamientos opuestos entre compañías, donde IB experimentó un deterioro de -5.48 puntos cayendo a 35.29 explicado principalmente por volatilidad estadística debido al tamaño reducido de muestra (solo 31 encuestas), mientras que YW mostró una mejora significativa de +14.65 puntos alcanzando 31.58. Esta evolución en YW se explica principalmente por el incremento del 45% en Mishandling (24.9% vs 17.1% promedio) concentrado en la ruta MAD-OPO, siendo especialmente visible entre viajeros Business/Work (NPS 16.7) comparado con Leisure (NPS 38.5), y entre clientes no españoles que mostraron mayor sensibilidad a las disrupciones operativas.

**ECONOMY LH: Estabilidad Semanal**
La cabina Economy de Long Haul mantuvo desempeño estable a nivel semanal durante este período. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción.

**BUSINESS LH: Mejora Aparente con Problemas Subyacentes**
La cabina Business de Long Haul experimentó una mejora aparente de +16.14 puntos alcanzando un NPS de 57.14 durante el 23 de agosto, aunque esta evolución enmascara problemas operativos severos. Los drivers principales fueron el incremento del 52% en Mishandling (27.2% vs 17.9% promedio) que se concentró específicamente en la flota A350 C generando un NPS catastrófico de -100.0, mientras que otras flotas como A332 y A350 next mantuvieron NPS de 100.0, impactando especialmente a clientes españoles (NPS 0.0) versus los de América Centro/Sur (NPS 100.0) y siendo los viajeros business menos reactivos que los de leisure ante estas disrupciones.

**PREMIUM LH: Deterioro por Problemas de Flota Específica**
El segmento Premium de Long Haul sufrió un deterioro de -5.56 puntos cayendo a un NPS de 20.0 durante el 23 de agosto debido al mismo incremento del 52% en Mishandling que afectó otras cabinas. Las causas dominantes fueron la falla operativa específica de la flota A333 que registró un NPS crítico de -66.7 con 100% de detractores, contrastando con el A350 next que logró +60.0 NPS el mismo día, especialmente evidentes en rutas hacia América Norte (NPS -33.3) versus América Centro (NPS +75.0) y entre pasajeros Premium de larga distancia que mostraron mayor sensibilidad a problemas de equipaje y saturación operativa.

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

# 🏢 DIAGNÓSTICO A NIVEL COMPAÑÍA (IB/YW)

## **A. ECONOMY SHORT HAUL**

### **Nodos Identificados:**
✅ **Economy SH/IB**: Existe (Global/SH/Economy/IB)  
❌ **Economy SH/YW**: No existe en el árbol

### **Análisis del Nodo Disponible (SH/Economy/IB):**
- **Driver Principal**: Mishandling crítico (+9.8 puntos vs baseline, 28.0% vs 18.2%)
- **Concentración**: MAD-VGO (NPS 0.0) y flota A319
- **Impacto por CodeShare**: LATAM severamente afectado (NPS -100.0) vs VY positivo (NPS 42.9)
- **Regiones**: ASIA (-66.7) y AMERICA NORTE (-33.3) más impactadas

### **Razón de Ausencia YW**: 
Posible falta de volumen suficiente de respuestas o concentración del problema específicamente en operaciones Iberia.

---

## **B. BUSINESS SHORT HAUL**

### **Nodos Identificados:**
✅ **Business SH/IB**: Existe (Global/SH/Business/IB)  
✅ **Business SH/YW**: Existe (Global/SH/Business/YW)

### **Comparación de Patrones:**

| **Aspecto** | **SH/Business/IB** | **SH/Business/YW** |
|-------------|-------------------|-------------------|
| **Driver Principal** | Mishandling (+9.8pts) | Mishandling (+7.8pts) |
| **Magnitud Mishandling** | 28.0% vs 18.2% baseline | 24.9% vs 17.1% baseline |
| **Caída NPS** | -5.77 puntos | -9.74 puntos |
| **Concentración Geográfica** | LHR-MAD exclusivamente | Disperso geográficamente |
| **Factores Específicos** | Codeshare AA problemático | Impacto "silencioso" generalizado |
| **Dispersión Regional** | 143.6 puntos (récord) | No especificada |

### **Diferencias Clave Identificadas:**
- **IB**: Problema localizado en LHR-MAD con impacto diferencial extremo por región
- **YW**: Problema disperso sin concentración geográfica específica
- **IB**: Mayor dispersión entre perfiles (143.6 puntos)
- **YW**: Mayor caída absoluta de NPS (-9.74 vs -5.77)

---

## **🎯 DIAGNÓSTICO FINAL**

### **Economy Short Haul:**
**CAUSA ESPECÍFICA DE IBERIA** - Solo existe evidencia en operaciones IB con problemas concentrados en rutas específicas y codeshares particulares.

### **Business Short Haul:**
**CAUSA COMÚN DE CABINA CON MANIFESTACIONES ESPECÍFICAS** - Ambas compañías (IB/YW) comparten el mismo driver principal (Mishandling), pero IB muestra concentración geográfica mientras YW presenta dispersión sistémica.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ DIAGNÓSTICO A NIVEL DE CABINA (Economy/Business/Premium)

## **A. SHORT HAUL - ANÁLISIS DE CABINAS**

### **Patrones Identificados por Cabina:**

| **Cabina** | **Driver Principal** | **Magnitud Mishandling** | **Caída NPS** | **Patrón de Impacto** |
|------------|---------------------|-------------------------|---------------|---------------------|
| **Economy SH** | Mishandling crítico | +9.3pts (27.2% vs 17.9%) | -2.56 puntos | Concentrado geográficamente |
| **Business SH** | Mishandling crítico | +9.3pts (27.2% vs 17.9%) | -2.31 puntos | Disperso con segmentación específica |

### **Análisis de Convergencia:**
✅ **CONVERGENCIA COMPLETA EN DRIVER**: Ambas cabinas comparten exactamente el mismo factor causal (Mishandling crítico)  
✅ **CONVERGENCIA EN MAGNITUD OPERATIVA**: Mismo incremento de +9.3pts en Mishandling  
⚠️ **DIVERGENCIA EN REACTIVIDAD**: Economy más reactiva (-2.56 vs -2.31 puntos de caída NPS)  
⚠️ **DIVERGENCIA EN MANIFESTACIÓN**: Economy concentrada geográficamente vs Business dispersa

### **Consistencia entre Compañías:**
- **IB Economy**: Concentración en MAD-VGO y codeshares específicos
- **IB Business**: Concentración en LHR-MAD con dispersión regional extrema (143.6pts)
- **YW Business**: Dispersión sistémica sin concentración geográfica

---

## **B. LONG HAUL - ANÁLISIS DE CABINAS**

### **Patrones Identificados por Cabina:**

| **Cabina** | **Driver Principal** | **Magnitud Mishandling** | **Caída NPS** | **Patrón de Impacto** |
|------------|---------------------|-------------------------|---------------|---------------------|
| **Economy LH** | Mishandling crítico | +9.3pts (27.2% vs 17.9%) | -7.6 puntos | Concentrado en MAD-SJU/Iberia |
| **Business LH** | **NO HUBO ANOMALÍA** | N/A | **NPS 50.0 (Positivo)** | Performance normal |
| **Premium LH** | Mishandling crítico | +9.3pts (27.2% vs 17.9%) | -5.56 puntos | Concentrado en EZE-MAD |

### **Análisis de Progresión por Clase:**
📈 **REACTIVIDAD DECRECIENTE POR CLASE**:
- **Economy LH**: Máxima reactividad (-7.6 puntos)
- **Premium LH**: Reactividad moderada (-5.56 puntos)  
- **Business LH**: **INMUNIDAD COMPLETA** (NPS +50.0)

### **Patrón de Concentración Geográfica:**
- **Economy LH**: MAD-SJU (ruta transatlántica Iberia)
- **Premium LH**: EZE-MAD (ruta transatlántica, codeshare IB)
- **Business LH**: EZE-MAD (sin impacto negativo)

---

## **🎯 DIAGNÓSTICO FINAL**

### **Short Haul:**
**CAUSA COMÚN DE RADIO CON REACTIVIDAD DIFERENCIAL** - Todas las cabinas SH comparten el mismo driver operativo (Mishandling crítico +9.3pts), pero Economy muestra mayor sensibilidad y concentración geográfica mientras Business presenta dispersión con segmentación específica.

### **Long Haul:**
**CAUSA ESPECÍFICA DE CABINA CON BUSINESS COMO AMORTIGUADOR** - Economy y Premium LH convergen en el mismo driver (Mishandling crítico), pero Business LH actúa como "amortiguador" completo, manteniendo performance positivo (+50.0 NPS) pese a los mismos problemas operativos, evidenciando inmunidad del segmento premium a disrupciones operativas menores.

**Hallazgo Clave**: Business LH demuestra que la calidad del servicio diferenciado puede neutralizar completamente el impacto de problemas operativos que afectan significativamente a otras cabinas.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

## **A. COMPARACIÓN ENTRE RADIOS**

### **Evaluación de Afectación:**
✅ **AMBOS RADIOS AFECTADOS** - Short Haul y Long Haul presentan anomalías significativas

### **Convergencia de Drivers:**

| **Radio** | **Driver Principal** | **Magnitud Mishandling** | **Caída NPS Promedio** | **Concentración Geográfica** |
|-----------|---------------------|-------------------------|------------------------|------------------------------|
| **Short Haul** | Mishandling crítico | +9.3pts (27.2% vs 17.9%) | -2.44 puntos | Múltiples rutas específicas |
| **Long Haul** | Mishandling crítico | +9.3pts (27.2% vs 17.9%) | -6.58 puntos* | Rutas transatlánticas |

*Promedio entre Economy LH (-7.6) y Premium LH (-5.56), excluyendo Business LH que no tuvo anomalía.

### **Consistencia Operativa:**
✅ **CONVERGENCIA TOTAL EN DRIVER**: Ambos radios comparten exactamente el mismo factor causal  
✅ **CONVERGENCIA EN MAGNITUD OPERATIVA**: Idéntico incremento de Mishandling (+9.3pts)  
⚠️ **DIVERGENCIA EN SENSIBILIDAD**: Long Haul muestra mayor reactividad (-6.58 vs -2.44 puntos)  
✅ **CONVERGENCIA EN MÉTRICAS SECUNDARIAS**: Ambos radios con Load Factor elevado y OTP mejorado

---

## **B. COHERENCIA CON NODO GLOBAL**

### **Validación de Agregación:**

| **Nivel** | **NPS Reportado** | **Caída vs Baseline** | **Driver Principal** |
|-----------|------------------|----------------------|---------------------|
| **Global** | 23.82 | -3.41 puntos | Mishandling +9.3pts |
| **Short Haul** | 27.29 | -2.52 puntos | Mishandling +9.3pts |
| **Long Haul** | 16.73 | -5.56 puntos | Mishandling +9.3pts |

### **Análisis de Coherencia:**
✅ **AGREGACIÓN MATEMÁTICA COHERENTE**: El Global (-3.41) representa un promedio ponderado lógico entre SH (-2.52) y LH (-5.56)  
✅ **DRIVER CONSISTENTE**: Los tres niveles convergen en el mismo factor causal (Mishandling crítico)  
✅ **AMPLIFICACIÓN PROPORCIONAL**: El Global no atenúa ni amplifica artificialmente, sino que refleja el impacto ponderado real

### **Concentración Geográfica Global:**
- **Epicentro identificado**: MAD-MEX con NPS -14.8 (extremadamente crítico)
- **Rutas críticas consolidadas**: MAD-VGO, MAD-SJU, LHR-MAD, EZE-MAD, ATH-MAD
- **Patrón común**: Todas las rutas problemáticas involucran Madrid como hub principal

---

## **C. ANÁLISIS DE EFECTOS COMPENSATORIOS**

### **Business Long Haul como Factor Estabilizador:**
- **Business LH**: NPS +50.0 (inmune a problemas operativos)
- **Efecto**: Atenúa parcialmente el impacto agregado de Long Haul
- **Sin este efecto**: Long Haul habría mostrado caída aún más severa

### **Dispersión vs Concentración:**
- **Short Haul**: Problemas dispersos en múltiples rutas menores
- **Long Haul**: Problemas concentrados en rutas transatlánticas de alto valor
- **Resultado Global**: Combinación que explica la magnitud intermedia (-3.41 puntos)

---

## **🎯 DIAGNÓSTICO FINAL**

**CAUSA GLOBAL HOMOGÉNEA CON REACTIVIDAD DIFERENCIAL POR RADIO** - Ambos radios (SH y LH) convergen completamente en el mismo driver operativo (Mishandling crítico +9.3pts) y métricas secundarias, pero Long Haul muestra mayor sensibilidad debido a las expectativas más altas en rutas transatlánticas y clientes internacionales. El nodo Global refleja coherentemente la agregación ponderada de ambos radios, con Madrid como epicentro operativo común y Business LH actuando como único factor estabilizador significativo.

**Hallazgo Clave**: Esta es una crisis operativa sistémica centrada en el hub de Madrid que afectó homogéneamente toda la red, pero con impacto diferencial según las expectativas del segmento de clientes y la distancia de las rutas.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

# 🔍 ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

## **CAUSA PRINCIPAL IDENTIFICADA: CRISIS OPERATIVA DE MISHANDLING**

---

### **A. NATURALEZA DE LA CAUSA**

**Hipótesis Causal**: El 22 de agosto de 2025 se produjo una **crisis sistémica en los procesos de ground handling** centrada en el hub de Madrid, que generó un incremento del 52% en el mishandling de equipajes (de 17.9% a 27.2%). Esta crisis operativa no fue un incidente puntual sino un **fallo sistémico en la cadena de servicios terrestres** que afectó múltiples rutas, flotas y operaciones de codeshare, manifestándose de forma diferencial según las expectativas del segmento de clientes y la complejidad operativa de cada ruta.

---

### **B. EVIDENCIA CONSOLIDADA Y ALCANCE**

#### **SEGMENTO MÁS GRANDE AFECTADO: GLOBAL**
*Seleccionado por representar el impacto consolidado más amplio de la crisis*

### **📊 OUTPUT CAUSAL DETALLADO - NODO GLOBAL**

**ANOMALÍA**: Caída de NPS de -3.41 puntos (23.82 vs 27.23 baseline)

**FACTOR OPERATIVO CRÍTICO**:
- **Mishandling de Equipajes**: 27.2% vs 17.9% promedio (+9.3 puntos)
- **Magnitud del problema**: Incremento del 52% respecto al baseline
- **Correlación directa**: Factor que explica completamente la anomalía del NPS

**FACTORES OPERATIVOS SECUNDARIOS**:
- **Load Factor elevado**: 88.3% vs 87.6% (+0.6pts) - Contribuyó a la presión operativa
- **OTP mejorado**: 85.8% vs 84.4% (+1.4pts) - No fue factor negativo
- **Misconex bajo**: 0.6% vs 0.9% (-0.3pts) - Rendimiento normal

---

### **🌍 ALCANCE DE AFECTACIÓN**

**TODOS LOS SUBSEGMENTOS BAJO EL NODO GLOBAL ESTÁN AFECTADOS**:
- ✅ Short Haul (Economy/Business) - Ambas cabinas IB/YW
- ✅ Long Haul (Economy/Premium) - Business LH inmune pero operativamente expuesto
- ✅ Todas las compañías (IB/YW)
- ✅ Todas las regiones geográficas con impacto diferencial

---

### **📈 MÉTRICAS CLAVE CONSOLIDADAS**

#### **Indicadores Operativos Críticos**:
- **Mishandling**: 27.2% (+9.3pts vs baseline, +52% incremento)
- **Load Factor**: 88.3% (+0.6pts, contribuyó a presión operativa)
- **OTP15**: 85.8% (+1.4pts, paradójicamente mejorado)
- **Misconex**: 0.6% (-0.3pts, estable)

#### **Impacto en NPS por Segmento**:
- **Global**: -3.41 puntos (23.82 vs 27.23)
- **Short Haul**: -2.52 puntos (27.29 vs 29.81)
- **Long Haul**: -5.56 puntos (16.73 vs 22.29)
- **Economy LH**: -7.6 puntos (más afectado)
- **Business LH**: +50.0 (inmune, anomalía positiva)

#### **Rutas Críticas Involucradas**:
- **MAD-MEX**: NPS -14.8 (epicentro de la crisis)
- **MAD-SJU**: NPS 14.3 (Economy LH/IB)
- **LHR-MAD**: NPS 60.0 (Business SH/IB)
- **EZE-MAD**: NPS 66.7 (Premium LH)
- **ATH-MAD**: NPS 0.0 (Economy SH)
- **MAD-VGO**: NPS 0.0 (Economy SH/IB)

#### **Perfiles Más Afectados**:
- **ORIENTE MEDIO**: NPS -50.0 (impacto más severo)
- **ASIA**: NPS -40.0 (impacto crítico)
- **AMÉRICA NORTE**: NPS -9.8 (mayor volumen afectado)
- **CodeShare LATAM**: NPS -83.3 a -100.0 (crítico)
- **Flota A319**: NPS 0.0 (completamente afectada)

---

### **💬 FEEDBACK CUALITATIVO**

#### **Verbatims Representativos**:
**AUSENCIA SISTEMÁTICA** - 0 verbatims registrados en todos los nodos analizados

#### **Incidentes NCS**:
**AUSENCIA TOTAL** - 0 incidentes reportados formalmente

#### **Interpretación del Silencio**:
La **ausencia sistemática de feedback cualitativo** durante la crisis representa un **gap crítico en los sistemas de captura** que impide la gestión proactiva. Los clientes expresaron su insatisfacción exclusivamente a través de puntuaciones NPS bajas, sin verbalizar problemas específicos, indicando un patrón de **"impacto silencioso"** donde los problemas operativos no se traducen en escalaciones formales.

---

### **🎯 SÍNTESIS CAUSAL**

**La crisis de mishandling del 22 de agosto de 2025 representa un evento operativo sistémico centrado en Madrid que afectó a toda la red de Iberia con manifestaciones diferenciales según el perfil del cliente**. La causa única (mishandling crítico +9.3pts) se propagó a través de todos los segmentos, pero con reactividad variable: máxima en Economy LH (-7.6pts), moderada en Premium LH (-5.56pts) y Short Haul (-2.52pts), e inmunidad completa en Business LH (+50.0pts). La concentración geográfica en rutas que involucran Madrid como hub confirma un fallo sistémico en ground handling que requiere investigación inmediata de los procesos operativos terrestres del 22 de agosto de 2025.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El día 22 de agosto de 2025 se registró una crisis operativa sistémica que impactó negativamente el NPS global, cayendo 3.41 puntos hasta alcanzar 23.82 puntos. Esta deterioro afectó de manera diferencial a los segmentos: Short Haul experimentó una caída de 2.52 puntos (NPS 27.29), mientras que Long Haul sufrió un impacto más severo con una reducción de 5.56 puntos (NPS 16.73). La causa raíz identificada fue una crisis crítica en el manejo de equipajes (mishandling) que se disparó a 27.2% versus un promedio de 17.9% (+9.3 puntos porcentuales, incremento del 52%), concentrada operativamente en el hub de Madrid. Esta causa única se propagó a través de todos los nodos del árbol jerárquico: Global/SH (-2.52 pts), Global/SH/Economy (-2.56 pts), Global/SH/Business (-2.31 pts), Global/LH (-5.56 pts), Global/LH/Economy (-7.6 pts), y Global/LH/Premium (-5.56 pts), con la notable excepción de Global/LH/Business que mantuvo un NPS positivo de 50.0 puntos, demostrando inmunidad completa a los problemas operativos.

Las rutas más severamente afectadas incluyen MAD-MEX con un NPS crítico de -14.8, MAD-VGO y ATH-MAD ambas con NPS de 0.0 (todos los pasajeros resultaron detractores), y MAD-SJU con NPS 14.3. Los grupos de clientes más reactivos fueron los residentes de Oriente Medio (NPS -50.0), Asia (NPS -40.0 a -66.7) y América Norte (NPS -9.8 a -37.5), así como los pasajeros de codeshare LATAM que registraron NPS entre -83.3 y -100.0, evidenciando que los clientes internacionales de larga distancia mostraron mayor sensibilidad a las disrupciones operativas de equipajes.

**ECONOMY SH: Impacto Operativo Distribuido**
La cabina Economy de Short Haul experimentó un deterioro de 2.56 puntos durante el 22 de agosto, registrando un NPS de 26.65 con una caída de 2.56 puntos respecto al baseline de 29.21. La causa principal fue el incremento crítico del mishandling a 27.2% (+9.3 puntos vs promedio), que se manifestó de forma concentrada geográficamente en la ruta ATH-MAD donde el 100% de los 15 pasajeros resultaron detractores (NPS 0.0), operada exclusivamente por flota A319. Este deterioro se complementó por un load factor elevado del 88.6% que incrementó la presión operativa, mientras que paradójicamente la puntualidad mejoró (OTP 86.2% vs 84.9%). El impacto fue universal en esta ruta específica, sin diferenciación por perfiles de clientes, evidenciando un fallo operativo localizado pero total.

**BUSINESS SH: Segmentación Crítica por Compañía**
El segmento Business de Short Haul mostró un deterioro de 2.31 puntos, registrando un NPS de 32.73 con una variación de -2.31 puntos vs el período anterior de 35.04. Esta evolución se explica principalmente por el mismo factor de mishandling crítico (+9.3 puntos), pero con manifestaciones diferenciadas por compañía: Iberia concentró el impacto en la ruta LHR-MAD con una dispersión récord de 143.6 puntos entre perfiles (clientes norteamericanos NPS -66.7 vs españoles NPS +76.9), mientras que Vueling experimentó problemas dispersos sistémicamente con una caída más pronunciada de 9.74 puntos. Los perfiles más reactivos incluyeron pasajeros norteamericanos, operaciones de codeshare con American Airlines (NPS -33.3) y flota A320 clásica, contrastando con el rendimiento estable de British Airways y A320neo.

**ECONOMY LH: Mayor Impacto Sectorial**
La cabina Economy de Long Haul sufrió el mayor deterioro sectorial con una caída de 7.6 puntos, registrando un NPS de 12.2 versus un baseline de 19.8. La causa principal fue la misma crisis de mishandling (+9.3 puntos) combinada con un load factor crítico del 92.1% que amplificó el impacto negativo, concentrándose específicamente en operaciones Iberia y la ruta transatlántica MAD-SJU (NPS 14.3). Esta severa degradación se reflejó especialmente entre clientes business que registraron NPS -14.3 versus leisure con NPS 14.1, y afectó desproporcionadamente a residentes europeos (NPS -7.7) y asiáticos, mientras que el codeshare LATAM mostró impacto crítico (NPS -16.7). Los perfiles más reactivos fueron los viajeros de negocios en rutas transatlánticas con expectativas de servicio premium.

**BUSINESS LH: Inmunidad Operativa Excepcional**
La cabina Business de Long Haul mantuvo un desempeño excepcionalmente estable, registrando un NPS positivo de 50.0 durante el 22 de agosto, demostrando inmunidad completa a los problemas operativos que afectaron al resto de la red. No se detectaron cambios significativos pese a estar expuesta a los mismos problemas de mishandling, manteniendo niveles consistentes de satisfacción y actuando como único factor estabilizador en el impacto global. Esta resiliencia se concentró en la ruta EZE-MAD donde el segmento mostró performance normal, evidenciando que la calidad del servicio diferenciado puede neutralizar completamente disrupciones operativas menores.

**PREMIUM LH: Reactividad Moderada Transatlántica**
El segmento Premium de Long Haul experimentó un deterioro de 5.56 puntos, registrando un NPS de 20.0 con una caída de 5.56 puntos vs el baseline de 25.56. Las causas dominantes fueron el mishandling crítico (+9.3 puntos) y sobrecarga operativa (load factor 91.5%), especialmente evidentes en la ruta EZE-MAD y operaciones de codeshare Iberia (NPS 23.8). El impacto mostró una dispersión geográfica extrema de 82.1 puntos entre regiones, afectando severamente a clientes de América Centro/Norte (NPS -25.0) mientras que América Sur y España mantuvieron performance positivo (NPS +57.1), y entre perfiles reactivos destacaron los pasajeros leisure (NPS 9.5) versus business (NPS 75.0) que demostraron mayor resistencia a las disrupciones operativas.

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