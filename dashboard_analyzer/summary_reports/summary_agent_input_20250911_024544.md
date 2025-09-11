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
[{'period': 1, 'date_range': '2025-08-31 to 2025-09-06', 'ai_interpretation': '📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**\n\n**Nodos analizados:** 9 (Global, Global/LH, Global/LH/Business, Global/LH/Economy, Global/LH/Premium, Global/SH/Business, Global/SH/Business/IB, Global/SH/Business/YW, Global/SH/Economy/YW)\n\n---\n\n## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA\n\nEconomy Short Haul  \n• Solo existe el nodo Global/SH/Economy/YW; falta por completo el nodo Economy SH/IB (limitación de datos). No es posible determinar si la causa es común a la cabina o específica de compañí\xada.  \n\nBusiness Short Haul  \n• Existen ambos nodos (Global/SH/Business/IB y Global/SH/Business/YW) y muestran patrones divergentes en drivers y evidencia operativa: IB muy impactado por cancelaciones y factor de carga, YW afectado también por puntualidad y elementos de producto (IFE, precio, etc.). Concluimos causas específicas por compañía.\n\n---\n\n## 💺 DIAGNÓSTICO A NIVEL DE CABINA\n\nShort Haul  \nEn Short Haul, Economy y Business muestran patrones divergentes (Economy SH/YW mejora por puntualidad mientras Business SH sufre por cancelaciones y carga, con diferencias marcadas entre IB y YW), por lo que las causas son específicas de cada cabina.  \n\nLong Haul  \nEn Long Haul, aunque las cancelaciones se repiten como driver principal en Economy, Business y Premium, la magnitud del impacto y los drivers secundarios (puntualidad en Economy/Business versus IFE y catering en Premium) divergen y muestran una progresión de reactividad (Economy amortigua, Premium intermedio, Business muy reactivo), por lo que las causas también son específicas de cabina.\n\n---\n\n## 🌎 DIAGNÓSTICO GLOBAL POR RADIO\n\nAmbos radios están afectados, pero con patrones distintos que se compensan en el agregado Global:  \n– Short Haul muestra divergencia entre cabinas (Economy mejora por puntualidad, Business se deteriora por cancelaciones y carga).  \n– Long Haul exhibe un empeoramiento común en todas las cabinas, centrado en puntualidad y cancelaciones.  \nA nivel Global, la fuerte subida en SH Economy contrarresta las caídas de SH Business y de LH, generando una mejora neta de +2.56 pts que oculta la heterogeneidad de las causas.\n\n---\n\n## 📋 ANÁLISIS DE CAUSAS DETALLADO\n\n1) Causa: Disrupciones operativas (cancelaciones y desvíos)  \nA. Naturaleza de la causa  \n   • Problemas de fiabilidad en la ejecución de vuelos que interrumpen el itinerario y deterioran la comunicación con el cliente.  \nB. Evidencia consolidada y alcance  \n   • Segmento más grande afectado: Nivel Global (todos los pasajeros).  \n   • Afecta a todos los subsegmentos bajo Global (Global/LH y Global/SH con sus cabinas).  \n   • Indicadores operativos: +33 desviaciones y +31 cancelaciones vs L7d.  \n   • Impacto en NPS Global: Actual 27.593 vs Baseline 25.034 (+2.559 pts).  \n   • Rutas involucradas (ejemplos): RJL-MAD, BCN-MAD, MAD-EZE, BCN-EZE, GYE-MAD, MAD-JFK, MAD-MIA.  \n   • Verbatims representativos: 1 742 de 6 918 comentarios mencionan “cancelaciones de último minuto”, “desvíos sin reubicación” y “falta de información en tierra”.  \n\n2) Causa: Retrasos y variabilidad de puntualidad  \nA. Naturaleza de la causa  \n   • Incremento en la frecuencia de retrasos y limitaciones de aeronave que reducen la percepción de confianza y gestión operativa.  \nB. Evidencia consolidada y alcance  \n   • Segmento más grande afectado: Global/Long Haul (todas las cabinas LH).  \n   • Afecta a todos los subsegmentos bajo Global/LH (Economy, Business, Premium).  \n   • Indicadores operativos: +9 retrasos, +4 limitaciones de aeronave y +2 desvíos vs L7d.  \n   • Impacto en NPS Global/LH: Actual 18.791 vs Baseline 19.845 (–1.054 pts).  \n   • Rutas más impactadas: DOH-MAD (NPS –52.5, 40 pax), JFK-MAD (2.4, 84 pax), DFW-MAD (0.0, 34 pax), MAD-SFO (0.0, 23 pax), MAD-SJU (35.0, 60 pax).  \n   • Verbatims representativos: 2 750 comentarios con quejas de “retrasos prolongados”, “escasa actualización de estado” y “esperas sin solución”.  \n\n3) Causa: Variación en Load Factor  \nA. Naturaleza de la causa  \n   • Cambios en la ocupación que generan percepción de sobreventa o de cabina vacía, afectando el confort y la sensación de valor.  \nB. Evidencia consolidada y alcance  \n   • Segmento más grande afectado: Nivel Global (todos los pasajeros).  \n   • Afecta a todos los subsegmentos bajo Global (infunde impacto en LH y SH, Economy y Business).  \n   • Indicadores operativos:  \n     – Global: Load factor diario –1.2 pts vs L7d.  \n     – Global/LH: Load factor – aumento de +2.0 pts (92.52 → 90.57).  \n     – Global/LH/Business: Load factor +2.18 pts.  \n     – Global/SH/Business: Load factor +2.57 pts (68.66 → 71.23).  \n   • SHAP del driver: variable pero hasta 0.165 en Global.  \n   • Rutas con evidencia por ocupación: BCN-VLC, SVQ-VLC, BIO-SCQ (según driver de producto).  \n   • Verbatims representativos: menciones a “asientos apretados” y “sobreventa” en verbatims de Economy y Business.  \n\n4) Causa: Deficiencias en servicios a bordo (IFE, catering, Wi-Fi)  \nA. Naturaleza de la causa  \n   • Calidad y disponibilidad de entretenimiento y catering debajo de expectativas en clases Premium, erosionando la percepción de valor.  \nB. Evidencia consolidada y alcance  \n   • Segmento más grande afectado: Global/Long Haul/Premium.  \n   • Afecta a todos los clientes bajo Global/LH/Premium.  \n   • Indicadores operativos y SHAP:  \n     – IFE SHAP = –5.342, Sat_diff = –11.888.  \n     – In-flight food & beverage SHAP = –2.708, Sat_diff = –0.152.  \n     – Wi-Fi SHAP = –1.713, Sat_diff = –6.200.  \n   • Impacto en NPS Premium LH: Actual 18.382 vs Baseline 22.772 (–4.390 pts).  \n   • Rutas clave: MAD-MEX (NPS 43.5, 23 pax), MAD-ORD (–100.0, 1 pax), MAD-MIA (–100.0, 1 pax).  \n   • Verbatims representativos: 224 comentarios con “IFE no funciona”, “comida escasa” y “Wi-Fi caído”.  \n\n5) Causa: Comunicación y gestión post-incidente  \nA. Naturaleza de la causa  \n   • Falta de información proactiva y de procesos ágiles de reubicación que agravan la frustración tras un incidente operativo.  \nB. Evidencia consolidada y alcance  \n   • Segmento más grande afectado: Global/Short Haul/Economy/YW.  \n   • Afecta a todo el subsegmento Global/SH/Economy/YW (y previsiblemente al Economy SH/IB, si existiera).  \n   • Verbatims: 1 232 comentarios con “sin notificación previa”, “tiempos de espera inaceptables” y “no me reubican rápido”.  \n   • Rutas afectadas: RJL-MAD, BCN-MAD, MAD-EZE, BCN-EZE, GYE-MAD, MAD-BOG, MAD-MIA, MAD-LAX, MAD-DFW, MAD-SDQ, MAD-ALG, MAD-NRT, MAD-SIN, MAD-DEL.  \n   • Impacto en NPS SH Economy YW: Actual 40.205 vs Baseline 29.952 (+10.253 pts, con alta variabilidad por CodeShare y Residence Region).\n\n---\n\n## 📋 SÍNTESIS EJECUTIVA FINAL\n\n\n\n---\n\n✅ **ANÁLISIS COMPLETADO**\n\n- **Nodos procesados:** 9\n- **Pasos de análisis:** 5\n- **Metodología:** Análisis conversacional paso a paso\n- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado\n\n*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*'}]

**ANÁLISIS DIARIO SINGLE:**


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