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
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

A. Economy Short Haul  
• Existen ambos nodos, SH/Economy/IB y SH/Economy/YW.  
• IB y YW convergen en un mismo patrón causal (puntualidad como driver principal con SHAP –4.93 vs –3.50, reducción de OTP15, aumento de cancelaciones y misconexiones), por lo que la causa es común a toda la cabina Economy SH.  

B. Business Short Haul  
• Existen ambos nodos, SH/Business/IB y SH/Business/YW.  
• Tanto IB (SHAP puntualidad –2.87) como YW (SHAP puntualidad –6.26) apuntan a impuntualidad, cancelaciones y misconexiones sin divergencias relevantes en los drivers operativos, de modo que la causa es general a toda la cabina Business SH.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

A. Short Haul  
• Economy SH y Business SH convergen en un patrón único: ambos muestran anomalías negativas impulsadas principalmente por impuntualidad (retrasos y cancelaciones) y problemas de conexiones.  
• La separación IB/YW no altera el diagnóstico: tanto en Economy (SHAP ~–4,3 a –4,9) como en Business (SHAP ~–2,9 a –6,3) los drivers operativos son idénticos y no hay cabina con menor reactividad.  
→ Causa común a toda la radio Short Haul.  

B. Long Haul  
• Economy LH, Business LH y Premium LH comparten la misma causa raíz: caída de puntualidad y aumento de misconexiones/cancelaciones.  
• No hay cabina aislada: las tres reflejan SHAP negativos en Punctuality (–7,3; –8,2; –5,2) y evidencias operativas coherentes.  
• Premium reacciona en menor medida (–3,8 pts vs –5,8/–6,7 pts) pero no rompe el patrón.  
→ Causa común a toda la radio Long Haul, con Premium actuando como amortiguador relativo.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Ambos radios (Short Haul y Long Haul) presentan anomalías negativas impulsadas por los mismos drivers operativos (puntualidad deficiente y aumento de cancelaciones/misconexiones), y el nodo Global refleja coherentemente ese impacto agregado (–6,7 pts), confirmando una causa global homogénea sin efectos compensatorios entre radios.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

1. Causa A: Deterioro de la puntualidad (retrasos y cancelaciones)  
A. Naturaleza del driver  
   • Driver operativo crítico: la variabilidad en tiempos de salida y llegada (OTP15), junto con cancelaciones y desvíos, genera una percepción de falta de fiabilidad y “mala gestión” que penaliza directamente la satisfacción (NPS).  

B. Evidencia consolidada y alcance  
   – Segmento seleccionado: Global (el mayor volumen de pasajeros, incluye Long Haul y Short Haul en todas las cabinas)  
   – Este nodo engloba a todos sus subsegmentos (LH, SH y sus clases Economy, Business, Premium; IB y YW).  
   – Output causal detallado (sección 2.1 del análisis Global):  
     • SHAP Punctuality: –5.237  
     • Sat_diff Punctuality vs L7d: –7.164  
     • OTP15 (operative_data_tool): cayó de 83.7% a 82.9% (–0.8 pts)  
     • Misconnections (operative_data_tool): subieron de 0.38 a 0.57 (+0.19)  
     • ncs_tool:  
       – Cancelaciones: 401 → 257 (–144)  
       – Retrasos y otras incidencias totales: 151 → 241 (+90)  
       – Limitación de aeronave: 0 → 27 (+27)  
       – Desvíos: 31 → 45 (+14)  
     • verbatims_tool: picos de quejas por “falta de información” y “mala gestión” durante retrasos y cancelaciones  
   – Impacto en NPS y rutas:  
     • NPS Global: 23.46 vs baseline 30.12 (–6.66 pts)  
     • Rutas con mayor caída: DOH–MAD (–46.7, 30 pax), DFW–MAD (–45.5, 33 pax), GRX–MAD (–3.1, 32 pax)  

2. Causa B: Incidencias de equipaje y otras incidencias operativas  
A. Naturaleza del driver  
   • Driver de servicio en tierra: la pérdida, demora o mal manejo de equipaje intensifica la insatisfacción, al añadirse a la frustración por la gestión de retrasos.  

B. Evidencia consolidada y alcance  
   – Segmento seleccionado: Global (abarca todo el tráfico y muestra el mayor número de casos)  
   – Afecta a todos los subsegmentos bajo Global (LH, SH y sus cabinas, IB y YW).  
   – Output causal detallado (sección 2.2 del análisis Global):  
     • ncs_tool – Otras incidencias (p. ej. equipaje perdido o retrasado): 151 → 241 (+90)  
     • ncs_tool – Limitación aeronave: 0 → 27 (+27)  
     • ncs_tool – Desvíos: 31 → 45 (+14)  
     • verbatims_tool: menciones crecientes de “pérdida y retraso de equipaje” y “poca asistencia para localizarlo”  
   – Impacto en NPS y rutas:  
     • NPS Global: 23.46 vs baseline 30.12 (–6.66 pts)  
     • Rutas con quejas de equipaje más frecuentes: MAD–VGO, MAD–TFN (identificadas por explanatory_drivers + ncs_tool)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
Durante la semana del 2025-10-18 al 2025-10-24, Global experimentó una bajada de NPS de 6.663 puntos, pasando de 30.120 (vs L7d) a 23.457. Este deterioro se reflejó tanto en Long Haul (NPS de 23.268→17.580, –5.688 pts vs L7d) como en Short Haul (NPS de 33.776→26.540, –7.236 pts vs L7d). A nivel de cabina, Economy LH cayó 5.820 pts (22.143→16.324), Business LH 6.672 pts (30.631→23.958) y Premium LH 3.751 pts (25.180→21.429). En Short Haul, Economy SH IB bajó 7.821 pts (28.411→20.590) y Economy SH YW 4.292 pts (39.193→34.901); Business SH IB cayó 10.949 pts (51.203→40.254) y Business SH YW 17.525 pts (44.628→27.103). Todas las caídas se explican por un empeoramiento de la puntualidad (SHAP Punctuality entre –5.237 en Global y –7.309 en Economy LH, –4.306 en SH), validado por una caída de OTP15 (–0.8 pts Global, –4.7 pts LH), un aumento de cancelaciones y “otras incidencias” (+90 en Global), y por un alza en quejas de “falta de información” y “pérdida de equipaje”.  

En rutas, DOH–MAD (–46.7 pts, 30 pax), DFW–MAD (–45.5 pts, 33 pax) y GRX–MAD (–3.1 pts, 32 pax) fueron las más impactadas, seguidas de MAD–VGO y FRA–MAD. Los pasajeros en vuelos CodeShare mostraron la mayor reactividad (spread de NPS_diff hasta 164.4 pts), seguidos por determinados modelos de Fleet (spread hasta 151.3 pts) y por Residence Region.  

ECONOMY SH: Erosionada por impuntualidad en IB y YW  
La cabina Economy SH IB experimentó un NPS de 20.590, con una caída de 7.821 pts (vs L7d 28.411→20.590), y Economy SH YW quedó en 34.901, con –4.292 pts (vs L7d 39.193→34.901). El driver principal fue la puntualidad (SHAP Punctuality –4.928 en IB y –3.499 en YW), reforzado por un aumento de misconexiones (+0.21 pts IB, +0.05 pts YW) y por el alza en cancelaciones y desvíos. En AGP–MAD y MAD–VGO se registraron las caídas más pronunciadas, y los pasajeros Fleet y CodeShare fueron los más sensibles.  

BUSINESS SH: Fuerte impacto operativo común  
Business SH IB cerró en 40.254 (–10.949 pts vs L7d 51.203→40.254) y Business SH YW en 27.103 (–17.525 pts vs L7d 44.628→27.103). Ambos reflejaron un SHAP Punctuality negativo (–2.871 IB, –6.256 YW), caída de OTP15 (–1.50 pts en IB), aumento de cancelaciones y “otras incidencias”. Los descensos más acentuados se observaron en MAD–VGO y FRA–MAD, con CodeShare como perfil de mayor reacción.  

ECONOMY LH: Puntualidad bajo presión  
La cabina Economy LH registró un NPS de 16.324, con una bajada de 5.820 pts (vs L7d 22.143→16.324). La caída responde a un SHAP Punctuality de –7.309, a una reducción de OTP15 de 82.91 % a 78.21 % (–4.7 pts) y a un aumento de retrasos (+20) y cancelaciones (+5). DOH–MAD y DFW–MAD fueron las rutas más golpeadas, y el perfil CodeShare mostró mayor sensibilidad.  

BUSINESS LH: Retrasos y conexiones penalizan 
La cabina Business LH quedó en 23.958, con –6.672 pts (vs L7d 30.631→23.958). El driver clave fue la puntualidad (SHAP –8.172), validado por OTP15 (–4.7 pts), misconexiones (+0.2) y aumento de cancelaciones. Destacan DFW–MAD y GIG–MAD, y los pasajeros según Residence Region presentaron amplia variabilidad.  

PREMIUM LH: Amortiguador parcial  
Premium LH se situó en 21.429, con una caída de 3.751 pts (vs L7d 25.180→21.429). La causa principal fue la puntualidad (SHAP –5.162), acompañada de un aumento de misconexiones (+0.2 pts). Las mayores caídas se dieron en MAD–MEX y MAD–MIA, con Residence Region como el perfil de mayor dispersión.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*



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