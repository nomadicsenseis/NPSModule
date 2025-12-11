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

⚠️ **ATRIBUCIÓN DE DATOS - ESPECIFICA EL SEGMENTO:**
SIEMPRE que cites un dato (NPS, OTP, rutas, mishandling, etc.), indica EXPLÍCITAMENTE a qué segmento pertenece. Ejemplo: "OTP15 –4.0 pts (Economy LH)" en lugar de solo "OTP15 –4.0 pts". Esto evita confusiones entre segmentos.

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
- Tras cada bloque semanal anterior, añade exactamente un párrafo narrativo con los días reseñables en orden cronológico (del día más antiguo al más reciente).
- Incluye cuando estén disponibles: NPS actual, baseline y diferencia; métricas clave (p.ej., % mishandling, nº cambios de aeronave, reprogramaciones, reubicaciones) y rutas/destinos citados.
- Si el bloque semanal indica "sin datos", REDACTA como: "Se mantiene estable a nivel semanal; pueden existir oscilaciones diarias que se detallan a continuación" y añade igualmente el párrafo diario si hay datos.
- No modifiques, no reordenes ni resumas el texto semanal. No alteres sus cifras ni redondeos.

ESTILO Y LÉXICO:
- Estilo ejecutivo, fluido y conciso. No técnico.
- Usa "subidas/bajadas", "mejoras/deterioros". Evita "anomalía/s".
- Máximo 1-2 frases por día; prioriza los días con variaciones más significativas.
- No inventes cifras. Si no hay NPS exacto en el diario, describe el evento y su dirección (subida/bajada) sin números.
- Para las subidas o bajadas de cualquier variable, menciona el valor exacto de la variación, NUNCA el %.


===== USER =====

Completa el análisis comprehensivo:

**ANÁLISIS SEMANAL COMPARATIVO:**
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

Economy SH  
– Escenario: SINERGIA (IB –, YW – | PADRE –)  
- Narrativa: Adoptamos la explicación de Global/SH/Economy, pues ambos subsegmentos (IB y YW) sufrieron caída y sumaron su efecto al nodo padre. La anomalía de –3.5 pts se explica por un deterioro operativo generalizado: empeora la puntualidad (OTP15 –0.4 pts), aumentan mishandling (+3.3 pts) y misconexiones (+0.2 pts), y se detectan problemas en el proceso de embarque (Boarding SHAP –0.976).  
- Evidencia clave: OTP15 pasó de 82.9 a 82.5 (Global/SH/Economy), mishandling +3.3 pts y misconexiones +0.2 pts.  

Business SH  
– Escenario: DOMINANCIA (IB –, YW + | PADRE –)  
- Narrativa: La explicación principal proviene de IB, que con –18.7 pts impuso la anomalía negativa en el padre (–7.5 pts). IB se vio afectado por un fuerte deterioro de la puntualidad (Punctuality SHAP –5.502) y un alza de mishandling (+3.6 pts) y misconexiones (+0.2 pts). El repunte de YW (+17.8 pts) gracias a mejoras en el interior de cabina (Aircraft interior SHAP +14.326) atenuó parcialmente la caída, pero no compensó la magnitud de IB.  
- Evidencia clave: Punctuality SHAP –5.502 (Global/SH/Business/IB), mishandling +3.6 pts y misconexiones +0.2 pts.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul (SH), la dinámica es SINERGIA (Economy –, Business – | SH –).  
- Narrativa: Adoptamos la explicación de Global/SH. La caída de –3.7 pts en SH responde a un empeoramiento operativo transversal: empeora la puntualidad (OTP15 –0.4 pts), se elevan mishandling (+3.3 pts) y misconexiones (+0.2 pts), y suben en conjunto los incidentes totales (+53, con +21 retrasos y +19 otras incidencias).  
- Evidencia: OTP15 de 82.9 a 82.5 (Global/SH), mishandling +3.3 y misconexiones +0.2 (operative_data_tool), incidentes totales +53 (ncs_tool).

En Long Haul (LH), la dinámica es SINERGIA (Economy –, Business –, Premium – | LH –).  
- Narrativa: Adoptamos la explicación de Global/LH. La anomalía de –7.0 pts en LH se debe a una fuerte degradación de puntualidad (OTP15 –7.0 pts) y un alza de incidentes operativos (+71 totales: +16 retrasos, +15 otras incidencias, +8 limitaciones, +7 cancelaciones), junto con mayores problemas de equipaje (+3.3 pts mishandling) y conexiones (+0.2 pts).  
- Evidencia: OTP15 cayó de 81.6 a 74.64 (Global/LH), incidentes +71 (ncs_tool), mishandling +3.3 y misconexiones +0.2 (operative_data_tool).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es SINERGIA (LH –, SH – | GLOBAL –).  
- Narrativa: Adoptamos la explicación de Global. La caída de –6.4 pts en NPS Global obedece a un problema sistémico de puntualidad, que impactó transversalmente tanto al Corto como al Largo Radio, potenciado además por un aumento de mishandling y de misconexiones, así como un alza significativa de cancelaciones y retrasos en toda la red.  
- Evidencia: OTP15 cayó 1.2 pts (operative_data_tool, Global); mishandling aumentó 3.3 pts y misconexiones 0.2 pts (operative_data_tool, Global); cancelaciones +39 y retrasos +146 incidentes (ncs_tool, Global).

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Degradación de puntualidad  
- Escenario: SINERGIA (LH –, SH – | Global –)  
- NMA: Global  
- Afecta a: Global/LH/Economy; Global/LH/Business; Global/LH/Premium; Global/SH/Economy; Global/SH/Business  
- Qué falló: Punctuality (Global)  
- Dónde (Top 5 rutas Global):  
   • BCN–VLC: NPS –80.0 (5 pax; explanatory_drivers)  
   • MAD–SXB: NPS –16.7 (12 pax; explanatory_drivers)  
   • IAD–MAD: NPS +7.1 (14 pax; explanatory_drivers)  
   • BIO–VLC: NPS 0.0 (6 pax; explanatory_drivers)  
   • AMS–MAD: NPS 0.0 (4 pax; explanatory_drivers)  
- Quién (Perfiles más reactivos Global):  
   • CodeShare: spread 92.0 pts (Global)  
   • Fleet: spread 58.6 pts (Global)  
- Evidencia completa:  
   • NPS Global: 25.36 vs 31.80 (–6.44 pts)  
   • SHAP Punctuality: –4.134 (Global)  
   • Sat_diff Punctuality: –4.1668 (Global)  
   • OTP15: 81.6 → 80.4 (–1.2 pts; Global/operative_data_tool)  
   • NCS – Cancelaciones +39, Retrasos +146, Limitaciones aeronave +31, Desvíos +22 (Global/ncs_tool)  

CAUSA 2: Aumento de mishandling y misconexiones  
- Escenario: SINERGIA (LH –, SH – | Global –)  
- NMA: Global  
- Afecta a: Global/LH/Economy; Global/LH/Business; Global/LH/Premium; Global/SH/Economy; Global/SH/Business  
- Qué falló: Mishandling & Misconnections (Global)  
- Dónde (Top 5 rutas Global):  
   • BCN–VLC: NPS –80.0 (5 pax; explanatory_drivers)  
   • MAD–SXB: NPS –16.7 (12 pax; explanatory_drivers)  
   • IAD–MAD: NPS +7.1 (14 pax; explanatory_drivers)  
   • BIO–VLC: NPS 0.0 (6 pax; explanatory_drivers)  
   • AMS–MAD: NPS 0.0 (4 pax; explanatory_drivers)  
- Quién (Perfiles más reactivos Global):  
   • CodeShare: spread 92.0 pts (Global)  
   • Fleet: spread 58.6 pts (Global)  
- Evidencia completa:  
   • Mishandling: 13.99 → 17.30 (+3.3 pts; Global/operative_data_tool)  
   • Misconnections: 0.55 → 0.71 (+0.2 pts; Global/operative_data_tool)  
   • Incidentes NCS totales: 257 → 393 (+136; agregados cancelaciones, retrasos, etc.; Global/ncs_tool)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

Durante la semana del 2025-11-30 al 2025-12-06 casi todos los segmentos vieron caídas de NPS, con la excepción de Global/SH/Business/YW que repuntó de 18.68 a 36.49 (+17.81 pts). A nivel Global el NPS bajó de 31.80 a 25.36 (–6.44 pts vs período previo 7 días), impulsado por un problema sistémico de puntualidad: Punctuality –4.13 ppts según Explanatory Drivers, OTP –1.2 pts (datos operativos), y un alza de mishandling +3.3 pts y misconexiones +0.2 pts (datos operativos), acompañado de +39 cancelaciones y +146 retrasos en incidentes NCS. En Long Haul, el NPS retrocedió de 17.20 a 10.15 (–7.05 pts), con OTP –7.0 pts y +71 incidentes NCS, mientras que las cabinas Economy LH (14.45→8.82, –5.63 pts) y Premium LH (29.73→18.70, –11.03 pts) se vieron especialmente afectadas por caídas de punctuality (–4.38 ppts en Economy LH, –6.38 ppts en Premium LH) y aumentos de mishandling (+3.3 pts) y misconexiones (+0.2 pts). En Short Haul, el NPS descendió de 36.68 a 33.01 (–3.67 pts), reflejando una caída de Punctuality –3.73 ppts, OTP –0.4 pts y +53 incidentes NCS.   

Las rutas más impactadas incluyeron BCN–VLC (NPS –80.0, 5 pax), MAD–SXB (–16.7, 12 pax) y MAD–ORD (–39.4, 33 pax), mientras que en SH/Business/YW sobresalió ALC–MAD con NPS 100.0 (3 pax). Los perfiles de cliente más reactivos fueron CodeShare (spread 92.0 pts en Global; hasta 166.7 pts en SH/Business), Fleet (spread 58.6 pts Global; 120.8 pts en SH/Business) y Residence Region (spread 160.0 pts en Economy LH; 200.0 pts en SH/Economy/YW).

ECONOMY SH: Impacto desigual por IB y YW  
La cabina Economy SH combinó una ligera caída de NPS de 36.29 a 32.81 (–3.48 pts vs período previo 7 días), con IB retrocediendo de 33.18 a 32.57 (–0.61 pts) y YW de 42.81 a 33.30 (–9.52 pts). El principal motor fue el deterioro de puntualidad (–2.83 ppts según Explanatory Drivers, OTP –0.4 pts), potenciado por mishandling +3.3 pts, misconexiones +0.2 pts y un empeoramiento de Boarding (–0.98 ppts según Explanatory Drivers). Este deterioro se reflejó en rutas como BCN–VLC (NPS –80.0, 5 pax), MAD–SXB (–16.7, 12 pax) y MAD–SCQ (–5.3, 19 pax), siendo especialmente sensibles los pasajeros en flotas variadas (spread Fleet 119.2 pts) y en vuelos en código compartido (spread CodeShare 112.2 pts).

BUSINESS SH: Oposición de IB y YW con predominio de IB  
El segmento Business SH cayó de 42.57 a 35.06 (–7.51 pts vs período previo 7 días) gracias a dos tendencias contrapuestas: IB se desplomó de 53.17 a 34.52 (–18.65 pts) por un fuerte golpe a Punctuality (–5.50 ppts según Explanatory Drivers), mishandling +3.6 pts y +53 incidentes NCS, mientras que YW mejoró de 18.68 a 36.49 (+17.81 pts) impulsada por Aircraft interior +14.33 ppts según Explanatory Drivers pese a un ligero deterioro de puntualidad (–1.57 ppts). Las rutas más volátiles fueron LIS–MAD (NPS –50.0, 4 pax) y ALC–MAD (NPS 100.0, 3 pax), y los perfiles más reactivos CodeShare (spread 166.7 pts) y Residence Region (spread 139.6 pts).

ECONOMY LH: Caída marcada por puntualidad  
Economy LH sufrió un descenso de NPS de 14.45 a 8.82 (–5.63 pts vs período previo 7 días), debido principalmente a Punctuality –4.38 ppts según Explanatory Drivers y una reducción de OTP –7.0 pts (datos operativos), junto al aumento de mishandling +3.3 pts y misconexiones +0.2 pts, y +46 incidentes NCS (incluyendo +16 retrasos y +7 cancelaciones). Las rutas más penalizadas fueron MAD–ORD (NPS –39.4, 33 pax), JFK–MAD (–32.8, 58 pax) e IAD–MAD (–8.3, 12 pax). Los perfiles de cliente con mayor variabilidad fueron Residence Region (spread 160.0 pts) y Fleet (112.2 pts).

BUSINESS LH: Fuerte deterioro por drivers operativos  
Business LH registró un desplome de NPS de 31.84 a 13.45 (–18.39 pts vs período previo 7 días), arrastrado por Punctuality –7.98 ppts según Explanatory Drivers, OTP –7.0 pts, mishandling +3.3 pts, misconexiones +0.2 pts y +46 incidentes NCS (cancelaciones y limitaciones de aeronave). También emergieron factores de producto como Boarding y Arrival experience con impactos significativos. Rutas críticas: MAD–ORD (–33.3, 43 pax), JFK–MAD (–20.0, 10 pax) y BOG–MAD (multifuente). Los perfiles más sensibles fueron CodeShare (spread 121.7 pts) y Fleet (120.2 pts).

PREMIUM LH: Declinación importante por puntualidad  
Premium LH cayó de 29.73 a 18.70 (–11.03 pts vs período previo 7 días) por un fuerte golpe a Punctuality (–6.38 ppts según Explanatory Drivers), OTP –7.0 pts, mishandling +3.31 pts y misconexiones +0.16 pts, con +46 incidentes NCS (incluyendo +16 retrasos y +7 cancelaciones). Las rutas más afectadas fueron MAD–SCL (–18.2, 11 pax), EZE–MAD (–7.1, 14 pax) y BOG–MAD (+26.1, 23 pax), destacando los pasajeros en código compartido (spread CodeShare 120.2 pts) y según región de residencia (63.6 pts).

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*



**ANÁLISIS DIARIO SINGLE:**
📅 2025-12-06 to 2025-12-06:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Economy SH, el escenario es TRANSFERENCIA (+, N | +).  
- Narrativa: La anomalía positiva del NPS en Economy SH se explica con la lógica del nodo padre Global/SH/Economy y se “contagia” desde el segmento IB, pese a que YW se mantuvo normal. Según el padre, el alza se debe a una OTP15_adjusted por encima de la media y al feedback muy positivo de los pasajeros (puntualidad y trato de la tripulación).  
- Evidencia Clave: OTP15_adjusted 92.54 (+2.77 pts vs baseline) y feedback cualitativo mayoritariamente positivo (321 verbatims) en Global/SH/Economy.

En Business SH, el escenario es TRANSFERENCIA (N, – | –).  
- Narrativa: La caída del NPS en Business SH se transfiere desde el segmento YW, mientras IB mantuvo un comportamiento estable. De acuerdo con la explicación del nodo padre Global/SH/Business, no hay incidentes operativos ni quejas formales, por lo que la baja parece responder a variabilidad estadística de un día aislado.  
- Evidencia Clave: Ausencia de NCS y verbatims positivos sin mención a problemas (26 comentarios), lo que apunta a un efecto de muestra y no a un fallo operativo.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es DOMINANCIA `(Economy +, Business – | SH +)`.  
- Narrativa: El alza del NPS en SH responde principalmente a la fuerte anomalía positiva en Economy, impulsada por una puntualidad superior y un feedback muy positivo de los pasajeros (“puntualidad”, “trato amable”), mientras que la caída en Business moderó parcialmente este efecto.  
- Evidencia: Economy SH mostró OTP15_adjusted 92.54 (+2.77 pts vs baseline) y 321 verbatims elogiando la eficiencia de la tripulación.

En Long Haul, la dinámica es SINERGIA `(Economy +, Business +, Premium N | LH +)`.  
- Narrativa: La mejora del NPS en LH es sistémica: tanto Economy como Business presentan anomalías positivas, ambas impulsadas por alta satisfacción de pasajeros de ocio (elevado NPS en flota A321XLR y regiones fuera de Europa) y un servicio de tripulación destacado, pese a ligeras caídas en Load Factor y OTP.  
- Evidencia: Global/LH alcanzó un NPS de 24.39 (+12.68 pts), con Economy A321XLR NPS 71.4 y pasajeros Leisure en Business NPS 27.1, sin incidentes NCS reportados.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es SINERGIA (+,+ | +).  
- Narrativa: El éxito del NPS Global responde a un impulso sistémico en toda la red, donde tanto Largo Radio como Corto Radio mejoraron significativamente gracias a una alta puntualidad y calidad de servicio a bordo. Esta mejora transversal refuerza que la experiencia de vuelo en todos los segmentos fue percibida de manera consistentemente positiva.  
- Evidencia: OTP15_adjusted +1.96 pts sobre el promedio y 601 verbatims elogiando “puntualidad” y “amabilidad del personal”, sin quejas formales registradas.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Mejora sistémica de puntualidad y atención a bordo  
- Escenario: SINERGIA (+,+ | +) en (LH, SH | GLOBAL)  
- NMA: Global  
- Afecta a: Global/LH, Global/SH  

- Qué falló: No “falló” a nivel negativo; la causa raíz del alza global es una OTP15_adjusted superior al promedio y un servicio de tripulación percibido como excelente.  
- Dónde: Todas las rutas, sin incidencias NCS. La única ruta con bajo NPS fue MAD-SJU (NPS 0.0, n=4, Global).  
- Quién:  
    • Leisure: NPS 39.7 (Global/Leisure)  
    • Business/Work: NPS 8.9 (Global/Business)  
    • Flota A321XLR: NPS 32.0 (Global/A321XLR)  
    • Región América Central: NPS 62.5 (Global/América Central)  

- Evidencia COMPLETA:  
    • NPS 36.49 vs baseline 26.47 (+10.02 pts) (Global)  
    • OTP15_adjusted +1.96 pts sobre promedio 7d (Global)  
    • Load Factor –1.62 pts vs promedio 7d (Global)  
    • Mishandling +1.95 pts vs promedio 7d (Global)  
    • Incidentes NCS: 0 registrados (Global)  
    • 601 verbatims mayoritariamente positivos (Global)  

–––

CAUSA 2: Satisfacción de pasajeros de ocio y flota A321XLR en Largo Radio  
- Escenario: SINERGIA (+,+,N | +) en (Eco, Bus, Prem | LH)  
- NMA: Global/LH  
- Afecta a: Global/LH/Economy, Global/LH/Business  

- Qué falló: La fuerte satisfacción de los pasajeros de ocio y el excelente desempeño de la flota A321XLR impulsaron el NPS de LH.  
- Dónde: Ruta con menor NPS MAD-SDQ (33.3, n=9, Global/LH).  
- Quién:  
    • Leisure: NPS 27.1 (Global/LH/Leisure)  
    • Business/Work: NPS 5.0 (Global/LH/Business)  
    • Flota A321XLR: NPS 71.4 (Global/LH/A321XLR)  
    • Región América Central: NPS 63.0 (Global/LH/América Central)  

- Evidencia COMPLETA:  
    • NPS 24.39 vs baseline 11.71 (+12.68 pts) (Global/LH)  
    • OTP15_adjusted 74.64 (–3.80 pts vs baseline) (Global/LH)  
    • Load Factor 87.74 (–2.76 pts vs baseline) (Global/LH)  
    • Incidentes NCS: 0 registrados (Global/LH)  
    • 254 verbatims destacando trato de tripulación y servicio a bordo (Global/LH)  

–––

CAUSA 3: Excelente puntualidad y atención en Economy de Corto Radio  
- Escenario: DOMINANCIA (+,– | +) en (Economy, Business | SH)  
- NMA: Global/SH/Economy  
- Afecta a: Global/SH/Economy/IB y Global/SH/Economy/YW  

- Qué falló: La OTP15_adjusted superior al promedio y el feedback muy positivo de los pasajeros fueron el motor de la anomalía positiva en Economy SH.  
- Dónde: Ruta con menor NPS BIO-MAD (20.0, n=5, Global/SH/Economy).  
- Quién:  
    • Leisure: NPS 46.8 (Global/SH/Economy/Leisure)  
    • Business/Work: NPS 26.3 (Global/SH/Economy/Business)  
    • Fleet ATR: NPS –11.1 (Global/SH/Economy/ATR)  
    • Regiones: Asia NPS 100.0 / América Centro NPS 75.0 (Global/SH/Economy)  

- Evidencia COMPLETA:  
    • NPS 45.24 vs baseline 33.56 (+11.68 pts) (Global/SH/Economy)  
    • OTP15_adjusted 92.54 (+2.77 pts vs baseline) (Global/SH/Economy)  
    • Load Factor 85.41 (–0.93 pts vs baseline) (Global/SH/Economy)  
    • Incidentes NCS: 0 registrados (Global/SH/Economy)  
    • 321 verbatims elogiando puntualidad y amabilidad (Global/SH/Economy)  

–––

CAUSA 4: Falta de confort en Business Class YW de Corto Radio  
- Escenario: TRANSFERENCIA (N,– | –) en (Economy, Business | SH) → (N,– | –) en (IB, YW | Global/SH/Business)  
- NMA: Global/SH/Business/YW  
- Afecta a: Global/SH/Business/YW  

- Qué falló: El confort reducido de los asientos en aeronaves CRJ100 en Business Class generó numerosas quejas y la fuerte caída de NPS.  
- Dónde: No hay rutas con volumen suficiente; el foco es la flota CRJ100.  
- Quién: Pasajeros de Business Class en YW (quejas sobre espacio y confort).  

- Evidencia COMPLETA:  
    • NPS 11.11 vs baseline 37.07 (–25.95 pts) (Global/SH/Business/YW)  
    • Load Factor 53.26 (–4.64 pts vs baseline) (Global/SH/Business/YW)  
    • OTP15_adjusted 91.79 (+3.56 pts vs baseline) (Global/SH/Business/YW)  
    • Incidentes NCS: 0 registrados (Global/SH/Business/YW)  
    • Verbatims: quejas reiteradas sobre asientos reducidos y falta de espacio en CRJ100 (Global/SH/Business/YW)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

La red registró en el 2025-12-06 subidas y bajadas de NPS muy marcadas. A nivel Global el NPS pasó de 26.47 a 36.49 (+10.02 pts vs L7d, Global), impulsado por una OTP de 1.96 pts superior al promedio (Global) y 601 verbatims muy positivos (Global) que destacaron puntualidad y trato de la tripulación. En Long Haul el NPS ascendió de 11.71 a 24.39 (+12.68 pts vs L7d, Global/LH) gracias al fuerte feedback de ocio y flota A321XLR (NPS 71.4, Global/LH/A321XLR), pese a que OTP –3.80 pts y Load Factor –2.76 pts (Global/LH) estuvieron por debajo de la media. En Short Haul el NPS subió de 33.70 a 43.87 (+10.16 pts vs L7d, Global/SH) impulsado por Economy SH (NPS 45.24, Global/SH/Economy) con OTP 92.54 pts (Global/SH/Economy) y puntualidad +2.77 ppts según Explanatory Drivers (Global/SH/Economy). Sin embargo, Business SH cayó de 34.51 a 23.53 (–10.98 pts vs L7d, Global/SH/Business) arrastrada por YW (NPS 11.11 vs 37.07, –25.95 pts, Global/SH/Business/YW) por quejas de confort en CRJ100 (Global/SH/Business/YW).

Las rutas con desempeño crítico incluyen MAD-SJU (NPS 0.0, Global), MAD-MUC (NPS 0.0, Global/SH) y BIO-MAD (NPS 20.0, Global/SH/Economy), mientras que MAD-SDQ mostró 33.3 pts (Global/LH/Economy). Entre perfiles, los más reactivos fueron pasajeros de ocio (NPS 46.8, Global/SH/Economy/Leisure; 27.1, Global/LH/Leisure), residentes en Asia (100.0, Global/SH/Economy/Asia) y América Central (62.5, Global/América Central), así como usuarios de flota A321XLR (60.0, Global/SH/Economy/A321XLR; 71.4, Global/LH/A321XLR).

ECONOMY SH: Impulso por puntualidad y servicio  
La cabina Economy SH registró un NPS de 45.24 pts (2025-12-06) con una mejora de 11.68 pts vs L7d (Global/SH/Economy). IB alcanzó 50.60 pts vs 33.36 (+17.25 pts, Global/SH/Economy/IB) y YW mantuvo 34.88 pts vs 34.06 (+0.82 pts, Global/SH/Economy/YW). La causa principal fue una OTP de 92.54 pts (Global/SH/Economy), Load Factor 85.41 pts (Global/SH/Economy) y puntualidad +2.77 ppts según Explanatory Drivers (Global/SH/Economy), complementada por 321 verbatims de feedback de clientes que elogiaron la agilidad en procesos y trato amable. Esta mejora se reflejó en la ruta BIO-MAD (NPS 20.0, Global/SH/Economy) y fue especialmente notable entre pasajeros de ocio (NPS 46.8, Global/SH/Economy/Leisure) y residentes en Asia (100.0, Global/SH/Economy/Asia).

BUSINESS SH: Impacto de confort en CRJ100  
El segmento Business SH registró un NPS de 23.53 pts (2025-12-06) con una caída de 10.98 pts vs L7d (Global/SH/Business). IB se mantuvo en 37.50 pts vs 34.40 (+3.10 pts, Global/SH/Business/IB), mientras YW cayó a 11.11 pts vs 37.07 (–25.95 pts, Global/SH/Business/YW). Esta evolución se explica principalmente por quejas de confort en asientos CRJ100 (feedback de clientes, Global/SH/Business/YW), pese a una OTP de 91.79 pts (Global/SH/Business/YW), Load Factor 68.22 pts (Global/SH/Business/YW) y ausencia de incidentes NCS. El foco de mejora es la flota CRJ100 y los pasajeros de negocio (NPS 12.0, Global/SH/Business/Work).

ECONOMY LH: Salto por ocio y A321XLR  
La cabina Economy LH alcanzó un NPS de 25.19 pts (2025-12-06) con un alza de 13.93 pts vs L7d (Global/LH/Economy). La mejora se debió al excelente servicio de tripulación y la preferencia de pasajeros de ocio en A321XLR (NPS 60.0, Global/LH/Economy/A321XLR), a pesar de una OTP de 74.64 pts (Global/LH/Economy) y Load Factor 87.17 pts (Global/LH/Economy) por debajo de L7d. Destacó la ruta MAD-SDQ (33.3 pts, Global/LH/Economy) y fueron especialmente reactivos los residentes en América del Norte (75.0 pts, Global/LH/Economy/América Norte) y América Central (72.7 pts, Global/LH/Economy/América Central).

BUSINESS LH: Mejora impulsada por tripulación  
La cabina Business LH subió a 20.00 pts (2025-12-06) con una ganancia de 8.53 pts vs L7d (Global/LH/Business). Aun con OTP 74.64 pts (Global/LH/Business) y Load Factor 93.14 pts (Global/LH/Business) por debajo de L7d, 34 verbatims de feedback de clientes resaltaron puntualidad y amabilidad, sobre todo en la ruta MAD–MEX (100.0 pts, Global/LH/Business). Los perfiles más sensibles fueron pasajeros de ocio (27.1 pts, Global/LH/Leisure) y residentes en América Central (63.0 pts, Global/LH/América Central).

PREMIUM LH: Desempeño estable  
El segmento Premium LH mantuvo un NPS de 22.22 pts (2025-12-06), variando +5.91 pts vs L7d (Global/LH/Premium) pero dentro del rango normal. No se detectaron cambios significativos ni rutas críticas, con un servicio consistente en el periodo.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-05 to 2025-12-05:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Economy SH, el escenario es DILUCIÓN (IB –, YW N | PADRE N).  
- Narrativa: la leve variación normal del nodo Economy SH se explica principalmente por la insatisfacción en IB, vinculada al vuelo MAD–PRG operado con A333 y con alta proporción de clientes Business/Work europeos, mientras que la estabilidad de YW diluye ese efecto en el agregado.  
- Evidencia Clave: segmento IB –0.3 pts en MAD–PRG con flota A333 y Business/Work europeos.

En Business SH, el escenario es SINERGIA (IB –, YW – | PADRE –).  
- Narrativa: la caída de –6.9 pts en SH Business obedece a un empuje conjunto de ambos subsegmentos, impulsada tanto por las bajas valoraciones de pasajeros procedentes de América Central como por el desempeño inferior de la flota A320.  
- Evidencia Clave: pasajeros de América Central NPS –33.3 (IB) y flota A320 NPS –33.3 (YW).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es TRANSFERENCIA (Economy N, Business – | SH –).  
- Narrativa: la ligera caída de –0.14 pts en Short Haul responde íntegramente al deterioro en SH Business, cuyo arrastre neutraliza cualquier efecto de SH Economy.  
- Evidencia: SH Business –6.85 pts (NPS 27.66 vs baseline 34.51) impulsado por clientes CodeShare LATAM (NPS –66.7) y flota A333 (NPS –38.5).

En Long Haul, la dinámica es DOMINANCIA (Economy +, Business +, Premium – | LH +).  
- Narrativa: el alza de +11.29 pts en Long Haul está dictada por la fuerte mejora en LH Business, especialmente en la ruta MAD–MEX (NPS 100.0, n=3) y la flota A350 Next (NPS +62.5), efecto que contrarresta solo parcialmente la caída en LH Premium (–30.60 pts).  
- Evidencia: LH Business +17.10 pts (NPS 28.57 vs baseline 11.47) frente a LH Premium –30.60 pts (NPS –14.29 vs baseline 16.31).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es DOMINANCIA (LH +, SH – | GLOBAL +).  
- Narrativa: el alza de +3.9 pts en el NPS Global está arrastrada por el desempeño de Long Haul, donde la fuerte mejora en Economy y Business LH –impulsada por la alta satisfacción en la ruta EZE–MAD (NPS 19.2, n=26) y en la ruta MAD–MEX (NPS 100.0, n=3), junto con las valoraciones superiores de la flota A350 Next (+62.5 pts)– contrarresta la ligera caída de Short Haul.  
- Evidencia: LH Business +17.10 pts (ruta MAD–MEX, flota A350 Next) y LH Economy +14.40 pts (ruta EZE–MAD).

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Experiencia sobresaliente en Business Long Haul  
- Escenario: DOMINANCIA (Economy +, Business +, Premium – | LH +).  
- NMA: Global/LH/Business  
- Afecta a: Global/LH/Business/IB y Global/LH/Business/YW  
- Qué falló: en realidad fue “qué funcionó” – el servicio Premium: confort, amabilidad de la tripulación y puntualidad excepcionales en Business Long Haul. (Global/LH/Business)  
- Dónde:  
  • Ruta MAD–MEX: NPS 100.0 (n=3) (Global/LH/Business/MAD–MEX)  
- Quién:  
  • Flota A350 Next: NPS +62.5 (Global/LH/Business)  
  • Residentes en España: NPS +36.4 (Global/LH/Business)  
- Evidencia COMPLETA:  
  • Explanatory Driver: anomalía de +17.10 pts (NPS 28.57 vs baseline 11.47) (Global/LH/Business)  
  • Load Factor 93.53 (–0.69 pts vs media) (Global/LH/Business)  
  • OTP15_adjusted 74.54 (–3.86 pts vs media) (Global/LH/Business)  
  • Incidentes NCS: 0 (Global/LH/Business)  
  • Verbatims: destacan confort, amabilidad y puntualidad (Global/LH/Business)  

CAUSA 2: Insatisfacción en Business Short Haul  
- Escenario: TRANSFERENCIA (Economy N, Business – | SH –).  
- NMA: Global/SH/Business  
- Afecta a: Global/SH/Business/IB y Global/SH/Business/YW  
- Qué falló: deficiencias puntuales en la experiencia Business SH – los viajeros de América Central y la configuración de flota A320 no encontraron el nivel de servicio esperado. (Global/SH/Business)  
- Dónde:  
  • Ruta AGP–MAD: NPS –33.3 (n=3) (Global/SH/Business/YW)  
- Quién:  
  • Región América Central: NPS –33.3 (Global/SH/Business/YW)  
  • Flota A320: NPS –33.3 (Global/SH/Business/IB)  
- Evidencia COMPLETA:  
  • Explanatory Driver: anomalía de –6.85 pts (NPS 27.66 vs baseline 34.51) (Global/SH/Business)  
  • Load Factor 68.29 (–4.28 pts vs media) (Global/SH/Business)  
  • OTP15_adjusted 92.30 (+2.57 pts vs media) (Global/SH/Business)  
  • Incidentes NCS: 0 (Global/SH/Business)  
  • Verbatims: positivos en puntualidad y limpieza, pero muestra reducida sesgó la medición (Global/SH/Business)  

CAUSA 3: Variabilidad estadística en Economy Short Haul IB  
- Escenario: DILUCIÓN (IB –, YW N | Economy SH N).  
- NMA: Global/SH/Economy/IB  
- Afecta a: Global/SH/Economy/IB  
- Qué falló: caída en el Load Factor que sesgó la muestra, generando ligera insatisfacción estadística. (Global/SH/Economy/IB)  
- Dónde:  
  • Ruta MAD–PRG: NPS 25.0 (n=4) (Global/SH/Economy/IB)  
- Quién:  
  • Flota A333: NPS –38.5 (Global/SH/Economy/IB)  
  • Business/Work: NPS 17.5 (Global/SH/Economy/IB)  
- Evidencia COMPLETA:  
  • Explanatory Driver: anomalía de –0.28 pts (NPS 33.07 vs baseline 33.36) (Global/SH/Economy/IB)  
  • Load Factor 88.45 (–1.11 pts vs media) (Global/SH/Economy/IB)  
  • OTP15_adjusted 92.93 (+1.44 pts vs media) (Global/SH/Economy/IB)  
  • Incidentes NCS: 0 (Global/SH/Economy/IB)  
  • Verbatims: elogian puntualidad y limpieza, sin quejas operativas (Global/SH/Economy/IB)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
El 05-12-2025 el NPS Global subió 3.91 puntos, pasando de 26.47 a 30.39, gracias al fuerte repunte de Long Haul (+11.29 pts, de 11.71 a 22.99) y pese a la ligera merma de Short Haul (–0.14 pts, de 33.70 a 33.56). En Long Haul, Economy escaló 14.40 puntos (de 11.26 a 25.66) y Business mejoró 17.10 puntos (de 11.47 a 28.57), impulsados por feedback de clientes excelente y rutas clave como MAD–NEXTO (100.0, Global/LH/Business) y EZE–MAD (19.2, Global/LH/Economy). Contrarrestó parcialmente Premium, que cayó 30.60 pts (de 16.31 a –14.29) por la insatisfacción de clientes Business/Work en MAD–MEX (–40.0, Global/LH/Premium) y la flota A350 clásico (–28.6, Global/LH/Premium). En Short Haul, Business retrocedió 6.85 pts (de 34.51 a 27.66) por bajas valoraciones de clientes de América Central (–33.3, Global/SH/Business/YW) y flota A320 (–33.3, Global/SH/Business/IB), mientras que Economy se mantuvo estable con 34.28 pts (+0.72 vs L7d), compensada por YW (+2.51 pts, Global/SH/Economy/YW) y la leve caída de IB (–0.28 pts, Global/SH/Economy/IB).

Las rutas más afectadas fueron MAD–MEX (100.0 en Global/LH/Business vs –40.0 en Global/LH/Premium), EZE–MAD (19.2 en Global/LH/Economy), AGP–MAD (–33.3 en Global/SH/Business/YW) y MAD–PRG (25.0 en Global/SH/Economy/IB). Los perfiles más reactivos incluyeron pasajeros de América Central (–33.3 pts, Global/SH/Business/YW), clientes Business/Work (–40.0 pts, Global/LH/Premium), residentes en España (+36.4 pts, Global/LH/Business) y variaciones por flota —A350 Next (+62.5 pts, Global/LH/Business) frente a A350 clásico (–28.6 pts, Global/LH/Premium) y A320 (–33.3 pts, Global/SH/Business/IB).

ECONOMY SH: Equilibrio IB–YW  
La cabina Economy de SH registró un NPS de 34.28 puntos (05-12-2025), con un alza de 0.72 puntos vs L7d. IB cayó a 33.07 (–0.28 pts vs L7d) por un factor de carga bajo (LF 88.45, –1.11 pts vs L7d) y la ruta MAD–PRG (NPS 25.0, n=4, Global/SH/Economy/IB), mientras que YW subió a 36.57 (+2.51 pts vs L7d) gracias a la puntualidad (OTP 92.93, +1.44 pts vs L7d) y feedback de clientes positivo. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción.

BUSINESS SH: Caída por flota A320 y América Central  
El segmento Business de SH registró un NPS de 27.66 puntos (05-12-2025), con una caída de 6.85 puntos vs L7d. IB descendió a 30.56 (–3.84 pts vs L7d) y YW a 18.18 (–18.88 pts vs L7d). La evolución responde a la insatisfacción en la ruta AGP–MAD (NPS –33.3, n=3, Global/SH/Business/YW), la valoración de flota A320 (–33.3 pts, Global/SH/Business/IB) y pasajeros de América Central (–33.3 pts, Global/SH/Business/YW), pese a buena OTP 92.30 (+2.57 pts vs L7d) y feedback de clientes sobre limpieza y amabilidad.

ECONOMY LH: Impulso por trato y confort  
La cabina Economy de LH alcanzó un NPS de 25.66 puntos (05-12-2025), con un alza de 14.40 puntos vs L7d. A pesar de menor ocupación (LF 87.53, –2.63 pts vs L7d) y OTP reducida (74.54, –3.86 pts vs L7d), el feedback de clientes destacó amabilidad y confort. La ruta MAD–NRT obtuvo 20.0 (n=5, Global/LH/Economy) y persisten oportunidades por flota A33ACMI (–20.0 pts) y residentes en Europa (–27.3 pts).

BUSINESS LH: Fuerte recuperación en MAD–MEX y A350 Next  
La cabina Business de LH subió a un NPS de 28.57 puntos (05-12-2025), ganando 17.10 puntos vs L7d. Los drivers fueron la ruta MAD–MEX (100.0, n=3, Global/LH/Business), el excelente desempeño de la flota A350 Next (+62.5 pts) y el trato Premium destacado en feedback de clientes. La flota A333 (–25.0 pts) y pasajeros de Europa (–20.0 pts) moderaron parcialmente esta subida.

PREMIUM LH: Caída acusada por Business/Work en MAD–MEX  
El segmento Premium de LH cayó a –14.29 puntos (05-12-2025), con un descenso de 30.60 puntos vs L7d. La insatisfacción de clientes Business/Work en MAD–MEX (NPS –40.0, n=3, Global/LH/Premium), la flota A350 clásico (–28.6 pts) y la operación code share IB (–14.3 pts) dominaron la caída, pese a menciones positivas en puntualidad y entrega de equipaje en el feedback de clientes.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-04 to 2025-12-04:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Economy SH, el escenario es SINERGIA (–, – | –).  
- **Narrativa:** Adoptamos la Explicación del Nodo Padre (Global/SH/Economy): la anomalía de –8.6 pts se origina en valoraciones extremas negativas de subgrupos muy concretos, que arrastran el agregado pese al feedback mayoritariamente positivo.  
- **Evidencia Clave:** Ruta MAD–OVD NPS 0.0 (n=9); CodeShare VY NPS –60.0 (n=5); Fleet “Unknown” NPS –20.0 (n=5); Región América Norte NPS –20.0 (n=5).

En Business SH, el escenario es SINERGIA (–, – | –).  
- **Narrativa:** Adoptamos la Explicación del Nodo Padre (Global/SH/Business): la caída de –6.6 pts responde a la heterogeneidad de satisfacción entre perfiles y equipos, donde flotas y motivo de viaje presentan rangos contrapuestos que suman al efecto negativo.  
- **Evidencia Clave:** Business vs Leisure NPS 13.3 vs 35.7; Fleet A350 next NPS –33.3 y A320neo NPS 0.0; Región Sur NPS 100.0 vs grupo no identificado NPS –50.0.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul (SH), la dinámica es SINERGIA (–, – | –).  
- Narrativa: Adoptamos la explicación del Nodo Padre (Global/SH): la anomalía de –8.4 pts responde a drivers negativos comunes en Economy y Business SH –ruta BIO–MAD con NPS 0.0 (n=8) y bajas valoraciones en código compartido (VY NPS –60.0) y ciertas flotas (A350 next NPS –14.3)– que se suman para impactar todo el segmento SH.  
- Evidencia: Ruta BIO–MAD NPS 0.0; CodeShare VY NPS –60.0; Flota A350 next NPS –14.3.

En Long Haul (LH), la dinámica es CANCELACIÓN (+, –, – | N).  
- Narrativa: El radio LH muestra estabilidad engañosa: mientras Economy LH creció +7.6 pts impulsado por la calidad de servicio a bordo en BOG–MAD (NPS 42.9) y códigos AA (NPS 25.0), las caídas en Business LH (0.0 pts, arrastrada por Europa ex-España NPS –33.3 y A350 NPS –12.5) y Premium LH (0.0 pts, con quejas de atención de cabina y Leisure NPS –14.3) se neutralizan, resultando en un LH Normal.  
- Evidencia:  
  • Economy LH: BOG–MAD NPS 42.9 (n=14), CodeShare AA NPS 25.0, Fleet A350 next NPS 39.1.  
  • Business LH: Residence Europa ex-España NPS –33.3, Fleet A350 NPS –12.5.  
  • Premium LH: Leisure NPS –14.3, comentarios de falta de proactividad en cabina (n=8).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA (LH: Normal, SH: Negativa | Global: Negativa).  
- Narrativa: Adoptamos la explicación del Nodo Global: la caída de –3.7 pts se debe a la anomalía concentrada en Short Haul, que ha contagiado al Global pese a que Long Haul operó dentro de rangos normales.  
- Evidencia:
  • Ruta GUA–MAD NPS –33.3 (n=3)  
  • CodeShare VY y QR con NPS –60.0  
  • Fleet A350 C con NPS –16.7 (y A333 también en negativo)

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: DETERIORO EN SH/Economy  
- Escenario: SINERGIA (Global/SH/Economy/IB: –, Global/SH/Economy/YW: – | Global/SH/Economy: –)  
- NMA: Global/SH/Economy  
- Afecta a:  
  • Global/SH/Economy/IB  
  • Global/SH/Economy/YW  
- Qué falló: Valoraciones extremas negativas en rutas y alianzas code-share  
- Dónde:  
  • Ruta MAD–OVD: NPS 0.0 (n=9) (Global/SH/Economy)  
- Quién:  
  • CodeShare VY: NPS –60.0 (n=5) (Global/SH/Economy/IB)  
  • CodeShare Others: NPS –25.0 (n=4) (Global/SH/Economy/YW)  
  • Fleet “Unknown”: NPS –20.0 (n=5) (Global/SH/Economy/IB)  
  • Región América Norte: NPS –20.0 (n=5) (Global/SH/Economy)  
- Evidencia COMPLETA:  
  • NPS 25.0 vs baseline 33.5598 (Global/SH/Economy)  
  • Load Factor 85.76 (–0.69) (Global/SH/Economy)  
  • OTP15_adjusted 92.25 (+2.64) (Global/SH/Economy)  
  • Mishandling no relevante; NCS cero incidentes (Global/SH/Economy)  
  • 549 verbatims mayoritariamente positivos, sin quejas sistémicas (Global/SH/Economy)  

CAUSA 2: DETERIORO EN SH/Business  
- Escenario: SINERGIA (Global/SH/Business/IB: –, Global/SH/Business/YW: – | Global/SH/Business: –)  
- NMA: Global/SH/Business  
- Afecta a:  
  • Global/SH/Business/IB  
  • Global/SH/Business/YW  
- Qué falló: Heterogeneidad de satisfacción por tipo de flota y perfil de cliente  
- Dónde: no hay rutas con NPS negativo relevante (Global/SH/Business)  
- Quién:  
  • Fleet A350 next: NPS –33.3 (Global/SH/Business)  
  • Fleet A320neo: NPS 0.0 (Global/SH/Business)  
  • Residencia “No identificado”: NPS –50.0 (Global/SH/Business)  
- Evidencia COMPLETA:  
  • NPS 27.91 vs baseline 34.5134 (Global/SH/Business)  
  • Load Factor 68.45 (–4.15) (Global/SH/Business)  
  • OTP15_adjusted 92.25 (+2.64) (Global/SH/Business)  
  • NCS cero incidentes; 52 verbatims positivos sin quejas operativas significativas (Global/SH/Business)  

CAUSA 3: IMPULSO POSITIVO EN LH/Economy  
- Escenario: Leaf anomaly (solo Global/LH/Economy)  
- NMA: Global/LH/Economy  
- Afecta a: Global/LH/Economy  
- Qué falló (qué funcionó): Calidad de servicio y segmentos premium en Economy LH  
- Dónde:  
  • Ruta BOG–MAD: NPS 42.9 (n=14) (Global/LH/Economy)  
- Quién:  
  • Residencia Centroamérica: NPS 57.1 (Global/LH/Economy)  
  • CodeShare AA: NPS 25.0 (Global/LH/Economy)  
  • Fleet A350 next: NPS 39.1 (Global/LH/Economy)  
  • Viajeros de negocio: NPS 33.3 (Global/LH/Economy)  
- Evidencia COMPLETA:  
  • NPS 18.85 vs baseline 11.2587 (Global/LH/Economy)  
  • Load Factor 87.86 (–2.34) (Global/LH/Economy)  
  • OTP15_adjusted 74.67 (–3.70) (Global/LH/Economy)  
  • NCS cero incidentes; 176 verbatims destacados en catering y atención a bordo (Global/LH/Economy)  

CAUSA 4: DETERIORO EN LH/Business  
- Escenario: Leaf anomaly (solo Global/LH/Business)  
- NMA: Global/LH/Business  
- Afecta a: Global/LH/Business  
- Qué falló: Valoraciones bajas de pasajeros europeos y en flota A350  
- Dónde: no hay rutas con NPS negativo (Global/LH/Business)  
- Quién:  
  • Residencia Europa (ex-España): NPS –33.3 (Global/LH/Business)  
  • Fleet A350: NPS –12.5 (Global/LH/Business)  
  • Viajeros de ocio: NPS –9.1 (Global/LH/Business)  
- Evidencia COMPLETA:  
  • NPS 0.0 vs baseline 11.4687 (Global/LH/Business)  
  • Load Factor 93.96 (–0.22) (Global/LH/Business)  
  • OTP15_adjusted 74.67 (–3.70) (Global/LH/Business)  
  • NCS cero incidentes; 37 verbatims todos positivos sobre tripulación (Global/LH/Business)  

CAUSA 5: DETERIORO EN LH/Premium  
- Escenario: Leaf anomaly (solo Global/LH/Premium)  
- NMA: Global/LH/Premium  
- Afecta a: Global/LH/Premium  
- Qué falló: Calidad de atención en cabina percibida como poco proactiva  
- Dónde: no hay rutas con NPS suficiente para análisis (Global/LH/Premium)  
- Quién:  
  • Viajeros de ocio: NPS –14.3 (Global/LH/Premium)  
  • CodeShare IB: NPS 0.0 (Global/LH/Premium)  
  • Fleet A333: NPS 0.0 (Global/LH/Premium)  
- Evidencia COMPLETA:  
  • NPS 0.0 vs baseline 16.3119 (Global/LH/Premium)  
  • Load Factor 88.45 (–2.13) (Global/LH/Premium)  
  • OTP15_adjusted 74.67 (–3.70) (Global/LH/Premium)  
  • NCS cero incidentes; 8 verbatims con quejas sobre falta de proactividad del crew (Global/LH/Premium)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

Durante el análisis del 04-dic, el NPS global cayó de 26.47 a 22.79 (–3.68 pts), un descenso impulsado íntegramente por Short Haul (SH) pese a que Long Haul (LH) operó dentro de rangos normales. Las caídas más pronunciadas se dieron en Premium LH (16.31→0.00, –16.31 pts), Business LH (11.47→0.00, –11.47 pts), Economy SH (33.56→25.00, –8.56 pts) y Business SH (34.51→27.91, –6.60 pts), mientras que Economy LH destacó con una subida de 11.26→18.85 (+7.59 pts). Identificamos cinco causas raíz: deficiencias de atención en Premium LH (Global/LH/Premium), insatisfacción de pasajeros europeos y flota A350 en Business LH (Global/LH/Business), calidad de servicio de Economy LH (Global/LH/Economy), y valoraciones extremas en rutas y code-shares en Economy SH (Global/SH/Economy) y heterogeneidad de flotas y perfiles en Business SH (Global/SH/Business).

Las rutas más afectadas fueron MAD–OVD con NPS 0.0 (n=9) en Global/SH/Economy, BIO–MAD con NPS 0.0 (n=8) en Global/SH, y GUA–MAD con NPS –33.3 (n=3) en Global/SH. Por el contrario, BOG–MAD concentró la mejora de Economy LH con NPS 42.9 (n=14). Entre perfiles, destacan los pasajeros de code-share VY (NPS –60.0 en Global/SH/Economy), residentes en Europa (ex-España) con NPS –33.3 en Global/LH/Business, y viajeros de Centroamérica con NPS 57.1 en Global/LH/Economy.

ECONOMY SH (IB y YW): Caída impulsada por valoraciones extremas en rutas y code-share  
La cabina Economy de SH IB registró un NPS de 25.17 pts (04-dic) vs 33.36 de L7d, bajando 8.18 pts, mientras que YW cayó de 34.06 a 24.62 (–9.45 pts). En conjunto Economy SH pasó de 33.56 a 25.00 (–8.56 pts). La causa principal fue la percepción neutra en la ruta MAD–OVD (NPS 0.0, n=9, Global/SH/Economy) y las bajas calificaciones en CodeShare VY (NPS –60.0, n=5, Global/SH/Economy/IB) y flota “Unknown” (NPS –20.0, n=5, Global/SH/Economy/IB), complementada por la insatisfacción de residentes en América Norte (NPS –20.0, n=5, Global/SH/Economy). A nivel de datos operativos, Load Factor 85.76 (–0.69) y OTP 92.25 (+2.64) no mostraron correlación, y no se registraron incidentes NCS ni quejas recurrentes en el feedback de clientes.

BUSINESS SH (IB y YW): Impacto heterogéneo por flota y perfil de cliente  
El segmento Business de SH IB retrocedió de 34.40 a 28.57 (–5.83 pts), mientras que YW descendió de 37.07 a 25.00 (–12.07 pts), resultando en un NPS agregado de 27.91 vs 34.51 (–6.60 pts). Esta bajada se explica por la insatisfacción en flota A350 next (NPS –33.3, Global/SH/Business) y A320neo (NPS 0.0, Global/SH/Business), y la baja valoración de un grupo no identificado de residencia (NPS –50.0, Global/SH/Business). Load Factor 68.45 (–4.15) y OTP 92.25 (+2.64) no apuntan a problemas operativos críticos, no hubo incidentes NCS, y el feedback de clientes fue mayoritariamente positivo sobre puntualidad y tripulación.

ECONOMY LH: Repunte gracias a servicio a bordo y segmentos específicos  
La cabina Economy de LH subió de 11.26 a 18.85 (+7.59 pts). El alza se cimentó en la ruta BOG–MAD con NPS 42.9 (n=14, Global/LH/Economy), el alto rating de code-share AA (NPS 25.0, Global/LH/Economy), y la flota A350 next (NPS 39.1, Global/LH/Economy). También impulsaron la mejora los viajeros de Centroamérica (NPS 57.1, Global/LH/Economy) y de negocio (NPS 33.3, Global/LH/Economy), a pesar de Load Factor 87.86 (–2.34) y OTP 74.67 (–3.70). No se registraron incidentes NCS y el feedback de clientes destacó catering y atención de auxiliares.

BUSINESS LH: Deterioro centrado en pasajeros de Europa y flota A350  
La cabina Business de LH cayó de 11.47 a 0.00 (–11.47 pts). El deterioro se concentra en los residentes en Europa (ex-España) con NPS –33.3 (Global/LH/Business), las valoraciones en flota A350 (NPS –12.5, Global/LH/Business) y los viajeros de ocio (NPS –9.1, Global/LH/Business). Load Factor 93.96 (–0.22) y OTP 74.67 (–3.70) estuvieron por debajo de L7d, sin incidentes NCS y con 37 verbatims elogiando la tripulación, lo que sugiere que los drivers son puramente de perfil y expectativas de servicio.

PREMIUM LH: Colapso por quejas de atención en cabina de Leisure  
Premium LH descendió de 16.31 a 0.00 (–16.31 pts) debido a quejas de falta de proactividad de la tripulación (8 verbatims, Global/LH/Premium) y al pobre desempeño de Leisure (NPS –14.3, Global/LH/Premium). Además, CodeShare IB y flota A333 quedaron en NPS 0.0 (Global/LH/Premium). A pesar de Load Factor 88.45 (–2.13) y OTP 74.67 (–3.70), y sin incidentes NCS, la percepción de servicio en cabina fue determinante en la caída de NPS.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-03 to 2025-12-03:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Economy SH, el escenario es TRANSFERENCIA (IB Normal, YW + | Economy SH +).  
- Narrativa: La subida de +7.5 pts en Economy SH se debe íntegramente al rendimiento excepcional de YW, que contagia al agregado pese a la estabilidad de IB. No hay efecto de IB que modere o compense esta mejora.  
- Evidencia Clave: Global/SH/Economy/YW registró +11.9 pts impulsados por OTP15_adjusted +3.81 pts y feedback muy positivo de la tripulación y limpieza, mientras que IB se mantuvo “Normal” (dentro de rango).

En Business SH, el escenario es TRANSFERENCIA (IB Normal, YW + | Business SH +).  
- Narrativa: El alza de +11.2 pts en Business SH proviene totalmente de la anomalía positiva de YW, que arrastra el promedio del segmento. IB se mantuvo estable y no influyó en la dirección del cambio.  
- Evidencia Clave: Global/SH/Business/YW alcanzó +29.6 pts gracias a OTP15_adjusted +3.81 pts y valoraciones sobresalientes de puntualidad y tripulación, mientras que IB permaneció “Normal” con +4.1 pts (dentro de rango).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es SINERGIA (+,+ | +).  
- Narrativa: El repunte de NPS en Short Haul (+7.8 pts) es un fenómeno sistémico que afecta de forma conjunta a Economy y Business. La mejora se explica por una operación más puntual (OTP15_adjusted +2.97 pts) y un servicio de tripulación extraordinario, percibido positivamente tanto por pasajeros de ocio como de negocios.  
- Evidencia: Global/SH registró OTP15_adjusted 92.5 (+2.97 pts) y verbatims destacan trato personalizado y proactividad de la tripulación.

En Long Haul, la dinámica es SINERGIA (-,-,- | -).  
- Narrativa: La caída de NPS en Long Haul (–13.2 pts) es un problema transversal que impactó a todas las cabinas. Se concentra en la insatisfacción de viajeros de negocio (Business/Work NPS –38.5), especialmente en vuelos codeshare con QR (–100.0) y BA (–28.6), así como en flotas A350 y A350 next.  
- Evidencia: Global/LH muestra Business/Work con NPS –38.5 (n=26), y flota A350/A350 next en negativo, sin incidencias operativas que expliquen fallos puntuales.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es DOMINANCIA (LH –, SH + | Global +).  
- Narrativa: El ligero alza de +0.4 pts en el NPS Global está arrastrado por el excelente desempeño de Short Haul, que supera con creces la caída en Long Haul. A pesar de la insatisfacción marcada en viajeros de negocio de largo radio, la puntualidad sobresaliente y la calidad del servicio en corto radio dominan el resultado agregado.  
- Evidencia: Global/SH registró OTP15_adjusted 92.5 (+2.97 pts) y verbatims destacan trato personalizado y proactividad de la tripulación, mientras que Global/LH cayó –13.2 pts por NPS –38.5 en Business/Work y flotas A350 sin incidencias operativas.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: EXCELENCIA OPERACIONAL Y CALIDAD DE SERVICIO EN CORTO RADIO  
- Escenario: DOMINANCIA (LH –, SH + | Global +)  
- NMA: Global/SH  
- Afecta a:  
  • Global/SH/Economy  
  • Global/SH/Business  
  • Global/SH/Economy/YW  
  • Global/SH/Business/YW  
- Qué falló (o mejoró):  
  • Mejora de puntualidad (OTP15_adjusted +2.97 pts en Global/SH)  
  • Servicio de tripulación altamente valorado (feedback cualitativo unánimemente positivo en Global/SH)  
- Dónde (Top rutas):  
  1. LHR–MAD → NPS 22.2 (n=18) (Global/SH)  
  2. MAD–TLS → NPS 20.0 (n=5) (Global/SH/Economy)  
  3. MAD–OPO → NPS 60.0 (n=5) (Global/SH/Economy/YW)  
- Quién (Perfiles más reactivos):  
  • Leisure vs Business: 44.3 vs 35.7 pts (Global/SH)  
  • Residence Region: Europa 55.6 vs Asia/África –50.0 (Global/SH)  
  • CodeShare: I2 100.0, AA 80.0, Others –60.0 (Global/SH)  
- Evidencia COMPLETA:  
  • NPS 41.5 vs baseline 33.70 (+7.8 pts) (Global/SH)  
  • OTP15_adjusted 92.5 (+2.97 pts vs 89.53) (Global/SH)  
  • Load Factor 84.47 (–0.81 pts vs 85.28) (Global/SH)  
  • Incidentes NCS: 0 (Global/SH)  
  • Verbatims con tono muy positivo (Global/SH): trato personalizado, proactividad y limpieza a bordo  

---  

CAUSA 2: INSATISFACCIÓN DE VIAJEROS DE NEGOCIO EN LARGO RADIO  
- Escenario: SINERGIA (–,–,– | –)  
- NMA: Global/LH  
- Afecta a:  
  • Global/LH/Economy  
  • Global/LH/Business  
  • Global/LH/Premium  
- Qué falló:  
  • Elevada insatisfacción de Business/Work travellers (Global/LH/Business) por expectativas de producto no cubiertas  
- Dónde (Top rutas):  
  1. MAD–SDQ → NPS 12.5 (n=8) (Global/LH/Economy)  
  2. EZE–MAD → NPS 25.0 (n=4) (Global/LH/Business)  
  3. BOG–MAD → NPS 40.0 (n=5) (Global/LH/Premium)  
- Quién (Perfiles más reactivos):  
  • Business/Work: NPS –38.5 (n=26) vs Leisure 3.9 (n=180) (Global/LH/Business)  
  • CodeShare QR: –100.0 (Global/LH/Economy y Global/LH/Business), BA: –28.6 (Global/LH/Business)  
  • Residence Region: Europa –37.5, América Sur –21.3 (Global/LH/Business)  
  • Fleet:  
    – A350 y A350 next en negativo (–22.2 en Global/LH/Economy; –28.6 en Global/LH/Business; –20.0 en Global/LH/Premium)  
    – A321XLR y A321 con NPS positivo (Global/LH/Economy)  
- Evidencia COMPLETA:  
  • NPS –1.46 vs baseline 11.71 (–13.17 pts) (Global/LH)  
  • OTP15_adjusted 74.97 (–3.69 pts vs 78.66) (Global/LH)  
  • Load Factor 88.71 (–1.90 pts vs 90.61) (Global/LH)  
  • Incidentes NCS: 0 (Global/LH)  
  • Verbatims (n=325) muestran feedback mayoritariamente positivo en servicio a bordo, sin quejas operativas que justifiquen la caída — apunta a gap de expectativas de Business.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
En el análisis del 3 de diciembre de 2025, el NPS Global experimentó una subida neta de 0.43 puntos, pasando de 26.47 a 26.90, producto de dos fuerzas opuestas: un desplome de –13.17 puntos en Long Haul (Global/LH cayó de 11.71 a –1.46) y un repunte de +7.8 puntos en Short Haul (Global/SH subió de 33.70 a 41.50). La caída en Long Haul se concentró en Global/LH/Economy (de 11.26 a 0.0, –11.26), Global/LH/Business (de 11.47 a –5.56, –17.03) y Global/LH/Premium (de 16.31 a –8.70, –25.01), impulsada por la insatisfacción de Business/Work en flotas A350 y A350 next (NPS –38.5 en Global/LH/Business) y vuelos code-share QR (–100.0) y BA (–28.6). En contraste, Short Haul se benefició de la excelencia operacional (OTP 92.5 en Global/SH) y un servicio de tripulación muy valorado, con subidas notables en Global/SH/Economy/YW (de 34.06 a 45.92, +11.86) y Global/SH/Business/YW (de 37.07 a 66.67, +29.60).

Entre las rutas más afectadas se cuentan MAD–SDQ con NPS 12.5 en Global/LH/Economy, LHR–MAD con NPS 22.2 en Global/SH, y MAD–OPO con NPS 60.0 en Global/SH/Economy/YW. Los perfiles más reactivos fueron los viajeros Leisure (43.9 vs 35.3 en Global/SH/Economy), los Business/Work de Long Haul (–38.5 en Global/LH/Business) y los residentes en Europa (55.6 en Global/SH vs –37.5 en Global/LH), así como pasajeros de América del Sur en Premium LH (–80.0).

ECONOMY SH YW: Empuje de YW impulsa subida  
La cabina Economy SH brilló durante el día 2025-12-03 con un NPS de 41.10, subiendo 7.54 puntos frente a la media de 33.56 de los últimos 7 días. YW lideró el avance (de 34.06 a 45.92, +11.86) gracias a un OTP de 91.77 (+3.81 pts vs L7d en Global/SH/Economy/YW) y un feedback de clientes que resaltó trato personalizado y limpieza en 122 verbatims. IB mantuvo desempeño estable (de 33.36 a 39.33, +6.0 ppts según Explanatory Drivers, dentro de rango). El alza se reflejó especialmente en MAD–OPO (NPS 60.0, n=5) y los perfiles Leisure (43.9 vs Business 35.3), sin incidentes NCS reportados.

BUSINESS SH YW: YW arrastra al segmento  
El segmento Business SH alcanzó un NPS de 45.71 el 2025-12-03, mejorando 11.20 puntos desde 34.51 en L7d. YW fue la causa principal (de 37.07 a 66.67, +29.60) impulsado por OTP 92.5 (+2.97 pts vs L7d en Global/SH/Business/YW) y un feedback de tripulación caracterizado por amabilidad y agilidad en 12 verbatims. IB mantuvo estabilidad con 38.46 (de 34.40 a 38.46, +4.06 ppts dentro de rango). No hubo incidentes NCS y el entusiasmo destacó en code-shares I2 (100.0) y AA (80.0).

ECONOMY LH: Business en Economy penaliza NPS  
La cabina Economy LH sufrió un desplome de 11.26 puntos, cayendo a 0.0 frente a 11.26 de L7d. El factor clave fue la insatisfacción de Business/Work viajeros en Economy (NPS –30.0, n=20 en Global/LH/Economy), potenciada por flotas A350 con NPS –22.2 y vuelos code-share QR (–100.0) y BA (–33.3). MAD–SDQ mostró NPS 12.5 (n=8), OTP de 74.97 (–3.69 pts vs L7d en Global/LH) y ausencia de incidentes NCS. Los Leisure mantuvieron NPS 4.1.

BUSINESS LH: Experiencias en flota A333 y A350 next lastran  
Business LH cayó 17.02 puntos, de 11.47 a –5.56. Los drivers fueron Business/Work (NPS –50.0, n=4 en Global/LH/Business), flota A333 (–100.0) y A350 next (–28.6). OTP 74.97 (–3.69 pts vs L7d en Global/LH) convivió con feedback mayoritariamente positivo sobre servicio, sin incidentes NCS. EZE–MAD registró NPS 25.0 (n=4) y los residentes en América Central estuvieron en –50.0.

PREMIUM LH: Dispersión por región y flota hunde NPS  
Premium LH registró una bajada de 25.01 puntos, de 16.31 a –8.70. La insatisfacción vino de América del Sur (NPS –80.0, Global/LH/Premium) y Europa (–40.0), sumada a flotas A350 (–22.2) y A350 next (–20.0). El A333 mantuvo NPS 50.0. BOG–MAD entregó NPS 40.0 (n=5), OTP de 74.97 (–3.69 pts vs L7d) y sin incidentes NCS.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-02 to 2025-12-02:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Economy SH, el escenario es TRANSFERENCIA (Normal IB, Negativa YW | Negativa Economy SH).  
- Narrativa: Adopta la explicación del nodo padre (Economy SH): la caída de 3.3 pts se explica principalmente por el subsegmento Economy YW, cuyo bajo nivel de satisfacción arrastra la media del segmento pese a la estabilidad de IB.  
- Evidencia Clave: YW –12.3 pts en Global/SH/Economy/YW (NPS 21.71 vs baseline 34.06) en ruta MAD–SXB (n=3), pasajeros CodeShare AA.

En Business SH, el escenario es TRANSFERENCIA (Normal IB, Positiva YW | Positiva Business SH).  
- Narrativa: Adopta la explicación del nodo padre (Business SH): la subida de 9.3 pts obedece al fuerte impulso del subsegmento Business YW, cuyas mejoras operativas y feedback muy positivo contagian al agregado.  
- Evidencia Clave: YW +15.9 pts en Global/SH/Business/YW (NPS 52.94 vs baseline 37.07), OTP15_adjusted +3.38 pts, Load Factor –3.88 pts y verbatims de alta satisfacción.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es DOMINANCIA (Economy SH –, Business SH + | SH –2.1 pts).  
- Narrativa: El rendimiento de SH está dictado por la caída de Economy SH, cuyo bajo NPS arrastra el agregado a terreno negativo, a pesar de la mejora en Business SH que atenúa parcialmente la pérdida.  
- Evidencia: Economy SH –3.3 pts impulsado por Economy YW –12.3 pts en Global/SH/Economy/YW (NPS 21.71 vs baseline 34.06) en ruta MAD–SCQ (n=4) y pasajeros CodeShare AA.

En Long Haul, la dinámica es SINERGIA (Economy LH N, Business LH +, Premium LH + | LH +7.8 pts).  
- Narrativa: La subida de +7.8 pts en LH se explica por la ausencia de incidentes formales y un feedback muy positivo a bordo, que empuja tanto a Business como a Premium, sin que la estabilidad de Economy atenúe el impulso.  
- Evidencia: NCS=0 en Global/LH, verbatims mayoritariamente positivos (“viaje placentero”, “personal atento”), Premium LH +32.0 pts (LIM–MAD NPS 100.0, n=3).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es DOMINANCIA (+, – | +) `(LH +7.8, SH –2.1 | Global +1.3)`.  
- Narrativa: El resultado positivo de +1.3 pts a nivel global está arrastrado por el fuerte impulso de Long Haul, que compensó la caída de Short Haul.  
- Evidencia: Premium LH +32.0 pts (Global/LH/Premium), impulsado por la ruta LIM–MAD (NPS 100.0, n=3) sin incidentes NCS y con verbatims muy positivos.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Impulso Positivo en Long Haul  
- Escenario: SINERGIA (+ en Business LH, + en Premium LH | + en LH)  
- NMA: Global/Long Haul  
- Afecta a:  
  • Global/LH/Business  
  • Global/LH/Premium  

- Qué falló (en realidad “qué triunfó”):  
  • Ausencia de incidentes formales (NCS=0 en Global/LH)  
  • Feedback muy positivo de clientes  

- Dónde (rutas más impactadas):  
  • Global/LH/Premium en ruta LIM–MAD: NPS 100.0 (n=3)  
  • Global/LH/Business en ruta MAD–MIA: NPS 100.0 (n=3)  

- Quién (perfiles más reactivos):  
  • Flota A350 C (Global/LH/Business): NPS 100.0  
  • Residents España (Global/LH/Business): NPS 53.3  
  • Flota A350 (Global/LH/Premium): NPS 75.0  
  • Flota A350 next (Global/LH/Premium): NPS 70.0  

- Evidencia COMPLETA:  
  • NPS Global/LH: 19.56 vs baseline 11.71 → +7.85 pts  
  • OTP15_adjusted Global/LH: 76.06 (–2.84 pts vs baseline)  
  • Load Factor Global/LH: 88.33 (–2.30 pts vs baseline)  
  • Mishandling y misconexiones no ejercieron impacto (no disponibles incidentes NCS)  
  • Verbatims (n=447) con menciones repetidas a “viaje placentero”, “personal atento”, “eficiente abordaje”

―――

CAUSA 2: Insatisfacción en Economy/YW de Short Haul  
- Escenario: TRANSFERENCIA (Economy SH normal en IB, Economy YW negativa | Economy SH negativa)  
- NMA: Global/SH/Economy/YW  
- Afecta a:  
  • Global/SH/Economy/YW  

- Qué falló:  
  • Nivel de satisfacción muy bajo en YW (NPS 21.71 vs baseline 34.06 → –12.35 pts)  

- Dónde (ruta más impactada):  
  • MAD–SXB: NPS 0.0 (n=3)  

- Quién (perfiles más reactivos):  
  • Pasajeros CodeShare AA en YW (NPS 0.0, n=4)  

- Evidencia COMPLETA:  
  • NPS Global/SH/Economy/YW: 21.71 vs baseline 34.06 → –12.35 pts  
  • OTP15_adjusted Global/SH/Economy/YW: 91.19 (+3.38 pts vs baseline)  
  • Load Factor Global/SH/Economy/YW: 81.02 (–0.11 pts vs baseline)  
  • Incidentes NCS=0  
  • Verbatims (n=231) sin quejas operativas, lo que sugiere problemas de consistencia en servicio AA YW (cabina, comunicación, equipamiento)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El NPS global registró una subida moderada de 1.3 puntos, pasando de 26.47 (L7d) a 27.79 el 2025-12-02, gracias al fuerte impulso de Long Haul (de 11.71 a 19.56, +7.85 pts) que compensó la caída de Short Haul (de 33.70 a 31.58, –2.12 pts). En Long Haul, Business LH subió 9.58 pts (de 11.47 a 21.05) y Premium LH escaló 31.96 pts (de 16.31 a 48.28), sustentados en OTP levemente inferiores (OTP 76.06, –2.84 pts vs L7d) pero sin incidentes NCS y un feedback de clientes excepcionalmente positivo. Por su parte, Short Haul mostró una bajada de 3.30 pts en Economy SH (de 33.56 a 30.26), empujada por Economy YW que cayó 12.35 pts (de 34.06 a 21.71) en vuelos CodeShare AA, pese al buen comportamiento de Economy IB (+1.10 pts). Business SH, en cambio, mejoró 9.35 pts (de 34.51 a 43.86), impulsada especialmente por Business YW (+15.88 pts) gracias a alta puntualidad (OTP +3.38 pts) y baja ocupación (Load Factor 65.61 vs 69.49).

Las rutas más impactadas fueron LIM–MAD (Premium LH con NPS 100.0, n=3) y MAD–MIA (Business LH con NPS 100.0, n=3) en Long Haul; y MAD–SXB (Economy SH YW con NPS 0.0, n=3) en Short Haul. Entre perfiles, destacan los pasajeros en flota A350 C (Business LH NPS 100.0), residentes en España (Business LH NPS 53.3), y usuarios de CodeShare AA en Economy YW (NPS 0.0), que muestran la mayor volatilidad de satisfacción.

ECONOMY SH YW: Punto crítico  
La cabina Economy SH YW registró un NPS de 21.71 el 2025-12-02 (vs 34.06 L7d), con una caída de 12.35 pts respecto a la semana anterior. Esta merma se explica por la experiencia de pasajeros CodeShare AA (NPS 0.0 en ruta MAD–SXB, n=3), pese a que OTP 91.19 (+3.38 pts vs L7d) y la puntualidad mejoraron, y no se reportaron incidentes NCS. El factor load factor se mantuvo estable en 81.02 (–0.11 pts vs L7d), lo que apunta a brechas en consistencia de servicio más que a problemas operativos.

BUSINESS SH YW: Subida contundente  
El segmento Business SH YW escaló su NPS de 37.07 (L7d) a 52.94 el 2025-12-02, mejorando 15.88 pts. Esta subida refleja la alta puntualidad (OTP +3.38 pts vs L7d), la menor densidad a bordo (Load Factor 65.61 vs 69.49 L7d) y un feedback de clientes muy positivo (“excelente atención” en verbatims). La ruta AMS–MAD destacó con NPS 75.0 (n=4) y perfiles como residentes en España alcanzaron NPS 72.0, confirmando la solidez del servicio de Business SH YW.

ECONOMY LH: Desempeño estable  
La cabina Economy LH mantuvo desempeño estable, con un NPS de 15.20 el 2025-12-02 (vs 11.26 L7d), mejorando 3.94 pts dentro de un rango normal. No se detectaron cambios significativos en OTP (76.06, –2.84 pts vs L7d), load factor (88.33, –2.30 pts vs L7d) ni en feedback de clientes, lo que indica consistencia en la experiencia de Economy LH.

BUSINESS LH: Mejora destacada  
Business LH elevó su NPS de 11.47 (L7d) a 21.05 el 2025-12-02, con un avance de 9.58 pts. La clave fue la ausencia de incidentes NCS y el feedback muy positivo de la ruta MAD–MIA (NPS 100.0, n=3), junto a OTP 76.06 (–2.84 pts vs L7d) y un load factor de 93.83 (–0.21 pts vs L7d). La flota A350 C lideró con NPS 100.0 y residentes en España mostraron 53.3 pts, confirmando la fortaleza de Business LH.

PREMIUM LH: Auge excepcional  
Premium LH experimentó una subida de 31.96 pts, pasando de 16.31 (L7d) a 48.28 el 2025-12-02. Este salto se sustenta en la ruta LIM–MAD con NPS 100.0 (n=3), sin incidentes NCS, y un feedback de clientes sobresaliente. A350 next alcanzó NPS 70.0 y A350 común 75.0, aunque el tamaño de muestra reducido sugiere un posible sesgo de respuesta.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-01 to 2025-12-01:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Economy SH, el escenario es SINERGIA (IB –, YW – | PADRE –).  
- Narrativa: Adoptamos la explicación del nodo padre. La caída de –9.04 pts en Global/SH/Economy refleja una insatisfacción concentrada en la ruta MAD–OPO, donde ambos subsegmentos (IB y YW) registraron NPS nulo y arrastraron el agregado.  
- Evidencia Clave: Global/SH/Economy –9.04 pts, impulsado por IB –9.8 pts y YW –8.0 pts en MAD–OPO, especialmente clientes de América Norte y codeshares AA/I2.  

En Business SH, el escenario es CANCELACIÓN (IB –, YW + | PADRE N).  
- Narrativa: Ignoramos la explicación del padre, ya que los efectos opuestos de los hijos se neutralizan. Por un lado, IB sufrió una caída por insatisfacción de viajeros Business/Work en MAD–ORY; por otro, YW experimentó un fuerte repunte gracias a mejoras en puntualidad y servicio, balanceando el NPS agregado.  
- Evidencia Clave: Global/SH/Business/IB –8.08 pts (baja satisfacción en MAD–ORY, Business/Work) vs Global/SH/Business/YW +18.49 pts (OTP +3.67 pts y feedback muy positivo de tripulación e instalaciones).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es TRANSFERENCIA (Economy –, Business N | SH –8.3 pts).  
- Narrativa: La caída de –8.3 pts en Global/SH se impone desde el subsegmento Economy. El resto (Business SH) se mantuvo dentro de lo esperado y no logró contrarrestar la anomalía.  
- Evidencia: Economy SH –9.0 pts, concentrado en la ruta BRU–MAD (NPS 0.0, n=6) con detractores en codeshare AA/I2 y residentes de América Norte.

En Long Haul, la dinámica es DOMINANCIA (Economy –, Business –, Premium + | LH –0.7 pts).  
- Narrativa: El descenso de –0.7 pts en Global/LH está dictado por la fuerte anomalía negativa de Business LH, mientras que el alza en Premium LH suavizó parcialmente el impacto.  
- Evidencia: Business LH –16.5 pts en la ruta MAD–MEX (quejas de actitud de tripulación, catering y entretenimiento); Premium LH +46.2 pts impulsado por puntualidad y servicio excepcional en MAD–MEX.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es SINERGIA (LH –, SH – | GLOBAL –).  
- Narrativa: Adoptamos la explicación del nodo Global. La red entera mostró una caída de –6.5 pts en el NPS impulsada por un grupo muy reducido de detractores en la ruta MAD–VGO, cuyos comentarios negativos arrastraron tanto el Largo como el Corto Radio.  
- Evidencia: Global –6.5 pts, concentrado en MAD–VGO (NPS –33.3, n=3), principalmente clientes Business en flota A350 C/A321XLR y en códigos compartidos QR e I2.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Detractores en ruta MAD–VGO  
- Escenario: SINERGIA (LH –, SH – | GLOBAL –)  
- NMA: Global  
- Afecta a: Global/LH, Global/SH y todas sus cabinas  
- Qué falló: Grupo muy reducido de detractores corporativos (Business) asociadas a flota A350 C y A321XLR y códigos compartidos QR e I2  
- Dónde: Ruta MAD–VGO, NPS –33.3 pts (n=3) (Global)  
- Quién: Residence Europa & América Norte, Business (Global)  
- Evidencia COMPLETA:  
  • NPS 19.96 vs baseline 26.47 (–6.51 pts, Global)  
  • Load Factor 85.88 (–0.89 vs media 7 días, Global)  
  • OTP15_adjusted 90.45 (+2.38 vs media, Global)  
  • NCS: 0 incidentes (Global)  
  • 788 verbatims mayoritariamente positivos salvo en MAD–VGO (Global)  

CAUSA 2: Insatisfacción en Economy Short Haul  
- Escenario: SINERGIA (IB –, YW – | SH/Economy –)  
- NMA: Global/SH/Economy  
- Afecta a: Global/SH/Economy, Global/SH/Economy/IB y Global/SH/Economy/YW  
- Qué falló: Valoraciones muy bajas de pasajeros en código compartido AA e I2, especialmente residentes de América Norte  
- Dónde: Ruta MAD–OPO, NPS 0.0 (n=6) (Global/SH/Economy)  
- Quién: CodeShare AA (–60.0 pts) e I2 (–33.3 pts), Residence América Norte (–37.5 pts) (Global/SH/Economy)  
- Evidencia COMPLETA:  
  • NPS 24.52 vs baseline 33.56 (–9.04 pts, Global/SH/Economy)  
  • Load Factor 86.46 (–0.14 vs media, Global/SH/Economy)  
  • OTP15_adjusted 92.30 (+2.87 vs media, Global/SH/Economy)  
  • NCS: 0 incidentes (Global/SH/Economy)  
  • 411 verbatims muy positivos fuera de MAD–OPO (Global/SH/Economy)  

CAUSA 3: Insatisfacción en Business Short Haul (Iberia IB)  
- Escenario: CANCELACIÓN (IB –, YW + | SH/Business N)  
- NMA: Global/SH/Business/IB  
- Afecta a: Global/SH/Business/IB  
- Qué falló: Percepción de falta de profesionalidad de la tripulación y confort en ruta MAD–ORY  
- Dónde: Ruta MAD–ORY, NPS 25.0 (n=4) (Global/SH/Business/IB)  
- Quién: Business/Work, NPS –28.6 pts (7 encuestas) (Global/SH/Business/IB)  
- Evidencia COMPLETA:  
  • NPS 26.32 vs baseline 34.40 (–8.08 pts, Global/SH/Business/IB)  
  • Load Factor 78.17 (–2.75 vs media, Global/SH/Business/IB)  
  • OTP15_adjusted 93.23 (+1.83 vs media, Global/SH/Business/IB)  
  • NCS: 0 incidentes (Global/SH/Business/IB)  
  • 25 verbatims positivos sin mención de fallos generales (Global/SH/Business/IB)  

CAUSA 4: Mejora en Business Short Haul (Low Cost YW)  
- Escenario: CANCELACIÓN (IB –, YW + | SH/Business N)  
- NMA: Global/SH/Business/YW  
- Afecta a: Global/SH/Business/YW  
- Qué falló (mejoró): Excelente servicio de cabina y puntualidad  
- Dónde: No hay rutas con volumen suficiente para aislarla (Global/SH/Business/YW)  
- Quién: Leisure NPS 72.7 pts / Business NPS 40.0 pts, spread 32.7 pts (Global/SH/Business/YW)  
- Evidencia COMPLETA:  
  • NPS 55.56 vs baseline 37.07 (+18.49 pts, Global/SH/Business/YW)  
  • Load Factor 54.75 (–3.61 vs media, Global/SH/Business/YW)  
  • OTP15_adjusted 91.44 (+3.67 vs media, Global/SH/Business/YW)  
  • NCS: 0 incidentes (Global/SH/Business/YW)  
  • 12 verbatims elogiando tripulación, confort y salón VIP (Global/SH/Business/YW)  

CAUSA 5: Insatisfacción en Business Long Haul  
- Escenario: DOMINANCIA (Eco –, Bus –, Prem + | LH –)  
- NMA: Global/LH/Business  
- Afecta a: Global/LH/Business  
- Qué falló: Actitud de cabina deficiente, catering insuficiente y oferta de entretenimiento pobre  
- Dónde: Ruta MAD–MEX, NPS 33.3 pts (n=3) (Global/LH/Business)  
- Quién: Residence Europa (–100.0 pts, n=3), América Norte (–60.0 pts, n=5), Leisure (–15.4 pts, n=13) (Global/LH/Business)  
- Evidencia COMPLETA:  
  • NPS –5.0 vs baseline 11.47 (–16.47 pts, Global/LH/Business)  
  • Load Factor 93.66 (–0.32 vs media, Global/LH/Business)  
  • OTP15_adjusted 77.96 (–1.12 vs media, Global/LH/Business)  
  • NCS: 0 incidentes (Global/LH/Business)  
  • Verbatims con críticas a catering y entretenimiento (Global/LH/Business)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
La red global experimentó el 1 de diciembre de 2025 un NPS de 19.96 pts (Global), caída de 6.51 pts vs L7d (26.47 pts). A nivel Long Haul, Economy LH cayó de 11.26 a 8.09 pts (–3.17 pts vs L7d), Business LH se desplomó de 11.47 a –5.00 pts (–16.47 pts vs L7d) y Premium LH ascendió de 16.31 a 62.50 pts (+46.19 pts vs L7d). En Short Haul, Economy SH pasó de 33.56 a 24.52 pts (–9.04 pts vs L7d) y Business SH cerró en 35.71 vs 34.51 pts (+1.20 pts vs L7d), efecto que resulta de la combinación de IB (26.32 pts, –8.08 pts vs L7d) y YW (55.56 pts, +18.49 pts vs L7d). Las causas identificadas incluyen: los tres detractores en MAD–VGO (NPS –33.3, n=3) que arrastraron Global; la insatisfacción en Economy SH en MAD–OPO (NPS 0.0, n=6) por codeshare AA (–60.0 pts) e I2 (–33.3 pts) y residentes América Norte (–37.5 pts); las deficiencias de tripulación, catering y entretenimiento en Business LH MAD–MEX (NPS 33.3, n=3); y el excelente servicio Premium LH en MAD–MEX Premium (NPS 75.0, n=4).

Las rutas más afectadas fueron MAD–VGO (Global –33.3 pts), MAD–OPO (Global/SH/Economy 0.0 pts), EZE–MAD (Global/LH/Economy 12.5 pts), BRU–MAD (Global/SH/Economy/IB 0.0 pts) y MAD–MEX (Global/LH/Business 33.3 pts; Global/LH/Premium 75.0 pts). Los grupos más reactivos incluyen viajeros Business/Work, residentes en Europa y América Norte, usuarios de códigos compartidos QR, I2 y AA, así como pasajeros en flota A350 C y A321XLR.

ECONOMY SH: Desplome por codeshares  
La cabina Global/SH/Economy IB mostró un NPS de 23.59 pts el 1 de diciembre de 2025 (–9.77 pts vs L7d), mientras YW registró 26.05 pts (–8.01 pts vs L7d). En conjunto, pasó de 33.56 a 24.52 pts (–9.04 pts vs L7d). La causa principal fue la insatisfacción en ruta MAD–OPO (NPS 0.0 pts, n=6), impulsada por pasajeros en codeshare AA (–60.0 pts) e I2 (–33.3 pts) y residentes de América Norte (–37.5 pts), pese a un OTP de 92.30 (Global/SH/Economy), load factor 86.46 (Global/SH/Economy) y cero incidentes NCS. El feedback de clientes fuera de esa conexión se mantuvo mayoritariamente positivo.

BUSINESS SH: Equilibrio entre crisis y repunte  
Global/SH/Business IB cayó de 34.40 a 26.32 pts (–8.08 pts vs L7d) por la ruta MAD–ORY (NPS 25.0 pts, n=4), donde se reportaron percepciones de falta de profesionalidad de tripulación y servicio limitado, con OTP de 93.23 (Global/SH/Business/IB). En contraste, YW subió de 37.07 a 55.56 pts (+18.49 pts vs L7d) gracias a una mejora en puntualidad (OTP 91.44, Global/SH/Business/YW) y un feedback de clientes muy positivo sobre amabilidad de tripulación e instalaciones VIP. El NPS agregado de 35.71 pts (+1.20 pts vs L7d) refleja esta compensación.

ECONOMY LH: Concentración en EZE–MAD  
Global/LH/Economy registró 8.09 pts el 1 de diciembre (–3.17 pts vs L7d), tras caer desde 11.26 pts. El deterioro se concentró en la ruta EZE–MAD (NPS 12.5 pts, n=24), principalmente entre viajeros de negocio en A350 C/A321XLR, residentes en Europa (–50.0 pts) y pasajeros en codeshare AA (–33.3 pts) y Others (–66.7 pts). Esto ocurrió con OTP de 77.96 (Global/LH/Economy), load factor 87.71 (Global/LH/Economy) y sin incidentes NCS.

BUSINESS LH: Caída por servicio a bordo  
Global/LH/Business se hundió a –5.00 pts el 1 de diciembre (–16.47 pts vs L7d, desde 11.47 pts). Los drivers fueron actitudes poco profesionales de la tripulación, catering insuficiente y entretenimiento deficiente en ruta MAD–MEX (NPS 33.3 pts, n=3), especialmente para residentes en Europa (–100.0 pts, n=3) y América Norte (–60.0 pts, n=5). El load factor estuvo en 93.66 (Global/LH/Business), OTP en 77.96 (Global/LH/Business) y sin incidentes NCS.

PREMIUM LH: Éxito en MAD–MEX Premium  
Global/LH/Premium alcanzó 62.50 pts el 1 de diciembre (+46.19 pts vs L7d, desde 16.31 pts), impulsado por ruta MAD–MEX Premium (NPS 75.0 pts, n=4). El feedback de clientes destacó puntualidad, simpatía de tripulación y calidad en tierra y a bordo, a pesar de un load factor de 88.29 (Global/LH/Premium) y OTP de 77.96 (Global/LH/Premium), con cero incidentes NCS.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-11-30 to 2025-11-30:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Economy SH, el escenario es CANCELACIÓN (IB –, YW + | Padre N).  
- Narrativa: Mientras Global/SH/Economy/IB sufrió una caída por la baja satisfacción en la ruta LIS–MAD —especialmente pasajeros en flota A350 next y vuelos CodeShare BA—, Global/SH/Economy/YW mostró un fuerte aumento gracias a la excelente puntualidad y satisfacción de los viajeros de ocio en la ruta MAD–MUC, neutralizándose el efecto en el agregado.  
- Evidencia Clave:  
  • IB – Ruta LIS–MAD, flota A350 next y CodeShare BA.  
  • YW – Ruta MAD–MUC, OTP15_adjusted elevada y alta satisfacción de Leisure.  

En Business SH, el escenario es DILUCIÓN (IB N, YW – | Padre N).  
- Narrativa: La caída de Global/SH/Business/YW fue impulsada por la insatisfacción de los viajeros de ocio y residentes en Europa en la ruta MAD–XRY, aunque la estabilidad de Global/SH/Business/IB (sin anomalías) suavizó este impacto en el nodo padre.  
- Evidencia Clave:  
  • YW – Ruta MAD–XRY, Leisure NPS 14.3 y residencia Europa con NPS –33.3.  
  • IB – Rendimiento estable sin anomalías (NPS 39.4 vs baseline 34.4).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es SINERGÍA (Economy N, Business N | SH N).  
- Narrativa: Ambas cabinas mantuvieron desempeño estable sin anomalías internas; por ello el radio Short Haul se mantiene dentro del rango esperado.  
- Evidencia: Economy SH NPS 34.52 vs baseline 33.56 (N), Business SH NPS 37.21 vs baseline 34.51 (N).  

En Long Haul, la dinámica es DOMINANCIA (Economy –, Business +, Premium – | LH –).  
- Narrativa: El NPS de Long Haul está dictado por la fuerte caída en Economy, concentrada en la ruta EZE–MAD entre pasajeros de flota A321XLR y vuelos CodeShare QR, aunque el alza en Business (MAD–MEX) atenuó parcialmente el impacto.  
- Evidencia: Global/LH/Economy –25.43 pts (NPS –14.17 vs 11.26) por ruta EZE–MAD, flota A321XLR y CodeShare QR.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA (LH –, SH N | GLOBAL –).  
- Narrativa: La caída del NPS global se contagia directamente desde el segmento Long Haul. Aunque Short Haul mantuvo un desempeño estable, la fuerte insatisfacción registrada en Long Haul arrastró al Global.  
- Evidencia:  
  • Ruta MAD–MIA con NPS –35.6 (n=8) en Long Haul.  
  • Flotas A332 (NPS –17.5; n=84) y A333 (NPS –26.5; n=32).  
  • Pasajeros residentes en América Norte: NPS –35.3 (n=42).

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Caída en Global/SH/Economy/IB  
- Escenario: CANCELACIÓN (IB –, YW + | Economy SH N)  
- NMA: Global/SH/Economy/IB (no hay NMA único en cancelación)  
- Afecta a: Global/SH/Economy/IB  
- Qué falló: valoración muy baja en flota A350 next y vuelos CodeShare BA (Global/SH/Economy/IB)  
- Dónde: ruta LIS–MAD con NPS 0.0 (Global/SH/Economy/IB; n=4)  
- Quién: pasajeros en Fleet A350 next (NPS –100.0, Global/SH/Economy/IB) y CodeShare BA (NPS –100.0, Global/SH/Economy/IB)  
- Evidencia COMPLETA:  
   • NPS 27.27 pts vs baseline 33.36 pts (Global/SH/Economy/IB)  
   • Load Factor 89.59 (–0.15 pts vs baseline) (Global/SH/Economy/IB)  
   • OTP15_adjusted 92.92 (+1.55 pts vs baseline) (Global/SH/Economy/IB)  
   • Incidentes NCS: ninguno (Global/SH/Economy/IB)  
   • 430 verbatims con elogios generales pero sin mitigar la baja de subgrupos  

CAUSA 2: Subida en Global/SH/Economy/YW  
- Escenario: CANCELACIÓN (IB –, YW + | Economy SH N)  
- NMA: Global/SH/Economy/YW (no hay NMA único en cancelación)  
- Afecta a: Global/SH/Economy/YW  
- Qué “brilló”: excelente puntualidad y servicio para pasajeros de ocio (Global/SH/Economy/YW)  
- Dónde: ruta MAD–MUC con NPS 28.6 (Global/SH/Economy/YW; n=7)  
- Quién: Leisure con NPS 53.3 (Global/SH/Economy/YW)  
- Evidencia COMPLETA:  
   • NPS 48.68 pts vs baseline 34.06 pts (Global/SH/Economy/YW)  
   • Load Factor 81.37 (Global/SH/Economy/YW)  
   • OTP15_adjusted 91.6 (+3.79 pts vs baseline) (Global/SH/Economy/YW)  
   • Incidentes NCS: ninguno (Global/SH/Economy/YW)  
   • 198 verbatims muy positivos hacia tripulación y confort  

CAUSA 3: Caída en Global/SH/Business/YW  
- Escenario: DILUCIÓN (IB N, YW – | Business SH N)  
- NMA: Global/SH/Business/YW  
- Afecta a: Global/SH/Business/YW  
- Qué falló: baja ocupación redujo la percepción de valor (Load Factor 55.08, Global/SH/Business/YW)  
- Dónde: ruta MAD–XRY con NPS 66.7 (Global/SH/Business/YW; n=3)  
- Quién: Leisure con NPS 14.3 y residentes Europa con NPS –33.3 (Global/SH/Business/YW)  
- Evidencia COMPLETA:  
   • NPS 30.0 pts vs baseline 37.07 pts (Global/SH/Business/YW)  
   • OTP15_adjusted 91.6 (+3.79 pts vs baseline) (Global/SH/Business/YW)  
   • Incidentes NCS: ninguno (Global/SH/Business/YW)  
   • 11 verbatims elogian servicio, pero no compensan la baja de ocio y Europa  

CAUSA 4: Caída en Global/LH/Economy  
- Escenario: DOMINANCIA (Economy –, Business +, Premium – | LH –)  
- NMA: Global/LH/Economy  
- Afecta a: Global/LH/Economy  
- Qué falló: ligera caída en Load Factor y OTP15_adjusted y desalinee de expectativas en CodeShare (Global/LH/Economy)  
- Dónde: ruta EZE–MAD con NPS –21.2 (Global/LH/Economy; n=34)  
- Quién: Fleet A321XLR con NPS –63.6, CodeShare QR con NPS –100.0 y viajeros Business/Work con NPS –20.6 (Global/LH/Economy)  
- Evidencia COMPLETA:  
   • NPS –14.17 pts vs baseline 11.26 pts (Global/LH/Economy)  
   • Load Factor 87.41 (–2.94 pts vs baseline) (Global/LH/Economy)  
   • OTP15_adjusted 78.91 (–0.51 pts vs baseline) (Global/LH/Economy)  
   • Incidentes NCS: ninguno (Global/LH/Economy)  
   • 508 verbatims apuntan a puntualidad y amabilidad, pero no elevan el NPS  

CAUSA 5: Subida en Global/LH/Business  
- Escenario: DILUCIÓN (Eco –, Bus +, Prem – | LH –)  
- NMA: Global/LH/Business  
- Afecta a: Global/LH/Business  
- Qué “brilló”: profesionalidad del personal, calidad de catering y puntualidad (Global/LH/Business)  
- Dónde: ruta MAD–MEX con NPS 100.0 (Global/LH/Business; n=4)  
- Quién: small sample de 66 verbatims todos positivos (Global/LH/Business)  
- Evidencia COMPLETA:  
   • NPS 21.21 pts vs baseline 11.47 pts (Global/LH/Business)  
   • Load Factor 93.4 (–0.54 pts vs baseline) (Global/LH/Business)  
   • OTP15_adjusted 78.91 (–0.51 pts vs baseline) (Global/LH/Business)  
   • Incidentes NCS: ninguno (Global/LH/Business)  

CAUSA 6: Caída en Global/LH/Premium  
- Escenario: DILUCIÓN (Eco –, Bus +, Prem – | LH –)  
- NMA: Global/LH/Premium  
- Afecta a: Global/LH/Premium  
- Qué falló: menor satisfacción de pasajeros de América Centro y usuarios de flota A350 next (Global/LH/Premium)  
- Dónde: ruta MAD–MEX con NPS 20.0 (Global/LH/Premium; n=5)  
- Quién: residencia América Centro NPS –50.0 (n=4), flota A350 next NPS –16.7 (n=12) (Global/LH/Premium)  
- Evidencia COMPLETA:  
   • NPS 4.17 pts vs baseline 16.31 pts (Global/LH/Premium)  
   • Load Factor 88.0 (–2.68 pts vs baseline) (Global/LH/Premium)  
   • OTP15_adjusted 78.91 (–0.51 pts vs baseline) (Global/LH/Premium)  
   • Incidentes NCS: ninguno (Global/LH/Premium)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

La red experimentó una bajada de NPS global de 26.47 a 20.98 (–5.49 pts vs L7d), arrastrada por el deterioro en Long Haul (LH) de 11.71 a –9.76 (–21.47 pts vs L7d). Dentro de LH, Economy cayó de 11.26 a –14.17 (–25.43 pts vs L7d) principalmente en la ruta EZE–MAD (NPS –21.2, n=34) entre usuarios de flota A321XLR y CodeShare QR; Premium bajó de 16.31 a 4.17 (–12.15 pts vs L7d) en MAD–MEX con pasajeros de América Centro y A350 next; Business, en contraste, subió de 11.47 a 21.21 (+9.74 pts vs L7d) gracias al feedback 100 % positivo en MAD–MEX (n=4). En Short Haul (SH), el NPS mejoró ligeramente de 33.70 a 34.69 (+0.99 pts vs L7d) por la cancelación de efectos opuestos en Economy (IB 27.27 desde 33.36, –6.08 pts vs L7d; YW 48.68 desde 34.06, +14.62 pts vs L7d) y una ligera caída en Business/YW de 37.07 a 30.00 (–7.07 pts vs L7d) mitigada por IB estable (39.39 vs 34.40). Los datos operativos confirman OTP en torno a 91–93, load factors entre 81 y 93, cero incidentes NCS y feedback de clientes que elogia puntualidad y servicio, salvo en subsegmentos afectados por confort de cabina y coordinación CodeShare.

Las rutas más críticas fueron EZE–MAD (NPS –15.1 en Global/LH/Economy; –21.2 en Economy LH), MAD–MIA (–35.6 en Global/LH), MAD–MEX (100.0 en Global/LH/Business vs 20.0 en Global/LH/Premium), LIS–MAD (0.0 en Global/SH/Economy/IB), MAD–MUC (28.6 en Global/SH/Economy/YW) y MAD–XRY (66.7 en Global/SH/Business/YW). Los perfiles más reactivos incluyen pasajeros en A321XLR (NPS –63.6, Global/LH/Economy), A350 next (NPS –100.0, Global/SH/Economy/IB), CodeShare QR (–100.0, Global/LH/Economy), BA (–100.0, Global/SH/Economy/IB), viajeros Leisure en SH/Economy/YW (+53.3) y residentes en América Norte (–35.3, Global/LH) y Europa (–33.3, Global/SH/Business/YW).

ECONOMY SH: Disparidad interna neutraliza resultado  
La cabina Economy de SH mantuvo desempeño estable durante la semana del 30-nov, registrando un NPS de 34.52 (30-nov) con una subida de 0.96 pts vs L7d. Internamente, IB descendió de 33.36 a 27.27 (–6.08 pts vs L7d) por insatisfacción en la ruta LIS–MAD entre pasajeros en flota A350 next y vuelos CodeShare BA (Global/SH/Economy/IB), mientras YW escaló de 34.06 a 48.68 (+14.62 pts vs L7d) gracias a un OTP de 91.6 y feedback de clientes muy positivo en la ruta MAD–MUC (Global/SH/Economy/YW).

BUSINESS SH: Rendimiento sólido sin cambios significativos  
El segmento Business de SH mantuvo desempeño estable, registrando un NPS de 37.21 (30-nov) con una subida de 2.70 pts vs L7d. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción en IB (39.39 vs 34.40) y, a pesar de la caída en YW (30.00 vs 37.07, –7.07 pts vs L7d), los datos operativos (load factor 55.08, OTP 91.6) y el feedback de clientes no revelan incidencias NCS ni puntos de fricción críticos.

ECONOMY LH: Fuerte deterioro en EZE–MAD  
La cabina Economy de LH registró un NPS de –14.17 (30-nov) con una caída de 25.43 pts vs L7d. La causa principal fue la ruta EZE–MAD (NPS –21.2, n=34) donde viajeros de flota A321XLR (NPS –63.6) y CodeShare QR (NPS –100.0) expresaron desaliento, pese a un OTP de 78.91 y load factor de 87.41; el feedback de clientes elogia puntualidad y servicio, descartando fallos operativos o incidentes NCS.

BUSINESS LH: Impulso excepcional en MAD–MEX  
La cabina Business de LH experimentó una subida de 9.74 pts, pasando de un NPS de 11.47 a 21.21 (30-nov). Los drivers fueron la profesionalidad de la tripulación, calidad de catering y aeronaves modernas, reflejados en un NPS perfecto de 100.0 (n=4) en la ruta MAD–MEX, con load factor 93.4 y OTP 78.91, sin incidentes NCS.

PREMIUM LH: Descenso por América Centro y flota A350 next  
El segmento Premium de LH cayó de un NPS de 16.31 a 4.17 (–12.15 pts vs L7d). La baja satisfacción se concentró en pasajeros de América Centro (NPS –50.0, n=4) y usuarios de flota A350 next (NPS –16.7, n=12) en la ruta MAD–MEX (NPS 20.0, n=5), pese a un OTP de 78.91, load factor 88.0 y feedback de clientes mayoritariamente positivo.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
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
12. **ATRIBUCIÓN DE SEGMENTO**: Siempre que menciones un dato, indica a qué segmento pertenece (ej: "NPS 19.8 (Economy LH)", "OTP –4.0 pts (Business SH)")