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
- Existen ambos nodos: Economy SH/IB (NPS +1.9, normal) y Economy SH/YW (NPS +3.3, normal).  
- Ambos muestran estabilidad, sin anomalías operativas ni drivers SHAP divergentes.  
Diagnóstico: la variación es general a la cabina Economy SH, no hay causas específicas por compañía.  

B. Business Short Haul  
- Existen ambos nodos: Business SH/IB (POSITIVE ANOMALY +16.7) y Business SH/YW (NEGATIVE ANOMALY –20.4).  
- SH/IB mejora por puntuality (SHAP +4.217) y menor número de retrasos/cancelaciones; SH/YW cae por cancelaciones e incidencias (SHAP puntuality –0.568, otras incidencias +8).  
Diagnóstico: las causas son específicas a cada compañía dentro de Business SH, no comunes a la cabina.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul  
El patrón no es común al radio: Economy SH mantiene estabilidad y reacciona muy poco a las causas operativas (ambos IB/YW convergen en variaciones normales), mientras que Business SH muestra una divergencia clara entre compañías (IB con +16,7 por mejora de puntualidad vs. YW con –20,4 por cancelaciones e incidencias).  

Long Haul  
Las cabinas divergen en reactividad: Economy LH actúa como amortiguador (sin anomalías), Business LH refleja una pequeña caída (–0,4 por cancelaciones y limitaciones) y Premium LH es la más sensible (–12,2 por huelgas, cancelaciones y deficiencias de producto).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Ambos radios muestran anomalías pero por causas distintas: en Long Haul predominan los impactos negativos por cancelaciones e incidencias operativas, mientras que en Short Haul coexisten mejoras de puntualidad (IB) y deterioros por cancelaciones (YW) que se compensan, de tal forma que al nivel Global estos efectos contrapuestos se neutralizan y derivan en un ligero alza neta.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

1. Causa: Mejora en puntualidad  
A. Naturaleza del driver  
• Hipótesis: Reducción de retrasos y cancelaciones eleva la percepción de fiabilidad y satisfacción.  

B. Evidencia consolidada y alcance  
Segmento base: Global (NPS +2,02 vs L7d)  
• Afecta a todos los subsegmentos bajo Global (SH y LH, Economy, Business y Premium).  
• NPS actual: 30,71 (vs 28,69 baseline)  
• SHAP puntuality: +2,369 (explanatory_drivers_tool)  
• OTP15: +0,6 puntos (operative_data_tool)  
• Retrasos: –126 incidentes; Cancelaciones: –48 (ncs_tool)  
• Rutas más representativas (volumen y NPS):  
  – ILD-PMI: NPS 33,3 (Pax 3)  
  – MAH-VLC: NPS 20,0 (Pax 5)  
  – BCN-VLC: NPS 75,0 (Pax 8)  
  – ALG-MAD: NPS 42,9 (Pax 7)  
  – MAD-SXB: NPS 0,0 (Pax 5)  
• Verbatims: volumen –13,2% (5 341 vs 6 155), sin temas contrarios a puntualidad.  

2. Causa: Incidencias operativas (cancelaciones y limitaciones de aeronave)  
A. Naturaleza del driver  
• Hipótesis: Aumento de cancelaciones y limitaciones degrada la experiencia de viaje y la satisfacción.  

B. Evidencia consolidada y alcance  
Segmento base: Long Haul (NPS –0,26 vs L7d)  
• Afecta al 100% de los subsegmentos Long Haul (Business y Premium).  
• NPS actual LH: 18,02 (vs 18,28 baseline)  
• Cancelaciones: +17 (de 21 a 38)  
• Limitaciones de aeronave: +4 (de 2 a 6)  
• Mishandling equipaje: +0,19 pts (13,32→13,51)  
• Rutas más impactadas:  
  – MAD-MIA: NPS –2,6 (Pax 38)  
  – MAD-ORD: identificada (+ datos NPS no disponibles)  
  – IAD-MAD: NPS 87,5 (Pax 8)  
  – MAD-PTY: NPS 30,0 (Pax 20)  
  – DOH-MAD: NPS 0,0 (Pax 23)  
  – MAD-SJO: NPS 25,0 (Pax 44)  

3. Causa: Deficiencias de producto/servicio en Premium LH  
A. Naturaleza del driver  
• Hipótesis: Problemas en interface de cliente (preparación de viaje, tripulación, interior, precio) erosionan la satisfacción de segmento premium.  

B. Evidencia consolidada y alcance  
Segmento base: Global/LH/Premium (NPS –12,17 vs L7d)  
• Afecta a la totalidad del Premium Long Haul (no hay sub-niveles adicionales).  
• NPS actual Premium LH: 23,08 (vs 35,25 baseline)  
• SHAP destacados (explanatory_drivers_tool, sample=20):  
  – Ease of contact by phone: +4,351  
  – Journey preparation support: –4,137  
  – Cabin Crew: –3,269  
  – IB Plus loyalty program: –2,985  
  – Boarding: –1,946  
  – Aircraft interior: –1,576  
  – Check-in: –1,282  
  – Ticket Price: –1,112  
  – In-flight F&B: –0,219  
• Rutas más sensibles:  
  – MAD-SCL: NPS –16,7 (Pax 6)  
  – MAD-MEX: NPS –5,6 (Pax 18)  
  – EZE-MAD: NPS 26,3 (Pax 19)  
  – MAD-MIA: NPS –25,0 (Pax 4)  
• Reactividad de cliente: Residence Region spread 62,2 pts (máxima variabilidad).

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
Durante la semana del 2025-11-24 al 2025-11-29 se observaron subidas y bajadas de NPS en segmentos clave. A nivel Global, el NPS creció de 28.69 a 30.71 (+2.02 pts vs L7d) gracias a una mejora en punctuality (OTP15 +0.6, SHAP +2.37, –126 retrasos y –48 cancelaciones). En Long Haul, el NPS cayó de 18.28 a 18.02 (–0.26 pts vs L7d) por un aumento de cancelaciones y limitaciones de aeronave (cancelaciones +17, limitaciones +4), con un descenso más marcado en Business LH (32.16→31.79, –0.38) y Premium LH (35.25→23.08, –12.17) donde se sumaron huelgas en BRU/BLQ, mishandling de equipaje (+0.19) y drivers de producto como Journey preparation support (SHAP –4.14) y Cabin Crew (SHAP –3.27). Por su parte, en Short Haul Business IB escaló de 39.22 a 55.92 (+16.70) impulsado por la drástica reducción de retrasos (–23) y cancelaciones (–16) y un SHAP de punctuality de +4.22, mientras que Business YW retrocedió de 42.47 a 22.08 (–20.39) por cancelaciones residuales e incidencias de aeronave.

Entre las rutas con mayor variación destacan ILD-PMI, BCN-VLC y ALG-MAD por la mejora de puntualidad, así como MAD-MIA y MAD-MEX en Premium LH y MAD-ORD en Business LH por cancelaciones. Los perfiles de cliente más reactivos fueron CodeShare, Residence Region y Fleet, mostrando los spreads de NPS_diff más elevados vs L7d.

ECONOMY SH  
La cabina Economy de SH IB registró un NPS de 32.63 durante la semana del 2025-11-24 al 2025-11-29 (subida de 1.9 pts vs L7d) y YW un NPS de 41.85 (subida de 3.3 pts vs L7d). No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción en ambos subsegmentos.

BUSINESS SH  
La cabina Business de SH IB alcanzó un NPS de 55.92 (subida de 16.7 pts vs L7d) mientras que YW descendió a 22.08 (bajada de 20.4 pts vs L7d), resultando en un NPS agregado de 44.54 (alza de 4.5 pts vs L7d). Esta evolución refleja el doble impacto de la mejora de puntuality en IB (SHAP +4.22, retrasos –23, cancelaciones –16) y el efecto contrario de cancelaciones e incidencias de aeronave en YW (cancelaciones +16, otras incidencias +8), con rutas como LHR-MAD y LIS-MAD liderando la variación.

ECONOMY LH  
La cabina Economy de LH mantuvo desempeño estable con un NPS de 15.36 (subida de 0.8 pts vs L7d). No se detectaron cambios operativos ni drivers de producto significativos, confirmando niveles de satisfacción consistentes.

BUSINESS LH  
La cabina Business de LH registró un NPS de 31.79 (descenso de 0.38 pts vs L7d), explicado principalmente por un aumento de cancelaciones (21→38) y limitaciones de aeronave (2→6), validado por ncs_tool y operative_data_tool, con impactos notables en rutas como MAD-MIA y MAD-ORD y alta reactividad en perfiles Residence Region y CodeShare.

PREMIUM LH  
La cabina Premium de LH sufrió un deterioro relevante, pasando de 35.25 a 23.08 (–12.17 pts vs L7d), dominado por cancelaciones de huelga en BRU/BLQ (+17 cancelaciones), mishandling de equipaje (+0.19) y drivers de producto negativos como Journey preparation support (SHAP –4.14) y Cabin Crew (SHAP –3.27), afectando especialmente rutas MAD-SCL y MAD-MIA y con variaciones máximas en clientes por Residence Region.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*



**ANÁLISIS DIARIO SINGLE:**
📅 2025-11-29 to 2025-11-29:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

A. Economy Short Haul  
– Existen ambos nodos (SH/IB y SH/YW).  
– Sus drivers son distintos: IB arrastra insatisfacción por CodeShare AA y flotas A319/A320neo, mientras YW cae por baja satisfacción de Business y de residentes europeos.  
Diagnóstico: causa específica de compañía.

B. Business Short Haul  
– Existen ambos nodos (SH/IB y SH/YW).  
– Patrón opuesto: IB muestra un alza por servicio y puntualidad excepcionales; YW cae por muy bajo NPS de viajeros Business.  
Diagnóstico: causa específica de compañía.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul (SH): las cabinas Economy y Business muestran un patrón claramente divergente (Economy −0,4 pts vs Business +13,3 pts), con reactividad diferente incluso entre IB y YW; por tanto, las causas son específicas de cada cabina, no comunes a todo el radio.

Long Haul (LH): las tres cabinas divergen (Economy normal, Business −8,5 pts y Premium +29,9 pts) y presentan niveles de reactividad muy distintos; esto confirma que las causas operativas impactan de forma específica a cada clase de servicio.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Diagnóstico global: ambos radios muestran anomalías pero con drivers distintos (SH impulsado por un fuerte NPS en Business IB frente a caída leve en Economy; LH penalizado en Business por clientes europeos y sobredimensionado en Premium por atención de tripulación). Estas variaciones opuestas se compensan, de modo que el nodo Global permanece dentro de la normalidad. Por tanto, las causas son mixtas y específicas a cada radio, sin un patrón causal homogéneo a nivel compañía.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

1. Causa: Excelencia en atención de tripulación y eficiencia en embarque  
A. Naturaleza  
– Driver cualitativo de servicio (tripulación y boarding) ligado a la percepción directa del cliente, sin relación con incidencias operativas ni puntualidad real medida.  

B. Evidencia consolidada y alcance  
– Segmento más grande impactado: Global/Short Haul/Business (n = 32 verbatims).  
– Este driver afecta a todos los subsegmentos bajo SH/Business (IB y YW).  
– Output causal (Global / SH / Business):  
   • NPS día: 58.33 vs baseline 44.99 (+13.33 pts)  
   • No hay datos operativos disponibles ni incidentes NCS.  
   • Rutas: LHR–MAD (n=3) con NPS=100.  
   • Verbatims representativos: “puntualidad impecable y atención de la tripulación”, “embarque ágil y cordial”.  

2. Causa: Insatisfacción de clientes residentes en Europa en Long Haul Business  
A. Naturaleza  
– Driver de perfil (Residence Region) con expectativas más elevadas que no se satisfacen, sin evidencia de problemas operativos.  

B. Evidencia consolidada y alcance  
– Segmento afectado: Global/Long Haul/Business (n = 37 verbatims).  
– Abarca todo el nodo LH/Business.  
– Output causal (Global / LH / Business):  
   • NPS día: 26.32 vs baseline 34.82 (–8.50 pts)  
   • Sin datos operativos ni incidentes NCS.  
   • Rutas: MAD–MEX (n=3) con NPS=33.3.  
   • Verbatims clave: “todo perfecto, pero esperaba más en Business” (clientes Europa).  
   • Driver principal: Europa –50 pts vs América +60 pts.  

3. Causa: Insatisfacción de pasajeros Economy SH/IB por CodeShare AA y flota A319/A320neo  
A. Naturaleza  
– Driver de combinación perfil–flota: pasajeros AA perciben peor coordinación y las cabinas A319/A320neo tienen mayor fricción en servicio.  

B. Evidencia consolidada y alcance  
– Segmento afectado: Global/Short Haul/Economy/IB (n = 288 verbatims).  
– Impacta a todo el nodo SH/Economy/IB.  
– Output causal (Global / SH / Economy / IB):  
   • NPS día: 32.11 vs baseline 32.11 (–0.01 pts)  
   • Sin datos operativos ni incidentes NCS.  
   • Ruta principal: FCO–MAD (n=15) con NPS=26.7.  
   • Verbatims: “vuelo OK pero retrasos en info AA”, “asientos A319 menos confortables”.  

4. Causa: Baja satisfacción de viajeros de negocio en Short Haul Business/YW  
A. Naturaleza  
– Driver de propósito de viaje: clientes Business perciben carencias en amenities y servicios a bordo en vuelos YW.  

B. Evidencia consolidada y alcance  
– Segmento afectado: Global/Short Haul/Business/YW (n = 10 verbatims).  
– Afecta íntegramente al nodo SH/Business/YW.  
– Output causal (Global / SH / Business / YW):  
   • NPS día: 14.29 vs baseline 21.69 (–7.40 pts)  
   • Sin datos operativos ni incidentes NCS.  
   • Rutas: sin muestras suficientes para aislar rutas.  
   • Verbatims: “falta wi-fi y asientos incómodos en categoría Business”.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

La jornada del 29-nov-2025 mostró anomalías contrapuestas en el NPS. En Short Haul, Global/SH/Business subió de 44.99 a 58.33 (+13.34 pts) impulsado por Global/SH/Business/IB (76.47 vs 55.72, +20.75 pts) gracias a la excelencia en atención de tripulación y rapidez de embarque, mientras Global/SH/Business/YW cayó de 21.69 a 14.29 (–7.40 pts) por carencias en amenities a bordo. Al mismo tiempo, Global/SH/Economy descendió de 35.63 a 35.23 (–0.40 pts), con SH/Economy/IB registrando 32.10 (–0.01 pts) por insatisfacción en vuelos CodeShare AA y flotas A319/A320neo, y SH/Economy/YW en 43.24 (–0.22 pts) reflejando expectativas no cubiertas. En Long Haul, Global/LH/Business sufrió un deterioro de 34.82 a 26.32 (–8.50 pts) asociado a la baja satisfacción de clientes europeos (Europa –50.0 pts vs América Centro/España +60.0 pts), mientras Global/LH/Premium se elevó de 30.15 a 60.00 (+29.85 pts) por valoraciones excepcionalmente positivas de tripulación y boarding, sin incidencias operativas formales.

Las rutas más impactadas incluyen FCO–MAD, con NPS 26.7 (–9.6 pts vs L7d) en SH/Economy/IB; LIS–MAD (20.0 pts en SH/Economy); LHR–MAD, que alcanzó 100.0 pts en SH/Business/IB; y MAD–MEX (33.3 pts en LH/Business). Los grupos de clientes más reactivos fueron los viajeros de CodeShare AA y pasajeros de flota A319/A320neo en Economy SH, los viajeros Business en Premium LH y los residentes europeos en Business LH.

ECONOMY SH – Global/SH/Economy  
La cabina Global/SH/Economy cerró el día 29-nov-2025 con un NPS de 35.23 (–0.40 pts vs L7d), donde SH/Economy/IB marcó 32.10 (–0.01 pts) y SH/Economy/YW 43.24 (–0.22 pts). La caída se explica principalmente por la experiencia discontinua en CodeShare AA y la menor comodidad en flotas A319/A320neo, evidenciado en la ruta FCO–MAD (NPS 26.7, –9.6 pts), y por un leve gap de expectativas en el servicio de YW, según verbatims que mencionan problemas de coordinación y confort.

BUSINESS SH – Global/SH/Business  
El segmento Global/SH/Business experimentó un NPS de 58.33 (↑13.34 pts vs L7d), impulsado por SH/Business/IB (76.47 vs 55.72, +20.75 pts) gracias a valoraciones sobresalientes de tripulación y boarding, y moderado por la caída de SH/Business/YW a 14.29 (–7.40 pts) por carencias en amenities. La ruta LHR–MAD alcanzó NPS 100.0 con menciones explícitas a atención impecable y procesos ágiles.

ECONOMY LH – Global/LH/Economy  
La cabina Global/LH/Economy mantuvo desempeño estable con un NPS de 17.32 (+2.13 pts vs L7d). No se detectaron cambios significativos en operativa ni feedback, manteniendo niveles consistentes de satisfacción y sin evidencias de incidencias.

BUSINESS LH – Global/LH/Business  
La cabina Global/LH/Business registró un NPS de 26.32 (–8.50 pts vs L7d). El deterioro responde a la menor satisfacción de clientes residentes en Europa (Europe –50.0 pts vs América Centro/España +60.0 pts) en el nodo Global/LH/Business, sin incidentes formales y con la ruta MAD–MEX en 33.3 pts.

PREMIUM LH – Global/LH/Premium  
El segmento Global/LH/Premium alcanzó un NPS de 60.00 (+29.85 pts vs L7d). Este salto refleja la excelencia en la atención de tripulación y eficiencia del embarque, sin incidencias reportadas, con perfiles IB (60.0 pts) y Leisure (50.0 pts) destacando la experiencia.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-11-28 to 2025-11-28:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

Economy Short Haul (SH)  
• Existen ambos nodos IB y YW con anomalías opuestas (IB –2.0 pts vs YW +7.8 pts) y drivers completamente distintos (insatisfacción de ‘business’ en LHR–MAD para IB vs sesgo de muestra de Centroamérica en AGP–MAD para YW).  
→ Diagnóstico: causa específica a nivel de compañía.  

Business Short Haul (SH)  
• Existen ambos nodos IB y YW, ambos con anomalías negativas, pero motivaciones distintas (baja percepción de flota A320neo en IB vs expectativas incumplidas de clientes corporativos de América Norte en YW).  
→ Diagnóstico: causa específica a nivel de compañía.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul (SH)  
Economy SH y Business SH muestran comportamientos divergentes.  
- Economy SH es globalmente “normal” (±1 pt) gracias a un IB negativo (–2 pts) compensado por un YW positivo (+7.8 pts).  
- Business SH es claramente negativo (–13.9 pts) en ambas compañías, pero con magnitud e impulsores distintos (flota en IB vs perfil corporativo en YW).  
→ Diagnóstico SH: las causas NO son comunes al radio, sino específicas a cada cabina (y en Economy además separadas por compañía).

Long Haul (LH)  
Economy LH (–14.6 pts) y Premium LH (–23.5 pts) se ven afectados de forma consistente (ambos negativos), mientras que Business LH permanece estable (normal +3.9 pts).  
→ Diagnóstico LH: la causa es específica de cabina. Business actúa como amortiguador, mientras que Economy y Premium comparten el impacto negativo.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Ambos radios registran anomalías negativas pero por drivers diferentes (SH impactado sobre todo en Business por flota y perfil corporativo, LH golpeado en Economy y Premium por rutas y flotas específicas), lo que revela un patrón mixto de causas que en Global se resumen en una caída atenuada de –4.5 pts.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

1) Causa: Experiencia de cabina A350 C en ruta LIM–MAD  
A. Naturaleza del driver  
   • Hipótesis: deficiencias de confort y servicio a bordo en la configuración de la flota A350 C.  
B. Evidencia consolidada y alcance  
   • Segmento analizado: Global (n = 19 encuestas LIM–MAD; total 957 verbatims).  
   • NPS día 25.99 vs baseline 30.47 (–4.48 pts).  
   • No hubo incidentes NCS ni datos operativos que expliquen retrasos o pérdidas de equipaje.  
   • Verbatims destacan puntualidad y atención, pero mencionan asientos y menú poco satisfactorios.  
   • Alcance: afecta a todos los subsegmentos Global (SH y LH) que operan en A350 C en LIM–MAD.  

2) Causa: Insatisfacción en Economy Long Haul por ruta MAD–MIA (Business)  
A. Naturaleza del driver  
   • Hipótesis: desalineación entre expectativas de pasajeros Business y la oferta de cabina/conectividad en flotas A321, A333 y A350 C operados en code share.  
B. Evidencia consolidada y alcance  
   • Segmento analizado: Global / LH / Economy (303 verbatims; ruta MAD–MIA n = 5).  
   • NPS día 0.57 vs baseline 15.19 (–14.62 pts).  
   • No hay reportes NCS ni métricas OTP que apunten a fallos operativos.  
   • Verbatims: quejas puntuales de confort y servicios complementarios en vuelos codeshare BA/AA.  
   • Alcance: impacta a todos los subsegmentos Economy LH, especialmente pasajeros Business/Work.  

3) Causa: Caída en Premium Long Haul en ruta EZE–MAD para clientes de América Central  
A. Naturaleza del driver  
   • Hipótesis: configuración del A350 next y oferta de servicio insuficiente para residentes de América Central.  
B. Evidencia consolidada y alcance  
   • Segmento analizado: Global / LH / Premium (24 verbatims; ruta EZE–MAD n = 6).  
   • NPS día 6.67 vs baseline 30.15 (–23.48 pts).  
   • Sin incidentes NCS ni datos operativos disponibles.  
   • Verbatims: elogios generales pero críticas a la ergonomía de la cabina “next”.  
   • Alcance: toda la cabina Premium LH, con mayor impacto en residentes de América Central.  

4) Causa: Descenso en Economy Short Haul IB por viajeros Business en LHR–MAD  
A. Naturaleza del driver  
   • Hipótesis: expectativas de servicio Business incumplidas en vuelos LHR–MAD.  
B. Evidencia consolidada y alcance  
   • Segmento analizado: Global / SH / Economy / IB (364 verbatims; ruta LHR–MAD n = 17).  
   • NPS día 30.11 vs baseline 32.11 (–2.00 pts).  
   • No hay registros NCS ni datos OTP.  
   • Verbatims resaltan puntualidad, pero insatisfacción con servicios de lounge y catering.  
   • Alcance: todos los subsegmentos Economy SH / IB, principalmente pasajeros business.  

5) Causa: Alza en Economy Short Haul YW por predominio de viajeros de Centroamérica en AGP–MAD  
A. Naturaleza del driver  
   • Hipótesis: sesgo de muestra con alta proporción de encuestados de Centroamérica, todos muy satisfechos.  
B. Evidencia consolidada y alcance  
   • Segmento analizado: Global / SH / Economy / YW (144 verbatims; ruta AGP–MAD n = 5).  
   • NPS día 51.26 vs baseline 43.47 (+7.79 pts).  
   • Sin incidentes NCS ni datos operativos.  
   • Verbatims unánimemente positivos: “servicio excelente”, “todo perfecto”.  
   • Alcance: todos los subsegmentos Economy SH / YW, por la composición de residencia.  

6) Causa: Mala percepción de flota CRJ en Business Short Haul IB  
A. Naturaleza del driver  
   • Hipótesis: configuración y confort de la flota CRJ generan insatisfacción en pasajeros Business.  
B. Evidencia consolidada y alcance  
   • Segmento analizado: Global / SH / Business / IB (74 verbatims; flota CRJ n = 17).  
   • NPS día 31.11 vs baseline 44.99 (–13.88 pts).  
   • No hay incidentes NCS ni métricas OTP.  
   • Verbatims: “buen servicio” pero críticas reiteradas al espacio y ruido en CRJ.  
   • Alcance: afecta a todos los vuelos Business SH / IB operados en CRJ.  

7) Causa: Insatisfacción de clientes corporativos de América Norte en Business Short Haul YW  
A. Naturaleza del driver  
   • Hipótesis: expectativas de servicio corporativo no satisfechas (lounge, prioridad, conectividad).  
B. Evidencia consolidada y alcance  
   • Segmento analizado: Global / SH / Business / YW (30 verbatims).  
   • NPS día 0.0 vs baseline 21.69 (–21.69 pts).  
   • No hay datos NCS ni OTP.  
   • Verbatims: positivos en puntualidad, pero quejas por falta de Wi-Fi premium y atención a ejecutivos.  
   • Alcance: todos los subsegmentos Business SH / YW, concentrado en viajeros Business/Work de América Norte.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

En el análisis del 28-11-2025 se detectaron movimientos de NPS significativos en varios niveles. Economy Long Haul se hundió de 15.19 a 0.57 (–14.62 pts) y Premium Long Haul de 30.15 a 6.67 (–23.48 pts), mientras que Business Long Haul se mantuvo estable (34.82→38.71, +3.90 pts). En Short Haul, Economy IB cayó de 32.11 a 30.11 (–2.00 pts) y Business IB bajó de 55.72 a 51.85 (–3.86 pts). El mayor desplome se dio en Business SH YW (21.69→0.00, –21.69 pts), mientras Economy SH YW subió de 43.47 a 51.26 (+7.79 pts). Estas variaciones responden a causas localizadas: deficiencias de confort en A350 C en LIM-MAD (Global), experiencias codeshare insatisfactorias en MAD-MIA para Economy LH, configuración “next” del A350 en EZE-MAD para Premium LH, expectativas de servicio de negocios incumplidas en LHR-MAD para SH/Economy IB, sesgo de muestra de Centroamérica en AGP-MAD para SH/Economy YW, mala percepción de la flota CRJ en SH/Business IB y carencia de Wi-Fi premium y lounge para corporativos de América Norte en SH/Business YW.

Las rutas más afectadas incluyen LIM-MAD (NPS –31.6, A350 C), MAD-MIA (–20.0, Economy LH), EZE-MAD (16.7, Premium LH), GRX-MAD (11.1, SH) y AGP-MAD (60.0, Economy SH YW). Los grupos más reactivos fueron residentes de América Central y América Norte, viajeros Business/Work y usuarios de flota CRJ, A321 y A350 C en operaciones codeshare.

ECONOMY SH IB/YW: desempeño mixto por compañía  
La cabina Economy de SH mantuvo un NPS de 36.60 (28-11-2025) con una variación de +1.0 pt vs L7d. Economy SH IB cayó 2.0 pts (32.11→30.11) por insatisfacción de pasajeros Business en LHR-MAD (NPS 23.5), sin reportes NCS ni datos OTP; en contraste, Economy SH YW subió 7.8 pts (43.47→51.26) impulsado por un sesgo de respuesta de clientes de América Central en AGP-MAD (NPS 60), con verbatims unánimes sobre servicio y atención.

BUSINESS SH IB/YW: fuerte caída por flota y perfil corporativo  
El segmento Business de SH registró un NPS de 31.11 (28-11-2025) y perdió 13.9 pts vs L7d. Business SH IB descendió 3.9 pts (55.72→51.85) debido a la percepción negativa sobre la flota CRJ (NPS –5.9, n=17), mientras Business SH YW se desplomó 21.7 pts (21.69→0.00) por la insatisfacción de clientes corporativos de América Norte (NPS –28.6), con críticas a la falta de Wi-Fi premium y servicios de lounge.

ECONOMY LH: caída marcada en rutas codeshare  
La cabina Economy de LH evidenció una fuerte caída, con un NPS de 0.57 (28-11-2025) y –14.6 pts vs L7d. La ruta MAD-MIA (NPS –20.0, n=5) concentró el impacto, donde pasajeros Business denunciaron problemas de confort y servicios en vuelos operados con BA/AA en flotas A321, A333 y A350 C, sin incidencias NCS ni métricas OTP que lo justifiquen.

BUSINESS LH: desempeño estable  
La cabina Business de LH mantuvo un NPS de 38.71 (28-11-2025), mejorando 3.9 pts vs L7d. No se registraron cambios significativos ni fallos operativos; las valoraciones sobresalientes en puntualidad y atención a bordo compensaron cualquier desviación.

PREMIUM LH: deterioro en configuración “next”  
El segmento Premium de LH cayó de 30.15 a 6.67 (–23.5 pts vs L7d). La ruta EZE-MAD (NPS 16.7, n=6) reveló insatisfacción de residentes de América Central con la ergonomía del A350 “next”, pese a la ausencia de incidentes formales y datos operativos.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-11-27 to 2025-11-27:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

A. Economy Short Haul  
– Existen ambos subnodos (IB y YW), cada uno con anomalía negativa. Ambos comparten drivers (feedback mayoritariamente positivo, ausencia de incidentes operativos) y atribuyen la caída a variabilidad muestral en rutas y perfiles específicos. Convergencia de patrones → causa común a la cabina Economy SH.  

B. Business Short Haul  
– Existen ambos subnodos (IB y YW) pero IB registra anomalía negativa mientras YW está dentro de lo normal. Divergencia en desempeño → causa específica de la compañía IB en la cabina Business SH.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul (SH):  
Las dos cabinas presentan anomalía negativa, pero con comportamientos distintos – Economy SH cae de forma homogénea en IB y YW, mientras que Business SH solo se resiente en IB (YW mantiene NPS normal). Esto indica causas específicas a cada cabina (y para Business, incluso a la compañía IB), no un factor común al radio SH.

Long Haul (LH):  
Economy LH se mantiene dentro de lo normal, Business LH muestra un pico positivo y Premium LH sufre una caída severa. Patrón completamente divergente entre clases: las causas operativas y de experiencia impactan de forma distinta por cabina, y Economy LH actúa como amortiguador frente a las variaciones.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Solo Short Haul registra un impacto neto negativo (–4.9 pts) con causas de variabilidad muestral en Economy y un sesgo en IB Business, mientras que Long Haul, pese a su Premium hundido (–35.7 pts) y Business al alza (+9.6 pts), queda en rango normal por compensación interna. A nivel Global, la caída de 2.4 pts refleja principalmente la debilidad de SH, amortiguada parcialmente por el impulso de LH/Business y reforzada en parte por la baja de LH/Premium, configurando un efecto mixto.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

1. Causa: Muestra reducida y sesgo de perfiles en Short Haul Economy  
A. Naturaleza de la causa  
   – Driver: variabilidad muestral en rutas de bajo volumen y concentración de detractores en perfiles específicos (Asia, determinadas flotas).  
B. Evidencia consolidada y alcance  
   – Segmento más grande: Global/SH/Economy (NPS 30.465 vs baseline 35.626).  
   – “Esta causa afecta a todos los subsegmentos bajo Global/SH/Economy (IB y YW).”  
   – Métricas clave:  
     • Rutas críticas: DSS–MAD con NPS 0.0 (n=3)  
     • Residencia Asia: NPS –20.0 (n=5)  
     • Flota A321: NPS 10.9 (muestra reducida)  
     • Total verbatims: 579, dominados por comentarios positivos de puntualidad y personal amable  
   – Output causal (Global/SH/Economy) resumen:  
     “No se registraron incidentes operativos; la caída de –5.16 pts obedece a un puñado de detractores en DSS–MAD y a viajeros de Asia en A321. Recomendación: investigar la experiencia de Asia en A321 y en DSS–MAD.”  

2. Causa: Experiencia inconsistente en Business SH operado por IB  
A. Naturaleza de la causa  
   – Driver: heterogeneidad de servicio en flota A350 C y en vuelos code-share con BA, generando insatisfacción puntual.  
B. Evidencia consolidada y alcance  
   – Segmento más grande: Global/SH/Business/IB (NPS 48.28 vs baseline 55.72).  
   – “Esta causa impacta exclusivamente al subsegmento IB dentro de Global/SH/Business.”  
   – Métricas clave:  
     • Ruta principal: MAD–MXP con NPS 66.7 (n=3)  
     • Flota A350 C: NPS –33.3 (clientes claramente disconformes)  
     • Code-share BA: NPS –25.0  
     • Total verbatims: 35, con elogios generales al servicio salvo en estos casos  
   – Output causal (Global/SH/Business/IB) resumen:  
     “No hay incidentes operativos; la baja de –7.44 pts se concentra en clientes de A350 C y vuelos BA. Recomendaciones: auditar la experiencia en A350 C y code-share BA.”  

3. Causa: Excelente percepción de servicio en Long Haul Business  
A. Naturaleza de la causa  
   – Driver: alto nivel de confort, atención de tripulación y puntualidad en la cabina Business LH.  
B. Evidencia consolidada y alcance  
   – Segmento más grande: Global/LH/Business (NPS 44.44 vs baseline 34.82).  
   – “Esta causa afecta a todos los viajes en cabina Business de Long Haul.”  
   – Métricas clave:  
     • Ruta de muestra: EZE–MAD con NPS 40.0 (n=5)  
     • Flota A332: NPS 75.0 (n=4)  
     • Región América Centro: NPS 100.0 (n=3)  
     • Total verbatims: 26, destacan confort y amabilidad  
   – Output causal (Global/LH/Business) resumen:  
     “Anomalía positiva de +9.62 pts sin incidencias operativas. Se recomienda reforzar mejoras en A350 y salones VIP para replicar el estándar A332.”  

4. Causa: Insatisfacción del perfil Business/Work en Long Haul Premium  
A. Naturaleza de la causa  
   – Driver: deficiencias específicas en la experiencia Premium para viajeros de negocios en ruta MAD–SJO.  
B. Evidencia consolidada y alcance  
   – Segmento más grande: Global/LH/Premium (NPS –5.56 vs baseline 30.15).  
   – “Esta causa impacta a todos los subsegmentos bajo Global/LH/Premium (Business/Work y Leisure).”  
   – Métricas clave:  
     • Ruta crítica: MAD–SJO con NPS 25.0 (n=4)  
     • Perfil Business/Work: NPS –83.3 (n=6)  
     • Leisure: NPS 33.3 (n=12)  
     • Total verbatims: 25, con menciones positivas pero fuerte insatisfacción de negocios  
   – Output causal (Global/LH/Premium) resumen:  
     “Caída de –35.7 pts sin incidentes operativos formales; se concentra en clientes Business/Work de MAD–SJO. Recomendación: contactar a los 6 clientes insatisfechos y auditar la experiencia Premium en esa ruta.”

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
El 27-nov-2025 detectamos dos dinámicas opuestas en nuestro NPS. En Short Haul se produjo una caída neta de 4.86 puntos (de 36.41 a 31.55 vs L7d), impulsada por Economy SH que retrocedió 5.16 puntos (de 35.63 a 30.47) y Business SH que cedió 0.55 puntos (de 44.99 a 44.44 vs L7d). En Long Haul, el NPS global fue normal (+0.26 pts, de 18.31 a 18.56 vs L7d) porque la subida de 9.62 pts de Business LH (de 34.82 a 44.44) compensó parcialmente el hundimiento de 35.70 pts de Premium LH (de 30.15 a ‑5.56 vs L7d).  
Las causas principales se concentran en:  
• Global/SH/Economy (30.47, –5.16): variabilidad muestral y sesgo de perfiles en rutas de bajo volumen (DSS–MAD con NPS 0.0, residentes en Asia con NPS –20.0), sin incidentes operativos.  
• Global/SH/Business/IB (48.28, –7.44): experiencia inconsistente en flota A350 C (NPS –33.3) y vuelos code-share BA (NPS –25.0) en MAD–MXP, sin fallos operativos.  
• Global/LH/Business (44.44, +9.62): confort, puntualidad y atención excepcional en EZE–MAD (NPS 40.0), especialmente en A332 (75.0) y entre residentes en América Centro (100.0).  
• Global/LH/Premium (–5.56, –35.70): alta insatisfacción de Business/Work en MAD–SJO (–83.3), pese a verbatims positivos del resto, sin incidentes formales.  

Las rutas más afectadas fueron DSS–MAD y MAD–OPO en Economy SH, MAD–MXP en Business SH, MAD–SJO en Premium LH y EZE–MAD en Business LH. Los grupos más reactivos incluyen residentes en Asia (NPS global hasta –42.9), viajeros de negocios en A350 C (NPS –33.3), clientes Business/Work en Premium LH (–83.3) y usuarios de ocio en flota CRJ.  

ECONOMY SH: Variabilidad muestral en rutas de bajo volumen  
La cabina Economy SH retrocedió 5.16 puntos vs L7d, pasando de un NPS de 35.63 a 30.47 el 27-nov-2025. IB bajó de 32.11 a 29.30 (–2.81) y YW de 43.47 a 32.48 (–10.98). La causa principal fue la concentración de detractores en rutas de muestra reducida como DSS–MAD (NPS 0.0, n=3) y entre residentes en Asia (–20.0), sin incidentes NCS que la justifiquen, lo que apunta a un sesgo de composición en vuelos A321.  

BUSINESS SH: Impacto localizado en IB  
El segmento Business SH cedió 0.55 puntos vs L7d, de 44.99 a 44.44 el 27-nov-2025. IB cayó de 55.72 a 48.28 (–7.44) mientras YW se mantuvo estable en 28.57 (+6.89, dentro del rango). Este deterioro se explica por experiencias inconsistentes en A350 C (NPS –33.3) y code-share con BA (–25.0) en ruta MAD–MXP, sin fallos operativos reportados.  

ECONOMY LH: Desempeño estable  
La cabina Economy LH mantuvo desempeño estable, registrando un NPS de 18.32 el 27-nov-2025 (+3.13 pts vs L7d). No se detectaron cambios significativos en verbatims ni incidentes operativos, consolidando niveles consistentes de satisfacción en puntualidad y confort.  

BUSINESS LH: Picos de satisfacción  
Business LH experimentó una subida de 9.62 puntos vs L7d, pasando de 34.82 a 44.44 el 27-nov-2025. Los drivers principales fueron el alto confort de cabina, la atención de la tripulación y la puntualidad, especialmente evidentes en la ruta EZE–MAD (NPS 40.0) y en flota A332 (75.0), con residentes en América Centro alcanzando 100.0.  

PREMIUM LH: Hundimiento por viajeros de negocio  
Premium LH registró un desplome de 35.70 puntos vs L7d, de 30.15 a –5.56 el 27-nov-2025. La causa dominante fue la insatisfacción de clientes Business/Work en la ruta MAD–SJO (NPS –83.3 en 6 respuestas), sin incidentes formales, lo que evidencia deficiencias en la experiencia Premium para este perfil.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-11-26 to 2025-11-26:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

Economy Short Haul (SH)  
• Existen ambos nodos (IB y YW) con comportamientos divergentes:  
  – IB muestra variación normal (+4.0 pts) sin drivers de impacto.  
  – YW presenta anomalía negativa (–2.6 pts) impulsada por bajo desempeño en flota ATR, pasajeros de América Norte y la ruta IBZ-VLC.  
→ Diagnóstico: causa específica de compañía (YW).  

Business Short Haul (SH)  
• Existen ambos nodos (IB y YW) con patrones distintos:  
  – IB mantiene variación normal (+1.4 pts) sin evidencia operativa adversa.  
  – YW reporta anomalía positiva (+22.1 pts) por feedback muy favorable de tripulación y servicio.  
→ Diagnóstico: causa específica de compañía (YW).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul: Economy SH y Business SH presentan patrones divergentes (anomalía negativa en YW-Economy vs positiva en Business), por lo que la causa es específica de cabinas y no común a todo el radio.  
Long Haul: Premium y Business muestran anomalías positivas mientras Economy LH se mantiene normal, evidenciando un patrón específico por cabina con Economy actuando como amortiguador.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Causa mixta: ambos radios muestran anomalías pero con drivers distintos (en SH la caída de YW-Economy contrasta con el alza de Business-YW, mientras que en LH Business y Premium suben consistentemente sobre una Economy neutra), y a nivel Global el fuerte impulso del Long Haul y del SH Business supera la baja en Economy SH/YW, generando la anomalía positiva agregada (+4.7 pts).

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

Causa 1: Experiencia deteriorada en Economy SH/YW  
A. Naturaleza de la causa  
  • Hipótesis: Insatisfacción ligada a flota ATR con menor confort, bajo rendimiento de clientes de América Norte y factores puntuales en la ruta IBZ–VLC.  
B. Evidencia consolidada y alcance  
  • Segmento mayor afectado: Global/SH/Economy/YW (anomalía –2.58 pts). Afecta a todos los pasajeros dentro de Economy Short Haul operados por YW.  
  • Métricas clave:  
    – NPS día: 40.88 vs baseline 43.47  
    – Sin incidentes NCS reportados  
    – Ruta con NPS más bajo: IBZ–VLC, NPS 33.3 (n=3)  
    – Flota ATR: NPS 31.6 (n=?); CRJ: NPS 42.1  
    – Región América Norte: NPS –25 (muestra reducida)  
  • Verbatim representativo: “El avión ATR es muy estrecho y el embarque tardó más de lo esperado.”  

Causa 2: Experiencia excepcional en Business SH/YW  
A. Naturaleza de la causa  
  • Hipótesis: Valor percibido muy alto por atención de tripulación, puntualidad y calidad de catering en vuelos gestionados por YW.  
B. Evidencia consolidada y alcance  
  • Segmento mayor afectado: Global/SH/Business/YW (anomalía +22.06 pts). Afecta a todos los pasajeros Business Short Haul operados por YW.  
  • Métricas clave:  
    – NPS día: 43.75 vs baseline 21.69  
    – Sin incidentes NCS reportados  
    – 23 verbatims, todos positivos  
    – No hay rutas con mínimo de encuestas para desglose adicional  
  • Verbatim representativo: “Tripulación de Business impecable: puntualidad y menú de primera.”  

Causa 3: Servicio de alto nivel en Business LH  
A. Naturaleza de la causa  
  • Hipótesis: Puntualidad, confort de cabina y amabilidad de la tripulación impulsaron la satisfacción en vuelos de largo radio Business.  
B. Evidencia consolidada y alcance  
  • Segmento mayor afectado: Global/LH/Business (anomalía +13.33 pts). Afecta a todos los pasajeros Business Long Haul.  
  • Métricas clave:  
    – NPS día: 48.15 vs baseline 34.82  
    – Sin incidentes NCS reportados  
    – Ruta con mayor impacto: EZE–MAD, NPS 100 (n=3)  
    – 45 verbatims positivos: embarque organizado y comodidad de asiento  
  • Verbatim representativo: “Vuelo a tiempo, butacas muy cómodas y servicio excepcional.”  

Causa 4: Experiencia premium sobresaliente en Premium LH  
A. Naturaleza de la causa  
  • Hipótesis: Combina modernidad de flota A350, alta calidad de atención y puntualidad que elevan de forma atípica el NPS.  
B. Evidencia consolidada y alcance  
  • Segmento mayor afectado: Global/LH/Premium (anomalía +28.19 pts). Afecta a todos los pasajeros Premium Long Haul.  
  • Métricas clave:  
    – NPS día: 58.33 vs baseline 30.15  
    – Sin incidentes NCS reportados  
    – Ruta disponible: MAD–MEX, NPS 33.3 (n=3)  
    – Flota A350: NPS 57.1 (n=7) vs A350 next: 33.3 (n=3)  
    – Regiones: América Centro y España, NPS 66.7; América Sur, NPS 40.0  
    – CodeShare IB: NPS 58.3 (n=12)  
  • Verbatim representativo: “La suite Premium en A350 es inmejorable: puntualidad y detalles de lujo.”

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El Global experimentó una subida de +4.67 pts, pasando de un NPS de 30.47 a 35.14 el 26-nov-2025, impulsado por el sólido desempeño de Long Haul Business (34.82→48.15, +13.33 pts) y Premium (30.15→58.33, +28.19 pts), así como por el fuerte repunte de Short Haul Business YW (21.69→43.75, +22.06 pts). Esta mejora se vio parcialmente contrarrestada por la caída en Short Haul Economy YW (43.47→40.88, –2.58 pts). La subida en LH Business y Premium responde a una experiencia de tripulación muy valorada, puntualidad y confort de cabina, mientras que el alza en SH Business YW se asocia a comentarios elogiosos sobre atención y catering. La baja en SH Economy YW refleja quejas por incomodidad de la flota ATR, tiempos de embarque y menor satisfacción de residentes en América Norte.

Las rutas más afectadas fueron EZE–MAD, que alcanzó un NPS de 100 pts en Business LH (n=3); MAD–MEX, con 33.3 pts en Premium LH (n=3); e IBZ–VLC, con 33.3 pts en Economy SH YW. Los grupos más reactivos incluyen pasajeros en flota ATR y residentes en América Norte en Economy SH YW, y usuarios de flota A350 y viajeros de ocio en Premium y Business LH.

ECONOMY SH: Rendimiento mixto IB vs YW  
La cabina Economy SH registró un NPS agregado de 37.91 pts el 26-nov-2025 (+2.29 pts vs L7d). IB obtuvo 36.12 pts con +4.01 pts vs L7d, manteniendo desempeño estable, mientras que YW cayó a 40.88 pts (–2.58 pts vs L7d). El descenso en YW se vinculó a las molestias en aviones ATR (NPS 31.6), al bajo índice entre residentes de América Norte (–25 pts) y al desempeño de la ruta IBZ–VLC (33.3 pts, n=3), según verbatims sobre espacio y embarque.

BUSINESS SH: Impulso por YW  
El segmento Business SH alcanzó un NPS consolidado de 52.27 pts el 26-nov-2025 (+7.28 pts vs L7d). IB se mantuvo en 57.14 pts (+1.43 pts vs L7d) con feedback estable, mientras YW mejoró de 21.69 a 43.75 pts (+22.06 pts vs L7d). Este avance en YW refleja verbatims que destacan la atención de tripulación, puntualidad y calidad de catering, sin incidentes reportados.

ECONOMY LH: Estable  
La cabina Economy LH mantuvo desempeño estable con un NPS de 18.99 pts el 26-nov-2025 (+3.80 pts vs L7d), sin variaciones significativas en puntualidad o servicio y sin incidentes reportados.

BUSINESS LH: Notable mejora  
La cabina Business LH experimentó una mejora de +13.33 pts, pasando de 34.82 a 48.15 pts el 26-nov-2025. No hubo incidentes NCS y la ruta EZE–MAD destacó con NPS 100 pts (n=3), sustentada en comentarios sobre puntualidad, embarque organizado y confort de asiento.

PREMIUM LH: Salto excepcional  
El segmento Premium LH saltó de 30.15 a 58.33 pts (+28.19 pts vs L7d) el 26-nov-2025. La flota A350 registró 57.1 pts (n=7) frente a 33.3 pts en A350 next (n=3), y la ruta MAD–MEX obtuvo 33.3 pts (n=3). Pasajeros de América Central y España alcanzaron 66.7 pts, impulsando la mejora gracias a puntualidad, modernidad de cabina y servicio de tripulación.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-11-25 to 2025-11-25:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

A. Economy Short Haul  
- Existen ambos nodos: IB (Normal +0.8 pts) y YW (Anomalía positiva +9.4 pts).  
- Diagnóstico: causa específica de compañía en YW, dado que sus drivers y magnitud de anomalía divergen claramente de IB.

B. Business Short Haul  
- Existen ambos nodos: IB (Anomalía negativa –1.9 pts) y YW (Normal +3.3 pts).  
- Diagnóstico: causa específica de compañía en IB, puesto que solo IB presenta desviación negativa mientras YW mantiene rendimiento esperado.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul  
El patrón es específico de cabina: Economy SH mantiene rendimiento normal (solo YW muestra subida puntual) mientras Business SH registra anomalía negativa, y esta divergencia se mantiene en ambas compañías (IB siempre por debajo de YW).

Long Haul  
El patrón es también específico de cabina: únicamente Business LH presenta la anomalía negativa, mientras Economy LH y Premium LH permanecen dentro de la variación esperada, sin una progresión uniforme de reactividad entre clases.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

El impacto es mixto: ambos radios registran anomalías pero con drivers distintos (SH impulsa al alza por Economy/YW pese a caídas en Business/IB, mientras LH sufre por Business), y ese balance divergente se refleja en un Global ligeramente positivo que atenúa los efectos negativos de LH.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

Causa A: Rendimiento Excepcional en Short-Haul Economy (YW)  
A. Naturaleza de la causa  
• Hipótesis: Ejecución operativa sobresaliente en puntos críticos (check-in ágil, puntualidad y amabilidad de tripulación) que potencia la satisfacción de los pasajeros de Economy YW.  

B. Evidencia consolidada y alcance  
• Segmento analizado: Global/SH/Economy/YW (NPS: 52.89 vs baseline 43.47, +9.43 pts; n=151 verbatims)  
• Afecta a todos los vuelos bajo SH – Economy/YW  
• Métricas clave:  
  – Incidentes reportados: 0  
  – Rutas:  
    • MAD–TLS: NPS 42.9 (n=7), única excepción sin incidente formal  
  – Flota:  
    • ATR: NPS 90.9  
    • CRJ: NPS 49.1  
• Verbatims representativos: “Vuelo salió a tiempo y el personal fue muy atento”, “Check-in rápido, embarque sin esperas”, “Servicio a bordo impecable”.  

Causa B: Insatisfacción en Short-Haul Business (IB)  
A. Naturaleza de la causa  
• Hipótesis: Desajuste entre expectativas de cliente Business y oferta en rutas IB, derivado de mayor uso de aeronaves A320 (con equipamiento más limitado) y concentración de pasajeros desde España.  

B. Evidencia consolidada y alcance  
• Segmento analizado: Global/SH/Business/IB (NPS: 53.85 vs baseline 55.72, –1.87 pts; n=37 verbatims)  
• Afecta a todos los vuelos bajo SH – Business/IB  
• Métricas clave:  
  – Incidentes reportados: 0  
  – Rutas:  
    • MAD–VIE: NPS 100.0 (n=3)  
  – Flota:  
    • A320: NPS 16.7  
    • A320neo: NPS 64.3  
  – Perfil cliente:  
    • Origen España: NPS 44.4 vs Europa 61.5  
    • Tipo de viaje Business/Work: 57.1 vs Leisure 50.0  
• Verbatims representativos: “Asiento demasiado estrecho en A320”, “Parecía un Economy mejorado, no un Business”.  

Causa C: Bajo desempeño en Long-Haul Business  
A. Naturaleza de la causa  
• Hipótesis: Fricciones latentes para pasajeros Business en larga distancia (conectividad, confort puntual en A350), no reflejadas en quejas directas pero sí en valoración global.  

B. Evidencia consolidada y alcance  
• Segmento analizado: Global/LH/Business (NPS: 20.0 vs baseline 34.82, –14.82 pts; n=43 verbatims)  
• Afecta a todos los vuelos bajo LH – Business  
• Métricas clave:  
  – Incidentes reportados: 0  
  – Rutas:  
    • GRU–MAD: NPS 100.0 (n=3)  
    • No hay rutas con NPS bajo identificadas de forma explícita  
  – Flota:  
    • A350: NPS –20.0  
    • Otras flotas: hasta +66.7  
  – Perfil cliente: Business/Work –28.6 vs Leisure 38.9  
• Verbatims representativos: “El catering y la tripulación perfectos, pero la conexión Wi-Fi fue pésima”, “Temperatura y ruido en cabina A350 afectaron descanso.”

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

En el día 2025-11-25 el NPS Global subió de 30.47 a 32.30 puntos (mejora de 1.82 pts vs L7d), impulsado por un feedback unánime sobre puntualidad y amabilidad en flotas ATR y CRJ. A nivel de radio, Short Haul mostró una ligera caída en Business (44.99 → 44.74, –0.25 pts) debido al subsegmento Business/IB (55.72 → 53.85, –1.87 pts) que sufrió desajustes en asientos A320 y menor satisfacción de pasajeros desde España, mientras que Economy/YW logró un fuerte repunte (43.47 → 52.89, +9.43 pts) por la excelencia operativa en check-in y servicio a bordo. En Long Haul, el NPS descendió de 18.31 a 17.67 (–0.63 pts) principalmente por Business LH (34.82 → 20.00, –14.82 pts), donde la flota A350 y los viajeros Business/Work mostraron las mayores fricciones operativas.

Las rutas más críticas incluyen HAM–MAD con un NPS de 0.0 (n=10) en el Global, y MAD–TLS en SH/Economy/YW con 42.9 pts (n=7) pese a no registrar incidencias formales. Entre los grupos de clientes, los más reactivos han sido los pasajeros Business/Work de Larga Distancia, los usuarios de flota A350 (LH) y A320 (SH/IB), así como los residentes en España y Norteamérica, que explican la mayoría de las bajadas de NPS.

ECONOMY SH  
La cabina Economy de SH experimentó una subida de NPS de 3.0 puntos, pasando de 35.63 a 38.63 vs L7d. IB mantuvo desempeño estable con un NPS de 32.89 (vs L7d 32.11; +0.78), mientras que YW lideró el alza con un salto de 9.43 pts (43.47 → 52.89), sustentado en 151 verbatims que resaltaron la puntualidad de salida, agilidad en check-in y trato de la tripulación. La única excepción local fue la ruta MAD–TLS (42.9, n=7), que sugiere oportunidades de mejora puntual.

BUSINESS SH  
El segmento Business de SH registró un ligero retroceso de 0.25 puntos, pasando de 44.99 a 44.74 vs L7d. Este comportamiento se explica por el descenso de 1.87 pts en IB (55.72 → 53.85), vinculado al uso intensivo de aviones A320 (NPS 16.7 vs 64.3 de A320neo) y menor satisfacción de residentes en España; YW, en cambio, mejoró 3.3 pts (21.69 → 25.00) pero no compensó la caída de IB.

ECONOMY LH  
La cabina Economy de LH mantuvo desempeño estable, registrando un NPS de 15.61 (vs L7d 15.19; +0.40). No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción y sin incidencias o quejas destacadas en los verbatims.

BUSINESS LH  
La cabina Business de LH sufrió una caída de 14.82 puntos, pasando de 34.82 a 20.00 vs L7d. Los drivers principales fueron la baja valoración de pasajeros Business/Work (–28.6 vs 38.9 de Leisure) y el pobre desempeño de la flota A350 (–20.0), con quejas latentes en conectividad Wi-Fi y confort de cabina, a pesar de no registrarse incidentes formales.

PREMIUM LH  
El segmento Premium de LH mantuvo desempeño estable, con un NPS de 35.29 (vs L7d 30.15; +5.15). No se detectaron cambios significativos, reflejando una satisfacción sostenida en rutas y perfiles habituales.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-11-24 to 2025-11-24:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

A. Economy Short Haul  
- Existen ambos nodos SH/Economy/IB y SH/Economy/YW.  
- IB muestra comportamiento normal (+3.1 pts), mientras que YW arrastra la caída (-8.4 pts) con drivers y patrones de mix de cliente claramente distintos.  
Diagnóstico: la causa es específica de la compañía YW en Economy SH.

B. Business Short Haul  
- Existen ambos nodos SH/Business/IB y SH/Business/YW.  
- Tanto IB (+0.3 pts) como YW (+1.8 pts) operan dentro del rango normal, compartiendo ausencia de incidencias y feedback positivo.  
Diagnóstico: la causa es común a la cabina Business SH (no atribuible a una sola compañía).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul  
Patrón divergente entre cabinas:  
- Economy SH sufre la caída (-0,5 pts) exclusivamente por el nodo YW (–8,4 pts), mientras que IB se mantiene normal.  
- Business SH presenta una leve caída común a ambas compañías (IB +0,3 pts; YW +1,8 pts), actuando como amortiguador.  

Long Haul  
Patrón claramente específico de cabina:  
– Economy LH registra un fuerte alza (+8,2 pts) por la excepcional experiencia en ocio.  
– Business LH (–18,7 pts) y Premium LH (–12,0 pts) padecen caídas marcadas, cada una con drivers operativos y de flota diferentes.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Diagnóstico global: las causas son mixtas y compensatorias: Short Haul sufre una leve caída focalizada en Economy YW, mientras que en Long Haul conviven subidas en Economy y caídas en Business/Premium que se neutralizan entre sí, resultando en un NPS global ligeramente al alza.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

1. Causa: Satisfacción excepcional del segmento Leisure en Economy Long Haul  
A. Naturaleza de la causa  
   • Hipótesis: la elevada percepción de confort y servicio en vuelos de ocio impulsa el NPS de Economy LH por encima del comportamiento habitual.  
B. Evidencia consolidada y alcance  
   • Segmento analizado: Global / Long Haul / Economy (n=265 comentarios)  
   • Output causal:  
     – NPS diario: 23.35 vs baseline 15.19 (+8.16 pts)  
     – Rutas: MAD–SJU con NPS 33.3 (6 respuestas), sin incidencias operativas  
     – Perfil: Leisure NPS 34.3 (n=134) vs Business/Work –21.2 (n=33)  
     – CodeShare: AA 75.0 pts, QR –40.0 pts, Others –100.0 pts  
     – Región: Norteamérica 80.0 pts, Europa –10.5 pts  
     – Flota A33ACMI: –27.3 pts  
     – Feedback representativo: “El vuelo fue muy cómodo, con excelente atención y oferta de comidas.”  
   • Alcance: afecta a todos los pasajeros subsegmento Economy LH.

2. Causa: Incidencias de flota A350 en Business Long Haul  
A. Naturaleza de la causa  
   • Hipótesis: variabilidad en la experiencia a bordo de la A350 (configuración, servicio, equipamiento) genera insatisfacción sostenida.  
B. Evidencia consolidada y alcance  
   • Segmento analizado: Global / Long Haul / Business (n=49 comentarios)  
   • Output causal:  
     – NPS diario: 16.13 vs baseline 34.82 (–18.69 pts)  
     – Ruta con datos: MAD–SCL NPS 66.7 (3 respuestas), sin reportes operativos  
     – Flota: A350 NPS –14.3 (14 encuestas) vs A350-next +60.0 (5)  
     – Perfil de viaje: Business/Work 5.6 pts, Leisure 28.6 pts  
     – CodeShare BA: 0.0 pts  
     – Feedback representativo: “Asientos de A350 demasiado estrechos y con desgaste evidente.”  
   • Alcance: afecta a todos los pasajeros subsegmento Business LH.

3. Causa: Gestión inadecuada de pausas de tripulación en Premium Long Haul  
A. Naturaleza de la causa  
   • Hipótesis: falta de respeto al horario de descanso de pasajeros en primera fila junto a la salida de emergencia impacta la percepción de Premium.  
B. Evidencia consolidada y alcance  
   • Segmento analizado: Global / Long Haul / Premium (n=11 comentarios)  
   • Output causal:  
     – NPS diario: 18.18 vs baseline 30.15 (–11.96 pts)  
     – Ruta con datos: MAD–MEX NPS 33.3 (3 respuestas)  
     – Región: España 50.0 pts vs resto de Europa –25.0 pts  
     – Flota: A350-next 50.0 pts vs A350 16.7 pts  
     – Tipología: Business 33.3 pts / Leisure 18.2 pts  
     – Feedback representativo: “La tripulación conversaba junto a mi asiento, impidiéndome descansar.”  
   • Alcance: afecta a todos los pasajeros subsegmento Premium LH.

4. Causa: Composición de mix y dispersión geográfica en Economy Short Haul (compañía YW)  
A. Naturaleza de la causa  
   • Hipótesis: predominio de viajeros de ocio y alta variabilidad regional (p.ej. América Norte) provoca la caída del NPS en Economy SH de YW, sin relación con operaciones.  
B. Evidencia consolidada y alcance  
   • Segmento analizado: Global / Short Haul / Economy / YW (n=162 comentarios)  
   • Output causal:  
     – NPS diario: 35.04 vs baseline 43.47 (–8.42 pts)  
     – Rutas: MAD–OVD NPS 25.0 (4 respuestas), sin incidencias  
     – Tipo de viaje: Leisure 27.2 pts (n=92) vs Business/Work 49.0 pts (n=49)  
     – Región: América Norte 0.0 pts (n=4) vs América Sur 50.0 pts (n=6)  
     – Feedback representativo: “El vuelo estaba bien, pero esperaba más comodidad en el trayecto corto.”  
   • Alcance: afecta a todos los pasajeros de YW en Economy SH.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

En la jornada del 24-nov, detectamos subidas y bajadas de NPS muy localizadas por cabina y compañía. En Short Haul, Economy pasó de 35.63 a 35.12 (−0.50 pts vs L7d) debido a la caída de Iberia City-Y (YW) de 43.47 a 35.04 (−8.42 pts vs L7d), mientras que Iberia (IB) mantuvo desempeño estable en 35.16 (+3.05 pts vs L7d). Business SH descendió de 44.99 a 42.86 (−2.13 pts vs L7d) sin que IB (56.0, +0.28 pts) ni YW (23.53, +1.84 pts) mostraran quejas operativas o de servicio. En Long Haul, Economy se disparó de 15.19 a 23.35 (+8.16 pts vs L7d) impulsada por pasajeros Leisure y la alta valoración de vuelos AA; Business LH cayó de 34.82 a 16.13 (−18.69 pts vs L7d) por la baja experiencia en la flota A350; y Premium LH bajó de 30.15 a 18.18 (−11.96 pts vs L7d) por problemas puntuales de pausas de tripulación en primera fila.

Las rutas más afectadas reflejan estos hallazgos: en SH Economy YW, MAD–OVD obtuvo un NPS de 25.0; en LH Economy, MAD–SJU alcanzó 33.3; y en Premium LH, MAD–MEX quedó en 33.3. Por perfiles, los viajeros de ocio en Economy LH y los pasajeros de América del Norte en SH Economy YW fueron los más reactivos, al igual que los usuarios de A350 en Business LH y los residentes europeos en Premium LH.

ECONOMY SH YW: Descenso por mix de ocio y variabilidad regional  
La cabina Economy de SH registró un NPS de 35.12 (24-nov) con una bajada de 0.50 puntos vs L7d. En Iberia (IB) se mantuvo estable en 35.16 (+3.05 pts vs L7d), mientras que en Iberia City-Y (YW) bajó de 43.47 a 35.04 (−8.42 pts vs L7d). La causa principal fue la alta proporción de viajeros de ocio (Leisure 27.2 pts, n=92) y la dispersión geográfica, con América Norte en 0.0 pts (n=4). Esta caída se reflejó especialmente en la ruta MAD–OVD (NPS 25.0), donde no hubo incidencias reportadas.

BUSINESS SH: Leve retroceso sin causas operativas  
El segmento Business de SH registró un NPS de 42.86 (24-nov) con una bajada de 2.13 puntos vs L7d. Iberia (IB) cerró en 56.0 (+0.28 pts vs L7d) y City-Y (YW) en 23.53 (+1.84 pts vs L7d). No se detectaron incidentes operativos ni feedback negativo significativo, por lo que la caída responde a ruido estadístico y muestra limitada. La ruta MAD–ZRH destacó con un NPS de 100 (n=3), y los perfiles Business/Work y Leisure mantuvieron satisfacciones consistentes.

ECONOMY LH: Impulso por confort de ocio  
La cabina Economy de LH experimentó un NPS de 23.35 (24-nov), con una subida de 8.16 puntos vs L7d. Este aumento se explica por la elevada satisfacción de pasajeros Leisure (NPS 34.3, n=134), residentes en América del Norte (80.0), y codeshare AA (75.0), sin incidencias operativas registradas. La ruta MAD–SJU alcanzó 33.3 (n=6), apoyando el alza general.

BUSINESS LH: Deterioro por experiencia en A350  
La cabina Business de LH cayó hasta 16.13 (24-nov), perdiendo 18.69 puntos vs L7d. Los drivers principales fueron las bajas valoraciones en flota A350 (–14.3 pts, n=14) frente a A350-next (60.0 pts, n=5). En MAD–SCL, el NPS llegó a 66.7 (n=3) sin incidentes. Los viajeros Business/Work valoraron 5.6 pts, muy por debajo de Leisure (28.6 pts), apuntando a una experiencia a bordo desigual en la A350.

PREMIUM LH: Impacto por gestión de descanso  
El segmento Premium de LH registró un NPS de 18.18 (24-nov), con una caída de 11.96 puntos vs L7d. La causa dominante fue la incorrecta gestión de pausas de tripulación junto a la salida de emergencia, reflejada en comentarios como “la tripulación conversaba junto a mi asiento, impidiéndome descansar”. MAD–MEX obtuvo 33.3 (n=3); España valoró 50.0 pts vs resto de Europa –25.0 pts; y la flota A350-next superó con 50.0 pts a la A350 con 16.7 pts.

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