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

En Short Haul Economy, el escenario es SINERGIA `(-, – | –)`.  
- **Narrativa:** La caída conjunta de IB y YW impulsó la anomalía negativa de –6,2 pts en Economy SH. Ambos subsegmentos registraron un alza de mishandling y misconexiones, lo que sumado a un aumento de retrasos y otras incidencias operativas impactó de forma homogénea al agregado.  
- **Evidencia Clave:**  
  • Mishandling ↑ 3,5 incidentes (operative_data_tool)  
  • Misconnections ↑ 0,2 incidentes (operative_data_tool)  
  • Otras_incidencias +16 y retrasos +12 (ncs_tool)  

En Short Haul Business, el escenario es DILUCIÓN `(N, + | N)`.  
- **Narrativa:** El fuerte impulso positivo de YW no logró imponerse en el agregado Business SH porque IB se mantuvo estable, neutralizando el efecto. Por tanto, la preocupación principal es la mejora en YW, con IB amortiguando la variación.  
- **Evidencia Clave:**  
  • YW: +23,0 pts por excelencia en servicio de cabina (drivers: Cabin Crew, Arrivals experience, Aircraft interior; verbatims centrados en amabilidad de la tripulación)  
  • IB: +0,3 pts, desempeño operativo estable sin variaciones significativas

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es TRANSFERENCIA `(-, N | -)`  
- **Narrativa:** La debacle operativa de Economy SH (–6,2 pts) arrastró al conjunto de Short Haul, pese a que Business SH se mantuvo dentro de la variación normal. La caída de puntualidad y el aumento de mishandling y misconexiones en Economy contagió el NPS del radio completo.  
- **Evidencia:**  
  • OTP15 disminuyó 0,7–4,0 pts y Punctuality SHAP negativo en Economy (operative_data_tool).  
  • Mishandling ↑ 3,5 incidencias y Misconnections ↑ 0,2 (operative_data_tool).  
  • Otras_incidencias +16 y Retrasos +12 vs. L7d (ncs_tool).  

En Long Haul, la dinámica es DOMINANCIA `(-, -, + | -)`  
- **Narrativa:** El resultado de Long Haul está dictado por la fuerte anomalía negativa de Business LH (–20,2 pts), que superó la mejora de Premium (+7,0 pts) y arrastró el NPS global de LH a –10,4 pts.  
- **Evidencia:**  
  • Business LH: OTP15 cayó 4,0 pts y Punctuality SHAP –8,189 (operative_data_tool + explanatory_drivers_tool).  
  • Aumento de cancelaciones (+3), mishandling (+3,5) y misconexiones (+0,2) en Business (operative_data_tool + ncs_tool).  
  • Premium LH compensó parcialmente con drivers de producto positivos, pero no impidió la caída dominada por Business.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es SINERGIA `(-, - | -)`.  
- **Narrativa:** Tanto el Long Haul como el Short Haul experimentaron caídas de NPS por problemas operativos similares (puntualidad, equipaje y conexiones), lo que desembocó en una anomalía negativa sistémica en toda la red.  
- **Evidencia:**  
  • Punctuality SHAP = –1,323 y OTP15 cayó 0,5 pts (operative_data_tool)  
  • Mishandling subió 3,5 incidentes y misconnections +0,2 (operative_data_tool)  
  • Total incidentes operativos ↑118% (retrazos, reroutes, cancelaciones) según ncs_tool  
  • Ruta más impactada: BIO–VLC con NPS –50,0 (4 pax)

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Disrupciones operativas (puntualidad, equipaje y conexiones)  
- NMA: Global  
- Afecta a:  
  • Global  
  • Long Haul (LH) → Economy, Business, Premium  
  • Short Haul (SH) → Economy, Business, IB, YW  
- Qué falló:  
  • Caída de puntualidad (OTP15 global –0,5 pts; LH –4,0 pts; SH –0,7 pts; SHAP puntualidad negativa)  
  • Aumento de mishandling (+3,5 incidentes)  
  • Aumento de misconexiones (+0,2 incidentes)  
- Dónde (Top 5 rutas con peor NPS):  
  1. BIO–VLC: –50,0 (4 pax)  
  2. MAD–OSL: –37,5 (8 pax)  
  3. GVA–MAD: –25,0 (4 pax)  
  4. MAD–OPO:   6,7 (15 pax)  
  5. DBV–MAD:  33,3 (6 pax)  
- Quién (perfiles con mayor spread de ΔNPS):  
  • Fleet: 95,1 pts  
  • Residence Region: 82,2 pts  
  • CodeShare: 80,3 pts  
  • Business/Leisure: 5,7 pts  
- Evidencia clave:  
  • NPS Global 22,57 vs 30,12 (–7,54 pts)  
  • Incidentes totales +118% (retrazos, reroutes, cancelaciones)  

CAUSAS AISLADAS  
1) Servicio de cabina excepcional en YW SH  
- NMA: Global/SH/Business/YW  
- Afecta a: YW SH  
- Qué “falló” (positivo): drivers de producto — Cabin Crew, Arrivals experience, Aircraft interior, Ticket Price, IFE, Lounge  
- Dónde:  
   • CMN–MAD: NPS 100,0 (3 pax)  
   • AGP–MAD: NPS   0,0 (3 pax)  
   • MAD–PNA: NPS 100,0 (3 pax)  
   • ALC–MAD: NPS  50,0 (4 pax)  
   • MAD–MUC: NPS   0,0 (4 pax)  
- Quién: Residence Region (spread 29,7 pts)  
- Evidencia: NPS +23,02 pts vs L7d; verbatims destacan amabilidad de tripulación  

2) Mejora de drivers de producto en Premium LH  
- NMA: Global/LH/Premium  
- Afecta a: Global/LH/Premium  
- Qué “falló” (positivo): drivers como Check-in, Cabin Crew, Aircraft interior, IB Plus, Lounge, Wi-Fi, Food&Beverage  
- Dónde:  
   • MAD–MIA: NPS 40,0 (5 pax)  
   • MAD–SCL: (datos consolidados)  
   • MAD–SJO: NPS –33,3 (3 pax)  
   • MAD–MCO: NPS 33,3 (3 pax)  
   • BOG–MAD: NPS 26,7 (15 pax)  
- Quién:  
   • CodeShare: 141,7 pts  
   • Residence Region: 104,6 pts  
   • Fleet: 36,8 pts  
- Evidencia: NPS +7,02 pts vs L7d; incremento en drivers de producto sin validación operativa (falta de verbatims/datos de uso)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
Durante la semana del 2025-11-29 al 2025-12-02, el NPS Global cayó de 30,12 a 22,57 (–7,54 pts) por la combinación de un fuerte desplome en Long Haul (17,21→6,84, –10,37 pts) y un deterioro en Short Haul (36,47→31,48, –4,99 pts). En Long Haul, Economy pasó de 13,63 a 3,19 (–10,43 pts) y Business de 32,70 a 12,50 (–20,20 pts), mientras Premium repuntó de 29,60 a 36,62 (+7,02 pts). En Short Haul, Economy IB bajó de 33,30 a 27,75 (–5,55 pts) e Economy YW de 42,07 a 34,62 (–7,45 pts), y aunque Business IB se mantuvo estable (47,25→47,57, +0,32 pts) y Business YW creció de 20,45 a 43,48 (+23,02 pts), el mal desempeño de Economy arrastró el radio. Las caídas se atribuyen principalmente a disrupciones operativas —caída de puntualidad (OTP15 hasta –4,0 pts), mishandling (+3,5 incidentes) y misconexiones (+0,2)—, mientras que las subidas en Premium LH y en YW Business SH obedecieron a excelentes drivers de producto y servicio de cabina.

Las rutas más afectadas fueron BIO–VLC (–50,0, 4 pax), MAD–OSL (–37,5, 8 pax) y GVA–MAD (–25,0, 4 pax), contrastando con DBV–MAD (+33,3, 6 pax) o MAD–OPO (+6,7, 15 pax). Los pasajeros más reactivos respondieron al tipo de flota (spread hasta 148,6 pts), a la región de residencia (hasta 170,5 pts en SH IB) y al code-share (hasta 141,7 pts en Premium LH), mientras que la categorización Business vs Leisure mostró variaciones moderadas (<15 pts).

ECONOMY SH: Impacto de disrupciones operativas  
La cabina Economy SH IB registró un NPS de 27,75 (2025-11-29→2025-12-02), con una caída de 5,55 pts vs L7d (33,30→27,75), y YW anotó 34,62 con –7,45 pts (42,07→34,62). El principal motor de esta bajada fue el incremento de mishandling (+3,5 incidentes) y misconexiones (+0,2), junto a un descenso de puntualidad (OTP15 –0,7), validado por +12 retrasos y +16 otras incidencias en ncs_tool. Este deterioro se concentró en rutas como BIO–VLC (–50,0), MAD–OSL (–37,5) y GVA–MAD (–25,0); los perfiles más sensibles son Fleet (spread 148,6 pts), Residence Region (170,5 pts) y CodeShare (95,4 pts).

BUSINESS SH: Estabilidad en la Clase Business a Corto Radio  
La cabina Business SH IB mantuvo desempeño estable con un NPS de 47,57 (2025-11-29→2025-12-02) y un leve alza de 0,32 pts vs L7d (47,25→47,57), mientras Business YW subió de 20,45 a 43,48 (+23,02 pts). No se detectaron variaciones operativas significativas en IB. El espectacular avance de 23,02 pts en YW se fundamentó en la excelencia del servicio de cabina (Cabin Crew, Arrivals experience, Aircraft interior) y comentarios muy positivos, que contrarrestaron un leve empeoramiento de puntualidad (OTP15 –2,48) y un alza de incidencias. Las rutas más destacadas fueron CMN–MAD (100,0), MAD–PNA (100,0) y ALC–MAD (50,0), y los viajeros según Residence Region (spread 29,7 pts) mostraron la mayor sensibilidad.

ECONOMY LH: Caída pronunciada por puntualidad y cancelaciones  
La cabina Economy LH registró un NPS de 3,19 (2025-11-29→2025-12-02), con un descenso de 10,43 pts vs L7d (13,63→3,19). La raíz de la caída fue la pérdida de puntualidad (OTP15 –4,0; SHAP –1,842), corroborada por +3 cancelaciones, +10 retrasos y +3,5 mishandling. Las mayores afectaciones se dieron en IAD–MAD (–71,4), MAD–ORD (–64,7) y JFK–MAD (–46,5); los perfiles más reactivos fueron Fleet (113,2 pts), CodeShare (98,2 pts) y Residence Region (37,1 pts).

BUSINESS LH: Deterioro por caída de puntualidad y acumulación de incidencias  
La cabina Business LH presentó un NPS de 12,50 (2025-11-29→2025-12-02), con un desplome de 20,20 pts vs L7d (32,70→12,50). El driver dominante fue la baja puntualidad (SHAP –8,189; OTP15 –4,0), potenciada por +3 cancelaciones, +3,5 mishandling y +0,2 misconexiones. MAD–ORD (–28,6), JFK–MAD (–16,7) y MAD–SCL mostraron los peores resultados; Residence Region (spread 123,5 pts) y CodeShare (103,7 pts) lideraron la reactividad.

PREMIUM LH: Mejora impulsada por drivers de producto  
El segmento Premium LH subió de 29,60 a 36,62 (+7,02 pts vs L7d). Aun con un empeoramiento operativo en puntualidad (SHAP –6,675; OTP15 –2,48) y +45 eventos críticos, la satisfacción se apoyó en drivers de producto como Check-in, Aircraft interior, Cabin Crew, IB Plus, Lounge y Wi-Fi. MAD–MIA (40,0), MAD–MCO (33,3) y BOG–MAD (26,7) mostraron avances, y los más sensibles fueron CodeShare (141,7 pts), Residence Region (104,6 pts) y Fleet (36,8 pts).

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*



**ANÁLISIS DIARIO SINGLE:**
📅 2025-12-02 to 2025-12-02:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Economy SH, el escenario es SINERGIA `(-, - | -)`.  
- Narrativa: La caída de –10.4 pts en Economy SH responde a un efecto conjunto de ambos subgrupos. Tanto IB como YW muestran anomalías negativas que se refuerzan mutuamente, transmitiéndose íntegramente al agregado.  
- Evidencia Clave: Insatisfacción concentrada en la ruta MAD–SDR (NPS 0.0) y perfiles de Oriente Medio, América Norte y clientes en code-share AA, sin incidencia de fallos operativos.

En Business SH, el escenario es DOMINANCIA `(-, + | +)`.  
- Narrativa: Aunque IB registró una ligera caída, la excepcional subida de +43.6 pts en YW impuso su efecto positivo al total de Business SH. Se adopta la explicación de YW como causa principal, matizando que IB moderó parcialmente el impacto.  
- Evidencia Clave: YW impulsada por puntualidad superior (OTP15_adjusted 91.19) y feedback muy positivo en servicio a bordo (amabilidad de tripulación, calidad de la comida, salidas/llegadas puntuales).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es DOMINANCIA `(-, + | -)`.  
- Narrativa: El descenso de –7.7 pts en SH está dictado por la fuerte caída en Economy SH, a pesar de la mejora en Business SH que no logró compensar.  
- Evidencia: Insatisfacción concentrada en la ruta MAD–SDR (NPS 0.0) y en pasajeros de Oriente Medio, América Norte y en code-share AA, que arrastraron el agregado.

En Long Haul, la dinámica es DOMINANCIA `(+, –, + | +)`.  
- Narrativa: El alza de +7.2 pts en LH responde sobre todo al fuerte empuje de Premium (+13.0 pts), mientras que la caída en Business LH moderó parcialmente el efecto.  
- Evidencia: Premium LH brilló por el excepcional feedback en MAD–MEX (NPS 100.0), la calidad de servicio, atención a movilidad reducida y confort de flota moderna.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es DOMINANCIA `(+, – | –)`.  
- Narrativa: El NPS Global se vio arrastrado por la fuerte caída en Short Haul, pese al buen desempeño en Long Haul.  
- Evidencia: La insatisfacción concentrada en la ruta MAD–SDR (NPS 0.0) y entre pasajeros de Oriente Medio, América Norte y code-share AA en SH fue más que suficiente para neutralizar y superar la mejora de LH.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Insatisfacción concentrada en Economy Short Haul  
- NMA: Global  
- Afecta a:  
  • Global  
  • SH (Short Haul)  
  • SH / Economy  
  • SH / Economy / IB  
  • SH / Economy / YW  
- Qué falló: La experiencia en Economy SH se deterioró sin incidentes operativos graves; la insatisfacción se concentró en segmentos con code-share AA y en flotas A319/A333.  
- Dónde (rutas):  
  1. MAD–SDR (NPS 0.0, 7 respuestas)  
  2. EAS–MAD (NPS 0.0, 6 respuestas)  
- Quién (perfiles):  
  • Residence Region: Oriente Medio (–33.3), América Norte (–41.7)  
  • Code-Share: AA (–55.6)  
- Evidencia operativa y cualitativa:  
  • NPS SH/Economy 21.64 vs baseline 32.09 (–10.45 pts)  
  • Load Factor 86.2 (–0.35 pts), OTP15_adjusted 92.17 (+2.72 pts)  
  • 430 verbatims mayoritariamente positivos en servicio, pero con focos de insatisfacción en las rutas señaladas  
  • 0 incidentes NCS reportados  

CAUSA 2: Excelencia en Premium Long Haul  
- NMA: Long Haul  
- Afecta a:  
  • LH (Long Haul)  
  • LH / Premium  
- Qué falló (realmente brilló): Máxima satisfacción por calidad de personal, atención a movilidad reducida y confort de flota moderna.  
- Dónde (ruta): MAD–MEX (NPS 100.0, 5 respuestas)  
- Quién (perfiles): Pasajeros Premium, especialmente con necesidades especiales de movilidad; flota moderna (A350)  
- Evidencia operativa y cualitativa:  
  • NPS LH/Premium 47.83 vs baseline 34.81 (+13.02 pts)  
  • Load Factor 88.33 (–2.30 pts), OTP15_adjusted 76.06 (–2.84 pts)  
  • 39 verbatims 100 % positivos  
  • 0 incidentes NCS  

CAUSA 3: Sobresaliente experiencia en Business Short Haul / YW  
- NMA: Business SH  
- Afecta a:  
  • SH / Business  
  • SH / Business / YW  
- Qué falló (realmente se potenció): Percepción muy positiva por puntualidad superior y servicio a bordo.  
- Dónde: no hay rutas con volumen suficiente, pero LHR–MAD (NPS 100.0, 3 respuestas) registra máxima satisfacción  
- Quién (perfiles):  
  • Flota CRJ (NPS 76.9)  
  • Residence Region España (NPS 84.2)  
- Evidencia operativa y cualitativa:  
  • NPS SH/Business/YW 76.92 vs baseline 33.36 (+43.57 pts)  
  • Load Factor 54.43 (–3.88 pts), OTP15_adjusted 91.19 (+3.38 pts)  
  • 15 verbatims 100 % positivos  
  • 0 incidentes NCS  

CAUSAS AISLADAS  
1. Caída en Business Long Haul (LH / Business)  
   - NMA: LH / Business  
   - Qué falló: Descenso de –11.13 pts por perfiles concretos con baja satisfacción (América Norte, A350 next, code-share AA)  
   - Quién: América Norte (–41.2), A350 next (–25.0), AA (–25.0)  
   - Evidencia: NPS 12.90 vs baseline 24.03; Load Factor 93.83; OTP15_adjusted 76.06; 0 NCS; verbatims sin quejas operativas  

2. Variación leve en Business SH / IB  
   - NMA: SH / Business / IB  
   - Qué falló: Heterogeneidad de percepción entre regiones (España vs Europa) y flotas (A320 vs A320neo)  
   - Quién: Europa (–20.0), flota A320 (0.0)  
   - Evidencia: NPS 45.83 vs baseline 48.00 (–2.17 pts); Load Factor 77.51; OTP15_adjusted 93.25; 0 NCS; verbatims mayoritariamente positivos.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El NPS Global se situó en 23.46 el 02-dic, cayendo 2.55 puntos desde 26.01 vs L7d, apuntalado por la fuerte pérdida en Short Haul (33.14→25.44, –7.70) que superó la ganancia en Long Haul (12.51→19.66, +7.15). En SH, Economy se desplomó de 32.09 a 21.64 (–10.45), con IB bajando de 29.65 a 23.53 (–6.12) y YW de 37.11 a 17.82 (–19.29), mientras Business subió de 43.53 a 56.76 (+13.23) gracias a YW (33.36→76.92, +43.56) que compensó la leve caída de IB (48.00→45.83, –2.17). En LH, Economy escaló de 8.94 a 16.13 (+7.19), Premium de 34.81 a 47.83 (+13.02) y Business retrocedió de 24.03 a 12.90 (–11.13).

Las rutas más afectadas en SH Economy fueron MAD–SDR y EAS–MAD, ambas con NPS 0.0, mientras que en LH Premium la ruta MAD–MEX alcanzó NPS 100.0. El feedback más reactivo provino de pasajeros residentes en Oriente Medio y América Norte (scores negativos en SH Economy/IB/YW), y del grupo Premium y Leisure en flota moderna (A350, A350 next) y clientes de Centroamérica y España (scores positivos en LH Economy y SH Business/YW).

ECONOMY SH: Caída pronunciada en IB y YW  
La cabina Economy de SH registró un NPS de 21.64 el 02-dic, con una caída de 10.45 puntos vs L7d. IB anotó 23.53 (–6.12 vs 29.65 L7d) y YW 17.82 (–19.29 vs 37.11 L7d). La causa principal fue la insatisfacción concentrada en MAD–SDR y EAS–MAD (ambas NPS 0.0), especialmente entre pasajeros de Oriente Medio (–33.3) y América Norte (–41.7) en vuelos code-share AA. No hubo incidentes NCS; la operación fue puntual (OTP15 +2.72 pts) con factor de carga estable (–0.35 pts), lo que apunta a fricciones en confort y producto de asientos.

BUSINESS SH: Impulso extraordinario de YW  
El segmento Business SH mostró un NPS de 56.76 el 02-dic, mejorando 13.23 puntos vs L7d (43.53). IB mantuvo desempeño estable con 45.83 (–2.17 vs 48.00 L7d), y YW destacó con 76.92 (+43.56 vs 33.36 L7d). Este alza se explica por puntualidad superior (OTP15 +2.72 pts) y feedback 100 % positivo en la ruta LHR–MAD (NPS 100.0), especialmente entre pasajeros en CRJ y residentes en España.

ECONOMY LH: Mejora sostenida  
La cabina Economy de LH alcanzó un NPS de 16.13 el 02-dic, subiendo 7.19 puntos vs L7d (8.94). El alza responde a elogios a la tripulación y puntualidad (OTP15 76.06), con outlier positivo en MAD–SDQ (NPS 33.3). Persisten focos de mejora en flotas A350 C (–66.7) y A321XLR (–30.0) y entre pasajeros de América Norte (–33.3).

BUSINESS LH: Retroceso relevante  
La cabina Business de LH registró un NPS de 12.90 el 02-dic, con una caída de 11.13 puntos vs L7d (24.03). A pesar de alta ocupación (LF 93.83) y sin incidentes NCS, la insatisfacción se concentró en América Norte (–41.2), vuelos con A350 XLR (–25.0) y code-share AA (–25.0).

PREMIUM LH: Desempeño sobresaliente  
El segmento Premium de LH brilló con un NPS de 47.83 el 02-dic, mejorando 13.02 puntos vs L7d (34.81). La causa dominante fue la excelencia en servicio a bordo y atención a movilidad reducida en MAD–MEX (NPS 100.0), con 39 verbatims 100 % positivos y sin incidentes operativos.

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

En Economy SH, el escenario es SINERGIA (-, - | -).  
- **Narrativa:** Adopta la explicación del nodo padre: la caída de 8.7 pts se explica por un subconjunto reducido de vuelos y perfiles críticos que afectan por igual a IB y YW.  
- **Evidencia Clave:** Ruta MAD–ZRH con NPS 0.0 (n=4) y pasajeros de América Norte en Economy SH con NPS –37.5, además de vuelos code-share AA con NPS –60.0.

En Business SH, el escenario es DOMINANCIA (-, + | -).  
- **Narrativa:** Adopta la explicación del hijo dominante (IB): la insatisfacción en IB arrastra el resultado agregado, aunque el alza de YW mitiga parcialmente la magnitud de la pérdida.  
- **Evidencia Clave:** Ruta MAD–ORY en IB con NPS 25.0 (n=4) y viajeros Business/Work con NPS –28.6, frente a YW que sube +22.2 pts.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul (SH), la dinámica es SINERGIA (Economy –, Business – | SH –).  
- Narrativa: Adopta la explicación del nodo padre SH: no hay señales de fallos operativos ni quejas en tierra o a bordo, y la caída de 8.7 pts responde a cambios en el mix de clientes y posibles deficiencias en la experiencia en tierra (check‐in/embarque) o servicios complementarios que no aparecen en los verbatims.  
- Evidencia: Volumen alto de comentarios (439) sin incidencias NCS, con rutas críticas como MAD–ZRH (NPS 0.0) y perfiles code‐share/América Norte rezagados.

En Long Haul (LH), la dinámica es DOMINANCIA (Economy –, Business –, Premium + | LH –).  
- Narrativa: Adopta la explicación de la cabina dominante (Business LH): la fuerte caída de –29.0 pts en Business, impulsada por la percepción negativa del servicio de cabina y catering en vuelos A333 (ruta MAD–MEX) para viajeros de Norteamérica y Europa, arrastra al conjunto de LH, pese al repunte de Premium.  
- Evidencia: Nodo Global/LH/Business –5.0 pts vs baseline 24.0 pts; quejas sobre actitud de tripulación y oferta de entretenimiento en MAD–MEX.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es SINERGIA `(-, - | -)`.  
- **Narrativa:** Adopta la explicación del nodo Global: la red completa sufrió una caída de –7.52 pts en NPS por factores comunes que impactaron tanto a largo como a corto radio.  
- **Evidencia:**  
  • Ruta LHR–MAD con NPS –26.3 (n=19)  
  • Viajeros de negocio con NPS 5.5 vs ocio 23.4  
  • Flotas A321XLR (–40.0) y A350 C (–37.5)  
  • Pasajeros en código compartido (AA, I2, QR) y residentes en Norteamérica/Europa con NPS negativos.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Descenso sistémico de NPS  
- NMA: Global  
- Afecta a: Global y todos sus sub-segmentos (LH, SH y todas sus cabinas)  
- Qué falló: Débil alineación de producto con las expectativas de viajeros de negocio y code-share en rutas críticas; falta de adaptación de servicios a bordo en flotas A321XLR y A350 C.  
- Dónde:  
  • LHR–MAD (NPS –26.3, n=19)  
  • BOS–MAD (NPS 0.0, n=6)  
  • EZE–MAD (NPS 12.5, n=24)  
- Quién:  
  • Viajeros de negocio (NPS 5.5 vs ocio 23.4)  
  • Residentes en Norteamérica y Europa (NPS negativos)  
  • Pasajeros en código compartido con AA, I2 y QR (NPS muy bajos)  
- Evidencia:  
  • NPS Global 18.49 vs baseline 26.01 (–7.52)  
  • Load Factor 85.88 (–0.89) y OTP15_adjusted 90.45 (+2.38) sin incidentes NCS  
  • 775 verbatims positivos en puntualidad y tripulación, sin quejas operativas  

CAUSA 2: Insatisfacción en Business de Largo Radio  
- NMA: Global/LH  
- Afecta a: Global/LH y Global/LH/Business  
- Qué falló: Actitud percibida poco profesional de la tripulación de cabina y deficiencias en catering y entretenimiento en vuelos A333.  
- Dónde:  
  • MAD–MEX (Business LH en A333, NPS 33.3 n=3)  
- Quién:  
  • Viajeros Business/Work de Norteamérica (NPS –60.0) y Europa (NPS –100.0)  
- Evidencia:  
  • NPS Business LH –5.0 vs baseline 24.03 (–29.03)  
  • OTP15_adjusted 77.96 (–1.12), Load Factor 93.66 (–0.32)  
  • Sin NCS; verbatims críticos sobre servicio de cabina y oferta de productos  

Causas aisladas (no burbujean hacia arriba)  
- Premium de Largo Radio (Global/LH/Premium):  
  • NMA: Global/LH/Premium  
  • Éxito en puntualidad y servicio de tripulación que dispara NPS a 62.5 vs baseline 34.8 (+27.7)  
  • Ruta MAD–MEX (NPS 75.0, n=4) y 22 verbatims unánimemente positivos  
  
- Business Short Haul Joven-Work (Global/SH/Business/YW):  
  • NMA: Global/SH/Business/YW  
  • Satisfacción elevada (NPS 55.6 vs baseline 33.4, +22.2) por puntualidad y atención al pasajero  
  • Volumen pequeño, sin rutas ni NCS reseñables

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
El 1 de diciembre de 2025 la red registró una caída global de NPS, pasando de 26.01 pts en la media de los últimos 7 días a 18.49 pts (–7.52 pts). Esta pérdida se distribuye de forma transversal: en Largo Radio el NPS bajó de 12.51 pts a 8.87 pts (–3.64 pts) con especial deterioro en Economy LH (de 8.94 pts a 5.39 pts, –3.55 pts) y un desplome crítico en Business LH (de 24.03 pts a –5.0 pts, –29.03 pts), mientras que Premium LH destacó con un salto de 34.81 pts a 62.50 pts (+27.69 pts). En Corto Radio el NPS global descendió de 33.14 pts a 24.46 pts (–8.68 pts), con Economy SH cayendo de 32.09 pts a 23.41 pts (–8.68 pts) y Business SH de 43.53 pts a 35.71 pts (–7.81 pts), pese al repunte de YW en ese segmento (de 33.36 pts a 55.56 pts, +22.20 pts). Identificamos como causa raíz en la red la falta de alineación del producto y servicio con las necesidades de viajeros de negocio y clientes code-share en flotas A321XLR y A350 C, con quejas puntuales de actitud de tripulación y catering en Business LH.

Las rutas más impactadas fueron LHR–MAD (NPS –26.3, n=19), BOS–MAD (0.0 pts, n=6) y EZE–MAD (12.5 pts, n=24) en Economy LH; MAD–MEX en Business LH con NPS 33.3 pts (n=3) y MAD–ORY en Business SH con 25.0 pts (n=4); en Economy SH destaca MAD–ZRH con 0.0 pts (n=4). Los perfiles más reactivos incluyen viajeros Business/Work de Norteamérica y Europa (NPS hasta –100 pts), residentes en España, pasajeros en código compartido con AA (hasta –60 pts) y clientes Leisure en Premium LH (NPS 72.7 pts).  

ECONOMY SH (IB y YW)  
La cabina Economy de SH sufrió una bajada de satisfacción: IB pasó de 29.65 pts a 22.28 pts (–7.37 pts vs L7d) y YW de 37.11 pts a 25.22 pts (–11.89 pts vs L7d), resultando un NPS agregado de 23.41 pts (–8.68 pts). La causa principal fue la concentración de detractores en la ruta MAD–ZRH (0.0 pts), amplificada por pasajeros de América Norte con –37.5 pts y vuelos code-share AA con –60.0 pts.  

BUSINESS SH (IB y YW)  
El segmento Business de SH registró un NPS de 35.71 pts el 1 dic vs 43.53 pts en la media anterior (–7.81 pts). IB cayó de 48.00 pts a 26.32 pts (–21.68 pts) por insatisfacción en la ruta MAD–ORY entre viajeros de negocio, mientras que YW subió de 33.36 pts a 55.56 pts (+22.20 pts), mitigando parcialmente el deterioro agregado.  

ECONOMY LH  
La cabina Economy de LH cayó a 5.39 pts el 1 dic desde 8.94 pts en la media L7d (–3.55 pts). El descontento provino de viajeros Business/Work (–23.8 pts vs Leisure 9.6 pts) y residentes en Europa (–50.0 pts), así como de clientes en flotas A321XLR y A350 C, con especial incidencia en EZE–MAD (12.5 pts).  

BUSINESS LH  
Business de LH sufrió un desplome a –5.0 pts desde 24.03 pts en la media anterior (–29.03 pts), impulsado por percepciones negativas del servicio de cabina, catering y entretenimiento en vuelos A333 de la ruta MAD–MEX, afectando fundamentalmente a clientes de Norteamérica (–60.0 pts) y Europa (–100.0 pts).  

PREMIUM LH  
El NPS Premium de LH saltó de 34.81 pts a 62.50 pts (+27.69 pts). La mejora se explica por la fuerte satisfacción de pasajeros Leisure (72.7 pts) en MAD–MEX, destacando puntualidad, eficiencia de tripulación y calidad de servicio a bordo, sin incidencias operativas ni NCS reportados.

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

En Economy SH, el escenario es CANCELACIÓN (IB –, YW + | PADRE N).  
- Narrativa: Mientras IB registró una caída de –1.9 pts por insatisfacción concentrada en la ruta MAD–VGO, especialmente en vuelos code-share (BA, VY, LATAM, AA) y con flota A319/“Unknown”, YW se disparó +11.6 pts gracias a una OTP15 extraordinaria y valoraciones muy positivas de la tripulación. Estos efectos opuestos se neutralizan en el agregado, mostrando un NPS “normal”.  
- Evidencia Clave:  
  • IB: NPS 0.0 en MAD–VGO, código compartido y flota A319/Unknown  
  • YW: OTP15_adjusted +3.79 pts vs baseline; NPS 48.68 impulsado por puntualidad y tripulación

En Business SH, el escenario es SINERGIA (IB –, YW – | PADRE –).  
- Narrativa: La caída de –9.4 pts en Business SH obedece al mismo conjunto de drivers en ambos subsegmentos: un mix elevado de vuelos code-share con BA, asignación de flotas CRJ/A321 y menor satisfacción de viajeros Business. Esa sinergia negativa en IB y YW se traslada íntegramente al nodo padre.  
- Evidencia Clave:  
  • Code-share BA: IB NPS –66.7 (n=3)  
  • Flotas CRJ/A321: YW NPS 18.2/16.7; fuerte incidencia en la valoración promedio

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es DILUCIÓN (Business –, Economy N | SH N).  
- Narrativa: el desempeño de Short Haul está marcado por la caída de Business SH (–9.4 pts), originada en un mix elevado de vuelos code-share BA y asignación de flotas CRJ/A321 con menor satisfacción de viajeros Business. Este impacto se vio atenuado por la estabilidad de Economy SH (NPS dentro del rango normal).  
- Evidencia:  
  • Business SH: NPS –9.4 pts, Code-share BA con NPS –66.7; flotas CRJ/A321 con puntuaciones bajas  
  • Economy SH: NPS 34.81 vs baseline 32.09 (normal)

En Long Haul, la dinámica es SINERGIA (Economy –, Business –, Premium – | LH –).  
- Narrativa: la caída de Long Haul (–21.1 pts) responde a un efecto sistémico que afectó a las tres cabinas. Economy LH sufrió por la ruta MAD–MEX, flota A321XLR y code-share QR; Business LH descendió por viajeros Leisure en flotas A333 y regiones Europa/América Norte; Premium LH cayó en clientes de América Centro a bordo de A350 next en MAD–MEX. Todos esos drivers se suman y arrastran el resultado del radio.  
- Evidencia:  
  • Economy LH: NPS –22.6 pts, MAD–MEX (–20.0 pts, A321XLR –63.6, QR –100.0)  
  • Business LH: segmentos Leisure NPS –11.3, A333 con –100.0, regiones Europa/América Norte hasta –80.0  
  • Premium LH: América Centro NPS –50.0, A350 next NPS –16.7 en MAD–MEX

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA (LH –, SH N | GLOBAL –).  
- Narrativa: La caída de –8.09 pts en el NPS global se explica con la misma lógica descrita en el nodo Global: una experiencia muy negativa concentrada en la ruta MAD–SCL generó un impacto sistémico que no fue contrarrestado por el buen desempeño del Short Haul.   
- Evidencia:  
  • Ruta MAD–SCL: NPS –36.4 (n=22)  
  • Perfiles implicados: clientes BA, QR, AA en A321XLR  
  • Resto de la operación (puntualidad, equipaje, servicio) dentro o por encima del baseline

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Experiencia negativa en ruta MAD–SCL  
- NMA: Global  
- Afecta a:  
  • Global  
  • Global/LH (y sus cabinas Economy, Business, Premium)  
  • Global/SH (todas las subdivisiones con anomalía negativa)  
- Qué falló: Experiencia muy negativa en vuelos MAD–SCL que no quedó reflejada en los reportes operativos formales.  
- Dónde: Ruta MAD–SCL (NPS –36.4, n=22)  
- Quién: Pasajeros code-share (BA, QR, AA) viajando en A321XLR (NPS de –63.6)  
- Evidencia:  
  • NPS Global 17.92 vs baseline 26.01 (caída –8.09)  
  • Sin incidentes NCS reportados  
  • Verbatims con quejas de confort y servicio a bordo

CAUSA 2: Insatisfacción en Business Short Haul por vuelos code-share BA y flotas CRJ/A321  
- NMA: Short Haul  
- Afecta a:  
  • Global/SH/Business  
  • Global/SH/Business/IB  
  • Global/SH/Business/YW  
- Qué falló: Mix elevado de vuelos code-share con BA combinado con asignación de aviones CRJ y A321, de confort inferior para clientes Business.  
- Dónde: Operaciones code-share BA (NPS –66.7, n=3)  
- Quién:  
  • Pasajeros Business (NPS 26.7 vs Leisure 37.9)  
  • Flota CRJ (NPS 18.2) y A321 (NPS 16.7)  
- Evidencia:  
  • Business SH NPS 34.09 vs baseline 43.53 (caída –9.43)  
  • Load Factor 70.63 (–2.12 pts) y OTP15_adjusted 92.23 (+2.80 pts)

CAUSA 3: Insatisfacción en Long Haul de viajeros Leisure en ruta LIM–MAD con flotas antiguas y code-share  
- NMA: Long Haul  
- Afecta a:  
  • Global/LH/Economy  
  • Global/LH/Business  
  • Global/LH/Premium  
- Qué falló: Grupo de pasajeros Leisure en la ruta LIM–MAD volando en flotas más antiguas y/o bajo acuerdos de code-share con IB, AY, AA, BA y QR.  
- Dónde: Ruta LIM–MAD (NPS –11.8, n=17)  
- Quién:  
  • Segmento Leisure (NPS –11.3 vs Business 3.5)  
  • Flotas A350, A332, A333, A321XLR con NPS entre –12.9 y –63.6  
  • Code-share QR (NPS –100)  
- Evidencia:  
  • Long Haul NPS –8.60 vs baseline 12.51 (caída –21.11)  
  • Load Factor 88.0 (–2.68 pts) y OTP15_adjusted 78.91 (–0.51 pts)  

CAUSA AISLADA: Dinámica opuesta en Economy Short Haul (IB vs YW)  
- NMA: Global/SH/Economy  
- Afecta a:  
  • Global/SH/Economy/IB  
  • Global/SH/Economy/YW  
- Qué falló: Mientras IB registró insatisfacción por la ruta MAD–VGO, vuelos code-share (BA, VY, LATAM, AA) y flota A319/“Unknown”, YW brilló por puntualidad excelente y fuerte percepción de tripulación.  
- Dónde: Ruta MAD–VGO (IB NPS 0.0 sobre 4 respuestas)  
- Quién:  
  • Pasajeros IB, VY, LATAM y AA en A319/“Unknown” (IB NPS 27.76 vs 29.65)  
  • Viajeros YW de ocio (NPS 53.3 en Leisure; 95–100 en Norteamérica/España)  
- Evidencia:  
  • IB NPS 27.76 vs baseline 29.65 (–1.9)  
  • YW NPS 48.68 vs baseline 37.11 (+11.6)  
  • OTP15_adjusted YW +3.79 pts, sin NCS reportados

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El análisis del 30-11-2025 revela que el NPS Global cayó de 26.01 en L7d a 17.92 (-8.09), impulsado por una experiencia muy negativa en la ruta MAD–SCL (NPS –36.4, n=22) que no quedó reflejada en los KPI operativos formales. En Long Haul, todas las cabinas registraron descensos: Economy LH se desplomó de 8.94 a –13.62 (-22.56) por la ruta MAD–MEX en A321XLR (NPS –63.6) y code-share QR (–100.0); Business LH bajó de 24.03 a 21.21 (-2.82) por la insatisfacción de viajeros Leisure en flotas antiguas (A333, NPS –100.0) y residentes de Europa/Norteamérica; Premium LH pasó de 34.81 a 4.17 (-30.64) centrado en clientes de América Centro a bordo del A350 next en MAD–MEX (NPS –16.7). En Short Haul, el NPS global se mantuvo estable (33.14→34.75, +1.60) gracias a la cancelación interna entre Economy SH y Business SH: Economy SH cerró en 34.81 (+2.73 vs L7d 32.09) como resultado de la caída leve de IB (27.76 vs 29.65, –1.89) en MAD–VGO code-share (BA, VY, LATAM, AA) y el fuerte repunte de YW (48.68 vs 37.11, +11.58) impulsado por una OTP15_adjusted de 91.6 (+3.79) y verbatims positivos; Business SH retrocedió de 43.53 a 34.09 (–9.43) por el alto mix de vuelos code-share BA (IB NPS –66.7) y la asignación de flotas CRJ/A321 con menor confort para clientes Business (YW 18.18 vs 33.36, –15.17).

Las rutas más afectadas fueron MAD–SCL, MAD–MEX, LIM–MAD y MAD–VGO, cada una ligada a drivers distintos: confort en flota A321XLR, amenities del A350 next, antigüedad de flotas A350/A332/A333 y coordinación con socios de code-share (QR, BA, AA). Los grupos de clientes más reactivos incluyen pasajeros Leisure en LH Economy y Business, viajeros de ocio de YW en SH, clientes Business en code-share BA y residentes de América Centro, Europa y Norteamérica.

ECONOMY SH (IB vs YW – Dinámica Opuesta)  
La cabina Economy de SH registró un NPS de 34.81 el 30-11-2025 (vs L7d=32.09, +2.73). IB tuvo una ligera caída a 27.76 (vs L7d=29.65, –1.89) por insatisfacción en la ruta MAD–VGO en vuelos code-share con BA, VY, LATAM y AA y flota A319/“Unknown”. En contraste, YW subió a 48.68 (vs L7d=37.11, +11.58), impulsado por una OTP15_adjusted de 91.6 (+3.79 vs L7d) y verbatims muy positivos sobre puntualidad y tripulación. La compensación interna evitó un impacto neto en SH Economy; los perfiles más reactivos fueron los usuarios de code-share en BA/VY para IB y los viajeros de ocio de América Norte y España para YW.

BUSINESS SH (Caída Consistente)  
El segmento Business de SH cayó a 34.09 el 30-11-2025 (vs L7d=43.53, –9.43), con IB en 39.39 (vs L7d=48.00, –8.61) e YW en 18.18 (vs L7d=33.36, –15.17). Esta retracción se explica por un mix elevado de vuelos code-share BA (IB NPS –66.7) y la asignación de flotas CRJ y A321 con menor confort percibido, lo que impactó de manera homogénea a los viajeros Business. Las rutas BA mostraron descensos agudos y los pasajeros Business fueron los más sensibles a la calidad de la cabina.

ECONOMY LH (Desplome por MAD–MEX y Code-Share)  
Economy de LH registró un NPS de –13.62 el 30-11-2025 (vs L7d=8.94, –22.56). El deterioro se concentró en la ruta MAD–MEX (NPS –20.0, n=15) a bordo de A321XLR con valoración de –63.6 y en vuelos code-share con QR (–100.0). Los pasajeros Leisure, especialmente desde Asia y Norteamérica, mostraron las caídas más pronunciadas, sin que ninguna incidencia operativa formal explicara el fenómeno.

BUSINESS LH (Leisure y Flotas Antiguas)  
La cabina Business de LH pasó a 21.21 el 30-11-2025 (vs L7d=24.03, –2.82). El descenso se atribuye al grupo de viajeros Leisure que voló en flotas antiguas (A333 con NPS –100.0) y a residentes de Europa y Norteamérica, con puntuaciones negativas de hasta –66.7, pese a mantener puntualidad y cero NCS en la operación.

PREMIUM LH (Impacto en A350 next y América Centro)  
Premium de LH cayó a 4.17 el 30-11-2025 (vs L7d=34.81, –30.64). La raíz está en la ruta MAD–MEX con A350 next (NPS –16.7) y en clientes de América Centro (–50.0). Aunque la puntualidad fue en línea con L7d y los verbatims no recogieron quejas de servicio, la percepción de amenidades en el A350 next dislocó la satisfacción de este segmento.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-11-29 to 2025-11-29:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En SH/Economy (IB N, YW N | Padre N)  
- No hay anomalías en ninguno de los nodos (todos “Normal”), por lo que no aplica ninguno de los cinco escenarios de interacción interna.  

En SH/Business (IB +, YW – | Padre +) → DOMINANCIA  
- Narrativa: La fuerte alza del NPS en Business SH se explica principalmente por el rendimiento excepcional de IB (Global/SH/Business/IB), que “arrastró” al nivel padre. El servicio y la puntuación a bordo de IB superan con creces la caída registrada en YW.  
- Evidencia Clave:  
  • IB: NPS +26.1 pts, impulsado por OTP15=92.89 (+1.51 vs. media) y Load Factor=78.98 (–1.93), con verbatims que destacan puntualidad y calidad de tripulación.  
  • YW: NPS –10.3 pts, afectado por Load Factor=55.15 (–3.41) y posible sesgo de muestra, pero su impacto no fue suficiente para contrarrestar el ímpetu de IB.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En SH (Eco, Bus | SH):  
- Dinámica: TRANSFERENCIA (Normal, + | +)  
- Narrativa: La subida del NPS en Short Haul se explica por el fuerte desempeño de la cabina Business, que “contagió” el resultado global pese a la estabilidad de Economy.  
- Evidencia:  
  • Business SH (+14.0 pts) impulsado por OTP15=91.84 (+2.42) y Load Factor=70.74 (–2.03), con verbatims sobresalientes en puntualidad y espacio.  
  • Economy SH se mantuvo en rango normal (NPS +3.2 pts), sin cambios operativos relevantes.  

En LH (Eco, Bus, Prem | LH):  
- Dinámica: CANCELACIÓN (Normal, – , + | Normal)  
- Narrativa: El NPS de Long Haul aparece estable, pero oculta dos fuerzas opuestas: la caída en Business por problemas en la flota A350 en la ruta GRU–MAD fue compensada por el fuerte alza en Premium gracias a la excelente atención de la tripulación.  
- Evidencia:  
  • Business LH (–9.7 pts) afectado por mal rendimiento en A350/GRU–MAD (NPS –28.6 en esa flota).  
  • Premium LH (+15.2 pts) favorecido por feedback positivo en calidad de servicio y menor saturación de cabina.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA (LH Normal, SH Normal | GLOBAL +).  
- Narrativa: El alza del NPS Global no se explica por un efecto agregado de corto o largo radio (ambos se mantuvieron dentro de su rango normal), sino por factores de red que “contagiaron” el resultado superior.  
- Evidencia:  
  • OTP15_adjusted Global en 90.02 (+1.9 vs. media): mejor puntualidad en toda la red  
  • Mishandling en 13.84 (–1.69): menos incidencias de equipaje  
  • Load Factor en 85.97 (–0.85): ligera reducción que mejoró percepción de confort  
  • Verbatims: destacan amabilidad y atención de la tripulación y procesos de viaje ágiles.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Mejora operativa de red (Global Positive Anomaly)  
- NMA: Global  
- Afecta a: todos los segmentos (LH/Economy, LH/Business, LH/Premium, SH/Economy, SH/Business, SH/Business/IB y SH/Business/YW)  
- Qué falló (en este caso, qué mejoró): puntalidad y manejo de equipaje por encima de la referencia  
- Dónde: red completa (no hay rutas específicas con quiebre operacional)  
- Quién: todos los perfiles de cliente, con especial alza en satisfacción de quien valora puntualidad y atención de tripulación  
- Evidencia:  
    • NPS Global 30.08 vs baseline 26.01 (+4.07 pts)  
    • OTP15_adjusted = 90.02 (+1.9 pts vs media)  
    • Mishandling = 13.84 (–1.69 vs media)  
    • Load Factor = 85.97 (–0.85 vs media)  
    • Verbatims: destacan amabilidad de tripulación, agilidad de procesos y puntualidad  

CAUSA 2: Experiencia A350 deficiente en Business Long Haul (LH Business Negative Anomaly)  
- NMA: Global/LH/Business  
- Afecta a: segmento LH Business (ruta GRU–MAD, flota A350)  
- Qué falló: calidad de la cabina A350 en GRU–MAD  
- Dónde: ruta GRU–MAD (NPS –28.6 en A350, n=7)  
- Quién: pasajeros residentes en Europa y Norteamérica (NPS –33.3)  
- Evidencia:  
    • NPS LH Business 14.29 vs baseline 24.03 (–9.75 pts)  
    • OTP15_adjusted = 77.8 (–1.75 vs media)  
    • Load Factor = 93.43 (–0.48 vs media)  
    • Feedback muy positivo en servicio, sin incidencias operativas, lo que apunta a un problema de experiencia de cabina  

CAUSA 3: Servicio excepcional en Premium Long Haul (LH Premium Positive Anomaly)  
- NMA: Global/LH/Premium  
- Afecta a: segmento LH Premium  
- Qué falló (qué potenció la mejora): atención de tripulación y percepción de mayor confort por menor densidad  
- Dónde: no hay rutas con volumen suficiente para aislar, el efecto es transversal en Premium LH  
- Quién: todos los clientes Premium (volumen reducido pero consistente en valoración de servicio)  
- Evidencia:  
    • NPS LH Premium 50.0 vs baseline 34.81 (+15.19 pts)  
    • Load Factor = 87.69 (–2.97 vs media)  
    • OTP15_adjusted = 77.8 (–1.75 vs media)  
    • Verbatims: destacan cordialidad de la tripulación y comodidad de cabina  

CAUSA 4: Impulso IB en Business Short Haul (SH Business Positive Anomaly)  
- NMA: Global/SH/Business  
- Afecta a: SH Business (incluye subsegmentos IB y YW)  
- Qué falló (qué potenció la mejora): excepcional rendimiento de IB en puntualidad y servicio  
- Dónde: ruta LHR–MAD (NPS 100.0, n=4, sin incidentes)  
- Quién: pasajeros de Leisure en A320neo (NPS 81.8) y Business/Work en general  
- Evidencia:  
    • NPS SH Business 57.5 vs baseline 43.53 (+13.97 pts)  
    • OTP15_adjusted = 91.84 (+2.42 vs media)  
    • Load Factor = 70.74 (–2.03 vs media)  
    • Verbatims (n=57): puntualidad, espacio y calidad de servicio  

CAUSA AISLADA: Sesgo de muestra en YW Short Haul Business (SH Business/YW Negative Anomaly)  
- NMA: Global/SH/Business/YW  
- Afecta a: SH Business/YW  
- Qué falló: baja ocupación (Load Factor muy bajo) que generó sesgo en la muestra de NPS  
- Dónde: no existe ruta con suficiente muestra para aislar impacto específico  
- Quién: pasajeros YW en días de baja demanda  
- Evidencia:  
    • NPS SH Business/YW 23.08 vs baseline 33.36 (–10.28 pts)  
    • Load Factor = 55.15 (–3.41 vs media)  
    • OTP15_adjusted = 90.88 (+3.10 vs media)  
    • Verbatims 100% positivos, sin incidencias operativas, sugiere efecto artefacto de muestra.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
La red registró un NPS global de 30.08 puntos, mejorando de 26.01 en la media de los últimos 7 días (+4.07), impulsado por una operación más puntual (OTP15 = 90.02, +1.9) y una reducción de mishandling (13.84, –1.69). Sin embargo, esta mejora engloba subidas y bajadas marcadas en distintos nodos: Business Long Haul cayó de 24.03 a 14.29 (–9.75) debido a la pobre experiencia en A350 en la ruta GRU–MAD (NPS –28.6, n=7), mientras Premium Long Haul ascendió de 34.81 a 50.00 (+15.19) gracias a la atención de cabina y menor carga (Load Factor 87.69, –2.97). En Short Haul, Business subió de 43.53 a 57.50 (+13.97), arrastrado por IB (de 48.00 a 74.07, +26.07) y mitigado solo en parte por la caída de YW (de 33.36 a 23.08, –10.28).

Las rutas más impactadas fueron GRU–MAD, responsable de la caída en Business LH, y LHR–MAD, donde IB Short Haul alcanzó NPS 100.0 (n=4) gracias a la puntualidad y el espacio extra. El tramo MAD–MUC mostró un NPS negativo aislado (–20.0, n=5). Los perfiles más reactivos incluyen a pasajeros Business residentes en Europa y Norteamérica en vuelos LH, y a viajeros Leisure en A320neo en vuelos SH.

ECONOMY SH: Estabilidad confirmada  
La cabina Economy SH mantuvo desempeño estable la semana del 29-11-2025 con un NPS de 35.27 (vs L7d: 32.09, +3.18). Economy SH IB alcanzó 33.54 (vs L7d: 29.65, +3.89) y Economy SH YW 39.68 (vs L7d: 37.11, +2.58). No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción.

BUSINESS SH: Impulso liderado por IB  
El segmento Business SH experimentó un alza de 43.53 a 57.50 (+13.97) con un NPS de 57.50. IB brilló con 74.07 (vs L7d: 48.00, +26.07) gracias a OTP15 = 92.89 (+1.51) y Load Factor = 78.98 (–1.93), especialmente en LHR–MAD (NPS 100.0, n=4), donde los verbatims elogian puntualidad y confort. YW cayó a 23.08 (vs L7d: 33.36, –10.28), un efecto atribuible a sesgo de muestra por baja ocupación (Load Factor 55.15, –3.41).

ECONOMY LH: Sin variaciones críticas  
La cabina Economy LH mantuvo desempeño estable con un NPS de 14.71 (vs L7d: 8.94, +5.77). No se registraron cambios operativos o comentarios relevantes que indiquen alteraciones en la percepción de este segmento.

BUSINESS LH: Caída por experiencia en A350  
Business LH descendió de 24.03 a 14.29 (–9.75), con un NPS de 14.29. La principal causa fue la experiencia a bordo en flota A350 en la ruta GRU–MAD (NPS –28.6, n=7), sin incidencias operativas ni retrasos, pero con un fuerte impacto negativo en pasajeros de Europa y Norteamérica (NPS –33.3).

PREMIUM LH: Subida sostenida por comodidad y servicio  
Premium LH escaló de 34.81 a 50.00 (+15.19), registrando NPS 50.00. Este crecimiento responde a la atención excepcional de la tripulación y a un menor nivel de ocupación (Load Factor 87.69, –2.97), destacándose en verbatims de confort y amabilidad, sin rutas o perfiles negativos significativos.

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