===== STEP 1: SECTION CONNECTIONS =====

--- ECONOMY SH ---
A nivel diario, la caída semanal de Economy SH se concentró en dos jornadas críticas de Youth: el 2-dic, cuando YW se desplomó –15.1 pts tras 295 incidentes NCS sin mención en verbatims y con MAD–MAH en –25.0, y el 4-dic, con –10.7 pts motivados por el mal desempeño de vuelos CodeShare QR en HAM–MAD (NPS –37.5). El retroceso menor del 6-dic (–0.4 pts en YW por cancelaciones y retrasos) y el repunte del 7-dic (+8.1 pts en YW gracias a la mejora de puntualidad) quedaron diluidos en la tendencia negativa acumulada de la semana.

--- BUSINESS SH ---
A nivel diario, el 6 de diciembre concentró casi todo el retroceso de Business SH, con IB desplomándose –11.0 pts (NPS 22.2) y provocando la caída total de –12.4 pts en la cabina. Una señal previa se vio el 4 de diciembre, cuando IB cedió –2.7 pts en la ruta MAD–BIO, anticipando la tendencia negativa. El repunte puntual de YW el 5 de diciembre (+20.0 pts) no logró compensar las fuertes pérdidas posteriores de IB.

--- ECONOMY LH ---
A nivel diario, la sólida ganancia semanal de +6.2 pts en Economy LH se concentró en los días con fuertes repuntes de satisfacción tras mejoras de puntualidad y feedback positivo en ruta. El 4-dic experimentó el mayor alza (+8.7 pts) gracias a mejor OTP en MAD–MIA y verbatims favorables pese a 85 retrasos y 50 cancelaciones, seguido por el 5-dic (+6.8 pts) impulsado por elogios en confort y atención. Estas subidas compensaron en parte las caídas del 3-dic (–14.1 pts por demoras en MAD–SCL) y del 6-dic (–11.9 pts tras incidencias en DOH–MAD), diluyendo variaciones extremas en el balance semanal.

--- BUSINESS LH ---
A nivel diario, el deterioro semanal de Business LH se concentró el 3-dic, con un hundimiento de –22,3 pts tras 79 retrasos y 66 cancelaciones que dejaron el NPS en 0,0 en rutas como MAD–SCL. A ello se sumaron las caídas del 6-dic (–3,1 pts por desviaciones de puntualidad y 310 incidentes, con BOG–MAD en NPS 0,0) y las correcciones del 2-dic y 4-dic (–1,8 pts sin drivers operativos claros). Los repuntes del 5-dic (+17,0 pts por feedback muy positivo en BOG–MAD) y 7-dic (+14,5 pts en JFK–MAD) no bastaron para compensar estas pérdidas en el cómputo semanal.

--- PREMIUM LH ---
A nivel diario, la debilidad semanal de Premium LH se explica sobre todo por tres jornadas críticas: el 04-dic acumuló la mayor caída (–25,5 pts) tras un desplome de OTP (–3,7 pts) y 85 retrasos con 50 cancelaciones en rutas como MAD–ORD; el 03-dic anotó –23,5 pts por 79 retrasos y 66 cancelaciones en MAD–SCL; y el 07-dic sumó –20,8 pts pese a feedback positivo. Aunque el 02-dic Premium LH llegó a repuntar +41,6 pts por elogios al servicio y confort, este impulso se diluyó ante las sucesivas incidencias operativas. Los descensos adicionales del 05-dic (–10,0 pts) y 06-dic (–8,8 pts) consolidaron las deficiencias en arrivals, conexiones y facilidad de contacto que marcaron el balance semanal.

--- GLOBAL ---
A nivel diario, la mejora semanal de +1,1 pts se sustentó sobre el alza del **6-dic** (+4,5 pts en el Global), impulsada por el empuje de Economy SH/IB y Premium LH pese a un OTP inferior. Los descensos del **4-dic** (–3,8 pts, arrastrado por Economy SH tras el mal desempeño en HAM–MAD) y del **7-dic** (–1,0 pts vinculado a Premium LH y Business SH) fueron amortiguados. Por su parte, los repuntes modestos del **2-dic** (+0,3 pts) y del **5-dic** (+0,4 pts) quedaron diluidos frente al fuerte impulso del 6-dic.

===== STEP 2: FINAL REPORT =====

📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Economy SH el escenario es TRANSFERENCIA (IB Normal, YW Negativa | Economy Negativa).  
- Narrativa: la caída del NPS en Economy SH se explica con la explicación del nodo padre (Economy), pues el deterioro de YW —empeoramiento de la puntualidad y aumento de mishandling— transfiere su efecto negativo al agregado, a pesar de la estabilidad de IB.  
- Evidencia Clave: Punctuality SHAP = –1.5 (Sat_diff = –0.7), OTP15 –0.3 pts y mishandling +0.3 pts (Global/SH/Economy).

En Business SH el escenario es TRANSFERENCIA (IB Negativa, YW Normal | Business Negativa).  
- Narrativa: la anomalía del NPS en Business SH adopta la explicación del nodo padre (Business), ya que la fuerte caída en satisfacción por puntualidad y drivers de producto negativos impone el signo, pese a la normalidad de YW.  
- Evidencia Clave: Punctuality SHAP = –3.1 (Sat_diff = –3.9) y Cabin Crew SHAP = –2.7 (Global/SH/Business).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Global la dinámica es DOMINANCIA (LH Normal, SH Negativa | Global Positiva).  
- Narrativa: el ascenso neto de NPS a nivel compañía responde al impulso de Long Haul, donde la cabina Economy LH obtuvo un sólido +6.2 pts, imponiéndose sobre la caída de Short Haul y arrastrando el resultado agregado hacia lo positivo, aunque la mejora se vio matizada por el descenso de SH.  
- Evidencia: Economy LH +6.2 pts (NPS 13.8 vs baseline 7.6) diluyó los impactos negativos de SH (Punctuality SHAP = –1.6, OTP15 –0.3 pts y retrasos +124 inc.).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es DOMINANCIA (LH Normal, SH Negativa | Global Positiva).  
- Narrativa: el alza neta de +0.8 pts del NPS Global está arrastrada por el aporte positivo de Long Haul, que compensó con creces la caída en Short Haul.  
- Evidencia: mejoras en drivers de producto en Global/LH – Cabin Crew (SHAP +0.5, Sat_diff +1.8), Check-in (SHAP +0.2, Sat_diff +0.4), Arrivals experience (SHAP +0.1, Sat_diff +0.2) e IB Plus loyalty program (SHAP +0.1, Sat_diff +1.0) – dominaron la disminución operacional y de satisfacción en SH.

A nivel diario, la mejora semanal de +1.1 pts se sustentó sobre el alza del 6-dic (+4.5 pts en el Global), impulsada por el empuje de Economy SH/IB y Premium LH pese a un OTP inferior. Los descensos del 4-dic (–3.8 pts, arrastrado por Economy SH tras el mal desempeño en HAM–MAD) y del 7-dic (–1.0 pts vinculado a Premium LH y Business SH) fueron amortiguados. Por su parte, los repuntes modestos del 2-dic (+0.3 pts) y del 5-dic (+0.4 pts) quedaron diluidos frente al fuerte impulso del 6-dic.

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

CAUSA 1: Deterioro de la puntualidad  
- Escenario: DOMINANCIA (LH Normal, SH Negativa | Global Positiva)  
- NMA: Global/LH  
- Afecta a:  
    • Global/LH/Business  
    • Global/LH/Premium  
    • Global/SH/Economy  
    • Global/SH/Business  
- Tipo de impacto: NEGATIVO  

CAUSA 2: Aumento de mishandling de equipaje  
- Escenario: TRANSFERENCIA en SH/Economy (IB Normal, YW Negativa | Economy SH Negativa)  
- NMA: Global/SH/Economy/YW  
- Afecta a: Global/SH/Economy/YW  
- Tipo de impacto: NEGATIVO  

CAUSA 3: Aumento de mishandling de equipaje  
- Escenario: DILUCIÓN (Economy LH Normal, Business LH Negativa, Premium LH Negativa | LH Normal)  
- NMA: Global/LH/Premium  
- Afecta a: Global/LH/Premium  
- Tipo de impacto: NEGATIVO  

CAUSA 4: Deficiencias en experiencia de producto (Arrivals experience, Aircraft interior, Connections, Ease of contact…)  
- Escenario: DILUCIÓN (Economy LH Normal, Business LH Negativa, Premium LH Negativa | LH Normal)  
- NMA: Global/LH/Premium  
- Afecta a: Global/LH/Premium  
- Tipo de impacto: NEGATIVO

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

=== NMA: Global/LH ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
No disponible

🚨 INCIDENTES NCS:  
No disponible

💬 FEEDBACK DE CLIENTES:  
No disponible

✈️ RUTAS AFECTADAS (Top 5):  
No disponible

👥 PERFILES REACTIVOS:  
No disponible


=== NMA: Global/SH/Economy/YW ===

📈 EXPLANATORY DRIVERS:  
• Punctuality SHAP = -4.4, Sat_diff = -4.1 (explanatory_drivers_tool)  
• Connections experience SHAP = -0.4, Sat_diff = -0.7 (explanatory_drivers_tool)  
• Flight cancellations – Identificado por ncs_tool (SHAP y Sat_diff no disponibles)

📊 DATOS OPERATIVOS:  
• Datos operativos (OTP15) no disponibles en la fuente  
• Datos de pérdidas de conexión (misconnections) no disponibles

🚨 INCIDENTES NCS:  
• NCS: no se especifican métricas de retrasos  
• Identificado por ncs_tool (cancelaciones de vuelo)

💬 FEEDBACK DE CLIENTES:  
• Verbatims no contienen menciones a estos aspectos negativos  

✈️ RUTAS AFECTADAS (Top 5):  
• BCN-MLN: NPS –100.0,   Pax 1 (fuente: explanatory_drivers)  
• MAD-SXB: NPS –20.0,   Pax 15 (fuente: explanatory_drivers)  
• ALG-MAD: NPS –16.7,   Pax 6 (fuente: explanatory_drivers)  
• BIO-VLC: NPS 14.3,    Pax 7 (fuente: explanatory_drivers)  
• MLN-SVQ: NPS 0.0,     Pax 4 (fuente: explanatory_drivers)

👥 PERFILES REACTIVOS:  
• Residence Region: spread 237.5 pts  
• CodeShare: spread 218.8 pts  
• Fleet: datos no disponibles  
• Business/Leisure: datos no disponibles  


=== NMA: Global/LH/Premium ===

📈 EXPLANATORY DRIVERS:  
• Punctuality (explanatory_drivers_tool): SHAP = –6.9, Sat_diff = –12.2  
• Load factor (explanatory_drivers_tool): SHAP = 0.0, Sat_diff = –3.6  
• Arrivals experience: SHAP = –1.5, Sat_diff = –9.6 (explanatory_drivers_tool)  
• Aircraft interior: SHAP = –1.4, Sat_diff = –8.5 (explanatory_drivers_tool)  
• Connections experience: SHAP = –0.9, Sat_diff = –16.6 (explanatory_drivers_tool)  
• Ease of contact by phone: SHAP = –0.7, Sat_diff = –10.1 (explanatory_drivers_tool)  
• Otros factores con SHAP negativos menores: Journey preparation support (–0.5), Check-in (–0.4), Boarding (–0.2), Wi-Fi (–0.2), Pilot’s announcements (–0.1)  
📊 DATOS OPERATIVOS:  
• OTP15 cayó 3.2 pts (74.3 → 71.1) (operative_data_tool)  
• Mishandling aumentó 0.3 pts (1.2 → 1.5) (operative_data_tool)

🚨 INCIDENTES NCS:  
• retrasos +124 incidentes (514 → 638) (ncs_tool)  
• desvíos +31 incidentes (0 → 31) (ncs_tool)  
• limitaciones de aeronave +24 incidentes (27 → 51) (ncs_tool)

💬 FEEDBACK DE CLIENTES:  
• Verbatims: no aportaron menciones cualitativas de retrasos, desvíos o equipaje  

✈️ RUTAS AFECTADAS (Top 5):  
• GRU-MAD: NPS 25.0, Pax 8 (solapamiento explanatory_drivers + ncs)  
• MAD-SCL: NPS –18.8, Pax 16 (solapamiento explanatory_drivers + ncs)  
• MAD-MIA: NPS 40.0, Pax 5 (solapamiento explanatory_drivers + ncs)  
• EZE-MAD: NPS 0.0, Pax 18 (solo explanatory_drivers)  
• JFK-MAD: NPS 0.0, Pax 4 (solo explanatory_drivers)

👥 PERFILES REACTIVOS:  
• Residence Region: spread = 285.7 pts  
• CodeShare: 93.3 pts  
• Fleet: 31.8 pts  
• Business/Leisure: 15.5 pts

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 2025-12-02 a 2025-12-08, hemos identificado tres causas principales que explican las variaciones de NPS. El resultado global mostró un aumento de 0.8 pts con respecto a la semana anterior.

El avance en el NPS global se sustentó en un sólido desempeño de LH. Economy LH experimentó una mejora de 6.2 pts con respecto a la semana anterior, compensando en parte el deterioro de Business LH (-4.4 pts) y el más acusado retroceso de Premium LH (-9.8 pts). En SH, la presión operativa sobre la puntualidad y el equipaje provocó una ligera caída en Economy SH (-0.2 pts) y un descenso más intenso en Business SH (-9.8 pts).

A continuación, el detalle por cabina:

ECONOMY SH: Impacto de puntualidad y mishandling en YW  
La cabina Economy de SH registró un NPS de 34.7, con una variación de –0.2 pts con respecto a la semana anterior. En IB, la satisfacción subió a 34.3 (+2.7 pts), mientras que en YW cayó a 35.6 (–6.1 pts), imponiendo la tendencia negativa. Explanatory Drivers señaló un impacto de puntualidad de –4.4 ppts según Explanatory Drivers, y las métricas operativas mostraron OTP –0.3 ppts y mishandling +0.3 ppts, respaldados por 124 retrasos, 31 desvíos y 24 limitaciones de aeronave según incidentes operativos. Las rutas con mayor deterioro fueron BCN-MLN (NPS –100.0,   Pax 1), MAD-SXB (NPS –20.0,   Pax 15) y ALG-MAD (NPS –16.7,   Pax 6), mientras que BIO-VLC (NPS 14.3,    Pax 7) y MLN-SVQ (NPS 0.0,     Pax 4) mostraron comportamientos mixtos. Los pasajeros en vuelos code-share (spread 183.3 pts) y según tipología de flota (spread 156.9 pts) fueron los más reactivos.

A nivel diario, la caída semanal de Economy SH se concentró en dos jornadas críticas de Youth: el 2-dic, cuando YW se desplomó –15.1 pts tras 295 incidentes NCS sin mención en verbatims y con MAD–MAH en –25.0, y el 4-dic, con –10.7 pts motivados por el mal desempeño de vuelos CodeShare QR en HAM–MAD (NPS –37.5). El retroceso menor del 6-dic (–0.4 pts en YW por cancelaciones y retrasos) y el repunte del 7-dic (+8.1 pts en YW gracias a la mejora de puntualidad) quedaron diluidos en la tendencia negativa acumulada de la semana.

BUSINESS SH: Presión de mishandling en IB  
La cabina Business de SH registró un NPS de 31.6, con una variación de –9.8 pts con respecto a la semana anterior. IB descendió a 33.5 (–16.0 pts), mientras que YW mejoró a 26.7 (+4.3 pts), pero el deterioro de IB dictó la caída global. En IB, el mishandling de equipaje aumentó de 18.4 a 18.8 (+0.4 ppts en métricas operativas) y se registraron 148 pérdidas de conexión nuevas según incidentes operativos. Explanatory Drivers identificó un impacto de puntualidad de –3.4 ppts. Las rutas más afectadas fueron AGP-MAD (NPS –100.0, Pax 1), LCG-MAD (NPS –50.0, Pax 2), LIS-MAD (NPS –42.9, Pax 7), MAD-OSL (NPS –66.7, Pax 3) y MAD-VCE (NPS 50.0, Pax 4). La segmentación de clientes mostró mayor variabilidad en Región de residencia (spread 246.2 pts) y flota (spread 178.6 pts).

A nivel diario, el 6 de diciembre concentró casi todo el retroceso de Business SH, con IB desplomándose –11.0 pts (NPS 22.2) y provocando la caída total de –12.4 pts en la cabina. Una señal previa se vio el 4 de diciembre, cuando IB cedió –2.7 pts en la ruta MAD–BIO, anticipando la tendencia negativa. El repunte puntual de YW el 5 de diciembre (+20.0 pts) no logró compensar las fuertes pérdidas posteriores de IB.

ECONOMY LH: Mantuvo desempeño estable  
La cabina Economy de LH registró un NPS de 13.8, con una mejora de +6.2 pts con respecto a la semana anterior. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción.

A nivel diario, la sólida ganancia semanal de +6.2 pts en Economy LH se concentró en los días con fuertes repuntes de satisfacción tras mejoras de puntualidad y feedback positivo en ruta. El 4-dic experimentó el mayor alza (+8.7 pts) gracias a mejor OTP en MAD–MIA y verbatims favorables pese a 85 retrasos y 50 cancelaciones, seguido por el 5-dic (+6.8 pts) impulsado por elogios en confort y atención. Estas subidas compensaron en parte las caídas del 3-dic (–14.1 pts por demoras en MAD–SCL) y del 6-dic (–11.9 pts tras incidencias en DOH–MAD), diluyendo variaciones extremas en el balance semanal.

BUSINESS LH: Impacto moderado por puntualidad y equipaje  
La cabina Business de LH presentó un NPS de 21.7, con una variación de –4.4 pts con respecto a la semana anterior. Explanatory Drivers reflejó un impacto de puntualidad de –1.2 ppts, respaldado por 124 retrasos adicionales según incidentes operativos, y el mishandling de equipaje sumó 24 incidentes nuevos en métricas operativas. Este deterioro se concentró en rutas como DFW-MAD (NPS –100.0, Pax 2), GUA-MAD (NPS 0.0, Pax 1), MAD-SJU (NPS 0.0, Pax 6), MAD-MIA (NPS  – , Pax 10) y GRU-MAD (NPS  – , Pax 14). Los perfiles más sensibles fueron los de flota (spread 196.7 pts) y Región de residencia (spread 92.8 pts).

A nivel diario, el deterioro semanal de Business LH se concentró el 3-dic, con un hundimiento de –22.3 pts tras 79 retrasos y 66 cancelaciones que dejaron el NPS en 0.0 en rutas como MAD–SCL. A ello se sumaron las caídas del 6-dic (–3.1 pts por desviaciones de puntualidad y 310 incidentes, con BOG–MAD en NPS 0.0) y las correcciones del 2-dic y 4-dic (–1.8 pts sin drivers operativos claros). Los repuntes del 5-dic (+17.0 pts por feedback muy positivo en BOG–MAD) y 7-dic (+14.5 pts en JFK–MAD) no bastaron para compensar estas pérdidas en el cómputo semanal.

PREMIUM LH: Deterioro por puntualidad, experiencia de producto y equipaje  
La cabina Premium de LH registró un NPS de 17.7, con una variación de –9.8 pts con respecto a la semana anterior. Explanatory Drivers mostró impactos de puntualidad de –6.9 ppts y de arrivals experience de –1.5 ppts, aircraft interior de –1.4 ppts, connections experience de –0.9 ppts y ease of contact de –0.7 ppts. En métricas operativas, OTP cayó 3.2 ppts y mishandling subió 0.3 ppts, mientras que incidentes operativos reportaron 124 retrasos, 31 desvíos y 24 limitaciones de aeronave. Las rutas más afectadas fueron GRU-MAD (NPS 25.0, Pax 8), MAD-SCL (NPS –18.8, Pax 16), MAD-MIA (NPS 40.0, Pax 5), EZE-MAD (NPS 0.0, Pax 18) y JFK-MAD (NPS 0.0, Pax 4). La mayor reactividad se observó en Región de residencia (spread 285.7 pts) y vuelos code-share (spread 93.3 pts).

A nivel diario, la debilidad semanal de Premium LH se explica sobre todo por tres jornadas críticas: el 04-dic acumuló la mayor caída (–25.5 pts) tras un desplome de OTP (–3.7 pts) y 85 retrasos con 50 cancelaciones en rutas como MAD–ORD; el 03-dic anotó –23.5 pts por 79 retrasos y 66 cancelaciones en MAD–SCL; y el 07-dic sumó –20.8 pts pese a feedback positivo. Aunque el 02-dic Premium LH llegó a repuntar +41.6 pts por elogios al servicio y confort, este impulso se diluyó ante las sucesivas incidencias operativas. Los descensos adicionales del 05-dic (–10.0 pts) y 06-dic (–8.8 pts) consolidaron las deficiencias en arrivals, conexiones y facilidad de contacto que marcaron el balance semanal.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0  
- **Pasos de análisis:** 6  
- **Metodología:** Análisis conversacional paso a paso  
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado  

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*