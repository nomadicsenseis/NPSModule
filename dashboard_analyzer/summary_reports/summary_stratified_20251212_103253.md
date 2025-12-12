===== STEP 1: SECTION CONNECTIONS =====

--- GLOBAL ---
A nivel diario, la caída semanal se explica en gran medida por el 30-nov, cuando el global cedió –7.3 pts con respecto a la media de los últimos 7 días tras 358 incidentes operativos (68 cancelaciones y 63 retrasos) que llevaron el NPS de Long Haul a –24.9 pts. La huelga en Italia aparecida el 27-nov y 28-nov, con picos de cancelaciones en MAD–MXP y BLQ–MAD, anticipó tensiones que neutralizaron parte de los repuntes del 24-nov (+11.3 pts por confort y servicio) y del 29-nov (+55.9 pts en Premium LH). Así, las subidas intermedias no alcanzaron a compensar el derrumbe final y quedaron diluidas en el balance semanal.

===== STEP 2: FINAL REPORT =====

**INFORME SEMANAL COMPLETO:**
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Economy SH, el escenario es SINERGIA (Normal IB, Normal YW | Normal Economy SH).  
- Narrativa: Adopto la explicación del nodo padre. No se identifican anomalías ni drivers atípicos; ambos subsegmentos mantienen un desempeño estable, lo que se traduce en un NPS Economy SH dentro de rango normal.  
- Evidencia clave: IB (NPS 32.87 vs 30.69, +2.18 pts); YW (NPS 42.97 vs 38.51, +4.46 pts).

En Business SH, el escenario es CANCELACIÓN (Positivo IB, Negativo YW | Normal Business SH).  
- Narrativa: Mientras IB obtuvo un impulso en puntualidad y boarding, YW sufrió un deterioro en el interior de la aeronave y el proceso de embarque, anulándose mutuamente y resultando en un NPS Business SH normal.  
- Evidencia clave: IB impulsado por Punctuality SHAP=4.266 y Boarding SHAP=3.989; YW lastrada por Aircraft interior SHAP=–6.757 y Boarding SHAP=–3.229.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es SINERGIA (Economy SH: Normal, Business SH: Normal | SH: Normal).  
- Narrativa: Adopto la explicación del nodo padre. Ninguna cabina presenta anomalías; ambas mantienen estabilidad operativa y de producto, por lo que el NPS SH se mantiene dentro de la variación esperada.  
- Evidencia: Economy SH (NPS 36.14 vs 33.26), Business SH (NPS 41.97 vs 40.07).

En Long Haul, la dinámica es SINERGIA (Economy LH: –, Business LH: –, Premium LH: – | LH: –).  
- Narrativa: Adopto la explicación del nodo padre. La caída de –5.74 pts en NPS LH se debe al agravamiento conjunto de factores operativos (puntualidad deteriorada, cancelaciones) y de producto (apoyo pre-viaje e in-flight service), afectando todas las cabinas.  
- Evidencia: Punctuality SHAP=–1.097 respaldado por OTP15 ↓ y + eventos de retraso/cancelación; Journey preparation support SHAP=–0.911; In flight food and beverage SHAP=–0.829.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es DOMINANCIA (Negativo LH, Positivo SH | Positivo Global).  
- Narrativa: El alza en el NPS Global (+0.8 pts) está dictada por el fuerte desempeño de Short Haul, que compensa la caída en Long Haul. En concreto, el segmento Business SH (IB) registró una mejora sustancial en puntualidad y boarding, arrastrando al global hacia arriba.  
- Evidencia: Business SH (IB) con Punctuality SHAP=4.266 y Boarding SHAP=3.989; reducción de cancelaciones y demoras en SH según ncs_tool.

A nivel diario, la caída semanal se explica en gran medida por el 30-nov, cuando el global cedió –7.3 pts con respecto a la media de los últimos 7 días tras 358 incidentes operativos (68 cancelaciones y 63 retrasos) que llevaron el NPS de Long Haul a –24.9 pts. La huelga en Italia aparecida el 27-nov y 28-nov, con picos de cancelaciones en MAD–MXP y BLQ–MAD, anticipó tensiones que neutralizaron parte de los repuntes del 24-nov (+11.3 pts por confort y servicio) y del 29-nov (+55.9 pts en Premium LH). Así, las subidas intermedias no alcanzaron a compensar el derrumbe final y quedaron diluidas en el balance semanal.

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

CAUSE 1: Impulso de puntualidad y boarding en SH Business IB  
- Escenario: DOMINANCIA  
- NMA: Global/SH/Business/IB  
- Afecta a: Global, Short Haul, Business IB  
- Tipo de impacto: POSITIVO  

CAUSE 2: Deterioro de producto en SH Business YW (interior, embarque, comidas…)  
- Escenario: CANCELACIÓN  
- NMA: Global/SH/Business/YW  
- Afecta a: Short Haul, Business YW  
- Tipo de impacto: NEGATIVO  

CAUSE 3: Fallos operativos y de producto en Long Haul (puntualidad, journey support, food & beverage…)  
- Escenario: SINERGIA  
- NMA: Global/LH  
- Afecta a: Economy LH, Business LH, Premium LH  
- Tipo de impacto: NEGATIVO

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

=== NMA: Global/SH/Business/IB ===

📈 EXPLANATORY DRIVERS:  
• Punctuality SHAP = 4.266 (Global/SH/Business/IB)  
• Load factor SHAP = 0.248 (Global/SH/Business/IB)  
• Boarding SHAP = 3.989 (Global/SH/Business/IB)  
• Journey preparation support SHAP = 1.867 (Global/SH/Business/IB)  
• Aircraft interior SHAP = 1.600 (Global/SH/Business/IB)  
• IB Plus loyalty program SHAP = 1.353 (Global/SH/Business/IB)  
• Connections experience SHAP = 0.358 (Global/SH/Business/IB)  
• In flight food and beverage SHAP = –0.821 (Global/SH/Business/IB)  
• Ticket Price SHAP = –0.333 (Global/SH/Business/IB)  
• Cabin Crew SHAP = –0.184 (Global/SH/Business/IB)  
• Check-in SHAP = –0.121 (Global/SH/Business/IB)  

📊 DATOS OPERATIVOS:  
• operative_data_tool: OTP15 prácticamente estable (↘️0.0) (Global/SH/Business/IB)  
• operative_data_tool: reducción de 0.1 en load factor (Global/SH/Business/IB)  

🚨 INCIDENTES NCS:  
• Flight cancellations (ncs_tool): causa identificada en 28 rutas (Global/SH/Business/IB)  
• ncs_tool: reducción global de cancelaciones y demoras (Global/SH/Business/IB)  

💬 FEEDBACK DE CLIENTES:  
• Volumen comentarios: 286 vs 280 (+2.1 %) (Global/SH/Business/IB)  
• Temas consistentes: puntualidad y amabilidad de la tripulación, sin nuevas quejas operativas (Global/SH/Business/IB)  

✈️ RUTAS AFECTADAS (Top 5):  
• LHR-MAD: 29 pax (fuente: ncs_tool) (Global/SH/Business/IB)  
• BRU-MAD: 9 pax (fuente: ncs_tool) (Global/SH/Business/IB)  
• MAD-MXP: 6 pax (fuente: ncs_tool) (Global/SH/Business/IB)  
• DSS-MAD: 5 pax, NPS –20.0 (fuente: explanatory_drivers_tool) (Global/SH/Business/IB)  
• LCG-MAD: 5 pax, NPS 40.0 (fuente: explanatory_drivers_tool) (Global/SH/Business/IB)  

👥 PERFILES REACTIVOS:  
• CodeShare: spread 250.0 pts (Global/SH/Business/IB)  
• Residence Region: spread 246.7 pts (Global/SH/Business/IB)  
• Fleet: datos de variabilidad no disponibles (Global/SH/Business/IB)  
• Business/Leisure: datos de variabilidad no disponibles (Global/SH/Business/IB)  

=== NMA: Global/SH/Business/YW ===

📈 EXPLANATORY DRIVERS:  
• Aircraft interior SHAP = –6.757 (Global/SH/Business/YW)  
• Boarding SHAP = –3.229 (Global/SH/Business/YW)  
• Journey preparation support SHAP = –3.196 (Global/SH/Business/YW)  
• Arrivals experience SHAP = –2.237 (Global/SH/Business/YW)  
• In flight food and beverage SHAP = –1.811 (Global/SH/Business/YW)  
• Check-in SHAP = –1.345 (Global/SH/Business/YW)  
• Cabin Crew SHAP = –1.223 (Global/SH/Business/YW)  
• Connections experience SHAP = –1.167 (Global/SH/Business/YW)  
• Ticket Price SHAP = –0.333 (Global/SH/Business/YW)  
• Lounge SHAP = –0.199 (Global/SH/Business/YW)  
• Ease of contact by phone SHAP = +0.389 (Global/SH/Business/YW)  

📊 DATOS OPERATIVOS:  
• Mishandling de equipaje aumentó +0.7 puntos vs L7d (Global/SH/Business/YW)  
• OTP15 (on-time performance 15’): +0.04110317460317463 puntos (Global/SH/Business/YW)  
• Load factor: –0.026214285714285693 puntos (Global/SH/Business/YW)  
• Misconnections: –0.0130600834592031 puntos (Global/SH/Business/YW)  

🚨 INCIDENTES NCS:  
• Flight cancellations: identificadas como causa (Global/SH/Business/YW)  
• Disminución de retrasos: –126 eventos (Global/SH/Business/YW)  
• Disminución de cancelaciones: –48 eventos (Global/SH/Business/YW)  
• Aumento de “LIMITACION_AERONAVE”: +15 eventos (Global/SH/Business/YW)  
• Aumento de “OTRAS_INCIDENCIAS”: +2 eventos (Global/SH/Business/YW)  

💬 FEEDBACK DE CLIENTES:  
• Volumen de comentarios: 128 vs 88 (+45.5 %) (Global/SH/Business/YW)  
• Temas constantes: amabilidad de tripulación, puntualidad (Global/SH/Business/YW)  
• No emergen quejas sobre equipaje, cancelaciones ni limitaciones de aeronave (Global/SH/Business/YW)  

✈️ RUTAS AFECTADAS (Top 5):  
No disponible

👥 PERFILES REACTIVOS:  
No disponible

=== NMA: Global/LH ===

📈 EXPLANATORY DRIVERS:  
• Punctuality SHAP = –1.097, Sat_diff = –2.9451844449161655 (Global/LH)  
• Load factor SHAP = –0.104, Sat_diff = 2.0857549169693215 (Global/LH)  
• Journey preparation support SHAP = –0.911, Sat_diff = –3.5926297719127263 (Global/LH)  
• In flight food and beverage SHAP = –0.829, Sat_diff = –3.9663852151046015 (Global/LH)  
• Cabin Crew SHAP = –0.667, Sat_diff = –2.252690757022947 (Global/LH)  
• Aircraft interior SHAP = –0.545, Sat_diff = –2.5473360150988356 (Global/LH)  
• Check-in SHAP = –0.418, Sat_diff = –2.5736388965001424 (Global/LH)  
• Connections experience SHAP = –0.406, Sat_diff = –2.488681301704105 (Global/LH)  
• Ticket Price SHAP = –0.354, Sat_diff = 15.88702099647901 (Global/LH)  
• Boarding SHAP = 0.301, Sat_diff = 0.9053855052102051 (Global/LH)  
• Ease of contact by phone SHAP = –0.282, Sat_diff = –9.70245358950374 (Global/LH)  
• IB Plus loyalty program SHAP = –0.194, Sat_diff = –4.567468340799103 (Global/LH)  
• Arrivals experience SHAP = –0.188, Sat_diff = –2.003793862317991 (Global/LH)  

📊 DATOS OPERATIVOS:  
No disponible

🚨 INCIDENTES NCS:  
• Flight cancellations (incidencia puntual señalada) – Herramienta: ncs_tool (Global/LH)  

💬 FEEDBACK DE CLIENTES:  
No disponible

✈️ RUTAS AFECTADAS (Top 5):  
• MAD-MCO: NPS –32.0 (Pax 17) – explanatory_drivers_tool (Global/LH)  
• JFK-MAD: NPS –14.5 (Pax 78) – explanatory_drivers_tool (Global/LH)  
• MAD-ORD: NPS –10.1 (Pax 31) – explanatory_drivers_tool & ncs_tool (Global/LH)  
• LIM-MAD: NPS 10.4 (Pax 104) – explanatory_drivers_tool (Global/LH)  
• IAD-MAD: NPS 24.0 (Pax 12) – explanatory_drivers_tool (Global/LH)  

👥 PERFILES REACTIVOS:  
• Residence Region: spread 142.4 pts (Global/LH)  
• CodeShare: spread 72.2 pts (Global/LH)  
• Fleet: spread 41.3 pts (Global/LH)  
• Business/Leisure: spread 1.5 pts (Global/LH)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 2025-11-24 al 2025-11-30, hemos identificado tres causas principales que explican las variaciones de NPS. El resultado global fue una subida de 0.8135026512014392 pts con respecto a la semana anterior.

El empuje más significativo se observó en SH Business IB, donde el NPS se situó en 51.90476190476191, con una subida de 12.689075630252113 pts con respecto a la semana anterior. Este avance se sustentó en una mejora de la puntualidad (4.266 ppts según Explanatory Drivers) y en un embarque más ágil (3.989 ppts según Explanatory Drivers), mientras que el OTP se mantuvo prácticamente estable (↘️0.0 según métricas operativas) y se registró reducción general de incidentes operativos. El feedback de clientes destacó la puntualidad y la amabilidad de la tripulación, y las rutas LHR-MAD (29 pax), BRU-MAD (9 pax) y MAD-MXP (6 pax) registraron los mejores resultados. Los segmentos más reactivos fueron CodeShare (spread 250.0 pts) y Residence Region (spread 246.7 pts).

En contraste, SH Business YW cayó a un NPS de 20.0, con una disminución de 22.465753424657528 pts con respecto a la semana anterior. El interior de la aeronave empeoró (–6.757 ppts según Explanatory Drivers), el embarque se retrasó (–3.229 ppts según Explanatory Drivers) y la preparación de viaje se resintió (–3.196 ppts según Explanatory Drivers), acompañado de un aumento de mishandling de equipaje (+0.7 según métricas operativas). Aunque los incidentes operativos mostraron una reducción de retrasos (–126 eventos) y cancelaciones (–48 eventos) y un ligero incremento de “LIMITACION_AERONAVE” (+15 eventos) y “OTRAS_INCIDENCIAS” (+2 eventos), el feedback de clientes no reflejó quejas específicas.

En LH la caída fue generalizada, con un descenso de 5.741522806478244 pts con respecto a la semana anterior, impactando a Economy (9.485294117647069, –5.029473814842392 pts), Business (27.45098039215687, –4.712762297901609 pts) y Premium (22.80701754385965, –12.438884095484617 pts). La puntualidad (–1.097 ppts según Explanatory Drivers) y las cancelaciones puntuales fueron los factores operativos clave, mientras que la experiencia de product service se deterioró en journey preparation support (–0.911 ppts según Explanatory Drivers), in flight food and beverage (–0.829 ppts según Explanatory Drivers) y cabin crew (–0.667 ppts según Explanatory Drivers). Las rutas MAD-MCO (NPS –32.0, 17 pax), JFK-MAD (NPS –14.5, 78 pax) y MAD-ORD (NPS –10.1, 31 pax) concentraron las mayores pérdidas, con los perfiles Residence Region (spread 142.4 pts) y CodeShare (spread 72.2 pts) como los más sensibles.

---

DETALLE POR CABINA:

ECONOMY SH: Desempeño equilibrado  
La cabina Economy de SH registró un NPS de 36.14495470165573 con una subida de 2.887121557072861 pts con respecto a la semana anterior. Desglose por compañía: IB obtuvo 32.87101248266297 con una variación de +2.1796128199311 pts, y YW alcanzó 42.967244701348754 con una variación de +4.461497574911964 pts. No se identificaron drivers operativos ni de producto atípicos, lo que indica estabilidad en puntualidad y servicios a bordo.

BUSINESS SH: Contraste que neutraliza el segmento  
La cabina Business de SH registró un NPS de 41.9672131147541 con una subida de 1.8950109486891198 pts con respecto a la semana anterior. Desglose por compañía: IB alcanzó 51.90476190476191 (subida de +12.689075630252113 pts según Explanatory Drivers), impulsado por Punctuality (4.266 ppts según Explanatory Drivers), Boarding (3.989 ppts según Explanatory Drivers) y Journey preparation support (1.867 ppts según Explanatory Drivers), con OTP estable (↘️0.0 según métricas operativas) y reducción de incidentes operativos. En contraste, YW descendió a 20.0 (–22.465753424657528 pts según Explanatory Drivers), afectada por Aircraft interior (–6.757 ppts según Explanatory Drivers), Boarding (–3.229 ppts según Explanatory Drivers), Journey preparation support (–3.196 ppts según Explanatory Drivers) y un aumento de mishandling de equipaje (+0.7 según métricas operativas). El feedback de clientes reforzó la percepción positiva en IB sin quejas específicas en YW, y los perfiles CodeShare (spread 250.0 pts) y Residence Region (spread 246.7 pts) mostraron alta reactividad en IB.

ECONOMY LH: Caída por deficiencias operativas y de producto  
La cabina Economy de LH registró un NPS de 9.485294117647069 con una disminución de 5.029473814842392 pts con respecto a la semana anterior. Entre los drivers operativos, Punctuality (–1.277 ppts según Explanatory Drivers) y Load factor (–0.106 ppts según Explanatory Drivers) se tradujeron en un aumento de mishandling de equipaje (+1.2 pts según métricas operativas) y un OTP sin variación (+0.0 pts según métricas operativas). En el frente de producto, In flight food and beverage (–0.819 ppts según Explanatory Drivers), Journey preparation support (–0.661 ppts según Explanatory Drivers) y Connections experience (–0.541 ppts según Explanatory Drivers) lideraron el deterioro. Los incidentes operativos totales bajaron de 1 145 a 978 (–167), con retrasos (–126) y cancelaciones (–48) pero con Limitación_aeronave (+15) y Otras_incidencias (+2) según incidentes operativos. Las rutas MAD-MCO (NPS –27.3, 11 pax), MAD-ORD (NPS –8.3, 24 pax), JFK-MAD (NPS –17.9, 67 pax), MAD-UIO (NPS 6.5, 47 pax) y LIM-MAD (NPS 11.9, 84 pax) sufrieron el mayor impacto, con perfiles Residence Region (NPS_diff rango –26.7 a +87.5) y CodeShare (NPS_diff rango –31.7 a +38.6) como los más volátiles.

BUSINESS LH: Impacto por fallos en interior y soporte  
La cabina Business de LH registró un NPS de 27.45098039215687 con una disminución de 4.712762297901609 pts con respecto a la semana anterior. Entre los drivers de producto, Aircraft interior (–2.950 ppts según Explanatory Drivers), Journey preparation support (–2.296 ppts según Explanatory Drivers) y Ease of contact by phone (–0.634 ppts según Explanatory Drivers) destacaron, junto al IB Plus loyalty program (–0.555 ppts según Explanatory Drivers). En lo operativo, Punctuality (–0.314 ppts según Explanatory Drivers) coincidió con flight cancellations en 23 rutas según incidentes operativos. Las rutas HAV-MAD (NPS 0.0, 5 pax), MAD-ORD (NPS 0.0, 6 pax) y MAD-PTY (NPS 66.7, 3 pax) mostraron contrastes, y el perfil CodeShare presentó spread 300.0 pts.

PREMIUM LH: Fuerte caída por cancelaciones y deficiencias de producto  
La cabina Premium de LH registró un NPS de 22.80701754385965 con una disminución de 12.438884095484617 pts con respecto a la semana anterior. Las cancelaciones, reflejadas en la reducción neta de incidentes operativos totales (–167) y en aumentos de Limitación_aeronave (+15) y Otras_incidencias (+2) según incidentes operativos, se combinaron con drivers de producto adversos: Cabin Crew (–4.675 ppts según Explanatory Drivers), IB Plus loyalty program (–2.555 ppts según Explanatory Drivers) y Check-in (–2.357 ppts según Explanatory Drivers). Rutas como MAD-ORD (NPS –100.0, 1 pax), MAD-NRT (NPS –25.0, 4 pax) y MAD-UIO (NPS –25.0, 4 pax) evidenciaron el impacto, con los perfiles Residence Region (spread 222.2 pts) y CodeShare (spread 122.6 pts) como los más sensibles.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 6
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*