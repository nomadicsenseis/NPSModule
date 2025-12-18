===== STEP 1: SECTION CONNECTIONS =====

--- GLOBAL ---
A nivel diario, el empeoramiento semanal (–6.6 pts vs semana anterior) se concentra en el 09-dic, cuando Premium Long Haul cayó –50.0 pts con respecto a la media de los últimos 7 días por baja puntualidad (OTP –6.99 pts) y críticas al asiento, tripulación y comida en MAD–UIO. El repunte del 06-dic (+6.1 pts global) por mejora de puntualidad en LH y alza en SH Business quedó diluido ante las caídas del 07-dic (–0.6 pts) y del 09-dic (–0.9 pts). No hubo huelgas ni cancelaciones masivas en el periodo ni otros días con variaciones extremas adicionales.

--- ECONOMY SH ---
A nivel diario, el leve ascenso semanal de +3.4 pts en Economy SH no se debe a un pico aislado sino a incrementos constantes y moderados. El 8 de diciembre, por ejemplo, la cabina registró un NPS de 36.80 (+0.7 pts vs media de los últimos 7 días), con IB en 34.9 vs 34.8 y YW en 40.9 vs 38.7, ambos en rango normal. No hubo días con variaciones extremas ni repuntes relevantes que se diluyeran en el consolidado semanal.

--- BUSINESS SH ---
A nivel diario, el **8-dic** concentró la mayor parte del deterioro en Business SH, con una caída de –16,7 pts con respecto a la media de los últimos 7 días tras 84 retrasos, 30 cancelaciones y 150 pérdidas de conexión (la ruta MAD–SXB quedó en NPS 0.0). A esto se sumaron descensos de –7,2 pts el **5-dic** y –5,1 pts el **7-dic**, ambos vinculados a picos de incidentes operativos en YW. Aunque el **6-dic** registró un repunte de +3,2 pts y el **9-dic** IB impulsó +15,4 pts, estos alzas se diluyeron frente a las fuertes caídas previas.

--- ECONOMY LH ---
A nivel diario, la aparente estabilidad de Economy LH (NPS 13.0, +6.9 con respecto a la semana anterior) oculta un fuerte deterioro el 7-dic, cuando un desplome de puntualidad (OTP15_adjusted –5.11 pts) y 68 cancelaciones con 58 retrasos hundieron el NPS en 2.38 (–9.9 con respecto a la media de los últimos 7 días) en rutas como DOH–MAD y MAD–MCO. Este episodio puntual contrastó con los repuntes del 6-dic (NPS 16.49, +4.2) y el 5-dic (NPS 19.03, +6.8), que amortiguaron el impacto y diluyeron la caída en el acumulado semanal.

--- BUSINESS LH ---
A nivel diario, la mejora semanal de Business LH (+18.2 pts con respecto a la semana anterior) se concentró el 8-dic, cuando el NPS repuntó +21.4 pts con respecto a la media de los últimos 7 días impulsado por la recuperación de la puntualidad y la valoración de la tripulación en la ruta GIG–MAD (NPS 100) pese a 57 retrasos y 36 cancelaciones. El 5-dic aportó un segundo empuje de +11.1 pts (NPS 35.48), apoyado en la satisfacción del perfil Leisure y la flota A350 next. En cambio, el 6-dic (+0.6 pts) y el 7-dic (+1.7 pts) mostraron variaciones leves que diluyeron parcialmente el avance.

--- PREMIUM LH ---
A nivel diario, el desplome del 09-dic (–50.0 pts) tras 57 retrasos, 36 cancelaciones y quejas por reasignación de asiento en MAD–UIO explica la mayor parte del deterioro semanal de Premium LH (–25.3 pts con respecto a la semana anterior). A pesar del repunte del 08-dic (+33.3 pts) por mejora de puntualidad y del alza del 05-dic (+10.5 pts) gracias al servicio de la tripulación, estos avances se diluyeron frente a las caídas de inicios de semana. Además, el 07-dic registró otra baja (–4.4 pts) asociada a elevada incidencia operativa, reforzando el balance neto negativo.

===== STEP 2: FULL REPORT =====

📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Short Haul Economy el escenario es SINERGIA (IB N, YW N | PADRE N).  
- Narrativa: Ambos carriers mantienen un desempeño estable sin anomalías relevantes, por lo que el NPS agregado se mantiene dentro de lo esperado.  
- Evidencia Clave: IB Normal (+3.5 pts) y YW Normal (+2.9 pts) en Global/SH/Economy.

En Short Haul Business el escenario es SINERGIA (IB –, YW – | PADRE –).  
- Narrativa: Las dos compañías presentan anomalías negativas que se suman en el nodo padre. La caída de puntualidad y la percepción deteriorada del producto explican la bajada conjunta del NPS en Global/SH/Business.  
- Evidencia Clave:  
  • NCS_TOOL: +64.0 retrasos y +44.0 desvíos  
  • explanatory_drivers_tool: Aircraft interior SHAP = –3.7, Cabin Crew SHAP = –2.2  

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul (SH), la dinámica es DILUCIÓN (Business –, Economy N | SH N).  
- Narrativa: El rendimiento de SH está dictado por la cabina Business, que sufrió una anomalía negativa, aunque la cabina Economy se mantuvo estable y diluyó ese impacto, dejando el NPS agregado en rango normal.  
- Evidencia:  
  • Global/SH/Business –10.0 pts explicado por +64.0 retrasos y +44.0 desvíos (NCS_TOOL) y drivers de producto negativos: Aircraft interior SHAP = –3.7, Cabin Crew SHAP = –2.2 (explanatory_drivers_tool).

En Long Haul (LH), la dinámica es CANCELACIÓN (Economy N, Business +, Premium – | LH N).  
- Narrativa: El alza de NPS en la cabina Business por mejoras operativas contrastó con la fuerte caída en Premium por un incremento de retrasos y desvíos y percepciones negativas de cabina, neutralizándose en el agregado y dejando el NPS de LH dentro del rango normal.  
- Evidencia:  
  • Global/LH/Business +18.2 pts impulsados por OTP15 +3.2 y –60.0 cancelaciones (operative_data_tool y NCS_TOOL) con Punctuality SHAP = 15.8.  
  • Global/LH/Premium –25.3 pts asociado a +64.0 retrasos y +44.0 desvíos (NCS_TOOL) y Aircraft interior SHAP = –12.4, Cabin Crew SHAP = –11.4 (explanatory_drivers_tool).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA (LH N, SH N | GLOBAL +).  
- Narrativa: La anomalía positiva de +3.2 pts en el NPS global se explica por las mismas causas operativas centrales identificadas en el nodo padre: la reducción de cancelaciones y la mejora de puntualidad, a pesar de que ambos radios por separado parecían “Normales” (sus cabinas ocultaron volatilidad interna).  
- Evidencia:  
  • OTP15 subió de 88.9 a 89.3 (+0.4 pts) con Punctuality SHAP = 1.1 (Explanatory Drivers)  
  • Cancelaciones –60.0 incidentes (ncs_tool)  
  • Retrasos +64.0 y Desvíos +44.0 (ncs_tool), que moderaron la mejora  
  • Drivers de producto secundarios: Arrivals experience SHAP = 0.7 y Check-in SHAP = 0.6  

A nivel diario, el empeoramiento semanal (–6.6 pts vs semana anterior) se concentra en el 09-dic, cuando Premium Long Haul cayó –50.0 pts con respecto a la media de los últimos 7 días por baja puntualidad (OTP –6.99 pts) y críticas al asiento, tripulación y comida en MAD–UIO. El repunte del 06-dic (+6.1 pts global) por mejora de puntualidad en LH y alza en SH Business quedó diluido ante las caídas del 07-dic (–0.6 pts) y del 09-dic (–0.9 pts). No hubo huelgas ni cancelaciones masivas en el periodo ni otros días con variaciones extremas adicionales.

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

CAUSA 1: Reducción de cancelaciones  
- Escenario: CANCELACIÓN  
- NMA: Global / LH / Business  
- Afecta a: LH Business  
- Tipo de impacto: POSITIVO  

CAUSA 2: Mejora de puntualidad (OTP15)  
- Escenario: CANCELACIÓN  
- NMA: Global / LH / Business  
- Afecta a: LH Business  
- Tipo de impacto: POSITIVO  

CAUSA 3: Incremento de retrasos y desvíos  
- Escenario: DILUCIÓN  
- NMA: Global / SH / Business  
- Afecta a: SH Business  
- Tipo de impacto: NEGATIVO  

CAUSA 4: Deterioro de producto en SH Business (interior de cabina y Cabin Crew)  
- Escenario: DILUCIÓN  
- NMA: Global / SH / Business  
- Afecta a: SH Business  
- Tipo de impacto: NEGATIVO  

CAUSA 5: Deterioro de producto en LH Premium (interior de cabina y Cabin Crew)  
- Escenario: CANCELACIÓN  
- NMA: Global / LH / Premium  
- Afecta a: LH Premium  
- Tipo de impacto: NEGATIVO  

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

=== NMA: Global / LH / Business ===

📈 EXPLANATORY DRIVERS:  
– Punctuality SHAP = 15.8, Sat_diff = 9.4  
– Arrivals experience SHAP = 5.4, Sat_diff = 5.9  

📊 DATOS OPERATIVOS:  
– operative_data_tool: OTP15 +3.2, mishandling –2.5, misconex –0.2, load factor –0.7  

🚨 INCIDENTES NCS:  
– ncs_tool: Cancelaciones –60.0 incidentes  

💬 FEEDBACK DE CLIENTES:  
No disponible  

✈️ RUTAS AFECTADAS (Top 5):  
• MAD–NRT: NPS 100.0 (1 pax)  
• MAD–MCO: NPS 50.0 (2 pax)  
• MAD–UIO: NPS 33.3 (3 pax)  
• JFK–MAD: NPS –14.3 (7 pax)  
• MAD–SDQ: NPS –100.0 (1 pax)  

👥 PERFILES REACTIVOS:  
• CodeShare: spread 366.7 pts  
• Residence Region: spread 98.8 pts  
• Fleet: spread 75.5 pts  
• Business/Leisure: spread 42.6 pts  

=== NMA: Global / SH / Business ===

📈 EXPLANATORY DRIVERS:  
– Punctuality: SHAP = –1.0; Sat_diff = –3.2  
– Load factor: SHAP = 0.0; Sat_diff = –3.6  
– Aircraft interior: SHAP = –3.7; Sat_diff = –6.4  
– Cabin Crew: SHAP = –2.2; Sat_diff = –2.3  
– Boarding: SHAP = –1.9; Sat_diff = –6.9  
– Journey preparation support: SHAP = –1.7; Sat_diff = –6.4  
– In flight food and beverage: SHAP = –1.5; Sat_diff = –6.9  
– Check-in: SHAP = –0.2; Sat_diff = –2.6  
– Connections experience: SHAP = 2.1; Sat_diff = 1.3  
– Ticket Price: SHAP = 0.1; Sat_diff = –15.3  

📊 DATOS OPERATIVOS:  
No disponible  

🚨 INCIDENTES NCS:  
– retrasos +64.0; desvíos +44.0; cancelaciones –60.0 (ncs_tool)  

💬 FEEDBACK DE CLIENTES:  
No disponible  

✈️ RUTAS AFECTADAS (Top 5):  
• MAD–VCE: NPS –100.0, Pax 2  
• AGP–MLN: NPS –100.0, Pax 1  
• FLR–MAD: NPS –100.0, Pax 1  
• AGP–MAD: NPS –100.0, Pax 1  
• DSS–MAD: NPS –100.0, Pax 2  

👥 PERFILES REACTIVOS:  
• Fleet: rango 250.0 pts  
• Residence Region: rango 182.7 pts  
• CodeShare: rango 104.2 pts  
• Business/Leisure: rango 11.0 pts  

=== NMA: Global / LH / Premium ===

📈 EXPLANATORY DRIVERS:  
– Aircraft interior: SHAP = –12.4, Sat_diff = –18.1  
– Cabin Crew: SHAP = –11.4, Sat_diff = –8.3  
– Journey preparation support: SHAP = –5.2, Sat_diff = –12.1  
– In flight food and beverage: SHAP = –5.2, Sat_diff = –4.3  
– Boarding: SHAP = –3.2, Sat_diff = –3.6  

📊 DATOS OPERATIVOS:  
No disponible  

🚨 INCIDENTES NCS:  
– Retrasos: +64.0 eventos (NCS_TOOL)  
– Desvíos: +44.0 eventos (NCS_TOOL)  
– Cancellaciones reducidas: –60.0 eventos (NCS_TOOL)  

💬 FEEDBACK DE CLIENTES:  
No disponible  

✈️ RUTAS AFECTADAS (Top 5):  
• MAD–ORD: NPS actual –50.0, Pax 4 (fuente: explanatory_drivers)  
• MAD–MCO: NPS actual 100.0, Pax 2 (fuente: explanatory_drivers)  
• MAD–NRT: NPS actual 33.3, Pax 3 (fuente: explanatory_drivers)  
• MAD–SJO: NPS actual 100.0, Pax 1 (fuente: explanatory_drivers)  
• JFK–MAD: NPS actual 0.0, Pax 2 (fuente: explanatory_drivers)  

👥 PERFILES REACTIVOS:  
• CodeShare: rango de impacto [–136.4, +100.0] (spread 236.4 pts)  
• Residence Region: spread 62.5 pts  
• Business/Leisure: spread reducido, todas las variaciones negativas  
• Fleet: spread reducido, todas las variaciones negativas  

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 5 al 9 de diciembre de 2025, hemos identificado tres causas principales que explican las variaciones de NPS. El resultado global mostró una subida de +3.2 puntos con respecto a la semana anterior.

En LH Business, la mejora de la puntualidad y la significativa reducción de cancelaciones impulsaron un alza de +18.2 puntos en el NPS de 33.3 con respecto a la semana anterior. Según Explanatory Drivers, la puntualidad aportó 15.8 ppts mientras que la experiencia de llegada añadió 5.4 ppts. Las métricas operativas respaldan esta mejora: OTP subió +3.2, mishandling –2.5, misconex –0.2 y load factor –0.7. Además, los incidentes operativos registraron –60.0 cancelaciones. Las rutas MAD–NRT (NPS 100.0, 1 pax), MAD–MCO (50.0, 2 pax), MAD–UIO (33.3, 3 pax), JFK–MAD (–14.3, 7 pax) y MAD–SDQ (–100.0, 1 pax) ilustran la dispersión de la experiencia. Los pasajeros en vuelos code-share (spread 366.7 pts), por región de residencia (98.8 pts), tipo de flota (75.5 pts) y Business/Leisure (42.6 pts) fueron especialmente reactivos.

El segmento Business de SH experimentó una caída de –10.0 puntos, hasta un NPS de 29.3 con respecto a la semana anterior. El aumento de +64.0 retrasos y +44.0 desvíos, junto a –60.0 cancelaciones, impactó la puntualidad (–1.0 ppts según Explanatory Drivers) y se tradujo en percepciones negativas sobre el interior de cabina (–3.7 ppts), la tripulación (–2.2 ppts), el embarque (–1.9 ppts), el apoyo en la preparación del viaje (–1.7 ppts), la oferta de comida y bebida a bordo (–1.5 ppts) y el check-in (–0.2 ppts), pese a la mejora en connections experience (+2.1 ppts) y ticket price (+0.1 ppts). Las rutas MAD–VCE, AGP–MLN, FLR–MAD, AGP–MAD y DSS–MAD mostraron NPS de –100.0, concentrando la insatisfacción. Los perfiles más sensibles fueron flota (spread 250.0 pts), región de residencia (182.7 pts), code-share (104.2 pts) y Business/Leisure (11.0 pts). No se detectaron variaciones relevantes en métricas operativas ni feedback de clientes documentado.

En LH Premium el NPS cayó –25.3 puntos, hasta 2.6 con respecto a la semana anterior. Las valoraciones de interior de cabina (–12.4 ppts) y tripulación (–11.4 ppts) fueron las más dañadas, secundadas por journey preparation support (–5.2 ppts), comida y bebida a bordo (–5.2 ppts) y embarque (–3.2 ppts). A pesar de –60.0 cancelaciones, los +64.0 retrasos y +44.0 desvíos no compensaron la insatisfacción. Las rutas MAD–ORD (–50.0, 4 pax), MAD–MCO (100.0, 2 pax), MAD–NRT (33.3, 3 pax), MAD–SJO (100.0, 1 pax) y JFK–MAD (0.0, 2 pax) ejemplifican la dispersión de resultados. Entre perfiles, code-share registró spread 236.4 pts, residence region 62.5 pts; Business/Leisure y Fleet mostraron spread reducido con variaciones negativas. No hubo feedback de clientes reportado.

---

📊 DETALLE POR CABINA:

**ECONOMY SH: Desempeño estable**  
La cabina Economy de SH registró un NPS de 36.6 con una variación de +3.4 puntos con respecto a la semana anterior. Desglose por compañía: IB obtuvo 35.3 (+3.5 ppts según Explanatory Drivers) y YW 39.0 (+2.9 ppts según Explanatory Drivers). Ambas compañías mantuvieron desempeño estable, diluyendo volatilidad interna y sosteniendo un NPS agregado dentro del rango normal. No se detectaron cambios significativos en métricas operativas ni incidentes operativos relevantes, y no se documentó feedback de clientes.

A nivel diario, el leve ascenso semanal de +3.4 pts en Economy SH no se debe a un pico aislado sino a incrementos constantes y moderados. El 8 de diciembre, por ejemplo, la cabina registró un NPS de 36.80 (+0.7 pts vs media de los últimos 7 días), con IB en 34.9 vs 34.8 y YW en 40.9 vs 38.7, ambos en rango normal. No hubo días con variaciones extremas ni repuntes relevantes que se diluyeran en el consolidado semanal.

**BUSINESS SH: Retrasos y experiencia de producto impactan**  
El segmento Business de SH registró un NPS de 29.3 con una variación de –10.0 puntos con respecto a la semana anterior. Desglose por compañía: IB obtuvo 37.4 (–5.5 ppts según Explanatory Drivers) y YW 9.8 (–21.1 ppts según Explanatory Drivers). El aumento de +64.0 retrasos y +44.0 desvíos impactó la puntuación de puntualidad (–1.0 ppts) y provocó percepciones negativas en interior de cabina (–3.7 ppts), tripulación (–2.2 ppts), embarque (–1.9 ppts), journey preparation support (–1.7 ppts), comida y bebida a bordo (–1.5 ppts) y check-in (–0.2 ppts), parcialmente compensado por conexiones (+2.1 ppts) y ticket price (+0.1 ppts). Las rutas MAD–VCE, AGP–MLN, FLR–MAD, AGP–MAD y DSS–MAD mostraron NPS de –100.0, concentrando la insatisfacción. Los más reactivos fueron flota (250.0 pts), región de residencia (182.7 pts), code-share (104.2 pts) y Business/Leisure (11.0 pts).

A nivel diario, el 8-dic concentró la mayor parte del deterioro en Business SH, con una caída de –16.7 pts con respecto a la media de los últimos 7 días tras 84 retrasos, 30 cancelaciones y 150 pérdidas de conexión (la ruta MAD–SXB quedó en NPS 0.0). A esto se sumaron descensos de –7.2 pts el 5-dic y –5.1 pts el 7-dic, ambos vinculados a picos de incidentes operativos en YW. Aunque el 6-dic registró un repunte de +3.2 pts y el 9-dic IB impulsó +15.4 pts, estos alzas se diluyeron frente a las fuertes caídas previas.

**ECONOMY LH: Desempeño estable**  
La cabina Economy de LH mantuvo desempeño estable, registrando un NPS de 13.0 con una variación de +6.9 puntos con respecto a la semana anterior. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción sin incidentes operativos o feedback de clientes relevantes.

A nivel diario, la aparente estabilidad de Economy LH (NPS 13.0, +6.9 con respecto a la semana anterior) oculta un fuerte deterioro el 7-dic, cuando un desplome de puntualidad (OTP15_adjusted –5.11 pts) y 68 cancelaciones con 58 retrasos hundieron el NPS en 2.38 (–9.9 con respecto a la media de los últimos 7 días) en rutas como DOH–MAD y MAD–MCO. Este episodio puntual contrastó con los repuntes del 6-dic (NPS 16.49, +4.2) y el 5-dic (NPS 19.03, +6.8), que amortiguaron el impacto y diluyeron la caída en el acumulado semanal.

**BUSINESS LH: Mejora notable por puntualidad y fiabilidad**  
La cabina Business de LH registró un NPS de 33.3 con una variación de +18.2 puntos con respecto a la semana anterior. La puntualidad aportó 15.8 ppts según Explanatory Drivers, acompañada por 5.4 ppts de arrivals experience. Las métricas operativas mostraron OTP +3.2, mishandling –2.5, misconex –0.2 y load factor –0.7, mientras que los incidentes operativos sumaron –60.0 cancelaciones. Las rutas MAD–NRT, MAD–MCO, MAD–UIO, JFK–MAD y MAD–SDQ destacaron por sus NPS de 100.0, 50.0, 33.3, –14.3 y –100.0, respectivamente. Code-share (366.7 pts), residence region (98.8 pts), fleet (75.5 pts) y Business/Leisure (42.6 pts) fueron los perfiles más sensibles.

A nivel diario, la mejora semanal de Business LH (+18.2 pts con respecto a la semana anterior) se concentró el 8-dic, cuando el NPS repuntó +21.4 pts con respecto a la media de los últimos 7 días impulsado por la recuperación de la puntualidad y la valoración de la tripulación en la ruta GIG–MAD (NPS 100) pese a 57 retrasos y 36 cancelaciones. El 5-dic aportó un segundo empuje de +11.1 pts (NPS 35.48), apoyado en la satisfacción del perfil Leisure y la flota A350 next. En cambio, el 6-dic (+0.6 pts) y el 7-dic (+1.7 pts) mostraron variaciones leves que diluyeron parcialmente el avance.

**PREMIUM LH: Deterioro crítico en experiencia de cabina**  
El segmento Premium de LH registró un NPS de 2.6 con una variación de –25.3 puntos con respecto a la semana anterior. El interior de cabina restó 12.4 ppts y la tripulación 11.4 ppts según Explanatory Drivers, además de journey preparation support (–5.2 ppts), comida y bebida a bordo (–5.2 ppts) y embarque (–3.2 ppts). A pesar de –60.0 cancelaciones, los +64.0 retrasos y +44.0 desvíos no mitigaron la insatisfacción. MAD–ORD (–50.0, 4 pax), MAD–MCO (100.0, 2 pax), MAD–NRT (33.3, 3 pax), MAD–SJO (100.0, 1 pax) y JFK–MAD (0.0, 2 pax) ilustran la dispersión. Code-share (236.4 pts), residence region (62.5 pts), Business/Leisure y fleet (ambos con spread reducido y variaciones negativas) registraron la mayor reactividad.

A nivel diario, el desplome del 09-dic (–50.0 pts) tras 57 retrasos, 36 cancelaciones y quejas por reasignación de asiento en MAD–UIO explica la mayor parte del deterioro semanal de Premium LH (–25.3 pts con respecto a la semana anterior). A pesar del repunte del 08-dic (+33.3 pts) por mejora de puntualidad y del alza del 05-dic (+10.5 pts) gracias al servicio de la tripulación, estos avances se diluyeron frente a las caídas de inicios de semana. Además, el 07-dic registró otra baja (–4.4 pts) asociada a elevada incidencia operativa, reforzando el balance neto negativo.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0  
- **Pasos de análisis:** 6  
- **Metodología:** Análisis conversacional paso a paso  
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado  

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*

===== STEP 3: EXECUTIVE SYNTHESIS =====

📈 <b>SÍNTESIS EJECUTIVA:</b><br>
Durante la semana del <b>5 al 9 de diciembre de 2025</b>, hemos identificado tres causas principales que explican las variaciones de NPS. El resultado global mostró una subida de <b>+3.2 puntos</b> con respecto a la semana anterior.<br><br>

<b>ECONOMY SH: Desempeño estable</b><br>
La cabina Economy de SH registró un NPS de <b>36.6</b> con una variación de <b>+3.4 puntos</b> con respecto a la semana anterior. Desglose por compañía: <b>IB</b> obtuvo 35.3 (<b>+3.5 ppts según Explanatory Drivers</b>) y <b>YW</b> 39.0 (<b>+2.9 ppts según Explanatory Drivers</b>). Ambas compañías mantuvieron desempeño estable, diluyendo volatilidad interna y sosteniendo un NPS agregado dentro del rango normal. No se detectaron cambios significativos en métricas operativas ni incidentes operativos relevantes, y no se documentó feedback de clientes.<br><br>
A nivel diario, el leve ascenso semanal de <b>+3.4 pts</b> en Economy SH no se debe a un pico aislado sino a incrementos constantes y moderados. El <b>8 de diciembre</b>, por ejemplo, la cabina registró un NPS de 36.80 (<b>+0.7 pts vs media de los últimos 7 días</b>), con IB en 34.9 vs 34.8 y YW en 40.9 vs 38.7, ambos en rango normal. No hubo días con variaciones extremas ni repuntes relevantes que se diluyeran en el consolidado semanal.<br><br>

<b>BUSINESS SH: Retrasos y experiencia de producto impactan</b><br>
El segmento Business de SH registró un NPS de <b>29.3</b> con una variación de <b>–10.0 puntos</b> con respecto a la semana anterior. Desglose por compañía: <b>IB</b> obtuvo 37.4 (<b>–5.5 ppts según Explanatory Drivers</b>) y <b>YW</b> 9.8 (<b>–21.1 ppts según Explanatory Drivers</b>). El aumento de <b>+64.0 retrasos</b> y <b>+44.0 desvíos</b>, junto a <b>–60.0 cancelaciones</b>, impactó la puntualidad (<b>–1.0 ppts según Explanatory Drivers</b>) y provocó percepciones negativas sobre el interior de cabina (<b>–3.7 ppts</b>), la tripulación (<b>–2.2 ppts</b>), el embarque (<b>–1.9 ppts</b>), el apoyo en la preparación del viaje (<b>–1.7 ppts</b>), la oferta de comida y bebida a bordo (<b>–1.5 ppts</b>) y el check-in (<b>–0.2 ppts</b>), pese a la mejora en connections experience (<b>+2.1 ppts</b>) y ticket price (<b>+0.1 ppts</b>). Las rutas MAD–VCE, AGP–MLN, FLR–MAD, AGP–MAD y DSS–MAD mostraron NPS de –100.0, concentrando la insatisfacción. Los perfiles más sensibles fueron flota (spread 250.0 pts), región de residencia (182.7 pts), code-share (104.2 pts) y Business/Leisure (11.0 pts). No se detectaron variaciones relevantes en métricas operativas ni feedback de clientes documentado.<br><br>
A nivel diario, el <b>8-dic</b> concentró la mayor parte del deterioro en Business SH, con una caída de <b>–16.7 pts</b> con respecto a la media de los últimos 7 días tras 84 retrasos, 30 cancelaciones y 150 pérdidas de conexión (la ruta MAD–SXB quedó en NPS 0.0). A esto se sumaron descensos de <b>–7.2 pts el 5-dic</b> y <b>–5.1 pts el 7-dic</b>, ambos vinculados a picos de incidentes operativos en YW. Aunque el <b>6-dic</b> registró un repunte de <b>+3.2 pts</b> y el <b>9-dic</b> IB impulsó <b>+15.4 pts</b>, estos alzas se diluyeron frente a las fuertes caídas previas.<br><br>

<b>ECONOMY LH: Desempeño estable</b><br>
La cabina Economy de LH mantuvo desempeño estable, registrando un NPS de <b>13.0</b> con una variación de <b>+6.9 puntos</b> con respecto a la semana anterior. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción sin incidentes operativos o feedback de clientes relevantes.<br><br>
A nivel diario, la aparente estabilidad de Economy LH (NPS 13.0, <b>+6.9 con respecto a la semana anterior</b>) oculta un fuerte deterioro el <b>7-dic</b>, cuando un desplome de puntualidad (OTP15_adjusted –5.11 pts) y 68 cancelaciones con 58 retrasos hundieron el NPS en 2.38 (<b>–9.9 con respecto a la media de los últimos 7 días</b>) en rutas como DOH–MAD y MAD–MCO. Este episodio puntual contrastó con los repuntes del <b>6-dic</b> (NPS 16.49, +4.2) y el <b>5-dic</b> (NPS 19.03, +6.8), que amortiguaron el impacto y diluyeron la caída en el acumulado semanal.<br><br>

<b>BUSINESS LH: Mejora notable por puntualidad y fiabilidad</b><br>
La cabina Business de LH registró un NPS de <b>33.3</b> con una variación de <b>+18.2 puntos</b> con respecto a la semana anterior. La puntualidad aportó <b>15.8 ppts</b> según Explanatory Drivers, acompañada por <b>5.4 ppts de arrivals experience</b>. Las métricas operativas mostraron OTP +3.2, mishandling –2.5, misconex –0.2 y load factor –0.7, mientras que los incidentes operativos sumaron –60.0 cancelaciones. Las rutas MAD–NRT, MAD–MCO, MAD–UIO, JFK–MAD y MAD–SDQ destacaron por sus NPS de 100.0, 50.0, 33.3, –14.3 y –100.0, respectivamente. Code-share (366.7 pts), residence region (98.8 pts), fleet (75.5 pts) y Business/Leisure (42.6 pts) fueron los perfiles más sensibles.<br><br>
A nivel diario, la mejora semanal de Business LH (<b>+18.2 pts con respecto a la semana anterior</b>) se concentró el <b>8-dic</b>, cuando el NPS repuntó +21.4 pts con respecto a la media de los últimos 7 días impulsado por la recuperación de la puntualidad y la valoración de la tripulación en la ruta GIG–MAD (NPS 100) pese a 57 retrasos y 36 cancelaciones. El <b>5-dic</b> aportó un segundo empuje de +11.1 pts (NPS 35.48), apoyado en la satisfacción del perfil Leisure y la flota A350 next. En cambio, el <b>6-dic</b> (+0.6 pts) y el <b>7-dic</b> (+1.7 pts) mostraron variaciones leves que diluyeron parcialmente el avance.<br><br>

<b>PREMIUM LH: Deterioro crítico en experiencia de cabina</b><br>
El segmento Premium de LH registró un NPS de <b>2.6</b> con una variación de <b>–25.3 puntos</b> con respecto a la semana anterior. El interior de cabina restó 12.4 ppts y la tripulación 11.4 ppts según Explanatory Drivers, además de journey preparation support (–5.2 ppts), comida y bebida a bordo (–5.2 ppts) y embarque (–3.2 ppts). A pesar de –60.0 cancelaciones, los +64.0 retrasos y +44.0 desvíos no mitigaron la insatisfacción. MAD–ORD (–50.0, 4 pax), MAD–MCO (100.0, 2 pax), MAD–NRT (33.3, 3 pax), MAD–SJO (100.0, 1 pax) y JFK–MAD (0.0, 2 pax) ilustran la dispersión. Code-share (236.4 pts), residence region (62.5 pts), Business/Leisure y fleet (ambos con spread reducido y variaciones negativas) registraron la mayor reactividad.<br><br>
A nivel diario, el desplome del <b>09-dic</b> (–50.0 pts) tras 57 retrasos, 36 cancelaciones y quejas por reasignación de asiento en MAD–UIO explica la mayor parte del deterioro semanal de Premium LH (<b>–25.3 pts con respecto a la semana anterior</b>). A pesar del repunte del <b>08-dic</b> (+33.3 pts) por mejora de puntualidad y del alza del <b>05-dic</b> (+10.5 pts) gracias al servicio de la tripulación, estos avances se diluyeron frente a las caídas de inicios de semana. Además, el <b>07-dic</b> registró otra baja (<b>–4.4 pts</b>) asociada a elevada incidencia operativa, reforzando el balance neto negativo.