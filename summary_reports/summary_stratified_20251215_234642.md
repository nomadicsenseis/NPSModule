===== STEP 1: SECTION CONNECTIONS =====

--- BUSINESS SH ---
A nivel diario, la caída semanal de Business SH encontró su momento crítico el 8-dic, cuando el NPS retrocedió –16.7 pts tras 30 cancelaciones, 84 retrasos y 150 reprogramaciones en MAD, reflejando el aumento de incidentes y desvíos señalado en el análisis semanal. El 7-dic añadió presión con 307 NCS en YW que explicaron un descenso extra de –5.1 pts en el segmento. Por su parte, el repunte del 9-dic (+15.4 pts gracias a un OTP15_adjusted +2.11 pts en IB) no bastó para revertir el deterioro acumulado.

--- ECONOMY SH ---
A nivel diario, la contribución de Economy SH se vio en los vaivenes que acabaron neutralizándose. El 6-dic registró un repunte de +6,3 pts (NPS 42,36) y el 7-dic otro de +5,1 pts (NPS 41,15), ambos sostenidos por IB, mientras que el 8-dic sufrió una caída moderada de –1,9 pts (NPS 34,15; IB –3,8 pts vs media de 7 días). El 9-dic remató con una NPS prácticamente estable (36,80; +0,7 pts), lo que confirma que Economy SH amortiguó el impacto negativo global.

--- BUSINESS LH ---
A nivel diario, la recuperación de Business LH que compensó la caída de Premium LH se concentró especialmente el 08-dic, cuando el segmento sumó 10,6 pts (NPS 35,0) gracias a verbatims muy positivos en la ruta GIG–MAD. Ese mismo día Premium LH se hundió 50,0 pts tras 57 retrasos y 36 cancelaciones, acentuando el desequilibrio entre ambos segmentos. Los incrementos más modestos del 06-dic (+0,6 pts) y del 07-dic (+1,7 pts) quedaron diluidos frente a esta intensa volatilidad.

--- ECONOMY LH ---
A nivel diario, la estabilidad semanal de Economy LH responde a oscilaciones compensadas entre el 6-dic (alza de +7.8 pts hasta NPS 20.10, impulsada por 336 verbatims positivos en rutas BOG–MAD y MAD–SDQ) y el 7-dic (caída de –9.9 pts hasta 2.38 por 308 incidentes y baja puntualidad en DOH–MAD). Además, el 8-dic sumó +4.2 pts (NPS 16.49) y el 9-dic restó –6.9 pts (NPS 5.42), de modo que estas variaciones extremas se neutralizaron en el cómputo semanal.

--- PREMIUM LH ---
A nivel diario, el deterioro semanal de Premium LH se explica principalmente por el 9-dic, cuando el NPS se desplomó a –50.0 pts (–50 vs la media de los últimos 7 días) tras 57 retrasos, 36 cancelaciones y 18 pérdidas de conexión en MAD, con 14 verbatims que reprocharon cambios de asiento, mala atención de tripulación y comida deficiente, reflejando exactamente las caídas en Aircraft interior y Cabin Crew detectadas en el análisis semanal. Ese fuerte descenso desbordó el repunte del 8-dic (+33.3 pts en MAD–MCO por confort y servicio) y se vio precedido por la presión operativa iniciada el 5-dic, cuando fenómenos meteorológicos en NTE y MLN y un fallo técnico en IB1212 generaron picos de desvíos y retrasos que agravaron la percepción de fiabilidad.

--- GLOBAL ---
A nivel diario, la baja semanal de –1.3 pts se explica sobre todo por dos jornadas negativas: el 7-dic cayó –0.6 pts presionado por el desplome de Premium LH tras 57 retrasos y 36 cancelaciones, y el 9-dic sumó –0.9 pts arrastrado por el NPS de Premium LH en –50 pts en la ruta MAD–UIO. El fuerte repunte del 6-dic (+6.1 pts gracias al feedback muy positivo en Economy SH) se diluyó con estas caídas, neutralizando su impulso en el cómputo semanal.

===== STEP 2: FULL REPORT =====

📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En SH/Economy el patrón es (N, N | N): ambos hijos (IB y YW) y el padre están etiquetados como “Normal”, por lo que no hay anomalía que explicar.

En SH/Business el patrón es (–, – | –): SINERGIA.

- Narrativa: Adoptamos la explicación del nodo padre Global/SH/Business. La caída de –10.0 pts en NPS de Business SH se explica principalmente por un empeoramiento de la puntualidad en el periodo actual, validado por el driver de puntualidad (SHAP = –1.0) y el aumento de retrasos (+64 incidentes) y desvíos (+44 incidentes), agravado por eventos meteorológicos y técnicos que impactaron a ambos IB y YW.

- Evidencia clave:  
  • Punctuality SHAP = –1.0 y Sat_diff = –3.2 (Global/SH/Business)  
  • Retrasos Δ=+64 y Desvíos Δ=+44 (ncs_tool para Global/SH/Business)

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En SH (Economy N, Business – | SH N):  
- **Dinámica:** DILUCIÓN  
- **Narrativa:** El resultado de SH está dictado por la caída en Business SH, aunque el buen desempeño de Economy SH amortigua el impacto en el agregado.  
- **Evidencia:** Punctuality SHAP = –1.0 y Sat_diff = –3.2 (Global/SH/Business); Retrasos Δ=+64 y Desvíos Δ=+44 (ncs_tool para Global/SH/Business).

En LH (Economy N, Business +, Premium – | LH N):  
- **Dinámica:** CANCELACIÓN  
- **Narrativa:** El NPS de LH parece neutro porque la fuerte mejora en Business LH se contrarresta con la caída en Premium LH, y Economy LH se mantuvo estable.  
- **Evidencia:**  
  • Business LH mejoró por OTP15 +3.2 y Punctuality SHAP = +15.8 (operative_data_tool + explanatory_drivers_tool).  
  • Premium LH cayó por Aircraft interior SHAP = –12.4 y Cabin Crew SHAP = –11.4 (explanatory_drivers_tool).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA (N, N | +).  
- Narrativa: Aunque ni LH ni SH muestran anomalías netas a nivel cabina, el NPS Global se vio impulsado por un efecto sistémico de mejora operacional. La excelente gestión de la puntualidad, junto con refuerzos en arrivals experience y check-in, arrastraron el indicador global al alza.  
- Evidencia:  
  • Punctuality SHAP = 1.1 y Sat_diff = 2.5 (Global)  
  • OTP15 +0.4 pts y cancelaciones –60 incidentes (operative_data_tool)  
  • Arrivals experience SHAP = 0.7 y Check-in SHAP = 0.6 (explanatory_drivers_tool)

A nivel diario, la baja semanal de –1.3 pts se explica sobre todo por dos jornadas negativas: el 7-dic cayó –0.6 pts presionado por el desplome de Premium LH tras 57 retrasos y 36 cancelaciones, y el 9-dic sumó –0.9 pts arrastrado por el NPS de Premium LH en –50 pts en la ruta MAD–UIO. El fuerte repunte del 6-dic (+6.1 pts gracias al feedback muy positivo en Economy SH) se diluyó con estas caídas, neutralizando su impulso en el cómputo semanal.

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

CAUSA 1: Deterioro de puntualidad en Short Haul Business  
- Escenario: DILUCIÓN `(–, N | N)`  
- NMA: Global/SH/Business  
- Afecta a: SH/Business/IB y SH/Business/YW  
- Tipo de impacto: NEGATIVO  

CAUSA 2: Mejora operativa de puntualidad y OTP en Long Haul Business  
- Escenario: CANCELACIÓN `(+,- | N)`  
- NMA: Global/LH/Business  
- Afecta a: LH/Business (consolidado)  
- Tipo de impacto: POSITIVO  

CAUSA 3: Deficiencias de producto en Long Haul Premium  
- Escenario: CANCELACIÓN `(+,- | N)`  
- NMA: Global/LH/Premium  
- Afecta a: LH/Premium (consolidado)  
- Tipo de impacto: NEGATIVO

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

=== NMA: Global/SH/Business ===

📈 EXPLANATORY DRIVERS:  
• Punctuality: SHAP = –1.0, Sat_diff = –3.2 (explanatory_drivers_tool)

📊 DATOS OPERATIVOS:  
No disponible

🚨 INCIDENTES NCS (CUANTITATIVO):  
• Retrasos: +64  
• Desvíos: +44  
• Cancelaciones: –60 (INCIDENTES_DELTA: Cancelaciones –60; Retrasos +64; Desvíos +44)

🧠 NCS (CUALITATIVO / REFLEXIÓN):  
• Período ACTUAL (05-dic a 11-dic):  
  – 2025-12-05: condiciones meteorológicas adversas en NTE y MLN que causaron desvíos y retrasos (IB1222, IB2271, IB1221)  
  – 2025-12-05: fallo técnico en IB1212  
  → Estos eventos se corresponderían con picos de insatisfacción al llegar cambios de ruta o demoras de última hora.  
• Período de COMPARACIÓN (L7d: 28-nov a 04-dic):  
  – 2025-11-28: huelga en Italia (IB0674, IB1238, IB1237, IB0671, IB0673, IB0672)  
  – 2025-11-28: fallos informáticos en I21873  
  → El baseline también estaba afectado por cancelaciones anticipadas, cuya gestión pudo haber sido mejor percibida que las demoras inesperadas actuales.

💬 FEEDBACK DE CLIENTES:  
No disponible

✈️ RUTAS AFECTADAS (Top 5):  
• MAD-VCE: Pax 2 (fuente: explanatory_drivers)  
• AGP-MLN: Pax 1 (fuente: explanatory_drivers)  
• FLR-MAD: Pax 1 (fuente: explanatory_drivers)  
• AGP-MAD: Pax 1 (fuente: explanatory_drivers)  
• DSS-MAD: Pax 2 (fuente: explanatory_drivers)

👥 PERFILES REACTIVOS:  
• Fleet: spread NPS_diff = 250.0 pts → pasajeros en determinadas flotas presentan reacciones extremas.  
• Residence Region: spread NPS_diff = 182.7 pts → variabilidad alta según región de residencia.  
• CodeShare: spread NPS_diff = 104.2 pts → niveles de insatisfacción muy distintos entre vuelos propios y compartidos.  
• Business/Leisure: spread NPS_diff bajo → similar impacto competitivo entre viajeros de negocios y ocio.

=== NMA: Global/LH/Business ===

📈 EXPLANATORY DRIVERS:  
• Punctuality SHAP = 15.8, Sat_diff = 9.4 (explanatory_drivers_tool)  
• Journey preparation support: SHAP = –4.5, Sat_diff = –13.3  
• Arrivals experience: SHAP = 5.4  
• In flight food and beverage: SHAP = 2.1  
• Check-in: SHAP = 1.6

📊 DATOS OPERATIVOS:  
• OTP15 subió en 3.2, Load Factor cayó en 0.7, Mishandling disminuyó en 2.5, Misconexiones en 0.2 (operative_data_tool)

🚨 INCIDENTES NCS (CUANTITATIVO):  
• Cancelaciones Δ=–60  
• Retrasos Δ=+64  
• Desvíos Δ=+44  
• Limitación de aeronave Δ=–14 (ncs_tool)

🧠 NCS (CUALITATIVO / REFLEXIÓN):  
• Período ACTUAL (2025-12-05 a 2025-12-11):  
  – 2025-12-05: condiciones meteorológicas que provocaron cancelación/desvío de IB1222 (NTE) e IB2271 (MLN)  
  ⇒ Generaron retrasos y desvíos (Δ+44), pero no afectaron significativamente el total de cancelaciones.  
• Período COMPARATIVO (2025-11-28 a 2025-12-04):  
  – 2025-11-28: huelga en Italia afectando vuelos IB0674, IB1238, IB1237, IB0671, IB0673, IB0672  
  – 2025-11-28: fallos informáticos en rotación de avión (vuelo I21873)  
  ⇒ Elevado nivel de cancelaciones y problemas de sistema en el baseline, lo que deprimió el NPS de los últimos 7 días anteriores.

💬 FEEDBACK DE CLIENTES:  
No disponible

✈️ RUTAS AFECTADAS (Top 5):  
• MAD-NRT: NPS 100.0, Pax 1 (explanatory_drivers_tool)  
• MAD-MCO: NPS 50.0, Pax 2 (explanatory_drivers_tool)  
• JFK-MAD: NPS –14.3, Pax 7 (explanatory_drivers_tool)  
• MAD-UIO: NPS 33.3, Pax 3 (explanatory_drivers_tool)  
• MAD-SDQ: NPS –100.0, Pax 1 (explanatory_drivers_tool)

👥 PERFILES REACTIVOS:  
• CodeShare: spread 366.7 pts  
• Residence Region: spread 98.8 pts  
• Fleet: spread 75.5 pts  
• Business/Leisure: spread 42.6 pts

=== NMA: Global/LH/Premium ===

📈 EXPLANATORY DRIVERS:  
• Aircraft interior: SHAP = –12.4 ; Sat_diff = –18.1  
• Cabin Crew: SHAP = –11.4 ; Sat_diff = –8.3  
• Journey preparation support: SHAP = –5.2 ; Sat_diff = –12.1  
• In flight food and beverage: SHAP = –5.2 ; Sat_diff = –4.3  
• Boarding: SHAP = –3.2 ; Sat_diff = –3.6 (explanatory_drivers_tool)

📊 DATOS OPERATIVOS:  
No disponible

🚨 INCIDENTES NCS (CUANTITATIVO):  
• Retrasos: +64  
• Desvíos: +44  
• Cancelaciones: –60 (ncs_tool)

🧠 NCS (CUALITATIVO / REFLEXIÓN):  
• Período ACTUAL (2025-12-05 a 2025-12-11):  
  – 2025-12-05: causas meteorológicas en NTE (IB1222) y MLN (IB2271)  
  – 2025-12-05: fallo técnico de avión en IB1212  
• Período COMPARATIVO (últimos 7 días previos):  
  – 2025-11-28: huelga en Italia (IB0674, IB0673, IB0672, IB0671, IB1237, IB1238)  
  – 2025-11-28: rotación de avión y fallos informáticos (I21873)  
• Impacto: en el período actual los fenómenos meteorológicos y el fallo técnico concentraron reubicaciones y desvíos, agravando la percepción de fiabilidad. En el baseline había huelgas masivas que influyeron en cancelaciones.

💬 FEEDBACK DE CLIENTES:  
No disponible

✈️ RUTAS AFECTADAS (Top 5):  
• MAD-ORD: NPS –50.0, Pax 4  
• MAD-MCO: NPS 100.0, Pax 2  
• MAD-NRT: NPS 33.3, Pax 3  
• MAD-SJO: NPS 100.0, Pax 1  
• JFK-MAD: NPS 0.0, Pax 2

👥 PERFILES REACTIVOS:  
• CodeShare: spread de NPS_diff = 236.4 pts (clientes en vuelos compartidos muy sensibles a los problemas)  
• Residence Region: spread de NPS_diff = 62.5 pts  
• Business vs Leisure: variación marginal  
• Fleet: variación marginal

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 5 al 9 de diciembre de 2025, la red experimentó una subida de +3.2 puntos en NPS con respecto a la semana anterior, impulsada por tendencias opuestas en diferentes cabinas. Mientras Business de LH repuntó con fuerza y Economy de SH mantuvo estabilidad, Premium de LH y Business de SH mostraron retrocesos significativos.

El segmento Business de LH registró un NPS de 33.3, +18.2 puntos con respecto a la semana anterior, gracias a una mejora de OTP de +3.2 ppts según métricas operativas en LH Business y un salto de puntualidad de +15.8 ppts según Explanatory Drivers en LH Business. La reducción de 60 cancelaciones según incidentes operativos y el refuerzo de la Arrivals experience (+5.4 ppts) y Check-in (+1.6 ppts) consolidaron el avance. Las rutas más beneficiadas fueron MAD-NRT (NPS 100.0, 1 pax) y MAD-MCO (NPS 50.0, 2 pax), con mayor sensibilidad en clientes code-share (spread 366.7 pts) y por región de residencia (spread 98.8 pts).

En contraste, Premium de LH se desplomó hasta un NPS de 2.6, –25.3 puntos con respecto a la semana anterior, por deficiencias de producto: interior de cabina (–12.4 ppts según Explanatory Drivers en LH Premium) y Cabin Crew (–11.4 ppts según Explanatory Drivers en LH Premium). A este deterioro se unió un aumento de +64 retrasos y +44 desvíos según incidentes operativos, exacerbados por condiciones meteorológicas en NTE y MLN y un fallo técnico en IB1212. Las caídas más notables se dieron en MAD-ORD (NPS –50.0, 4 pax) y JFK-MAD (NPS 0.0, 2 pax), con reacciones extremas en clientes code-share (spread de NPS_diff = 236.4 pts) y según región de residencia (spread de NPS_diff = 62.5 pts).

La cabina Business de SH vio su NPS descender a 29.3, –10.0 puntos con respecto a la semana anterior, por un empeoramiento de puntualidad de –1.0 ppts según Explanatory Drivers en SH Business y un alza de +64 retrasos y +44 desvíos según incidentes operativos. Las condiciones meteorológicas adversas en NTE y MLN y el incidente técnico en IB1212 intensificaron la frustración. IB marcó 37.4 (–5.5 ppts) y YW 9.8 (–21.1 ppts), afectando especialmente rutas como MAD-VCE (NPS –100.0, 2 pax) y FLR-MAD (NPS –100.0, 1 pax), con gran sensibilidad en ciertas flotas (spread 250.0 pts) y regiones de residencia (spread 182.7 pts).

**DETALLE POR CABINA:**

Economy SH: mantuvo desempeño estable, con un NPS de 36.6 y una subida de +3.4 puntos con respecto a la semana anterior. Desglose por compañía: IB alcanzó 35.4 (+3.5 ppts) y YW registró 39.0 (+2.9 ppts).  
A nivel diario, la contribución de Economy SH se vio en los vaivenes que acabaron neutralizándose. El 6-dic registró un repunte de +6.3 pts (NPS 42.4) y el 7-dic otro de +5.1 pts (NPS 41.2), ambos sostenidos por IB, mientras que el 8-dic sufrió una caída moderada de –1.9 pts (NPS 34.2; IB –3.8 pts vs media de 7 días). El 9-dic remató con una NPS prácticamente estable (36.8; +0.7 pts), lo que confirma que Economy SH amortiguó el impacto negativo global.

Business SH: registró un NPS de 29.3, –10.0 puntos con respecto a la semana anterior. Desglose por compañía: IB obtuvo 37.4 (–5.5 ppts según Explanatory Drivers en SH Business) y YW 9.8 (–21.1 ppts según Explanatory Drivers en SH Business). El retroceso se explica por un deterioro de puntualidad de –1.0 ppts según Explanatory Drivers en SH Business y un incremento de +64 retrasos y +44 desvíos según incidentes operativos, agravados por condiciones meteorológicas en NTE y MLN y un incidente técnico en IB1212.  
A nivel diario, la caída semanal de Business SH encontró su momento crítico el 8-dic, cuando el NPS retrocedió –16.7 pts tras 30 cancelaciones, 84 retrasos y 150 reprogramaciones en MAD, reflejando el aumento de incidentes y desvíos señalado en el análisis semanal. El 7-dic añadió presión con 307 NCS en YW que explicaron un descenso extra de –5.1 pts en el segmento. Por su parte, el repunte del 9-dic (+15.4 pts gracias a un OTP15_adjusted +2.1 pts en IB) no bastó para revertir el deterioro acumulado.

Economy LH: mantuvo desempeño estable, con un NPS de 13.0 y una subida de +6.9 puntos con respecto a la semana anterior. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción.  
A nivel diario, la estabilidad semanal de Economy LH responde a oscilaciones compensadas entre el 6-dic (alza de +7.8 pts hasta NPS 20.1, impulsada por 336 verbatims positivos en rutas BOG–MAD y MAD–SDQ) y el 7-dic (caída de –9.9 pts hasta 2.4 por 308 incidentes y baja puntualidad en DOH–MAD). Además, el 8-dic sumó +4.2 pts (NPS 16.5) y el 9-dic restó –6.9 pts (NPS 5.4), de modo que estas variaciones extremas se neutralizaron en el cómputo semanal.

Business LH: registró un NPS de 33.3 y una subida de +18.2 puntos con respecto a la semana anterior. Los drivers principales fueron la puntualidad (+15.8 ppts según Explanatory Drivers en LH Business) y la mejora de OTP (+3.2 ppts según métricas operativas en LH Business), junto con una reducción de 60 cancelaciones según incidentes operativos. Arrivals experience (+5.4 ppts) y Check-in (+1.6 ppts) añadieron valor. Las rutas más beneficiadas incluyeron MAD-NRT (100.0, 1 pax) y MAD-UIO (33.3, 3 pax), con mayor impacto en clientes code-share (spread 366.7 pts) y por región de residencia (spread 98.8 pts).  
A nivel diario, la recuperación de Business LH que compensó la caída de Premium LH se concentró especialmente el 08-dic, cuando el segmento sumó 10.6 pts (NPS 35.0) gracias a verbatims muy positivos en la ruta GIG–MAD. Ese mismo día Premium LH se hundió 50.0 pts tras 57 retrasos y 36 cancelaciones, acentuando el desequilibrio entre ambos segmentos. Los incrementos más modestos del 06-dic (+0.6 pts) y del 07-dic (+1.7 pts) quedaron diluidos frente a esta intensa volatilidad.

Premium LH: registró un NPS de 2.6, –25.3 puntos con respecto a la semana anterior. Las causas dominantes fueron deficiencias de producto —interior de cabina (–12.4 ppts según Explanatory Drivers en LH Premium) y Cabin Crew (–11.4 ppts según Explanatory Drivers en LH Premium)— combinadas con +64 retrasos y +44 desvíos según incidentes operativos, agravados por condiciones meteorológicas en NTE y MLN y un fallo técnico en IB1212. Las rutas más afectadas fueron MAD-ORD (–50.0, 4 pax) y MAD-MCO (100.0, 2 pax), con reacciones especialmente intensas entre clientes code-share (spread de NPS_diff = 236.4 pts) y según región de residencia (spread de NPS_diff = 62.5 pts).  
A nivel diario, el deterioro semanal de Premium LH se explica principalmente por el 9-dic, cuando el NPS se desplomó a –50.0 pts (–50 vs la media de los últimos 7 días) tras 57 retrasos, 36 cancelaciones y 18 pérdidas de conexión en MAD, con 14 verbatims que reprocharon cambios de asiento, mala atención de tripulación y comida deficiente, reflejando exactamente las caídas en Aircraft interior y Cabin Crew detectadas en el análisis semanal. Ese fuerte descenso desbordó el repunte del 8-dic (+33.3 pts en MAD–MCO por confort y servicio) y se vio precedido por la presión operativa iniciada el 5-dic, cuando fenómenos meteorológicos en NTE y MLN y un fallo técnico en IB1212 generaron picos de desvíos y retrasos que agravaron la percepción de fiabilidad.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0  
- **Pasos de análisis:** 6  
- **Metodología:** Análisis conversacional paso a paso  
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*

===== STEP 3: EXECUTIVE SYNTHESIS =====

📈 <b>SÍNTESIS EJECUTIVA:</b><br>
Durante la semana del 5 al 9 de diciembre de 2025, la red experimentó una subida de +3.2 puntos en NPS con respecto a la semana anterior, impulsada por tendencias opuestas en diferentes cabinas. Mientras Business de LH repuntó con fuerza y Economy de SH mantuvo estabilidad, Premium de LH y Business de SH mostraron retrocesos significativos.<br><br>

Economy SH: mantuvo desempeño estable, con un NPS de 36.6 y una subida de +3.4 puntos con respecto a la semana anterior. Desglose por compañía: IB alcanzó 35.4 (+3.5 ppts) y YW registró 39.0 (+2.9 ppts).<br>
A nivel diario, la contribución de Economy SH se vio en los vaivenes que acabaron neutralizándose. El 6-dic registró un repunte de +6.3 pts (NPS 42.4) y el 7-dic otro de +5.1 pts (NPS 41.2), ambos sostenidos por IB, mientras que el 8-dic sufrió una caída moderada de –1.9 pts (NPS 34.2; IB –3.8 pts vs media de 7 días). El 9-dic remató con una NPS prácticamente estable (36.8; +0.7 pts), lo que confirma que Economy SH amortiguó el impacto negativo global.<br><br>

Business SH: registró un NPS de 29.3, –10.0 puntos con respecto a la semana anterior. Desglose por compañía: IB obtuvo 37.4 (–5.5 ppts según Explanatory Drivers en SH Business) y YW 9.8 (–21.1 ppts según Explanatory Drivers en SH Business). El retroceso se explica por un deterioro de puntualidad de –1.0 ppts según Explanatory Drivers en SH Business y un incremento de +64 retrasos y +44 desvíos según incidentes operativos, agravados por condiciones meteorológicas en NTE y MLN y un incidente técnico en IB1212.<br>
A nivel diario, la caída semanal de Business SH encontró su momento crítico el 8-dic, cuando el NPS retrocedió –16.7 pts tras 30 cancelaciones, 84 retrasos y 150 reprogramaciones en MAD, reflejando el aumento de incidentes y desvíos señalado en el análisis semanal. El 7-dic añadió presión con 307 NCS en YW que explicaron un descenso extra de –5.1 pts en el segmento. Por su parte, el repunte del 9-dic (+15.4 pts gracias a un OTP15_adjusted +2.1 pts en IB) no bastó para revertir el deterioro acumulado.<br><br>

Economy LH: mantuvo desempeño estable, con un NPS de 13.0 y una subida de +6.9 puntos con respecto a la semana anterior. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción.<br>
A nivel diario, la estabilidad semanal de Economy LH responde a oscilaciones compensadas entre el 6-dic (alza de +7.8 pts hasta NPS 20.1, impulsada por 336 verbatims positivos en rutas BOG–MAD y MAD–SDQ) y el 7-dic (caída de –9.9 pts hasta 2.4 por 308 incidentes y baja puntualidad en DOH–MAD). Además, el 8-dic sumó +4.2 pts (NPS 16.5) y el 9-dic restó –6.9 pts (NPS 5.4), de modo que estas variaciones extremas se neutralizaron en el cómputo semanal.<br><br>

Business LH: registró un NPS de 33.3 y una subida de +18.2 puntos con respecto a la semana anterior. Los drivers principales fueron la puntualidad (+15.8 ppts según Explanatory Drivers en LH Business) y la mejora de OTP (+3.2 ppts según métricas operativas en LH Business), junto con una reducción de 60 cancelaciones según incidentes operativos. Arrivals experience (+5.4 ppts) y Check-in (+1.6 ppts) añadieron valor. Las rutas más beneficiadas incluyeron MAD-NRT (100.0, 1 pax) y MAD-UIO (33.3, 3 pax), con mayor impacto en clientes code-share (spread 366.7 pts) y por región de residencia (spread 98.8 pts).<br>
A nivel diario, la recuperación de Business LH que compensó la caída de Premium LH se concentró especialmente el 08-dic, cuando el segmento sumó 10.6 pts (NPS 35.0) gracias a verbatims muy positivos en la ruta GIG–MAD. Ese mismo día Premium LH se hundió 50.0 pts tras 57 retrasos y 36 cancelaciones, acentuando el desequilibrio entre ambos segmentos. Los incrementos más modestos del 06-dic (+0.6 pts) y del 07-dic (+1.7 pts) quedaron diluidos frente a esta intensa volatilidad.<br><br>

Premium LH: registró un NPS de 2.6, –25.3 puntos con respecto a la semana anterior. Las causas dominantes fueron deficiencias de producto —interior de cabina (–12.4 ppts según Explanatory Drivers en LH Premium) y Cabin Crew (–11.4 ppts según Explanatory Drivers en LH Premium)— combinadas con +64 retrasos y +44 desvíos según incidentes operativos, agravados por condiciones meteorológicas en NTE y MLN y un fallo técnico en IB1212. Las rutas más afectadas fueron MAD-ORD (–50.0, 4 pax) y MAD-MCO (100.0, 2 pax), con reacciones especialmente intensas entre clientes code-share (spread de NPS_diff = 236.4 pts) y según región de residencia (spread de NPS_diff = 62.5 pts).<br>
A nivel diario, el deterioro semanal de Premium LH se explica principalmente por el 9-dic, cuando el NPS se desplomó a –50.0 pts (–50 vs la media de los últimos 7 días) tras 57 retrasos, 36 cancelaciones y 18 pérdidas de conexión en MAD, con 14 verbatims que reprocharon cambios de asiento, mala atención de tripulación y comida deficiente, reflejando exactamente las caídas en Aircraft interior y Cabin Crew detectadas en el análisis semanal. Ese fuerte descenso desbordó el repunte del 8-dic (+33.3 pts en MAD–MCO por confort y servicio) y se vio precedido por la presión operativa iniciada el 5-dic, cuando fenómenos meteorológicos en NTE y MLN y un fallo técnico en IB1212 generaron picos de desvíos y retrasos que agravaron la percepción de fiabilidad.