===== STEP 1: SECTION CONNECTIONS =====

--- ECONOMY SH ---
A nivel diario, Economy SH confirmó la estabilidad reflejada en el informe semanal, sin que ningún día registrara variaciones superiores a ±3 pts con respecto a la media de los últimos 7 días. Por ejemplo, el 9-dic IB cerró en 34.85 (vs 34.80, +0.1 pts) y YW en 40.87 (vs 38.75, +2.1 pts), ambos sin incidencias operativas relevantes. Los ligeros repuntes de YW a mediados de semana se diluyeron en el consolidado, reforzando el estado “Normal” de la cabina.

--- BUSINESS SH ---
A nivel diario, la caída semanal de Business SH se explica sobre todo por el **08-dic**, cuando 326 incidentes operativos (84 retrasos, 30 cancelaciones y 20 pérdidas de conexión) hundieron el NPS 16.7 pts (IB –21.9, YW –5.4) y la ruta MAD–ZRH tocó –40.0 pts. El **06-dic**, un nuevo pico de 103 retrasos, 38 cancelaciones y 93 misconexiones reforzó ese deterioro. Los repuntes del **09-dic** (+15.4 pts por puntualidad y catering en IB) y del **07-dic** (alza de IB neutralizada por caída de YW) se diluyeron en el agregado semanal.

--- BUSINESS LH ---
A nivel diario, el alza de +18.2 puntos de Business LH se fraguó principalmente el 06-dic, cuando el NPS de este segmento escaló +21.4 pts con respecto a la media de los últimos 7 días gracias al feedback de calidad de servicio y confort en la ruta GRU–MAD (NPS 7.7, 13 encuestas), coincidiendo además con un descenso a 30 cancelaciones, el mínimo de la semana. La finalización de la huelga en Italia a inicios de semana se materializó ese día, impulsando la drástica reducción de cancelaciones. Por su parte, el repunte inicial del 05-dic (+11.1 pts) se vio parcialmente diluido por la persistencia de demoras.

--- PREMIUM LH ---
A nivel diario, la caída semanal de –25.3 pts en Premium LH se concentró principalmente el 09-dic, cuando un OTP de 71.27 (–6.99 pts) junto a 57 retrasos y 36 cancelaciones hundieron el NPS hasta –50.0 pts con respecto a la media de los últimos 7 días. Un nuevo descenso el 06-dic (–4.4 pts) por 68 cancelaciones y 58 retrasos reforzó la tendencia negativa, mientras que el repunte del 05-dic (+10.5 pts gracias a feedback positivo de tripulación y confort) quedó neutralizado por las caídas posteriores.

--- GLOBAL ---
A nivel diario, la mejora global de la semana (+3,4 pts con respecto a la semana anterior) se explica fundamentalmente por el repunte del 6-dic (+6,1 pts). En esa jornada destacó el alza de +10,7 pts en Economy SH IB y +7,8 pts en Economy LH, impulsados por feedback muy positivo en puntualidad y calidad de servicio. En contraste, las caídas del 5-dic (–1,3 pts tras 103 retrasos y 38 cancelaciones en Short Haul) y del 9-dic (–0,9 pts por degradación de puntualidad en Economy LH) atenuaron parcialmente este avance, sin que se registraran variaciones extremas no reflejadas en el agregado.

--- ECONOMY LH ---
A nivel diario, la mejora semanal de +5.7 pts en Economy LH se sustenta en los repuntes del 5 de diciembre (+6.8 pts, con feedback favorable sobre puntualidad y comodidad en rutas como MAD–SDQ) y del 6 de diciembre (+7.8 pts, impulsado por un OTP mejorado y comentarios positivos en BOG–MAD). El 9 de diciembre, la caída de –6.9 pts por la degradación de puntualidad (OTP 71.27 %) en la ruta MAD–UIO moderó parcialmente este avance. No se registraron huelgas, cancelaciones masivas ni variaciones extremas (>±20 pts) que alteren la consistencia del desempeño.

===== STEP 2: FULL REPORT =====

📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

Economy SH  
- Estados: (IB: N, YW: N | Economy SH: N)  
- Escenario: Consistencia de normalidad (N,N|N).  
- Narrativa: Ambos operadores mantienen performance estable, sin anomalías que requieran explicación adicional.  

Business SH  
- Estados: (IB: –, YW: – | Business SH: –)  
- Escenario: SINERGIA (–,– | –)  
- Narrativa: La caída conjunta de –10.0 pts en SH Business se explica con la misma lógica raíz del nodo padre. Se observa un deterioro en factores de producto y un repunte operativo negativo que impactan de forma homogénea a IB y YW.  
- Evidencia Clave:  
  • Producto: Aircraft interior (SHAP = –3.7; Sat_diff = –6.4) y Cabin Crew (SHAP = –2.2; Sat_diff = –2.4)  
  • Operativo: Retrasos +64.0 y Desvíos +44.0 (NCS_TOOL)

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul (SH)  
- Dinámica: DILUCIÓN (Economy: N, Business: – | SH: N)  
- Narrativa: El nivel agregado de SH aparece “normal” porque la caída de –10.0 pts en Business se vio parcialmente absorbida por la performance estable de Economy.  
- Evidencia: Business SH sufrió un deterioro en Aircraft interior (SHAP=–3.7; Sat_diff=–6.4) y Cabin Crew (SHAP=–2.2; Sat_diff=–2.4), además de un pico de retrasos (+64.0) y desvíos (+44.0) (NCS_TOOL).  

En Long Haul (LH)  
- Dinámica: CANCELACIÓN (Economy: N, Business: +, Premium: – | LH: N)  
- Narrativa: El NPS global de LH resulta “normal” por compensación interna:  
   • Business LH sube +18.2 pts gracias al fin de la huelga en Italia y la drástica reducción de cancelaciones (NCS_TOOL).  
   • Premium LH cae –25.3 pts por un mal estado del interior de cabina (SHAP=–12.4; Sat_diff=–18.1), deficiencias en Cabin Crew (SHAP=–11.4; Sat_diff=–8.3) y aumento de retrasos (+64.0) y desvíos (+44.0).  
- Evidencia: Contraste directo entre el impulso operativo y de producto en Business vs. el deterioro de producto y operaciones en Premium, que neutralizan el efecto en el agregado.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA (LH: N, SH: N | GLOBAL: +).  
- Narrativa: Aunque ni el largo radio ni el corto exhiben anomalías formales, el NPS Global se ve impulsado por factores sistémicos que no alcanzan a sesgar cada radio aislado, pero sí el agregado de la red.  
- Evidencia:  
  • Operativo: reducción de cancelaciones (–60.0 vs L7d) y leve alza de OTP15 (+0.4 pts) que mejoraron la percepción de fiabilidad (explanatory_drivers_tool: Punctuality SHAP=1.1; Sat_diff=+2.5).  
  • Producto: mejoras en arrivals experience (SHAP=0.7; Sat_diff=+2.3) y check-in (SHAP=0.6; Sat_diff=+1.2) contribuyeron al alza global.

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

CAUSA 1: Impulso positivo global  
- Escenario: TRANSFERENCIA (LH: N, SH: N | GLOBAL: +)  
- NMA: Global  
- Afecta a: toda la red  
- Tipo de impacto: POSITIVO  

CAUSA 2: Mejora operativa en Long Haul Business (fin de huelga, –60.0 cancelaciones)  
- Escenario: CANCELACIÓN (Economy: N, Business: +, Premium: – | LH: N)  
- NMA: Global/LH/Business  
- Afecta a: segmento LH Business  
- Tipo de impacto: POSITIVO  

CAUSA 3: Deterioro de producto y operaciones en Long Haul Premium (interior cabina, tripulación, +64.0 retrasos, +44.0 desvíos)  
- Escenario: CANCELACIÓN (Economy: N, Business: +, Premium: – | LH: N)  
- NMA: Global/LH/Premium  
- Afecta a: segmento LH Premium  
- Tipo de impacto: NEGATIVO  

CAUSA 4: Caída en Short Haul Business (interior de avión, tripulación, +64.0 retrasos, +44.0 desvíos)  
- Escenario: DILUCIÓN (Economy: N, Business: – | SH: N)  
- NMA: Global/SH/Business  
- Afecta a: segmento SH Business  
- Tipo de impacto: NEGATIVO

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

=== NMA: Global ===

📈 EXPLANATORY DRIVERS:  
- Punctuality: SHAP=1.1, Sat_diff=+2.5  
- Load factor:   SHAP=0.1, Sat_diff=–0.6  
- Arrivals experience:     SHAP=0.7, Sat_diff=+2.3  
- Check-in:                SHAP=0.6, Sat_diff=+1.2  
- Ticket Price:            SHAP=–0.4, Sat_diff=+6.0  
- Boarding:                SHAP=0.3, Sat_diff=+0.6  
- In flight food & bev.:   SHAP=0.2, Sat_diff=+2.1  
- Connections experience:  SHAP=0.2, Sat_diff=+1.4  
- Cabin Crew:              SHAP=0.2, Sat_diff=+1.2  
- Aircraft interior:       SHAP=0.1, Sat_diff=+0.4  
- IB Plus loyalty program: SHAP=0.1, Sat_diff=+0.9  

📊 DATOS OPERATIVOS:  
- OTP15 (subida): +0.4 puntos  
- Mishandling (bajada): –2.5 incidentes  
- Misconnections (bajada): –0.2 incidentes  

🚨 INCIDENTES NCS:  
- Cancelaciones: –60.0  
- Retrasos:        +64.0  
- Desvíos:         +44.0  
- Limitación aeronave: –14.0  
- Otras incidencias:   –21.0  

💬 FEEDBACK DE CLIENTES:  
No se dispone de feedback cualitativo segmentado (Global) que confirme o refute los drivers operativos o de producto.  

✈️ RUTAS AFECTADAS (Top 5):  
- MAD–RVN: NPS 83.3 (6 pax)  
- BRU–MAD: NPS 66.7 (3 pax)  
- MAD–TLS: NPS 0.0  (5 pax)  
- LCG–MAD: NPS100.0 (3 pax)  
- MAD–SCQ: NPS 64.3 (14 pax)  

👥 PERFILES REACTIVOS:  
- Residence Region: spread 98.4 pts  
- Fleet:            spread 91.3 pts  
- CodeShare:        spread 57.3 pts  
- Business/Leisure: spread 5.8 pts  

=== NMA: Global/LH/Business ===

📈 EXPLANATORY DRIVERS:  
- Punctuality con SHAP=15.8, Sat_diff=9.4  

📊 DATOS OPERATIVOS:  
No disponible  

🚨 INCIDENTES NCS:  
- cancelaciones –60.0 (NCS_TOOL)  
- retrasos +64.0 (NCS_TOOL)  
- desvíos +44.0 (NCS_TOOL)  

💬 FEEDBACK DE CLIENTES:  
No disponible  

✈️ RUTAS AFECTADAS (Top 5):  
- JFK-MAD: NPS –14.3, 7 pax (EXPLANATORY_DRIVERS_TOOL)  
- MAD-SDQ: NPS –100.0, 1 pax (EXPLANATORY_DRIVERS_TOOL)  
- MAD-MCO: NPS 50.0, 2 pax (EXPLANATORY_DRIVERS_TOOL)  
- MAD-UIO: NPS 33.3, 3 pax (EXPLANATORY_DRIVERS_TOOL)  
- MAD-NRT: NPS 100.0, 1 pax (EXPLANATORY_DRIVERS_TOOL)  

👥 PERFILES REACTIVOS:  
- CodeShare: spread de NPS 366.7 pts  
- Residence Region: spread de NPS 98.8 pts  
- Fleet: spread de NPS 75.5 pts  

=== NMA: Global/LH/Premium ===

📈 EXPLANATORY DRIVERS:  
- Check-in: SHAP = 14.9 (Sat_diff = +3.7)  
- Aircraft interior: SHAP = –12.4 (Sat_diff = –18.1)  
- Cabin Crew: SHAP = –11.4 (Sat_diff = –8.3)  
- Journey preparation support: SHAP = –5.2 (Sat_diff = –12.1)  
- In flight food and beverage: SHAP = –5.2 (Sat_diff = –4.3)  
- Boarding: SHAP = –3.2 (Sat_diff = –3.6)  
- Ticket Price: SHAP = –1.7 (Sat_diff = +73.2)  
- Ease of contact by phone: SHAP = –1.1 (Sat_diff = –5.5)  
- Load factor: SHAP = 0.0 (Sat_diff = –4.2)  
- Punctuality: SHAP = 0.0 (Sat_diff = –0.8)  

📊 DATOS OPERATIVOS:  
No disponible  

🚨 INCIDENTES NCS:  
- Cancelaciones: –60.0  
- Retrasos: +64.0  
- Desvíos: +44.0  
- Limitación de aeronave: –14.0  
- Otras incidencias: –21.0  

💬 FEEDBACK DE CLIENTES:  
- Volumen de comentarios: 137 vs 227 (–39.6%)  
- Principalmente valoraciones genéricas positivas (confort, trato, eficiencia)  
- No se mencionan retrasos, desvíos ni interior de cabina → no valida drivers SHAP  

✈️ RUTAS AFECTADAS (Top 5):  
- MAD-ORD: NPS –50.0, Pax 4 (explanatory_drivers)  
- JFK-MAD: NPS   0.0, Pax 2 (explanatory_drivers)  
- MAD-NRT: NPS  33.3, Pax 3 (explanatory_drivers)  
- MAD-MCO: NPS 100.0, Pax 2 (explanatory_drivers)  
- MAD-SJO: NPS 100.0, Pax 1 (explanatory_drivers)  

👥 PERFILES REACTIVOS:  
- CodeShare: spread de 236.4 pts  
- Residence Region: spread de 62.5 pts  
- Business/Leisure: spread de 20.3 pts  
- Fleet: spread de 15.2 pts  

=== NMA: Global/SH/Business ===

📈 EXPLANATORY DRIVERS:  
- Aircraft interior: SHAP = –3.7, Sat_diff = –6.4  
- Cabin Crew: SHAP = –2.2, Sat_diff = –2.3  
- Boarding: SHAP = –1.9, Sat_diff = –6.9  
- Journey preparation support: SHAP = –1.7, Sat_diff = –6.4  
- In flight food and beverage: SHAP = –1.5, Sat_diff = –6.9  
- Punctuality driver: SHAP = –1.0, Sat_diff = –3.2  

📊 DATOS OPERATIVOS:  
No disponible  

🚨 INCIDENTES NCS:  
- Retrasos: +64.0 incidentes  
- Desvíos: +44.0 incidentes  

💬 FEEDBACK DE CLIENTES:  
No disponible  

✈️ RUTAS AFECTADAS (Top 5):  
- MAD-VCE: NPS −100.0, Pax 2 (Producto)  
- AGP-MLN: NPS −100.0, Pax 1 (Producto)  
- FLR-MAD: NPS −100.0, Pax 1 (Producto)  
- AGP-MAD: NPS −100.0, Pax 1 (Producto)  
- DSS-MAD: NPS −100.0, Pax 2 (Producto)  

👥 PERFILES REACTIVOS:  
- Fleet: spread de 250.0 pts  
- Residence Region: spread de 182.7 pts  
- CodeShare: spread de 104.2 pts  
- Business/Leisure: spread de 11.0 pts

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 **SÍNTESIS EJECUTIVA:**  
Durante la semana del 5 al 9 de diciembre de 2025 hemos identificado cuatro vectores clave que explican las variaciones de NPS. El indicador global registró una subida de 3.2 puntos con respecto a la semana anterior.

A nivel diario, la mejora global de la semana (+3.4 pts con respecto a la semana anterior) se explica fundamentalmente por el repunte del 6-dic (+6.1 pts). En esa jornada destacó el alza de +10.7 pts en Economy SH IB y +7.8 pts en Economy LH, impulsados por feedback muy positivo en puntualidad y calidad de servicio. En contraste, las caídas del 5-dic (–1.3 pts tras 103 retrasos y 38 cancelaciones en Short Haul) y del 9-dic (–0.9 pts por degradación de puntualidad en Economy LH) atenuaron parcialmente este avance, sin que se registraran variaciones extremas no reflejadas en el agregado.

El alza global se sustentó en la mejora de la fiabilidad operativa y en puntos fuertes de producto. La puntualidad mejoró 1.1 ppts según Explanatory Drivers, apoyada en un OTP que subió 0.4 pts y en la reducción de incidentes operativos: –60.0 cancelaciones, +64.0 retrasos y +44.0 desvíos, junto a –2.5 equipajes perdidos y –0.2 misconexiones. En producto, arrivals experience ganó 0.7 ppts, check-in 0.6 ppts, boarding 0.3 ppts, catering 0.2 ppts, cabin crew 0.2 ppts y aircraft interior 0.1 ppts, mientras que la percepción del precio mejoró 6.0 ppts. Estas mejoras brillaron en rutas como LCG–MAD (NPS 100.0, 3 pax), MAD–RVN (83.3, 6 pax) y MAD–SCQ (64.3, 14 pax), y se sintieron con especial intensidad en pasajeros de determinadas flotas (spread 91.3 pts) y regiones de residencia (spread 98.4 pts).

En LH se observó un fuerte contraste entre Business y Premium. El segmento Business subió 18.2 pts con respecto a la semana anterior hasta un NPS de 33.3, impulsado por un fin de huelga en Italia que redujo 60.0 cancelaciones y otorgó +15.8 ppts de puntualidad según Explanatory Drivers, a pesar de un repunte de +64.0 retrasos y +44.0 desvíos. Las rutas más impactadas incluyeron MAD–NRT (100.0, 1 pax) y MAD–MCO (50.0, 2 pax), mientras que JFK–MAD (–14.3, 7 pax) y MAD–SDQ (–100.0, 1 pax) amortiguaron parcialmente la mejora. Los pasajeros en vuelos code-share mostraron la mayor variabilidad (spread 366.7 pts), seguidos por residencia (98.8 pts) y tipo de flota (75.5 pts).  

En Premium LH el NPS cayó 25.3 pts hasta 2.6, reflejando un deterioro de producto y operación. A pesar de que el check-in mejoró 14.9 ppts según Explanatory Drivers, el aircraft interior perdió 12.4 ppts, cabin crew –11.4 ppts, journey support –5.2 ppts, catering –5.2 ppts y boarding –3.2 ppts. La operativa siguió tensionada con –60.0 cancelaciones, +64.0 retrasos, +44.0 desvíos y –14.0 limitaciones de aeronave. El feedback de clientes cayó un 39.6 % en volumen (137 vs 227), con valoraciones genéricas positivas pero sin mención a los retrasos ni a la cabina. Las rutas con peor desempeño fueron MAD–ORD (–50.0, 4 pax) y JFK–MAD (0.0, 2 pax), mientras que MAD–MCO y MAD–SJO alcanzaron 100.0. La sensibilidad mayor se apreció en pasajeros code-share (spread 236.4 pts) y de regiones específicas (62.5 pts).

---

**ECONOMY SH: Mantenimiento de satisfacción estable**  
La cabina Economy de SH registró un NPS de 36.6 con una subida de 3.4 pts con respecto a la semana anterior. IB obtuvo 35.3 (+3.5 pts) y YW 39.0 (+2.9 pts), con un rendimiento armonizado que refleja la estabilidad de parámetros operativos y de producto.

A nivel diario, Economy SH confirmó la estabilidad reflejada en el informe semanal, sin que ningún día registrara variaciones superiores a ±3 pts con respecto a la media de los últimos 7 días. Por ejemplo, el 9-dic IB cerró en 34.85 (vs 34.80, +0.1 pts) y YW en 40.87 (vs 38.75, +2.1 pts), ambos sin incidencias operativas relevantes. Los ligeros repuntes de YW a mediados de semana se diluyeron en el consolidado, reforzando el estado “Normal” de la cabina.

---

**BUSINESS SH: Deterioro por producto y clima**  
El segmento Business de SH cerró en 29.3, con una bajada de 10.0 pts con respecto a la semana anterior. IB marcó 37.4 (–5.5 pts) y YW 9.8 (–21.1 pts). El interior de cabina perdió 3.7 ppts según Explanatory Drivers, cabin crew cayó 2.2 ppts, boarding 1.9 ppts, journey support 1.7 ppts, catering 1.5 ppts y puntualidad 1.0 ppts. Además, los incidentes operativos se incrementaron con +64.0 retrasos y +44.0 desvíos, especialmente en rutas como MAD–VCE, AGP–MLN, FLR–MAD, AGP–MAD y DSS–MAD (todas con NPS –100.0). Los clientes más sensibles fueron los que viajaron en flotas específicas (spread 250.0 pts) y residentes en regiones con mayor variabilidad (182.7 pts).

A nivel diario, la caída semanal de Business SH se explica sobre todo por el 08-dic, cuando 326 incidentes operativos (84 retrasos, 30 cancelaciones y 20 pérdidas de conexión) hundieron el NPS 16.7 pts (IB –21.9, YW –5.4) y la ruta MAD–ZRH tocó –40.0 pts. El 06-dic, un nuevo pico de 103 retrasos, 38 cancelaciones y 93 misconexiones reforzó ese deterioro. Los repuntes del 09-dic (+15.4 pts por puntualidad y catering en IB) y del 07-dic (alza de IB neutralizada por caída de YW) se diluyeron en el agregado semanal.

---

**ECONOMY LH: Desempeño estable**  
La cabina Economy de LH mostró un NPS de 14.5 con una subida de 5.7 pts con respecto a la semana anterior, manteniendo desempeño estable sin cambios operativos ni de producto que alteren los niveles de satisfacción.

A nivel diario, la mejora semanal de +5.7 pts en Economy LH se sustenta en los repuntes del 5 de diciembre (+6.8 pts, con feedback favorable sobre puntualidad y comodidad en rutas como MAD–SDQ) y del 6 de diciembre (+7.8 pts, impulsado por un OTP mejorado y comentarios positivos en BOG–MAD). El 9 de diciembre, la caída de –6.9 pts por la degradación de puntualidad (OTP 71.27 %) en la ruta MAD–UIO moderó parcialmente este avance. No se registraron huelgas, cancelaciones masivas ni variaciones extremas (>±20 pts) que alteren la consistencia del desempeño.

---

**BUSINESS LH: Recuperación tras fin de huelga**  
En LH Business el NPS llegó a 33.3, con una subida de 18.2 pts con respecto a la semana anterior. La conclusión del conflicto en Italia redujo 60.0 cancelaciones y aportó +15.8 ppts de puntualidad según Explanatory Drivers, pese al aumento de +64.0 retrasos y +44.0 desvíos. Las rutas MAD–NRT (100.0, 1 pax) y MAD–UIO (33.3, 3 pax) reflejaron la mejora, mientras que JFK–MAD (–14.3, 7 pax) atenuó el avance. El alza fue más marcada en code-share (spread 366.7 pts), regiones de residencia (98.8 pts) y tipo de flota (75.5 pts).

A nivel diario, el alza de +18.2 puntos de Business LH se fraguó principalmente el 06-dic, cuando el NPS de este segmento escaló +21.4 pts con respecto a la media de los últimos 7 días gracias al feedback de calidad de servicio y confort en la ruta GRU–MAD (NPS 7.7, 13 encuestas), coincidiendo además con un descenso a 30 cancelaciones, el mínimo de la semana. La finalización de la huelga en Italia a inicios de semana se materializó ese día, impulsando la drástica reducción de cancelaciones. Por su parte, el repunte inicial del 05-dic (+11.1 pts) se vio parcialmente diluido por la persistencia de demoras.

---

**PREMIUM LH: Deterioro de producto y gestión climática**  
El segmento Premium de LH cayó a un NPS de 2.6, con una bajada de 25.3 pts con respecto a la semana anterior. Aunque el check-in mejoró 14.9 ppts según Explanatory Drivers, el aircraft interior perdió 12.4 ppts, cabin crew –11.4 ppts, journey support –5.2 ppts, catering –5.2 ppts y boarding –3.2 ppts. La operativa mostró –60.0 cancelaciones, +64.0 retrasos, +44.0 desvíos y –14.0 limitaciones de aeronave. Las rutas MAD–ORD (–50.0, 4 pax) y JFK–MAD (0.0, 2 pax) registraron los peores resultados; en contraste, MAD–MCO y MAD–SJO alcanzaron 100.0. Los perfiles más sensibles fueron code-share (spread 236.4 pts) y determinadas regiones de residencia (62.5 pts).

A nivel diario, la caída semanal de –25.3 pts en Premium LH se concentró principalmente el 09-dic, cuando un OTP de 71.27 (–6.99 pts) junto a 57 retrasos y 36 cancelaciones hundieron el NPS hasta –50.0 pts con respecto a la media de los últimos 7 días. Un nuevo descenso el 06-dic (–4.4 pts) por 68 cancelaciones y 58 retrasos reforzó la tendencia negativa, mientras que el repunte del 05-dic (+10.5 pts gracias a feedback positivo de tripulación y confort) quedó neutralizado por las caídas posteriores.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0  
- **Pasos de análisis:** 6  
- **Metodología:** Análisis conversacional paso a paso  
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*

===== STEP 3: EXECUTIVE SYNTHESIS =====

📈 <b>SÍNTESIS EJECUTIVA:</b><br>
Durante la semana del 5 al 9 de diciembre de 2025 hemos identificado cuatro vectores clave que explican las variaciones de NPS. El indicador global registró una subida de 3.2 puntos con respecto a la semana anterior.<br><br>

<b>ECONOMY SH: Mantenimiento de satisfacción estable</b><br>
La cabina Economy de SH registró un NPS de 36.6 con una subida de 3.4 pts con respecto a la semana anterior. IB obtuvo 35.3 (+3.5 pts) y YW 39.0 (+2.9 pts), con un rendimiento armonizado que refleja la estabilidad de parámetros operativos y de producto.<br><br>
A nivel diario, Economy SH confirmó la estabilidad reflejada en el informe semanal, sin que ningún día registrara variaciones superiores a ±3 pts con respecto a la media de los últimos 7 días. Por ejemplo, el 9-dic IB cerró en 34.85 (vs 34.80, +0.1 pts) y YW en 40.87 (vs 38.75, +2.1 pts), ambos sin incidencias operativas relevantes. Los ligeros repuntes de YW a mediados de semana se diluyeron en el consolidado, reforzando el estado “Normal” de la cabina.<br><br>

<b>BUSINESS SH: Deterioro por producto y clima</b><br>
El segmento Business de SH cerró en 29.3, con una bajada de 10.0 pts con respecto a la semana anterior. IB marcó 37.4 (–5.5 pts) y YW 9.8 (–21.1 pts). El interior de cabina perdió 3.7 ppts según Explanatory Drivers, cabin crew cayó 2.2 ppts, boarding 1.9 ppts, journey support 1.7 ppts, catering 1.5 ppts y puntualidad 1.0 ppts. Además, los incidentes operativos se incrementaron con +64.0 retrasos y +44.0 desvíos, especialmente en rutas como MAD–VCE, AGP–MLN, FLR–MAD, AGP–MAD y DSS–MAD (todas con NPS –100.0). Los clientes más sensibles fueron los que viajaron en flotas específicas (spread 250.0 pts) y residentes en regiones con mayor variabilidad (182.7 pts).<br><br>
A nivel diario, la caída semanal de Business SH se explica sobre todo por el 08-dic, cuando 326 incidentes operativos (84 retrasos, 30 cancelaciones y 20 pérdidas de conexión) hundieron el NPS 16.7 pts (IB –21.9, YW –5.4) y la ruta MAD–ZRH tocó –40.0 pts. El 06-dic, un nuevo pico de 103 retrasos, 38 cancelaciones y 93 misconexiones reforzó ese deterioro. Los repuntes del 09-dic (+15.4 pts por puntualidad y catering en IB) y del 07-dic (alza de IB neutralizada por caída de YW) se diluyeron en el agregado semanal.<br><br>

<b>ECONOMY LH: Desempeño estable</b><br>
La cabina Economy de LH mostró un NPS de 14.5 con una subida de 5.7 pts con respecto a la semana anterior, manteniendo desempeño estable sin cambios operativos ni de producto que alteren los niveles de satisfacción.<br><br>
A nivel diario, la mejora semanal de +5.7 pts en Economy LH se sustenta en los repuntes del 5 de diciembre (+6.8 pts, con feedback favorable sobre puntualidad y comodidad en rutas como MAD–SDQ) y del 6 de diciembre (+7.8 pts, impulsado por un OTP mejorado y comentarios positivos en BOG–MAD). El 9 de diciembre, la caída de –6.9 pts por la degradación de puntualidad (OTP 71.27 %) en la ruta MAD–UIO moderó parcialmente este avance. No se registraron huelgas, cancelaciones masivas ni variaciones extremas (>±20 pts) que alteren la consistencia del desempeño.<br><br>

<b>BUSINESS LH: Recuperación tras fin de huelga</b><br>
En LH Business el NPS llegó a 33.3, con una subida de 18.2 pts con respecto a la semana anterior. La conclusión del conflicto en Italia redujo 60.0 cancelaciones y aportó +15.8 ppts de puntualidad según Explanatory Drivers, pese al aumento de +64.0 retrasos y +44.0 desvíos. Las rutas MAD–NRT (100.0, 1 pax) y MAD–UIO (33.3, 3 pax) reflejaron la mejora, mientras que JFK–MAD (–14.3, 7 pax) atenuó el avance. El alza fue más marcada en code-share (spread 366.7 pts), regiones de residencia (98.8 pts) y tipo de flota (75.5 pts).<br><br>
A nivel diario, el alza de +18.2 puntos de Business LH se fraguó principalmente el 06-dic, cuando el NPS de este segmento escaló +21.4 pts con respecto a la media de los últimos 7 días gracias al feedback de calidad de servicio y confort en la ruta GRU–MAD (NPS 7.7, 13 encuestas), coincidiendo además con un descenso a 30 cancelaciones, el mínimo de la semana. La finalización de la huelga en Italia a inicios de semana se materializó ese día, impulsando la drástica reducción de cancelaciones. Por su parte, el repunte inicial del 05-dic (+11.1 pts) se vio parcialmente diluido por la persistencia de demoras.<br><br>

<b>PREMIUM LH: Deterioro de producto y gestión climática</b><br>
El segmento Premium de LH cayó a un NPS de 2.6, con una bajada de 25.3 pts con respecto a la semana anterior. Aunque el check-in mejoró 14.9 ppts según Explanatory Drivers, el aircraft interior perdió 12.4 ppts, cabin crew –11.4 ppts, journey support –5.2 ppts, catering –5.2 ppts y boarding –3.2 ppts. La operativa mostró –60.0 cancelaciones, +64.0 retrasos, +44.0 desvíos y –14.0 limitaciones de aeronave. Las rutas MAD–ORD (–50.0, 4 pax) y JFK–MAD (0.0, 2 pax) registraron los peores resultados; en contraste, MAD–MCO y MAD–SJO alcanzaron 100.0. Los perfiles más sensibles fueron code-share (spread 236.4 pts) y determinadas regiones de residencia (62.5 pts).<br><br>
A nivel diario, la caída semanal de –25.3 pts en Premium LH se concentró principalmente el 09-dic, cuando un OTP de 71.27 (–6.99 pts) junto a 57 retrasos y 36 cancelaciones hundieron el NPS hasta –50.0 pts con respecto a la media de los últimos 7 días. Un nuevo descenso el 06-dic (–4.4 pts) por 68 cancelaciones y 58 retrasos reforzó la tendencia negativa, mientras que el repunte del 05-dic (+10.5 pts gracias a feedback positivo de tripulación y confort) quedó neutralizado por las caídas posteriores.