===== STEP 1: IDENTIFY RELEVANT DAYS =====

INPUT:
Analiza los siguientes 7 días e identifica cuáles son relevantes para incluir en el informe semanal:

📅 2025-11-30 to 2025-11-30:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Economy SH, el escenario es CANCELACIÓN (IB: –6.6 pts, YW: +14.2 pts | Padre: Normal).  
- Narrativa: Mientras Global/SH/Economy/IB sufrió una caída atribuible al incremento de incidentes operativos en NCS (12 cancelaciones, 7 demoras y 5 mishandlings de equipaje), Global/SH/Economy/YW experimentó una subida sin causas operativas claras, con verbatims positivos centrados en amabilidad y puntualidad. Estos efectos opuestos se neutralizaron en el agregado, dejando al nodo Economy SH dentro del rango normal.  
- Evidencia Clave:  
  • IB (ncs_tool): 12 cancelaciones, 7 retrasos, 5 incidencias de equipaje (Global/SH/Economy/IB).  
  • YW (verbatims_tool): 198 comentarios destacando amabilidad y eficiencia sin mención de incidencias (Global/SH/Economy/YW).  

En Business SH, el escenario es DILUCIÓN (IB: Normal, YW: –9.2 pts | Padre: Normal).  
- Narrativa: La anomalía negativa de Global/SH/Business/YW, impulsada por los mismos incidentes operativos (cancelaciones, retrasos e incidencias de equipaje), actúa como causa principal de presión en el nodo padre, aunque su impacto fue parcialmente suavizado por el desempeño estable de Global/SH/Business/IB.  
- Evidencia Clave:  
  • YW (ncs_tool): 12 cancelaciones, 7 retrasos y 5 mishandlings de equipaje vinculados a una caída de 9.2 pts (Global/SH/Business/YW).  
  • IB (customer_profile_tool & operative_data_tool): NPS estable en 39.4 pts sin desviaciones operativas significativas (Global/SH/Business/IB).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, ninguna cabina muestra anomalía (Economy SH: Normal, Business SH: Normal | SH: Normal).  
- Narrativa: Ambas cabinas mantuvieron su desempeño dentro del rango esperado, por lo que el radio SH no presenta variaciones atípicas.  
- Evidencia:  
  • Economy SH +0.5 pts vs baseline (34.52 vs 34.02)  
  • Business SH +1.9 pts vs baseline (37.21 vs 35.35)  

En Long Haul, la dinámica es SINERGIA (Economy LH: –26.4 pts, Business LH: –3.2 pts, Premium LH: –11.9 pts | LH: –23.6 pts).  
- Narrativa: La caída de NPS en Long Haul obedece a un problema sistémico de operaciones, donde un elevado número de incidentes impactó de forma transversal a todas las cabinas.  
- Evidencia:  
  • NCS_tool: 32 incidentes operativos (12 cancelaciones, 6 retrasos, 3 mishandlings de equipaje, 2 cambios de aeronave)  
  • Caso crítico SDQ–MAD: 159 maletas retenidas y 10 pasajeros desembarcados.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA (LH: –23.6 pts, SH: +0.5 pts | Global: –6.4 pts).  
- Narrativa: La caída en NPS Global se contagia desde el Largo Radio, donde un elevado número de incidentes operativos impactó transversalmente la percepción del pasajero. El desempeño estable del Corto Radio no fue suficiente para contrarrestar este efecto.  
- Evidencia:  
  • NCS_tool (Global): 358 incidentes totales (68 cancelaciones, 63 retrasos, 36 otras incidencias, 11 mishandlings de equipaje), incl. caso SDQ–MAD con 159 maletas retenidas y 10 pax desembarcados.

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

CAUSA 1: Incidentes operativos en Economy SH – IB  
- Escenario: CANCELACIÓN  
- NMA: Global/SH/Economy/IB  
- Afecta a: Global/SH/Economy/IB  
- Tipo de impacto: Negativo  

CAUSA 2: Factores de satisfacción en Economy SH – YW  
- Escenario: CANCELACIÓN  
- NMA: Global/SH/Economy/YW  
- Afecta a: Global/SH/Economy/YW  
- Tipo de impacto: Positivo  

CAUSA 3: Incidentes operativos en Short Haul – Business YW  
- Escenario: DILUCIÓN  
- NMA: Global/SH/Business/YW  
- Afecta a: Global/SH/Business/YW  
- Tipo de impacto: Negativo  

CAUSA 4: Incidentes operativos en Long Haul  
- Escenario: SINERGIA  
- NMA: Global/LH  
- Afecta a: Global/LH/Economy, Global/LH/Business, Global/LH/Premium  
- Tipo de impacto: Negativo

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

=== NMA: Global/SH/Economy/IB ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
No disponible

🚨 INCIDENTES NCS:  
• 12 cancelaciones  
• 7 demoras  
• 5 problemas de equipaje  
• Caso puntual SDQ-MAD: limitación de peso que dejó 159 maletas y 10 pasajeros sin viajar

💬 FEEDBACK DE CLIENTES:  
• 430 comentarios mayoritariamente positivos: puntualidad, amabilidad del personal, correcto manejo de equipaje

✈️ RUTAS AFECTADAS (Top 5):  
• MAD-VIE: NPS 0.0 con 4 encuestas

👥 PERFILES REACTIVOS:  
No disponible  


=== NMA: Global/SH/Economy/YW ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
OTP15_adjusted +3.79 pts vs baseline (operative_data_tool)

🚨 INCIDENTES NCS:  
34 incidencias (12 cancelaciones, 7 retrasos, mishandling, 1 incidente grave SDQ-MAD)

💬 FEEDBACK DE CLIENTES:  
198 comentarios centrados en amabilidad, eficiencia y puntualidad; no mencionan incidencias operativas

✈️ RUTAS AFECTADAS (Top 5):  
CMN-MAD: NPS 25.0 (5 encuestas)

👥 PERFILES REACTIVOS:  
• Leisure: NPS 53.3 (138 encuestas)  
• Business: NPS 6.7 (15 encuestas)  
• Fleet ATR: NPS 50.0 (26 encuestas)  
• Fleet CRJ: NPS 48.4 (127 encuestas)  
• Residence Region Europa: 28.9 (39 encuestas)  
• Residence Region América Centro: 25.0 (4 encuestas)  
• Residence Region América Sur: –100.0 (1 encuesta)  


=== NMA: Global/SH/Business/YW ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
OTP15: 67.0 (vs 63.21 baseline) → +3.79  
Load Factor: 69.47 (vs 72.83 baseline) → –3.36

🚨 INCIDENTES NCS:  
• 12 cancelaciones  
• 7 retrasos  
• 5 incidencias de equipaje  
• 10 otras incidencias  
• 2 vuelos afectados  
• Limitación de peso que dejó en tierra 159 maletas y 10 pasajeros

💬 FEEDBACK DE CLIENTES:  
11 comentarios positivos sobre la amabilidad y servicio de la tripulación; no hay mención de cancelaciones, retrasos o equipaje

✈️ RUTAS AFECTADAS (Top 5):  
MAD-OPO: 1 encuesta, NPS no disponible

👥 PERFILES REACTIVOS:  
• Business/Work: NPS 66.7 (4 encuestas)  
• Leisure: NPS 14.3 (7 encuestas)  
• Flota CRJ: NPS 30.0 (11 encuestas)  
• Región Europa: NPS –33.3 (número de encuestas no disponible)  


=== NMA: Global/LH ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
No disponible

🚨 INCIDENTES NCS:  
• 12 cancelaciones  
• 6 retrasos  
• 3 incidencias de equipaje  
• 2 cambios de aeronave por limitaciones técnicas  
• Incidente crítico en SDQ-MAD: 159 maletas retenidas y 10 pasajeros desembarcados  
• Se documentaron 32 eventos operativos en el día

💬 FEEDBACK DE CLIENTES:  
624 comentarios: enfoque globalmente positivo en servicio a bordo y puntualidad; ninguna mención a cancelaciones, equipaje retenido o cambios de aeronave

✈️ RUTAS AFECTADAS (Top 5):  
EZE-MAD: NPS -15.1 (n=38)

👥 PERFILES REACTIVOS:  
• Business/Work: NPS -0.1 (58 encuestas)  
• Leisure: NPS -11.8 (261 encuestas)  
• Flota más penalizada:  
  – A321XLR: NPS -63.6  
  – A333: NPS -26.5  
  – A332: NPS -22.4  
• Codeshare con peor NPS: AY (-48.0), AA (-55.7), BA (-74.3), QR (-100.0)  
• Regiones de residencia más afectadas: ASIA (NPS -80.0), EUROPA (NPS -48.2)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 2025-11-30, hemos identificado cuatro causas principales que explican las variaciones de NPS. El resultado global fue una caída de 6.4 puntos con respecto a la media de los últimos 7 días.

El factor dominante provino del LH, donde un conjunto de 32 incidentes operativos —12 cancelaciones, 6 retrasos, 3 mishandlings de equipaje y 2 cambios de aeronave por limitaciones técnicas, más un caso crítico en SDQ–MAD con 159 maletas retenidas y 10 pasajeros desembarcados— arrastró el NPS de –9.76, cayendo 23.6 puntos con respecto a la media de los últimos 7 días. A pesar de un feedback de 624 comentarios positivos sobre servicio a bordo y puntualidad, no se mencionaron los problemas operativos. La ruta EZE–MAD mostró NPS –15.1 (38 encuestas) y los perfiles más afectados fueron Leisure (NPS –11.8, 261 encuestas), Business/Work (NPS –0.1, 58), flota A321XLR (NPS –63.6), A333 (–26.5) y A332 (–22.4), así como residentes en ASIA (NPS –80.0) y EUROPA (NPS –48.2).

En SH, la dinámica opuesta de los dos operadores y cabinas resultó en un leve alza agregada de 0.5 puntos con respecto a la media de los últimos 7 días, ocultando volatilidad interna en Economy y Business.

**ECONOMY SH: Compensación de impactos opuestos**  
La cabina Economy de SH registró un NPS de 34.52, con un alza de 0.5 puntos con respecto a la media de los últimos 7 días. Desglose por compañía: IB obtuvo 27.27 (–6.6 puntos) y YW 48.68 (+14.2 puntos). El descenso de IB se asocia a 12 cancelaciones, 7 demoras y 5 problemas de equipaje (Global/SH/Economy/IB), pese a 430 comentarios positivos sobre puntualidad, amabilidad y correcto manejo de equipaje. En contraste, YW contó con OTP +3.79 puntos con respecto a la media de los últimos 7 días y 34 incidencias (12 cancelaciones, 7 retrasos, mishandling y un incidente grave en SDQ–MAD), sin reflejo en 198 comentarios de amabilidad, eficiencia y puntualidad. CMN–MAD marcó NPS 25.0 (5 encuestas) y sobresalieron los perfiles Leisure (NPS 53.3, 138), Business (NPS 6.7, 15), flota ATR (NPS 50.0, 26), CRJ (NPS 48.4, 127) y residentes en Europa (NPS 28.9, 39), América Centro (25.0, 4) y América Sur (–100.0, 1).

**BUSINESS SH: Presión por incidentes en operador YW**  
El segmento Business de SH alcanzó un NPS de 37.21, con un alza de 1.9 puntos con respecto a la media de los últimos 7 días. Desglose por compañía: IB obtuvo 39.39 (+4.6 puntos) manteniendo desempeño estable sin desviaciones operativas relevantes; YW se situó en 30.0 (–9.2 puntos), afectado por 12 cancelaciones, 7 retrasos, 5 mishandlings de equipaje, 10 otras incidencias y limitación de peso en SDQ–MAD que retuvo 159 maletas y 10 pasajeros. En YW, las métricas operativas incluyeron OTP +3.79 puntos y Load Factor –3.36 puntos con respecto a la media de los últimos 7 días. El feedback de 11 comentarios destacó amabilidad y servicio de tripulación sin mención de incidencias. Los perfiles más sensibles fueron Leisure (NPS 14.3, 7 encuestas), residentes en Europa (NPS –33.3), Business/Work (NPS 66.7, 4) y flota CRJ (NPS 30.0, 11). La ruta MAD–OPO registró una encuesta, sin NPS disponible.

**ECONOMY LH: Deterioro por incidentes operativos**  
La cabina Economy de LH registró un NPS de –14.17, con una caída de 26.4 puntos con respecto a la media de los últimos 7 días. Los 32 eventos operativos (12 cancelaciones, 6 retrasos, 3 mishandlings de equipaje y 2 cambios de aeronave, incluido el caso SDQ–MAD con 159 maletas y 10 pasajeros) explican el deterioro, pese a que Load Factor registró –2.94 puntos y OTP –0.51 puntos con respecto a la media de los últimos 7 días. En los 508 comentarios, predominó la valoración positiva de servicio y puntualidad sin alusión a incidentes. EZE–MAD mostró NPS –21.2 (34 encuestas). Los perfiles más afectados incluyeron Leisure (NPS –13.2, 227), Business/Work (–20.6, 35), flota A321XLR (–63.6, 11) y regiones de Asia (–75.0) y Europa (–50.0). En CodeShare, QR (–100.0), BA (–71.4) y AA (–53.3) evidenciaron alta volatilidad.

**BUSINESS LH: Leve impacto secundario**  
La cabina Business de LH mostró un NPS de 21.21, con una caída de 3.2 puntos con respecto a la media de los últimos 7 días. Las métricas operativas registraron OTP –0.51 puntos y Load Factor –0.54 puntos con respecto a la media de los últimos 7 días. Sin embargo, los mismos 12 cancelaciones, 6 retrasos, 3 mishandlings, 2 cambios de aeronave y una limitación de peso en SDQ–MAD ocasionaron insatisfacción, pese a 66 comentarios que valoraron la puntualidad y amabilidad. La ruta MAD–SJO alcanzó NPS 50.0 (2 encuestas). Los perfiles Business/Work (NPS 52.9, 17) se mantuvieron más satisfechos frente a Leisure (NPS –12.5, 16) y flota A332 (–100.0, 3), así como codeshare BA y AY (–100.0).

**PREMIUM LH: Efecto moderado de incidentes**  
El segmento Premium de LH registró un NPS de 4.17, con una caída de 11.91 puntos con respecto a la media de los últimos 7 días. Aunque OTP –0.51 puntos y Load Factor –2.68 puntos se mantuvieron dentro de umbrales no críticos, los 12 cancelaciones, 6 retrasos, 3 mishandlings de equipaje y un caso de 159 maletas retenidas en SDQ–MAD generaron descontento, pese a 50 comentarios positivos sobre puntualidad y confort. GRU–MAD mostró NPS 50.0 (2 encuestas). Los perfiles Leisure (5.6, 18) y Business (0.0, 6), junto con la flota A350 next (–16.7), evidenciaron mayor sensibilidad.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 6
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*

---

📅 2025-11-29 to 2025-11-29:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Economy SH, el escenario es DILUCIÓN `(IB –, YW N | PADRE N)`.  
- Narrativa: La caída leve de –0.2 pts en SH/Economy/IB fue totalmente absorbida por la estabilidad de SH/Economy/YW, manteniendo el agregado en rango normal.  
- Evidencia Clave: en SH/Economy/IB se registraron 7 retrasos, 4 cancelaciones y 1 limitación de equipaje en la ruta LHR-MAD, afectando de forma desproporcionada a pasajeros residentes en Asia (NPS –53.3) y en vuelos code-share AA (NPS –50.0).

En Business SH, el escenario es DOMINANCIA `(IB +, YW – | PADRE +)`.  
- Narrativa: El fuerte alza de +39.3 pts en SH/Business/IB impuso el signo positivo en el nodo padre, a pesar de la caída de –22.6 pts en SH/Business/YW.  
- Evidencia Clave: en SH/Business/IB mejoraron las métricas operativas —OTP15_adjusted +1.51 pts (SH/Business/IB) y Load Factor –1.93 pts (SH/Business/IB)—, y rutas clave como LHR-MAD y MAD-VIE alcanzaron NPS 100.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es DILUCIÓN (Economy N, Business + | SH N).  
- Narrativa: El rendimiento de SH está dictado por la fuerte mejora en SH/Business (+21.1 pts) atribuida a puntualidad, espacio para piernas y excelente atención de tripulación en rutas clave, aunque este efecto fue atenuado por la estabilidad de SH/Economy, que no mostró desviaciones significativas.  
- Evidencia: SH/Business – verbatims con “puntualidad” y “atención de tripulación” y ruta LHR-MAD con NPS 100.

En Long Haul, la dinámica es CANCELACIÓN (Economy N, Business -, Premium + | LH N).  
- Narrativa: El NPS de LH resulta neutro porque el descenso de –10.2 pts en LH/Business, provocado por off-load de equipaje y pérdidas de conexión, fue compensado por la fuerte subida de +55.4 pts en LH/Premium, impulsada por feedback positivo en rutas como BOG-MAD.  
- Evidencia:  
  • LH/Business – off-load de 116 maletas y pérdidas de conexión (MAD/LHR).  
  • LH/Premium – comentarios de alta calidad de servicio y ruta BOG-MAD con NPS 100.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA (LH N, SH N | GLOBAL +).  
- Narrativa: La mejora de +3.9 pts en el NPS Global no aparece en los radios porque proviene de anomalías localizadas en cabinas específicas (LH/Premium y SH/Business) que, pese a no alterar el nivel agregado de cada radio, sí se transmitieron al resultado Global.  
- Evidencia:  
  • LH/Premium: +55.4 pts impulsado por feedback muy positivo en calidad de servicio y ruta BOG-MAD (NPS 100).  
  • SH/Business: +21.1 pts respaldado por verbatims de “puntualidad” y “excelente atención de tripulación” en rutas como LHR-MAD (NPS 100).

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

CAUSA 1: Caída en SH/Economy/IB  
- Escenario: DILUCIÓN `(IB –, YW N | Economy SH N)`  
- NMA: Global/SH/Economy/IB  
- Afecta a: pasajeros SH Economy IB (LHR-MAD, residentes en Asia, code-share AA)  
- Tipo de impacto: NEGATIVE  

CAUSA 2: Subida en SH/Business/IB  
- Escenario: DOMINANCIA `(IB +, YW – | Business SH +)`  
- NMA: Global/SH/Business/IB  
- Afecta a: pasajeros SH Business IB (ruta LHR-MAD y MAD-VIE con verbatims de “puntualidad” y “excelente atención de tripulación”)  
- Tipo de impacto: POSITIVE  

CAUSA 3: Contraste en LH/Business vs LH/Premium  
- Escenario: CANCELACIÓN `(Economy N, Business –, Premium + | LH N)`  
- NMA: Global/LH/Business y Global/LH/Premium (ambos nodos)  
- Afecta a:  
   • Global/LH/Business (–10.2 pts) por off-load de 116 equipajes y pérdidas de conexión  
   • Global/LH/Premium (+55.4 pts) por alta satisfacción de servicio en BOG-MAD  
- Tipo de impacto: MIXED (NEGATIVE en Business, POSITIVE en Premium)

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

=== NMA: Global/SH/Economy/IB ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
Load Factor: –0.14 (no significativa)  
OTP15_adjusted: +1.51 (no significativa)

🚨 INCIDENTES NCS:  
7 retrasos, 4 cancelaciones, 1 limitación de equipaje (116 maletas), cambios de avión y pérdidas de conexión

💬 FEEDBACK DE CLIENTES:  
Comodidad de asientos; Rapidez y eficiencia en embarque; Amabilidad de la tripulación

✈️ RUTAS AFECTADAS (Top 5):  
LHR-MAD: NPS 10.7 (n=28 encuestas)

👥 PERFILES REACTIVOS:  
Residence Region (Asia): NPS –53.3  
CodeShare (vuelos AA): NPS –50.0  

=== NMA: Global/SH/Business/IB ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
OTP15_adjusted aumentó +1.51 puntos  
Load Factor disminuyó –1.93 puntos

🚨 INCIDENTES NCS:  
Se registraron 20 incidentes (retrasos y mishandling)

💬 FEEDBACK DE CLIENTES:  
Puntualidad y cumplimiento de horario; Excelente atención de la tripulación; Comodidad de espacio y calidad del servicio a bordo

✈️ RUTAS AFECTADAS (Top 5):  
LHR–MAD: NPS 100.0 (5 encuestas)  
MAD–VIE: NPS 100.0 (1 encuesta)

👥 PERFILES REACTIVOS:  
Leisure: NPS 76.5 (18 encuestas)  
Business/Work: NPS 70.0 (10 encuestas)  

=== NMA: Global/LH/Business ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
Load Factor: 93.43 vs baseline –0.48 pts  
OTP15_adjusted: 77.8 vs baseline –1.75 pts

🚨 INCIDENTES NCS:  
24 incidentes totales (4 cancelaciones, 5 retrasos, 3 limitaciones de aeronave, 8 otros); offload de 116 equipajes; pérdidas de conexión en MAD y LHR

💬 FEEDBACK DE CLIENTES:  
“Servicio y puntualidad impecables”; “Buen servicio”; “Comodidad y buen servicio”

✈️ RUTAS AFECTADAS (Top 5):  
MAD-SJU: NPS 100.0 (1 encuesta)

👥 PERFILES REACTIVOS:  
Business/Work: NPS 30.8 (13 encuestas)  
Leisure: NPS 0.0 (15 encuestas)  
Flota A332: NPS 75.0 (4 encuestas)  
A350 next: NPS 71.4 (7 encuestas)  
A333: NPS 16.0 (8 encuestas)  

=== NMA: Global/LH/Premium ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
Load Factor –2.97  
OTP15 –1.75

🚨 INCIDENTES NCS:  
4 cancelaciones; 5 retrasos; 3 limitaciones de aeronave; 8 otras incidencias; 116 equipajes no cargados; 109 pérdidas de conexión en MAD; 5 gestiones de reprogramación

💬 FEEDBACK DE CLIENTES:  
Menciones únicamente positivas a calidad de servicio y amabilidad de la tripulación; no se hace referencia a equipaje, conexiones ni retrasos

✈️ RUTAS AFECTADAS (Top 5):  
BOG-MAD: NPS 100.0 (n=2)  
No hay rutas con baja NPS ni coincidencia con los incidentes reportados

👥 PERFILES REACTIVOS:  
Business/Work: NPS 100.0 (n=1)  
Leisure: NPS 66.7 (n=7)  
Flota A350: NPS 100.0 (n=4)  
Flota A350 next: NPS 33.3 (n=4)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 **SÍNTESIS EJECUTIVA:**  
Durante la semana del 29 de noviembre de 2025, hemos identificado tres focos de impacto que explican la subida de 3.9 pts en el NPS global con respecto a la media de los últimos 7 días. La mejora global proviene de picos de satisfacción en Premium LH y en Business SH, mientras que pequeñas caídas en Economy SH e impactos contrapuestos en Business LH quedaron neutralizados.

La mayor contribución vino de la cabina Premium en LH, donde el NPS escaló a 71.4286 (+55.4 pts con respecto a la media de los últimos 7 días) aun registrando 4 cancelaciones, 5 retrasos, 3 limitaciones de aeronave, 8 otras incidencias, 116 equipajes no cargados y 109 pérdidas de conexión en MAD. El feedback de clientes subrayó únicamente la calidad de servicio y la amabilidad de la tripulación, y la ruta BOG–MAD alcanzó un NPS de 100.0 (2 encuestas), con pasajeros de Business/Work (NPS 100.0, 1 encuesta) y flota A350 (NPS 100.0, 4 encuestas) liderando la reacción positiva. De manera similar, en la cabina Business de SH el NPS se situó en 56.4103 (+21.1 pts con respecto a la media de los últimos 7 días), dominado por IB con un NPS de 74.0741 (+39.3 pts), respaldado por métricas operativas (OTP15_adjusted +1.51 pts y Load Factor –1.93 pts) y verbatims que destacaron puntualidad y excelente atención de tripulación en LHR–MAD (NPS 100.0, 5 encuestas) y MAD–VIE (NPS 100.0, 1 encuesta).

En contraste, la cabina Economy de SH absorbió un ligero retroceso de IB a un NPS de 33.6449 (–0.2 pts con respecto a la media de los últimos 7 días) debido a 7 retrasos, 4 cancelaciones y 1 limitación de equipaje (116 maletas) en LHR–MAD (NPS 10.7, 28 encuestas), afectando especialmente a residentes en Asia (NPS –53.3) y pasajeros en code-share AA (NPS –50.0), pero fue diluido por YW con NPS 39.5161 (+5.0 pts). Asimismo, la cabina Business de LH cayó a un NPS de 14.2857 (–10.2 pts con respecto a la media de los últimos 7 días), pese a un Load Factor de 93.43 vs baseline –0.48 pts, OTP15_adjusted de 77.8 vs baseline –1.75 pts y 24 incidentes operativos (4 cancelaciones, 5 retrasos, 3 limitaciones de aeronave, 8 otros; offload de 116 equipajes; pérdidas de conexión en MAD y LHR), aunque su feedback siguió resaltando “servicio y puntualidad impecables” y la ruta MAD–SJU registró un NPS de 100.0 (1 encuesta). Estos impactos negativos no opacaron los picos en Premium LH y Business SH que impulsaron la variación global.

---

**ECONOMY SH: Dilución de un retroceso localizado**  
La cabina Economy de SH registró un NPS de 35.2809 con 1.3 pts de variación con respecto a la media de los últimos 7 días. Desglose por compañía: IB obtuvo un NPS de 33.6449 (–0.2 pts) y YW un NPS de 39.5161 (+5.0 pts). La caída de IB se diluyó por el desempeño sobresaliente de YW. En IB se reportaron 7 retrasos, 4 cancelaciones y 1 limitación de equipaje (116 maletas) en LHR–MAD (NPS 10.7, 28 encuestas), con impactos concentrados en pasajeros residentes en Asia (NPS –53.3) y en vuelos code-share AA (NPS –50.0).

**BUSINESS SH: Dominancia de IB impulsando la mejora**  
El segmento Business de SH registró un NPS de 56.4103 con 21.1 pts de variación con respecto a la media de los últimos 7 días. Desglose por compañía: IB obtuvo un NPS de 74.0741 (+39.3 pts) y YW un NPS de 16.6667 (–22.6 pts). El alza fue liderado por IB, con OTP15_adjusted +1.51 pts y Load Factor –1.93 pts según métricas operativas, complementado por verbatims que resaltan puntualidad y excelente atención de la tripulación. Las rutas clave LHR–MAD (NPS 100.0, 5 encuestas) y MAD–VIE (NPS 100.0, 1 encuesta) aglutinaron el mayor impacto, con fuerte reacción de pasajeros Leisure (NPS 76.5, 18 encuestas) y Business/Work (NPS 70.0, 10 encuestas).

**ECONOMY LH: Desempeño estable**  
La cabina Economy de LH mantuvo desempeño estable, registrando un NPS de 14.3564 con 2.2 pts de variación con respecto a la media de los últimos 7 días. No se detectaron cambios significativos en métricas operativas ni en el feedback de clientes que alteraran su nivel de satisfacción.

**BUSINESS LH: Deterioro compensado**  
La cabina Business de LH registró un NPS de 14.2857 con –10.2 pts de variación con respecto a la media de los últimos 7 días. En esta cabina, el Load Factor fue de 93.43 vs baseline –0.48 pts y el OTP15_adjusted de 77.8 vs baseline –1.75 pts, junto a 24 incidentes operativos (4 cancelaciones, 5 retrasos, 3 limitaciones de aeronave, 8 otros; offload de 116 equipajes; pérdidas de conexión en MAD y LHR). Aun así, el feedback de clientes mencionó “servicio y puntualidad impecables” y la ruta MAD–SJU alcanzó un NPS de 100.0 (1 encuesta).

**PREMIUM LH: Mejora excepcional**  
El segmento Premium de LH registró un NPS de 71.4286 con +55.4 pts de variación con respecto a la media de los últimos 7 días. A pesar de 4 cancelaciones; 5 retrasos; 3 limitaciones de aeronave; 8 otras incidencias; 116 equipajes no cargados; 109 pérdidas de conexión en MAD; 5 gestiones de reprogramación, el feedback de clientes destacó exclusivamente calidad de servicio y amabilidad de la tripulación. La ruta BOG–MAD alcanzó un NPS de 100.0 (2 encuestas), con pasajeros de Business/Work (100.0, 1 encuesta), Leisure (66.7, 7 encuestas) y flotas A350 (100.0, 4 encuestas) y A350 next (33.3, 4 encuestas) mostrando la mayor reacción positiva.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 6
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*

---

📅 2025-11-28 to 2025-11-28:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Economy SH, el escenario es CANCELACIÓN `(IB −, YW + | N)`  
- Narrativa: Mientras IB registró una caída por cancelaciones y retrasos derivados de la huelga en Italia en la ruta MAD–MXP (IB: NPS −0.7 pts explicada por NCS – huelga Italia), YW experimentó un alza impulsada por la alta satisfacción del segmento Leisure y la flota ATR (YW: NPS +14.5 pts, Leisure 56.4 y ATR 73.9), neutralizándose en el NPS agregado de Economy SH.  
- Evidencia Clave:  
  • IB – Causas: cancelaciones y retrasos por huelga en Italia (ruta MAD–MXP)  
  • YW – Drivers: segmento Leisure (NPS 56.4) y flota ATR (NPS 73.9)

En Business SH, el escenario es DOMINANCIA `(IB +, YW − | −)`  
- Narrativa: La fuerte caída de NPS en Business SH se explica principalmente por las cancelaciones y retrasos de la huelga general en Italia que impactaron la ruta BLQ–MAD (YW: NPS −10.0, −49.2 pts), dominando el resultado a pesar del repunte aislado de IB (+15.2 pts en NPS, apoyado en MAD–ORY y valoración de flota A321).  
- Evidencia Clave:  
  • YW – Causa dominante: cancelaciones y retrasos por huelga en Italia en BLQ–MAD  
  • IB – Efecto opuesto atenuado: anomalía positiva en MAD–ORY (NPS 66.7)

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En SH, la dinámica es DILUCIÓN (Economy SH N, Business SH – | SH N)  
- Narrativa: El SH aparece estable porque, aunque la cabina Business SH sufrió una caída por cancelaciones y retrasos ocasionados por la huelga en Italia en la ruta BLQ–MAD, este efecto fue parcialmente suavizado por el desempeño normal de Economy SH.  
- Evidencia:  
  • Business SH –7.6 pts por huelga en Italia en BLQ–MAD (ncs_tool)  
  • Economy SH +3.8 pts mantuvo normalidad sin desviaciones operativas significativas  

En LH, la dinámica es DOMINANCIA (Economy LH –, Business LH +, Premium N | LH –)  
- Narrativa: El LH hereda la caída porque Economy LH sufrió cancelaciones y retrasos por la huelga en Italia (vuelos IB1237/28NOV/MAD–MXP y IB1238/28NOV/BLQ–MAD), efecto que solo fue atenuado de forma parcial por el repunte aislado de Business LH.  
- Evidencia:  
  • Economy LH –10.4 pts por cancelaciones/retrasos por huelga en Italia (ncs_tool, rutas MAD–MXP y MAD–BLQ)  
  • Business LH +10.9 pts (sin causa operativa tangible; ruta MAD–MVD)

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es DILUCIÓN `(LH –, SH N | Global +)`  
- Narrativa: El NPS global refleja principalmente la caída del Largo Radio, que fue causada por las cancelaciones y retrasos derivados de la huelga en Italia en las rutas MAD–MXP y BLQ–MAD, efecto que se vio parcialmente atenuado por el desempeño normal del Corto Radio.  
- Evidencia:  
  • LH –7.6 pts: Economy LH con –10.4 pts explicado por cancelaciones/retrasos (huelga general en Italia, vuelos IB1237/1238/672 en MAD–MXP y MAD–BLQ)  
  • SH +2.9 pts (normal): sin desviaciones operativas significativas ni incidentes que impacten el NPS agregado.

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

CAUSA 1: Huelga general en Italia (cancelaciones y retrasos)  
- Escenario: DOMINANCIA en LH → DILUCIÓN en Global `(LH –, SH N | Global N)` y luego DOMINANCIA en LH `(Eco –, Bus +, Prem N | LH –)`  
- NMA: Global/LH/Economy  
- Afecta a:  
  • Global/LH/Economy (–10.4 pts)  
  • Global/SH/Economy/IB (–0.7 pts)  
  • Global/SH/Business/YW (–49.2 pts)  
- Tipo de impacto: Negativo  

CAUSA 2: Alta satisfacción de segmento Leisure y flota ATR  
- Escenario: TRANSFERENCIA en SH/Economy `(IB N, YW + | Economy SH N)`  
- NMA: Global/SH/Economy/YW  
- Afecta a:  
  • Global/SH/Economy/YW (+14.5 pts)  
- Tipo de impacto: Positivo

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

=== NMA: Global/LH/Economy ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
No disponible

🚨 INCIDENTES NCS:  
12 cancelaciones, 8 retrasos, cambios de aeronave y pérdidas de conexión (total 33 incidentes) · Vuelos cancelados por huelga: IB1237, IB1238, IB672

💬 FEEDBACK DE CLIENTES:  
Atención, puntualidad, comodidad y calidad de servicio destacados positivamente · No se mencionan cancelaciones, retrasos ni huelga

✈️ RUTAS AFECTADAS (Top 5):  
MAD–NRT: NPS –20.0 (n=5)  
(No hay más rutas reportadas por routes_tool)

👥 PERFILES REACTIVOS:  
Business/Work: NPS –16.0 (25 encuestas)  
Leisure: NPS 4.2 (194 encuestas)  
Fleet A321: NPS –62.5 (8 encuestas)  
Fleet A333: NPS –25.9 (23 encuestas)  
Fleet A350 C: NPS –25.0 (8 encuestas)  
Fleet A33ACMI: NPS –12.5 (8 encuestas)  
Residence Region Europa: NPS –50.0 (4 encuestas)  
Residence Region España: NPS –9.3 (216 encuestas)  
CodeShare BA: NPS –60.0  
CodeShare AA: NPS –28.6  
CodeShare Others: NPS –100.0  
CodeShare IB: NPS 5.7 (296 encuestas)

=== NMA: Global/SH/Economy/YW ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
Load Factor: +0.16  
OTP15: +2.48

🚨 INCIDENTES NCS:  
33 incidentes totales: 18 cancelaciones, 9 retrasos, reprogramaciones, cambios de equipo, conexiones perdidas (MAD-MXP, MAD-BLQ, BLQ-MAD)

💬 FEEDBACK DE CLIENTES:  
Verbatims del día (n=179): predominio de elogios a atención y servicio a bordo · No hay menciones a cancelaciones ni retrasos

✈️ RUTAS AFECTADAS (Top 5):  
BLQ–MAD: NPS 66.7 (n=6)  
LEU–MAD: NPS 0.0 (n=1)  
(Otras rutas con incidentes no tienen dato de NPS reportado)

👥 PERFILES REACTIVOS:  
Leisure: NPS 56.4 (102 encuestas)  
Business/Work: NPS 31.8 (44 encuestas)  
Fleet ATR: NPS 73.9 (23 encuestas)  
Fleet CRJ: NPS 44.3 (123 encuestas)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 2025-11-28, hemos identificado dos causas principales que explican las variaciones de NPS. El resultado global mostró una subida de 1.7 puntos con respecto a la media de los últimos 7 días.

En primer lugar, la caída del NPS en LH fue impulsada por el segmento Economy de LH, que registró un NPS de 1.8433 con una variación de –10.4 puntos con respecto a la media de los últimos 7 días. Este deterioro se debió a la huelga general en Italia, que provocó 12 cancelaciones, 8 retrasos y pérdidas de conexión en los vuelos IB1237, IB1238 e IB672 (incidentes operativos). Aunque el feedback de clientes destacó puntualidad, amabilidad y calidad de servicio, no hubo menciones de retrasos o cancelaciones. La desviación más notable se vio en la ruta MAD–NRT (NPS –20.0, n=5). Los perfiles más sensibles fueron Business/Work (NPS –16.0, 25 encuestas), flotas A321 (–62.5, 8), A333 (–25.9, 23), A350 C (–25.0, 8) y A33ACMI (–12.5, 8), así como pasajeros de Europa (–50.0, 4) y España (–9.3, 216), y clientes de code-share BA (–60.0), AA (–28.6) y Others (–100.0).

En contraste, la cabina Economy de SH en YW impulsó el desempeño agregado con un NPS de 48.9655 y una variación de +14.5 puntos con respecto a la media de los últimos 7 días. Este repunte estuvo soportado por métricas operativas sólidas (Load Factor +0.16, OTP +2.48), un feedback de clientes muy positivo en atención y servicio a bordo, y la alta satisfacción del segmento Leisure (NPS 56.4, 102 encuestas) y de la flota ATR (NPS 73.9, 23 encuestas). La ruta BLQ–MAD obtuvo NPS 66.7 (n=6), mientras que LEU–MAD cerró en NPS 0.0 (n=1), evidenciando consistencia a pesar de los 33 incidentes operativos del día (18 cancelaciones, 9 retrasos, reprogramaciones y conexiones perdidas).

**ECONOMY SH: Compensación de experiencias divergentes**  
La cabina Economy de SH registró un NPS de 37.8323 con una variación de +3.8 puntos con respecto a la media de los últimos 7 días. Desglose por compañía: IB obtuvo NPS 33.1395 (–0.7 pts), afectada por 18 cancelaciones y 9 retrasos en MAD–MXP y BLQ–MAD vinculados a la huelga en Italia; métrica operativa Load Factor –0.16, OTP +1.34; feedback de 438 comentarios mayoritariamente positivo sin menciones a cancelaciones; rutas más impactadas: MAD–MXP (NPS –66.7, n=6) y BIO–MAD (NPS 18.2, n=11); perfiles sensibles: Business/Work (NPS 8.5, n=59) y Leisure (NPS 38.2, n=286), flotas A320neo (44.3, n=158) y A320 (32.1, n=78). Por su parte, YW alcanzó NPS 48.9655 (+14.5 pts), respaldado por Load Factor +0.16 y OTP +2.48, un feedback muy positivo (n=179) y la fortaleza del segmento Leisure (56.4) y flota ATR (73.9), con ruta BLQ–MAD en NPS 66.7 (n=6). Estos efectos opuestos se cancelan en el agregado, dando estabilidad al segmento.

**BUSINESS SH: Impacto desigual por compañía**  
El segmento Business de SH registró un NPS de 27.7778 con una variación de –7.6 puntos con respecto a la media de los últimos 7 días. IB obtuvo NPS 50.0 (+15.2 pts), sin causas operativas claras (Load Factor –1.63, OTP +1.34), con 51 comentarios elogiando puntualidad y servicio en MAD–ORY (NPS 66.7, n=3) y alta satisfacción de Business/Work (52.9, n=17) y flotas A321 (66.7, n=6). En cambio, YW cayó a NPS –10.0 (–49.2 pts) por 18 cancelaciones y 9 retrasos en BLQ–MAD debidos a la huelga en Italia; ruta BLQ–MAD marcó NPS –100.0 (n=1); perfiles más afectados: Business/Work (–37.5, n=8), flota CRJ (–15.8, n=19) y pasajeros de Europa (–16.7, n=6). La debacle de YW arrastró al conjunto de Business SH.

**ECONOMY LH: Impacto de huelga en Italia**  
La cabina Economy de LH registró un NPS de 1.8433 con una variación de –10.4 puntos con respecto a la media de los últimos 7 días. La principal causa fue la huelga general en Italia, que generó 12 cancelaciones, 8 retrasos y pérdidas de conexión en vuelos IB1237, IB1238 e IB672 (incidentes operativos); no se menciona estos eventos en el feedback de clientes, que sigue destacando atención y comodidad. La ruta más afectada fue MAD–NRT (NPS –20.0, n=5). Los pasajeros más reactivos fueron Business/Work (–16.0, 25 encuestas), Leisure (4.2, 194), flotas A321 (–62.5), A333 (–25.9), A350 C (–25.0) y A33ACMI (–12.5), así como residentes de Europa (–50.0, 4) y España (–9.3, 216), y clientes de code-share BA (–60.0), AA (–28.6) y Others (–100.0).

**BUSINESS LH: Repunte sin correlación operativa**  
El segmento Business de LH registró un NPS de 35.2941 con una variación de +10.9 puntos con respecto a la media de los últimos 7 días. No se identificaron métricas operativas desviadas (Load Factor –0.39, OTP –2.48) ni validación en feedback de clientes (51 comentarios muy positivos sin menciones de incidentes). La ruta MAD–MVD destacó con NPS 50.0 (n=2). Los perfiles más satisfechos fueron Business/Work (70.0, 10), Leisure (20.8, 24), flotas A332 (66.7, 6) y A350 (50.0, 6); los code-share IB alcanzaron 37.9 (29), BA 100.0 (1) y LATAM –100.0 (1). La falta de causas operativas claras sugiere factores cualitativos no capturados.

**PREMIUM LH: Estabilidad consolidada**  
El segmento Premium de LH mantuvo desempeño estable, con NPS de 16.6667 y una variación de +0.6 puntos con respecto a la media de los últimos 7 días. No se detectaron cambios significativos, preservándose los niveles de satisfacción.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 6
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*

---

📅 2025-11-27 to 2025-11-27:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Economy SH, el escenario es TRANSFERENCIA (–, N | –).  
- Narrativa: La caída de NPS en Economy SH se explica por el deterioro en IB (–2.1 pts) originado por 20 cancelaciones y 13 retrasos debidos a la huelga en Italia en rutas clave (p. ej. MAD–MXP). Ese impacto de IB se contagió al agregado a pesar de que YW se mantuvo en rango normal.  
- Evidencia Clave: Incidentes NCS IB – 20 cancelaciones y 13 retrasos en rutas IB (MAD–MXP), nivel de confianza medio.

En Business SH, el escenario es DOMINANCIA (+, – | +).  
- Narrativa: El alza de NPS en Business SH responde al fuerte desempeño de IB (+12.2 pts), impulsado por la alta valoración del servicio a bordo (tripulación, calidad de menú y puntualidad), que superó la caída simultánea de YW.  
- Evidencia Clave: Verbatims IB – comentarios mayoritarios positivos sobre atención de la tripulación, calidad de comida y puntualidad (39 comentarios).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es DOMINANCIA (–, + | –).  
- Narrativa: El ligero descenso de NPS en SH responde principalmente al desplome de Economy SH provocado por las 20 cancelaciones y 13 retrasos de IB en la huelga de Italia (p. ej. MAD–MXP), efecto que dominó el agregado pese al fuerte repunte de Business SH.  
- Evidencia: Economy SH IB – 2.1 pts por 20 cancelaciones y 13 retrasos (nivel de confianza medio).  

En Long Haul, la dinámica es CANCELACIÓN (+,+,– | N).  
- Narrativa: La estabilidad de NPS en LH es engañosa: las subidas en Economy LH (+8.3 pts) y Business LH (+18.4 pts), impulsadas por feedback muy positivo en servicio y menú, se anularon con la fuerte caída de Premium LH (–11.9 pts) debida a 14 cancelaciones y 15 retrasos por la huelga en Italia (rutas MAD–BLQ, MAD–MXP).  
- Evidencia:  
  • Economy LH: NPS +8.3 pts, comentarios positivos sobre puntualidad y amabilidad.  
  • Business LH: NPS +18.4 pts, verbatims elogian profesionalidad de tripulación y calidad de comida.  
  • Premium LH: –11.9 pts por 14 cancelaciones y 15 retrasos (NCS_tool).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es DOMINANCIA (LH Normal, SH Negativo | Global Positivo).  
- Narrativa: El alza de 3.8 pts del NPS Global está arrastrada por el sólido desempeño de Long Haul: sus cabinas Economy LH (+8.3 pts) y Business LH (+18.4 pts), impulsadas por verbatims que destacan amabilidad, puntualidad y calidad de servicio, superaron la leve caída de Short Haul.  
- Evidencia:  
  • Economy LH +8.3 pts (amabilidad del personal y puntualidad en rutas DOH–MAD, n=273 comentarios positivos).  
  • Business LH +18.4 pts (profesionalidad de tripulación y calidad del menú, n=30 comentarios positivos).

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

CAUSA 1: Impacto de cancelaciones/retrasos en Economy SH (IB)  
- Escenario: TRANSFERENCIA (–, N | –)  
- NMA: Global/SH/Economy/IB  
- Afecta a: Global/SH/Economy, Global/SH  
- Tipo de impacto: NEGATIVO  

CAUSA 2: Mejora de servicio a bordo en Business SH (IB)  
- Escenario: DOMINANCIA (+, – | +)  
- NMA: Global/SH/Business/IB  
- Afecta a: Global/SH/Business, Global/SH  
- Tipo de impacto: POSITIVO  

CAUSA 3: Desacople de cabinas en Long Haul (Eco LH +, Bus LH + vs Prem LH –)  
- Escenario: CANCELACIÓN (+, +, – | N)  
- NMA: Global/LH/Economy; Global/LH/Business; Global/LH/Premium  
- Afecta a: Economy LH (pos), Business LH (pos), Premium LH (neg)  
- Tipo de impacto: MIXTO  

CAUSA 4: Empuje de Long Haul arrastrando el Global  
- Escenario: DOMINANCIA (LH, SH | +)  
- NMA: Global/LH  
- Afecta a: Global  
- Tipo de impacto: POSITIVO

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

=== NMA: Global/SH/Economy/IB ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
- Ninguna métrica operativa se desvió >3 pts vs baseline (OTP15 aumentó ligeramente, Load Factor disminuyó ligeramente)

🚨 INCIDENTES NCS:  
• 20 cancelaciones  
• 13 retrasos  
• Zona afectada: Italia

💬 FEEDBACK DE CLIENTES:  
415 comentarios mayoritariamente positivos (puntualidad, amabilidad)  
No se mencionan cancelaciones ni retrasos

✈️ RUTAS AFECTADAS (Top 5):  
• MAD–MXP: NPS 0.0 (12 encuestas) (incidentes NCS presentes)  
• BRU–MAD: NPS 0.0 (11 encuestas)

👥 PERFILES REACTIVOS:  
• Leisure: NPS 36.8 (229 encuestas)  
• Business/Work: NPS 19.4 (93 encuestas)  


=== NMA: Global/SH/Business/IB ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
- Métricas operativas: ninguna se desvió >3 pts vs baseline (Load Factor –1.72, OTP15 +1.28)

🚨 INCIDENTES NCS:  
• 34 incidentes en total (20 cancelaciones, 13 retrasos), vinculados a convocatoria de huelga en Italia

💬 FEEDBACK DE CLIENTES:  
39 comentarios, todos positivos, destacando servicio de a bordo, atención de la tripulación, calidad de comida y puntualidad  
No se mencionan cancelaciones ni retrasos

✈️ RUTAS AFECTADAS (Top 5):  
• MAD–MXP: NPS 66.7 (3 encuestas)  
• MAD–BLQ: incidentes reportados (encuestas no disponible)  
• BLQ–MAD: incidentes reportados (encuestas no disponible)

👥 PERFILES REACTIVOS:  
• Región EUROPA: NPS 0.0 (12 encuestas)  
• España: NPS 58.8 (17 encuestas)  
• Segmento Business: NPS 53.3 (15 encuestas)  
• Segmento Leisure: NPS 42.1 (20 encuestas)  


=== NMA: Global/LH/Economy ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
- Sin desviaciones significativas (>3 pts) vs baseline (operative_data_tool)

🚨 INCIDENTES NCS:  
• 37 incidentes: 14 cancelaciones, 15 retrasos y 7 mishandling  
• Huelga en Italia afectando rutas MAD–BLQ y MAD–MXP

💬 FEEDBACK DE CLIENTES:  
273 comentarios, mayoritariamente positivos (amabilidad, buen servicio, puntualidad)  
No hay menciones a cancelaciones, retrasos ni equipaje

✈️ RUTAS AFECTADAS (Top 5):  
• DOH–MAD: NPS 20.0 (5 encuestas)  
• MAD–BLQ: NPS no disponible / encuestas no disponible  
• MAD–MXP: NPS no disponible / encuestas no disponible

👥 PERFILES REACTIVOS:  
• Leisure: NPS 23.0 (139 encuestas)  
• Business/Work: NPS 4.5 (22 encuestas)  
• Fleet A350 C: NPS –40.0 (10 encuestas)  
• Residence Region – Europa: NPS –18.2 (encuestas no disponible)  
• Residence Region – Asia: NPS –100.0 (encuestas no disponible)  


=== NMA: Global/LH/Business ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
- Load Factor: –0.11 pts vs baseline  
- OTP15_adjusted: –2.39 pts vs baseline

🚨 INCIDENTES NCS:  
• 14 cancelaciones, 15 retrasos, 7 problemas de equipaje (total 37)  
• Rutas con más incidencias: MAD–MXP (3), MAD–BLQ (1)

💬 FEEDBACK DE CLIENTES:  
30 comentarios, todos positivos, destacando comodidad en cabina ejecutiva y profesionalidad del personal  
No hay menciones a cancelaciones, retrasos ni equipaje

✈️ RUTAS AFECTADAS (Top 5):  
• MAD–SJO: NPS 100.0 (n=2)

👥 PERFILES REACTIVOS:  
• Leisure: NPS 46.2 (13 encuestas)  
• Business/Work: NPS 37.5 (8 encuestas)  
• Fleet: A333 100.0 (2), A332 75.0 (4), A33ACMI 50.0 (2), A350 14.3 (7), A321XLR 0.0 (2)  
• Región: España 36.4 (11), América Sur 0.0 (2)  
• CodeShare: IB 36.8 (19)  


=== NMA: Global/LH/Premium ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
- Load Factor: −2.86 pts vs baseline (mejora, no correlaciona)  
- OTP15: −2.39 pts vs baseline (no significativa)

🚨 INCIDENTES NCS:  
• 14 cancelaciones (MAD–BLQ, MAD–MXP por huelga en Italia)  
• 15 retrasos (p. ej. ORY–MAD, reprogramación 25 min)

💬 FEEDBACK DE CLIENTES:  
Total comentarios: 28  
Todos positivos: elogios al servicio y actitud de la tripulación  
No se mencionan cancelaciones, retrasos ni quejas operativas

✈️ RUTAS AFECTADAS (Top 5):  
• No se identificaron rutas con desviación negativa  
• Única ruta con datos: GRU–MAD, NPS 100.0 (n=2)

👥 PERFILES REACTIVOS:  
• Business/Work: NPS −66.7 (9 encuestas)  
• Leisure: NPS 46.7 (15 encuestas)  
• CodeShare IB: NPS 9.5 (21 encuestas)  
• Otras aerolíneas code share y regiones: <3 encuestas cada una  


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

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 27 de noviembre de 2025, hemos identificado cuatro focos de influencia que explican las oscilaciones en el NPS Global, que resultó en una mejora de 3.8 puntos con respecto a la media de los últimos 7 días. El alza global está impulsada por el sólido desempeño de LH, mientras que SH muestra dinámicas opuestas en sus cabinas Economy y Business.

En SH, la cabina Economy sostuvo un retroceso motivado por los 20 cancelaciones y 13 retrasos de IB derivados de la huelga en Italia. Con un NPS de 33.532934131736525 y una variación de –0.5 pts con respecto a la media de los últimos 7 días, IB cayó a 31.775700934579444 (–2.1 pts) frente a un repunte de YW a 36.66666666666667 (+2.2 pts). A pesar de 415 comentarios mayoritariamente positivos sobre puntualidad y amabilidad, las rutas MAD–MXP (NPS 0.0, n=12) y BRU–MAD (NPS 0.0, n=11) reflejaron el impacto, con mayor sensibilidad en clientes Leisure (NPS 36.8, 229 encuestas) y especialmente en Business/Work (NPS 19.4, 93 encuestas).

Por contraste, SH Business escaló hasta un NPS de 43.90243902439025 (+8.6 pts). IB alcanzó 47.05882352941177 (+12.2 pts) gracias a 39 comentarios sobresalientes sobre servicio de a bordo, atención de la tripulación, calidad de comida y puntualidad, mientras YW descendió a 28.571428571428573 (–10.7 pts). Aunque hubo 34 incidentes operativos (20 cancelaciones, 13 retrasos), las rutas MAD–MXP (NPS 66.7, n=3) y los perfiles España (NPS 58.8, 17 encuestas) y Business (NPS 53.3, 15 encuestas) fuertemente positivos sostuvieron la subida.

En LH, Economy registró un NPS de 20.496894409937905 (+8.3 pts) sin desviaciones significativas en métricas operativas. Aun con 37 incidentes (14 cancelaciones, 15 retrasos, 7 mishandling) en rutas como DOH–MAD (NPS 20.0, n=5), el feedback de 273 comentarios positivos sobre amabilidad, buen servicio y puntualidad y la fuerte valoración del segmento Leisure (NPS 23.0, 139 encuestas) sostuvieron la mejora, contrarrestando la menor satisfacción de Business/Work (NPS 4.5, 22 encuestas).

La cabina Business en LH sorprendió con un NPS de 42.85714285714285 (+18.4 pts), a pesar de métricas operativas estables (Load Factor –0.11 pts, OTP15_adjusted –2.39 pts). Los 37 incidentes (14 cancelaciones, 15 retrasos, 7 problemas de equipaje) no opacaron los 30 comentarios muy positivos sobre comodidad en cabina ejecutiva y profesionalidad del personal. MAD–SJO alcanzó NPS 100.0 (n=2), con perfiles Leisure (NPS 46.2, 13 encuestas), Business/Work (NPS 37.5, 8 encuestas), flota A333 (100.0, 2) y code-share IB (NPS 36.8, 19).

El segmento Premium LH cayó a un NPS de 4.166666666666671 (–11.9 pts), derivado principalmente de 14 cancelaciones en rutas MAD–BLQ y MAD–MXP y 15 retrasos (p. ej. ORY–MAD, reprogramación 25 min). Aunque los 28 comentarios elogiaron actitud de tripulación y servicio, el perfil Business/Work registró NPS –66.7 (9 encuestas) frente a Leisure 46.7 (15), y en code-share IB solo 9.5 (21). La única ruta con datos fue GRU–MAD: NPS 100.0 (n=2).

---

**DETALLE POR CABINA:**

**ECONOMY SH: Impacto de Cancelaciones IB**  
La cabina Economy de SH registró un NPS de 33.532934131736525 con una variación de –0.5 pts con respecto a la media de los últimos 7 días. Desglose por compañía: IB obtuvo 31.775700934579444 (–2.1 pts) y YW 36.66666666666667 (+2.2 pts). El descenso se explica por 20 cancelaciones y 13 retrasos en IB vinculados a la huelga en Italia, que se reflejó en rutas críticas como MAD–MXP (NPS 0.0, n=12) y BRU–MAD (NPS 0.0, n=11). Aun con 415 comentarios mayoritariamente positivos sobre puntualidad y amabilidad, la sensibilidad de clientes Leisure (NPS 36.8, 229 encuestas) y Business/Work (NPS 19.4, 93 encuestas) acentuó la baja.

**BUSINESS SH: Impulso del Servicio IB**  
El segmento Business de SH alcanzó un NPS de 43.90243902439025 con un alza de +8.6 pts con respecto a la media de los últimos 7 días. IB destacó con 47.05882352941177 (+12.2 pts) gracias a 39 comentarios positivos sobre servicio de a bordo, atención de la tripulación, calidad de comida y puntualidad, mientras YW cayó a 28.571428571428573 (–10.7 pts). A pesar de 34 incidentes operativos (20 cancelaciones, 13 retrasos), rutas como MAD–MXP (NPS 66.7, n=3) y perfiles España (NPS 58.8, 17 encuestas) y Business (NPS 53.3, 15 encuestas) sostuvieron la fuerte mejora.

**ECONOMY LH: Resiliencia a Incidentes**  
La cabina Economy de LH registró un NPS de 20.496894409937905 con una mejora de +8.3 pts con respecto a la media de los últimos 7 días. No se detectaron cambios significativos en métricas operativas pese a 37 incidentes (14 cancelaciones, 15 retrasos, 7 mishandling) en rutas como DOH–MAD (NPS 20.0, n=5). Los 273 comentarios positivos centrados en amabilidad, buen servicio y puntualidad, junto al sólido comportamiento del segmento Leisure (NPS 23.0, 139 encuestas), explican esta recuperación, a pesar de la menor valoración de Business/Work (NPS 4.5, 22 encuestas).

**BUSINESS LH: Excelencia en la Experiencia**  
La cabina Business de LH registró un NPS de 42.85714285714285 con una variación de +18.4 pts con respecto a la media de los últimos 7 días. Los 37 incidentes (14 cancelaciones, 15 retrasos, 7 problemas de equipaje) no impidieron que 30 comentarios destacados elogiando la comodidad en cabina ejecutiva y la profesionalidad del personal sostuvieran la mejora. En rutas como MAD–SJO (NPS 100.0, n=2), los perfiles Leisure (NPS 46.2, 13 encuestas), Business/Work (NPS 37.5, 8 encuestas), flota A333 (NPS 100.0, 2) y code-share IB (NPS 36.8, 19) fueron los más reactivos.

**PREMIUM LH: Impacto Operativo de Huelga**  
El segmento Premium de LH sufrió un descenso a un NPS de 4.166666666666671, con una caída de –11.9 pts con respecto a la media de los últimos 7 días. Las 14 cancelaciones en MAD–BLQ y MAD–MXP y 15 retrasos (p. ej. ORY–MAD con reprogramación de 25 min) fueron las causas dominantes, pese a comentarios positivos de servicio (28 comentarios). Los perfiles Business/Work (NPS –66.7, 9 encuestas) y code-share IB (NPS 9.5, 21 encuestas) reflejaron la mayor insatisfacción, mientras Leisure mostró resistencia (NPS 46.7, 15 encuestas).

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 6
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*

---

📅 2025-11-26 to 2025-11-26:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

Economy SH  
- Escenario: DILUCIÓN (IB N, YW + | Economy SH N)  
- Narrativa: La anomalía positiva de +7.8 pts en Young Women (YW) no fue suficiente para mover el NPS agregado de Economy Short Haul, porque Iberia (IB) se mantuvo en rango normal y diluyó el efecto.  
- Evidencia Clave: YW mostró comentarios muy positivos sobre puntualidad, rapidez y trato (verbatims_tool, n=215), sin que ninguna métrica operacional superase ±3 pts frente al baseline (OTP15_adjusted +2.46 pts; Load Factor –0.16 pts).

Business SH  
- Escenario: TRANSFERENCIA (IB +, YW N | Business SH +)  
- Narrativa: Adoptamos la explicación del nodo padre. El alza de +13.6 pts en Business Short Haul se vincula principalmente a las 12 cancelaciones de vuelos BRU–MAD y MAD–BRU por huelga (ncs_tool), que generaron un NPS 0.0 muy bajo en esa ruta y arrastraron el promedio del segmento, a pesar de un feedback cualitativo general muy favorable.  
- Evidencia Clave: Incidentes NCS – 12 cancelaciones por huelga en BRU–MAD/MAD–BRU (ncs_tool), correlacionado con NPS 0.0 en IB (routes_tool); verbatims de Business SH sin mención de problemas operativos (verbatims_tool, 56 comentarios).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es DILUCIÓN (Economy SH N, Business SH + | SH N).  
- Narrativa: El segmento SH habría mostrado un alza si no fuera porque Economy SH se mantuvo estable y absorbió parcialmente el empuje de Business SH. El rendimiento sobresaliente de Business SH (por los impactos de las cancelaciones de BRU–MAD/MAD–BRU) quedó diluido en el promedio.  
- Evidencia: Business SH sufrió 12 cancelaciones por huelga en BRU–MAD/MAD–BRU (ncs_tool) con NPS 0.0 en esa ruta (routes_tool); Economy SH no presentó desviaciones operativas ni cualitativas significativas.

En Long Haul, la dinámica es SINERGIA (Economy LH N, Business LH +, Premium LH + | LH +).  
- Narrativa: El alza de +9.3 pts en NPS Long Haul obedece a la sinergia de Business LH y Premium LH, cuyos fuertes incrementos arrastraron el promedio de todo el radio.  
- Evidencia: Business LH subió +22.2 pts (OTP15_adjusted –3.3 pts y 8 cancelaciones, 4 retrasos en rutas clave; operative_data_tool & ncs_tool) y Premium LH subió +42.3 pts (Load Factor –3.01 pts y verbatims muy positivos sobre comodidad y servicio; operative_data_tool & verbatims_tool).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA (LH +, SH N | Global +).  
- Narrativa: El alza global de +8.1 pts está impulsada íntegramente por el excelente desempeño de Long Haul, mientras que Short Haul se mantuvo en rango normal y no contrarrestó el efecto.  
- Evidencia: Long Haul registró +9.3 pts, potenciado por Business LH (+22.2 pts) con OTP15_adjusted –3.3 pts y 8 cancelaciones por huelga en BRU–MAD/MAD–BRU (ncs_tool), y Premium LH (+42.3 pts) asociado a un Load Factor –3.01 pts y feedback muy positivo en verbatims (operative_data_tool & verbatims_tool).

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

CAUSA 1: Mejora cualitativa en Young Women (Economy SH)  
- Escenario: DILUCIÓN  
- NMA: Global / SH / Economy / YW  
- Afecta a: Global / SH / Economy / YW  
- Tipo de impacto: POSITIVO  

CAUSA 2: Cancelaciones por huelga BRU–MAD/MAD–BRU (Business SH)  
- Escenario: TRANSFERENCIA  
- NMA: Global / SH / Business  
- Afecta a: Global / SH / Business  
- Tipo de impacto: POSITIVO  

CAUSA 3: Caída de puntualidad (OTP) en Business Long Haul  
- Escenario: SINERGIA  
- NMA: Global / LH  
- Afecta a: Global / LH (Business y Premium)  
- Tipo de impacto: POSITIVO  

CAUSA 4: Menor Load Factor en Premium Long Haul  
- Escenario: SINERGIA  
- NMA: Global / LH  
- Afecta a: Global / LH (Business y Premium)  
- Tipo de impacto: POSITIVO

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

=== NMA: Global/SH/Economy/YW ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
OTP15_adjusted: +2.46 pts  
Load_Factor: –0.16 pts  

🚨 INCIDENTES NCS:  
12 cancelaciones (huelga en BRU, vuelos BRU-MAD e IB601/26NOV/MAD-BRU)  
2 problemas de equipaje  
6 otras incidencias  

💬 FEEDBACK DE CLIENTES:  
Verbatims (n=215): predominan comentarios sobre puntualidad, rapidez y buen trato; no se mencionan cancelaciones ni huelga  

✈️ RUTAS AFECTADAS (Top 5):  
LEI-PMI: NPS 0.0 (n=2)  

👥 PERFILES REACTIVOS:  
Leisure: NPS 43.9 (n=98)  
Business/Work: NPS 40.3 (n=72)  
Fleet – CRJ: NPS 43.7 (n=151); ATR: NPS 31.6 (n=19)  
Residence Region – África: NPS –100.0 (n=2); Centroamérica: NPS 100.0 (n=4)  

=== NMA: Global/SH/Business ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
Load Factor: 70.8 (–1.94 pts vs baseline)  
OTP15_adjusted: 91.47 (+2.07 pts vs baseline)  

🚨 INCIDENTES NCS:  
12 de 18 incidentes fueron cancelaciones de los vuelos BRU–MAD y MAD–BRU por huelga anunciada en BRU  

💬 FEEDBACK DE CLIENTES:  
Temas principales en verbatims (56 comentarios): todos positivos sobre atención en Business Short Haul; sin menciones a cancelaciones, demoras o mishandling  

✈️ RUTAS AFECTADAS (Top 5):  
BRU–MAD: NPS 0.0 (1 encuesta)  
MAD–ORY: NPS 25.0 (4 encuestas)  

👥 PERFILES REACTIVOS:  
Leisure: NPS 52.6 (19 encuestas)  
Business/Work: NPS 46.4 (28 encuestas)  
Flota – A321: NPS 25.0 (número de encuestas no disponible); CRJ: NPS 37.5 (número de encuestas no disponible)  
Residence Region – América Centro: NPS 0.0 (número de encuestas no disponible)  

=== NMA: Global/LH ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
OTP15_adjusted 26-Nov: 76.71  
Variación vs baseline: –3.3 pts  
Load Factor • Variación vs baseline: –3.01 pts  
OTP15_adjusted • Variación vs baseline: –3.30 pts  

🚨 INCIDENTES NCS:  
8 cancelaciones (incluyendo BRU–MAD y MAD–BRU por huelga en BRU)  
4 retrasos (uno de 1h50 por rotación de avión)  
3 cambios de aeronave y varias pérdidas de conexión en MAD  

💬 FEEDBACK DE CLIENTES:  
Verbatims del día (352 comentarios): “Embarque organizado, tripulación muy atenta, buen avión y asiento…”, “Excelente atención…”, “Todo excelente. La comida. Los chicos asistentes muy cordiales.” No se mencionan retrasos, cancelaciones ni problemas operativos.  

✈️ RUTAS AFECTADAS (Top 5):  
DOH–MAD: NPS 33.3 (n=3)  

👥 PERFILES REACTIVOS:  
Business/Work: NPS 28.1 (29 encuestas)  
Leisure: NPS 22.4 (189 encuestas)  
Fleet: A321XLR: 19; A33ACMI: 7  
Residence Region: Europa: NPS –29.6 (31 encuestas)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 26 de noviembre de 2025, hemos identificado cuatro causas principales que explican las variaciones de NPS. El NPS global subió 8.1 pts con respecto a la media de los últimos 7 días, impulsado fundamentalmente por el desempeño de LH y matizado por dinámicas contrapuestas en SH.

El segmento Economy de SH mantuvo un desempeño estable, aunque dentro de él Young Women mostró un alza de 7.8 pts. En este caso, métricas operativas moderadas (OTP +2.46 ppts según métricas operativas; Load Factor –0.16 pts según métricas operativas) y un feedback de clientes muy positivo («puntualidad», «rapidez» y «buen trato») sostuvieron el nivel, pese a 12 cancelaciones por huelga en BRU–MAD/MAD–BRU, 2 problemas de equipaje y 6 otras incidencias reportadas por incidentes operativos. La ruta LEI–PMI registró un NPS 0.0 (n=2) y los perfiles más reactivos fueron Leisure (NPS 43.9, 98 encuestas), Business/Work (NPS 40.3, 72 encuestas), flota CRJ (NPS 43.7, 151 encuestas), ATR (NPS 31.6, 19 encuestas) y residentes en África (NPS –100.0, 2 encuestas) y Centroamérica (NPS 100.0, 4 encuestas).

En Business SH, el NPS creció 13.6 pts hasta 48.94 con respecto a la media de los últimos 7 días, en gran parte por la gestión de Iberia, que alcanzó un NPS 53.33 (+18.5 pts según Explanatory Drivers) frente a YW en 41.18 (+1.9 pts según Explanatory Drivers). Aunque la puntualidad mejoró ligeramente (OTP +2.07 ppts según métricas operativas) y el Load Factor cayó –1.94 pts según métricas operativas, fueron las 12 cancelaciones por huelga en BRU–MAD/MAD–BRU las que explicaron el impacto, con un NPS 0.0 en esa ruta (1 encuesta). El feedback de clientes en Business SH (56 verbatims) fue uniformemente positivo, sin menciones a demoras o cancelaciones, y los perfiles más sensibles fueron Leisure (NPS 52.6, 19 encuestas), Business/Work (NPS 46.4, 28 encuestas), y usuarios de flota A321 (NPS 25.0) y CRJ (NPS 37.5), así como residentes en América Centro (NPS 0.0).

En LH, ambas cabinas Business y Premium actuaron en sinergia para elevar el NPS Long Haul en 9.3 pts con respecto a la media de los últimos 7 días. Business LH alcanzó 46.67 (+22.2 pts con respecto a la media de los últimos 7 días) a pesar de una caída de puntualidad (OTP 76.71, –3.3 pts según métricas operativas) y de 8 cancelaciones, 4 retrasos, 3 cambios de avión y pérdidas de conexión según incidentes operativos. En paralelo, el feedback de clientes (51 verbatims) resaltó «embarque organizado», «puntualidad», «excelente atención» y «buenas instalaciones», y la ruta MAD–SDQ registró un NPS 100.0 (1 encuesta). Los perfiles más sensibles en Business LH fueron Business/Work (NPS 75.0, 8 encuestas), Leisure (NPS 36.4, 22 encuestas), flota A321XLR (NPS 19.0, 19 encuestas), A33ACMI (NPS 7.0, 7 encuestas) y residentes en Europa (NPS –29.6, 31 encuestas).

Premium LH presentó un NPS de 58.33 (+42.3 pts con respecto a la media de los últimos 7 días), impulsado por una carga más baja (Load Factor –3.01 pts según métricas operativas) a pesar de una menor puntualidad (OTP –3.30 pts según métricas operativas) y de 8 cancelaciones y 4 retrasos según incidentes operativos. El feedback de clientes (18 verbatims) destacó «comodidad», «amabilidad», «todo a tiempo», «servicio a bordo» y «accesibilidad digital», con la ruta MAD–SCL en NPS 100.0 (1 encuesta). Los viajeros Business/Work alcanzaron NPS 100.0 (2 encuestas) y Leisure NPS 50.0 (10 encuestas).

---

**DETALLE POR CABINA:**

**ECONOMY SH: Estabilidad con pockets de satisfacción**  
La cabina Economy de SH mantuvo un NPS de 38.52813852813853 con una variación de +4.5 pts con respecto a la media de los últimos 7 días. Desglose por compañía: IB obtuvo un NPS de 36.3013698630137 (+2.4 pts) y YW 42.352941176470594 (+7.8 pts). No se detectaron picos de Explanatory Drivers por encima de 3 ppts. Las métricas operativas mostraron OTP +2.46 ppts y Load Factor –0.16 pts. El feedback de clientes resaltó «puntualidad», «rapidez» y «buen trato», pese a 12 cancelaciones, 2 problemas de equipaje y 6 incidencias según incidentes operativos. La ruta LEI–PMI registró NPS 0.0 (n=2). Los perfiles más reactivos fueron Leisure, Business/Work, flota CRJ y ATR, y residentes en África y Centroamérica.

**BUSINESS SH: Impacto de las cancelaciones de huelga**  
El segmento Business de SH registró un NPS de 48.936170212765965 con una variación de +13.6 pts con respecto a la media de los últimos 7 días. Desglose por compañía: IB alcanzó 53.333333333333336 (+18.5 pts según Explanatory Drivers) y YW 41.17647058823529 (+1.9 pts según Explanatory Drivers). La puntualidad mejoró levemente (OTP +2.07 ppts) y el Load Factor cayó –1.94 pts, pero 12 cancelaciones por huelga en BRU–MAD/MAD–BRU provocaron un NPS 0.0 en esa ruta (1 encuesta). El feedback de clientes (56 verbatims) fue uniformemente positivo y las rutas BRU–MAD y MAD–ORY reflejaron NPS 0.0 (1 encuesta) y 25.0 (4 encuestas), respectivamente. Los perfiles más sensibles incluyeron Leisure, Business/Work, usuarios de flota A321 y CRJ, y residentes en América Centro.

**ECONOMY LH: Mantiene desempeño estable**  
La cabina Economy de LH mantuvo desempeño estable, con un NPS de 17.613636363636374 y una variación de +5.4 pts con respecto a la media de los últimos 7 días. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción.

**BUSINESS LH: Mejora a pesar de la merma de puntualidad**  
La cabina Business de LH alcanzó un NPS de 46.66666666666668 con una variación de +22.2 pts con respecto a la media de los últimos 7 días. La caída de puntualidad (OTP 76.71, –3.3 pts) y 8 cancelaciones, 4 retrasos, 3 cambios de avión y pérdidas de conexión según incidentes operativos no impidieron una respuesta muy positiva de los clientes (51 verbatims). La ruta MAD–SDQ registró NPS 100.0 (1 encuesta) y los perfiles más sensibles fueron Business/Work, Leisure y flotas A321XLR y A33ACMI.

**PREMIUM LH: Éxito por menor ocupación y excelente servicio**  
El segmento Premium de LH registró un NPS de 58.33333333333332 con una variación de +42.3 pts con respecto a la media de los últimos 7 días. La menor ocupación (Load Factor –3.01 pts) y una leve merma en puntualidad (OTP –3.30 pts) convivieron con un feedback excepcional (18 verbatims) centrado en comodidad, amabilidad, «todo a tiempo», servicio a bordo y accesibilidad digital. La ruta MAD–SCL alcanzó NPS 100.0 (1 encuesta) y los viajeros Business/Work y Leisure mostraron NPS 100.0 y 50.0, respectivamente.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 6
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*

---

📅 2025-11-25 to 2025-11-25:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En SH/Economy, el escenario es CANCELACIÓN (IB: –0.6, YW: +17.1 | Normal).  
- Narrativa: Mientras la cabina IB sufrió por cancelaciones y retrasos en la ruta BRU-MAD vinculados a huelga, la cabina YW se benefició de una mejora en puntualidad (OTP15 +2.26 pts) y de menor ocupación (Load Factor –0.42 pts), neutralizándose mutuamente en el agregado.  
- Evidencia Clave:  
  • Economy IB: Incidentes NCS en BRU-MAD (10 cancelaciones, 2 retrasos)  
  • Economy YW: OTP15 +2.26 pts vs baseline; Load Factor –0.42 pts vs baseline  

En SH/Business, el escenario es DOMINANCIA (IB: +19.0, YW: –14.2 | +9.4).  
- Narrativa: La subida de SH/Business responde principalmente a la cabina IB, impulsada por un alza de puntualidad (OTP15 +1.61 pts) y una menor carga de ocupación (Load Factor –2.04 pts); este efecto dominante fue parcialmente suavizado por la anomalía negativa de YW, ligada a bajas de Load Factor (–3.72 pts) e incidentes por huelga.  
- Evidencia Clave:  
  • Business IB: OTP15 +1.61 pts y Load Factor –2.04 pts vs baseline  
  • Business YW: Load Factor –3.72 pts vs baseline; incidencias NCS en BRU-MAD/MAD-BRU

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En SH, la dinámica es DILUCIÓN (Economy: N, Business: + | SH: N).  
- Narrativa: El rendimiento de SH está dictado por SH/Business, cuya anomalía positiva, impulsada por la mejora en puntualidad de IB (OTP15 +1.61 pts) y la menor ocupación (Load Factor –2.04 pts), se vio parcialmente suavizada por la estabilidad de SH/Economy.  
- Evidencia:  
  • SH/Business IB: OTP15 +1.61 pts vs baseline; Load Factor –2.04 pts vs baseline  

En LH, la dinámica es CANCELACIÓN (Economy: N, Business: –, Premium: + | LH: N).  
- Narrativa: LH muestra estabilidad engañosa: LH/Business cayó por un empeoramiento de puntualidad (OTP15_adjusted –4.97 pts) y 12 cancelaciones + 3 retrasos en rutas BRU-MAD/MAD-BRU, mientras LH/Premium subió +22.8 pts sin causas operativas claras ni reflejo en verbatims, neutralizándose mutuamente.  
- Evidencia:  
  • LH/Business: OTP15_adjusted –4.97 pts vs baseline; 12 cancelaciones y 3 retrasos (ncs_tool)  
  • LH/Premium: NPS +22.8 pts sin desviaciones en OTP o Load Factor ni menciones de incidentes en verbatims

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es SINERGIA (LH: +, SH: + | GLOBAL: +).  
- Narrativa: El alza de NPS de +6.3 pts responde a un efecto de red: tanto Largo Radio como Corto Radio mostraron mejoras (aunque dentro de rangos normales) y el Global heredó este impulso, potenciado por la gestión de incidentes operacionales.  
- Evidencia:  
  • 58 cancelaciones y 24 retrasos (ncs_tool) que catalizaron feedback positivo  
  • OTP15 +1.14 pts vs baseline (operative_data_tool)  
  • Load Factor –1.18 pts vs baseline (operative_data_tool)

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

CAUSA 1: Cancelaciones y retrasos por huelga en rutas BRU-MAD/MAD-BRU  
- Escenario: CANCELACIÓN (LH: Economy N, Business –, Premium + | LH N) y CANCELACIÓN (SH/Economy: IB –, YW + | N)  
- NMA:  
  • Global/LH/Business  
  • Global/SH/Economy/IB  
- Afecta a: LH/Business; SH/Economy/IB  
- Tipo de impacto: NEGATIVO  

CAUSA 2: Mejora de puntualidad (OTP15) y menor carga (Load Factor) en SH/Business/IB  
- Escenario: DOMINANCIA (SH/Business: IB +, YW – | +)  
- NMA: Global/SH/Business/IB  
- Afecta a: SH/Business  
- Tipo de impacto: POSITIVO  

CAUSA 3: Mejora de puntualidad (OTP15) y menor carga (Load Factor) en SH/Economy/YW  
- Escenario: CANCELACIÓN (SH/Economy: IB –, YW + | N)  
- NMA: Global/SH/Economy/YW y Global/SH/Economy/IB  
- Afecta a: SH/Economy  
- Tipo de impacto: POSITIVO  

CAUSA 4: Gestión de incidentes y ligera mejora operativa de red (OTP15 +1.14, Load Factor –1.18)  
- Escenario: SINERGIA (LH +, SH + | Global +)  
- NMA: Global  
- Afecta a: toda la red  
- Tipo de impacto: POSITIVO

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

=== NMA: Global/LH/Business ===

📈 EXPLANATORY DRIVERS:  
– Empeoramiento de la puntualidad (OTP15_adjusted 74.95 vs baseline ~79.92 → −4.97 pts)

📊 DATOS OPERATIVOS:  
– OTP15_adjusted 74.95 vs baseline ~79.92 → −4.97 pts

🚨 INCIDENTES NCS:  
– 12 cancelaciones y 3 retrasos

💬 FEEDBACK DE CLIENTES:  
– Feedback 100 % positivo: énfasis en “excelente servicio”, “aviones cómodos” y “atención al cliente”

✈️ RUTAS AFECTADAS (Top 5):  
– EZE-MAD: NPS 100.0 (2 encuestas)

👥 PERFILES REACTIVOS:  
– Business/Work: NPS −25.0 (8 encuestas)  
– Leisure: NPS 38.9 (18 encuestas)  
– Region Residence – Europa: NPS −50.0 (n no disponible)  
– CodeShare – LATAM: NPS −100.0 (1 encuesta)  
– Fleet A321XLR: NPS 66.7 (3 encuestas); A350 y A33ACMI en extremo inferior (datos de n no disponibles)  

=== NMA: Global/SH/Economy/IB ===

📈 EXPLANATORY DRIVERS:  
– Métricas operativas: no se identificaron desviaciones >3 pts vs baseline (Load Factor −0.45 pts; OTP15_adjusted +1.61 pts) → no correlación con la caída de NPS

📊 DATOS OPERATIVOS:  
– Load Factor −0.45 pts; OTP15_adjusted +1.61 pts

🚨 INCIDENTES NCS:  
– 10 cancelaciones, 2 retrasos (17 incidentes totales)  
– Cancelaciones BRU-MAD y MAD-BRU por huelga  
– Retrasos de 45 min y 1 h 25 min en vuelos de conexión

💬 FEEDBACK DE CLIENTES:  
– No hay datos disponibles

✈️ RUTAS AFECTADAS (Top 5):  
– BRU-MAD: NPS 15.0 (n=20)  
– DUS-MAD: NPS 20.0 (n=10)

👥 PERFILES REACTIVOS:  
– Business/Work: NPS 26.3 (n=95)  
– Leisure: NPS 36.3 (n=215)  
– Región América Norte: NPS −20.0 (n=10)  

=== NMA: Global/SH/Business/IB ===

📈 EXPLANATORY DRIVERS:  
– OTP15: +1.61 pts vs baseline  
– Load Factor: −2.04 pts vs baseline

📊 DATOS OPERATIVOS:  
– OTP15: +1.61 pts vs baseline  
– Load Factor: −2.04 pts vs baseline

🚨 INCIDENTES NCS:  
– 17 incidentes (10 cancelaciones, 7 reprogramaciones)

💬 FEEDBACK DE CLIENTES:  
– Total comentarios: 37  
– Temas principales: puntualidad de la tripulación, amabilidad, avión nuevo, comodidad, calidad del servicio

✈️ RUTAS AFECTADAS (Top 5):  
– MAD-VIE: NPS 100.0 (n=3)  
– No hay mapeo de rutas con incidentes NCS (BRU-MAD, MAD-BRU) en el set de NPS del día

👥 PERFILES REACTIVOS:  
– Business: NPS 57.1 (n=14)  
– Leisure: NPS 50.0 (n=12)  
– Fleet A350 C: NPS 100.0 (n=1)  
– Fleet A319: NPS 100.0 (n=1)  
– Fleet A320: NPS 16.7 (n=6)  
– Resto de flota con NPS ≥ 80.0 (n datos no completos)  
– CodeShare IB: NPS 50.0 (n=24)  
– CodeShare AA: NPS 100.0 (n=2)  

=== NMA: Global/SH/Economy/YW ===

📈 EXPLANATORY DRIVERS:  
– OTP15_adjusted subió 2.26 pts vs baseline → correlación directa con aumento de NPS  
– Load_Factor bajó 0.42 pts vs baseline → correlación inversa con aumento de NPS

📊 DATOS OPERATIVOS:  
– OTP15_adjusted subió 2.26 pts vs baseline  
– Load_Factor bajó 0.42 pts vs baseline

🚨 INCIDENTES NCS:  
– Incidentes NCS (n=17):  
  • 10 cancelaciones, 2 retrasos, 2 cambios de aeronave, 3 otros  
  • A pesar de estos incidentes, la mejora en puntualidad y menor carga de ocupación parece haber mitigado su impacto

💬 FEEDBACK DE CLIENTES:  
– Temas principales en verbatims (n=161):  
  • Enfoque en amabilidad y eficacia del personal  
  • No se mencionan quejas por retrasos, cancelaciones o handling

✈️ RUTAS AFECTADAS (Top 5):  
– MAD-MRS: NPS 0.0 con 2 encuestas  
– Otras rutas: no se dispone de datos de NPS por ruta con desviación significativa

👥 PERFILES REACTIVOS:  
– Información no disponible

=== NMA: Global ===

📈 EXPLANATORY DRIVERS:  
– OTP15 + 1.14 pts (vs baseline)  
– Mishandling – 1.52 pts  
– Load Factor – 1.18 pts

📊 DATOS OPERATIVOS:  
– OTP15 + 1.14 pts (vs baseline)  
– Mishandling – 1.52 pts  
– Load Factor – 1.18 pts

🚨 INCIDENTES NCS:  
– 173 incidentes totales detectados en el día  
  • 58 cancelaciones  
  • 24 retrasos  
  • 21 otras incidencias  
  • 16 vuelos afectados con 5 cambios de equipo y cancelaciones por restricciones

💬 FEEDBACK DE CLIENTES:  
– Verbatims del día (verbatims_tool, n=1 041): feedback mayoritariamente positivo (“buen servicio”, “comodidad”, “personal amable”), sin menciones a cancelaciones o retrasos

✈️ RUTAS AFECTADAS (Top 5):  
– MAD-OVD: NPS −20.0 (n=5)  
– BRU-MAD: NPS 15.0 (n=20)

👥 PERFILES REACTIVOS:  
– Leisure: NPS 33.9 (n=513)  
– Business/Work: NPS 32.9 (n=185)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 2025-11-25 hemos identificado cuatro factores clave que explican las variaciones de NPS. El NPS Global experimentó un incremento de 6.3 pts con respecto a la media de los últimos 7 días, impulsado por una combinación de acciones correctivas en red y dinámicas contrastantes entre cabinas y compañías.

La primera dinámica relevante fue el aumento de cancelaciones y retrasos ligado al aviso de huelga en rutas BRU-MAD y MAD-BRU. En LH Business, el NPS retrocedió a 19.23, 5.2 pts por debajo de la media de los últimos 7 días, sustentado por un empeoramiento de puntualidad (OTP de 74.95, 4.97 ppts según Explanatory Drivers) y 12 cancelaciones con 3 retrasos, pese a un feedback 100 % positivo de 44 comentarios. Entre los perfiles, Business/Work registró NPS –25.0 (8 encuestas) y Leisure 38.9 (18 encuestas), mientras la ruta EZE-MAD consiguió NPS 100.0 (2 encuestas). De manera paralela, en SH Economy la compañía IB vio su NPS descender a 33.23 (–0.6 pts) por 10 cancelaciones y 2 retrasos, afectando especialmente BRU-MAD (NPS 15.0, 20 encuestas) y DUS-MAD (NPS 20.0, 10 encuestas), con un NPS de 26.3 en Business/Work (95 encuestas).

En contraste, SH Business registró un NPS de 44.74, 9.4 pts por encima de la media de los últimos 7 días, liderado por IB que subió 19.0 pts hasta 53.85 gracias a una puntualidad mejorada en 1.61 ppts y una menor carga de ocupación (Load Factor –2.04 ppts), según métricas operativas. El feedback de 37 comentarios destacó puntualidad de la tripulación y comodidad, y la ruta MAD-VIE alcanzó NPS 100.0 (3 encuestas). Entre los perfiles más satisfechos figuraron Business con 57.1 (14 encuestas) y flota A350 C con 100.0 (1 encuesta).

La cabina SH Economy de YW aportó también un impulso positivo, con un NPS de 51.59, 17.1 pts por encima de la media de los últimos 7 días. Las métricas operativas muestran una mejora de puntualidad en 2.26 ppts y una reducción de ocupación en 0.42 ppts, y aun con 10 cancelaciones, 2 retrasos, 2 cambios de aeronave y 3 otras incidencias, el feedback de 161 verbatims valoró la amabilidad y eficacia del personal. La ruta MAD-MRS registró NPS 0.0 (2 encuestas).

Finalmente, la gestión global de incidentes y una ligera recuperación operativa impulsaron el NPS Global hasta 33.64. Se observó una mejora de puntualidad de 1.14 ppts, una reducción de mishandling en 1.52 ppts y una minoración de carga de ocupación en 1.18 ppts. A pesar de 58 cancelaciones, 24 retrasos y 21 otras incidencias, los 1 041 verbatims mostraron feedback mayoritariamente positivo (“buen servicio”, “comodidad”, “personal amable”). La ruta BRU-MAD alcanzó NPS 15.0 (20 encuestas) y MAD-OVD se mantuvo en –20.0 (5 encuestas), mientras Leisure y Business/Work se situaron en 33.9 (513 encuestas) y 32.9 (185 encuestas), respectivamente.

**DETALLE POR CABINA:**

ECONOMY SH: Equilibrio compensado  
La cabina Economy de SH registró un NPS de 38.53, 4.5 pts por encima de la media de los últimos 7 días. Desglose por compañía: IB obtuvo 33.23 (–0.6 pts) atenuado por 10 cancelaciones y 2 retrasos en BRU-MAD/DUS-MAD, mientras YW alcanzó 51.59 (+17.1 pts) gracias a una puntualidad 2.26 ppts superior y un Load Factor 0.42 ppts inferior.

BUSINESS SH: Impulso IB  
El segmento Business de SH reportó un NPS de 44.74, 9.4 pts por encima de la media de los últimos 7 días. Desglose por compañía: IB obtuvo 53.85 (+19.0 pts) con un OTP 1.61 ppts superior y carga de ocupación 2.04 ppts inferior, y YW se situó en 25.0 (–14.2 pts) impactada por un Load Factor 3.72 ppts inferior y 10 cancelaciones más 2 retrasos.

ECONOMY LH: Desempeño estable  
La cabina Economy de LH mantuvo desempeño estable con un NPS de 15.56, 3.4 pts por encima de la media de los últimos 7 días. No se detectaron cambios significativos en métricas operativas ni en feedback de clientes.

BUSINESS LH: Impacto de huelga  
La cabina Business de LH registró un NPS de 19.23, 5.2 pts por debajo de la media de los últimos 7 días, motivado por un empeoramiento de puntualidad de 4.97 ppts y 12 cancelaciones más 3 retrasos en rutas BRU-MAD/MAD-BRU. El feedback de 44 comentarios fue 100 % positivo, y la ruta EZE-MAD alcanzó NPS 100.0 (2 encuestas).

PREMIUM LH: Mejora inexplicable  
El segmento Premium de LH alcanzó un NPS de 38.89, 22.8 pts por encima de la media de los últimos 7 días. A pesar de una caída de puntualidad de 4.97 ppts y una reducción de Load Factor de 3.08 ppts, no se registraron menciones a incidencias en verbatims de 36 comentarios. La ruta JFK-MAD obtuvo NPS 100.0 (1 encuesta) y los perfiles más satisfechos fueron Business/Work con 66.7 (3 encuestas) y flota A350 con 50.0 (10 encuestas).

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 6
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*

---

📅 2025-11-24 to 2025-11-24:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Economy SH, el escenario es “Sin anomalías” (N, N | N).  
- Narrativa: No hay dinámicas de anomalía interna ni en los hijos ni en el padre, todos los nodos se mantienen dentro de rangos esperados.  
- Evidencia Clave:  
  • IB SH: NPS +1.0 pts (34.89 vs 33.87 baseline) – normal  
  • YW SH: NPS +0.2 pts (34.75 vs 34.51 baseline) – normal  
  • Economy SH: NPS +0.8 pts (34.84 vs 34.02 baseline) – normal  

En Business SH, el escenario es CANCELACIÓN (+, – | N).  
- Narrativa: Las anomalías opuestas de IB y YW se neutralizan en el agregado. Mientras SH/Business/IB experimentó una subida por mejoras operativas, SH/Business/YW sufrió una caída sin causa operativa clara, compensándose mutuamente y resultando en un nivel normal para Business SH.  
- Evidencia Clave:  
  • SH/Business/IB (+15.2 pts): aumento de puntualidad (OTP15_adjusted +1.82 pts) y menor ocupación (Load Factor –2.05 pts).  
  • SH/Business/YW (–15.7 pts): reportes de 6 cancelaciones y 6 retrasos (NCS), nivel de confianza bajo y sin correlación operativa confirmada.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul (SH), la dinámica es SINERGIA (+, + | +).  
- Narrativa: El ligero incremento de 1.0 pt en NPS SH se explica por la mejora simultánea en ambas cabinas. Economy SH y Business SH registraron percepciones más positivas de puntualidad y servicio, lo que sumó su efecto al nivel de corto radio.  
- Evidencia:  
  • Economy SH: NPS +0.8 pts (34.84 vs 34.02) – verbatims destacan puntualidad y cortesía.  
  • Business SH: NPS +4.2 pts (39.53 vs 35.35) – OTP15_adjusted +1.82 pts correlacionado con mejor experiencia.  

En Long Haul (LH), la dinámica es DOMINANCIA (+, –, N | +).  
- Narrativa: El alza de 8.4 pts en NPS LH está dictada por la fuerte subida de Economy LH. La cabina Economy (+11.3 pts), potenciada por feedback de confort y servicio amable en la ruta LIM–MAD, impuso su efecto, aunque fue parcialmente mitigada por la caída de Business LH y la estabilidad de Premium LH.  
- Evidencia:  
  • Economy LH: NPS +11.3 pts (23.53 vs 12.20) – verbatims on-board resaltan confort y servicio amable (ruta LIM–MAD).  
  • Business LH: NPS –8.8 pts (15.62 vs 24.44) – cancelaciones/retrasos.  
  • Premium LH: NPS +2.1 pts (18.18 vs 16.08) – desempeño estable.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es SINERGIA (LH +8.4 pts, SH +1.0 pts | GLOBAL +4.7 pts).  
- Narrativa: El incremento de 4.7 pts en el NPS Global se explica por el empuje simultáneo de ambos radios. Aunque ninguna métrica operativa de Global supera el umbral de 3 pts (OTP15_adjusted +1.15 pts (Global); Load Factor –1.21 pts (Global)), el feedback cualitativo a nivel red revela una mejora generalizada en puntualidad, cortesía de la tripulación y confort a bordo.  
- Evidencia:  
  • NPS Global: 32.03 vs baseline 27.37 (+4.7 pts)  
  • OTP15_adjusted +1.15 pts (Global)  
  • Load Factor –1.21 pts (Global)  
  • Verbatims (896 comentarios): temas principales puntualidad, cortesía y confort.

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

CAUSA 1: Mejora generalizada en puntualidad, cortesía y confort  
- Escenario: SINERGIA (LH +8.4 pts, SH +1.0 pts | Global +4.7 pts)  
- NMA: Global  
- Afecta a: Global/LH y Global/SH (todos los radios)  
- Tipo de impacto: Positivo  

CAUSA 2: Mejora de NPS en Short Haul Business/IB  
- Escenario: CANCELACIÓN (+15.2, –15.7 | N)  
- NMA: Global/SH/Business/IB  
- Afecta a: SH/Business/IB  
- Tipo de impacto: Positivo  

CAUSA 3: Caída de NPS en Short Haul Business/YW  
- Escenario: CANCELACIÓN (+15.2, –15.7 | N)  
- NMA: Global/SH/Business/YW  
- Afecta a: SH/Business/YW  
- Tipo de impacto: Negativo  

CAUSA 4: Incremento de NPS en Long Haul Economy  
- Escenario: DOMINANCIA (+11.3, –8.8, +2.1 | +8.4)  
- NMA: Global/LH/Economy  
- Afecta a: LH/Economy  
- Tipo de impacto: Positivo

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

=== NMA: Global ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
• Load Factor: 85.71 vs 86.92 (–1.21)  
• OTP15_adjusted: 89.09 vs 87.94 (+1.15)  
• Mishandling: 14.26 vs 16.08 (–1.82)  
• Misconex: 0.67 (sin dato de variación)

🚨 INCIDENTES NCS:  
254 totales (58 cancelaciones, 61 retrasos)

💬 FEEDBACK DE CLIENTES:  
Temas principales en verbatims (896 comentarios):  
 • Puntualidad  
 • Cortesía del personal a bordo  
 • Confort y servicio amable

✈️ RUTAS AFECTADAS (Top 5):  
• BRU-MAD: NPS 61.2 (n=15), desviación +33.8 pts vs baseline  
• GRX-MAD: NPS –40.0 (n=5), desviación –67.4 pts vs baseline

👥 PERFILES REACTIVOS:  
• Leisure: NPS 34.2 (483 encuestas)  
• Business/Work: NPS 26.6 (192 encuestas)  
• Fleet – 32S: NPS 100.0 (1 encuesta)  
• Fleet – A333: NPS 60.9 (23 encuestas)  
• Fleet – A321: NPS 44.9 (85 encuestas)  
• CodeShare – QR: NPS –40.0 (5 encuestas)

=== NMA: Global/SH/Business/IB ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
• OTP15_adjusted superior en 1.82 pts vs baseline  
• Load_Factor inferior en 2.05 pts vs baseline

🚨 INCIDENTES NCS:  
• Total incidentes: 17 (6 cancelaciones, 6 retrasos)  
• Vuelo BRU-MAD cancelado (huelga)  
• Vuelo en MUC afectado por meteorología

💬 FEEDBACK DE CLIENTES:  
Temas principales en verbatims (33 comentarios):  
 • “su personal de cabina fue un sueño”  
 • “fantástico servicio a bordo”  
 • “buena atención de la tripulación”

✈️ RUTAS AFECTADAS (Top 5):  
• BRU-MAD: NPS 0.0 (1 encuesta)

👥 PERFILES REACTIVOS:  
• Leisure: NPS 68.8 (16 encuestas)  
• Business/Work: NPS 20.0 (10 encuestas)  
• Flota A319: NPS 0.0 (1 encuesta)  
• Residence España: NPS 35.3 (17 encuestas)

=== NMA: Global/SH/Business/YW ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
• Load Factor: –3.69 pts vs baseline  
• OTP15: +2.29 pts vs baseline

🚨 INCIDENTES NCS:  
• 6 cancelaciones y 6 retrasos reportados (ncs_tool), incluidas rutas BRU-MAD y MAD-BRU por huelga  
• Incidentes NCS relevantes (17 totales):  
  – 6 cancelaciones (BRU-MAD, MAD-BRU por huelga)  
  – 6 retrasos

💬 FEEDBACK DE CLIENTES:  
Temas principales en verbatims (16 comentarios):  
 • Atención del personal  
 • Cortesía  
 • Puntualidad  
 • No hay menciones a cancelaciones, retrasos ni huelga

✈️ RUTAS AFECTADAS (Top 5):  
• SVQ-VLC: NPS 100 (n=1), sin relación con incidentes

👥 PERFILES REACTIVOS:  
• Business/Work: NPS 28.6 (7 encuestas)  
• Leisure: NPS 20.0 (10 encuestas)  
• Fleet CRJ: NPS 23.5 (17 encuestas)  
• Residence Region “Europa”: NPS –50.0 (4 encuestas)  
• CodeShare “LATAM”: NPS –100.0 (1 encuesta)

=== NMA: Global/LH/Economy ===

📈 EXPLANATORY DRIVERS:  
No disponible

📊 DATOS OPERATIVOS:  
• OTP15_adjusted∆ = –5.55 pts  
• Load_Factor se situó 3.51 pts por debajo del baseline

🚨 INCIDENTES NCS:  
• 14 cancelaciones registradas  
• 4 retrasos asociados a huelga y reprogramaciones

💬 FEEDBACK DE CLIENTES:  
• Total de comentarios: 268 (feedback centrado en confort y servicio amable)  
• No se mencionan retrasos, cancelaciones ni problemas de puntualidad

✈️ RUTAS AFECTADAS (Top 5):  
• LIM–MAD: NPS 29.4 (n=17 encuestas)

👥 PERFILES REACTIVOS:  
• Business/Work: NPS –21.2 (33 encuestas)  
• Leisure: NPS 34.3 (137 encuestas)  
• Flota A33ACMI: NPS –27.3 (11 encuestas)  
• CodeShare QR: NPS –40.0 (5 encuestas)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 2025-11-24, hemos identificado tres causas principales que explican las variaciones de NPS. El resultado global mostró una subida de 4.7 pts con respecto a la media de los últimos 7 días.

El punto de partida de esta mejora fue un impulso transversal en puntualidad, cortesía de la tripulación y confort a bordo, recogido en 896 comentarios. A nivel operativo, la red registró OTP de 89.09 vs 87.94 (+1.15 pts) y Load Factor de 85.71 vs 86.92 (–1.21 pts), sin que superen el umbral de 3 pts, mientras que incidentes operativos se mantuvieron en 254 totales (58 cancelaciones, 61 retrasos). Las rutas BRU–MAD (NPS 61.2, +33.8 pts) y GRX–MAD (NPS –40.0, –67.4 pts) mostraron las mayores desviaciones y los perfiles Leisure (NPS 34.2) y Business/Work (NPS 26.6) fueron los más activos. Este efecto conjunto en SH y LH impulsó el NPS Global a 32.03.

En SH Business se observó una dinámica opuesta que se compensó en el agregado: IB elevó su NPS a 50.0 (+15.2 pts) gracias a un OTP de +1.82 pts y Load Factor –2.05 pts, con 6 cancelaciones y 6 retrasos que no empañaron el feedback (“su personal de cabina fue un sueño”), mientras que YW descendió a 23.53 (–15.7 pts) tras 6 cancelaciones y 6 retrasos, pese a menciones positivas en atención y puntualidad. La ruta BRU–MAD tuvo NPS 0.0 (1 encuesta) y SVQ–VLC NPS 100 (1 encuesta); Leisure alcanzó 68.8 y CodeShare LATAM registró –100.0. Estos efectos opuestos se neutralizaron, resultando en un Business SH de 39.53 (+4.2 pts).

En LH Economy, el NPS subió a 23.53 (+11.3 pts) gracias a 268 comentarios centrados en confort y servicio amable, a pesar de una caída de OTP en 5.55 pts, Load Factor –3.51 pts y 14 cancelaciones con 4 retrasos. La ruta LIM–MAD marcó NPS 29.4 (17 encuestas), Leisure llegó a 34.3 y Business/Work cayó a –21.2; la flota A33ACMI registró –27.3 y CodeShare QR –40.0. Este fuerte impulso de Economy LH, junto con el mantenimiento de Premium LH (NPS 18.18, +2.1 pts) y pese al descenso de Business LH (NPS 15.62, –8.8 pts), explica el alza de +8.4 pts en LH.

**DETALLE POR CABINA:**

ECONOMY SH: Desempeño consolidado  
La cabina Economy de SH registró un NPS de 34.84 con 0.8 pts de mejora con respecto a la media de los últimos 7 días. Desglose por compañía: IB obtuvo 34.89 (+1.0 pts) y YW 34.75 (+0.2 pts). Ambas respondieron positivamente en puntualidad y cortesía, sumando su efecto al nivel general de Economy SH.

BUSINESS SH: Dicotomía compensada  
El segmento Business de SH registró un NPS de 39.53 con 4.2 pts de mejora con respecto a la media de los últimos 7 días. IB alcanzó 50.0 (+15.2 pts) apalancado en OTP de +1.82 pts, Load Factor –2.05 pts y un feedback excepcional (“fantástico servicio a bordo”), pese a 6 cancelaciones y 6 retrasos. YW descendió a 23.53 (–15.7 pts) tras 6 cancelaciones y 6 retrasos, aunque el feedback resaltó la atención y puntualidad. Las rutas BRU–MAD (NPS 0.0) y SVQ–VLC (NPS 100) ilustran esta compensación interna.

ECONOMY LH: Alto impacto positivo  
La cabina Economy de LH alcanzó un NPS de 23.53 con 11.3 pts de mejora con respecto a la media de los últimos 7 días. El alza se sustentó en 268 comentarios sobre confort y servicio amable, pese a un OTP de –5.55 pts, Load Factor –3.51 pts y 14 cancelaciones con 4 retrasos. La ruta LIM–MAD registró NPS 29.4, mientras el perfil Leisure llegó a 34.3 y Business/Work cayó a –21.2; la flota A33ACMI marcó –27.3 y CodeShare QR –40.0.

BUSINESS LH: Deterioro significativo  
La cabina Business de LH registró un NPS de 15.62 con 8.8 pts de caída con respecto a la media de los últimos 7 días. Esta merma se vinculó a 14 cancelaciones y 4 retrasos por huelga y reprogramaciones, reflejados en una reducción de OTP en 5.55 pts. El feedback no registró menciones a demoras (50 comentarios) y la ruta MAD–SCL alcanzó NPS 66.7, con clientes Business/Work en 5.6.

PREMIUM LH: Rendimiento estable  
El segmento Premium de LH mantuvo desempeño estable con un NPS de 18.18 y 2.1 pts de mejora con respecto a la media de los últimos 7 días. No se detectaron cambios significativos, sosteniendo niveles consistentes de satisfacción.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 6
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*

**INSTRUCCIONES:**
1. Para cada día, evalúa si tiene variaciones significativas (>±5 pts NPS)
2. Identifica incidentes operativos destacables
3. Marca los días SIN datos relevantes como "OMITIR"

**FORMATO DE RESPUESTA:**
Para cada día relevante, responde con:
```
📅 [FECHA]
- Cabinas afectadas: [lista]
- Variación más significativa: [cabina] [+/-X.X pts]
- Razón principal: [causa breve]
- Rutas destacadas: [si las hay]
```

Para días sin relevancia:
```
📅 [FECHA]: OMITIR - [razón breve]
```


OUTPUT:
📅 2025-11-30  
- Cabinas afectadas: Economy LH, Business LH  
- Variación más significativa: Economy LH –26.4 pts  
- Razón principal: 32 incidentes operativos en Long Haul (12 cancelaciones, 6 retrasos, 3 mishandlings de equipaje, 2 cambios de aeronave) y caso crítico SDQ–MAD (159 maletas retenidas, 10 pax desembarcados)  
- Rutas destacadas: EZE–MAD (NPS –15.1)

📅 2025-11-29  
- Cabinas afectadas: Premium LH, Business SH  
- Variación más significativa: Premium LH +55.4 pts  
- Razón principal: pico excepcional de satisfacción en Premium LH pese a 4 cancelaciones, 5 retrasos y limitaciones de aeronave; verbatims destacan calidad de servicio  
- Rutas destacadas: BOG–MAD (NPS 100)

📅 2025-11-28  
- Cabinas afectadas: Economy SH (YW), Business SH (IB & YW), Economy LH  
- Variación más significativa: Business SH / YW –49.2 pts  
- Razón principal: cancelaciones y retrasos por huelga en Italia en BLQ–MAD y MAD–MXP que impactan a YW; IB contrarresta con +15.2 pts y Economy LH cae –10.4 pts  
- Rutas destacadas: BLQ–MAD (NPS –100), MAD–MXP (afectado por huelga)

📅 2025-11-27  
- Cabinas afectadas: Business SH (IB), Economy LH, Business LH, Premium LH  
- Variación más significativa: Business LH +18.4 pts  
- Razón principal: fuerte feedback de servicio en Business LH y recuperación en Economy LH pese a huelga en Italia; Premium LH penalizada por cancelaciones y retrasos  
- Rutas destacadas: MAD–SJO (Business LH NPS 100)

📅 2025-11-26  
- Cabinas afectadas: Economy SH (YW), Business SH, Business LH, Premium LH  
- Variación más significativa: Premium LH +42.3 pts  
- Razón principal: menor ocupación y verbatims muy positivos en Premium LH; mejoras operativas y satisfacción en Business LH y SH / YW  
- Rutas destacadas: MAD–SCL (NPS 100)

📅 2025-11-25  
- Cabinas afectadas: Economy SH (YW), Business SH, Business LH, Premium LH  
- Variación más significativa: Premium LH +22.8 pts  
- Razón principal: alta satisfacción en Premium LH pese a incidencias; puntualidad mejorada en Economy SH / YW y gestión de cancelaciones en SH Business  
- Rutas destacadas: JFK–MAD (NPS 100), EZE–MAD (Business LH NPS 100)

📅 2025-11-24  
- Cabinas afectadas: Business SH (IB & YW), Economy LH  
- Variación más significativa: Business SH / YW –15.7 pts  
- Razón principal: dinámica opuesta en SH Business (IB +15.2 pts vs YW –15.7 pts) y fuerte alza en Economy LH (+11.3 pts) por confort y servicio amable  
- Rutas destacadas: GRX–MAD (NPS –40.0), BRU–MAD (NPS +61.2), LIM–MAD (Economy LH NPS 29.4)

===== STEP 2: EXTRACT DAILY DATA =====

OUTPUT:
📅 2025-11-30 – Economy LH  
• Variación vs baseline: –26.4 pts  
• Causa: 32 incidentes operativos en Long Haul (12 cancelaciones, 6 retrasos, 3 mishandlings de equipaje, 2 cambios de aeronave) y caso crítico SDQ–MAD (159 maletas retenidas, 10 pax desembarcados)  
• Rutas: EZE–MAD (NPS –15.1)  
• Perfiles reactivos: Business/Work (NPS –0.1, 58 encuestas), Leisure (NPS –11.8, 261 encuestas)  

📅 2025-11-30 – Business LH  
• (No hay dato de variación exacta vs baseline)  
• Causa: mismo incidente de Long Haul que impactó a Economy LH  
• Rutas: EZE–MAD (NPS –15.1)  
• Perfiles reactivos: Business/Work (NPS 52.9, 17 encuestas), Leisure (NPS –12.5, 16 encuestas)  

📅 2025-11-29 – Premium LH  
• Variación vs baseline: +55.4 pts  
• Causa: pico excepcional de satisfacción pese a 4 cancelaciones, 5 retrasos y limitaciones de aeronave; verbatims destacan calidad de servicio  
• Rutas: BOG–MAD (NPS 100)  
• Perfiles reactivos: Business/Work (NPS 100, 1 encuesta), Leisure (NPS 66.7, 7 encuestas)  

📅 2025-11-29 – Business SH  
• Variación vs baseline: +21.1 pts  
• Causa: alta valoración de puntualidad y atención de tripulación en Business SH pese a algunas cancelaciones y retrasos  
• Rutas: LHR–MAD (NPS 100)  
• Perfiles reactivos: Leisure (NPS 76.5, 18 encuestas), Business/Work (NPS 70.0, 10 encuestas)  

📅 2025-11-28 – Economy SH (YW)  
• Variación vs baseline: +14.5 pts  
• Causa: cancelaciones y retrasos por huelga en Italia (BLQ–MAD, MAD–MXP)  
• Rutas: BLQ–MAD (NPS –100), MAD–MXP (afectada por huelga)  
• Perfiles reactivos: Leisure (NPS 56.4, 102 encuestas), Flota ATR (NPS 73.9, 23 encuestas)  

📅 2025-11-28 – Business SH  
• Variación vs baseline: –49.2 pts  
• Causa: cancelaciones y retrasos por huelga en Italia en rutas BLQ–MAD y MAD–MXP  
• Rutas: BLQ–MAD (NPS –100), MAD–MXP  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-28 – Economy LH  
• Variación vs baseline: –10.4 pts  
• Causa: cancelaciones y retrasos por huelga en Italia en rutas BLQ–MAD y MAD–MXP  
• Rutas: BLQ–MAD (NPS –100), MAD–MXP  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-27 – Business LH  
• Variación vs baseline: +18.4 pts  
• Causa: fuerte feedback de servicio a bordo en Business LH y recuperación de Economy LH pese a huelga en Italia; Premium LH penalizada por cancelaciones y retrasos  
• Rutas: MAD–SJO (NPS 100)  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-27 – Economy LH  
• (No hay dato de variación exacta vs baseline)  
• Causa: recuperación de satisfacción pese a huelga en Italia  
• Rutas: MAD–SJO (NPS 100)  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-27 – Business SH  
• (No hay dato de variación exacta vs baseline)  
• Causa: recuperación de la cabina Business SH impulsada por la valoración de servicio de tripulación  
• Rutas: MAD–SJO (NPS 100)  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-27 – Premium LH  
• (No hay dato de variación exacta vs baseline)  
• Causa: penalización por cancelaciones y retrasos en Premium LH  
• Rutas: MAD–SJO (NPS 100)  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-26 – Premium LH  
• Variación vs baseline: +42.3 pts  
• Causa: menor ocupación y verbatims muy positivos en Premium LH; mejoras operativas y satisfacción en Business LH y Economy SH/YW  
• Rutas: MAD–SCL (NPS 100)  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-26 – Business LH  
• (No hay dato de variación exacta vs baseline)  
• Causa: mejoras operativas en Business LH y efecto positivo de Premium LH  
• Rutas: MAD–SCL (NPS 100)  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-26 – Business SH  
• (No hay dato de variación exacta vs baseline)  
• Causa: mejoras operativas y satisfacción en Business SH  
• Rutas: MAD–SCL (NPS 100)  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-26 – Economy SH (YW)  
• (No hay dato de variación exacta vs baseline)  
• Causa: alta satisfacción en Economy SH/YW  
• Rutas: MAD–SCL (NPS 100)  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-25 – Premium LH  
• Variación vs baseline: +22.8 pts  
• Causa: alta satisfacción en Premium LH pese a incidencias; puntualidad mejorada en Economy SH/YW y gestión de cancelaciones en SH Business  
• Rutas: JFK–MAD (NPS 100), EZE–MAD (Business LH NPS 100)  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-25 – Business LH  
• (No hay dato de variación exacta vs baseline)  
• Causa: alta valoración en Business LH y buena gestión de cancelaciones  
• Rutas: EZE–MAD (NPS 100)  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-25 – Business SH  
• (No hay dato de variación exacta vs baseline)  
• Causa: gestión efectiva de cancelaciones y retrasos en SH Business  
• Rutas: JFK–MAD (NPS 100)  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-25 – Economy SH (YW)  
• (No hay dato de variación exacta vs baseline)  
• Causa: puntualidad mejorada en Economy SH/YW  
• Rutas: JFK–MAD (NPS 100)  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-24 – Business SH (IB & YW)  
• Variación vs baseline: Business SH/IB +15.2 pts, Business SH/YW –15.7 pts  
• Causa: dinámica opuesta en SH Business (IB +15.2 pts vs YW –15.7 pts)  
• Rutas: GRX–MAD (NPS –40.0), BRU–MAD (NPS +61.2)  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-24 – Economy LH  
• Variación vs baseline: +11.3 pts  
• Causa: fuerte alza por confort y servicio amable en Economy LH  
• Rutas: LIM–MAD (NPS 29.4)  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-24 – Economy SH  
• (No hay dato de variación exacta vs baseline)  
• Causa: estabilidad sin anomalías en Economy SH  
• Rutas: GRX–MAD (NPS –40.0), BRU–MAD (NPS +61.2)  
• Perfiles reactivos: (no disponibles)  

📅 2025-11-24 – Business SH  
• (Ver Business SH/IB & YW arriba)  
• Perfiles excluidos para evitar duplicación  

📅 2025-11-24 – Premium LH  
• (No hubo cambio significativo)  
• Causa: desempeño estable en Premium LH  
• Rutas: (no hay rutas destacadas con incidencia)  
• Perfiles reactivos: (no disponibles)  

===== STEP 3: FINAL REPORT =====

**📈 Análisis semana del (2025-11-24 al 2025-11-30) con respecto a la semana anterior (Global):**

📊 **DIAGNÓSTICO A NIVEL DE EMPRESA**

Economy SH  
- Escenario: No aplica anomalía interna, ambos hijos estables (IB N, YW N | Economy N).

• **2025-11-24**: estabilidad en **Economy SH** sin cambios significativos en NPS, con rutas como GRX–MAD (NPS –40.0) y BRU–MAD (NPS +61.2).  
• **2025-11-28**: repunte de **Economy SH/YW** +14.5 pts vs baseline debido a cancelaciones y retrasos por huelga en Italia en BLQ–MAD y MAD–MXP, con Leisure (NPS 56.4, 102 encuestas) y Flota ATR (NPS 73.9, 23 encuestas) más reactivos.  


Business SH  
- Escenario: CANCELACIÓN (+,– | N)  
- Narrativa: Mientras la cabina IB obtuvo un alza de +12.7 pts impulsada por la mejora en puntualidad (reducción de 23 retrasos y 16 cancelaciones; Punctuality SHAP=4.266 en Business SH/IB; OTP15 taxi 0.97 vs 0.96), la cabina YW sufrió una caída de –22.5 pts debido al incremento de mishandling en equipaje (+0.7 eventos; Operative_data_tool en Business SH/YW) y fuertes impactos de producto (Aircraft interior SHAP=–6.757; Boarding SHAP=–3.229 en Business SH/YW). Estos efectos opuestos se neutralizaron, dejando el nodo padre en rango normal.  
- Evidencia clave:  
  • Business SH/IB – Punctuality SHAP=4.266 (Sat_diff=3.487) validado por –23 delays y –16 cancellations  
  • Business SH/YW – Mishandling equipaje 11.84 vs 11.15 (+0.7) y Aircraft interior SHAP=–6.757

• **2025-11-24**: choque de dinámicas en **Business SH**, con IB +15.2 pts vs YW –15.7 pts en NPS en rutas GRX–MAD y BRU–MAD.  
• **2025-11-28**: caída de **Business SH** –49.2 pts vs baseline por cancelaciones y retrasos por huelga en Italia en BLQ–MAD y MAD–MXP.  
• **2025-11-29**: pico de **Business SH** +21.1 pts vs baseline gracias a alta valoración de puntualidad y atención de tripulación, destacando LHR–MAD (NPS 100.0), con Leisure (NPS 76.5, 18 encuestas) y Business/Work (NPS 70.0, 10 encuestas) más reactivos.  


💺 **DIAGNÓSTICO A NIVEL DE CABINA**

En Long Haul, la dinámica es SINERGIA (Economy –, Business –, Premium – | Long Haul –).  
- Narrativa: La caída de NPS en Long Haul (–5.7 pts) responde a un impacto sistémico que afectó por igual a las tres cabinas: un aumento de cancelaciones (+17 incidentes) y limitaciones de aeronave (+4 incidentes) deterioró la puntualidad (Punctuality SHAP = –1.097), y se sumaron deficiencias en el servicio a bordo (In flight food and beverage SHAP = –0.829; Cabin Crew SHAP = –0.667).

• **2025-11-24**: **Economy LH** subió +11.3 pts vs baseline por confort y servicio amable en LIM–MAD (NPS 29.4).  
• **2025-11-25**: **Premium LH** ganó +22.8 pts vs baseline pese a incidencias, destacando JFK–MAD (NPS 100.0).  
• **2025-11-26**: **Premium LH** repuntó +42.3 pts vs baseline con menor ocupación y verbatims muy positivos en MAD–SCL (NPS 100.0).  
• **2025-11-27**: **Business LH** experimentó +18.4 pts vs baseline por fuerte feedback de servicio a bordo en MAD–SJO (NPS 100.0).  
• **2025-11-28**: **Economy LH** cayó –10.4 pts vs baseline por huelga en Italia en BLQ–MAD (NPS –100.0) y MAD–MXP.  
• **2025-11-29**: **Premium LH** alcanzó +55.4 pts vs baseline pese a 4 cancelaciones y 5 retrasos, con BOG–MAD (NPS 100.0).  
• **2025-11-30**: **Economy LH** se hundió –26.4 pts vs baseline tras 12 cancelaciones, 6 retrasos y caso crítico SDQ–MAD (159 maletas retenidas), con EZE–MAD (NPS –15.1) y Business/Work (NPS –0.1, 58 encuestas) y Leisure (NPS –11.8, 261 encuestas) más sensibles.  


🌎 **DIAGNÓSTICO GLOBAL POR RADIO**

A nivel GLOBAL, la dinámica es DOMINANCIA (Long Haul –, Short Haul + | Global +).  
- Narrativa: El alza global de +0.8 pts responde fundamentalmente al buen desempeño de Short Haul, donde la mejora en puntualidad (OTP15 +0.6, retrasos –126, cancelaciones –48) generó un NPS +2.8 pts, superando la caída de Long Haul.

• **2025-11-28**: **Short Haul** se benefició del +14.5 pts vs baseline en **Economy SH/YW**, superando la presión en LH.  
• **2025-11-29**: **Short Haul** impulsó +21.1 pts en **Business SH** gracias a puntualidad y atención de tripulación.  
• **2025-11-30**: **Short Haul** cerró estable compensando la fuerte caída de **Long Haul** en Economy LH (–26.4 pts).