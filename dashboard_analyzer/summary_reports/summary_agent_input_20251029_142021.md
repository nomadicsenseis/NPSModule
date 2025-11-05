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
Tanto IB (–11,1 pts) como YW (–5,8 pts) presentan anomalías con drivers operativos coincidentes (puntualidad e incidentes) y respaldo en NCS, por lo que la causa es general a la cabina Economy SH.  

B. Business Short Haul  
Solo IB muestra anomalía (–10,7 pts) mientras YW permanece normal (+0,3 pts); la causa es específica de la compañía IB en la cabina Business SH.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul (SH):  
El deterioro operativo en puntualidad y misconexiones afecta a toda la radio SH, pero la reactividad es específica de cabina/compañía: tanto Economy SH (IB y YW) como Business IB sufren caídas de NPS, mientras que Business YW permanece estable, actuando como amortiguador.

Long Haul (LH):  
La causa operativa (puntualidad) es común a todas las cabinas LH, sin embargo Premium LH reacciona con mucha mayor intensidad (–13,5 pts) y Economy/Business amortiguan el impacto (ambas alrededor de –2,7 pts), mostrando sensibilidad diferencial por clase.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Ambos radios (Short Haul y Long Haul) están afectados y convergen en el mismo driver operativo principal (deterioro de puntualidad con impactos adicionales de misconexiones y cancelaciones), con valores SHAP comparables (SH ≈ –5.19; LH ≈ –4.93) y evidencia operativa coherente (OTP15, NCS). El nodo Global refleja un impacto agregado coherente (–7 pts), sin efectos compensatorios entre radios, y amplifica la caída detectada, mostrando la naturaleza homogénea y de alcance compañía de la causa.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

Causa 1: Deterioro de la puntualidad operacional  
A. Naturaleza de la causa  
• Empeoramiento de la puntualidad de las operaciones (retrasos, cancelaciones, desvíos, limitaciones de aeronave) que erosiona la percepción de fiabilidad de la compañía.  

B. Evidencia consolidada y alcance  
• Segmento más grande: Global (NPS actual 22,05 vs 29,06; variación –7,00 pts)  
• Afecta a todos los subsegmentos bajo Global (Long Haul, Short Haul, y todas sus cabinas)  
• Indicadores operativos:  
  – SHAP Punctuality = –4,944  
  – OTP15 cayó 1,3 puntos (de últimos 7 días)  
  – NCS totales +133 incidentes (Otras +98; Limitación aeronave +23; Desvíos +13; Cancelaciones +6)  
• Rutas involucradas (muestra >50 pax): LHR–MAD (176 pax), MAD–MEX (156), MAD–MIA (100), LIM–MAD (102); clave MAD–VGO  
• Verbatims representativos: “retrasos excesivos”, “no recibimos información en tiempo real”  

Causa 2: Gestión deficiente de conexiones y cancelaciones  
A. Naturaleza de la causa  
• Aumento de misconexiones y cancelaciones sin apoyo ni soluciones ágiles, generando costes y frustración en itinerarios con conexiones.  

B. Evidencia consolidada y alcance  
• Segmento más grande: Global / Long Haul (NPS actual 18,39 vs 21,82; variación –3,43 pts)  
• Afecta a todos los subsegmentos bajo Global / Long Haul (Economy, Business, Premium)  
• Indicadores operativos:  
  – SHAP Punctuality = –4,927; Sat_diff Punctuality = –6,48  
  – Retrasos +21 incidentes; Desvíos +5; Cancelaciones +9 (ncs_tool)  
  – OTP15 LH cayó 6,48 puntos  
• Rutas involucradas (pax >30): DOH–MAD (31 pax, NPS –58,1), JFK–MAD (94, –2,1), DFW–MAD (32, –37,5), MAD–MVD (44, –15,9)  
• Verbatims representativos: “conexiones perdidas y costes extra”, “cambios de vuelo sin aviso previo”  

Causa 3: Falta de comunicación e información al pasajero  
A. Naturaleza de la causa  
• Deficiencias en la comunicación ante incidencias operativas, intensificando la insatisfacción y la sensación de abandono.  

B. Evidencia consolidada y alcance  
• Segmento más grande: Global (NPS actual 22,05 vs 29,06; variación –7,00 pts)  
• Afecta a todos los subsegmentos bajo Global  
• Indicadores cualitativos:  
  – Verbatims: “mala comunicación”, “sin avisos de cambios”, “atención en tierra deficiente”  
• Impacto por perfil (NPS_diff vs L7d): CodeShare +59,9 pts; Residence Region +42,0 pts; Fleet +35,7 pts  
• Rutas más mencionadas: MAD–VGO, LHR–MAD, FRA–MAD, DOH–MAD, MAD–MEX

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
Durante la semana del 2025-10-19 al 2025-10-25 se registraron bajadas de NPS en 11 de los 12 segmentos analizados. A nivel Global, el NPS cayó de 29.06 a 22.05 (–7.00 pts vs L7d), reflejo de descensos en Long Haul (21.82→18.39, –3.43 pts) y Short Haul (32.88→23.97, –8.91 pts). En Long Haul, Premium sufrió la mayor pérdida (29.50→16.03, –13.47 pts), mientras Economy LH pasó de 20.21 a 17.49 (–2.72 pts) y Business LH de 29.09 a 26.40 (–2.69 pts). En Short Haul, Economy IB cayó de 27.80 a 16.72 (–11.09 pts) y Economy YW de 39.25 a 33.40 (–5.85 pts), con un NPS consolidado de 22.40 (–9.16 pts); Business IB bajó de 48.75 a 38.10 (–10.66 pts) mientras Business YW se mantuvo estable en 33.64 (+0.31 pts), dando un Business SH total de 36.74 (–7.32 pts). Estas bajadas se explican principalmente por un empeoramiento de la puntualidad (SHAP Punctuality entre –4.9 y –9.5), caída de OTP15 (hasta –6.48 p.p.), incremento de incidentes operativos (retrasos +21, cancelaciones +9, desvíos +5, limitaciones de aeronave +23) y deficiencias en la comunicación al pasajero.

Las rutas más impactadas incluyen DOH–MAD (NPS –58.1, 31 pax), DFW–MAD (–37.5, 32 pax), MAD–VGO (–11.1 en Economy IB, 60 pax) y MAD–MVD (–15.9, 44 pax). Los pasajeros en código compartido (CodeShare spread hasta 200.0 pts en Business LH y 176.2 pts en SH) y según región de residencia (hasta 93.3 pts en Premium LH) fueron los más reactivos, seguidos por la dimensión Fleet.

ECONOMY SH: Deterioro generalizado por puntualidad  
La cabina Global/SH/Economy IB registró un NPS de 16.72 (2025-10-19 a 2025-10-25) tras caer de 27.80 (–11.09 pts vs L7d), y Economy SH YW bajó de 39.25 a 33.40 (–5.85 pts vs L7d), dejando un NPS consolidado de 22.40 (–9.16 pts vs L7d). La causa principal fue el empeoramiento de la puntualidad (SHAP Punctuality –3.84), validado con una caída de OTP15 de 80.94 a 77.31 (–3.63 p.p.) y 21 incidentes adicionales de retrasos, complementada por un aumento de misconexiones (0.51→0.65, +0.14 p.p.) y cancelaciones (+9). Esta bajada se reflejó especialmente en DOH–MAD (–67.9 pts, 28 pax) y MAD–MVD (–13.2 pts, 38 pax), y los pasajeros CodeShare (spread 145.2 pts) y por región de residencia (37.0 pts) mostraron mayor sensibilidad.

BUSINESS SH: Impacto concentrado en IB  
El segmento Global/SH/Business IB cayó de 48.75 a 38.10 (–10.66 pts vs L7d), mientras Business SH YW se mantuvo estable en 33.64 (+0.31 pts vs L7d), resultando un NPS total de 36.74 (–7.32 pts vs L7d). Esta evolución se explica principalmente por la pérdida de puntualidad (SHAP Punctuality –3.54, OTP15 89.41→88.44, –1.0 p.p.), un aumento de incidentes de retrasos (–13 en este subsegmento) y desvíos (+3), y la persistencia de “otras incidencias” (+17). Las rutas GRX–MAD (–20.0, 5 pax) y BRU–MAD (–9.1, 11 pax) fueron las más afectadas, y los viajeros CodeShare (spread 143.8 pts) y por región de residencia (118.3 pts) reaccionaron con mayor intensidad.

ECONOMY LH: Caída moderada por retrasos y cancelaciones  
La cabina Global/LH/Economy pasó de un NPS de 20.21 a 17.49 (–2.72 pts vs L7d). La causa principal fue el deterioro operativo en puntualidad (SHAP Punctuality –3.84), con OTP15 descendiendo de 80.94 a 77.31 (–3.63 p.p.) y 21 incidentes adicionales de retrasos, junto con 9 cancelaciones y 5 desvíos. DOH–MAD (–58.1, 31 pax) y DFW–MAD (–37.5, 32 pax) lideraron las caídas, y los pasajeros CodeShare (spread 110.9 pts) mostraron mayor reactividad.

BUSINESS LH: Sensibilidad homogénea  
Global/LH/Business descendió de 29.09 a 26.40 (–2.69 pts vs L7d), explicado por puntualidad (SHAP –4.20), OTP15 –3.6 p.p., aumentos de misconexiones (+0.1 p.p.) y 9 cancelaciones. Rutas clave incluyen MAD–SJU (–75.0, 8 pax) y MAD–ORD (mencionada en varias fuentes), con CodeShare (200.0 pts) y Residence Region (90.0 pts) como perfiles más sensibles.

PREMIUM LH: Caída pronunciada por incidentes operativos  
Global/LH/Premium registró un NPS de 16.03 tras descender desde 29.50 (–13.47 pts vs L7d). El driver dominante fue puntualidad (SHAP –9.53), apoyado por un incremento de 105 incidentes (retrasos +21, cancelaciones +9, desvíos +5, limitaciones +5) y OTP15 deteriorado. Rutas críticas abarcan MAD–LIM, MAD–JFK y MAD–VGO, y el perfil Residence Region (93.3 pts) fue el más reactivo.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*



**ANÁLISIS DIARIO SINGLE:**
📅 2025-10-25 to 2025-10-25:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

A. Economy Short Haul  
- Existen ambos nodos (IB: –6.6 pts; YW: –1.9 pts)  
- Aunque difieren en magnitud, comparten el mismo patrón de incidencias no reflejadas en el feedback (cancelaciones/retrasos vs verbatims mayoritariamente positivos y muestreo parcial)  
→ Diagnóstico: causa general de cabina Economy SH, no específica de IB ni de YW.  

B. Business Short Haul  
- Existen ambos nodos (IB: +0.9 pts; YW: +5.9 pts), ambos dentro del rango normal  
→ Diagnóstico: no hay anomalía ni causa desviada ni a nivel de compañía ni de cabina.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

A. Short Haul (SH)  
• Economy SH y Business SH divergen claramente:  
  – Economy SH muestra una anomalía negativa (–5.2 pts), impulsada por cancelaciones, retrasos y cambios de aeronave no reflejados en el feedback.  
  – Business SH está dentro del rango normal, sin impacto relevante.  
• Dentro de Economy SH, ambos operadores (IB –6.6 pts y YW –1.9 pts) comparten el mismo patrón causal (incidencias operativas vs muestreo de feedback), pese a distinta magnitud. En Business SH IB y YW coinciden en mantener niveles normales.  
→ Diagnóstico SH: la causa es específica de la cabina Economy, mientras que Business actúa de “amortiguador” y no se ve afectada.  

B. Long Haul (LH)  
• Las tres cabinas reaccionan de modo divergente:  
  – Economy LH sufre una fuerte caída (–9.2 pts).  
  – Business LH presenta una subida muy elevada (+18.4 pts).  
  – Premium LH también sube significativamente (+9.3 pts).  
• No hay convergencia: cada clase responde de forma distinta a las mismas incidencias operativas.  
→ Diagnóstico LH: las causas son específicas de cada cabina; Economy es la más vulnerable, mientras que Business y Premium amortiguan o incluso invierten el impacto en satisfacción.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Ambos radios (SH y LH) están afectados por las mismas incidencias operativas no reflejadas en el feedback, pero con patrones divergentes en LH (Economy sufre fuerte caída mientras Business y Premium repuntan), de modo que en Global se observa un impacto mixto y compensatorio que se traduce en una anomalía negativa de –3.8 pts.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

Análisis profundo de las causas identificadas

Causa 1: Incidentes operativos frecuentes (retrasos, cancelaciones y cambios de aeronave)  
A. Naturaleza de la causa  
• Hipótesis: Un volumen elevado de disturbios operacionales genera frustración acumulada, especialmente en cabina Economy, reduciendo el NPS.  

B. Evidencia consolidada y alcance  
– Segmento seleccionado: Global/Short Haul / Economy (el mayor grupo de Economy afectados, n≈382 verbatims)  
– Afecta a todos los subsegmentos bajo Economy SH (IB y YW)  
– Indicadores operativos  
  • Total incidentes: 40  
    – Cancelaciones: 12  
    – Retrasos: 10  
    – Cambios de avión: 1  
    – Incidencias de equipaje: 1  
    – Otras: 11  
  • Vuelo más impactado: IB0921/24OCT/MAD-DSS con 40 maletas no cargadas  
– NPS  
  • Periodo: 16.79 pts vs baseline 22.03 pts (−5.24 pts)  
– Rutas involucradas  
  • MAD-DSS, MAD-GYE, MVD-MAD (1 incidencia cada una)  
  • Ruta crítica con muy mala valoración pero sin incidente NCS: MAD-VIE NPS −14.3 (n=7)  
– Feedback (382 verbatims)  
  • Tono mayoritariamente positivo: tripulación amable, puntualidad, comodidad  
  • Ausencia de menciones a cancelaciones o retrasos  
– Conclusión: Los problemas operativos están detrás de la caída de NPS en Economy SH, pero no llegan a traducirse en quejas escritas, lo que amplifica su efecto no mitigado.  

Causa 2: Percepción negativa en rutas puntuales sin registro de incidentes formales  
A. Naturaleza de la causa  
• Hipótesis: Deficiencias en servicio (horarios, confort, enlaces) impactan la experiencia de cliente en vuelos específicos, aunque no se refleje en NCS.  

B. Evidencia consolidada y alcance  
– Segmento seleccionado: Global (todos los radios y cabinas)  
– Afecta a la totalidad de nodos hijos (LH y SH, Economy/Business/Premium)  
– Indicadores operativos  
  • Total incidentes NCS: 407  
    – Retrasos: 86  
    – Cancelaciones: 46  
    – Cambios de aeronave: 19  
    – Problemas técnicos: 8  
    – Vuelos afectados: 48 (IB0273 con 4 incidentes)  
– NPS  
  • Periodo: 17.73 pts vs baseline 21.56 pts (−3.83 pts)  
– Rutas críticas con baja valoración y sin incidencias NCS:  
  • MAD-VCE: NPS −40.0 (n=5)  
  • GIG-MAD (LH): NPS 0.0 (n=9)  
  • HAM-MAD (SH): NPS −28.6 (n=7)  
– Feedback cualitativo: No disponible para vincular operativa con percepción  
– Conclusión: Hay segmentos de ruta donde la experiencia es mala pese a no figurar en registros de incidentes, lo que sugiere fallos de servicio o expectativas no cubiertas.  

Causa 3: Sesgo de cobertura en el feedback (clientes afectados no encuestados)  
A. Naturaleza de la causa  
• Hipótesis: Los pasajeros que sufren un incidente tienden a no responder encuesta, sesgando el muestreo hacia valoraciones más positivas y ocultando el alcance real de la insatisfacción.  

B. Evidencia consolidada y alcance  
– Segmento seleccionado: Global/SH/Economy (IB & YW)  
– Afecta igual a ambos operadores en Economy SH  
– Feedback (382 verbatims)  
  • Altamente positivo sin alusiones a cancelaciones, retrasos o equipaje  
  • Las rutas con más incidencias operativas no aparecen en el muestreo de verbatims  
– Conclusión: El sesgo de muestra impide capturar la voz de los clientes que vivieron los incidentes, ampliando el gap entre operativa real y percepción reportada.  

Causa 4: Reactividad diferencial de cabinas Business y Premium en Long Haul  
A. Naturaleza de la causa  
• Hipótesis: Clientes de alto valor (Business y Premium LH) ven mitigado el impacto de incidentes gracias a flotas de última generación y altos estándares de servicio, incluso invierten la tendencia negativa.  

B. Evidencia consolidada y alcance  
1) Business Long Haul  
   – Nodo: Global/LH/Business  
   – NPS: 48.0 pts vs baseline 29.57 pts (+18.4 pts)  
   – Incidentes: 50 (8 cancelaciones, 11 retrasos, 16 otras, 8 cambios de avión, 4 técnicos)  
   – Rutas con más reportes: SAL–SJO, MAD–DSS  
   – Feedback (principalmente positivos): atención de tripulación, puntualidad, embarque ordenado, wi-fi, oferta gastronómica  
   – Conclusión: A pesar de la operativa adversa, la satisfacción en Business LH crece, demostrando su rol de amortiguador.  

2) Premium Long Haul  
   – Nodo: Global/LH/Premium  
   – NPS: 23.53 pts vs baseline 14.25 pts (+9.3 pts)  
   – Incidentes: mismo conteo de 50 (11 retrasos, 8 cancelaciones, 1 desvío, 1 equipaje, 16 otras)  
   – Rutas muestreadas sin incidencia: EZE–MAD NPS 75.0 (n=4)  
   – Feedback (26 verbatims): comodidad, limpieza, atención, puntualidad  
   – Conclusión: Al igual que Business, Premium LH invierte el efecto de los incidentes, gracias a la selección de flotas y perfiles de cliente.  

Resumen  
Las principales causas de la anomalía negativa de NPS provienen de incidentes operativos concentrados en Economy (SH y LH) y de rutas puntuales con deficiencias de servicio no registradas en NCS. Estas causas se ven además amplificadas por un sesgo de muestreo de feedback, mientras que las cabinas de alto valor (Business y Premium LH) actúan como amortiguadores o generan anomalías positivas gracias a su resiliencia frente a las mismas incidencias.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El NPS global sufrió una caída de 3.83 puntos, pasando de 21.56 a 17.73, impulsada por el deterioro en cabina Economy tanto en Short Haul (–3.86 pts, de 23.80 a 19.94) como en Long Haul Economy (–9.20 pts, de 16.58 a 7.38). Estos retrocesos se explican por un pico de incidentes operativos (Short Haul: 12 cancelaciones y 10 retrasos; Long Haul: 8 cancelaciones, 11 retrasos y 8 cambios de aeronave) que no quedaron plasmados en los verbatims, generando frustración acumulada. En paralelo, las cabinas de alto valor en Long Haul revirtieron la tendencia: Business subió 18.43 puntos (de 29.57 a 48.00) y Premium escaló 9.28 puntos (de 14.25 a 23.53), gracias a la resiliencia de flotas A350 y una experiencia de servicio muy valorada.

Las rutas más afectadas incluyen MAD-DSS (40 equipajes no cargados), MAD-VIE (SH/Economy NPS –14.3 sin incidentes NCS), MAD-SJU (LH/Economy NPS –11.1) y GIG-MAD (LH/Economy NPS 0.0), donde fallos de horarios, confort y enlaces deterioraron la percepción sin quedar reflejados en NCS. Los grupos más reactivos fueron los business travelers de Economy SH en IB (bajada de 6.59 pts) y los residentes en América Norte y Asia en Economy (hasta –33.3 pts), así como pasajeros en vuelos AA (SH/Economy IB con NPS –66.7). En contraste, los usuarios de LH/Business y LH/Premium en flotas A350 next mostraron niveles de satisfacción superiores, impulsando las subidas de NPS.

ECONOMY SH: Caída concentrada en IB y atenuada en YW  
La cabina Economy de SH registró un NPS combinado de 16.79 el 25-10-2025, 5.24 puntos por debajo del promedio L7d. IB cayó a 9.50 (–6.59 vs L7d) y YW a 31.46 (–1.93 vs L7d). La causa principal fue el cúmulo de incidentes operativos (12 cancelaciones, 10 retrasos y cambios de aeronave en rutas como MAD-DSS) sin reflejo en feedback, unido a un sesgo de muestreo que dejó fuera a los pasajeros más afectados.

BUSINESS SH: Estabilidad pese a ligeras subidas  
La cabina Business de SH mantuvo desempeño estable con un NPS de 39.53 el 25-10-2025 (+1.97 vs L7d). IB alcanzó 38.71 (+0.87) y YW 41.67 (+5.94), sin cambios significativos en drivers operativos ni en verbatims, lo que confirma su papel de amortiguador frente a las incidencias que impactaron a Economy.

ECONOMY LH: Fuerte deterioro en Economy Long Haul  
La cabina Economy de LH sufrió un NPS de 7.38 el 25-10-2025, 9.20 puntos por debajo de los 16.58 del L7d. Este deterioro obedeció a 50 incidentes (8 cancelaciones, 11 retrasos, 8 cambios de aeronave) en rutas como IB243/24OCT MAD-SJO y MAD-SJU (NPS –11.1), sin correlación en verbatims, lo que sugiere problemas de servicio en horarios y confort.

BUSINESS LH: Empuje positivo en Business Long Haul  
La cabina Business de LH escaló su NPS a 48.00 el 25-10-2025, 18.43 puntos por encima de los 29.57 del L7d. A pesar de los 50 incidentes operativos, el feedback destacó atención de tripulación, puntualidad, wi-fi y oferta gastronómica en rutas SAL-SJO y MAD-DSS, revirtiendo el impacto de los inconvenientes.

PREMIUM LH: Recuperación en Premium Long Haul  
La cabina Premium de LH alcanzó un NPS de 23.53 el 25-10-2025, con una mejora de 9.28 puntos respecto a los 14.25 del L7d. Los 26 verbatims de rutas como EZE-MAD (NPS 75.0) resaltaron comodidad, limpieza y atención, amortiguando los efectos de los 50 incidentes operativos.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-10-24 to 2025-10-24:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

A. Economy Short Haul  
• Existen ambos nodos, SH/Economy/IB (–27.7 pts) y SH/Economy/YW (–11.9 pts).  
• Ambos muestran el mismo driver raíz: ola de cancelaciones y retrasos (caso IB273 MAD–HAV) no captada en verbatims, y rutas con NPS muy bajo sin incidentes formales (CDG–MAD vs MAD–MXP).  
• Diagnóstico: causa general a la cabina Economy SH, no específica de IB o YW.

B. Business Short Haul  
• Existen ambos nodos, SH/Business/IB (–18.8 pts) y SH/Business/YW (–20.7 pts).  
• Ambos convergen en patrones: ausencia de drivers operativos en SH/Business y caída explicada por perfil de cliente y región, más que por servicio o incidentes.  
• Diagnóstico: causa general a la cabina Business SH, no específica de IB o YW.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

A. Short Haul  
Economy SH (–21.3 pts) y Business SH (–19.8 pts) presentan ambas anomalías negativas de magnitud similar y comparten los mismos drivers operativos (ola de cancelaciones, retrasos, fallos en embarque y equipaje). El comportamiento IB vs YW es coherente en las dos cabinas.  
Diagnóstico SH: causa común a todo el radio Short Haul, sin cabina atenuadora.

B. Long Haul  
Economy LH (–5.9 pts) y Premium LH (–39.3 pts) caen en NPS, mientras que Business LH registra un alza de +20.4 pts. La respuesta a las disrupciones operativas varía claramente por clase: Premium es la más reactiva a la baja, Economy muestra un impacto menor y Business actúa como amortiguador (incluso positivo).  
Diagnóstico LH: patrón específico de cabina, con Business divergente frente a Economy y Premium.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A. Comparación Entre Radios  
– Ambos radios presentan anomalías: Short Haul con caída pronunciada (–20.9 pts) y Long Haul con patrón mixto (Economy y Premium caen, Business sube).  
– Los drivers de SH (cancelaciones, retrasos, fallos en embarque/equipaje y comunicación) se repiten en Economy y Business SH, pero en LH el impacto operativo es amortiguado en Business por una muy buena valoración del servicio a bordo.  
– Conclusión: ambos radios están afectados, pero con patrones causales divergentes entre SH (solo drivers operativos) y LH (drivers operativos vs. calidad de servicio diferencial).

B. Coherencia con Nodo Global  
– El Global refleja correctamente un impacto agregado negativo (–16.0 pts), resultado de combinar la fuerte caída de SH con el mix de LH (dos cabinas a la baja y una al alza).  
– No hay anulación total de efectos: el positivo de Business LH atenúa pero no compensa las caídas de SH y de las otras dos cabinas LH.  
– Conclusión: causas mixtas/compensatorias a nivel global, donde el perfil operativo negativo es la base común y la excepción positiva de Business LH modera el impacto sin revertirlo.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

A continuación, el diagnóstico consolidado por causa raíz, con el segmento “más grande” afectado y su evidencia clave.

1. Causa: Operaciones disruptivas (cancelaciones, retrasos y cambios de avión)  
A. Naturaleza de la causa  
  • Hipótesis: Un pico sostenido de cancelaciones, retrasos y reprogramaciones incrementó la insatisfacción, sobre todo en rutas cortas donde el factor “tiempo perdido” resulta más sensible.  
B. Evidencia consolidada y alcance  
  • Segmento: Global / SH / Economy (NPS 0.70 vs baseline 22.03; Δ –21.33 pts)  
  • Afecta a todos los subsegmentos bajo Global / SH / Economy (IB y YW).  
  • Incidentes operativos (ncs_tool): 48 total – 20 cancelaciones, 21 retrasos, 7 cambios de avión. Caso crítico: vuelo IB273 (MAD–HAV) retornó por emergencia técnica, 24 cambios de aeronave y 12 reprogramaciones.  
  • Rutas con peor NPS (routes_tool): CDG–MAD NPS –50.0 (8 encuestas), GUA–MAD NPS 0.0 (4).  
  • Verbatims (verbatims_tool): 661 comentarios, todos positivos en trato y organización, pero sin reflejo de los incidentes operativos (indica sesgo en la captura de feedback).  

2. Causa: Deficiencias en embarque, gestión de equipaje y comunicaciones  
A. Naturaleza de la causa  
  • Hipótesis: Procesos de embarque/desembarque lentos, falta de espacio en bodega y avisos tardíos provocaron frustración acumulada en vuelos de corto radio.  
B. Evidencia consolidada y alcance  
  • Segmento: Global / SH (NPS 2.87 vs baseline 23.80; Δ –20.93 pts)  
  • Afecta a todos los subsegmentos bajo Global / SH (Economy y Business, IB y YW).  
  • Incidentes operativos (ncs_tool): 48 total – 20 cancelaciones, 21 retrasos, 7 otras incidencias (equipos/procesos). Principal driver: MAD–HAV con emergencia técnica y 24 cambios de avión.  
  • Rutas impactadas (routes_tool): AMS–MAD NPS –44.4 (9), MAD–HAV baja percepción ligada a retrasos y problemas técnicos.  
  • Verbatims (verbatims_tool): quejas de embarque descoordinado, falta de espacio de equipaje de mano y comunicación deficiente sobre puertas y cambios.  

3. Causa: Problemas de confort y percepción de la flota  
A. Naturaleza de la causa  
  • Hipótesis: La incomodidad de asientos e imagen negativa de ciertas aeronaves (Air Nostrum, 32S, A320) acentuaron la insatisfacción global, especialmente cuando se sumaron incidencias operativas.  
B. Evidencia consolidada y alcance  
  • Segmento: Global (NPS 5.55 vs baseline 21.56; Δ –16.01 pts)  
  • Afecta a todos los subsegmentos bajo Global (LH y SH, Economy/Business/Premium).  
  • Incidentes operativos (ncs_tool): 527 total – 151 retrasos, 84 cancelaciones; temas recurrentes: “passenger”, “aircraft_change”, “technical_issues”. Ruta más golpeada: MAD–HAV (49 vuelos).  
  • Verbatims (verbatims_tool): quejas por mala gestión de equipaje, retrasos/cancelaciones y “asientos incómodos”; percepción negativa de flota Air Nostrum.  

4. Causa: Excelente calidad de servicio en Business Long Haul (amortiguador)  
A. Naturaleza de la causa  
  • Hipótesis: Una experiencia de servicio excepcional (tripulación, catering, confort) eclipsó las disrupciones operativas, generando un efecto compensatorio en el NPS global.  
B. Evidencia consolidada y alcance  
  • Segmento: Global / LH / Business (NPS 50.0 vs baseline 29.57; Δ +20.43 pts)  
  • Afecta al subsegmento Global / LH / Business en su totalidad.  
  • Incidentes operativos (ncs_tool): 45 – 14 cancelaciones, 16 retrasos, 7 cambios de aeronave, 8 “otros”. Caso destacado: IB273/MAD–HAV regresó por emergencia técnica.  
  • Verbatims (verbatims_tool): 34 comentarios muy positivos: elogios a tripulación, catering y comodidad; ausencia total de quejas operativas.  
  • Ruta destacada (routes_tool): GRU–MAD NPS 100.0 (3 encuestas).  

5. Causa: Incidencias críticas en Premium Long Haul  
A. Naturaleza de la causa  
  • Hipótesis: Altísimo nivel de disrupción operativa en Premium (cancelaciones, retrasos, cambios de avión) y problemas específicos en la flota A350 next generaron la peor caída de NPS.  
B. Evidencia consolidada y alcance  
  • Segmento: Global / LH / Premium (NPS –25.0 vs baseline 14.25; Δ –39.25 pts)  
  • Afecta al subsegmento Global / LH / Premium íntegramente.  
  • Incidentes operativos (ncs_tool): 45 – 14 cancelaciones, 16 retrasos, 7 cambios de aeronave, 12 reprogramaciones.  
  • Verbatims (verbatims_tool): 34 comentarios mayoritariamente positivos, sin mención de fallos, lo que sugiere falta de captura del feedback de los pasaje­s afectados.  
  • Rutas críticas: BOG–MAD NPS –25.0 (4 respuestas); flota A350 next NPS –66.7.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El análisis jerárquico de NPS del 24-octubre-2025 revela un fuerte descenso en el indicador global, que pasó de un NPS de 21.56 (vs período previo 7 días) a 5.55 (-16.01 pts). Esta caída responde a dos fenómenos principales: en Short Haul, Global/SH cayó de 23.80 a 2.87 (-20.93 pts) por un pico de 48 incidencias operativas (20 cancelaciones, 21 retrasos y 7 cambios de avión, con foco en el vuelo IB273 MAD–HAV) y deficiencias en embarque, equipaje y comunicación; en Long Haul, las cabinas Economy y Premium también sufrieron descensos (Economy LH de 16.58 a 10.69, -5.89 pts; Premium LH de 14.25 a –25.00, -39.25 pts) por reprogramaciones y problemas en la flota A350 next, mientras Business LH se disparó de 29.57 a 50.00 (+20.43 pts) gracias a un servicio a bordo valorado de forma sobresaliente.  

Las rutas más afectadas incluyen MAD–HAV (49 vuelos impactados y emergencia técnica), CDG–MAD (Economy SH IB NPS –50.0), AMS–MAD (SH NPS –44.4) y BOG–MAD (Premium LH NPS –25.0). Los pasajeros de negocio, residentes en Europa y usuarios de flota A350 y A321 fueron los perfiles más reactivos negativamente, mientras que viajeros de ocio, América Central, Oriente Medio y code-share VY/LATAM mostraron niveles de satisfacción superiores.

ECONOMY SH IB & YW: Operaciones Disruptivas y Captura Parcial de Feedback  
En la cabina Global/SH/Economy, Iberia (IB) mostró un NPS de –11.57 (vs L7d 16.09, -27.66 pts) y Level (YW) un NPS de 21.52 (vs L7d 33.39, -11.87 pts), resultando en un consolidado de 0.70 (vs L7d 22.03, -21.33 pts). El deterioro se explica principalmente por 48 incidencias operativas (20 cancelaciones, 21 retrasos, 7 cambios de avión) en rutas como MAD–HAV (24 cambios de avión) y por la falta de reflejo de estas disrupciones en los 661 verbatims recibidos. CDG–MAD cayó a –50.0 (n=8) y GUA–MAD a 0.0 (n=4), con mayor insatisfacción entre viajeros de negocio y usuarios de A350.

BUSINESS SH IB & YW: Insatisfacción de Clientes de Negocio  
En Global/SH/Business, IB alcanzó un NPS de 19.05 (vs L7d 37.84, -18.79 pts) y YW de 15.00 (vs L7d 35.73, -20.73 pts), para un promedio de 17.74 (vs L7d 37.57, -19.82 pts). A pesar de que las 48 incidencias operativas afectaron fundamentalmente vuelos de largo radio, el feedback de 93 verbatims en SH Business solo recoge elogios a puntualidad y servicio; la caída responde a la insatisfacción específica de pasajeros de negocio en ciertas regiones y aviones A321, sin que las operaciones disruptivas aparezcan en las encuestas.

ECONOMY LH: Impacto Moderado de Disrupciones  
La cabina Global/LH/Economy registró un NPS de 10.69 (vs L7d 16.58, -5.89 pts), reflejando un repunte de cancelaciones (14), retrasos (16) y cambios de aeronave (7), con punto crítico en el vuelo IB273 MAD–HAV. Aunque 260 verbatims destacan trato y comida, la ruta GUA–MAD marcó un NPS 0.0 (n=4), con mayor descontento en viajeros de negocio y usuarios de A350 C y A33ACMI.

BUSINESS LH: Servicio a Bordo como Amortiguador  
En Global/LH/Business, el NPS subió de 29.57 a 50.00 (+20.43 pts) a pesar de 45 incidencias (14 cancelaciones, 16 retrasos, 7 cambios de avión). Los 34 verbatims fueron unánimemente positivos sobre tripulación, catering y confort, y la ruta GRU–MAD alcanzó un NPS de 100.0 (n=3), compensando las disrupciones operativas.

PREMIUM LH: Disrupciones Críticas en Flota A350 next  
La cabina Global/LH/Premium cayó de 14.25 a –25.00 (-39.25 pts) bajo un alto número de cancelaciones (14), retrasos (16), cambios de avión (7) y 12 reprogramaciones. Aunque los 34 verbatims hablan bien de limpieza y menú, no recogen las incidencias reales; BOG–MAD registró –25.0 (n=4) y la flota A350 next marcó un NPS de –66.7, penalizando especialmente a viajeros de ocio.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-10-23 to 2025-10-23:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

Economy Short Haul – causa específica de compañía  
• Existen ambos nodos (IB y YW) y presentan patrones de drivers distintos:  
  – IB (–6.5 pts) se vio impactada por incidencias operativas concentradas en EAS–MAD y problemas de fiabilidad de la flota A350 C.  
  – YW (–9.1 pts) refleja quejas de gestión de equipaje y un posible sesgo de muestreo más que las mismas rutas o flotas.  

Business Short Haul – causa específica de compañía  
• Existen ambos nodos (IB y YW) con anomalías opuestas:  
  – IB (+8.3 pts) muestra un alza impulsada por feedback positivo de rutas no afectadas y baja respuesta de los clientes impactados.  
  – YW (–24.0 pts) evidencia fuertes quejas por cancelaciones/retrasos y gestión de equipaje.  

En ambos casos, los drivers y la evidencia operativa divergen para IB y YW, por lo que las causas son específicas de cada compañía, no generales a la cabina.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul – causa específica de cabina  
• Economy SH (–7.9 pts) y Business SH (–1.9 pts) ambos reflejan el impacto de cancelaciones y retrasos, pero reaccionan de forma distinta:  
  – Economy SH sufre de manera consistente y homogénea (IB –6.5 pts, YW –9.1 pts).  
  – Business SH muestra alta heterogeneidad (IB +8.3 pts vs YW –24.0 pts).  
Conclusión: las causas operativas (retrasos/cancelaciones) afectan de forma diferente a cada cabina, por lo que el patrón es específico de cabina en Short Haul.

Long Haul – causa común al radio con reactividad diferencial  
• Economy LH (–3.9 pts), Business LH (–6.0 pts) y Premium LH (–33.0 pts) apuntan al mismo driver raíz (retrasos, cancelaciones y gestión de asistencia),  
• pero con niveles de sensibilidad muy distintos (Economy amortigua, Premium amplifica).  
Conclusión: la causa es compartida por todo el radio Long Haul, aunque las cabinas muestran distinta reactividad.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Ambos radios afectados por la misma raíz operativa; el nodo Global refleja coherentemente su impacto  
• Short Haul (–7.4 pts) y Long Haul (–5.8 pts) comparten como drivers principales el elevado número de cancelaciones y retrasos, agravado por falta de comunicación y deficiencias en la atención al pasajero.  
• Aunque cada radio incorpora matices (SH añade quejas de equipaje y cambios de asiento; LH suma flota específica y código compartido), los patrones convergen en la causa operativa de fondo.  
• El Global (–6.5 pts) no “normaliza” ninguna anomalía ni las cancela entre sí, sino que atenúa levemente los extremos de cada radio y presenta un impacto agregado coherente con los dos niveles.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

1. Causa: Elevado número de cancelaciones y retrasos  
A. Naturaleza del driver  
   • Hipótesis: La acumulación de incidencias críticas (cancelaciones y demoras) genera una sensación de inestabilidad y pérdida de control en el pasajero, que se traduce en una caída generalizada del NPS.  

B. Evidencia consolidada y alcance  
   • Segmento más grande afectado: Global (todos los pasajeros de cualquier radio y cabina)  
   • Output causal detallado:  
     – NPS del día: 15.08 vs baseline 21.56 (–6.47 pts)  
     – Incidentes totales: 520 (107 retrasos, 98 cancelaciones)  
     – Rutas con más incidencias: MAD–MRS (4 eventos, NPS –33.3, n=9), EAS–MAD (4 eventos, NPS 0.0, n=6)  
     – Verbatims representativos: “No recibimos información clara durante el retraso”, “Me alojaron de urgencia sin aviso previo”  
   • Afecta a todos los subsegmentos bajo Global (Long Haul, Short Haul, Economy, Business y Premium).  

2. Causa: Deficiente comunicación y gestión de asistencia  
A. Naturaleza del driver  
   • Hipótesis: La falta de información proactiva y apoyo en tierra (alojamiento, reembolsos) agrava el impacto de cualquier incidencia operativa, incluso de demoras moderadas.  

B. Evidencia consolidada y alcance  
   • Segmento más grande afectado: Global / Short Haul  
   • Output causal detallado:  
     – NPS del día: 16.45 vs baseline 23.80 (–7.35 pts)  
     – Incidentes NCS: 62 (24 cancelaciones, 15 retrasos, 4 desvíos, 15 otras)  
     – Rutas clave: BLQ–MAD (NPS –20.0, n=10), EAS–MAD (NPS 0.0, n=6)  
     – Verbatims: “Nos cambiaron de asiento sin avisar”, “Retraso de 2h30 sin nadie que informe”  
   • Afecta a todos los subsegmentos bajo Global / Short Haul: Economy SH e Business SH.  

3. Causa: Problemas de fiabilidad y confort de la flota en Long Haul  
A. Naturaleza del driver  
   • Hipótesis: Incidentes técnicos y deficiencias de confort en aeronaves de largo radio (A321XLR, A332, A350) erosionan la percepción de servicio premium y desplazan el NPS a niveles muy bajos.  

B. Evidencia consolidada y alcance  
   • Segmento más grande afectado: Global / Long Haul  
   • Output causal detallado:  
     – NPS del día: 12.08 vs baseline 17.86 (–5.78 pts)  
     – Incidentes NCS: 59 (18 cancelaciones, 13 retrasos, 6 cambios de avión, 4 por meteorología)  
     – Ruta más crítica: vuelo IB0157 (6 h 55 min de retraso; 170 pax pernoctados)  
     – Perfiles de flota con peor NPS: A321XLR (–16.7), A332 (–2.5), A350 estándar (–40.0), A350 “next” (–20.0)  
     – Verbatims: “Cancelaron mi vuelo sin ofrecer hotel adecuado”, “Cabina muy ruidosa y sin aviso de cambio”  
   • Afecta a todos los subsegmentos bajo Global / Long Haul: Economy LH, Business LH y Premium LH.  

4. Causa: Fallos en la gestión de equipaje  
A. Naturaleza del driver  
   • Hipótesis: Pérdidas, demoras o daños en equipaje impactan de forma desproporcionada a clientes fidelizados, erosionando la lealtad especialmente en trayectos de negocio.  

B. Evidencia consolidada y alcance  
   • Segmento más grande afectado: Global / Short Haul / Business /YW  
   • Output causal detallado:  
     – NPS del día: 11.76 vs baseline 35.73 (–23.96 pts)  
     – Incidentes NCS: 62 totales (24 cancelaciones, 15 retrasos, 8 problemas con pasajeros, 6 cambios de avión, 3 meteorología)  
     – Verbatims: “Mi maleta llegó 24 h después”, “No me informaron dónde recoger equipaje dañado”  
     – Rutas no desglosadas (n<2 encuestas)  
   • Afecta a todos los subsegmentos bajo Global / Short Haul / Business /YW.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El análisis del 23-oct-2025 revela descensos generalizados de NPS en todos los niveles operativos. A nivel Global, el NPS cayó de 21.56 a 15.08 (–6.47 pts). En corto radio, Global/SH pasó de 23.80 a 16.45 (–7.35 pts), con Economy SH desplomándose de 22.03 a 14.16 (–7.87 pts) e impulsado por IB (16.09→9.54, –6.55) y YW (33.39→24.32, –9.07), mientras Business SH moderó su caída neta (37.57→35.71, –1.85) por la ganancia de IB (37.84→46.15, +8.32) y la fuerte pérdida de YW (35.73→11.76, –23.96). En largo radio, Global/LH retrocedió de 17.86 a 12.08 (–5.78), con Economy LH menor (16.58→12.63, –3.95), Business LH notablemente afectada (29.57→23.53, –6.04) y Premium LH sufriendo la mayor caída (14.25→–18.75, –33.00 pts).

Las rutas más afectadas corresponden a los enlaces Madrid–Marsella (Global NPS –33.3), BLQ–MAD (SH Economy –20.0) y MAD–MIA (Premium LH –33.3), todas marcadas por múltiples cancelaciones, retrasos y deficiencias en la gestión de equipaje. Los grupos más reactivos incluyen pasajeros de negocio en YW Short Haul, residentes en Europa en Economy LH, usuarios de A321XLR/A350 estándar y viajeros de ocio de América Norte en Premium LH.

ECONOMY SH: Desaceleración en percepción de pasajeros Economy de corto radio  
La cabina Economy de SH durante la semana del 23-oct-2025 registró un NPS de 14.16 (día 23-oct-2025) con una caída de 7.87 puntos vs L7d. IB marcó 9.54 (16.09→9.54, –6.55) y YW 24.32 (33.39→24.32, –9.07). La causa principal fue el aumento de cancelaciones (24) y retrasos (15) combinado con deficiencias en comunicación y handling de equipaje, tal como reflejan quejas sobre cambios de asiento sin aviso y maletas tardías. El deterioro se concentró en rutas como MAD–MUC (NPS –28.6, n=7) y EAS–MAD (0.0, n=6). Los perfiles más sensibles fueron viajeros de negocio (NPS 1.4) y residentes en Europa (–4.5).

BUSINESS SH: Divergencia entre compañías impulsa ligera caída neta  
El segmento Business de SH registró un NPS de 35.71 (23-oct-2025) con un descenso de 1.85 puntos vs L7d. IB mejoró de 37.84 a 46.15 (+8.32), sustentado en verbatims muy positivos en rutas sin incidencias (BCN–MAD, NPS 75.0). En contraste, YW se desplomó de 35.73 a 11.76 (–23.96) por quejas de equipaje y falta de información en cancelaciones y retrasos. Esta evolución refleja la heterogeneidad operativa entre carriers, afectando principalmente a usuarios de A321 y clientes en código compartido con LATAM.

ECONOMY LH: Impacto moderado en Economy de largo radio  
La cabina Economy de LH cerró el día con un NPS de 12.63 (23-oct-2025), 3.95 puntos por debajo de la media L7d (16.58→12.63). El descenso se debió a 18 cancelaciones y 13 retrasos en rutas de larga distancia, especialmente MAD–EAS (2 incidencias) y LIM–MAD (NPS 11.1, n=9), y a problemas de confort en A321XLR (–16.7) y flota en codeshare con BA (–33.3). Los viajeros de leisure y residentes en Europa fueron los más sensibles.

BUSINESS LH: Caída significativa por protocolo de asistencia y flota  
La cabina Business de LH registró un NPS de 23.53 el 23-oct-2025, retrocediendo 6.04 puntos vs L7d (29.57→23.53). Los drivers principales fueron 18 cancelaciones, 13 retrasos y deficiencias en la gestión de alojamiento tras incidentes críticos como el vuelo IB0157 (6 h 55 min de demora). La flota A332 (–20.0) y clientes norteamericanos mostraron las peor valoración; rutas como EZE–MAD (40.0, n=5) sin incidentes destacan la falta de correlación directa en todos los trayectos.

PREMIUM LH: Colapso de satisfacción tras demoras extremas  
El segmento Premium de LH sufrió un NPS de –18.75 (23-oct-2025), 33.00 puntos por debajo de L7d (14.25→–18.75). El colapso se concentró en ocio de América Norte (–100.0) en MAD–MIA (–33.3, n=3), afectado por la reprogramación del vuelo IB0157 con pernocta de 170 pasajeros. A350 estándar (–40.0) y “next” (–20.0) incumplieron expectativas de confort y puntualidad, elevando la insatisfacción.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-10-22 to 2025-10-22:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

A. Economy Short Haul  
• Existen los dos nodos: SH/Economy – IB y SH/Economy – YW.  
• Ambos muestran drivers muy similares (alta percepción de puntualidad y servicio pese a incidentes), patrones de feedback convergentes y mismas limitaciones en datos operativos.  
Diagnóstico: la anomalía positiva es común a toda la cabina Economy SH, no específica de IB o de YW.  

B. Business Short Haul  
• Existen los dos nodos: SH/Business – IB y SH/Business – YW.  
• Ambos mantienen un desempeño estable (sin anomalías), con drivers y evidencia operativa homogéneos.  
Diagnóstico: no hay divergencia por compañía; el comportamiento es general de la cabina Business SH.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul (SH)  
• Economy SH vs Business SH: patrón divergente  
  – Economy SH presenta la anomalía positiva (+8.8 pts) de manera consistente en IB y YW.  
  – Business SH es normal, sin desviaciones relevantes.  
• Diagnóstico SH: la causa del alza de NPS es específica de la cabina Economy; la cabina Business no reacciona a esos drivers operativos (actúa como amortiguador).  

Long Haul (LH)  
• Economy LH, Business LH y Premium LH: patrón divergente  
  – Business LH muestra una anomalía negativa (–16.2 pts).  
  – Economy LH (+3.0 pts) y Premium LH (+5.7 pts) están dentro de variación normal.  
• Diagnóstico LH: la caída de NPS es específica de la cabina Business; Economy y Premium mitigan el impacto de los problemas operativos.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Solo Short Haul muestra un impacto claro (+7.9 pts) impulsado por la alta satisfacción en Economy SH; Long Haul permanece neutral (Business LH cae pero Economy-Premium LH compensan). A nivel Global, la anomalía positiva moderada (+5.8 pts) refleja un efecto compensatorio entre el fuerte alza en SH y la estabilidad en LH.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

Causa 1  
A. NATURALEZA DE LA CAUSA  
• Alto nivel de satisfacción ligado a puntualidad percibida, comodidad de cabina y profesionalidad de tripulación en vuelos de corto recorrido clase Economy.  

B. EVIDENCIA CONSOLIDADA Y ALCANCE  
• Segmento más grande afectado: Global / SH / Economy (n≈634 verbatims)  
• Output causal clave:  
  – NPS el día: 30.86 vs baseline 22.03 (anomalía +8.82 pts)  
  – Incidentes registrados: 21 (10 cancelaciones, 9 retrasos, 2 otros)  
  – Caso crítico: IB0157 con 6 h 55 min de retraso y 170 pasajeros pernoctando en MAD  
  – Verbatims representativos: “puntualidad impecable”, “comodidad en cabina”, “tripulación muy profesional”  
  – Ruta con impacto aislado: MAD–VCE NPS –5.0 (n=20), sin incidente NCS asociado  
• Este driver positivo se extiende a ambos subsegmentos SH / Economy / IB y SH / Economy / YW, sin divergencias en patrones ni en evidencia operativa.  

Causa 2  
A. NATURALEZA DE LA CAUSA  
• Insatisfacción concentrada en pasajeros de negocios de largo recorrido debido a demoras graves y expectativas de servicio elevadas (flota A350, vuelos con código IB).  

B. EVIDENCIA CONSOLIDADA Y ALCANCE  
• Segmento más grande afectado: Global / LH / Business (40 encuestas)  
• Output causal clave:  
  – NPS el día: 13.33 vs baseline 29.57 (anomalía –16.23 pts)  
  – Incidentes totales: 28 (10 cancelaciones, 11 retrasos, 1 equipaje extraviado, 7 otros)  
  – Caso más grave: IB0157, 6 h 55 min de retraso, 170 pasajeros sin hotel en MAD  
  – Verbatims: prácticamente neutros, pero Business / Work NPS –40.0 (n=5)  
  – Perfil flota: A350 NPS –60.0 (n=5) vs A321XLR NPS +66.7 (n=3)  
  – Code-share: IB NPS 10.5 (n=19) vs AA NPS 66.7 (n=6)  
  – Ruta con datos: LIM–MAD NPS 50.0 (n=4), sin incidentes NCS  
• La caída de NPS impacta todo el segmento Business LH sin matices regionales o de compañía.  

Causa 3  
A. NATURALEZA DE LA CAUSA  
• Efecto compensatorio generado por la sólida satisfacción de viajeros de negocio, flotas preferidas (32S, A321XLR) y vuelos code-share I2, que eleva el NPS global.  

B. EVIDENCIA CONSOLIDADA Y ALCANCE  
• Segmento más grande afectado: Global (12 segmentos, n total de encuestas no especificado)  
• Output causal clave:  
  – NPS el día: 27.37 vs baseline 21.56 (anomalía +5.81 pts)  
  – Incidentes totales: 303 (81 retrasos, 66 cancelaciones, 156 otros)  
  – Verbatims destacados: quejas por retrasos, equipaje y comunicación, pero fuerte alza en Business vs Leisure (+5.5 pts)  
  – Flotas con mejor rendimiento: 32S y A321XLR (NPS hasta 53.3)  
  – Code-share: I2 NPS 66.7 vs BA NPS –33.3  
  – Ruta más crítica: LAX–MAD NPS –18.2 (n=11)  
• Este driver positivo engloba todos los subsegmentos bajo Global, compensando localmente la insatisfacción de Business LH.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
El análisis del 22-oct-2025 revela contrastes claros entre corto y largo radio. A nivel Global, el NPS pasó de 21.56 a 27.37 (+5.81 pts), impulsado por una extraordinaria mejora en Short Haul (SH), donde Economy SH escaló de 22.03 a 30.86 (+8.82 pts) y el segmento SH Global alcanzó 31.68 vs 23.80 (+7.88 pts). Este auge se explica por la percepción de puntualidad, comodidad de cabina y profesionalidad de tripulación, a pesar de 21 incidentes (10 cancelaciones, 9 retrasos). En contraste, Business Long Haul (LH) sufrió un desplome de 29.57 a 13.33 (–16.23 pts), originado en demoras severas—pico de 6 h 55 min en IB0157—y elevadas expectativas de viajeros de negocios en flota A350, sin que los verbatims expresen quejas directas. Economy y Premium LH, con variaciones de +3.00 y +5.75 pts respectivamente, amortiguaron la caída de Business LH, manteniendo el radio largo dentro de rangos normales.

En cuanto a rutas y perfiles, LAX–MAD emerge como la más afectada (NPS –18.2, n=11), seguida de MAD–VCE (NPS –5.0, n=20) y PMI–VLC (NPS 33.3, n=3). Los viajeros de negocio de largo radio y los vuelos en A350 concentran la insatisfacción más marcada, mientras que los pasajeros de Economy SH—tanto IB como YW—muestran la mayor reactividad positiva, con verbatims que destacan puntualidad, cortesías a bordo y agilidad en procesos.

ECONOMY SH: Subida impulsada por puntualidad y servicio  
La cabina Economy de SH combinada (Global/SH/Economy) registró un NPS de 30.86 el 22-oct-2025 (vs L7d 22.03, +8.82 pts). En SH/Economy/IB el NPS mejoró de 16.09 a 26.17 (+10.09 pts) y en SH/Economy/YW subió de 33.39 a 40.41 (+7.02 pts). La causa principal fue la alta percepción de puntualidad, comodidad de cabina y profesionalidad de la tripulación, validada por más de 600 verbatims positivos, pese a 21 incidentes (10 cancelaciones, 9 retrasos). Esta mejora se reflejó incluso en rutas con baja muestra como MAD–VCE (NPS –5.0), y los perfiles más reactivos incluyeron pasajeros de flota A332 (NPS 100.0) y code share I2 (66.7).

BUSINESS SH: Estabilidad de alto nivel  
El segmento Business de SH (Global/SH/Business) mantuvo un NPS de 41.03 el 22-oct-2025 (vs L7d 37.57, +3.46 pts), dentro de rangos normales. SH/Business/IB subió de 37.84 a 40.74 (+2.90 pts) y SH/Business/YW de 35.73 a 41.67 (+5.94 pts). No se detectaron cambios significativos en drivers u operativa, y las valoraciones siguieron enfocadas en servicios de cabina y atención al cliente sin incidencias críticas.

ECONOMY LH: Desempeño estable  
La cabina Economy de LH (Global/LH/Economy) registró un NPS de 19.58 el 22-oct-2025 (vs L7d 16.58, +3.00 pts), dentro de la variación normal. No se identificaron drivers negativos ni positivos destacados; el servicio y la regularidad operativa se mantuvieron consistentes, sin rutas con impacto significativo bajo este nodo.

BUSINESS LH: Deterioro marcado  
La cabina Business de LH (Global/LH/Business) sufrió un NPS de 13.33 el 22-oct-2025 (vs L7d 29.57, –16.23 pts). Los principales drivers fueron cancelaciones (10), retrasos críticos (11) y equipaje extraviado, con especial incidencia en vuelo IB0157 (6 h 55 min de demora y 170 pasajeros sin alojamiento). Los viajeros de negocios y flota A350 (NPS –60.0) encabezaron la insatisfacción, sin que las encuestas de rutas (p.ej. LIM–MAD NPS 50.0) reflejaran los incidentes operativos.

PREMIUM LH: Estabilidad dentro de rango  
El segmento Premium de LH (Global/LH/Premium) alcanzó un NPS de 20.00 el 22-oct-2025 (vs L7d 14.25, +5.75 pts), manteniéndose dentro de la variación esperada. No se registraron drivers operativos o verbatims que sugieran cambios significativos, confirmando un desempeño estable en esta clase de servicio.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-10-21 to 2025-10-21:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

Economy Short Haul: existen ambos nodos (IB y YW) y divergen claramente en sus drivers (IB crece por calidad de servicio y resiliencia frente a incidencias; YW cae por un sesgo de muestra ligado a code-shares y la ruta LEI-MAD). Causa específica de compañía.  
Business Short Haul: existen ambos nodos (IB y YW) y comparten el mismo patrón de alza (fuerte énfasis en servicio, puntualidad y tripulación), por lo que la causa es general a la cabina.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul: patrón específico de cabina.  
- Economy SH muestra divergencia entre compañías (IB sube, YW baja).  
- Business SH converge en ambas (IB y YW suben).  

Long Haul: patrón específico de cabina.  
- Economy LH no anómala, Business LH positiva y Premium LH negativa.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Mixed/compensatorio: ambos radios muestran anomalías opuestas en sub-cabinas (SH con Business al alza y Economy IB≠YW; LH con Business positivo y Premium negativo) y al consolidarse generan el leve repunte global de +2.4 pts.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

Causa 1: Excelencia del servicio a bordo  
A. Naturaleza  
  • Driver: percepción muy alta de confort, atención de tripulación y procesos (check-in, embarque, comunicación).  
  • Hipótesis: un “efecto servicio” que domina sobre las incidencias operativas reportadas.  

B. Evidencia consolidada y alcance  
  • Segmento mayor: Global/Short-Haul/Business (NPS 49.02 vs baseline 37.57, anomalía +11.45 pts, n ≈ 65 verbatims).  
  • Incidentes NCS: 38 totales (15 retrasos, 8 cancelaciones, otros técnicos y bird-strikes).  
  • Rutas clave: DUS-MAD (NPS 100, n = 3, sin incidencias).  
  • Verbatims representativos: “servicio de abordo excelente”, “puntualidad percibida pese a retrasos oficiales”, “amabilidad de la tripulación”.  
  • Alcance: afecta a todos los subsegmentos bajo Global/SH/Business (IB y YW) y también se replica en Global/LH/Business.  

Causa 2: Sesgo de muestra y perfiles de cliente  
A. Naturaleza  
  • Driver: composición del sample (code-shares, ruta LEI-MAD con n pequeño, mayor peso de Leisure y regiones menos satisfechas).  
  • Hipótesis: la diferencia en quién responde (partners específicos y pocos encuestados) sesga el NPS a la baja.  

B. Evidencia consolidada y alcance  
  • Segmento mayor: Global/SH/Economy/YW (NPS 23.08 vs baseline 33.39, anomalía –10.31 pts, n = 204 verbatims).  
  • Incidentes NCS: 38 totales (15 retrasos, 8 cancelaciones, 5 otras incidencias, 2 desvíos).  
  • Ruta crítica: LEI-MAD (NPS 0.0, n = 4, sin incidencias operativas).  
  • Drivers de perfil: code-share VY (–40), QR (–25), AA (+50); región América Sur (+75) vs Norte (–15).  
  • Verbatims: tono general positivo (limpieza, amabilidad), sin quejas operativas, lo que reafirma que el descenso no es por calidad de servicio.  
  • Alcance: afecta únicamente al nodo Global/SH/Economy/YW.  

(Estas dos causas explican de forma consolidada los principales saltos positivos en Business y la caída en Economy YW. Las incidencias operativas, aunque presentes, no son el motor de las anomalías detectadas sino un telón de fondo contrastado contra el feedback cualitativo.)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

Durante la semana del 2025-10-21, el NPS Global subió de 21.56 a 23.98 (+2.42 pts vs L7d) gracias al fuerte desempeño en Business Short Haul (de 37.57 a 49.02, +11.45) y en Business Long Haul (de 29.57 a 42.11, +12.54), mientras Premium Long Haul retrocedió de 14.25 a 10.53 (-3.73) y Economy Short Haul YW cayó de 33.39 a 23.08 (-10.31). El alza en Business responde a un “efecto servicio” dominado por confort de cabina, puntualidad percibida y calidad de atención —aun frente a 38 incidentes operativos—; la baja en Economy SH YW se explica por un sesgo de muestra (ruta LEI-MAD con NPS 0.0, n=4 y puntuaciones bajas en code-shares VY/QR) sin evidencia de quejas de servicio.

En términos de rutas, DUS-MAD destacó con NPS 100 (SH Business) y LEI-MAD mostró NPS 0 (SH Economy YW). MAD-VGO demostró resiliencia con NPS 7.1 en Economy IB pese a 4 incidentes. Los grupos más reactivos fueron viajeros de codeshare VY (–40) y QR (–25) en SH Economy YW, clientes de larga distancia en flota A350 (NPS 75.0) y viajeros de ocio en Premium LH.

ECONOMY SH: Compensación entre IB y YW  
La cabina Economy de SH mantuvo un NPS agregado de 23.73 durante la semana del 2025-10-21 (vs L7d: +1.69). Desagregado por compañía, IB subió de 16.09 a 24.07 (+7.98) impulsada por alta percepción de puntualidad, eficiencia en check-in y atención de tripulación pese a 38 incidentes, mientras que YW cayó de 33.39 a 23.08 (-10.31) por un sesgo de muestra en la ruta LEI-MAD (NPS 0.0, n=4) y puntuaciones bajas en code-shares VY/QR. El neto se neutralizó, manteniendo estable la satisfacción global.

BUSINESS SH: Alza sostenida por “efecto servicio”  
El segmento Business de SH experimentó un NPS de 49.02 (vs L7d: +11.45), pasando de 37.57 a 49.02. Este avance refleja comentarios muy positivos sobre confort de cabina, atención de tripulación y percepción de puntualidad, con 65 verbatims elogiosos y sin alusiones a los 38 incidentes reportados. La ruta DUS-MAD alcanzó NPS 100 (n=3), y los perfiles más satisfechos incluyeron pasajeros corporativos de América del Sur.

ECONOMY LH: Desempeño estable  
La cabina Economy de LH mantuvo desempeño estable, registrando un NPS de 18.39 durante la semana del 2025-10-21 (vs L7d: +1.81). No se detectaron cambios significativos en la satisfacción, a pesar de 37 incidentes operativos (11 retrasos, 4 cancelaciones) y comentarios consistentemente positivos sobre puntualidad y limpieza de cabina.

BUSINESS LH: Salto por percepción de valor  
La cabina Business de LH registró un NPS de 42.11 (vs L7d: +12.54), mejorando de 29.57 a 42.11. La evolución fue impulsada por evaluaciones muy positivas de confort en A350, trato de la tripulación y relación precio-valor, sin mención a los incidentes (37 NCS). La ruta EZE-MAD alcanzó NPS 100 (n=3) y el perfil “Business/Work” mostró NPS 60.0.

PREMIUM LH: Deterioro por perfil de sample  
El segmento Premium de LH descendió de 14.25 a 10.53 (-3.73 vs L7d). Aunque los 36 verbatims fueron 100 % positivos, el peso de viajeros Leisure (NPS –8.3) y de ciertas regiones con tendencia a puntuar bajo explica la baja, sin vinculación directa a las 37 incidencias operativas ni a rutas concretas.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-10-20 to 2025-10-20:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

Economy Short Haul: Existen ambos subnodos (IB y YW) pero muestran comportamientos opuestos – IB registra una anomalía positiva de +8.9 pts mientras YW se mantiene en rango normal – por lo que la mejora de NPS en Economy SH es específica de la aerolínea IB.  
Business Short Haul: Tanto IB (+14.8 pts) como YW (+14.3 pts) presentan anomalías positivas de magnitud similar y comparten drivers de servicio y puntualidad, señalando una causa común al segmento Business SH.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul (SH): aunque tanto Economy SH (+8,3 pts) como Business SH (+14,3 pts) registran anomalías positivas, Business mejora de forma homogénea en IB y YW, mientras que el repunte en Economy sólo ocurre en IB (YW se mantiene normal), de modo que no existe un patrón común al radio SH, sino respuestas específicas por cabina y compañía.  
Long Haul (LH): las tres cabinas divergen claramente —Premium LH lidera con +33,1 pts, Economy LH modera con +9,0 pts y Business LH permanece estable—, lo que confirma un comportamiento específico de cada clase de servicio más que un efecto común al radio LH.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Short Haul y Long Haul muestran ambas anomalías positivas, pero con drivers diferenciales:  
- En SH, el alza se explica por un fuerte repunte en Business (IB y YW) y una mejora de Economy centrada en IB.  
- En LH, el crecimiento está dominado por Premium y en menor medida por Economy (Business mantiene estabilidad).  

Al consolidarse en Global (+9,4 pts), los dos radios no se cancelan sino que se suman, amplificando el efecto positivo. Por tanto, tenemos causas mixtas: cada radio presenta su propio patrón operativo—mejora de servicio y sesgos de muestra en distintos segmentos—pero todas convergen en un impulso global de NPS.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

A continuación detallo las causas principales, el segmento “padre” más representativo y la evidencia consolidada que las sustenta:

1) Causa: Sesgo de muestra hacia viajeros Leisure y regiones de alta satisfacción  
A. Naturaleza de la causa  
- No es un cambio operativo puntual, sino un desplazamiento en la composición de respuestas: mayor proporción de pasajeros de ocio y de Norteamérica, cuyos niveles de satisfacción triplican o cuadruplican a los de otros perfiles.  
B. Evidencia consolidada y alcance  
- Segmento: Global (NPS 30.98 vs baseline 21.56; +9.42 pts)  
- Afecta a todos los subnodos bajo Global (Long Haul y Short Haul, todas las cabinas).  
- Incidentes operativos: 199 totales (34 cancelaciones, 35 retrasos, 11 problemas de equipaje) → elevado “friction score” que, sin embargo, no tradujo una caída global.  
- Ruta crítica: IAD–MAD con NPS –14.3 (n=7).  
- Perfiles: Leisure NPS 35.5 (n=558) vs Business 16.9 (n=178); Norteamérica 45.5, Europa 24.0, Latinoamérica 23.3, Asia –40.0.  
- Verbatims representativos:  
   • “La atención fue excelente y puntual” (Leisure, NA)  
   • “Retraso injustificado y mal manejo de equipaje” (Business, Europa)  
- Conclusión: la elevada satisfacción de Leisure y Norteamérica compensó holgadamente los puntos de fricción operativa, impulsando el NPS global.

2) Causa: Excelencia de servicio en Premium Long Haul  
A. Naturaleza de la causa  
- Mejora real en la experiencia de cabina Premium: confort, limpieza y atención de tripulación fueron percibidos como muy superiores al estándar.  
B. Evidencia consolidada y alcance  
- Segmento: Global / LH / Premium (NPS 47.37 vs baseline 14.25; +33.11 pts)  
- Afecta a todo lo que cuelga de LH / Premium (no hay subnodos adicionales).  
- Incidentes operativos del día: 24 (5 retrasos, 2 cancelaciones, 4 cambios de avión, 3 equipaje, 10 “otros”) → no impactaron la percepción Premium según verbatims.  
- Ruta analizada: BOG–MAD con NPS 100.0 y 0 incidencias.  
- Perfiles: Leisure 62.5 vs Business –33.3 (muestra muy sesgada hacia ocio).  
- Verbatims representativos:  
   • “La suite a bordo y el menú fueron increíbles” (Leisure)  
   • “No noté ningún retraso ni fallo en el servicio” (Leisure)  
- Conclusión: el repunte masivo se explica por el alto nivel de satisfacción de los pocos pasajeros Premium que volaron en rutas sin incidencias, más que por mejoras operativas globales.

3) Causa: Servicio y puntualidad en Business Short Haul  
A. Naturaleza de la causa  
- Alta consistencia en cumplimiento de horarios y trato de tripulación en vuelos de corta distancia, especialmente en aeronaves 32S.  
B. Evidencia consolidada y alcance  
- Segmento: Global / SH / Business (NPS 51.85 vs baseline 37.57; +14.29 pts)  
- Afecta a ambos subnodos Business / SH / IB y Business / SH / YW.  
- Incidentes operativos: 23 totales (8 cancelaciones, 4 retrasos, 4 cambios de avión) → en la ruta muestreada (MAD–PNA, n=3) no hubo ningún evento y se obtuvo NPS 100.  
- Flota: 32S NPS 75.0 vs A321 NPS 28.6 vs A350 NPS 33.3 → la excelencia de 32S marcó la diferencia.  
- Verbatims representativos:  
   • “Vuelo puntual y crew muy profesional” (Business, IB)  
   • “Todo perfecto en sala VIP y cabina” (Business, YW)  
- Conclusión: el fuerte ascenso se debe a una experiencia de alta calidad y puntualidad en la flota mejor gestionada, homologada en ambas compañías de corto radio.

Estos tres drivers explican conjuntamente el +9.4 pts de NPS global: un sesgo favorable en la muestra (causa 1) potenciado por picos de excelencia en Premium LH (causa 2) y Business SH (causa 3).

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

La compañía cerró el 20-oct-2025 con una subida de NPS global de 21.56 a 30.98 (+9.42 pts vs L7d), impulsada por aumentos en ambos radios. En Long Haul, Economy pasó de 16.58 a 25.59 (+9.01 pts) y Premium se disparó de 14.25 a 47.37 (+33.11 pts), mientras Business LH se mantuvo estable en 30.0 (+0.43 pts). En Short Haul, Economy escaló de 22.03 a 30.33 (+8.30 pts) con un repunte de IB/Economy de 16.09 a 25.0 (+8.91 pts) y estabilidad de YW/Economy en 40.0 (+6.61 pts vs L7d), y Business SH subió de 37.57 a 51.85 (+14.29 pts), tanto IB/Business SH (37.84→52.63, +14.79 pts) como YW/Business SH (35.73→50.0, +14.27 pts).  
Las causas principales fueron: un sesgo de muestra favorable a pasajeros Leisure y residentes en Norteamérica que compensó 199 incidencias operativas (34 cancelaciones, 35 retrasos), un pico de excelencia en Premium Long Haul gracias a la ruta BOG–MAD sin incidencias (NPS 100) y un sólido desempeño en Business Short Haul, especialmente en aviones 32S y en la ruta MAD–PNA (NPS 100), donde la puntualidad y la atención de tripulación destacaron en los verbatims.  

Las rutas más afectadas incluyen IAD–MAD (NPS –14.3, n=7) y MAD-OVD (NPS 0.0, n=6) en Short Haul, y MAD-PTY (NPS 12.5, n=8) en Economy Long Haul; en contraste, Premium LH brilló con BOG–MAD (NPS 100) y Business SH con MAD–PNA (NPS 100). Los viajeros Leisure mostraron la mayor reactividad con NPS hasta de 62.5 en Premium y 55.0 en Business SH, mientras Business LH y Asia en SH/Economy fueron los grupos menos flexibles frente a incidencias.

ECONOMY SH: Diferenciación por compañía  
La cabina Economy de SH experimentó una mejora de 22.03 a 30.33 pts (+8.30 pts vs L7d) el 20-oct-2025. IB/Economy impulsó el alza, subiendo de 16.09 a 25.0 (+8.91 pts) gracias a 370 verbatims que destacaron puntualidad y amabilidad de la tripulación pese a 23 incidencias operativas (8 cancelaciones, 4 cambios de avión, 1 demora de 6h30 en IB157). YW/Economy mantuvo su NPS en 40.0 (+6.61 pts vs L7d), sin cambios significativos. El avance de la cabina se vio atenuado en rutas como MAD-OVD y MAD-OPO (ambas con NPS 0.0) y los perfiles más reactivos fueron los viajeros Leisure (NPS 34.8) frente a Business (NPS 19.8).

BUSINESS SH: Alineación y flota 32S como motor  
La cabina Business de SH registró un alza de 37.57 a 51.85 pts (+14.29 pts vs L7d). IB/Business SH pasó de 37.84 a 52.63 (+14.79 pts) y YW/Business SH de 35.73 a 50.0 (+14.27 pts), impulsados por la experiencia en la flota 32S (NPS 75.0) y la ruta MAD–PNA (NPS 100, n=3) sin incidencias de 23 reportadas. Los verbatims elogiaron el cumplimiento de horarios y la calidad del servicio a bordo, siendo los pasajeros de 32S los más entusiastas.

ECONOMY LH: Mejora compartida por ocio  
La cabina Economy de LH escaló de 16.58 a 25.59 pts (+9.01 pts vs L7d) el 20-oct-2025. A pesar de 24 incidentes (2 cancelaciones, 5 retrasos, 1 desvío, 3 equipaje), los verbatims valoraron favorablemente la cortesía de la tripulación y la puntualidad, aunque la ruta MAD-PTY mostró un NPS bajo de 12.5 sin NCS formales. Los pasajeros Leisure alcanzaron 32.9 pts, mientras Business/LH se situó en –2.6 pts.

BUSINESS LH: Estabilidad en niveles altos  
La cabina Business de LH mantuvo desempeño estable con un NPS de 30.0 el 20-oct-2025 (+0.43 pts vs L7d). No se detectaron cambios significativos en las métricas de satisfacción pese a las 24 incidencias del día, mostrando coherencia con su baseline de 29.57 pts.

PREMIUM LH: Experiencia “5 estrellas”  
La cabina Premium de LH registró un salto de 14.25 a 47.37 pts (+33.11 pts vs L7d). El principal motor fue la percepción de confort y servicio en la ruta BOG–MAD (NPS 100) sin incidencias, según 41 verbatims que destacaron limpieza y menú a bordo. Aunque se documentaron 24 eventos operativos, ninguno impactó la valoración de los pasajeros Leisure (62.5 pts), en contraste con el escaso grupo Business (–33.3 pts).

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-10-19 to 2025-10-19:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

Economy Short Haul  
– Existen ambos nodos (IB y YW), ambos con anomalía positiva. Los análisis de drivers (feedback muy positivo de clientes no impactados y sesgo de composición de muestra) convergen en IB y en YW, por lo que la causa es común a la cabina Economy SH.  

Business Short Haul  
– Existen dos nodos, pero solo IB muestra anomalía negativa (-19,1 pts) mientras YW está en rango normal. La caída se localiza en IB, por lo que la causa es específica de la compañía Iberia (IB).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul: Economy y Business muestran patrones divergentes. Economy SH registra una anomalía positiva común a IB y YW, mientras que Business SH sufre una caída concentrada en IB y se mantiene estable en YW, lo que evidencia causas específicas tanto de cabina (Business) como de compañía (IB).  

Long Haul: Las tres cabinas reaccionan de modo divergente. Premium LH sube notablemente, Business LH cae drásticamente y Economy LH se mantiene dentro de rango, lo que apunta a drivers operativos y de muestra distintos por clase de servicio, es decir, causas específicas de cada cabina.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

La anomalía es específica de Short Haul: SH muestra un alza clara (+12,8 pts) impulsada por Economy y segmentos de alto desempeño, mientras que Long Haul se mantiene normal al compensarse Premium y Business. A nivel Global (+8,7 pts) se refleja principalmente el impulso de SH, sin que LH modifique sustancialmente el resultado.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

Causa 1: Sesgo de muestra y alta satisfacción en Short Haul Economy  
A. Naturaleza de la causa  
• Hipótesis: Aun con un número elevado de incidencias operativas, el NPS sube porque responde un mayor porcentaje de clientes no afectados (vuelos ATR/CRJ, Leisure, code-share VY, América Norte) que valoran muy positivamente el servicio.  

B. Evidencia consolidada y alcance  
• Segmento mayor afectado: Global/SH/Economy/IB (404 verbatims)  
• “Síntesis integral para Global/SH/Economy/IB” (NPS 29.90 vs baseline 16.09; +13.81 pts)  
  – Incidentes NCS: 38 (22 cancelaciones, 10 retrasos, 5 cambios de aeronave, 1 desvío)  
  – Rutas: MAD–VCE (n=8) NPS=12.5, sin NCS – indica feedback sesgado de pasajeros no impactados.  
  – Verbatims (404): tono 100 % positivo (“puntualidad excelente”, “trato impecable”) y ausencia de quejas operativas.  
• El mismo patrón se observa en Global/SH/Economy/YW (NPS 52.94 vs baseline 33.39; +19.55 pts)  
• Afecta a todos los subsegmentos bajo Global / SH / Economy (IB y YW)  

---  

Causa 2: Disrupciones operativas en Short Haul Business (Iberia)  
A. Naturaleza de la causa  
• Hipótesis: Los múltiples cambios de avión, cancelaciones y reubicaciones en flota A321 generan frustración en pasajeros Business de Iberia, cuyo feedback negativo reduce drásticamente el NPS.  

B. Evidencia consolidada y alcance  
• Segmento mayor afectado: Global/SH/Business/IB (47 verbatims)  
• “Síntesis integral para Global/SH/Business/IB” (NPS 18.75 vs baseline 37.84; –19.09 pts)  
  – Incidentes NCS: 38 (22 cancelaciones, 10 retrasos, 6 otros)  
  – Flota: A321 NPS –75.0 (n=4 encuestas) vs A320neo +42.9 (n=14)  
  – CodeShare: AA 100.0, BA 50.0, IB 18.2  
  – Verbatims (47): tono mayoritariamente positivo, pero muestra un sesgo de no recoger a los más afectados.  
  – Rutas: única con datos FCO–MAD (n=3) NPS=66.7, sin incidencias NCS – refuerza la falta de feedback de pasajeros impactados.  
• Afecta al único subsegmento bajo Global / SH / Business/IB  

---  

Causa 3: Composición de muestra premium y flota en Long Haul Premium  
A. Naturaleza de la causa  
• Hipótesis: Predomina la voz de clientes muy satisfechos (flota A350 next, viajeros Leisure y residentes en Europa/Centroamérica), lo que provoca un pico de NPS que eclipsa los incidentes operativos.  

B. Evidencia consolidada y alcance  
• Segmento mayor afectado: Global/LH/Premium – flota A350 next (n=5)  
• “Síntesis integral de la anomalía NPS del día 2025-10-19 para Global/LH/Premium”  
  – NPS 42.11 vs baseline 14.25; +27.85 pts  
  – Incidentes NCS: 28 (8 retrasos, 6 cancelaciones, 26 cambios de avión, 109 reubicaciones)  
  – Ruta: MAD–MEX (n=3) NPS=100.0, sin incidencias capturadas  
  – Verbatims (31): tono muy positivo (“horario perfecto”, “avión impecable”, “tripulación amable”), cero menciones a demoras.  
  – Región: Europa/Centroamérica NPS=100.0; Travel Type Leisure 46.7 vs Business 25.0  
• Afecta a todos los subsegmentos bajo Global / LH / Premium  

---  

Causa 4: Incidencias críticas en Long Haul Business  
A. Naturaleza de la causa  
• Hipótesis: Un pico de cancelaciones, retrasos y cambios de aeronave en vuelos críticos de Business LH provoca un desplome de la satisfacción.  

B. Evidencia consolidada y alcance  
• Segmento mayor afectado: Global/LH/Business (64 verbatims)  
• “Síntesis integral del día 2025-10-19 para Global/LH/Business”  
  – NPS 0.0 vs baseline 29.57; –29.57 pts  
  – Incidentes NCS: 28 (6 cancelaciones, 8 retrasos, 1 desvío, 2 limitaciones, 6 otras incidencias)  
  – Cambios de avión: 26; Reubicaciones en DFW: 109  
  – Ruta: JFK–MAD (n=4) NPS=50.0, sin correlato de NCS en esa ruta  
  – Verbatims (64): predominio de comentarios positivos en trato y catering, ausencia de feedback de los vuelos más impactados.  
  – Perfil: Business/Work NPS –8.7 vs Leisure +14.3; flota A33ACMI y A333 con NPS muy bajos  
• Afecta a todos los subsegmentos bajo Global / LH / Business

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
El análisis del 19-oct-2025 pone de manifiesto subidas y bajadas de NPS muy marcadas por cabina y radio. En Short Haul Economy (Global/SH/Economy) el NPS escaló de 22.03 pts (L7d) a 37.67 pts (+15.63), con Iberia subiendo de 16.09 a 29.90 (+13.81) y Vueling de 33.39 a 52.94 (+19.55), gracias a un fuerte sesgo de muestra —pasajeros Leisure en ATR/CRJ y code-share VY— que generó 404 verbatims 100 % positivos sin referencia a cancelaciones ni retrasos. En contraste, Business SH (Global/SH/Business) se dejó 9.50 pts, cayendo de 37.57 a 28.07; el impacto se localiza en Iberia (37.84→18.75, –19.09) por múltiples cancelaciones, retrasos y cambios de avión en A321, mientras Vueling mantuvo 40.00 (+4.27) dentro de rango normal. En Long Haul, Premium (Global/LH/Premium) registró un salto de 14.25 a 42.11 (+27.85), impulsado por flota A350 next y viajeros Leisure/Europa-Centroamérica que otorgaron NPS 100.0 en MAD–MEX, pese a 28 incidencias operativas. Por el contrario, Business LH (Global/LH/Business) colapsó de 29.57 a 0.00 (–29.57) debido a 28 incidencias (6 cancelaciones, 8 retrasos, 109 reubicaciones en DFW) que, al no quedar reflejadas en verbatims, arrastraron el promedio.  

En términos de rutas, MAD–MEX emergió con NPS 100.0 (n=3), MAD–VCE quedó en 12.5 (n=8) y FCO–MAD en 66.7 (n=3). Los grupos más reactivos fueron los pasajeros Leisure, los residentes en América Central y Norte, las flotas ATR/CRJ en SH y A350 next en LH, junto a los code-shares VY y AA, mientras que los usuarios de A321, A33ACMI y los pasajeros Business/Work mostraron la mayor insatisfacción, especialmente en Iberia.  

ECONOMY SH: Consolidación de la satisfacción por sesgo de muestra  
La cabina Global/SH/Economy cerró el 19-10-2025 con un NPS de 37.67 (15.63 pts más vs L7d de 22.03). Iberia alcanzó 29.90 (+13.81 vs 16.09 L7d) y Vueling 52.94 (+19.55 vs 33.39 L7d). La mejora se explica por un predominio de respuestas de viajeros Leisure en flotas ATR y CRJ, así como de pasajeros de code-share VY, que elogiaron trato y puntualidad en 404 verbatims sin mencionar cancelaciones ni retrasos. Aunque rutas como MAD–VCE (12.5) y MAD–NTE (0.0) mostraron niveles más bajos, no socavaron el alza general.  

BUSINESS SH: Caída localizada en Iberia  
El segmento Global/SH/Business registró un NPS de 28.07 el 19-10-2025, tras caer 9.50 pts vs L7d (37.57→28.07). Iberia sufrió un descenso de 37.84 a 18.75 (–19.09), impulsado por cancelaciones, retrasos y cambios de avión en A321, mientras Vueling se mantuvo en 40.00 (+4.27) sin desviaciones significativas. El efecto negativo se concentró en pasajeros Business/Work de A321, y el sesgo de muestra ocultó quejas en rutas como FCO–MAD (66.7, n=3).  

ECONOMY LH: Desempeño estable ante incidencias  
Global/LH/Economy mantuvo desempeño estable con un NPS de 21.79 (+5.21 vs L7d de 16.58) el 19-10-2025, dentro de rango normal. A pesar de 38 incidencias (22 cancelaciones, 10 retrasos), el feedback de 604 verbatims fue equilibrado, elogiando servicio y personal, sin quejas operativas significativas ni impacto en rutas.  

BUSINESS LH: Efecto letal de las disrupciones  
Global/LH/Business colapsó hasta 0.00 el 19-10-2025, desplomándose 29.57 pts vs L7d (29.57→0.00). Un pico de 28 incidencias —incluyendo 109 reubicaciones en DFW y cancelaciones críticas en vuelos como IB363 MAD–DFW— generó un feedback insuficiente (64 verbatims) que no reflejó directamente las quejas, arrastrando el NPS global. Las flotas A33ACMI y A333 y los pasajeros Business/Work fueron los más afectados.  

PREMIUM LH: Picos de excelencia en flota y perfil  
Global/LH/Premium subió de 14.25 a 42.11 (+27.85) el 19-10-2025, potenciado por la flota A350 next (NPS 100.0, n=5) y viajeros de ocio residentes en Europa/Centroamérica (NPS 100.0), como se refleja en la ruta MAD–MEX (100.0, n=3). Aun con 28 incidencias operativas, 31 verbatims alabaron la puntualidad, el estado de la aeronave y el trato de la tripulación.

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