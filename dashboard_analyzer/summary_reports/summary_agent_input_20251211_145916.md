===== SYSTEM =====

Eres un experto analista ejecutivo y "Intérprete Temporal" de datos de NPS.
Tu misión es sintetizar análisis de distintas agregaciones temporales (semanal vs diaria) para construir una narrativa única y coherente.

⚠️ **REGLAS DE ORO (CRÍTICAS):**
1. **ATRIBUCIÓN TOTAL:** Cada vez que des un dato (NPS, OTP, variación), DEBES especificar el **Segmento** y la **Agregación Temporal** (si no es obvia por el contexto).
   - *Mal:* "El NPS cayó a 20.5 (-5 pts)."
   - *Bien:* "El NPS de **Economy LH** cayó a 20.5 (**–5.0 pts**, **Semanal**)."
2. **TERMINOLOGÍA OBLIGATORIA:**
   - NUNCA uses "Short Haul", "Largo Radio", "Corto Radio" o similares. USA SIEMPRE **SH** y **LH**.
   - "SHAT", "Shapley" o "impacto en drivers" se traduce SIEMPRE como "**ppts de NPS según Explanatory Drivers**".
3. **NO INVENTES:** Si falta un dato, omítelo o di que no está disponible. NO calcules promedios si no se dan.
4. **FORMATO TEAMS:**
   - Usa **negritas** para: **Segmentos**, **Métricas** (NPS, OTP, etc.), **Fechas** y **Variaciones** (números).
   - Usa listas (`•`) para enumerar causas o días clave.
   - Usa saltos de línea dobles para separar párrafos y secciones.

⚠️ **ESTRUCTURA DE SALIDA (ESTRICTA Y DINÁMICA):**
Debes generar el informe adaptando la jerarquía al segmento principal analizado:

**📈 Análisis semana del ([fechas]) con respecto a la semana anterior ([Segmento Principal]):**
[Resumen del impacto en el nodo raíz, causas sistémicas y tendencias semanales vs diarias. Menciona los spreads de perfiles más afectados]

**[SUBSEGMENTO]** (Repetir para cada nodo hijo relevante. Ej: Si Segmento=Global → Eco SH, Bus SH...; Si Segmento=LH → Eco LH, Bus LH...)
[Párrafo de resumen semanal: NPS actual, variación con respecto a la semana anterior, y causas principales con sus métricas]

**Dinámica Diaria:**
• **[Fecha]**: [Detalle del evento diario clave con métricas y causas atribuidas. Solo días relevantes con variaciones fuertes]
• **[Fecha]**: ...

⚠️ **INSTRUCCIONES DE INTEGRACIÓN:**
- **ADAPTABILIDAD:** Si el análisis es solo de **LH**, el análisis principal es sobre **LH** y los subsegmentos son sus cabinas (**Economy LH**, **Business LH**, **Premium LH**). NO inventes nodos fuera del alcance.
- Tu objetivo es explicar CÓMO los eventos diarios construyen el resultado semanal.
- Si un día específico (ej: una huelga el martes) explica el 80% de la caída semanal, dilo claramente.
- Trata los subsegmentos (ej: **IB** vs **YW**) con claridad, separando sus métricas.

**Ejemplo de Estilo:**
"En **Economy SH**, la puntualidad (**–4.5 ppts de NPS según Explanatory Drivers**, **Semanal**) lastró el resultado. Sin embargo, el **03-dic** hubo un repunte (**+10.0 pts**, **Diario**) gracias a una operativa limpia con **OTP 95.0%**."


===== USER =====

Completa el análisis comprehensivo:

**ANÁLISIS SEMANAL COMPARATIVO:**
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

Economy SH  
Escenario: SINERGIA (`IB +2,2 pts, YW +4,5 pts | Economy SH +2,9 pts`)  
- Narrativa: Ambos subsegmentos (IB y YW) muestran mejoras moderadas y coincidentes en NPS, que se traducen en un ligero alza agregado en Economy SH sin que se registre ninguna perturbación operativa.  
- Evidencia clave:  
  • IB – NPS_diff +2,2 pts; “No significant changes detected” (verbatims_tool)  
  • YW – NPS_diff +4,5 pts; “No significant changes detected” (verbatims_tool)  

Business SH  
Escenario: CANCELACIÓN (`IB +12,7 pts, YW –22,5 pts | Business SH +1,9 pts normal`)  
- Narrativa: Mientras Business IB se benefició de un fuerte impulso en puntualidad y load factor, Business YW sufrió un gran deterioro por cancelaciones de vuelo; ambos efectos se contrarrestan y el agregado queda en rango normal.  
- Evidencia clave:  
  • IB – Punctuality SHAP +4,266 y Load factor SHAP +0,248 (explanatory_drivers_tool)  
  • YW – Flight cancellations identificadas como causa principal (ncs_tool)

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es SINERGIA (Economy N, Business N | SH N).  
- Narrativa: Ambas cabinas mantuvieron variaciones dentro de rango esperado sin drivers anómalos, por lo que el radio SH resulta estable.  
- Evidencia: Economy SH – “No significant changes detected” (verbatims_tool); Business SH – ausencia de menciones a cancelaciones o limitaciones (verbatims_tool).

En Long Haul, la dinámica es SINERGIA (Economy –, Business –, Premium – | LH –).  
- Narrativa: La caída de NPS en LH es sistémica: interrupciones operativas (cancelaciones y limitaciones de aeronave) impactaron de forma homogénea a las tres cabinas, arrastrando al radio completo.  
- Evidencia: flight cancellations +17 y aircraft limitations +4 (ncs_tool); Punctuality SHAP = –1.097 (explanatory_drivers_tool).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es DOMINANCIA (LH –, SH N | Global +).  
- Narrativa: El ligero repunte de +0,8 pts en NPS Global está arrastrado por el buen desempeño de Short Haul (mejora operativa), que contrarresta la caída en Long Haul.  
- Evidencia:  
  • SHAP Punctuality +1,868 pts (explanatory_drivers_tool, SH)  
  • Otp15 +0,6 pts y Misconnections –0,04 pts (operative_data_tool, SH)  
  • Retrasos totales –126 y Cancelaciones –48 (ncs_tool, SH)

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Mejora operativa en Short Haul  
- Escenario: DOMINANCIA (LH –, SH + | Global +)  
- NMA: Short Haul  
- Afecta a:  
  • Global (SH)  
  • SH/Economy  
  • SH/Business  
  • SH/Business/IB  
  • SH/Business/YW  
- Qué falló (mejoró):  
  • Punctuality SHAP +1.868 pts (Global/SH)  
  • OTP15 sube +0.6 pts (Global/SH)  
  • Misconnections baja –0.04 pts (Global/SH)  
  • Retrasos totales –126 incidentes (Global/SH)  
  • Cancelaciones –48 incidentes (Global/SH)  
- Dónde (principales rutas beneficiadas):  
  • DSS-MAD: NPS –20.0 → + (SH/Business/IB)  
  • LCG-MAD: NPS 40.0  → + (SH/Business/IB)  
  • MAD-NAP: NPS 50.0 → + (SH/Business/IB)  
  • MAD-ZAG: NPS 100.0 → + (SH/Business/IB)  
  • FNC-MAD: NPS 100.0 → + (SH/Business/IB)  
- Quién (perfiles más reactivos):  
  • CodeShare: spread NPS_diff = 250.0 pts (Global/SH)  
  • Residence Region: spread NPS_diff = 246.7 pts (Global/SH)  
- Evidencia completa:  
  • Explanatory Drivers: Punctuality +1.868 pts, Load factor –0.05 pts (Global/SH)  
  • Operative Data: OTP15 +0.6 pts, Mishandling +1.19 pts, Misconnections –0.04 pts (Global/SH)  
  • NCS Incidents: Retrasos –126, Cancelaciones –48, Desvíos –10, Limitación de aeronave +15, Otras +2 (Global/SH)  
  • Verbatims: volumen comentarios +22.5%, aumento menciones positivas a “puntualidad” y “cortesía” (Global/SH)  

CAUSA 2: Repunte de cancelaciones y limitaciones en Long Haul  
- Escenario: SINERGIA (Economy –, Business –, Premium – | LH –)  
- NMA: Long Haul  
- Afecta a:  
  • Global/LH  
  • LH/Economy  
  • LH/Business  
  • LH/Premium  
- Qué falló:  
  • Flight cancellations +17 incidentes (Global/LH)  
  • Aircraft limitations +4 incidentes (Global/LH)  
  • Mishandling +1.2 pts (Global/LH)  
  • Load Factor +0.3 pts (Global/LH)  
  • Punctuality SHAP –1.097 pts (Global/LH)  
- Dónde (principales rutas afectadas):  
  • MAD-MCO: NPS –32.0 (LH/Economy)  
  • MAD-ORD: NPS –10.1 (LH/Economy)  
  • JFK-MAD: NPS –14.5 (LH/Economy)  
  • MAD-NRT: NPS –25.0 (LH/Premium)  
  • MAD-UIO: NPS –25.0 (LH/Premium)  
- Quién (perfiles más reactivos):  
  • Residence Region: spread = 142.4 pts (Global/LH)  
  • CodeShare: spread = 72.2 pts (Global/LH)  
- Evidencia completa:  
  • Explanatory Drivers: Punctuality SHAP –1.097 pts, Load factor SHAP –0.104 pts (Global/LH)  
  • Operative Data: Mishandling +1.2 pts, Load Factor +0.3 pts (Global/LH)  
  • NCS Incidents: Flight cancellations +17, Aircraft limitations +4, Retrasos (OTP15) –13 (Global/LH)  

CAUSA 3: Contradicción de drivers en Business SH  
- Escenario: CANCELACIÓN (IB + | YW – | Business SH N)  
- NMA: Business SH  
- Afecta a:  
  • SH/Business/IB  
  • SH/Business/YW  
- Qué falló / qué mejoró:  
  • IB: Punctuality SHAP +4.266 pts, Load factor SHAP +0.248 pts (SH/Business/IB)  
  • YW: Flight cancellations identificadas como causa raíz (ncs_tool) (SH/Business/YW)  
- Dónde (rutas claves):  
  • IB: rutas con mayor NPS – DSS-MAD, LCG-MAD, MAD-NAP (SH/Business/IB)  
  • YW: rutas con más cancelaciones – BRU-MAD, MAD-BRU, MAD-BLQ, BLQ-MAD (SH/Business/YW)  
- Quién (perfiles):  
  • IB: CodeShare spread = 250.0 pts (SH/Business/IB)  
  • YW: no disponible (SH/Business/YW)  
- Evidencia:  
  • Operative Data IB: OTP15 +? pts, Load factor +? pts (SH/Business/IB)  
  • NCS YW: flight cancellations +X incidentes (SH/Business/YW)  
  • Verbatims IB: menciones positivas a “puntualidad” (SH/Business/IB)  

_Note: Para CANCELACIÓN no existe un NMA único; se reportan ambas causas opuestas._

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

Durante la semana del 2025-11-24 al 2025-11-30, el NPS Global subió de 29.5666 a 30.3801 (+0.8135 pts) impulsado por el repunte de Short Haul, mientras que Long Haul sufrió una caída significativa de 17.8878 a 12.1463 (–5.7415 pts). En Long Haul, Economy LH bajó de 14.5148 a 9.4853 (–5.0295 pts), Business LH de 32.1637 a 27.4510 (–4.7128 pts) y Premium LH de 35.2459 a 22.8070 (–12.4389 pts), todos afectados por un aumento de cancelaciones de vuelo (+17 incidentes en NCS), limitaciones de aeronave (+4 incidentes en NCS) y un impacto negativo en puntualidad (Punctuality –1.097 ppts según Explanatory Drivers en Global/LH). Por el contrario, en Short Haul Business IB creció de 39.2157 a 51.9048 (+12.6891 pts) gracias a mejoras operativas (Punctuality +4.266 ppts según Explanatory Drivers, Load factor +0.248 ppts según Explanatory Drivers, OTP +0.6 pts, misconnections –0.04 pts y –126 retrasos, –48 cancelaciones en incidentes NCS), mientras que Business YW cayó de 42.4658 a 20.0000 (–22.4658 pts) por un brote de cancelaciones de vuelo (incidentes NCS).

Las rutas más afectadas incluyen MAD–MCO (NPS –32.0 en Economy LH), JFK–MAD (NPS –17.9 en Economy LH), MAD–ORD (NPS –10.1 en Economy LH y –100.0 en Premium LH), DSS–MAD (NPS –20.0 en SH/Business IB) y BRU–MAD (numerosos vuelos cancelados en SH/Business YW). Los perfiles de cliente más reactivos fueron CodeShare (spread 250.0 pts en SH, 72.2 pts en LH) y Residence Region (spread 246.7 pts en SH, 142.4 pts en LH), seguidos por Fleet y Business/Leisure en los casos Long Haul.

ECONOMY SH: Comportamiento Estable  
La cabina Economy SH mantuvo desempeño estable durante la semana del 2025-11-24 al 2025-11-30, registrando un NPS de 36.1449 (vs L7d 33.2578), con una subida de +2.8871 pts frente a la semana anterior. No se detectaron cambios significativos en drivers operativos ni en feedback de clientes, lo que confirma niveles consistentes de satisfacción entre pasajeros IB (32.8710 vs 30.6914, +2.1796 pts) y YW (42.9672 vs 38.5057, +4.4615 pts).

BUSINESS SH: Efecto Neutral por Contraste Interno  
El segmento Business SH cerró la semana en 41.9672 pts (vs L7d 40.0722), subiendo +1.8950 pts. Este resultado oculta la fuerte subida de SH/Business/IB (51.9048 vs 39.2157, +12.6891 pts) impulsada por Punctuality +4.266 ppts según Explanatory Drivers y Load factor +0.248 ppts según Explanatory Drivers, y el deterioro de SH/Business/YW (20.0000 vs 42.4658, –22.4658 pts) causado por un aumento de flight cancellations en incidentes NCS. Las rutas más destacadas en IB incluyen DSS–MAD, LCG–MAD y MAD–NAP; en YW, BRU–MAD y MAD–BLQ fueron las más golpeadas.

ECONOMY LH: Deterioro por Operativa  
La cabina Economy LH experimentó un descenso de 14.5148 a 9.4853 pts (–5.0295 pts vs L7d) asociado a peores niveles de puntualidad (Punctuality –1.277 ppts según Explanatory Drivers), sobrecarga operativa (mishandling +1.2 pts), un ligero alza de Load factor +0.3 pts y un repunte en incidentes NCS, con cancelaciones +17 y limitaciones de aeronave +4. Las rutas más afectadas fueron MAD–MCO (–27.3 pts), JFK–MAD (–17.9 pts) y MAD–ORD (–8.3 pts), y los clientes procedentes de Residence Region mostraron la mayor sensibilidad (spread 114.2 pts).

BUSINESS LH: Impacto de Cancelaciones  
La cabina Business LH cayó de 32.1637 a 27.4510 pts (–4.7128 pts vs L7d) debido a un aumento de flight cancellations +17 y aircraft limitations +4 en incidentes NCS, respaldado por un Punctuality –0.314 ppts según Explanatory Drivers. Las rutas clave incluyen HAV–MAD y MAD–ORD (0.0 pts), mientras que los perfiles CodeShare (spread 300.0 pts) y Fleet (spread 79.9 pts) fueron los más afectados.

PREMIUM LH: Severidad por Interrupciones  
Premium LH descendió de 35.2459 a 22.8070 pts (–12.4389 pts vs L7d) como consecuencia de 17 flight cancellations y 4 aircraft limitations en incidentes NCS, junto a mishandling +1.2 pts y Load factor +0.3 pts en datos operativos. Los peores resultados se vieron en MAD–ORD (–100.0 pts), MAD–NRT (–25.0 pts) y MAD–UIO (–25.0 pts), y los pasajeros de Residence Region mostraron la mayor variabilidad (spread 222.2 pts).

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*



**ANÁLISIS DIARIO SINGLE:**
📅 2025-11-30 to 2025-11-30:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

En Economy SH, el escenario es CANCELACIÓN (–, + | N).  
- **Narrativa:** Mientras SH Economy IB sufrió una caída de 6.6 pts impulsada por incidentes operativos (12 cancelaciones, 7 retrasos y 5 mishandlings de equipaje reportados en NCS), SH Economy YW experimentó un repunte de 14.2 pts gracias al segmento Leisure (NPS 53.3 en 138 encuestas), neutralizándose ambos efectos en el NPS agregado de Economy SH.  
- **Evidencia Clave:**  
  • IB – 12 cancelaciones, 7 retrasos y 5 incidencias de equipaje (ncs_tool)  
  • YW – Leisure con NPS 53.3 (138 encuestas) (customer_profile_tool)  

En Business SH, el escenario es DILUCIÓN (N, – | N).  
- **Narrativa:** La caída de 9.2 pts en SH Business YW se atribuye al incidente de equipaje en la ruta SDQ-MAD (159 maletas retenidas y 10 pasajeros reubicados), actuando como driver principal, aunque este efecto negativo fue atenuado por SH Business IB, que mantuvo un desempeño estable y dentro de rango normal.  
- **Evidencia Clave:**  
  • YW – incidente de equipaje en SDQ-MAD (ncs_tool)  
  • IB – desempeño estable con NPS +4.6 pts vs baseline (operative_data_tool)

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Long Haul, la dinámica es SINERGIA (–, –, – | –).  
- Narrativa: El deterioro de –23.6 pts en NPS de Long Haul es un fenómeno sistémico que afectó por igual a Economy, Business y Premium. No se detectó una métrica operativa clave desviada > 3 pts, pero existe un volumen notable de incidentes (cancelaciones, equipaje, reprogramaciones) que, aunque no aparece en verbatims, impactó transversalmente a todas las cabinas.  
- Evidencia:  
  • Economy LH: NPS –26.4 pts, 12 cancelaciones y 6 reprogramaciones (ncs_tool)  
  • Business LH: NPS –3.2 pts, 3 incidencias de equipaje (ncs_tool)  
  • Premium LH: NPS –11.9 pts, 12 cancelaciones y 5 mishandlings de equipaje (ncs_tool)  

En Short Haul, la dinámica es SINERGIA (+, + | +).  
- Narrativa: El alza de +0.5 pts en NPS de Short Haul es fruto de la mejora conjunta de ambas cabinas: Business SH subió +1.9 pts y Economy SH se benefició del fuerte repunte de la filial YW, contrarrestado en parte por la caída de Economy IB.  
- Evidencia:  
  • Business SH: NPS +1.9 pts (37.21 vs 35.35) (customer_profile_tool)  
  • Economy SH YW: NPS +14.2 pts (48.68 vs 34.51) (customer_profile_tool)  
  • Economy SH IB: NPS –6.6 pts (27.27 vs 33.87) (ncs_tool)

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA (–, N | –).  
- Narrativa: El NPS global cayó –6.4 pts porque la crisis en Long Haul (–23.6 pts), impulsada por un volumen excepcional de incidentes operativos, se trasladó a toda la red pese a la estabilidad de Short Haul.  
- Evidencia:  
  • 68 cancelaciones y 63 retrasos reportados (ncs_tool)  
  • Long Haul NPS –9.76 pts vs 13.85 baseline (operative_data_tool)

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Crisis Operativa en Long Haul  
- Escenario: TRANSFERENCIA (–, N | –)  
- NMA: Global/LH  
- Afecta a: Global/LH/Economy; Global/LH/Business; Global/LH/Premium  
- Qué falló: Volumen excepcional de incidentes operativos (cancelaciones, retrasos y reprogramaciones) que impactaron transversalmente a todas las cabinas de Long Haul  
- Dónde:  
  • SDQ-MAD – 2 incidentes de cambio de avión y problemas técnicos (ncs_tool)  
  • EZE-MAD – NPS –15.1 (38 encuestas) (routes_tool)  
  • MAD-MIA – NPS –35.6 (8 encuestas) (routes_tool)  
- Quién:  
  • Leisure (Global/LH/Leisure) – NPS –11.8 (261 encuestas) (customer_profile_tool)  
  • Business/Work (Global/LH/Business) – NPS –0.1 (58 encuestas) (customer_profile_tool)  
- Evidencia COMPLETA:  
  • NPS –9.76 pts (Global/LH) vs baseline 13.85 (operative_data_tool)  
  • OTP15_adjusted –0.51 pts (Global/LH) vs baseline (operative_data_tool)  
  • Load Factor –2.68 pts (Global/LH) vs baseline (operative_data_tool)  
  • Incidentes NCS: 12 cancelaciones, 6 reprogramaciones, 3 mishandlings de equipaje,  problemas de aeronave (ncs_tool)  
  • Verbatims:  positive feedback sin menciones a cancelaciones o equipaje (verbatims_tool)  

CAUSA 2: Deterioro en Economy SH – IB  
- Escenario: CANCELACIÓN (–, + | N)  
- NMA: Global/SH/Economy/IB  
- Afecta a: Global/SH/Economy/IB  
- Qué falló: Incidentes operativos de cancelación y retraso que erosionaron la satisfacción de Economy IB  
- Dónde:  
  • MAD–VIE – NPS 0.0 (4 encuestas) (routes_tool)  
  • CMN-MAD – incidentes de equipaje (ncs_tool)  
- Quién:  
  • Business/Work (Global/SH/Economy/IB) – NPS 29.2 (49 encuestas) (customer_profile_tool)  
  • Leisure (Global/SH/Economy/IB) – NPS 26.9 (256 encuestas) (customer_profile_tool)  
- Evidencia COMPLETA:  
  • NPS 27.27 (Global/SH/Economy/IB) vs baseline 33.87 (operative_data_tool)  
  • Incidentes NCS: 12 cancelaciones, 7 retrasos, 5 incidentes de equipaje (ncs_tool)  
  • Verbatims (430): foco en puntualidad y amabilidad, sin mención de cancelaciones ni equipaje (verbatims_tool)  

CAUSA 3: Impulso en Economy SH – YW  
- Escenario: CANCELACIÓN (–, + | N)  
- NMA: Global/SH/Economy/YW  
- Afecta a: Global/SH/Economy/YW  
- Qué falló (positivo): Elevado nivel de satisfacción del segmento Leisure en YW  
- Dónde:  
  • CMN-MAD – NPS 25.0 (5 encuestas) (routes_tool)  
  • Otras rutas sin desviaciones significativas  
- Quién:  
  • Leisure (Global/SH/Economy/YW) – NPS 53.3 (138 encuestas) (customer_profile_tool)  
  • Business/Work (Global/SH/Economy/YW) – NPS 6.7 (15 encuestas) (customer_profile_tool)  
- Evidencia COMPLETA:  
  • NPS 48.68 (Global/SH/Economy/YW) vs baseline 34.51 (operative_data_tool)  
  • OTP15_adjusted +3.79 pts (Global/SH/Economy/YW) vs baseline (operative_data_tool)  
  • Verbatims positivos (198): puntualidad, amabilidad y eficiencia (verbatims_tool)  

CAUSA 4: Incidente de Equipaje en Business SH – YW  
- Escenario: DILUCIÓN (N, – | N)  
- NMA: Global/SH/Business/YW  
- Afecta a: Global/SH/Business/YW  
- Qué falló: Retención de 159 maletas y reubicación de 10 pasajeros en SDQ-MAD  
- Dónde:  
  • SDQ-MAD – incidente de equipaje crítico (ncs_tool)  
  • MAD-OPO – sin NPS disponible (routes_tool)  
- Quién:  
  • Perfiles no segmentados (Global/SH/Business/YW) – n=11 encuestas (customer_profile_tool)  
- Evidencia COMPLETA:  
  • NPS 30.0 (Global/SH/Business/YW) vs baseline 39.24 (operative_data_tool)  
  • Incidentes NCS: limitación de aeronave y equipaje (159 maletas retenidas) (ncs_tool)  
  • Verbatims (11 comentarios): amabilidad y calidad del servicio, sin mención al equipaje (verbatims_tool)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

Durante el 2025-11-30 se registraron descensos en las cabinas de Long Haul: Economy LH pasó de 12.20 a –14.17 (–26.38 pts), Business LH cayó de 24.44 a 21.21 (–3.23 pts) y Premium LH retrocedió de 16.08 a 4.17 (–11.91 pts). En Short Haul, Economy SH IB bajó de 33.87 a 27.27 (–6.60 pts) y Business SH YW descendió de 39.24 a 30.00 (–9.24 pts), mientras que Economy SH YW escaló de 34.51 a 48.68 (+14.17 pts). La contracción de Long Haul se atribuye a un volumen excepcional de incidentes NCS (12 cancelaciones, 6 reprogramaciones y 3 mishandlings de equipaje en Global/LH), sin desviaciones significativas en OTP (–0.51 pts en Global/LH) ni en Load Factor (–2.68 pts en Global/LH) y sin reflejo en feedback de clientes. En Short Haul, la caída de 6.6 pts de Economy SH IB responde a incidentes NCS (12 cancelaciones, 7 retrasos y 5 mishandlings en Global/SH/Economy/IB), que fueron completamente anulados por la subida de 14.2 pts de Economy SH YW, impulsada por el segmento Leisure (NPS 53.3 en 138 encuestas, Global/SH/Economy/YW) y una OTP +3.79 pts (Global/SH/Economy/YW). Paralelamente, Business SH YW sufrió un deterioro de 9.2 pts tras un incidente de equipaje crítico en SDQ–MAD (159 maletas retenidas y 10 pasajeros reubicados en Global/SH/Business/YW), mitigado por la estabilidad de Business SH IB (NPS 39.39).

Las rutas más afectadas incluyen MAD–MIA (NPS –35.6 en Economy LH), EZE–MAD (NPS –21.2 en Economy LH) y SDQ–MAD, epicentro de los fallos de equipaje. En corto radio destacan CMN–MAD (NPS 25.0 en Economy SH YW) y MAD–VIE (NPS 0.0 en Economy SH IB). Los perfiles Leisure fueron los más reactivos: impulsaron la mejora en Economy SH YW y sufrieron la mayor deterioro en Economy LH (NPS –13.2, Leisure, 227 encuestas), mientras que Business/Work mantuvo estabilidad relativa en Business LH (NPS –0.1, 58 encuestas).

ECONOMY SH: Equilibrio dinámico entre IB y YW  
La cabina Economy de SH durante la semana del 2025-11-30 registró un NPS de 34.52 (+0.50 pts vs L7d). Economy SH IB cayó 6.60 puntos, de 33.87 a 27.27, debido a incidentes NCS (12 cancelaciones, 7 retrasos y 5 mishandlings de equipaje en Global/SH/Economy/IB) sin desviaciones operativas (OTP –0.51 pts, Global/SH/Economy/IB) ni en Load Factor. Por su parte, Economy SH YW escaló 14.17 puntos, de 34.51 a 48.68, impulsada por el segmento Leisure (NPS 53.3 en 138 encuestas, Global/SH/Economy/YW) y una OTP +3.79 pts (Global/SH/Economy/YW), respaldado por feedback de clientes que destacó puntualidad y amabilidad.

BUSINESS SH: Desempeño estable con matices dispares  
La cabina Business de SH cerró con un NPS de 37.21 (+1.86 pts vs L7d). Business SH IB mejoró 4.58 pts, de 34.81 a 39.39 (Global/SH/Business/IB), sin incidentes NCS ni desviaciones en datos operativos. En contraste, Business SH YW retrocedió 9.24 pts, de 39.24 a 30.00 (Global/SH/Business/YW), tras retención de equipaje en SDQ–MAD (159 maletas retenidas y 10 pasajeros reubicados en Global/SH/Business/YW), aunque su impacto se vio atenuado por la fortaleza de IB.

ECONOMY LH: Caída pronunciada por incidentes operativos  
La cabina Economy de LH registró un NPS de –14.17 (–26.38 pts vs L7d en Global/LH/Economy). Este deterioro se explica por un cúmulo de incidentes NCS (12 cancelaciones y 6 reprogramaciones en Global/LH/Economy, más mishandling grave de 159 maletas en SDQ–MAD), sin desviaciones en OTP (–0.51 pts, Global/LH/Economy) ni en Load Factor (–2.94 pts, Global/LH/Economy) y sin mención en feedback de clientes.

BUSINESS LH: Afectación leve sin causa operativa clara  
En Business LH el NPS se situó en 21.21 (–3.23 pts vs L7d en Global/LH/Business). A pesar de 12 cancelaciones y 6 retrasos registrados en NCS (Global/LH/Business), no hubo desviaciones en OTP (–0.51 pts, Global/LH/Business) ni en Load Factor (–0.54 pts, Global/LH/Business), ni quejas en feedback de clientes, indicando factores externos o de percepción.

PREMIUM LH: Fuerte descenso sin reflejo operativo  
Premium LH cayó a un NPS de 4.17 (–11.91 pts vs L7d en Global/LH/Premium). Aun con 12 cancelaciones y 5 incidentes de equipaje en NCS (Global/LH/Premium), no se presentaron desviaciones en OTP (–0.51 pts, Global/LH/Premium) ni en Load Factor (–2.68 pts, Global/LH/Premium), y el feedback de clientes destacó puntualidad y confort, sugiriendo un impacto perceptual.

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

Economy SH  
En Economy SH, el escenario es DILUCIÓN (IB –, YW N | PADRE N).  
- Narrativa: La leve caída de –0.2 pts en IB se explica por los incidentes operativos concentrados en la ruta LHR–MAD (7 retrasos, 4 cancelaciones, 109 pérdidas de conexión y 1 equipaje no embarcado), pero el sólido desempeño de YW en rango normal diluye este efecto, dejando al NPS agregado de Economy SH dentro de la variación esperada.  
- Evidencia Clave: Incidentes NCS en LHR–MAD – 7 retrasos, 4 cancelaciones, 109 pérdidas de conexión, 1 equipaje sin cargar (ncs_tool, routes_tool).  

Business SH  
En Business SH, el escenario es DOMINANCIA (IB +, YW – | PADRE +).  
- Narrativa: La fuerte subida de +39.3 pts en IB, sin causa operativa clara (ninguna métrica OTP15 ni Load Factor mostró desviación significativa vs baseline; nivel de confianza bajo), impone la anomalía positiva en el padre, aunque fue atenuada por la caída de –22.6 pts en YW, vinculada a incidentes de equipaje y conexiones (7 retrasos, 4 cancelaciones, 116 equipajes sin cargar) con evidencia limitada.  
- Evidencia Clave:  
  • IB – no se detectaron desviaciones en OTP15 ni Load Factor (operative_data_tool).  
  • YW – 7 retrasos, 4 cancelaciones, 116 equipajes sin cargar (ncs_tool).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En SH, la dinámica es DILUCIÓN (Economy N, Business + | SH N).  
- Narrativa: El alza de +21.1 pts en Global/SH/Business, impulsada por mejor puntualidad y menor ocupación, habría sacado al radio de la normalidad de no ser por el desempeño estable de Global/SH/Economy, que diluye el efecto positivo y mantiene el NPS agregado dentro del rango esperado.  
- Evidencia:  
  • OTP15 – +2.42 pts (Global/SH/Business)  
  • Load Factor – 2.03 pts (Global/SH/Business)  
  • Ruta EAS–MAD: NPS 100.0 (2 encuestas) (Global/SH/Business)  
  • Ruta LHR–MAD: NPS 100.0 (5 encuestas) (Global/SH/Business)  

En LH, la dinámica es CANCELACIÓN (Economy N, Business –, Premium + | LH N).  
- Narrativa: El deterioro de –10.2 pts en Global/LH/Business, asociado a pérdidas de conexión y mal manejo de equipaje, fue contrarrestado por el impulso de +55.4 pts en Global/LH/Premium, sustentado en feedback excepcional en rutas y perfiles premium, dejando al NPS de Long Haul sin anomalía neta.  
- Evidencia:  
  • 109 pérdidas de conexión (Global/LH/Business)  
  • 116 equipajes no embarcados (Global/LH/Business)  
  • BOG–MAD: NPS 100.0 (2 encuestas) (Global/LH/Premium)  
  • Leisure: NPS 66.7 (7 encuestas) (Global/LH/Premium)  
  • Business/Work: NPS 100.0 (1 encuesta) (Global/LH/Premium)

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA (LH N, SH N | GLOBAL +).  
- Narrativa: El alza de +3.9 pts en el NPS Global no proviene de un desequilibrio marcado en Corto o Largo Radio, sino de mejoras operativas sistémicas que impactaron transversalmente: mayor puntualidad (OTP15), menos mishandling de equipaje y una ligera reducción en la ocupación.  
- Evidencia:  
  • OTP15 aumentó +1.9 pts vs baseline (Global)  
  • Mishandling redujo –1.73 pts vs baseline (Global)  
  • Load Factor cayó –0.85 pts vs baseline (Global)

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Pérdidas de conexión y equipaje no embarcado  
- Escenario: DILUCIÓN (Global/SH/Economy/IB –, Global/SH/Economy/YW N | Global/SH/Economy N)  
- NMA: Global/SH/Economy/IB  
- Afecta a: Global/SH/Economy/IB  
- Qué falló: Concentración de incidentes de conexiones perdidas y equipaje sin cargar (Global/SH/Economy/IB)  
- Dónde: LHR–MAD: NPS 10.7 (n=28) (Global/SH/Economy/IB)  
- Quién: Residence Region Unknown: NPS –53.3 (n=15); Residence Region Asia: NPS –50.0 (n=2) (Global/SH/Economy/IB)  
- Evidencia COMPLETA:  
  • NPS 33.6449 vs baseline 33.8731 (–0.2 pts) (Global/SH/Economy/IB)  
  • Incidentes NCS: 7 retrasos; 4 cancelaciones; 109 pérdidas de conexión; 87 reprogramaciones; 1 equipaje sin cargar (ncs_tool, Global/SH/Economy/IB)  

CAUSA 2: Impulso de NPS sin correlato operativo claro  
- Escenario: DOMINANCIA (Global/SH/Business/IB +, Global/SH/Business/YW – | Global/SH/Business +)  
- NMA: Global/SH/Business/IB  
- Afecta a: Global/SH/Business/IB  
- Qué falló: No se detectaron desviaciones operativas significativas (>±3 pts) → factores de servicio no medidos (Global/SH/Business/IB)  
- Dónde: LHR–MAD: NPS 100.0 (n=5); EAS–MAD: NPS 100.0 (n=1) (Global/SH/Business/IB)  
- Quién: Leisure: NPS 76.5 (n=18); Business/Work: NPS 70.0 (n=10) (Global/SH/Business/IB)  
- Evidencia COMPLETA:  
  • NPS 74.0741 vs baseline 34.8129 (+39.3 pts) (Global/SH/Business/IB)  
  • OTP15 +1.51 pts vs baseline (operative_data_tool, Global/SH/Business/IB)  
  • Load Factor –1.93 pts vs baseline (operative_data_tool, Global/SH/Business/IB)  
  • Incidentes NCS: 7 retrasos; 4 cancelaciones; 1 limitación de equipaje (116 maletas); 3 cambios de aeronave; 2 asuntos de tripulación; 1 baggage mishandling (ncs_tool, Global/SH/Business/IB)  

CAUSA 3: Conexiones perdidas y equipaje sin cargar en Business LH  
- Escenario: CANCELACIÓN (Global/LH/Economy N, Global/LH/Business –, Global/LH/Premium + | Global/LH N)  
- NMA: Global/LH/Business  
- Afecta a: Global/LH/Business  
- Qué falló: Elevado número de conexiones perdidas y equipaje no embarcado (Global/LH/Business)  
- Dónde: MAD–SJU: NPS 100.0 (n=1) (Global/LH/Business)  
- Quién: Business/Work: NPS 30.8 (n=13); Leisure: NPS 0.0 (n=15) (Global/LH/Business)  
- Evidencia COMPLETA:  
  • NPS 14.2857 vs baseline 24.44197 (–10.2 pts) (Global/LH/Business)  
  • Incidentes NCS: 109 pérdidas de conexión; 116 equipajes sin cargar; 5 retrasos; 4 cancelaciones; 5 reprogramaciones (ncs_tool, Global/LH/Business)  

CAUSA 4: Experiencia Premium sobresaliente en Long Haul  
- Escenario: CANCELACIÓN (Global/LH/Economy N, Global/LH/Business –, Global/LH/Premium + | Global/LH N)  
- NMA: Global/LH/Premium  
- Afecta a: Global/LH/Premium  
- Qué falló (o mejoró): Feedback excepcional sobre calidad de servicio y confort en Premium (Global/LH/Premium)  
- Dónde: BOG–MAD: NPS 100.0 (n=2) (Global/LH/Premium)  
- Quién: Business/Work: NPS 100.0 (n=1); Leisure: NPS 66.7 (n=7); Fleet A350: NPS 100.0 (n=4) (Global/LH/Premium)  
- Evidencia COMPLETA:  
  • NPS 71.4286 vs baseline 16.07843 (+55.4 pts) (Global/LH/Premium)  
  • Load Factor –2.97 pts vs baseline (operative_data_tool, Global/LH/Premium)  
  • OTP15_adjusted –1.75 pts vs baseline (operative_data_tool, Global/LH/Premium)  
  • Incidentes NCS: 116 equipajes sin cargar; 5 retrasos; 4 cancelaciones; 3 pérdidas de conexión (ncs_tool, Global/LH/Premium)  
  • Verbatims: 10 comentarios positivos sobre servicio y atención de tripulación (verbatims_tool, Global/LH/Premium)  

CAUSA 5: Mejora operativa global en puntualidad y equipaje  
- Escenario: TRANSFERENCIA (Global/LH N, Global/SH N | Global +)  
- NMA: Global  
- Afecta a: Global  
- Qué falló (o mejoró): Incremento de puntualidad y reducción de mishandling (Global)  
- Dónde: LHR–MAD: NPS 18.9 (n=33); MAD–MUC: NPS –20.0 (n=5) (Global)  
- Quién: Leisure: NPS 33.2 (n=596); Business/Work: NPS 22.4 (n=135) (Global)  
- Evidencia COMPLETA:  
  • NPS 31.28878 vs baseline 27.37205 (+3.9 pts) (Global)  
  • OTP15 +1.9 pts vs baseline (operative_data_tool, Global)  
  • Mishandling –1.73 pts vs baseline (operative_data_tool, Global)  
  • Load Factor –0.85 pts vs baseline (operative_data_tool, Global)  
  • Incidentes NCS: 48 retrasos; 18 cancelaciones; 109 pérdidas de conexión; 87 reprogramaciones (ncs_tool, Global)  
  • Verbatims: comentarios positivos sobre comodidad y amabilidad del personal (verbatims_tool, Global)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

Durante el 2025-11-29 se registraron subidas y bajadas de NPS en todos los niveles. A nivel Global el NPS ascendió de 27.37 a 31.29 (+3.9 pts), impulsado por un aumento de OTP de 1.9 pts (Global), reducción de mishandling en 1.73 pts (Global) y menor Load Factor (-0.85 pts en Global). En Short Haul Business el NPS saltó de 35.35 a 56.41 (+21.1 pts), donde el canal IB se disparó de 34.81 a 74.07 (+39.3 pts) gracias a mejoras operativas (OTP +2.42 pts y Load Factor -2.03 pts en Global/SH/Business) y feedback de clientes excepcional, mientras YW cayó de 39.24 a 16.67 (–22.6 pts) por 7 retrasos, 4 cancelaciones y 116 equipajes sin cargar en Global/SH/Business/YW. En Long Haul Premium el NPS escaló de 16.08 a 71.43 (+55.4 pts) avalado por verbatims muy positivos y alta satisfacción en BOG–MAD y flota A350, mientras en Long Haul Business se hundió de 24.44 a 14.29 (–10.2 pts) tras 109 pérdidas de conexión, 116 equipajes sin cargar, 5 retrasos y 4 cancelaciones en Global/LH/Business. En Economy SH IB bajó de 33.87 a 33.64 (–0.2 pts) por incidencias NCS en LHR–MAD (7 retrasos, 4 cancelaciones, 109 pérdidas de conexión, 1 equipaje sin cargar), contrarrestado por YW, que subió de 34.51 a 39.52 (+5.0 pts), dejando el agregado en rango normal. Economy LH mantuvo desempeño estable con NPS de 14.36 (+2.2 pts vs L7d).

Las rutas más afectadas mostraron contrastes marcados: LHR–MAD acumuló las peores valoraciones en Economy SH IB (NPS 10.7, n=28) y SH Business YW (NPS 16.7, n=??), mientras que EAS–MAD y BOG–MAD alcanzaron NPS 100.0 en Global/SH/Business/IB y Global/LH/Premium respectivamente. Los perfiles más reactivos incluyen Leisure (NPS 33.2 en Global, 76.5 en Global/SH/Business/IB) y Business/Work (NPS 22.4 en Global, 100.0 en Global/LH/Premium), además de regiones como Residence Region Unknown (–53.3 en Global/SH/Economy/IB) y Asia (–50.0 en Global/SH/Economy/IB).

ECONOMY SH IB & YW  
La cabina Economy SH IB registró una ligera caída de 33.87 a 33.64 (–0.2 pts vs L7d) y YW experimentó un repunte de 34.51 a 39.52 (+5.0 pts vs L7d) durante la semana del 2025-11-29. La causa principal de la baja en IB fue la concentración de incidentes NCS en la ruta LHR–MAD (7 retrasos, 4 cancelaciones, 109 pérdidas de conexión, 1 equipaje no embarcado), afectando especialmente a perfiles Residence Region Unknown (NPS –53.3, n=15) y Residence Region Asia (NPS –50.0, n=2). El canal YW mantuvo desempeño estable sin incidencias significativas, equilibrando el agregado de Economy SH dentro del rango normal.

BUSINESS SH IB & YW  
El segmento Business SH anotó un NPS de 56.41 (25.11 pts por encima de L7d 35.35). Internamente, IB pasó de 34.81 a 74.07 (+39.3 pts vs L7d) impulsado por OTP +2.42 pts y Load Factor –2.03 pts (datos operativos Global/SH/Business) y un feedback de clientes centrado en puntualidad y confort, con rutas como EAS–MAD y LHR–MAD alcanzando NPS 100.0. Por el contrario, YW cayó de 39.24 a 16.67 (–22.6 pts vs L7d) debido a 7 retrasos, 4 cancelaciones y 116 equipajes no embarcados (incidentes NCS Global/SH/Business/YW), especialmente entre flota CRJ y perfiles de América Sur.

ECONOMY LH  
La cabina Economy LH mantuvo desempeño estable con un NPS de 14.36 (+2.2 pts vs L7d 12.20) durante la semana del 2025-11-29. No se detectaron cambios significativos, manteniéndose niveles consistentes de satisfacción.

BUSINESS LH  
En Business LH, el NPS descendió de 24.44 a 14.29 (–10.2 pts vs L7d). El deterioro se atribuye a 109 pérdidas de conexión, 116 equipajes sin cargar, 5 retrasos y 4 cancelaciones en Global/LH/Business, afectando rutas como MAD–SJU (NPS 100.0, n=1) y penalizando perfiles Leisure (NPS 0.0, n=15) y Business/Work (NPS 30.8, n=13).

PREMIUM LH  
El segmento Premium LH elevó su NPS de 16.08 a 71.43 (+55.4 pts vs L7d) en la semana del 2025-11-29. Las causas dominantes fueron un feedback de clientes excepcional sobre servicio y confort (10 verbatims positivos), con rutas como BOG–MAD alcanzando NPS 100.0 (n=2) y perfiles Business/Work (NPS 100.0, n=1), Leisure (NPS 66.7, n=7) y flota A350 (NPS 100.0, n=4) liderando la subida.

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

En SH/Economy, el escenario es CANCELACIÓN (IB –, YW + | Normal).  
- Narrativa: Mientras SH/Economy IB sufrió una caída de 0.7 pts atribuida a cancelaciones y retrasos por la huelga general en Italia (ncs_tool), SH/Economy YW se elevó 14.5 pts gracias al feedback mayoritariamente positivo sobre atención y servicio a bordo (verbatims_tool), neutralizándose el efecto en el agregado.  
- Evidencia Clave:  
  • IB –0.7 pts por cancelaciones y retrasos (huelga en Italia, rutas MAD–MXP/BLQ–MAD).  
  • YW +14.5 pts por comentarios positivos de servicio a bordo (179 verbatims).

En SH/Business, el escenario es DOMINANCIA (IB +, YW – | –).  
- Narrativa: La caída de SH/Business de 7.6 pts se explica principalmente por el desplome de YW (–49.2 pts) por cancelaciones masivas y retrasos motivados por la huelga en Italia en rutas clave (MAD–MXP, BLQ–MAD), imponiéndose al agregado, aunque SH/Business IB había repuntado +15.2 pts con feedback positivo en puntualidad y calidad de servicio, atenuando parcialmente la baja.  
- Evidencia Clave:  
  • YW –49.2 pts por cancelaciones y retrasos (ncs_tool, huelga en Italia).  
  • IB +15.2 pts por alta satisfacción en puntualidad y atención en cabina (verbatims_tool).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En SH, la dinámica es DILUCIÓN (Economy N, Business – | SH N).  
- **Narrativa:** El rendimiento de Short Haul está dictado por la caída de SH/Business (–7.6 pts), originada por el desplome de YW (–49.2 pts) debido a cancelaciones y retrasos por la huelga en Italia en rutas MAD–MXP y BLQ–MAD, aunque este efecto fue parcialmente mitigado por la estabilidad de SH/Economy.  
- **Evidencia:**  
  • SH/Business YW –49.2 pts por cancelaciones y retrasos (ncs_tool).  
  • SH/Economy +3.8 pts (no anomalía significativa).

En LH, la dinámica es DOMINANCIA (Economy –, Business +, Premium N | LH –).  
- **Narrativa:** El desempeño de Long Haul está dictado por la caída de LH/Economy (–10.4 pts), atribuida a las 12 cancelaciones y 8 retrasos detectados por ncs_tool en rutas MAD–BLQ y MAD–MXP, efecto dominante que solo se vio atenuado parcialmente por el repunte de LH/Business (+10.9 pts) gracias al feedback positivo en calidad de servicio y cabina.  
- **Evidencia:**  
  • LH/Economy –10.4 pts por cancelaciones y retrasos (ncs_tool, rutas MAD–BLQ, MAD–MXP).  
  • LH/Business +10.9 pts por comentarios positivos de servicio a bordo (verbatims_tool).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es DOMINANCIA (LH –, SH N | +).  
- Narrativa: El NPS Global de +1.7 pts está arrastrado por el sólido desempeño de Short Haul (NPS SH 37.1 pts vs baseline 34.16), que contrarrestó la caída de Long Haul. SH no registró desviaciones operativas significativas y mantuvo feedback positivo de puntualidad y servicio.  
- Evidencia:  
  • SH/Economy y SH/Business sin anomalías críticas (NPS SH +2.9 pts; operative_data_tool sin variaciones >3 pts).  
  • Verbatims SH: énfasis en puntualidad y amabilidad del personal (verbatims_tool).  
  • LH/Business y LH/Economy presentaron caídas de –7.6 pts y –10.4 pts respectivamente (ncs_tool: cancelaciones y retrasos en MAD–BLQ, MAD–MXP), pero no bastaron para frenar el efecto de SH.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Cancelaciones y retrasos por huelga en Italia (Impacto en Economy LH)  
- Escenario: DOMINANCIA (Economy –, Business +, Premium N | LH –)  
- NMA: Long Haul Economy  
- Afecta a: LH/Economy  
- Qué falló: Cancelaciones (12 vuelos) y retrasos (8 vuelos) motivados por la huelga en Italia (LH/Economy)  
- Dónde:  
  • MAD–NRT: NPS –20.0 (n=5, LH/Economy)  
  • MAD–BLQ: rutas con cancelaciones (encuestas no disponibles, LH/Economy)  
  • MAD–MXP: rutas con cambios de avión (encuestas no disponibles, LH/Economy)  
- Quién:  
  • Business/Work: NPS –16.0 (25 encuestas, LH/Economy)  
  • Leisure: NPS 4.2 (194 encuestas, LH/Economy)  
- Evidencia COMPLETA:  
  • NPS 1.84 vs baseline 12.20 (LH/Economy)  
  • Load Factor –3.24 pts; OTP15 –2.48 pts (operative_data_tool, LH/Economy)  
  • 12 cancelaciones y 8 retrasos (ncs_tool, LH/Economy)  

CAUSA 2: Cancelaciones y retrasos por huelga en Italia (Impacto en YW SH/Business)  
- Escenario: DOMINANCIA (IB +, YW – | SH/Business –)  
- NMA: SH/Business YW  
- Afecta a: SH/Business YW  
- Qué falló: Cancelaciones (18 vuelos) y retrasos (9 vuelos) por huelga en Italia (SH/Business YW)  
- Dónde:  
  • BLQ–MAD: NPS –100.0 (n=1, SH/Business YW)  
  • MAD–MRS: NPS 0.0 (n=2, SH/Business YW)  
- Quién:  
  • Business/Work: NPS –37.5 (8 encuestas, SH/Business YW)  
  • Leisure: NPS 8.3 (12 encuestas, SH/Business YW)  
- Evidencia COMPLETA:  
  • NPS –10.0 vs baseline 39.24 (SH/Business YW)  
  • Load Factor –3.05 pts; OTP15 +2.48 pts (operative_data_tool, SH/Business YW)  
  • 18 cancelaciones y 9 retrasos (ncs_tool, SH/Business YW)  

CAUSA 3: Cancelaciones y retrasos por huelga en Italia (Impacto en IB SH/Economy)  
- Escenario: CANCELACIÓN (IB –, YW + | SH/Economy N)  
- NMA: SH/Economy IB  
- Afecta a: SH/Economy IB  
- Qué falló: Cancelaciones (18 vuelos) y retrasos (9 vuelos) por huelga en Italia (SH/Economy IB)  
- Dónde:  
  • MAD–MXP: NPS –66.7 (n=6, SH/Economy IB)  
- Quién:  
  • Business/Work: NPS 8.5 (59 encuestas, SH/Economy IB)  
  • Leisure: NPS 38.2 (286 encuestas, SH/Economy IB)  
- Evidencia COMPLETA:  
  • NPS 33.14 vs baseline 33.87 (SH/Economy IB)  
  • Load Factor –0.16 pts; OTP15 +1.34 pts (operative_data_tool, SH/Economy IB)  
  • 18 cancelaciones y 9 retrasos (ncs_tool, SH/Economy IB)  

CAUSA 4: Excelente servicio a bordo y atención (Impacto en Business LH)  
- Escenario: DOMINANCIA (Economy –, Business +, Premium N | LH –) pero positiva en Business  
- NMA: LH/Business  
- Afecta a: LH/Business  
- Qué falló: No hubo fallo; más bien, calidad de servicio en cabina y comida excepcional (LH/Business)  
- Dónde:  
  • MAD–MVD: NPS 50.0 (n=2, LH/Business)  
- Quién:  
  • Business/Work: NPS 70.0 (10 encuestas, LH/Business)  
  • Leisure: NPS 20.8 (24 encuestas, LH/Business)  
- Evidencia COMPLETA:  
  • NPS 35.29 vs baseline 24.44 (LH/Business)  
  • Load Factor –0.39 pts; OTP15 –2.48 pts (operative_data_tool, LH/Business)  
  • Temas clave: “Calidad del servicio en cabina”, “Comida muy buena”, “Atención de check-in muy agradable” (verbatims_tool, LH/Business)  

CAUSA 5: Excelente servicio a bordo y atención (Impacto en IB SH/Business)  
- Escenario: DOMINANCIA (IB +, YW – | SH/Business –) pero positiva en IB  
- NMA: SH/Business IB  
- Afecta a: SH/Business IB  
- Qué falló: No hubo fallo; el servicio en cabina y la puntualidad destacaron (SH/Business IB)  
- Dónde:  
  • MAD–ORY: NPS 66.7 (n=3, SH/Business IB)  
- Quién:  
  • Business/Work: NPS 52.9 (17 encuestas, SH/Business IB)  
  • Leisure: NPS 47.1 (17 encuestas, SH/Business IB)  
- Evidencia COMPLETA:  
  • NPS 50.0 vs baseline 34.81 (SH/Business IB)  
  • Load Factor –1.63 pts; OTP15 +1.34 pts (operative_data_tool, SH/Business IB)  
  • Temas clave: “Puntualidad”, “Servicio a bordo excelente”, “Amabilidad de la tripulación” (verbatims_tool, SH/Business IB)  

CAUSA 6: Excelente servicio a bordo y atención (Impacto en YW SH/Economy)  
- Escenario: CANCELACIÓN (IB –, YW + | SH/Economy N)  
- NMA: SH/Economy YW  
- Afecta a: SH/Economy YW  
- Qué falló: No hubo fallo; muy buena atención al pasajero y servicio a bordo (SH/Economy YW)  
- Dónde:  
  • BLQ–MAD: NPS 66.7 (n=6, SH/Economy YW)  
- Quién:  
  • Leisure: NPS 56.4 (102 encuestas, SH/Economy YW)  
  • Business/Work: NPS 31.8 (44 encuestas, SH/Economy YW)  
- Evidencia COMPLETA:  
  • NPS 48.97 vs baseline 34.51 (SH/Economy YW)  
  • OTP15 +2.48 pts; Load Factor +0.16 pts (operative_data_tool, SH/Economy YW)  
  • 179 verbatims positivos sobre “atención” y “servicio a bordo” (verbatims_tool, SH/Economy YW)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  

La red global cerró el 28 de noviembre de 2025 con un NPS Global de 29.11, mejorando 1.7 pts vs L7d. En Short Haul Economy, IB pasó de 33.87 a 33.14 (–0.7 pts) y YW de 34.51 a 48.97 (+14.5 pts), aunque el agregado se mantuvo estable en 37.83 (+3.8 pts). Short Haul Business se deterioró de 35.35 a 27.78 (–7.6 pts) por el desplome de YW de 39.24 a –10.0 (–49.2 pts), pese al repunte de IB de 34.81 a 50.0 (+15.2 pts). En Long Haul, Economy cayó de 12.20 a 1.84 (–10.4 pts) y Business subió de 24.44 a 35.29 (+10.9 pts).  

Las rutas más afectadas incluyen BLQ–MAD (NPS –100.0, SH/Business YW; NPS 66.7, SH/Economy YW), MAD–MXP (NPS –66.7, SH/Economy IB), MAD–NRT (NPS –20.0, LH/Economy) y MAD–MVD (NPS 50.0, LH/Business). Los incidentes NCS de huelga en Italia (18 cancelaciones y 9 retrasos en SH/Business YW; 12 cancelaciones y 8 retrasos en LH/Economy) explican las mayores caídas de NPS. En perfiles, los pasajeros Business/Work en SH/Business YW (–37.5 pts) y en LH/Economy (–16.0 pts), así como viajeros en flota A321 (–62.5 pts, LH/Economy), fueron los más sensibles, mientras que Business/Work en LH/Business (+70.0) y Leisure en SH/Economy YW (+56.4) lideraron las subidas.

ECONOMY SH: Volatilidad interna con efecto neutro  
La cabina SH/Economy mantuvo desempeño estable durante la semana del 28 de noviembre, registrando un NPS de 37.83 (2025-11-28) con una variación de +3.8 pts vs L7d. IB experimentó una ligera caída de 33.87 a 33.14 (–0.7 pts) atribuible a 18 cancelaciones y 9 retrasos en rutas MAD–MXP y BLQ–MAD por huelga en Italia (incidentes NCS), mientras que YW escaló de 34.51 a 48.97 (+14.5 pts) gracias a un feedback de clientes muy positivo sobre atención y servicio a bordo (179 comentarios). Estos movimientos opuestos se neutralizaron en el agregado. Las rutas más volátiles fueron MAD–MXP (NPS –66.7, SH/Economy IB) y BLQ–MAD (NPS 66.7, SH/Economy YW), y los perfiles más reactivos incluyeron Leisure (38.2, SH/Economy IB) y Leisure (56.4, SH/Economy YW).

BUSINESS SH: Caída liderada por YW pese a impulso de IB  
El segmento SH/Business registró un NPS de 27.78 (2025-11-28), con un deterioro de –7.6 pts vs L7d. YW se desplomó de 39.24 a –10.0 (–49.2 pts) por 18 cancelaciones y 9 retrasos motivados por la huelga en Italia en rutas BLQ–MAD y MAD–MRS (incidentes NCS), mientras que IB subió de 34.81 a 50.0 (+15.2 pts) gracias a puntualidad y calidad de servicio en cabina (feedback de clientes). La presión negativa de YW se impuso, afectando especialmente al perfil Business/Work (–37.5 pts, SH/Business YW) y al segmento Leisure en flota CRJ (–15.8 pts, SH/Business IB).

ECONOMY LH: Impacto severo de cancelaciones y retrasos  
La cabina LH/Economy sufrió un retroceso de 12.20 a 1.84 (–10.4 pts vs L7d), registrando un NPS de 1.84 (2025-11-28). Este deterioro se explica por 12 cancelaciones y 8 retrasos en MAD–BLQ y MAD–MXP debido a la huelga en Italia (incidentes NCS), pese a que OTP cayó –2.48 pts y Load Factor mejoró –3.24 pts (datos operativos). La mayor caída se observó en MAD–NRT (NPS –20.0, LH/Economy) y entre Business/Work (–16.0 pts, LH/Economy) y flota A321 (–62.5 pts, LH/Economy).

BUSINESS LH: Repunte impulsado por calidad de servicio  
La cabina LH/Business escaló de 24.44 a 35.29 (+10.9 pts vs L7d), con un NPS de 35.29 (2025-11-28). Este alza se sustenta en comentarios destacados sobre “calidad de servicio en cabina”, “comida muy buena” y “atención de check-in” (feedback de clientes), sin desviaciones significativas en OTP (–2.48 pts) ni Load Factor (–0.39 pts, datos operativos). Destacan la ruta MAD–MVD (NPS 50.0, 2 encuestas, LH/Business) y perfiles Business/Work (70.0 pts, LH/Business) y flota A332 (66.7 pts, LH/Business).

PREMIUM LH: Rendimiento estable  
El segmento Premium de LH mantuvo desempeño estable, registrando un NPS de 16.67 (2025-11-28) con una variación de +0.6 pts vs L7d. No se detectaron cambios significativos en Explanatory Drivers, datos operativos ni incidentes NCS, manteniendo niveles de satisfacción consistentes.

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

Economy (SH): escenario TRANSFERENCIA (IB −2.1, YW N | Economy SH −0.5)  
- Narrativa: la caída en el NPS de Economy SH se impone desde el segmento IB; aunque YW se mantuvo en rango normal y suavizó el impacto, el deterioro “se transfirió” desde IB.  
- Evidencia Clave: Global/SH/Economy/IB registró −2.1 pts con 20 cancelaciones y 13 retrasos vinculados a la huelga general en Italia (ncs_tool, confianza alta).  

Business (SH): escenario DOMINANCIA (IB +12.2, YW −10.7 | Business SH +8.6)  
- Narrativa: la subida de NPS en Business SH obedece a la anomalía positiva de IB, que superó el efecto negativo de YW; se adopta la explicación de IB como causa principal.  
- Evidencia Clave: Global/SH/Business/IB registró +12.2 pts pese a 20 cancelaciones y 13 retrasos por la huelga en Italia (ncs_tool, confianza media).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Long Haul, la dinámica es CANCELACIÓN (+,+,- | N).  
- Narrativa: mientras Economy LH repuntó (+8.3 pts) impulsado por el feedback muy positivo a la tripulación y puntualidad tras cancelaciones masivas por huelga en Italia, y Business LH subió (+18.4 pts) sin un motor operativo claro (confianza baja en métricas), Premium LH cayó (−11.9 pts) por el pico de cancelaciones y retrasos vinculados a la misma huelga. Esos efectos contradictorios se neutralizaron en el agregado, dejando LH en rango normal.  
- Evidencia:  
  · Economy LH: +8.3 pts, 273 comentarios elogiando puntualidad y servicio (verbatims_tool).  
  · Business LH: +18.4 pts, feedback cualitativo positivo sin métrica operativa significativa (verbatims_tool + operative_data_tool).  
  · Premium LH: –11.9 pts, 14 cancelaciones y 15 retrasos por huelga en Italia (ncs_tool).  

En Short Haul, la dinámica es DOMINANCIA (–,+ | –).  
- Narrativa: el ligero deterioro de SH (–0.1 pts) está dictado por la caída de Economy SH, que impuso su signo negativo pese al fuerte repunte de Business SH; el buen desempeño de Business mitigó parcialmente el impacto, pero no evitó que SH cierre en negativo.  
- Evidencia:  
  · Economy SH (Global/SH/Economy/IB): –2.1 pts por 20 cancelaciones y 13 retrasos causados por huelga en Italia (ncs_tool, confianza alta).  
  · Business SH: +8.6 pts, correlacionado con mejora de puntualidad (OTP15 +1.74 pts) y menor ocupación (Load Factor –1.68 pts) (operative_data_tool, confianza media).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es DOMINANCIA (LH +6.5 pts, SH –0.1 pts | Global +3.8 pts)  
- Narrativa: el alza neta en NPS Global se debe principalmente al comportamiento de Long Haul, cuyo saldo positivo de +6.5 pts arrastró el resultado de la red. En LH, las subidas en Economy y Business, impulsadas por feedback muy positivo en puntualidad y servicio, superaron la caída de Premium por cancelaciones y retrasos, en tanto que la mínima caída de SH fue insuficiente para contrarrestar ese empuje.  
- Evidencia:  
  • Economy LH: +8.3 pts con 273 comentarios elogiando puntualidad y calidad del servicio (verbatims_tool).  
  • Business LH: +18.4 pts sin desviaciones operativas significativas (operative_data_tool).  
  • Premium LH: –11.9 pts por 14 cancelaciones y 15 retrasos relacionados con la huelga en Italia (ncs_tool).

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Incidentes por huelga en Italia  
- Escenario: TRANSFERENCIA (Economy SH/IB −2.1 pts, YW N | Economy SH −0.5 pts)  
- NMA: Global/SH/Economy/IB  
- Afecta a: Global/SH/Economy/IB  
- Qué falló: 20 cancelaciones y 13 retrasos por huelga general en Italia (ncs_tool)  
- Dónde:  
  • MAD–MXP: NPS 0.0 (Global/SH/Economy, n=12)  
  • BRU–MAD: NPS 0.0 (Global/SH/Economy, n=11)  
  • CMN–MAD: NPS –20.0 (Global/SH/Economy, n=5)  
- Quién:  
  • Leisure: NPS 36.8 (Global/SH/Economy/IB, n=229)  
  • Business/Work: NPS 19.4 (Global/SH/Economy/IB, n=93)  
- Evidencia COMPLETA:  
  • NPS 31.78 vs 33.87 (Global/SH/Economy/IB, −2.1 pts)  
  • Incidentes NCS: 20 cancelaciones y 13 retrasos en vuelos IB1237/38/672/674 (ncs_tool)  
  • Verbatims: 415 comentarios sobre puntualidad y servicio sin mención a cancelaciones/retrasos (verbatims_tool)  

CAUSA 2: Excelencia en atención y puntualidad  
- Escenario: DOMINANCIA (Business SH/IB +12.2 pts, YW −10.7 pts | Business SH +8.6 pts)  
- NMA: Global/SH/Business/IB  
- Afecta a: Global/SH/Business/IB  
- Qué falló: Driver cualitativo – feedback muy positivo sobre profesionalidad de la tripulación, atención a bordo y calidad de la comida (verbatims_tool)  
- Dónde:  
  • MAD–MXP: NPS 66.7 (Global/SH/Business/IB, n=3)  
  • MAD–OPO: NPS 100.0 (Global/SH/Business/IB, n=2)  
- Quién:  
  • Business/Work: NPS 53.3 (Global/SH/Business/IB, n=15)  
  • Leisure: NPS 42.1 (Global/SH/Business/IB, n=20)  
  • CodeShare IB: NPS 55.2 (Global/SH/Business/IB, n=30)  
- Evidencia COMPLETA:  
  • NPS 47.06 vs 34.81 (Global/SH/Business/IB, +12.2 pts)  
  • OTP15 +1.28 pts y Load Factor –1.72 pts (Global/SH/Business/IB, operative_data_tool)  
  • 39 verbatims elogiando puntualidad, servicio y comida sin mención de incidentes (verbatims_tool)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

Durante la semana del 27 noviembre 2025 se observaron subidas y bajadas de NPS en todos los segmentos. En Short Haul, Economy SH cayó 0.49 puntos, pasando de 34.02 a 33.53, mientras que Business SH repuntó 8.56 puntos, de 35.35 a 43.90. En Long Haul, Economy LH mejoró 8.29 puntos (12.20 → 20.50), Business LH subió 18.42 puntos (24.44 → 42.86) y Premium LH cayó 11.91 puntos, de 16.08 a 4.17. La baja de Economy SH se explicó por 20 cancelaciones y 13 retrasos por huelga general en Italia (Global/SH/Economy/IB, NPS 31.78 vs 33.87), mitigada parcialmente por YW (NPS 36.67 vs 34.51). El alza de Business SH se impulsó desde IB gracias a mejora de puntualidad (OTP +1.74 pts) y menor Load Factor (–1.68 pts) (Global/SH/Business/IB, NPS 47.06 vs 34.81), pese al desplome de YW por los mismos incidentes NCS. En LH, las subidas en Economy LH y Business LH se deben a un feedback de clientes muy positivo sobre puntualidad y servicio (273 comentarios en Economy LH; 30 en Business LH), mientras que Premium LH sufrió el impacto de la huelga en Italia con 14 cancelaciones y 15 retrasos (Premium LH, NPS 4.17 vs 16.08).

Las rutas más afectadas incluyen MAD–MXP, donde Economy SH registró NPS 0.0 (n=12) tras incidentes NCS, y CMN–MAD con NPS –20.0 (n=5) en el mismo segmento. En Long Haul, DOH–MAD obtuvo NPS 20.0 (n=5) en Economy LH y MAD–SJO alcanzó 100.0 (n=2) en Business LH. Entre perfiles, los más reactivos fueron los viajeros Business/Work en Premium LH (NPS –66.7, n=9) y en Economy SH (NPS 19.4, n=93), así como los Leisure en Economy SH (NPS 36.8, n=229) y los CodeShare IB en Business SH (NPS 55.2, n=30).

ECONOMY SH: Huelga en Italia Impacta Economy SH IB  
La cabina Economy SH mostró un NPS de 33.53 el 27 noviembre 2025, con una bajada de 0.49 puntos vs L7d. IB cayó de 33.87 a 31.78 (–2.10 pts) debido a 20 cancelaciones y 13 retrasos por huelga general en Italia (incidentes NCS), mientras que YW subió de 34.51 a 36.67 (+2.16 pts) por estabilidad operativa (OTP estable, Load Factor sin cambios), diluyendo parcialmente el impacto. El deterioro se reflejó en rutas como CMN–MAD (NPS –20.0, n=5), MAD–MXP (NPS 0.0, n=12) y BRU–MAD (NPS 0.0, n=11). Los perfiles más reactivos fueron Leisure (NPS 36.8, n=229) y Business/Work (NPS 19.4, n=93).

BUSINESS SH: Puntualidad y Menor Ocupación Impulsan Business SH IB  
El segmento Business SH registró un NPS de 43.90 el 27 noviembre 2025, con un alza de 8.56 puntos vs L7d. IB subió de 34.81 a 47.06 (+12.25 pts) gracias a mejora de puntualidad (OTP +1.74 pts) y Load Factor reducido (–1.68 pts) (datos operativos), más un feedback muy positivo sobre comodidad y servicio a bordo (39 comentarios). En contraste, YW bajó de 39.24 a 28.57 (–10.66 pts) por 20 cancelaciones y 13 retrasos de la huelga en Italia (incidentes NCS). La tendencia positiva se reflejó en rutas como MAD–MXP (NPS 66.7, n=3) y MAD–MRS (NPS 50.0, n=2). Destacan los perfiles Business/Work (NPS 55.6, n=18) y Leisure (NPS 34.8, n=24).

ECONOMY LH: Feedback Positivo Refuerza Economy LH  
La cabina Economy LH alcanzó un NPS de 20.50 el 27 noviembre 2025, con una mejora de 8.29 puntos vs L7d. No se detectaron desviaciones operativas significativas (OTP –2.39 pts, Load Factor –3.19 pts) que expliquen el alza; el motor principal fue el feedback de clientes muy positivo sobre puntualidad y amabilidad de la tripulación (273 comentarios). Esta mejora se materializó en la ruta DOH–MAD (NPS 20.0, n=5). Los perfiles más reactivos fueron Leisure (NPS 23.0, n=139) y Business/Work (NPS 4.5, n=22).

BUSINESS LH: Feedback Sobresaliente Eleva Business LH  
La cabina Business LH presentó un NPS de 42.86 el 27 noviembre 2025, con un incremento de 18.42 puntos vs L7d. Sin variaciones operativas significativas, el alza se atribuye al feedback exclusivo y muy positivo en comodidad, profesionalidad y atención (30 comentarios). El impacto fue notable en la ruta MAD–SJO (NPS 100.0, n=2). Los perfiles más reactivos fueron Residence España (NPS 36.4, n=11) y CodeShare IB (NPS 36.8, n=19).

PREMIUM LH: Huelga en Italia Deteriora Premium LH  
La cabina Premium LH cayó a un NPS de 4.17 el 27 noviembre 2025, perdiendo 11.91 puntos vs L7d. La causa dominante fueron 14 cancelaciones y 15 retrasos por huelga general en Italia (incidentes NCS), sin desviaciones relevantes en OTP (–2.39 pts) ni Load Factor (–2.86 pts). No hubo menciones a estos eventos en el feedback de clientes (28 comentarios). Aunque GRU–MAD registró NPS 100.0 (n=2), el impacto negativo se concentró en perfiles Business/Work (NPS –66.7, n=9) y en regiones de Asia/Europa/América Norte (NPS –100.0 en muestras pequeñas).

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

En Economy SH, el escenario es DILUCIÓN (IB: Normal, YW: Positivo | Economy SH: Normal).  
- Narrativa: La anomalía positiva de YW (+7.8 pts) está presente, pero el IB mantiene su comportamiento “dentro de lo esperado” y diluye el efecto en el agregado, dejando al nodo padre en un nivel normal.  
- Evidencia Clave: Feedback cualitativo excepcional de YW sobre puntualidad, rapidez y trato (verbatims_tool).

En Business SH, el escenario es TRANSFERENCIA (IB: Positivo, YW: Normal | Business SH: Positivo).  
- Narrativa: Adoptamos la explicación del nodo padre: el alza de NPS (+13.6 pts) se debe principalmente a la elevada satisfacción con la atención del personal en Business Class Short Haul, contagiosa al agregado pese a que YW no mostró anomalía.  
- Evidencia Clave: Verbatims unánimemente positivos sobre amabilidad y atención de tripulación en Business Class (verbatims_tool).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Long Haul, la dinámica es SINERGIA (Economy: Normal, Business: Positivo, Premium: Positivo | LH: Positivo).  
- **Narrativa:** Adoptamos la explicación del nodo Long Haul: la subida de NPS +9.3 pts se explica principalmente por la caída del Load Factor, que al reducir la ocupación mejoró la experiencia en todas las cabinas.  
- **Evidencia:** Load Factor –3.01 pts (Long Haul).

En Short Haul, la dinámica es DILUCIÓN (Economy: Normal, Business: Positivo | SH: Normal).  
- **Narrativa:** El fuerte alza en Business SH (+13.6 pts), impulsada por la excelente atención de la tripulación, no alcanza a trasladarse al agregado porque el volumen mayoritario de Economy mantiene un desempeño estable y diluye el efecto.  
- **Evidencia:** Verbatims Global/SH/Business: “Atención excelente de las TCPs” (n=56 comentarios).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA `(LH: +, SH: N | GLOBAL: +)`.  
- **Narrativa:** Adoptamos la explicación del nodo Global: el alza de 8.1 pts en NPS Global se contagia desde el incremento de 9.3 pts en Long Haul, impulsado por la caída del Load Factor, mientras que Short Haul mantuvo un desempeño estable.  
- **Evidencia:** Load Factor –3.01 pts (Long Haul).

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Reducción de Load Factor en Long Haul  
- Escenario: SINERGIA en LH (+, +, + | +) y TRANSFERENCIA en Global (+, N | +)  
- NMA: Global/LH  
- Afecta a:  
  • Global/LH/Economy  
  • Global/LH/Business  
  • Global/LH/Premium  
- Qué falló: Load Factor –3.01 pts (Global/LH)  
- Dónde:  
  • DOH–MAD: NPS 33.3 (n=3) (Global/LH)  
  • Rutas con incidentes (huelga BRU): BRU–MAD, MAD–BRU (Global/LH)  
- Quién:  
  • Business/Work: NPS 28.1 (n=29) (Global/LH)  
  • Leisure: NPS 22.4 (n=189) (Global/LH)  
  • Fleet A350 C: NPS 69.3 (n=6) (Global/LH)  
- Evidencia COMPLETA:  
  • Explanatory Driver: Load Factor –3.01 pts vs baseline 13.8535→10.8435 (Global/LH)  
  • Operativa: OTP15_adjusted –3.3 pts vs baseline  (Global/LH)  
  • Incidentes NCS: 19 totales, 8 cancelaciones (incluyendo BRU–MAD por huelga), 4 retrasos (Global/LH)  
  • Verbatims: 352 comentarios con feedback mayoritariamente positivo en embarque, tripulación y catering (Global/LH)  

CAUSA 2: Excelente atención de la tripulación en Business SH  
- Escenario: TRANSFERENCIA en SH (+, N | +)  
- NMA: Global/SH/Business  
- Afecta a:  
  • Global/SH/Business/IB  
  • Global/SH/Business/YW  
- Qué falló: No falló – la calidad de servicio en Business Short Haul fue excepcional (verbatims)  
- Dónde:  
  • MAD–ORY: NPS 25.0 (n=4) (Global/SH/Business)  
  • BRU–MAD: NPS 0.0 (n=1) – caso aislado de cancelación no representativo (Global/SH/Business)  
- Quién:  
  • Leisure: NPS 52.6 (n=19) (Global/SH/Business)  
  • Business/Work: NPS 46.4 (n=28) (Global/SH/Business)  
- Evidencia COMPLETA:  
  • Explanatory Driver: NPS 48.936 (vs baseline 35.347) = +13.6 pts (Global/SH/Business)  
  • Incidentes NCS: 18 totales (12 cancelaciones – BRU–MAD/MAD–BRU, 2 equipaje, 6 otras) (Global/SH/Business)  
  • Verbatims: 56 comentarios “Atención excelente de las TCPs” y “Vuelo agradable en Business Class” (Global/SH/Business)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
Durante el día 2025-11-26 el NPS Global subió de 27.37 a 35.50 (+8.1 pts), impulsado por Long Haul (Global/LH) que mejoró de 13.85 a 23.11 (+9.3 pts) gracias a una caída de Load Factor (–3.01 ppts según Explanatory Drivers), y por repuntes notables en las cabinas Business LH (Global/LH/Business pasó de 24.44 a 46.67, +22.2 pts) y Premium LH (Global/LH/Premium de 16.08 a 58.33, +42.3 pts), ambas amparadas en feedback de clientes muy positivo pese al deterioro de OTP (–3.3 pts) y 19 incidentes NCS (8 cancelaciones en BRU-MAD por huelga y 4 retrasos). En Short Haul, el NPS Business SH ascendió de 35.35 a 48.94 (+13.6 pts) impulsado por la atención de tripulación (56 verbatims Global/SH/Business), y la sub-cabina Economy YW creció de 34.51 a 42.35 (+7.8 pts) gracias a comentarios de puntualidad y rapidez (215 verbatims Global/SH/YW), mientras que Economy IB (36.30 vs 33.87, +2.4 pts) y Business YW (41.18 vs 39.24, +1.9 pts) mantuvieron desempeño estable.

En cuanto a rutas, BRU-MAD y MAD-BRU concentraron 8 cancelaciones y 4 retrasos por huelga en Bruselas (incidentes NCS), generando NPS de 0.0 (Global/LH y Global/SH/Business) en encuestas aisladas. DOH-MAD (NPS 33.3, n=3, Global/LH) y las conexiones Premium MAD-SCL (100.0, n=1) y MAD-SDQ (100.0, n=1) mostraron divergencias de satisfacción. En perfiles, Long Haul registró Business/Work con NPS 28.1 (n=29) y Leisure con 22.4 (n=189), mientras que Short Haul Business destacó Leisure 52.6 (n=19) y Business/Work 46.4 (n=28), y Short Haul YW presentó Leisure 43.9 (n=98) frente a Business/Work 40.3 (n=72).

ECONOMY SH  
La cabina Economy de SH mantuvo desempeño estable durante el 2025-11-26, registrando un NPS de 39.23 (Global/SH, +5.1 pts vs L7d). IB marcó 36.30 (Global/SH/IB, +2.4 pts vs L7d) y YW alcanzó 42.35 (Global/SH/YW, +7.8 pts vs L7d). No se detectaron desviaciones operativas relevantes (OTP +2.46 pts, Load Factor –0.16 pts) y los 18 incidentes NCS no generaron quejas en feedback de clientes, manteniendo niveles consistentes de satisfacción.

BUSINESS SH  
El segmento Business de SH registró un NPS de 48.94 (Global/SH/Business) el 2025-11-26, con una mejora de +13.6 pts vs L7d. IB subió a 53.33 (Global/SH/Business/IB, +18.5 pts vs L7d) y YW marcó 41.18 (Global/SH/Business/YW, +1.9 pts vs L7d). Esta evolución se explica principalmente por la calidad de atención de tripulación (56 verbatims Global/SH/Business), pese a 12 cancelaciones y 6 otros ajustes de equipo en rutas BRU-MAD/MAD-BRU (incidentes NCS).

ECONOMY LH  
La cabina Economy de LH mantuvo desempeño estable, con NPS de 17.61 (Global/LH/Economy) vs 12.20 en L7d (+5.4 pts). No se reportaron variaciones significativas en OTP ni en Load Factor, y los datos operativos no revelan causas adicionales.

BUSINESS LH  
La cabina Business de LH experimentó un salto de NPS de 24.44 a 46.67 (+22.2 pts) el 2025-11-26. Los principales drivers fueron la baja ocupación (Load Factor –3.01 ppts) y 8 cancelaciones NCS en BRU-MAD/MAD-BRU, acompañados de un OTP –3.3 pts, sin impacto negativo en feedback de clientes. Entre perfiles, Leisure mostró 36.4 (Global/LH/Business) y Business/Work 75.0 (Global/LH/Business).

PREMIUM LH  
El segmento Premium de LH subió de 16.08 a 58.33 (+42.3 pts) durante el 2025-11-26, apoyado en un feedback excelente de comodidad y servicio (18 verbatims Global/LH/Premium), a pesar de 8 cancelaciones y 4 retrasos (19 incidentes NCS) y un OTP –3.3 pts que no afectó la satisfacción reportada.

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

En Economy SH, el escenario es CANCELACIÓN (IB –, YW + | Padre N).  
- Narrativa: Mientras el subsegmento IB sufrió una caída por 10 cancelaciones en la ruta BRU–MAD (huelga en BRU), YW registró un alza impulsada por la excelente atención al cliente (amabilidad del personal y agilidad en cambios de vuelo), neutralizándose el efecto en el agregado.  
- Evidencia Clave:  
  • IB: 10 cancelaciones en BRU–MAD (ncs_tool)  
  • YW: feedback positivo en verbatims (verbatims_tool)  

En Business SH, el escenario es DOMINANCIA (IB +, YW – | Padre +).  
- Narrativa: La anomalía positiva del nodo padre responde principalmente al impulso de IB, donde el feedback cualitativo destacó puntualidad, limpieza y atención de la tripulación; este efecto fue parcialmente suavizado por la caída de YW.  
- Evidencia Clave:  
  • IB: excelente puntuación en verbatims por puntualidad, limpieza y amabilidad de la tripulación (verbatims_tool)

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es DILUCIÓN (Economy N, Business + | SH N).  
- Narrativa: El comportamiento de SH está dictado por el subsegmento Business, cuyo alza se apoya en el impulso de IB, mientras que Economy, con desempeño estable, diluye parcialmente ese efecto.  
- Evidencia:  
  • Business SH (IB): +19.0 pts respaldados por verbatims que destacan puntualidad, limpieza y atención de la tripulación (verbatims_tool).  
  • Economy SH: desempeño normal sin variaciones significativas (NPS Period: 15.6 vs Baseline: 12.2).  

En Long Haul, la dinámica es CANCELACIÓN (Economy N, Business –, Premium + | LH N).  
- Narrativa: El descenso en Business LH, motivado por la caída de puntualidad (OTP15_adjusted –4.97 pts vs baseline) y 12 cancelaciones + 3 retrasos, fue contrarrestado por la mejora en Premium LH, impulsada por feedback positivo en verbatims sobre tripulación, limpieza y confort, resultando en un NPS global equilibrado.  
- Evidencia:  
  • Business LH: OTP15_adjusted –4.97 pts vs baseline; 12 cancelaciones y 3 retrasos (operative_data_tool & ncs_tool).  
  • Premium LH: elogios a la tripulación, limpieza y comodidad en verbatims (verbatims_tool).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA (LH N, SH N | Global +).  
- Narrativa: El alza de 6.3 pts en NPS Global se explica por un impulso neto de satisfacción presente en toda la red, documentado en verbatims, sin un driver operativo consolidado. Aunque ambos radios (LH y SH) están dentro de sus rangos normales, el efecto positivo agregado de la experiencia de servicio se transmitió al nivel Global.  
- Evidencia:  
  • NPS Global +6.3 pts (33.64 vs 27.37 baseline)  
  • 1 041 comentarios con tono positivo: elogios a tripulación, comodidad y limpieza, sin menciones de cancelaciones o retrasos (verbatims_tool)

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Cancelaciones por huelga en Bruselas  
- Escenario: CANCELACIÓN (SH/Economy/IB –, SH/Economy/YW + | SH/Economy N)  
- NMA: SH/Economy/IB  
- Afecta a: Global / SH / Economy / IB  
- Qué falló: 10 cancelaciones por huelga en aeropuerto BRU (SH/Economy/IB)  
- Dónde: Ruta BRU–MAD con NPS 15.0 (n=20) (SH/Economy/IB)  
- Quién:  
  • Leisure: NPS 36.3 (215 encuestas) (SH/Economy)  
  • Business/Work: NPS 26.3 (95 encuestas) (SH/Economy)  
- Evidencia completa:  
  • NPS 33.2258 vs baseline 33.8731 (–0.6 pts) (SH/Economy/IB)  
  • OTP15_adjusted +1.61 pts vs baseline (SH/Economy/IB)  
  • Load Factor –0.45 pts vs baseline (SH/Economy/IB)  
  • Incidentes NCS: 10 cancelaciones, 2 retrasos, 7 otras incidencias (SH/Economy/IB)  

CAUSA 2: Mejora en la atención al cliente  
- Escenario: CANCELACIÓN (SH/Economy/IB –, SH/Economy/YW + | SH/Economy N)  
- NMA: SH/Economy/YW  
- Afecta a: Global / SH / Economy / YW  
- Qué falló: Refuerzo de amabilidad de la tripulación y agilidad en cambios de vuelo (verbatims) (SH/Economy/YW)  
- Dónde: No hay rutas con desviación positiva significativa (SH/Economy/YW)  
- Quién:  
  • Fleet ATR: NPS 91.7 (12 encuestas) (SH/Economy/YW)  
  • Business/Work: NPS 56.8 (44 encuestas) (SH/Economy/YW)  
- Evidencia completa:  
  • NPS 51.5873 vs baseline 34.5112 (+17.1 pts) (SH/Economy/YW)  
  • OTP15_adjusted +2.26 pts vs baseline (SH/Economy/YW)  
  • Load Factor –0.42 pts vs baseline (SH/Economy/YW)  
  • Incidentes NCS: 10 cancelaciones, 2 retrasos, 7 incidencias (SH/Economy/YW)  
  • 161 verbatims positivos sin menciones a cancelaciones o retrasos (SH/Economy/YW)  

CAUSA 3: Feedback positivo en puntualidad, limpieza y atención  
- Escenario: DOMINANCIA (SH/Business/IB +, SH/Business/YW – | SH/Business +)  
- NMA: SH/Business/IB  
- Afecta a: Global / SH / Business / IB  
- Qué falló: Elogios a puntualidad, limpieza y amabilidad de tripulación (verbatims) (SH/Business/IB)  
- Dónde: Ruta MAD–VIE con NPS 100.0 (n=3) (SH/Business/IB)  
- Quién:  
  • Business/Work: NPS 57.1 (14 encuestas) (SH/Business/IB)  
  • Leisure: NPS 50.0 (12 encuestas) (SH/Business/IB)  
- Evidencia completa:  
  • NPS 53.8462 vs baseline 34.8129 (+19.0 pts) (SH/Business/IB)  
  • OTP15_adjusted +1.61 pts vs baseline (SH/Business/IB)  
  • Load Factor –2.04 pts vs baseline (SH/Business/IB)  
  • Incidentes NCS: 17 totales (10 cancelaciones, 2 reprogramaciones) (SH/Business/IB)  
  • 37 verbatims positivos sin menciones operativas (SH/Business/IB)  

CAUSA 4: Caída de puntualidad y cancelaciones  
- Escenario: CANCELACIÓN (LH/Economy N, LH/Business –, LH/Premium + | LH N)  
- NMA: LH/Business  
- Afecta a: Global / LH / Business  
- Qué falló: OTP15_adjusted –4.97 pts vs baseline (74.95 vs 79.94) y 12 cancelaciones + 3 retrasos (LH/Business)  
- Dónde: No se identificaron rutas con caída de NPS vinculables a esos incidentes (LH/Business)  
- Quién:  
  • Leisure: NPS 38.9 (18 encuestas) (LH/Business)  
  • Business/Work: NPS –25.0 (8 encuestas) (LH/Business)  
- Evidencia completa:  
  • NPS 19.2308 vs baseline 24.4420 (–5.2 pts) (LH/Business)  
  • Load Factor: dato operativo faltante para evaluación (LH/Business)  
  • Incidentes NCS: 12 cancelaciones, 3 retrasos (LH/Business)  
  • 44 verbatims sin menciones a cancelaciones o retrasos (LH/Business)  

CAUSA 5: Elogios a tripulación, limpieza y confort  
- Escenario: CANCELACIÓN (LH/Economy N, LH/Business –, LH/Premium + | LH N)  
- NMA: LH/Premium  
- Afecta a: Global / LH / Premium  
- Qué falló: Feedback muy positivo en tripulación, limpieza y comodidad de cabina (verbatims) (LH/Premium)  
- Dónde: Ruta JFK–MAD con NPS 100.0 (n=1) (LH/Premium)  
- Quién:  
  • Leisure: NPS 33.3 (15 encuestas) (LH/Premium)  
  • Business/Work: NPS 66.7 (3 encuestas) (LH/Premium)  
  • Flota A350: NPS 50.0 (10 encuestas) (LH/Premium)  
  • Flota A333: NPS 0.0 (3 encuestas) (LH/Premium)  
  • Residencia Europa: NPS –33.3 (3 encuestas) (LH/Premium)  
- Evidencia completa:  
  • NPS 38.8889 vs baseline 16.0784 (+22.8 pts) (LH/Premium)  
  • OTP15_adjusted –4.97 pts vs baseline (LH/Premium)  
  • Load Factor –3.08 pts vs baseline (LH/Premium)  
  • Incidentes NCS: 12 cancelaciones, 3 retrasos (LH/Premium)  
  • 36 verbatims positivos sin menciones operativas (LH/Premium)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El NPS Global experimentó una subida de 6.3 puntos, pasando de 27.37 a 33.64 entre el L7d y el día 2025-11-25. Este alza se explica por la fuerte mejora de Premium LH (de 16.08 a 38.89, +22.8 pts) impulsada por feedback de clientes sobre limpieza, confort y amabilidad de la tripulación, y por el repunte de Business SH (de 35.35 a 44.74, +9.4 pts) gracias a elogios a puntualidad y calidad de servicio. Economy SH contribuyó positivamente con un NPS de 38.53 (de 34.02 a 38.53, +4.5 pts) donde YW escaló de 34.51 a 51.59 (+17.1 pts) neutralizando la leve caída de IB de 33.87 a 33.23 (–0.6 pts). En contrapartida, Business LH retrocedió de 24.44 a 19.23 (–5.2 pts) por deterioro de puntualidad (OTP –4.97 pts) y 12 cancelaciones + 3 retrasos, y Economy SH/IB también se contrajo por 10 cancelaciones en BRU–MAD.

Las rutas más afectadas fueron BRU–MAD con NPS 15.0 (SH/Economy/IB) tras 10 cancelaciones y 2 retrasos, la ruta MAD–VIE alcanzó NPS 100.0 (SH/Business/IB) por feedback excepcional, y JFK–MAD registró NPS 100.0 (LH/Premium) a partir de comentarios positivos. Entre los perfiles de cliente, Business/Work fue el más sensible en Business LH con NPS –25.0 (8 encuestas) y Leisure destacó en Premium LH con NPS 33.3 (15 encuestas).

ECONOMY SH: Contraste de dinámicas internas  
La cabina Economy de SH mantuvo desempeño estable el día 2025-11-25, registrando un NPS de 38.53 (+4.5 pts vs L7d). IB anotó 33.23 (–0.6 pts vs 33.87 L7d) por 10 cancelaciones y 2 retrasos en BRU–MAD (incidentes NCS) pese a un OTP de +1.61 pts (datos operativos), mientras YW subió a 51.59 (+17.1 pts vs 34.51 L7d) potenciado por la amabilidad de la tripulación y ágil gestión de cambios de vuelo (feedback de clientes). Esta dicotomía interna se equilibró, resultando en estabilidad en el agregado.

BUSINESS SH: Impulso de IB suavizado por YW  
La cabina Business de SH experimentó un NPS de 44.74 (+9.4 pts vs L7d) el día 2025-11-25. IB se disparó a 53.85 (+19.0 pts vs 34.81 L7d) por elogios a puntualidad, limpieza y atención de la tripulación (feedback de clientes), especialmente en la ruta MAD–VIE con NPS 100.0 (SH/Business/IB), mientras YW cayó a 25.00 (–14.2 pts vs 39.24 L7d) pese a mejoras en OTP (+2.26 pts) y un Load Factor menor (–3.72 pts), moderando el ascenso global.

ECONOMY LH: Desempeño estable  
La cabina Economy de LH mantuvo desempeño estable el día 2025-11-25, con un NPS de 15.56 (+3.4 pts vs 12.20 L7d). No se detectaron desviaciones operativas o de feedback de clientes significativas: el Load Factor cerró en –1.18 pts y OTP en +1.14 pts (datos operativos), ambos dentro del rango normal.

BUSINESS LH: Retroceso por puntualidad e incidencias  
La cabina Business de LH registró un NPS de 19.23 (–5.2 pts vs 24.44 L7d) el día 2025-11-25. El driver principal fue la caída de puntualidad (OTP –4.97 pts, datos operativos) junto a 12 cancelaciones y 3 retrasos (incidentes NCS), concentrados en rutas como EZE–MAD, afectando especialmente a perfiles Business/Work con NPS –25.0 (8 encuestas).

PREMIUM LH: Alza apoyada en calidad de servicio  
El segmento Premium de LH alcanzó un NPS de 38.89 (+22.8 pts vs 16.08 L7d) el día 2025-11-25. El impulso dominante provino del feedback de clientes sobre limpieza, confort y amabilidad de la tripulación (36 verbatims), pese a que la puntualidad bajó (OTP –4.97 pts, datos operativos) y se contabilizaron 12 cancelaciones + 3 retrasos (incidentes NCS). La ruta JFK–MAD destacó con NPS 100.0 (1 encuesta) y los perfiles Business/Work (66.7, 3 encuestas) y Leisure (33.3, 15 encuestas) fueron los más reactivos.

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

Economy SH  
- Escenario: Ninguno de los cinco (IB Normal, YW Normal | Economy SH Normal).  
  • Ambos hijos están dentro de rango “Normal”, por lo que no hay dinámica interna que explicar; el padre refleja estabilidad.

Business SH  
- Escenario: CANCELACIÓN (+15.2, –15.7 | Normal)  
- Narrativa:  
  Ignoramos la explicación del padre, ya que el alza de IB y la caída de YW se anulan mutuamente.  
  • Mientras SH-Business IB ganó +15.2 pts impulsado por elogios al servicio a bordo y una experiencia positiva en la ruta BRU-MAD pese a 6 cancelaciones y 6 retrasos (nivel de confianza medio),  
  • SH-Business YW cayó –15.7 pts atribuible a los mismos incidentes de cancelación y retraso (nivel de confianza baja),  
  neutralizándose ambos efectos en el NPS agregado de Business SH.  
- Evidencia Clave:  
  – IB: 6 cancelaciones y 6 retrasos en BRU-MAD (n=1), confianza media.  
  – YW: 6 cancelaciones y 6 retrasos (misma ruta), confianza baja.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul (SH)  
- Dinámica: SINERGIA (Economy SH Normal, Business SH Normal | Short Haul Normal)  
- Narrativa: Ambas cabinas mantuvieron resultados dentro del rango esperado, reflejando la estabilidad de Short Haul sin que ninguna anomalía interna desequilibre el agregado.  
- Evidencia:  
  • Economy SH: NPS 34.8449 vs baseline 34.0192 (Normal)  
  • Business SH: NPS 39.5349 vs baseline 35.3476 (Normal)  

Long Haul (LH)  
- Dinámica: DOMINANCIA (Economy +11.3, Business –8.8, Premium N | LH +8.4)  
- Narrativa: El alza neta de NPS en Long Haul se explica por la fuerte anomalía positiva de Economy, pese a la caída en Business y la estabilidad de Premium. El empuje de Economy dictó el comportamiento del radio, moderado parcialmente por el desempeño opuesto de Business.  
- Evidencia:  
  • Economy LH: anomalía +11.3 pts impulsada por la caída de puntualidad (OTP15_adjusted –5.55 pts) y 14 cancelaciones/4 retrasos (BRU-MAD, LIM-MAD) con fuerte alza en Leisure (NPS 34.3, n=137).  
  • Business LH: anomalía –8.8 pts asociada a la misma caída de OTP y los incidentes operativos en rutas BRU-MAD.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA `(LH +8.4, SH N | Global +4.7)`.  
- Narrativa: El impulso positivo registrado en Long Haul se contagió al Global, a pesar de que Short Haul se mantuvo estable. La explicación operativa global no arroja desviaciones de métricas > 3 pts (OTP15 +1.15, Load Factor –1.21, Mishandling –1.82) ni soporta una causa clara, por lo que la confianza en esta atribución es baja.  
- Evidencia:  
  • Métricas Global vs baseline: Load Factor –1.21 pts, OTP15 +1.15 pts (ninguna supera ±3 pts).  
  • Incidentes NCS globales: 58 cancelaciones y 61 retrasos (incoherente con subida de NPS).  
  • Temas de verbatims (896 comentarios): puntualidad, cortesía del personal y confort.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Deterioro de puntualidad y aumento de cancelaciones en Economy Long Haul  
- Escenario: DOMINANCIA (Economy +11.3, Business –8.8, Premium N | LH +8.4)  
- NMA: Global/LH/Economy  
- Afecta a:  
  • Global/LH/Economy  
  • Global/LH  
  • Global  
- Qué falló:  
  • OTP15_adjusted 74.38 vs baseline 79.93 (–5.55 pts) (Global/LH/Economy)  
  • 14 cancelaciones y 4 retrasos (ncs_tool) (Global/LH/Economy)  
- Dónde (rutas más afectadas):  
  1. BRU-MAD: NPS 13.3 (Global/LH/Economy, n=6)  
  2. LIM-MAD: NPS 29.4 (Global/LH/Economy, n=17)  
- Quién (perfiles más reactivos):  
  • Leisure: NPS 34.3 (Global/LH/Economy, 137 encuestas)  
  • Business/Work: NPS –21.2 (Global/LH/Economy, 33 encuestas)  
- Evidencia completa:  
  • NPS 23.53 vs baseline 12.20 (+11.3 pts) (Global/LH/Economy)  
  • OTP15_adjusted –5.55 pts (Global/LH/Economy)  
  • Load Factor 87.09 vs 90.85 (–3.51 pts, descartado) (Global/LH/Economy)  
  • Incidentes NCS: 14 cancelaciones, 4 retrasos (Global/LH/Economy)  

CAUSA 2: Efectos contrapuestos en Short Haul Business (no hay NMA único)  
- Escenario: CANCELACIÓN (+15.2, –15.7 | Normal)  
- NMA: – (no existe nodo único; causas opuestas)  
- Afecta a:  
  • Global/SH/Business/IB  
  • Global/SH/Business/YW  
- Qué falló:  
  • SH/Business/IB: elevada satisfacción por “cortesía del personal” y “confort” (verbatims_tool, 33 comentarios) → NPS 50.0 vs 34.81 (+15.2 pts)  
  • SH/Business/YW: 6 cancelaciones y 6 retrasos (ncs_tool) → NPS 23.53 vs 39.24 (–15.7 pts)  
- Dónde (rutas más afectadas):  
  • BRU-MAD: NPS 0.0 (Global/SH/Business/IB, n=1)  
  • SVQ-VLC: NPS 100.0 (Global/SH/Business/YW, n=1)  
- Quién (perfiles):  
  • SH/Business/IB – Leisure: NPS 68.8 (n=16); Business/Work: NPS 20.0 (n=10)  
  • SH/Business/YW – perfiles no identificados  
- Evidencia completa:  
  • NPS 50.0 vs 34.81 (Global/SH/Business/IB)  
  • NPS 23.53 vs 39.24 (Global/SH/Business/YW)  
  • Incidentes NCS: 6 cancelaciones, 6 retrasos (Global/SH/Business)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

Durante el 2025-11-24 detectamos subidas y bajadas de NPS en seis segmentos clave. Global pasó de 27.37 a 32.03 (+4.7 pts) por transferencia desde Long Haul, sin métricas operativas que superaran ±3 pts (OTP +1.15 pts, Load Factor –1.21 pts, mishandling –1.82 pts) y con 58 cancelaciones y 61 retrasos (incidentes NCS), lo que sugiere un efecto cualitativo de feedback positivo en puntualidad, cortesía y confort. En Long Haul/Economy el NPS creció de 12.20 a 23.53 pts (+11.3 pts) debido al deterioro de puntualidad (OTP 74.38 vs 79.93, –5.55 pts) y 14 cancelaciones/4 retrasos (incidentes NCS) en BRU-MAD y LIM-MAD, acompañado de menciones favorables a confort y amabilidad del personal. Contrariamente, Long Haul/Business cayó de 24.44 a 15.62 pts (–8.8 pts) por la misma pérdida de puntualidad e incidentes operativos en BRU-MAD y MAD-SCL. En Short Haul/Business IB subió de 34.81 a 50.0 pts (+15.2 pts) gracias a la experiencia a bordo en BRU-MAD (feedback de clientes), mientras YW bajó de 39.24 a 23.53 pts (–15.7 pts) por 6 cancelaciones y 6 retrasos (incidentes NCS), neutralizando el agregado de Business SH (39.53 pts, +4.2 pts). Economy SH mantuvo desempeño estable, pasando de 34.02 a 34.84 pts (+0.8 pts) sin drivers operativos relevantes.

Las rutas más afectadas fueron BRU-MAD/MAD-BRU, con 20 cancelaciones/retrasos combinados en Long Haul y Short Haul/Business, además de LIM-MAD (14 cancelaciones) y MAD-SCL (incidentes NCS en Business LH). Los perfiles más reactivos fueron Leisure en Economy LH (NPS 34.3, n=137) y Business/Work en Economy LH (NPS –21.2, n=33) y Business LH (NPS 5.6, n=18). En Short Haul/Business IB, Leisure premió el servicio a bordo (NPS 68.8, n=16).

ECONOMY SH: Estabilidad moderada  
La cabina Economy de SH mantuvo desempeño estable durante el 2025-11-24, registrando un NPS de 34.84 pts (2025-11-24) con una mejora de 0.8 pts vs L7d. No se detectaron cambios significativos en drivers operativos ni incidencias NCS, manteniendo niveles consistentes de satisfacción.

BUSINESS SH: Volatilidad interna oculta  
El segmento Business de SH mantuvo desempeño estable, registrando un NPS de 39.53 pts (2025-11-24) con una mejora de 4.2 pts vs L7d. No se identificaron drivers operativos agregados, ya que la subida de 15.2 pts en Global/SH/Business/IB (NPS 50.0 vs 34.81) por elogios a cortesía y confort en BRU-MAD (feedback de clientes) se compensó con la caída de 15.7 pts en Global/SH/Business/YW (NPS 23.53 vs 39.24) atribuible a 6 cancelaciones y 6 retrasos (incidentes NCS), resultando en un agregado equilibrado.

ECONOMY LH: Aumento de satisfacción pese a incidencias  
La cabina Economy de LH registró un NPS de 23.53 pts (2025-11-24), subiendo 11.3 pts vs L7d. La causa principal fue el deterioro de puntualidad (OTP 74.38 vs 79.93, –5.55 pts) y 14 cancelaciones/4 retrasos (incidentes NCS) en BRU-MAD (NPS 13.3, n=6) y LIM-MAD (NPS 29.4, n=17), complementada por menciones a confort y amabilidad del personal (feedback de clientes). El alza fue más visible en Leisure (NPS 34.3, n=137), mientras Business/Work reaccionó negativamente (NPS –21.2, n=33).

BUSINESS LH: Impacto de cancelaciones en la experiencia  
La cabina Business de LH registró un NPS de 15.62 pts (2025-11-24), cayendo 8.8 pts vs L7d. Los drivers principales fueron la pérdida de puntualidad (OTP 74.38 vs 79.93, –5.55 pts) y 14 cancelaciones/4 retrasos (incidentes NCS) en BRU-MAD y MAD-SCL (NPS 66.7, n=3), con fuerte impacto en Business/Work (NPS 5.6, n=18) y menor en Leisure (NPS 28.6, n=14).

PREMIUM LH: Rendimiento estable  
El segmento Premium de LH mantuvo desempeño estable, registrando un NPS de 18.18 pts (2025-11-24) con una mejora de 2.1 pts vs L7d. No se detectaron cambios significativos en datos operativos ni incidencias NCS, manteniendo niveles consistentes de satisfacción.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

TAREA:
1. Usa el header: **📈 Análisis semana del ([fechas]) con respecto a la semana anterior ([Segmento Principal]):**
2. Copia el contenido del análisis semanal del interpreter TAL COMO ESTÁ
3. Para cada sección (Párrafo 1, Párrafo 2, y cada sección de cabina/radio):
   - Mantén el contenido semanal TAL COMO ESTÁ
   - Añade DESPUÉS un párrafo adicional con el detalle diario correspondiente
   - Integra de forma fluida y natural, sin títulos ni separadores
   - El análisis diario debe fluir naturalmente después del análisis semanal
4. Orden de integración: Segmento Principal (párrafos 1 y 2), luego subsegmentos según jerarquía
5. Identifica días especialmente reseñables en el detalle diario (en orden cronológico)
6. NO cambies el contenido del análisis semanal del interpreter (ni cifras ni redondeos)
7. **TERMINOLOGÍA:** Reemplaza TODAS las menciones de "vs L7d", "vs L7D", "vs L7 días" por "con respecto a la semana anterior"
6. NO añadas recomendaciones adicionales
7. Haz el texto fluido y ejecutivo, no técnico, evitando la palabra "anomalía"
8. Solo incluye días que tengan análisis relevantes (con caídas/subidas o datos significativos)
9. Para cabinas/radio con "sin datos": REDACTA como estabilidad semanal y añade, si existen, las oscilaciones diarias relevantes a continuación
10. **CRÍTICO**: Si hay datos en "ANÁLISIS DIARIO SINGLE", DEBES usarlos. NO digas que "no están disponibles" si están presentes en el input.
11. **FORMATO DE NÚMEROS**: Todos los números, porcentajes, métricas y valores NPS deben mostrarse con exactamente UN decimal (ej: 19.8, -4.4, 93.5%)
12. **ATRIBUCIÓN DE SEGMENTO**: Siempre que menciones un dato, indica a qué segmento pertenece (ej: "NPS 19.8 (Economy LH)", "OTP –4.0 pts (Business SH)")