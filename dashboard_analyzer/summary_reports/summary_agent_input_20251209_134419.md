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

En Global/SH/Economy, el escenario es SINERGIA `(IB –5.6, YW –7.5 | Economy –6.2)`.  
- **Narrativa:** Ambas subcompañías (IB y YW) presentan anomalías negativas que se suman en el nodo padre. La caída de –6.2 pts en Economy SH obedece a un repunte de incidentes operativos (mishandling y pérdidas de conexión) en rutas clave, sin contrapeso interno.  
- **Evidencia Clave:** Mishandling +3.5 incidentes y misconexiones +0.2 (operative_data_tool); NCS “OTRAS_INCIDENCIAS” +16 y “RETRASOS” +12 (ncs_tool).

En Global/SH/Business, el escenario es DILUCIÓN `(IB N, YW +23.0 | Business N)`.  
- **Narrativa:** La normalidad del nodo padre se explica porque la anomalía positiva de YW (+23.0 pts) por mejoras en producto/servicio se ve neutralizada por la estabilidad de IB (NPS_diff +0.3), diluyendo su efecto en el agregado.  
- **Evidencia Clave:** Mejoras en Aircraft interior, Arrivals experience y Punctuality (SHAP 4.596) en YW (explanatory_drivers_tool).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul (SH), la dinámica es TRANSFERENCIA `(Economy –6.2, Business N | SH –5.0)`.  
- **Narrativa:** La caída de –5.0 pts en SH se transmite íntegramente desde Economy, ya que Business mantuvo su desempeño.  
- **Evidencia:** Economy SH sufrió un repunte de mishandling +3.5 y misconexiones +0.2 (operative_data_tool) y un alza de “OTRAS_INCIDENCIAS” +16 y “RETRASOS” +12 (ncs_tool).

En Long Haul (LH), la dinámica es DOMINANCIA `(Economy –10.4, Business –20.2, Premium +7.0 | LH –10.4)`.  
- **Narrativa:** El performance de LH está dictado por la fuerte caída en Business, debido al deterioro operacional en puntualidad y manejo de equipaje, pese a la mejora en Premium que atenuó parcialmente el impacto.  
- **Evidencia:** Business LH mostró SHAP Punctuality –8.189 y caída de OTP15 –4.0 pts, además de mishandling +3.5 (explanatory_drivers_tool + operative_data_tool).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es SINERGÍA `(LH –10.4, SH –5.0 | Global –7.5)`.  
- **Narrativa:** La red entera sufrió un deterioro operativo común que impactó tanto al Largo Radio como al Corto Radio, arrastrando el NPS Global –7.54 pts.  
- **Evidencia:**  
  • Punctuality (SHAP Global = –1.323) con OTP15 –0.5 pts (operative_data_tool)  
  • Mishandling +3.5 y Misconnections +0.2 (operative_data_tool)  
  • Incidentes NCS totales +205 (otras_incidencias +70, limitación_aeronave +43, cancelaciones +10, gestiones de conexión +137%)

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Deterioro en Puntualidad  
- Escenario: SINERGIA (LH –, SH – | Global –)  
- NMA: Global  
- Afecta a: Global/LH (NPS_diff –10.4), Global/SH (NPS_diff –5.0)  
- Qué falló: Punctuality (SHAP = –1.323) (Global)  
- Dónde:  
  • BIO-VLC: NPS –50.0 pts, 4 pax (Global)  
  • MAD-OSL: NPS –37.5 pts, 8 pax (Global)  
  • GVA-MAD: NPS –25.0 pts, 4 pax (Global)  
- Quién:  
  • Fleet: spread 95.1 pts (Global)  
  • Residence Region: spread 82.2 pts (Global)  
  • CodeShare: spread 80.3 pts (Global)  
- Evidencia:  
  • NPS 22.57 vs baseline 30.12 (Global)  
  • OTP15 –0.5 pts (operative_data_tool, Global)  
  • Flight cancellations +10 eventos (ncs_tool, Global)  

CAUSA 2: Fallos en Manejo de Equipaje y Conexiones  
- Escenario: SINERGIA (LH –, SH – | Global –)  
- NMA: Global  
- Afecta a: Global/LH (NPS_diff –10.4), Global/SH (NPS_diff –5.0)  
- Qué falló:  
  • Mishandling +3.5 incidentes (operative_data_tool, Global)  
  • Misconnections +0.2 incidentes (operative_data_tool, Global)  
- Dónde:  
  • BIO-VLC: NPS –50.0 pts, 4 pax (Global)  
  • MAD-OSL: NPS –37.5 pts, 8 pax (Global)  
  • GVA-MAD: NPS –25.0 pts, 4 pax (Global)  
- Quién:  
  • Fleet: spread 95.1 pts (Global)  
  • Residence Region: spread 82.2 pts (Global)  
  • CodeShare: spread 80.3 pts (Global)  
- Evidencia:  
  • NCS totales +205 incidentes (ncs_tool, Global)  
  • “otras_incidencias” +70, “gestiones de conexión” +137 % (ncs_tool, Global)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

Durante el periodo 2025-11-29 a 2025-12-02 el NPS Global cayó de 30.12 a 22.57 (–7.54 pts vs L7d), reflejo de descensos en Long Haul (LH) de 17.21 a 6.84 (–10.37 pts vs L7d) y en Short Haul (SH) de 36.47 a 31.48 (–4.99 pts vs L7d). A nivel de cabina, Economy SH descendió de 36.17 a 30.01 (–6.16 pts vs L7d), con IB SH de 33.30 a 27.75 (–5.55 pts vs L7d) e YW SH de 42.07 a 34.62 (–7.45 pts vs L7d); Economy LH cayó de 13.63 a 3.19 (–10.43 pts vs L7d); Business LH bajó de 32.70 a 12.50 (–20.20 pts vs L7d); mientras Premium LH subió de 29.60 a 36.62 (+7.02 pts vs L7d). El deterioro global obedece principalmente a un empeoramiento en Punctuality (SHAP –1.323, OTP15 –0.5 pts Global) junto a un incremento de mishandling (+3.5 incidentes Global), misconnections (+0.2 incidentes Global) y 205 incidentes NCS (otras_incidencias +70, limitación_aeronave +43, cancelaciones +10, gestiones de conexión +137 %). En contraste, la mejora de +7.02 pts en Premium LH se atribuye a la excelencia en producto/servicio (Aircraft interior, Cabin Crew, Arrivals experience; apoyado en verbatims como “Excelente trato del personal…”), pese a que factores operativos (Punctuality SHAP –6.675, OTP15 –4.0 pts Premium LH, mishandling +3.5, misconnections +0.2) habían actuado en sentido contrario.

Las rutas más críticas incluyen IAD–MAD con NPS –71.4 (7 pax, Economy LH), MAD–ORD con NPS –57.7 (26 pax, Economy LH) y –28.6 (7 pax, Business LH), BIO–VLC con NPS –50.0 (4 pax, Economy SH) y MAD–OSL con –37.5 (8 pax, Economy SH). Los casos de alza destacan MAD–MIA con +40.0 (5 pax, Premium LH) y MAD–MCO con +33.3 (3 pax, Premium LH). Entre perfiles, Fleet mostró spread de 113.2 pts (Economy LH) y 95.1 pts (Global), Residence Region spread de 97.2 pts (LH) y 82.2 pts (Global), CodeShare spread de 98.2 pts (Economy LH) y 80.3 pts (Global), mientras Business/Leisure mantuvo variaciones moderadas.

ECONOMY SH: Deterioro por Incidentes Operativos  
La cabina Economy de SH registró un NPS de 30.01 (Global/SH/Economy) durante la semana del 2025-11-29 a 2025-12-02, con una caída de –6.16 pts vs L7d. IB SH pasó de 33.30 a 27.75 (–5.55 pts vs L7d) y YW SH de 42.07 a 34.62 (–7.45 pts vs L7d). La causa principal fue un repunte de mishandling +3.5 incidentes (operative_data_tool, Global/SH/Economy) y misconnections +0.2 incidentes (operative_data_tool, Global/SH/Economy), junto con “OTRAS_INCIDENCIAS” +16 y “RETRASOS” +12 (ncs_tool, Global/SH/Economy). Esta bajada se reflejó especialmente en BIO–VLC (NPS –50.0, 4 pax) y MAD–OSL (NPS –37.5, 8 pax), siendo los más sensibles los clientes según Residence Region (spread 142.0 pts, Global/SH/Economy) y CodeShare (spread 133.9 pts, Global/SH/Economy).

BUSINESS SH: Desempeño Estable  
El segmento Business de SH mantuvo desempeño estable con un NPS de 46.31 (Global/SH/Business) durante la misma semana, mostrando una variación de +6.77 pts vs L7d que se ubica dentro de la normalidad. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción.

ECONOMY LH: Impacto Crítico de Puntualidad y Equipaje  
La cabina Economy de LH sufrió un descenso de –10.43 pts vs L7d, pasando de un NPS de 13.63 a 3.19 (Global/LH/Economy). El driver principal fue Punctuality (SHAP –1.842, explanatory_drivers_tool) agravado por una caída de OTP15 –4.0 pts (operative_data_tool, Economy LH), junto al aumento de mishandling +3.5 incidentes y misconnections +0.2 incidentes (operative_data_tool, Economy LH), reforzado por +70 incidentes totales (retrasos +10, cancelaciones +3; ncs_tool, Economy LH). Esta baja se concentró en IAD–MAD (NPS –71.4, 7 pax) y MAD–ORD (NPS –64.7, 17 pax), y los perfiles de Fleet (spread 113.2 pts, Global/LH/Economy) y CodeShare (spread 98.2 pts, Global/LH/Economy) resultaron los más afectados.

BUSINESS LH: Caída Brusca por Punctuality  
La cabina Business de LH registró un NPS de 12.50 (Global/LH/Business), cayendo de 32.70 a 12.50 (–20.20 pts vs L7d). El principal driver operativo fue Punctuality (SHAP –8.189, explanatory_drivers_tool) con OTP15 –4.0 pts (operative_data_tool, Business LH) y mishandling +3.5 incidentes (operative_data_tool, Business LH), junto a 70 incidentes totales (ncs_tool, Business LH). El impacto fue más evidente en MAD–ORD (NPS –28.6, 7 encuestas) y JFK–MAD (NPS –16.7, 6 encuestas), con Residence Region (spread 123.5 pts, Global/LH/Business) y CodeShare (spread 103.7 pts, Global/LH/Business) como los perfiles más sensibles.

PREMIUM LH: Excepcional Alza por Calidad de Servicio  
El segmento Premium de LH subió de 29.60 a 36.62 (Global/LH/Premium), con una ganancia de +7.02 pts vs L7d. Las causas dominantes fueron mejoras en drivers de producto/servicio: Aircraft interior, Cabin Crew y Arrivals experience (explanatory_drivers_tool), respaldadas por verbatims como “Excelente trato del personal…” (verbatims_tool). Este repunte se reflejó en MAD–MIA (NPS +40.0, 5 pax) y MAD–MCO (NPS +33.3, 3 pax), y los grupos más reactivos fueron CodeShare (spread 141.7 pts, Global/LH/Premium) y Residence Region (spread 104.6 pts, Global/LH/Premium).

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

Economy SH  
- Escenario: SINERGIA (IB –6.1, YW –19.3 | Padre –10.4)  
- Narrativa: Ambas subcabinas presentan anomalías negativas que se suman para arrastrar el NPS de Economy SH. Por coherencia con el nodo padre, la caída de –10.4 pts se explica por la concentración de valoraciones neutras o negativas en rutas y perfiles específicos.  
- Evidencia Clave: Ruta EAS–MAD con NPS 0.0 (n=6); flotas A333/A319 con puntuaciones bajas; code‐share AA y residentes en América Norte/Oriente Medio.  

Business SH  
- Escenario: DOMINANCIA (IB –2.2, YW +43.6 | Padre +13.2)  
- Narrativa: El fuerte impulso positivo de YW prevalece sobre la ligera caída de IB. El NPS de +13.2 pts en Business SH se explica fundamentalmente por el salto de +43.6 pts de YW, mientras que la anomalía negativa de IB (–2.2 pts) apenas atenúa el resultado.  
- Evidencia Clave: YW – baja ocupación (Load Factor 54.43), alta puntualidad (OTP15_adjusted 91.19) y verbatims 100 % positivos (NPS 76.92).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul (SH)  
- Dinámica: DOMINANCIA (Economy –, Business + | SH –)  
- Narrativa: El NPS de Short Haul está dictado por la caída de Economy SH, cuyo descenso de –10.4 pts arrastra el radio a –7.7 pts, mientras que la mejora de Business SH (+13.2 pts) solo atenúa parcialmente ese efecto.  
- Evidencia: Economy SH afectada por la ruta EAS–MAD con NPS 0.0 (n=6), bajas valoraciones en flotas A333/A319 y code-share AA con residentes en América Norte/Oriente Medio.  

Long Haul (LH)  
- Dinámica: DOMINANCIA (Economy +, Business –, Premium + | LH +)  
- Narrativa: El NPS de Long Haul responde principalmente al fuerte impulso positivo de Premium LH, que con +13.0 pts supera la caída de Business LH (–11.1 pts) y refuerza el resultado global de +7.2 pts.  
- Evidencia: Premium LH potenciado por vuelos en A350 next (NPS 77.8) y A350 (NPS 66.7), frente a la insatisfacción en A333 (NPS –40.0).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es DOMINANCIA (LH +7.2, SH –7.7 | Global –2.6).  
- Narrativa: El NPS global está arrastrado por el bajo rendimiento de Short Haul, cuya caída de –7.7 pts—y en particular el desplome de Economy SH (–10.4 pts)—superó el impulso positivo de Long Haul (+7.2 pts).  
- Evidencia: Economy SH impactada por la ruta EAS–MAD con NPS 0.0 (n=6), bajas valoraciones en flotas A333/A319 y code-share AA con pasajeros de América Norte/Oriente Medio.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Caída de NPS en Economy Short Haul  
- Escenario: SINERGIA (IB –6.12 pts, YW –19.28 pts | Economy SH –10.45 pts)  
- NMA: Global/SH/Economy  
- Afecta a:  
  • Global/SH/Economy  
  • Global/SH/Economy/IB  
  • Global/SH/Economy/YW  
- Qué falló: concentración de valoraciones neutrales o negativas sin fallos operativos, indicando insatisfacción puntual en Economy SH (Global/SH/Economy)  
- Dónde: ruta EAS–MAD con NPS 0.0 (n=6) (Global/SH/Economy)  
- Quién: pasajeros residentes en América Norte (NPS –34.4 pts, Global/SH/Economy) y Oriente Medio (NPS –33.3 pts, Global/SH/Economy), y vuelos code-share AA (Global/SH/Economy)  
- Evidencia:  
  • NPS 21.64 vs baseline 32.09 pts (Global/SH/Economy)  
  • Load Factor 86.20 vs 86.55 (–0.35) (Global/SH/Economy)  
  • OTP15_adjusted 92.17 vs 89.45 (+2.72) (Global/SH/Economy)  
  • Sin incidentes NCS registrados (Global/SH/Economy)  
  • Dispersión de flota 137.5 pts (Global/SH/Economy)  

CAUSA 2: Impulso de NPS en Young & Young de Business Short Haul  
- Escenario: DOMINANCIA (IB –2.17 pts, YW +43.57 pts | Business SH +13.23 pts)  
- NMA: Global/SH/Business/YW  
- Afecta a:  
  • Global/SH/Business/YW  
- Qué falló: no es un fallo, sino un driver positivo: menor hacinamiento y alta puntualidad mejoraron la experiencia (Global/SH/Business/YW)  
- Dónde: no hay rutas con muestra suficiente para segmentar (Global/SH/Business/YW)  
- Quién: pasajeros YW (Global/SH/Business/YW)  
- Evidencia:  
  • NPS 76.92 vs baseline 33.36 pts (Global/SH/Business/YW)  
  • Load Factor 54.43 vs 58.31 (–3.88) (Global/SH/Business/YW)  
  • OTP15_adjusted 91.19 vs 87.81 (+3.38) (Global/SH/Business/YW)  
  • 15 verbatims 100 % positivos (Global/SH/Business/YW)  
  • Sin incidentes NCS (Global/SH/Business/YW)  

CAUSA 3: Alza de NPS en Premium Long Haul  
- Escenario: DOMINANCIA (Economy +7.16 pts, Business –11.13 pts, Premium +13.02 pts | LH +7.16 pts)  
- NMA: Global/LH/Premium  
- Afecta a:  
  • Global/LH/Premium  
- Qué falló: no es un fallo, sino un driver positivo: mayor peso de vuelos en A350/A350 next elevó la satisfacción (Global/LH/Premium)  
- Dónde: sin rutas destacadas con suficientes encuestas (Global/LH/Premium)  
- Quién: pasajeros en A350 next (NPS 77.8 pts, Global/LH/Premium) y A350 (NPS 66.7 pts, Global/LH/Premium)  
- Evidencia:  
  • NPS 47.83 vs baseline 34.81 pts (Global/LH/Premium)  
  • Load Factor 88.33 vs 90.63 (–2.30) (Global/LH/Premium)  
  • OTP15_adjusted 76.06 vs 78.90 (–2.84) (Global/LH/Premium)  
  • 39 verbatims 100 % positivos (Global/LH/Premium)  
  • Dispersión de flota 117.8 pts (Global/LH/Premium)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El 2 de diciembre, el NPS global retrocedió de 26.01 a 23.46 (–2.55 pts) como resultado de dos movimientos opuestos: Short Haul cayó 33.14→25.44 (–7.70 pts) y Long Haul subió 12.51→19.66 (+7.16 pts). En Short Haul, Economy SH se hundió 32.09→21.64 (–10.45 pts) con IB en 29.65→23.53 (–6.12 pts) y YW en 37.11→17.82 (–19.28 pts), debido a valoraciones negativas en la ruta EAS–MAD (Global/SH/Economy NPS 0.0, n=6), bajas calificaciones en flotas A333 (–100.0, n=5) y A319 (–26.7, n=15), y decepción de clientes code-share AA y residentes en América Norte (Global/SH/Economy NPS –34.4) y Oriente Medio (Global/SH/Economy NPS –33.3). En paralelo, Business SH subió 43.53→56.76 (+13.23 pts), liderado por Young & Young con 33.36→76.92 (+43.57 pts), impulsado por baja ocupación (Load Factor 69.49, –3.19 pts vs L7d) y alta puntualidad (OTP15_adjusted 92.17, +2.72 pts vs L7d), reflejados en verbatims 100 % positivos. Por su parte, en Long Haul, Economy LH mejoró 8.94→16.13 (+7.19 pts) gracias a la percepción de servicio y puntualidad (212 verbatims positivos), pese a ligeras caídas operativas (Load Factor 87.74, –2.54 pts vs L7d; OTP15_adjusted 76.06, –2.84 pts vs L7d). Business LH se contrajo 24.03→12.90 (–11.13 pts), sin incidentes NCS, por la insatisfacción de perfiles América Norte (Global/LH/Business NPS –50.0, n=4) y Europa (Global/LH/Business NPS –42.9, n=14) en aviones A350 next (–25.0, n=2) y code-share AA Business Beginning (–100.0, n=2). Premium LH escaló 34.81→47.83 (+13.02 pts) gracias al mayor peso de A350 next (Global/LH/Premium NPS 77.8) y A350 (66.7), con 39 verbatims 100 % positivos.

Entre las rutas, la más afectada fue EAS–MAD (Global/SH/Economy NPS 0.0, n=6), seguida de MAD–SDQ en Economy LH (Global/LH/Economy NPS 33.3, n=3), mientras que LIM–MAD (Global/LH/Business NPS 100.0, n=3) y LHR–MAD (Global/SH/Business NPS 100.0, n=3) marcaron los picos positivos. Los grupos de clientes más reactivos incluyeron a residentes de América Norte y Oriente Medio en Economy SH, pasajeros Leisure de América Centro y España en Economy LH, así como usuarios de code-share AA y flotas A333/A319 con mayor dispersión de NPS.

ECONOMY SH: Desplome por perfiles críticos  
La cabina Economy de SH durante la semana del 2 de diciembre registró un NPS de 21.64 (2 de diciembre) con un descenso de 10.45 pts vs L7d. El NPS de IB cayó de 29.65 a 23.53 (–6.12 pts) y el de YW de 37.11 a 17.82 (–19.28 pts). La causa principal fue la acumulación de valoraciones negativas en la ruta EAS–MAD (Global/SH/Economy NPS 0.0, n=6), junto a bajas calificaciones en flotas A333 (Global/SH/Economy NPS –100.0, n=5) y A319 (Global/SH/Economy NPS –26.7, n=15) y entre clientes code-share AA (Global/SH/Economy NPS –54.5) y residentes en América Norte (Global/SH/Economy NPS –34.4) y Oriente Medio (Global/SH/Economy NPS –33.3).

BUSINESS SH: Impulso por baja ocupación  
La cabina Business de SH durante la semana del 2 de diciembre registró un NPS de 56.76 (2 de diciembre) con una mejora de 13.23 pts vs L7d. Mientras IB retrocedió de 48.00 a 45.83 (–2.17 pts), YW escaló de 33.36 a 76.92 (+43.57 pts). Esta evolución se explica por la baja ocupación (Load Factor 69.49, –3.19 pts vs L7d) y la alta puntualidad (OTP15_adjusted 92.17, +2.72 pts vs L7d), reflejados en verbatims 100 % positivos (n=15, Global/SH/Business/YW).

ECONOMY LH: Recuperación por calidad de servicio  
La cabina Economy de LH durante la semana del 2 de diciembre registró un NPS de 16.13 (2 de diciembre) con una mejora de 7.19 pts vs L7d. A pesar de una ligera degradación operacional (Load Factor 87.74, –2.54 pts vs L7d; OTP15_adjusted 76.06, –2.84 pts vs L7d), predominó la valoración positiva de la tripulación y la puntualidad, con 212 verbatims favorables (Global/LH/Economy). La subida fue más notoria en la ruta MAD–SDQ (Global/LH/Economy NPS 33.3, n=3).

BUSINESS LH: Caída atribuida a perfiles específicos  
La cabina Business de LH durante la semana del 2 de diciembre registró un NPS de 12.90 (2 de diciembre) con un descenso de 11.13 pts vs L7d. Sin incidentes NCS, el deterioro responde a la insatisfacción de pasajeros de América Norte (Global/LH/Business NPS –50.0, n=4) y Europa (Global/LH/Business NPS –42.9, n=14), así como a vuelos en A350 next (Global/LH/Business NPS –25.0, n=2) y code-share AA Business Beginning (Global/LH/Business NPS –100.0, n=2).

PREMIUM LH: Sólida mejora por flotas  
La cabina Premium de LH durante la semana del 2 de diciembre registró un NPS de 47.83 (2 de diciembre) con una mejora de 13.02 pts vs L7d. El alza se explica por el mayor peso de vuelos en A350 next (Global/LH/Premium NPS 77.8) y A350 (Global/LH/Premium NPS 66.7), reflejado en 39 verbatims 100 % positivos, mientras que la baja valoración de A333 (Global/LH/Premium NPS –40.0) solo mitigó marginalmente la tendencia.

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

En Economy SH, el escenario es SINERGIA (–, – | –).  
- Narrativa: Adoptamos la explicación del nodo padre. La caída de –8.7 pts en NPS se debe a la concentración de valoraciones extremas en la ruta MAD–ZRH, con viajeros Business y residentes en América Norte aportando puntuaciones muy bajas, reforzado además por vuelos en flota 32S y code-share AA.  
- Evidencia Clave: Ruta MAD–ZRH (Global/SH/Economy); perfiles Business, América Norte, CodeShare AA, Flota 32S.  

En Business SH, el escenario es DOMINANCIA (–, + | –).  
- Narrativa: Adoptamos la explicación del hijo dominante (IB). La anomalía negativa de –7.8 pts se origina principalmente en Global/SH/Business/IB, donde clientes Business en la ruta MAD–ORY registraron una insatisfacción notable. Este impacto fue parcialmente suavizado por el rendimiento positivo del subsegmento YW.  
- Evidencia Clave: Global/SH/Business/IB – caída de –21.7 pts por clientes Business en ruta MAD–ORY (NPS 25.0, n=4).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es SINERGIA (Economy SH –, Business SH – | SH –).  
- Narrativa: La caída de –8.7 pts en NPS Short Haul se explica por un punto de dolor común a ambas cabinas: valoraciones extremadamente bajas en la ruta BRU–MAD, impulsadas por vuelos en code-share con AA e I2 y clientes residentes en América Norte.  
- Evidencia: Global/SH –8.7 pts; ruta BRU–MAD (NPS 0.0, n=6), CodeShare AA e I2 y región América Norte.  

En Long Haul, la dinámica es DOMINANCIA (Economy LH –, Business LH –, Premium LH + | LH –).  
- Narrativa: El descenso de –3.6 pts en NPS Long Haul está liderado por la fuerte caída de Business LH (–29.0 pts), atribuible a deficiencias en servicio de cabina (actitud de tripulación, catering y entretenimiento) con mayor impacto en pasajeros de Europa y Norteamérica. Este efecto fue parcialmente mitigado por el sobresaliente desempeño de Premium LH (+27.7 pts).  
- Evidencia: Global/LH/Business –29.0 pts en NPS; Global/LH/Premium +27.7 pts en NPS.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es SINERGIA (–, – | –).  
- Narrativa: Adoptamos la explicación del nodo Global. La caída de –7.52 pts en el NPS global responde a un problema sistémico de red: la insatisfacción de viajeros de negocio en rutas clave (sin incidencias operativas), que se reflejó tanto en Long Haul como en Short Haul.  
- Evidencia:  
  • Global –7.5 pts (NPS 18.49 vs 26.01)  
  • Ruta LHR–MAD –26.3 pts  
  • Business Global NPS 5.5  
  • Flotas A321XLR y A350 C  
  • CodeShare QR e I2

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Insatisfacción sistémica en Short Haul  
- Escenario: SINERGIA (Global/SH/Economy –, Global/SH/Business – | Global/SH –)  
- NMA: Global/SH  
- Afecta a: Global/SH/Economy y Global/SH/Business  
- Qué falló: Valoraciones extremas negativas en vuelos code-share y en ruta crítica (Global/SH)  
- Dónde:  
  • BRU–MAD: NPS 0.0 (Global/SH, n=6)  
  • MAD–ZRH: NPS 0.0 (Global/SH/Economy, n=4)  
  • MAD–ORY: NPS 25.0 (Global/SH/Business, n=4)  
- Quién:  
  • Business (NPS 8.9, Global/SH/Business)  
  • Residentes en América Norte (NPS –8.3, Global/SH/Economy)  
  • CodeShare AA (NPS –60.0, Global/SH/Economy) e I2 (NPS –33.3, Global/SH/Economy)  
- Evidencia:  
  • NPS 24.46 vs baseline 33.14 (Global/SH)  
  • Load Factor 85.07% vs +0.31 (Global/SH)  
  • OTP15_adjusted 92.30% vs +2.87 (Global/SH)  
  • 0 incidentes NCS reportados  

CAUSA 2: Deficiencias de servicio en Long Haul Business  
- Escenario: DOMINANCIA (Economy LH –, Business LH –, Premium LH + | LH –)  
- NMA: Global/LH/Business  
- Afecta a: Global/LH/Business  
- Qué falló: Actitud poco profesional de tripulación, insatisfacción con catering y entretenimiento (Global/LH/Business)  
- Dónde:  
  • MAD–MEX: NPS 33.3 (Global/LH/Business, n=3) – sin volumen suficiente para conclusiones por ruta  
  • BOS–MAD: NPS 0.0 (Global/LH, n=6) como indicador de contexto en LH  
- Quién:  
  • Residentes en Europa (NPS –100.0, Global/LH/Business)  
  • Residentes en América Norte (NPS –60.0, Global/LH/Business)  
- Evidencia:  
  • NPS –5.0 vs baseline 24.03 (Global/LH/Business)  
  • Load Factor 93.66% vs –0.32 (Global/LH/Business)  
  • OTP15_adjusted 77.96% vs –1.12 (Global/LH)  
  • 24 verbatims con quejas sobre tripulación, catering y entretenimiento  
  • 0 incidentes NCS reportados

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
El NPS global se contrajo de 26.01 a 18.49 pts (–7.52 pts vs L7d Global) el 1 de diciembre. En Long Haul, el NPS pasó de 12.51 a 8.87 pts (–3.64 pts vs L7d Global/LH) debido principalmente a la fuerte caída de Business LH, que descendió de 24.03 a –5.00 pts (–29.03 pts vs L7d Global/LH/Business), aunque Premium LH logró un repunte desde 34.81 hasta 62.50 pts (+27.69 pts vs L7d Global/LH/Premium). En Short Haul, el NPS bajó de 33.14 a 24.46 pts (–8.68 pts vs L7d Global/SH), con Economy SH reduciéndose de 32.09 a 23.41 pts (–8.68 pts vs L7d Global/SH/Economy) y Business SH de 43.53 a 35.71 pts (–7.82 pts vs L7d Global/SH/Business).

Las causas identificadas apuntan a dos focos: en Short Haul, valoraciones extremas negativas en rutas como BRU–MAD (NPS 0.0, Global/SH) y MAD–ZRH (NPS 0.0, Global/SH/Economy), impulsadas por vuelos codeshare (AA e I2) y clientes Business y residentes en América Norte. En Long Haul, el deterioro de Business LH se originó en deficiencias de servicio a bordo (actitud de tripulación, catering y entretenimiento) con impacto crítico en pasajeros de Europa y Norteamérica, mientras que el impulso de Premium LH vino de la sobresatisfacción del segmento Leisure en la ruta MAD–MEX (NPS 75.0, Global/LH/Premium).

Rutas más afectadas incluyen LHR–MAD (NPS –26.3 pts, Global), BRU–MAD (0.0 pts, Global/SH), MAD–ZRH (0.0 pts, Global/SH/Economy) y EZE–MAD (12.5 pts, Global/LH/Economy). Los grupos más reactivos fueron los viajeros de negocio (Business)—especialmente residentes en Europa y América Norte—y el segmento Leisure en Premium LH, junto a pasajeros de vuelos codeshare QR, AA e I2.

  
ECONOMY SH: Caída pronunciada en Economy SH  
En Global/SH/Economy/IB el NPS fue 22.28 pts (–7.36 pts vs L7d) y en Global/SH/Economy/YW 25.22 pts (–11.89 pts vs L7d). En agregado, la cabina Economy de SH registró un NPS de 23.41 pts (1 dic. 2025) con una bajada de 8.68 pts vs L7d Global/SH/Economy. La principal causa fue la concentración de valoraciones en la ruta MAD–ZRH (NPS 0.0, Global/SH/Economy) y la ruta BRU–MAD (NPS 0.0, Global/SH), junto a la insatisfacción de clientes Business (NPS 9.0, Global/SH/Economy), residentes en América Norte (NPS –37.5, Global/SH/Economy) y usuarios de CodeShare AA (NPS –60.0, Global/SH/Economy) e I2 (NPS –33.3, Global/SH/Economy).

BUSINESS SH: Divergencia marcada en Business SH  
En Global/SH/Business/IB el NPS cayó a 26.32 pts (–21.69 pts vs L7d) mientras que Global/SH/Business/YW subió a 55.56 pts (+22.20 pts vs L7d). En conjunto, la cabina Business de SH registró un NPS de 35.71 pts (1 dic. 2025) con una bajada de 7.82 pts vs L7d Global/SH/Business. El descenso se explica principalmente por la insatisfacción de pasajeros Business en la ruta MAD–ORY (NPS 25.0, Global/SH/Business/IB), especialmente residentes en España (NPS 0.0, Global/SH/Business/A320neo) y viajando en A320neo.

ECONOMY LH: Descenso moderado en Economy LH  
La cabina Economy de LH registró un NPS de 5.39 pts (1 dic. 2025) con una bajada de 3.55 pts vs L7d Global/LH/Economy. El deterioro provino de viajeros Business/Work (NPS –23.8, Global/LH/Economy) en la ruta EZE–MAD (NPS 12.5, Global/LH/Economy), con fuerte insatisfacción de residentes en Europa (NPS –50.0, Global/LH/Economy) y América Norte (NPS –21.1, Global/LH/Economy), y por la experiencia en flotas A321XLR y A350 C (ambas con NPS –33.3, Global/LH/Economy).

BUSINESS LH: Fuerte caída en Business LH  
La cabina Business de LH registró un NPS de –5.00 pts (1 dic. 2025) con una bajada de 29.03 pts vs L7d Global/LH/Business. Los drivers principales fueron deficiencias de servicio a bordo—actitud de tripulación, catering y entretenimiento (24 verbatims, Global/LH/Business)—con impacto crítico en residentes en Europa (NPS –100.0, Global/LH/Business) y América Norte (NPS –60.0, Global/LH/Business).

PREMIUM LH: Sólido repunte en Premium LH  
El segmento Premium de LH alcanzó un NPS de 62.50 pts (1 dic. 2025), mejorando 27.69 pts vs L7d Global/LH/Premium. Este avance se debe a la sobresatisfacción del segmento Leisure (NPS 72.7, Global/LH/Premium) en la ruta MAD–MEX (NPS 75.0, Global/LH/Premium), apoyada en 22 verbatims mayoritariamente positivos y cero incidentes operativos.

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

En SH Economy, el escenario es CANCELACIÓN `(-, + | N)`.  
- Narrativa: Se contrastan las causas opuestas de los hijos IB y YW, que en conjunto neutralizan el efecto en el padre. Mientras Global/SH/Economy/IB sufrió una caída de –1.9 pts por la concentración de respuestas insatisfechas en la ruta MAD–VGO con vuelos code share BA y pasajeros de Norteamérica, Global/SH/Economy/YW registró un alza de +11.6 pts gracias a la elevada satisfacción de pasajeros de ocio en la ruta MAD–MUC, con un desempeño operativo sólido. Estos movimientos contrarios se anularon, dejando el NPS agregado dentro de la variación normal.  
- Evidencia Clave:  
  • Global/SH/Economy/IB: Ruta MAD–VGO, NPS 0.0 (n=4), spread de CodeShare BA de 133.3 pts.  
  • Global/SH/Economy/YW: Ruta MAD–MUC, NPS 53.3 (Leisure), OTP15_adjusted +3.79 pts.

En SH Business, el escenario es SINERGIA `(-, - | -)`.  
- Narrativa: Ambos subsegmentos IB y YW empujan en la misma dirección negativa y sus efectos se suman en el padre. La explicación del nodo Global/SH/Business aplica por igual a IB (–8.6 pts) y a YW (–15.2 pts): la caída de –9.4 pts en el NPS Business SH se debe primordialmente a la percepción muy negativa de clientes en vuelos code share BA, sin que existan incidencias operativas formales que lo justifiquen.  
- Evidencia Clave:  
  • Global/SH/Business: NPS 34.09 vs baseline 43.53 (–9.43 pts).  
  • CodeShare BA en Business SH: NPS –66.7, spread de 106.1 pts frente a IB.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es DILUCIÓN `(Economy N, Business – | SH N)`.  
- Narrativa: El desempeño agregado de Short Haul está dictado por la caída en Business SH, cuyo NPS bajó –9.4 pts debido a la percepción muy negativa de pasajeros en vuelos code share BA, aunque este efecto fue parcialmente atenuado por la estabilidad de Economy SH.  
- Evidencia:  
  • Global/SH/Business –9.4 pts (NPS 34.09 vs baseline 43.53) impulsado por CodeShare BA con NPS –66.7.  
  • Global/SH/Economy Normal (NPS 34.81 vs baseline 32.09).

En Long Haul, la dinámica es SINERGIA `(-, -, - | -)`.  
- Narrativa: Todas las cabinas reflejan la misma anomalía negativa, por lo que la explicación de Global/LH aplica de forma transversal: la fuerte caída de –21.1 pts se concentra en la ruta LIM–MAD, donde viajeros Leisure y pasajeros en flotas A350, A332, A333 y A321XLR procedentes de Europa, Norteamérica y Asia manifestaron una experiencia insatisfecha.  
- Evidencia:  
  • Global/LH –21.1 pts (NPS –8.60 vs baseline 12.51) en ruta LIM–MAD con NPS –11.8 (n=17).  
  • Global/LH/Economy –22.6 pts, Global/LH/Business –2.8 pts y Global/LH/Premium –30.6 pts, todos impactados por la misma ruta y perfiles.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA `(-, N | -)`.  
- Narrativa: El resultado Global está arrastrado por la anomalía en Long Haul. La caída de –8.1 pts en el NPS global se explica por el descenso de –21.1 pts en la red de Largo Radio, concentrado en la ruta LIM–MAD, donde viajeros Leisure y pasajeros en flotas A350, A332, A333 y A321XLR procedentes de Europa, Norteamérica y Asia manifestaron percepciones de servicio negativas.  
- Evidencia:  
  • Global/LH –21.1 pts (NPS –8.60 vs baseline 12.51) en ruta LIM–MAD con NPS –11.8 (n=17).  
  • Global/SH Normal (+1.6 pts), sin impacto significativo en el Global.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Insatisfacción sistémica en Long Haul  
- Escenario: SINERGIA `(Global/LH/Economy –, Global/LH/Business –, Global/LH/Premium – | Global/LH –)`  
- NMA: Global/LH  
- Afecta a: Global/LH/Economy, Global/LH/Business, Global/LH/Premium  
- Qué falló: Percepción de calidad de servicio en la ruta LIM–MAD (Global/LH), sin que incidencias operativas formales (NCS) lo justifiquen.  
- Dónde: Ruta LIM–MAD NPS –11.8 (n=17) (Global/LH)  
- Quién: Pasajeros Leisure y Business en flotas A350, A332, A333 y A321XLR procedentes de Europa, Norteamérica y Asia (Global/LH, perfiles)  
- Evidencia:  
   • NPS –8.5987 vs baseline 12.5078 (Global/LH)  
   • Load_Factor 88.0 (–2.68 pts vs media) (Global/LH)  
   • OTP15_adjusted 78.91 (–0.51 pts vs media) (Global/LH)

CAUSA 2: Experiencia negativa en vuelos code-share BA (Business SH)  
- Escenario: SINERGIA `(Global/SH/Business/IB –, Global/SH/Business/YW – | Global/SH/Business –)`  
- NMA: Global/SH/Business  
- Afecta a: Global/SH/Business/IB, Global/SH/Business/YW  
- Qué falló: Servicio en vuelos operados o comercializados con BA, sin incidencias formales registradas.  
- Dónde: Muestra mínima en DUS–MAD (n=4), pero CodeShare BA arrastró la media.  
- Quién: Pasajeros de Business SH en CodeShare BA (Global/SH/Business)  
- Evidencia:  
   • NPS 34.0909 vs baseline 43.5255 (Global/SH/Business)  
   • CodeShare BA NPS –66.7 (Global/SH/Business)  
   • Load_Factor 70.63 (–2.12 pts vs media) (Global/SH/Business)

CAUSA 3a: Insatisfacción puntual en SH Economy/IB  
- Escenario: CANCELACIÓN `(- (IB), + (YW) | N)`  
- NMA: – (no hay único NMA para cancelación)  
- Afecta a: Global/SH/Economy/IB  
- Qué falló: Concentración de feedback negativo en ruta MAD–VGO, potenciado por vuelos CodeShare BA.  
- Dónde: Ruta MAD–VGO NPS 0.0 (n=4) (Global/SH/Economy/IB)  
- Quién: Pasajeros IB y CodeShare BA, con amplia dispersión de regiones (Global/SH/Economy/IB)  
- Evidencia:  
   • NPS 27.7592 vs baseline 29.6476 (Global/SH/Economy/IB)  
   • CodeShare BA NPS –100.0 (Global/SH/Economy/IB)

CAUSA 3b: Exceso de satisfacción en SH Economy/YW  
- Escenario: CANCELACIÓN `(- (IB), + (YW) | N)`  
- NMA: – (no hay único NMA para cancelación)  
- Afecta a: Global/SH/Economy/YW  
- Qué falló (bien): Elevada satisfacción de ocio en ruta MAD–MUC, sin problemas operativos.  
- Dónde: Ruta MAD–MUC NPS 28.6 (n=7) (Global/SH/Economy/YW)  
- Quién: Pasajeros Leisure (Global/SH/Economy/YW)  
- Evidencia:  
   • NPS 48.6842 vs baseline 37.1058 (Global/SH/Economy/YW)  
   • OTP15_adjusted 91.6 (+3.79 pts vs media) (Global/SH/Economy/YW)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

Global presentó el 30/11/2025 un descenso de 26.0147 → 17.9234 en su NPS (–8.0913 pts). Esta caída está arrastrada por Long Haul, cuyo NPS pasó de 12.5078 → –8.5987 (–21.1156 pts). En LH, Economy LH retrocedió de 8.9370 → –13.6187 (–22.5557 pts) principalmente en la ruta MAD–MEX (NPS –20.0, Global/LH/Economy), Business LH bajó de 24.0313 → 21.2121 (–2.8192 pts) y Premium LH cayó de 34.8085 → 4.1667 (–30.6418 pts), especialmente para pasajeros de América Centro en A350 next. Por el contrario, Short Haul mantuvo un ligero alza de 33.1425 → 34.7475 (+1.6050 pts), equilibrando internamente la caída en Business SH (43.5255 → 34.0909, –9.4346 pts) con la cancelación de efectos en Economy SH, donde IB pasó de 29.6476 → 27.7592 (–1.8884 pts) y YW de 37.1058 → 48.6842 (+11.5784 pts).

Las rutas más afectadas según el árbol fueron LIM–MAD con un NPS de –11.8 (Global/LH) y MAD–MEX con –20.0 (Global/LH/Economy), junto a MAD–VGO en SH Economy IB (NPS 0.0, Global/SH/Economy/IB) y MAD–MUC en SH Economy YW (NPS 28.6, Global/SH/Economy/YW). Los perfiles más reactivos incluyen pasajeros Leisure en SH Economy YW (NPS 53.3), usuarios de CodeShare BA en Business SH (NPS –66.7), viajeros de América Norte y Europa en Economy LH (NPS –59.3 y –52.6) y pasajeros de América Centro en Premium LH (NPS –50.0).

ECONOMY SH: Estabilidad engañosa por cancelación interna  
La cabina Economy de SH mantuvo desempeño estable durante la semana del 2025-11-30, registrando un NPS de 34.8115 (30/11/2025) con una subida de 2.7253 pts vs L7d. No se detectaron cambios significativos en el agregado, pero internamente se compensaron un descenso de 29.6476 → 27.7592 (–1.8884 pts) en Global/SH/Economy/IB, centrado en la ruta MAD–VGO (NPS 0.0, Global/SH/Economy/IB), y un ascenso de 37.1058 → 48.6842 (+11.5784 pts) en Global/SH/Economy/YW, impulsado por la alta satisfacción de ocio en MAD–MUC (NPS 28.6, Global/SH/Economy/YW).

BUSINESS SH: Deficiencias en CodeShare BA arrastran la cabina  
El segmento Business de SH registró un NPS de 34.0909 (30/11/2025) con una caída de 9.4346 pts vs L7d. Esta merma se explica por la pobre experiencia en vuelos CodeShare BA, donde BA obtuvo un NPS de –66.7 (Global/SH/Business), generando un spread de 106.1 pts con IB. Afectó especialmente a usuarios de flotas A333 (NPS –100.0, Global/SH/Business) y pasajeros de Norteamérica y Europa, pese a la puntualidad destacada (OTP15_adjusted 92.23, +2.80 pts vs L7d, Global/SH/Business).

ECONOMY LH: Desplome en MAD–MEX y flotas A321XLR/A332  
La cabina Economy de LH experimentó un deterioro significativo, con NPS de –13.6187 (30/11/2025) y una bajada de 22.5557 pts vs L7d. La caída se concentró en la ruta MAD–MEX (NPS –20.0, n=15, Global/LH/Economy) y se agravó en pasajeros Business/Work (NPS –20.6, Global/LH/Economy) y Leisure (NPS –12.6, Global/LH/Economy) volando en flotas A321XLR (NPS –63.6, Global/LH/Economy) y A332 (NPS –24.7, Global/LH/Economy), especialmente desde América Norte (NPS –59.3, Global/LH/Economy) y Europa (NPS –52.6, Global/LH/Economy).

BUSINESS LH: Heterogeneidad de flota y CodeShare BA  
El segmento Business de LH cerró en 21.2121 (30/11/2025) con un descenso de 2.8192 pts vs L7d. Esta leve baja obedece a la disparidad en CodeShare BA (NPS –66.7, Global/LH/Business) y en la flota A333 (NPS –63.6, Global/LH/Business) frente a la buena valoración en A350 next (NPS 50.0, Global/LH/Business). Pasajeros de Norteamérica y Europa registraron puntuaciones inferiores, pese a la puntualidad óptima.

PREMIUM LH: Hundimiento en MAD–MEX entre pasajeros de América Centro  
El segmento Premium de LH sufrió un hundimiento crítico, con NPS de 4.1667 (30/11/2025) y una caída de 30.6418 pts vs L7d. El descenso se concentró en la ruta MAD–MEX (NPS 20.0, n=5, Global/LH/Premium), donde cinco pasajeros de América Centro viajando en A350 next dieron puntuaciones de –50.0 (Global/LH/Premium), sin que se reportaran incidentes operativos formales.

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

En SH/Economy, el escenario es No Aplica (N, N | N).  
- Narrativa: Tanto IB (NPS +3.9 pts vs baseline) como YW (NPS +2.6 pts) se mantienen dentro de la variación normal; consecuentemente, el padre Economy SH (+3.2 pts) refleja un desempeño estable sin anomalías internas.  
- Evidencia Clave: IB NPS 33.54 vs 29.65 (baseline); YW NPS 39.68 vs 37.11; sin incidentes reportados ni quejas operativas.

En SH/Business, el escenario es DOMINANCIA (+, – | +).  
- Narrativa: La anomalía positiva del padre (+14.0 pts) se impone desde Global/SH/Business/IB, cuyo fuerte ascenso arrastra al agregado, a pesar de la caída en YW que matiza el resultado.  
- Evidencia Clave: IB registró +26.1 pts por OTP15_adjusted 92.89 (+1.51) y baja densidad de pasajeros, con feedback muy positivo en la ruta LHR-MAD (NPS 100.0) y en A320neo; YW presentó –10.3 pts con Load_Factor 55.15 (–3.41) sin incidentes.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En SH, la dinámica es DILUCIÓN (Economy N, Business + | SH N).  
- Narrativa: El rendimiento de Short Haul quedó estabilizado pese al fuerte impulso de la cabina Business; la mejora no logró permear al agregado por la estabilidad de Economy.  
- Evidencia: Global/SH/Business presentó +14.0 pts impulsados por OTP15_adjusted 91.84 (+2.42) y baja densidad en A320neo con NPS 100.0 en LHR–MAD, mientras Economy SH se mantuvo dentro de rango normal (NPS 35.27 vs baseline 32.09).

En LH, la dinámica es CANCELACIÓN (Economy N, Business –, Premium + | LH N).  
- Narrativa: El NPS de Long Haul aparenta normalidad, pero esconde causas opuestas: la caída en Business LH neutralizó la subida en Premium LH y la estabilidad de Economy.  
- Evidencia: Global/LH/Business cayó –9.7 pts por OTP15_adjusted 77.8 (–1.75) y concentración de detractores en A350/Europa-Norteamérica, mientras Global/LH/Premium subió +15.2 pts gracias a feedback muy positivo y heterogeneidad en flotas (A350 vs A350 next). Economía LH permaneció estable.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA (LH N, SH N | Global +).  
- Narrativa: El repunte de +4.07 pts en NPS Global no proviene de un único radio, sino de factores operativos y de servicio que, si bien no fueron lo suficientemente fuertes para generar anomalías aisladas en Long Haul ni Short Haul, sí se acumularon y elevaron el indicador agregado.  
- Evidencia:  
  • OTP15_adjusted Global: 90.02 (+1.90 pts vs media 7 días)  
  • Mishandling index Global: 13.84 (–1.69 pts vs media)  
  • Verbatims Global: 1.093 comentarios con tono altamente positivo (puntualidad, amabilidad de tripulación).

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Mejora Operativa y Feedback Global  
- Escenario: TRANSFERENCIA (LH N, SH N | Global +)  
- NMA: Global  
- Afecta a: Global  
- Qué falló: No aplica – el alza proviene de la mejora de puntualidad y manejo de equipaje  
- Dónde: no hay rutas con incidencia operativa (MAD–MUC NPS –20.0, Global/Ruta MAD–MUC)  
- Quién: todos los perfiles contribuyeron con comentarios positivos de tripulación (Global/Verbatims)  
- Evidencia:  
  • NPS 30.08 vs 26.01 (Global)  
  • OTP15_adjusted 90.02 (+1.90 pts vs media 7d; Global)  
  • Mishandling index 13.84 (–1.69 pts vs media 7d; Global)  

CAUSA 2: Excelencia en Short Haul Business/IB  
- Escenario: DOMINANCIA (+, – | +) en SH/Business  
- NMA: Global/SH/Business/IB  
- Afecta a: Global/SH/Business/IB  
- Qué falló: No es fallo – impulso por puntualidad y espacio extra  
- Dónde: LHR–MAD NPS 100.0 (Global/SH/Business/IB – Ruta LHR–MAD)  
- Quién: pasajeros en A320neo NPS 81.8 (Global/SH/Business/IB – Fleet A320neo)  
- Evidencia:  
  • NPS 74.07 vs 48.00 (SH/Business/IB)  
  • OTP15_adjusted 92.89 (+1.51 pts vs media 7d; SH/Business/IB)  
  • Load_Factor 78.98 (–1.93 pts vs media 7d; SH/Business/IB)  

CAUSA 3: Baja Percepción de Valor en Short Haul Business/YW  
- Escenario: DOMINANCIA (+, – | +) en SH/Business  
- NMA: Global/SH/Business/YW  
- Afecta a: Global/SH/Business/YW  
- Qué falló: Load_Factor reducido afectó el ambiente y la percepción de valor  
- Dónde: no hubo rutas con muestra suficiente  
- Quién: todos los pasajeros YW sin variaciones significativas por perfil  
- Evidencia:  
  • NPS 23.08 vs 33.36 (SH/Business/YW)  
  • Load_Factor 55.15 (–3.41 pts vs media 7d; SH/Business/YW)  
  • OTP15_adjusted 90.88 (+3.10 pts vs media 7d; SH/Business/YW)  

CAUSA 4: Detractores en Long Haul Business  
- Escenario: CANCELACIÓN (Economy N, Business –, Premium + | LH N)  
- NMA: Global/LH/Business  
- Afecta a: Global/LH/Business  
- Qué falló: Pequeño grupo de detractores en A350  
- Dónde: GRU–MAD NPS 33.3 (Global/LH/Business – Ruta GRU–MAD)  
- Quién: clientes de Europa y Norteamérica NPS –33.3 (Global/LH/Business – Residence Region Europa/Norteamérica)  
- Evidencia:  
  • NPS 14.29 vs 24.03 (LH/Business)  
  • OTP15_adjusted 77.8 (–1.75 pts vs media 7d; LH/Business)  
  • Load_Factor 93.43 (–0.48 pts vs media 7d; LH/Business)  

CAUSA 5: Heterogeneidad en Long Haul Premium  
- Escenario: CANCELACIÓN (Economy N, Business –, Premium + | LH N)  
- NMA: Global/LH/Premium  
- Afecta a: Global/LH/Premium  
- Qué falló: Variabilidad de experiencia según tipo de avión (A350 vs A350 next)  
- Dónde: no hubo rutas con muestra suficiente  
- Quién: flota con spread de NPS (0–100 pts; Global/LH/Premium – Fleet A350 vs A350 next)  
- Evidencia:  
  • NPS 50.0 vs 34.81 (LH/Premium)  
  • Load_Factor 87.69 (–2.97 pts vs media 7d; LH/Premium)  
  • OTP15_adjusted 77.8 (–1.75 pts vs media 7d; LH/Premium)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El NPS Global experimentó un repunte de 4.07 puntos, pasando de 26.01 a 30.08, gracias a una combinación de mayor puntualidad (OTP15_adjusted 90.02 vs L7d 88.12, +1.90 pts) y menor incidencia de equipaje extraviado (mishandling index 13.84 vs L7d 15.53, –1.69 pts), junto a 1.093 verbatims con tono altamente positivo. A nivel de cabina y radio, destacan cinco movimientos extremos:  
– Long Haul Business cayó 9.75 puntos (24.03→14.29) por un OTP15_adjusted reducido (77.8 vs L7d 79.55, –1.75 pts) y una concentración de detractores en flota A350 (NPS –28.6) y clientes de Europa/Norteamérica (NPS –33.3).  
– Long Haul Premium subió 15.19 puntos (34.81→50.0) impulsado por la heterogeneidad en flota (spread 0–100 entre A350 y A350 next) y comentarios positivos de tripulación, pese a ligeras caídas operativas (Load_Factor 87.69 vs L7d 90.66, –2.97 pts; OTP15_adjusted 77.8 vs L7d 79.55, –1.75 pts).  
– Short Haul Business ascendió 13.97 puntos (43.53→57.50) jalonado por Global/SH/Business/IB, que mejoró 26.07 puntos (48.00→74.07) gracias a puntualidad superior (OTP15_adjusted 92.89 vs L7d 91.38, +1.51 pts), menor densidad (Load_Factor 78.98 vs L7d 80.91, –1.93 pts) y NPS 100.0 en LHR–MAD (n=4) con A320neo (81.8 de NPS). Este avance contrarrestó la caída de 10.28 puntos en Global/SH/Business/YW (33.36→23.08), donde la baja ocupación (Load_Factor 55.15 vs L7d 58.56, –3.41 pts) afectó la percepción de valor.  

Las rutas más afectadas incluyeron LHR–MAD con NPS 100.0 en Global/SH/Business/IB, y GRU–MAD con NPS 33.3 en Global/LH/Business, ambas con muestras pequeñas pero críticas. Los grupos de clientes más sensibles fueron pasajeros en A320neo en SH/Business/IB (NPS 81.8) y residentes en Europa/Norteamérica en LH/Business (NPS –33.3).  

ECONOMY SH: Desempeño estable  
La cabina Economy de SH mantuvo desempeño estable durante el día 2025-11-29, registrando un NPS de 35.27 (vs L7d 32.09) con una subida de 3.18 puntos respecto a la media de los últimos 7 días. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción sin quejas operativas ni NCS.  

BUSINESS SH: Impulso marcado por IB, matizado por YW  
En Business SH el NPS global del segmento subió 13.97 puntos, pasando de 43.53 a 57.50. Global/SH/Business/IB registró un salto de 26.07 puntos (48.00→74.07) gracias a OTP15_adjusted 92.89 vs L7d 91.38 (+1.51 pts), Load_Factor 78.98 vs L7d 80.91 (–1.93 pts) y NPS 100.0 en LHR–MAD con A320neo (81.8 de NPS). Este avance se vio contrarrestado por Global/SH/Business/YW, que cayó 10.28 puntos (33.36→23.08) por una ocupación reducida (Load_Factor 55.15 vs L7d 58.56, –3.41 pts), afectando la percepción de valor.  

ECONOMY LH: Desempeño estable  
La cabina Economy de LH mantuvo desempeño estable el 2025-11-29, con un NPS de 15.83 (vs L7d 12.51) y una subida de 3.33 puntos frente a la media de los últimos 7 días. No se registraron anomalías operativas ni variaciones significativas en verbatims.  

BUSINESS LH: Caída focalizada en A350  
La cabina Business de LH sufrió un deterioro de 9.75 puntos, bajando de un NPS de 24.03 a 14.29. El OTP15_adjusted cayó a 77.8 vs L7d 79.55 (–1.75 pts) y la alta densidad (Load_Factor 93.43 vs L7d 93.91, –0.48 pts) no generó quejas explícitas, pero un grupo de detractores en flota A350 (NPS –28.6) y clientes de Europa/Norteamérica (NPS –33.3) concentró el impacto.  

PREMIUM LH: Subida por heterogeneidad de flota  
El segmento Premium de LH experimentó un alza de 15.19 puntos, pasando de un NPS de 34.81 a 50.0. La variabilidad en experiencia según flota (spread 0–100 entre A350 y A350 next) y el feedback muy positivo de tripulación impulsaron esta mejora, pese a una ligera caída de Load_Factor (87.69 vs L7d 90.66, –2.97 pts) y de OTP15_adjusted (77.8 vs L7d 79.55, –1.75 pts).

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