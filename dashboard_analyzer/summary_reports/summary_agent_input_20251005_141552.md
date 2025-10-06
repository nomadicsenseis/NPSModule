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
- Existen ambos subnodos: SH/Economy/IB y SH/Economy/YW.  
- Patrón y drivers opuestos:  
  • IB muestra una anomalía negativa (–4.0 pts) impulsada por empeoramiento de puntualidad e incremento de “other_incidencias”.  
  • YW presenta anomalía positiva (+7.1 pts) gracias a mejora de puntualidad y caída de cancelaciones.  
Diagnóstico: las causas son específicas a cada compañía.

B. Business Short Haul  
- Existen ambos subnodos: SH/Business/IB y SH/Business/YW.  
- Ambos registran anomalía positiva (IB +20.3 pts; YW +19.8 pts) con mismas palancas: mejora de puntualidad y reducción de cancelaciones.  
Diagnóstico: la causa es común a la cabina Business SH.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul: Economy SH y Business SH divergen claramente (Economy –0.2 pts vs Business +20.6 pts) e incluso dentro de Economy los subsegmentos IB y YW muestran señales opuestas, por lo que las causas operativas son específicas de cada cabina.  
Long Haul: Economy (–3.3 pts), Business (–5.1 pts) y Premium (–24.6 pts) comparten el mismo driver principal (deterioro de puntualidad y capacidad), con una progresión de reactividad creciente desde Economy hasta Premium, lo que indica una causa común al radio LH con distinta sensibilidad por cabina.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Diagnóstico global: Long Haul es el único radio con un impacto negativo consistente (–5.0 pts), impulsado por la caída de puntualidad y presión de ocupación, mientras que Short Haul cierra como normal (+1.9 pts) gracias a la compensación entre Business (+20.6 pts) y Economy (–0.2 pts). A nivel Global, el fuerte descenso de LH se atenúa parcialmente por el alza de SH, resultando en una leve anomalía negativa (–0.6 pts).

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

1. Causa: DETERIORO DE LA PUNTUALIDAD  
A. Naturaleza  
   • Driver operativo crítico: empeoramiento de OTP15 y mayor frecuencia de retrasos. Impacto directo en satisfacción y percepción de fiabilidad.  

B. Evidencia consolidada y alcance  
   • Segmento “más grande”: Global/Long Haul (NPS 17.42 vs 22.45; variación –5.03 pts; 2 879 verbatims).  
   • Output causal destacado:  
     – SHAP Punctuality: –2.901  
     – Sat_diff puntualidad: –3.15 pts  
     – OTP15: 78.32 vs 83.23 (–4.91 pts)  
     – NCS Tool: +6 retrasos, +26 “other_incidencias”, +5 limitaciones aeronave, +4 desvíos  
     – Verbatims: >40 % mencionan “retrasos frecuentes” y “falta de comunicación horaria”  
   • Rutas más afectadas: GYE-MAD, LAX-MAD, MAD-SAL, GUA-MAD, CCS-MAD  
   • Alcance: afecta a todos los subsegmentos Long Haul (Economy, Business y Premium).  

2. Causa: AUMENTO DE LA OCUPACIÓN (LOAD FACTOR)  
A. Naturaleza  
   • Driver operativo secundario pero relevante: mayor presión en embarque y cabina, generando sensación de caos.  

B. Evidencia consolidada y alcance  
   • Segmento “más grande”: Global/Long Haul (mismo perfil que puntualidad).  
   • Output causal destacado:  
     – SHAP Load factor: –0.206  
     – Sat_diff ocupación: +2.01 pts  
     – Load factor: 92.73 % vs 91.81 % (+0.92 pts)  
     – Verbatims: frecuentes referencias a “vuelos completos” y “embarque caótico”  
   • Rutas clave: GYE-MAD, LAX-MAD, MAD-ORD, MAD-MIA, MAD-SCL  
   • Alcance: presente en todos los subsegmentos Long Haul.  

3. Causa: INCREMENTO DE INCIDENTES OPERATIVOS  
A. Naturaleza  
   • Acumulación de “other_incidencias”, limitaciones de aeronave, desvíos y pérdidas de conexión que tensionan la experiencia.  

B. Evidencia consolidada y alcance  
   • Segmento “más grande”: Global (NPS 24.42 vs 25.03; variación –0.61 pts; ncs_tool).  
   • Output causal destacado:  
     – Flight cancellations: –73 (vs baseline)  
     – Other_incidencias: +93  
     – Aircraft limitations: +18  
     – Desvíos: +7  
     – Pérdidas de conexión: 108 vs 0  
   • Rutas con mayor concentración de incidencias: JMK-MAD, FLR-MAD, JTR-MAD, CDT-MAD, AGP-MAD  
   • Alcance: afecta a todos los subsegmentos bajo el nodo Global (LH y SH).  

4. Causa: FALLOS DE SISTEMAS E INFRAESTRUCTURALES  
A. Naturaleza  
   • Problemas con kioscos de autoservicio, aplicación y equipamiento de puerta que impactan principalmente en Economy SH/IB.  

B. Evidencia consolidada y alcance  
   • Segmento “más grande”: Global/SH/Economy/IB (NPS 22.45 vs 26.44; variación –4.00 pts; 1 045 menciones de cancelaciones y cambios de avión).  
   • Output causal destacado:  
     – SHAP Punctuality: –1.978; Sat_diff: –3.94  
     – SHAP Load factor: –0.319; Sat_diff: +3.44  
     – OTP15: –0.4 pts; Load factor: +0.3 pts  
     – NCS Tool: +15 other_incidencias; +4 limitaciones aeronave  
     – Verbatims: 685 menciones de equipaje retrasado, 512 de mala comunicación de cambios  
   • Rutas principales: JMK-MAD, FLR-MAD, JTR-MAD, ATH-MAD, MAD-NCE  
   • Alcance: afecta a todos los subsegmentos SH/Economy de la compañía IB.  

5. Causa: MEJORA DE PUNTUALIDAD Y REDUCCIÓN DE CANCELACIONES  
A. Naturaleza  
   • Factores operativos positivos: aumento de OTP15, caída de retrasos y cancelaciones, elevando masivamente la percepción de fiabilidad.  

B. Evidencia consolidada y alcance  
   • Segmento “más grande”: Global/SH/Business (NPS 41.10 vs 20.50; variación +20.61 pts; 463 verbatims).  
   • Output causal destacado:  
     – Driver Punctuality confirmado por reducción de retrasos y cancelaciones (ncs_tool).  
     – Load factor identificado pero no validado operativamente como limitante.  
     – Verbatims: 463 comentarios que alaban vuelos a tiempo y gestión de puertas.  
   • Rutas más beneficiadas: BRU-MAD, MAD-SDR, BUD-MAD, MAD-TLS, GVA-MAD  
   • Alcance: afecta a todos los subsegmentos bajo SH/Business (IB y YW).

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

Durante la semana del 2025-09-25 al 2025-10-01 se detectaron anomalías contrapuestas por radio y cabina. En Short Haul, la cabina Economy cerró con un ligero descenso de 0.24 puntos (de 27.04 a 26.80) mientras que Business repuntó 20.61 puntos (de 20.50 a 41.10). A nivel de subsegmentos, Economy SH IB cayó 3.99 pts (de 26.44 a 22.45) por empeoramiento de puntualidad (OTP15 –0.4 pts, SHAP Punctuality –1.98) y alza de incidencias operativas, mientras que Economy SH YW mejoró 7.07 pts (de 28.11 a 35.18) gracias a un OTP15 en +3.36 pts y reducción de retrasos y cancelaciones. Business SH IB y Business SH YW compartieron un alza mayor a 19 pts cada uno, impulsada por mejoras de puntualidad y caída de cancelaciones. En Long Haul, las tres cabinas empeoraron: Economy LH cayó 3.30 pts (de 19.66 a 16.36) por retrasos frecuentes (OTP15 –4.91 pts, SHAP Punctuality –2.38), alta ocupación (+0.92 pts) y +26 “otras incidencias”; Business LH perdió 5.12 pts (de 36.19 a 31.07) por los mismos drivers operativos reforzados con +36 incidentes adicionales; y Premium LH se desplomó 24.65 pts (de 31.68 a 7.03) ante un fuerte deterioro de puntualidad (SHAP –6.64, OTP15 –4.91 pts) y tensiones operativas.  

Las rutas más afectadas por caídas de NPS en Long Haul incluyen MAD–ORD (–51.5 pts en Economy LH), GYE–MAD (–66.7 pts en Business LH) y LAX–MAD (–33.3 pts en Premium LH). En Short Haul, BRU–MAD lidera las mejoras (+71.4 pts en Business SH) y JMK–MAD las caídas (–66.7 pts en Economy SH IB). Los pasajeros en vuelos code-share y según región de residencia mostraron la mayor reactividad, con spreads de hasta 133 pts en Business LH y 121 pts en Business SH.  

ECONOMY SH  
La cabina Economy de SH IB experimentó un descenso de 3.99 pts, pasando de un NPS de 26.44 (vs L7d) a 22.45, debido a un empeoramiento de la puntualidad (OTP15 –0.4 pts; SHAP Punctuality –1.98) y al incremento de “other_incidencias” (+15) y limitaciones de aeronave (+4). Por su parte, Economy SH YW mejoró de 28.11 a 35.18 (+7.07 pts) impulsada por un aumento de OTP15 en +3.36 pts y reducción de retrasos y cancelaciones (–30 retrasos, –18 cancelaciones). El efecto neto dejó a Economy SH en un NPS de 26.80 (–0.24 pts vs L7d). Las caídas más notables se vieron en JMK–MAD (–66.7 pts) y ATH–MAD (–20.6 pts), mientras que los pasajeros code-share fueron los más sensibles (spread de 88.9 pts).  

BUSINESS SH  
El segmento Business de SH cerró en 41.10 (+20.61 pts vs L7d), con subidas de 20.27 pts en IB (de 27.90 a 48.17) y 19.82 pts en YW (de 7.03 a 26.85). Ambos registraron mejoras sustanciales de puntualidad y reducción de cancelaciones, confirmadas por verbatims que destacan “vuelos a tiempo” y “gestión fluida de puertas”. Las rutas BRU–MAD (+71.4 pts) y MAD–SDR (+100.0 pts) concentraron el alza, y los pasajeros de determinadas flotas mostraron mayor reactividad (spread de 115.0 pts).  

ECONOMY LH  
La cabina Economy de LH cayó de 19.66 a 16.36 (–3.30 pts vs L7d) tras un OTP15 en 78.32 (–4.91 pts), SHAP Punctuality de –2.38 y un incremento de load factor (91.81→92.73 %) junto a +26 “other_incidencias”. Esta caída se reflejó en rutas como GYE–MAD (0.0 pts) y MAD–ORD (–51.5 pts), y los pasajeros code-share fueron los más afectados (spread de 50.7 pts).  

BUSINESS LH  
La cabina Business de LH descendió de 36.19 a 31.07 (–5.12 pts vs L7d), motivado por la misma combinación de puntualidad deteriorada (OTP15 –4.91 pts; SHAP –2.95) y alta ocupación (+0.92 pts; SHAP –0.17) junto a +36 incidentes operativos. GYE–MAD (–66.7 pts) y LAX–MAD (0.0 pts) lideraron las caídas, con clientes de residencia regional como los más sensibles (spread de 178.0 pts).  

PREMIUM LH  
El segmento Premium de LH sufrió un desplome de 24.65 pts, de 31.68 a 7.03 (vs L7d), debido a una fuerte degradación de la puntualidad (SHAP –6.64; OTP15 –4.91 pts) y acumulación de incidentes (otras_incidencias +26, retrasos +6). Las peores caídas se dieron en MAD–ORD (–42.9 pts) y LAX–MAD (–33.3 pts), y los pasajeros según región de residencia mostraron la mayor variabilidad (spread de 103.3 pts).

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*



**ANÁLISIS DIARIO SINGLE:**
📅 2025-10-01 to 2025-10-01:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

Economy Short Haul  
• Existen ambos nodos SH/Economy/IB y SH/Economy/YW, y muestran patrones claramente divergentes en sus drivers y evidencias operativas:  
  – IB (+10.0 pts): impulsado por flotas A332, alta satisfacción europea y ausencia de incidencias en rutas clave.  
  – YW (–17.3 pts): penalizado por un volumen significativo de cancelaciones, retrasos y pobre desempeño de la flota ATR.  
Diagnóstico: la causa del movimiento en Economy SH es específica de cada compañía, no un fenómeno común a la cabina.  

Business Short Haul  
• Solo existe el nodo SH/Business/IB (+26.3 pts); no hay datos para SH/Business/YW (muestra insuficiente).  
Diagnóstico: la anomalía de Business SH responde a dinamicas particulares de IB y no puede atribuirse de forma válida a toda la cabina.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul  
“El patrón de anomalías en SH es específico de cada cabina y no común al radio: Business muestra un fuerte alza generalizado (IB +31.8 pts), mientras que Economy presenta respuestas opuestas por compañía (IB +10.0 pts vs. YW –17.3 pts), lo que impide una causa única de radio.”  

Long Haul  
“El patrón en LH es también específico de cabina: Economy actúa como amortiguador (normal), Business se ve penalizada (–6.8 pts) y Premium se beneficia (+34.0 pts), sin convergencia en la reacción de las tres clases.”

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

“Las anomalías responden a causas mixtas y compensatorias entre radios: Short Haul y Long Haul exhiben patrones cabin-específicos divergentes que se equilibran mutuamente, resultando en un Global con un alza moderada de +3.6 pts sin un driver homogéneo a nivel compañía o radio.”

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

Causa 1  
A. Naturaleza de la causa  
   Un alza transversal de la satisfacción impulsada por viajeros de ocio (“leisure”) y el uso de flota moderna, que elevó el NPS global a pesar de incidencias operativas.  

B. Evidencia consolidada y alcance  
   Segmento: Global (afecta a todos los sub‐nodos Global, incluidos LH y SH).  
   – NPS 28.61 vs baseline 24.99 (+3.62 pts)  
   – Incidencias: 449 reportes (108 retrasos, 70 cancelaciones; 50 vuelos afectados; 11 technical_issues, 11 cambios de avión, 10 “passenger”).  
   – Rutas: MAD–PRG NPS 16.7 (n=6), sin incidencias.  
   – Perfiles: Leisure 31.0 vs Business 21.8; A319 55.6, A320neo 50.9 vs ATR –10.0, 32S 0.0; Región SurAm 47.3, Europa 44.0, Asia –33.3; code‐share spread 134.7.  
   – Verbatims: no disponibles.  
   “El incremento de +3.62 pts se explica por la alta valoración de clientes de ocio en rutas de flota moderna, compensando retrasos y cancelaciones.”  

Causa 2  
A. Naturaleza de la causa  
   Problemas de fiabilidad operativa en Long Haul Business (retrasos, overbooking y pérdidas de conexión) concentrados en la flota A350 estándar.  

B. Evidencia consolidada y alcance  
   Segmento: Global/LH/Business (afecta a todos los sub‐nodos de LH Business).  
   – NPS 19.05 vs baseline 25.82 (–6.77 pts)  
   – Incidencias: 39 (11 retrasos, 12 “otras”, 72 cambios de equipo, 49 pérdidas de conexión; overbooking en IB271/272; avería IB243 con retorno MAD; 3 vuelos afectados).  
   – Flota: A350 next 100.0 (n=5), A333 33.3 (n=3), A350 estándar –10.0 (n=10).  
   – Región: SurAm 66.7 vs NorteAm 33.3; Business vs Leisure spread 7.1; CodeShare IB 35.3.  
   – Verbatims: no disponibles.  
   “La caída de –6.8 pts responde a contratiempos operativos en LH Business, especialmente en A350 estándar, generando reacomodos y conexiones perdidas.”  

Causa 3  
A. Naturaleza de la causa  
   Salto anómalo en Long Haul Premium, probablemente derivado de sesgo de muestreo o mejoras puntuales en touchpoints no medidos (atención a bordo, servicios digitales).  

B. Evidencia consolidada y alcance  
   Segmento: Global/LH/Premium (afecta a todos los sub‐nodos de LH Premium).  
   – NPS 44.44 vs baseline 10.48 (+33.96 pts)  
   – Incidencias: 39 (11 retrasos, 12 “otras”, 5 cambios de avión, 4 technical_issues); rutas: MAD–GIG, IB271/272, IB243/MADSJO.  
   – Métricas operativas: no disponibles.  
   – Verbatims: no disponibles; no rutas con ≥2 encuestas.  
   “Sin datos de puntualidad ni feedback, el fuerte alza sugiere un sesgo de muestra o mejoras de servicio no captadas por las herramientas operativas.”  

Causa 4  
A. Naturaleza de la causa  
   Elevado volumen de cancelaciones, retrasos y pobre desempeño de la flota ATR que penaliza la satisfacción en Economy Short Haul de YW.  

B. Evidencia consolidada y alcance  
   Segmento: Global/SH/Economy/YW (afecta a todos los sub‐nodos SH Economy/YW).  
   – NPS 16.67 vs baseline 33.93 (–17.27 pts)  
   – Incidencias: 46 (18 cancelaciones, 12 retrasos, 2 desvíos, 2 equipaje, 12 otras; 5 cambios de avión, 4 technical_issues, 1 bird_strike; MAD–GIG con 3 vuelos afectados).  
   – Flota: CRJ 22.0 (n=50), ATR –10.0 (n=10); Leisure 22.2 vs Business 8.3; Región rango –20 a 28.6; CodeShare IB 18.2 (n=55).  
   – Verbatims: no disponibles.  
   “El drop de –17.3 pts es atribuible a la alta incidencia de cancelaciones/retrasos y al mal desempeño de la flota ATR en Economy SH/YW.”  

Causa 5  
A. Naturaleza de la causa  
   Concentración de respuestas en rutas sin incidencias y muestra reducida que infló el NPS de Business Short Haul en IB.  

B. Evidencia consolidada y alcance  
   Segmento: Global/SH/Business/IB (afecta a ese sub‐nodo).  
   – NPS 77.78 vs baseline 51.44 (+26.34 pts)  
   – Incidencias: 46 (18 cancelaciones, 12 retrasos, 12 otras, 5 cambios de avión, 4 technical_issues, 1 bird_strike; overbooking IB271/272; avería IB243; 3 vuelos afectados; 72 cambios de equipo; 49 pérdidas de conexión; 4 reprogramaciones).  
   – Rutas: LHR–MAD NPS 100.0 (n=3), sin incidencias.  
   – Perfil: NorteAm 25.0, Europa 100.0, CentroAm 100.0; A320 50.0, A320neo 100.0.  
   – Verbatims: no disponibles.  
   “El fuerte alza responde a un sesgo de ruta/μestra: la mayoría de encuestas proviene de LHR–MAD sin incidentes, más que a cambios operativos reales.”

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

Durante el día 2025-10-01 se observaron subidas y bajadas de NPS en seis nodos clave: Global ganó 3.62 puntos, pasando de 24.99 a 28.61; Long Haul Business cayó 6.77 puntos (de 25.82 a 19.05) por 39 incidencias –entre ellas overbooking en IB271/272, avería en IB243 y 49 pérdidas de conexión– y el bajo desempeño de la flota A350 estándar (NPS –10.0 vs 100.0 en A350 next); Long Haul Premium se disparó 33.96 puntos (de 10.48 a 44.44), un salto no respaldado por métricas operativas ni verbatims, apuntando a un sesgo de muestra o mejoras puntuales en servicios no medidos; Short Haul Economy/IB subió 9.99 puntos (de 23.89 a 33.88) gracias a la alta satisfacción de pasajeros europeos en flota A332 y rutas sin incidencias; Short Haul Economy/YW bajó 17.27 puntos (de 33.93 a 16.67) por un volumen elevado de 46 cancelaciones y retrasos y el pobre desempeño de ATR en rutas como MAD–GIG; y Short Haul Business aumentó 31.81 puntos (de 44.38 a 76.19), impulsado por el sesgo de muestreo en la ruta LHR–MAD, que alcanzó NPS 100.0 (n=3) sin incidencias, y la valoración de flota A320neo.

Las rutas más afectadas incluyeron MAD–GIG e IB271/272/IB243, donde se concentraron cancelaciones, cambios de avión y retrasos, mientras que corredores como LHR–MAD y DUS–MAD, libres de incidencias, registraron NPS perfectos que inflaron las subidas. Los grupos de clientes más reactivos fueron los viajeros de ocio y residentes en América del Sur y Europa –apoyados en flota moderna–, en contraste con pasajeros de negocio, usuarios de ATR y viajeros asiáticos o de América Central, que mostraron mayor insatisfacción ante las disrupciones.

ECONOMY SH: Rendimiento divergente entre IB y YW  
La cabina Economy de SH registró un NPS de 33.88 en Global/SH/Economy/IB (subida de 9.99 pts vs L7d) y de 16.67 en Global/SH/Economy/YW (descenso de 17.27 pts vs L7d). IB se benefició de la alta satisfacción de pasajeros europeos, NPS 100.0 en DUS–MAD y flota A332, mientras YW sufrió 46 incidentes –18 cancelaciones y 12 retrasos– y el bajo NPS de ATR, especialmente en la ruta MAD–GIG.

BUSINESS SH: Fuerte alza por sesgo de muestra  
El segmento Business de SH alcanzó un NPS de 76.19 en Global/SH/Business (subida de 31.81 pts vs L7d), con IB en 77.78 pts (subida de 26.34 pts vs L7d) y YW estable. Este repunte refleja un sesgo de muestreo en LHR–MAD (NPS 100.0, n=3) sin incidencias y la valoración de flota A320neo entre pasajeros europeos.

ECONOMY LH: Desempeño estable  
La cabina Economy de LH mantuvo desempeño estable con un NPS de 20.56 en Global/LH/Economy (subida de 3.52 pts vs L7d). No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción sin incidencias destacadas.

BUSINESS LH: Caída marcada por incidencias en A350 estándar  
La cabina Business de LH registró un NPS de 19.05 en Global/LH/Business (descenso de 6.77 pts vs L7d), golpeada por 39 incidentes –overbooking en IB271/272, avería en IB243 y 49 pérdidas de conexión– y el bajo NPS de la flota A350 estándar (–10.0 vs 100.0 en A350 next), afectando rutas como EZE–MAD sin verbatims de respaldo.

PREMIUM LH: Subida atípica por posible sesgo de muestra  
El segmento Premium de LH logró un NPS de 44.44 en Global/LH/Premium (subida de 33.96 pts vs L7d), un salto no explicado por métricas operativas ni comentarios de clientes. Aun con 39 incidencias reportadas en MAD–GIG e IB271/272, la fuerte subida apunta a un sesgo de muestra o mejoras puntuales en touchpoints no capturados por las herramientas.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-09-30 to 2025-09-30:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

Economy Short Haul: existen ambos nodos (SH/Economy/IB y SH/Economy/YW), los cuales muestran patrones distintos en perfiles-drivers (IB impulsado por residentes en España y codeshare LATAM/Others; YW por Oriente Medio y LH), lo que indica que la causa de la anomalía es específica de cada compañía.

Business Short Haul: existen ambos nodos (SH/Business/IB con anomalía positiva y SH/Business/YW con anomalía negativa) y divergencia total en la dirección y evidencia operativa de la variación, por lo que la causa es también específica de cada compañía.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul: los dos cabins reaccionan de forma distinta—Economy SH muestra una anomalía positiva convergente en IB y YW, mientras que Business SH, aunque globalmente “normal”, evidencia divergencia (IB positiva vs YW negativa), por lo que las causas son específicas de cada cabina y compañía.

Long Haul: las tres cabinas divergen—Economy LH al alza, Business LH a la baja y Premium LH estable—de modo que no hay un factor común al radio y Premium actúa como amortiguador ante la variabilidad operativa.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Tanto Short Haul (+15.3 pts) como Long Haul (+7.0 pts) presentan anomalías positivas netas impulsadas por un repunte común en Economy; los drivers y la evidencia operativa convergen en ambos radios y el nodo Global (+12.4 pts) amplifica esta tendencia de alcance transversal.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

1) Naturaleza de la causa: Recuperación operativa y perfil de ocio en Short Haul Economy  
A. Hipótesis  
   • La mejora del NPS en Economy SH responde a la combinación de dos drivers:  
     1. Predominio de pasajeros de ocio y residentes en España, con expectativas de servicio más alineadas y tolerancia a pequeñas incidencias.  
     2. Protocolos de recuperación de disruptivos (cambios de aeronave, reubicaciones, gestión rápida de conexiones perdidas) aplicados de forma efectiva en rutas críticas.  

B. Evidencia consolidada y alcance  
   • Segmento “más grande”: Global/Short Haul/Economy (NPS 43.77 vs baseline 27.35; +16.42 pts)  
   • Este driver positivo se observa de forma consistente en todos los sub-nodos bajo SH/Economy (IB +16.66 pts; YW +17.47 pts).  
   • Métricas clave:  
     – Incidentes operativos totales: 23 (6 cancelaciones, 9 retrasos, 4 equipaje, 1 limitación de aeronave, 2 otros)  
     – Rutas con mayor impacto:  
         • BOG-MAD: NPS 42.9 (n=21) con 6 retrasos y 4 cancelaciones  
         • MAD-JFK: NPS 40.0 (n=5) con 3 retrasos y 2 cancelaciones  
         • GRU-MAD: NPS 33.3 (n=6) sin incidentes  
     – Verbatims: ninguno disponible para ese día  
   • Conclusión: un elevado volumen de encuestas de ocio (n=266) y residentes en España (n=190), junto a protocolos de contingencia efectivos, impulsan una anomalía positiva homogénea en todo SH/Economy.  

2) Naturaleza de la causa: Disrupciones técnicas en A350 next y paros de control aéreo golpean Business  
A. Hipótesis  
   • Los clientes Business experimentaron una notable degradación del servicio por:  
     1. Fallos técnicos recurrentes en la flota A350 next (AOG, inspecciones adicionales).  
     2. Huelga parcial de controladores aéreos que provocó retrasos, cancelaciones y reprogramaciones masivas.  

B. Evidencia consolidada y alcance  
   • Segmento “más grande”: Global/Long Haul/Business (NPS 16.0 vs baseline 25.82; –9.82 pts)  
   • Afecta a todos los sub-nodos bajo LH/Business (no hay separación IB/YW en LH).  
   • Métricas clave:  
     – Incidentes operativos totales: 21 (6 retrasos, 4 cancelaciones, 5 reclamaciones de equipaje, 3 overbooking, 3 otros)  
     – Rutas con mayor impacto:  
         • BOG-MAD: NPS –100.0 (n=1) por cancelación y reprogramación con A350 next  
         • JFK-MAD: NPS 50.0 (n=2) tras demoras y problemas de equipaje en A350 next  
         • EZE-MAD: NPS 100.0 (n=3) sin incidentes  
     – Flota A350 next: NPS –75.0 (principal foco de insatisfacción); A332 y A333 por encima de 40 pts  
     – Verbatims: no disponibles  
   • Conclusión: las fallas en la flota A350 next y la huelga de ATC explican la anomalía negativa de Business LH y requieren revisión de mantenimiento preventivo, planes de contingencia y comunicaciones proactivas.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
Durante el día 2025-09-30 se registraron subidas de NPS en los segmentos Global (de 24.99 a 37.42, +12.43 pts vs L7d), Long Haul Economy (Global/LH/Economy: de 17.04 a 26.92, +9.88 pts vs L7d), Short Haul Economy (Global/SH/Economy: de 27.35 a 43.77, +16.42 pts vs L7d) y Short Haul Business IB (Global/SH/Business/IB: de 51.44 a 67.74, +16.30 pts vs L7d). En contrapartida, Long Haul Business sufrió una caída (Global/LH/Business: de 25.82 a 16.00, –9.82 pts vs L7d) y Short Haul Business YW experimentó un deterioro crítico (Global/SH/Business/YW: de 29.58 a 0.00, –29.58 pts vs L7d). Estos movimientos responden a:  
- Eficiente gestión de ocio y protocolos de contingencia en Economy SH (23 incidentes NCS atendidos proactivamente, predominio de residentes en España y pasajeros Leisure).  
- Sesgo muestral y rutas puntuales de alta satisfacción en SH Business IB (100 pts en BCN-MAD con n=4, flotas A319/A321 sin incidencias).  
- Fallos técnicos en A350 next y huelga de controladores que lastraron Business LH (21 incidentes en rutas BOG-MAD y JFK-MAD, A350 next con NPS –75.0).  
- Problemas operativos y falta de feedback en SH Business YW que derivaron en una caída absoluta del NPS.

Las rutas más afectadas por volumen de incidencias fueron BOG-MAD y MAD-JFK (retrasos, cancelaciones y equipaje). MAD-NRT mostró un NPS de 0.0 sin incidentes NCS, alertando de brechas no capturadas. Los perfiles más reactivos incluyen viajeros Leisure en SH Economy (NPS ocio 45.9 vs Business 37.9), residentes en España y partners de codeshare LATAM/Others; por el contrario, pasajeros Business en A350 next y codeshare YW presentaron la mayor insatisfacción.

  
ECONOMY SH: Contención y recuperación de satisfacción  
La cabina Economy SH IB experimentó una subida de 16.66 puntos, pasando de un NPS de 23.89 a 40.55 (2025-09-30 vs L7d), y SH YW aumentó 17.47 puntos de 33.93 a 51.40. En conjunto, Economy SH registró un NPS de 43.77 con una subida de 16.42 pts vs L7d. La causa principal fue el dominio de pasajeros de ocio y residentes en España, con protocolos de recuperación de 23 NCS (6 cancelaciones, 9 retrasos y 4 incidencias de equipaje) aplicados eficazmente. Esta mejora se reflejó en rutas como BOG-MAD (NPS 42.9, n=21) y GRU-MAD (33.3, n=6 sin incidentes).  

BUSINESS SH: Estabilidad con divergencia por compañía  
En SH Business, IB subió 16.30 puntos de 51.44 a 67.74 y YW cayó 29.58 puntos de 29.58 a 0.00, pero el neto Global/SH/Business se mantuvo estable en 50.0 (+5.6 pts vs L7d). No se detectaron cambios significativos a nivel global, aunque la divergencia IB vs YW evidencia un fenómeno de sesgo muestral en rutas como BCN-MAD (100.0, n=4) y altos niveles de disrupción sin respuesta en YW.  

ECONOMY LH: Mejora moderada impulsada por protocolos y flotas fiables  
La cabina Economy LH subió 9.88 puntos, pasando de 17.04 a 26.92 (2025-09-30 vs L7d). El incremento se atribuye a la gestión satisfactoria de incidencias en rutas clave (BOG-MAD, MAD-JFK, JFK-MAD) y a un mejor desempeño de flotas como A321XLR y A332 (NPS 50.0 y 42.6). Los pasajeros Leisure lideraron la subida frente a Business, y los residentes en América Centro mostraron la mayor reactividad (45.6 pts).  

BUSINESS LH: Caída por fallos técnicos y huelga de ATC  
La cabina Business LH empeoró 9.82 puntos, de 25.82 a 16.00 (2025-09-30 vs L7d). Los drivers principales fueron los repetidos problemas técnicos en A350 next y la huelga parcial de controladores aéreos, que generaron 21 NCS (6 retrasos, 4 cancelaciones y 5 reclamaciones de equipaje). Las rutas más impactadas fueron BOG-MAD (–100.0, n=1) y JFK-MAD (50.0, n=2).  

PREMIUM LH: Desempeño estable dentro de rango  
El segmento Premium LH registró un NPS de 16.67, con una subida de 6.18 puntos vs L7d, pero se considera variación normal. No se detectaron cambios significativos, manteniendo niveles de satisfacción consistentes en rutas SHARED sin registrar incidencias críticas.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-09-29 to 2025-09-29:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

Economy Short Haul  
• Ambos nodos existen: SH/Economy/IB y SH/Economy/YW.  
• Diagnóstico: patrones y drivers distintos (IB impactado sobre todo por cuestiones de flota A33ACMI y reclasificaciones de asiento; YW condicionado por cancelaciones/retrasos y variabilidad fuerte según región y operador), por lo que la causa es específica de compañía.  

Business Short Haul  
• Ambos nodos existen: SH/Business/IB y SH/Business/YW.  
• Diagnóstico: impactos divergentes (IB mostró un sesgo positivo por falta de respuestas de los vuelos problemáticos; YW sufrió un desplome ligado a cancelaciones y retrasos no captados en rutas con encuestas), así que la causa es específica de compañía.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul  
El patrón de anomalías en SH no es común al radio, sino que varía claramente por cabina y compañía: Economy y Business muestran signos y magnitudes distintas (IB apenas cae en Economy y sube en Business, mientras que YW sufre desplomes en ambas), lo que indica causas específicas de cabina/compañía y no un efecto homogéneo en todo SH.  

Long Haul  
Todas las clases en LH comparten la misma dirección de impacto (anomalías negativas) atribuible a los picos de incidencias operativas, por lo que la causa es común al radio. Sin embargo, hay reactividad diferencial (Business –30.8 pts > Premium –22.2 pts > Economy –2.6 pts) y ninguna cabina amortigua totalmente el efecto.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Tanto Short Haul como Long Haul muestran anomalías negativas impulsadas por picos de cancelaciones, retrasos y fallos técnicos, y el nodo Global agrega coherentemente ese impacto sin compensaciones entre radios. Aunque la magnitud y los segmentos más afectados varían por cabina y compañía, la causa subyacente es homogénea a nivel operativo y se refleja fielmente en el Global.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

1) Causa: Picos de incidencias operativas (retrasos, cancelaciones y fallos técnicos)  
A. Naturaleza de la causa  
   • Hipótesis: Un elevado volumen de alteraciones en la operación (retrasos masivos, cancelaciones y cambios forzados de aeronave/asiento) redujo drásticamente la experiencia percibida.  
B. Evidencia consolidada y alcance  
   • Segmento más grande: Global (12 segmentos hijos, NPS 19.53 vs baseline 24.99; caída –5.46 pts)  
   • Output causal detallado para Global:  
     – Incidencias NCS totales: 373 (92 retrasos, 56 cancelaciones, 13 cambios de aeronave, 9 fallos técnicos)  
     – Vuelos afectados: 40; vuelo más impactado IB0379 con 2 incidentes  
     – Ruta DOH–MAD: NPS –37.5 (n=8), sin incidentes documentados en NCS  
     – No hay verbatims disponibles ni datos operativos de puntualidad u ocupación  
   • Afecta a todos los subsegmentos bajo Global (LH y SH, todas las clases y compañías)  

2) Causa: Deficiencias en operadores terceros (ACMI y code-share) en vuelos Short Haul Economy  
A. Naturaleza de la causa  
   • Hipótesis: La calidad de servicio y la gestión de incidencias en aeronaves arrendadas (ACMI) y vuelos en code-share degradaron la experiencia Economy en SH.  
B. Evidencia consolidada y alcance  
   • Segmento más grande: Global/SH/Economy (NPS 22.73 vs baseline 27.35; caída –4.62 pts)  
   • Output causal detallado para SH/Economy:  
     – Incidencias NCS: 34 (16 cancelaciones, 10 retrasos, 2 desvíos, 1 limitación de aeronave, 8 otras)  
     – Flota A33ACMI: NPS –61.1 (n=18 encuestas)  
     – Code-share VY: NPS –33.3; AA: NPS –20.0  
     – Ruta MAD–MUC: NPS –28.6 (n=7), sin incidencias NCS  
     – No hay verbatims disponibles ni datos operativos de puntualidad  
   • Afecta a todos los subsegmentos bajo SH/Economy (IB y YW)  

3) Causa: Sesgo de muestra en Short Haul Business para IB  
A. Naturaleza de la causa  
   • Hipótesis: La ausencia de respuestas de los pasajeros más afectados (vuelos cancelados o muy retrasados) en la muestra elevó artificialmente el NPS en IB Business SH.  
B. Evidencia consolidada y alcance  
   • Segmento: Global/SH/Business/IB (NPS 59.26 vs baseline 51.44; alza +7.82 pts)  
   • Output causal detallado para SH/Business/IB:  
     – Incidencias NCS: 16 cancelaciones, 10 retrasos, 3 fallos técnicos con retorno a origen  
     – Vuelo crítico IB0123 MAD–LIM reprogramado como IB0127 con 58 cambios de asiento  
     – Única ruta con NPS: FCO–MAD NPS 100.0 (n=3), sin incidencias reportadas  
     – Flota A320neo con NPS 0.0, otras flotas hasta 83.3  
     – No hay verbatims ni datos de puntualidad  
   • Afecta exclusivamente al subsegmento SH/Business/IB; el mismo patrón no se observa en SH/Business/YW.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
El 29-sep-2025 el NPS global de Iberia/YW cayó de 24.99 a 19.53 (–5.46 pts), arrastrado por incidencias operativas masivas (373 eventos NCS: 92 retrasos, 56 cancelaciones, 13 cambios de avión y 9 fallos técnicos) que afectaron por igual a Long Haul y Short Haul. En SH/Economy el NPS combinado retrocedió de 27.35 a 22.73 (–4.62 pts), con IB manteniendo 22.78 (–1.11 pts vs L7d) y YW en 22.63 (–11.31 pts) debido a deficiencias en flota ACMI (A33ACMI: –61.1 pts) y vuelos code-share (VY –33.3 pts, AA –20.0 pts). En SH/Business el indicador mixto bajó de 44.38 a 36.17 (–8.21 pts), pero con IB sorprendentemente al alza (59.26 vs 51.44, +7.82 pts) por un sesgo de muestra—solo respondieron pasajeros no afectados—mientras YW se desplomó de 29.58 a 5.00 (–24.58 pts) por cancelaciones/retrasos concentrados fuera de las rutas encuestadas. En LH/Economy el NPS pasó de 17.04 a 14.43 (–2.62 pts), impulsado por el pobre rendimiento de vuelos code-share (QR –44.4 pts, Others –66.7 pts) sin incidentes puntuales que justifiquen la insatisfacción. En LH/Business la caída fue de 25.82 a –5.00 (–30.82 pts), principalmente por el fallo técnico en IB0123 (58 reubicaciones de asiento, reprogramación MAD–LIM) que impactó especialmente a Business/Work (–33.3 pts) y flota A332 (–50.0 pts). Finalmente, LH/Premium retrocedió de 10.48 a –11.76 (–22.25 pts) con cancelaciones y fallos técnicos reiterados, destacando España (–40.0 pts) y code-share AA (–66.7 pts) entre los perfiles más insatisfechos.

Las rutas más afectadas incluyen DOH–MAD (NPS –37.5, n=8), MAD–MUC (SH/Economy, –28.6, n=7) y BOD–MAD (SH/Economy/YW, 0.0, n=4), pese a que en ninguno de estos casos hubo incidentes formales reportados, lo que sugiere brechas en servicio o comunicación. Los grupos de clientes más reactivos fueron pasajeros en flota ACMI y code-share (SH/Economy), viajeros de ocio en SH/Business/YW, y el perfil Business/Work en LH, reflejando una alta sensibilidad a cancelaciones, retrasos y reubicaciones de asiento.

ECONOMY SH IB & YW  
La cabina Economy de SH registró un NPS combinado de 22.73 el 29-sep (–4.62 pts vs L7d). En IB quedó en 22.78 (–1.11 pts) y en YW en 22.63 (–11.31 pts), debido principalmente a la operación de vuelos ACMI (A33ACMI NPS –61.1) y código compartido (VY –33.3, AA –20.0). Esta bajada se reflejó especialmente en la ruta MAD–MUC (NPS –28.6, n=7) y entre pasajeros de América Sur y Asia, sin verbatims ni datos operativos de puntualidad disponibles.

BUSINESS SH IB & YW  
El segmento Business de SH bajó a 36.17 el 29-sep (–8.21 pts vs L7d), pero con fuerte divergencia: IB subió a 59.26 (+7.82 pts) por un sesgo de muestra en FCO–MAD (100.0 pts, n=3) y ausencia de respuestas de vuelos afectados, mientras YW cayó a 5.00 (–24.58 pts) por 16 cancelaciones y 10 retrasos, sin encuestas en las rutas impactadas.

ECONOMY LH  
La cabina Economy de LH registró un NPS de 14.43 (–2.62 pts vs L7d) el 29-sep. La causa principal fue la percepción negativa de vuelos code-share (QR –44.4 pts, Others –66.7 pts) y un único trayecto con NPS 0.0 (n=5), apuntando a factores de servicio no capturados por incidentes NCS.

BUSINESS LH  
La cabina Business de LH cayó de 25.82 a –5.00 (–30.82 pts vs L7d) el 29-sep, impulsada por el fallo técnico en IB0123 MAD–LIM (58 cambios de asiento y reprogramación), que afectó sobre todo a pasajeros Business/Work (–33.3 pts) y en flota A332 (–50.0 pts). La ruta MAD–MEX mantuvo un NPS de 25.0 (n=4) sin incidencias, indicando un sesgo de muestreo y necesidad de ampliar la recogida de feedback.

PREMIUM LH  
En Premium LH el NPS retrocedió de 10.48 a –11.76 (–22.25 pts vs L7d) el 29-sep, por 26 incidencias operativas (11 retrasos, 12 otras, 3 cancelaciones) y alto impacto en pasajeros de España (–40.0 pts), código compartido AA (–66.7 pts) y flota A350 next (–50.0 pts), pese a que la ruta MAD–PTY registró un NPS de 75.0 (n=4) sin incidentes formales.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-09-28 to 2025-09-28:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

PASO 1 – DIAGNÓSTICO A NIVEL DE COMPAÑÍA (IB vs. YW) EN SHORT HAUL

A. Economy SH  
- Existen ambos nodos: Global/SH/Economy/IB (NEGATIVE ANOMALY –17.9 pts) y Global/SH/Economy/YW (Normal +5.7 pts).  
- Patrones y evidencia:  
  • IB presenta fuerte caída de NPS ligada a un pico de retrasos y cambios de avión (IB341/29sep, sustitución A330→A332) y alta insatisfacción en DUS-MAD.  
  • YW no muestra anomalías operativas ni descenso de NPS.  
Diagnóstico: la anomalía es específica de la operación IB en Economy SH, no un problema general de la cabina.

B. Business SH  
- Existen ambos nodos: Global/SH/Business/IB y Global/SH/Business/YW, pero ambos están dentro de rango normal (variaciones +0.3 pts para IB y +3.7 pts para YW).  
- No hay patrones de drivers ni evidencia operativa que difieran ni causen anomalía.  
Diagnóstico: no hay anomalía ni divergencia por compañía en Business SH; la cabina mantiene un desempeño homogéneo y dentro de lo esperado.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

PASO 2 – DIAGNÓSTICO A NIVEL DE CABINA

A. Short Haul  
• Economy SH vs Business SH divergen claramente: Economy registra anomalía negativa, mientras Business permanece dentro de rango normal.  
• El patrón es consistente entre compañías (solo IB/Economy cae; YW/Economy y ambas Business están estables).  
Diagnóstico: la causa de insatisfacción en Short Haul es específica de la cabina Economy, no un problema general de todo el radio.

B. Long Haul  
• Economy LH, Business LH y Premium LH convergen en anomalías negativas, señalando un driver común a todo el radio.  
• Sin embargo, hay reactividad diferencial: Premium sufre la caída más intensa, Economy moderada y Business la más amortiguada.  
Diagnóstico: las incidencias operativas afectan a todo el Long Haul, con respuesta escalonada por cabina (Business actúa como “amortiguador”).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

PASO 3 – DIAGNÓSTICO A NIVEL RADIO Y GLOBAL

Ambos radios (SH y LH) están afectados por los mismos drivers operativos (principalmente retrasos, cambios de aeronave y problemas técnicos), y el nodo Global refleja de forma coherente ese impacto agregado sin que unas caídas compensen a otras, indicando una causa de alcance verdaderamente global.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

PASO 4 – ANÁLISIS PROFUNDO DE CAUSAS IDENTIFICADAS

A continuación se consolidan las dos causas raíz que explican la caída de NPS, el alcance sobre los segmentos más amplios y la evidencia soportante.

1. Causa: Elevado número de incidencias operativas (retrasos, cambios de avión y problemas técnicos)  
A. Naturaleza de la causa  
 • Hipótesis: La acumulación de incidencias durante el día (principalmente retrasos y cambios de aeronave) ha erosionado de forma homogénea la percepción de la experiencia de vuelo en Long Haul y, en menor medida, en Short Haul Economy de IB.  

B. Evidencia consolidada y alcance  
 • Segmento “más grande” afectado: Global / Long Haul (NPS día 5.36 vs baseline 17.87; anomalía –12.51 pts).  
 • Afecta a todos los subsegmentos bajo Long Haul:  
   – LH / Economy (–11.58 pts)  
   – LH / Business (–0.82 pts)  
   – LH / Premium (–24.12 pts)  
 • Métricas operativas clave (ncs_tool):  
   – Total incidentes LH: 47  
     · Retrasos: 19  
     · Cambios de aeronave: 5  
     · Problemas técnicos: 4  
     · Otros (equipaje, demoras adicionales…): 19  
   – Vuelos críticos: IB341 (MAD-ORD) con sustitución A330→A332 y 6 h de demora en cabina J/C  
 • Rutas más impactadas:  
   – GYE-MAD (NPS 0.0, n=10)  
   – BOS-MAD (NPS 12.5, n=8)  
   – LAX-MAD (NPS 28.6, n=7)  
 • Verbatims representativos: No hay comentarios abiertos disponibles para este día.  

2. Causa: Fallos operativos específicos de IB en Economy Short Haul (retrases y cambios en flota)  
A. Naturaleza de la causa  
 • Hipótesis: La operación de Iberia en vuelos cortos de cabina Economy sufrió incidencias concentradas (retrasos y sustituciones de avión) que solo afectaron a ese subsegmento, generando una caída de casi 18 pts en NPS.  

B. Evidencia consolidada y alcance  
 • Segmento “más grande” afectado: Global / SH / Economy / IB (NPS 6.01 vs baseline 23.89; anomalía –17.89 pts).  
 • Afecta a todos los subsegmentos bajo SH/Economy/IB (no hay SH/Economy/YW con anomalía).  
 • Métricas operativas clave (ncs_tool):  
   – Total incidentes SH: 35  
     · Retrasos: 17  
     · Cambios de aeronave: 5 (destaca IB341/29sep y IB342)  
     · Otras (cancelaciones menores, equipaje…): 13  
   – Rutas críticas: DUS-MAD (NPS 0.0, n=11), BRU-MAD (NPS 0.0, n=13)  
 • Verbatims representativos: No hay comentarios abiertos disponibles para este día.  

Resumen consolidado  
- La causa principal es la elevada tasa de incidencias operativas, que impacta globalmente el Long Haul y, de forma más localizada, la cabina Economy de Short Haul en Iberia.  
- Los subsegmentos Business Short Haul y las operaciones YW no presentan anomalías, actuando de “amortiguador” frente a problemas operativos.  
- La ausencia de verbatims subraya la necesidad de recuperar canales de feedback cualitativo para profundizar en las sensaciones de los pasajeros.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
El análisis del 28-sep-2025 revela fuertes bajadas de NPS en múltiples niveles. A nivel global, el NPS cayó de 24,99 a 14,76 (–10,23 pts), impulsado por Long Haul (de 17,87 a 5,36; –12,51 pts) y, en Short Haul, por la cabina Economy (de 27,35 a 17,59; –9,76 pts). En Long Haul, la clase Premium sufrió la mayor caída (de 10,48 a –13,64; –24,12 pts), seguida de Economy (de 17,04 a 5,46; –11,58 pts) y Business (de 25,82 a 25,00; –0,82 pts). En Short Haul /Economy de Iberia (Global/SH/Economy/IB), el NPS se desplomó de 23,89 a 6,01 (–17,89 pts), mientras que Vueling (Global/SH/Economy/YW) mejoró ligeramente de 33,93 a 39,60 (+5,67 pts). Business SH mantuvo desempeño estable (IB: 51,44→51,72; +0,29 pts; YW: 29,58→33,33; +3,75 pts).  
Se identificaron dos causas principales:  
• Incidencias operativas elevadas (47 en LH, 35 en SH) – retrasos, reemplazos de avión y problemas técnicos – que impactaron de forma homogénea todo Long Haul y erosionaron el NPS en sus subsegmentos Economy, Business y Premium.  
• Problemas específicos en la operación Iberia de Economy Short Haul – concentrados en vuelos como IB341/IB342 con sustitución de A330 a A332 y múltiples retrasos – responsables de la caída en Global/SH/Economy/IB.  

Las rutas más afectadas incluyen GYE-MAD (LH Economy NPS 0,0), BOS-MAD y LAX-MAD en Long Haul, así como DUS-MAD y BRU-MAD en Short Haul/Economy, ambas con NPS 0,0 pese a ausencia de incidencias reportadas. Los grupos más reactivos fueron pasajeros residentes en Asia y África, clientes en código compartido con LATAM, BA y AA, y usuarios de flotas A333 y A33ACMI, que mostraron los descensos de NPS más acusados.  

ECONOMY SH: Impacto Concentrado en IB  
La cabina Economy de SH experimentó un deterioro de 9,76 puntos vs L7d, pasando de un NPS de 27,35 a 17,59. Iberia cayó de 23,89 a 6,01 (–17,89 pts) debido a 17 retrasos, 5 cambios de avión (IB341/29sep y IB342) y 13 incidencias varias; Vueling, en cambio, mejoró de 33,93 a 39,60 (+5,67 pts), manteniendo desempeño estable. Esta bajada se reflejó especialmente en las rutas DUS-MAD y BRU-MAD con NPS 0,0, y los perfiles más reactivos fueron los pasajeros de las flotas A33ACMI, codeshare “Others” y residentes en Asia y África.  

BUSINESS SH: Desempeño Estable  
El segmento Business de SH mantuvo desempeño estable, con un NPS combinado de 44,38→44,68 (+0,30 pts vs L7d). Iberia subió de 51,44 a 51,72 (+0,29 pts) y Vueling de 29,58 a 33,33 (+3,75 pts). No se detectaron cambios significativos, manteniéndose niveles consistentes de satisfacción en todas las rutas y perfiles.  

ECONOMY LH: Deterioro Generalizado  
La cabina Economy de LH cayó de 17,04 a 5,46 (–11,58 pts vs L7d), afectada por 19 retrasos, 5 cambios de aeronave y 19 incidencias adicionales en 47 eventos operativos. Las rutas MAD-UIO (NPS 7,7), GYE-MAD (0,0) y LAX-MAD (28,6) fueron los focos de mayor insatisfacción, especialmente entre pasajeros de código compartido con BA y AA y usuarios de flota A350 C y A333.  

BUSINESS LH: Ligera Caída  
La cabina Business de LH registró un NPS de 25,82→25,00 (–0,82 pts vs L7d), arrastrada por 19 retrasos, 9 incidencias varias y 5 cambios de avión, con foco en IB341/MAD-ORD. El malestar fue notable en clientes de flota A333 (NPS –50,0) y residentes en España (–25,0 pts), mientras que los viajeros en A350 y de América Centro mostraron niveles altos de satisfacción.  

PREMIUM LH: Caída Extrema  
El segmento Premium de LH sufrió la mayor bajada, de 10,48 a –13,64 (–24,12 pts vs L7d), impulsada por los mismos 47 incidentes operativos y, en particular, por la sustitución de aeronave y retraso de 6 h en IB341/29sep. El código compartido AA (–100,0 pts), la flota A333 (–50,0 pts) y los pasajeros desde Norteamérica (–66,7 pts) registraron las reacciones más negativas.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-09-27 to 2025-09-27:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

Economy Short Haul:  
• Existen ambos nodos:  
  – SH/Economy/IB (+8.0 pts)  
  – SH/Economy/YW (–6.7 pts)  
• Patrones distintos:  
  – IB: impulso positivo por alta satisfacción de clientes de América Centro, sin incidencias operativas relevantes en rutas clave.  
  – YW: caída atribuible a cancelaciones y retrasos (BOG–MAD, MAD–BOG), mayor insatisfacción de viajeros de negocio y residentes en Norteamérica.  
Diagnóstico: la causa es específica de compañía.

Business Short Haul:  
• Existen ambos nodos:  
  – SH/Business/IB (–8.1 pts)  
  – SH/Business/YW (+20.4 pts)  
• Patrones opuestos:  
  – IB: impacto negativo por incidentes operativos (IFE inoperativo, cancelaciones) y perfiles corporativos insatisfechos.  
  – YW: fuerte alza sustentada en ocio y clientes residentes en España, pese a incidencias NCS.  
Diagnóstico: la causa es específica de compañía.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul  
Patrón divergente entre cabinas: tanto Economy SH como Business SH presentan cancelación de efectos en el agregado (nodos “Normal”) fruto de divergencias IB vs YW. Este mismo esquema de despliegue (IB positiva–YW negativa en Economy y IB negativa–YW positiva en Business) se replica en ambas cabinas.  
Diagnóstico SH: la causa no es común al radio sino específica a la combinación cabina + compañía (IB/YW).

Long Haul  
Las tres cabinas (Economy LH, Business LH y Premium LH) convergen claramente en una caída de NPS atribuible a incidentes operativos el mismo día. No hay desviaciones de signo entre clases; varía solo la magnitud (Premium más impactada).  
Diagnóstico LH: la causa es común al radio, con Premium como el segmento más reactivo y Economy/Business amortiguando parcialmente el impacto.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Diagnóstico global: las causas son mixtas y compensatorias – Long Haul muestra una anomalía negativa homogénea en todas sus cabinas por incidencias operativas, mientras Short Haul exhibe efectos opuestos en IB vs YW que se cancelan, derivando en un Global ligeramente positivo.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

Causa 1. Incidentes operativos en ruta BOG–MAD (y vuelos asociados)  
A. Naturaleza de la causa  
   • Hipótesis: fallos de entretenimiento (IFE inoperativo), cancelaciones y retrasos concentrados en BOG–MAD (vuelo IB154/27SEP) generaron un deterioro generalizado de la experiencia, que arrastró a todos los segmentos de Long Haul y a los subsegmentos de Short Haul expuestos a esas mismas incidencias.  

B. Evidencia consolidada y alcance  
   • Segmento más impactado (mayor caída absoluta): Global/LH/Premium  
     – NPS 4.35 vs baseline 10.48 (–6.14 pts)  
     – Incidentes NCS: 28 totales (4 cancelaciones, 6 retrasos, 2 limitaciones de aeronave, 4 equipaje, 10 “otras”); vuelo IB154/BOG-MAD con IFE inoperativo  
     – Rutas críticas: BOG-MAD (4 incidentes; NPS 0.0), MAD-BOG (incidencias recurrentes)  
     – Perfiles más insatisfechos: pasajeros en flota A333 (NPS –100.0), código AA (–100.0), residente América Norte (–66.7)  
     – Verbátims: no disponibles (limitación de feedback cualitativo)  
   • Alcance: afecta a todos los subsegmentos bajo Global/LH (Economy, Business y Premium) y, como extensión, a los subsegmentos de Short Haul que registraron anomalías negativas (Global/SH/Economy/YW y Global/SH/Business/IB).  

Causa 2. Dinámicas de satisfacción específicas por compañía en Short Haul  
A. Naturaleza de la causa  
   • Hipótesis: diferencias en mix de cliente (perfil, origen, propósito de viaje) y en percepción de servicio entre Iberia (IB) y Level (YW) generan anomalías opuestas que se cancelan a nivel agregado.  

B. Evidencia consolidada y alcance  
   1. Sub-causa A – Alto NPS por Economy/IB  
     – Segmento: Global/SH/Economy/IB  
     – NPS 31.90 vs baseline 23.89 (+8.01 pts)  
     – Incidentes: 26 totales (8 cancelaciones, 5 retrasos, 13 “otras”); ninguna ruta clave (MAD-PRG) registró impacto operativo  
     – Motor principal: elevada satisfacción de pasajeros de América Centro (NPS +86.7) y ausencia de incidencias graves en rutas con mayor encuesta  
     – Alcance: todos los vuelos y clientes bajo SH/Economy/IB  
   2. Sub-causa B – Alto NPS por Business/YW  
     – Segmento: Global/SH/Business/YW  
     – NPS 50.0 vs baseline 29.58 (+20.42 pts)  
     – Incidentes: 26 totales (8 cancelaciones, 8 “otras”); la ruta DUS-MAD (n=3) no registró eventos NCS  
     – Motor principal: fuerte sesgo positivo de viajeros de ocio y residentes en España (NPS 100.0), que contrarrestó la caída de perfiles Business y de otras regiones  
     – Alcance: todos los vuelos y clientes bajo SH/Business/YW  

En conjunto, las dos causas explican tanto las caídas pronunciadas en Long Haul y en ciertos subsegmentos de Short Haul afectados por incidencias operativas, como los picos de NPS de los clientes mejor servidos o con mix más favorable en SH/IB y SH/YW.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El 27 de septiembre de 2025 el NPS Global subió de 24.99 a 25.82 (+0.83 pts) tras la combinación de dos fenómenos contrapuestos. En Long Haul, todas las cabinas retrocedieron por incidentes operativos en la ruta BOG–MAD (IFE inoperativo, cancelaciones y retrasos): Economy LH cayó de 17.04 a 16.53 (–0.51 pts), Business LH de 25.82 a 25.00 (–0.82 pts) y Premium LH se desplomó de 10.48 a 4.35 (–6.14 pts). Por el contrario, en Short Haul las variaciones dependieron de la compañía: SH/Economy/IB subió de 23.89 a 31.90 (+8.01 pts) mientras SH/Economy/YW bajó de 33.93 a 27.22 (–6.72 pts); en Business SH IB cayó de 51.44 a 43.33 (–8.10 pts) y YW escaló de 29.58 a 50.00 (+20.42 pts).

Las incidencias en BOG–MAD y MAD–BOG lideraron las presiones negativas, afectando con máxima intensidad a Premium LH (NPS 0.0 en BOG–MAD) y arrastrando a Economy LH y Business LH. En sentido contrario, la elevada satisfacción de viajeros de América Centro en SH/Economy/IB (NPS +86.7) y de residentes en España en SH/Business/YW (NPS +100.0) aportó los picos positivos. Las rutas más afectadas fueron BOG–MAD, MAD–BOG, MAD–PRG y DUS–MAD; los perfiles más reactivos incluyeron a pasajeros en flota A333 (–100.0 en Premium LH), code-share AA (–100.0 en Premium LH), viajeros de América Centro (+86.7 en Economy SH/IB) y residentes en Norteamérica (–17.6 en Economy SH/YW).

ECONOMY SH: Divergencia de Compañía  
La cabina Economy SH experimentó movimientos contrapuestos el 27-SEP. SH/Economy/IB alcanzó un NPS de 31.90 (vs L7d 23.89, +8.01 pts), impulsada por la satisfacción de pasajeros de América Centro y sin incidencias operativas en MAD–PRG. En paralelo, SH/Economy/YW descendió a 27.22 (vs L7d 33.93, –6.72 pts) debido a cancelaciones y retrasos en BOG–MAD y MAD–BOG, con peor desempeño de clientes de negocio y residentes en Norteamérica. Esta divergencia refleja mix de perfiles y diferencias operativas entre compañías.

BUSINESS SH: Contrastes por Operador  
En Business SH, SH/Business/IB cayó a 43.33 (vs L7d 51.44, –8.10 pts) por incidencias de IFE inoperativo en IB154 (BOG–MAD) y baja satisfacción de viajeros corporativos en flota A333. Al mismo tiempo, SH/Business/YW escaló a 50.00 (vs L7d 29.58, +20.42 pts), sostenido por viajeros de ocio y residentes en España (NPS +100.0) en la ruta DUS–MAD sin reportes NCS. La compensación de ambos efectos mantuvo estable el NPS agregado de Business SH.

ECONOMY LH: Deterioro por Incidencias  
La cabina Economy LH retrocedió de 17.04 a 16.53 (–0.51 pts) el 27-SEP, impactada por 6 retrasos, 4 cancelaciones y un IFE inoperativo en el vuelo IB154/BOG–MAD. Las rutas BOG–MAD y MAD–BOG agruparon la mayoría de incidencias, mientras EZE–MAD, sin reportes, mostró NPS 10.8. Los pasajeros de Asia y Europa fueron los más críticos.

BUSINESS LH: Impacto Generalizado  
Business LH descendió de 25.82 a 25.00 (–0.82 pts) por 6 retrasos y 4 cancelaciones en Long Haul; BOG–MAD cerró con NPS 0.0 y GRU–MAD con 12.5 sin incidencias. El fallo de IFE y problemas de equipaje en flota A333 arrastraron la percepción de clientes corporativos y residentes en Europa y Asia.

PREMIUM LH: Caída Aguda  
Premium LH sufrió el mayor golpe, cayendo de 10.48 a 4.35 (–6.14 pts) por 28 incidentes (4 cancelaciones, 6 retrasos y IFE inoperativo en IB154/BOG–MAD). La ruta BOG–MAD registró NPS 0.0, con pasajeros en A333 y code-share AA puntuando –100.0, y residentes en Norteamérica en –66.7.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-09-26 to 2025-09-26:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

Economy Short Haul: existen ambos nodos (SH/Economy/IB y SH/Economy/YW) y muestran patrones opuestos (IB con fuerte caída ligada a cancelaciones/retrasos y mala valoración en FCO–MAD; YW con alza impulsada por Leisure, ATR e IB code-share). Diagnóstico: causa específica de compañía.

Business Short Haul: existen ambos nodos (SH/Business/IB y SH/Business/YW) y ambos presentan caída pronunciada atribuible al mismo driver (alto volumen de cancelaciones y retrasos que penaliza a business travelers, sin verbatims disponibles). Diagnóstico: causa común a la cabina.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul (SH): patrón específico de cabina.  
– Business SH sufre una anomalía negativa uniforme (IB y YW convergen en drivers de cancelaciones y retrasos),  
– mientras que Economy SH muestra respuestas divergentes por compañía (IB con fuerte caída, YW con alza) y una menor reactividad global (–3.0 pts vs –29.3 pts).  

Long Haul (LH): patrón específico de cabina.  
– Economy LH se mantiene estable (normal),  
– Business LH (+30.4 pts) y Premium LH (+22.8 pts) presentan anomalías positivas crecientes,  
– mostrando una progresión lógica de reactividad (Business > Premium > Economy, que actúa como amortiguador).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Ambos radios están afectados pero con drivers opuestos: Short Haul cae por cancelaciones y retrasos mientras Long Haul sube por el sesgo de muestra en Business y Premium, y a nivel Global estos efectos se compensan (–1,0 pts neto), atenuando los patrones individuales.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

Causa 1: Incumplimientos operativos (cancelaciones y retrasos)  
A. Naturaleza de la causa  
  Hipótesis: La alta tasa de cancelaciones y retrasos en Short Haul generó frustración generalizada, especialmente entre viajeros de negocio, penalizando la percepción de la cabina independientemente de la compañía.  

B. Evidencia consolidada y alcance  
  Segmento de referencia: Global/Short Haul/Business (anomalía –29.28 pts)  
  • NPS día 15.09 vs baseline 44.38 pts  
  • Incidentes NCS (ncs_tool): 36 totales  
    – 26 cancelaciones (control ATC, restricciones NCE)  
    – 7 retrasos  
    – 2 técnicos y 1 equipaje  
  • Rutas clave (routes_tool):  
    – Única con NPS: BCN–MAD NPS 100.0 (n=6) sin incidentes NCS  
    – Principales incidencias en BOG–MAD (IFE inoperativo)  
  • Perfiles (customer_profile_tool):  
    – Business/Work: NPS –21.4  
    – Leisure: +28.2  
    – Residence Region: –23.1 a +40.0 según zona  
  • Verbatims: no disponibles  
  Alcance: afecta a todos los subsegmentos bajo Global/SH (Economy y Business, IB y YW).  

Causa 2: Sesgo de composición de muestra en Long Haul  
A. Naturaleza de la causa  
  Hipótesis: Aun con incidencias operativas, predominó el feedback de subgrupos muy satisfechos (viajeros Business/Leisure en flotas ATR y vuelos con NPS perfectos), generando un alza artificial del indicador.  

B. Evidencia consolidada y alcance  
  Segmento de referencia: Global/Long Haul/Business (anomalía +30.43 pts)  
  • NPS día 56.25 vs baseline 25.82 pts  
  • Incidentes NCS: 29 totales  
    – 12 cancelaciones, 12 retrasos, 5 “otros”  
  • Rutas (routes_tool):  
    – BOG–MAD: NPS 100.0 (IFE fuera de servicio, 5 pasajeros)  
    – MAD–SDQ: NPS 100.0 sin incidencias  
  • Perfiles (customer_profile_tool):  
    – Residence Region: –12.5 a +85.7  
    – Fleet: variación según familia (A332/A350 peor)  
    – Business vs Leisure: spread 0.9  
  • Verbatims: no disponibles  
  Alcance: afecta a Global/LH y todos sus subsegmentos (Business y Premium).

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:  
El análisis de NPS para el 26-sep-2025 revela subidas y bajadas muy pronunciadas en los distintos segmentos. En Short Haul Economy el NPS cayó de 27.35 a 24.33 (–3.02 pts), impulsado por Iberia (SH/Economy/IB) que pasó de 23.89 a 13.31 (–10.58 pts) a causa de 26 cancelaciones, 7 retrasos y una valoración de 4.3 pts en FCO–MAD, mientras que Vueling (SH/Economy/YW) mejoró de 33.93 a 43.50 (+9.57 pts) gracias al alto NPS de pasajeros Leisure en ATR. El segmento Business de SH sufrió un desplome de 44.38 a 15.09 (–29.28 pts): Iberia (SH/Business/IB) bajó de 51.44 a 18.92 (–32.52 pts) y Vueling (SH/Business/YW) de 29.58 a 6.25 (–23.33 pts), reflejo directo de 26 cancelaciones y 7 retrasos que penalizaron a viajeros corporativos. En Long Haul, Economy se mantuvo estable (17.04→17.22, +0.18 pts), mientras Business saltó de 25.82 a 56.25 (+30.43 pts) por puntuaciones perfectas en BOG–MAD y MAD–SDQ, y Premium escaló de 10.48 a 33.33 (+22.85 pts) gracias al sesgo de muestra de clientes de Suramérica y business en A333, a pesar de 12 cancelaciones y 12 retrasos.

Las rutas más afectadas fueron BCN–MAD (SH/Economy con NPS –8.8), FCO–MAD (SH/Economy/IB con 4.3), BOG–MAD (incidentes en SH/Business e IFE inoperativo), y MAD–SDQ (100 pts en LH/Business). Los grupos más reactivos incluyen viajeros de negocio —especialmente en Asia y usuarios de flota A33ACMI en SH— y pasajeros de ocio en ATR y code-share IB en LH.

ECONOMY SH (SH/Economy)  
La cabina Economy de SH registró un NPS de 24.33 pts el 26-sep-2025, cayendo 3.02 pts vs L7d. Iberia (SH/Economy/IB) anotó 13.31 pts (–10.58 pts) ante 26 cancelaciones, 7 retrasos y la mala valoración de FCO–MAD (4.3 pts), mientras que Vueling (SH/Economy/YW) subió a 43.50 pts (+9.57 pts) apoyada en viajeros Leisure en ATR. Esta caída neta se reflejó especialmente en BCN–MAD (–8.8 pts) y en la alta insatisfacción de business travelers y residentes en Asia.

BUSINESS SH (SH/Business)  
El segmento Business de SH descendió a 15.09 pts el 26-sep, perdiendo 29.28 pts vs L7d. Iberia (SH/Business/IB) cayó a 18.92 pts (–32.52 pts) y Vueling (SH/Business/YW) a 6.25 pts (–23.33 pts) debido a 26 cancelaciones y 7 retrasos que penalizaron a viajeros corporativos en rutas como BOG–MAD, y que no pudo compensar ni siquiera el NPS de 100 pts en BCN–MAD.

ECONOMY LH (LH/Economy)  
La cabina Economy de LH mantuvo desempeño estable con 17.22 pts el 26-sep (↑0.18 pts vs L7d). No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción.

BUSINESS LH (LH/Business)  
La cabina Business de LH experimentó un fuerte repunte hasta 56.25 pts el 26-sep, mejorando 30.43 pts vs L7d. Esta subida obedece a puntuaciones perfectas en BOG–MAD y MAD–SDQ, que compensaron 12 cancelaciones y 12 retrasos gracias al sesgo de muestra de viajeros altamente satisfechos.

PREMIUM LH (LH/Premium)  
El segmento Premium de LH ascendió a 33.33 pts el 26-sep, +22.85 pts vs L7d. Las principales causas fueron las valoraciones elevadas de clientes de América Sur y business en flota A333 —por ejemplo NPS 66.7 en BOG–MAD— que contrarrestaron los impactos operativos negativos.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-09-25 to 2025-09-25:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

Economy Short Haul: existen ambos nodos (SH/IB con anomalía negativa de –5.1 pts y SH/YW con variación normal). La divergencia—IB muy impactada y YW estable—indica que la causa es específica de la operación de IB, no general a toda la cabina Economy.

Business Short Haul: también hay dos nodos (SH/IB con –10.1 pts y SH/YW con +16.2 pts). La oposición en resultados y drivers revela causas diferenciadas por compañía, no un factor común a la cabina Business.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

Short Haul: Economy SH y Business SH convergen en una anomalía negativa a nivel de cabina (ambas impactadas por cancelaciones/retrasos), aunque internamente Business muestra divergencia IB (-) vs YW (+). En conjunto, la caída es común al radio SH, pero la intensidad varía por compañía.

Long Haul: Economy LH (normal), Business LH (+) y Premium LH (–) divergen claramente entre sí. No hay un patrón único para todo el radio LH: cada clase reacciona de forma distinta ante los mismos eventos operativos.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

Solo Short Haul registra un deterioro generalizado por cancelaciones y retrasos, mientras que en Long Haul las clases reaccionan de forma mixta (Business al alza, Premium a la baja). Al combinarse ambos radios, sus efectos contrapuestos se compensan parcialmente y el Global queda apenas negativo (-0,2 pts).

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

1. Causa: Incidentes operativos (cancelaciones y retrasos)  
A. Naturaleza de la causa  
   • Volumen elevado de cancelaciones y retrasos, potenciado por la huelga general en Italia y fallos técnicos, que impactó la puntualidad y la experiencia de embarque.  
B. Evidencia consolidada y alcance  
   • Segmento más grande afectado: Short Haul (Global/SH)  
     – NPS Día: 26,86 pts vs Baseline: 29,14 pts (–2,27 pts)  
     – Incidentes totales: 48 (22 cancelaciones, 20 retrasos, 6 otros)  
     – Rutas con mayor incidencia: BRU-MAD, BCN-SCL  
     – Verbatims: no disponibles  
   • Esta causa afecta a todos los subsegmentos bajo Short Haul:  
     – Economy SH (IB y YW) y Business SH (IB y YW)  
   • Métricas clave:  
     – Reactividad Business vs Leisure: 23,1 vs 29,0 pts  
     – Flotas: A350 C NPS 80,0; A320 NPS –1,1  
     – CodeShare I2 NPS –66,7  

2. Causa: Cancelaciones y reprogramaciones masivas en Premium Long Haul  
A. Naturaleza de la causa  
   • Huelga en Italia derivó en reprogramaciones y cancelaciones concentradas en rutas clave de Premium LH, agravadas por limitaciones de avión y fallos de equipaje.  
B. Evidencia consolidada y alcance  
   • Segmento afectado: Premium Long Haul (Global/LH/Premium)  
     – NPS Día: 0,0 pts vs Baseline: 10,48 pts (–10,48 pts)  
     – Incidentes totales: 36 (17 retrasos, 12 cancelaciones, 7 otros)  
     – Rutas con mayor alteración: BRU-MAD, BCN-SCL  
     – Ruta sin incidentes pero baja satisfacción: EZE-MAD NPS 20,0 (n=5)  
   • Afecta a todos los subsegmentos Premium bajo LH (no hay distinción IB/YW en Premium LH).  
   • Métricas clave:  
     – Flota A333: NPS 50,0 (n=4) vs A350 next: NPS –16,7  
     – Región España: NPS –40,0 (n=5) vs América Sur: NPS 18,2 (n=11)  

3. Causa: Alta satisfacción de pasajeros Business Long Haul en vuelos sin incidencias  
A. Naturaleza de la causa  
   • Un pequeño grupo de rutas Long Haul Business estuvo libre de incidentes y generó valoraciones excepcionalmente altas que compensaron el resto.  
B. Evidencia consolidada y alcance  
   • Segmento afectado: Business Long Haul (Global/LH/Business)  
     – NPS Día: 44,44 pts vs Baseline: 25,82 pts (+18,62 pts)  
     – Incidentes totales: 36 (17 retrasos, 12 cancelaciones, 7 otros)  
     – Única ruta sin incidencias y con NPS 100: JFK-MAD (n=3)  
     – Reactividad promedio: 54,6 pts (amplia dispersión por flota y región)  
   • Afecta al conjunto de Business en Long Haul, sin separación por compañía.  
   • Métricas clave:  
     – Dispersión NPS por flota: 0–100 pts  
     – Dispersión NPS por región: 14,3–100 pts  

4. Causa: Elevada valoración de segmento Leisure y flota CRJ en SH Business YW  
A. Naturaleza de la causa  
   • Clientes Leisure, especialmente residentes en América Sur, con alta valoración de la flota CRJ en la ruta MAD-MLN, compensaron las disrupciones generales.  
B. Evidencia consolidada y alcance  
   • Segmento afectado: Business Short Haul / YW (Global/SH/Business/YW)  
     – NPS Día: 45,83 pts vs Baseline: 29,58 pts (+16,25 pts)  
     – Incidentes totales: 48 (mismas cancelaciones y retrasos de SH)  
     – Única ruta con encuestas y sin incidentes: MAD-MLN NPS 33,3 (n=3)  
     – NPS Leisure vs Business: 61,5 vs 27,3 pts  
     – Flota CRJ: NPS 47,6; ATR: NPS 33,3  
     – Región América Sur: NPS 100,0; España: NPS 36,4  
     – CodeShare IB: NPS 45,5  
   • Esta causa aplica específicamente a YW dentro de Business SH; el subsegmento IB muestra el efecto opuesto.  
   • Métricas clave:  
     – Reactividad promedio: 28,0 pts  
     – Dispersión amplia entre subgrupos (flota, región)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El análisis del árbol de anomalías NPS para el día 2025-09-25 detectó nueve disparos relevantes en el árbol jerárquico. A nivel Global, el NPS cayó levemente de 24.99 a 24.79 (–0.20 pts vs L7d), equilibrio de dos tendencias contrarias. Short Haul registró un descenso de 29.14 a 26.86 (–2.27 pts) motivado por la caída de Economy SH (27.35→24.65, –2.70 pts) y Business SH (44.38→42.86, –1.52 pts), con impactos extremos en IB: Economy SH IB bajó de 23.89 a 18.82 (–5.07 pts) y Business SH IB de 51.44 a 41.30 (–10.13 pts). En paralelo, Business SH YW rebotó de 29.58 a 45.83 (+16.25 pts). En Long Haul, Business LH subió de 25.82 a 44.44 (+18.62 pts) gracias a vuelos sin incidencias como JFK–MAD (NPS 100, n=3), mientras Premium LH se hundió de 10.48 a 0.00 (–10.48 pts) por cancelaciones y retrasos masivos.

Las rutas BRU–MAD y BCN–SCL concentraron la mayoría de los 48 incidentes en Short Haul y 36 en Premium LH, siendo epicentro de cancelaciones (22) y retrasos (20 en SH, 17 en LH). En SH, MAD–NCE mostró NPS 0.0 sin incidentes, y MAD–MLN alcanzó 33.3 sin disrupciones, apoyando la subida de Business SH YW. Los grupos más reactivos fueron los pasajeros IB en SH (flota A320 y code‐share I2 con NPS muy bajos) y los viajeros Leisure residentes en América Sur en Business SH YW, que elevaron su satisfacción hasta 61.5 pts.

ECONOMY SH: Impacto diferencial por compañía  
La cabina Economy de SH experimentó un deterioro de 27.35 a 24.65 (–2.70 pts vs L7d) durante el día 2025-09-25. Economy SH IB sufrió la mayor caída, de 23.89 a 18.82 (–5.07 pts), mientras Economy SH YW se mantuvo dentro de lo normal con un NPS de 36.53 (+2.60 pts vs L7d). El principal motor de este descenso fueron los 22 cancelaciones y 20 retrasos concentrados en rutas como BRU–MAD y BCN–SCL, particularmente en operaciones IB con aviones A320 y partners I2.

BUSINESS SH: Contraste de percepción IB vs YW  
El segmento Business de SH registró un ligero retroceso de 44.38 a 42.86 (–1.52 pts vs L7d). No obstante, Business SH IB cayó de 51.44 a 41.30 (–10.13 pts) por cancelaciones y retrasos en la ruta BRU–MAD, mientras Business SH YW ascendió de 29.58 a 45.83 (+16.25 pts) impulsada por la excelente valoración de pasajeros Leisure en vuelos CRJ sin incidencias, sobre todo en MAD–MLN. Los perfiles Leisure y residentes en América Sur mostraron la reactividad más positiva.

ECONOMY LH: Rendimiento estable  
La cabina Economy de LH mantuvo desempeño estable, con un NPS de 18.18 (vs L7d 17.04, +1.14 pts) durante el día 2025-09-25. No se detectaron cambios significativos, preservándose niveles consistentes de satisfacción en rutas regulares sin concentraciones críticas de incidentes.

BUSINESS LH: Alza impulsada por vuelos sin incidencia  
La cabina Business de LH mejoró notablemente, pasando de 25.82 a 44.44 (+18.62 pts vs L7d). El driver principal fueron los vuelos sin incidencias, en especial la ruta JFK–MAD con NPS 100 (n=3), que contrarrestaron 17 retrasos y 12 cancelaciones en otros tramos. La dispersión de valoración por flota y región sugiere que, aunque algunos clientes vivieron disrupciones, un núcleo de vuelos robustos elevó el promedio.

PREMIUM LH: Caída pronunciada por huelgas  
El segmento Premium de LH descendió de 10.48 a 0.00 (–10.48 pts vs L7d) debido a cancelaciones masivas y retrasos (17 y 12 respectivamente) motivados por la huelga en Italia, especialmente en BRU–MAD y BCN–SCL. Los pasajeros en A350 next y residentes en España fueron los más afectados (NPS –16.7 y –40.0), mientras la ruta EZE–MAD, sin incidencias, mostró un NPS de 20.0 (n=5).

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