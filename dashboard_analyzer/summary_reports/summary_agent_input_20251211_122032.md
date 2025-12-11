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

En Economy SH, el escenario es TRANSFERENCIA (IB Normal, YW – | SH –).  
- Narrativa: La anomalía negativa del nodo padre se explica íntegramente por el empeoramiento operacional del subsegmento YW, que contagia su efecto al agregado pese a que IB se mantuvo estable.  
- Evidencia Clave: YW –10.1 pts impulsado por Punctuality (SHAP = –5.550), caída de OTP15 en 1.45 pts, aumento de mishandling +1.20 y misconexiones +0.13 (operative_data_tool), con rutas críticas BCN-MLN y BCN-VLC y alta reactividad en perfiles CodeShare y Residence Region.

En Business SH, el escenario es DOMINANCIA (IB –, YW + | SH –).  
- Narrativa: Aunque YW registró una mejora significativa en Load factor, la fuerte caída de IB impuso el signo negativo al nodo padre. Adoptamos la explicación de IB, matizando que el avance de YW moderó parcialmente el impacto.  
- Evidencia Clave: IB –16.9 pts liderado por Punctuality (SHAP = –6.462), incremento de mishandling +2.1 y misconexiones +0.1 (operative_data_tool), principales rutas MAD-OPO y CDG-MAD, con perfiles más sensibles en Residence Region y CodeShare.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es SINERGIA (Economy –, Business – | SH –).  
- Narrativa: Ambas cabinas sufrieron un deterioro operativo que se reforzó mutuamente, trasladándose con fuerza al nivel Short Haul.  
- Evidencia: Punctuality (SHAP = –4.099), caída de OTP15 de 92.93 % a 92.49 %, mishandling +1.9 incidentes, misconexiones +0.1; rutas críticas BCN-MLN y BCN-VLC; alta sensibilidad en perfiles CodeShare y Fleet.

En Long Haul, la dinámica es DOMINANCIA (Economy N, Business –, Premium – | LH N).  
- Narrativa: La solidez de Economy, con un aumento de NPS de +3.2 pts sin anomalías, neutralizó las caídas en Business y Premium, manteniendo el nivel Long Haul en rango normal.  
- Evidencia: Economy LH estable (+3.2 pts vs baseline); Business LH –6.7 pts por Punctuality (SHAP = –4.255), OTP15 –5.7 pts, mishandling +1.9; Premium LH –1.4 pts por Punctuality (SHAP = –1.476), mismas tendencias de incidentes.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA (Largo Radio Normal, Corto Radio – | Global –).  
- Narrativa: La caída de NPS Global (–2.9 pts) se explica exclusivamente por el deterioro operativo en Short Haul, que contagia su efecto al conjunto de la red pese a la estabilidad de Long Haul.  
- Evidencia:  
  • Punctuality (SHAP = –4.099) en Short Haul  
  • OTP15 –0.44 pts (Short Haul)  
  • Mishandling +1.9 incidentes (Short Haul)  
  • Misconnections +0.1 incidentes (Short Haul)  
  • Rutas críticas: BCN-MLN (NPS –100.0) y BCN-VLC (NPS –28.6)  
  • Perfiles más sensibles: CodeShare y Fleet

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

CAUSA 1: Deterioro de puntualidad  
- Escenario: SINERGIA (Economy SH –, Business SH – | Short Haul –)  
- NMA: Short Haul  
- Afecta a:  
  • SH/Economy/YW  
  • SH/Business/IB  
- Qué falló: Puntuality (SHAP = –4.099 en Short Haul)  
- Dónde:  
  1. BCN-MLN: NPS –100.0, 1 pax (explanatory_drivers_tool)  
  2. BCN-VLC: NPS –28.6, 7 pax (explanatory_drivers_tool)  
  3. AMS-MAD: NPS 0.0, 4 pax (explanatory_drivers_tool)  
  4. BIO-VLC: NPS 0.0, 6 pax (explanatory_drivers_tool)  
  5. MLN-SVQ: NPS 20.0, 5 pax (explanatory_drivers_tool)  
- Quién:  
  • CodeShare: spread 166.2 pts (customer_profile_tool / Short Haul)  
  • Fleet: spread 115.4 pts (customer_profile_tool / Short Haul)  
  • Residence Region: spread 90.6 pts (customer_profile_tool / Short Haul)  
- Evidencia COMPLETA:  
  • NPS 33.6898 vs 36.5130 (Short Haul)  
  • OTP15 92.49% vs 92.93% (Short Haul)  
  • Punctuality SHAP = –4.099 (Short Haul)  

CAUSA 2: Incremento de incidencias NCS (mishandling y misconexiones y cancelaciones)  
- Escenario: SINERGIA (Economy SH –, Business SH – | Short Haul –)  
- NMA: Short Haul  
- Afecta a:  
  • SH/Economy/YW  
  • SH/Business/IB  
- Qué falló:  
  • Mishandling: 17.57 vs 15.68 (+1.89 incidentes) (operative_data_tool / Short Haul)  
  • Misconnections: 0.71 vs 0.58 (+0.13 incidentes) (operative_data_tool / Short Haul)  
  • Flight cancellations: +5 incidentes (ncs_tool / Short Haul)  
- Dónde: mismas rutas que en Puntuality (BCN-MLN, BCN-VLC, AMS-MAD, BIO-VLC, MLN-SVQ)  
- Quién: mismos perfiles (CodeShare 166.2 pts, Fleet 115.4 pts, Residence Region 90.6 pts en Short Haul)  
- Evidencia COMPLETA:  
  • NCS – Retrasos +22, Otras_incidencias +8, Desvíos +7, Cancelaciones +5, Limitación_aeronave +3 (ncs_tool / Short Haul)  

— Fin de consolidación de causas —

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

La red global registró una caída de NPS de –2.9 puntos, pasando de 30.38 a 27.46 (Global), impulsada por el deterioro en Short Haul pese al desempeño estable de Long Haul. Short Haul redujo su NPS de 36.51 a 33.69 (–2.8 pts, SH), mientras que Long Haul se mantuvo normal en 14.29 (+2.1 pts, LH) gracias a la fortaleza de Economy LH. Dentro de Short Haul, Economy SH mostró un NPS de 33.46 (–2.7 pts vs L7d) por la caída de YW de 42.97 a 32.88 (–10.1 pts), pese a que IB pasó de 32.87 a 33.75 (+0.9 pts). Business SH se situó en 36.04 (–5.9 pts vs L7d), dominado por IB que cayó de 51.90 a 34.98 (–16.9 pts), contrarrestando la mejora de YW de 20.00 a 38.75 (+18.8 pts). En Long Haul, Business LH experimentó un descenso de 27.45 a 20.71 (–6.7 pts) y Premium LH bajó de 22.81 a 21.43 (–1.4 pts), ambas impactadas por el deterioro de puntualidad (Punctuality –4.3 ppts según Explanatory Drivers en Business LH y –1.5 ppts en Premium LH), caídas de OTP, aumento de mishandling y misconexiones, mientras Economy LH mantuvo estabilidad con un alza de 3.2 pts (9.49 → 12.64).

Las rutas más afectadas reflejan estas tendencias: en SH/Economy YW, BCN-MLN (NPS –100.0, 1 pax) y BCN-VLC (–28.6, 7 pax) concentran los peores descensos; en SH/Business IB destacan cancelaciones recurrentes en MCO-MAD y HAV-MAD; en LH Business, MAD-MIA brilló al pasar de 7 pax con NPS 42.9; en Premium LH, GRU-MAD (NPS 25.0, 8 pax) y MAD-SCL (–15.4, 13 pax) fueron los nodos críticos. Los perfiles más reactivos fueron CodeShare (spread 166.2 pts en SH/Economy, 166.7 pts en SH/Business YW), Fleet (115.4 pts en SH, 105.5 pts en LH Business) y Residence Region (90.6 pts en SH, 270.1 pts en LH Premium), evidenciando alta sensibilidad a la puntualidad y al manejo de equipaje.

ECONOMY SH YW: Impacto operativa  
La cabina Economy SH YW exhibió un descenso de NPS de 42.97 a 32.88 (–10.1 pts vs L7d), debido a un empeoramiento de puntualidad (Punctuality –5.6 ppts según Explanatory Drivers), caída de OTP en 1.45 pts (datos operativos), +1.20 mishandling y +0.13 misconexiones (datos operativos), y +45 incidentes NCS (retrasos +22, otras_incidencias +8, desvíos +7, cancelaciones +5, limitación_aeronave +3). Rutas críticas: BCN-MLN, BCN-VLC, AMS-MAD; perfiles más sensibles: CodeShare, Fleet, Residence Region.

ECONOMY SH IB: Rendimiento estable  
La cabina Economy SH IB mantuvo desempeño estable con un NPS de 33.75 (vs 32.87, +0.9 pts vs L7d). No se detectaron cambios significativos en puntualidad, mishandling o feedback de clientes, manteniendo consistencia de satisfacción.

BUSINESS SH IB: Deterioro extremo  
Business SH IB cayó de 51.90 a 34.98 (–16.9 pts vs L7d) por un fuerte deterioro de puntualidad (Punctuality –6.5 ppts según Explanatory Drivers), mishandling +2.10 y misconexiones +0.10 (datos operativos), y +45 incidentes NCS (42 → 64 retrasos, +22; +5 cancelaciones). Rutas clave: MAD-OPO, CDG-MAD, LIS-MAD; perfiles más reactivos: Residence Region, CodeShare, Fleet.

BUSINESS SH YW: Mejora destacada  
Business SH YW registró un salto de NPS de 20.00 a 38.75 (+18.8 pts vs L7d), motivado por la reducción de load factor en 1.6 pts (datos operativos) que mejoró confort y espacio (load factor +1.95 ppts según Explanatory Drivers). Rutas: MCO-MAD, HAV-MAD, VGO-MAD; perfiles más sensibles: Residence Region, CodeShare.

ECONOMY LH: Estabilidad positiva  
Economy LH mantuvo desempeño estable con un NPS de 12.64 (vs 9.49, +3.2 pts vs L7d). No se detectaron cambios significativos en drivers operativos, load factor o feedback de clientes, sosteniendo niveles de satisfacción consistentes.

BUSINESS LH: Caída por puntualidad  
Business LH disminuyó de 27.45 a 20.71 (–6.7 pts vs L7d) principalmente por Punctuality –4.3 ppts según Explanatory Drivers, OTP –5.7 pts (datos operativos), mishandling +1.9 y misconexiones +0.10, y +17 retrasos y +9 otras_incidencias en NCS. Ruta principal: MAD-MIA; perfiles clave: Fleet, Residence Region, CodeShare.

PREMIUM LH: Leve deterioro  
Premium LH bajó de 22.81 a 21.43 (–1.4 pts vs L7d) por Punctuality –1.5 ppts según Explanatory Drivers, OTP –5.7 pts, mishandling +1.9, misconexiones +0.10 y +17 retrasos en NCS. Rutas: GRU-MAD, MAD-SCL, EZE-MAD; perfiles más afectados: Residence Region, CodeShare.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 5
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*



**ANÁLISIS DIARIO SINGLE:**


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