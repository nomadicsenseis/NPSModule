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

En Economy SH, el escenario es SINERGIA (IB Negativa, YW Negativa | Padre Negativa).  
- Narrativa: Ambos subgrupos sufrieron la misma caída de NPS y la presión operativa se suma de forma homogénea al nodo padre. La caída de –6.2 pts en Economy SH se explica por el aumento de problemas de equipaje y conexiones, corroborado tanto en IB como en YW.  
- Evidencia Clave: operative_data_tool muestra mishandling +16 incidentes y desvíos/misconexiones +28; ncs_tool registra retrasos +12.  

En Business SH, el escenario es DILUCIÓN (IB Normal, YW Positiva | Padre Normal).  
- Narrativa: La fuerte alza de +23 pts en YW, impulsada por drivers de producto y servicio, no alcanza a imponer esa variación al agregado porque IB se mantiene estable, resultando en una variación normal de +6.8 pts para Business SH.  
- Evidencia Clave: YW: Aircraft interior SHAP=14.123 y Arrivals experience SHAP=4.727; IB: NPS 47.6 vs 47.2 (sin cambio significativo).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En SH, la dinámica es TRANSFERENCIA (Economy Negativa, Business Normal | SH Negativa).  
- Narrativa: La caída de –6.2 pts en Economy SH, motivada por el aumento de mishandling y misconexiones y cancelaciones de vuelo, arrastró al radio SH completo, aun cuando Business se mantuvo estable.  
- Evidencia: operative_data_tool registra mishandling +3.5 y misconex +0.2; ncs_tool señala incremento en flight cancellations.  

En LH, la dinámica es DOMINANCIA (Economy Negativa, Business Negativa, Premium Positiva | LH Negativa).  
- Narrativa: El desempeño de LH está dictado por la fuerte caída en Economy (–10.4 pts) debido a la reducción de OTP15 (–4.0 pp) y al alza de mishandling (+3.5) y misconexiones (+0.2); este efecto negativo prevaleció sobre el repunte en Premium.  
- Evidencia: operative_data_tool en Economy muestra OTP15 81.06→77.09 (–4.0), mishandling 13.67→17.14 (+3.5) y misconex 0.47→0.63 (+0.2).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es SINERGIA (LH Negativa, SH Negativa | Global Negativa).  
- **Narrativa:** La red entera se vio impactada por un deterioro sistémico en la puntualidad y un alza de incidencias operativas, que arrastró hacia abajo tanto el Largo como el Corto Radio.  
- **Evidencia:** OTP15 cayó 0.5 pts; mishandling subió 3.5 pts; misconexiones +0.2 pts; ncs_tool reportó +205 incidentes con incremento de flight cancellations.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

Causa 1: Deterioro de la puntualidad operativa (retrasos y cancelaciones)  
A. NATURALEZA DE LA CAUSA  
• Hipótesis: Fallos en la gestión de operaciones (turnarounds, planificación de tripulaciones y slots) provocaron retrasos encadenados y un aumento de cancelaciones, erosionando la percepción de fiabilidad.  

B. EVIDENCIA CONSOLIDADA Y ALCANCE  
• Segmento «Global» (el mayor volumen de clientes):  
  – NPS actual 22.57 vs baseline 30.12 (∆ –7.54 pts).  
  – OTP15 cayó 0.5 pp vs L7d.  
  – ncs_tool: 2 416 incidentes operativos totales (+205), con incrementos en retrasos, limitación de aeronave y flight cancellations.  
  – Rutas críticas:  
     • BIO–VLC: NPS –50.0 (4 pax)  
     • MAD–OSL: NPS –37.5 (8 pax)  
     • GVA–MAD: NPS –25.0 (4 pax)  
  – Verbatim representativo: “Nuestro vuelo salió con tres horas de retraso y cancelaron la conexión sin aviso.”  
• Alcance: afecta a todos los subsegmentos bajo «Global» (LH, SH, Economy, Business, Premium).  

—  

Causa 2: Incremento en incidentes de manipulación de equipaje (mishandling)  
A. NATURALEZA DE LA CAUSA  
• Hipótesis: Picos de tráfico no acompañados de refuerzo de personal y equipamiento en tierra generaron más equipajes dañados, retrasados o extraviados.  

B. EVIDENCIA CONSOLIDADA Y ALCANCE  
• Segmento «Global»:  
  – NPS actual 22.57 vs baseline 30.12.  
  – operative_data_tool: mishandling subió de 13.67 a 17.14 (+3.47 pts).  
  – ncs_tool: OTRAS_INCIDENCIAS +16 vs L7d.  
  – Rutas más impactadas por quejas de equipaje:  
     • IAD–MAD: NPS –50.0 (8 pax)  
     • MAD–ORD: NPS –57.7 (26 pax)  
  – Verbatim representativo: “Mi maleta llegó rota y tardó días en aparecer.”  
• Alcance: presente en todos los subsegmentos de «Global», con especial incidencia en Economy LH y SH.  

—  

Causa 3: Aumento de pérdida de conexiones y reprogramaciones (misconnections)  
A. NATURALEZA DE LA CAUSA  
• Hipótesis: Reajustes de flota y cancelaciones en ejes hubs obligaron a rebookings masivos, generando frustración por escalas perdidas.  

B. EVIDENCIA CONSOLIDADA Y ALCANCE  
• Segmento «Global»:  
  – NPS actual 22.57 vs baseline 30.12.  
  – operative_data_tool: misconexiones subieron de 0.47 a 0.63 (+0.16 pts).  
  – ncs_tool: pérdida de conexiones +63 incidentes vs L7d; resequencing/re-book +22; maintenance failures +23.  
  – Rutas con mayor impacto por misconexiones:  
     • MAD–ORD: NPS –57.7 (26 pax)  
     • JFK–MAD: NPS –44.0 (50 pax)  
     • GUA–MAD: NPS –17.6 (17 pax)  
  – Verbatim representativo: “Perdí mi conexión y me cambiaron tres veces de puerta sin explicación.”  
• Alcance: afecta transversalmente a todos los subsegmentos de «Global», con especial gravedad en Economy y Business LH.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

Durante la semana del 29 de noviembre al 2 de diciembre de 2025, el NPS global cayó de 30.12 a 22.57 (–7.54 pts), arrastrado por descensos de 10.37 pts en Global/LH (de 17.21 a 6.84) y de 4.99 pts en Global/SH (de 36.47 a 31.48). En Global/SH/Economy, la satisfacción pasó de 36.17 a 30.01 (–6.16 pts), con SH IB de 33.30 a 27.75 (–5.55 pts) y SH YW de 42.07 a 34.62 (–7.45 pts). Global/LH/Economy sufrió un hundimiento de 13.63 a 3.19 (–10.43 pts) y Global/LH/Business cayó de 32.70 a 12.50 (–20.20 pts), mientras que Global/LH/Premium repuntó de 29.60 a 36.62 (+7.02 pts). Estas bajadas se explican por un deterioro sistémico en la puntualidad (OTP15 –4.0 pp), un alza de mishandling (+3.47 incidentes) y misconexiones (+0.16 incidentes), y un incremento de 205 incidentes operativos NCS, con especial incidencia en cancelaciones y retrasos.  

Las rutas más afectadas incluyen BIO–VLC (NPS –50.0), MAD–ORD (–57.7 en Economy LH; –28.6 en Business LH), IAD–MAD (–50.0 en LH/Economy) y JFK–MAD (–44.0 en LH/Economy). Los viajeros por Residence Region (spread de hasta 142.0 pts en SH/Economy) y en CodeShare (spread 133.9 pts en SH/Economy) fueron los perfiles más reactivos, seguidos por flota y tipo de viaje.

ECONOMY SH YW e IB – Caída generalizada por equipaje y conexiones  
La cabina Global/SH/Economy registró un NPS de 30.01 (vs L7d 36.17, –6.16 pts). SH IB bajó de 33.30 a 27.75 (–5.55 pts) y SH YW de 42.07 a 34.62 (–7.45 pts). El principal factor fue el aumento de mishandling (+3.47 incidentes) y misconexiones (+0.16 incidentes), junto con 12 retrasos adicionales reportados por NCS, que erosionaron la percepción de fiabilidad y manejo de equipaje. Los impactos más severos se dieron en BIO–VLC (–50.0), MAD–OSL (–37.5) y GVA–MAD (–25.0). Los viajeros más sensibles fueron los segmentados por Residence Region (spread 142.0 pts) y CodeShare (spread 133.9 pts).

BUSINESS SH – Desempeño estable a nivel agregado  
El segmento Global/SH/Business mantuvo un NPS de 46.31 (vs L7d 39.54, +6.8 pts), dentro de la variación normal. SH IB se mantuvo estable en 47.57 (vs L7d 47.25, +0.33 pts), mientras que SH YW trepó de 20.45 a 43.48 (+23.02 pts) por drivers de producto como Aircraft interior (SHAP +14.123) y Arrivals experience (SHAP +4.727). Este repunte de YW se diluyó en el agregado debido al volumen superior de IB, resultando en un comportamiento estable para Business SH.

ECONOMY LH – Fuerte deterioro por puntualidad y equipaje  
La cabina Global/LH/Economy cayó de 13.63 a 3.19 (–10.43 pts vs L7d). El descenso se explica por una puntualidad muy deficiente (OTP15 81.06 %→77.09 %, –4.0 pp), mishandling +3.47 incidentes y misconexiones +0.16 incidentes, respaldado por 70 incidentes NCS adicionales (retrasos, limitaciones de aeronave y cancellations +3). Las rutas con mayor impacto fueron IAD–MAD (–71.4), MAD–ORD (–64.7) y JFK–MAD (–46.5). Los más sensibles fueron los viajeros según Fleet (spread 113.2 pts) y CodeShare (98.2 pts).

BUSINESS LH – Máxima caída por puntualidad y arrivals experience  
Global/LH/Business descendió de 32.70 a 12.50 (–20.20 pts vs L7d). La causa principal fue la fuerte incidencia en puntualidad (SHAP –8.189) y arrivals experience (SHAP –4.253), además de boarding (–3.621) y cabin crew (–2.838), con OTP15 –4.0 pp, mishandling +3.47 y misconex +0.16. Las rutas críticas fueron MAD–ORD (–28.6) y JFK–MAD (–16.7). Residence Region (spread 123.5 pts) y CodeShare (103.7 pts) fueron los perfiles más reactivos.

PREMIUM LH – Mejora impulsada por producto a pesar de incidencias  
Global/LH/Premium subió de 29.60 a 36.62 (+7.02 pts vs L7d). Los drivers dominantes fueron Aircraft interior (SHAP +14.123) y Arrivals experience (SHAP +4.727), validados en explanatory_drivers. Aun con un incremento de 203 incidentes NCS (otras_incidencias +203, retrasos +50, cancelaciones +19), la fortaleza del servicio Premium compensó deficiencias operativas. Destacan las rutas MAD–MIA (40.0) y MAD–MCO (33.3), y los perfiles más reactivos fueron CodeShare (141.7 pts) y Residence Region (104.6 pts).

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

En Economy SH, el escenario es SINERGIA `(-, - | -)`.  
- Narrativa: La anomalía negativa se origina en ambos subsegmentos (IB y YW) que presentan caídas, y se suman para generar el descenso en Economy SH.  
- Evidencia Clave: Ruta EAS-MAD con NPS 0.0, pasajeros en flota A333 con NPS –100.0 y CodeShare AA con NPS –55.6.

En Business SH, el escenario es DOMINANCIA `(-, + | +)`.  
- Narrativa: El alza del NPS en Business SH está impulsada principalmente por YW, cuyas valoraciones muy positivas imponen el signo al padre, aunque el efecto fue parcialmente suavizado por IB.  
- Evidencia Clave: YW alcanzó NPS 76.92 gracias a un Load Factor de 54.43 y feedback unánimemente positivo; IB cayó a NPS 45.83 por bajas puntuaciones de clientes europeos en A320 (NPS 0.0).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En LH, la dinámica es DOMINANCIA `(+, -, + | +)`.  
- Narrativa: El alza de 7.2 pts en Long Haul se explica principalmente por el fuerte rendimiento de Premium, donde los vuelos en A350 y la ruta MAD–MEX obtuvieron NPS unánimes de 100, impulsando el radio completo. La caída en Business (–11.1 pts por insatisfacción en A350 next, residentes en América Norte/Europa y code-share AA) limitó el potencial de crecimiento, pero no impidió que Premium arrastrara el resultado global.  
- Evidencia: Premium: NPS 47.8 vs 34.8, elogios unánimes al confort del A350 y servicio en MAD–MEX; Business: NPS 12.9 vs 24.0, críticas de viajeros de negocio en A350 next y socios AA.

En SH, la dinámica es DOMINANCIA `(-, + | -)`.  
- Narrativa: La caída de 7.7 pts en Short Haul responde sobre todo a Economy, donde un puñado de valoraciones muy negativas en la ruta EAS–MAD, flota A333 (NPS –100) y CodeShare AA (–55.6) arrastraron el índice. El fuerte repunte en Business (especialmente YW con NPS 76.9 gracias a baja ocupación y verbatims muy positivos) atenuó la pérdida, pero no logró revertirla.  
- Evidencia: Economy: NPS 21.6 vs 32.1, EAS–MAD NPS 0.0 y A333 con calificaciones extremas; Business YW: NPS 76.9 vs 33.4, Load Factor 54.4 y feedback unánime positivo.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es DOMINANCIA `(+, - | -)`.  
- Narrativa: El NPS global está arrastrado por el deterioro en Short Haul, pese al buen desempeño de Long Haul. La caída en Economy SH, con valoraciones extremadamente bajas en la ruta EAS–MAD y en flota A333/code-share AA, impuso la anomalía negativa al total de la red.  
- Evidencia:  
  • Economy SH: NPS 21.64 vs baseline 32.09, ruta EAS–MAD con NPS 0.0 (6 encuestas)  
  • Flota A333 en SH: NPS –100.0 y CodeShare AA: NPS –55.6 criticó la experiencia en corto radio.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

1. Causa: Valoraciones negativas extremas en Economy Short Haul  
A. NATURALEZA DE LA CAUSA  
   • Hipótesis: Concentración de detractores aislados en vuelos cortos por una combinación de factores puntuales en la ruta (EAS–MAD), configuración de la cabina A333 y discrepancias en el servicio de CodeShare AA. Aunque la operación fue puntual y limpia, estos “outliers” sesgan la media y arrastran el NPS.  

B. EVIDENCIA CONSOLIDADA Y ALCANCE  
   • Segmento “más grande” afectado: Global/SH/Economy/IB (295 verbatims). Abarca todo el nodo Global/SH/Economy, incluyendo IB y YW.  
   • Detalle IB:  
     – NPS: 23.53 vs baseline 29.65 (–6.12 pts)  
     – Load Factor: 89.07 (–0.61 pts vs media)  
     – OTP15_adjusted: 93.25 (+1.85 pts vs media)  
     – Ruta principal: LIS–MAD, NPS 16.7 (n=6)  
     – Flota: A321 NPS 16.7 (n=42) / A319 NPS –26.7 (n=3) / A333 NPS –100.0 (n=2)  
     – Verbátim representativo: “Espacio muy justo en el A333 y la conexión con AA resultó caótica.”  
   • Este patrón de outliers en rutas y flotas similares (EAS–MAD, A333, AA) se repite también en YW, afectando todo Global/SH/Economy.  

2. Causa: Calidad sobresaliente del producto Premium en Long Haul  
A. NATURALEZA DE LA CAUSA  
   • Hipótesis: El Airbus A350-900, combinado con un servicio de tripulación muy valorado (puntualidad, atención a movilidad reducida), genera un nivel de satisfacción extraordinario que empuja al alza el NPS.  

B. EVIDENCIA CONSOLIDADA Y ALCANCE  
   • Segmento “más grande” afectado: Global/LH/Premium (39 verbatims). Impacta la totalidad de Premium en largo radio.  
   • Detalle Premium:  
     – NPS: 47.83 vs baseline 34.81 (+13.02 pts)  
     – Load Factor: 88.33 (–2.30 pts vs media)  
     – OTP15_adjusted: 76.06 (–2.84 pts vs media)  
     – Ruta clave: MAD–MEX, NPS 100.0 (n=5)  
     – Flota: A350 NPS 100.0 (n=5) / A333 NPS –40.0 (n=5)  
     – Verbátim representativo: “El A350 es un avión de otro nivel y la tripulación estuvo impecable.”  
   • La excelencia observada en A350/Premium es consistente en todas las rutas y subnodos de Global/LH/Premium.  

3. Causa: Discrepancia de servicio en Business Long Haul  
A. NATURALEZA DE LA CAUSA  
   • Hipótesis: Altas expectativas de viajeros de negocio no cumplidas en flota A350 next y fallas de coordinación con CodeShare AA generan detractores, pese a la operación estable.  

B. EVIDENCIA CONSOLIDADA Y ALCANCE  
   • Segmento “más grande” afectado: Global/LH/Business (57 verbatims). Abarca todo Business en largo radio.  
   • Detalle Business LH:  
     – NPS: 12.90 vs baseline 24.03 (–11.13 pts)  
     – Load Factor: 93.83 (–0.21 pts vs media)  
     – OTP15_adjusted: 76.06 (–2.84 pts vs media)  
     – Ruta analizada: LIM–MAD, NPS 100.0 (n=3)  
     – Flota: A350 next NPS –25.0 (n=4) / Unknown NPS –33.3 (n=3)  
     – Residence Region América Norte: NPS –50.0 (n=4); CodeShare AA: NPS –33.3 (n=3)  
     – Verbátim representativo: “El A350 next no está a la altura de lo que prometen para Business y en el tramo AA faltó cohesión.”  
   • Estas quejas subyacen en todos los subnodos de Global/LH/Business.  

4. Causa: Confort extremo y atención en Business Short Haul YW  
A. NATURALEZA DE LA CAUSA  
   • Hipótesis: Factor de carga muy bajo (54.43) permite mayor espacio y servicio personalizado; la puntualidad reforzó la percepción de excelencia.  

B. EVIDENCIA CONSOLIDADA Y ALCANCE  
   • Segmento “más grande” afectado: Global/SH/Business/YW (15 verbatims). Se replica en todo el subnodo YW de Business SH.  
   • Detalle YW SH:  
     – NPS: 76.92 vs baseline 33.36 (+43.56 pts)  
     – Load Factor: 54.43 (–3.88 pts vs media)  
     – OTP15_adjusted: 91.19 (+3.38 pts vs media)  
     – Verbátims: unanimidad de elogios a servicio, comida y puntualidad  
     – No hay rutas con muestra suficiente para identificar un punto único  
   • Este desempeño sobresaliente caracteriza a todo Global/SH/Business/YW.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El NPS global cayó de 26.01 a 23.46 (–2.55 pts vs media últimos 7 días) como reflejo de la divergencia entre Long Haul y Short Haul. En Long Haul (Global/LH) el indicador subió de 12.51 a 19.66 (+7.16 pts) gracias al impulso de Premium LH, mientras que en Short Haul (Global/SH) se perdió de 33.14 a 25.44 (–7.70 pts). Dentro de Short Haul, la cabina Economy (Global/SH/Economy) retrocedió de 32.09 a 21.64 (–10.45 pts) por la sumatoria de caídas en IB (23.53 vs 29.65, –6.12 pts) y YW (17.82 vs 37.11, –19.28 pts), originadas en la ruta EAS–MAD y en flota A333/CodeShare AA. En ese mismo radio, Business (Global/SH/Business) escaló de 43.53 a 56.76 (+13.23 pts), impulsado por YW (76.92 vs 33.36, +43.57 pts) pese a la leve bajada de IB (45.83 vs 48.00, –2.17 pts). En Long Haul Economy (Global/LH/Economy) subió de 8.94 a 16.13 (+7.19 pts) por comentarios muy positivos sobre tripulación y limpieza, mientras que Business LH (Global/LH/Business) retrocedió de 24.03 a 12.90 (–11.13 pts) debido a quejas de viajeros de negocio en A350 next y CodeShare AA. Premium LH (Global/LH/Premium) registró un salto de 34.81 a 47.83 (+13.02 pts) soportado en vuelos A350-900 en ruta MAD–MEX.

Las rutas más afectadas fueron EAS–MAD en Economy SH (NPS 0.0, n=6) y LIS–MAD en Economy IB SH (16.7, n=6), mientras que en Premium LH la ruta MAD–MEX alcanzó un NPS de 100.0 (n=5). En Business LH, los usuarios de A350 next en CodeShare AA residentes en América Norte registraron las puntuaciones más bajas. Los clientes más reactivos incluyen pasajeros de CodeShare AA y residentes en América Norte/Oriente Medio en Economy SH, así como viajeros de negocio en A350 next en Business LH.

ECONOMY SH (Global/SH/Economy)  
La cabina Economy de SH el día 2025-12-02 registró un NPS de 21.64 puntos, con una variación de –10.45 puntos vs media últimos 7 días. La causa principal fue la convergencia de caídas en sus subnodos: IB retrocedió 6.12 puntos (23.53 vs 29.65) y YW perdió 19.28 puntos (17.82 vs 37.11), derivadas de insatisfacción en la ruta EAS–MAD (NPS 0.0 en 6 encuestas) y percepciones negativas en flota A333 (NPS –100.0) y CodeShare AA (NPS –55.6).

BUSINESS SH (Global/SH/Business)  
El segmento Business de SH el día 2025-12-02 registró un NPS de 56.76 puntos, con una mejora de +13.23 puntos vs media últimos 7 días. Esta subida se explica por el repunte de YW (+43.57 pts, 76.92 vs 33.36), favorecido por un Load Factor bajo (54.43), alta puntualidad (OTP15 91.19) y verbatims elogiosos, mitigada ligeramente por IB (–2.17 pts, 45.83 vs 48.00) por bajas valoraciones de clientes europeos en A320.

ECONOMY LH (Global/LH/Economy)  
La cabina Economy de LH el día 2025-12-02 obtuvo un NPS de 16.13 puntos, con un alza de +7.19 puntos vs media últimos 7 días. La mejora se sustentó en feedback positivos sobre la amabilidad de la tripulación, limpieza de cabina y puntualidad, a pesar de ligeras caídas en Load Factor (87.74, –2.54) y OTP15 (76.06, –2.84). Fue especialmente notoria en la ruta MAD–SDQ (33.3, n=3) y entre residentes de América Centro.

BUSINESS LH (Global/LH/Business)  
La cabina Business de LH el día 2025-12-02 registró un NPS de 12.90 puntos, con un retroceso de –11.13 puntos vs media últimos 7 días. El descenso responde a insatisfacción de viajeros de negocios en flota A350 next (NPS –25.0) y CodeShare AA (–33.3), así como residentes en Norteamérica (–50.0) y Europa (–42.9), pese a la ruta LIM–MAD que mantuvo un 100.0 (n=3).

PREMIUM LH (Global/LH/Premium)  
El segmento Premium de LH el día 2025-12-02 registró un NPS de 47.83 puntos, con una mejora de +13.02 puntos vs media últimos 7 días. El alza estuvo dominada por vuelos en Airbus A350-900, que lograron NPS 100.0 (n=5) en la ruta MAD–MEX gracias al confort y profesionalidad de la tripulación, aunque la flota A333 anotó –40.0 (n=5).

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

En Economy SH, el escenario es SINERGIA `(-, - | -)`.  
- Narrativa: Adopta la explicación del nodo padre. La caída de –8.7 pts en Economy SH obedece a la insatisfacción concentrada en la ruta MAD–ZRH, especialmente entre pasajeros de negocio de Norteamérica y billetes en code-share AA. No se registraron incidencias operativas formales, de modo que los “soft factors” (expectativas de comodidad en flota 32S y perfil de viajeros de negocio) impulsaron el NPS a la baja.  
- Evidencia Clave: Ruta MAD–ZRH con NPS 0.0, pasajeros de negocio y América Norte (NPS –37.5), code-share AA (NPS –60.0).

En Business SH, el escenario es DOMINANCIA `(-, + | -)`.  
- Narrativa: Adopta la explicación del hijo dominante (IB), añadiendo que el impulso positivo de YW suavizó parcialmente el resultado. La caída de –21.7 pts en IB SH se explica por la insatisfacción de viajeros de negocio en la ruta MAD–ORY, sin incidencias operativas, por brecha entre expectativas de servicio y la experiencia real en A320neo.  
- Evidencia Clave: Ruta MAD–ORY (NPS 25.0, n=4), pasajeros de negocio (NPS –28.6) en IB SH.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es SINERGIA `(- Economy SH, - Business SH | - SH)`.  
- Narrativa: La caída de –8.7 pts en Short Haul obedece a un problema común en ambas cabinas: la insatisfacción de viajeros Business/Work en la ruta BRU–MAD (especialmente bajo código compartido AA e I2 y procedentes de Europa y Norteamérica). A pesar de la puntualidad y la actitud general positiva de la tripulación, “soft factors” como catering, espacio y proceso de embarque en esa ruta arrastraron el NPS global de SH.  
- Evidencia: BRU–MAD con NPS 0.0 (n=6), Business/Work SH: NPS 8.9, Code-share AA e I2: NPS –33.3, Residencia Europa/Norteamérica.

En Long Haul, la dinámica es DOMINANCIA `(- Economy LH, - Business LH, + Premium LH | - LH)`.  
- Narrativa: El NPS de –3.6 pts en Long Haul está dictado por la fuerte caída en Business LH, debido a la percepción negativa del servicio a bordo (actitud de cabina, calidad del catering y entretenimiento) en la ruta MAD–MEX. El alza en Premium LH (+27.7 pts) mitigó parcialmente ese impacto, pero no revirtió la baja general.  
- Evidencia: Global/LH/Business en MAD–MEX: NPS –5.0 (n=24), drivers: actitud de cabina, catering y entretenimiento.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es SINERGIA `(- LH, - SH | - Global)`.  
- **Narrativa:** La red entera se vio impactada por la mala experiencia de business travelers europeos en la ruta LHR–MAD, operada principalmente con aviones A321XLR y A350 C, lo que arrastró el NPS global a la baja.  
- **Evidencia:** Ruta LHR–MAD con NPS –26.3 (n=19); business travelers en esa ruta: NPS 5.5; flotas A321XLR (NPS –40.0) y A350 C (NPS –37.5).

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

A continuación presento las cuatro causas raíz detectadas, con su naturaleza y evidencia consolidada:

1) Causa: Mala experiencia de business travelers en la ruta LHR–MAD  
A. Naturaleza de la causa  
  • Driver “soft”: expectativas de servicio premium no cubiertas (configuración de cabina, amenities) en flotas A321XLR y A350 C.  
B. Evidencia consolidada y alcance  
  • Segmento más grande: Global (todos los pasajeros de LHR–MAD).  
  • NPS ruta LHR–MAD: –26.3 pts (n=19) vs baseline no disponible local, aporte neto al Global de –7.52 pts.  
  • Perfiles afectados: Business travelers (NPS 5.5), Leisure (23.4).  
  • Flotas implicadas: A321XLR (NPS –40.0), A350 C (–37.5).  
  • Operacionales: Load Factor 85.88 (–0.89 pts vs media), OTP15 90.45 (+2.38), Mishandling 0.83 (–0.24).  
  • Verbatims representativos (775 totales): “Cabina demasiado cerrada”, “amenities insuficientes para Business”.  
  • Alcance: afecta a todos los subsegmentos de la ruta LHR–MAD, contribuyendo al descenso global de –7.52 pts.

2) Causa: Percepción negativa del servicio a bordo en Business Long Haul (ruta MAD–MEX)  
A. Naturaleza de la causa  
  • Drivers “soft”: actitud de la tripulación con desgana, catering deficiente y oferta de entretenimiento insuficiente.  
B. Evidencia consolidada y alcance  
  • Segmento más grande: Global / LH / Business.  
  • NPS observado: –5.0 vs baseline 24.03 (caída de –29.03 pts).  
  • Ruta afectada: MAD–MEX (n=3, NPS 33.3 aportó negatividad al LH).  
  • Perfil: ocio (NPS –15.4), Business/Work (–14.3); región Europa (–100), Norteamérica (–60); flota A333 (–50).  
  • Operacionales: Load Factor 93.66 (–0.32), OTP15 77.96 (–1.12).  
  • Verbatims (24): “Tripulación poco atenta”, “comida fría”, “sin entretenimiento decente”.  
  • Alcance: cubre a todos los pasajeros de Business LH, arrastrando el NPS Long Haul –3.6 pts.

3) Causa: Insatisfacción de clientes Economy Short Haul en ruta MAD–ZRH  
A. Naturaleza de la causa  
  • Drivers “soft”: percepción de cabina menos confortable (flota 32S), expectativas de occupancy y servicios en code-share AA no cubiertas.  
B. Evidencia consolidada y alcance  
  • Segmento más grande: Global / SH / Economy.  
  • NPS del día: 23.41 vs baseline 32.09 (caída de –8.67 pts).  
  • Ruta crítica: MAD–ZRH (n=4, NPS 0.0).  
  • Perfiles: Norteamérica (–37.5), código compartido AA (–60.0), Business (9.0), Leisure (30.7).  
  • Operacionales: Load Factor 86.46 (–0.14), OTP15 92.30 (+2.87).  
  • Verbatims (verbatim_tool): predominio de comentarios positivos sobre puntualidad pero menciones a espacio reducido.  
  • Alcance: impacta a todos los subsegmentos de Economy SH (IB y YW incluidos), causando –8.7 pts en SH Economy.

4) Causa: Insatisfacción de viajeros Business Short Haul IB en ruta MAD–ORY  
A. Naturaleza de la causa  
  • Drivers “soft”: configuración de cabina, amenities insuficientes en A320neo, expectativas de servicio no cumplidas.  
B. Evidencia consolidada y alcance  
  • Segmento más grande: Global / SH / Business / IB.  
  • NPS observado: 25.0 pts vs baseline 48.00 (caída de –21.69 pts).  
  • Ruta clave: MAD–ORY (n=4, NPS 25.0).  
  • Perfil: Business/Work (8.3), Leisure (56.2), residentes en España (0.0), flota A320neo (0.0).  
  • Operacionales: Load Factor 70.08 (–2.65), OTP15 92.30 (+2.87).  
  • Verbatims (37): elogios a confort y tripulación, ausencia de quejas operativas — indica brecha de expectativas de servicio.  
  • Alcance: engloba a todos los pasajeros Business IB SH, explicando gran parte del –7.8 pts en Business SH.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El NPS global sufrió una caída de 7.52 puntos, pasando de 26.01 a 18.49 el 2025-12-01, como resultado de un debilitamiento tanto en Long Haul (–3.64 pts, de 12.51 a 8.87) como en Short Haul (–8.68 pts, de 33.14 a 24.46). En Long Haul, el factor más determinante fue el colapso de Business LH, que descendió 29.03 puntos (de 24.03 a –5.0) por percepciones negativas de servicio a bordo (actitud de la tripulación, catering y entretenimiento) en MAD–MEX, mientras que Premium LH se recuperó con un alza de 27.69 puntos (de 34.81 a 62.5) gracias a la excelencia en puntualidad y atención en cabina. En Short Haul, Economy SH retrocedió 8.68 puntos (de 32.09 a 23.41) por quejas de confort y espacio en MAD–ZRH, y Business SH cayó 7.81 puntos (de 43.53 a 35.71) dominado por la performance de IB (–21.69 pts, de 48.00 a 26.32) pese al fuerte empujón de YW (+22.20 pts, de 33.36 a 55.56).

Las rutas más impactadas fueron LHR–MAD (NPS –26.3, n=19) por insatisfacción de viajeros Business en A321XLR/A350 C; MAD–MEX, donde Business LH criticó catering y entretenimiento; MAD–ZRH, con un NPS 0.0 en Economy SH; y MAD–ORY, con NPS 25.0 en Business IB SH. Los grupos más reactivos fueron los business travelers europeos y norteamericanos, pasajeros en código compartido AA y usuarios de flotas A321XLR, A350 C y A320neo.

ECONOMY SH: Caída sinérgica en IB y YW  
En Global/SH/Economy, IB registró un NPS de 22.28 el 2025-12-01 (descendió 7.36 puntos vs L7d) y YW un NPS de 25.22 (bajó 11.89 puntos vs L7d), resultando un NPS agregado de 23.41 (–8.68 pts vs L7d). La causa principal fue la insatisfacción en la ruta MAD–ZRH (NPS 0.0, n=4), donde pasajeros de Norteamérica (NPS –37.5) y billetes code-share AA (NPS –60.0) percibieron un espacio reducido en flota 32S, a pesar de una puntualidad alta (OTP15 92.30).

BUSINESS SH: Dominancia de IB mitigada por YW  
En Global/SH/Business, IB cayó a 26.32 el 2025-12-01 (–21.69 pts vs L7d) mientras YW subió a 55.56 (+22.20 pts vs L7d), resultando un NPS de 35.71 (–7.81 pts vs L7d). El desplome de IB se concentró en la ruta MAD–ORY (NPS 25.0, n=4) por expectativas incumplidas en configuración de cabina y amenities en A320neo, y la mejora de YW suavizó parcialmente ese impacto.

ECONOMY LH: Deterioro por segmentos Business y code-share  
La cabina Economy de LH registró un NPS de 5.39 el 2025-12-01 (–3.55 pts vs L7d, de 8.94 a 5.39). La principal causa fue la insatisfacción del segmento Business/Work (NPS –23.8) y de code-share “Others” (NPS –66.7) en la ruta EZE–MAD (NPS 12.5, n=24), junto a bajas valoraciones en flotas A321XLR/A350 C, pese a una operación puntual (OTP15 77.96) y sin incidentes.

BUSINESS LH: Colapso por servicio deficiente  
La cabina Business de LH cayó a –5.0 el 2025-12-01 (–29.03 pts vs L7d, de 24.03 a –5.0). Los drivers fueron la actitud apagada de la tripulación, la calidad del catering y la oferta de entretenimiento en MAD–MEX (n=3, NPS 33.3), impactando a viajeros de ocio (–15.4) y a flota A333 (–50.0), sin registro de incidentes formales.

PREMIUM LH: Fuerte recuperación por soft factors  
El segmento Premium de LH alcanzó un NPS de 62.5 el 2025-12-01 (+27.69 pts vs L7d, de 34.81 a 62.5). La mejora responde a una percepción muy positiva de puntualidad, actitud de tripulación y claridad del piloto, sin quejas relevantes en verbatims (n=22), a pesar de una ligera caída en ocupación (LF 88.29) y puntualidad (OTP15 77.96).

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

Economy SH  
- Escenario: CANCELACIÓN (–1.9, +11.6 | Normal)  
- Narrativa: Mientras IB arrastró al segmento por baja satisfacción en vuelos code-share (especialmente en MAD–VGO con BA, VY y LATAM), YW registró un desempeño operativo sobresaliente (OTP ajustado +3.79 pts, cero NCS) y feedback muy positivo por puntualidad y confort. Estos efectos opuestos se neutralizaron, ocultando la volatilidad interna en el resultado “Normal” del segmento Economy SH.  
- Evidencia Clave:  
  • IB: “Insatisfacción de pasajeros en code-share (BA, VY, LATAM), ruta MAD–VGO con NPS 0.0”  
  • YW: “OTP15_adjusted +3.79 pts, cero incidentes y verbatims muy positivos”  

Business SH  
- Escenario: SINERGIA (–8.6, –15.2 | –9.4)  
- Narrativa: Ambos sub-segmentos IB y YW confluyeron en una fuerte caída del NPS Business SH. Adopta la explicación del nodo padre: el golpe principal vino de vuelos code-share con BA con NPS extremadamente bajo, reforzado por diferencias de satisfacción según flota (A321 vs A320) y región de residencia.  
- Evidencia Clave: “Code-share BA con NPS –66.7 (IB) y caída de –15.2 pts en Business YW, que suman la anomalía de –9.43 pts en el agregado.”

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Short Haul, la dinámica es DILUCIÓN (Economy SH: Normal, Business SH: Negativo | Normal).  
- Narrativa: El rendimiento de SH estuvo dictado por la caída en Business SH —principalmente en vuelos code-share con BA (NPS –66.7) y diferencias de satisfacción por flota y región—; sin embargo, el volumen y estabilidad de Economy SH (OTP y Load Factor habituales, feedback muy positivo) amortiguaron ese efecto, resultando en una variación global dentro de lo esperado.  
- Evidencia Clave:  
  • Business SH: Code-share BA con NPS –66.7 (n=3), A321 con NPS 16.7, residentes en Europa fuera de España con NPS 5.3  
  • Economy SH: OTP15_adjusted +1.55 pts, cero NCS y verbatims mayoritariamente positivos  

En Long Haul, la dinámica es SINERGIA (Economy LH: Negativo, Business LH: Negativo, Premium LH: Negativo | Negativo).  
- Narrativa: La caída en NPS LH es sistémica: todas las cabinas se vieron afectadas por el elevado peso de la ruta LIM–MAD con pasajeros de ocio en flotas históricamente con bajas puntuaciones (A321XLR, A333, A332, A350) y por vuelos en código compartido con IB, AA y QR, que registraron NPS negativos.  
- Evidencia Clave:  
  • Ruta LIM–MAD: NPS –11.8 (n=17)  
  • Flotas A321XLR: NPS –63.6, A333: –28.1, A332: –21.8, A350: –12.9  
  • Code-share IB/AA/QR con puntuaciones muy bajas en Leisure y regiones de Europa/América/Asia

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es TRANSFERENCIA (LH: Negativo, SH: Normal | Global: Negativo).  
- Narrativa: La caída global de –8.1 pts fue contagiada íntegramente por el deterioro del largo radio. Aun cuando el corto radio mostró estabilidad e incluso ligera mejora, el fuerte descenso en NPS de LH arrastró el resultado agregado.  
- Evidencia:  
  • Experiencia en vuelos code-share con BA (NPS –82.4) y QR (–75.0)  
  • Ruta MAD–SCL con NPS –36.4  
  • Flota A321XLR con NPS –63.6

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

1. Causa: Deficiencias en la experiencia de vuelos código compartido  
A. Naturaleza de la causa  
 • Hipótesis: La coordinación de servicio y los estándares difieren entre la aerolínea principal y sus socios (IB, AA, QR), generando frustración en el pasajero.  

B. Evidencia consolidada y alcance  
 Segmento más grande: Global / Long Haul (todos sus sub-segmentos: Economy, Business, Premium)  
 • NPS: –8.60 pts (vs baseline 12.51) → caída de –21.11 pts  
 • Load Factor: 88.0 (–2.68 pts vs histórico)  
 • OTP15_adjusted: 78.91 (–0.51 pts vs histórico)  
 • Rutas: LIM–MAD NPS –11.8 (n=17)  
 • Code-share con baja valoración: IB, AA, QR (NPS individuales muy negativos)  
 • Verbatims: mayoría positivos en puntualidad y servicio, sin quejas operativas formales  
  
 “Esta disrupción en códigos compartidos impacta uniformemente a Economy, Business y Premium de largo radio.”  

2. Causa: Insatisfacción en flota de largo radio  
A. Naturaleza de la causa  
 • Hipótesis: La percepción de confort y amenities en aviones A321XLR, A333, A332 y A350 está por debajo de expectativas, especialmente en cabina Economy.  

B. Evidencia consolidada y alcance  
 Segmento más grande: Global / Long Haul / Economy (abarca la mayoría de pasajeros LH)  
 • NPS: –13.62 pts (vs baseline 8.94) → caída de –22.56 pts  
 • Load Factor: 87.41 (–2.94 pts)  
 • OTP15_adjusted: 78.91 (–0.51 pts)  
 • Verbatims: sin feedback específico de producto, pero descenso concomitante en ocupación y puntualidad sugiere incomodidad o cancelaciones/reasignaciones  
 • Rutas: no desglosadas, pero coincide con los mismos vuelos LIM–MAD de alto impacto  
   
 “La baja valoración de la flota afecta a todo el Economy de LH, incidiendo en la experiencia de viaje y reforzando la anomalía global de largo radio.”  

3. Causa: Rutas de alto impacto con predominio Leisure (LIM–MAD y MAD–SCL)  
A. Naturaleza de la causa  
 • Hipótesis: El peso de pasajeros de ocio en rutas específicas, donde históricamente se registran puntuaciones bajas, concentra la insatisfacción.  

B. Evidencia consolidada y alcance  
 Segmento más grande: Global (toda la red, pero muy visible en LH)  
 • NPS Global: 17.92 (vs baseline 26.01) → caída de –8.09 pts  
 • Ruta MAD–SCL: NPS –36.4 (n=22), la peor del día  
 • Ruta LIM–MAD: NPS –11.8 (n=17)  
 • Perfil: Leisure (NPS –11.3 en LH)  
 • Regiones más críticas: América Norte (–42.9), Asia (–37.5)  
   
 “Estas dos rutas impactan transversalmente a todos los sub-segmentos de la red, generando un sesgo negativo en el NPS global.”  

4. Causa: Fragmentación de experiencia en SH Business por code-share y flota  
A. Naturaleza de la causa  
 • Hipótesis: En corto radio Business, los vuelos code-share con BA ofrecen un servicio distinto al operado directamente; además, la heterogeneidad de aviones (A321 vs A320) y origen de pasajeros (residentes en Europa) incrementa la variabilidad.  

B. Evidencia consolidada y alcance  
 Segmento: Global / SH / Business  
 • NPS: 34.09 (vs baseline 43.53) → caída de –9.43 pts  
 • Load Factor: 70.63 (–2.12 pts)  
 • OTP15_adjusted: 92.23 (+2.80 pts)  
 • Code-share BA: NPS –66.7 (n=3)  
 • Flota A321: NPS 16.7 vs A320: NPS 100.0  
 • Residence Region Europa (excl. España): NPS 5.3  
 • Verbatims: positivos en tripulación y comida, sin quejas operativas formales  
   
 “El desbalance de experiencia en Business SH dicta la tendencia del segmento, afectando todas sus rutas code-share y aviones operados.”  

5. Motor positivo: Desempeño operativo y feedback en SH Economy YW  
A. Naturaleza de la causa  
 • Hipótesis: Una operación muy puntual (OTP15_adjusted +3.79 pts) y un Load Factor adecuado, unido a una cabina renovada, genera satisfacción excepcional en YW.  

B. Evidencia consolidada y alcance  
 Segmento: Global / SH / Economy YW  
 • NPS: 48.68 (vs baseline 37.11) → subida de +11.58 pts  
 • Load Factor: 81.37  
 • OTP15_adjusted: 91.60 (+3.79 pts)  
 • Incidentes NCS: 0  
 • Verbatims: 197 comentarios muy positivos (puntualidad, confort, amabilidad)  
 • Rutas: todas ≥ 48.7, excepción MAD–MUC (28.6, n=7)  
 • Perfil: Leisure (NPS 53.3) vs Business (6.7)  
   
 “Este nivel de servicio en Economy YW actúa como contrapeso positivo en el segmento Economy de corto radio.”

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

La red registró caídas de satisfacción en Global, con un NPS de 17.92 (30-11-2025) frente a 26.01 en L7d, bajando –8.09 pts, debido sobre todo al largo radio. LH mostró un NPS de –8.60 vs 12.51 (–21.11 pts), donde Economy cayó de 8.94 a –13.62 (–22.56 pts) por Load Factor reducido (87.41, –2.94 pts vs L7d) y puntualidad inferior (OTP15_adjusted 78.91, –0.51 pts), Business bajó de 24.03 a 21.21 (–2.82 pts) por heterogeneidad de flota y perfil Leisure, y Premium sufrió la mayor caída, de 34.81 a 4.17 (–30.64 pts) por un sesgo de muestra con clientes de América Central en A350 next. En contraste, Short Haul mantuvo estabilidad global (34.75 vs 33.14, +1.60 pts), con Business SH disminuyendo de 43.53 a 34.09 (–9.43 pts) por vuelos code-share con BA (–66.7, n=3) y variabilidad de flota/región, mientras que Economy SH quedó estable en 34.81 vs 32.09 (+2.72 pts) al compensar la caída de IB (27.76 vs 29.65, –1.89 pts) con la subida de YW (48.68 vs 37.11, +11.58 pts) tras una operación muy puntual (OTP15_adjusted +3.79 pts) y cero NCS.

Las rutas más impactadas fueron MAD–SCL (NPS –36.4, n=22) y LIM–MAD (–11.8, n=17) en LH, donde predominan viajeros Leisure en flotas A321XLR (–63.6), A333 (–28.1), A332 (–21.8) y A350 (–12.9). En SH, sobresalió MAD–VGO (0.0, n=4) en Economy IB y los code-share BA de Business (–66.7, n=3). Los grupos más reactivos incluyen pasajeros Leisure en Europa fuera de España, residentes en América Central en Premium LH y clientes de code-share con BA y QR.

ECONOMY SH: Equilibrio entre IB y YW  
La cabina Economy SH mantuvo desempeño estable durante el día 30-11-2025, registrando un NPS de 34.81 (30-11-2025) con una variación de +2.72 pts vs L7d. IB cayó de 29.65 a 27.76 (–1.89 pts) por insatisfacción en vuelos code-share (BA, VY y LATAM) en rutas como MAD–VGO (0.0, n=4), mientras que YW subió de 37.11 a 48.68 (+11.58 pts) gracias a una operación muy puntual (OTP15_adjusted +3.79 pts) y verbatims muy positivos en puntualidad y confort. Estos efectos opuestos se neutralizaron, manteniendo niveles consistentes de satisfacción.

BUSINESS SH: Impacto de code-share BA  
El segmento Business SH registró un NPS de 34.09 (30-11-2025) con una variación de –9.43 pts vs L7d. Esta evolución se explica principalmente por vuelos code-share con BA (NPS –66.7, n=3), la heterogeneidad de flota —A321 con 16.7 vs A320 con 100.0— y la baja valoración de residentes en Europa fuera de España (5.3). Rutas como DUS–MAD (75.0, n=4) mostraron muestras limitadas que acentuaron la caída.

ECONOMY LH: Retrasos y ocupación inferior  
La cabina Economy LH registró un NPS de –13.62 (30-11-2025) con una variación de –22.56 pts vs L7d. El Load Factor bajó a 87.41 (–2.94 pts) y el OTP15_adjusted quedó en 78.91 (–0.51 pts), reflejando vuelos menos llenos y retrasos que impactaron la experiencia. La ruta LIM–MAD marcó NPS –11.8 (n=17) con predominio de pasajeros Leisure.

BUSINESS LH: Ligera caída por perfil y flota  
El segmento Business LH registró un NPS de 21.21 (30-11-2025) con una variación de –2.82 pts vs L7d. A pesar de verbatims positivos en puntualidad y servicio, la variabilidad de flota —A333 con –100.0 vs A350 next con +50.0— y la mayor insatisfacción de Leisure (–12.5 vs Business +52.9) en rutas como MAD–MEX (100.0, n=4) explican este leve deterioro.

PREMIUM LH: Sesgo de muestra y flota  
El segmento Premium LH registró un NPS de 4.17 (30-11-2025) con una variación de –30.64 pts vs L7d. La ruta MAD–MEX (20.0, n=5) mostró un grupo concentrado de clientes de América Central (–50.0) en A350 next (–16.7), creando un sesgo de muestra que exacerbó la caída pese a verbatims positivos en confort y servicio.

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

En Economy SH, el escenario es SINERGIA (N, N | N).  
- Narrativa: Ambos subsegmentos (IB y YW) presentan rendimiento normal y contribuyen alineados al NPS normal del agregado.  
- Evidencia clave: IB y YW sin anomalías, sin variaciones operativas ni de feedback que alteren el resultado.  

En Business SH, el escenario es DOMINANCIA (+26.1, –10.3 | +14.0).  
- Narrativa: La subida excepcional de IB impone su signo al agregado de Business SH, aunque este efecto fue parcialmente suavizado por la caída de YW.  
- Evidencia clave: IB registró +26.07 pts impulsados por OTP15 mejorada y feedback destacado en la ruta LHR–MAD; YW cayó –10.28 pts sin incidencias operativas, apuntando a variabilidad muestral.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

En Long Haul (LH), la dinámica es CANCELACIÓN (+, – , N | N).  
- Narrativa: El NPS agregado de LH parece estable, pero oculta efectos opuestos:  
  • LH/Premium subió +15.2 pts impulsado por la experiencia excepcional en A350 “clásicos” y feedback de tripulación.  
  • LH/Business cayó –9.7 pts por insatisfacción de pasajeros en A350 y residentes en Europa/América Norte.  
  • LH/Economy se mantuvo normal, neutralizando el impacto.  
- Evidencia: Premium ganó por NPS 100 en A350 “clásicos”; Business sufrió NPS –28.6 en A350 y –33.3 en regiones Europa/América Norte.

En Short Haul (SH), la dinámica es DILUCIÓN (N, + | N).  
- Narrativa: El alza del radio SH está dictada por SH/Business, aunque su efecto fue amortiguado por la estabilidad de SH/Economy.  
- Evidencia: SH/Business subió +14.0 pts gracias a OTP15 +2.42 y feedback muy positivo en ruta LHR–MAD (A320neo y pasajeros leisure); SH/Economy permaneció en rango normal.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

A nivel GLOBAL, la dinámica es SINERGIA (+, + | +).  
- Narrativa: La red entera experimentó un impulso simultáneo en puntualidad y reducción de mishandling tanto en Largo Radio como en Corto Radio, lo que sumó efectos positivos hasta generar la anomalía global.  
- Evidencia:  
  • OTP15_adjusted en global aumentó 1.90 pts (90.02 vs 88.12).  
  • Mishandling se redujo 1.69 pts (13.84 vs 15.53).  
  • Feedback cualitativo muy positivo en 1 093 verbatims, destacando servicio y puntualidad.

---

## 📋 ANÁLISIS DE CAUSAS DETALLADO

1. Causa: Mejora operativa de puntualidad y reducción de mishandling  
A. Naturaleza de la causa  
   • Un impulso sistémico en eficiencia de red (OTP15 más alta y menos equipaje dañado) que elevó la satisfacción general.  
B. Evidencia consolidada y alcance  
   • Segmento más grande: GLOBAL  
   • Output causal:  
     – OTP15_adjusted: 90.02 (+1.90 pts sobre media 7d)  
     – Mishandling: 13.84 (–1.69 pts sobre media 7d)  
     – NPS Global: 30.08 vs baseline 26.01 (+4.07)  
     – 1 093 verbatims con sentimiento mayoritariamente positivo (destacan puntualidad y amabilidad)  
     – Ruta MAD–MUC: NPS –20.0 en 5 respuestas, identificada como único punto de fricción  
   • Afecta a todos los subsegmentos bajo GLOBAL (LH y SH, todas las cabinas)  

2. Causa: Experiencia excepcional a bordo de A350 “clásicos”  
A. Naturaleza de la causa  
   • Calidad de producto y servicio en la flota A350 que genera altos niveles de recomendación en Premium LH.  
B. Evidencia consolidada y alcance  
   • Segmento: GLOBAL / LH / Premium  
   • Output causal:  
     – NPS: 50.0 vs baseline 34.81 (+15.19)  
     – Load Factor: 87.69 (–2.97 pts)  
     – OTP15_adjusted: 77.8 (–1.75 pts)  
     – 10 verbatims, todos positivos (resaltan amabilidad y calidad de atención)  
     – Por tipo de avión: A350 “clásico” NPS 100.0 vs A350 next NPS 0.0  
   • Afecta a todo el subsegmento Premium en Largo Radio  

3. Causa: Desajuste de expectativas en Business LH para pasajeros en A350 y residentes Europa/América Norte  
A. Naturaleza de la causa  
   • Perfil de cliente insatisfecho por experiencia percibida en ciertos aviones y origen geográfico, no por fallos operativos.  
B. Evidencia consolidada y alcance  
   • Segmento: GLOBAL / LH / Business  
   • Output causal:  
     – NPS: 14.29 vs baseline 24.03 (–9.74)  
     – Load Factor: 93.43 (–0.48 pts)  
     – OTP15_adjusted: 77.8 (–1.75 pts)  
     – Flota: A350 NPS –28.6 vs A332 NPS 75.0 (spread 103.6)  
     – Región de residencia: Europa/NA NPS –33.3 vs España NPS 66.7 (spread 100.0)  
     – Ruta GRU–MAD: NPS 33.3 en 3 respuestas  
   • Afecta a todo el subsegmento Business en Largo Radio  

4. Causa: Puntualidad superior y experiencia destacada en SH Business (especialmente A320neo y pasajeros leisure)  
A. Naturaleza de la causa  
   • Mayor fiabilidad operativa y confort en rutas clave de Corto Radio que impulsaron la percepción de valor.  
B. Evidencia consolidada y alcance  
   • Segmento: GLOBAL / SH / Business  
   • Output causal:  
     – NPS: 57.5 vs baseline 43.53 (+13.97)  
     – OTP15_adjusted: 91.84 (+2.42 pts)  
     – Load Factor: 70.74 (–2.03 pts)  
     – 57 verbatims positivos (puntuación a espacio, servicio y procesos ágiles)  
     – Ruta LHR–MAD: NPS 100.0 en 4 respuestas  
     – Flota: A320neo NPS 81.8 vs CRJ NPS 16.7 (spread 65.2)  
     – Tipo de viaje: Leisure NPS 68.0 vs Business NPS 40.0  
   • Afecta a todo el subsegmento Business en Corto Radio

---

## 📋 SÍNTESIS EJECUTIVA FINAL

📈 SÍNTESIS EJECUTIVA:

El análisis detectó subidas y bajadas de NPS muy localizadas: el NPS Global mejoró de 26.01 a 30.08 (+4.07 pts) impulsado por una red más puntual (OTP15 en 90.02, +1.90 pts vs L7d) y menos mishandling (13.84, –1.69 pts vs L7d). En Long Haul Business el NPS cayó de 24.03 a 14.29 (–9.74 pts), atribuible a baja satisfacción de pasajeros en A350 (NPS –28.6) y residentes en Europa/América Norte (NPS –33.3). Contrariamente, Long Haul Premium subió de 34.81 a 50.00 (+15.19 pts) por la experiencia excepcional a bordo de A350 clásicos (NPS 100 vs 0 en A350 next). En Short Haul Business el NPS ascendió de 43.53 a 57.50 (+13.97 pts) gracias a OTP15 en 91.84 (+2.42 pts) y al servicio destacado en A320neo con pasajeros leisure (NPS 68.0 vs 40.0 en business), mientras que en ese mismo nodo IB escaló de 48.00 a 74.07 (+26.07 pts) y YW bajó de 33.36 a 23.08 (–10.28 pts).  

Las rutas más afectadas confirman este esquema: MAD–MUC exhibió un NPS de –20.0 (5 respuestas) como único punto de fricción global, GRU–MAD quedó en 33.3 (3 respuestas) y LHR–MAD alcanzó 100.0 (4 respuestas) impulsando la suba de Short Haul Business. Los grupos más reactivos fueron los usuarios de A350 clásicos y A320neo, residentes en Europa/América Norte, y el segmento Leisure, que marcó brechas de hasta 103.6 pts entre flotas y 28 pts entre tipos de viaje.  

ECONOMY SH YW e IB  
La cabina Economy de Short Haul mantuvo desempeño estable durante el 29-11-2025, registrando un NPS de 35.27 (vs 32.09 en L7d, +3.18 pts). IB obtuvo 33.54 (vs 29.65, +3.89 pts) y YW 39.68 (vs 37.11, +2.58 pts). No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción sin rutas ni perfiles que alteraran el balance.  

BUSINESS SH IB/YW  
El segmento Business de Short Haul mostró una evolución divergente: el agregado pasó de 43.53 a 57.50 (+13.97 pts). IB subió espectacularmente de 48.00 a 74.07 (+26.07 pts) por una puntualidad OTP15 en 92.89 (+1.51 pts), espacio extra en A320neo (NPS 81.8) y rutas como LHR–MAD (100.0). YW, en cambio, cayó de 33.36 a 23.08 (–10.28 pts), sin incidencias operativas, sugiriendo variabilidad muestral. El empuje de IB dominó pese a la resistencia de YW.  

ECONOMY LH  
La cabina Economy de Long Haul mantuvo desempeño estable durante el 29-11-2025, con un NPS de 14.71 (vs 8.94 en L7d, +5.77 pts). No se registraron desviaciones operativas ni feedback negativos destacados: la ocupación y la puntualidad se mantuvieron en niveles habituales y el feedback de 17 verbatims fue neutro.  

BUSINESS LH  
La cabina Business de Long Haul sufrió un deterioro notable, pasando de un NPS de 24.03 a 14.29 (–9.74 pts) vs L7d. Los drivers principales fueron la insatisfacción de pasajeros en A350 (NPS –28.6), especialmente residentes en Europa/América Norte (NPS –33.3), sin incidencias operativas que justifiquen la caída. La ruta GRU–MAD mostró un NPS de 33.3 (3 respuestas) como foco de análisis.  

PREMIUM LH  
El segmento Premium de Long Haul experimentó una fuerte subida, de 34.81 a 50.00 (+15.19 pts) vs L7d. El motor de este incremento fue la experiencia en A350 clásicos (NPS 100.0 frente a 0.0 en A350 next), junto a verbatims 100 % positivos sobre amabilidad y calidad de servicio, pese a una ligera merma operativa (OTP15 77.8, –1.75 pts; Load Factor 87.69, –2.97 pts).

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