===== SYSTEM =====

Eres un experto analista ejecutivo y "Intérprete Temporal" de datos de NPS.
Tu misión es sintetizar análisis de distintas agregaciones temporales (semanal vs diaria) para construir una narrativa única y coherente.

⚠️ **REGLAS DE ORO (CRÍTICAS):**
1. **NOMENCLATURA DE COMPAÑÍAS:** Las compañías SIEMPRE se escriben como **IB** e **YW**. NUNCA uses "Iberia", "Young Wings", "Yanky Whiskey", "India Bravo", ni ninguna otra variación. Solo IB e YW, sin excepciones.
2. **ATRIBUCIÓN TOTAL:** Cada vez que des un dato (NPS, OTP, variación), DEBES especificar el **Segmento** y la **Agregación Temporal** (si no es obvia por el contexto).
   - *Mal:* "El NPS cayó a 20.5 (-5 pts)."
   - *Bien:* "El NPS de **Economy LH** cayó a 20.5 (**–5.0 pts**, **Semanal**)."
3. **TERMINOLOGÍA OBLIGATORIA:**
   - NUNCA uses "Short Haul", "Largo Radio", "Corto Radio" o similares. USA SIEMPRE **SH** y **LH**.
   - "SHAT", "Shapley" o "impacto en drivers" se traduce SIEMPRE como "**ppts de NPS según Explanatory Drivers**".
   - **PRESERVA** la terminología de comparación del interpreter (ej: "con respecto a la media de los últimos 7 días")
   - **NUNCA** uses abreviaturas como "vs baseline", "vs L7d" o similares.
4. **NO INVENTES:** Si falta un dato, omítelo o di que no está disponible. NO calcules promedios si no se dan.
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
<b>SÍNTESIS EJECUTIVA</b><br>
La red global registró un NPS de 27.9 (+0.6 pts) con respecto a la semana anterior, impulsado por un sólido desempeño en SH que compensó las caídas en LH.<br><br>
En SH la satisfacción se mantuvo elevada. Economy SH alcanzó un NPS de 35.2 (+1.4 pts) gracias al impulso de YW, mientras que Business SH alcanzó 38.8 (+3.7 pts) por la mejora de la puntualidad y el check-in de IB. Estas dinámicas generaron una aportación neta positiva a nivel global.<br><br>
En LH, el NPS cayó a 11.1 (–2.5 pts). Business LH mejoró a 26.4 (+10.3 pts) por un OTP15 de 72.3 (+4.5 pts) y drivers de puntualidad (+4.7 ppts) y check-in (+2.2 ppts), pero Economy LH y Premium LH retrocedieron. Economy LH descendió a 8.7 (–3.7 pts) por incidencias en equipaje y embarque, y Premium LH cayó a 13.8 (–8.2 pts) debido a deficiencias de tripulación (–6.9 ppts), catering (–4.0 ppts) y check-in (–3.6 ppts).<br><br>
<b><u>DETALLE POR CABINA</u></b><br>
<b><u>ECONOMY SH: Dificultades puntuales en IB compensadas por YW</u></b><br>
La cabina Economy de SH registró un NPS de 35.2 (+1.4 pts) con respecto a la semana anterior. <b>Desglose por compañía:</b> IB obtuvo 32.8 (–1.3 pts) y YW 39.8 (+6.5 pts). El avance neto se basó en el sólido desempeño de YW, mientras IB sufrió por mishandling de equipaje y caos en el embarque, reflejados en 87 incidentes operativos (–15 vs baseline) y huelgas en Portugal y VCE.<br><br>
<b><u>BUSINESS SH: Impacto mixto de interior en YW y puntualidad en IB</u></b><br>
El segmento Business de SH registró un NPS de 38.8 (+3.7 pts) con respecto a la semana anterior. <b>Desglose por compañía:</b> IB obtuvo 42.8 (+6.4 pts) y YW 29.8 (–2.6 pts). IB mejoró con puntualidad (+4.7 ppts según Explanatory Drivers) y check-in (+2.2 ppts), en rutas como DFW-MAD y MAD-SCL, mientras YW se vio penalizada por el mal estado del interior de cabina (–4.1 ppts) y verbatims en FRA-MAD y MAD-VCE.<br><br>
<b><u>ECONOMY LH: Deterioro por handling de equipaje y embarque</u></b><br>
La cabina Economy de LH registró un NPS de 8.7 (–3.7 pts) con respecto a la semana anterior. La causa principal fue un repunte de incidentes de equipaje irregular (+3 casos) y desorganización en facturación y embarque, reflejado en Cabin Crew –2.4 ppts, Arrivals –1.7 ppts e In flight food –0.9 ppts. Esto se tradujo en quejas en GYE-MAD (NPS 5), IAD-MAD (NPS 0) y LAX-MAD (NPS 6), impactando sobre todo a clientes corporativos y de residencia LATAM.<br><br>
<b><u>BUSINESS LH: Fuerte alza por puntualidad y OTP</u></b><br>
La cabina Business de LH registró un NPS de 26.4 (+10.3 pts) con respecto a la semana anterior. Los drivers principales fueron Punctuality +4.7 ppts y Boarding +2.1 ppts, respaldados por un OTP15 de 72.3 (+4.5 pts). Se observó una mayor satisfacción en rutas DFW-MAD y MAD-SCL, sin variaciones relevantes en perfiles de cliente.<br><br>
<b><u>PREMIUM LH: Caída pronunciada por fallos de producto</u></b><br>
El segmento Premium de LH registró un NPS de 13.8 (–8.2 pts) con respecto a la semana anterior. Las causas dominantes fueron Cabin Crew –6.9 ppts, In flight food & beverage –4.0 ppts y Check-in –3.6 ppts, a pesar de un OTP15 de 72.3 (+4.5 pts). Las rutas BOG-MAD y MAD-MIA concentraron las mayores quejas de equipaje perdido y atención en mostrador, afectando especialmente a pasajeros en código compartido y de residencia Europa.

**ANÁLISIS DIARIO SINGLE:**
📅 2025-12-08 to 2025-12-08:
<b>SÍNTESIS EJECUTIVA</b><br>
La red global registró un <b>NPS de 29.0 (+1.5 pts)</b> con respecto a la media de los últimos 7 días. A nivel agregado, la mejora en LH superó el deterioro en SH, impulsando el resultado global.

<br><br>
En SH, la caída de 1.4 pts en el NPS de 33.6 se concentró en Business SH, que anotó un <b>NPS de 15.9 (–23.4 pts)</b> debido a 4 cancelaciones en MAD-OVD e IBZ-VLC, 6 retrasos incluyendo un aplazamiento de más de 7 h en MAD-ZRH, y quejas por downgrades de flota y asientos rígidos en AMS-MAD y FRA-MAD. Economy SH mantuvo su NPS en 35.5 (+0.9 pts), diluyendo parcialmente el impacto negativo; sin embargo, IB retrocedió a 31.8 (–0.4 pts) por equipaje de mano mandado a bodega, caos en embarque y overbooking en rutas como BCN-MAD y MAD-OPO, mientras que YW se mantuvo estable en 43.1 (+3.5 pts).

<br><br>
En LH se registró un repunte significativo: Business LH alcanzó <b>46.4 (+21.4 pts)</b> y Economy LH <b>17.8 (+9.1 pts)</b>, impulsados por la gestión de 13 incidentes operativos (5 demoras y una reprogramación de 7 h 50 min en IB155/08DEC/MAD-BOG), junto con una menor incidencia de mishandling de equipaje, pese a una puntualidad inferior (OTP15 ajustado de 72.2, –6.2 pts). Premium LH subió a <b>23.1 (+10.7 pts)</b> sin causas operativas claras, reflejando verbatims muy positivos sobre el servicio en tierra y a bordo en EZE-MAD y BOG-MAD.

<br><br>
Este contraste geográfico explica cómo el sólido desempeño de LH arrastró la media global, contrarrestando los incidentes concentrados en SH.

<br><br>
<b><u>DETALLE POR CABINA</u></b><br>

<b><u>ECONOMY SH: Déficit en servicio a tierra y equipaje.</u></b><br>
La cabina Economy de SH registró un NPS de 35.5 (+0.9 pts) con respecto a la media de los últimos 7 días. <b>Desglose por compañía:</b> IB obtuvo 31.8 (–0.4 pts) y YW 43.1 (+3.5 pts). IB concentró quejas por equipaje de mano facturado en bodega pese a prioridad, desorden en grupos de embarque y overbooking en rutas como BCN-MAD, LIS-MAD y MAD-OPO. YW mantuvo desempeño estable, diluyendo el impacto de IB.

<br><br>
<b><u>BUSINESS SH: Impacto de cancelaciones y flotas pequeñas.</u></b><br>
El segmento Business de SH registró un NPS de 15.9 (–23.4 pts) con respecto a la media de los últimos 7 días. <b>Desglose por compañía:</b> IB obtuvo 22.6 (–21.1 pts) y YW 0.0 (–29.3 pts). Ambas sufrieron cancelaciones en MAD-OVD e IBZ-VLC y 6 retrasos, incluido un lapso >7 h en MAD-ZRH. IB registró downgrades de flota y asientos rígidos en FRA-MAD y AMS-MAD, mientras que YW recibió feedback de oferta Business de Air Nostrum sin espacio para equipaje de mano en AMS-MAD y FRA-MAD.

<br><br>
<b><u>ECONOMY LH: Alza destacada pese a demoras operativas.</u></b><br>
La cabina Economy de LH registró un NPS de 17.8 (+9.1 pts) con respecto a la media de los últimos 7 días. La causa principal fue la gestión de 13 incidentes operativos, incluyendo 5 demoras y la reprogramación de 7 h 50 min en IB155/08DEC/MAD-BOG, respaldada por verbatims que, pese a citar demoras y falta de agua en la espera, destacaron menos mishandling de equipaje. Esta mejora se reflejó en rutas como MAD-MIA (8.3), EZE-MAD (6.0) y BOG-MAD (0.0), con perfiles Leisure más reactivos (23.8) que Business (0.0).

<br><br>
<b><u>BUSINESS LH: Mejora robusta impulsada por reducción de retrasos.</u></b><br>
La cabina Business de LH alcanzó un NPS de 46.4 (+21.4 pts) con respecto a la media de los últimos 7 días. Los drivers principales fueron la menor incidencia de retrasos operativos y la gestión de la reprogramación de IB155/08DEC/MAD-BOG por 7 h 50 min. Los verbatims registraron menciones de esperas en EZE-MAD y MAD-MIA. El perfil Leisure mostró un NPS de 62.5 y Business/Work de 25.0.

<br><br>
<b><u>PREMIUM LH: Subida moderada sin causa operativa tangible.</u></b><br>
El segmento Premium de LH registró un NPS de 23.1 (+10.7 pts) con respecto a la media de los últimos 7 días. No se identificó una causa operativa clara en las métricas de puntualidad o load factor. Los verbatims destacaron servicios de primer nivel en tierra y a bordo en rutas EZE-MAD y BOG-MAD, reflejados en tres comentarios con NPS 10.
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-09 to 2025-12-09:
<b>SÍNTESIS EJECUTIVA</b><br>
La red global registró un NPS de 29.3 (+1.9 pts) con respecto a la semana anterior, impulsado por el sólido desempeño de SH que compensó la contracción en LH.<br><br>
En LH el NPS fue de 8.0 (–2.5 pts), lastrado por el colapso en Premium, donde un OTP15 de 71.3 (–7.0 pts) y 6 retrasos técnicos junto a 2 cancelaciones impactaron fuertemente la satisfacción. Los verbatims reportaron mala calidad de comida, cambios de asiento inapropiados, falta de entretenimiento y mishandling de equipaje en rutas como MAD-MEX y EZE-MAD. Este deterioro predominó pese a que Business LH mejoró gracias al feedback de pasajeros Leisure y a menciones de puntualidad y profesionalidad de la tripulación.<br><br>
SH mantuvo desempeño con un NPS de 40.2 (+5.2 pts), con un alza concentrada en Business SH. Este segmento se benefició de YW, donde incidentes operativos (2 cancelaciones, 3 retrasos y 1 desvío) y quejas de embarque en pista en MAD-MUC y configuración de flota regional en GVA-MAD y MAD-PNA generaron un aumento de 16.9 pts. IB se mantuvo estable, absorbiendo el impulso de YW.<br><br>
<b><u>DETALLE POR CABINA</u></b><br>
<b><u>ECONOMY SH: Desempeño estable</u></b><br>
La cabina Economy de SH registró un NPS de 39.6 con una variación de +5.0 pts con respecto a la semana anterior. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción. Desglose por compañía: IB obtuvo 37.5 (+5.2 pts) y YW 43.7 (+4.1 pts), ambas subidas reflejan un efecto conjunto derivado de un contexto operativo estable.<br><br>
<b><u>BUSINESS SH: Alza impulsada por YW</u></b><br>
El segmento Business de SH registró un NPS de 46.5 con una variación de +7.2 pts con respecto a la semana anterior. Desglose por compañía: IB obtuvo 46.7 (+2.9 pts) y YW 46.2 (+16.9 pts). El crecimiento responde exclusivamente a YW, donde incidentes operativos (2 cancelaciones, 3 retrasos y 1 desvío) y verbatims que señalaron embarque en pista en MAD-MUC, flota “mini” sin espacio en GVA-MAD y precios elevados frente al tren en MAD-PNA impulsaron la satisfacción.<br><br>
<b><u>ECONOMY LH: Deterioro por puntualidad</u></b><br>
La cabina Economy de LH registró un NPS de 5.7 con una variación de –3.0 pts con respecto a la semana anterior. La causa principal fue la baja puntualidad, con un OTP15 de 71.3 (–7.0 pts) y 14 incidentes operativos (6 retrasos técnicos y 2 cancelaciones), complementada por mishandling de equipaje y falta de servicio de cabina reportados en verbatims, especialmente en MAD-UIO (NPS de 0.0 con 7 encuestas). Los perfiles más reactivos fueron Leisure (NPS de 8.8 con 193 encuestas) y Business/Work (–26.3 con 19 encuestas).<br><br>
<b><u>BUSINESS LH: Mejora por satisfacción Leisure</u></b><br>
La cabina Business de LH registró un NPS de 36.1 con una variación de +11.1 pts con respecto a la semana anterior. El alza se explica por la elevada satisfacción de pasajeros Leisure (NPS de 43.8 con 32 encuestas) y la percepción de profesionalidad de la tripulación, confort en asientos y puntualidad, pese a que las métricas operativas no mejoraron. La ruta HAV-MAD (NPS de 50.0 con 2 encuestas) reflejó esta tendencia.<br><br>
<b><u>PREMIUM LH: Caída abrupta</u></b><br>
El segmento Premium de LH registró un NPS de –30.8 con una variación de –43.1 pts con respecto a la semana anterior. Las causas dominantes fueron la baja puntualidad (OTP15 de 71.3, –7.0 pts) y 6 retrasos significativos junto a 2 cancelaciones, respaldadas por verbatims que mencionaron mala calidad de comida, cambios de asiento inapropiados, falta de entretenimiento y mishandling de equipaje en rutas como MAD-MEX (NPS de –33.3 con 3 encuestas) y EZE-MAD (NPS de –80.0 con 5 encuestas). El perfil Leisure (NPS de –41.7 con 12 encuestas) fue el más insatisfecho.
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-10 to 2025-12-10:
<b>SÍNTESIS EJECUTIVA</b><br>
La red global registró un NPS de 32.6 (+5.1 pts) con respecto a la semana anterior. Este avance se explica por el fuerte empuje de LH, mientras que SH mantuvo un desempeño estable con ligeras variaciones.<br><br>
En LH el NPS alcanzó 21.7 (+11.1 pts) gracias al notable alza en Economy (NPS de 19.5 +10.8 pts) y Premium (NPS de 42.9 +30.5 pts). Economy LH obtuvo 6 menciones positivas sobre puntualidad, atención de tripulación, catering y entretenimiento en flotas A321XLR (NPS 50.0) y A350 C (NPS 44.4), a pesar de 10 incidentes operativos (3 retrasos, 2 mishandling). Premium LH destacó por la satisfacción de pasajeros Business/Work (NPS 100.0) y flota A350 next (NPS 57.1) en rutas como MAD-MEX (NPS 57.1), pese a 10 incidentes (3 retrasos, 2 mishandling, 2 cambios de aeronave, 1 incidencia técnica).<br><br>
En SH el NPS fue 37.8 (+2.8 pts) pero concentró contratiempos en Business SH (NPS de 36.4 –2.9 pts) por 16 cancelaciones, 3 retrasos, 1 desvío y 1 incidente de equipaje. Verbatims en MAD-VIE, FCO-MAD y MAD-MUC refirieron retrasos en embarque/desembarque, sobreventa de plazas y problemas con equipaje prioritario.<br><br>
YW en Economy SH registró un NPS de 37.8 (–1.8 pts) por 16 cancelaciones, 3 retrasos prolongados, 1 desvío y un mishandling de 109 maletas. Los comentarios en MAD-VLC y PMI-VLC citaron cancelaciones sin explicación y pérdida de equipaje tras varios días. El sólido volumen de IB en Economy SH atenuó ese impacto.<br><br>
<b><u>DETALLE POR CABINA</u></b><br>
<b><u>ECONOMY SH: Desempeño estable</u></b><br>
La cabina Economy de SH registró un NPS de 37.9 (+3.3 pts) con respecto a la semana anterior. <b>Desglose por compañía:</b> IB obtuvo un NPS de 37.9 (+5.7 pts) y YW un NPS de 37.8 (–1.8 pts). El ligero descenso en YW se diluyó gracias al buen volumen de IB.<br><br>
<b><u>BUSINESS SH: Afectado por cancelaciones y retrasos</u></b><br>
El segmento Business de SH registró un NPS de 36.4 (–2.9 pts) con respecto a la semana anterior. <b>Desglose por compañía:</b> IB obtuvo un NPS de 42.3 (–1.4 pts) y YW un NPS de 14.3 (–15.0 pts). Ambos sufrieron 16 cancelaciones, 3 retrasos, 1 desvío y 1 mishandling, con verbatims que citaron demoras en embarque y desbordes de mostrador en MAD-VIE, LHR-MAD y ATH-MAD, sobreventa en MAD-MUC y fallos en equipaje prioritario en FCO-MAD.<br><br>
<b><u>ECONOMY LH: Impulso por calidad de servicio</u></b><br>
La cabina Economy de LH registró un NPS de 19.5 (+10.8 pts) con respecto a la semana anterior. La causa principal fue la excelente percepción de la tripulación y la puntualidad, reforzada en flotas A321XLR (NPS 50.0) y A350 C (NPS 44.4). Se contabilizaron 10 incidentes operativos (3 retrasos, 2 mishandling), sin impacto en la valoración general. La ruta MAD-SCL mantuvo un NPS de 15.4.<br><br>
<b><u>BUSINESS LH: Desempeño estable</u></b><br>
La cabina Business de LH registró un NPS de 28.6 (+3.6 pts) con respecto a la semana anterior. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción.<br><br>
<b><u>PREMIUM LH: Fuerte repunte</u></b><br>
El segmento Premium de LH registró un NPS de 42.9 (+30.5 pts) con respecto a la semana anterior. Las causas dominantes fueron la alta satisfacción de pasajeros Business/Work (NPS 100.0) y la experiencia en flota A350 next (NPS 57.1), especialmente en rutas como MAD-MEX (NPS 57.1), pese a 10 incidentes (3 retrasos, 2 mishandling, 2 cambios de aeronave, 1 incidencia técnica).
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-11 to 2025-12-11:
<b>SÍNTESIS EJECUTIVA</b><br>
La red global registró un NPS de 24.1 (–3.3 pts) con respecto a la semana anterior, reflejando un impacto sistémico en manejo de equipaje y conexiones que afectó a todos los segmentos.<br><br>
El desempeño de LH cayó a un NPS de 5.0 (–5.5 pts) debido a un fuerte descenso en puntualidad (OTP15 –8.2 pts) acompañado de cinco retrasos operativos y siete menciones de pérdida de conexión en feedback. Además, surgió mishandling de equipaje con un incidente por limitación de peso y seis menciones de maletas extraviadas, especialmente en la ruta BOG-MAD.<br><br>
En SH el NPS bajó a 32.8 (–2.1 pts) por un pico de 12 cancelaciones y nueve retrasos, reflejados en verbatims con seis referencias a vuelos cancelados y cinco a equipaje mal manejado, así como quince quejas sobre caos en embarque. La ruta LHR-MAD destacó con un NPS de –7.7.<br><br>
Dentro de Business SH, IB mejoró su servicio de cabina y comunicación de demoras, alcanzando un NPS de 56.7 (+12.9 pts) en rutas como LHR-MAD y DUS-MAD, mientras que YW cayó a 23.5 (–5.8 pts) por doce cancelaciones y nueve retrasos en MAD-SDR y BCN-VLC.<br><br>
<b><u>DETALLE POR CABINA</u></b><br>
<b><u>ECONOMY SH: Impacto de cancelaciones y retrasos</u></b><br>
La cabina Economy de SH registró un NPS de 31.6 (–3.0 pts) con respecto a la semana anterior. <b>Desglose por compañía:</b> IB obtuvo 31.0 (–1.3 pts) y YW 32.5 (–7.1 pts). Ambas compañías sufrieron 12 cancelaciones y nueve retrasos, reflejados en tres menciones de cancelaciones y ocho de retrasos en verbatims, especialmente en la ruta LHR-MAD.<br><br>
<b><u>BUSINESS SH: Desempeño estabilizado</u></b><br>
La cabina Business de SH registró un NPS de 44.7 (+5.4 pts) con respecto a la semana anterior. No se detectaron cambios significativos en el agregado. <b>Desglose por compañía:</b> IB obtuvo 56.7 (+12.9 pts) gracias a mejoras en servicio de cabina y comunicación de demoras en LHR-MAD y DUS-MAD, mientras que YW registró 23.5 (–5.8 pts) afectada por cancelaciones y retrasos en MAD-SDR y BCN-VLC.<br><br>
<b><u>ECONOMY LH: Deterioro por puntualidad</u></b><br>
La cabina Economy de LH registró un NPS de 6.4 (–2.3 pts) con respecto a la semana anterior. La causa principal fue el descenso en puntualidad (OTP15 –8.2 pts) con cinco retrasos operativos y siete menciones de pérdida de conexión en verbatims, complementada por mishandling de equipaje con un incidente y seis menciones de maletas perdidas, especialmente en la ruta BOG-MAD.<br><br>
<b><u>BUSINESS LH: Impacto crítico de retrasos</u></b><br>
La cabina Business de LH registró un NPS de –6.5 (–31.4 pts) con respecto a la semana anterior. La caída se explicó por un descenso en puntualidad (OTP15 –8.2 pts) y cinco retrasos operativos, reforzada por mishandling de equipaje (un incidente de limitación de peso) y quejas de equipamiento y tripulación, especialmente en las rutas BOG-MAD (NPS –28.0) y EZE-MAD.<br><br>
<b><u>PREMIUM LH: Leve deterioro de puntualidad</u></b><br>
La cabina Premium de LH registró un NPS de 11.8 (–0.6 pts) con respecto a la semana anterior. El principal driver fue una caída de puntualidad con cinco retrasos y un incidente de mishandling de 44 maletas en IB152/11DIC, reflejado en verbatims de pérdida de equipaje en EZE-MAD y LIM-MAD.
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-12 to 2025-12-12:
<b>SÍNTESIS EJECUTIVA</b><br>
La red global registró un <b>NPS de 27.3 (–0.1 pts)</b> con respecto a la media de los últimos 7 días. Este ligero descenso se originó principalmente en LH, mientras SH mantuvo estabilidad.<br><br>
En términos globales, la operatividad se vio lastrada por un aumento de 2 incidentes de equipaje reportados en incidentes operativos y 29 menciones de quejas sobre maltrato y pérdida de maletas en verbatims, concentradas en la ruta EZE-MAD (NPS de –10.3 pts, 29 encuestas). Estos problemas de mishandling trasladaron al Global un impacto negativo en la experiencia de los pasajeros.<br><br>
El deterioro se acentuó en Economy LH, donde la puntualidad retrocedió 7.3 pts en OTP15_adjusted, se registraron 3 retrasos reportados y 12 menciones de esperas prolongadas y falta de comunicación. A esto se sumaron 3 incidencias de asiento no respetado y 8 reportes de equipaje extraviado o cobros inesperados, con efectos destacados en JFK-MAD (NPS de –60.0 pts, 10 encuestas) y LAX-MAD (NPS de –40.0 pts, 10 encuestas).<br><br>
Por su parte, Business LH mostró una recuperación con un <b>NPS de 39.1 (+14.1 pts)</b>, impulsada por la gestión de 3 retrasos en MAD-UIO y 2 menciones de compensación en verbatims, mientras Premium LH alcanzó un <b>NPS de 33.3 (+21.0 pts)</b> gracias a intervenciones centradas en pasajeros en MAD-SJO, BOG-MAD y EZE-MAD tras retrasos y problemas de equipaje.<br><br>
<b><u>DETALLE POR CABINA</u></b><br>
<b><u>ECONOMY SH: Compensación de dinámicas opuestas</u></b><br>
La cabina Economy de SH registró un NPS de 35.0 (+0.5 pts) con respecto a la media de los últimos 7 días. Desglose por compañía: IB obtuvo 29.8 (–2.5 pts) y YW alcanzó 47.0 (+7.4 pts). El descenso de IB se atribuyó al mishandling de 70 maletas no cargadas y 17 menciones en verbatims, mientras la elevada satisfacción de clientes Leisure de YW compensó el efecto.<br><br>
<b><u>BUSINESS SH: Neutralidad operativa</u></b><br>
El segmento Business de SH registró un NPS de 40.8 (+1.5 pts) con respecto a la media de los últimos 7 días. Desglose por compañía: IB obtuvo 38.2 (–5.5 pts) y YW 46.7 (+17.4 pts). IB sufrió un incidente de mishandling de 70 maletas y menciones de equipaje, mientras YW potenció su satisfacción reduciendo la ocupación de los vuelos.<br><br>
<b><u>ECONOMY LH: Afectada por puntualidad y equipaje</u></b><br>
La cabina Economy de LH registró un NPS de 1.1 (–7.6 pts) con respecto a la media de los últimos 7 días. La causa principal fue la caída de puntualidad, con OTP15_adjusted retrocediendo 7.3 pts, acompañada de 3 retrasos reportados y 12 menciones de demoras y falta de comunicación. A esto se sumaron 3 incidencias de asiento no respetado y 8 reportes de equipaje dañado o cobros inesperados, especialmente en JFK-MAD (NPS de –60.0 pts, 10 encuestas), LAX-MAD (NPS de –40.0 pts, 10 encuestas), MAD-SDQ (NPS de –16.7 pts, 6 encuestas), GIG-MAD (NPS de –14.3 pts, 14 encuestas) y otras rutas de menor volumen.<br><br>
<b><u>BUSINESS LH: Mejora tras gestión de retrasos</u></b><br>
La cabina Business de LH registró un NPS de 39.1 (+14.1 pts) con respecto a la media de los últimos 7 días. La mejora estuvo vinculada a la gestión de 3 retrasos en MAD-UIO y 2 menciones en verbatims sobre información y compensaciones, elevando la percepción de atención en ruta.<br><br>
<b><u>PREMIUM LH: Recuperación enfocada en servicio</u></b><br>
El segmento Premium de LH registró un NPS de 33.3 (+21.0 pts) con respecto a la media de los últimos 7 días. Los pasajeros destacaron la atención tras 3 retrasos reportados, junto a 1 incidente de mishandling y 10 menciones de mejoras en la app, reservas y asignación de asientos, con especial resonancia en MAD-SJO (NPS de –100.0 pts, 1 encuesta) y BOG-MAD (NPS de 33.3 pts, 3 encuestas).
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-13 to 2025-12-13:
🎯 **SÍNTESIS EJECUTIVA FINAL - ANÁLISIS NPS**

**Enfoque requerido:** Genera un informe ejecutivo NARRATIVO orientado a causas. Usa las evidencias extraídas en el paso anterior (4B).

**⚠️ FORMATO TEAMS (CRÍTICO):**
- El output FINAL debe ser **HTML** (NO Markdown).
- Usa solo: `<b>...</b>`, `<u>...</u>`, `<br>`, `<ul>`, `<li>`.
- No uses `**negrita**` (usa `<b>`), `#`, `-` ni backticks.
- No uses emojis.

**⚠️ REGLAS DE LENGUAJE EJECUTIVO:**
- 🚫 **PROHIBICIÓN ESTRICTA:** NUNCA escribas "Young Wings". Usa SIEMPRE **YW**.
- NUNCA uses términos técnicos internos: "nodo", "árbol", "NMA", "burbujeo".
- USA "LH" y "SH" (NO "Largo Radio" ni "Corto Radio").
- **NO ENUMERES las causas**. Nárralas de forma fluida.

**🔢 FORMATO NUMÉRICO (CRÍTICO):**
- Redondea TODOS los valores numéricos (NPS, SHAP, OTP, porcentajes) a **1 decimal** máximo.
- **OBLIGATORIO:** Siempre que menciones una variación de NPS, debes incluir primero el valor absoluto actual. 
  *   **Formato:** "NPS de X (Y pts)" o "NPS de X con una variación de Y pts".
  *   Ejemplos: `NPS de 27.9 (+0.6 pts)`; `NPS de 8.7 (–3.7 pts)`.
- Para números enteros (conteos de incidentes, pasajeros), NO uses decimales.

**Formato esperado (HTML):**

Empieza SIEMPRE con:
`<b>SÍNTESIS EJECUTIVA</b><br>`

**PÁRRAFO INTRODUCTORIO (2-3 líneas):**
Debe resumir el resultado GLOBAL mencionando el NPS absoluto y su variación.
Ejemplo: "La red global registró un **NPS de 27.9 (+0.6 pts)** con respecto a la semana anterior..."

**DESARROLLO NARRATIVO DE LAS CAUSAS (2-4 párrafos):**
Texto corrido. Separa párrafos con `<br><br>`.

**✍️ NARRATIVA DE INCIDENTES (NCS):** 
No uses listas densas entre paréntesis. Integra los datos en la frase de forma fluida y natural.
- **⚠️ ESPECIFICIDAD:** NUNCA digas simplemente "otras incidencias" o "limitaciones" si el dato original especifica el motivo (ej: equipaje, puertas, huelgas). DEBES mencionar el motivo concreto.
- **Bien:** "La operatividad se vio lastrada por un aumento de **15 cancelaciones** y **9 incidencias de equipaje**, provocadas principalmente por..."

**✍️ DETALLE DE VERBATIMS:**
No generalices el feedback. Cita los problemas concretos (ej: asientos que no reclinan, falta de comida, trato descortés) y menciona al menos 1 o 2 rutas donde el feedback fue más intenso.

**DETALLE POR CABINA (HTML):**
Añade `<br><br><b><u>DETALLE POR CABINA</u></b><br>` y luego pega `
    **ECONOMY SH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Economy de SH registró un NPS de [valor cabina] con [variación] pts vs L7d. **Desglose por compañía:** IB obtuvo [NPS IB] ([diff IB] pts) y YW [NPS YW] ([diff YW] pts). [Breve explicación de la dinámica: si ambas suben/bajan = efecto conjunto; si una sube y otra baja = se compensan; si solo una tiene variación = esa domina]
    
    **BUSINESS SH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] El segmento Business de SH registró un NPS de [valor cabina] con [variación] pts vs L7d. **Desglose por compañía:** IB obtuvo [NPS IB] ([diff IB] pts) y YW [NPS YW] ([diff YW] pts). [Breve explicación de la dinámica y causas principales de cada compañía con sus evidencias clave]
    
    **ECONOMY LH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Economy de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha período]) con una [variación de NPS_diff cabina] puntos respecto a la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "La causa principal fue [hipótesis con datos (drivers, operativa, NCS, verbatims) que la respaldan], complementada por [hipótesis secundarias (si las hubiera)]. Esta [mejora/deterioro] se reflejó especialmente en rutas como [top rutas con NPS y diff], mientras que los perfiles más reactivos incluyen [perfiles específicos]."]
    
    **BUSINESS LH: [Título]**  
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Business de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha período]) con una [variación de NPS_diff cabina] puntos respecto al período anterior.     [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "Los drivers principales fueron [causas con SHAP], impactando especialmente las rutas [rutas específicas] y perfiles [perfiles específicos]."]
    
    **PREMIUM LH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] El segmento Premium de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha]) con [diff cabina] puntos de [variación] vs la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "Las causas dominantes fueron [drivers SHAP], especialmente evidentes en [rutas top] y entre [perfiles reactivos]."]
            ` (también en HTML) **SOLAMENTE SI** hay más de una cabina analizada.
*   Usa este formato para los títulos de cabina dentro de `
    **ECONOMY SH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Economy de SH registró un NPS de [valor cabina] con [variación] pts vs L7d. **Desglose por compañía:** IB obtuvo [NPS IB] ([diff IB] pts) y YW [NPS YW] ([diff YW] pts). [Breve explicación de la dinámica: si ambas suben/bajan = efecto conjunto; si una sube y otra baja = se compensan; si solo una tiene variación = esa domina]
    
    **BUSINESS SH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] El segmento Business de SH registró un NPS de [valor cabina] con [variación] pts vs L7d. **Desglose por compañía:** IB obtuvo [NPS IB] ([diff IB] pts) y YW [NPS YW] ([diff YW] pts). [Breve explicación de la dinámica y causas principales de cada compañía con sus evidencias clave]
    
    **ECONOMY LH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Economy de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha período]) con una [variación de NPS_diff cabina] puntos respecto a la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "La causa principal fue [hipótesis con datos (drivers, operativa, NCS, verbatims) que la respaldan], complementada por [hipótesis secundarias (si las hubiera)]. Esta [mejora/deterioro] se reflejó especialmente en rutas como [top rutas con NPS y diff], mientras que los perfiles más reactivos incluyen [perfiles específicos]."]
    
    **BUSINESS LH: [Título]**  
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Business de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha período]) con una [variación de NPS_diff cabina] puntos respecto al período anterior.     [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "Los drivers principales fueron [causas con SHAP], impactando especialmente las rutas [rutas específicas] y perfiles [perfiles específicos]."]
    
    **PREMIUM LH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] El segmento Premium de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha]) con [diff cabina] puntos de [variación] vs la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "Las causas dominantes fueron [drivers SHAP], especialmente evidentes en [rutas top] y entre [perfiles reactivos]."]
            `: `<b><u>ECONOMY SH: [Titular]</u></b><br>`

**⚠️ REGLA DE REDUNDANCIA:** 
Si el análisis se centra en una única cabina (nodo raíz = Economy/Business), **OMITE la sección 'DETALLE POR CABINA'**.


    **ECONOMY SH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Economy de SH registró un NPS de [valor cabina] con [variación] pts vs L7d. **Desglose por compañía:** IB obtuvo [NPS IB] ([diff IB] pts) y YW [NPS YW] ([diff YW] pts). [Breve explicación de la dinámica: si ambas suben/bajan = efecto conjunto; si una sube y otra baja = se compensan; si solo una tiene variación = esa domina]
    
    **BUSINESS SH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] El segmento Business de SH registró un NPS de [valor cabina] con [variación] pts vs L7d. **Desglose por compañía:** IB obtuvo [NPS IB] ([diff IB] pts) y YW [NPS YW] ([diff YW] pts). [Breve explicación de la dinámica y causas principales de cada compañía con sus evidencias clave]
    
    **ECONOMY LH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Economy de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha período]) con una [variación de NPS_diff cabina] puntos respecto a la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "La causa principal fue [hipótesis con datos (drivers, operativa, NCS, verbatims) que la respaldan], complementada por [hipótesis secundarias (si las hubiera)]. Esta [mejora/deterioro] se reflejó especialmente en rutas como [top rutas con NPS y diff], mientras que los perfiles más reactivos incluyen [perfiles específicos]."]
    
    **BUSINESS LH: [Título]**  
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Business de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha período]) con una [variación de NPS_diff cabina] puntos respecto al período anterior.     [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "Los drivers principales fueron [causas con SHAP], impactando especialmente las rutas [rutas específicas] y perfiles [perfiles específicos]."]
    
    **PREMIUM LH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] El segmento Premium de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha]) con [diff cabina] puntos de [variación] vs la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "Las causas dominantes fueron [drivers SHAP], especialmente evidentes en [rutas top] y entre [perfiles reactivos]."]
            

**⚠️ IMPORTANTE - CONTENIDO:**
- **INCLUYE TODOS LOS DATOS:** SHAP, rutas (pos/neg), verbatims, perfiles.
- **ATRIBUCIÓN:** Indica siempre a qué segmento pertenece el dato.
- **TERMINOLOGÍA DE COMPARACIÓN:** Usa la fórmula "con respecto a [TEXTO DEL BASELINE REFERENCE]" (ej: "con respecto a la semana anterior").
- **PARA CABINAS SH:** Menciona NPS de IB y YW por separado.

**📊 TRADUCCIÓN DE TERMINOLOGÍA:**
- "SHAP = X" → "X ppts según Explanatory Drivers"
- "ncs_tool" → "incidentes operativos"
- "operative_data_tool" → "métricas operativas"
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-14 to 2025-12-14:
<b>SÍNTESIS EJECUTIVA</b><br>
La red global registró un <b>NPS de 22.8 (–4.7 pts)</b> con respecto a la media de los últimos 7 días. La disminución está liderada por problemas de mishandling de equipaje en Economy SH y se ve acentuada por caídas en Business y Premium de LH. Los avances en SH Business de IB y la mejora de Economy LH no lograron neutralizar el impacto negativo.<br><br>

Los pasajeros de Economy SH enfrentaron un total de 20 incidentes operativos, incluidos 3 casos de equipaje mal manejado que provocaron la pérdida de 27 maletas en el vuelo IB458 (LHR-MAD). A pesar de una puntualidad ajustada del 91.8 % (+1.9 pts) y un Load Factor de 83.5 % (–2.6 pts), los verbatims destacaron fallos en la facturación de equipaje en bodega y demoras en la entrega.<br><br>

En LH Business, la satisfacción cayó por una puntualidad ajustada 5.8 pts por debajo de la media y la combinación de 1 retraso en embarque con 2 incidentes de equipaje. Los comentarios criticaron la mala comunicación en MAD-MIA y el manejo inadecuado de maletas en BOG-MAD. Premium LH registró un NPS de 0.0 (–12.4 pts) tras 1 retraso con autobús sin explicación y 2 incidentes de equipaje (26 maletas AKH y 27 maletas BA458), lo que generó quejas sobre esperas de 1.5 h en MAD-MIA y pesaje manual de equipaje en BOG-MAD.<br><br>

El alza de SH Business en IB, con un <b>NPS de 57.9 (+14.2 pts)</b>, y la recuperación de Economy LH a <b>NPS de 17.6 (+9.0 pts)</b> resultaron insuficientes frente a la magnitud de las incidencias en las otras cabinas.<br><br>

<b><u>DETALLE POR CABINA</u></b><br>

<b><u>ECONOMY SH: Manejo de equipaje impacta percepción</u></b><br>
La cabina Economy de SH registró un <b>NPS de 21.6 (–12.9 pts)</b> con respecto a la media de los últimos 7 días. <b>Desglose por compañía:</b> IB obtuvo un <b>NPS de 15.0 (–17.3 pts)</b> y YW un <b>NPS de 36.0 (–3.5 pts)</b>. Ambas compañías sufrieron mishandling de equipaje con un total de 3 incidentes, incluyendo la pérdida de 27 maletas en IB458 (LHR-MAD). A pesar de una puntualidad ajustada del 91.8 % (+1.9 pts) y un Load Factor de 83.5 % (–2.6 pts), los verbatims mencionaron gestión de equipaje de mano, facturación en bodega y demoras en la entrega de maletas. La ruta más afectada fue LHR-MAD (NPS –31.2, n=16).<br><br>

<b><u>BUSINESS SH: Incidencias operativas en IB superan expectativas</u></b><br>
El segmento Business de SH registró un <b>NPS de 48.3 (+9.0 pts)</b> con respecto a la media de los últimos 7 días. <b>Desglose por compañía:</b> IB obtuvo un <b>NPS de 57.9 (+14.2 pts)</b> y YW un <b>NPS de 30.0 (+0.7 pts)</b>. En IB, el Load Factor estuvo 6.0 pts por debajo de la media y la puntualidad ajustada fue 2.4 pts superior, en un contexto de 20 incidentes operativos que incluyeron 3 casos de equipaje, 6 retrasos/desvíos y 2 cancelaciones. Los verbatims señalaron asientos deteriorados en AMS-MAD y BIO-MAD, demoras de equipaje en BCN-MAD y caos en mostradores de MAD-VIE.<br><br>

<b><u>ECONOMY LH: Recuperación impulsada por menor mishandling</u></b><br>
La cabina Economy de LH registró un <b>NPS de 17.6 (+9.0 pts)</b> con respecto a la media de los últimos 7 días. La causa principal fue la baja incidencia de mishandling de equipaje —solo 2 incidentes operativos y 1 retraso— pese a una caída de la puntualidad ajustada de 5.8 pts. En verbatims se mencionaron casos en EZE-MAD, GIG-MAD y LIM-MAD, con quejas de equipaje no entregado y caos en conexiones MAD-UIO. Los pasajeros Leisure mostraron un NPS de 20.0 (n=30) y Business/Work de 0.0 (n=4).<br><br>

<b><u>BUSINESS LH: Comunicación y equipaje debilitan satisfacción</u></b><br>
La cabina Business de LH registró un <b>NPS de 11.1 (–13.9 pts)</b> con respecto a la media de los últimos 7 días. La puntualidad ajustada cayó 5.8 pts y se sumaron 1 retraso en embarque y 2 incidentes de equipaje. En verbatims se criticó la mala comunicación en MAD-MIA y el manejo de maletas en BOG-MAD, mientras que MAD-UIO y EZE-MAD registraron caídas de –100.0 en NPS.<br><br>

<b><u>PREMIUM LH: Retraso en bus y equipaje generan quejas</u></b><br>
El segmento Premium de LH registró un <b>NPS de 0.0 (–12.4 pts)</b> con respecto a la media de los últimos 7 días. La puntualidad ajustada estuvo 5.8 pts por debajo y se produjeron 1 retraso con bus sin explicación y 2 incidentes de equipaje (26 maletas AKH, 27 maletas BA458). Los verbatims describieron esperas de 1.5 h en MAD-MIA, pesaje manual en BOG-MAD y falta de respuesta en MAD-SCL. Los perfiles más afectados fueron Business/Work (NPS –100.0, n=1) y CodeShare AA (NPS –100.0, n=1).
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
7. **TERMINOLOGÍA:** Preserva la terminología de comparación del interpreter. NO la cambies.
6. NO añadas recomendaciones adicionales
7. Haz el texto fluido y ejecutivo, no técnico, evitando la palabra "anomalía"
8. Solo incluye días que tengan análisis relevantes (con caídas/subidas o datos significativos)
9. Para cabinas/radio con "sin datos": REDACTA como estabilidad semanal y añade, si existen, las oscilaciones diarias relevantes a continuación
10. **CRÍTICO**: Si hay datos en "ANÁLISIS DIARIO SINGLE", DEBES usarlos. NO digas que "no están disponibles" si están presentes en el input.
11. **FORMATO DE NÚMEROS**: Todos los números, porcentajes, métricas y valores NPS deben mostrarse con exactamente UN decimal (ej: 19.8, -4.4, 93.5%)
12. **ATRIBUCIÓN DE SEGMENTO**: Siempre que menciones un dato, indica a qué segmento pertenece (ej: "NPS 19.8 (Economy LH)", "OTP –4.0 pts (Business SH)")
