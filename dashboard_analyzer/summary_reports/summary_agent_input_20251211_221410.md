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
📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 2025-11-24 al 2025-11-30, hemos identificado cuatro causas principales que explican las variaciones de NPS. El resultado global registró una subida de 0.8 pts vs L7d, impulsado por el sólido desempeño de SH que compensó la caída en LH.

El motor de crecimiento estuvo en SH, donde el NPS aumentó +2.8 pts vs L7d gracias a una mejora de la puntualidad y de la experiencia a bordo. En Business SH (Global/SH/Business/IB), Iberia elevó su NPS a 51.90476190476191 (+12.689 ppts según Explanatory Drivers), apoyada en puntuality +4.266 ppts, boarding +3.989 ppts y journey preparation support +1.867 ppts; las métricas operativas mostraron OTP +0.6 pts, misconnect +0.4 pts y mishandling +1.2 pts, mientras que el feedback de clientes creció de 286 a 280 comentarios (+2.1%), destacando “puntualidad” y “servicio a bordo”. En contraste, Business SH Air Europa (Global/SH/Business/YW) cayó a 20.0 (–22.466 ppts según Explanatory Drivers) por un aumento de mishandling de equipaje +0.7 incidentes, sin otros incidentes operativos ni quejas en verbatims, neutralizando parte del avance. Economy SH mantuvo desempeño estable: IB 32.87101248266297 (+2.2 pts) y YW 42.967244701348754 (+4.5 pts), sin drivers operativos ni de producto destacados.

En LH, el NPS descendió –5.7 pts vs L7d por un empeoramiento generalizado de puntualidad y atributos de producto. Economy LH (Global/LH/Economy) cayó a 9.485294117647069 (–5.029 ppts según Explanatory Drivers), con punctuality –1.277 ppts y in flight food and beverage –0.819 ppts; el load factor subió +0.3, no hubo incidentes operativos y el feedback creció +24.5% sin menciones negativas. Las rutas más golpeadas fueron MAD–MCO (–27.3 pts, 11 pax) y JFK–MAD (–17.9 pts, 67 pax), mientras los residentes mostraron un spread de hasta 114.2 pts. Business LH (Global/LH/Business) descendió a 27.45098039215687 (–4.713 ppts), impactada por aircraft interior –2.950 ppts y journey preparation support –2.296 ppts, sin incidentes operativos y con feedback estable; rutas como GUA–MAD lograron 33.3 pts (3 pax) y el perfil CodeShare exhibió spread de 300.0 pts. Premium LH (Global/LH/Premium) experimentó la mayor caída, a 22.80701754385965 (–12.439 ppts), dominada por cabin crew –4.675 ppts, IB Plus loyalty program –2.555 ppts y check-in –2.357 ppts; no se registraron incidentes operativos ni cambios en feedback, y las rutas más afectadas fueron MAD–ORD (–100.0 pts, 1 pax), MAD–NRT (–25.0 pts, 4 pax) y MAD–UIO (–25.0 pts, 4 pax), con residencia Region spread de 222.2 pts.

---

**DETALLE POR CABINA:**

**ECONOMY SH: Mantenimiento de satisfacción**  
La cabina Economy de SH registró un NPS de 36.14495470165573 con +2.88712155707286 pts vs L7d. Desglose por compañía: IB obtuvo 32.87101248266297 (+2.17961354993109 pts) y YW 42.967244701348754 (+4.461491574911964 pts). Ambas compañías experimentaron mejoras sinérgicas, sin drivers operativos ni de producto significativos y manteniendo niveles consistentes de satisfacción.

**BUSINESS SH: Impulso y retroceso compensados**  
El segmento Business de SH alcanzó un NPS de 41.9672131147541 con +1.89501094868912 pts vs L7d. Desglose por compañía: IB obtuvo 51.90476190476191 (+12.689075630252113 pts), impulsada por puntuality +4.266 ppts y boarding +3.989 ppts según Explanatory Drivers; las métricas operativas mostraron OTP +0.6 pts, misconnect +0.4 pts y mishandling +1.2 pts, mientras el feedback de clientes creció de 286 a 280 comentarios (+2.1%), destacando “puntualidad” y “servicio a bordo”. YW cayó a 20.0 (–22.466 ppts según Explanatory Drivers) por mishandling de equipaje +0.7 incidentes, sin otros incidentes operativos ni quejas en verbatims, neutralizando parte del impulso de IB.

**ECONOMY LH: Deterioro de puntualidad y producto**  
La cabina Economy de LH descendió a un NPS de 9.485294117647069 con –5.029473814842392 pts vs L7d. La caída responde a una reducción de punctuality –1.277 ppts y de in flight food and beverage –0.819 ppts según Explanatory Drivers, a pesar de un ligero aumento de load factor +0.3. No se registraron incidentes operativos y el feedback creció +24.5% sin menciones negativas. Las rutas más afectadas fueron MAD–MCO (–27.3 pts, 11 pax) y MAD–ORD (–8.3 pts, 24 pax), y los residentes exhibieron spread de hasta 114.2 pts.

**BUSINESS LH: Experiencia de producto degradada**  
La cabina Business de LH cayó a un NPS de 27.45098039215687 con –4.712762297901609 pts vs L7d. Los drivers principales fueron aircraft interior –2.950 ppts y journey preparation support –2.296 ppts según Explanatory Drivers; no se registraron incidentes operativos y el feedback se mantuvo positivo. Entre las rutas, HAV–MAD mostró 0.0 pts (5 pax) y GUA–MAD 33.3 pts (3 pax). Los vuelos en code-share presentaron spread de 300.0 pts.

**PREMIUM LH: Fuerte retroceso por servicio de cabina**  
El segmento Premium de LH se situó en un NPS de 22.80701754385965 con –12.438884095484617 pts vs L7d. La caída estuvo dominada por cabin crew –4.675 ppts y IB Plus loyalty program –2.555 ppts según Explanatory Drivers; no hubo incidentes operativos y el feedback permaneció estable. Las rutas más afectadas fueron MAD–ORD (–100.0 pts, 1 pax), MAD–NRT (–25.0 pts, 4 pax) y MAD–UIO (–25.0 pts, 4 pax), mientras la residencia Region presentó spread de 222.2 pts.

**ANÁLISIS DIARIO SINGLE:**
📅 2025-11-242025-11-24:
📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 2025-11-24 hemos identificado cinco causas principales que explican las variaciones de NPS. El resultado global mostró una subida de 4.7 pts vs L7d.

En LH se detectaron disparidades operativas por code-share: los vuelos bajo QR registraron un NPS de –40.0 (5 encuestas), mientras que los asociados a AA alcanzaron 80.0 (5 encuestas), sin incidentes operativos y con feedback positivo en 329 comentarios sobre puntualidad, confort y amabilidad. Este contexto heterogéneo qued a diluido por un fuerte desempeño de Economy LH, que elevó su NPS a 23.53 (+11.3 pts vs L7d) gracias a la reducción del Load factor (–3.51 pts vs baseline). La ruta LIM-MAD (NPS 29.4, 17 encuestas) y los pasajeros Leisure (NPS 34.3, 137 encuestas) fueron los más beneficiados, mientras que Business/Work reportó  –21.2 pts (33 encuestas).

La cabina Business LH retrocedió a un NPS de 15.62 (–8.8 pts vs L7d), impactada por una caída de puntualidad (OTP 5.55 pts por debajo del baseline). A pesar de 50 verbatims que elogiaron comodidad y amabilidad y la ausencia de incidentes operativos, la ruta MAD-SCL (NPS 66.7, 3 encuestas) y los clientes Business (NPS 5.6, 18 encuestas) mostraron alta sensibilidad, especialmente en flotas A350 (–14.3, 14 encuestas), A350 next (60.0, 5 encuestas) y A332 (50.0, 6 encuestas), y un outlier en code-share AY (–100.0, 1 encuesta).

En SH, el segmento Business quedó estable con un NPS de 39.53 (+4.2 pts vs L7d) gracias a la compensación entre IB (+15.2 pts, NPS 50.0 vs baseline 34.81) y YW (–15.7 pts, NPS 23.53 vs baseline 39.24). IB fue impulsado por un mix de perfiles (Leisure 68.8, 16 encuestas; Business 20.0, 10 encuestas), rutas como LHR-MAD (NPS 100.0, 1 encuesta) y métricas operativas estables (Load 78.72, –2.05 pts; OTP 93.06, +1.82 pts), con feedback exclusivo de 33 comentarios positivos (“un sueño”, “fantástico servicio”). YW mostró una caída sin drivers operativos claros (OTP +2.29 pts; Load –3.69 pts), sin incidentes operativos, feedback positivo y la ruta SVQ-VLC (NPS 100.0, 1 encuesta).

Economy SH y Premium LH mantuvieron desempeño estable, con un NPS de 34.84 (+0.8 pts vs L7d) y 18.18 (+2.1 pts vs L7d), respectivamente, sin drivers operativos, incidentes ni feedback que alterara la tendencia.

**DETALLE POR CABINA:**

ECONOMY SH: Rendimiento estable  
La cabina Economy de SH registró un NPS de 34.84 con +0.8 pts vs L7d. Desglose por compañía: IB obtuvo 34.89 (+1.0 pts) y YW 34.75 (+0.2 pts). Ambas compañías mostraron estabilidad sin anomalías, diluyendo cualquier variación interna.

BUSINESS SH: Compensación entre IB y YW  
El segmento Business de SH alcanzó un NPS de 39.53 con +4.2 pts vs L7d. IB obtuvo 50.0 (+15.2 pts), impulsado por un mix de perfiles (Leisure 68.8; Business 20.0) y la ruta LHR-MAD (NPS 100.0, 1 encuesta), con métricas operativas estables (Load factor 78.72, –2.05 pts; OTP 93.06, +1.82 pts) y feedback positivo. YW, en cambio, descendió a 23.53 (–15.7 pts) sin drivers operativos claros (OTP +2.29 pts; Load factor –3.69 pts), ni incidentes operativos, con feedback favorable y la ruta SVQ-VLC (NPS 100.0, 1 encuesta).

ECONOMY LH: Alza sostenida  
La cabina Economy de LH registró un NPS de 23.53 (2025-11-24) con +11.3 pts vs L7d. Esta mejora responde a la reducción del Load factor (–3.51 pts vs baseline), que elevó la experiencia en rutas como LIM-MAD (NPS 29.4, 17 encuestas). Los pasajeros Leisure fueron los más beneficiados (NPS 34.3, 137 encuestas), mientras que Business/Work reportó –21.2 pts (33 encuestas).

BUSINESS LH: Impacto de puntualidad  
La cabina Business de LH registró un NPS de 15.62 (2025-11-24) con –8.8 pts vs L7d. La causa principal fue la caída de puntualidad (OTP 5.55 pts por debajo del baseline), afectando especialmente la ruta MAD-SCL (NPS 66.7, 3 encuestas) y a clientes Business (NPS 5.6, 18 encuestas). A pesar de 50 verbatims positivos y ausencia de incidentes operativos, el segmento mostró alta sensibilidad a la puntualidad.

PREMIUM LH: Desempeño consistente  
La cabina Premium de LH mantuvo desempeño estable, con un NPS de 18.18 (2025-11-24) y +2.1 pts vs L7d. No se detectaron cambios significativos en métricas operativas, incidentes operativos ni feedback de clientes.

📅 2025-11-252025-11-25:
📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 25-nov-2025 hemos identificado siete causas principales que explican las variaciones de NPS. El NPS global subió 6.3 pts vs L7d, impulsado por un feedback extraordinariamente positivo en toda la red, aunque coexisten focos de detracción muy localizados.

En toda la red destacó un consenso muy favorable en servicio, relación calidad-precio, comodidad y amabilidad de la tripulación, reflejado en 1 041 verbatims positivos sin incidentes operativos relevantes. Sin embargo, en la ruta MAD-OVD surgió un grupo de detractores de codeshare QR (4 encuestas, NPS –20.7) y AY (1 encuesta, NPS –100.0) a bordo de flota A33ACMI (4 encuestas, NPS –46.9), que generó un impacto localizado negativo.

**ECONOMY SH: Equilibrio compensado entre compañías**  
La cabina Economy de SH registró un NPS de 38.53 (+4.5 pts vs L7d). Desglose por compañía: IB obtuvo 33.23 (–0.6 pts) y YW 51.59 (+17.1 pts). IB sufrió la insatisfacción de clientes de “AMERICA NORTE” en la ruta DUS-MAD (NPS 20.0, 5 encuestas), pese a mejoras en métricas operativas (OTP +1.61 pts vs L7d en Global/SH/Economy/IB; Load Factor –0.45 pts vs L7d en Global/SH/Economy/IB). No hubo incidentes operativos y los verbatims reflejaron comentarios positivos sobre servicio, comodidad y amabilidad. Por su parte, YW se benefició de un excelente feedback en mostradores y gestión de equipaje (161 verbatims), sin desviaciones operativas ni incidentes, con perfiles Business/Work (56.8, 44 encuestas), Leisure (48.8, 82 encuestas), ATR (91.7, 12) y CRJ (47.4, 114).

**BUSINESS SH: Pulso de puntualidad en IB domina**  
Business de SH cerró con NPS 44.74 (+9.4 pts vs L7d). IB alcanzó 53.85 (+19.0 pts) frente a YW con 25.00 (–14.2 pts). El empuje lo aportó IB gracias a mejoría de puntualidad (OTP +1.61 pts vs L7d en Global/SH/Business/IB) y menor ocupación (Load Factor –2.04 pts vs L7d en Global/SH/Business/IB), avalado por verbatims que destacaron puntualidad de tripulación, limpieza de cabina, comodidad y calidad de servicio. YW descendió sin causas operativas claras ni incidentes, pero su efecto negativo quedó mitigado por el fuerte desempeño de IB.

**ECONOMY LH: desempeño estable**  
Economy de LH mantuvo desempeño estable con NPS 15.56 (+3.4 pts vs L7d). No se detectaron cambios significativos ni en métricas operativas, incidentes o verbatims, manteniendo niveles consistentes de satisfacción.

**BUSINESS LH: tensión en puntualidad lastró NPS**  
Business de LH cayó a 19.23 (–5.2 pts vs L7d) principalmente por la baja de puntualidad (OTP 4.97 pts por debajo del L7d en Global/LH/Business). No hubo incidentes operativos; los 44 verbatims destacaron “excelente servicio” y “aviones cómodos” sin mención de retrasos. La ruta EZE-MAD, con NPS 100.0 (2 encuestas), no compensó la percepción negativa de pasajeros Leisure (38.9, 18) y Business/Work (–25.0, 8).

**PREMIUM LH: feedback excepcional impulsa subida**  
Premium de LH se disparó a 38.89 (+22.8 pts vs L7d) pese a registrar la misma baja de puntualidad (OTP 4.97 pts por debajo del L7d en Global/LH/Premium). Sin incidentes operativos, 36 verbatims elogiaron tripulación, limpieza y comodidad. La ruta JFK-MAD alcanzó NPS 100.0 (1 encuesta). Los perfiles más reactivos fueron Business/Work (66.7, 3), Leisure (33.3, 15), las flotas A350 (50.0, 10), A350 next (40.0, 5) y A333 (0.0, 9), con variaciones por región: Europa –33.3 y América Central +100.0.

📅 2025-11-262025-11-26:
📈 **SÍNTESIS EJECUTIVA:**

Durante la jornada del 26 de noviembre de 2025, hemos identificado tres focos de mejora que explican el alza de +8.1 pts en el NPS global vs L7d. El principal impulso provino de LH, complementado por avances relevantes en el segmento Business de SH y en la subcabina Economy YW de SH.

En LH, el NPS pasó de 13.85 a 23.11 (+9.3 pts vs L7d) pese a un OTP de 76.71 (–3.3 pts vs L7d) y un Load Factor de 87.68 (–3.01 pts vs L7d), sin incidentes operativos. El feedback de clientes destacó “embarque organizado”, “tripulación atenta y cordial” y “buen asiento y comida” en 352 comentarios. Esta mejora se reflejó en la ruta DOH–MAD (NPS 33.3, n=3) y fue especialmente sensible entre el perfil Business/Work (NPS 28.1, 29 encuestas) y la flota A350 C (NPS 69.3, 6 encuestas).

El segmento Business de SH elevó su NPS de 35.35 a 48.94 (+13.6 pts vs L7d), con métricas operativas de OTP +2.07 pts vs L7d y Load Factor –1.94 pts vs L7d. En 56 comentarios de feedback de clientes se subrayó la amabilidad y el servicio del personal en Business Class, impulsando la ruta MAD–ORY (NPS 25.0, n=4) y obteniendo alta valoración entre Leisure (NPS 52.6, 19 encuestas) y Business/Work (NPS 46.4, 28 encuestas).

La subcabina Economy YW de SH alcanzó un NPS de 42.35 vs 34.51 (+7.8 pts vs L7d), con OTP +2.46 pts vs L7d y Load Factor +0.16 pts vs L7d. En 215 comentarios de feedback de clientes se enfatizaron la rapidez, el buen trato y la puntualidad, sin alusiones a demoras o equipaje. El único punto crítico fue la ruta LEI–PMI (NPS 0.0, n=2), mientras los perfiles más reactivos fueron Leisure (NPS 43.9, 98 encuestas) y flota CRJ (NPS 43.7, 151 encuestas).

---

📊 DETALLE POR CABINA:

ECONOMY SH: Equilibrio ante variaciones opuestas  
La cabina Economy de SH registró un NPS de 38.53 con +4.5 pts vs L7d. Desglose por compañía: IB alcanzó 36.30 (+2.4 pts según Explanatory Drivers) y YW 42.35 (+7.8 pts según Explanatory Drivers). La subcabina YW impulsó la mejora, mientras IB se mantuvo estable, diluyendo parcialmente el efecto en el consolidado de Economy SH.

BUSINESS SH: Impulso por servicio en Business Class  
Business de SH alcanzó un NPS de 48.94 con +13.6 pts vs L7d. Desglose por compañía: IB 53.33 (+18.5 pts según Explanatory Drivers) y YW 41.18 (+1.9 pts según Explanatory Drivers). La fuerte valoración en IB, sustentada por 56 comentarios señalando amabilidad y servicio del personal, arrastró el resultado global de la cabina, pese a la estabilidad de YW.

ECONOMY LH: Desempeño estable  
La cabina Economy de LH mantuvo desempeño estable con un NPS de 17.61 (+5.4 pts vs L7d). No se detectaron cambios significativos en métricas operativas ni feedback de clientes, manteniendo niveles consistentes de satisfacción.

BUSINESS LH: Sobresale con feedback de alta calidad  
Business de LH registró un NPS de 46.67 con +22.2 pts vs L7d. Aun con un OTP de 76.71 (–3.3 pts vs L7d) y un Load Factor de 87.68 (–3.01 pts vs L7d), el feedback de clientes resaltó el embarque organizado, la tripulación atenta y cordial y la calidad de asiento y comida en 352 comentarios. La ruta DOH–MAD (NPS 33.3, n=3) y el perfil Business/Work (NPS 28.1, 29 encuestas) fueron los más sensibles al alza.

PREMIUM LH: Comodidad y servicio impulsan la mejora  
Premium de LH presentó un NPS de 58.33 con +42.3 pts vs L7d. A pesar de que el OTP cayó 3.3 pts vs L7d, los 18 comentarios de feedback de clientes valoraron positivamente la comodidad, la amabilidad y la puntualidad. Destacó la ruta MAD–SCL (NPS 100.0, n=1) y el perfil Business/Work (NPS 100.0, 2 encuestas) junto a flotas como A333 (NPS 100.0, 2 encuestas) y A350 (NPS 57.1, 7 encuestas).

📅 2025-11-272025-11-27:
📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 27 de noviembre de 2025, hemos identificado tres focos que explican las variaciones de NPS. El indicador global registró una subida de 3.8 pts vs L7d, gracias al empuje de LH y al resultado combinado de las cabinas en SH.

El descenso en Economy de SH (NPS 33.53, –0.5 pts vs L7d) obedece a la caída de Iberia, que pasó de 33.87 a 31.78 (–2.1 pts) en la ruta BRU-MAD (NPS 0.0 con 11 encuestas). A pesar de que Vueling se mantuvo estable (36.67; +2.2 pts vs L7d), la variación de Iberia no llegó a compensarse. Las métricas operativas de SH/Economy/IB fueron estables (Load Factor 89.6 vs 89.81, –0.21 pts; OTP 92.66 vs 91.38, +1.28 pts) y no se registraron incidentes operativos. El feedback de clientes destacó puntualidad, profesionalidad de la tripulación y alta calidad de servicio. Los perfiles más reactivos fueron Leisure (36.8; 229 encuestas), Business/Work (19.4; 93), Fleet A321 (12.5; 56) y CodeShare BA (0.0; 9).

En contraste, Business de SH experimentó un fuerte repunte (NPS 43.90, +8.6 pts vs L7d) impulsado por Iberia, que subió de 34.81 a 47.06 (+12.2 pts), mientras Vueling cayó de 39.24 a 28.57 (–10.7 pts). Iberia obtuvo NPS 100 en la ruta MAD-OPO (2 encuestas), sin desviaciones operativas de relevancia (Load Factor –1.72 pts vs baseline; OTP +1.28 pts) ni incidentes. El feedback resaltó servicio a bordo, atención de tripulación, calidad de la comida, confort y puntualidad. Los pasajeros Business/Work fueron los más satisfechos (53.3; 15 encuestas), seguidos de Leisure (42.1; 20), con flotas y vuelos code-share mostrando amplias oscilaciones en muestras reducidas.

La mejora de LH fue contundente (NPS +6.5 pts vs L7d) gracias a Economy (20.50; +8.3 pts vs L7d) y Business (42.86; +18.4 pts vs L7d), pese al retroceso en Premium (4.17; –11.9 pts vs L7d). Economy LH no mostró indicadores operativos, incidentes o feedback concluyentes para explicar el alza, y no hay rutas específicas ni perfiles asociados. En Business LH, la mejora se reflejó en la ruta MAD-SJO con NPS 100 (2 enc.), y el feedback destacó comodidad, profesionalidad de tripulación y calidad general. Tampoco hubo desviaciones operativas ni incidentes. El descenso en Premium LH carece de drivers claros: no se detectaron cambios operativos ni incidentes, y el feedback fue mayoritariamente positivo. GRU-MAD alcanzó NPS 100 (2 encuestas), mientras que los perfiles Business/Work (–66.7; 9), Leisure (46.7; 15) y varias flotas mostraron elevada dispersión con muestras reducidas.

**DETALLE POR CABINA:**

   
  Economy SH: Insatisfacción de IB dominó la caída  
La cabina Economy de SH registró un NPS de 33.53 (–0.5 pts vs L7d). IB cayó a 31.78 (–2.1 pts vs L7d) debido a la ruta BRU-MAD con NPS 0.0 (11 encuestas), a pesar de que Vueling mantuvo 36.67 (+2.2 pts). Las métricas operativas de IB fueron estables (Load Factor 89.6 vs 89.81, –0.21 pts; OTP 92.66 vs 91.38, +1.28 pts), sin incidentes, y el feedback destacó puntualidad y profesionalidad de la tripulación. Los perfiles Leisure, Business/Work, Fleet A321 y CodeShare BA fueron los más sensibles.

Business SH: Impulso de IB amortiguado por YW  
La cabina Business de SH alcanzó un NPS de 43.90 (+8.6 pts vs L7d). Iberia lideró con 47.06 (+12.2 pts) gracias al NPS 100 en MAD-OPO (2 encuestas), sin impactos operativos ni incidentes, y un feedback focalizado en servicio, confort y calidad de comida. Vueling descendió a 28.57 (–10.7 pts) por quejas menores de catering y separación de cabina, lo que suavizó el alza general. Los pasajeros Business/Work y Leisure fueron los perfiles más reactivos.

Economy LH: Mejora sin indicadores concluyentes  
La cabina Economy de LH mejoró a 20.50 (+8.3 pts vs L7d) sin que las métricas operativas, los incidentes operativos ni el feedback ofrezcan pistas claras. No se identificaron rutas específicas ni perfiles que expliquen el avance, lo que sugiere un efecto de muestra limitada o factores ajenos a los datos disponibles.

Business LH: Fuerte subida impulsada por experiencia premium  
Business LH registró 42.86 (+18.4 pts vs L7d), destacando la ruta MAD-SJO con NPS 100 (2 encuestas). No hubo desviaciones operativas ni incidentes, y el feedback de clientes elogió la comodidad, la profesionalidad de la tripulación y la calidad general, confirmando la percepción de mejora integral en cabina premium.

Premium LH: Caída sin causas operativas claras  
Premium LH cayó a 4.17 (–11.9 pts vs L7d) sin métricas operativas ni incidentes que lo justifiquen, y con feedback mayoritariamente positivo. Aun cuando GRU-MAD alcanzó NPS 100 (2 enc.), los perfiles Business/Work, Leisure y varias flotas mostraron elevada dispersión con pocas encuestas, sin un patrón consistente de insatisfacción.

📅 2025-11-282025-11-28:
📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 28 de noviembre de 2025, hemos identificado cuatro focos principales que explican las variaciones de NPS. El índice global mostró una subida de 1.7 pts vs L7d, balanceando dinámicas muy dispares entre SH y LH.

En SH/Economy coexisten dos movimientos opuestos que se anulan en el agregado. Por un lado, IB registró un leve descenso de –0.7 pts (NPS 33.14 vs L7d 33.87), motivado por la baja satisfacción de clientes Business/Work en la ruta BIO–MAD (NPS 18.2, 11 encuestas), mientras que Leisure se mantuvo en 38.2 (286 encuestas). No se reportaron incidentes operativos y el feedback de clientes (438 comentarios) fue netamente positivo en puntualidad y amabilidad sin quejas recurrentes. Al mismo tiempo, YW experimentó un repunte de +14.5 pts (NPS 48.97 vs L7d 34.51), respaldado por un OTP +2.48 ppts vs L7d y un load factor +0.16 ppts, con feedback muy positivo (179 comentarios) y perfiles Leisure (56.4, 102 encuestas) y Business/Work (31.8, 44). La ruta LEU–MAD (NPS 0.0, 1 encuesta), de baja muestra, no altera este patrón.

El segmento SH/Business fue arrastrado por la fuerte caída de YW, que anotó –49.2 pts (NPS –10.0 vs L7d 39.24). Este deterioro se concentra en la ruta MAD–MRS donde Business/Work en CRJ obtuvo un NPS –37.5 (8 encuestas) y la flota CRJ –15.8 (19). A pesar de un OTP +2.48 ppts vs L7d y sin incidentes operativos, el feedback positivo en puntualidad y servicio (84 comentarios) no alcanzó a contrarrestar la insatisfacción de ese nicho. IB, por su parte, mejoró +15.2 pts (NPS 50.0 vs L7d 34.81) pero su volumen no fue suficiente para mitigar el impacto.

En LH/Economy se observó un deterioro significativo de –10.4 pts (NPS 1.84 vs L7d 12.20), impulsado por la insatisfacción de clientes Business/Work (NPS –16.0, 25 encuestas) y travellers en flotas A321 (–62.5), A333 (–25.9) y A350 C (–25.0). La ruta MAD–NRT registró NPS –20.0 (5), mientras que las métricas operativas mostraron un load factor –3.24 ppts y un OTP –2.48 ppts vs L7d. No hubo incidentes operativos y el feedback (374 comentarios) destacó comodidad y comida, sin menciones a mishandling o misconex.

Aunque LH/Business subió +10.9 pts (NPS 35.29 vs L7d 24.44), la magnitud del descenso en LH/Economy arrastró el balance de LH, y la estabilidad de Premium LH (+0.6 pts, NPS 16.67 vs L7d 16.08) no alcanzó a revertir la tendencia.

DETALLE POR CABINA:

ECONOMY SH: Compensación de fuerzas  
La cabina Economy de SH registró un NPS de 37.83 con +3.81 pts vs L7d. Desglose por compañía: IB obtuvo 33.14 (–0.73 pts) y YW 48.97 (+14.46 pts). Estas variaciones se neutralizaron en el total: la caída de IB, originada en la ruta BIO–MAD (NPS 18.2, 11), y el fuerte repunte de YW, apoyado en OTP +2.48 ppts vs L7d y load factor +0.16 ppts, confluyeron en estabilidad. El feedback de clientes (438 comentarios en IB y 179 en YW) fue mayoritariamente positivo, y no se detectaron incidentes operativos.

BUSINESS SH: Impacto dominado por YW  
El segmento Business de SH registró un NPS de 27.78 con –7.6 pts vs L7d. Desglose por compañía: IB alcanzó 50.0 (+15.19 pts) y YW –10.0 (–49.24 pts). La caída de YW en la ruta MAD–MRS, donde Business/Work en CRJ cayó a –37.5 (8) y flota CRJ a –15.8 (19), fue el factor clave, pese a un OTP +2.48 ppts vs L7d y sin incidentes operativos. Los verbatims (84) resaltaron puntualidad y amabilidad, pero no convencieron al viajero de negocio en CRJ.

ECONOMY LH: Deterioro focalizado  
La cabina Economy de LH sufrió un NPS de 1.84, con –10.36 pts vs L7d, debido a la insatisfacción concentrada en Business/Work (–16.0, 25), flotas A321 (–62.5), A333 (–25.9) y A350 C (–25.0), y la ruta MAD–NRT (–20.0, 5). Las métricas operativas mostraron load factor –3.24 ppts y OTP –2.48 ppts vs L7d. No hubo incidentes operativos y el feedback (374) destacó aspectos positivos de servicio y comida sin referencias a errores de operación.

BUSINESS LH: Subida sin causa operativa clara  
La cabina Business de LH marcó un NPS de 35.29 con +10.85 pts vs L7d. No se detectaron desviaciones significativas en OTP (–2.48 ppts vs L7d) ni load factor (–0.39 ppts), ni hubo incidentes operativos. El feedback de clientes (51 comentarios) resaltó calidad de servicio en cabina, puntualidad, comida y atención en check-in. La ruta MAD–MVD obtuvo NPS 50.0 (2), y entre perfiles destacaron Business/Work 70.0 (10), Leisure 20.8 (24), clientes europeos –50.0 (6), code-share LATAM –100.0 (1) y cabina A350 C –50.0 (4).

PREMIUM LH: Mantuvo desempeño estable  
La cabina Premium de LH registró un NPS de 16.67 con +0.59 pts vs L7d. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción.

📅 2025-11-292025-11-29:
📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 29 de noviembre de 2025, hemos identificado cinco focos de variación que explican las fluctuaciones de NPS. El resultado global mostró una mejora de 3.9 pts vs L7d, sustentada en una sólida percepción de comodidad y atención al cliente que permeó toda la red.

A nivel de LH se observó una marcada dicotomía: Premium ganó 55.4 pts gracias a la brecha de 66.7 pts entre la flota A350 (NPS 100.0, 4 encuestas) y A350 next (NPS 33.3, 4 encuestas), mientras que Business cayó 10.2 pts sin que métricas operativas (OTP –1.75 pts, Load Factor –0.48 pts), incidentes operativos o feedback negativo (servicio impecable, puntualidad, comodidad) expliquen el deterioro. Estos efectos opuestos se compensaron, manteniendo LH dentro de la variación normal (+2.2 pts).

En SH, la cabina Business tiró del segmento con un alza de 21.1 pts, impulsada por mejoras en puntualidad (OTP +2.42 pts vs baseline) y menor ocupación (Load Factor –2.03 pts vs baseline), reforzadas por 60 comentarios que destacaron la atención en tierra y a bordo, el espacio para las piernas y la calidad del servicio. Sin embargo, esta subida de IB (+39.3 pts) convivió con un desplome de YW (–22.6 pts), en un escenario de dominancia interna que resultó en el NPS agregado de SH (+2.4 pts). Economy SH, por su parte, mantuvo desempeño estable (+1.3 pts) gracias a la compensación de una caída mínima en IB (–0.2 pts) y la solidez de YW (+5.0 pts).

**ECONOMY SH: Dinámica compensada entre compañías**  
La cabina Economy de SH registró un NPS de 35.28 con un alza de 1.3 pts vs L7d. Desglose por compañía:  
• IB obtuvo 33.64 (–0.2 pts) con Load Factor –0.14 pts vs baseline y OTP +1.51 pts vs baseline;  
• YW anotó 39.52 (+5.0 pts).  
La caída ligera en IB se diluyó gracias a la estabilidad de YW, manteniendo el segmento dentro de la variación normal. No hubo incidentes operativos. El feedback de clientes (475 comentarios) destacó comodidad de asientos, amabilidad de la tripulación y embarque rápido. En la ruta LHR–MAD el NPS fue 10.7 (28 encuestas), y los perfiles Business vs Leisure mostraron 33.3 (48) vs 33.7 (274).

**BUSINESS SH: Impacto de puntualidad y ocupación**  
El segmento Business de SH alcanzó un NPS de 56.41 con un alza de 21.1 pts vs L7d. Desglose por compañía:  
• IB obtuvo 74.07 (+39.3 pts) con OTP +1.51 pts vs baseline y Load Factor –1.93 pts vs baseline;  
• YW registró 16.67 (–22.6 pts) con OTP +3.10 pts vs baseline y Load Factor –3.41 pts vs baseline.  
En IB, la mejora de puntualidad y la menor ocupación, junto a un feedback de 60 comentarios que valoraron la atención en tierra y a bordo, el espacio para las piernas y la calidad del servicio, explican el impulso positivo. En YW, la caída se produjo sin incidentes operativos y con un feedback mayoritariamente positivo. No se dispusieron datos de rutas ni de segmentación de clientes.

**ECONOMY LH: Mantuvo desempeño estable**  
La cabina Economy de LH registró un NPS de 14.36, con un alza de 2.2 pts vs L7d. No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción.

**BUSINESS LH: Caída sin causa operativa clara**  
La cabina Business de LH presentó un NPS de 14.29 con una caída de 10.2 pts vs L7d. A pesar de una ligera desviación en métricas operativas (OTP –1.75 pts vs baseline, Load Factor –0.48 pts vs baseline), no se registraron incidentes operativos y el feedback (servicio impecable, puntualidad y comodidad) fue positivo. Afectó especialmente la ruta MAD–SJU (NPS 100.0, 1 encuesta), y los perfiles Business/Work vs Leisure mostraron 30.8 (13 encuestas) vs 0.0 (15 encuestas).

**PREMIUM LH: Brecha de flota impulsa subidón**  
El segmento Premium de LH alcanzó un NPS de 71.43 con un alza de 55.4 pts vs L7d. Las métricas operativas se mantuvieron dentro de rangos normales (Load Factor –2.97 pts vs baseline, OTP –1.75 pts vs baseline) y no hubo incidentes operativos. El despegue se explainó por la brecha de segmentación de clientes por flota: A350 consiguió NPS 100.0 (4 encuestas) frente a A350 next con 33.3 (4 encuestas). En la ruta BOG–MAD el NPS fue 100.0 (2 encuestas), y entre los perfiles más reactivos Business/Work alcanzó 100.0 (1 encuesta) y Leisure 66.7 (7 encuestas).

📅 2025-11-302025-11-30:
📈 **SÍNTESIS EJECUTIVA:**

Durante la semana del 2025-11-30, hemos identificado cuatro drivers clave que explican la caída de 6.4 pts en el NPS global (20.98 vs L7d 27.37). El desplome en LH, originado en el mal desempeño de la ruta EZE–MAD, y las dinámicas contrapuestas en SH determinaron el resultado final.

La principal presión negativa provino de LH, donde el segmento Economy cerró en –14.17 pts tras caer –26.4 pts vs L7d y Business cayó –3.2 pts, mientras Premium retrocedió –11.9 pts. A pesar de que las métricas operativas se mantuvieron dentro de umbrales aceptables (Load Factor entre –2.68 y –2.94 pts vs baseline; OTP –0.51 pts vs baseline) y no se registraron incidentes operativos, la ruta EZE–MAD mostró un NPS de –15.1 (38 encuestas), asociado a un deterioro transversal en pasajeros Leisure (NPS –11.8, 261 encuestas) y Business/Work (–0.1, 58 encuestas). El feedback de más de 600 comentarios, aunque mayoritariamente positivo en servicio a bordo y amabilidad, no bastó para contrarrestar esta caída.

En SH, el impacto neto resultó neutral. En Economy, IB descendió a 27.27 (–6.6 pts vs L7d) sin hallazgos operativos o incidentes y con feedback que resaltó puntualidad y amabilidad (ruta MAD–VIE: NPS 0.0, 4 encuestas; perfiles Business/Work 29.2, 49 encuestas; Leisure 26.9, 256; A320neo 34.4, 75; A321 20.0, 75; A320 20.0, 62). Por su parte, YW destacó con 48.68 pts (+14.2 vs L7d) gracias a un OTP +3.79 pts vs baseline, validado en verbatims de puntualidad y eficiencia (ruta CMN–MAD: NPS 25.0, 5; perfiles Leisure 53.3, 138; Business/Work 6.7, 15; ATR 50.0, 26; CRJ 48.4, 127). En Business SH, IB subió a 39.39 (+4.6 pts) con feedback muy favorable, mientras YW cayó a 30.0 (–9.2 pts) pese a un OTP +3.79 pts vs baseline y un Load Factor –3.36 pts vs baseline, sin incidentes reportados (ruta MAD–OPO: NPS nan, 1 encuesta; perfiles Business/Work 66.7, 4; Leisure 14.3, 7; Residence Region EUROPA –33.3, 3).

**DETALLE POR CABINA:**

**ECONOMY SH: Dinámica compensada**  
La cabina Economy de SH registró un NPS de 34.52 con +0.5 pts vs L7d. Desglose por compañía: IB obtuvo 27.27 (–6.6 pts vs L7d) sin métricas operativas ni incidentes operativos y con feedback centrado en puntualidad y amabilidad, destacando la ruta MAD–VIE (NPS 0.0, 4 encuestas) y los perfiles Business/Work (29.2, 49), Leisure (26.9, 256), Fleet A320neo (34.4, 75), A321 (20.0, 75) y A320 (20.0, 62). En contraste, YW alcanzó 48.68 (+14.2 pts vs L7d), impulsado por un OTP +3.79 pts vs baseline validado en verbatims sobre puntualidad y eficiencia, con destaque en CMN–MAD (NPS 25.0, 5 encuestas) y los perfiles Leisure (53.3, 138), Business/Work (6.7, 15), ATR (50.0, 26) y CRJ (48.4, 127).

**BUSINESS SH: Estabilidad con volatilidad interna**  
El segmento Business de SH cerró en 37.21 con +1.9 pts vs L7d. IB alcanzó 39.39 (+4.6 pts) sin variaciones operativas ni incidentes operativos y con verbatims muy positivos. YW, en cambio, cayó a 30.0 (–9.2 pts vs L7d) pese a un OTP +3.79 pts vs baseline y un Load Factor –3.36 pts vs baseline, sin hallazgos en incidentes. La ruta MAD–OPO reportó NPS nan (1 encuesta) y los perfiles más sensibles fueron Business/Work (66.7, 4 encuestas), Leisure (14.3, 7) y Residence Region EUROPA (–33.3, 3).

**ECONOMY LH: Deterioro centrado en EZE–MAD**  
La cabina Economy de LH sufrió un colapso a –14.17 pts tras caer –26.4 pts vs L7d. La ruta EZE–MAD registró NPS –21.2 (34 encuestas), sin métricas operativas críticas (Load Factor –2.94 pts, OTP –0.51 pts vs baseline) ni incidentes operativos, y con feedback de 508 comentarios mayoritariamente positivos. Los perfiles más afectados incluyeron CodeShare QR (–100, 4 encuestas), BA (–71.4, 7), Fleet A321XLR (–63.6, 11), A332 (–24.7), A350 (–23.6), Business/Work (–20.6, 35) y Leisure (–13.2, 227).

**BUSINESS LH: Leve retroceso sin causas claras**  
La cabina Business de LH registró 21.21 pts con –3.2 pts vs L7d. No se detectaron métricas operativas ni incidentes operativos que expliquen la baja, mientras la ruta MAD–SJO alcanzó 50.0 (2 encuestas). El feedback fue uniformemente positivo y los perfiles más reactivos fueron Business/Work (52.9, 17 encuestas), Leisure (–12.5, 16), Fleet A350 C (100.0, 1) y A350 next (50.0, 14).

**PREMIUM LH: Caída moderada con buena percepción**  
El segmento Premium de LH cerró en 4.17 pts tras descender –11.9 pts vs L7d. A pesar de un Load Factor –2.68 pts y OTP –0.51 pts vs baseline sin superar umbrales críticos y ausencia de incidentes operativos, el feedback de 50 comentarios destacó puntualidad, confort y amabilidad. La ruta GRU–MAD mostró 50.0 (2 encuestas) y los perfiles con mayor variabilidad fueron Business/Work (0.0, 6), Leisure (5.6, 18), Fleet A350 (30.0, 10), A333 (0.0, 2) y A350 next (–16.7).

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
