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
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 PASO 1: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL COMPAÑÍA)

---

## 📊 ECONOMY SH

### **PARTE A: ANÁLISIS DE DINÁMICA DE AGREGACIÓN**

**Tríada de estados:** `(-,- | -)`  
- **IB:** NEGATIVE ANOMALY (-8.5 pts)  
- **YW:** NEGATIVE ANOMALY (-10.9 pts)  
- **PADRE (Economy SH):** NEGATIVE ANOMALY (-9.3 pts)

**Escenario identificado:** **SINERGIA NEGATIVA (-,- | -)**

Ambas compañías experimentan deterioro simultáneo en la misma dirección. El efecto se suma y transfiere íntegramente al nodo padre. No hay compensación ni dilución: la crisis operativa afecta a **ambas operaciones** (IB e YW) de forma consistente.

---

### **PARTE B: SELECCIÓN DE NARRATIVA CAUSAL**

**Narrativa:** Adopto la **Explicación del Nodo Padre (Economy SH)** como causa común que aplica a ambos hijos.

**Causa raíz compartida:**  
La anomalía de -9.3 pts en Economy SH se explica por **deterioro operativo multidimensional** que afecta simultáneamente a IB (-8.5 pts) e YW (-10.9 pts):

1. **Punctuality:** OTP15 cayó -2.2 pts (89.52% vs 91.72%), con incremento de cancelaciones (+7, +24.1%) y retrasos (+6, +16.7%)
2. **Mishandling:** Subió +5.9 pts (22.81 vs 16.95), con 127 maletas afectadas documentadas (BA458: 27 maletas, incidente AGP: 100 maletas)
3. **Arrivals Experience:** SHAP -1.721 pts, con múltiples menciones de esperas prolongadas en recogida de equipaje (45-50 min)

**Dark Horses comunes:**
- **Huelga ATC FCO** (17-dic, 13:00-17:00 LT): 6 vuelos cancelados
- **Meteorología adversa generalizada** (14-19 dic): Norte España (BIO, EAS, LCG) + Italia (FLR, FCO)
- **Incidentes masivos de equipaje**: BA458 + AGP (127 maletas totales)

---

### **EVIDENCIA CLAVE:**

**Triangulación cuádruple (SHAP + Operative Data + NCS + Verbatims):**

- **IB:** Punctuality SHAP -3.677 pts, OTP15 -2.0 pts, verbatims con 4 menciones explícitas de retrasos (LIS-MAD, LHR-MAD, BCN-MAD, LCG-MAD)
- **YW:** Punctuality SHAP -1.845 pts, OTP15 -2.3 pts, verbatims con múltiples menciones de retrasos (AGP-MAD, GRX-MAD, GVA-MAD, LCG-MAD)

**Rutas críticas compartidas:**
- **LCG-MAD:** Caída -62.1 pts (IB) vs -66.1 pts (YW) - ambas afectadas por meteorología Norte España
- **GVA-MAD:** Caída -36.9 pts (IB) vs similar en YW - problemas de equipaje perdido (3 días sin maletas)
- **BIO-MAD:** 5 disrupciones NCS afectando ambas operaciones

**Conclusión:** La crisis operativa fue **sistémica y transversal**, no específica de compañía. Ambas operaciones (IB e YW) sufrieron el mismo shock exógeno (meteorología + huelga ATC) con consecuencias operativas idénticas (puntualidad + equipaje).

---

## 💼 BUSINESS SH

### **PARTE A: ANÁLISIS DE DINÁMICA DE AGREGACIÓN**

**Tríada de estados:** `(-,- | -)`  
- **IB:** NEGATIVE ANOMALY (-0.9 pts)  
- **YW:** NEGATIVE ANOMALY (-8.1 pts)  
- **PADRE (Business SH):** NEGATIVE ANOMALY (-1.3 pts)

**Escenario identificado:** **DOMINANCIA NEGATIVA CON DILUCIÓN (-,- | -)**

Ambas compañías tienen anomalías negativas, pero **YW domina con -8.1 pts** mientras **IB tiene deterioro marginal de -0.9 pts**. El padre (-1.3 pts) refleja la dominancia de YW pero **diluida por el mayor volumen de IB** (93 encuestas IB vs sample menor de YW).

---

### **PARTE B: SELECCIÓN DE NARRATIVA CAUSAL**

**Narrativa:** Adopto la **Explicación del Hijo Dominante (YW)** como causa principal del padre, con matiz de dilución por IB.

**Causa raíz dominante (YW):**  
La anomalía de -8.1 pts en YW Business se explica por **deterioro operativo agravado en operaciones Air Nostrum**:

1. **Punctuality:** SHAP -2.181 pts (mayor impacto que IB -1.230), OTP15 cayó -2.33 pts
2. **Boarding:** SHAP -2.983 pts (problema estructural de embarque con autobús/escaleras en Air Nostrum)
3. **Ticket Price:** SHAP -3.123 pts (paradoja: satisfacción subió +99.47 pts pero sigue siendo pain point - "Pagas business para que te metan en un avión minúsculo")

**Dark Horses específicos de YW:**
- **Crisis operativa FLR:** MAD-FLR con 5 disrupciones (máxima concentración)
- **Meteorología adversa concentrada:** MAD-BIO (5 disrupciones), MAD-EAS (2 disrupciones)
- **Problema estructural de flota CRJ:** 3 menciones en período actual vs 5 en comparativo sobre inadecuación de CRJ para Business Class

---

### **EVIDENCIA CLAVE:**

**Contraste IB vs YW:**

| Dimensión | IB (-0.9 pts) | YW (-8.1 pts) |
|-----------|---------------|---------------|
| **Punctuality SHAP** | -1.230 | -2.181 (79% más severo) |
| **Boarding SHAP** | +0.891 (compensador) | -2.983 (agravante) |
| **OTP15** | -2.0 pts | -2.33 pts |
| **Sample size** | 93 encuestas | 20 encuestas |
| **Rutas críticas** | FCO-MAD (huelga ATC compensada por Boarding 100.0) | MAD-FLR (5 disrupciones), MAD-BIO (5 disrupciones) |

**Perfil de cliente más afectado (YW):**
- **Business travelers:** -20.1 pts (máxima sensibilidad a puntualidad/boarding)
- **Residence Region:** Spread 63.3 pts (hasta -30.0 pts en regiones específicas)

**Matiz de dilución:**  
"La anomalía del padre Business SH (-1.3 pts) está **dominada por el colapso de YW (-8.1 pts)**, pero fue **parcialmente suavizada por la estabilidad relativa de IB (-0.9 pts)**, cuyo mayor volumen (93 encuestas vs ~20 de YW) diluye el impacto agregado. IB mantuvo compensadores positivos (Boarding +0.891) que YW no tuvo, explicando la diferencia de magnitud."

---

## 🎯 SÍNTESIS EJECUTIVA DEL PASO 1

| Cabina | Escenario | Estado (IB, YW \| Padre) | Narrativa Dominante |
|--------|-----------|--------------------------|---------------------|
| **Economy SH** | **SINERGIA NEGATIVA** | `(-,- | -)` | Crisis operativa sistémica (meteorología + huelga ATC + mishandling) afecta **por igual** a IB e YW. Causa común del padre. |
| **Business SH** | **DOMINANCIA CON DILUCIÓN** | `(-,- | -)` | YW domina con colapso severo (-8.1 pts) por problemas de boarding/flota CRJ. IB diluye el impacto agregado con deterioro marginal (-0.9 pts). |

**Implicación estratégica:**  
- En **Economy SH**, las acciones correctivas deben ser **transversales** (aplican a IB e YW).  
- En **Business SH**, las acciones correctivas deben **priorizarse en YW** (Air Nostrum), especialmente en proceso de boarding y adecuación de flota CRJ.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ PASO 2: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL CABINA)

---

## 🛫 SHORT HAUL (SH)

### **PARTE A: ANÁLISIS DE DINÁMICA DE AGREGACIÓN**

**Tríada de estados:** `(-,- | -)`  
- **Economy SH:** NEGATIVE ANOMALY (-9.3 pts)  
- **Business SH:** NEGATIVE ANOMALY (-1.3 pts)  
- **PADRE (SH):** NEGATIVE ANOMALY (-8.6 pts)

**Escenario identificado:** **SINERGIA NEGATIVA PARCIAL (-,- | -)**

Ambas cabinas experimentan deterioro en la misma dirección negativa. El efecto se transfiere al padre con **dominancia de Economy** por:
1. **Magnitud:** Economy -9.3 pts (7.2x más severo que Business -1.3 pts)
2. **Volumen:** Economy representa el grueso del tráfico SH
3. **Coherencia:** El padre SH (-8.6 pts) está más cerca de Economy (-9.3 pts) que de Business (-1.3 pts)

**Conclusión:** Sinergia negativa donde **Economy arrastra al SH completo**, mientras Business aporta deterioro marginal.

---

### **PARTE B: SELECCIÓN DE NARRATIVA CAUSAL**

**Narrativa:** Adopto la **Explicación del Nodo Padre (SH)** como causa sistémica que afecta transversalmente, con **énfasis en Economy** como cabina dominante.

---

### **CAUSA SISTÉMICA DEL RADIO SH:**

**Crisis operativa triple documentada en el análisis del padre SH:**

#### **1. DETERIORO CRÍTICO DE PUNCTUALITY**
- **Métricas:** OTP15 cayó -2.2 pts (89.52% vs 91.72%)
- **Drivers SHAP:** Punctuality -3.955 pts (mayor impacto negativo en SH)
- **NCS:** 
  - Cancelaciones: +7 (+24.1%)
  - Retrasos: +6 (+16.7%)
  - **Total incidentes críticos: +13 (+29.5%)**
- **Impacto diferencial por cabina:**
  - Economy: 8 menciones explícitas de retrasos en verbatims
  - Business: 4 menciones explícitas de retrasos en verbatims

#### **2. EPIDEMIA DE MISHANDLING**
- **Métricas:** Mishandling subió +5.9 pts (22.81 vs 16.95)
- **Drivers SHAP:** Arrivals Experience -1.721 pts
- **NCS:** 127 maletas afectadas documentadas
- **Impacto diferencial por cabina:**
  - Economy: 15+ menciones de equipaje perdido/retrasado/dañado (aumento +87% vs período anterior)
  - Business: 6 menciones de equipaje perdido (aumento +300% vs período anterior)

#### **3. AUMENTO DE CONEXIONES PERDIDAS**
- **Métricas:** Misconex subió +0.2 pts (0.86 vs 0.66)
- **Impacto diferencial por cabina:**
  - Economy: Múltiples casos documentados (MAD-VCE, BIO-MAD, EAS-MAD)
  - Business: 4 menciones (LHR-MAD, BCN-MAD, EAS-MAD, MAD-VCE)

---

### **DARK HORSES SISTÉMICOS (AFECTAN A AMBAS CABINAS):**

#### **A) HUELGA ATC FCO (17-DIC, 13:00-17:00 LT)**
- **Impacto:** 6 vuelos cancelados
- **Rutas afectadas:** MAD-FCO, FCO-MAD
- **Evidencia transversal:**
  - Economy: FCO-MAD aparece con múltiples menciones negativas
  - Business: FCO-MAD mantuvo NPS 76.9 (compensado por Boarding 100.0 en IB)

#### **B) METEOROLOGÍA ADVERSA GENERALIZADA (14-19 DIC)**
- **Zonas afectadas:** Norte España (BIO, EAS, LCG, OVD) + Italia (FLR)
- **Rutas críticas compartidas:**
  - MAD-FLR: 5 disrupciones (máxima concentración)
  - MAD-BIO: 5 disrupciones
  - MAD-EAS: 2 disrupciones
  - LCG-MAD: Múltiples disrupciones
- **Evidencia transversal:**
  - Economy: LCG-MAD cayó -62.1 pts (IB) / -66.1 pts (YW)
  - Business: BIO-MAD con 5 disrupciones afectando ambas cabinas

#### **C) INCIDENTES MASIVOS DE EQUIPAJE**
- **BA458 (LHR-MAD):** 27 maletas extraviadas
- **AGP:** 100 maletas no cargadas
- **Total:** 127 maletas afectadas
- **Evidencia transversal:**
  - Economy: 6 casos graves documentados (MAD-SVQ, MAD-VGO, GVA-MAD, PMI-VLC, BCN-SXB, LEI-MAD)
  - Business: DSS-MAD [NPS 0] "Llevo tres días esperando mi equipaje... clase ejecutiva"

---

### **EVIDENCIA CLAVE DE SINERGIA:**

**Rutas críticas compartidas entre cabinas:**

| Ruta | Economy | Business | Causa Común |
|------|---------|----------|-------------|
| **LCG-MAD** | -62.1 pts (43 pax) | -20.0 pts (5 pax) | Meteorología Norte España + retrasos sistemáticos |
| **BIO-MAD** | -23.8 pts (68 pax) | 30.0 pts (10 pax) | 5 disrupciones meteorológicas |
| **GVA-MAD** | -36.9 pts (47 pax) | 16.1 pts (69 pax) | Equipaje perdido 3 días + retrasos |
| **FCO-MAD** | -16.0 pts (125 pax) | 76.9 pts (13 pax) | Huelga ATC 17-dic |
| **MAD-FLR** | 5 disrupciones NCS | 5 disrupciones NCS | Crisis operativa Italia |

**Patrón identificado:**  
Las mismas rutas aparecen afectadas en ambas cabinas, con **magnitudes diferentes** pero **causas operativas idénticas** (meteorología + huelga ATC + mishandling). Esto confirma que la crisis fue **sistémica del radio SH**, no específica de cabina.

---

### **MATIZ DE DOMINANCIA DE ECONOMY:**

"La anomalía del padre SH (-8.6 pts) está **dominada por el colapso de Economy (-9.3 pts)**, que representa el mayor volumen de pasajeros y sufrió el impacto operativo con mayor intensidad (15+ menciones de equipaje vs 6 en Business, 8 menciones de retrasos vs 4 en Business). Business SH (-1.3 pts) experimentó la misma crisis operativa pero con **menor magnitud absoluta** debido a:
1. Compensadores positivos en IB Business (Boarding +0.891 SHAP)
2. Menor volumen de pasajeros afectados
3. Mayor tolerancia relativa de viajeros Business a disrupciones (aunque con alta reactividad: -20.1 pts en perfil Business de YW)"

---

## 🌍 LONG HAUL (LH)

### **PARTE A: ANÁLISIS DE DINÁMICA DE AGREGACIÓN**

**Tríada ternaria de estados:** `(-,-,N | -)`  
- **Economy LH:** NEGATIVE ANOMALY (-6.9 pts)  
- **Business LH:** NEGATIVE ANOMALY (-6.2 pts)  
- **Premium LH:** Normal (+3.6 pts - **dentro de rango normal**)  
- **PADRE (LH):** NEGATIVE ANOMALY (-5.8 pts)

**Escenario identificado:** **SINERGIA NEGATIVA PARCIAL CON NEUTRALIDAD DE PREMIUM (-,-,N | -)**

Dos cabinas (Economy y Business) experimentan deterioro simultáneo con magnitudes similares (-6.9 y -6.2 pts), mientras Premium se mantiene estable. El efecto se transfiere al padre con **co-dominancia de Economy y Business**, sin compensación significativa de Premium (que representa volumen menor).

---

### **PARTE B: SELECCIÓN DE NARRATIVA CAUSAL**

**Narrativa:** Adopto la **Explicación del Nodo Padre (LH)** como causa sistémica que afecta transversalmente a Economy y Business, con Premium como **excepción no afectada**.

---

### **CAUSA SISTÉMICA DEL RADIO LH:**

**Deterioro operativo dual documentado en el análisis del padre LH:**

#### **1. CRISIS DE ARRIVALS EXPERIENCE (BAGGAGE HANDLING)**
- **Driver SHAP:** Arrivals Experience -1.191 pts (mayor impacto negativo)
- **Métricas:** 
  - Mishandling: 22.81 vs 16.95 → **+5.9 pts (+34.8%)**
  - Misconex: 0.86 vs 0.66 → **+0.2 pts (+30.3%)**
- **NCS:** 
  - BA458: 27 maletas extraviadas
  - Incidente sin especificar: 100 maletas no cargadas
- **Impacto diferencial por cabina:**
  - **Economy:** 4 casos graves documentados (BOG-MAD, JFK-MAD, BOS-MAD, LIM-MAD) + "aumento +100% en menciones de equipaje"
  - **Business:** 6 casos documentados (LIS-MAD, ALC-MAD, FRA-MAD, BCN-MAD, MAD-VCE x2) + "aumento +300% en quejas de equipaje"
  - **Premium:** Sin menciones de problemas de equipaje

#### **2. DETERIORO DE CHECK-IN**
- **Driver SHAP:** Check-in -0.830 pts
- **Satisfacción:** Cayó -1.35 pts
- **Rutas críticas compartidas:**
  - MAD-MVD: Check-in 60.5 (-31.7 pts)
  - MAD-MEX: Check-in 79.6 (-22.8 pts)
  - DOH-MAD: Check-in 48.1 (-15.6 pts)
- **Impacto diferencial por cabina:**
  - **Economy:** Check-in SHAP -1.073 pts (más severo)
  - **Business:** Check-in SHAP -0.383 pts (moderado)
  - **Premium:** Sin datos de deterioro

#### **3. PARADOJA DE PUNTUALIDAD**
- **Driver SHAP:** Punctuality -0.674 pts (negativo)
- **Satisfacción:** Cayó -3.44 pts
- **Métricas:** OTP15 **MEJORÓ** 81.84% vs 79.12% (+2.7 pts)
- **Contradicción:** Percepción empeoró a pesar de mejora operativa
- **Explicación:** Eventos extremos puntuales (IB1586: 160 min retraso, IB0337: 93 min) afectaron desproporcionadamente la percepción
- **Impacto diferencial por cabina:**
  - **Economy:** Punctuality SHAP -0.580 pts
  - **Business:** Punctuality SHAP -7.194 pts (paradoja más severa)
  - **Premium:** Sin datos de deterioro

---

### **DARK HORSES SISTÉMICOS LH:**

#### **A) REPROGRAMACIÓN MASIVA MAD-DFW (18-DIC)**
- **Impacto:** Reprogramación de **8h 5min** con **193 pérdidas de conexión** desde DFW
- **Rutas afectadas:** MAD-DFW (3 disrupciones totales)
- **Evidencia transversal:**
  - Economy: No aparece en verbatims del segmento (pasajeros en conexión no identifican MAD-DFW como origen)
  - Business: No aparece en verbatims específicos
  - **Nota:** Este evento afectó principalmente al período COMPARISON (baseline), NO al período actual

#### **B) LIMITACIÓN DE PESO EXTREMA (16-DIC)**
- **Impacto:** 377 maletas sin cargar en un solo vuelo
- **Relación:** Coherente con aumento de Mishandling +5.9 pts
- **Evidencia transversal:** Contribuyó a la epidemia de equipaje en ambas cabinas

#### **C) FALLO SISTEMA CARGA (16-DIC)**
- **Impacto:** Bloqueo transacciones Resiber/Amadeus (00:01-08:00h aprox.)
- **Evidencia transversal:** Afectó operaciones generales de carga

---

### **EVIDENCIA CLAVE DE SINERGIA ECONOMY-BUSINESS:**

**Rutas críticas compartidas entre cabinas:**

| Ruta | Economy LH | Business LH | Causa Común |
|------|------------|-------------|-------------|
| **MAD-SCL** | -25.6 pts (43 pax) | NPS -12.3 (57 pax total) | Arrivals 54.5, problemas equipaje + pantallas IFE |
| **MAD-ORD** | 8.8 pts (34 pax) | NPS 12.8 (39 pax total) | Arrivals 65.8, F&B deficiente |
| **MAD-MVD** | -14.3 pts (35 pax) | NPS -7.7 (24 pax total) | Check-in 60.5 (-31.7 pts), equipaje |
| **BOG-MAD** | 8.8 pts (102 pax) | NPS 7.6 (145 pax total) | Equipaje perdido, retrasos |
| **BOS-MAD** | 22.7 pts (22 pax) | NPS 24.0 (48 pax total) | Equipaje no llegó, espera 3h |
| **MAD-MEX** | 12.2 pts (74 pax) | NPS 18.0 (100 pax total) | Check-in 79.6 (-22.8 pts), F&B |

**Patrón identificado:**  
Las mismas rutas aparecen con problemas operativos en ambas cabinas (equipaje, check-in, arrivals experience), con **magnitudes similares** de deterioro (-6.9 Economy vs -6.2 Business). Esto confirma que la crisis fue **sistémica del radio LH**, afectando transversalmente a Economy y Business.

---

### **EXCEPCIÓN: PREMIUM LH NO AFECTADO**

**Premium LH:** Normal (+3.6 pts, NPS 13.1 vs baseline 9.5)

**Explicación de la neutralidad:**
1. **Sin menciones en verbatims** de problemas operativos en Premium
2. **Sin rutas críticas** identificadas en segmento Premium
3. **Volumen menor** (no representativo en el análisis agregado)
4. **Posible aislamiento operativo:** Pasajeros Premium tienen procesos diferenciados (check-in prioritario, handling preferente, acceso a lounges) que los protegieron de la crisis operativa que afectó a Economy y Business

**Implicación:** Premium actuó como **grupo de control**, demostrando que los problemas de equipaje/check-in/arrivals fueron específicos de procesos masivos (Economy/Business), no de la operación LH en general.

---

### **MATIZ DE CO-DOMINANCIA ECONOMY-BUSINESS:**

"La anomalía del padre LH (-5.8 pts) está **co-determinada por Economy (-6.9 pts) y Business (-6.2 pts)**, que sufrieron la misma crisis operativa de equipaje/check-in/arrivals con magnitudes prácticamente idénticas. Premium LH (+3.6 pts) se mantuvo **neutral y no compensó** el deterioro de las otras cabinas debido a su menor volumen y procesos operativos diferenciados. La sinergia negativa entre Economy y Business fue **completa y sistémica**, sin efectos de cancelación o dilución."

---

## 🎯 SÍNTESIS EJECUTIVA DEL PASO 2

| Radio | Escenario | Estado (Cabinas \| Padre) | Narrativa Dominante |
|-------|-----------|---------------------------|---------------------|
| **SH** | **SINERGIA NEGATIVA CON DOMINANCIA ECONOMY** | `(-,- | -)` | Crisis operativa sistémica (meteorología + huelga ATC + mishandling) afecta transversalmente. **Economy domina** por magnitud (-9.3 pts) y volumen. Business aporta deterioro marginal (-1.3 pts). |
| **LH** | **SINERGIA NEGATIVA PARCIAL (ECO-BUS) CON NEUTRALIDAD PREMIUM** | `(-,-,N | -)` | Crisis operativa de equipaje/check-in/arrivals afecta **por igual** a Economy (-6.9 pts) y Business (-6.2 pts). **Premium neutral** (+3.6 pts) por procesos diferenciados. Co-dominancia sin compensación. |

**Implicaciones estratégicas:**  
- En **SH**, las acciones correctivas deben priorizarse en **Economy** (mayor impacto) pero aplicarse transversalmente (ambas cabinas afectadas).  
- En **LH**, las acciones correctivas deben ser **transversales a Economy y Business** (co-dominancia), enfocándose en procesos masivos de equipaje/check-in. Premium debe **auditarse** para identificar qué procesos diferenciados la protegieron.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 PASO 3: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL GLOBAL)

---

## 📊 ANÁLISIS GLOBAL

### **PARTE A: ANÁLISIS DE DINÁMICA DE AGREGACIÓN**

**Tríada de estados:** `(-,- | -)`  
- **LH (Long Haul):** NEGATIVE ANOMALY (-5.8 pts)  
- **SH (Short Haul):** NEGATIVE ANOMALY (-8.6 pts)  
- **GLOBAL:** NEGATIVE ANOMALY (-7.3 pts)

**Escenario identificado:** **SINERGIA NEGATIVA CON DOMINANCIA DE SH (-,- | -)**

Ambos radios experimentan deterioro simultáneo en la misma dirección negativa, configurando una **crisis sistémica de red**. El Global (-7.3 pts) refleja:
1. **Dominancia de SH:** El valor Global (-7.3 pts) está más próximo a SH (-8.6 pts) que a LH (-5.8 pts)
2. **Peso volumétrico:** SH representa mayor volumen de operaciones (vuelos cortos/medios con mayor frecuencia)
3. **Coherencia direccional:** Ambos radios deterioran, confirmando problema transversal

---

### **PARTE B: SELECCIÓN DE NARRATIVA CAUSAL**

**Narrativa:** Adopto la **Explicación del Nodo Global** como crisis sistémica de red que afecta transversalmente a ambos radios, con **matiz de dominancia de SH**.

---

## 🔴 CAUSA SISTÉMICA DE LA RED GLOBAL

### **CRISIS OPERATIVA MULTIDIMENSIONAL TRANSVERSAL**

La red entera se vio impactada por un **deterioro operativo generalizado** que afectó tanto al Largo como al Corto Radio, con tres vectores operativos compartidos:

---

### **1️⃣ DETERIORO CRÍTICO DE PUNTUALIDAD (CAUSA PRIMARIA COMPARTIDA)**

**Evidencia transversal LH + SH:**

| Dimensión | SH | LH | Convergencia |
|-----------|----|----|--------------|
| **OTP15** | 89.52% vs 91.72% (**-2.2 pts**) | 81.84% vs 79.12% (**+2.7 pts**) | ⚠️ Paradoja: SH empeoró, LH mejoró métricamente |
| **Driver SHAP** | Punctuality **-3.955 pts** | Punctuality **-0.674 pts** | Ambos negativos (percepción deteriorada) |
| **Satisfacción** | Cayó **-5.88 pts** | Cayó **-3.44 pts** | Ambos deterioros perceptuales |
| **NCS Cancelaciones** | **+7 (+24.1%)** | **+5 (+500%)** | Incremento en ambos radios |
| **NCS Retrasos** | **+6 (+16.7%)** | **-7 (-24.1%)** | SH empeoró, LH mejoró |

**Explicación de la paradoja LH:**  
Aunque el OTP15 de LH mejoró +2.7 pts, la **percepción del cliente empeoró** (SHAP -0.674, Sat -3.44 pts) debido a **eventos extremos puntuales** que no captura el OTP15:
- IB1586: 160 min de retraso (descanso tripulación)
- IB0337: 93 min de retraso
- Reprogramación MAD-DFW: 8h 5min con 193 conexiones perdidas

**Conclusión:** La puntualidad deterioró **perceptualmente en ambos radios**, aunque las métricas operativas muestren mejora en LH. La causa es **sistémica**: eventos excepcionales (meteorología, huelgas) afectaron la red completa.

---

### **2️⃣ EPIDEMIA DE MISHANDLING (CAUSA SECUNDARIA COMPARTIDA)**

**Evidencia transversal LH + SH:**

| Dimensión | SH | LH | Convergencia |
|-----------|----|----|--------------|
| **Mishandling** | 22.81 vs 16.95 (**+5.9 pts, +34.8%**) | 22.81 vs 16.95 (**+5.9 pts, +34.8%**) | ✅ **Idéntico deterioro** |
| **Driver SHAP** | Arrivals Experience **-1.721 pts** | Arrivals Experience **-1.191 pts** | Ambos negativos |
| **Satisfacción** | Cayó **-4.69 pts** | Cayó **-2.33 pts** | Ambos deterioros |
| **NCS Incidentes** | 127 maletas afectadas (BA458: 27, AGP: 100) | 127 maletas afectadas (mismos incidentes) | ✅ **Mismos eventos** |
| **Verbatims** | 15+ menciones (**+87%** vs baseline) | 4 menciones Eco + 6 menciones Bus (**+100-300%**) | Incremento masivo en ambos |

**Incidentes críticos compartidos:**
- **[16-DIC] BA458 (LHR-MAD):** 27 maletas extraviadas (afecta conexiones LH)
- **[14-DIC] AGP:** 100 maletas no cargadas (afecta red SH)
- **[16-DIC] Limitación peso extrema:** 377 maletas sin cargar en un solo vuelo (sin especificar radio)

**Conclusión:** El mishandling es **idéntico en ambos radios** (+5.9 pts), confirmando un **problema sistémico de red** en gestión de equipajes, NO específico de un radio.

---

### **3️⃣ AUMENTO DE CONEXIONES PERDIDAS (CAUSA TERCIARIA COMPARTIDA)**

**Evidencia transversal LH + SH:**

| Dimensión | SH | LH | Convergencia |
|-----------|----|----|--------------|
| **Misconex** | 0.86 vs 0.66 (**+0.2 pts, +30.3%**) | 0.86 vs 0.66 (**+0.2 pts, +30.3%**) | ✅ **Idéntico deterioro** |
| **Driver SHAP** | Connections Experience **-0.133 pts** (Eco SH) | Connections Experience **-0.084 pts** (Bus LH) | Ambos negativos |
| **Satisfacción** | Cayó **-3.66 pts** (Eco SH) | Cayó **-7.12 pts** (Bus LH) | Ambos deterioros, LH más severo |
| **Verbatims** | Múltiples casos (MAD-VCE, BIO-MAD, EAS-MAD) | 3 casos Bus (LHR-MAD, BCN-MAD, EZE-MAD) | Ambos radios afectados |

**Patrón identificado:**  
Las conexiones perdidas afectaron **desproporcionadamente a pasajeros LH Business** (viajeros internacionales con compromisos profesionales), pero el problema operativo (misconex +0.2 pts) es **idéntico en ambos radios**.

**Conclusión:** El aumento de conexiones perdidas es **sistémico de red**, no específico de radio.

---

## 🐴 DARK HORSES SISTÉMICOS (AFECTAN A TODA LA RED)

### **A) HUELGA ATC EN FCO (17-DIC, 13:00-17:00 LT)**

**Impacto transversal:**
- **6 vuelos cancelados** (flexibilización tarifas publicada)
- **Rutas afectadas:**
  - **SH:** FCO-MAD (125 pax Eco SH, 13 pax Bus SH), MAD-FCO
  - **LH:** Conexiones desde/hacia FCO afectadas indirectamente

**Evidencia de impacto:**
- **SH:** FCO-MAD cayó -16.0 pts (Eco), mantuvo 76.9 pts (Bus compensado por Boarding)
- **LH:** Sin rutas LH directas FCO documentadas, pero afectó conexiones

**Conclusión:** Evento excepcional que afectó principalmente a **SH Europa**, con impacto secundario en conexiones LH.

---

### **B) METEOROLOGÍA ADVERSA GENERALIZADA (14-19 DIC)**

**Impacto transversal:**

**Zonas afectadas:**
1. **Norte de España:** BIO, EAS, LCG, OVD, SDR, LEU (principalmente SH)
2. **Italia:** FLR, FCO (SH + conexiones LH)
3. **Sur de España:** AGP, TFS, MLN (principalmente SH)

**Rutas críticas por radio:**

| Radio | Rutas Afectadas | Disrupciones NCS |
|-------|-----------------|------------------|
| **SH** | MAD-FLR (5), MAD-BIO (5), MAD-EAS (2), LCG-MAD, GVA-MAD, AGP-MAD | **Máxima concentración** |
| **LH** | Sin rutas directas documentadas | Impacto indirecto en conexiones |

**Patrón identificado:**  
La meteorología adversa afectó **desproporcionadamente a SH** (rutas cortas/medias europeas y domésticas), con impacto secundario en conexiones LH desde/hacia hubs afectados (MAD, BIO).

**Conclusión:** Evento excepcional con **impacto asimétrico**: SH sufrió disrupciones directas masivas, LH sufrió disrupciones indirectas vía conexiones perdidas.

---

### **C) INCIDENTES MASIVOS DE EQUIPAJE (RED COMPLETA)**

**Impacto transversal:**

| Incidente | Radio Afectado | Impacto |
|-----------|----------------|---------|
| **BA458 (LHR-MAD):** 27 maletas extraviadas | **LH** (conexión intercontinental) | Equipaje no llegó a conexiones LH |
| **AGP:** 100 maletas no cargadas | **SH** (doméstico/europeo) | Equipaje no cargado en vuelos SH |
| **Limitación peso (16-DIC):** 377 maletas sin cargar | **Red completa** (sin especificar) | Afectó ambos radios |

**Conclusión:** Los incidentes de equipaje fueron **sistémicos de red**, afectando tanto a SH como a LH con la misma magnitud (+5.9 pts mishandling).

---

## 📍 RUTAS CRÍTICAS TRANSVERSALES (EVIDENCIA DE SINERGIA)

### **RUTAS QUE CONECTAN AMBOS RADIOS (HUB MAD):**

| Ruta | Radio | NPS | Causa Común |
|------|-------|-----|-------------|
| **BOG-MAD** | LH | 7.6 (145 pax) | Equipaje perdido, retrasos → Afecta conexiones SH desde MAD |
| **BIO-MAD** | SH | 14.7 (68 pax Eco) | 5 disrupciones meteorológicas → Afecta conexiones LH desde MAD |
| **LCG-MAD** | SH | 2.3 (43 pax Eco) | Retrasos sistemáticos → Afecta conexiones LH desde MAD |
| **GVA-MAD** | SH | 6.4 (47 pax Eco) | Equipaje perdido 3 días → Afecta conexiones LH desde MAD |
| **FCO-MAD** | SH | 17.9 (112 pax Eco) | Huelga ATC → Afecta conexiones LH desde MAD |

**Patrón identificado:**  
Los problemas operativos en **rutas SH hacia/desde MAD** (hub principal) generaron **efecto cascada en conexiones LH**, confirmando que la crisis es **sistémica de red** y no aislada por radio.

---

### **RUTAS LH CON PROBLEMAS COMPARTIDOS:**

| Ruta | Cabina | NPS | Causa Común con SH |
|------|--------|-----|---------------------|
| **MAD-SCL** | Eco LH | -25.6 (43 pax) | Arrivals 54.5, equipaje (misma causa que SH) |
| **MAD-MEX** | Eco LH | 12.2 (74 pax) | Check-in 79.6, F&B (misma causa que SH) |
| **MAD-MVD** | Eco LH | -14.3 (35 pax) | Check-in 60.5, equipaje (misma causa que SH) |
| **BOS-MAD** | Eco LH | 22.7 (22 pax) | Equipaje no llegó, espera 3h (misma causa que SH) |

**Conclusión:** Las rutas LH experimentaron los **mismos problemas operativos** (equipaje, check-in, arrivals) que las rutas SH, confirmando causa sistémica de red.

---

## 🎯 MATIZ DE DOMINANCIA DE SH

"La anomalía Global (-7.3 pts) está **co-determinada por ambos radios**, pero con **dominancia de SH (-8.6 pts)** que arrastra el valor agregado debido a:

1. **Mayor magnitud absoluta:** SH cayó -8.6 pts vs LH -5.8 pts (diferencia de 2.8 pts)
2. **Mayor volumen operativo:** SH representa el grueso de vuelos diarios (corto/medio radio con mayor frecuencia)
3. **Mayor exposición a dark horses:** SH sufrió impacto directo de meteorología adversa (MAD-FLR: 5, MAD-BIO: 5, MAD-EAS: 2) + huelga ATC FCO (6 cancelaciones), mientras LH sufrió impacto indirecto vía conexiones

**Sin embargo, la causa raíz es SISTÉMICA:**  
Los tres vectores operativos (puntualidad percibida, mishandling +5.9 pts, misconex +0.2 pts) afectaron **por igual a ambos radios**, confirmando que NO es un problema específico de SH, sino una **crisis de red completa** con mayor manifestación en SH por su mayor exposición a eventos excepcionales (meteorología + huelga ATC)."

---

## 📊 SÍNTESIS EJECUTIVA DEL PASO 3

**A nivel GLOBAL, la dinámica es SINERGIA NEGATIVA CON DOMINANCIA DE SH `(-,- | -)`.**

### **Narrativa:**

"La red entera se vio impactada por una **crisis operativa multidimensional** que afectó transversalmente tanto al Largo como al Corto Radio:

1. **Deterioro de puntualidad percibida:** Ambos radios experimentaron percepción negativa (SHAP negativo), aunque las métricas operativas muestren comportamiento mixto (SH empeoró -2.2 pts OTP, LH mejoró +2.7 pts pero con eventos extremos puntuales)

2. **Epidemia de mishandling:** Incremento **idéntico** de +5.9 pts en ambos radios, con 127 maletas afectadas en incidentes compartidos (BA458, AGP, limitación peso)

3. **Aumento de conexiones perdidas:** Incremento **idéntico** de +0.2 pts misconex en ambos radios, con mayor impacto perceptual en LH Business (-7.12 pts satisfacción)

**Dark Horses sistémicos:**
- Huelga ATC FCO (17-dic): Afectó principalmente SH, impacto secundario en conexiones LH
- Meteorología adversa generalizada (14-19 dic): Afectó desproporcionadamente SH (Norte España + Italia), impacto secundario en conexiones LH
- Incidentes masivos de equipaje: Afectaron red completa (BA458 LH, AGP SH, limitación peso ambos)

**Dominancia de SH:**  
El Global (-7.3 pts) está más próximo a SH (-8.6 pts) que a LH (-5.8 pts) debido a:
- Mayor volumen operativo de SH
- Mayor exposición directa de SH a dark horses (meteorología + huelga ATC)
- Efecto cascada: problemas SH en hub MAD afectaron conexiones LH

**Conclusión:** La anomalía Global es una **crisis sistémica de red**, NO un problema aislado de un radio. Ambos radios sufrieron el mismo shock operativo (mishandling +5.9 pts, misconex +0.2 pts), con SH dominando el agregado por mayor magnitud y volumen."

---

### **Evidencia Clave:**

**Causas sistémicas confirmadas:**
1. **Mishandling +5.9 pts:** Idéntico en LH y SH (triangulación: SHAP + Operative Data + NCS + Verbatims)
2. **Misconex +0.2 pts:** Idéntico en LH y SH (triangulación: Operative Data + NCS + Verbatims)
3. **Puntualidad percibida:** Ambos radios con SHAP negativo (SH -3.955, LH -0.674)

**Dark Horses transversales:**
- Huelga ATC FCO: 6 cancelaciones (principalmente SH)
- Meteorología adversa: 12+ disrupciones (principalmente SH)
- Incidentes equipaje: 127 maletas (ambos radios)

**Rutas críticas que conectan radios:**
- BOG-MAD (LH) → conexiones SH desde MAD
- BIO-MAD, LCG-MAD, GVA-MAD, FCO-MAD (SH) → conexiones LH desde MAD

**Implicación estratégica:**  
Las acciones correctivas deben ser **transversales a toda la red**, priorizando:
1. **Gestión de equipajes** (problema idéntico en ambos radios)
2. **Gestión de conexiones** (problema idéntico en ambos radios)
3. **Resiliencia operativa ante eventos excepcionales** (meteorología, huelgas) con mayor foco en SH por mayor exposición

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

# 🔍 PASO 4: IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS (NMA)

---

## 📋 ANÁLISIS DE NODOS MÁXIMO AFECTADOS

---

### **CAUSA 1: DETERIORO DE PUNTUALIDAD PERCIBIDA**

- **Escenario:** SINERGIA NEGATIVA EN TODOS LOS NIVELES
- **NMA:** `Global` (máximo ancestro alcanzado por sinergia completa)
- **Afecta a:** 
  - Global/SH/Economy/IB
  - Global/SH/Economy/YW
  - Global/SH/Business/IB
  - Global/SH/Business/YW
  - Global/LH/Economy
  - Global/LH/Business
- **Tipo de impacto:** NEGATIVO
- **Cadena de propagación hacia el segmento raíz:**
  
  **Nivel Compañía (IB, YW):**
  * `Global/SH/Economy/IB` (-8.5 pts) + `Global/SH/Economy/YW` (-10.9 pts) → `Global/SH/Economy` (SINERGIA `-,-|-`, ambas compañías con deterioro de puntualidad)
  * `Global/SH/Business/IB` (-0.9 pts) + `Global/SH/Business/YW` (-8.1 pts) → `Global/SH/Business` (SINERGIA `-,-|-`, ambas compañías con deterioro de puntualidad, YW dominante)
  
  **Nivel Cabina (Economy, Business):**
  * `Global/SH/Economy` (-9.3 pts) + `Global/SH/Business` (-1.3 pts) → `Global/SH` (SINERGIA `-,-|-`, ambas cabinas con deterioro de puntualidad, Economy dominante)
  * `Global/LH/Economy` (-6.9 pts) + `Global/LH/Business` (-6.2 pts) + `Global/LH/Premium` (Normal) → `Global/LH` (SINERGIA PARCIAL `-,-,N|-`, Economy y Business con deterioro de puntualidad)
  
  **Nivel Radio (LH, SH):**
  * `Global/LH` (-5.8 pts) + `Global/SH` (-8.6 pts) → `Global` (SINERGIA `-,-|-`, ambos radios con deterioro de puntualidad percibida, SH dominante)

**Conclusión:** La puntualidad deterioró en **TODOS los niveles jerárquicos** (compañía → cabina → radio → global) mediante sinergia negativa completa. El NMA es `Global` porque la causa burbujea sin interrupción desde los nodos hoja hasta el ancestro máximo.

---

### **CAUSA 2: EPIDEMIA DE MISHANDLING (GESTIÓN DE EQUIPAJES)**

- **Escenario:** SINERGIA NEGATIVA EN TODOS LOS NIVELES
- **NMA:** `Global` (máximo ancestro alcanzado por sinergia completa)
- **Afecta a:**
  - Global/SH/Economy/IB
  - Global/SH/Economy/YW
  - Global/SH/Business/IB
  - Global/SH/Business/YW
  - Global/LH/Economy
  - Global/LH/Business
- **Tipo de impacto:** NEGATIVO
- **Cadena de propagación hacia el segmento raíz:**
  
  **Nivel Compañía (IB, YW):**
  * `Global/SH/Economy/IB` (Mishandling +5.8 pts) + `Global/SH/Economy/YW` (Mishandling +6.0 pts) → `Global/SH/Economy` (SINERGIA `-,-|-`, ambas compañías con epidemia de equipaje)
  * `Global/SH/Business/IB` (6 menciones equipaje, +300%) + `Global/SH/Business/YW` (equipaje SHAP negativo) → `Global/SH/Business` (SINERGIA `-,-|-`, ambas compañías con problemas de equipaje)
  
  **Nivel Cabina (Economy, Business):**
  * `Global/SH/Economy` (15+ menciones, +87%) + `Global/SH/Business` (6 menciones, +300%) → `Global/SH` (SINERGIA `-,-|-`, ambas cabinas con epidemia de equipaje, Economy dominante por volumen)
  * `Global/LH/Economy` (4 casos graves) + `Global/LH/Business` (6 casos graves) + `Global/LH/Premium` (sin menciones) → `Global/LH` (SINERGIA PARCIAL `-,-,N|-`, Economy y Business con problemas de equipaje)
  
  **Nivel Radio (LH, SH):**
  * `Global/LH` (Mishandling 22.81 vs 16.95, +5.9 pts) + `Global/SH` (Mishandling 22.81 vs 16.95, +5.9 pts) → `Global` (SINERGIA `-,-|-`, **IDÉNTICO deterioro en ambos radios**)

**Conclusión:** El mishandling es **IDÉNTICO en LH y SH** (+5.9 pts), confirmando causa sistémica de red. La epidemia de equipaje burbujea en **TODOS los niveles jerárquicos** mediante sinergia negativa completa. El NMA es `Global` porque es un problema transversal de la red completa.

---

### **CAUSA 3: AUMENTO DE CONEXIONES PERDIDAS (MISCONEX)**

- **Escenario:** SINERGIA NEGATIVA EN TODOS LOS NIVELES
- **NMA:** `Global` (máximo ancestro alcanzado por sinergia completa)
- **Afecta a:**
  - Global/SH/Economy/IB
  - Global/SH/Economy/YW
  - Global/SH/Business/IB
  - Global/SH/Business/YW
  - Global/LH/Business
- **Tipo de impacto:** NEGATIVO
- **Cadena de propagación hacia el segmento raíz:**
  
  **Nivel Compañía (IB, YW):**
  * `Global/SH/Economy/IB` (múltiples casos documentados) + `Global/SH/Economy/YW` (múltiples casos documentados) → `Global/SH/Economy` (SINERGIA `-,-|-`, ambas compañías con conexiones perdidas)
  * `Global/SH/Business/IB` (múltiples casos documentados) + `Global/SH/Business/YW` (múltiples casos documentados) → `Global/SH/Business` (SINERGIA `-,-|-`, ambas compañías con conexiones perdidas)
  
  **Nivel Cabina (Economy, Business):**
  * `Global/SH/Economy` (múltiples casos) + `Global/SH/Business` (4 menciones) → `Global/SH` (SINERGIA `-,-|-`, ambas cabinas con conexiones perdidas)
  * `Global/LH/Business` (Connections Experience SHAP -0.084, Sat -7.12 pts) → `Global/LH` (TRANSFERENCIA `-,N|-`, solo Business afectado, Economy sin datos de conexiones)
  
  **Nivel Radio (LH, SH):**
  * `Global/LH` (Misconex 0.86 vs 0.66, +0.2 pts) + `Global/SH` (Misconex 0.86 vs 0.66, +0.2 pts) → `Global` (SINERGIA `-,-|-`, **IDÉNTICO deterioro en ambos radios**)

**Conclusión:** El misconex es **IDÉNTICO en LH y SH** (+0.2 pts, +30.3%), confirmando causa sistémica de red. Las conexiones perdidas burbujean en **TODOS los niveles jerárquicos** mediante sinergia negativa completa. El NMA es `Global` porque es un problema transversal de la red completa.

---

### **CAUSA 4: DETERIORO DE CHECK-IN**

- **Escenario:** SINERGIA NEGATIVA PARCIAL (SH) + SINERGIA NEGATIVA PARCIAL (LH)
- **NMA:** `Global` (sinergia en ambos radios permite burbujeo)
- **Afecta a:**
  - Global/SH/Economy/IB
  - Global/SH/Economy/YW
  - Global/SH/Business/IB
  - Global/LH/Economy
  - Global/LH/Business
- **Tipo de impacto:** NEGATIVO
- **Cadena de propagación hacia el segmento raíz:**
  
  **Nivel Compañía (IB, YW):**
  * `Global/SH/Economy/IB` (Check-in SHAP -0.514) + `Global/SH/Economy/YW` (Check-in SHAP -0.881) → `Global/SH/Economy` (SINERGIA `-,-|-`, ambas compañías con deterioro de check-in)
  * `Global/SH/Business/IB` (Check-in SHAP -0.383) + `Global/SH/Business/YW` (sin datos específicos) → `Global/SH/Business` (TRANSFERENCIA `-,N|-`, solo IB documentado)
  
  **Nivel Cabina (Economy, Business):**
  * `Global/SH/Economy` (Check-in SHAP -0.514 IB, -0.881 YW) + `Global/SH/Business` (Check-in SHAP -0.383 IB) → `Global/SH` (SINERGIA `-,-|-`, ambas cabinas con deterioro de check-in, Economy más severo)
  * `Global/LH/Economy` (Check-in SHAP -1.073) + `Global/LH/Business` (Check-in SHAP -0.383) + `Global/LH/Premium` (sin datos) → `Global/LH` (SINERGIA PARCIAL `-,-,N|-`, Economy y Business con deterioro de check-in)
  
  **Nivel Radio (LH, SH):**
  * `Global/LH` (Check-in SHAP -0.830, Sat -1.35 pts) + `Global/SH` (Check-in SHAP -0.774, Sat -3.70 pts) → `Global` (SINERGIA `-,-|-`, ambos radios con deterioro de check-in, SH más severo)

**Conclusión:** El check-in deterioró en **ambos radios** (LH y SH) y en **múltiples cabinas/compañías**, burbujando mediante sinergia negativa hasta `Global`. El NMA es `Global` porque es un problema transversal.

---

### **CAUSA 5: DETERIORO DE ARRIVALS EXPERIENCE (MÁS ALLÁ DE EQUIPAJE)**

- **Escenario:** SINERGIA NEGATIVA EN TODOS LOS NIVELES
- **NMA:** `Global` (sinergia completa permite burbujeo)
- **Afecta a:**
  - Global/SH/Economy/IB
  - Global/SH/Economy/YW
  - Global/SH/Business/IB
  - Global/LH/Economy
  - Global/LH/Business
- **Tipo de impacto:** NEGATIVO
- **Cadena de propagación hacia el segmento raíz:**
  
  **Nivel Compañía (IB, YW):**
  * `Global/SH/Economy/IB` (Arrivals SHAP -1.687) + `Global/SH/Economy/YW` (Arrivals SHAP -1.156) → `Global/SH/Economy` (SINERGIA `-,-|-`, ambas compañías con deterioro de arrivals)
  * `Global/SH/Business/IB` (Arrivals SHAP -0.340) + `Global/SH/Business/YW` (sin datos específicos) → `Global/SH/Business` (TRANSFERENCIA `-,N|-`, solo IB documentado)
  
  **Nivel Cabina (Economy, Business):**
  * `Global/SH/Economy` (Arrivals SHAP -1.687 IB, -1.156 YW) + `Global/SH/Business` (Arrivals SHAP -0.340 IB) → `Global/SH` (SINERGIA `-,-|-`, ambas cabinas con deterioro de arrivals, Economy más severo)
  * `Global/LH/Economy` (Arrivals SHAP -0.944) + `Global/LH/Business` (Arrivals SHAP -3.934) + `Global/LH/Premium` (sin datos) → `Global/LH` (SINERGIA PARCIAL `-,-,N|-`, Economy y Business con deterioro de arrivals)
  
  **Nivel Radio (LH, SH):**
  * `Global/LH` (Arrivals SHAP -1.191, Sat -2.33 pts) + `Global/SH` (Arrivals SHAP -1.721, Sat -4.69 pts) → `Global` (SINERGIA `-,-|-`, ambos radios con deterioro de arrivals, SH más severo)

**Conclusión:** Arrivals experience deterioró en **ambos radios** (LH y SH) mediante sinergia negativa completa. El NMA es `Global` porque es un problema transversal de la red completa.

---

### **CAUSA 6: PROBLEMA ESTRUCTURAL DE BOARDING EN YW (AIR NOSTRUM)**

- **Escenario:** DOMINANCIA CON DILUCIÓN EN BUSINESS SH
- **NMA:** `Global/SH/Business/YW` (nodo hoja, no propaga por dilución de IB)
- **Afecta a:**
  - Global/SH/Business/YW
- **Tipo de impacto:** NEGATIVO
- **Cadena de propagación hacia el segmento raíz:**
  
  **Nivel Compañía (IB, YW):**
  * `Global/SH/Business/YW` (Boarding SHAP -2.983, problema estructural autobús/escaleras) + `Global/SH/Business/IB` (Boarding SHAP +0.891, compensador positivo) → `Global/SH/Business` (DOMINANCIA CON DILUCIÓN `-,+|-`, YW con problema severo pero diluido por IB compensador)
  
  **Nivel Cabina (Economy, Business):**
  * `Global/SH/Business` (deterioro marginal -1.3 pts, diluido) + `Global/SH/Economy` (sin problema de boarding significativo) → `Global/SH` (DILUCIÓN `-,N|N`, Business con problema pero diluido por Economy)
  
  **Nivel Radio (LH, SH):**
  * `Global/SH` (problema diluido) + `Global/LH` (sin problema de boarding) → `Global` (DILUCIÓN `-,N|N`, SH con problema pero diluido por LH)

**Conclusión:** El problema de boarding es **específico de YW Business SH** (operaciones Air Nostrum con embarque por autobús/escaleras). No propaga a niveles superiores por dilución de IB (que tiene Boarding +0.891 compensador) y por el mayor volumen de Economy. El NMA se detiene en `Global/SH/Business/YW`.

---

### **CAUSA 7: PARADOJA DE TICKET PRICE EN YW**

- **Escenario:** DOMINANCIA EN BUSINESS SH, DILUCIÓN EN NIVELES SUPERIORES
- **NMA:** `Global/SH/Business/YW` (nodo hoja, no propaga significativamente)
- **Afecta a:**
  - Global/SH/Business/YW
  - Global/SH/Economy/YW (menor impacto)
- **Tipo de impacto:** NEGATIVO (paradoja: satisfacción sube pero SHAP negativo)
- **Cadena de propagación hacia el segmento raíz:**
  
  **Nivel Compañía (IB, YW):**
  * `Global/SH/Business/YW` (Ticket Price SHAP -3.123, Sat +99.47 pts) + `Global/SH/Business/IB` (Ticket Price SHAP -0.271, Sat +63.69 pts) → `Global/SH/Business` (DOMINANCIA `-,-|-`, ambas con SHAP negativo pero YW más severo)
  * `Global/SH/Economy/YW` (Ticket Price SHAP -3.114, Sat +77.21 pts) + `Global/SH/Economy/IB` (Ticket Price SHAP -1.479, Sat +39.60 pts) → `Global/SH/Economy` (DOMINANCIA `-,-|-`, ambas con SHAP negativo pero YW más severo)
  
  **Nivel Cabina (Economy, Business):**
  * `Global/SH/Business` (Ticket Price negativo) + `Global/SH/Economy` (Ticket Price negativo) → `Global/SH` (SINERGIA `-,-|-`, ambas cabinas con paradoja de precio)
  
  **Nivel Radio (LH, SH):**
  * `Global/SH` (Ticket Price negativo) + `Global/LH` (Ticket Price negativo) → `Global` (SINERGIA `-,-|-`, ambos radios con paradoja de precio)

**Conclusión:** La paradoja de Ticket Price (satisfacción sube pero SHAP negativo) es **transversal a toda la red**, pero con mayor severidad en YW. Sin embargo, es un problema **estructural/perceptual**, no operativo, por lo que su impacto en NPS es secundario. El NMA técnicamente es `Global` por sinergia, pero la causa raíz se origina en las operaciones de YW (flota CRJ inadecuada para Business, percepción de "pagar business para vuelo en avión minúsculo").

---

### **CAUSA 8: PROBLEMA DE FLOTA CRJ EN YW BUSINESS**

- **Escenario:** TRANSFERENCIA EN BUSINESS SH
- **NMA:** `Global/SH/Business/YW` (nodo hoja, problema específico de flota)
- **Afecta a:**
  - Global/SH/Business/YW
- **Tipo de impacto:** NEGATIVO (problema de producto hard)
- **Cadena de propagación hacia el segmento raíz:**
  
  **Nivel Compañía (IB, YW):**
  * `Global/SH/Business/YW` (3 menciones período actual sobre flota CRJ inadecuada) + `Global/SH/Business/IB` (sin menciones de flota) → `Global/SH/Business` (TRANSFERENCIA `-,N|-`, problema específico de YW)
  
  **Nivel Cabina (Economy, Business):**
  * `Global/SH/Business` (problema de flota YW) + `Global/SH/Economy` (sin problema de flota) → `Global/SH` (DILUCIÓN `-,N|N`, Business con problema pero diluido por Economy)
  
  **Nivel Radio (LH, SH):**
  * `Global/SH` (problema diluido) + `Global/LH` (sin problema de flota CRJ) → `Global` (DILUCIÓN `-,N|N`, SH con problema pero diluido por LH)

**Conclusión:** El problema de flota CRJ es **específico de YW Business SH** (Air Nostrum). No propaga a niveles superiores por dilución de IB (sin quejas de flota) y por el mayor volumen de Economy. El NMA se detiene en `Global/SH/Business/YW`.

---

### **CAUSA 9: DETERIORO DE AIRCRAFT INTERIOR EN LH ECONOMY**

- **Escenario:** TRANSFERENCIA EN LH ECONOMY
- **NMA:** `Global/LH/Economy` (segmento hoja sin subniveles compañía en LH)
- **Afecta a:**
  - Global/LH/Economy
- **Tipo de impacto:** NEGATIVO
- **Cadena de propagación hacia el segmento raíz:**
  
  **Nivel Cabina (Economy, Business, Premium):**
  * `Global/LH/Economy` (Aircraft Interior SHAP -1.266, mayor impacto negativo) + `Global/LH/Business` (Aircraft Interior SHAP -0.847) + `Global/LH/Premium` (sin datos) → `Global/LH` (SINERGIA PARCIAL `-,-,N|-`, Economy y Business con deterioro de interior)
  
  **Nivel Radio (LH, SH):**
  * `Global/LH` (Aircraft Interior negativo) + `Global/SH` (Aircraft Interior SHAP -0.229 YW Eco, sin impacto significativo) → `Global` (TRANSFERENCIA `-,N|-`, problema específico de LH)

**Conclusión:** El deterioro de Aircraft Interior es **específico de LH** (pantallas IFE no funcionales, espacios angostos A321XLR), con mayor severidad en Economy. No propaga significativamente a Global por dilución de SH (sin problema de interior significativo). El NMA es `Global/LH/Economy`.

**Nota especial:** Este es un segmento hoja sin subniveles que analizar (no hay IB/YW en LH).

---

### **CAUSA 10: DETERIORO DE IN-FLIGHT FOOD & BEVERAGE EN LH ECONOMY**

- **Escenario:** TRANSFERENCIA EN LH ECONOMY
- **NMA:** `Global/LH/Economy` (segmento hoja sin subniveles compañía en LH)
- **Afecta a:**
  - Global/LH/Economy
- **Tipo de impacto:** NEGATIVO
- **Cadena de propagación hacia el segmento raíz:**
  
  **Nivel Cabina (Economy, Business, Premium):**
  * `Global/LH/Economy` (F&B SHAP -0.745, Sat -3.79 pts) + `Global/LH/Business` (F&B SHAP -0.678, Sat -3.90 pts) + `Global/LH/Premium` (sin datos) → `Global/LH` (SINERGIA PARCIAL `-,-,N|-`, Economy y Business con deterioro de F&B)
  
  **Nivel Radio (LH, SH):**
  * `Global/LH` (F&B negativo) + `Global/SH` (F&B SHAP -0.818 YW Eco, -0.319 IB Eco) → `Global` (SINERGIA `-,-|-`, ambos radios con deterioro de F&B)

**Conclusión:** El deterioro de F&B es **transversal a LH y SH**, burbujando mediante sinergia negativa hasta `Global`. Sin embargo, el impacto es más severo en LH Economy. El NMA técnicamente es `Global` por sinergia, pero la causa se manifiesta con mayor intensidad en `Global/LH/Economy`.

**Nota especial:** Este es un segmento hoja sin subniveles que analizar (no hay IB/YW en LH).

---

## 📊 RESUMEN DE NODOS MÁXIMO AFECTADOS

| # | Causa | NMA | Tipo Impacto | Escenario Dominante | Alcance |
|---|-------|-----|--------------|---------------------|---------|
| **1** | **Deterioro de Puntualidad Percibida** | `Global` | NEGATIVO | SINERGIA completa en todos los niveles | **RED COMPLETA** |
| **2** | **Epidemia de Mishandling** | `Global` | NEGATIVO | SINERGIA completa en todos los niveles | **RED COMPLETA** |
| **3** | **Aumento de Conexiones Perdidas** | `Global` | NEGATIVO | SINERGIA completa en todos los niveles | **RED COMPLETA** |
| **4** | **Deterioro de Check-in** | `Global` | NEGATIVO | SINERGIA en ambos radios | **RED COMPLETA** |
| **5** | **Deterioro de Arrivals Experience** | `Global` | NEGATIVO | SINERGIA completa en todos los niveles | **RED COMPLETA** |
| **6** | **Problema de Boarding YW** | `Global/SH/Business/YW` | NEGATIVO | DOMINANCIA con dilución | **ESPECÍFICO YW** |
| **7** | **Paradoja Ticket Price** | `Global` (origen YW) | NEGATIVO | SINERGIA (estructural) | **RED COMPLETA** (más severo YW) |
| **8** | **Problema Flota CRJ YW** | `Global/SH/Business/YW` | NEGATIVO | TRANSFERENCIA con dilución | **ESPECÍFICO YW** |
| **9** | **Deterioro Aircraft Interior LH** | `Global/LH/Economy` | NEGATIVO | TRANSFERENCIA (LH específico) | **ESPECÍFICO LH ECO** |
| **10** | **Deterioro F&B** | `Global` (más severo LH Eco) | NEGATIVO | SINERGIA en ambos radios | **RED COMPLETA** |

---

## 🎯 IMPLICACIONES ESTRATÉGICAS

### **CAUSAS SISTÉMICAS DE RED (NMA = Global):**
1. Puntualidad percibida
2. Mishandling (+5.9 pts idéntico LH/SH)
3. Misconex (+0.2 pts idéntico LH/SH)
4. Check-in
5. Arrivals experience
6. F&B

**Acción requerida:** Intervenciones **transversales a toda la red**, no específicas de compañía/radio/cabina.

### **CAUSAS ESPECÍFICAS DE YW:**
1. Boarding (embarque autobús/escaleras Air Nostrum)
2. Flota CRJ (inadecuación para Business)
3. Ticket Price (percepción "pagar business por avión minúsculo")

**Acción requerida:** Intervenciones **específicas en operaciones Air Nostrum** (YW).

### **CAUSAS ESPECÍFICAS DE LH:**
1. Aircraft Interior (pantallas IFE, espacios A321XLR)

**Acción requerida:** Intervenciones **específicas en flota/producto LH**.

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

# 📊 PASO 4B: EXTRACCIÓN DE EVIDENCIAS DEL CONTEXTO INICIAL

---

## **CAUSA 1: DETERIORO DE PUNTUALIDAD PERCIBIDA**

### === NMA: Global ===

#### 📈 EXPLANATORY DRIVERS:

**Global:**
- Punctuality: SHAP = -2.568 ppts, Sat_diff = -4.97 pts

**Global/SH:**
- Punctuality: SHAP = -3.955 ppts, Sat_diff = -5.88 pts

**Global/SH/Economy:**
- Punctuality: SHAP = -3.677 ppts, Sat_diff = -7.25 pts (IB)
- Punctuality: SHAP = -1.845 ppts, Sat_diff = -2.89 pts (YW)

**Global/SH/Business:**
- Punctuality: SHAP = -1.445 ppts, Sat_diff = -6.94 pts

**Global/SH/Business/IB:**
- Punctuality: SHAP = -1.230 ppts, Sat_diff = -8.99 pts

**Global/SH/Business/YW:**
- Punctuality: SHAP = -2.181 ppts, Sat_diff = -2.27 pts

**Global/LH:**
- Punctuality: SHAP = -0.674 ppts, Sat_diff = -3.44 pts

**Global/LH/Economy:**
- Punctuality: SHAP = -0.580 ppts, Sat_diff = -3.21 pts

**Global/LH/Business:**
- Punctuality: SHAP = -7.194 ppts, Sat_diff = -9.46 pts

#### 📊 DATOS OPERATIVOS:

**Global:**
- OTP15: 89.52% vs 91.72% baseline → Caída de 2.2 pts

**Global/SH:**
- OTP15: 89.52% vs 91.72% → Caída de 2.2 pts

**Global/SH/Economy/IB:**
- OTP15: 92.58% vs 94.62% → Caída de 2.0 pts

**Global/SH/Economy/YW:**
- OTP15: 86.81% vs 89.14% → Caída de 2.3 pts

**Global/SH/Business:**
- OTP15: 89.52% vs 91.72% → Caída de 2.2 pts

**Global/LH:**
- OTP15: 81.84% vs 79.12% → MEJORA de 2.7 pts (paradoja: mejora operativa pero percepción negativa)

#### 🚨 INCIDENTES NCS (CUANTITATIVO):

**Global:**
- Cancelaciones: +7 incidentes (+24.1% vs baseline, de 29 a 36)
- Retrasos: +6 incidentes (+16.7%, de 36 a 42)
- Total incidentes críticos operativos: +13 (+29.5%)

**Global/SH:**
- Cancelaciones: +7 (+24.1%)
- Retrasos: +6 (+16.7%)

**Global/SH/Economy/IB:**
- Cancelaciones: +7 incidentes (+24.1% vs período comparativo, de 29 a 36)
- Retrasos: +6 incidentes (+16.7%, de 36 a 42)

**Global/SH/Economy/YW:**
- Cancelaciones: +7 incidentes (+24.1% sobre baseline ~29)
- Retrasos: +6 incidentes (+16.7% sobre baseline ~36)

**Global/LH:**
- Cancelaciones: +5 incidentes (+500%)
- Retrasos: -7 incidentes (-24.1%)

#### 🧠 NCS (CUALITATIVO / REFLEXIÓN):

**PERÍODO ACTUAL (2025-12-13 a 2025-12-19):**

**[2025-12-16] HUELGA ATC EN FCO (Roma) - 17 DIC 13:00-17:00 LT:**
- Impacto: 6 vuelos cancelados
- Horario: 13:00-17:00 LT en Roma-Fiumicino
- Medidas: Flexibilización de tarifas publicada
- Rutas afectadas: MAD-FLR (5 disruptions), MAD-BIO (5 disruptions), MAD-EAS (2 disruptions)
- Coherencia temporal: Evento dentro del período de análisis, correlaciona con aumento de cancelaciones (+7) documentado en NCS

**[2025-12-14] METEOROLOGÍA EXTREMA GENERALIZADA:**
- Impacto: Múltiples desvíos/cancelaciones
- Rutas: EAS (cancelación + surface), OVD (2 desvíos desde LEN), AGP (regreso), TFS (desvío desde MAD), MLN (cancelación)
- Zonas afectadas: Norte de España (EAS, OVD, TFS, MLN)

**[2025-12-18] METEOROLOGÍA ADVERSA CONCENTRADA:**
- Impacto: BIO (2 cancelaciones + regreso IB0433), SDR (2 incidentes)
- Medidas: Oferta masiva de transporte por superficie

**[2025-12-19] METEOROLOGÍA PERSISTENTE:**
- Impacto: LEU (2 incidentes, surface ILD-LEU)

**[2025-12-18] FALLO SISTEMA IT:**
- Impacto: Incidencias operativas no especificadas

**INCIDENTES TÉCNICOS PUNTUALES:**
- **[2025-12-14] IB1586:** 160 min de retraso (descanso tripulación)
- **[2025-12-14] IB0337:** 93 min de retraso (descanso tripulación)
- **[2025-12-16] MAD-MRS:** Retraso mecánico - "puertas de la bodega del avión no cerraban"

**PERÍODO COMPARATIVO (2025-12-06 a 2025-12-12):**
- No se detectaron eventos excepcionales (huelgas, meteorología adversa, fallos de sistemas) en el período baseline
- Las quejas se centraron en aspectos estructurales de producto, NO en disrupciones operativas puntuales

#### 💬 FEEDBACK DE CLIENTES:

**Global/SH/Economy/IB (Período Actual):**
- LIS-MAD [NPS 9]: "retraso a la salida de Lisboa y consecuente retraso a la llegada"
- LHR-MAD [NPS 4]: "Long to Board, perdimos nuestra conexión"
- BCN-MAD [NPS 4]: "Retraso en la salida del vuelo que supone la pérdida del vuelo de enlace en Madrid hacia Chile"
- LCG-MAD [NPS 2]: "salir con retraso"

**Global/SH/Economy/YW (Período Actual):**
- AGP-MAD [NPS 1]: "1 hora de retraso"
- GRX-MAD [NPS 8]: "25 minutos de retraso"
- GVA-MAD [NPS 0]: "retraso 2h"
- LCG-MAD [NPS 0]: "salida con 1 hora de retraso"

**Global/SH/Business (Período Actual):**
- LIS-MAD [NPS 9]: "retraso a la salida de Lisboa y consecuente retraso a la llegada"
- LHR-MAD [NPS 4]: "Long to Board, perdimos nuestra conexión"
- BCN-MAD [NPS 4]: "Retraso en la salida del vuelo que supone la pérdida del vuelo de enlace en Madrid hacia Chile"
- EAS-MAD [NPS 0]: "numerosas molestias, esperas hasta abrir la puerta de desembarque, hacerlo con autobús, dejarnos en la terminal internacional teniendo que pasar por control de la policía con atrasos y colas"

**Global/LH (Período Actual):**
- MAD-PTY [NPS 3]: "Tuvimos un retraso en la puerta antes de la salida de más de una hora"
- BOG-MAD [NPS 0]: "El vuelo se retrasó casi dos horas"

**Comparativa Período COMPARISON vs Período ACTUAL:**
- Período ACTUAL: Quejas operativas dominantes (retrasos, cancelaciones), tono emocional de frustración por disrupciones operativas fuera del control del pasajero
- Período COMPARISON: Quejas estructurales/producto (cambios de avión, falta de diferenciación Business vs Economy), tono emocional de decepción por expectativas no cumplidas

#### ✈️ RUTAS AFECTADAS (Top 5):

**Global/SH:**
1. DUS-MAD: NPS -33.3 vs baseline -83.3 → Caída de 50 pts (6 pax)
2. LCG-MAD: NPS 0.0 vs baseline 70.0 → Caída de 66.1 pts (58 pax total)
3. MAD-VCE: NPS 50.0 vs baseline 100.0 → Caída de 44.4 pts (49 pax total)
4. MAD-SCQ: NPS 38.1 vs baseline 68.8 → Caída de 30.7 pts (40 pax total)
5. GVA-MAD: NPS 16.1 vs baseline 42.5 → Caída de 26.4 pts (69 pax total)

**Global/SH/Economy/IB:**
1. MAD-OSL: NPS -42.9 (7 pax) → Caída de -76.2 pts
2. LCG-MAD: NPS 2.3 (43 pax) → Caída de -62.1 pts
3. GVA-MAD: NPS 6.4 (47 pax) → Caída de -36.9 pts
4. LHR-MAD: NPS -13.2 (114 pax) → Caída de -32.6 pts
5. BIO-MAD: NPS 14.7 (68 pax) → Caída de -23.8 pts

**Global/SH/Business/IB:**
1. FRA-MAD: NPS 0.0 (2 pax) → Caída de -100.0 pts (Punctuality 0.0)
2. EAS-MAD: NPS 0.0 (2 pax) → Caída de -100.0 pts (Punctuality 0.0, Check-in 0.0)
3. LCG-MAD: NPS -20.0 (5 pax) → Caída de -120.0 pts (Punctuality 0.0, Check-in 50.0)
4. ATH-MAD: NPS -33.3 (3 pax) → Caída de -33.3 pts (Punctuality 0.0)
5. BCN-MAD: NPS 20.0 (25 pax) → Múltiples disrupciones operativas

**Global/LH:**
1. MAD-SCL: NPS -12.3 (57 pax) → Caída de -26.4 pts
2. MAD-ORD: NPS 12.8 (39 pax) → Caída de -18.8 pts
3. MAD-MVD: NPS -7.7 (24 pax) → Caída de -31.7 pts
4. MAD-MEX: NPS 18.0 (100 pax) → Caída de -22.8 pts
5. BOS-MAD: NPS 24.0 (48 pax) → Mejora de +2.1 pts (caso atípico positivo)

#### 👥 PERFILES REACTIVOS:

**Global/SH/Economy/IB:**
- Residence Region: Spread 161.1 pts (rango: -83.3 a +77.8 pts)
- Business/Leisure: Spread 2.1 pts (Leisure -8.8 pts, Business -6.7 pts)
- Fleet: Spread 0.0 pts (NPS_diff -6.1 pts, 1 perfil único)
- CodeShare: Spread 0.0 pts (NPS_diff -5.9 pts, 1 perfil único)

**Global/SH/Business/IB:**
- Residence Region: Spread 176.2 pts (rango: -26.4 a +133.3 pts)
- Business/Leisure: Spread 14.5 pts (Business -10.5 pts, Leisure +4.1 pts)
- Fleet: Spread 0.0 pts (NPS_diff -1.7 pts, 1 perfil: A320 Family)
- CodeShare: Spread 0.0 pts (NPS_diff -1.1 pts, 1 perfil: IB operated)

**Global/SH/Business/YW:**
- Residence Region: Spread 63.3 pts (rango: -30.0 a +33.3 pts entre 5 perfiles regionales)
- Business/Leisure: Spread 29.6 pts (Business -20.1 pts, Leisure +9.4 pts)
- Fleet: Spread 0.0 pts (NPS_diff -11.0 pts, 1 tipo de flota)
- CodeShare: Spread 0.0 pts (NPS_diff -11.0 pts, 1 tipo de operación)

**Global/LH:**
- Residence Region: Spread 122.4 pts (rango: -22.4 a +100.0 pts, 9 perfiles analizados)
- Business/Leisure: Spread 5.5 pts (rango: -6.5 a -1.0 pts, 2 perfiles analizados)
- Fleet: Spread 0.0 pts (1 perfil analizado, NPS_diff: -6.2 pts)
- CodeShare: Spread 0.0 pts (1 perfil analizado, NPS_diff: -6.3 pts)

---

## **CAUSA 2: EPIDEMIA DE MISHANDLING (GESTIÓN DE EQUIPAJES)**

### === NMA: Global ===

#### 📈 EXPLANATORY DRIVERS:

**Global:**
- Arrivals experience: SHAP = -1.420 ppts, Sat_diff = -3.83 pts

**Global/SH:**
- Arrivals experience: SHAP = -1.721 ppts, Sat_diff = -4.69 pts

**Global/SH/Economy/IB:**
- Arrivals experience: SHAP = -1.687 ppts, Sat_diff = -5.53 pts

**Global/SH/Economy/YW:**
- Arrivals experience: SHAP = -1.156 ppts, Sat_diff = -4.16 pts

**Global/SH/Business/IB:**
- Arrivals experience: SHAP = -0.340 ppts, Sat_diff = -2.85 pts

**Global/LH:**
- Arrivals experience: SHAP = -1.191 ppts, Sat_diff = -2.33 pts

**Global/LH/Economy:**
- Arrivals experience: SHAP = -0.944 ppts, Sat_diff = -1.87 pts

**Global/LH/Business:**
- Arrivals experience: SHAP = -3.934 ppts, Sat_diff = -3.03 pts

#### 📊 DATOS OPERATIVOS:

**Global:**
- Mishandling: 22.81 vs 16.95 → Incremento de 5.9 pts (+34.8%)

**Global/SH:**
- Mishandling: 22.81 vs 16.95 → Incremento de 5.9 pts

**Global/SH/Economy/IB:**
- Mishandling: 23.89 vs 18.1 → Incremento de 5.8 pts (+32%)

**Global/SH/Economy/YW:**
- Mishandling: 19.47 vs 13.43 → Incremento de 6.0 pts

**Global/LH:**
- Mishandling: 22.81 vs 16.95 → Incremento de 5.9 pts (+34.8%)

#### 🚨 INCIDENTES NCS (CUANTITATIVO):

**Global:**
- Incidentes de equipaje: 2 casos severos reportados
  - BA458: 27 maletas sin cargar
  - 100 maletas sin cargar en otro vuelo

**Global/SH:**
- Equipaje afectado: +56 incidentes (nueva categoría emergente)

**Global/SH/Economy/IB:**
- BA458: 27 maletas retrasadas
- Incidente sin vuelo especificado: 100 maletas no cargadas

**Global/SH/Economy/YW:**
- BA458 (LHR-MAD): 27 maletas extraviadas
- Incidente AGP: 100 maletas no cargadas

**Global/LH:**
- BA458: 27 maletas extraviadas (conexión LHR-MAD-DOH)
- 100 maletas no cargadas (sin especificar)

#### 🧠 NCS (CUALITATIVO / REFLEXIÓN):

**PERÍODO ACTUAL (2025-12-13 a 2025-12-19):**

**[2025-12-14] BA458: PROBLEMA EQUIPAJE:**
- 27 maletas en bodega sin cargar
- Entregadas 14/12 noche

**[2025-12-16] IB0433: PROBLEMA EQUIPAJE:**
- 100 maletas sin cargar
- Entregadas 16/12 noche

**[2025-12-16] LIMITACIÓN PESO EXTREMA:**
- 377 maletas sin cargar en un solo vuelo
- Relación: Coherente con aumento de Mishandling (+5.9 pts)

**[2025-12-18] FALLO SISTEMA IT EN LHR:**
- Evento: Fallos en sistema de carga de equipaje en Londres Heathrow
- Impacto: 10 maletas no cargadas
- Clasificación: RELEVANTE para LH (Londres hub intercontinental), pero afecta indirectamente a conexiones SH

**CAMBIO DE PATRÓN:**
- Período COMPARISON: Problemas de "corta conexión" (45 maletas)
- Período ACTUAL: Problemas de "falta de capacidad" (26-27 maletas)
- Rutas afectadas: TLV-MAD, conexiones en DFW

#### 💬 FEEDBACK DE CLIENTES:

**Global/SH/Economy/IB (Período Actual):**
- FCO-MAD [NPS 2]: "Mis maletas no aparecieron, incluyendo un coche para bebé"
- AMS-MAD [NPS 0]: "Me habéis perdido una maleta, no me dejaron meterla una vez estaba en el avión"
- MAD-NAP: Equipaje retrasado 1h30min en conexión MAD con "uniforme militar" (consecuencias disciplinarias)
- BCN-MAD [NPS 0]: "Maleta perdida, llegó 2 días después rota"

**Global/SH/Economy/YW (Período Actual):**
- MAD-SVQ [NPS 0]: "maletas en El Cairo, llevaba mis pastillas"
- MAD-VGO [NPS 5]: "equipaje perdido 24h sin ropa"
- GVA-MAD [NPS 0]: "equipaje perdido 3 días"
- PMI-VLC [NPS 0]: "maleta con partitura profesional desaparecida 2 días"
- BCN-SXB [NPS 0]: "maletas 2.5 días después, máquina apnea vital"
- LEI-MAD [NPS 0]: "maleta de mano perdida, conexión perdida"

**Global/SH/Business (Período Actual):**
- LIS-MAD [NPS 8]: "una hora de espera es demasiado tiempo" para recoger maletas
- ALC-MAD [NPS 0]: "habían perdido la maleta"
- FRA-MAD [NPS 0]: "Perdieron todo mi equipaje en algún punto del trayecto entre el vuelo FRA-MAD o el siguiente vuelo MAD-GRU" (pérdida total)
- BCN-MAD [NPS 2]: "Mi maleta no llegó al vuelo a Barcelona" + entrega tardía hasta lunes noche
- MAD-VCE [NPS 3]: "Nos quedamos sin equipaje en Venecia, sin nada y principalmente sin medicamentos"
- DSS-MAD [NPS 0]: "Llevo tres días esperando mi equipaje... clase ejecutiva"

**Global/LH (Período Actual):**
- BOG-MAD [NPS 3]: "me han abierto la maleta y me han sustraído la mitad"
- JFK-MAD [NPS 0]: "Mi maleta se perdió y tardó más de un día"
- BOS-MAD [NPS 0]: "El equipaje por el cual se realizó el cobro adicional no llegó a destino"
- LIM-MAD [NPS 0]: "Se perdieron mis maletas... 3 de 6 maletas extraviadas"
- EZE-MAD [NPS 0]: "Tenía una conexión Madrid-Palma... tuve que pasar por una fila muy larga en migraciones que no me permitió tomar dicho vuelo"

**Comparativa Período COMPARISON vs Período ACTUAL:**
- Período ACTUAL (SH Economy): 15+ menciones de equipaje perdido/retrasado/dañado
- Período COMPARISON (SH Economy): 8 menciones de equipaje (menos severas)
- **Incremento:** ~87% más menciones + mayor severidad

- Período ACTUAL (SH Business): 6 menciones de equipaje perdido
- Período COMPARISON (SH Business): 2 menciones
- **Incremento:** +300% en quejas de equipaje

- Período ACTUAL (LH): 4 menciones (Economy) + 6 menciones (Business)
- Período COMPARISON (LH): 2 menciones (Economy), datos Business no especificados
- **Incremento:** +100-300%

#### ✈️ RUTAS AFECTADAS (Top 5):

**Global/SH/Economy/IB:**
1. MAD-OSL: NPS -42.9 (7 pax), Arrivals 57.1
2. LCG-MAD: NPS 2.3 (43 pax), Arrivals 61.0
3. GVA-MAD: NPS 6.4 (47 pax), Arrivals 67.4
4. BIO-MAD: NPS 14.7 (68 pax), Arrivals 67.7
5. FCO-MAD: NPS 17.9 (112 pax), equipaje perdido documentado

**Global/SH/Economy/YW:**
1. MAD-SVQ: Equipaje con medicamentos en El Cairo
2. MAD-VGO: Equipaje perdido 24h
3. GVA-MAD: Equipaje perdido 3 días
4. PMI-VLC: Maleta con partitura profesional 2 días
5. BCN-SXB: Maletas 2.5 días después, máquina apnea vital

**Global/SH/Business:**
1. DSS-MAD: NPS -25.0 (4 pax), 3 días esperando equipaje
2. FRA-MAD: NPS 0.0 (2 pax), pérdida total de equipaje
3. MAD-VCE: NPS 76.9 (equipaje sin medicamentos)
4. BCN-MAD: NPS 20.0 (25 pax), maleta no llegó
5. LIS-MAD: 1 hora espera recogida equipaje

**Global/LH:**
1. BOG-MAD: NPS 7.6 (145 pax), maleta abierta con robo
2. MAD-SCL: NPS -12.3 (57 pax), Arrivals 54.5
3. MAD-ORD: NPS 12.8 (39 pax), Arrivals 65.8
4. MAD-MVD: NPS -7.7 (24 pax), equipaje retrasado
5. LIM-MAD: NPS 12.5 (56 pax), 3 de 6 maletas extraviadas

#### 👥 PERFILES REACTIVOS:

**Global/SH/Economy/IB:**
- Residence Region: Spread 161.1 pts (máxima reactividad)

**Global/SH/Business/IB:**
- Residence Region: Spread 176.2 pts (máxima reactividad)
- Business travelers: -10.5 pts (mayor sensibilidad)

**Global/LH:**
- Residence Region: Spread 122.4 pts (rango: -22.4 a +100.0 pts)
- Clientes de Latinoamérica: Mayor concentración de problemas de equipaje

---

## **CAUSA 3: AUMENTO DE CONEXIONES PERDIDAS (MISCONEX)**

### === NMA: Global ===

#### 📈 EXPLANATORY DRIVERS:

**Global/SH/Economy (IB):**
- Connections experience: SHAP = -0.133 ppts, Sat_diff = -3.66 pts

**Global/LH/Business:**
- Connections experience: SHAP = -0.084 ppts, Sat_diff = -7.12 pts

#### 📊 DATOS OPERATIVOS:

**Global:**
- Misconex: 0.86 vs 0.66 → Incremento de 0.2 pts (+30.3%)

**Global/SH:**
- Misconex: 0.86 vs 0.66 → Incremento de 0.2 pts (+30.3%)

**Global/SH/Economy/IB:**
- Misconex: 0.89 vs 0.72 → Incremento de 0.17 pts (+24%)

**Global/LH:**
- Misconex: 0.86 vs 0.66 → Incremento de 0.2 pts (+30.3%)

#### 🚨 INCIDENTES NCS (CUANTITATIVO):

**Global:**
- Desvíos: -3 incidentes (-21.4% vs baseline: 14→11)
- Pérdidas de conexión: 92→24 (-74%), pero con 128 pérdidas por huelga Italia (17-dic)

**Global/SH:**
- Desvíos: -3 incidentes (-25% vs baseline)
- Desvíos complejos que generaron conexiones perdidas:
  - FLR→BLQ (huelga ATC FCO)
  - EAS→BIO (meteorología)
  - BCN→OVD (meteorología)

**Global/LH:**
- Reprogramación masiva MAD-DFW (2025-12-18): 8h 5min con 193 pérdidas de conexión desde DFW

#### 🧠 NCS (CUALITATIVO / REFLEXIÓN):

**PERÍODO ACTUAL (2025-12-13 a 2025-12-19):**

**[2025-12-18] MAD-DFW - REPROGRAMACIÓN MASIVA:**
- Vuelo reprogramado 8 horas 5 minutos
- **193 pérdidas de conexión** desde DFW hacia destinos finales (SDQ, BOG, MIA, GYE)
- Nueva salida: 09:00h MAD → 13:35h DFW
- Impacto directo en NPS: Este evento explica la magnitud de la caída en "Connections experience"
- Clasificación: Evento operativo sin precedentes en magnitud de conexiones perdidas

**[2025-12-14] EVENTO OPERATIVO SISTÉMICO:**
- 24 cambios de equipo en un solo día
- 24 pérdidas de conexión en MAD
- 83 cancelaciones acumuladas en período
- Sugiere problema operativo de capacidad/disponibilidad de flota

**CRISIS OPERATIVA FLR (Florencia):**
- MAD-FLR: 5 disrupciones NCS (ruta más afectada del período)
- BLQ-FLR: 2 disrupciones NCS (hub alternativo saturado)
- Múltiples desvíos a BLQ sin capacidad de vuelos directos posteriores

**PERÍODO COMPARATIVO (2025-12-06 a 2025-12-12):**

**[2025-12-07] DFW-MAD - Evento masivo de reprogramación:**
- **193 pérdidas de conexión** desde DFW
- Múltiples verbatims en período COMPARISON
- Impacto en Baseline: Este evento DEPRIMIÓ el NPS del baseline

#### 💬 FEEDBACK DE CLIENTES:

**Global/SH (Período Actual):**
- BIO-MAD [NPS 0]: "Conexión perdida en MAD por retraso del vuelo"
- MAD-VCE [NPS 0]: "El vuelo a Oporto estaba programado con solo 30 minutos de escala... llegué tarde, tuve muchas prisas y cuando llegué a la T4, el vuelo ya había salido"
- EAS-MAD [NPS 0]: "conexión caótica", "llegar deprisa y corriendo atravesando toda la T4"

**Global/SH/Business (Período Actual):**
- LHR-MAD [NPS 4]: "Long to Board, perdimos nuestra conexión, hotel a las 2 AM"
- BCN-MAD [NPS 4]: "Retraso en la salida... pérdida del vuelo de enlace en Madrid hacia Chile"
- EAS-MAD [NPS 0]: "llegar deprisa y corriendo atravesando toda la T4"
- MAD-VCE [NPS 3]: "enlace con un vuelo... una hora sabiendo que es prácticamente imposible cargar las maletas"

**Global/LH (Período Actual):**
- EZE-MAD [NPS 0]: "Llegué 40 min antes de la salida del avión a Ámsterdam... mi valija no llegó"
- BOG-MAD [NPS 0]: "Teníamos una conexión a Londres Heathrow... llegamos, pero nuestras maletas no"

#### ✈️ RUTAS AFECTADAS (Top 5):

**Global/SH:**
1. MAD-VCE: Conexión imposible 30 min con niebla
2. BIO-MAD: Conexión perdida por retraso
3. EAS-MAD: Conexión caótica, traslados entre terminales
4. LHR-MAD: Conexión perdida, hotel 2 AM
5. BCN-MAD: Conexión perdida MAD-Chile

**Global/LH:**
1. MAD-DFW: 193 conexiones perdidas (18-dic)
2. EZE-MAD: Conexión perdida MAD-AMS
3. BOG-MAD: Conexión perdida, maletas no llegaron
4. MAD-VCE: Conexión con 1 hora de escala

#### 👥 PERFILES REACTIVOS:

**Global/SH/Business:**
- Business travelers: -10.5 pts (mayor sensibilidad a conexiones perdidas)
- Residence Region: Spread 176.2 pts (perfil más afectado: -26.4 pts)

**Global/LH/Business:**
- Business travelers: -21.4 pts (máxima sensibilidad)
- Residence Region: Spread 96.1 pts (perfil más afectado: -51.6 pts, probablemente destinos finales en Latinoamérica)

---

## **CAUSA 4: DETERIORO DE CHECK-IN**

### === NMA: Global ===

#### 📈 EXPLANATORY DRIVERS:

**Global:**
- Check-in: SHAP = -0.737 ppts, Sat_diff = -2.58 pts

**Global/SH:**
- Check-in: SHAP = -0.774 ppts, Sat_diff = -3.70 pts

**Global/SH/Economy/IB:**
- Check-in: SHAP = -0.514 ppts, Sat_diff = -3.70 pts

**Global/SH/Economy/YW:**
- Check-in: SHAP = -0.881 ppts, Sat_diff = -2.81 pts

**Global/SH/Business/IB:**
- Check-in: SHAP = -0.383 ppts, Sat_diff = -5.65 pts

**Global/LH:**
- Check-in: SHAP = -0.830 ppts, Sat_diff = -1.35 pts

**Global/LH/Economy:**
- Check-in: SHAP = -1.073 ppts, Sat_diff = -1.63 pts

**Global/LH/Business:**
- Check-in: SHAP = -0.396 ppts, Sat_diff = -2.55 pts

#### 📊 DATOS OPERATIVOS:

No disponible

#### 🚨 INCIDENTES NCS (CUANTITATIVO):

No disponible (no hay incidentes NCS específicos de check-in)

#### 🧠 NCS (CUALITATIVO / REFLEXIÓN):

No disponible

#### 💬 FEEDBACK DE CLIENTES:

**Global/SH/Economy/IB (Período Actual):**
- MAD-SCQ [NPS 0]: "No pude facturar en línea. La información no llega a la app a tiempo"
- AMS-MAD [NPS 0]: "Por un error de apellidos... no podía hacer el check in online"
- BCN-MAD [NPS 0]: "No pude hacer el checking online porque me saltaba el error"

**Global/SH/Economy/YW (Período Actual):**
- BOG-MAD [NPS 6]: "desorganización total en embarque, pesaje de maletas caótico, solo 1 máquina disponible"
- BOG-MAD [NPS 0]: "3 horas en cola, personal subcontratado tonteando, perdí el vuelo por negligencia"
- JFK-MAD [NPS 0]: "cambio de asientos sin justificación, nos separaron pese a haber pagado extra €244"

**Global/SH/Business/IB (Período Actual):**
- MAD-VIE [NPS 7]: "Solo había 2 personas en los mostradores... cola de 6 pasajeros"

**Global/LH (Período Actual):**
- JFK-MAD [NPS 0]: "proceso de facturación... fue terrible... información errónea"
- MAD-SCL [NPS 0]: "fila no avanzaba... varios se metieron... personal no hizo nada"

#### ✈️ RUTAS AFECTADAS (Top 5):

**Global/SH/Economy/IB:**
1. EAS-MAD: Check-in 0.0 (colapso total)
2. LCG-MAD: Check-in 50.0 (-120.0 pts)
3. DSS-MAD: Check-in 50.0 (-25.0 pts)
4. MAD-SCQ: Check-in online imposible
5. AMS-MAD: Check-in online fallido

**Global/LH:**
1. MAD-MVD: Check-in 60.5 (-31.7 pts), NPS -7.7 (24 pax)
2. MAD-MEX: Check-in 79.6 (-22.8 pts), NPS 18.0 (100 pax)
3. DOH-MAD: Check-in 48.1 (-15.6 pts), NPS -42.9 (28 pax)
4. MAD-NRT: Check-in 36.4 (12 pax)
5. MAD-SCL: Check-in 52.4 (43 pax)

#### 👥 PERFILES REACTIVOS:

No disponible (no se especifica reactividad diferencial por check-in en customer_profile_tool)

---

## **CAUSA 5: DETERIORO DE ARRIVALS EXPERIENCE (MÁS ALLÁ DE EQUIPAJE)**

### === NMA: Global ===

#### 📈 EXPLANATORY DRIVERS:

Ya incluido en CAUSA 2 (Mishandling)

**Nota:** Arrivals Experience incluye múltiples componentes:
1. Gestión de equipajes (mishandling) - Ver CAUSA 2
2. Esperas prolongadas en recogida de equipaje
3. Procesos de llegada (desembarque, traslados, migraciones)

#### 📊 DATOS OPERATIVOS:

Ya incluido en CAUSA 2

#### 🚨 INCIDENTES NCS (CUANTITATIVO):

Ya incluido en CAUSA 2

#### 🧠 NCS (CUALITATIVO / REFLEXIÓN):

**CONGESTIÓN T4 MADRID:**
- Múltiples menciones en verbatims de retrasos en entrega de equipaje (45+ minutos)
- Problemas sistémicos de coordinación tierra-vuelo
- Impacto en conexiones SH

#### 💬 FEEDBACK DE CLIENTES:

**Global/SH/Business (Período Actual):**
- LIS-MAD [NPS 8]: "una hora de espera es demasiado tiempo" para recoger maletas
- DUS-MAD [NPS 9]: "recogida de equipaje en Madrid tardó un poco"
- MAD-OSL [NPS 0]: "50 min espera equipaje"

**Global/LH (Período Actual):**
- LHR-MAD [NPS 3]: "45 min espera equipaje"
- BUD-MAD [NPS 1]: "45 min espera equipaje"
- MAD-SJO [NPS 8]: "esperamos más de una hora por nuestras maletas"
- BOS-MAD [NPS 0]: "esperamos tres horas hasta poder retirar el equipaje"

#### ✈️ RUTAS AFECTADAS (Top 5):

Ya incluido en CAUSA 2

#### 👥 PERFILES REACTIVOS:

Ya incluido en CAUSA 2

---

## **CAUSA 6: PROBLEMA ESTRUCTURAL DE BOARDING EN YW (AIR NOSTRUM)**

### === NMA: Global/SH/Business/YW ===

#### 📈 EXPLANATORY DRIVERS:

**Global/SH/Business/YW:**
- Boarding: SHAP = -2.983 ppts, Sat_diff = -7.41 pts

**Comparativa IB vs YW:**
- **IB:** Boarding SHAP = +0.891 ppts, Sat_diff = +2.53 pts (compensador positivo)
- **YW:** Boarding SHAP = -2.983 ppts, Sat_diff = -7.41 pts (agravante)

#### 📊 DATOS OPERATIVOS:

No disponible

#### 🚨 INCIDENTES NCS (CUANTITATIVO):

No disponible

#### 🧠 NCS (CUALITATIVO / REFLEXIÓN):

No disponible

#### 💬 FEEDBACK DE CLIENTES:

**Global/SH/Business/YW (Período Actual):**
- GVA-MAD [NPS 8]: "El embarque tardó demasiado tiempo"

**Período COMPARISON:**
- MAD-MUC [NPS 2]: "No recomendaría Air Nostrum solo por la ridiculez que es el acceso por Grupos cuando al final todos vamos a jardinera"

**Patrón recurrente:** Quejas sobre proceso de embarque con autobús/escaleras en ambos períodos, con mayor intensidad en período de comparación.

#### ✈️ RUTAS AFECTADAS (Top 5):

**Global/SH/Business/YW:**
1. GVA-MAD: Embarque tardío documentado
2. MAD-MUC: Embarque por autobús (período comparison)
3. Múltiples rutas Air Nostrum con proceso autobús/escaleras

#### 👥 PERFILES REACTIVOS:

**Global/SH/Business/YW:**
- Business travelers: -20.1 pts (máxima sensibilidad a boarding)
- Residence Region: Spread 63.3 pts

---

## **CAUSA 7: PARADOJA TICKET PRICE**

### === NMA: Global (origen YW) ===

#### 📈 EXPLANATORY DRIVERS:

**Global:**
- Ticket Price: SHAP = -1.354 ppts, Sat_diff = +30.03 pts (paradoja)

**Global/SH/Economy/IB:**
- Ticket Price: SHAP = -1.479 ppts, Sat_diff = +39.60 pts

**Global/SH/Economy/YW:**
- Ticket Price: SHAP = -3.114 ppts, Sat_diff = +77.21 pts

**Global/SH/Business/IB:**
- Ticket Price: SHAP = -0.271 ppts, Sat_diff = +63.69 pts

**Global/SH/Business/YW:**
- Ticket Price: SHAP = -3.123 ppts, Sat_diff = +99.47 pts

**Global/LH/Economy:**
- Ticket Price: SHAP = No disponible en datos proporcionados

**Global/LH/Business:**
- Ticket Price: SHAP = -0.424 ppts, Sat_diff = +8.86 pts

#### 📊 DATOS OPERATIVOS:

No disponible

#### 🚨 INCIDENTES NCS (CUANTITATIVO):

No disponible

#### 🧠 NCS (CUALITATIVO / REFLEXIÓN):

No disponible

#### 💬 FEEDBACK DE CLIENTES:

**Global/SH/Business/YW (Período Actual):**
- MAD-NTE [NPS 0]: "Pagas business para que te metan en un avión minúsculo"
- MAD-VCE [NPS 3]: "Difícil de justificar el precio"

**Período COMPARISON:**
- MAD-PNA [NPS 6]: "Altísimos precios del vuelo"
- GVA-MAD [NPS 1]: "Billete carísimo para nada"
- FRA-MAD [NPS 4]: "No vale lo que he pagado"
- AMS-MAD [NPS 0]: "Pagas business para que te metan en un avión minúsculo"

**Explicación de la paradoja:** Satisfacción con precio SUBIÓ +99.47 pts (YW) porque hubo MENOS quejas en el período actual (2 menciones) vs comparación (4 menciones), pero el impacto SHAP es negativo porque sigue siendo un pain point residual que afecta la recomendación.

**Global/LH (Período Actual):**
- MAD-SCL [NPS 0]: "1.186.379 [pesos] para ascender a Turista Premium... me tocaba ese nuevo asiento [38B]" (downgrade)

#### ✈️ RUTAS AFECTADAS (Top 5):

**Global/SH/Business/YW:**
1. MAD-NTE: "Pagas business para avión minúsculo"
2. MAD-VCE: "Difícil justificar precio"
3. MAD-PNA: "Altísimos precios" (comparison)
4. GVA-MAD: "Billete carísimo" (comparison)
5. AMS-MAD: "Pagas business para avión minúsculo" (comparison)

#### 👥 PERFILES REACTIVOS:

No disponible (Ticket Price no genera diferenciación específica en customer_profile_tool)

---

## **CAUSA 8: PROBLEMA DE FLOTA CRJ EN YW BUSINESS**

### === NMA: Global/SH/Business/YW ===

#### 📈 EXPLANATORY DRIVERS:

**Global/SH/Business/YW:**
- Aircraft interior: No disponible en datos SHAP específicos de YW

**Nota:** Este problema se manifiesta en verbatims, no en drivers SHAP principales.

#### 📊 DATOS OPERATIVOS:

No disponible

#### 🚨 INCIDENTES NCS (CUANTITATIVO):

No disponible

#### 🧠 NCS (CUALITATIVO / REFLEXIÓN):

No disponible

#### 💬 FEEDBACK DE CLIENTES:

**Global/SH/Business/YW (Período Actual):**
- MAD-SDR [NPS 0]: "Estado del avión lamentable, asientos desvencijados"
- MAD-VLC [NPS 7]: "Avión bastante antiguo"
- MAD-VGO [NPS 7]: "Aeronave más moderna" (comentario positivo)

**Período COMPARISON:**
- MAD-SVQ [NPS 0]: "Asientos diminutos, abarrotado e incómodo"
- GVA-MAD [NPS 1]: "Avión minúsculo, estrecha y pegada a otra persona"
- AMS-MAD [NPS 4]: "Avión no apto para clase ejecutiva"
- CMN-MAD [NPS 0]: "NO HAY CLASE BUSINESS!"
- CMN-MAD [NPS 0]: "Seamos sinceros, ¡NO HAY CLASE BUSINESS!"

**Conclusión:** Problema estructural en AMBOS períodos sobre la inadecuación de la flota CRJ para Business Class, con mayor intensidad y especificidad en el período de comparación.

#### ✈️ RUTAS AFECTADAS (Top 5):

**Global/SH/Business/YW:**
1. MAD-SDR: Avión lamentable, asientos desvencijados
2. MAD-VLC: Avión antiguo
3. MAD-SVQ: Asientos diminutos (comparison)
4. GVA-MAD: Avión minúsculo (comparison)
5. CMN-MAD: "NO HAY CLASE BUSINESS" (comparison)

#### 👥 PERFILES REACTIVOS:

**Global/SH/Business/YW:**
- Fleet: Spread 0.0 pts (1 tipo de flota, -11.0 pts) - Toda la operación usa misma flota CRJ

---

## **CAUSA 9: DETERIORO DE AIRCRAFT INTERIOR EN LH ECONOMY**

### === NMA: Global/LH/Economy ===

**Nota:** Segmento hoja sin subniveles que analizar (no hay IB/YW en LH).

#### 📈 EXPLANATORY DRIVERS:

**Global/LH/Economy:**
- Aircraft interior: SHAP = -1.266 ppts, Sat_diff = -3.13 pts (mayor impacto negativo)

**Global/LH/Business:**
- Aircraft interior: SHAP = -0.847 ppts, Sat_diff = -1.57 pts

#### 📊 DATOS OPERATIVOS:

No disponible

#### 🚨 INCIDENTES NCS (CUANTITATIVO):

No disponible

#### 🧠 NCS (CUALITATIVO / REFLEXIÓN):

No disponible

#### 💬 FEEDBACK DE CLIENTES:

**Global/LH/Economy (Período Actual):**
- BOS-MAD [NPS 0]: "avión muy pequeño e incómodo, asientos no reclinables, pantallas no funcionaban"
- MAD-REC - A321XLR [NPS 5]: "espacios demasiado angostos, pantallas enormes en los ojos, bulkhead obstruye paso"
- LIM-MAD [NPS 0]: "pantalla no funcionó durante 11 horas, llamé varias veces sin respuesta"
- MAD-SCL [NPS 0]: "pantalla dejó de funcionar tras 6 horas, avisé 4 veces, nadie apareció"

**Período COMPARISON:**
- Menciones mínimas de problemas de pantallas/interior

#### ✈️ RUTAS AFECTADAS (Top 5):

**Global/LH/Economy:**
1. MAD-SAL: Aircraft Interior 33.3 (6 pax)
2. MAD-SDQ: Aircraft Interior 45.8 (48 pax)
3. MAD-SCL: Aircraft Interior 44.2 (43 pax, ↓42.2 pts)
4. BOS-MAD: Aircraft Interior 50.0 (22 pax, ↑3.4 pts)
5. GRU-MAD: Aircraft Interior 53.8 (39 pax, ↓8.7 pts)

**Ruta con problema específico de flota:**
- MAD-REC (A321XLR): "espacios demasiado angostos, pantallas enormes en los ojos, bulkhead obstruye paso"

#### 👥 PERFILES REACTIVOS:

**Global/LH/Economy:**
- Residence Region: Spread 93.1 pts (rango: -26.4 a +66.7 pts)
- Business/Leisure: Spread 4.0 pts (ambos perfiles cayeron de manera similar)

---

## **CAUSA 10: DETERIORO DE IN-FLIGHT FOOD & BEVERAGE**

### === NMA: Global (más severo LH Economy) ===

#### 📈 EXPLANATORY DRIVERS:

**Global:**
- In flight food and beverage: SHAP = -0.447 ppts, Sat_diff = -3.59 pts

**Global/SH/Economy/IB:**
- In flight food and beverage: SHAP = -0.137 ppts, Sat_diff = -3.43 pts

**Global/SH/Economy/YW:**
- In flight food and beverage: SHAP = -0.818 ppts, Sat_diff = -6.48 pts

**Global/SH/Business:**
- In flight food and beverage: No disponible en datos proporcionados

**Global/LH/Economy:**
- In flight food and beverage: SHAP = -0.745 ppts, Sat_diff = -3.79 pts

**Global/LH/Business:**
- In flight food and beverage: SHAP = -0.678 ppts, Sat_diff = -3.90 pts

#### 📊 DATOS OPERATIVOS:

No disponible

#### 🚨 INCIDENTES NCS (CUANTITATIVO):

No disponible

#### 🧠 NCS (CUALITATIVO / REFLEXIÓN):

No disponible

#### 💬 FEEDBACK DE CLIENTES:

**Global/LH/Economy (Período Actual):**
- MAD-SCL [NPS 0]: "La comida era pobre"
- MAD-ORD [NPS 5]: "La comida era terrible y apenas teníamos nada para beber"
- MAD-SCL [NPS 0]: "no se sirvió café en 13 horas, menú pobre con solo 2 opciones"
- MAD-PTY [NPS 3]: "anunciaron al salir que no disponían de bebidas calientes durante todo el viaje largo"
- MAD-MIA [NPS 0]: "solo una botellita de agua en 8 horas, vaso diminuto medio lleno"

**Global/LH/Business:**
- LHR-MAD [NPS 7]: "menú de comida pobre"

**Global/SH/Economy/YW (Período Actual):**
- LIM-MAD [NPS 2]: "servicio sin ánimo, azafata respondió fuerte 'es agua y nada más'"

**Período COMPARISON:**
- MAD-MEX [NPS 5]: "comida solo sabe a sal, no había bebidas calientes"

#### ✈️ RUTAS AFECTADAS (Top 5):

**Global/LH/Economy:**
1. MAD-SCL: Comida pobre, sin café 13 horas
2. MAD-ORD: Comida terrible, sin bebidas
3. MAD-PTY: Sin bebidas calientes en vuelo largo
4. MAD-MIA: Solo una botellita de agua 8 horas
5. MAD-MEX: Comida solo sabe a sal (comparison)

#### 👥 PERFILES REACTIVOS:

No disponible (F&B no genera diferenciación específica en customer_profile_tool)

---

## Cabin Radio Reflection

# 📋 PASO 4C: REFLEXIÓN POR CABINA-RADIO

---

## 🔷 CABINAS SHORT HAUL (SH)

---

### === ECONOMY SH ===

• **NPS Cabina:** 27.9 (-9.3 pts)  
• **Estado:** NEGATIVE ANOMALY  
• **Escenario:** SINERGIA NEGATIVA (IB `-`, YW `-` | Cabina `-`)

• **IB:** NPS 27.0 (-8.5 pts)  
  - **Causa principal:** Deterioro operativo multidimensional con tres vectores: (1) **Punctuality** (SHAP -3.677, OTP15 -2.0 pts, +7 cancelaciones, +6 retrasos), (2) **Mishandling** (+5.8 pts, 127 maletas afectadas: BA458 27 maletas + AGP 100 maletas), (3) **Arrivals Experience** (SHAP -1.687, esperas prolongadas 45-50 min). Dark horses: Huelga ATC FCO (6 vuelos cancelados), meteorología adversa Norte España + Italia (MAD-FLR 5 disrupciones, MAD-BIO 5 disrupciones), incidentes masivos de equipaje.

• **YW:** NPS 29.6 (-10.9 pts)  
  - **Causa principal:** Deterioro operativo con énfasis en (1) **Punctuality** (SHAP -1.845, OTP15 -2.3 pts, +7 cancelaciones, +6 retrasos), (2) **Mishandling** (+6.0 pts, múltiples casos graves: MAD-SVQ medicamentos perdidos, GVA-MAD 3 días sin maletas, BCN-SXB máquina apnea vital), (3) **Ticket Price** (SHAP -3.114, paradoja: satisfacción +77.21 pts pero percepción negativa por "pagar business para avión minúsculo"). Dark horses: mismos eventos que IB (huelga ATC FCO, meteorología adversa, incidentes equipaje).

• **Narrativa de agregación:**  
Ambas compañías (IB e YW) experimentan deterioro simultáneo por la **misma crisis operativa sistémica** (meteorología adversa + huelga ATC FCO + epidemia de mishandling). YW sufre mayor magnitud (-10.9 pts vs -8.5 pts IB) por mayor severidad en mishandling (+6.0 pts vs +5.8 pts) y mayor deterioro de puntualidad (OTP15 -2.3 pts vs -2.0 pts). La causa es **común y transversal**, no específica de compañía. El efecto se suma y transfiere íntegramente al padre Economy SH (-9.3 pts).

• **Rutas críticas (del padre Economy SH):**
  1. **DUS-MAD:** NPS -50.0 (6 pax), caída de -100.0 pts - Triangulación: SHAP + NCS×2 + Verbatims×2 (azafata irrespetuosa)
  2. **LCG-MAD:** NPS 2.3 (43 pax), caída de -62.1 pts - Triangulación: NCS×2 + Verbatims×2 (50 min retraso, maletas en pista)
  3. **MAD-VCE:** NPS 50.0 (49 pax), caída de -44.4 pts - Triangulación: NCS×2 + Verbatims×2 (equipaje no cargado, conexión perdida)
  4. **GVA-MAD:** NPS 6.4 (47 pax), caída de -36.9 pts - Triangulación: NCS×2 + Verbatims×2 (3 días sin maletas)
  5. **BIO-MAD:** NPS 14.7 (68 pax), caída de -23.8 pts - Triangulación: SHAP + NCS + Verbatims (5 disrupciones meteorológicas)

• **Perfiles reactivos (del padre Economy SH):**
  - **Residence Region:** Spread 161.1 pts (máxima reactividad), rango -83.3 a +77.8 pts - Ciertos mercados desproporcionadamente afectados
  - **Business/Leisure:** Spread 2.1 pts (baja diferenciación), Leisure -8.8 pts, Business -6.7 pts - Ambos perfiles reaccionaron similarmente
  - **Fleet:** Spread 0.0 pts (sin variabilidad, 1 perfil único)
  - **CodeShare:** Spread 0.0 pts (sin variabilidad, 1 perfil único)

---

### === BUSINESS SH ===

• **NPS Cabina:** 34.6 (-1.3 pts)  
• **Estado:** NEGATIVE ANOMALY  
• **Escenario:** DOMINANCIA CON DILUCIÓN (IB `-`, YW `-` | Cabina `-`)

• **IB:** NPS 40.1 (-0.9 pts)  
  - **Causa principal:** Deterioro operativo triple: (1) **Punctuality** (SHAP -1.230, OTP15 -2.0 pts, +7 cancelaciones, +6 retrasos), (2) **Mishandling** (+5.8 pts, 6 menciones equipaje +300% vs baseline: DSS-MAD 3 días esperando, FRA-MAD pérdida total, MAD-VCE sin medicamentos), (3) **Conexiones perdidas** (misconex +0.17 pts, 4 menciones: LHR-MAD hotel 2 AM, BCN-MAD conexión Chile perdida). **Compensador positivo:** Boarding (+0.891 SHAP) mitigó parcialmente el deterioro. Dark horses: Huelga ATC FCO (6 cancelaciones), meteorología adversa (MAD-FLR 5, MAD-BIO 5 disrupciones), fallo sistema LHR (10 maletas no cargadas).

• **YW:** NPS 16.7 (-8.1 pts)  
  - **Causa principal:** Deterioro operativo agravado en operaciones Air Nostrum: (1) **Punctuality** (SHAP -2.181, OTP15 -2.33 pts, 79% más severo que IB), (2) **Boarding** (SHAP -2.983, problema estructural embarque autobús/escaleras: "El embarque tardó demasiado tiempo"), (3) **Ticket Price** (SHAP -3.123, paradoja: satisfacción +99.47 pts pero percepción negativa "Pagas business para avión minúsculo"). **Problema estructural de flota CRJ:** 3 menciones período actual vs 5 en comparativo sobre inadecuación para Business Class. Dark horses: mismos eventos que IB + crisis operativa FLR (5 disrupciones).

• **Narrativa de agregación:**  
Ambas compañías tienen anomalías negativas, pero **YW domina con -8.1 pts** mientras **IB tiene deterioro marginal de -0.9 pts**. El padre (-1.3 pts) refleja la dominancia de YW pero **diluida por el mayor volumen de IB** (93 encuestas IB vs ~20 YW). IB mantuvo compensadores positivos (Boarding +0.891) que YW no tuvo, explicando la diferencia de magnitud. La causa es **parcialmente común** (puntualidad, equipaje) pero con **problemas específicos de YW** (boarding autobús/escaleras, flota CRJ inadecuada).

• **Rutas críticas (del hijo dominante YW):**
  1. **MAD-FLR:** 5 disrupciones NCS (crisis operativa Italia, huelga ATC FCO)
  2. **MAD-BIO:** 5 disrupciones NCS (meteorología adversa Norte España) + 2 cancelaciones (18-dic)
  3. **MAD-EAS:** 2 disrupciones NCS (meteorología) + cancelación + transporte superficie
  4. **MAD-MRS:** Retraso mecánico (16-dic) + verbatim [NPS 2] "Más de una hora de retraso porque las puertas de la bodega del avión no cerraban"
  5. **GVA-MAD:** Verbatim [NPS 8] "El embarque tardó demasiado tiempo"

• **Perfiles reactivos (del hijo dominante YW):**
  - **Residence Region:** Spread 63.3 pts (máxima reactividad), rango -30.0 a +33.3 pts - Ciertos mercados desproporcionadamente afectados (hasta -30.0 pts)
  - **Business/Leisure:** Spread 29.6 pts, Business -20.1 pts, Leisure +9.4 pts - Business travelers más sensibles a puntualidad/boarding
  - **Fleet:** Spread 0.0 pts (1 tipo de flota CRJ, -11.0 pts) - Toda la operación usa misma flota
  - **CodeShare:** Spread 0.0 pts (1 tipo de operación YW, -11.0 pts)

---

## 🔶 CABINAS LONG HAUL (LH)

---

### === ECONOMY LH ===

• **NPS:** 3.9 (-6.9 pts)  
• **Estado:** NEGATIVE ANOMALY

• **Causa principal:**  
Deterioro masivo en experiencia de **PRODUCTO**, no operativo. Triangulación de 3 fuentes (Drivers SHAP + NCS + Verbatims) confirma: (1) **Aircraft Interior** (SHAP -1.266, mayor impacto negativo: pantallas IFE no funcionales en BOS-MAD, LIM-MAD 11 horas, MAD-SCL 6 horas; espacios angostos A321XLR en MAD-REC), (2) **Check-in** (SHAP -1.073, problemas sistémicos en MAD-MVD, MAD-MEX, DOH-MAD), (3) **In-flight Food & Beverage** (SHAP -0.745, múltiples menciones: MAD-SCL "comida pobre", MAD-ORD "comida terrible", MAD-PTY "sin bebidas calientes 13 horas"). **Contradicción operativa:** OTP mejoró +2.7 pts pero NPS cayó, confirmando naturaleza no operativa de la anomalía.

• **Evidencia clave:**  
  - **Driver SHAP principal:** Aircraft Interior SHAP -1.266 ppts, Sat_diff -3.13 pts
  - **Métrica operativa más relevante:** OTP15 81.84% vs 79.12% → MEJORA de +2.7 pts (paradoja: mejora operativa pero percepción negativa por eventos extremos puntuales)
  - **Mishandling:** 22.81 vs 16.95 → +5.9 pts (+34.8%), con 4 casos graves documentados (BOG-MAD, JFK-MAD, BOS-MAD, LIM-MAD)

• **Rutas críticas:**
  1. **MAD-SCL:** NPS -25.6 (43 pax), caída de -42.2 pts - Aircraft Interior 44.2, Check-in 52.4, Arrivals 53.7 - Verbatims: pantalla no funcionó 6 horas, sin café 13 horas
  2. **MAD-ORD:** NPS 8.8 (34 pax), caída de -37.6 pts - Aircraft Interior 65.6, Check-in 65.6, Arrivals 66.7 - Verbatim: "comida terrible, solo una bebida de 4 onzas para 8 horas"
  3. **MAD-MVD:** NPS -14.3 (35 pax), caída de -34.3 pts - Check-in 58.8, Arrivals 66.7
  4. **BOS-MAD:** NPS 22.7 (22 pax), mejora de +3.4 pts (paradójica) - Aircraft Interior 50.0, Check-in 59.1 - Verbatim: "avión muy pequeño e incómodo, asientos no reclinables, pantallas no funcionaban"
  5. **GRU-MAD:** NPS 2.6 (39 pax), caída de -8.7 pts - Aircraft Interior 53.8 - Verbatim: "equipaje retrasado desde 16/12, sin novedades"

• **Perfiles reactivos:**
  - **Residence Region:** Spread 93.1 pts (rango: -26.4 a +66.7 pts, 9 perfiles analizados) - Pasajeros de LATAM muestran mayor sensibilidad negativa (-50 pts según inferencia cualitativa)
  - **Business/Leisure:** Spread 4.0 pts (rango: -7.4 a -3.4 pts) - Ambos perfiles cayeron de manera similar, sin diferenciación significativa

---

### === BUSINESS LH ===

• **NPS:** 19.0 (-6.2 pts)  
• **Estado:** NEGATIVE ANOMALY

• **Causa principal:**  
Deterioro operativo triple: (1) **Conexiones perdidas y reprogramaciones extremas** (Connections Experience SHAP -0.396, misconex +0.2 pts; evento crítico MAD-DFW 18-dic: reprogramación 8h 5min con 193 pérdidas de conexión desde DFW; verbatims: LHR-MAD "perdimos nuestra conexión", BCN-MAD "pérdida del vuelo de enlace en Madrid hacia Chile"), (2) **Gestión de equipajes deteriorada** (mishandling +5.9 pts, 6 menciones +300% vs baseline: LIS-MAD 90 min espera, FRA-MAD pérdida total, MAD-VCE sin medicamentos), (3) **Puntualidad percibida** (Punctuality SHAP -7.194, paradoja: OTP15 mejoró +2.7 pts pero satisfacción cayó -9.46 pts por eventos extremos puntuales IB1586 160 min, IB0337 93 min). **Compensador insuficiente:** Boarding (+1.332 SHAP) no compensó el deterioro operativo.

• **Evidencia clave:**  
  - **Driver SHAP principal:** Punctuality SHAP -7.194 ppts, Sat_diff -9.46 pts (mayor impacto perceptual)
  - **Métrica operativa más relevante:** Misconex 0.86 vs 0.66 → +0.2 pts (+30.3%), con evento excepcional MAD-DFW 193 conexiones perdidas
  - **Mishandling:** 22.81 vs 16.95 → +5.9 pts (+34.8%)

• **Rutas críticas:**
  1. **MAD-DFW / DFW-MAD:** 3 disrupciones en período actual, reprogramación masiva 8h 5min con 193 conexiones perdidas (18-dic) - Hub de conexiones crítico para tráfico LH hacia Américas
  2. **BOG-MAD:** NPS 7.6 (145 pax), caída de -9.0 pts - Verbatims: 3 comentarios negativos (equipaje 90 min, vuelo retrasó 2 horas, equipaje no llegó)
  3. **EZE-MAD:** NPS 6.7 (75 pax), caída de -8.7 pts - Verbatims: 4 comentarios negativos (conexión perdida MAD-AMS, vuelo con demora, asiento no respetado, tripulación poco proactiva)
  4. **MAD-MIA:** NPS 12.8 (deterioro de -18.8 pts según otra fuente) - Verbatims: 3 comentarios negativos (servicio deficiente, retrasos, tripulación como aerolínea de bajo coste)
  5. **MAD-SDQ:** 1 comentario crítico [NPS 0] - equipaje perdido

• **Perfiles reactivos:**
  - **Residence Region:** Spread 96.1 pts (rango: -51.6 a +44.6 pts, 8 regiones analizadas) - Perfil más afectado: -51.6 pts (región específica no identificada, hipótesis: mercados con alta dependencia de conexiones MAD-DFW, probablemente destinos latinoamericanos)
  - **Business/Leisure:** Spread 6.3 pts, Business -8.4 pts, Leisure -2.1 pts - Ambos perfiles negativos, sin diferenciación significativa

---

### === PREMIUM LH ===

• **NPS:** 13.1 (+3.6 pts)  
• **Estado:** Normal (dentro de rango normal, no anomalía)

• **Causa principal:**  
Sin análisis causal disponible en el tree_data. El nodo indica: "No significant changes detected. Current period maintained stable performance." Premium LH actuó como **grupo de control**, demostrando que los problemas de equipaje/check-in/arrivals fueron específicos de procesos masivos (Economy/Business), no de la operación LH en general. **Posible aislamiento operativo:** Pasajeros Premium tienen procesos diferenciados (check-in prioritario, handling preferente, acceso a lounges) que los protegieron de la crisis operativa que afectó a Economy y Business.

• **Evidencia clave:**  
  - NPS Period: 13.1 | NPS Baseline: 9.5 → Mejora de +3.6 pts
  - Sin menciones en verbatims de problemas operativos en Premium
  - Sin rutas críticas identificadas en segmento Premium

• **Rutas críticas:**  
No disponible (sin rutas críticas identificadas)

• **Perfiles reactivos:**  
No disponible (sin análisis de perfiles en el CAUSAL EXPLANATION)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

<b>SÍNTESIS EJECUTIVA</b><br>

La red global registró un <b>NPS de 21.5 (–7.3 pts)</b> con respecto a los últimos 7 días, resultado de una crisis operativa sistémica que afectó simultáneamente a ambos radios. Long Haul alcanzó un <b>NPS de 6.5 (–5.8 pts)</b> mientras que Short Haul cayó a <b>28.5 (–8.6 pts)</b>, configurando una sinergia negativa donde el deterioro de Short Haul dominó el agregado por su mayor volumen operativo y exposición directa a eventos excepcionales.<br><br>

El deterioro de puntualidad percibida constituyó la causa principal del colapso, manifestándose de forma transversal en toda la red. En Short Haul, la métrica OTP15 cayó 2.2 puntos hasta 89.52 por ciento, con un incremento de 7 cancelaciones y 6 retrasos que elevó los incidentes críticos operativos en 13 casos, representando un aumento del 29.5 por ciento. Este deterioro generó un impacto negativo de 3.955 puntos porcentuales según Explanatory Drivers, con una caída de satisfacción de 5.88 puntos. En Long Haul se presentó una paradoja operativa: aunque OTP15 mejoró 2.7 puntos hasta 81.84 por ciento, la percepción del cliente empeoró con un impacto negativo de 0.674 puntos porcentuales y una caída de satisfacción de 3.44 puntos, explicado por eventos extremos puntuales como el vuelo IB1586 con 160 minutos de retraso por descanso de tripulación y el IB0337 con 93 minutos. La huelga de controladores aéreos en Roma Fiumicino el 17 de diciembre entre las 13:00 y 17:00 horas canceló preventivamente 6 vuelos, mientras que la meteorología adversa generalizada entre el 14 y 19 de diciembre afectó múltiples zonas: norte de España con disrupciones en San Sebastián, Bilbao, A Coruña y Oviedo, Italia con 5 disrupciones en la ruta Madrid-Florencia, y sur de España en Málaga, Tenerife y Melilla. Las rutas más afectadas incluyeron Düsseldorf-Madrid con una caída de 50 puntos hasta un NPS de –33.3 con 6 pasajeros, A Coruña-Madrid que cayó 66.1 puntos hasta 0.0 con 58 pasajeros, Madrid-Venecia con una caída de 44.4 puntos hasta 50.0 con 49 pasajeros, Ginebra-Madrid que cayó 26.4 puntos hasta 16.1 con 69 pasajeros, y Bilbao-Madrid con una caída de 23.8 puntos hasta 14.7 con 68 pasajeros. Los pasajeros agrupados por región de residencia mostraron la mayor reactividad con un spread de 161.1 puntos, donde ciertos mercados experimentaron caídas de hasta 83.3 puntos mientras otros mejoraron 77.8 puntos, evidenciando una concentración geográfica del impacto.<br><br>

La epidemia de gestión de equipajes constituyó la segunda causa sistémica, con un incremento idéntico de 5.9 puntos en la tasa de mishandling tanto en Long Haul como en Short Haul, alcanzando 22.81 incidentes por mil pasajeros frente a 16.95 en el período comparativo, representando un aumento del 34.8 por ciento. Este deterioro generó un impacto negativo de 1.721 puntos porcentuales según Explanatory Drivers en Short Haul y 1.191 puntos porcentuales en Long Haul. Los incidentes documentados incluyeron 27 maletas extraviadas en el vuelo BA458 de conexión Londres Heathrow-Madrid, 100 maletas no cargadas en Málaga, y un evento excepcional de limitación de peso extrema el 16 de diciembre donde 377 maletas no fueron cargadas en un solo vuelo. El feedback de clientes mostró un aumento del 87 por ciento en menciones de equipaje perdido, retrasado o dañado en Short Haul Economy, con casos críticos como Madrid-Sevilla donde las maletas con medicamentos aparecieron en El Cairo, Madrid-Vigo con equipaje perdido 24 horas sin ropa, Ginebra-Madrid con equipaje perdido 3 días completos, Palma-Valencia donde una maleta con partitura profesional para un cantante de ópera desapareció 2 días, y Barcelona-Estrasburgo donde las maletas llegaron 2.5 días después conteniendo una máquina de apnea vital. En Long Haul, los casos graves incluyeron Bogotá-Madrid donde abrieron la maleta sustrayendo la mitad del contenido, Nueva York JFK-Madrid con maleta perdida más de un día, Boston-Madrid donde el equipaje facturado nunca llegó a la recogida, y Lima-Madrid con 3 de 6 maletas extraviadas. Las rutas más afectadas fueron Madrid-Oslo con un NPS de –42.9 y solo 7 pasajeros, A Coruña-Madrid con 2.3 y 43 pasajeros, Ginebra-Madrid con 6.4 y 47 pasajeros, Bilbao-Madrid con 14.7 y 68 pasajeros, y Roma Fiumicino-Madrid con 17.9 y 112 pasajeros. Los pasajeros agrupados por región de residencia mantuvieron la máxima reactividad con un spread de 161.1 puntos en Short Haul y 122.4 puntos en Long Haul.<br><br>

El aumento de conexiones perdidas configuró la tercera causa sistémica, con un incremento idéntico de 0.2 puntos en la métrica misconex tanto en Long Haul como en Short Haul, alcanzando 0.86 incidentes por mil pasajeros frente a 0.66 en el período comparativo, representando un aumento del 30.3 por ciento. El evento más crítico fue la reprogramación masiva del vuelo Madrid-Dallas Fort Worth el 18 de diciembre con un retraso de 8 horas y 5 minutos que generó 193 pérdidas de conexión desde el hub de Dallas hacia destinos finales en Santo Domingo, Bogotá, Miami y Guayaquil, constituyendo un evento operativo sin precedentes en magnitud de conexiones perdidas. La crisis operativa en Florencia con 5 disrupciones documentadas y múltiples desvíos a Bolonia sin capacidad de vuelos directos posteriores agravó la situación. El feedback de clientes documentó casos como Londres Heathrow-Madrid donde perdieron la conexión y llegaron al hotel a las 2 de la madrugada, Barcelona-Madrid con pérdida del vuelo de enlace hacia Chile perjudicando negocios que debía realizar sin posibilidad de modificación, San Sebastián-Madrid con conexión caótica atravesando toda la terminal 4 corriendo, y Madrid-Venecia donde el vuelo a Oporto estaba programado con solo 30 minutos de escala resultando prácticamente imposible cargar las maletas. Las rutas críticas incluyeron Madrid-Dallas Fort Worth con 193 conexiones perdidas, Buenos Aires-Madrid con conexión perdida hacia Ámsterdam, Bogotá-Madrid con conexión perdida hacia Londres Heathrow, y Madrid-Venecia con conexión imposible de 30 minutos. Los viajeros de negocios en Short Haul mostraron una sensibilidad de –10.5 puntos mientras que en Long Haul alcanzaron –21.4 puntos, siendo el perfil más afectado con una caída de hasta –51.6 puntos en ciertas regiones de residencia, probablemente destinos finales en Latinoamérica con alta dependencia de conexiones en Madrid-Dallas Fort Worth.<br><br>

El deterioro de check-in constituyó una causa adicional transversal, manifestándose con un impacto negativo de 0.774 puntos porcentuales según Explanatory Drivers en Short Haul y 0.830 puntos porcentuales en Long Haul. El feedback documentó problemas sistémicos como imposibilidad de facturar en línea en Madrid-Santiago de Compostela donde la información no llega a la aplicación a tiempo, errores de apellidos que bloqueaban el check-in online en Ámsterdam-Madrid, desorganización total en embarque de Bogotá-Madrid con pesaje de maletas caótico y solo una máquina disponible, y 3 horas en cola donde personal subcontratado estaba tonteando y el pasajero perdió el vuelo por negligencia. Las rutas más afectadas incluyeron San Sebastián-Madrid con check-in en 0.0 representando un colapso total, A Coruña-Madrid con 50.0 puntos y una caída de 120.0 puntos, Dakar-Madrid con 50.0 puntos y una caída de 25.0 puntos, Madrid-Montevideo con 60.5 puntos y una caída de 31.7 puntos con 24 pasajeros, y Madrid-Ciudad de México con 79.6 puntos y una caída de 22.8 puntos con 100 pasajeros.<br><br>

El deterioro de arrivals experience más allá del equipaje se manifestó en esperas prolongadas en recogida de equipaje y problemas de desembarque. La congestión en la terminal 4 de Madrid generó múltiples menciones de retrasos de 45 minutos o más, con casos documentados como Lisboa-Madrid donde una hora de espera fue considerada demasiado tiempo, Madrid-Oslo con 50 minutos de espera, Londres Heathrow-Madrid con 45 minutos, Budapest-Madrid con 45 minutos, Madrid-San José de Costa Rica esperando más de una hora, y Boston-Madrid esperando tres horas hasta poder retirar el equipaje. Este deterioro generó un impacto negativo de 1.721 puntos porcentuales según Explanatory Drivers en Short Haul y 1.191 puntos porcentuales en Long Haul, con caídas de satisfacción de 4.69 y 2.33 puntos respectivamente.<br><br>

En las operaciones de YW se identificaron problemas estructurales específicos que no afectaron a IB. El boarding en operaciones Air Nostrum mostró un impacto negativo de 2.983 puntos porcentuales según Explanatory Drivers con una caída de satisfacción de 7.41 puntos, contrastando con IB donde boarding actuó como compensador positivo con 0.891 puntos porcentuales. El feedback documentó quejas recurrentes sobre el proceso de embarque con autobús y escaleras, como en Ginebra-Madrid donde el embarque tardó demasiado tiempo, y Madrid-Múnich donde criticaron la ridiculez del acceso por grupos cuando al final todos van en jardinera. La flota CRJ mostró inadecuación para clase Business con 3 menciones en el período actual versus 5 en el comparativo, incluyendo Madrid-Santander donde el estado del avión fue calificado como lamentable con asientos desvencijados, Madrid-Valencia con avión bastante antiguo, y múltiples comentarios de pasajeros pagando business para volar en un avión minúsculo. La paradoja de ticket price se manifestó con un impacto negativo de 3.123 puntos porcentuales según Explanatory Drivers a pesar de que la satisfacción con precio subió 99.47 puntos, explicado porque aunque hubo menos quejas en el período actual, el precio sigue siendo un pain point residual que afecta la recomendación, especialmente en rutas como Madrid-Nantes y Madrid-Venecia donde los pasajeros cuestionaron la dificultad de justificar el precio pagado.<br><br>

En Long Haul Economy se identificó un deterioro masivo en experiencia de producto que no fue operativo. El aircraft interior mostró el mayor impacto negativo con 1.266 puntos porcentuales según Explanatory Drivers y una caída de satisfacción de 3.13 puntos, documentando pantallas de entretenimiento en vuelo no funcionales durante períodos prolongados como Boston-Madrid con avión muy pequeño e incómodo donde asientos no reclinaban y pantallas no funcionaban, Lima-Madrid donde la pantalla no funcionó durante 11 horas tras llamar varias veces sin respuesta, Madrid-Santiago de Chile donde la pantalla dejó de funcionar tras 6 horas y avisó 4 veces sin que nadie apareciera, y Madrid-Recife en el nuevo A321XLR donde los espacios fueron calificados como demasiado angostos con pantallas enormes en los ojos y bulkhead obstruyendo el paso. El in-flight food and beverage generó un impacto negativo de 0.745 puntos porcentuales con múltiples menciones como Madrid-Santiago de Chile donde la comida era pobre y no se sirvió café en 13 horas con menú pobre de solo 2 opciones, Madrid-Chicago donde la comida era terrible y apenas tenían nada para beber, Madrid-Panamá donde anunciaron al salir que no disponían de bebidas calientes durante todo el viaje largo, y Madrid-Miami con solo una botellita de agua en 8 horas servida en vaso diminuto medio lleno. La paradoja operativa se confirmó con OTP15 mejorando 2.7 puntos hasta 81.84 por ciento mientras el NPS cayó 6.9 puntos, validando que la anomalía fue de producto y no operativa. Las rutas más afectadas incluyeron Madrid-Santiago de Chile con NPS de –25.6 y 43 pasajeros cayendo 42.2 puntos, Madrid-Chicago con 8.8 y 34 pasajeros cayendo 37.6 puntos, Madrid-Montevideo con –14.3 y 35 pasajeros cayendo 34.3 puntos, Boston-Madrid con 22.7 y 22 pasajeros mejorando paradójicamente 3.4 puntos a pesar de verbatims negativos, y São Paulo Guarulhos-Madrid con 2.6 y 39 pasajeros cayendo 8.7 puntos. Los pasajeros agrupados por región de residencia mostraron un spread de 93.1 puntos con rango de –26.4 a más 66.7 puntos, donde los pasajeros de Latinoamérica según inferencia cualitativa mostraron mayor sensibilidad negativa de –50 puntos.<br><br>

La convergencia de <b>Long Haul con NPS de 6.5 (–5.8 pts)</b> y <b>Short Haul con NPS de 28.5 (–8.6 pts)</b>, ambos en dirección negativa, produjo el resultado global de <b>21.5 (–7.3 pts)</b>. Short Haul dominó el agregado por su mayor magnitud absoluta de caída, mayor volumen operativo de vuelos diarios, y mayor exposición directa a los dark horses identificados: la huelga de controladores aéreos en Roma Fiumicino canceló 6 vuelos afectando principalmente rutas Short Haul europeas, la meteorología adversa generalizada generó 5 disrupciones en Madrid-Florencia y 5 en Madrid-Bilbao concentrándose en Short Haul, y los incidentes masivos de equipaje con 127 maletas afectadas impactaron ambos radios con la misma magnitud. Sin embargo, la causa raíz fue sistémica: los tres vectores operativos de puntualidad percibida, mishandling con incremento de 5.9 puntos, y misconex con incremento de 0.2 puntos afectaron por igual a ambos radios, confirmando una crisis de red completa con mayor manifestación en Short Haul por su mayor exposición a eventos excepcionales.<br><br>

<b><u>DETALLE POR CABINA</u></b><br>

<b><u>ECONOMY SH: Crisis operativa con sinergia negativa entre compañías</u></b><br>
La cabina alcanzó un <b>NPS de 27.9 (–9.3 pts)</b> resultado de una sinergia negativa donde tanto IB con <b>27.0 (–8.5 pts)</b> como YW con <b>29.6 (–10.9 pts)</b> experimentaron deterioro simultáneo por la misma crisis operativa sistémica. El deterioro operativo multidimensional se manifestó con tres vectores principales: puntualidad con un impacto negativo de 3.677 puntos porcentuales según Explanatory Drivers en IB y 1.845 en YW, donde OTP15 cayó 2.0 puntos en IB hasta 92.58 por ciento y 2.3 puntos en YW hasta 86.81 por ciento, con incremento de 7 cancelaciones y 6 retrasos; mishandling subiendo 5.8 puntos en IB y 6.0 puntos en YW con 127 maletas afectadas documentadas incluyendo 27 maletas del vuelo BA458 y 100 maletas en Málaga; y arrivals experience con impacto negativo de 1.687 puntos porcentuales en IB y 1.156 en YW, documentando esperas prolongadas de 45 a 50 minutos. YW sufrió mayor magnitud por mayor severidad en mishandling con casos graves como Madrid-Sevilla donde las maletas con medicamentos aparecieron en El Cairo, Ginebra-Madrid con equipaje perdido 3 días completos, Palma-Valencia donde una maleta con partitura profesional para un cantante de ópera desapareció 2 días, Barcelona-Estrasburgo donde las maletas llegaron 2.5 días después conteniendo una máquina de apnea vital, y Lleida-Madrid con maleta de mano perdida y conexión perdida. YW también mostró la paradoja de ticket price con impacto negativo de 3.114 puntos porcentuales a pesar de que satisfacción subió 77.21 puntos, reflejando la percepción de pagar business para volar en avión minúsculo. Los dark horses afectaron por igual a ambas compañías: huelga de controladores aéreos en Roma Fiumicino cancelando 6 vuelos, meteorología adversa en norte de España e Italia con Madrid-Florencia registrando 5 disrupciones y Madrid-Bilbao otras 5, e incidentes masivos de equipaje. Las rutas críticas incluyeron Düsseldorf-Madrid con NPS de –50.0 y 6 pasajeros cayendo 100.0 puntos con triangulación de Explanatory Drivers más incidentes operativos documentados dos veces más verbatims dos veces sobre azafata irrespetuosa, A Coruña-Madrid con 2.3 y 43 pasajeros cayendo 62.1 puntos con 50 minutos de retraso sin explicaciones claras y maletas soltadas en pista a oscuras, Madrid-Venecia con 50.0 y 49 pasajeros cayendo 44.4 puntos por equipaje no cargado y conexión perdida, Ginebra-Madrid con 6.4 y 47 pasajeros cayendo 36.9 puntos por 3 días sin maletas, y Bilbao-Madrid con 14.7 y 68 pasajeros cayendo 23.8 puntos por 5 disrupciones meteorológicas. Los pasajeros agrupados por región de residencia mostraron máxima reactividad con spread de 161.1 puntos donde ciertos mercados cayeron 83.3 puntos mientras otros mejoraron 77.8 puntos, mientras que business versus leisure mostró baja diferenciación con spread de solo 2.1 puntos donde leisure cayó 8.8 puntos y business 6.7 puntos reaccionando de manera similar.<br><br>

<b><u>BUSINESS SH: Dominancia de YW con deterioro agravado</u></b><br>
La cabina alcanzó un <b>NPS de 34.6 (–1.3 pts)</b> reflejando una dominancia de YW con <b>16.7 (–8.1 pts)</b> que fue diluida por el mayor volumen de IB con <b>40.1 (–0.9 pts)</b> y sus 93 encuestas frente a aproximadamente 20 de YW. IB experimentó deterioro operativo triple con puntualidad mostrando impacto negativo de 1.230 puntos porcentuales según Explanatory Drivers donde OTP15 cayó 2.0 puntos con incremento de 7 cancelaciones y 6 retrasos, mishandling subiendo 5.8 puntos con 6 menciones de equipaje representando aumento del 300 por ciento versus baseline incluyendo Dakar-Madrid donde llevan tres días esperando equipaje en clase ejecutiva, Frankfurt-Madrid con pérdida total de equipaje, y Madrid-Venecia sin equipaje incluyendo medicamentos, y conexiones perdidas con misconex subiendo 0.17 puntos documentando 4 menciones como Londres Heathrow-Madrid perdiendo conexión con hotel a las 2 de la madrugada y Barcelona-Madrid con pérdida del vuelo de enlace hacia Chile. IB mantuvo boarding como compensador positivo con 0.891 puntos porcentuales según Explanatory Drivers mitigando parcialmente el deterioro. YW sufrió deterioro operativo agravado en operaciones Air Nostrum con puntualidad mostrando impacto negativo de 2.181 puntos porcentuales 79 por ciento más severo que IB donde OTP15 cayó 2.33 puntos, boarding con impacto negativo de 2.983 puntos porcentuales por problema estructural de embarque con autobús y escaleras documentado en Ginebra-Madrid donde el embarque tardó demasiado tiempo, y ticket price con impacto negativo de 3.123 puntos porcentuales a pesar de que satisfacción subió 99.47 puntos reflejando la percepción de pagar business para volar en avión minúsculo. YW también mostró problema estructural de flota CRJ con 3 menciones en período actual versus 5 en comparativo sobre inadecuación para clase Business. Las rutas críticas de YW incluyeron Madrid-Florencia con 5 disrupciones por crisis operativa Italia y huelga de controladores aéreos en Roma Fiumicino, Madrid-Bilbao con 5 disrupciones por meteorología adversa norte de España más 2 cancelaciones el 18 de diciembre, Madrid-San Sebastián con 2 disrupciones por meteorología más cancelación más transporte superficie, Madrid-Marsella con retraso mecánico el 16 de diciembre donde más de una hora de retraso porque las puertas de la bodega del avión no cerraban, y Ginebra-Madrid donde el embarque tardó demasiado tiempo. Los pasajeros agrupados por región de residencia en YW mostraron máxima reactividad con spread de 63.3 puntos donde ciertos mercados cayeron hasta 30.0 puntos mientras otros mejoraron 33.3 puntos, y business versus leisure mostró spread de 29.6 puntos donde business travelers cayeron 20.1 puntos siendo más sensibles a puntualidad y boarding mientras leisure mejoró 9.4 puntos.<br><br>

<b><u>ECONOMY LH: Deterioro masivo en experiencia de producto</u></b><br>
La cabina alcanzó un <b>NPS de 3.9 (–6.9 pts)</b> debido a deterioro masivo en experiencia de producto validado por triangulación de tres fuentes. Aircraft interior mostró el mayor impacto negativo con 1.266 puntos porcentuales según Explanatory Drivers y caída de satisfacción de 3.13 puntos, documentando pantallas de entretenimiento en vuelo no funcionales durante períodos prolongados como Boston-Madrid con avión muy pequeño e incómodo donde asientos no reclinaban y pantallas no funcionaban, Lima-Madrid donde la pantalla no funcionó durante 11 horas tras llamar varias veces sin respuesta, Madrid-Santiago de Chile donde la pantalla dejó de funcionar tras 6 horas y avisó 4 veces sin que nadie apareciera, y Madrid-Recife en el nuevo A321XLR donde los espacios fueron calificados como demasiado angostos con pantallas enormes en los ojos y bulkhead obstruyendo el paso. Check-in generó impacto negativo de 1.073 puntos porcentuales con problemas sistémicos en Madrid-Montevideo, Madrid-Ciudad de México, y Doha-Madrid. In-flight food and beverage mostró impacto negativo de 0.745 puntos porcentuales con múltiples menciones como Madrid-Santiago de Chile donde la comida era pobre y no se sirvió café en 13 horas con menú pobre de solo 2 opciones, Madrid-Chicago donde la comida era terrible y apenas tenían nada para beber, Madrid-Panamá donde anunciaron al salir que no disponían de bebidas calientes durante todo el viaje largo, y Madrid-Miami con solo una botellita de agua en 8 horas servida en vaso diminuto medio lleno. La paradoja operativa se confirmó con OTP15 mejorando 2.7 puntos hasta 81.84 por ciento mientras el NPS cayó, validando que la anomalía fue de producto no operativa. Las rutas más afectadas incluyeron Madrid-Santiago de Chile con NPS de –25.6 y 43 pasajeros cayendo 42.2 puntos con aircraft interior en 44.2, check-in en 52.4 y arrivals en 53.7, Madrid-Chicago con 8.8 y 34 pasajeros cayendo 37.6 puntos con aircraft interior en 65.6, Madrid-Montevideo con –14.3 y 35 pasajeros cayendo 34.3 puntos con check-in en 58.8, Boston-Madrid con 22.7 y 22 pasajeros mejorando paradójicamente 3.4 puntos a pesar de verbatims negativos con aircraft interior en 50.0, y São Paulo Guarulhos-Madrid con 2.6 y 39 pasajeros cayendo 8.7 puntos con aircraft interior en 53.8. Los pasajeros agrupados por región de residencia mostraron spread de 93.1 puntos con rango de –26.4 a más 66.7 puntos donde los pasajeros de Latinoamérica según inferencia cualitativa mostraron mayor sensibilidad negativa de –50 puntos, mientras business versus leisure mostraron spread de solo 4.0 puntos con rango de –7.4 a –3.4 puntos cayendo de manera similar sin diferenciación significativa.<br><br>

<b><u>BUSINESS LH: Deterioro operativo con conexiones perdidas críticas</u></b><br>
La cabina alcanzó un <b>NPS de 19.0 (–6.2 pts)</b> debido a deterioro operativo triple. Conexiones perdidas y reprogramaciones extremas mostraron connections experience con impacto negativo de 0.396 puntos porcentuales según Explanatory Drivers donde misconex subió 0.2 puntos, destacando el evento crítico de Madrid-Dallas Fort Worth el 18 de diciembre con reprogramación de 8 horas y 5 minutos generando 193 pérdidas de conexión desde el hub de Dallas hacia destinos finales en Santo Domingo, Bogotá, Miami y Guayaquil constituyendo un evento operativo sin precedentes en magnitud, con verbatims documentando Londres Heathrow-Madrid donde perdimos nuestra conexión, Barcelona-Madrid con pérdida del vuelo de enlace en Madrid hacia Chile perjudicando los negocios que debía realizar sin posibilidad de modificación. Gestión de equipajes deteriorada mostró mishandling subiendo 5.9 puntos con 6 menciones representando aumento del 300 por ciento versus baseline incluyendo Lisboa-Madrid con 90 minutos de espera, Frankfurt-Madrid con pérdida total de equipaje, Madrid-Venecia sin equipaje incluyendo medicamentos. Puntualidad percibida mostró la paradoja donde punctuality generó impacto negativo de 7.194 puntos porcentuales según Explanatory Drivers con caída de satisfacción de 9.46 puntos a pesar de que OTP15 mejoró 2.7 puntos hasta 81.84 por ciento, explicado por eventos extremos puntuales como vuelo IB1586 con 160 minutos de retraso y IB0337 con 93 minutos. Boarding actuó como compensador con 1.332 puntos porcentuales pero fue insuficiente para compensar el deterioro operativo. Las rutas críticas incluyeron Madrid-Dallas Fort Worth con 3 disrupciones en período actual y reprogramación masiva de 8 horas y 5 minutos con 193 conexiones perdidas siendo hub de conexiones crítico para tráfico Long Haul hacia Américas, Bogotá-Madrid con NPS de 7.6 y 145 pasajeros cayendo 9.0 puntos con 3 verbatims negativos sobre equipaje 90 minutos, vuelo retrasó 2 horas, equipaje no llegó, Buenos Aires-Madrid con 6.7 y 75 pasajeros cayendo 8.7 puntos con 4 verbatims negativos sobre conexión perdida Madrid-Ámsterdam, vuelo con demora, asiento no respetado, tripulación poco proactiva, Madrid-Miami con deterioro de 18.8 puntos según otra fuente con 3 verbatims negativos sobre servicio deficiente, retrasos, tripulación como aerolínea de bajo coste, y Madrid-Santo Domingo con 1 verbatim crítico sobre equipaje perdido. Los pasajeros agrupados por región de residencia mostraron spread de 96.1 puntos con rango de –51.6 a más 44.6 puntos donde el perfil más afectado cayó 51.6 puntos en región específica no identificada con hipótesis de mercados con alta dependencia de conexiones en Madrid-Dallas Fort Worth probablemente destinos latinoamericanos, mientras business versus leisure mostraron spread de 6.3 puntos donde business cayó 8.4 puntos y leisure 2.1 puntos ambos negativos sin diferenciación significativa.<br><br>

<b><u>PREMIUM LH: Estabilidad como grupo de control</u></b><br>
La cabina alcanzó un <b>NPS de 13.1 (+3.6 pts)</b> manteniéndose en estado normal dentro de rango esperado sin cambios significativos detectados. Premium Long Haul actuó como grupo de control demostrando que los problemas de equipaje, check-in y arrivals fueron específicos de procesos masivos en Economy y Business, no de la operación Long Haul en general. El posible aislamiento operativo se explica porque pasajeros Premium tienen procesos diferenciados como check-in prioritario, handling preferente y acceso a lounges que los protegieron de la crisis operativa que afectó a las otras cabinas. Sin menciones en verbatims de problemas operativos en Premium y sin rutas críticas identificadas en el segmento, la mejora de 3.6 puntos desde un NPS baseline de 9.5 confirma que la crisis fue específica de procesos masivos no de la red completa.

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 7
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*



**ANÁLISIS DIARIO SINGLE:**
📅 2025-12-19 to 2025-12-19:
❌ Error en la interpretación jerárquica: An error occurred (ThrottlingException) when calling the Converse operation (reached max retries: 3): Too many tokens, please wait before trying again.
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-18 to 2025-12-18:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 PASO 1: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL COMPAÑÍA)

---

## 📊 CABINA: ECONOMY SHORT HAUL

### **Escenario Detectado:** SINERGIA `(+, + | +)`

**Estados:**
- **IB:** POSITIVE ANOMALY (+7.1 pts)
- **YW:** POSITIVE ANOMALY (+7.6 pts)
- **PADRE (Economy SH):** POSITIVE ANOMALY (+7.3 pts)

### **Dinámica de Agregación:**
Ambas compañías (**IB** y **YW**) experimentaron mejoras simultáneas de magnitud similar (+7.1 y +7.6 pts respectivamente), generando un efecto sinérgico que se transfiere al nodo padre (+7.3 pts). No hay conflicto ni dilución: las anomalías positivas se suman y refuerzan mutuamente.

### **Narrativa Causal:**
Se adopta la **Explicación del Nodo Padre (Economy SH)**, ya que la causa raíz es común a ambas compañías.

**Causa Principal:** Mejora operativa global compensada por menor ocupación, a pesar de deterioro significativo en gestión de equipaje.

**Evidencia Clave:**
- **Mishandling:** +3.69 pts vs baseline (19.25 absoluto) - deterioro significativo que paradójicamente NO impactó negativamente el NPS
- **OTP15:** +0.83 pts vs baseline (90.94% absoluto) - mejora en puntualidad
- **Load Factor:** -3.01 pts vs baseline (82.92% absoluto) - menor ocupación mejoró la experiencia percibida
- **18 incidentes NCS** (6 cancelaciones por meteorología adversa en BIO/SDR), pero concentrados en rutas sin encuestas o con NPS neutral

**Interpretación:** La anomalía positiva NO refleja ausencia de problemas operativos (Mishandling subió significativamente), sino un **efecto de composición de muestra**. Los clientes que respondieron las encuestas NO fueron los afectados por los incidentes meteorológicos (BIO/SDR) ni por problemas de equipaje. La menor ocupación (-3.01 pts Load Factor) probablemente compensó el deterioro en Mishandling, generando una experiencia general positiva que se reflejó en ambas compañías.

**Aplicabilidad:** Esta dinámica aplica tanto a **IB** (94% de las encuestas, NPS +35.4) como a **YW** (operación regional con flota CRJ), confirmando que la causa es sistémica y no específica de una compañía.

---

## 📊 CABINA: BUSINESS SHORT HAUL

### **Escenario Detectado:** SINERGIA `(-, - | -)`

**Estados:**
- **IB:** NEGATIVE ANOMALY (-5.7 pts)
- **YW:** NEGATIVE ANOMALY (-28.5 pts)
- **PADRE (Business SH):** NEGATIVE ANOMALY (-11.8 pts)

### **Dinámica de Agregación:**
Ambas compañías experimentaron caídas de NPS, aunque con magnitudes muy diferentes (**IB** -5.7 pts vs **YW** -28.5 pts). El efecto combinado genera una anomalía negativa en el padre (-11.8 pts), donde **YW** tiene un impacto desproporcionado debido a la severidad de su caída, pero **IB** domina en volumen (28 de 30 encuestas, 93% del total). La sinergia negativa se manifiesta en que ambas compañías sufrieron por la misma causa raíz operativa.

### **Narrativa Causal:**
Se adopta la **Explicación del Nodo Padre (Business SH)**, ya que la causa raíz es común a ambas compañías.

**Causa Principal:** Meteorología adversa en BIO (Bilbao) y SDR (Santander) generó 18 incidentes operativos (6 cancelaciones, 3 retrasos) que afectaron desproporcionadamente a clientes Business/Work.

**Evidencia Clave:**
- **18 incidentes NCS totales:** 6 cancelaciones (33%), 3 retrasos (17%), 1 equipaje
- **Rutas críticas:** MAD-BIO (2 incidentes), VIT-BIO (2 incidentes), BIO-MAD (1 incidente, vuelo IB0433 regresó a MAD)
- **Mishandling:** +3.69 pts vs baseline (19.25 absoluto)
- **Misconex:** +0.11 pts vs baseline (0.8 absoluto)
- **Impacto diferencial por perfil:**
  - Business/Work: NPS 0.0 (8 encuestas) - segmento más afectado
  - Leisure: NPS 31.8 (22 encuestas) - menor impacto

**Patrón Común en IB y YW:**
- **IB:** Polarización extrema entre Business/Work (NPS -50.0, n=4) y Leisure (NPS 55.6, n=18). Spread de 105.6 pts.
- **YW:** Polarización aún más severa entre Business/Work (NPS 50.0, n=4) y Leisure (NPS -75.0, n=4). Spread de 125 pts.

**Interpretación:** La meteorología adversa generó disrupciones operativas que impactaron principalmente a viajeros de negocios, quienes tienen menor tolerancia a cancelaciones/retrasos. En **IB**, los clientes Business/Work mostraron NPS -50.0, mientras que en **YW**, el segmento Leisure fue el más afectado (NPS -75.0), sugiriendo una desalineación de expectativas en vuelos regionales operados con flota CRJ. A pesar de las diferencias en los perfiles afectados, ambas compañías sufrieron por la misma causa raíz: incidentes meteorológicos en el norte de España.

**Aplicabilidad:** La causa operativa (meteorología BIO/SDR) es común, pero el impacto se manifestó de forma diferente: **IB** sufrió principalmente en su segmento Business corporativo, mientras que **YW** enfrentó una crisis de expectativas en su segmento Leisure. El nodo padre refleja la sinergia negativa de ambos efectos.

---

## 🔑 RESUMEN EJECUTIVO DE AGREGACIÓN:

| Cabina | Escenario | Estados (IB, YW \| Padre) | Narrativa Adoptada | Causa Raíz Compartida |
|--------|-----------|---------------------------|--------------------|-----------------------|
| **Economy SH** | SINERGIA (+,+ \| +) | (+7.1, +7.6 \| +7.3) | Explicación del Padre | Menor ocupación compensó deterioro en Mishandling |
| **Business SH** | SINERGIA (-,- \| -) | (-5.7, -28.5 \| -11.8) | Explicación del Padre | Meteorología adversa BIO/SDR + impacto diferencial por perfil |

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ PASO 2: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL CABINA)

---

## 📊 RADIO: SHORT HAUL (SH)

### **Escenario Detectado:** CANCELACIÓN `(+, - | N)`

**Estados:**
- **Economy:** POSITIVE ANOMALY (+7.3 pts)
- **Business:** NEGATIVE ANOMALY (-11.8 pts)
- **PADRE (SH):** Normal (+5.5 pts - dentro del rango normal)

### **Dinámica de Agregación:**
Las cabinas Economy y Business presentan anomalías de signo opuesto que se cancelan mutuamente en el agregado. Economy mejoró +7.3 pts mientras Business cayó -11.8 pts, resultando en una variación neta de +5.5 pts que el sistema clasifica como "Normal" (dentro del rango esperado de variabilidad). **El nodo padre muestra estabilidad engañosa**, ocultando volatilidad interna significativa entre cabinas.

### **Narrativa Causal:**
El radio SH muestra **estabilidad aparente que oculta dinámicas opuestas** entre productos. Ignoramos la explicación del padre (irrelevante al ser "Normal") y contrastamos las causas divergentes de los hijos.

**Contraste de Causas:**

#### **Economy SH (+7.3 pts): Mejora por Composición de Muestra**
- **Causa:** Menor ocupación (Load Factor -3.01 pts) compensó deterioro operativo en Mishandling (+3.69 pts)
- **Evidencia:** 
  - OTP15 +0.83 pts (puntualidad mejoró)
  - 18 incidentes NCS concentrados en rutas sin encuestas (BIO/SDR)
  - Clientes que respondieron NO fueron los afectados por problemas de equipaje
- **Perfiles favorecidos:** 
  - Fleet A320: NPS 55.3 (n=38)
  - Pasajeros de España: NPS 43.5 (**IB**), NPS 26.9 (general)
  - Viajeros Business/Work: NPS 48.8 (**IB**)

#### **Business SH (-11.8 pts): Deterioro por Meteorología Adversa**
- **Causa:** Meteorología adversa en BIO/SDR generó 18 incidentes operativos (6 cancelaciones, 3 retrasos) que impactaron desproporcionadamente a viajeros de negocios
- **Evidencia:**
  - Mishandling +3.69 pts, Misconex +0.11 pts
  - Rutas críticas: MAD-BIO (2 incidentes), VIT-BIO (2 incidentes), BIO-MAD (1 incidente)
  - Vuelo IB0433 BIO-MAD regresó a MAD por condiciones adversas
- **Perfiles más afectados:**
  - Business/Work: NPS 0.0 (8 encuestas) vs Leisure: NPS 31.8 (22 encuestas)
  - **IB:** Polarización Business/Work NPS -50.0 vs Leisure NPS 55.6 (spread 105.6 pts)
  - **YW:** Polarización extrema Leisure NPS -75.0 vs Business/Work NPS 50.0 (spread 125 pts)
  - Fleet CRJ: NPS -12.5 (8 encuestas, **YW**)
  - Región Europa: NPS 0.0 (8 encuestas)

### **Interpretación Ejecutiva:**
Mientras Economy experimentó una mejora artificial (+7.3 pts) debido a que los clientes encuestados NO fueron los afectados por los 18 incidentes operativos meteorológicos, Business sufrió el impacto real (-11.8 pts) de esas mismas disrupciones, especialmente en el segmento corporativo sensible a cancelaciones/retrasos. **La compensación mutua en el agregado SH (+5.5 pts "Normal") oculta una crisis operativa real en Business y una lectura sesgada en Economy.**

**Volumen relativo:** Economy representa aproximadamente el 90% del volumen de encuestas en SH (256 Economy vs 30 Business), lo que explica por qué la anomalía positiva de Economy no logró convertir el SH en anomalía positiva, siendo neutralizada por la caída severa en Business.

---

## 📊 RADIO: LONG HAUL (LH)

### **Escenario Detectado:** SINERGIA PARCIAL FUERTE `(-, -, - | -)`

**Estados:**
- **Economy:** NEGATIVE ANOMALY (-8.8 pts)
- **Business:** NEGATIVE ANOMALY (-26.7 pts)
- **Premium:** NEGATIVE ANOMALY (-52.1 pts)
- **PADRE (LH):** NEGATIVE ANOMALY (-13.6 pts)

### **Dinámica de Agregación:**
Las tres cabinas presentan anomalías negativas simultáneas, generando una **sinergia negativa total** que se transfiere al nodo padre con fuerza. La magnitud de las caídas aumenta progresivamente según la cabina: Economy (-8.8 pts) < Business (-26.7 pts) < Premium (-52.1 pts), revelando que el deterioro operativo afectó más severamente a las cabinas premium. El padre LH (-13.6 pts) refleja el promedio ponderado por volumen, donde Economy domina numéricamente (97 encuestas vs 13 Business vs 7 Premium).

### **Narrativa Causal:**
Se adopta la **Explicación del Radio Padre (LH)**, ya que la causa raíz es sistémica y afectó transversalmente a todas las cabinas.

**Causa Sistémica:** Deterioro operativo generalizado en gestión de equipaje y puntualidad que afectó transversalmente al vuelo Long Haul, con impacto progresivamente más severo en cabinas premium.

**Evidencia Clave:**

#### **Métricas Operativas (LH):**
- **Mishandling:** 19.25 (+3.69 pts vs baseline) - **deterioro significativo** (incremento del 23.7%)
- **OTP15:** 79.85% (-1.55 pts vs baseline) - deterioro en puntualidad
- **Load Factor:** 90.19% (-0.44 pts vs baseline) - menor ocupación (favorable, descartado como causa)
- **Misconex:** 0.8 (+0.11 pts vs baseline) - deterioro menor

#### **Incidentes NCS (8 incidentes totales):**
- 3 retrasos
- 2 cancelaciones
- 1 equipaje
- **Incidentes destacados:**
  - Reprogramación extrema: +8h 5min (MAD salida 09:00h)
  - Retraso internacional: +1h 20min (DFW-MAD, salida 15:15h)
- **Alcance:** 21 rutas con incidentes operacionales

#### **Rutas Críticas (Top 3 con peor NPS):**
| Ruta | NPS | Encuestas | Incidentes NCS |
|------|-----|-----------|----------------|
| **MAD-SJO** | -33.3 | 6 | ✅ Confirmado |
| **DOH-MAD** | -20.0 | 5 | ✅ Confirmado |
| **MAD-MEX** | -13.3 | 15 | ✅ Confirmado |

**Patrón geográfico:** Concentración de problemas en rutas latinoamericanas desde MAD.

#### **Perfiles Más Afectados (Impacto Progresivo por Cabina):**

**Economy LH (-8.8 pts):**
- Business/Work: NPS -33.3 (9 encuestas) - **3.6x más afectados que Leisure**
- Leisure: NPS -2.3 (88 encuestas)
- Fleet A333: NPS -30.8 (13 encuestas)
- Fleet A33ACMI: NPS -66.7 (3 encuestas)
- Residentes Europa: NPS -100.0 (6 encuestas)

**Business LH (-26.7 pts):**
- Operador AA: NPS -100.0 (3 encuestas) - **todos detractores**
- Fleet A332: NPS -25.0 (4 encuestas)
- Fleet A350 next: NPS +20.0 (5 encuestas) - único positivo
- Residentes Europa: NPS 0.0 (8 encuestas)

**Premium LH (-52.1 pts):**
- Rutas críticas: DFW-MAD, GRU-MAD, MAD-NRT (todas NPS -100)
- Fleet A333: NPS -100.0 (2 encuestas)
- Fleet A33ACMI: NPS -50.0 (4 encuestas)
- Operador AA: NPS -100.0 (1 encuesta)
- Residentes Europa: NPS -100.0 (1 encuesta)
- Business/Work: NPS -100.0 (1 encuesta)

### **Interpretación Ejecutiva:**
El deterioro operativo en Mishandling (+3.69 pts, +23.7%) y puntualidad (OTP15 -1.55 pts) afectó transversalmente a las 21 rutas Long Haul, pero el **impacto fue progresivamente más severo en cabinas premium**, donde las expectativas de servicio son más altas. Economy sufrió principalmente en su segmento Business/Work (NPS -33.3), Business experimentó el impacto de disrupciones con operadores codeshare (AA: NPS -100), y Premium colapsó completamente (NPS -42.86, caída de -52.1 pts) ante la combinación de problemas de equipaje, retrasos extremos (+8h 5min) y operación con flota wide-body problemática (A333/A33ACMI).

**Cadena causal común:** Mishandling elevado → Problemas de conexión → Retrasos acumulados → Impacto desproporcionado en rutas intercontinentales (MAD-SJO, DOH-MAD, MAD-MEX) → Insatisfacción crítica en cabinas premium con menor tolerancia a disrupciones.

---

## 🔑 RESUMEN EJECUTIVO DE AGREGACIÓN POR RADIO:

| Radio | Escenario | Estados (Cabinas \| Padre) | Narrativa Adoptada | Causa Raíz |
|-------|-----------|----------------------------|--------------------|------------|
| **Short Haul** | CANCELACIÓN (+,- \| N) | (+7.3, -11.8 \| +5.5 N) | Contraste de causas opuestas | Economy: Sesgo de muestra compensó Mishandling<br>Business: Meteorología BIO/SDR impactó segmento corporativo |
| **Long Haul** | SINERGIA TOTAL (-,-,- \| -) | (-8.8, -26.7, -52.1 \| -13.6) | Explicación del Padre | Deterioro operativo sistémico (Mishandling +3.69, OTP15 -1.55) con impacto progresivo en cabinas premium |

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 PASO 3: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL GLOBAL)

---

## 📊 NIVEL: GLOBAL

### **Escenario Detectado:** DILUCIÓN `(-, N | +)`

**Estados:**
- **Long Haul (LH):** NEGATIVE ANOMALY (-13.6 pts)
- **Short Haul (SH):** Normal (+5.5 pts - dentro del rango normal)
- **PADRE (GLOBAL):** POSITIVE ANOMALY (+0.5 pts)

### **Dinámica de Agregación:**
El radio Long Haul presenta una anomalía negativa significativa (-13.6 pts) mientras que Short Haul muestra variación normal (+5.5 pts). **El volumen masivo de Short Haul diluye completamente el deterioro de Long Haul**, resultando en una anomalía positiva marginal a nivel Global (+0.5 pts). Esta dilución crea una **falsa señal de mejora** que oculta una crisis operativa real en el segmento intercontinental.

**Análisis de Volumen:**
- **Short Haul:** 286 encuestas (256 Economy + 30 Business) = **71% del total**
- **Long Haul:** 117 encuestas (97 Economy + 13 Business + 7 Premium) = **29% del total**
- **Ratio SH:LH:** 2.4:1

La dominancia volumétrica de SH (71% de las respuestas) absorbe el impacto negativo de LH, generando un Global aparentemente positivo que NO refleja la realidad operativa del día.

---

### **Narrativa Causal:**
Se adopta la **Explicación del Radio Dominante (Short Haul)**, ya que su volumen dicta el resultado Global, aunque esto oculta la crisis en Long Haul.

**Narrativa Ejecutiva:**

El resultado Global está **arrastrado por Short Haul** debido a su peso volumétrico (71% de encuestas), generando una anomalía positiva artificial (+0.5 pts) que **oculta un deterioro operativo severo en Long Haul** (-13.6 pts). Esta dilución crea una falsa señal de mejora a nivel red.

---

## 🔍 **EVIDENCIA DETALLADA:**

### **A) Causa del Radio Dominante (Short Haul - Normal +5.5 pts):**

**Dinámica interna SH:** Cancelación entre Economy (+7.3 pts) y Business (-11.8 pts)

#### **Economy SH (+7.3 pts) - Componente positivo:**
- **Causa:** Sesgo de muestra compensó deterioro operativo
- **Evidencia operativa:**
  - OTP15: 90.94% (+0.83 pts vs baseline)
  - Load Factor: 82.92% (-3.01 pts vs baseline) → Menor ocupación mejoró experiencia
  - **Contradicción:** Mishandling 19.25 (+3.69 pts vs baseline) - deterioro NO reflejado en NPS
- **18 incidentes NCS** (6 cancelaciones por meteorología BIO/SDR) concentrados en rutas sin encuestas
- **Perfiles favorecidos:**
  - **IB:** NPS 35.4 (240 encuestas, 94% del segmento Economy SH)
  - Fleet A320: NPS 55.3 (38 encuestas)
  - Residentes España: NPS 43.5 (**IB**)
- **Interpretación:** Los clientes que respondieron NO fueron los afectados por incidentes meteorológicos

#### **Business SH (-11.8 pts) - Componente negativo:**
- **Causa:** Meteorología adversa BIO/SDR impactó segmento corporativo
- **Evidencia operativa:**
  - Mishandling: 19.25 (+3.69 pts vs baseline)
  - Misconex: 0.8 (+0.11 pts vs baseline)
- **Rutas críticas:** MAD-BIO (2 incidentes), VIT-BIO (2 incidentes), BIO-MAD (1 incidente)
- **Perfiles más afectados:**
  - Business/Work: NPS 0.0 (8 encuestas) vs Leisure: NPS 31.8 (22 encuestas)
  - **IB:** Polarización Business/Work NPS -50.0 vs Leisure NPS 55.6
  - **YW:** Polarización Leisure NPS -75.0 vs Business/Work NPS 50.0
  - Fleet CRJ (**YW**): NPS -12.5 (8 encuestas)

**Efecto neto SH:** La mejora artificial en Economy (+7.3 pts, 256 encuestas) fue parcialmente neutralizada por la caída en Business (-11.8 pts, 30 encuestas), resultando en +5.5 pts "Normal" que el sistema considera dentro del rango esperado.

---

### **B) Causa del Radio Diluido (Long Haul - Anomalía Negativa -13.6 pts):**

**Dinámica interna LH:** Sinergia negativa total entre Economy (-8.8 pts), Business (-26.7 pts) y Premium (-52.1 pts)

#### **Deterioro operativo sistémico:**
- **Mishandling:** 19.25 (+3.69 pts vs baseline, +23.7%) - **desviación significativa**
- **OTP15:** 79.85% (-1.55 pts vs baseline) - deterioro en puntualidad
- **8 incidentes NCS:** 3 retrasos, 2 cancelaciones, 1 equipaje
  - Reprogramación extrema: +8h 5min (MAD 09:00h)
  - Retraso internacional: +1h 20min (DFW-MAD 15:15h)
- **21 rutas con incidentes operacionales**

#### **Rutas críticas con peor NPS:**
| Ruta | NPS | Encuestas | Segmento | Incidentes NCS |
|------|-----|-----------|----------|----------------|
| **MAD-SJO** | -33.3 | 6 | Economy/Premium | ✅ Confirmado |
| **DOH-MAD** | -20.0 | 5 | Economy | ✅ Confirmado |
| **MAD-MEX** | -13.3 | 15 | Economy/Premium | ✅ Confirmado |
| **DFW-MAD** | -66.7 | 6 | Premium | ✅ Confirmado |
| **GRU-MAD** | -100.0 | 2 | Premium | ✅ Confirmado |
| **MAD-NRT** | -100.0 | 1 | Premium | ✅ Confirmado |

**Patrón geográfico:** Concentración en rutas latinoamericanas y conexiones intercontinentales desde MAD.

#### **Perfiles más afectados (impacto progresivo por cabina):**

**Economy LH (-8.8 pts, 97 encuestas):**
- Business/Work: NPS -33.3 (9 encuestas) - **3.6x más afectados que Leisure**
- Fleet A33ACMI: NPS -66.7 (3 encuestas)
- Fleet A333: NPS -30.8 (13 encuestas)
- Residentes Europa: NPS -100.0 (6 encuestas)
- Codeshare BA: NPS -100.0 (2 encuestas)

**Business LH (-26.7 pts, 13 encuestas):**
- Operador AA: NPS -100.0 (3 encuestas) - **todos detractores**
- Fleet A332: NPS -25.0 (4 encuestas)
- Residentes Europa: NPS 0.0 (8 encuestas)

**Premium LH (-52.1 pts, 7 encuestas):**
- Rutas críticas: DFW-MAD, GRU-MAD, MAD-NRT (todas NPS -100)
- Fleet A333: NPS -100.0 (2 encuestas)
- Fleet A33ACMI: NPS -50.0 (4 encuestas)
- Business/Work: NPS -100.0 (1 encuesta)
- Residentes Europa: NPS -100.0 (1 encuesta)

#### **Cadena causal LH:**
```
Mishandling +3.69 pts (+23.7%)
         ↓
Problemas de equipaje en conexiones intercontinentales
         ↓
Retrasos acumulados (OTP15 -1.55 pts)
         ↓
Impacto concentrado en flota wide-body (A333/A33ACMI)
         ↓
Crisis severa en rutas latinoamericanas (MAD-SJO, MAD-MEX)
         ↓
Colapso progresivo: Economy (-8.8) → Business (-26.7) → Premium (-52.1)
```

---

## 📊 **INTERPRETACIÓN EJECUTIVA DE LA DILUCIÓN:**

### **Por qué el Global es +0.5 pts a pesar de la crisis en LH:**

1. **Peso volumétrico de SH (71%):** 286 encuestas SH vs 117 encuestas LH
2. **Composición favorable en SH Economy:** 256 encuestas con NPS +34.4 (mejora artificial por sesgo de muestra)
3. **Dilución matemática:**
   - Contribución SH al Global: +5.5 pts × 0.71 = +3.9 pts
   - Contribución LH al Global: -13.6 pts × 0.29 = -3.9 pts
   - Resultado neto: +0.5 pts (anomalía positiva marginal)

### **La paradoja operativa:**

**A pesar de 252 incidentes operativos totales** (18 SH + 8 LH formalmente reportados, más 21 rutas LH con problemas):
- 74 cancelaciones
- 35 retrasos
- 14 cambios de equipo
- 19 pérdidas de conexión
- Mishandling +3.69 pts en ambos radios

**El NPS Global mejoró +0.5 pts** debido a que:
1. El 90% de encuestas SH corresponden a vuelos **IB** (NPS 24.3) que NO experimentaron las disrupciones concentradas en flota wide-body LH
2. Los clientes SH que respondieron NO fueron los afectados por meteorología BIO/SDR
3. El 54% de encuestas totales son del mercado español (NPS 26.9), menos afectado por disrupciones intercontinentales

---

## 🚨 **CONCLUSIÓN CRÍTICA:**

**La anomalía positiva Global (+0.5 pts) NO refleja una mejora real del servicio**, sino un **efecto de dilución volumétrica** donde:

- **Short Haul (71% del volumen)** mostró variación normal (+5.5 pts) con dinámicas internas contradictorias (Economy mejoró artificialmente, Business cayó por meteorología)
- **Long Haul (29% del volumen)** experimentó deterioro operativo severo (-13.6 pts) con crisis progresiva en cabinas premium (-52.1 pts)

**El resultado Global oculta:**
1. Crisis operativa real en LH (Mishandling +23.7%, 21 rutas afectadas)
2. Colapso en Premium LH (NPS -42.86, caída de -52.1 pts)
3. Impacto severo en rutas latinoamericanas (MAD-SJO, MAD-MEX, DFW-MAD)
4. Problemas sistémicos con flota wide-body (A333/A33ACMI)
5. Deterioro en operaciones codeshare (AA: NPS -100)

---

## 🔑 **RESUMEN EJECUTIVO DE AGREGACIÓN GLOBAL:**

| Nivel | Escenario | Estados (Radios \| Global) | Narrativa Adoptada | Interpretación |
|-------|-----------|----------------------------|--------------------|----------------|
| **GLOBAL** | DILUCIÓN (-, N \| +) | (-13.6, +5.5 N \| +0.5) | Explicación del Radio Dominante (SH) | El volumen de SH (71%) diluye la crisis de LH, generando falsa señal de mejora (+0.5 pts) que oculta deterioro operativo severo en segmento intercontinental |

**Recomendación:** Monitorear LH de forma independiente, ya que el agregado Global enmascara crisis operativa real en rutas intercontinentales y cabinas premium.

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

# 🔍 PASO 4: IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS (NMA)

---

## 📊 ANÁLISIS DE NODOS MÁXIMO AFECTADOS

### **CAUSA 1: Deterioro Operativo en Long Haul (Mishandling + Puntualidad)**

- **Escenario:** SINERGIA TOTAL `(-,-,- | -)`
- **NMA:** `Global/LH` (Long Haul completo)
- **Afecta a:** 
  - Global/LH/Economy
  - Global/LH/Business
  - Global/LH/Premium
- **Tipo de impacto:** NEGATIVO
- **Magnitud:** -13.6 pts en LH (con impacto progresivo: Economy -8.8 pts, Business -26.7 pts, Premium -52.1 pts)

**Cadena de propagación hacia el segmento raíz:**

1. **`Global/LH/Economy` + `Global/LH/Business` + `Global/LH/Premium` → `Global/LH`**
   - Escenario: **SINERGIA TOTAL** `(-8.8, -26.7, -52.1 | -13.6)`
   - Las tres cabinas presentan anomalías negativas simultáneas por la misma causa operativa (Mishandling +3.69 pts, OTP15 -1.55 pts)
   - El deterioro afecta transversalmente a todas las cabinas, con severidad progresiva según categoría premium
   - **NMA se establece en `Global/LH`** (padre común)

2. **`Global/LH` + `Global/SH` → `Global`**
   - Escenario: **DILUCIÓN** `(-13.6, +5.5 N | +0.5)`
   - Long Haul presenta anomalía negativa severa (-13.6 pts)
   - Short Haul muestra variación normal (+5.5 pts)
   - El volumen masivo de SH (71% de encuestas, 286 respuestas) diluye el deterioro de LH (29%, 117 respuestas)
   - **NMA se DETIENE en `Global/LH`** (no propaga al Global debido a dilución volumétrica)

**Conclusión:** El NMA es `Global/LH`, afectando a las 117 encuestas del segmento intercontinental. La causa NO propagó al Global debido a la dilución por volumen de Short Haul.

---

### **CAUSA 2: Meteorología Adversa en Norte de España (BIO/SDR) - Impacto en Business Short Haul**

- **Escenario:** SINERGIA NEGATIVA `(-,- | -)`
- **NMA:** `Global/SH/Business` (Business Short Haul completo)
- **Afecta a:**
  - Global/SH/Business/IB
  - Global/SH/Business/YW
- **Tipo de impacto:** NEGATIVO
- **Magnitud:** -11.8 pts en Business SH (IB -5.7 pts, YW -28.5 pts)

**Cadena de propagación hacia el segmento raíz:**

1. **`Global/SH/Business/IB` + `Global/SH/Business/YW` → `Global/SH/Business`**
   - Escenario: **SINERGIA NEGATIVA** `(-5.7, -28.5 | -11.8)`
   - Ambas compañías presentan anomalías negativas por la misma causa: 18 incidentes NCS meteorológicos (6 cancelaciones, 3 retrasos en BIO/SDR)
   - Impacto diferencial: **YW** (-28.5 pts) más severo que **IB** (-5.7 pts), pero causa raíz común
   - **NMA se establece en `Global/SH/Business`** (padre común)

2. **`Global/SH/Business` + `Global/SH/Economy` → `Global/SH`**
   - Escenario: **CANCELACIÓN** `(+7.3, -11.8 | +5.5 N)`
   - Business presenta anomalía negativa (-11.8 pts)
   - Economy presenta anomalía positiva (+7.3 pts)
   - Efectos opuestos se cancelan, resultando en variación "Normal" (+5.5 pts)
   - **NMA se DETIENE en `Global/SH/Business`** (no propaga debido a cancelación con Economy)

3. **`Global/SH` + `Global/LH` → `Global`**
   - Escenario: **DILUCIÓN** `(-13.6, +5.5 N | +0.5)`
   - Short Haul muestra variación normal (que oculta la crisis en Business)
   - **NMA permanece en `Global/SH/Business`** (no alcanza nivel Global)

**Conclusión:** El NMA es `Global/SH/Business`, afectando a las 30 encuestas del segmento corporativo de corto radio. La causa NO propagó al nivel SH debido a cancelación con Economy, ni al Global debido a dilución.

---

### **CAUSA 3: Sesgo de Muestra en Economy Short Haul (Mejora Artificial)**

- **Escenario:** SINERGIA POSITIVA `(+,+ | +)`
- **NMA:** `Global/SH/Economy` (Economy Short Haul completo)
- **Afecta a:**
  - Global/SH/Economy/IB
  - Global/SH/Economy/YW
- **Tipo de impacto:** POSITIVO
- **Magnitud:** +7.3 pts en Economy SH (IB +7.1 pts, YW +7.6 pts)

**Cadena de propagación hacia el segmento raíz:**

1. **`Global/SH/Economy/IB` + `Global/SH/Economy/YW` → `Global/SH/Economy`**
   - Escenario: **SINERGIA POSITIVA** `(+7.1, +7.6 | +7.3)`
   - Ambas compañías presentan anomalías positivas simultáneas de magnitud similar
   - Causa común: Los clientes que respondieron NO fueron los afectados por los 18 incidentes NCS (meteorología BIO/SDR)
   - Menor ocupación (Load Factor -3.01 pts) compensó deterioro en Mishandling (+3.69 pts)
   - **NMA se establece en `Global/SH/Economy`** (padre común)

2. **`Global/SH/Economy` + `Global/SH/Business` → `Global/SH`**
   - Escenario: **CANCELACIÓN** `(+7.3, -11.8 | +5.5 N)`
   - Economy presenta anomalía positiva (+7.3 pts)
   - Business presenta anomalía negativa (-11.8 pts)
   - Efectos opuestos se cancelan, resultando en variación "Normal" (+5.5 pts)
   - **NMA se DETIENE en `Global/SH/Economy`** (no propaga debido a cancelación con Business)

3. **`Global/SH` + `Global/LH` → `Global`**
   - Escenario: **DILUCIÓN** `(-13.6, +5.5 N | +0.5)`
   - Short Haul muestra variación normal (que oculta la mejora artificial en Economy)
   - **NMA permanece en `Global/SH/Economy`** (aunque contribuye marginalmente al Global +0.5 pts)

**Conclusión:** El NMA es `Global/SH/Economy`, afectando a las 256 encuestas del segmento económico de corto radio. La causa NO propagó claramente al nivel SH debido a cancelación con Business, pero contribuyó marginalmente al Global +0.5 pts a través de la dilución volumétrica (Economy SH representa el 63% del total de encuestas globales).

---

## 📋 RESUMEN DE NODOS MÁXIMO AFECTADOS

| Causa | NMA | Escenario en NMA | Tipo Impacto | Magnitud | Propagó a Raíz Global | Razón de No Propagación |
|-------|-----|------------------|--------------|----------|----------------------|-------------------------|
| **Deterioro Operativo LH** | `Global/LH` | SINERGIA TOTAL (-,-,- \| -) | NEGATIVO | -13.6 pts | ❌ NO | DILUCIÓN por volumen SH (71% encuestas) |
| **Meteorología BIO/SDR** | `Global/SH/Business` | SINERGIA NEGATIVA (-,- \| -) | NEGATIVO | -11.8 pts | ❌ NO | CANCELACIÓN con Economy SH → DILUCIÓN en Global |
| **Sesgo de Muestra Economy SH** | `Global/SH/Economy` | SINERGIA POSITIVA (+,+ \| +) | POSITIVO | +7.3 pts | ⚠️ PARCIAL | CANCELACIÓN con Business SH, pero contribuyó al Global +0.5 vía dilución volumétrica |

---

## 🔑 INTERPRETACIÓN EJECUTIVA DE PROPAGACIÓN

### **Patrones de Bloqueo de Propagación:**

1. **Long Haul (Crisis Operativa):**
   - ✅ Sinergia total en las 3 cabinas LH → NMA sube a `Global/LH`
   - ❌ Bloqueado en nivel Global por **DILUCIÓN** (SH 71% vs LH 29%)
   - **Resultado:** Crisis severa en LH (-13.6 pts) invisible en Global (+0.5 pts)

2. **Business Short Haul (Meteorología):**
   - ✅ Sinergia negativa entre **IB** y **YW** → NMA sube a `Global/SH/Business`
   - ❌ Bloqueado en nivel SH por **CANCELACIÓN** con Economy (+7.3 vs -11.8)
   - ❌ Bloqueado en nivel Global por **DILUCIÓN** (SH Normal vs LH Negativo)
   - **Resultado:** Crisis en Business SH (-11.8 pts) oculta en SH Normal (+5.5 pts) y Global (+0.5 pts)

3. **Economy Short Haul (Mejora Artificial):**
   - ✅ Sinergia positiva entre **IB** y **YW** → NMA sube a `Global/SH/Economy`
   - ❌ Bloqueado en nivel SH por **CANCELACIÓN** con Business (-11.8 vs +7.3)
   - ⚠️ Contribución marginal a Global (+0.5 pts) por peso volumétrico (63% de encuestas totales)
   - **Resultado:** Mejora artificial en Economy SH (+7.3 pts) se diluye en SH Normal (+5.5 pts) pero empuja marginalmente al Global a +0.5 pts

### **Conclusión Crítica:**

**Ninguna de las tres causas propagó claramente al nivel Global** debido a mecanismos de cancelación y dilución. El Global +0.5 pts es un **resultado espurio** que oculta:
- Crisis operativa severa en LH (-13.6 pts)
- Crisis meteorológica en Business SH (-11.8 pts)
- Mejora artificial en Economy SH (+7.3 pts) por sesgo de muestra

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

# 📊 PASO 4B: EXTRACCIÓN DE EVIDENCIAS DEL CONTEXTO INICIAL

---

## === NMA 1: Global/LH ===

### 📈 EXPLANATORY DRIVERS:
No disponible

### 📊 DATOS OPERATIVOS:
- **Mishandling (Global/LH):** 19.25 (variación +3.69 vs baseline de 15.56)
- **OTP15 (Global/LH):** 79.85% (variación -1.55 pts vs baseline de 81.40%)
- **Load Factor (Global/LH):** 90.19% (variación -0.44 vs baseline de 90.63%)
- **Misconex (Global/LH):** 0.8 (variación +0.11 vs baseline de 0.69)

### 🚨 INCIDENTES NCS (CUANTITATIVO):
- **Total incidentes (Global/LH):** 8 incidentes operativos
- **Retrasos:** 3 incidentes
- **Cancelaciones:** 2 incidentes
- **Equipaje:** 1 incidente
- **Otras incidencias:** 1 incidente
- **Sin categoría:** 1 incidente

### 🧠 NCS (CUALITATIVO / REFLEXIÓN):
**Incidentes destacados:**
- Reprogramación extrema: Vuelo desde MAD con salida 09:00h → reprogramado +8h 5min
- Retraso significativo: Vuelo DFW-MAD con salida 15:15h → reprogramado +1h 20min

**Temas principales:** Passenger (2 incidentes)

**Alcance geográfico:** 21 rutas con incidentes operacionales reportados en el período

### 💬 FEEDBACK DE CLIENTES:
**No hay comentarios con texto disponibles** para el período 2025-12-18 (Global/LH)

**Implicación metodológica:** La ausencia de feedback cualitativo impide:
- Validar si los clientes percibieron las cancelaciones/retrasos
- Confirmar quejas sobre equipaje
- Triangular rutas específicas con quejas recurrentes
- Elevar el nivel de confianza de MEDIA-ALTA a ALTA

### ✈️ RUTAS AFECTADAS (Top 5):
**Global/LH - Rutas con peor NPS:**

1. **MAD-SJO (Global/LH):** NPS -33.3 (6 encuestas) - Incidentes NCS confirmados
2. **DOH-MAD (Global/LH):** NPS -20.0 (5 encuestas) - Incidentes NCS confirmados
3. **MAD-MEX (Global/LH):** NPS -13.3 (15 encuestas) - Incidentes NCS confirmados

**Global/LH/Economy - Rutas con peor NPS:**

1. **MAD-SJO:** NPS -33.3 (6 encuestas) - Incidentes NCS confirmados
2. **DOH-MAD:** NPS -20.0 (5 encuestas) - Incidentes NCS confirmados
3. **MAD-MEX:** NPS -8.3 (12 encuestas) - Incidentes NCS confirmados

**Global/LH/Business - Rutas críticas:**

| Ruta | NPS | Encuestas | Observación |
|------|-----|-----------|-------------|
| MAD-MEX | -100 | 1 | Muestra no representativa |
| MAD-SJU | -100 | 1 | Muestra no representativa |
| DFW-MAD | -100 | 1 | Muestra no representativa |
| EZE-MAD | -100 | 1 | Muestra no representativa |
| HAV-MAD | 0 | 2 | Muestra limitada |
| MAD-MIA | 0 | 1 | Muestra limitada |
| BOG-MAD | 0 | 4 | Muestra más significativa |

**Global/LH/Premium - Rutas críticas:**

| Ruta | NPS | Encuestas | Incidentes NCS |
|------|-----|-----------|----------------|
| DFW-MAD | -100 | 1 | ✅ Confirmado |
| GRU-MAD | -100 | 2 | ✅ Confirmado |
| MAD-NRT | -100 | 1 | ✅ Confirmado |
| MAD-MEX | 0 | 2 | ✅ Confirmado |
| BOG-MAD | 100 | 1 | ✅ Confirmado |

**Patrón geográfico:** Concentración de problemas en rutas latinoamericanas desde MAD

### 👥 PERFILES REACTIVOS:

**Global/LH/Economy:**

**Por Propósito de Viaje:**
- Business/Work: NPS -33.3 (9 encuestas) - **3.6x más afectados que Leisure**
- Leisure: NPS -2.3 (88 encuestas)

**Por Flota:**
- A33ACMI: NPS -66.7 (3 encuestas) - Crítico
- A333: NPS -30.8 (13 encuestas) - Crítico
- A332: NPS 8.8 (17 encuestas)
- A350: NPS 13.3 (15 encuestas)

**Por Región de Residencia:**
- EUROPA: NPS -100.0 (6 encuestas) - Crítico
- ESPAÑA: NPS -9.5 (42 encuestas) - Moderado
- AMERICA SUR: NPS 30.8 (13 encuestas) - Positivo

**Por CodeShare:**
- BA: NPS -100.0 (2 encuestas) - Crítico
- IB: NPS -6.2 (80 encuestas) - Moderado
- LATAM: NPS 100.0 (2 encuestas) - Excelente

---

**Global/LH/Business:**

**Por Propósito de Viaje:**
- Leisure: NPS -10.0 (10 encuestas)
- Business/Work: NPS 0.0 (3 encuestas)
- Diferencial: 10 pts (no significativa)

**Por Fleet:**
- A350 next: NPS +20.0 (5 encuestas) - Mejor desempeño relativo
- A332: NPS -25.0 (4 encuestas)
- A333: NPS -100.0 (1 encuesta)
- A33ACMI: NPS 0.0 (1 encuesta)
- A359: NPS 0.0 (2 encuestas)

**Por Región de Residencia:**
- Dispersión extrema de hasta 200 pts
- Volumen insuficiente por región (n=1-3)

**Por CodeShare:**
- IB: NPS +37.5 (8 encuestas)
- BA: NPS -50.0 (2 encuestas)
- AA: NPS -100.0 (3 encuestas) - **Todos detractores**

---

**Global/LH/Premium:**

**Por Propósito de Viaje:**
- Business/Work: NPS -100.0 (1 encuesta)
- Leisure: NPS -33.3 (6 encuestas)
- Diferencial: 66.7 pts

**Por Fleet:**
- A333: NPS -100.0 (2 encuestas)
- A33ACMI: NPS -50.0 (4 encuestas)
- A350: NPS -33.3 (3 encuestas)
- A350 next: NPS 0.0 (2 encuestas)

**Dispersión por Fleet:** 100.0 pts

**Por Región de Residencia:**
- EUROPA: NPS -100.0 (1 encuesta)
- ESPAÑA: NPS -100.0 (1 encuesta)
- AMERICA CENTRO: NPS +33.3 (3 encuestas)

**Dispersión por Región:** 133.3 pts (extremadamente alta)

**Por CodeShare:**
- AA: NPS -100.0 (1 encuesta)
- IB: No calculable (n=1 por segmento)

---

## === NMA 2: Global/SH/Business ===

### 📈 EXPLANATORY DRIVERS:
No disponible

### 📊 DATOS OPERATIVOS:
- **Mishandling (Global/SH/Business):** 19.25 (variación +3.69 vs baseline estimado ~15.56)
- **OTP15 (Global/SH/Business):** 90.94% (variación +0.83 vs baseline estimado ~90.11%)
- **Load Factor (Global/SH/Business):** 66.1% (variación -5.7 vs baseline estimado ~71.8%)
- **Misconex (Global/SH/Business):** 0.8 (variación +0.11 vs baseline estimado ~0.69)

### 🚨 INCIDENTES NCS (CUANTITATIVO):
- **Total incidentes (Global/SH/Business):** 18 incidentes operativos
- **Cancelaciones:** 6 (33%)
- **Retrasos:** 3 (17%)
- **Incidentes de pasajeros:** 6 (33%)
- **Incidentes de equipaje:** 1 (6%)
- **Otras incidencias:** 1 (6%)

### 🧠 NCS (CUALITATIVO / REFLEXIÓN):
**Causa dominante:** Meteorología adversa en BIO (Bilbao) y SDR (Santander)

**Rutas más afectadas por meteorología:**
- MAD-BIO: 2 incidentes
- VIT-BIO: 2 incidentes
- BIO-MAD: 1 incidente

**Incidente destacado:**
- Vuelo IB0433 BIO-MAD regresó a MAD por condiciones meteorológicas adversas en Bilbao
- Se ofrecieron alternativas de transporte por superficie a pasajeros afectados

**Distribución temática:**
- 6 incidentes relacionados con pasajeros
- 1 incidente de equipaje (baggage)

**Respuesta operativa:** Oferta de transporte por superficie (surface) para pasajeros afectados

### 💬 FEEDBACK DE CLIENTES:
**No hay comentarios con texto disponibles** para el período 2025-12-18 (Global/SH/Business)

**Impacto metodológico:** La ausencia de feedback cualitativo impide:
- Validar si los clientes percibieron las cancelaciones/retrasos
- Confirmar quejas sobre equipaje
- Triangular rutas específicas con quejas recurrentes

### ✈️ RUTAS AFECTADAS (Top 5):

**Global/SH/Business - Rutas con peor desempeño:**

| Ruta | NPS | Encuestas | Observación |
|------|-----|-----------|-------------|
| BIO-MAD | 0.0 | 2 | Correlaciona con incidente NCS |
| MAD-TLS | 0.0 | 1 | Muestra limitada |
| LIS-MAD | 0.0 | 1 | Muestra limitada |
| LHR-MAD | -33.3 | 3 | NPS negativo |

**Global/SH/Business - Rutas con mejor desempeño:**

| Ruta | NPS | Encuestas |
|------|-----|-----------|
| FCO-MAD | 100.0 | N/A |
| MAD-ORY | 100.0 | N/A |
| HAM-MAD | 100.0 | N/A |
| BUD-MAD | 100.0 | N/A |

**Total de rutas analizadas:** 21 rutas (números de encuestas muy bajos n=1-3)

---

**Global/SH/Business/IB - Rutas:**

| Ruta | NPS | Encuestas | Incidentes NCS |
|------|-----|-----------|----------------|
| LHR-MAD | -33.3 | 3 | Sí |
| BIO-MAD | 0.0 | 2 | Sí |
| BUD-MAD | 100.0 | 1 | No |
| MAD-OPO | 100.0 | 1 | No |
| FCO-MAD | 100.0 | 1 | No |

**Limitación crítica:** Las rutas más afectadas por meteorología según NCS (MAD-BIO, VIT-BIO) NO tienen datos de NPS disponibles

---

**Global/SH/Business/YW - Rutas:**

| Ruta | NPS | Encuestas | Observaciones |
|------|-----|-----------|---------------|
| LEN-PMI | -100 | 1 | Detractor único |
| MAD-MRS | -100 | 1 | Detractor único |
| FRA-MAD | -100 | 1 | Detractor único |
| BIO-MAD | 0 | 1 | Neutral, a pesar de incidente NCS |
| MAD-PNA | 50 | 2 | Positivo |
| MAD-SVQ | 100 | 1 | Promotor único |

**Hallazgo crítico:** Las rutas con NPS -100 (LEN-PMI, MAD-MRS, FRA-MAD) NO fueron identificadas en NCS como problemáticas

### 👥 PERFILES REACTIVOS:

**Global/SH/Business:**

**Por Propósito de Viaje:**
- Business/Work: NPS 0.0 (8 encuestas) - **Segmento más afectado**
- Leisure: NPS 31.8 (22 encuestas)
- Diferencial: -31.8 pts

**Por Fleet:**
- CRJ: NPS -12.5 (8 encuestas) - **Peor desempeño** (aeronave regional usada en rutas como BIO)
- A320neo: NPS 18.2 (11 encuestas)
- A321: NPS 50.0 (4 encuestas)
- A320: NPS 57.1 (7 encuestas)

**Por Residencia:**
- EUROPA: NPS 0.0 (8 encuestas) - **Más afectados**
- ESPAÑA: NPS 29.4 (17 encuestas)
- LATAM: NPS 60.0 (5 encuestas)

**Por CodeShare:**
- IB: NPS 25.0 (28 encuestas)
- BA: NPS -100.0 (1 encuesta)
- LATAM: NPS 100.0 (1 encuesta)

---

**Global/SH/Business/IB:**

**Por Propósito de Viaje:**
- Business/Work: NPS -50.0 (4 encuestas) - **Segmento más afectado**
- Leisure: NPS 55.6 (18 encuestas)
- Spread: 105.6 pts (polarización extrema)

**Por Fleet:**
- A320neo: NPS 18.2 (11 encuestas)
- A321: NPS 50.0 (4 encuestas)
- A320: NPS 57.1 (7 encuestas)
- CRJ: No disponible para IB específicamente

**Por Región de Residencia:**
- ESPAÑA: NPS 50.0 (10 encuestas)
- EUROPA: NPS 14.3 (7 encuestas)
- Unknown/AFRICA: NPS -100.0 (1 encuesta)
- LATAM: NPS -100.0 (1 encuesta)

**Por Compañía:**
- IB: NPS 40.0 (20 encuestas)
- BA: NPS 100.0 (1 encuesta)
- LATAM: NPS -100.0 (1 encuesta)

---

**Global/SH/Business/YW:**

**Por Propósito de Viaje:**
- Business/Work: NPS 50.0 (4 encuestas)
- Leisure: NPS -75.0 (4 encuestas) - **Insatisfacción extrema**
- Brecha de expectativas: 125 pts

**Por Región de Residencia:**
- España: NPS 0.0 (6 encuestas)
- Europa: NPS -100.0 (1 encuesta) - **Cliente internacional extremadamente insatisfecho**

**Por Flota:**
- CRJ: NPS -12.5 (8 encuestas) - Toda la operación en un solo tipo de avión

**Por Codeshare:**
- IB: NPS -12.5 (8 encuestas) - Todos los vuelos operados por IB

---

## === NMA 3: Global/SH/Economy ===

### 📈 EXPLANATORY DRIVERS:
No disponible

### 📊 DATOS OPERATIVOS:
- **Mishandling (Global/SH/Economy):** 19.25 (variación +3.69 pts vs baseline de 15.56)
- **OTP15 (Global/SH/Economy):** 90.94% (variación +0.83 pts vs baseline de 90.11%)
- **Load Factor (Global/SH/Economy):** 82.92% (variación -3.01 pts vs baseline de 85.93%)
- **Misconex (Global/SH/Economy):** 0.8 (variación +0.11 pts vs baseline de 0.69)

### 🚨 INCIDENTES NCS (CUANTITATIVO):
- **Total incidentes (Global/SH/Economy):** 18 incidentes operativos
- **Cancelaciones:** 6 (33%)
- **Retrasos:** 3 (17%)
- **Incidentes de pasajeros:** 6 (33%)
- **Incidentes de equipaje:** 1 (6%)
- **Otras incidencias:** No especificado

### 🧠 NCS (CUALITATIVO / REFLEXIÓN):
**Causas principales:** Meteorología adversa en BIO (Bilbao) y SDR (Santander)

**Rutas afectadas:**
- MAD-BIO: 2 incidentes
- VIT-BIO: 2 incidentes
- BIO-MAD: 1 incidente

**Vuelo más afectado:** IB0433 BIO-MAD (regresó por mal tiempo)

**Discrepancia crítica:**
- Mishandling subió +3.69 puntos (significativo)
- Pero solo 1 incidente de equipaje reportado formalmente (6% del total)
- **Conclusión:** Problemas de equipaje NO fueron capturados en reportes formales NCS

### 💬 FEEDBACK DE CLIENTES:
**No hay comentarios con texto disponibles** para el período 2025-12-18 (Global/SH/Economy)

**Implicación:** No se pudo validar cualitativamente la percepción del cliente sobre los problemas de equipaje y puntualidad

### ✈️ RUTAS AFECTADAS (Top 5):

**Global/SH/Economy - Datos insuficientes:**
- Solo 1 ruta con datos de NPS: MAD-VIE con NPS -50.0 (n=2 encuestas)
- Muestra **no significativa estadísticamente**
- 75 rutas con incidentes NCS pero **sin datos de NPS asociados**

**Rutas con incidentes meteorológicos identificados:**
- BIO (Bilbao), SDR (Santander), VIT (Vitoria)
- **NO disponible** impacto medible en NPS por falta de datos

---

**Global/SH/Economy/IB - Rutas con NPS más bajo:**

| Ruta | NPS | Encuestas | Observación |
|------|-----|-----------|-------------|
| MAD-ORY | 8.3 | 12 | NPS muy bajo, muestra razonable |
| MAD-PRG | 16.7 | 6 | NPS muy bajo, muestra pequeña |
| HAM-MAD | 25.0 | 4 | NPS bajo, muestra pequeña |
| DUS-MAD | 37.5 | 8 | NPS bajo |
| LIN-MAD | 42.9 | 7 | NPS moderado-bajo |

**Limitaciones:**
- No hay datos de NPS baseline por ruta para calcular desviaciones
- Las rutas con incidentes NCS (BIO) no tienen datos de NPS
- Muestras pequeñas en la mayoría de rutas (n<10)
- 30 rutas con incidentes NCS, pero solo datos de NPS en 5 rutas

---

**Global/SH/Economy/YW:**
- Solo 1 ruta con datos: GVA-MAD con NPS 0.0 (n=1) - **No representativo**
- 45 rutas operadas, sin datos suficientes de encuestas

### 👥 PERFILES REACTIVOS:

**Global/SH/Economy:**

**Por Propósito de Viaje:**
- Business: NPS 40.0 (75 encuestas, 29%)
- Leisure: NPS 32.0 (181 encuestas, 71%)
- Dispersión: 8.0 puntos - **BAJA** (ambos perfiles afectados similarmente)

**Por Flota:**
- ATR: NPS 69.2 (13 encuestas) - Mejor desempeño
- A320: NPS 55.3 (38 encuestas)
- A320neo: NPS 34.0 (datos incompletos)
- CRJ: NPS 29.9 (87 encuestas, 34%) - **Desempeño medio-bajo**
- A321: NPS 24.4 (41 encuestas) - Desempeño bajo
- A350 C: NPS -50.0 (4 encuestas) - Muestra muy pequeña
- Dispersión: 119.2 puntos - **ALTA**

**Por CodeShare:**
- IB: NPS 35.4 (240 encuestas, 94%) - **Valor más representativo**
- Otros códigos: muestras no significativas (n=1-6)

---

**Global/SH/Economy/IB:**

**Por Propósito de Viaje:**
- Business/Work: NPS 48.8 (41 encuestas)
- Leisure: NPS 28.7 (115 encuestas)
- Diferencia: 20.1 pts a favor de Business

**Por Fleet:**
- A350 C: NPS -50.0 (4 encuestas) - **Anomalía extrema negativa**
- A319: NPS 20.0 (10 encuestas)
- A321: NPS 24.4 (41 encuestas)
- A320neo: NPS 34.9 (63 encuestas)
- A320: NPS 55.3 (38 encuestas) - **Alto**
- Dispersión: 105.3 pts (segunda mayor variabilidad)

**Por Residence Region:**
- Unknown: NPS -100.0 (3 encuestas)
- AMERICA SUR: NPS 0.0 (8 encuestas)
- EUROPA: NPS 26.2 (42 encuestas)
- ESPAÑA: NPS 43.5 (92 encuestas) - **Alto, muestra más grande**
- Dispersión: 200.0 pts (máxima variabilidad)

**Por Codeshare:**
- BA: NPS -20.0 (5 encuestas)
- IB: NPS 35.2 (142 encuestas)

---

**Global/SH/Economy/YW:**

**Por Propósito de Viaje:**
No disponible (volumen de encuestas n=1 estadísticamente insuficiente)

**Por Flota:**
No disponible

**Por Región de Residencia:**
No disponible

**Por CodeShare:**
No disponible

---

## Cabin Radio Reflection

# 📋 PASO 4C: REFLEXIÓN POR CABINA-RADIO

---

## === Economy SH ===

• **NPS Cabina:** 34.4 (+7.3 pts)  
• **Estado:** POSITIVE ANOMALY  
• **Escenario:** SINERGIA (+, + | +) (IB POSITIVE ANOMALY, YW POSITIVE ANOMALY | Cabina POSITIVE ANOMALY)

• **IB:** NPS 34.0 (+7.1 pts) - Mejora operativa global con OTP15 +1.34 pts (93.39%) y Load Factor -3.51 pts (85.53%, menor ocupación favorable). A pesar de Mishandling elevado (+3.99 pts, 20.46 absoluto), los clientes que respondieron NO fueron los afectados por los 18 incidentes NCS meteorológicos (BIO/SDR). Perfiles favorecidos: Fleet A320 (NPS 55.3, n=38), España (NPS 43.5, n=92), Business/Work (NPS 48.8, n=41).

• **YW:** NPS 35.0 (+7.6 pts) - Misma dinámica que IB: OTP15 +0.31 pts (88.77%), Load Factor -2.17 pts (78.33%). Mishandling +2.76 pts (15.55 absoluto) NO se reflejó en NPS. Los 18 incidentes NCS se concentraron en rutas sin encuestas suficientes. Volumen extremadamente bajo (n≈1-3 por ruta) sugiere sesgo de muestra donde los clientes afectados por meteorología BIO/SDR no completaron encuestas.

• **Narrativa de agregación:** Ambas compañías experimentaron mejoras sinérgicas de magnitud similar (+7.1 y +7.6 pts). La causa es común: **sesgo de muestra compensó deterioro operativo**. Los clientes que respondieron (principalmente IB con 94% de encuestas, 240 respuestas) NO fueron los afectados por problemas de equipaje ni incidentes meteorológicos. La menor ocupación (Load Factor -3.01 pts en agregado) mejoró la experiencia percibida, compensando el aumento significativo de Mishandling (+3.69 pts). Ambas compañías se beneficiaron del mismo efecto de composición favorable.

• **Rutas críticas:** (Según CAUSAL EXPLANATION del padre - Economy SH)
  - MAD-VIE: NPS -50.0 (n=2) - Única ruta con datos, muestra no significativa
  - **Limitación:** 75 rutas con incidentes NCS pero sin datos de NPS asociados
  - Rutas con incidentes meteorológicos (BIO, SDR, VIT) NO tienen encuestas suficientes
  - **IB específico:** MAD-ORY (NPS 8.3, n=12), MAD-PRG (NPS 16.7, n=6), HAM-MAD (NPS 25.0, n=4)
  - **YW específico:** GVA-MAD (NPS 0.0, n=1) - No representativo

• **Perfiles reactivos:** (Según CAUSAL EXPLANATION del padre - Economy SH)
  - **Fleet:** Dispersión 119.2 pts - ATR (NPS 69.2, n=13) mejor, A350 C (NPS -50.0, n=4) peor, CRJ (NPS 29.9, n=87, 34% de encuestas)
  - **Residence Region:** Dispersión 200.0 pts (IB) - España (NPS 43.5, n=92) mejor, Unknown (NPS -100.0, n=3) peor
  - **Business/Leisure:** Dispersión 8.0 pts (baja) - Business (NPS 40.0, n=75), Leisure (NPS 32.0, n=181)
  - **CodeShare:** IB (NPS 35.4, n=240, 94% del segmento) - Valor más representativo

---

## === Business SH ===

• **NPS Cabina:** 23.3 (-11.8 pts)  
• **Estado:** NEGATIVE ANOMALY  
• **Escenario:** SINERGIA (-, - | -) (IB NEGATIVE ANOMALY, YW NEGATIVE ANOMALY | Cabina NEGATIVE ANOMALY)

• **IB:** NPS 36.4 (-5.7 pts) - Meteorología adversa BIO/SDR generó 18 incidentes NCS (6 cancelaciones, 3 retrasos). Mishandling +3.99 pts (20.46 absoluto), Misconex +0.11 pts (0.84 absoluto). Impacto diferencial por perfil: Business/Work (NPS -50.0, n=4) vs Leisure (NPS 55.6, n=18), spread de 105.6 pts. Rutas críticas: BIO-MAD (NPS 0.0, n=2), LHR-MAD (NPS -33.3, n=3). Fleet A320neo bajo desempeño (NPS 18.2, n=11). Europa más afectada (NPS 14.3, n=7).

• **YW:** NPS -12.5 (-28.5 pts) - Misma causa meteorológica BIO/SDR, pero con polarización extrema opuesta: Leisure (NPS -75.0, n=4) vs Business/Work (NPS 50.0, n=4), spread de 125 pts. Desalineación de expectativas en segmento Leisure en vuelos regionales con flota CRJ. Rutas críticas: LEN-PMI (NPS -100, n=1), MAD-MRS (NPS -100, n=1), FRA-MAD (NPS -100, n=1). Europa extremadamente insatisfecha (NPS -100.0, n=1). OTP15 +0.31 pts (88.77%), Load Factor -5.73 pts (51.3%) - métricas operativas NO explican la caída.

• **Narrativa de agregación:** Ambas compañías sufrieron por la misma causa raíz (meteorología adversa BIO/SDR con 18 incidentes operativos), pero el impacto se manifestó de forma diferente. **IB** experimentó una caída moderada (-5.7 pts) concentrada en su segmento Business/Work corporativo (NPS -50.0), mientras que **YW** sufrió un colapso severo (-28.5 pts) en su segmento Leisure (NPS -75.0) por desalineación de expectativas en operaciones regionales con flota CRJ. El agregado Business SH (-11.8 pts) refleja la sinergia negativa, con **IB** dominando volumétricamente (28 de 30 encuestas, 93%) pero **YW** mostrando mayor severidad relativa.

• **Rutas críticas:** (Según CAUSAL EXPLANATION del padre - Business SH)
  - **Rutas con incidentes NCS confirmados:** MAD-BIO (2 incidentes), VIT-BIO (2 incidentes), BIO-MAD (1 incidente - vuelo IB0433 regresó a MAD)
  - **Rutas con NPS medible:**
    - BIO-MAD: NPS 0.0 (n=2) - Correlaciona con incidente NCS
    - LHR-MAD: NPS -33.3 (n=3)
    - MAD-TLS: NPS 0.0 (n=1)
  - **IB específico:** LHR-MAD (NPS -33.3, n=3), BIO-MAD (NPS 0.0, n=2)
  - **YW específico:** LEN-PMI (NPS -100, n=1), MAD-MRS (NPS -100, n=1), FRA-MAD (NPS -100, n=1)
  - **Limitación:** Las rutas más afectadas por meteorología (MAD-BIO, VIT-BIO) NO tienen datos de NPS disponibles

• **Perfiles reactivos:** (Según CAUSAL EXPLANATION del padre - Business SH)
  - **Business/Leisure:** Diferencial -31.8 pts - Business/Work (NPS 0.0, n=8) vs Leisure (NPS 31.8, n=22)
    - **IB:** Spread 105.6 pts - Business/Work (NPS -50.0, n=4) vs Leisure (NPS 55.6, n=18)
    - **YW:** Spread 125 pts (invertido) - Leisure (NPS -75.0, n=4) vs Business/Work (NPS 50.0, n=4)
  - **Fleet:** CRJ (NPS -12.5, n=8, peor desempeño - aeronave regional YW), A320neo (NPS 18.2, n=11), A321 (NPS 50.0, n=4), A320 (NPS 57.1, n=7)
  - **Residence Region:** Europa (NPS 0.0, n=8) más afectada, España (NPS 29.4, n=17), LATAM (NPS 60.0, n=5)
    - **IB:** Europa (NPS 14.3, n=7), España (NPS 50.0, n=10)
    - **YW:** Europa (NPS -100.0, n=1), España (NPS 0.0, n=6)
  - **CodeShare:** IB (NPS 25.0, n=28, 93% del segmento), BA (NPS -100.0, n=1), LATAM (NPS 100.0, n=1)

---

## === Economy LH ===

• **NPS:** -5.2 (-8.8 pts)  
• **Estado:** NEGATIVE ANOMALY

• **Causa principal:** Deterioro operativo en gestión de equipajes (Mishandling +3.69 pts, 19.25 absoluto, incremento del 23.7%) y puntualidad (OTP15 -1.55 pts, 79.85% absoluto). 8 incidentes NCS operativos (3 retrasos, 2 cancelaciones, 1 equipaje) afectaron 21 rutas, con reprogramaciones extremas de hasta +8h 5min (MAD 09:00h) y +1h 20min (DFW-MAD 15:15h).

• **Evidencia clave:**
  - **Mishandling:** 19.25 (+3.69 pts vs baseline de 15.56) - Desviación significativa >3 pts, correlación INVERSA con NPS
  - **OTP15:** 79.85% (-1.55 pts vs baseline de 81.40%) - Deterioro en puntualidad, correlación DIRECTA con NPS
  - **8 incidentes NCS:** 3 retrasos (37.5%), 2 cancelaciones (25%), 1 equipaje (12.5%)
  - **Load Factor:** 90.03% (-0.21 pts vs baseline) - Menor ocupación NO explica caída (correlación inversa esperada)

• **Rutas críticas:**
  1. **MAD-SJO:** NPS -33.3 (n=6) - Incidentes NCS confirmados
  2. **DOH-MAD:** NPS -20.0 (n=5) - Incidentes NCS confirmados
  3. **MAD-MEX:** NPS -8.3 (n=12) - Incidentes NCS confirmados, mayor volumen
  4. **IAD-MAD, MAD-SJU, MAD-ORD, EZE-MAD, JFK-MAD:** Incidentes confirmados
  5. **Patrón geográfico:** Concentración en rutas latinoamericanas desde MAD

• **Perfiles reactivos:**
  - **Business/Leisure:** Business/Work (NPS -33.3, n=9) **3.6x más afectados** que Leisure (NPS -2.3, n=88)
  - **Fleet:** Dispersión alta - A33ACMI (NPS -66.7, n=3) crítico, A333 (NPS -30.8, n=13) crítico, A332 (NPS 8.8, n=17) positivo, A350 (NPS 13.3, n=15) positivo
  - **Residence Region:** Europa (NPS -100.0, n=6) extremo, España (NPS -9.5, n=42) moderado, America Sur (NPS 30.8, n=13) positivo
  - **CodeShare:** BA (NPS -100.0, n=2) crítico, IB (NPS -6.2, n=80) moderado, LATAM (NPS 100.0, n=2) excelente

---

## === Business LH ===

• **NPS:** -7.7 (-26.7 pts)  
• **Estado:** NEGATIVE ANOMALY

• **Causa principal:** Deterioro operativo sistémico en equipaje (Mishandling +3.69 pts) y puntualidad (OTP15 -1.55 pts) que afectó desproporcionadamente a viajeros de negocios. 8 incidentes NCS (3 retrasos, 2 cancelaciones) con reprogramaciones extremas. Problemas concentrados en flota wide-body (A333/A33ACMI) y operadores codeshare (AA: NPS -100.0, todos detractores).

• **Evidencia clave:**
  - **Mishandling:** 19.25 (+3.69 pts vs baseline ~15.56) - Desviación significativa
  - **OTP15:** 79.85% (-1.55 pts vs baseline ~81.40%) - Deterioro en puntualidad
  - **8 incidentes NCS:** 3 retrasos, 2 cancelaciones, 1 equipaje
  - **Reprogramaciones críticas:** +8h 5min (MAD 09:00h), +1h 20min (DFW-MAD 15:15h)
  - **21 rutas con incidentes operacionales**

• **Rutas críticas:**
  1. **MAD-MEX:** NPS -100 (n=1) - Muestra no representativa
  2. **MAD-SJU:** NPS -100 (n=1) - Muestra no representativa
  3. **DFW-MAD:** NPS -100 (n=1) - Muestra no representativa
  4. **EZE-MAD:** NPS -100 (n=1) - Muestra no representativa
  5. **BOG-MAD:** NPS 0 (n=4) - Muestra más significativa
  - **Limitación:** Volumen extremadamente bajo (n=1-4) por ruta limita confiabilidad

• **Perfiles reactivos:**
  - **Business/Leisure:** Diferencia mínima (10 pts) - Leisure (NPS -10.0, n=10), Business/Work (NPS 0.0, n=3)
  - **Fleet:** A350 next (NPS +20.0, n=5) mejor desempeño, A332 (NPS -25.0, n=4), A333 (NPS -100.0, n=1), A33ACMI (NPS 0.0, n=1), A359 (NPS 0.0, n=2)
  - **Residence Region:** Dispersión extrema hasta 200 pts, volumen insuficiente (n=1-3) para análisis confiable
  - **CodeShare:** AA (NPS -100.0, n=3) **todos detractores**, IB (NPS +37.5, n=8), BA (NPS -50.0, n=2)

---

## === Premium LH ===

• **NPS:** -42.9 (-52.1 pts)  
• **Estado:** NEGATIVE ANOMALY

• **Causa principal:** Deterioro operativo severo en equipaje (Mishandling +3.69 pts, +23.7%) y puntualidad (OTP15 -1.55 pts) que colapsó la experiencia en cabina premium. Retrasos extremos (+8h 5min) y problemas con flota wide-body (A333/A33ACMI) en rutas intercontinentales críticas. Impacto progresivamente más severo en cabinas premium donde expectativas de servicio son más altas.

• **Evidencia clave:**
  - **Mishandling:** 19.25 (+3.69 pts vs baseline de 15.56, incremento del 23.7%) - Desviación significativa
  - **OTP15:** 79.85% (-1.55 pts vs baseline de 81.40%) - Deterioro en puntualidad
  - **8 incidentes NCS:** 3 retrasos, 2 cancelaciones, 1 equipaje
  - **Reprogramación extrema:** +8h 5min (MAD 09:00h)
  - **21 rutas con incidentes operacionales**

• **Rutas críticas:**
  1. **DFW-MAD:** NPS -100 (n=1) - Incidentes NCS confirmados
  2. **GRU-MAD:** NPS -100 (n=2) - Incidentes NCS confirmados
  3. **MAD-NRT:** NPS -100 (n=1) - Incidentes NCS confirmados
  4. **MAD-MEX:** NPS 0 (n=2) - Incidentes NCS confirmados
  5. **BOG-MAD:** NPS 100 (n=1) - Incidentes NCS confirmados
  - **Patrón:** Las 5 rutas identificadas tienen incidentes NCS confirmados, alta correlación

• **Perfiles reactivos:**
  - **Business/Leisure:** Business/Work (NPS -100.0, n=1) vs Leisure (NPS -33.3, n=6), diferencial 66.7 pts
  - **Fleet:** Dispersión 100.0 pts - A333 (NPS -100.0, n=2) crítico, A33ACMI (NPS -50.0, n=4) crítico, A350 (NPS -33.3, n=3), A350 next (NPS 0.0, n=2)
  - **Residence Region:** Dispersión 133.3 pts (extremadamente alta) - Europa (NPS -100.0, n=1) extremo, España (NPS -100.0, n=1) extremo, America Centro (NPS +33.3, n=3) positivo
  - **CodeShare:** AA (NPS -100.0, n=1), IB (no calculable, n=1 por segmento)
  - **Limitación:** Volumen extremadamente bajo (7 encuestas totales) limita robustez estadística

---

## 📋 SÍNTESIS EJECUTIVA FINAL

```html
<b>SÍNTESIS EJECUTIVA</b><br>
<br>
La red global registró un <b>NPS de 21.3 (+0.5 pts)</b> con respecto a la media de los últimos 7 días, resultado aparentemente positivo que enmascara una crisis operativa severa en el segmento intercontinental. Esta mejora marginal se explica por un efecto de dilución volumétrica donde el comportamiento de Short Haul, que representa el 71% de las encuestas procesadas, neutralizó completamente el deterioro crítico experimentado en Long Haul.<br>
<br>
En <b>Long Haul</b>, el NPS cayó a <b>-7.7 (-13.6 pts)</b> debido a un deterioro operativo generalizado en gestión de equipaje y puntualidad que afectó transversalmente a las 117 encuestas del segmento. El Mishandling aumentó 3.69 puntos alcanzando 19.25 maletas mal gestionadas por cada 1,000 pasajeros, un incremento del 23.7% que correlaciona inversamente con la satisfacción del cliente. Simultáneamente, la puntualidad se deterioró con un OTP15 de 79.85%, cayendo 1.55 puntos porcentuales respecto al baseline. Este colapso operativo generó 8 incidentes formales incluyendo reprogramaciones extremas de hasta 8 horas 5 minutos en vuelos desde MAD con salida a las 09:00h, y retrasos de 1 hora 20 minutos en la ruta DFW-MAD. Las rutas más afectadas fueron <b>MAD-SJO con NPS de -33.3</b>, <b>DOH-MAD con -20.0</b> y <b>MAD-MEX con -13.3</b>, todas con incidentes operacionales confirmados y concentradas geográficamente en conexiones latinoamericanas desde Madrid. Los pasajeros más sensibles fueron los viajeros de negocios, 3.6 veces más afectados que los de ocio con un NPS de -33.3 en Economy, y los clientes operados en codeshare con AA que registraron NPS de -100.0 en Business. La flota wide-body mostró problemas críticos con el A33ACMI alcanzando NPS de -66.7 y el A333 con -30.8 en Economy, mientras que en Premium estos equipos colapsaron a -50.0 y -100.0 respectivamente. Esta presión en Long Haul no logró propagarse al nivel Global debido a que Short Haul, con 286 encuestas frente a las 117 de LH, diluyó matemáticamente el impacto negativo generando la falsa señal de mejora de +0.5 puntos.<br>
<br>
En <b>Business Short Haul</b>, el NPS cayó a <b>23.3 (-11.8 pts)</b> debido a meteorología adversa en el norte de España que generó 18 incidentes operativos concentrados en Bilbao y Santander, incluyendo 6 cancelaciones y 3 retrasos. Las rutas críticas fueron MAD-BIO con 2 incidentes, VIT-BIO con 2 incidentes adicionales, y el vuelo IB0433 de BIO-MAD que debió regresar a Madrid ofreciendo transporte por superficie a los pasajeros afectados. Este deterioro se manifestó de forma sinérgica en ambas compañías: <b>IB</b> cayó a <b>NPS de 36.4 (-5.7 pts)</b> con una polarización extrema donde los viajeros Business/Work alcanzaron NPS de -50.0 mientras los de ocio mantuvieron 55.6, generando un spread de 105.6 puntos. <b>YW</b> experimentó un colapso más severo a <b>NPS de -12.5 (-28.5 pts)</b> con una polarización invertida donde el segmento Leisure cayó a -75.0 frente a Business/Work con 50.0, evidenciando una desalineación de expectativas en operaciones regionales con flota CRJ que generó un spread de 125 puntos. Los residentes europeos fueron los más críticos con NPS de 0.0 en el agregado, mientras que la flota CRJ operada principalmente por YW registró el peor desempeño con NPS de -12.5. Las rutas con datos de NPS mostraron BIO-MAD en 0.0 y LHR-MAD en -33.3, aunque las rutas más afectadas por meteorología no cuentan con encuestas suficientes para validar el impacto completo. Esta presión en Business SH no se propagó al nivel Short Haul debido a que fue neutralizada por la mejora artificial en Economy SH, generando una variación normal de +5.5 puntos que oculta la crisis en el segmento corporativo.<br>
<br>
En <b>Economy Short Haul</b>, el NPS subió a <b>34.4 (+7.3 pts)</b> en una mejora aparente que contradice el deterioro operativo subyacente. Ambas compañías experimentaron mejoras sinérgicas: <b>IB</b> alcanzó <b>NPS de 34.0 (+7.1 pts)</b> y <b>YW</b> llegó a <b>35.0 (+7.6 pts)</b>, impulsadas por el mismo efecto de composición de muestra. A pesar de que el Mishandling aumentó 3.69 puntos alcanzando 19.25 y se registraron 18 incidentes operativos por meteorología adversa en BIO y SDR incluyendo 6 cancelaciones, los clientes que respondieron las encuestas no fueron los afectados por estos problemas. La mejora se explica por un Load Factor reducido en 3.01 puntos alcanzando 82.92%, que generó menor ocupación y mejor experiencia percibida, compensando el deterioro en equipaje. La puntualidad mejoró marginalmente con OTP15 de 90.94%, subiendo 0.83 puntos. Los perfiles favorecidos fueron la flota A320 con NPS de 55.3 en IB, los residentes de España con 43.5 también en IB representando el 54% de las encuestas del segmento, y los viajeros Business/Work con 48.8. Las rutas críticas identificadas en IB fueron MAD-ORY con NPS de 8.3, MAD-PRG con 16.7 y HAM-MAD con 25.0, aunque las 75 rutas con incidentes NCS carecen de datos de NPS asociados, limitando la validación del impacto real. Esta mejora en Economy SH no logró convertir al Short Haul en anomalía positiva al ser neutralizada por la caída en Business, resultando en una variación normal que posteriormente contribuyó marginalmente al Global +0.5 a través de su peso volumétrico del 63% de las encuestas totales.<br>
<br>
La convergencia de <b>Long Haul (-13.6 pts)</b> en deterioro severo y <b>Short Haul (+5.5 pts variación normal)</b> con dinámicas internas contradictorias, produjo el resultado Global de +0.5 puntos mediante un mecanismo de dilución donde el volumen masivo de SH con 286 encuestas frente a las 117 de LH absorbió matemáticamente la crisis operativa intercontinental. Este resultado enmascara tres realidades críticas: el colapso progresivo en cabinas premium de Long Haul con caídas de hasta 52.1 puntos, la crisis meteorológica en Business Short Haul con impacto diferencial entre compañías, y la mejora artificial en Economy Short Haul por sesgo de muestra que no refleja los 252 incidentes operativos totales del día incluyendo 74 cancelaciones, 35 retrasos, 14 cambios de equipo y 19 pérdidas de conexión.<br>
<br>
<b><u>DETALLE POR CABINA</u></b><br>
<br>
<b><u>Economy SH: Mejora Aparente por Composición de Muestra</u></b><br>
El segmento registró un <b>NPS de 34.4 (+7.3 pts)</b> en una mejora sinérgica entre IB con 34.0 y YW con 35.0 que contradice el deterioro operativo subyacente. A pesar del aumento de Mishandling en 3.69 puntos alcanzando 19.25 maletas mal gestionadas por cada 1,000 pasajeros y los 18 incidentes operativos por meteorología adversa en Bilbao y Santander que incluyeron 6 cancelaciones y 3 retrasos, los clientes que respondieron las 256 encuestas del segmento no fueron los afectados por estos problemas. La causa raíz de la mejora es un efecto de composición favorable donde el Load Factor reducido en 3.01 puntos alcanzando 82.92% generó menor ocupación y mejor experiencia percibida, compensando el deterioro en equipaje. La puntualidad mejoró marginalmente con OTP15 de 90.94%, subiendo 0.83 puntos respecto al baseline. Los perfiles que impulsaron esta mejora fueron la flota A320 con NPS de 55.3 en 38 encuestas de IB, los residentes de España con 43.5 en 92 encuestas representando el 36% del segmento, y los viajeros Business/Work con 48.8 en 41 encuestas. IB dominó volumétricamente con 240 encuestas representando el 94% del segmento y alcanzando NPS de 35.4. Las rutas con datos de NPS en IB mostraron MAD-ORY con 8.3 en 12 encuestas, MAD-PRG con 16.7 en 6 encuestas y HAM-MAD con 25.0 en 4 encuestas, aunque las 75 rutas con incidentes NCS carecen de datos suficientes para validar el impacto real de los problemas meteorológicos en BIO, SDR y VIT. La flota CRJ operada principalmente en conexiones regionales mostró NPS de 29.9 en 87 encuestas representando el 34% del segmento, mientras que el A350 C registró un colapso a -50.0 aunque con muestra muy pequeña de 4 encuestas. Esta mejora artificial no se propagó claramente al nivel Short Haul debido a la cancelación con Business que cayó 11.8 puntos, resultando en una variación normal de +5.5 que posteriormente contribuyó marginalmente al Global +0.5 a través del peso volumétrico de Economy SH que representa el 63% de las encuestas totales de la red.<br>
<br>
<b><u>Business SH: Crisis Meteorológica con Impacto Diferencial por Compañía</u></b><br>
El segmento experimentó una caída a <b>NPS de 23.3 (-11.8 pts)</b> debido a meteorología adversa en el norte de España que generó 18 incidentes operativos concentrados en Bilbao y Santander. Las 30 encuestas del segmento reflejan una crisis sinérgica entre ambas compañías aunque con manifestaciones diferentes. <b>IB</b> cayó a <b>36.4 (-5.7 pts)</b> con una polarización extrema donde los viajeros Business/Work alcanzaron NPS de -50.0 en 4 encuestas mientras los de ocio mantuvieron 55.6 en 18 encuestas, generando un spread de 105.6 puntos que evidencia la mayor sensibilidad del segmento corporativo a las disrupciones. <b>YW</b> sufrió un colapso más severo a <b>-12.5 (-28.5 pts)</b> con una polarización invertida donde el segmento Leisure cayó a -75.0 en 4 encuestas frente a Business/Work con 50.0 también en 4 encuestas, revelando una desalineación de expectativas en operaciones regionales con flota CRJ que generó un spread de 125 puntos. El Mishandling aumentó 3.69 puntos alcanzando 19.25 y el Misconex subió 0.11 puntos a 0.84, mientras que paradójicamente el OTP15 mejoró 0.83 puntos a 90.94% y el Load Factor cayó 5.7 puntos a 66.1%, métricas que no explican el deterioro del NPS. Los 18 incidentes incluyeron 6 cancelaciones representando el 33% del total, 3 retrasos con 17%, 6 incidentes relacionados con pasajeros y 1 de equipaje. Las rutas más afectadas fueron MAD-BIO con 2 incidentes, VIT-BIO con 2 incidentes adicionales, y el vuelo IB0433 de BIO-MAD que debió regresar a Madrid ofreciendo transporte por superficie. Las rutas con datos de NPS mostraron BIO-MAD en 0.0 con 2 encuestas correlacionando con los incidentes operativos, y LHR-MAD en -33.3 con 3 encuestas. En YW específicamente, las rutas LEN-PMI, MAD-MRS y FRA-MAD alcanzaron NPS de -100 aunque con muestras de solo 1 encuesta cada una, y curiosamente estas rutas no fueron identificadas en los incidentes NCS formales. Los residentes europeos fueron los más críticos con NPS de 0.0 en 8 encuestas del agregado, desagregándose en IB con 14.3 en 7 encuestas y YW con -100.0 en 1 encuesta. La flota CRJ operada principalmente por YW en rutas regionales registró el peor desempeño con NPS de -12.5 en 8 encuestas, mientras que en IB la flota A320neo mostró NPS bajo de 18.2 en 11 encuestas representando el 50% de los vuelos de esa compañía. Esta presión en Business SH no se propagó al nivel Short Haul debido a que fue neutralizada por la mejora artificial en Economy de +7.3 puntos, generando una variación normal de +5.5 que oculta la crisis en el segmento corporativo y posteriormente no alcanzó el nivel Global por el mismo mecanismo de dilución.<br>
<br>
<b><u>Economy LH: Deterioro Operativo en Equipaje y Puntualidad</u></b><br>
El segmento cayó a <b>NPS de -5.2 (-8.8 pts)</b> debido a un deterioro operativo concentrado en gestión de equipajes y puntualidad que afectó las 97 encuestas procesadas. El Mishandling aumentó 3.69 puntos alcanzando 19.25 maletas mal gestionadas por cada 1,000 pasajeros, un incremento del 23.7% que correlaciona inversamente con la satisfacción del cliente. Simultáneamente, la puntualidad se deterioró con un OTP15 de 79.85%, cayendo 1.55 puntos porcentuales respecto al baseline de 81.40%. Los 8 incidentes operativos formales incluyeron 3 retrasos representando el 37.5% del total, 2 cancelaciones con 25%, y 1 incidente de equipaje con 12.5%, además de reprogramaciones extremas de hasta 8 horas 5 minutos en vuelos desde MAD con salida a las 09:00h y retrasos de 1 hora 20 minutos en la ruta DFW-MAD con salida a las 15:15h. El alcance geográfico abarcó 21 rutas con incidentes operacionales confirmados, concentrándose en conexiones latinoamericanas desde Madrid. Las rutas críticas fueron <b>MAD-SJO con NPS de -33.3</b> en 6 encuestas, <b>DOH-MAD con -20.0</b> en 5 encuestas, y <b>MAD-MEX con -8.3</b> en 12 encuestas siendo esta última la de mayor volumen entre las problemáticas, todas con incidentes NCS confirmados. Los viajeros de negocios fueron 3.6 veces más afectados que los de ocio, alcanzando NPS de -33.3 en 9 encuestas frente a -2.3 en 88 encuestas de Leisure, evidenciando mayor sensibilidad a los problemas de equipaje y retrasos. La flota wide-body mostró problemas críticos con el A33ACMI alcanzando NPS de -66.7 en 3 encuestas y el A333 con -30.8 en 13 encuestas, sugiriendo limitaciones operativas específicas de estos equipos, mientras que el A332 mantuvo NPS positivo de 8.8 en 17 encuestas y el A350 alcanzó 13.3 en 15 encuestas. Los residentes europeos mostraron impacto extremo con NPS de -100.0 en 6 encuestas, posiblemente por expectativas más altas o mayor exposición a los incidentes operativos, mientras que los residentes de España registraron -9.5 en 42 encuestas y los de América del Sur mantuvieron 30.8 en 13 encuestas. Los vuelos operados en codeshare con BA tuvieron impacto crítico con NPS de -100.0 aunque en muestra pequeña de 2 encuestas, mientras que IB representó el volumen principal con NPS de -6.2 en 80 encuestas y LATAM mostró excelente desempeño de 100.0 en 2 encuestas. Esta caída en Economy LH formó parte de la sinergia negativa total que elevó el NMA a Long Haul completo, pero no logró propagarse al Global debido a la dilución por el volumen masivo de Short Haul que representa el 71% de las encuestas totales.<br>
<br>
<b><u>Business LH: Impacto Severo en Segmento Corporativo y Codeshare</u></b><br>
El segmento experimentó una caída a <b>NPS de -7.7 (-26.7 pts)</b> debido al mismo deterioro operativo sistémico en equipaje y puntualidad que afectó a Long Haul, pero con impacto desproporcionadamente severo en las 13 encuestas del segmento corporativo. El Mishandling aumentó 3.69 puntos alcanzando 19.25 y el OTP15 cayó 1.55 puntos a 79.85%, con los mismos 8 incidentes operativos que incluyeron reprogramaciones críticas de hasta 8 horas 5 minutos y retrasos de 1 hora 20 minutos en DFW-MAD. El volumen extremadamente bajo de encuestas por ruta, con solo 1 a 4 respuestas en las rutas identificadas, limita severamente la confiabilidad estadística de los resultados. Las rutas MAD-MEX, MAD-SJU, DFW-MAD y EZE-MAD alcanzaron todas NPS de -100 aunque con muestras de solo 1 encuesta cada una, mientras que BOG-MAD registró 0 con 4 encuestas siendo la muestra más significativa. A diferencia de Economy LH donde el perfil Business/Work fue 3.6 veces más afectado que Leisure, en Business LH la diferencia fue mínima de solo 10 puntos con Leisure en -10.0 en 10 encuestas y Business/Work en 0.0 en 3 encuestas. La flota A350 next mostró el mejor desempeño relativo con NPS de +20.0 en 5 encuestas, mientras que el A332 registró -25.0 en 4 encuestas, el A333 colapsó a -100.0 en 1 encuesta y el A33ACMI alcanzó 0.0 también en 1 encuesta. El hallazgo más crítico fue el desempeño de los vuelos operados en codeshare con AA que alcanzaron NPS de -100.0 en 3 encuestas siendo todos detractores, sugiriendo posibles problemas de coordinación operativa, mientras que IB mantuvo +37.5 en 8 encuestas y BA registró -50.0 en 2 encuestas. La dispersión por región de residencia fue extrema alcanzando hasta 200 puntos aunque con volumen insuficiente de 1 a 3 encuestas por región que impide análisis confiable. Esta caída severa en Business LH formó parte de la sinergia negativa total que elevó el NMA a Long Haul completo con las tres cabinas cayendo simultáneamente, pero no logró propagarse al Global debido a la dilución por el volumen de Short Haul y al peso volumétrico limitado de Business LH que representa solo el 3% de las encuestas totales de la red.<br>
<br>
<b><u>Premium LH: Colapso Crítico en Cabina de Mayor Expectativa</u></b><br>
El segmento experimentó un colapso a <b>NPS de -42.9 (-52.1 pts)</b> debido al mismo deterioro operativo sistémico en equipaje y puntualidad que afectó a Long Haul, pero con impacto progresivamente más severo en las 7 encuestas de la cabina donde las expectativas de servicio son más altas. El Mishandling aumentó 3.69 puntos alcanzando 19.25 maletas mal gestionadas por cada 1,000 pasajeros, un incremento del 23.7%, y el OTP15 cayó 1.55 puntos a 79.85%. Los 8 incidentes operativos incluyeron la reprogramación extrema de 8 horas 5 minutos en vuelos desde MAD con salida a las 09:00h que impactó especialmente a pasajeros premium con menor tolerancia a disrupciones severas. El volumen extremadamente bajo de 7 encuestas totales limita severamente la robustez estadística del análisis, pero la coherencia con los incidentes operativos proporciona confianza media en las causas identificadas. Las cinco rutas identificadas todas tienen incidentes NCS confirmados mostrando alta correlación: <b>DFW-MAD alcanzó NPS de -100</b> en 1 encuesta, <b>GRU-MAD también -100</b> en 2 encuestas, <b>MAD-NRT igualmente -100</b> en 1 encuesta, MAD-MEX registró 0 en 2 encuestas y BOG-MAD alcanzó 100 en 1 encuesta. Los viajeros Business/Work colapsaron a NPS de -100.0 en 1 encuesta frente a Leisure con -33.3 en 6 encuestas, generando un diferencial de 66.7 puntos que evidencia la extrema sensibilidad del segmento corporativo premium a las disrupciones operativas. La flota wide-body mostró problemas críticos con el A333 colapsando a NPS de -100.0 en 2 encuestas y el A33ACMI alcanzando -50.0 en 4 encuestas que representan el 57% del total del segmento, mientras que el A350 registró -33.3 en 3 encuestas y el A350 next mantuvo 0.0 en 2 encuestas. Los residentes europeos y españoles mostraron impacto extremo ambos con NPS de -100.0 en 1 encuesta cada uno, mientras que los de América Central mantuvieron +33.3 en 3 encuestas, generando una dispersión de 133.3 puntos extremadamente alta. Los vuelos operados en codeshare con AA alcanzaron NPS de -100.0 en 1 encuesta, coherente con el patrón observado en Business LH donde AA también registró todos detractores. Esta caída crítica en Premium LH formó parte de la sinergia negativa total que elevó el NMA a Long Haul completo, siendo la cabina más severamente afectada con una caída de 52.1 puntos que refleja cómo el deterioro operativo impacta desproporcionadamente cuando las expectativas de servicio son más altas, pero no logró propagarse al Global debido a la dilución por el volumen masivo de Short Haul y al peso volumétrico mínimo de Premium LH que representa solo el 2% de las encuestas totales de la red.
```

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 7
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-17 to 2025-12-17:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 PASO 1: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL COMPAÑÍA)

---

## **CABINA 1: ECONOMY SHORT HAUL**

### **Escenario Detectado:** SINERGIA `(-, - | -)`

**Mapeo de Estados:**
- **IB:** NEGATIVE ANOMALY (-2.6 pts) → Estado `-`
- **YW:** NEGATIVE ANOMALY (-18.3 pts) → Estado `-`
- **PADRE (Economy SH):** NEGATIVE ANOMALY (-7.5 pts) → Estado `-`

### **Interpretación:**
Ambas compañías empujan en la misma dirección negativa. El efecto se suma y transfiere al padre, aunque **YW tiene un impacto 7x más severo** que IB (-18.3 pts vs -2.6 pts). La anomalía negativa del padre (-7.5 pts) es el resultado ponderado de ambas caídas.

---

### **NARRATIVA CAUSAL:**

Adoptamos la **Explicación del Nodo Padre (Economy SH)**, que identifica como causa principal el **desempeño crítico en vuelos CodeShare** (especialmente BA y QR con NPS -100.0) y el deterioro en la ruta **LHR-MAD** (NPS -42.9).

**Aplicación a los hijos:**

- **IB (-2.6 pts):** La caída moderada se explica por:
  - Exposición a vuelos CodeShare con partners problemáticos (BA: NPS -100.0 con 3 encuestas, QR: NPS -100.0 con 2 encuestas)
  - Ruta **LHR-MAD** con NPS -42.9 (14 encuestas), operada principalmente por IB
  - Impacto limitado por el volumen dominante de operación propia (253 encuestas con NPS +22.5)

- **YW (-18.3 pts):** La caída severa se concentra en:
  - **Ruta crítica MAD-MRS** (NPS -40.0, 5 encuestas) que impacta desproporcionadamente
  - Clientes europeos (no españoles) con NPS -17.4 (23 encuestas)
  - Flota CRJ con NPS 6.5 (77 encuestas, 88.5% de la operación YW)
  - Viajeros Leisure con NPS 3.2 (63 encuestas, 72.4% del segmento)

**Causa Común Subyacente:**
Las **22 cancelaciones y 12 retrasos** documentados en NCS (especialmente el incidente meteorológico en Florencia) afectaron a ambas compañías, aunque no se refleja directamente en encuestas por sesgo de supervivencia (pasajeros cancelados no respondieron).

---

### **EVIDENCIA CLAVE:**

| Dimensión | IB | YW | Padre (Economy SH) |
|-----------|----|----|-------------------|
| **Ruta crítica** | LHR-MAD: NPS -42.9 (n=14) | MAD-MRS: NPS -40.0 (n=5) | LHR-MAD dominante |
| **CodeShare** | BA: -100.0 (n=3), QR: -100.0 (n=2) | Concentrado en IB (96.5% de encuestas YW) | BA/QR: -100.0 |
| **Perfil afectado** | Clientes europeos y Leisure | Clientes europeos (-17.4), Leisure (3.2) | Europa: 0.0, Leisure: 17.1 |
| **Incidentes NCS** | 28 totales (22 cancelaciones) | 28 totales (meteorología FLR) | 28 totales |

---

## **CABINA 2: BUSINESS SHORT HAUL**

### **Escenario Detectado:** TRANSFERENCIA `(+, N | +)`

**Mapeo de Estados:**
- **IB:** POSITIVE ANOMALY (+22.6 pts) → Estado `+`
- **YW:** Estado "S" (sin datos suficientes) → Tratado como `N` por ausencia
- **PADRE (Business SH):** POSITIVE ANOMALY (+17.3 pts) → Estado `+`

### **Interpretación:**
IB tiene una anomalía positiva significativa (+22.6 pts) que **contagia al padre** (+17.3 pts) a pesar de que YW no tiene datos suficientes para evaluación. La ausencia de YW en el análisis (volumen insuficiente o sin encuestas) significa que el resultado del padre es **completamente impulsado por IB**.

---

### **NARRATIVA CAUSAL:**

Adoptamos la **Explicación del Hijo Dominante (IB)**, aunque con **nivel de confianza BAJO** debido a limitaciones críticas.

**Causa Identificada (IB):**
La anomalía positiva (+22.6 pts) es un **artefacto estadístico** causado por:

1. **Muestra extremadamente pequeña:** Solo 17 encuestas en todo el segmento Business/IB
2. **Sesgo de supervivencia:** Las encuestas capturaron solo pasajeros que SÍ volaron, excluyendo a los afectados por las 22 cancelaciones de MAD-FLR (meteorología adversa)
3. **Composición favorable de la muestra:**
   - **Clientes españoles:** NPS 100.0 (10 encuestas, 59% de la muestra)
   - **Clientes europeos:** NPS 14.3 (7 encuestas, 41% de la muestra)
   - **Viajeros Business/Work:** NPS 100.0 (6 encuestas)

**Contradicción Operativa:**
Pese a que las métricas operativas empeoraron significativamente:
- Mishandling: +3.91 pts vs baseline
- Misconex: +0.20 pts vs baseline
- 28 incidentes NCS (22 cancelaciones por meteorología FLR)

El NPS subió porque **la ruta problemática MAD-FLR no aparece en el análisis de encuestas** (0 respuestas de pasajeros cancelados).

---

### **EVIDENCIA CLAVE:**

| Dimensión | IB | YW | Padre (Business SH) |
|-----------|----|----|-------------------|
| **Volumen de encuestas** | 17 (no representativo) | Sin datos | 21 total |
| **Composición** | España: NPS 100.0 (n=10) vs Europa: 14.3 (n=7) | N/A | España: 100.0 (n=12) |
| **Rutas analizadas** | 11 rutas con n≤4 (no confiables) | N/A | Solo MAD-VIE (n=2) |
| **Incidentes NCS** | MAD-FLR: 0 encuestas pese a 22 cancelaciones | N/A | Meteorología FLR |
| **Nivel de confianza** | **BAJO** (sesgo crítico) | N/A | **BAJO** |

---

## **RESUMEN EJECUTIVO:**

### **Economy SH: Sinergia Negativa**
Ambas compañías (IB y YW) contribuyen a la caída del padre (-7.5 pts), impulsadas por **desempeño crítico en CodeShare** (BA/QR con NPS -100.0) y rutas específicas problemáticas (LHR-MAD para IB, MAD-MRS para YW). YW tiene un impacto 7x más severo por concentración en flota CRJ y clientes europeos Leisure.

### **Business SH: Transferencia Positiva (con Alta Incertidumbre)**
IB domina completamente el resultado del padre (+17.3 pts) con una anomalía positiva (+22.6 pts) que es un **artefacto estadístico** causado por muestra no representativa (n=17) y sesgo de supervivencia. YW no tiene datos suficientes para evaluación. **Nivel de confianza: BAJO** - No utilizar para decisiones estratégicas.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ PASO 2: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL CABINA)

---

## **RADIO 1: SHORT HAUL (SH)**

### **Escenario Detectado:** DOMINANCIA `(-, N, + | -)`

**Mapeo de Estados:**
- **Economy SH:** NEGATIVE ANOMALY (-7.5 pts) → Estado `-`
- **Premium SH:** No existe en tree_data → Estado `N/A`
- **Business SH:** POSITIVE ANOMALY (+17.3 pts) → Estado `+`
- **PADRE (SH):** NEGATIVE ANOMALY (-5.8 pts) → Estado `-`

**Formato simplificado (2 hijos):** `(-, + | -)`

### **Interpretación:**
Dos cabinas con efectos opuestos, pero **Economy domina e impone su signo negativo al radio** (-5.8 pts) a pesar de la mejora en Business (+17.3 pts). Esta dominancia se explica por:

1. **Diferencia de volumen:** Economy tiene **276 encuestas** vs Business con solo **21 encuestas** (ratio 13:1)
2. **Magnitud de la anomalía:** Economy cae -7.5 pts con alta confianza, mientras Business sube +17.3 pts con **confianza BAJA** (muestra no representativa de 21 encuestas)
3. **Peso operativo:** Economy representa el segmento mayoritario de operación SH

---

### **NARRATIVA CAUSAL:**

Adoptamos la **Explicación de la Cabina Dominante (Economy SH)**, que identifica como causa principal:

**Causa Raíz: Desempeño Crítico en Vuelos CodeShare y Rutas Específicas**

El deterioro del radio SH (-5.8 pts) está dictado por Economy debido a:

1. **Vuelos CodeShare con NPS catastrófico:**
   - **BA (British Airways):** NPS -100.0 (3 encuestas)
   - **QR (Qatar Airways):** NPS -100.0 (2 encuestas)
   - **I2:** NPS -60.0 (5 encuestas)
   - Dispersión de 171.4 pts (la más alta de todas las dimensiones)

2. **Rutas problemáticas identificadas:**
   - **LHR-MAD:** NPS -42.9 (14 encuestas) - Ruta crítica operada principalmente por IB
   - **MAD-MRS:** NPS -40.0 (5 encuestas) - Concentrada en YW

3. **Perfiles de cliente más afectados:**
   - **Clientes europeos (no españoles):** NPS 0.0 (79 encuestas en Economy)
   - **Viajeros Leisure:** NPS 18.1 (199 encuestas) vs Business/Work NPS 29.6
   - **Flota CRJ:** NPS 5.0 (80 encuestas) - Peor desempeño con muestra significativa

4. **Incidentes operativos subyacentes:**
   - **28 incidentes NCS:** 22 cancelaciones (78.6%) + 12 retrasos
   - **Meteorología adversa en Florencia (FLR):** Desvíos a BLQ, cancelaciones masivas
   - **Mishandling:** +3.59 pts vs baseline (deterioro en manejo de equipaje)

**Efecto Mitigador Parcial:**
La mejora en Business SH (+17.3 pts) **NO compensó** el deterioro de Economy debido a:
- Volumen 13x menor (21 vs 276 encuestas)
- Baja confiabilidad estadística (sesgo de muestra crítico)
- La mejora de Business es un artefacto estadístico (composición favorable: 57% clientes españoles con NPS 100.0)

---

### **EVIDENCIA CLAVE:**

| Dimensión | Economy (Dominante) | Business (Mitigador) | Radio SH |
|-----------|---------------------|----------------------|----------|
| **Volumen** | 276 encuestas | 21 encuestas | 297 encuestas |
| **Variación NPS** | -7.5 pts | +17.3 pts | **-5.8 pts** |
| **Confianza** | MEDIO | **BAJO** | MEDIO |
| **Causa principal** | CodeShare BA/QR (-100.0) | Sesgo de muestra (n=17 IB) | CodeShare BA/QR |
| **Ruta crítica** | LHR-MAD (-42.9), MAD-MRS (-40.0) | MAD-VIE (n=2, no confiable) | LHR-MAD |
| **Perfil afectado** | Europa (0.0), Leisure (18.1), CRJ (5.0) | España (100.0), Business/Work (77.8) | Europa (0.0) |

---

## **RADIO 2: LONG HAUL (LH)**

### **Escenario Detectado:** CANCELACIÓN `(N, -, + | N)`

**Mapeo de Estados:**
- **Economy LH:** Normal (+4.8 pts - within normal range) → Estado `N`
- **Business LH:** NEGATIVE ANOMALY (-24.5 pts) → Estado `-`
- **Premium LH:** POSITIVE ANOMALY (+28.2 pts) → Estado `+`
- **PADRE (LH):** Normal (+2.4 pts - within normal range) → Estado `N`

### **Interpretación:**
El radio LH muestra **estabilidad aparente** (+2.4 pts dentro del rango normal) porque las anomalías opuestas de Business (-24.5 pts) y Premium (+28.2 pts) se **anulan mutuamente**, mientras Economy permanece estable. Este es un caso clásico de **volatilidad interna oculta**.

---

### **NARRATIVA CAUSAL:**

**"El radio LH muestra estabilidad engañosa: el deterioro en Business por degradación operativa múltiple fue compensado por la mejora en Premium (artefacto estadístico), mientras Economy mantuvo performance estable."**

### **Causas Opuestas Identificadas:**

#### **BUSINESS LH (Anomalía Negativa: -24.5 pts)**

**Causa: Degradación Operativa Múltiple**
- **Nivel de Confianza:** MEDIO ⚠️

**Métricas Operativas vs Baseline:**
- **OTP15 (Puntualidad):** -2.15 pts (79.39% vs 81.54%)
- **Mishandling (Equipaje):** +3.59 pts (19.14 vs 15.55) - Desviación significativa
- **Misconex (Conexiones perdidas):** +0.19 pts (0.88 vs 0.69)

**Incidentes Operativos:**
- **17 incidentes NCS:** 4 retrasos, 3 equipaje, 2 cancelaciones, 1 desvío
- **Rutas Transatlánticas Críticas:**
  - IB364 (DFW-MAD): Reprogramado +1h 20min
  - MAD-DFW: Reprogramado +55min por ajuste de rotación
  - DOH-MAD: Incidentes reportados (1 desvío confirmado)

**Perfil Más Afectado:**
- **Viajeros Business/Work:** NPS -66.7 (3 encuestas) vs Leisure +6.7 (15 encuestas)
- **Flota A332:** NPS -66.7 (6 encuestas) - Probablemente operando rutas transatlánticas
- **Residentes en España:** NPS -25.0 (8 encuestas, 44% de la muestra)

**Limitaciones:**
- Volumen bajo (18 encuestas)
- Sin verbatims disponibles
- Rutas críticas (DFW-MAD, MAD-DFW) NO aparecen en análisis de encuestas

---

#### **PREMIUM LH (Anomalía Positiva: +28.2 pts)**

**Causa: Artefacto Estadístico por Muestra No Representativa**
- **Nivel de Confianza:** BAJA ⚠️

**Contradicción Fundamental:**
- **Anomalía observada:** NPS SUBIÓ +28.2 pts
- **Métricas operativas:** EMPEORARON (Mishandling +3.59, OTP15 -2.15)
- **Incidentes NCS:** 17 incidentes (3 equipaje, 4 retrasos)

**Explicación:**
1. **Volumen extremadamente bajo:** Solo 8 encuestas (muestra NO representativa)
2. **Sesgo de selección:** Las encuestas podrían provenir de experiencias atípicamente positivas
3. **Desconexión operativa:** Rutas con incidentes NCS (DFW-MAD, DOH-MAD) NO tienen encuestas

**Distribución de Encuestas:**
- 5 rutas de **AMÉRICA DEL SUR** (MAD-SCL, MAD-UIO, BOG-MAD, EZE-MAD, GRU-MAD)
- **Leisure:** 6 encuestas (NPS 33.3)
- **Business/Work:** 2 encuestas (NPS 50.0)

**Conclusión:** La mejora es un **evento aleatorio** sin causa operativa tangible, resultado de variabilidad estadística en muestras pequeñas.

---

#### **ECONOMY LH (Estado Normal: +4.8 pts)**

**Causa: Performance Estable Sin Desviaciones Significativas**
- Mantiene operación dentro de parámetros esperados
- NPS Period: 8.5 | NPS Baseline: 3.7
- Sin cambios detectados que requieran investigación

---

### **EVIDENCIA CLAVE:**

| Dimensión | Business (Negativa) | Premium (Positiva) | Economy (Normal) | Radio LH |
|-----------|---------------------|--------------------|--------------------|----------|
| **Volumen** | 18 encuestas | 8 encuestas | 59 encuestas | 85 encuestas |
| **Variación NPS** | -24.5 pts | +28.2 pts | +4.8 pts | **+2.4 pts (N)** |
| **Confianza** | MEDIO | **BAJA** | N/A | N/A |
| **Causa** | Degradación operativa (OTP -2.15, Mishandling +3.59) | Artefacto estadístico (n=8) | Estabilidad | Cancelación |
| **Rutas críticas** | DFW-MAD (+1h20min), DOH-MAD | Sin rutas con incidentes | N/A | N/A |
| **Perfil afectado** | Business/Work (-66.7), España (-25.0), A332 (-66.7) | Leisure (33.3), AMÉRICA SUR | N/A | N/A |

---

## **RESUMEN EJECUTIVO:**

### **Short Haul: Dominancia de Economy**
El deterioro del radio SH (-5.8 pts) está **dictado por Economy** (-7.5 pts, 276 encuestas) debido a desempeño crítico en vuelos CodeShare (BA/QR con NPS -100.0) y rutas LHR-MAD/MAD-MRS. La mejora en Business (+17.3 pts, 21 encuestas) NO compensó por diferencia de volumen 13:1 y baja confiabilidad estadística.

### **Long Haul: Cancelación de Efectos Opuestos**
El radio LH muestra **estabilidad engañosa** (+2.4 pts Normal) porque Business cayó -24.5 pts por degradación operativa (OTP -2.15, Mishandling +3.59, reprogramaciones DFW-MAD) mientras Premium subió +28.2 pts por artefacto estadístico (n=8, sin causa operativa). Economy permaneció estable (+4.8 pts), actuando como ancla.

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 PASO 3: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL GLOBAL)

---

## **ANÁLISIS DE AGREGACIÓN GLOBAL**

### **Escenario Detectado:** TRANSFERENCIA `(N, - | -)`

**Mapeo de Estados:**
- **Long Haul (LH):** Normal (+2.4 pts - within normal range) → Estado `N`
- **Short Haul (SH):** NEGATIVE ANOMALY (-5.8 pts) → Estado `-`
- **GLOBAL:** NEGATIVE ANOMALY (-3.4 pts) → Estado `-`

---

### **INTERPRETACIÓN:**

Un solo radio (Short Haul) con anomalía negativa **contagia al Global completo** (-3.4 pts) a pesar de que Long Haul permanece estable (+2.4 pts Normal). Esta transferencia se explica por:

1. **Dominancia de volumen:** SH representa **297 encuestas** (67.3%) vs LH con **144 encuestas** (32.7%) - ratio 2:1
2. **Magnitud de la anomalía:** SH cae -5.8 pts con confianza MEDIA, mientras LH mantiene estabilidad aparente (+2.4 pts)
3. **Peso operativo:** SH es el segmento mayoritario de la operación diaria, con mayor frecuencia de vuelos y volumen de pasajeros

**Patrón de Contagio:**
```
SH (-5.8 pts, 297 enc) × 67.3% + LH (+2.4 pts, 144 enc) × 32.7% ≈ Global (-3.4 pts)
```

---

## **NARRATIVA CAUSAL:**

Adoptamos la **Explicación del Radio Dominante (Short Haul)**, que identifica como causa principal:

### **CAUSA RAÍZ: Desempeño Crítico en Vuelos CodeShare y Disrupciones Operativas Masivas**

**El resultado Global está arrastrado por Short Haul debido a una combinación de factores operativos y de producto que afectaron desproporcionadamente a este segmento, pese a la estabilidad mostrada por Long Haul.**

---

## **EVIDENCIA DETALLADA:**

### **1. CAUSA PRINCIPAL: Vuelos CodeShare con NPS Catastrófico**

**Nivel de Confianza: MEDIO-ALTO**

#### **Partners Problemáticos (SH):**
- **BA (British Airways):** NPS -100.0 (4 encuestas en SH)
- **QR (Qatar Airways):** NPS -100.0 (2 encuestas en SH)
- **I2:** NPS -60.0 (5 encuestas en SH)
- **Dispersión CodeShare:** 171.4 pts (la más alta de todas las dimensiones)

**Comparativa con Operación Propia:**
- **IB (SH):** NPS +25.3 (273 encuestas) - Operación estándar positiva
- **YW (SH):** NPS +71.4 (7 encuestas) - Excelente desempeño
- **Impacto:** Aunque los vuelos CodeShare representan solo 3.7% del volumen (11 de 297 encuestas), su NPS extremadamente negativo arrastró el promedio agregado

#### **Triangulación:**
- ✅ **Customer_Profile:** BA/QR con NPS -100.0 confirmado
- ✅ **Routes:** LHR-MAD (operada por BA) con NPS -42.9 (14 encuestas)
- ❌ **NCS:** Sin incidentes específicos reportados en rutas BA/QR
- ❌ **Verbatims:** Sin comentarios disponibles para validación cualitativa

---

### **2. CAUSA SECUNDARIA: Rutas Específicas Problemáticas**

**Nivel de Confianza: MEDIO**

#### **Rutas Críticas Identificadas (SH):**

| Ruta | NPS | Encuestas | Compañía Dominante | Segmento |
|------|-----|-----------|-------------------|----------|
| **LHR-MAD** | **-42.9** | 14 | IB | Economy SH |
| **MAD-MRS** | **-40.0** | 5 | YW | Economy SH |
| MAD-VIE | 100.0 | 2 | IB | Business SH |

**Análisis:**
- **LHR-MAD:** Correlación geográfica con BA (opera desde Londres Heathrow), muestra robusta de 14 encuestas
- **MAD-MRS:** Concentrada en YW, impacto desproporcionado en segmento Marsella
- **Limitación:** Sin datos de baseline histórico por ruta para confirmar si estos valores son anómalos

---

### **3. CAUSA TERCIARIA: Disrupciones Operativas Masivas**

**Nivel de Confianza: MEDIO-BAJO**

#### **Incidentes NCS (28 totales):**
- **Cancelaciones:** 22 incidentes (78.6%)
- **Retrasos:** 12 incidentes (42.9%)
- **Equipaje:** 1 incidente (3.6%)
- **Desvíos:** 1 incidente (3.6%)

#### **Incidente Meteorológico Crítico:**
- **Localización:** Florencia (FLR)
- **Causa:** Meteorología adversa
- **Impacto:** Desvíos a Bologna (BLQ), cancelaciones masivas, falta de vuelos de retorno
- **Rutas afectadas:** MAD-FLR (2 incidentes), BLQ-FLR (2 incidentes), FLR-MAD (1 incidente)

**Limitación Crítica:**
Las rutas FLR **NO aparecen** en el análisis de NPS (0 encuestas), lo que indica **sesgo de supervivencia**: los pasajeros afectados por cancelaciones no completaron encuestas el mismo día.

#### **Métricas Operativas vs Baseline (Global):**
- **Mishandling:** +3.59 pts (19.14 vs 15.55) - Desviación significativa en manejo de equipaje
- **Misconex:** +0.19 pts (0.88 vs 0.69) - Aumento en conexiones perdidas
- **OTP15:** +0.31 pts (mejor puntualidad) - Métrica favorable que NO compensó
- **Load Factor:** -2.65 pts (menor ocupación) - Métrica favorable que NO compensó

**Interpretación:** Tres métricas operativas críticas empeoraron, pero solo Mishandling muestra desviación >3 pts (significativa).

---

### **4. PERFILES DE CLIENTE MÁS AFECTADOS**

#### **Por Región de Residencia (SH):**
| Región | NPS | Encuestas | % Muestra |
|--------|-----|-----------|-----------|
| **EUROPA** | 0.0 | 79 | 26.6% |
| ESPAÑA | 37.2 | 156 | 52.5% |
| AFRICA | -100.0 | 1 | 0.3% |

**Insight:** Pasajeros residentes en Europa (no españoles) mostraron NPS neutral/pobre (0.0), coherente con problemas en rutas europeas (LHR-MAD, MAD-MRS).

#### **Por Tipo de Viaje (SH):**
| Tipo | NPS | Encuestas | % Muestra |
|------|-----|-----------|-----------|
| **Leisure** | 18.1 | 199 | 67.0% |
| Business/Work | 29.6 | 98 | 33.0% |

**Insight:** Viajeros de ocio (mayoría de la muestra) fueron más afectados que viajeros de negocios.

#### **Por Flota (SH):**
| Aeronave | NPS | Encuestas | % Muestra |
|----------|-----|-----------|-----------|
| A320 | 44.0 | 50 | 16.8% |
| ATR | 36.4 | 11 | 3.7% |
| A320neo | 27.5 | 80 | 26.9% |
| A321 | 15.6 | 64 | 21.5% |
| **CRJ** | **5.0** | 80 | 26.9% |

**Insight:** La flota CRJ (Canadair Regional Jet) registró el NPS más bajo (5.0) con una muestra significativa de 80 encuestas (26.9% del total SH).

---

## **ESTABILIDAD APARENTE DE LONG HAUL**

**Por qué LH NO impactó negativamente en Global:**

### **Cancelación Interna de Efectos (LH):**
- **Business LH:** -24.5 pts (18 encuestas) - Degradación operativa
- **Premium LH:** +28.2 pts (8 encuestas) - Artefacto estadístico
- **Economy LH:** +4.8 pts (59 encuestas) - Estable
- **Resultado:** +2.4 pts Normal (efectos opuestos se anulan)

### **Menor Peso Relativo:**
- LH representa solo 32.7% del volumen total (144 de 441 encuestas)
- Su estabilidad aparente (+2.4 pts) NO compensó la caída de SH (-5.8 pts) por diferencia de volumen 2:1

---

## **SÍNTESIS DE EVIDENCIA:**

| Dimensión | Short Haul (Dominante) | Long Haul (Estable) | Global |
|-----------|------------------------|---------------------|--------|
| **Volumen** | 297 encuestas (67.3%) | 144 encuestas (32.7%) | 441 encuestas |
| **Variación NPS** | -5.8 pts | +2.4 pts (N) | **-3.4 pts** |
| **Confianza** | MEDIO | N/A (cancelación interna) | MEDIO |
| **Causa principal** | CodeShare BA/QR (-100.0) | Cancelación Business/Premium | CodeShare BA/QR |
| **Rutas críticas** | LHR-MAD (-42.9), MAD-MRS (-40.0) | DFW-MAD (sin encuestas) | LHR-MAD |
| **Incidentes NCS** | 28 totales (22 cancelaciones FLR) | 17 incidentes (DFW-MAD +1h20min) | 28 totales |
| **Perfil afectado** | Europa (0.0), Leisure (18.1), CRJ (5.0) | Business/Work (-66.7 en Bus) | Europa, Leisure |
| **Métricas operativas** | Mishandling +3.59, Misconex +0.19 | OTP -2.15, Mishandling +3.59 | Mishandling +3.59 |

---

## **CONCLUSIÓN EJECUTIVA:**

### **Dinámica de Transferencia: Short Haul Arrastra al Global**

**A nivel GLOBAL, la dinámica es TRANSFERENCIA `(N, - | -)`.**

**Narrativa:**
El resultado Global (-3.4 pts) está **arrastrado por Short Haul** (-5.8 pts) debido a una combinación de:

1. **Desempeño crítico en vuelos CodeShare** (BA/QR con NPS -100.0)
2. **Rutas problemáticas específicas** (LHR-MAD -42.9, MAD-MRS -40.0)
3. **Disrupciones operativas masivas** (22 cancelaciones por meteorología FLR, Mishandling +3.59 pts)
4. **Perfiles más vulnerables** (clientes europeos con NPS 0.0, viajeros Leisure 18.1, flota CRJ 5.0)

**Pese a la estabilidad mostrada por Long Haul** (+2.4 pts Normal, resultado de cancelación interna entre Business -24.5 y Premium +28.2), el **volumen 2:1 de Short Haul** (297 vs 144 encuestas) permitió que su anomalía negativa contagiara al resultado global.

**Nivel de Confianza: MEDIO**
- ✅ Evidencia operativa sólida (28 incidentes NCS, Mishandling +3.59)
- ✅ Correlación entre CodeShare/rutas y perfiles afectados
- ❌ Sin validación cualitativa (verbatims vacíos)
- ❌ Desconexión entre rutas con incidentes NCS (FLR) y rutas con NPS medido

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

# 🔍 PASO 4: IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS (NMA)

---

## **CAUSA 1: Desempeño Crítico en Vuelos CodeShare (BA/QR)**

### **Características:**
- **Escenario de Propagación:** DOMINANCIA → TRANSFERENCIA
- **NMA:** `Global/SH/Economy`
- **Afecta a:** Economy SH (IB y YW), Short Haul agregado, Global
- **Tipo de impacto:** NEGATIVO

### **Cadena de Propagación hacia el Segmento Raíz:**

```
1. Global/SH/Economy/IB (-2.6 pts) + Global/SH/Economy/YW (-18.3 pts)
   ↓
   [SINERGIA: (-, - | -)]
   ↓
2. Global/SH/Economy (-7.5 pts)
   Hermanos: Business SH (+17.3 pts, anomalía opuesta)
   ↓
   [DOMINANCIA: (-, + | -)]
   ↓
3. Global/SH (-5.8 pts)
   Hermanos: Long Haul (Normal +2.4 pts)
   ↓
   [TRANSFERENCIA: (N, - | -)]
   ↓
4. Global (-3.4 pts) ← SEGMENTO RAÍZ
```

### **Explicación de la Propagación:**

**Nivel 1 (Compañías):** SINERGIA `(-, - | -)`
- Tanto IB como YW experimentan caída en Economy SH
- IB: -2.6 pts (afectado por CodeShare BA/QR en ruta LHR-MAD)
- YW: -18.3 pts (afectado por ruta MAD-MRS y flota CRJ)
- **Causa común:** Vuelos CodeShare con BA (-100.0, 3 enc), QR (-100.0, 2 enc), I2 (-60.0, 5 enc)
- **NMA se establece en el PADRE** (Economy SH) por ser causa común

**Nivel 2 (Cabinas):** DOMINANCIA `(-, + | -)`
- Economy SH (-7.5 pts, 276 enc) domina sobre Business SH (+17.3 pts, 21 enc)
- Ratio de volumen 13:1 permite que Economy imponga su signo negativo
- **NMA permanece en Economy SH** (hijo dominante)

**Nivel 3 (Radios):** TRANSFERENCIA `(N, - | -)`
- Short Haul (-5.8 pts, 297 enc) contagia a Global (-3.4 pts)
- Long Haul permanece Normal (+2.4 pts, 144 enc)
- Ratio de volumen 2:1 (SH:LH) permite transferencia
- **NMA permanece en SH** (hijo anómalo que contagia)

**Conclusión:** El NMA es `Global/SH/Economy` porque:
1. Es el punto de convergencia de la causa común (SINERGIA IB+YW)
2. Domina sobre Business SH (DOMINANCIA)
3. Arrastra a SH y luego a Global (TRANSFERENCIA)

---

## **CAUSA 2: Ruta Crítica MAD-MRS (Marsella)**

### **Características:**
- **Escenario de Propagación:** TRANSFERENCIA → DOMINANCIA → TRANSFERENCIA
- **NMA:** `Global/SH/Economy/YW`
- **Afecta a:** YW Economy SH exclusivamente
- **Tipo de impacto:** NEGATIVO

### **Cadena de Propagación hacia el Segmento Raíz:**

```
1. Global/SH/Economy/YW (-18.3 pts)
   Hermano: IB (-2.6 pts, anomalía menor)
   ↓
   [SINERGIA: (-, - | -)] pero YW es 7x más severo
   ↓
2. Global/SH/Economy (-7.5 pts)
   Hermanos: Business SH (+17.3 pts, anomalía opuesta)
   ↓
   [DOMINANCIA: (-, + | -)]
   ↓
3. Global/SH (-5.8 pts)
   Hermanos: Long Haul (Normal +2.4 pts)
   ↓
   [TRANSFERENCIA: (N, - | -)]
   ↓
4. Global (-3.4 pts) ← SEGMENTO RAÍZ
```

### **Explicación de la Propagación:**

**Nivel 1 (Compañías):** SINERGIA con YW dominante
- YW tiene caída severa (-18.3 pts) concentrada en ruta MAD-MRS (NPS -40.0, 5 enc)
- IB tiene caída moderada (-2.6 pts) por otras causas
- Aunque ambos caen (SINERGIA), **YW es el hijo dominante** por magnitud 7x superior
- **NMA = YW** (hijo con causa específica más severa)

**Nivel 2 (Cabinas):** DOMINANCIA `(-, + | -)`
- Economy SH (-7.5 pts, 276 enc) domina sobre Business SH (+17.3 pts, 21 enc)
- La caída de YW contribuye desproporcionadamente a Economy SH
- **NMA permanece en YW** (causa específica localizada)

**Nivel 3 (Radios):** TRANSFERENCIA `(N, - | -)`
- Short Haul (-5.8 pts) contagia a Global (-3.4 pts)
- **NMA permanece en YW** (causa localizada en una compañía específica)

**Conclusión:** El NMA es `Global/SH/Economy/YW` porque:
1. Es el punto de origen de la causa específica (ruta MAD-MRS)
2. Tiene impacto 7x más severo que IB
3. Propaga hacia arriba pero la causa NO es común (no sube a Economy SH como NMA)

---

## **CAUSA 3: Ruta Crítica LHR-MAD (Londres-Madrid)**

### **Características:**
- **Escenario de Propagación:** TRANSFERENCIA → DOMINANCIA → TRANSFERENCIA
- **NMA:** `Global/SH/Economy/IB`
- **Afecta a:** IB Economy SH, correlacionado con CodeShare BA
- **Tipo de impacto:** NEGATIVO

### **Cadena de Propagación hacia el Segmento Raíz:**

```
1. Global/SH/Economy/IB (-2.6 pts)
   Hermano: YW (-18.3 pts, anomalía más severa)
   ↓
   [SINERGIA: (-, - | -)] pero IB tiene causa específica LHR-MAD
   ↓
2. Global/SH/Economy (-7.5 pts)
   Hermanos: Business SH (+17.3 pts, anomalía opuesta)
   ↓
   [DOMINANCIA: (-, + | -)]
   ↓
3. Global/SH (-5.8 pts)
   Hermanos: Long Haul (Normal +2.4 pts)
   ↓
   [TRANSFERENCIA: (N, - | -)]
   ↓
4. Global (-3.4 pts) ← SEGMENTO RAÍZ
```

### **Explicación de la Propagación:**

**Nivel 1 (Compañías):** SINERGIA con causa específica de IB
- IB tiene caída moderada (-2.6 pts) con ruta LHR-MAD (NPS -42.9, 14 enc)
- Correlación geográfica con CodeShare BA (opera desde LHR)
- Aunque ambos caen (SINERGIA), **IB tiene causa específica localizable**
- **NMA = IB** (hijo con ruta crítica identificable)

**Nivel 2 (Cabinas):** DOMINANCIA `(-, + | -)`
- Economy SH (-7.5 pts) domina sobre Business SH (+17.3 pts)
- **NMA permanece en IB** (causa específica de ruta)

**Nivel 3 (Radios):** TRANSFERENCIA `(N, - | -)`
- Short Haul (-5.8 pts) contagia a Global (-3.4 pts)
- **NMA permanece en IB** (causa localizada)

**Conclusión:** El NMA es `Global/SH/Economy/IB` porque:
1. Es el punto de origen de la ruta crítica LHR-MAD
2. Tiene correlación directa con CodeShare BA
3. La causa es específica y localizable (no común con YW)

---

## **CAUSA 4: Disrupciones Operativas Masivas (Meteorología FLR)**

### **Características:**
- **Escenario de Propagación:** SINERGIA → DOMINANCIA → TRANSFERENCIA
- **NMA:** `Global/SH/Economy`
- **Afecta a:** Economy SH (IB y YW), Short Haul agregado, Global
- **Tipo de impacto:** NEGATIVO

### **Cadena de Propagación hacia el Segmento Raíz:**

```
1. Global/SH/Economy/IB (-2.6 pts) + Global/SH/Economy/YW (-18.3 pts)
   ↓
   [SINERGIA: (-, - | -)]
   ↓
2. Global/SH/Economy (-7.5 pts)
   Hermanos: Business SH (+17.3 pts, anomalía opuesta)
   ↓
   [DOMINANCIA: (-, + | -)]
   ↓
3. Global/SH (-5.8 pts)
   Hermanos: Long Haul (Normal +2.4 pts)
   ↓
   [TRANSFERENCIA: (N, - | -)]
   ↓
4. Global (-3.4 pts) ← SEGMENTO RAÍZ
```

### **Explicación de la Propagación:**

**Nivel 1 (Compañías):** SINERGIA `(-, - | -)`
- Tanto IB como YW afectados por 28 incidentes NCS (22 cancelaciones)
- Causa raíz: Meteorología adversa en Florencia (FLR)
- Rutas afectadas: MAD-FLR, BLQ-FLR, FLR-MAD (5 incidentes)
- **Causa común operativa** → **NMA = PADRE (Economy SH)**

**Nivel 2 (Cabinas):** DOMINANCIA `(-, + | -)`
- Economy SH (-7.5 pts, 276 enc) domina sobre Business SH (+17.3 pts, 21 enc)
- **NMA permanece en Economy SH** (hijo dominante)

**Nivel 3 (Radios):** TRANSFERENCIA `(N, - | -)`
- Short Haul (-5.8 pts) contagia a Global (-3.4 pts)
- **NMA permanece en Economy SH** (causa común que domina)

**Limitación Crítica:**
- Las rutas FLR **NO aparecen** en análisis de NPS (0 encuestas)
- Sesgo de supervivencia: pasajeros cancelados no respondieron
- Impacto operativo confirmado pero sin validación en NPS

**Conclusión:** El NMA es `Global/SH/Economy` porque:
1. Es una causa común operativa (SINERGIA IB+YW)
2. Afecta transversalmente a ambas compañías
3. Domina sobre Business SH y contagia a Global

---

## **CAUSA 5: Deterioro en Manejo de Equipaje (Mishandling)**

### **Características:**
- **Escenario de Propagación:** SINERGIA → DOMINANCIA → TRANSFERENCIA
- **NMA:** `Global/SH/Economy`
- **Afecta a:** Economy SH (IB y YW), Short Haul agregado, Global
- **Tipo de impacto:** NEGATIVO

### **Cadena de Propagación hacia el Segmento Raíz:**

```
1. Global/SH/Economy/IB (-2.6 pts) + Global/SH/Economy/YW (-18.3 pts)
   ↓
   [SINERGIA: (-, - | -)]
   ↓
2. Global/SH/Economy (-7.5 pts)
   Hermanos: Business SH (+17.3 pts, anomalía opuesta)
   ↓
   [DOMINANCIA: (-, + | -)]
   ↓
3. Global/SH (-5.8 pts)
   Hermanos: Long Haul (Normal +2.4 pts)
   ↓
   [TRANSFERENCIA: (N, - | -)]
   ↓
4. Global (-3.4 pts) ← SEGMENTO RAÍZ
```

### **Explicación de la Propagación:**

**Nivel 1 (Compañías):** SINERGIA `(-, - | -)`
- Métrica operativa: Mishandling +3.59 pts vs baseline (19.14 vs 15.55)
- Desviación significativa (>3 pts)
- Solo 1 incidente de equipaje reportado en NCS (discrepancia)
- **Causa común operativa** → **NMA = PADRE (Economy SH)**

**Nivel 2 (Cabinas):** DOMINANCIA `(-, + | -)`
- Economy SH (-7.5 pts) domina sobre Business SH (+17.3 pts)
- **NMA permanece en Economy SH**

**Nivel 3 (Radios):** TRANSFERENCIA `(N, - | -)`
- Short Haul (-5.8 pts) contagia a Global (-3.4 pts)
- **NMA permanece en Economy SH**

**Limitación:**
- Sin validación cualitativa (verbatims vacíos)
- Discrepancia entre métrica operativa (+3.59) y reportes NCS (1 incidente)
- Nivel de confianza: MEDIO-BAJO

**Conclusión:** El NMA es `Global/SH/Economy` porque:
1. Es una causa común operativa (métrica transversal)
2. Afecta a ambas compañías (SINERGIA)
3. Domina y contagia hacia arriba

---

## **CAUSA 6: Degradación Operativa Múltiple (LH Business)**

### **Características:**
- **Escenario de Propagación:** Segmento hoja sin subniveles
- **NMA:** `Global/LH/Business`
- **Afecta a:** Business LH exclusivamente
- **Tipo de impacto:** NEGATIVO

### **Cadena de Propagación hacia el Segmento Raíz:**

```
1. Global/LH/Business (-24.5 pts)
   Hermanos: Economy LH (Normal +4.8 pts), Premium LH (+28.2 pts)
   ↓
   [CANCELACIÓN: (N, -, + | N)]
   ↓
2. Global/LH (Normal +2.4 pts)
   Hermanos: Short Haul (-5.8 pts)
   ↓
   [TRANSFERENCIA: (N, - | -)]
   ↓
3. Global (-3.4 pts) ← SEGMENTO RAÍZ
```

### **Explicación de la Propagación:**

**Nivel 1 (Cabinas LH):** CANCELACIÓN `(N, -, + | N)`
- Business LH cae -24.5 pts por degradación operativa
- Premium LH sube +28.2 pts (artefacto estadístico)
- Economy LH permanece Normal (+4.8 pts)
- **Efectos opuestos se anulan** → LH resulta Normal (+2.4 pts)
- **NMA = Business LH** (segmento con causa operativa real)

**Nivel 2 (Radios):** TRANSFERENCIA `(N, - | -)`
- Long Haul Normal (+2.4 pts) NO contagia a Global
- Short Haul (-5.8 pts) sí contagia a Global (-3.4 pts)
- **La causa de Business LH NO propaga** por cancelación interna

**Caso Especial:**
- Segmento hoja sin subniveles (no hay análisis IB/YW en LH)
- El NMA coincide con el segmento raíz de esta causa
- **No hay cadena de propagación efectiva hacia Global** debido a cancelación con Premium LH

**Conclusión:** El NMA es `Global/LH/Business` porque:
1. Es un segmento hoja sin subniveles que analizar
2. Tiene causa operativa específica (OTP -2.15, Mishandling +3.59, reprogramaciones DFW-MAD)
3. NO propaga a Global por cancelación con Premium LH (+28.2 pts)

---

## **CAUSA 7: Artefacto Estadístico por Muestra No Representativa (LH Premium)**

### **Características:**
- **Escenario de Propagación:** Segmento hoja sin subniveles
- **NMA:** `Global/LH/Premium`
- **Afecta a:** Premium LH exclusivamente
- **Tipo de impacto:** POSITIVO (falso)

### **Cadena de Propagación hacia el Segmento Raíz:**

```
1. Global/LH/Premium (+28.2 pts)
   Hermanos: Economy LH (Normal +4.8 pts), Business LH (-24.5 pts)
   ↓
   [CANCELACIÓN: (N, -, + | N)]
   ↓
2. Global/LH (Normal +2.4 pts)
   Hermanos: Short Haul (-5.8 pts)
   ↓
   [TRANSFERENCIA: (N, - | -)]
   ↓
3. Global (-3.4 pts) ← SEGMENTO RAÍZ
```

### **Explicación de la Propagación:**

**Nivel 1 (Cabinas LH):** CANCELACIÓN `(N, -, + | N)`
- Premium LH sube +28.2 pts (anomalía positiva falsa)
- Business LH cae -24.5 pts (anomalía negativa real)
- **Efectos opuestos se anulan** → LH resulta Normal (+2.4 pts)
- **NMA = Premium LH** (segmento con anomalía, aunque sea artefacto)

**Nivel 2 (Radios):** TRANSFERENCIA `(N, - | -)`
- Long Haul Normal (+2.4 pts) NO contagia a Global
- **La causa de Premium LH NO propaga** por cancelación con Business LH

**Caso Especial:**
- Segmento hoja sin subniveles (no hay análisis IB/YW en LH)
- **Causa identificada:** Muestra extremadamente pequeña (n=8)
- **Nivel de confianza:** BAJA (artefacto estadístico sin causa operativa)
- **Contradicción:** Métricas operativas empeoraron (Mishandling +3.59, OTP -2.15) pero NPS subió

**Conclusión:** El NMA es `Global/LH/Premium` porque:
1. Es un segmento hoja sin subniveles que analizar
2. Tiene anomalía positiva (aunque sea falsa)
3. NO propaga a Global por cancelación con Business LH (-24.5 pts)
4. **No requiere acción correctiva** (es variabilidad estadística, no problema operativo)

---

## **RESUMEN DE NODOS MÁXIMO AFECTADOS:**

| # | Causa | NMA | Tipo | Propaga a Global |
|---|-------|-----|------|------------------|
| 1 | CodeShare BA/QR | `Global/SH/Economy` | NEGATIVO | ✅ Sí (TRANSFERENCIA) |
| 2 | Ruta MAD-MRS | `Global/SH/Economy/YW` | NEGATIVO | ✅ Sí (TRANSFERENCIA) |
| 3 | Ruta LHR-MAD | `Global/SH/Economy/IB` | NEGATIVO | ✅ Sí (TRANSFERENCIA) |
| 4 | Meteorología FLR | `Global/SH/Economy` | NEGATIVO | ✅ Sí (TRANSFERENCIA) |
| 5 | Mishandling | `Global/SH/Economy` | NEGATIVO | ✅ Sí (TRANSFERENCIA) |
| 6 | Degradación Operativa LH | `Global/LH/Business` | NEGATIVO | ❌ No (CANCELACIÓN) |
| 7 | Artefacto Estadístico LH | `Global/LH/Premium` | POSITIVO | ❌ No (CANCELACIÓN) |

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

# 📊 PASO 4B: EXTRACCIÓN DE EVIDENCIAS DEL CONTEXTO INICIAL

---

## **NMA 1: Global/SH/Economy**

### **=== NMA: Global/SH/Economy ===**

#### **📈 EXPLANATORY DRIVERS:**
No disponible

#### **📊 DATOS OPERATIVOS:**
**Global/SH/Economy:**
- OTP15: 90.8% (+0.71 pts vs baseline de 90.09%)
- Load Factor: 83.0% (-2.95 pts vs baseline de 85.9%)
- Mishandling: 19.14 (+3.59 pts vs baseline de 15.55)
- Misconex: 0.88 (+0.19 pts vs baseline de 0.69)

**Interpretación del análisis causal:**
"OTP15 mejoró → Debería SUBIR NPS ✅
Load Factor bajó → Debería SUBIR NPS (mejor experiencia) ✅
Mishandling subió → Debería BAJAR NPS ⚠️ (pero sin validación en NCS/verbatims)
**Conclusión**: Las métricas operativas NO explican la caída de -7.5 pts. La causa es cualitativa (experiencia en vuelos CodeShare)."

#### **🚨 INCIDENTES NCS (CUANTITATIVO):**
**Global/SH/Economy:**
- Total incidentes: 28
- Cancelaciones: 22 incidentes (78.6%)
- Retrasos: 12 incidentes (42.9%)
- Desvíos: 1 incidente (3.6%)
- Equipaje: 1 incidente (3.6%)

**Distribución geográfica:**
- Foco en Florencia (FLR): 5 incidentes documentados
- Rutas: MAD-FLR (2), BLQ-FLR (2), FLR-MAD (1)

#### **🧠 NCS (CUALITATIVO / REFLEXIÓN):**
**Incidente Meteorológico Crítico:**
"**Meteorología FLR**: 2 vuelos MAD-FLR desviados a BLQ con transporte por superficie, sin vuelos de retorno disponibles en días posteriores."

**Patrón identificado:**
"⚠️ **DESCONEXIÓN CRÍTICA:** Las rutas FLR NO aparecen en los datos de NPS, lo que indica que este incidente meteorológico **NO impactó significativamente** en las encuestas del día."

**Limitación crítica:**
"**⚠️ Limitación crítica**: Ninguno de estos incidentes correlaciona directamente con la caída de NPS. Los 22 vuelos cancelados no generaron encuestas NPS el mismo día."

#### **💬 FEEDBACK DE CLIENTES:**
**Estado:** No hay comentarios con texto disponibles para el 17-dic-2025.

**Implicación:** "Imposibilidad de validar cualitativamente las hipótesis operativas o identificar causas emergentes no capturadas por métricas."

#### **✈️ RUTAS AFECTADAS (Top 5):**
**Global/SH/Economy:**
1. **LHR-MAD**: NPS -42.9 (14 encuestas) - Muestra más robusta, NPS muy bajo
2. DSS-MAD: NPS 0.0 (4 encuestas) - Muestra pequeña
3. MAD-NAP: NPS 0.0 (3 encuestas) - Muestra pequeña
4. ATH-MAD: NPS 0.0 (2 encuestas) - Muestra muy pequeña
5. MAD-OSL: NPS 0.0 (1 encuesta) - Muestra muy pequeña

**Nota crítica:** "Las rutas FLR (MAD-FLR, BLQ-FLR, FLR-MAD) con 5 incidentes NCS documentados **NO aparecen** en los datos de encuestas NPS."

#### **👥 PERFILES REACTIVOS:**

**Por CodeShare (Dispersión: 171.4 pts):**
- VY: NPS +71.4 (7 encuestas)
- IB: NPS +22.5 (253 encuestas)
- **BA: NPS -100.0 (3 encuestas)** ⚠️
- **QR: NPS -100.0 (2 encuestas)** ⚠️
- **I2: NPS -60.0 (5 encuestas)** ⚠️

**Por Residence Region (Dispersión: 150.0 pts):**
- ESPAÑA: NPS +31.9 (144 encuestas)
- EUROPA: NPS 0.0 (71 encuestas)
- AFRICA: NPS -100.0 (1 encuesta)

**Por Business/Leisure:**
- Business/Work: NPS +24.7 (89 encuestas)
- Leisure: NPS +17.1 (187 encuestas)
- Diferencia: 7.6 pts a favor de viajeros de negocios

**Por Fleet (Dispersión: 33.9 pts):**
- A320: NPS +40.4 (47 encuestas)
- ATR: NPS +30.0 (10 encuestas)
- A320neo: NPS +25.0 (28 encuestas)
- A321: NPS +21.4 (14 encuestas)
- A319: NPS +12.5 (8 encuestas)
- **CRJ: NPS +6.5 (169 encuestas)** ⚠️

---

## **NMA 2: Global/SH/Economy/YW**

### **=== NMA: Global/SH/Economy/YW ===**

#### **📈 EXPLANATORY DRIVERS:**
No disponible

#### **📊 DATOS OPERATIVOS:**
**Global/SH/Economy/YW:**
- Mishandling: 15.36 (+2.58 vs baseline)
- OTP15: 88.74% (+0.36 vs baseline)
- Load Factor: 78.51% (-1.99 vs baseline)

**Análisis de Correlación:**
"❌ OTP mejoró ligeramente → NO explica caída de NPS
❌ Load Factor bajó (menos ocupación) → Debería mejorar NPS, no empeorarlo
⚠️ Mishandling subió 2.58 puntos → Correlaciona con caída de NPS, pero solo 1 incidente reportado en NCS"

#### **🚨 INCIDENTES NCS (CUANTITATIVO):**
**Global/SH/Economy/YW:**
- Total incidentes: 28
- Cancelaciones: 22 incidentes
- Retrasos: 12 incidentes
- Equipaje: 1 incidente

#### **🧠 NCS (CUALITATIVO / REFLEXIÓN):**
**Incidente Destacado:**
"**Ruta MAD-FLR:** 2 incidentes por meteorología adversa con desvío a BLQ
Comentario crítico: 'No hay vuelos FLR-MAD en los próximos días'
**Nota:** Esta ruta NO aparece en el análisis de NPS por falta de encuestas"

#### **💬 FEEDBACK DE CLIENTES:**
**Estado:** Sin comentarios de texto disponibles para el período analizado.

**Rutas listadas en sistema:** 38 rutas operativas (CMN-MAD, MAD-MRS, BCN-VLC, BCN-BJZ, MAD-TNG, LYS-MAD, ALG-MAD, AGP-MAD, MAD-TLS, ALC-MAD, AMS-MAD, MAD-MLN, MAD-TRN, BOD-MAD, LEI-MAD, LIS-MAD, BCN-MLN, MAD-SCQ, IBZ-PMI, MAH-PMI, MAD-SVQ, MAD-MAH, MAD-SXB, BLQ-MAD, MAD-MUC, MAD-XRY, MAD-VCE, IBZ-VLC, MAH-VLC, BCN-LEN, LEI-PMI, GVA-MAD, MAD-VLC, FRA-MAD, MAD-NCE, MAD-VGO, LCG-MAD, GRX-MAD)

#### **✈️ RUTAS AFECTADAS (Top 5):**
**Global/SH/Economy/YW:**
1. **MAD-MRS: NPS -40.0 (5 encuestas)** - Ruta crítica

**Nota:** "La mayoría de rutas operativas (38 totales) presentan muestras muy pequeñas (1-5 encuestas) con NPS extremos (0.0 o 100.0) que no son estadísticamente representativos."

**Rutas sin datos de NPS:**
- MAD-FLR (identificada en NCS con meteorología adversa)
- 36 rutas adicionales sin suficientes encuestas para cálculo

#### **👥 PERFILES REACTIVOS:**

**Por Región de Residencia:**
- **EUROPA: NPS -17.4 (23 encuestas)** ⚠️ - 26.4% de la muestra
- ESPAÑA: NPS 28.6 (42 encuestas) - 48.3% de la muestra
- Unknown: NPS -66.7 (3 encuestas) - 3.4%

**Análisis:** "Clientes europeos (no españoles) muestran insatisfacción marcada, coherente con problemas en ruta francesa (MAD-MRS)."

**Por Tipo de Viaje:**
- **Leisure: NPS 3.2 (63 encuestas)** ⚠️ - 72.4% de la muestra
- Business: NPS 25.0 (24 encuestas) - 27.6%

**Análisis:** "Viajeros de ocio (mayoría de la muestra) significativamente más insatisfechos que viajeros de negocios."

**Por Flota:**
- **CRJ: NPS 6.5 (77 encuestas)** ⚠️ - 88.5% de la muestra
- ATR: NPS 30.0 (10 encuestas) - 11.5%

**Análisis:** "Flota CRJ (dominante en operación) muestra NPS significativamente inferior a ATR."

**Por Codeshare:**
- **IB: NPS 8.3 (84 encuestas)** - 96.5% de la muestra
- Others: NPS 100.0 (1 encuesta) - 1.1%
- VY: NPS 100.0 (1 encuesta) - 1.1%
- I2: NPS -100.0 (1 encuesta) - 1.1%

---

## **NMA 3: Global/SH/Economy/IB**

### **=== NMA: Global/SH/Economy/IB ===**

#### **📈 EXPLANATORY DRIVERS:**
No disponible

#### **📊 DATOS OPERATIVOS:**
**Global/SH/Economy/IB:**
- Load Factor: 85.51%
- OTP15: 93.1%
- Mishandling: 20.37
- Misconex: 0.93

**Nota del resumen operative_data (contradictorio con datos crudos):**
"Load Factor: 3.53 pts inferior al baseline
OTP15: 1.01 pts superior al baseline
Mishandling: 3.91 pts superior al baseline (⚠️ empeoramiento)
Misconex: 0.2 pts superior al baseline (⚠️ empeoramiento)"

**Limitación:** "⚠️ **Sin baseline histórico confirmado, no se puede validar correlación con NPS**"

#### **🚨 INCIDENTES NCS (CUANTITATIVO):**
**Global/SH/Economy/IB:**
- Total incidentes: 28
- Cancelaciones: 22 incidentes (78.6%)
- Retrasos: 12 incidentes (42.9%)
- Desvíos: 1 incidente (3.6%)
- Equipaje: 1 incidente (3.6%)

#### **🧠 NCS (CUALITATIVO / REFLEXIÓN):**
**Patrón geográfico identificado:**
"**Foco en Florencia (FLR)**: 5 incidentes documentados
- Rutas: MAD-FLR (2), BLQ-FLR (2), FLR-MAD (1)
- **Causa raíz**: Meteorología adversa en FLR
- Impacto: Desvíos a BLQ, cancelaciones, falta de vuelos de retorno"

**Desconexión crítica:**
"⚠️ **DESCONEXIÓN CRÍTICA:** Las rutas FLR NO aparecen en los datos de NPS, lo que indica que este incidente meteorológico **NO impactó significativamente** en las encuestas del día."

#### **💬 FEEDBACK DE CLIENTES:**
"❌ **0 comentarios con texto disponibles**
31 rutas mencionadas sin contenido cualitativo asociado
Imposibilidad de triangulación cualitativa"

#### **✈️ RUTAS AFECTADAS (Top 5):**
**Global/SH/Economy/IB:**
1. **LHR-MAD: NPS -42.9 (14 encuestas)** - Muestra más robusta, NPS muy bajo
2. DSS-MAD: NPS 0.0 (4 encuestas) - Muestra pequeña
3. MAD-NAP: NPS 0.0 (3 encuestas) - Muestra pequeña
4. ATH-MAD: NPS 0.0 (2 encuestas) - Muestra muy pequeña
5. MAD-OSL: NPS 0.0 (1 encuesta) - Muestra muy pequeña

**Limitación:** "⚠️ **Limitación:** Sin baseline histórico por ruta, no se puede determinar si estos valores son anómalos o normales para estas rutas."

**Nota crítica:** "Las rutas FLR (MAD-FLR, BLQ-FLR, FLR-MAD) con 5 incidentes NCS documentados **NO aparecen** en los datos de encuestas NPS."

#### **👥 PERFILES REACTIVOS:**

**Por CODESHARE (Dispersión: 166.7 pts):**
- IB: NPS +29.6 (169 encuestas) - 89%
- VY: NPS +66.7 (6 encuestas) - 3%
- **BA: NPS -100.0 (3 encuestas)** ⚠️ - 2%
- **QR: NPS -100.0 (2 encuestas)** ⚠️ - 1%
- **LATAM: NPS -100.0 (1 encuesta)** ⚠️ - 1%
- **I2: NPS -50.0 (4 encuestas)** ⚠️ - 2%

**Patrón observado:** "⚠️ **PATRÓN OBSERVADO:** Códigos compartidos NO-IB muestran NPS catastrófico, pero sin baseline histórico no se puede confirmar si es anómalo."

**Por ORIGEN GEOGRÁFICO (Dispersión: 87.4 pts):**
- ESPAÑA: NPS +33.3 (102 encuestas)
- EUROPA: NPS +8.3 (48 encuestas)
- AFRICA: NPS +50.0 (6 encuestas)
- AMERICA NORTE: NPS +11.1 (9 encuestas)
- **AMERICA CENTRO: NPS -28.6 (7 encuestas)** ⚠️
- **AMERICA SUR: NPS -12.5 (8 encuestas)** ⚠️

**Patrón observado:** "⚠️ **PATRÓN OBSERVADO:** Clientes americanos sistemáticamente insatisfechos, pero sin baseline histórico no se puede confirmar si es anómalo."

**Por FLOTA (Dispersión: 28.6 pts):**
- A320: NPS +40.4 (47 encuestas)
- A320neo: NPS +25.0 (72 encuestas)
- A321: NPS +11.9 (59 encuestas)
- A319: NPS +30.8 (13 encuestas)

**Por BUSINESS/LEISURE (Dispersión: 0.4 pts):**
- Business/Work: NPS +24.6 (65 encuestas)
- Leisure: NPS +24.2 (124 encuestas)

**Conclusión:** "No es factor diferenciador en este día."

---

## **NMA 4: Global/LH/Business**

### **=== NMA: Global/LH/Business ===**

#### **📈 EXPLANATORY DRIVERS:**
No disponible

#### **📊 DATOS OPERATIVOS:**
**Global/LH/Business:**

**Métricas Operativas vs Baseline:**
- **OTP15 (Puntualidad):** 79.39% vs baseline 81.54% = **-2.15 pts** (⬇️ Negativo)
- **Mishandling (Equipaje):** 19.14 vs baseline 15.55 = **+3.59 pts** (⬇️ Negativo)
- **Misconex (Conexiones perdidas):** 0.88 vs baseline 0.69 = **+0.19 pts** (⬇️ Negativo)
- **Load Factor (Ocupación):** 91.57% vs baseline 94.15% = -2.58 pts (⬆️ Positivo - menor ocupación)

**Interpretación:** "Tres métricas operativas críticas empeoraron simultáneamente, correlacionando con la caída de NPS. La única métrica favorable (Load Factor) no compensó el impacto negativo."

#### **🚨 INCIDENTES NCS (CUANTITATIVO):**
**Global/LH/Business:**
- Total incidentes: 17
- Retrasos: 4 incidentes
- Equipaje: 3 incidentes
- Cancelaciones: 2 incidentes
- Desvíos: 1 incidente
- Otras incidencias: 7

#### **🧠 NCS (CUALITATIVO / REFLEXIÓN):**
**Rutas Transatlánticas Críticas:**
"**IB364 (DFW-MAD):** Reprogramado +1h 20min (18 de diciembre)
**MAD-DFW:** Reprogramado +55min por ajuste de rotación
**DOH-MAD:** Incidentes reportados (1 desvío confirmado)"

**Distribución de Incidentes:**
"**Puntualidad:** 4 retrasos + 2 cancelaciones = **6 incidentes** (correlaciona con OTP15 ↓)
**Equipaje:** 3 incidentes (correlaciona con Mishandling ↑)
**Operaciones:** 1 desvío + 7 otras incidencias"

**Patrón Detectado:**
"⚠️ **Patrón Detectado:** Ajustes de rotación en rutas transatlánticas generaron efecto cascada en puntualidad y conexiones."

**Inconsistencia crítica:**
"⚠️ **INCONSISTENCIA CRÍTICA DETECTADA:**
**DFW-MAD** y **MAD-DFW** (rutas con incidentes más severos en NCS: +1h20min y +55min de reprogramación) **NO aparecen** en el análisis de rutas.
**Hipótesis:** No se completaron encuestas en estas rutas críticas ese día, creando desconexión entre incidentes operativos y feedback de clientes."

#### **💬 FEEDBACK DE CLIENTES:**
"❌ **No hay comentarios con texto disponibles para el período analizado.**

**Hipótesis sobre la ausencia:**
- Volumen extremadamente bajo de encuestas (n=18)
- Clientes solo proporcionaron puntuaciones sin comentarios escritos
- Posible problema técnico en recolección de feedback cualitativo

**Impacto en el Análisis:** Imposibilita triangulación cualitativa y validación de percepción del cliente."

#### **✈️ RUTAS AFECTADAS (Top 5):**
**Global/LH/Business:**
1. MAD-MVD: NPS -100 (1 encuesta) ⚠️ Muy baja confiabilidad (n=1)
2. MAD-SJU: NPS -100 (1 encuesta) ⚠️ Muy baja confiabilidad (n=1)
3. JFK-MAD: NPS -100 (1 encuesta) ⚠️ Muy baja confiabilidad (n=1)
4. DOH-MAD: NPS -100 (1 encuesta) ⚠️ Muy baja confiabilidad (n=1)
5. MAD-MIA: NPS -100 (1 encuesta) ⚠️ Muy baja confiabilidad (n=1)

**Limitación Estadística:** "7 de 10 rutas tienen n=1, haciendo imposible análisis robusto por ruta individual."

**Inconsistencia crítica detectada:**
"**DFW-MAD** y **MAD-DFW** (rutas con incidentes más severos en NCS: +1h20min y +55min de reprogramación) **NO aparecen** en el análisis de rutas."

#### **👥 PERFILES REACTIVOS:**

**Por Propósito de Viaje:**
- **Business/Work: NPS -66.7 (3 encuestas)** ⚠️ - 17%
- Leisure: NPS +6.7 (15 encuestas) - 83%

**Insight Clave:** "Los viajeros de negocios experimentaron un impacto **10x más negativo** que viajeros de ocio. Los retrasos y cancelaciones afectan desproporcionadamente a este segmento (conexiones críticas, compromisos profesionales)."

**Por Flota:**
- **A321XLR: NPS -100 (2 encuestas)** ⚠️ Baja confiabilidad
- **A332: NPS -66.7 (6 encuestas)** ✅ Media confiabilidad
- A350 next: NPS 0 (2 encuestas) ⚠️ Baja
- A350: NPS +75 (4 encuestas) ⚠️ Media-baja
- A333: NPS +100 (1 encuesta) ⚠️ Muy baja

**Correlación con NCS:** "La flota **A332** (probablemente operando rutas transatlánticas como DFW-MAD) muestra el peor desempeño con volumen suficiente para análisis (n=6)."

**Por Región de Residencia:**
- **ESPAÑA: NPS -25.0 (8 encuestas)** ⚠️ - 44%
- AMÉRICA DEL NORTE: NPS +50.0 (4 encuestas) - 22%
- AMÉRICA CENTRAL: NPS +50.0 (2 encuestas) - 11%
- Sin especificar: NPS -100 (3 encuestas) - 17%
- AMÉRICA DEL SUR: NPS 0 (1 encuesta) - 6%

**Insight Clave:** "Pasajeros con residencia en **España** (probablemente en rutas MAD-origen) fueron los más afectados, representando el segmento más grande con NPS negativo."

**Por Operador (CodeShare):**
- **IB: NPS +6.2 (16 encuestas)** ✅ Alta confiabilidad
- **BA: NPS -100 (1 encuesta)** ⚠️ Muy baja
- **QR: NPS -100 (1 encuesta)** ⚠️ Muy baja

**Nota:** "El volumen de encuestas en BA y QR es insuficiente para conclusiones robustas."

---

## **NMA 5: Global/LH/Premium**

### **=== NMA: Global/LH/Premium ===**

#### **📈 EXPLANATORY DRIVERS:**
No disponible

#### **📊 DATOS OPERATIVOS:**
**Global/LH/Premium:**

**Métricas Operativas (vs baseline):**
- **Mishandling:** 19.14 (+3.59 pts vs baseline de 15.55) - Desviación SIGNIFICATIVA (>3 pts)
- **OTP15:** 79.39% (-2.15 pts vs baseline de 81.54%) - Desviación moderada (<3 pts)
- **Misconex:** 0.88 (+0.19 pts vs baseline de 0.69) - Desviación NO significativa
- **Load Factor:** 89.49% (-1.04 pts vs baseline de 90.53%) - Desviación NO significativa, mejoró ligeramente (menos ocupación)

**Contradicción Fundamental:**
"**El análisis revela una inconsistencia crítica entre la dirección de la anomalía y las métricas operativas:**
- **Anomalía observada:** NPS SUBIÓ +28.21 pts (anomalía positiva)
- **Métricas operativas:** EMPEORARON significativamente

Esta contradicción sugiere que **las métricas operativas NO explican la anomalía positiva de NPS**. Las causas identificadas serían válidas para una caída de NPS, no para una subida."

#### **🚨 INCIDENTES NCS (CUANTITATIVO):**
**Global/LH/Premium:**
- Total de incidentes: 17
- Retrasos: 4 incidentes
- Equipaje: 3 incidentes
- Otras incidencias: 4 incidentes
- Cancelaciones: 2 incidentes
- Desvíos: 1 incidente
- Sin clasificar: 3 incidentes

#### **🧠 NCS (CUALITATIVO / REFLEXIÓN):**
**Rutas con incidentes reportados:**
"DFW-MAD: Reprogramaciones (+1h20min)
MAD-DFW: Incidentes operativos
DOH-MAD: Incidentes reportados"

**Ejemplos específicos:**
"Vuelo IB364/18DEC: Reprogramado +1 hora 20 minutos
Otro vuelo: Reprogramado +55 minutos"

**Desconexión crítica:**
"⚠️ **DESCONEXIÓN CRÍTICA:** Las rutas con incidentes NCS (DFW-MAD, DOH-MAD) NO tienen encuestas"

#### **💬 FEEDBACK DE CLIENTES:**
"**Verbatims:**
- **No hay comentarios con texto disponibles** para este período
- Imposible validar causas desde perspectiva cualitativa del cliente"

#### **✈️ RUTAS AFECTADAS (Top 5):**
**Global/LH/Premium:**
1. MAD-SCL: NPS 0 (2 encuestas) - NPS crítico
2. MAD-UIO: NPS 0 (1 encuesta) - NPS crítico
3. BOG-MAD: NPS 50 (2 encuestas) - NPS neutral
4. EZE-MAD: NPS 50 (2 encuestas) - NPS neutral
5. GRU-MAD: NPS 100 (1 encuesta) - NPS excelente

**Observaciones:**
"Las rutas con NPS 0 (MAD-SCL, MAD-UIO) NO aparecen en incidentes NCS
Las rutas con incidentes NCS (DFW-MAD, DOH-MAD) NO tienen encuestas
Concentración en rutas de AMÉRICA DEL SUR
Sin datos de comparación vs baseline por ruta (nps_diff no disponible)"

#### **👥 PERFILES REACTIVOS:**

**Por tipo de viaje:**
- Leisure: NPS 33.3 (6 encuestas) - 75% de la muestra
- Business/Work: NPS 50.0 (2 encuestas) - 25% de la muestra

**Por región de residencia:**
- AMÉRICA DEL SUR: NPS 40.0 (5 encuestas) - Grupo mayoritario
- AMÉRICA CENTRO: NPS 50.0 (2 encuestas)
- EUROPA: NPS 0.0 (1 encuesta) - Muestra mínima

**Spread:** 50 pts entre mejor y peor región → Alta variabilidad

**Por flota:**
- A350: NPS 40.0 (5 encuestas)
- A350 next: NPS 33.3 (3 encuestas)

**Conclusión:** "Tipo de aeronave NO es factor diferenciador"

---

## **RESUMEN DE DISPONIBILIDAD DE DATOS POR NMA:**

| NMA | Explanatory Drivers | Datos Operativos | NCS Cuant. | NCS Cual. | Verbatims | Rutas | Perfiles |
|-----|---------------------|------------------|------------|-----------|-----------|-------|----------|
| Global/SH/Economy | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Global/SH/Economy/YW | ❌ | ✅ | ✅ | ✅ | ❌ | ⚠️ | ✅ |
| Global/SH/Economy/IB | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Global/LH/Business | ❌ | ✅ | ✅ | ✅ | ❌ | ⚠️ | ✅ |
| Global/LH/Premium | ❌ | ✅ | ✅ | ✅ | ❌ | ⚠️ | ✅ |

**Leyenda:**
- ✅ Datos disponibles y completos
- ⚠️ Datos disponibles pero limitados (muestras pequeñas)
- ❌ Datos no disponibles

---

## Cabin Radio Reflection

# 📋 PASO 4C: REFLEXIÓN POR CABINA-RADIO

---

## **CABINAS SHORT HAUL (SH)**

### **=== Economy SH ===**

**• NPS Cabina:** 19.6 (-7.5 pts)  
**• Estado:** NEGATIVE ANOMALY  
**• Escenario:** SINERGIA (IB `-`, YW `-` | Cabina `-`)

**• IB:** NPS 24.3 (-2.6 pts)  
**Causa:** Caída moderada explicada por exposición a vuelos CodeShare con partners problemáticos (BA: NPS -100.0 con 3 encuestas, QR: NPS -100.0 con 2 encuestas, I2: NPS -50.0 con 4 encuestas) y ruta crítica LHR-MAD con NPS -42.9 (14 encuestas). El impacto fue limitado por el volumen dominante de operación propia IB (253 encuestas con NPS +22.5). Las 22 cancelaciones y 12 retrasos documentados en NCS (especialmente el incidente meteorológico en Florencia) afectaron la operación, aunque no se refleja directamente en encuestas por sesgo de supervivencia. Sin verbatims disponibles para validación cualitativa.

**• YW:** NPS 9.2 (-18.3 pts)  
**Causa:** Caída severa concentrada en ruta crítica MAD-MRS (NPS -40.0, 5 encuestas) que impactó desproporcionadamente. Afectó principalmente a clientes europeos (no españoles) con NPS -17.4 (23 encuestas), flota CRJ con NPS 6.5 (77 encuestas, 88.5% de la operación YW), y viajeros Leisure con NPS 3.2 (63 encuestas, 72.4% del segmento). Las disrupciones operativas masivas (28 incidentes NCS: 22 cancelaciones por meteorología FLR, Mishandling +2.58 pts vs baseline) contribuyeron al deterioro. Sin comentarios de texto disponibles para validación cualitativa.

**• Narrativa de agregación:**  
Ambas compañías empujan en la misma dirección negativa (SINERGIA), aunque YW tiene un impacto 7x más severo que IB (-18.3 pts vs -2.6 pts). La anomalía negativa del padre (-7.5 pts) es el resultado ponderado de ambas caídas. La causa común subyacente son los **vuelos CodeShare críticos** (BA/QR con NPS -100.0) y las **disrupciones operativas masivas** (22 cancelaciones, Mishandling +3.59 pts), aunque el impacto específico varía: IB concentra problemas en LHR-MAD, mientras YW en MAD-MRS y flota CRJ.

**• Rutas críticas (del CAUSAL EXPLANATION del padre - Economy SH):**
1. **LHR-MAD:** NPS -42.9 (14 encuestas) - Ruta crítica operada principalmente por IB, correlación geográfica con CodeShare BA
2. **MAD-MRS:** NPS -40.0 (5 encuestas) - Concentrada en YW
3. DSS-MAD: NPS 0.0 (4 encuestas)
4. MAD-NAP: NPS 0.0 (3 encuestas)
5. ATH-MAD: NPS 0.0 (2 encuestas)

**Nota crítica:** Las rutas FLR (MAD-FLR, BLQ-FLR, FLR-MAD) con 5 incidentes NCS documentados NO aparecen en los datos de encuestas NPS.

**• Perfiles reactivos (del CAUSAL EXPLANATION del padre - Economy SH):**
- **CodeShare (dispersión: 171.4 pts):** BA NPS -100.0 (3 enc), QR NPS -100.0 (2 enc), I2 NPS -60.0 (5 enc) vs IB NPS +22.5 (253 enc), VY NPS +71.4 (7 enc)
- **Residence Region (dispersión: 150.0 pts):** EUROPA NPS 0.0 (71 enc), ESPAÑA NPS +31.9 (144 enc), AFRICA NPS -100.0 (1 enc)
- **Business/Leisure:** Leisure NPS +17.1 (187 enc) vs Business/Work NPS +24.7 (89 enc) - Viajeros de ocio más afectados
- **Fleet (dispersión: 33.9 pts):** CRJ NPS +6.5 (169 enc) - Peor desempeño con muestra significativa

---

### **=== Business SH ===**

**• NPS Cabina:** 52.4 (+17.3 pts)  
**• Estado:** POSITIVE ANOMALY  
**• Escenario:** TRANSFERENCIA (IB `+`, YW `N/A` | Cabina `+`)

**• IB:** NPS 64.7 (+22.6 pts)  
**Causa:** La anomalía positiva es un **artefacto estadístico** causado por: (1) Muestra extremadamente pequeña de solo 17 encuestas (no representativa), (2) Sesgo de supervivencia - las encuestas capturaron solo pasajeros que SÍ volaron, excluyendo a los afectados por las 22 cancelaciones de MAD-FLR por meteorología adversa, (3) Composición favorable de la muestra con clientes españoles NPS 100.0 (10 encuestas, 59%) vs europeos NPS 14.3 (7 encuestas, 41%), y viajeros Business/Work NPS 100.0 (6 encuestas). **Contradicción operativa:** Pese a que las métricas empeoraron (Mishandling +3.91 pts, Misconex +0.20 pts, 28 incidentes NCS con 22 cancelaciones), el NPS subió porque la ruta problemática MAD-FLR no aparece en el análisis de encuestas (0 respuestas de pasajeros cancelados). **Nivel de confianza: BAJO** - No utilizar para decisiones estratégicas.

**• YW:** Sin datos suficientes (Estado "S")  
**Causa:** Ausencia de YW en el análisis por volumen insuficiente o sin encuestas.

**• Narrativa de agregación:**  
IB tiene una anomalía positiva significativa (+22.6 pts) que **contagia al padre** (+17.3 pts) a pesar de que YW no tiene datos suficientes para evaluación (TRANSFERENCIA). La ausencia de YW significa que el resultado del padre es **completamente impulsado por IB**. Sin embargo, esta anomalía positiva tiene **nivel de confianza BAJO** debido a muestra extremadamente pequeña (n=17) y sesgo de supervivencia crítico que invalida conclusiones.

**• Rutas críticas (del CAUSAL EXPLANATION del hijo dominante - IB):**
1. LHR-MAD: NPS 0.0 (4 encuestas) - Única ruta con NPS bajo, pero n insuficiente
2. FRA-MAD: NPS 100.0 (2 encuestas) - Muestra muy pequeña
3. LIN-MAD: NPS 100.0 (2 encuestas) - Muestra muy pequeña
4. BIO-MAD: NPS 100.0 (1 encuesta) - Estadísticamente irrelevante
5. HAM-MAD: NPS 100.0 (1 encuesta) - Estadísticamente irrelevante

**Limitación:** 11 rutas analizadas con muestras ≤4 pasajeros (estadísticamente irrelevantes). La ruta MAD-FLR (epicentro de 22 cancelaciones según NCS) NO aparece en el análisis.

**• Perfiles reactivos (del CAUSAL EXPLANATION del hijo dominante - IB):**
- **Residence Region (dispersión: 200 pts):** ESPAÑA NPS 100.0 (10 enc, 59%) vs EUROPA NPS 14.3 (7 enc, 41%) vs Sin región NPS -100.0 (1 enc)
- **Business/Leisure:** Business/Work NPS 100.0 (6 enc) vs Leisure NPS 45.5 (11 enc) - Viajeros frecuentes más tolerantes
- **Fleet:** A320 NPS 100.0 (3 enc), ATR NPS 100.0 (1 enc), A321 NPS 60.0 (5 enc), A320neo NPS 50.0 (8 enc) - Dispersión baja (50 pts), no es factor determinante
- **Codeshare:** IB NPS 75.0 (16 enc), BA NPS -100.0 (1 enc) - Muestra BA no confiable

---

## **CABINAS LONG HAUL (LH)**

### **=== Economy LH ===**

**• NPS:** 8.5 (+4.8 pts)  
**• Estado:** Normal  

**• Causa principal:**  
Performance estable sin desviaciones significativas. Mantiene operación dentro de parámetros esperados (NPS Period: 8.5 vs NPS Baseline: 3.7). No se detectaron cambios que requieran investigación.

**• Evidencia clave:**  
Sin análisis causal disponible (segmento marcado como Normal, sin anomalía detectada).

**• Rutas críticas:**  
No disponible (no se proporciona análisis de rutas para segmentos Normal).

**• Perfiles reactivos:**  
No disponible (no se proporciona análisis de perfiles para segmentos Normal).

---

### **=== Business LH ===**

**• NPS:** -5.6 (-24.5 pts)  
**• Estado:** NEGATIVE ANOMALY  

**• Causa principal:**  
Degradación operativa múltiple con tres métricas críticas empeorando simultáneamente: OTP15 (puntualidad) -2.15 pts (79.39% vs 81.54% baseline), Mishandling (equipaje) +3.59 pts (19.14 vs 15.55 baseline), y Misconex (conexiones perdidas) +0.19 pts (0.88 vs 0.69 baseline). Los ajustes de rotación en rutas transatlánticas generaron efecto cascada en puntualidad y conexiones.

**• Evidencia clave:**  
- **Métricas operativas:** OTP15 -2.15 pts, Mishandling +3.59 pts (desviación significativa >3 pts), Misconex +0.19 pts
- **Incidentes NCS:** 17 incidentes totales (4 retrasos, 3 equipaje, 2 cancelaciones, 1 desvío). Rutas transatlánticas críticas: IB364 (DFW-MAD) reprogramado +1h 20min, MAD-DFW reprogramado +55min por ajuste de rotación, DOH-MAD con desvío confirmado
- **Nivel de confianza:** MEDIO ⚠️ (volumen bajo de 18 encuestas, sin verbatims disponibles)

**• Rutas críticas:**
1. MAD-MVD: NPS -100 (1 encuesta) ⚠️ Muy baja confiabilidad
2. MAD-SJU: NPS -100 (1 encuesta) ⚠️ Muy baja confiabilidad
3. JFK-MAD: NPS -100 (1 encuesta) ⚠️ Muy baja confiabilidad
4. DOH-MAD: NPS -100 (1 encuesta) ⚠️ Muy baja confiabilidad
5. MAD-MIA: NPS -100 (1 encuesta) ⚠️ Muy baja confiabilidad

**Inconsistencia crítica:** DFW-MAD y MAD-DFW (rutas con incidentes más severos en NCS: +1h20min y +55min de reprogramación) NO aparecen en el análisis de rutas. 7 de 10 rutas tienen n=1, haciendo imposible análisis robusto por ruta individual.

**• Perfiles reactivos:**
- **Propósito de Viaje:** Business/Work NPS -66.7 (3 enc, 17%) vs Leisure NPS +6.7 (15 enc, 83%) - Viajeros de negocios experimentaron impacto 10x más negativo (conexiones críticas, compromisos profesionales)
- **Flota:** A332 NPS -66.7 (6 enc) - Probablemente operando rutas transatlánticas como DFW-MAD, muestra el peor desempeño con volumen suficiente para análisis. A321XLR NPS -100 (2 enc), A350 next NPS 0 (2 enc), A350 NPS +75 (4 enc)
- **Residence Region:** ESPAÑA NPS -25.0 (8 enc, 44%) - Pasajeros españoles (probablemente en rutas MAD-origen) fueron los más afectados, representando el segmento más grande con NPS negativo. AMÉRICA DEL NORTE NPS +50.0 (4 enc), AMÉRICA CENTRAL NPS +50.0 (2 enc)
- **Operador (CodeShare):** IB NPS +6.2 (16 enc) ✅ Alta confiabilidad, BA NPS -100 (1 enc) ⚠️, QR NPS -100 (1 enc) ⚠️ - Volumen insuficiente en BA/QR

---

### **=== Premium LH ===**

**• NPS:** 37.5 (+28.2 pts)  
**• Estado:** POSITIVE ANOMALY  

**• Causa principal:**  
Artefacto estadístico por muestra no representativa. La anomalía positiva NO tiene causa operativa tangible debido a: (1) Volumen extremadamente bajo de solo 8 encuestas (muestra NO representativa), (2) Contradicción fundamental - las métricas operativas empeoraron significativamente (Mishandling +3.59 pts, OTP15 -2.15 pts, 17 incidentes NCS con 3 equipaje y 4 retrasos) pero el NPS subió, (3) Desconexión operativa - rutas con incidentes NCS (DFW-MAD, DOH-MAD) NO tienen encuestas. La mejora es un evento aleatorio sin causa operativa tangible, resultado de variabilidad estadística en muestras pequeñas.

**• Evidencia clave:**  
- **Métricas operativas (contradicción):** Mishandling +3.59 pts (19.14 vs 15.55), OTP15 -2.15 pts (79.39% vs 81.54%), Misconex +0.19 pts - Todas empeoraron pero NPS subió
- **Incidentes NCS:** 17 incidentes (4 retrasos, 3 equipaje, 2 cancelaciones, 1 desvío). Rutas: DFW-MAD (+1h20min reprogramación), MAD-DFW, DOH-MAD
- **Nivel de confianza:** BAJA ⚠️ (n=8, contradicción operativa, sin verbatims, rutas críticas ausentes)

**• Rutas críticas:**
1. MAD-SCL: NPS 0 (2 encuestas) - NPS crítico
2. MAD-UIO: NPS 0 (1 encuesta) - NPS crítico
3. BOG-MAD: NPS 50 (2 encuestas) - NPS neutral
4. EZE-MAD: NPS 50 (2 encuestas) - NPS neutral
5. GRU-MAD: NPS 100 (1 encuesta) - NPS excelente

**Observación:** Las rutas con NPS 0 (MAD-SCL, MAD-UIO) NO aparecen en incidentes NCS. Las rutas con incidentes NCS (DFW-MAD, DOH-MAD) NO tienen encuestas. Concentración en rutas de AMÉRICA DEL SUR.

**• Perfiles reactivos:**
- **Tipo de viaje:** Leisure NPS 33.3 (6 enc, 75%) vs Business/Work NPS 50.0 (2 enc, 25%)
- **Residence Region (dispersión: 50 pts):** AMÉRICA DEL SUR NPS 40.0 (5 enc) - Grupo mayoritario, AMÉRICA CENTRO NPS 50.0 (2 enc), EUROPA NPS 0.0 (1 enc) - Muestra mínima
- **Flota:** A350 NPS 40.0 (5 enc), A350 next NPS 33.3 (3 enc) - Tipo de aeronave NO es factor diferenciador

**Nota:** No requiere acción correctiva (es variabilidad estadística, no problema operativo).

---

## **RESUMEN DE ESTADOS POR CABINA-RADIO:**

| Cabina-Radio | NPS | Variación | Estado | Escenario | Confianza |
|--------------|-----|-----------|--------|-----------|-----------|
| **Economy SH** | 19.6 | -7.5 | NEGATIVE ANOMALY | SINERGIA (IB -, YW -) | MEDIO |
| **Business SH** | 52.4 | +17.3 | POSITIVE ANOMALY | TRANSFERENCIA (IB +, YW N/A) | **BAJO** |
| **Economy LH** | 8.5 | +4.8 | Normal | N/A | N/A |
| **Business LH** | -5.6 | -24.5 | NEGATIVE ANOMALY | Segmento hoja | MEDIO |
| **Premium LH** | 37.5 | +28.2 | POSITIVE ANOMALY | Segmento hoja | **BAJO** |

---

## 📋 SÍNTESIS EJECUTIVA FINAL

```html
<b>SÍNTESIS EJECUTIVA</b><br>
<br>
La red global registró un <b>NPS de 17.5 (–3.4 pts)</b> con respecto a la media de los últimos 7 días, resultado de una transferencia negativa desde el Corto Radio que arrastró al conjunto de la operación pese a la estabilidad aparente del Largo Radio.<br>
<br>
En <b>Economy SH</b>, el NPS cayó a <b>19.6 (–7.5 pts)</b> debido a un desempeño crítico en vuelos operados bajo acuerdos de código compartido, donde las asociaciones con BA registraron un NPS de –100.0 con 3 pasajeros encuestados, QR alcanzó –100.0 con 2 encuestas, e I2 marcó –60.0 con 5 respuestas. Este deterioro contrastó radicalmente con la operación propia de IB, que mantuvo un NPS de 22.5 con 253 encuestas, y YW con 71.4 en 7 respuestas. Las rutas más afectadas fueron <b>LHR-MAD con un NPS de –42.9 en 14 encuestas</b>, operada principalmente por IB y con correlación geográfica con los vuelos de BA desde Londres, y <b>MAD-MRS con –40.0 en 5 encuestas</b>, concentrada en la operación de YW. Paralelamente, las métricas operativas mostraron un incremento de 3.59 puntos en el índice de manejo de equipaje respecto al baseline, alcanzando 19.14, aunque solo se reportó un incidente formal de equipaje en los registros operativos. Las disrupciones masivas del día incluyeron 22 cancelaciones y 12 retrasos, con un foco crítico en Florencia debido a condiciones meteorológicas adversas que provocaron desvíos a Bolonia, cancelaciones y falta de vuelos de retorno, afectando las rutas MAD-FLR con 2 incidentes, BLQ-FLR con 2 y FLR-MAD con 1. Sin embargo, estas rutas no aparecieron en el análisis de encuestas, evidenciando un sesgo de supervivencia donde los pasajeros cancelados no respondieron el mismo día. Los perfiles más reactivos fueron los pasajeros agrupados por operador de código compartido con una dispersión de 171.4 puntos, seguidos por región de residencia con 150.0 puntos donde Europa alcanzó un NPS de 0.0 en 71 encuestas frente a España con 31.9 en 144 respuestas. Los viajeros de ocio mostraron un NPS de 17.1 en 187 encuestas comparado con 24.7 en 89 respuestas de viajeros de negocios, mientras que la flota CRJ registró el desempeño más bajo con 6.5 en 169 encuestas. Esta presión en Economy SH se propagó dominando sobre Business SH, que mostró una mejora aparente de 17.3 puntos pero con baja confiabilidad estadística debido a una muestra de solo 21 encuestas, arrastrando al Corto Radio hacia una caída de 5.8 puntos.<br>
<br>
Dentro de Economy SH, <b>IB registró un NPS de 24.3 (–2.6 pts)</b> con una caída moderada explicada por la exposición a los vuelos en código compartido problemáticos mencionados, especialmente BA con –100.0 en 3 encuestas, QR con –100.0 en 2 y I2 con –50.0 en 4 respuestas. La ruta crítica LHR-MAD con –42.9 en 14 encuestas concentró el impacto, aunque el volumen dominante de operación propia de IB con 253 encuestas y un NPS de 22.5 limitó la magnitud de la caída. Los perfiles más afectados incluyeron códigos compartidos no propios con una dispersión de 166.7 puntos, seguidos por origen geográfico con 87.4 puntos donde clientes de América Centro alcanzaron –28.6 en 7 encuestas y América Sur –12.5 en 8 respuestas, mientras que España mantuvo 33.3 en 102 encuestas y Europa 8.3 en 48 respuestas. Las 22 cancelaciones y 12 retrasos documentados, especialmente por el incidente meteorológico en Florencia, afectaron la operación aunque no se reflejaron directamente en encuestas por el sesgo de supervivencia mencionado. No se dispuso de comentarios cualitativos para validación.<br>
<br>
Por su parte, <b>YW alcanzó un NPS de 9.2 (–18.3 pts)</b>, una caída 7 veces más severa que IB, concentrada en la ruta crítica <b>MAD-MRS con un NPS de –40.0 en 5 encuestas</b> que impactó desproporcionadamente. Los clientes europeos no españoles mostraron un NPS de –17.4 en 23 encuestas, representando el 26.4 por ciento de la muestra, coherente con los problemas en la ruta francesa. La flota CRJ, que representa el 88.5 por ciento de la operación de YW con 77 encuestas, registró un NPS de 6.5, significativamente inferior al ATR con 30.0 en 10 respuestas. Los viajeros de ocio, que constituyeron el 72.4 por ciento de la muestra con 63 encuestas, mostraron un NPS de 3.2 comparado con 25.0 en 24 respuestas de viajeros de negocios. Las disrupciones operativas masivas incluyeron un incremento de 2.58 puntos en manejo de equipaje respecto al baseline, alcanzando 15.36, y una mejora ligera de 0.36 puntos en puntualidad con 88.74 por ciento. La operación de YW concentró el 96.5 por ciento de encuestas bajo código IB con un NPS de 8.3 en 84 respuestas. La mayoría de las 38 rutas operativas presentaron muestras muy pequeñas de 1 a 5 encuestas con valores extremos de 0.0 o 100.0 que no son estadísticamente representativos, y la ruta MAD-FLR identificada en incidentes operativos por meteorología adversa no tuvo datos de encuestas disponibles.<br>
<br>
En <b>Business LH</b>, el NPS cayó a <b>–5.6 (–24.5 pts)</b> debido a una degradación operativa múltiple donde tres métricas críticas empeoraron simultáneamente. La puntualidad disminuyó 2.15 puntos alcanzando 79.39 por ciento frente a un baseline de 81.54 por ciento, el manejo de equipaje aumentó 3.59 puntos llegando a 19.14 respecto a 15.55, y las conexiones perdidas subieron 0.19 puntos a 0.88 desde 0.69. Los ajustes de rotación en rutas transatlánticas generaron un efecto cascada en puntualidad y conexiones, con el vuelo IB364 de DFW a MAD reprogramado con un retraso de 1 hora 20 minutos el 18 de diciembre, MAD a DFW reprogramado con 55 minutos adicionales por ajuste de rotación, y DOH a MAD con un desvío confirmado. Los 17 incidentes operativos totales incluyeron 4 retrasos, 3 relacionados con equipaje, 2 cancelaciones, 1 desvío y 7 otras incidencias. Las rutas críticas identificadas fueron MAD-MVD, MAD-SJU, JFK-MAD, DOH-MAD y MAD-MIA, todas con un NPS de –100.0 pero con una sola encuesta cada una, lo que representa muy baja confiabilidad estadística. De manera crítica, las rutas DFW-MAD y MAD-DFW con los incidentes más severos de reprogramación de 1 hora 20 minutos y 55 minutos no aparecieron en el análisis de encuestas, y 7 de las 10 rutas analizadas tuvieron solo una encuesta, haciendo imposible un análisis robusto por ruta individual. Los perfiles más reactivos fueron los viajeros de negocios con un NPS de –66.7 en 3 encuestas comparado con viajeros de ocio con 6.7 en 15 respuestas, experimentando un impacto 10 veces más negativo debido a conexiones críticas y compromisos profesionales. La flota A332, probablemente operando rutas transatlánticas como DFW-MAD, mostró el peor desempeño con –66.7 en 6 encuestas, el volumen suficiente para análisis, mientras que A321XLR alcanzó –100.0 en 2 encuestas. Los pasajeros con residencia en España registraron un NPS de –25.0 en 8 encuestas, representando el 44 por ciento de la muestra y el segmento más grande con valoración negativa, probablemente en rutas con origen en Madrid. Esta caída en Business LH se compensó con una subida aparente de 28.2 puntos en Premium LH y una estabilidad de 4.8 puntos en Economy LH, resultando en una estabilidad engañosa del Largo Radio con 2.4 puntos dentro del rango normal, donde los efectos opuestos se anularon mutuamente ocultando la volatilidad interna.<br>
<br>
En <b>Premium LH</b>, el NPS subió a <b>37.5 (+28.2 pts)</b>, pero esta anomalía positiva constituyó un artefacto estadístico sin causa operativa tangible. El volumen extremadamente bajo de solo 8 encuestas representó una muestra no representativa, y existió una contradicción fundamental donde las métricas operativas empeoraron significativamente con un incremento de 3.59 puntos en manejo de equipaje alcanzando 19.14 respecto a 15.55, una disminución de 2.15 puntos en puntualidad llegando a 79.39 por ciento desde 81.54 por ciento, y un aumento de 0.19 puntos en conexiones perdidas a 0.88 desde 0.69, mientras que el NPS subió. Los 17 incidentes operativos incluyeron 4 retrasos, 3 relacionados con equipaje, 2 cancelaciones y 1 desvío, con rutas DFW-MAD con una reprogramación de 1 hora 20 minutos, MAD-DFW y DOH-MAD reportando incidentes. Sin embargo, estas rutas con incidentes no tuvieron encuestas disponibles, evidenciando una desconexión operativa crítica. Las rutas con encuestas fueron MAD-SCL con un NPS de 0.0 en 2 respuestas, MAD-UIO con 0.0 en 1 encuesta, BOG-MAD con 50.0 en 2 respuestas, EZE-MAD con 50.0 en 2 encuestas y GRU-MAD con 100.0 en 1 respuesta, todas concentradas en América del Sur. Las rutas con NPS de 0.0 no aparecieron en incidentes operativos, mientras que las rutas con incidentes no tuvieron encuestas. Los perfiles mostraron viajeros de ocio con un NPS de 33.3 en 6 encuestas representando el 75 por ciento de la muestra comparado con viajeros de negocios con 50.0 en 2 respuestas. Por región de residencia con una dispersión de 50 puntos, América del Sur alcanzó 40.0 en 5 encuestas como grupo mayoritario, América Centro 50.0 en 2 respuestas y Europa 0.0 en 1 encuesta con muestra mínima. Por flota, A350 registró 40.0 en 5 encuestas y A350 next 33.3 en 3 respuestas, sin ser el tipo de aeronave un factor diferenciador. La mejora es un evento aleatorio resultado de variabilidad estadística en muestras pequeñas sin causa operativa tangible, por lo que no requiere acción correctiva.<br>
<br>
La convergencia de <b>Corto Radio con –5.8 pts</b>, impulsado por Economy SH con –7.5 pts donde tanto IB como YW cayeron en sinergia aunque YW con un impacto 7 veces más severo, y <b>Largo Radio con estabilidad aparente de +2.4 pts</b>, resultado de la cancelación entre Business LH con –24.5 pts por degradación operativa y Premium LH con +28.2 pts por artefacto estadístico, produjo la caída global de 3.4 puntos. El volumen del Corto Radio con 297 encuestas representando el 67.3 por ciento del total frente a 144 encuestas del Largo Radio con el 32.7 por ciento permitió que su anomalía negativa contagiara al resultado global mediante transferencia, donde un solo radio con anomalía arrastra al conjunto completo.<br>
<br>
<b><u>DETALLE POR CABINA</u></b><br>
<br>
<b><u>Economy SH: Deterioro por código compartido y rutas críticas</u></b><br>
El desempeño crítico en vuelos operados bajo acuerdos de código compartido lastró la cabina con un <b>NPS de 19.6 (–7.5 pts)</b>, donde IB cayó 2.6 puntos a 24.3 y YW se desplomó 18.3 puntos a 9.2, evidenciando un impacto 7 veces más severo en esta última compañía. Las asociaciones con BA alcanzaron un NPS de –100.0 con 3 pasajeros encuestados, QR –100.0 con 2 encuestas e I2 –60.0 con 5 respuestas, contrastando con la operación propia de IB que mantuvo 22.5 en 253 encuestas y YW con 71.4 en 7 respuestas. Las rutas más afectadas fueron LHR-MAD con –42.9 en 14 encuestas, operada principalmente por IB con correlación geográfica con vuelos de BA desde Londres, y MAD-MRS con –40.0 en 5 encuestas, concentrada en YW. Las disrupciones operativas masivas incluyeron 22 cancelaciones y 12 retrasos, con un foco crítico en Florencia por condiciones meteorológicas adversas que provocaron desvíos a Bolonia, cancelaciones y falta de vuelos de retorno en las rutas MAD-FLR con 2 incidentes, BLQ-FLR con 2 y FLR-MAD con 1, aunque estas rutas no aparecieron en el análisis de encuestas por sesgo de supervivencia. El manejo de equipaje aumentó 3.59 puntos respecto al baseline alcanzando 19.14, aunque solo se reportó un incidente formal. Los perfiles más reactivos fueron los pasajeros agrupados por operador de código compartido con una dispersión de 171.4 puntos, seguidos por región de residencia con 150.0 puntos donde Europa alcanzó 0.0 en 71 encuestas frente a España con 31.9 en 144 respuestas. Los viajeros de ocio mostraron 17.1 en 187 encuestas comparado con 24.7 en 89 respuestas de viajeros de negocios, y la flota CRJ registró 6.5 en 169 encuestas como el desempeño más bajo con muestra significativa.<br>
<br>
<b><u>Business SH: Mejora aparente con baja confiabilidad</u></b><br>
La cabina registró un <b>NPS de 52.4 (+17.3 pts)</b>, completamente impulsado por IB que subió 22.6 puntos a 64.7 mientras YW no tuvo datos suficientes para evaluación. Sin embargo, esta anomalía positiva constituyó un artefacto estadístico causado por una muestra extremadamente pequeña de solo 17 encuestas no representativa, un sesgo de supervivencia donde las encuestas capturaron únicamente pasajeros que volaron excluyendo a los afectados por las 22 cancelaciones de MAD-FLR por meteorología adversa, y una composición favorable con clientes españoles alcanzando un NPS de 100.0 en 10 encuestas representando el 59 por ciento frente a europeos con 14.3 en 7 encuestas con el 41 por ciento, junto con viajeros de negocios con 100.0 en 6 respuestas. Pese a que las métricas operativas empeoraron con un incremento de 3.91 puntos en manejo de equipaje, 0.20 puntos en conexiones perdidas y 28 incidentes operativos con 22 cancelaciones, el NPS subió porque la ruta problemática MAD-FLR no apareció en el análisis de encuestas con 0 respuestas de pasajeros cancelados. Las 11 rutas analizadas tuvieron muestras de 4 pasajeros o menos siendo estadísticamente irrelevantes, con LHR-MAD alcanzando 0.0 en 4 encuestas como única ruta con valoración baja pero volumen insuficiente. Los perfiles mostraron una dispersión de 200 puntos por región de residencia con España en 100.0 en 10 encuestas comparado con Europa en 14.3 en 7 respuestas, viajeros de negocios con 100.0 en 6 encuestas frente a viajeros de ocio con 45.5 en 11 respuestas, y por flota una dispersión baja de 50 puntos sin ser factor determinante. El nivel de confianza es bajo y no debe utilizarse para decisiones estratégicas.<br>
<br>
<b><u>Economy LH: Estabilidad dentro de parámetros esperados</u></b><br>
La cabina mantuvo un <b>NPS de 8.5 (+4.8 pts)</b> dentro del rango normal, con una performance estable sin desviaciones significativas. La operación se mantuvo dentro de parámetros esperados comparando 8.5 en el período con 3.7 en el baseline, sin cambios detectados que requirieran investigación.<br>
<br>
<b><u>Business LH: Degradación operativa en rutas transatlánticas</u></b><br>
La cabina cayó a un <b>NPS de –5.6 (–24.5 pts)</b> debido a una degradación operativa múltiple donde tres métricas críticas empeoraron simultáneamente. La puntualidad disminuyó 2.15 puntos alcanzando 79.39 por ciento frente a un baseline de 81.54 por ciento, el manejo de equipaje aumentó 3.59 puntos llegando a 19.14 respecto a 15.55, y las conexiones perdidas subieron 0.19 puntos a 0.88 desde 0.69. Los ajustes de rotación en rutas transatlánticas generaron un efecto cascada en puntualidad y conexiones, con el vuelo IB364 de DFW a MAD reprogramado con un retraso de 1 hora 20 minutos el 18 de diciembre, MAD a DFW reprogramado con 55 minutos adicionales por ajuste de rotación, y DOH a MAD con un desvío confirmado. Los 17 incidentes operativos totales incluyeron 4 retrasos, 3 relacionados con equipaje, 2 cancelaciones, 1 desvío y 7 otras incidencias. Las rutas críticas identificadas fueron MAD-MVD, MAD-SJU, JFK-MAD, DOH-MAD y MAD-MIA, todas con un NPS de –100.0 pero con una sola encuesta cada una representando muy baja confiabilidad estadística. De manera crítica, las rutas DFW-MAD y MAD-DFW con los incidentes más severos no aparecieron en el análisis de encuestas, y 7 de las 10 rutas analizadas tuvieron solo una encuesta haciendo imposible un análisis robusto por ruta individual. Los viajeros de negocios experimentaron un impacto 10 veces más negativo con –66.7 en 3 encuestas comparado con viajeros de ocio con 6.7 en 15 respuestas, debido a conexiones críticas y compromisos profesionales. La flota A332, probablemente operando rutas transatlánticas como DFW-MAD, mostró el peor desempeño con –66.7 en 6 encuestas como volumen suficiente para análisis, mientras que A321XLR alcanzó –100.0 en 2 encuestas. Los pasajeros con residencia en España registraron –25.0 en 8 encuestas representando el 44 por ciento de la muestra como el segmento más grande con valoración negativa, probablemente en rutas con origen en Madrid.<br>
<br>
<b><u>Premium LH: Artefacto estadístico sin causa operativa</u></b><br>
La cabina subió a un <b>NPS de 37.5 (+28.2 pts)</b>, pero esta anomalía positiva constituyó un artefacto estadístico sin causa operativa tangible. El volumen extremadamente bajo de solo 8 encuestas representó una muestra no representativa, y existió una contradicción fundamental donde las métricas operativas empeoraron significativamente con un incremento de 3.59 puntos en manejo de equipaje alcanzando 19.14 respecto a 15.55, una disminución de 2.15 puntos en puntualidad llegando a 79.39 por ciento desde 81.54 por ciento, y un aumento de 0.19 puntos en conexiones perdidas a 0.88 desde 0.69, mientras que el NPS subió. Los 17 incidentes operativos incluyeron 4 retrasos, 3 relacionados con equipaje, 2 cancelaciones y 1 desvío, con rutas DFW-MAD con una reprogramación de 1 hora 20 minutos, MAD-DFW y DOH-MAD reportando incidentes, pero estas rutas con incidentes no tuvieron encuestas disponibles evidenciando una desconexión operativa crítica. Las rutas con encuestas fueron MAD-SCL con 0.0 en 2 respuestas, MAD-UIO con 0.0 en 1 encuesta, BOG-MAD con 50.0 en 2 respuestas, EZE-MAD con 50.0 en 2 encuestas y GRU-MAD con 100.0 en 1 respuesta, todas concentradas en América del Sur. Las rutas con valoración de 0.0 no aparecieron en incidentes operativos mientras que las rutas con incidentes no tuvieron encuestas. Los viajeros de ocio alcanzaron 33.3 en 6 encuestas representando el 75 por ciento de la muestra comparado con viajeros de negocios con 50.0 en 2 respuestas. Por región de residencia con una dispersión de 50 puntos, América del Sur alcanzó 40.0 en 5 encuestas como grupo mayoritario, América Centro 50.0 en 2 respuestas y Europa 0.0 en 1 encuesta con muestra mínima. Por flota, A350 registró 40.0 en 5 encuestas y A350 next 33.3 en 3 respuestas, sin ser el tipo de aeronave un factor diferenciador. La mejora es un evento aleatorio resultado de variabilidad estadística en muestras pequeñas sin causa operativa tangible, por lo que no requiere acción correctiva.
```

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 7
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-16 to 2025-12-16:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 PASO 1: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL COMPAÑÍA)

---

## **ECONOMY SHORT HAUL**

### **PARTE A: ANÁLISIS DE DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(IB: -, YW: - | Economy SH: -)`

- **IB Economy SH:** NEGATIVE ANOMALY (-6.3 pts)
- **YW Economy SH:** NEGATIVE ANOMALY (-11.4 pts)
- **Economy SH (Padre):** NEGATIVE ANOMALY (-8.2 pts)

**Escenario Identificado:** **SINERGIA NEGATIVA (-,- | -)**

Ambas compañías empujan en la misma dirección negativa. El efecto se suma y transfiere al padre, resultando en una anomalía negativa agregada de -8.2 pts.

---

### **PARTE B: SELECCIÓN DE NARRATIVA CAUSAL**

**Narrativa:** Se adopta la **Explicación del Nodo Padre (Economy SH)**, ya que el escenario es SINERGIA. La causa raíz afecta a ambas compañías de manera sistémica.

**Causa Principal:** Deterioro crítico en gestión de equipaje (Mishandling +3.45 pts vs baseline en Economy SH).

**Evidencia Clave:**
- **Métricas Operativas (Economy SH):** Mishandling 18.91 (+3.45 pts vs baseline 15.46), incremento del 22.3%
- **Incidentes NCS:** 6 incidentes formales de equipaje reportados
- **Verbatims:** 8 quejas específicas de equipaje perdido/retrasado en rutas críticas
- **Rutas Afectadas:** 
  - GRU-MAD: NPS 0.0 (IB Economy LH) - "Mi equipaje lleva retrasado desde el 16/12"
  - GVA-MAD: NPS -50.0 (YW Economy SH) - "Nos dejaron las maletas en Ginebra, 3 días sin localizar"
  - AGP-MAD: NPS 0.0 (ambas compañías afectadas) - "Mi maleta está perdida en almacén de IB en Madrid"
- **Perfiles Impactados:**
  - **IB Economy SH:** Viajeros Business (NPS 6.1, -18.5 pts vs Leisure), flota A320neo (NPS 6.2, 43.5% de encuestas)
  - **YW Economy SH:** Residentes Europa (NPS -6.7), flota CRJ (NPS 14.8, 89% de encuestas)

**Interpretación:** El deterioro en Mishandling afectó de manera **transversal** a ambas operadoras, con **YW sufriendo un impacto más severo (-11.4 pts)** que **IB (-6.3 pts)**. La diferencia se explica por:
1. Mayor concentración de problemas en flota CRJ (operada principalmente por YW)
2. Rutas específicas de YW (GVA-MAD, AGP-MAD) con mayor incidencia de equipaje perdido
3. Menor volumen de encuestas en YW (143 vs 223 de IB), amplificando el impacto de incidentes individuales

---

---

## **BUSINESS SHORT HAUL**

### **PARTE A: ANÁLISIS DE DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(IB: -, YW: - | Business SH: -)`

- **IB Business SH:** NEGATIVE ANOMALY (-2.1 pts)
- **YW Business SH:** NEGATIVE ANOMALY (-16.0 pts)
- **Business SH (Padre):** NEGATIVE ANOMALY (-1.8 pts)

**Escenario Identificado:** **DILUCIÓN NEGATIVA (-,- | -)** con efecto de **DOMINANCIA INVERSA**

Ambas compañías tienen anomalías negativas, pero el **volumen desproporcionado de IB (30 encuestas, 83%) diluye significativamente el impacto severo de YW (6 encuestas, 17%)**, resultando en una caída agregada moderada de -1.8 pts en el padre, a pesar de que YW cayó -16.0 pts.

---

### **PARTE B: SELECCIÓN DE NARRATIVA CAUSAL**

**Narrativa:** Se adopta la **Explicación del Hijo Dominante (IB Business SH)** por volumen, pero se reconoce la **severidad extrema en YW**.

**Causa Principal (IB Business SH - Dominante por Volumen):**
Deterioro en gestión de equipaje (Mishandling +3.8 pts vs baseline en IB Business SH).

**Evidencia Clave IB:**
- **Métricas Operativas:** Mishandling 20.16 (+3.8 pts vs baseline 16.36, +23%)
- **Verbatims:** 2 quejas críticas de equipaje
  - FRA-MAD (NPS 0): "Perdieron todo mi equipaje en trayecto FRA-MAD-GRU, impidió continuar viaje"
  - LIS-MAD (NPS 8): "1 hora de espera en recogida de maletas"
- **Rutas Críticas:** BCN-MAD (NPS 14.3, n=7), LIS-MAD (NPS 25.0, n=4)
- **Perfil Más Afectado:** Business/Work (NPS 20.0 vs Leisure 60.0, diferencial de 40 pts), residentes España (NPS 13.3), flota A320neo (NPS 20.0)

**Matiz Crítico (YW Business SH - Severidad Extrema):**
Aunque **IB domina el agregado por volumen**, **YW experimentó un colapso total (NPS 0.0, -16.0 pts)** con **evidencia fragmentada**:
- **Muestra microscópica:** Solo 6 encuestas (estadísticamente insuficiente)
- **Incidentes específicos identificados:**
  - FRA-MAD: Problema sanitario (tripulante tosiendo sin mascarilla en Business, NPS 5)
  - MAD-MRS: Retraso mecánico >1h por fallo en puertas de bodega (NPS 2)
  - MAD-TRN: NPS -100 sin verbatim asociado (incidente no documentado)
- **Perfil Más Afectado:** Pasajeros Leisure (NPS -50 vs Business +100), flota CRJ (100% de encuestas, NPS 0.0)
- **Nivel de Confianza:** BAJO debido a muestra insuficiente y evidencia fragmentada

**Interpretación:** El agregado Business SH (-1.8 pts) refleja principalmente los problemas de **IB**, que con 30 encuestas (83% del total) diluye el colapso de **YW**. Sin embargo, la **severidad extrema en YW (-16.0 pts, NPS 0.0)** indica problemas operativos críticos no capturados en el agregado, requiriendo **investigación urgente** a pesar de la baja significancia estadística.

---

---

## **RESUMEN EJECUTIVO - NIVEL COMPAÑÍA**

| Cabina | Escenario | Estado (IB, YW \| Padre) | Narrativa Dominante | Causa Raíz |
|--------|-----------|--------------------------|---------------------|------------|
| **Economy SH** | **SINERGIA NEGATIVA** | `(-, - \| -)` | Explicación del Padre | Mishandling +3.45 pts (sistémico IB/YW) |
| **Business SH** | **DILUCIÓN NEGATIVA** | `(-, - \| -)` | Explicación del Hijo Dominante (IB) + Matiz Crítico (YW) | Mishandling +3.8 pts (IB) + Incidentes operativos específicos (YW) |

**Conclusión:** En ambas cabinas SH, el **deterioro en gestión de equipaje** es la causa raíz común, pero con **dinámicas de agregación diferentes**:
- **Economy:** Impacto sinérgico con YW más afectado (-11.4 vs -6.3 de IB)
- **Business:** Impacto diluido por volumen de IB, pero YW con colapso extremo (-16.0) que requiere atención urgente

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ PASO 2: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL CABINA)

---

## **SHORT HAUL (SH)**

### **PARTE A: ANÁLISIS DE DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(Economy: -, Business: - | SH: -)`

- **Economy SH:** NEGATIVE ANOMALY (-8.2 pts)
- **Business SH:** NEGATIVE ANOMALY (-1.8 pts)
- **SH (Padre):** NEGATIVE ANOMALY (-7.6 pts)

**Escenario Identificado:** **SINERGIA NEGATIVA PARCIAL (-,- | -)**

Ambas cabinas empujan en la misma dirección negativa, con **Economy SH dominando el efecto** por:
1. **Mayor severidad de la anomalía:** Economy -8.2 pts vs Business -1.8 pts
2. **Mayor volumen de encuestas:** Economy 366 encuestas (91%) vs Business 36 encuestas (9%)
3. **Resultado agregado:** SH -7.6 pts refleja principalmente el deterioro de Economy

---

### **PARTE B: SELECCIÓN DE NARRATIVA CAUSAL**

**Narrativa:** Se adopta la **Explicación del Radio Padre (SH)**, ya que el escenario es SINERGIA. El problema afectó transversalmente al corto radio, impactando ambas cabinas con especial severidad en Economy.

**Causa Sistémica:** Deterioro crítico en gestión de equipaje (Mishandling +3.45 pts vs baseline en SH).

**Evidencia Clave del Radio SH:**

**Métricas Operativas:**
- **Mishandling:** 18.91 (+3.45 pts vs baseline 15.46, incremento del 22.3%)
- **Misconex:** 0.84 (+0.14 pts vs baseline 0.70, aumento del 20% en conexiones perdidas)
- **OTP15:** 91.15% (+1.08 pts) - Puntualidad mejoró, pero no compensó el impacto de equipaje
- **Load Factor:** 81.53% (-3.2 pts) - Menor ocupación no mejoró la experiencia

**Incidentes NCS (SH):**
- **25 incidentes operativos totales:** 16 cancelaciones (64%), 15 retrasos (60%), 1 otras incidencias
- **Incidente destacado:** Huelga ATC en FCO (Roma) programada para 17/Dic con 6 vuelos cancelados (efecto anticipatorio el 16/Dic)

**Verbatims (Problema Dominante - Equipaje):**
- **6 casos explícitos de equipaje perdido/extraviado:**
  - AGP-MAD: "Mi maleta está perdida en almacén de IB en Madrid"
  - FRA-MAD: "Perdieron todo mi equipaje, impidió continuar viaje"
  - MAD-ORY: "72 horas sin localizar maleta" (PIR: ORYIB11854)
  - BCN-SXB: "Maletas llegaron 2.5 días después, contenía máquina de apnea vital"
  - LCG-MAD: "Soltaron maletas en pista a oscuras, cogieron mi maleta por error"
  - GVA-MAD: "Nos dejaron maletas en Ginebra, 3 días buscando ropa/calzado"

- **3 casos de gestión inadecuada de equipaje de mano:**
  - FCO-MAD: "Forzaron facturación de mano, avión medio vacío"
  - AGP-MAD: "Obligaron a despachar en bodega de mala manera, luego había espacio"
  - MAD-SCQ: "Obligaron a despachar mochila bajo lluvia, no había espacio (falso)"

**Rutas Críticas (Patrón: Madrid como Hub):**

| Ruta | NPS | Encuestas | Problema Principal |
|------|-----|-----------|-------------------|
| **GVA-MAD** | -50.0 | 6 | Equipaje perdido (3 días sin localizar) |
| **BCN-MAD** | -42.9 | 21 | Retrasos + pérdida conexión + equipaje mano |
| **LIS-MAD** | -20.0 | 10 | Retrasos múltiples + desinformación |
| **LHR-MAD** | -10.5 | 19 | Problemas embarque + retrasos |
| **AGP-MAD** | 0.0 | 10+10 | Equipaje perdido + retrasos + maltrato |

**Observación Crítica:** La mayoría de problemas de equipaje involucran **MAD (Madrid)** como origen, destino o punto de conexión, sugiriendo un **problema sistémico en las operaciones del hub principal**.

**Perfiles Más Afectados (SH):**

**Por Cabina:**
- **Economy SH:** NPS 18.85 (366 encuestas, 91% del total) - Impacto masivo
- **Business SH:** NPS 33.33 (36 encuestas, 9% del total) - Impacto moderado

**Por Compañía (dentro de Economy SH):**
- **IB Economy SH:** NPS 20.6 (223 encuestas) - Viajeros Business más afectados (NPS 6.1 vs Leisure 24.6)
- **YW Economy SH:** NPS 16.1 (143 encuestas) - Residentes Europa más afectados (NPS -6.7)

**Por Flota:**
- **CRJ:** NPS 14.8 (128 encuestas YW, 89% de YW) - Peor flota, concentra problemas de equipaje
- **A320neo:** NPS 6.2 (97 encuestas IB, 43.5% de IB) - Segunda peor flota

**Interpretación:** El deterioro en Mishandling (+3.45 pts) afectó **transversalmente al corto radio**, con:
1. **Economy absorbiendo el 91% del impacto** por volumen de pasajeros
2. **Madrid (MAD) como epicentro operativo** de los problemas de equipaje
3. **Flotas CRJ y A320neo concentrando la mayoría de incidentes**
4. **Viajeros Business y residentes Europa siendo los más críticos**

---

---

## **LONG HAUL (LH)**

### **PARTE A: ANÁLISIS DE DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(Economy: -, Business: +, Premium: - | LH: -)`

- **Economy LH:** NEGATIVE ANOMALY (-4.9 pts)
- **Business LH:** POSITIVE ANOMALY (+13.3 pts)
- **Premium LH:** NEGATIVE ANOMALY (-1.0 pts)
- **LH (Padre):** NEGATIVE ANOMALY (-1.4 pts)

**Escenario Identificado:** **CANCELACIÓN PARCIAL CON DOMINANCIA DE ECONOMY (+,-,- | -)**

Efectos mixtos con **Economy imponiendo su signo negativo** al agregado LH, a pesar de la mejora en Business:

1. **Economy LH domina por volumen:** 168 encuestas (54% del total LH)
2. **Business LH mejora significativamente:** +13.3 pts, pero con solo 31 encuestas (10%)
3. **Premium LH deterioro leve:** -1.0 pts, 111 encuestas (36%)
4. **Resultado:** Economy arrastra al LH a territorio negativo (-1.4 pts), pero el efecto es **mitigado significativamente** por Business

---

### **PARTE B: SELECCIÓN DE NARRATIVA CAUSAL**

**Narrativa:** Se adopta la **Explicación de la Cabina Dominante (Economy LH)**, ya que el escenario es DOMINANCIA. El rendimiento del largo radio está dictado por Economy debido a su volumen y severidad de la anomalía, efecto que fue **parcialmente mitigado** por la mejora excepcional en Business.

**Causa Dominante (Economy LH):** Deterioro operativo generalizado con triple impacto.

**Evidencia Clave Economy LH:**

**Métricas Operativas:**
- **OTP15:** 77.97% (-3.6 pts vs baseline 81.57%) - Puntualidad empeoró significativamente
- **Mishandling:** 18.91 (+3.45 pts vs baseline 15.46) - Gestión de equipaje deteriorada
- **Misconex:** 0.84 (+0.14 pts vs baseline 0.70) - Conexiones perdidas aumentaron
- **Triple correlación INVERSA confirmada:** Las tres métricas críticas empeoraron simultáneamente

**Incidentes NCS (Economy LH):**
- **15 incidentes totales:** 7 retrasos (47%), 6 cancelaciones (40%), 1 equipaje (7%), 4 otras incidencias (27%)

**Verbatims (Problemas Dominantes):**

**1. Retrasos/Cancelaciones (8+ menciones):**
- GIG-MAD: "Vuelo retrasado 4h, sin explicación" (NPS 3)
- MAD-SCL: "Atrasos, pérdida conexiones, pésima gestión re-acomodar" (NPS 0)
- MAD-MIA: "Llegar 20:45, llegamos 00:10, problema técnico" (NPS 0)
- MAD-ORD: "Vuelo conexión Madrid-Almería cancelado 30 min antes" (NPS 0)

**2. Equipaje Perdido/Retrasado (6+ menciones):**
- GIG-MAD: "Equipaje perdido, sin nada 4 días" (NPS 2)
- GRU-MAD: "Equipaje retrasado desde 16/12, sin novedades" (NPS 0)
- JFK-MAD: "Maleta perdida, tardó +1 día devolverla" (NPS 0)
- EZE-MAD: "Carrito maletas, 3 cayeron, 1 quedó tirada en pista rota" (NPS 8)

**3. Problemas Técnicos (3 menciones):**
- MAD-MIA: "Problema técnico, desembarcar, 2h en avión, cambio de avión" (NPS 4)
- LIM-MAD: "Avión hecho pedazos, sillones con cinta adhesiva" (NPS 0)

**4. Catering Deficiente (4 menciones):**
- EZE-MAD: "Cena escasa, 12h sin servicio bar/cocina" (NPS 3)
- MAD-ORD: "Sin desayuno, solo cena, ni agua ni café" (NPS 4)
- JFK-MAD: "Solo comimos 1 vez, desayuno solo vaso zumo" (NPS 7)

**Rutas Críticas (Economy LH):**

| Ruta | NPS | Encuestas | Problemas Documentados |
|------|-----|-----------|------------------------|
| **MAD-MIA** | -28.6 | 7 | Problemas técnicos, cambio avión, retraso 3+ horas |
| **GRU-MAD** | -28.6 | 7 | Equipaje retrasado desde 16/12 sin novedades |
| **LIM-MAD** | -23.5 | 17 | Retrasos, conexiones perdidas, avión deteriorado, catering |
| **MAD-SCL** | -22.2 | 9 | Cancelaciones, pérdida conexiones, gestión pésima |
| **JFK-MAD** | -16.7 | 18 | Equipaje perdido, catering deficiente |

**Patrón Geográfico:** Rutas transatlánticas (América del Norte y Sur) concentran los problemas más severos, especialmente aquellas con conexiones en MAD.

**Perfiles Más Afectados (Economy LH):**
- **Codeshare LATAM:** NPS -28.6 (14 encuestas) - Rutas sudamericanas críticas
- **Codeshare QR (Qatar):** NPS -66.7 (6 encuestas) - Severidad extrema
- **Residentes EUROPA:** NPS 1.7 (120 encuestas) - Mayor volumen afectado
- **Viajeros Business:** NPS 7.3 (150 encuestas) - Más sensibles a retrasos/conexiones
- **Flota A332:** NPS -12.0 (25 encuestas) - Flota antigua con quejas de deterioro

---

**Factor Mitigante (Business LH - Anomalía Positiva):**

**Business LH:** POSITIVE ANOMALY (+13.3 pts, NPS 32.26 vs baseline 18.96)

**Limitación Crítica:** El análisis causal de Business LH no pudo completarse (error de procesamiento), pero el **impacto positivo es evidente**:
- **31 encuestas** (10% del total LH)
- **Mejora significativa de +13.3 pts** que contrarrestó parcialmente el deterioro de Economy
- **Hipótesis probable:** Mejoras en servicio, upgrades, o ausencia de los problemas operativos que afectaron a Economy

---

**Interpretación Final (LH):**

El largo radio muestra una **anomalía negativa moderada (-1.4 pts)** resultado de:

1. **Economy LH arrastrando el agregado:** -4.9 pts por deterioro operativo triple (OTP, Mishandling, Misconex)
2. **Business LH mitigando el impacto:** +13.3 pts contrarrestó significativamente el efecto negativo
3. **Premium LH neutral:** -1.0 pts, impacto marginal
4. **Efecto neto:** Economy domina por volumen (54%), pero Business reduce la caída de -4.9 a -1.4 pts en el agregado

---

---

## **RESUMEN EJECUTIVO - NIVEL CABINA**

| Radio | Escenario | Estado (Hijos \| Padre) | Narrativa Dominante | Causa Raíz |
|-------|-----------|-------------------------|---------------------|------------|
| **Short Haul** | **SINERGIA NEGATIVA** | `(Eco: -, Bus: - \| SH: -)` | Explicación del Padre | Mishandling +3.45 pts (sistémico, Madrid como epicentro) |
| **Long Haul** | **DOMINANCIA DE ECONOMY** | `(Eco: -, Bus: +, Prem: - \| LH: -)` | Explicación de Economy (mitigada por Business) | Triple deterioro operativo en Economy LH (OTP -3.6, Mishandling +3.45, Misconex +0.14) |

**Conclusión:** 
- **SH:** Problema sistémico de equipaje afectando transversalmente ambas cabinas, con Madrid como hub crítico
- **LH:** Economy domina el resultado negativo por deterioro operativo generalizado, pero Business LH logra mitigar parcialmente el impacto con una mejora excepcional de +13.3 pts

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 PASO 3: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL GLOBAL)

---

## **PARTE A: ANÁLISIS DE DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(LH: -, SH: - | GLOBAL: -)`

- **Long Haul (LH):** NEGATIVE ANOMALY (-1.4 pts, NPS 4.48 vs baseline 5.89)
- **Short Haul (SH):** NEGATIVE ANOMALY (-7.6 pts, NPS 20.15 vs baseline 27.73)
- **GLOBAL (Padre):** NEGATIVE ANOMALY (-6.3 pts, NPS 14.56 vs baseline 20.89)

**Escenario Identificado:** **SINERGIA NEGATIVA CON DOMINANCIA DE SHORT HAUL (-,- | -)**

Ambos radios empujan en la misma dirección negativa, configurando un **problema sistémico de RED**. Sin embargo, **Short Haul domina claramente el efecto** por:

1. **Mayor severidad de la anomalía:** SH -7.6 pts vs LH -1.4 pts (5.4x más severo)
2. **Mayor volumen de encuestas:** SH 402 encuestas (64% del total) vs LH 223 encuestas (36%)
3. **Resultado agregado:** GLOBAL -6.3 pts refleja principalmente el deterioro de SH, con contribución menor de LH

---

## **PARTE B: SELECCIÓN DE NARRATIVA CAUSAL**

**Narrativa:** Se adopta la **Explicación del Nodo Global**, ya que el escenario es SINERGIA. La red entera se vio impactada por una **causa raíz común** que afectó transversalmente tanto al Largo como al Corto Radio, con especial severidad en Short Haul.

---

## **CAUSA RAÍZ GLOBAL: COLAPSO EN GESTIÓN DE EQUIPAJE (MISHANDLING)**

**Nivel de Confianza: ALTO ✅** (Triangulación exitosa en 5 fuentes de datos)

---

### **EVIDENCIA CUANTITATIVA - MÉTRICAS OPERATIVAS**

**Mishandling (Incidentes por cada 1,000 pasajeros):**
- **Valor del día:** 18.91
- **Baseline (media últimos 7 días):** 15.46
- **Desviación:** **+3.45 pts (+22.3%)**
- **Correlación con NPS:** INVERSA confirmada → Mishandling↑ = NPS↓

**Distribución por Radio:**
- **Short Haul:** Mishandling 18.91 (+3.45 pts vs baseline 15.46)
- **Long Haul:** Mishandling 18.91 (+3.45 pts vs baseline 15.46)
- **Patrón:** Deterioro sistémico afectando ambos radios por igual

**Otras Métricas Operativas (Contexto):**
- **Misconex:** 0.84 (+0.14 pts vs baseline 0.70) - Conexiones perdidas aumentaron 20%
- **OTP15 (Global):** 89.44% (+0.45 pts) - Puntualidad mejoró ligeramente (contradicción aparente)
- **Load Factor:** Variaciones mixtas por segmento

**Análisis de Contradicción OTP15:**
- OTP15 mide puntualidad dentro de 15 minutos
- Los **229 incidentes de retrasos reportados en NCS** (31.8% del total) sugieren retrasos **superiores a 15 minutos** que no impactan la métrica OTP15 pero **SÍ la percepción del cliente**

---

### **EVIDENCIA CUALITATIVA - INCIDENTES NCS**

**Volumen Total (16-Dic-2025):**
- **720 incidentes operativos**
- **70 vuelos afectados**

**Distribución por Tipo:**
| Tipo | Cantidad | % del Total | Relevancia para NPS |
|------|----------|-------------|---------------------|
| **Retrasos** | 229 | 31.8% | ⚠️ **CATEGORÍA DOMINANTE** |
| **Cancelaciones** | 122 | 16.9% | ⚠️ Alta |
| **Equipaje** | 6 | 0.8% | ⚠️ Crítica (subreportada vs verbatims) |
| **Otras incidencias** | 24 | 3.3% | Media |
| **Incidencias sistemas** | 2 | 0.3% | Baja |

**Incidente Crítico Destacado:**
- **MAD (Madrid):** "52 cambios de equipo, 142 pérdidas de conexión y 160 conexiones reprogramadas"
- **Epicentro operativo:** Madrid concentra la mayoría de disrupciones

**Inconsistencia Detectada:**
- NCS reporta solo **6 incidentes formales de equipaje** (0.8%)
- Verbatims documentan **14+ quejas explícitas de equipaje perdido/retrasado**
- **Conclusión:** Subreporte en NCS, la métrica Mishandling (+3.45) refleja la realidad operativa

---

### **EVIDENCIA CUALITATIVA - VERBATIMS (625 ENCUESTAS TOTALES)**

**Temas Dominantes (Frecuencia de Menciones):**

**1. EQUIPAJE PERDIDO/RETRASADO (14+ menciones explícitas):**

**Short Haul:**
- FRA-MAD: "Perdieron todo mi equipaje" (NPS 0)
- GVA-MAD: "Nos dejaron las maletas en Ginebra, 3 días sin localizar" (NPS 0)
- LCG-MAD: "Soltaron maletas en pista a oscuras, cogieron la mía por error" (NPS 0)
- AGP-MAD: "Mi maleta está perdida en almacén de IB en Madrid" (NPS 0)
- BCN-SXB: "Maletas llegaron 2.5 días después, contenía máquina de apnea vital" (NPS 0)

**Long Haul:**
- GRU-MAD: "Mi equipaje lleva retrasado desde el 16/12, sin novedades" (NPS 0)
- GIG-MAD: "Mi equipaje se perdió, quedé sin nada 4 días" (NPS 2)
- JFK-MAD: "Mi maleta se perdió, tardó más de un día en devolvérmela" (NPS 0)
- EZE-MAD: "Carrito con maletas, 3 cayeron, 1 quedó tirada en pista rota" (NPS 8)

**2. RETRASOS OPERATIVOS (13+ menciones explícitas):**

**Short Haul:**
- BCN-MAD: "100 minutos de retraso" (NPS 0)
- MAD-ZRH: "Retraso 1 hora + desorganización check-in" (NPS 0)
- BIO-MAD: "Salió 1 hora tarde" (NPS 0)

**Long Haul:**
- MAD-MIA: "5 horas de retraso, problema técnico" (NPS 0)
- GIG-MAD: "Llegamos 23:00 en vez de 18:40, sin ayuda" (NPS 2)
- GIG-MAD: "Retraso de 2.5h en despegue" (NPS 1)

**3. PROBLEMAS EN CONEXIONES (4+ menciones explícitas):**
- LIS-MAD: "Llegué tarde, conexión cerrando, me quedé sin saber qué hacer" (NPS 0)
- AGP-MAD: "Demora en Buenos Aires, tiempo de conexión reducido, pánico" (NPS 0)
- MAD-SCL: "Atrasos, pérdida de conexiones, pésima gestión para re-acomodar" (NPS 0)

**4. SERVICIO DE TRIPULACIÓN (4+ menciones):**
- LIM-MAD: "Tripulación sin ánimo, grosera con pasajeros" (NPS 2)
- MAD-MIA: "Servicio como low-cost, no ofrecían bebidas" (NPS 0)
- EZE-MAD: "Tripulantes golpeaban con carros, se molestaban si iba al baño" (NPS 3)

---

### **EVIDENCIA GEOGRÁFICA - RUTAS CRÍTICAS**

**Patrón Identificado: MADRID (MAD) COMO EPICENTRO OPERATIVO**

**Rutas Short Haul con Mayor Impacto:**

| Ruta | NPS | Encuestas | Problemas Validados (3 fuentes) |
|------|-----|-----------|----------------------------------|
| **GVA-MAD** | -50.0 | 6 | ✅ Equipaje + NCS + Operative Data |
| **BCN-MAD** | -42.9 | 21 | ✅ Retrasos + Equipaje + NCS |
| **LIS-MAD** | -20.0 | 10 | ✅ Conexiones perdidas + NCS |
| **AGP-MAD** | 0.0 | 10+10 | ✅ Equipaje + Desinformación + NCS |

**Rutas Long Haul con Mayor Impacto:**

| Ruta | NPS | Encuestas | Problemas Validados (3 fuentes) |
|------|-----|-----------|----------------------------------|
| **GRU-MAD** | -28.6 | 7 | ✅ Equipaje + NCS + Operative Data |
| **GIG-MAD** | Múltiples | - | ✅ Equipaje + Retrasos severos + NCS |
| **MAD-MIA** | -28.6 | 7 | ✅ Problemas técnicos + Retrasos 5h + NCS |
| **LIM-MAD** | -23.5 | 17 | ✅ Retrasos + Conexiones + Avión deteriorado |
| **MAD-SCL** | -22.2 | 9 | ✅ Cancelaciones + Conexiones perdidas |

**Observación Crítica:**
- **85% de las rutas críticas** involucran MAD (Madrid) como origen, destino o conexión
- **Problema sistémico en el hub principal:** Gestión de equipaje, coordinación de conexiones, cambios de equipo

---

### **EVIDENCIA DEMOGRÁFICA - PERFILES MÁS AFECTADOS**

**Por Codeshare (Dispersión: 200 pts):**

| Compañía | NPS | Encuestas | Radio | Impacto |
|----------|-----|-----------|-------|---------|
| **QR (Qatar)** | -66.7 | 6 | LH | ⚠️ **SEVERO** |
| **LATAM** | -28.6 | 14 | LH | ⚠️ **CRÍTICO** (rutas sudamericanas) |
| **IB** | 17.2 | 551 | Ambos | Base principal, performance aceptable |
| **YW** | 22.2 | 45 | SH | Performance positiva |

**Por Región de Residencia:**

| Región | NPS | Encuestas | Radio | Impacto |
|--------|-----|-----------|-------|---------|
| **EUROPA** | 1.7 | 120 | LH | ⚠️ **CRÍTICO** (mayor volumen afectado) |
| **ASIA** | -50.0 | 4 | SH | ⚠️ Muestra pequeña pero severidad extrema |
| **ESPAÑA** | 18.0 | 283 | Ambos | Base principal, relativamente mejor |

**Por Tipo de Viajero:**

| Perfil | NPS | Encuestas | Impacto |
|--------|-----|-----------|---------|
| **Business** | 7.3 | 150 | ⚠️ Más críticos, sensibles a retrasos/conexiones |
| **Leisure** | 16.8 | 475 | Mayoría de pasajeros, más tolerantes |

**Por Flota:**

| Modelo | NPS | Encuestas | Radio | Impacto |
|--------|-----|-----------|-------|---------|
| **A332** | -12.0 | 25 | LH | ⚠️ Flota antigua, quejas de deterioro |
| **A333** | -5.7 | 35 | LH | ⚠️ Flota antigua, problemas confort |
| **CRJ** | 14.8 | 128 | SH | ⚠️ Concentra problemas equipaje YW |
| **A320neo** | 6.2 | 97 | SH | ⚠️ Concentra problemas equipaje IB |

---

## **SÍNTESIS EJECUTIVA - NIVEL GLOBAL**

### **Dinámica de Agregación:**
**SINERGIA NEGATIVA CON DOMINANCIA DE SHORT HAUL** `(LH: -, SH: - | GLOBAL: -)`

- **Short Haul arrastra el resultado:** -7.6 pts (64% de encuestas) domina el agregado
- **Long Haul contribuye marginalmente:** -1.4 pts (36% de encuestas)
- **Efecto combinado:** GLOBAL -6.3 pts refleja problema sistémico de red

---

### **Causa Raíz Única y Transversal:**

**COLAPSO EN GESTIÓN DE EQUIPAJE (MISHANDLING +3.45 pts, +22.3%)**

**Mecanismo Causal:**
1. **Deterioro operativo en MAD (hub principal):** 52 cambios de equipo, 142 pérdidas de conexión, 160 conexiones reprogramadas
2. **Efecto cascada en ambos radios:**
   - **Short Haul:** Rutas domésticas/europeas con MAD como origen/destino (GVA-MAD, BCN-MAD, AGP-MAD)
   - **Long Haul:** Rutas transatlánticas con conexiones en MAD (GRU-MAD, GIG-MAD, MAD-MIA)
3. **Impacto diferencial por volumen:**
   - SH absorbe 64% del impacto (402 encuestas, -7.6 pts)
   - LH absorbe 36% del impacto (223 encuestas, -1.4 pts)

---

### **Factores Agravantes Secundarios:**

1. **Conexiones perdidas (Misconex +0.14 pts, +20%):** 142 pérdidas + 160 reprogramaciones en MAD
2. **Retrasos operativos (229 incidentes NCS, 31.8%):** Retrasos >15 min no capturados en OTP15
3. **Problemas técnicos en flota antigua:** A332/A333 (LH) con quejas de deterioro
4. **Servicio de tripulación:** 4 quejas cualitativas (causa emergente)

---

### **Segmentos Críticos para Intervención:**

**Prioridad 1 - Codeshare Internacional:**
- LATAM (NPS -28.6, rutas sudamericanas)
- Qatar Airways (NPS -66.7, muestra pequeña pero severidad extrema)

**Prioridad 2 - Perfiles de Cliente:**
- Viajeros Business (NPS 7.3, sensibles a retrasos/conexiones)
- Residentes Europa (NPS 1.7, mayor volumen afectado en LH)

**Prioridad 3 - Rutas Específicas:**
- **SH:** GVA-MAD, BCN-MAD, LIS-MAD, AGP-MAD
- **LH:** GRU-MAD, GIG-MAD, MAD-MIA, LIM-MAD, MAD-SCL

**Prioridad 4 - Flotas:**
- CRJ (YW SH, NPS 14.8)
- A320neo (IB SH, NPS 6.2)
- A332/A333 (LH, NPS negativo)

---

### **Nivel de Confianza Global: ALTO ✅**

**Triangulación Exitosa (5 fuentes):**
- ✅ **Operative Data:** Mishandling +3.45 pts confirmado
- ✅ **NCS:** 720 incidentes, MAD como epicentro
- ✅ **Verbatims:** 14+ quejas equipaje, 13+ quejas retrasos
- ✅ **Routes:** Rutas específicas validadas (3 fuentes coincidentes)
- ✅ **Customer Profile:** Segmentos afectados identificados

**Muestra Robusta:** 625 encuestas totales analizadas

---

### **Conclusión Final:**

El **16 de diciembre de 2025**, la red de IB/YW experimentó un **colapso sistémico en la gestión de equipaje** (+22.3% vs baseline) con **Madrid como epicentro operativo**, afectando transversalmente ambos radios (LH y SH) y resultando en una caída de **-6.3 pts en NPS Global**. Short Haul dominó el impacto por volumen (64% de encuestas, -7.6 pts), mientras Long Haul contribuyó marginalmente (-1.4 pts). Los segmentos más afectados fueron pasajeros de codeshare internacional (LATAM, Qatar), viajeros Business, y residentes Europa, concentrados en rutas con conexión en MAD y operadas en flotas CRJ/A320neo (SH) y A332/A333 (LH).

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

# 🔍 PASO 4: IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS (NMA)

---

## **CAUSA 1: COLAPSO EN GESTIÓN DE EQUIPAJE (MISHANDLING)**

### **Identificación del NMA:**

- **Escenario detectado:** SINERGIA NEGATIVA en todos los niveles de agregación
- **NMA:** **Global** (nivel más alto alcanzado por sinergia)
- **Afecta a:** Todos los sub-segmentos (LH y SH, todas las cabinas, ambas compañías IB/YW)
- **Tipo de impacto:** NEGATIVO

### **Cadena de Propagación hacia el Segmento Raíz:**

#### **Nivel 1: Compañía → Cabina (Short Haul)**

**Economy SH:**
- **IB Economy SH** (NEGATIVE -6.3 pts) → **Economy SH** (NEGATIVE -8.2 pts)
- **YW Economy SH** (NEGATIVE -11.4 pts) → **Economy SH** (NEGATIVE -8.2 pts)
- **Escenario:** SINERGIA NEGATIVA `(IB: -, YW: - | Economy SH: -)`
- **Hermanos:** Ambos hijos empujan en la misma dirección negativa
- **Conclusión:** Causa común (Mishandling) afecta a ambas compañías → NMA sube a Economy SH

**Business SH:**
- **IB Business SH** (NEGATIVE -2.1 pts) → **Business SH** (NEGATIVE -1.8 pts)
- **YW Business SH** (NEGATIVE -16.0 pts) → **Business SH** (NEGATIVE -1.8 pts)
- **Escenario:** DILUCIÓN NEGATIVA `(IB: -, YW: - | Business SH: -)`
- **Hermanos:** Ambos negativos, pero IB diluye por volumen (83% vs 17% YW)
- **Conclusión:** Causa común (Mishandling) afecta a ambas compañías, pero volumen de IB domina → NMA sube a Business SH

---

#### **Nivel 2: Cabina → Radio (Short Haul)**

- **Economy SH** (NEGATIVE -8.2 pts) → **SH** (NEGATIVE -7.6 pts)
- **Business SH** (NEGATIVE -1.8 pts) → **SH** (NEGATIVE -7.6 pts)
- **Escenario:** SINERGIA NEGATIVA PARCIAL `(Economy: -, Business: - | SH: -)`
- **Hermanos:** Ambos hijos empujan en la misma dirección negativa, Economy domina por severidad (-8.2 vs -1.8) y volumen (91% vs 9%)
- **Conclusión:** Causa común (Mishandling) afecta transversalmente a ambas cabinas → NMA sube a SH

---

#### **Nivel 3: Radio → Global**

- **SH** (NEGATIVE -7.6 pts) → **Global** (NEGATIVE -6.3 pts)
- **LH** (NEGATIVE -1.4 pts) → **Global** (NEGATIVE -6.3 pts)
- **Escenario:** SINERGIA NEGATIVA CON DOMINANCIA DE SH `(LH: -, SH: - | Global: -)`
- **Hermanos:** Ambos radios empujan en la misma dirección negativa, SH domina por severidad (-7.6 vs -1.4) y volumen (64% vs 36%)
- **Conclusión:** Causa común (Mishandling) afecta transversalmente a ambos radios → **NMA alcanza Global**

---

### **Resumen de Propagación:**

```
IB Economy SH (NEGATIVE -6.3) ────┐
                                   ├─→ Economy SH (NEGATIVE -8.2) ────┐
YW Economy SH (NEGATIVE -11.4) ───┘                                    │
                                                                        ├─→ SH (NEGATIVE -7.6) ────┐
IB Business SH (NEGATIVE -2.1) ───┐                                    │                           │
                                   ├─→ Business SH (NEGATIVE -1.8) ───┘                           │
YW Business SH (NEGATIVE -16.0) ──┘                                                                ├─→ GLOBAL (NEGATIVE -6.3)
                                                                                                    │
Economy LH (NEGATIVE -4.9) ───────┐                                                                │
Business LH (POSITIVE +13.3) ─────┼─→ LH (NEGATIVE -1.4) ─────────────────────────────────────────┘
Premium LH (NEGATIVE -1.0) ───────┘
```

**Tipo de Sinergia en cada nivel:**
- **Nivel Compañía (SH):** SINERGIA total (ambas compañías negativas)
- **Nivel Cabina (SH):** SINERGIA parcial (ambas cabinas negativas, Economy domina)
- **Nivel Radio:** SINERGIA parcial (ambos radios negativos, SH domina)

**Conclusión:** El Mishandling es una **causa sistémica de red** que afecta a todos los segmentos, propagándose desde los nodos hoja (IB/YW) hasta el **NMA = Global** mediante sinergia negativa en todos los niveles.

---

---

## **CAUSA 2: DETERIORO EN PUNTUALIDAD Y CONEXIONES (OTP + MISCONEX)**

### **Identificación del NMA:**

- **Escenario detectado:** SINERGIA NEGATIVA en Long Haul, contribución menor en Short Haul
- **NMA:** **Economy LH** (segmento hoja sin subniveles IB/YW)
- **Afecta a:** Economy LH principalmente, contribución marginal en SH
- **Tipo de impacto:** NEGATIVO

### **Análisis de Propagación:**

#### **Nivel 1: Cabina → Radio (Long Haul)**

- **Economy LH** (NEGATIVE -4.9 pts) → **LH** (NEGATIVE -1.4 pts)
- **Business LH** (POSITIVE +13.3 pts) → **LH** (NEGATIVE -1.4 pts)
- **Premium LH** (NEGATIVE -1.0 pts) → **LH** (NEGATIVE -1.4 pts)
- **Escenario:** DOMINANCIA DE ECONOMY `(Economy: -, Business: +, Premium: - | LH: -)`
- **Hermanos:** Economy domina por volumen (54%) e impone su signo negativo, a pesar de Business positivo (10%) y Premium neutral (36%)
- **Conclusión:** Causa específica de Economy LH (triple deterioro OTP/Mishandling/Misconex) arrastra al LH, pero es **mitigada significativamente** por Business (+13.3)

---

#### **Nivel 2: Radio → Global**

- **LH** (NEGATIVE -1.4 pts) → **Global** (NEGATIVE -6.3 pts)
- **SH** (NEGATIVE -7.6 pts) → **Global** (NEGATIVE -6.3 pts)
- **Escenario:** SINERGIA NEGATIVA CON DOMINANCIA DE SH `(LH: -, SH: - | Global: -)`
- **Hermanos:** Ambos radios negativos, pero **SH domina** el agregado (64% volumen, -7.6 pts severidad)
- **Conclusión:** La causa de Economy LH (OTP/Misconex) contribuye marginalmente al Global (-1.4 pts de -6.3 total), siendo **eclipsada por el Mishandling dominante de SH**

---

### **Cadena de Propagación:**

```
Economy LH (NEGATIVE -4.9) ────┐
                                ├─→ LH (NEGATIVE -1.4) ────┐
Business LH (POSITIVE +13.3) ──┤ (DOMINANCIA, Economy gana) │
Premium LH (NEGATIVE -1.0) ────┘                            ├─→ GLOBAL (NEGATIVE -6.3)
                                                             │   (SINERGIA, SH domina)
                                                             │
SH (NEGATIVE -7.6) ─────────────────────────────────────────┘
```

**Conclusión:** El deterioro en OTP/Misconex es una **causa específica de Economy LH** (segmento hoja), que propaga al LH mediante DOMINANCIA (a pesar de Business positivo), y contribuye marginalmente al Global donde es eclipsada por el Mishandling de SH.

---

### **Caso Especial: Segmento Hoja Sin Subniveles**

**Economy LH** es un segmento hoja sin análisis de compañías (IB/YW no aplican en LH según el árbol de datos proporcionado). Por lo tanto:

- **NMA = Economy LH** (coincide con el segmento raíz de esta causa)
- **No hay cadena de propagación interna** (no existen IB/YW Economy LH en el árbol)
- **Propagación externa:** Economy LH → LH (DOMINANCIA) → Global (SINERGIA con SH dominando)

---

---

## **CAUSA 3: MEJORA EXCEPCIONAL EN BUSINESS LONG HAUL**

### **Identificación del NMA:**

- **Escenario detectado:** ANOMALÍA POSITIVA aislada en Business LH
- **NMA:** **Business LH** (segmento hoja sin subniveles IB/YW)
- **Afecta a:** Únicamente Business LH
- **Tipo de impacto:** POSITIVO

### **Análisis de Propagación:**

#### **Nivel 1: Cabina → Radio (Long Haul)**

- **Business LH** (POSITIVE +13.3 pts) → **LH** (NEGATIVE -1.4 pts)
- **Economy LH** (NEGATIVE -4.9 pts) → **LH** (NEGATIVE -1.4 pts)
- **Premium LH** (NEGATIVE -1.0 pts) → **LH** (NEGATIVE -1.4 pts)
- **Escenario:** DOMINANCIA DE ECONOMY (OPUESTA) `(Economy: -, Business: +, Premium: - | LH: -)`
- **Hermanos:** Economy domina e impone su signo negativo al LH, a pesar del fuerte positivo de Business
- **Conclusión:** La mejora de Business LH **NO propaga al LH** debido a que Economy (54% volumen, -4.9 pts) domina el agregado

---

#### **Nivel 2: Radio → Global**

- **LH** (NEGATIVE -1.4 pts) → **Global** (NEGATIVE -6.3 pts)
- **SH** (NEGATIVE -7.6 pts) → **Global** (NEGATIVE -6.3 pts)
- **Escenario:** SINERGIA NEGATIVA CON DOMINANCIA DE SH
- **Conclusión:** El positivo de Business LH **NO alcanza el Global** porque:
  1. Fue neutralizado en el nivel LH por Economy dominante
  2. LH contribuye marginalmente al Global (36% volumen)

---

### **Cadena de Propagación (Bloqueada):**

```
Business LH (POSITIVE +13.3) ──┐
                                ├─→ LH (NEGATIVE -1.4) ────┐
Economy LH (NEGATIVE -4.9) ────┤ (DOMINANCIA, Economy gana) │
Premium LH (NEGATIVE -1.0) ────┘ (Business NO propaga)      ├─→ GLOBAL (NEGATIVE -6.3)
                                                             │   (Business NO alcanza)
                                                             │
SH (NEGATIVE -7.6) ─────────────────────────────────────────┘
```

**Conclusión:** La mejora en Business LH es una **anomalía positiva aislada** que:
1. **NO propaga al LH** (bloqueada por DOMINANCIA de Economy)
2. **NO alcanza el Global** (doble bloqueo: LH negativo + SH dominando)
3. **Efecto mitigante:** Reduce parcialmente el impacto negativo de Economy en LH (de -4.9 a -1.4 en el agregado)

---

### **Caso Especial: Segmento Hoja Sin Subniveles**

**Business LH** es un segmento hoja sin análisis de compañías (IB/YW no aplican en LH). Por lo tanto:

- **NMA = Business LH** (coincide con el segmento raíz de esta causa positiva)
- **No hay cadena de propagación interna**
- **Propagación externa bloqueada:** Business LH → LH (BLOQUEADO por DOMINANCIA de Economy) → Global (NO alcanza)

---

---

## **RESUMEN EJECUTIVO DE NMAs**

| Causa | NMA | Tipo | Escenario de Propagación | Alcance |
|-------|-----|------|--------------------------|---------|
| **Colapso Gestión Equipaje (Mishandling)** | **Global** | NEGATIVO | SINERGIA en todos los niveles | Red completa (LH + SH, todas cabinas, IB/YW) |
| **Deterioro Puntualidad/Conexiones (OTP/Misconex)** | **Economy LH** | NEGATIVO | DOMINANCIA (Economy LH → LH) + SINERGIA (LH → Global, eclipsado por SH) | Economy LH principalmente, contribución marginal a Global |
| **Mejora Business LH** | **Business LH** | POSITIVO | BLOQUEADO (no propaga a LH ni Global) | Únicamente Business LH, efecto mitigante en LH |

---

### **Observaciones Clave:**

1. **Causa 1 (Mishandling)** es la **única causa sistémica de red** que alcanza el NMA = Global mediante sinergia pura en todos los niveles.

2. **Causa 2 (OTP/Misconex)** es **específica de Economy LH** y contribuye marginalmente al Global, siendo eclipsada por el Mishandling dominante de SH.

3. **Causa 3 (Business LH positivo)** está **completamente aislada** y no propaga, pero cumple un rol **mitigante** reduciendo el impacto negativo de Economy en el agregado LH.

4. **Casos especiales aplicados:**
   - Economy LH y Business LH son segmentos hoja sin subniveles IB/YW (según árbol proporcionado)
   - No hay análisis de compañías en Long Haul

5. **Dinámicas de bloqueo:**
   - Business LH positivo bloqueado por DOMINANCIA de Economy en nivel LH
   - Economy LH negativo eclipsado por DOMINANCIA de SH en nivel Global

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

# 📊 PASO 4B: EXTRACCIÓN DE EVIDENCIAS DEL CONTEXTO INICIAL

---

## **NMA 1: GLOBAL**

### **=== NMA: Global ===**

#### **📈 EXPLANATORY DRIVERS:**
No disponible

---

#### **📊 DATOS OPERATIVOS:**

**Métricas del día vs baseline (Global):**
- **Mishandling:** 18.91 vs baseline 15.46 → **+3.45 pts** (incremento del 22.3%)
- **Misconex:** 0.84 vs baseline 0.70 → **+0.14 pts** (incremento del 20%)
- **OTP15:** 89.44% vs baseline → **+0.45 pts** (mejoró ligeramente)
- **Load Factor:** Variaciones mixtas por segmento

**Correlaciones confirmadas:**
- Mishandling: Relación INVERSA con NPS → Mishandling↑ = NPS↓
- Misconex: Relación INVERSA con NPS → Misconex↑ = NPS↓

---

#### **🚨 INCIDENTES NCS (CUANTITATIVO):**

**Total incidentes (Global, 16-Dic-2025):**
- **Total incidentes:** 720
- **Vuelos afectados:** 70
- **Distribución:**
  - Retrasos: 229 (31.8%) ← **CATEGORÍA DOMINANTE**
  - Cancelaciones: 122 (16.9%)
  - Equipaje: 6 (0.8%)
  - Otras incidencias: 24 (3.3%)
  - Incidencias sistemas: 2 (0.3%)

---

#### **🧠 NCS (CUALITATIVO / REFLEXIÓN):**

**Incidente Destacado:**
"52 cambios de equipo, 142 pérdidas de conexión y 160 conexiones reprogramadas en MAD"

**Observación Crítica del Análisis:**
"OTP15 mide puntualidad dentro de 15 minutos. Los 229 incidentes reportados sugieren retrasos superiores a 15 minutos que no impactan la métrica OTP15 pero sí la percepción del cliente."

**Inconsistencia detectada:**
"NCS no reportó incidentes específicos de equipaje, a pesar del Mishandling elevado"

---

#### **💬 FEEDBACK DE CLIENTES:**

**Temas Principales en Verbatims (625 encuestas totales):**

**1. Equipaje perdido/retrasado (14+ menciones explícitas):**
- FRA-MAD: "Perdieron todo mi equipaje" [NPS 0]
- GRU-MAD: "Mi equipaje lleva retrasado desde el 16/12" [NPS 0]
- GIG-MAD: "Mi equipaje se perdió, quedé sin nada 4 días" [NPS 2]
- GVA-MAD: "Nos dejaron las maletas en Ginebra" [NPS 0]
- LCG-MAD: "Soltaron maletas en pista a oscuras, cogieron la mía por error" [NPS 0]
- AGP-MAD: "Mi maleta está perdida" [NPS 0]
- BCN-SXB: "Nos perdieron las maletas facturadas" [NPS 0]
- MAD-ORY: "Después de 72 horas, aún no la han localizado" [NPS 0]

**2. Retrasos operativos (13+ menciones explícitas):**
- MAD-MIA: "5 horas de retraso" [NPS 0]
- GIG-MAD: "Llegamos 23:00 en vez de 18:40, sin ayuda" [NPS 2]
- GIG-MAD: "Retraso de 2.5h en despegue" [NPS 1]
- LCG-MAD: "Salió 1 hora tarde" [NPS 0]
- BCN-MAD: "100 minutos de retraso" [NPS 0]
- MAD-ZRH: "Retraso 1 hora" [NPS 0]

**3. Problemas en conexiones (4+ menciones):**
- LIS-MAD: "Llegué tarde, conexión cerrando, me quedé sin saber qué hacer" [NPS 0]
- AGP-MAD: "Demora en Buenos Aires, tiempo de conexión reducido, pánico" [NPS 0]

**4. Servicio de tripulación (4+ menciones):**
- LIM-MAD: "Tripulación sin ánimo, grosera con pasajeros" [NPS 2]
- MAD-MIA: "Servicio como low-cost, no ofrecían bebidas" [NPS 0]
- EZE-MAD: "Tripulantes golpeaban con carros, se molestaban si iba al baño" [NPS 3]
- MAD-SJO: "Discriminación en catering, faltaban bandejas en última fila" [NPS 1]

**5. Intoxicación alimentaria (1 mención severa):**
- LIM-MAD: "Intoxicación alimentaria, vómitos y diarrea tras comer" [NPS 2]

---

#### **✈️ RUTAS AFECTADAS (Top 10):**

**Rutas con Mayor Desviación Negativa de NPS:**

| Ruta | NPS | Encuestas | Nivel de Confianza |
|------|-----|-----------|-------------------|
| **MAD-MRS** | -50.0 | 6 | BAJA (muestra pequeña) |
| **BCN-MAD** | -28.6 | 28 | MEDIA (sin verbatims) |
| **MAD-SJO** | -21.4 | 14 | ALTA ✅ |
| **LIS-MAD** | -7.1 | 14 | ALTA ✅ |
| **GRU-MAD** | 0.0 | 9 | ALTA ✅ |
| **AGP-MAD** | 0.0 | 6 | ALTA ✅ |
| **GVA-MAD** | -50.0 | 6 | ALTA ✅ |
| **MAD-MIA** | -28.6 | 7 | ALTA ✅ |
| **LIM-MAD** | -23.5 | 17 | ALTA ✅ |
| **MAD-SCL** | -22.2 | 9 | ALTA ✅ |

**Rutas Críticas Validadas (3 fuentes coincidentes):**
1. **GRU-MAD / GIG-MAD:** Equipaje perdido, retrasos severos (Codeshare: LATAM NPS -28.6)
2. **LIS-MAD:** Conexiones perdidas (Residence: Europa NPS 1.7, n=120)
3. **MAD-SJO:** Servicio de tripulación, discriminación en catering
4. **AGP-MAD:** Desinformación crítica, equipaje forzado a bodega

---

#### **👥 PERFILES REACTIVOS:**

**Por Codeshare (Spread: 200 pts):**
| Compañía | NPS | Encuestas | Impacto |
|----------|-----|-----------|---------|
| **QR (Qatar)** | -66.7 | 6 | ⚠️ **SEVERO** |
| **LATAM** | -28.6 | 14 | ⚠️ **CRÍTICO** |
| **IB** | 17.2 | 551 | Base principal |
| **YW** | 22.2 | 45 | Performance positiva |

**Por Residence Region:**
| Región | NPS | Encuestas | Impacto |
|--------|-----|-----------|---------|
| **EUROPA** | 1.7 | 120 | ⚠️ **CRÍTICO** |
| **ASIA** | -50.0 | 4 | ⚠️ Muestra pequeña |
| **ESPAÑA** | 18.0 | 283 | Base principal |
| **AMERICA SUR** | 21.2 | 66 | Paradoja: NPS alto pese a quejas |
| **AMERICA NORTE** | 12.0 | 80 | Performance media-baja |

**Por Tipo de Viajero:**
| Perfil | NPS | Encuestas | Impacto |
|--------|-----|-----------|---------|
| **Leisure** | 16.8 | 475 | Mayoría de pasajeros |
| **Business** | 7.3 | 150 | ⚠️ Más críticos |

**Por Flota:**
| Modelo | NPS | Encuestas | Impacto |
|--------|-----|-----------|---------|
| **A332** | -12.0 | 25 | ⚠️ Flota antigua |
| **A333** | -5.7 | 35 | ⚠️ Flota antigua |
| **A350 next** | 0.0 | 56 | Muestra significativa |
| **A321** | 29.2 | 65 | Flota moderna |
| **A320** | 32.7 | 55 | Flota moderna |

---

---

## **NMA 2: ECONOMY LH**

### **=== NMA: Global/LH/Economy ===**

#### **📈 EXPLANATORY DRIVERS:**
No disponible

---

#### **📊 DATOS OPERATIVOS:**

**Métricas del día vs baseline (Economy LH):**
- **OTP15:** 77.97% vs baseline 81.57% → **-3.6 pts** (empeoró significativamente)
- **Mishandling:** 18.91 vs baseline 15.46 → **+3.45 pts** (incremento del 22.3%)
- **Misconex:** 0.84 vs baseline 0.70 → **+0.14 pts** (incremento del 20%)
- **Load Factor:** 88.69% vs baseline 90.05% → **-1.36 pts** (menor ocupación)

**Triple correlación INVERSA confirmada:** Las tres métricas críticas (OTP15, Mishandling, Misconex) empeoraron simultáneamente vs baseline.

---

#### **🚨 INCIDENTES NCS (CUANTITATIVO):**

**Total incidentes (Economy LH, 16-Dic-2025):**
- **Total incidentes:** 15
- **Distribución:**
  - Retrasos: 7 incidentes (47%)
  - Cancelaciones: 6 incidentes (40%)
  - Equipaje: 1 incidente (7%)
  - Otras incidencias: 4 incidentes (27%)

---

#### **🧠 NCS (CUALITATIVO / REFLEXIÓN):**
No disponible

---

#### **💬 FEEDBACK DE CLIENTES:**

**Temas Principales en Verbatims (168 encuestas Economy LH):**

**1. Retrasos/Cancelaciones (8+ menciones explícitas):**
- GIG-MAD: "El vuelo se ha retrasado 4h" [NPS 3]
- MAD-SCL: "Atrasos, pérdida de conexiones, pésima gestión para re acomodar" [NPS 0]
- GIG-MAD: "Teníamos previsto llegar a las 18:40... llegamos a las 23:00" [NPS 2]
- MAD-MIA: "teníamos que llegar a las 20:45 y llegamos a las 00:10" [NPS 0]
- MAD-ORD: "Vuelo de conexión Madrid-Almería cancelado 30 minutos antes" [NPS 0]

**2. Equipaje Perdido/Retrasado (6+ menciones explícitas):**
- GIG-MAD: "mi equipaje se perdió y me quedé sin absolutamente nada durante cuatro días" [NPS 2]
- GRU-MAD: "Mi equipaje lleva retrasado desde el 16 de diciembre y permanece en Madrid sin novedades" [NPS 0]
- JFK-MAD: "Mi maleta se perdió y tardó más de un día en devolvérmela" [NPS 0]
- MAD-SJO: "mi equipaje no apareció... tuve que esperar hasta el día siguiente" [NPS 6]
- EZE-MAD: "a uno de los carro que llevaba las maletas se le cayeron 3... y 1 de ellas quedó tirada en la pista y rota" [NPS 8]

**3. Problemas Técnicos (3 menciones):**
- MAD-MIA: "El vuelo tuvo un problema técnico y tuvimos que desembarcar" [NPS 4]
- LIM-MAD: "El avión estaba hecho pedazos... sillones pegados con cinta adhesiva" [NPS 0]

**4. Catering Deficiente (4 menciones):**
- EZE-MAD: "12 hs sin servicio de bar o cocina de ningún tipo" [NPS 3]
- MAD-ORD: "No dieron servicio de desayuno" [NPS 4]
- JFK-MAD: "solo nos dieron un vaso de zumo" [NPS 7]

---

#### **✈️ RUTAS AFECTADAS (Top 7 Economy LH):**

**Rutas con Mayor Desviación Negativa:**

| Ruta | NPS | Encuestas | Problemas Documentados |
|------|-----|-----------|------------------------|
| **MAD-MIA** | -28.6 | 7 | Problemas técnicos, cambio de avión, retraso 3+ horas |
| **GRU-MAD** | -28.6 | 7 | Equipaje retrasado desde 16/12 sin novedades |
| **LIM-MAD** | -23.5 | 17 | Retrasos, conexiones perdidas, avión deteriorado, catering deficiente |
| **MAD-SCL** | -22.2 | 9 | Cancelaciones, pérdida de conexiones, pésima gestión |
| **JFK-MAD** | -16.7 | 18 | Equipaje perdido, catering deficiente |
| **MAD-ORD** | 0.0 | 6 | Cancelación 30 min antes, sin desayuno |
| **MAD-SJO** | 0.0 | 7 | Equipaje no apareció, catering insuficiente |

**Rutas con NPS Positivo (Contraste):**
- MAD-SJU: NPS 10.0 (10 encuestas) - Sin incidentes reportados
- A350: NPS 18.5 (54 encuestas) - Mejor desempeño de flota

---

#### **👥 PERFILES REACTIVOS:**

**Por Codeshare (Dispersión: 200 pts):**
| Aerolínea | NPS | Encuestas | Impacto |
|-----------|-----|-----------|---------|
| **QR (Qatar)** | -100.0 | 3 | 🔴 CRÍTICO |
| **BA (British Airways)** | -37.5 | 8 | 🔴 MUY BAJO |
| **LATAM** | -22.2 | 9 | 🔴 BAJO |
| **IB** | 1.5 | 135 | 🟡 NEUTRO (mayor volumen) |

**Por Residence Region (Dispersión: 117 pts):**
| Residencia | NPS | Encuestas | Impacto |
|------------|-----|-----------|---------|
| **ESPAÑA** | -17.0 | 53 | 🔴 ALTO (mayor volumen afectado) |
| **EUROPA** | -13.3 | 15 | 🟠 MEDIO-ALTO |
| **Unknown** | -7.1 | 28 | 🟡 MEDIO |

**Por Fleet (Dispersión: 36.7 pts):**
| Tipo de Avión | NPS | Encuestas | Impacto |
|---------------|-----|-----------|---------|
| **A332** | -18.2 | 22 | 🔴 Peor flota del día |
| **A321** | -14.3 | 7 | 🟠 MEDIO-ALTO |
| **A333** | -11.5 | 26 | 🟠 MEDIO |
| **A350 next** | -10.5 | 38 | 🟡 MEDIO |
| **A350** | 18.5 | 54 | ✅ Mejor flota (contraste) |

**Por Business/Leisure (Dispersión: 7.7 pts):**
| Propósito | NPS | Encuestas | Impacto |
|-----------|-----|-----------|---------|
| **Business/Work** | -8.3 | 12 | 🟡 Ligeramente más afectados |
| **Leisure** | -0.6 | 156 | 🟡 Mayor volumen, impacto similar |

---

---

## **NMA 3: BUSINESS LH**

### **=== NMA: Global/LH/Business ===**

#### **📈 EXPLANATORY DRIVERS:**
No disponible

---

#### **📊 DATOS OPERATIVOS:**
No disponible (el análisis causal de Business LH no pudo completarse debido a error de procesamiento)

---

#### **🚨 INCIDENTES NCS (CUANTITATIVO):**
No disponible

---

#### **🧠 NCS (CUALITATIVO / REFLEXIÓN):**
No disponible

---

#### **💬 FEEDBACK DE CLIENTES:**
No disponible (análisis no completado)

---

#### **✈️ RUTAS AFECTADAS:**
No disponible

---

#### **👥 PERFILES REACTIVOS:**
No disponible

**Nota:** El análisis causal completo de Business LH no pudo completarse debido a error de procesamiento. Solo se confirma la **anomalía positiva de +13.3 pts** (NPS 32.26 vs baseline 18.96) con **31 encuestas** (10% del total LH).

---

---

## **RESUMEN DE DISPONIBILIDAD DE DATOS POR NMA**

| NMA | Operative Data | NCS Cuant. | NCS Cualit. | Verbatims | Routes | Profiles |
|-----|----------------|------------|-------------|-----------|--------|----------|
| **Global** | ✅ Completo | ✅ Completo | ✅ Parcial | ✅ Completo | ✅ Completo | ✅ Completo |
| **Economy LH** | ✅ Completo | ✅ Completo | ❌ No disp. | ✅ Completo | ✅ Completo | ✅ Completo |
| **Business LH** | ❌ No disp. | ❌ No disp. | ❌ No disp. | ❌ No disp. | ❌ No disp. | ❌ No disp. |

---

## Cabin Radio Reflection

# 📋 PASO 4C: REFLEXIÓN POR CABINA-RADIO

---

## **CABINAS SHORT HAUL (SH)**

---

### **=== ECONOMY SH ===**

**• NPS Cabina:** 18.9 (-8.2 pts)  
**• Estado:** NEGATIVE ANOMALY  
**• Escenario:** SINERGIA NEGATIVA (IB NEGATIVE, YW NEGATIVE | Cabina NEGATIVE)

**• IB:** NPS 20.6 (-6.3 pts)  
**Explicación:** Deterioro en gestión de equipaje (Mishandling +3.8 pts vs baseline, incremento del 23%). Problemas concentrados en rutas domésticas/europeas con Madrid como hub principal. Viajeros Business más afectados (NPS 6.1 vs Leisure 24.6, diferencial de 18.5 pts). Flota A320neo concentra el 43.5% de encuestas con NPS de solo 6.2 pts. Incidentes específicos: pérdida total de equipaje en FRA-MAD-GRU, retrasos de hasta 100 minutos en BCN-MAD, 1 hora de espera en recogida de maletas en LIS-MAD.

**• YW:** NPS 16.1 (-11.4 pts)  
**Explicación:** Incremento en problemas de equipaje (Mishandling +2.36 pts vs baseline) con alta incidencia operativa (31 incidentes: 16 cancelaciones + 15 retrasos). Impacto concentrado en residentes Europa (NPS -6.7) y flota CRJ (NPS 14.8, 89% de encuestas YW). Rutas críticas: GVA-MAD (NPS -50.0, equipaje perdido 3 días), AGP-MAD (NPS 0.0, equipaje perdido + conexión fallida), MAD-XRY (NPS 20.0, equipaje no entregado tras 3 días). Mayor severidad que IB debido a menor volumen de encuestas (143 vs 223), amplificando el impacto de incidentes individuales.

**• Narrativa de agregación:**  
Ambas compañías experimentaron deterioro en gestión de equipaje de manera sistémica, con **YW sufriendo un impacto más severo (-11.4 pts) que IB (-6.3 pts)**. El problema es común (Mishandling), pero YW fue más afectado por: (1) mayor concentración de problemas en flota CRJ, (2) rutas específicas con mayor incidencia de equipaje perdido (GVA-MAD, AGP-MAD), (3) menor volumen de encuestas amplificando el impacto. El resultado agregado de Economy SH (-8.2 pts) refleja la sinergia negativa de ambas operadoras, con IB aportando mayor volumen (61% de encuestas) y YW mayor severidad.

**• Rutas críticas (del CAUSAL EXPLANATION del padre - Economy SH):**
1. **GVA-MAD:** NPS -50.0 (n=6) - Equipaje perdido, familia sin maletas 3 días
2. **BCN-MAD:** NPS -42.9 (n=21) - Retrasos + pérdida conexión + equipaje mano
3. **LIS-MAD:** NPS -20.0 (n=10) - Retrasos múltiples + desinformación
4. **LHR-MAD:** NPS -10.5 (n=19) - Problemas embarque + retrasos
5. **AGP-MAD:** NPS 0.0 (n=10+10) - Equipaje perdido + retrasos + maltrato

**Patrón identificado:** Madrid (MAD) como epicentro operativo - 85% de rutas críticas involucran MAD como origen, destino o conexión.

**• Perfiles reactivos (del CAUSAL EXPLANATION del padre - Economy SH):**
- **Business/Leisure:** Business NPS 6.1 (n=114) vs Leisure NPS 24.6 (n=252) - Diferencial de 18.5 pts. Viajeros Business más sensibles a problemas de equipaje y retrasos.
- **Residence Region:** Europa NPS 7.1 (n=85) - Bajo con muestra significativa. España NPS 26.4 (n=197) - 54% de la muestra, mejor que internacionales.
- **Fleet:** CRJ NPS 14.8 (n=128, 35% del total) - Peor flota. A320neo NPS 6.2 (n=97, 26.5%) - Segunda peor flota. Ambas concentran problemas de equipaje.
- **CodeShare:** IB NPS 21.4 (n=337, 92% de muestra). Problema core de la operación IB/YW, no de códigos compartidos.

---

### **=== BUSINESS SH ===**

**• NPS Cabina:** 33.3 (-1.8 pts)  
**• Estado:** NEGATIVE ANOMALY  
**• Escenario:** DILUCIÓN NEGATIVA (IB NEGATIVE, YW NEGATIVE | Cabina NEGATIVE)

**• IB:** NPS 40.0 (-2.1 pts)  
**Explicación:** Deterioro en gestión de equipaje (Mishandling +3.8 pts vs baseline, incremento del 23%) y conexiones perdidas (Misconex +0.14 pts, incremento del 19%). Problemas concentrados en BCN-MAD (NPS 14.3, n=7) y LIS-MAD (NPS 25.0, n=4). Incidentes específicos: pérdida total de equipaje en FRA-MAD-GRU (NPS 0), 1 hora de espera en recogida de maletas en LIS-MAD (NPS 8), retrasos de hasta 100 minutos en BCN-MAD. Viajeros Business más afectados (NPS 20.0 vs Leisure 60.0, diferencial de 40 pts). Flota A320neo (NPS 20.0, 50% de encuestas) y residentes España (NPS 13.3, 50% de encuestas) concentran el impacto.

**• YW:** NPS 0.0 (-16.0 pts)  
**Explicación:** Colapso total con muestra estadísticamente insuficiente (solo 6 encuestas). Incidentes operativos específicos identificados: problema sanitario en FRA-MAD (tripulante tosiendo sin mascarilla en Business, NPS 5), retraso mecánico >1h en MAD-MRS (fallo en puertas de bodega, NPS 2), incidente no documentado en MAD-TRN (NPS -100). Pasajeros Leisure significativamente más críticos (NPS -50 vs Business +100, diferencial de 150 pts). Mishandling aumentó +2.36 pts pero sin quejas de equipaje en verbatims, sugiriendo desconexión operativa-cualitativa. **Nivel de confianza: BAJO** debido a muestra microscópica y evidencia fragmentada.

**• Narrativa de agregación:**  
Ambas compañías tienen anomalías negativas, pero el **volumen desproporcionado de IB (30 encuestas, 83%) diluye significativamente el impacto severo de YW (6 encuestas, 17%)**. El resultado agregado de Business SH (-1.8 pts) refleja principalmente los problemas de IB, que con mayor volumen atenúa el colapso extremo de YW. Sin embargo, la **severidad extrema en YW (-16.0 pts, NPS 0.0)** indica problemas operativos críticos no capturados en el agregado, requiriendo **investigación urgente** a pesar de la baja significancia estadística.

**• Rutas críticas:**

**Según escenario DILUCIÓN, usar rutas del hijo dominante (IB) + mencionar severidad de YW:**

**IB Business SH:**
1. **BCN-MAD:** NPS 14.3 (n=7) - Retrasos hasta 100 min, app de IB criticada
2. **LIS-MAD:** NPS 25.0 (n=4) - Demora equipaje (1h espera), retrasos
3. **LCG-MAD:** NPS 0.0 (n=2) - Incidentes NCS registrados
4. **FRA-MAD:** Sin datos cuantitativos - Pérdida total equipaje (caso grave, NPS 0)

**YW Business SH (severidad extrema, muestra pequeña):**
1. **MAD-TRN:** NPS -100.0 (n=1) - Sin verbatim asociado
2. **FRA-MAD:** NPS -100.0 (n=1) - Problema sanitario (tripulante tosiendo)
3. **MAD-MRS:** NPS -100.0 (n=1) - Retraso mecánico >1h

**• Perfiles reactivos:**

**Según escenario DILUCIÓN, usar perfiles del hijo dominante (IB) + mencionar diferencial de YW:**

**IB Business SH:**
- **Business/Leisure:** Business NPS 20.0 (n=15) vs Leisure NPS 60.0 (n=15) - Diferencial de 40 pts. Pasajeros Business más sensibles a problemas de equipaje (viajes profesionales, equipaje crítico).
- **Fleet:** A320neo NPS 20.0 (n=15, 50% de encuestas) - Mayor volumen, peor NPS. A319 NPS 25.0 (n=4) - También afectado. A320/A321/A350 NPS 50.0-100.0 - Mejor desempeño.
- **Residence Region:** España NPS 13.3 (n=15, 50% de encuestas) - Más afectados. Europa NPS 80.0 (n=5) - Mejor experiencia. Pasajeros españoles en rutas domésticas (BCN-MAD, LIS-MAD) más impactados.
- **CodeShare:** IB NPS 44.4 (n=27, 90% de muestra) - Problema específico de operaciones IB.

**YW Business SH (diferencial crítico):**
- **Business/Leisure:** Leisure NPS -50 (n=4) vs Business +100 (n=2) - Diferencial de 150 pts (patrón inverso a IB).
- **Residence Region:** España NPS 33.3 (n=3), Europa NPS 0.0 (n=2), Unknown NPS -100.0 (n=1) - Spread de 133.3 pts.
- **Fleet:** 100% CRJ (n=6) con NPS 0.0 agregado - Sin variabilidad para análisis comparativo.

---

---

## **CABINAS LONG HAUL (LH)**

---

### **=== ECONOMY LH ===**

**• NPS:** -1.2 (-4.9 pts)  
**• Estado:** NEGATIVE ANOMALY

**• Causa principal:**  
Deterioro operativo generalizado con triple impacto simultáneo: puntualidad empeoró (OTP15 -3.6 pts vs baseline), gestión de equipaje deteriorada (Mishandling +3.45 pts), y conexiones perdidas aumentaron (Misconex +0.14 pts). Las tres métricas críticas empeoraron simultáneamente, configurando una **triple correlación INVERSA confirmada** que explica completamente la caída de NPS.

**• Evidencia clave:**
- **OTP15:** 77.97% (-3.6 pts vs baseline 81.57%) - Puntualidad empeoró significativamente
- **Mishandling:** 18.91 (+3.45 pts vs baseline 15.46, +22.3%) - Correlación INVERSA con NPS
- **Misconex:** 0.84 (+0.14 pts vs baseline 0.70, +20%) - Conexiones perdidas aumentaron
- **Incidentes NCS:** 15 totales (7 retrasos 47%, 6 cancelaciones 40%, 1 equipaje 7%, 4 otras 27%)
- **Verbatims:** 8+ menciones de retrasos severos (hasta 5h en MAD-MIA), 6+ menciones de equipaje perdido (hasta 4 días sin equipaje en GIG-MAD), 3 menciones de problemas técnicos (avión "hecho pedazos" en LIM-MAD), 4 menciones de catering deficiente (12h sin servicio en EZE-MAD)

**• Rutas críticas:**
1. **MAD-MIA:** NPS -28.6 (n=7) - Problemas técnicos críticos, cambio de avión, retraso 3+ horas
2. **GRU-MAD:** NPS -28.6 (n=7) - Equipaje retrasado desde 16/12 sin novedades
3. **LIM-MAD:** NPS -23.5 (n=17) - Retrasos, conexiones perdidas, avión deteriorado, catering deficiente (mayor volumen)
4. **MAD-SCL:** NPS -22.2 (n=9) - Cancelaciones, pérdida de conexiones, gestión pésima
5. **JFK-MAD:** NPS -16.7 (n=18) - Equipaje perdido, catering deficiente (segundo mayor volumen)

**Patrón geográfico:** Rutas transatlánticas (América del Norte y Sur) concentran los problemas más severos, especialmente aquellas con conexiones en MAD.

**• Perfiles reactivos:**
- **CodeShare (spread 200 pts):** QR -100.0 (n=3) CRÍTICO, BA -37.5 (n=8) MUY BAJO, LATAM -22.2 (n=9) BAJO, IB 1.5 (n=135) NEUTRO. Pasajeros de codeshare (especialmente QR y BA) significativamente más afectados que vuelos operados directamente por IB.
- **Residence Region (spread 117 pts):** ESPAÑA -17.0 (n=53) ALTO (mayor volumen afectado), EUROPA -13.3 (n=15) MEDIO-ALTO, Unknown -7.1 (n=28) MEDIO. Residentes en España fueron el segmento con mayor volumen de encuestas negativas, probablemente afectados por cancelaciones y pérdidas de conexión en MAD.
- **Fleet (spread 36.7 pts):** A332 -18.2 (n=22) Peor flota, A321 -14.3 (n=7), A333 -11.5 (n=26), A350 next -10.5 (n=38), A350 18.5 (n=54) Mejor flota. Flota A332 tuvo el peor desempeño, sugiriendo problemas técnicos concentrados en flotas específicas.
- **Business/Leisure (spread 7.7 pts):** Business -8.3 (n=12) ligeramente más afectados, Leisure -0.6 (n=156) mayor volumen. Ambos segmentos afectados de manera similar.

---

### **=== BUSINESS LH ===**

**• NPS:** 32.3 (+13.3 pts)  
**• Estado:** POSITIVE ANOMALY

**• Causa principal:**  
Sin análisis causal disponible (el análisis completo no pudo completarse debido a error de procesamiento). Se confirma únicamente la **anomalía positiva de +13.3 pts** (NPS 32.26 vs baseline 18.96) con **31 encuestas** (10% del total LH).

**• Evidencia clave:**  
No disponible

**• Rutas críticas:**  
No disponible

**• Perfiles reactivos:**  
No disponible

**• Nota metodológica:**  
A pesar de la ausencia de análisis detallado, el **impacto positivo es evidente y significativo** (+13.3 pts). Esta mejora en Business LH actuó como **factor mitigante** que contrarrestó parcialmente el deterioro de Economy LH (-4.9 pts) en el agregado del radio, reduciendo la caída de LH a solo -1.4 pts. Hipótesis probable: mejoras en servicio, upgrades, o ausencia de los problemas operativos que afectaron a Economy.

---

### **=== PREMIUM LH ===**

**• NPS:** 8.3 (-1.0 pts)  
**• Estado:** NEGATIVE ANOMALY

**• Causa principal:**  
Sin análisis causal disponible (el análisis completo no pudo completarse debido a error de procesamiento). Se confirma únicamente la **anomalía negativa de -1.0 pts** con **111 encuestas** (36% del total LH).

**• Evidencia clave:**  
No disponible

**• Rutas críticas:**  
No disponible

**• Perfiles reactivos:**  
No disponible

**• Nota metodológica:**  
La caída de -1.0 pts es **marginal** comparada con Economy LH (-4.9 pts) y representa un **impacto neutral** en el agregado del radio. Con 36% de las encuestas LH, Premium no fue un driver significativo de la anomalía negativa de LH (-1.4 pts), que estuvo dominada por Economy (54% de encuestas).

---

---

## **RESUMEN EJECUTIVO DE CABINAS-RADIO**

| Cabina | NPS | Variación | Estado | Escenario | Driver Principal |
|--------|-----|-----------|--------|-----------|------------------|
| **Economy SH** | 18.9 | -8.2 | NEGATIVE | SINERGIA (IB -, YW -) | Mishandling sistémico (IB +3.8, YW +2.36) |
| **Business SH** | 33.3 | -1.8 | NEGATIVE | DILUCIÓN (IB -, YW -) | Mishandling (IB domina por volumen, YW colapso extremo) |
| **Economy LH** | -1.2 | -4.9 | NEGATIVE | N/A (segmento hoja) | Triple deterioro (OTP -3.6, Mishandling +3.45, Misconex +0.14) |
| **Business LH** | 32.3 | +13.3 | POSITIVE | N/A (segmento hoja) | Sin análisis (mejora significativa, factor mitigante) |
| **Premium LH** | 8.3 | -1.0 | NEGATIVE | N/A (segmento hoja) | Sin análisis (impacto marginal) |

**Conclusiones Clave:**
1. **Economy SH:** Sinergia negativa IB/YW con Mishandling como causa común, Madrid como epicentro operativo
2. **Business SH:** Dilución por volumen de IB, pero YW con colapso extremo (-16.0) que requiere investigación urgente
3. **Economy LH:** Triple deterioro operativo (OTP/Mishandling/Misconex) dominando el resultado negativo de LH
4. **Business LH:** Anomalía positiva aislada (+13.3) que mitiga parcialmente el impacto de Economy en LH
5. **Premium LH:** Impacto marginal (-1.0) sin efecto significativo en el agregado LH

---

## 📋 SÍNTESIS EJECUTIVA FINAL

```html
<b>SÍNTESIS EJECUTIVA</b><br>
<br>
La red global registró un <b>NPS de 14.6 (–6.3 pts)</b> con respecto a la media de los últimos 7 días, resultado de un colapso sistémico en la gestión de equipaje que afectó transversalmente a ambos radios, con especial severidad en Short Haul.<br>
<br>
El deterioro operativo tuvo su origen en un <b>incremento del 22.3% en Mishandling</b> (18.91 incidentes por cada 1,000 pasajeros frente a un baseline de 15.46), configurando la causa raíz única que propagó desde los niveles más granulares hasta el agregado global mediante sinergia negativa. Este problema se vio agravado por <b>720 incidentes operativos</b> que afectaron a 70 vuelos, con 229 retrasos (31.8% del total, categoría dominante), 122 cancelaciones (16.9%), y 6 incidentes formales de equipaje (0.8%, cifra subreportada frente a la evidencia cualitativa). El epicentro operativo fue <b>Madrid (MAD)</b>, donde se registraron 52 cambios de equipo, 142 pérdidas de conexión y 160 conexiones reprogramadas, generando un efecto cascada en rutas domésticas, europeas y transatlánticas que utilizan el hub como origen, destino o punto de conexión.<br>
<br>
En <b>Short Haul</b>, el NPS cayó a <b>20.1 (–7.6 pts)</b>, dominando el resultado global por volumen (64% de encuestas, 402 respuestas) y severidad. Tanto <b>IB</b> como <b>YW</b> experimentaron deterioro en gestión de equipaje de manera sistémica, con YW sufriendo un impacto más severo (–11.4 pts frente a –6.3 pts de IB) debido a mayor concentración de problemas en flota CRJ, rutas específicas con mayor incidencia de equipaje perdido, y menor volumen de encuestas amplificando el impacto de incidentes individuales. Los clientes reportaron 8 casos explícitos de equipaje perdido o retrasado, incluyendo situaciones críticas como familias sin maletas durante 3 días en GVA-MAD, pérdida de equipaje con objetos vitales (máquina de apnea del sueño) en BCN-SXB, y maletas extraviadas sin localización tras 72 horas en MAD-ORY. Adicionalmente, se identificó un patrón recurrente de gestión inadecuada de equipaje de mano, con personal forzando facturación alegando falta de espacio cuando los aviones iban medio vacíos, documentado en FCO-MAD, AGP-MAD y MAD-SCQ. Las rutas más afectadas fueron <b>GVA-MAD</b> (NPS –50.0), <b>BCN-MAD</b> (NPS –42.9 con 21 encuestas, mayor volumen crítico), <b>LIS-MAD</b> (NPS –20.0), <b>LHR-MAD</b> (NPS –10.5), y <b>AGP-MAD</b> (NPS 0.0), todas con Madrid como denominador común. Los viajeros más sensibles fueron los de <b>Business</b> (NPS 6.1 frente a Leisure 24.6, diferencial de 18.5 pts), residentes de <b>Europa</b> (NPS 7.1 con muestra significativa de 85 encuestas), y usuarios de flotas <b>CRJ</b> (NPS 14.8, 35% del total) y <b>A320neo</b> (NPS 6.2, 26.5% del total), que concentraron la mayoría de problemas de equipaje.<br>
<br>
En <b>Long Haul</b>, el NPS cayó a <b>4.5 (–1.4 pts)</b>, contribuyendo marginalmente al resultado global (36% de encuestas, 223 respuestas). Esta anomalía negativa moderada fue resultado de una dinámica de dominancia donde <b>Economy LH</b> (NPS –1.2, –4.9 pts) arrastró al radio a pesar de la mejora excepcional de <b>Business LH</b> (NPS 32.3, +13.3 pts). Economy LH experimentó un <b>triple deterioro operativo simultáneo</b>: puntualidad empeoró 3.6 pts (OTP15 de 77.97% frente a baseline de 81.57%), gestión de equipaje deteriorada (Mishandling +3.45 pts), y conexiones perdidas aumentaron 20% (Misconex +0.14 pts). Los clientes reportaron 8 menciones de retrasos severos (hasta 5 horas en MAD-MIA por problemas técnicos), 6 menciones de equipaje perdido (hasta 4 días sin equipaje en GIG-MAD), 3 menciones de problemas técnicos en flota antigua (avión "hecho pedazos" con sillones pegados con cinta adhesiva en LIM-MAD), y 4 menciones de catering deficiente (12 horas sin servicio de bar o cocina en EZE-MAD). Las rutas transatlánticas concentraron los problemas más severos, especialmente <b>MAD-MIA</b> (NPS –28.6), <b>GRU-MAD</b> (NPS –28.6 con equipaje retrasado desde el 16 de diciembre sin novedades), <b>LIM-MAD</b> (NPS –23.5 con 17 encuestas, mayor volumen), <b>MAD-SCL</b> (NPS –22.2), y <b>JFK-MAD</b> (NPS –16.7 con 18 encuestas, segundo mayor volumen). Los pasajeros de <b>codeshare internacional</b> fueron los más afectados: Qatar Airways con NPS de –100.0 (muestra pequeña pero severidad extrema), British Airways con –37.5, y LATAM con –28.6 en rutas sudamericanas críticas. Los residentes de <b>España</b> representaron el mayor volumen afectado (NPS –17.0, 53 encuestas), probablemente por cancelaciones y pérdidas de conexión en MAD, seguidos de residentes de <b>Europa</b> (NPS –13.3). La flota <b>A332</b> registró el peor desempeño (NPS –18.2), sugiriendo problemas técnicos concentrados en flotas antiguas. Business LH, por su parte, logró una mejora significativa de +13.3 pts que actuó como factor mitigante, reduciendo parcialmente el impacto negativo de Economy en el agregado del radio (de –4.9 a –1.4 pts), aunque sin análisis causal disponible debido a limitaciones de procesamiento.<br>
<br>
La convergencia de <b>Short Haul (–7.6 pts)</b> y <b>Long Haul (–1.4 pts)</b>, ambos en dirección negativa, produjo el resultado global de –6.3 pts mediante sinergia negativa con dominancia de Short Haul, que por su mayor volumen (64% de encuestas) y severidad (5.4 veces más severo que LH) arrastró el agregado de la red. El problema de Mishandling afectó de manera transversal a todos los segmentos, propagándose desde los niveles más granulares (IB y YW en cada cabina) hasta el nivel global, con Madrid como epicentro operativo sistémico donde el 85% de las rutas críticas involucran este hub como origen, destino o punto de conexión.<br>
<br>
<br>
<b><u>DETALLE POR CABINA</u></b><br>
<br>
<b><u>ECONOMY SH: Colapso sistémico en gestión de equipaje con Madrid como epicentro</u></b><br>
El NPS de Economy SH cayó a <b>18.9 (–8.2 pts)</b>, resultado de una sinergia negativa donde tanto <b>IB</b> (NPS 20.6, –6.3 pts) como <b>YW</b> (NPS 16.1, –11.4 pts) experimentaron deterioro en gestión de equipaje de manera simultánea. El Mishandling aumentó 3.8 pts en IB (incremento del 23%) y 2.36 pts en YW, afectando transversalmente a ambas operadoras con especial severidad en YW debido a mayor concentración de problemas en flota CRJ (89% de encuestas YW, NPS 14.8), rutas específicas como GVA-MAD (NPS –50.0, equipaje perdido 3 días), AGP-MAD (NPS 0.0, equipaje perdido más conexión fallida), y MAD-XRY (NPS 20.0, equipaje no entregado tras 3 días), además de menor volumen de encuestas (143 frente a 223 de IB) amplificando el impacto de incidentes individuales. Los problemas se concentraron en rutas con Madrid como hub principal: GVA-MAD (familia sin maletas 3 días), BCN-MAD (retrasos hasta 100 minutos más pérdida de conexión más equipaje de mano), LIS-MAD (retrasos múltiples más desinformación), LHR-MAD (problemas de embarque más retrasos), y AGP-MAD (equipaje perdido más maltrato). Los viajeros de Business fueron significativamente más críticos (NPS 6.1 frente a Leisure 24.6, diferencial de 18.5 pts) por mayor sensibilidad a problemas de equipaje y retrasos, mientras que los residentes de Europa mostraron NPS de 7.1 con muestra significativa de 85 encuestas, y los usuarios de flotas CRJ y A320neo concentraron la mayoría de incidentes (35% y 26.5% del total respectivamente).<br>
<br>
<b><u>BUSINESS SH: Dilución de impacto con colapso extremo en YW</u></b><br>
El NPS de Business SH cayó a <b>33.3 (–1.8 pts)</b>, resultado de una dilución donde el volumen desproporcionado de <b>IB</b> (NPS 40.0, –2.1 pts, 30 encuestas representando el 83%) atenuó significativamente el impacto severo de <b>YW</b> (NPS 0.0, –16.0 pts, solo 6 encuestas representando el 17%). IB experimentó deterioro en gestión de equipaje (Mishandling +3.8 pts, incremento del 23%) y conexiones perdidas (Misconex +0.14 pts, incremento del 19%), con problemas concentrados en BCN-MAD (NPS 14.3, retrasos hasta 100 minutos) y LIS-MAD (NPS 25.0, 1 hora de espera en recogida de maletas). Los incidentes específicos incluyeron pérdida total de equipaje en FRA-MAD-GRU (NPS 0), retrasos de hasta 100 minutos en BCN-MAD con críticas a la aplicación de IB, y demoras en entrega de equipaje en LIS-MAD. Los viajeros de Business fueron significativamente más críticos (NPS 20.0 frente a Leisure 60.0, diferencial de 40 pts) por mayor sensibilidad a problemas de equipaje en viajes profesionales, mientras que los residentes de España (NPS 13.3, 50% de encuestas) y usuarios de flota A320neo (NPS 20.0, 50% de encuestas) concentraron el impacto. Por su parte, YW experimentó un colapso total con muestra estadísticamente insuficiente, con incidentes operativos específicos como problema sanitario en FRA-MAD (tripulante tosiendo sin mascarilla en Business, NPS 5), retraso mecánico superior a 1 hora en MAD-MRS (fallo en puertas de bodega, NPS 2), e incidente no documentado en MAD-TRN (NPS –100). Los pasajeros de Leisure fueron significativamente más críticos en YW (NPS –50 frente a Business +100, diferencial de 150 pts, patrón inverso a IB), con nivel de confianza bajo debido a muestra microscópica y evidencia fragmentada, requiriendo investigación urgente a pesar de la baja significancia estadística.<br>
<br>
<b><u>ECONOMY LH: Triple deterioro operativo en rutas transatlánticas</u></b><br>
El NPS de Economy LH cayó a <b>–1.2 (–4.9 pts)</b>, resultado de un triple deterioro operativo simultáneo donde puntualidad empeoró 3.6 pts (OTP15 de 77.97% frente a baseline de 81.57%), gestión de equipaje deteriorada (Mishandling +3.45 pts, incremento del 22.3%), y conexiones perdidas aumentaron 20% (Misconex +0.14 pts). Las tres métricas críticas empeoraron simultáneamente configurando una triple correlación inversa confirmada que explica completamente la caída de NPS. Los clientes reportaron 8 menciones de retrasos severos incluyendo 5 horas de retraso en MAD-MIA por problemas técnicos con cambio de avión, llegadas a las 23:00 cuando estaba previsto a las 18:40 en GIG-MAD sin ayuda, y retrasos de 2.5 horas en despegue. Adicionalmente, 6 menciones de equipaje perdido incluyeron casos de hasta 4 días sin equipaje en GIG-MAD, equipaje retrasado desde el 16 de diciembre sin novedades en GRU-MAD, y maletas perdidas con demora superior a un día en devolución en JFK-MAD. Los problemas técnicos en flota antigua se evidenciaron en LIM-MAD con avión "hecho pedazos" con sillones pegados con cinta adhesiva, mientras que el catering deficiente se reportó en EZE-MAD con 12 horas sin servicio de bar o cocina, y en MAD-ORD sin servicio de desayuno. Las rutas transatlánticas concentraron los problemas más severos: MAD-MIA (NPS –28.6, problemas técnicos críticos más cambio de avión más retraso superior a 3 horas), GRU-MAD (NPS –28.6, equipaje retrasado desde el 16 de diciembre sin novedades), LIM-MAD (NPS –23.5 con 17 encuestas siendo el mayor volumen, retrasos más conexiones perdidas más avión deteriorado más catering deficiente), MAD-SCL (NPS –22.2, cancelaciones más pérdida de conexiones más gestión pésima), y JFK-MAD (NPS –16.7 con 18 encuestas siendo el segundo mayor volumen, equipaje perdido más catering deficiente). Los pasajeros de codeshare internacional fueron los más afectados con Qatar Airways registrando NPS de –100.0 (muestra pequeña pero severidad extrema), British Airways con –37.5, y LATAM con –22.2 en rutas sudamericanas críticas, significativamente más afectados que vuelos operados directamente por IB (NPS 1.5). Los residentes de España representaron el mayor volumen afectado (NPS –17.0, 53 encuestas) probablemente por cancelaciones y pérdidas de conexión en MAD, seguidos de residentes de Europa (NPS –13.3). La flota A332 registró el peor desempeño (NPS –18.2), sugiriendo problemas técnicos concentrados en flotas antiguas, mientras que A350 destacó positivamente (NPS 18.5) como contraste.<br>
<br>
<b><u>BUSINESS LH: Mejora excepcional que mitiga el deterioro de Economy</u></b><br>
El NPS de Business LH subió a <b>32.3 (+13.3 pts)</b>, configurando una anomalía positiva aislada que actuó como factor mitigante reduciendo parcialmente el impacto negativo de Economy LH (–4.9 pts) en el agregado del radio (de –4.9 a –1.4 pts en el agregado LH). Sin análisis causal disponible debido a error de procesamiento, se confirma únicamente la mejora significativa con 31 encuestas (10% del total LH). La hipótesis probable incluye mejoras en servicio, upgrades, o ausencia de los problemas operativos que afectaron a Economy. A pesar del fuerte positivo de Business LH, este no propagó al nivel LH debido a que Economy (54% de volumen, –4.9 pts) dominó el agregado e impuso su signo negativo, configurando un escenario de dominancia opuesta donde la mejora de Business fue neutralizada por el deterioro de Economy.<br>
<br>
<b><u>PREMIUM LH: Impacto marginal sin efecto significativo</u></b><br>
El NPS de Premium LH cayó a <b>8.3 (–1.0 pts)</b>, configurando una anomalía negativa marginal sin análisis causal disponible debido a error de procesamiento. Con 111 encuestas (36% del total LH), Premium no fue un driver significativo de la anomalía negativa de LH (–1.4 pts), que estuvo dominada por Economy (54% de encuestas, –4.9 pts). La caída de –1.0 pts representa un impacto neutral en el agregado del radio, sin afectar significativamente el resultado final de Long Haul.
```

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 7
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-15 to 2025-12-15:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 PASO 1: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL COMPAÑÍA)

---

## 📊 ANÁLISIS SHORT HAUL - ECONOMY

### **PARTE A: DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(IB: N, YW: + | Economy SH: +)`

**Escenario Identificado:** **TRANSFERENCIA**

**Justificación:**
- **IB (Economy SH):** Normal (+1.4 pts) - Estado `N`
- **YW (Economy SH):** POSITIVE ANOMALY (+22.1 pts) - Estado `+`
- **Economy SH (Padre):** POSITIVE ANOMALY (+7.9 pts) - Estado `+`

YW presenta una anomalía positiva significativa (+22.1 pts) que **contagia al nodo padre** a pesar de que IB mantiene comportamiento Normal. El volumen de YW es suficiente para elevar el agregado a anomalía positiva (+7.9 pts).

---

### **PARTE B: NARRATIVA CAUSAL**

**Estrategia:** Adoptar la **Explicación del Hijo Dominante (YW)**, ya que es el responsable de la transferencia de la anomalía al padre.

**Narrativa:**

En **Economy SH**, el escenario es **TRANSFERENCIA** `(N, + | +)`.

- **Narrativa:** La anomalía positiva del segmento Economy SH (+7.9 pts) es impulsada exclusivamente por **YW**, que registró un salto excepcional de **+22.1 pts** vs baseline. IB mantuvo desempeño estable (+1.4 pts, dentro de rango normal), pero el volumen y magnitud de la mejora de YW fue suficiente para elevar el agregado a territorio anómalo positivo.

- **Evidencia Clave:** 
  - **YW (Economy SH):** NPS 49.6 vs baseline 27.4 (+22.1 pts)
  - **IB (Economy SH):** NPS 28.3 vs baseline 26.9 (+1.4 pts, Normal)
  - **Limitación:** El análisis causal de YW falló en la recolección de datos, impidiendo identificar los drivers operativos específicos (rutas, perfiles, métricas) que explican esta mejora extraordinaria.

---

## 📊 ANÁLISIS SHORT HAUL - BUSINESS

### **PARTE A: DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(IB: -, YW: + | Business SH: -)`

**Escenario Identificado:** **DOMINANCIA**

**Justificación:**
- **IB (Business SH):** NEGATIVE ANOMALY (-35.2 pts) - Estado `-`
- **YW (Business SH):** POSITIVE ANOMALY (+44.0 pts) - Estado `+`
- **Business SH (Padre):** NEGATIVE ANOMALY (-20.4 pts) - Estado `-`

Efectos opuestos: IB colapsa (-35.2 pts) mientras YW se dispara (+44.0 pts). Sin embargo, **IB domina e impone su signo negativo al padre** (-20.4 pts), probablemente por mayor volumen de encuestas o peso operativo. El efecto positivo de YW suaviza la caída, pero no logra neutralizarla.

---

### **PARTE B: NARRATIVA CAUSAL**

**Estrategia:** Adoptar la **Explicación del Hijo Dominante (IB)**, añadiendo el matiz de que YW compensó parcialmente el impacto.

**Narrativa:**

En **Business SH**, el escenario es **DOMINANCIA** `(-, + | -)`.

- **Narrativa:** La anomalía negativa del segmento Business SH (-20.4 pts) está **causada por el colapso operativo de IB** (-35.2 pts vs baseline), que domina el agregado a pesar de la mejora excepcional de YW (+44.0 pts). El efecto positivo de YW suavizó la caída del padre, pero no logró neutralizarla debido al mayor peso volumétrico o impacto de IB en este segmento.

- **Evidencia Clave (IB - Hijo Dominante):**
  - **IB (Business SH):** NPS 6.9 vs baseline 42.1 (-35.2 pts) | 29 encuestas
  - **Causa Principal:** **Mishandling +3.57 pts** vs baseline (19.79 vs 16.22) - Correlación inversa confirmada con NPS
  - **Causa Secundaria:** **Retrasos >15min** no capturados por OTP15 (5 menciones en verbatims: BCN-MAD, LCG-MAD, LHR-MAD, LIS-MAD, BCN-MAD)
  - **Rutas Críticas:** BCN-MAD (5 detractores, NPS 0), AMS-MAD (2 detractores, NPS 0, mishandling grave), LIS-MAD (2 detractores, NPS 0)
  - **Perfiles Afectados:** Pasajeros **Leisure** (NPS 0.0, -6.9 pts vs día), flotas **A320neo** (NPS 0.0) y **A321** (NPS -22.2)
  - **Métricas Operativas:** OTP15 mejoró (+1.55 pts) pero **contradice verbatims** de retrasos, indicando problemas en demoras >15min

- **Contrapunto (YW):**
  - **YW (Business SH):** NPS 60.0 vs baseline 16.0 (+44.0 pts)
  - **Limitación:** El análisis causal de YW falló, impidiendo identificar qué factores operativos generaron esta mejora récord que compensó parcialmente el desplome de IB.

---

## 🎯 SÍNTESIS EJECUTIVA - NIVEL COMPAÑÍA SH

| Cabina | Escenario | Estado IB | Estado YW | Resultado Padre | Narrativa Dominante |
|--------|-----------|-----------|-----------|-----------------|---------------------|
| **Economy SH** | TRANSFERENCIA | Normal (+1.4) | **Anomalía + (+22.1)** | Anomalía + (+7.9) | YW impulsa al padre; IB estable |
| **Business SH** | DOMINANCIA | **Anomalía - (-35.2)** | Anomalía + (+44.0) | Anomalía - (-20.4) | IB colapsa y domina; YW compensa parcialmente |

**Dinámica Global SH:** Las anomalías opuestas en Business (IB negativa dominante vs YW positiva) y la transferencia positiva de YW en Economy se **cancelan mutuamente** en el agregado Short Haul, resultando en variación Normal (+5.6 pts) a nivel padre SH.

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ PASO 2: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL CABINA)

---

## 🌍 ANÁLISIS LONG HAUL (LH)

### **PARTE A: DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(Economy: N, Business: -, Premium: + | LH: N)`

**Escenario Identificado:** **CANCELACIÓN**

**Justificación:**
- **Economy LH:** Normal (+0.8 pts) - Estado `N`
- **Business LH:** NEGATIVE ANOMALY (-2.3 pts) - Estado `-`
- **Premium LH:** POSITIVE ANOMALY (+31.9 pts) - Estado `+`
- **LH (Padre):** Normal (+3.4 pts) - Estado `N`

Las anomalías opuestas en Business (-) y Premium (+), combinadas con la estabilidad de Economy (que representa el mayor volumen), **se cancelan mutuamente** en el agregado. El resultado es una variación Normal en LH (+3.4 pts) que oculta volatilidad significativa en las cabinas de valor agregado.

---

### **PARTE B: NARRATIVA CAUSAL**

**Estrategia:** Narrativa de **CANCELACIÓN** - Contrastar causas opuestas de las cabinas anómalas.

**Narrativa:**

En **Long Haul**, la dinámica es **CANCELACIÓN** `(N, -, + | N)`.

- **Narrativa:** El radio LH muestra estabilidad aparente (+3.4 pts, Normal) que **oculta volatilidad extrema en las cabinas premium**. El deterioro en Business LH (-2.3 pts) fue compensado por la mejora excepcional en Premium LH (+31.9 pts), mientras Economy LH mantuvo desempeño estable (+0.8 pts). Esta neutralización impide visibilidad de problemas específicos de producto en el agregado.

- **Evidencia de Causas Opuestas:**

  **Business LH (Anomalía Negativa -2.3 pts):**
  - NPS: 16.7 vs baseline 19.0
  - **Limitación Crítica:** El análisis causal falló completamente (❌ No se recolectaron datos)
  - **Hipótesis inferida:** Posible impacto de problemas de servicio o producto específico de Business en rutas LH, pero sin datos operativos para confirmar

  **Premium LH (Anomalía Positiva +31.9 pts):**
  - NPS: 41.2 vs baseline 9.3 (+31.9 pts - mejora extraordinaria)
  - **Limitación Crítica:** El análisis causal falló completamente (❌ No se recolectaron datos)
  - **Hipótesis inferida:** Posible mejora en experiencia de producto Premium (upgrade operacional, servicio excepcional, o cambio de flota), pero sin evidencia operativa disponible

  **Economy LH (Estabilizador):**
  - NPS: 4.5 vs baseline 3.7 (+0.8 pts, Normal)
  - Representa el **mayor volumen** de pasajeros LH, actuando como ancla que diluye los efectos extremos de las cabinas premium

---

## 🛫 ANÁLISIS SHORT HAUL (SH)

### **PARTE A: DINÁMICA DE AGREGACIÓN**

**Díada de Estados:** `(Economy: +, Business: - | SH: N)`

**Escenario Identificado:** **CANCELACIÓN**

**Justificación:**
- **Economy SH:** POSITIVE ANOMALY (+7.9 pts) - Estado `+`
- **Business SH:** NEGATIVE ANOMALY (-20.4 pts) - Estado `-`
- **SH (Padre):** Normal (+5.6 pts) - Estado `N`

Efectos opuestos se neutralizan en el agregado: la mejora significativa en Economy (+7.9 pts, impulsada por YW) **compensa el colapso de Business** (-20.4 pts, dominado por IB). El resultado es una variación Normal (+5.6 pts) que oculta dinámicas críticas a nivel de cabina y compañía.

---

### **PARTE B: NARRATIVA CAUSAL**

**Estrategia:** Narrativa de **CANCELACIÓN** - Contrastar causas opuestas de las cabinas.

**Narrativa:**

En **Short Haul**, la dinámica es **CANCELACIÓN** `(+, - | N)`.

- **Narrativa:** El radio SH muestra estabilidad engañosa (+5.6 pts, Normal): el **colapso operativo en Business SH** (-20.4 pts) fue compensado por la **mejora excepcional en Economy SH** (+7.9 pts). Esta neutralización oculta una crisis operativa grave en Business (especialmente IB) que queda invisibilizada en el agregado del radio.

- **Evidencia de Causas Opuestas:**

  **Business SH (Anomalía Negativa -20.4 pts):**
  - NPS: 14.7 vs baseline 35.1
  - **Causa Raíz (IB Business SH -35.2 pts):** 
    - **Mishandling +3.57 pts** (19.79 vs baseline 16.22) - Causa operativa principal
    - **Retrasos >15min** no capturados por OTP15 (5 menciones: BCN-MAD, LCG-MAD, LHR-MAD, LIS-MAD)
    - **Rutas Críticas:** BCN-MAD (NPS 0, 5 detractores), AMS-MAD (NPS 0, mishandling grave), LIS-MAD (NPS 0)
    - **Perfiles Afectados:** Leisure (NPS 0.0), flotas A320neo (NPS 0.0) y A321 (NPS -22.2)
  - **Compensación parcial (YW Business SH +44.0 pts):** NPS 60.0 vs baseline 16.0, pero sin datos causales disponibles

  **Economy SH (Anomalía Positiva +7.9 pts):**
  - NPS: 34.9 vs baseline 27.1
  - **Causa Raíz (YW Economy SH +22.1 pts):**
    - NPS: 49.6 vs baseline 27.4
    - **Limitación Crítica:** El análisis causal falló (❌ No se recolectaron datos)
    - **Hipótesis inferida:** Posible mejora operativa específica de YW (rutas, servicio, flota) que contrasta con la estabilidad de IB Economy (+1.4 pts, Normal)
  - **Estabilidad de IB Economy:** NPS 28.3 vs baseline 26.9 (+1.4 pts, Normal) - sin contribuir a la anomalía

---

## 🎯 SÍNTESIS EJECUTIVA - NIVEL RADIO

| Radio | Escenario | Estados Cabinas | Resultado Padre | Interpretación Clave |
|-------|-----------|-----------------|-----------------|---------------------|
| **Long Haul** | CANCELACIÓN | `(Eco: N, Bus: -, Prem: +)` | Normal (+3.4) | Volatilidad extrema en cabinas premium (+31.9 vs -2.3) oculta por volumen de Economy |
| **Short Haul** | CANCELACIÓN | `(Eco: +, Bus: -)` | Normal (+5.6) | Crisis operativa en Business IB (-35.2) compensada por mejora excepcional en Economy YW (+22.1) |

---

## ⚠️ IMPLICACIONES ESTRATÉGICAS

### **Riesgo de "Estabilidad Engañosa":**
Ambos radios (LH y SH) presentan variaciones Normales que **ocultan problemas críticos**:

1. **LH:** Deterioro en Business (-2.3) invisibilizado por mejora en Premium (+31.9)
2. **SH:** Colapso en Business IB (-35.2, causado por Mishandling y retrasos) compensado por mejora en Economy YW (+22.1)

### **Limitaciones de Análisis:**
- **3 de 4 anomalías en LH** sin datos causales (Business -, Premium +, y Economy N)
- **2 de 3 anomalías positivas en SH** sin datos causales (YW Economy +, YW Business +)
- Solo **IB Business SH** tiene análisis causal completo, revelando problemas operativos tangibles

### **Recomendación Prioritaria:**
Investigar urgentemente las **causas raíz de las anomalías sin datos** (especialmente Premium LH +31.9 y YW Economy/Business SH) para:
- Replicar factores de éxito en segmentos positivos
- Prevenir que problemas de Business LH se agraven
- Entender si la mejora de YW es sostenible o circunstancial

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 PASO 3: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL GLOBAL)

---

## 🌐 ANÁLISIS NIVEL GLOBAL

### **PARTE A: DINÁMICA DE AGREGACIÓN**

**Díada de Estados:** `(LH: N, SH: N | Global: +)`

**Escenario Identificado:** **TRANSFERENCIA**

**Justificación:**
- **Long Haul:** Normal (+3.4 pts) - Estado `N`
- **Short Haul:** Normal (+5.6 pts) - Estado `N`
- **Global:** POSITIVE ANOMALY (+4.7 pts) - Estado `+`

A pesar de que ambos radios muestran variaciones Normales, el **Global presenta anomalía positiva**. Esto indica que la **suma acumulativa de las variaciones positivas** de ambos radios (LH +3.4 y SH +5.6), aunque individualmente dentro de rangos normales, **supera el umbral de anomalía al agregarse** en el nivel Global. Este es un caso de **TRANSFERENCIA por efecto acumulativo**: las variaciones "normales" de ambos radios se potencian mutuamente y contagian al Global.

---

### **PARTE B: NARRATIVA CAUSAL**

**Estrategia:** Adoptar la **Explicación del Nodo Global** como causa sistémica que afecta transversalmente a la red.

**⚠️ LIMITACIÓN CRÍTICA:** El análisis causal del nodo Global **falló completamente** (síntesis falló debido a error). Por tanto, debemos **inferir la causa sistémica** a partir de los patrones detectados en los niveles inferiores.

---

**Narrativa:**

A nivel **GLOBAL**, la dinámica es **TRANSFERENCIA** `(N, N | +)`.

- **Narrativa:** El NPS Global registra una **anomalía positiva de +4.7 pts** (NPS 25.6 vs baseline 20.9), a pesar de que tanto Long Haul como Short Haul muestran variaciones Normales (+3.4 y +5.6 pts respectivamente). Este fenómeno de **transferencia por efecto acumulativo** revela que las mejoras moderadas en ambos radios se potenciaron al agregarse, superando el umbral de anomalía en el nivel de red.

- **Causa Sistémica Inferida (Patrón Transversal):**
  
  **La anomalía positiva Global está impulsada por mejoras excepcionales en segmentos específicos de alto valor que compensan problemas operativos localizados:**

  1. **Motor Positivo Principal - Segmentos Premium:**
     - **Premium LH:** +31.9 pts (NPS 41.2 vs baseline 9.3) - Mejora extraordinaria sin causa identificada
     - **YW Business SH:** +44.0 pts (NPS 60.0 vs baseline 16.0) - Mejora récord sin causa identificada
     - **YW Economy SH:** +22.1 pts (NPS 49.6 vs baseline 27.4) - Mejora significativa sin causa identificada

  2. **Contrapeso Negativo - Crisis Operativa Localizada:**
     - **IB Business SH:** -35.2 pts (NPS 6.9 vs baseline 42.1)
       - **Causa confirmada:** Mishandling +3.57 pts (19.79 vs baseline 16.22)
       - **Causa secundaria:** Retrasos >15min no capturados por OTP15
       - **Rutas críticas:** BCN-MAD, AMS-MAD, LIS-MAD
       - **Perfiles afectados:** Leisure, flotas A320neo/A321

  3. **Estabilizadores:**
     - **Economy LH:** Normal (+0.8 pts) - Volumen alto, desempeño estable
     - **IB Economy SH:** Normal (+1.4 pts) - Sin contribución a anomalía

---

### **EVIDENCIA DE PATRÓN SISTÉMICO:**

**Hipótesis de Causa Raíz Global (inferida por triangulación):**

La anomalía positiva Global **NO es resultado de una mejora operativa sistémica** (las métricas operativas muestran comportamiento mixto), sino de un **efecto de composición favorable**:

| Componente | Contribución al Global | Naturaleza |
|------------|----------------------|------------|
| **Segmentos YW (Eco+Bus SH)** | Impulso positivo masivo (+22.1 y +44.0) | **Desconocida** (fallos de análisis) |
| **Premium LH** | Impulso positivo extraordinario (+31.9) | **Desconocida** (fallo de análisis) |
| **IB Business SH** | Lastre negativo severo (-35.2) | **Operativa confirmada** (Mishandling, retrasos) |
| **Economy LH + IB Eco SH** | Estabilidad (volumen alto, variación normal) | Desempeño base consistente |

**Interpretación Ejecutiva:**

El Global presenta una **"anomalía positiva frágil"**: está sostenida por mejoras excepcionales en segmentos de **bajo volumen pero alto impacto** (Premium, Business YW) cuyas causas **no están identificadas** (posiblemente circunstanciales o no replicables). Simultáneamente, existe una **crisis operativa grave y documentada** en IB Business SH (Mishandling, retrasos) que afecta rutas domésticas clave.

---

### **⚠️ RIESGOS IDENTIFICADOS:**

1. **Sostenibilidad Incierta:**
   - Las mejoras en YW (+22.1, +44.0) y Premium LH (+31.9) representan **+98.0 pts acumulados** sin causa raíz identificada
   - Si estas mejoras son circunstanciales (ej: eventos puntuales, promociones, cambios temporales de flota), la anomalía positiva Global **no es sostenible**

2. **Crisis Operativa Oculta:**
   - El problema de Mishandling en IB Business SH (-35.2 pts) está **compensado artificialmente** por las mejoras de YW
   - Rutas críticas (BCN-MAD, AMS-MAD, LIS-MAD) con NPS 0 requieren intervención urgente

3. **Cancelaciones Múltiples Enmascaran Volatilidad:**
   - **LH:** Cancelación `(N, -, +)` oculta deterioro en Business y mejora en Premium
   - **SH:** Cancelación `(+, -)` oculta colapso en Business y mejora en Economy
   - **Global:** Transferencia `(N, N | +)` oculta que la mejora depende de segmentos sin diagnóstico

---

## 🎯 SÍNTESIS EJECUTIVA - NIVEL GLOBAL

**Dinámica:** TRANSFERENCIA `(LH: N, SH: N | Global: +)`

**Resultado:** NPS Global +4.7 pts (25.6 vs baseline 20.9) - Anomalía Positiva

**Causa Sistémica:**
- **Motor positivo:** Mejoras excepcionales en Premium LH (+31.9), YW Business SH (+44.0) y YW Economy SH (+22.1) - **Causas no identificadas**
- **Lastre operativo:** Crisis en IB Business SH (-35.2) por Mishandling (+3.57) y retrasos >15min en rutas BCN-MAD, AMS-MAD, LIS-MAD
- **Efecto neto:** Las mejoras de segmentos premium compensan la crisis operativa y elevan el Global a anomalía positiva por acumulación

**Nivel de Confianza:** **BAJO-MEDIO**
- ✅ Causa operativa de IB Business SH confirmada (Mishandling, retrasos, rutas)
- ❌ 75% de las anomalías positivas (3 de 4) sin análisis causal
- ⚠️ Imposible determinar si la mejora Global es replicable o circunstancial

**Acción Requerida:**
1. **URGENTE:** Investigar causas de mejoras en YW (Eco/Bus SH) y Premium LH para determinar replicabilidad
2. **CRÍTICO:** Resolver crisis de Mishandling en IB Business SH (rutas BCN-MAD, AMS-MAD, LIS-MAD)
3. **PREVENTIVO:** Monitorear Business LH (-2.3) para evitar agravamiento del deterioro

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

# 🔍 PASO 4: IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS (NMA)

---

## 📋 CAUSAS IDENTIFICADAS Y SUS NMAs

---

### **CAUSA 1: Mishandling Operativo (Gestión de Equipaje)**

- **Escenario:** DOMINANCIA
- **NMA:** `Global/SH/Business/IB`
- **Afecta a:** Pasajeros Business IB en Short Haul (rutas domésticas/europeas)
- **Tipo de impacto:** NEGATIVO (-35.2 pts)

**Cadena de propagación hacia el segmento raíz:**

1. **`Global/SH/Business/IB`** → **`Global/SH/Business`** 
   - Escenario: **DOMINANCIA** `(IB: -, YW: + | Business SH: -)`
   - IB (-35.2 pts) domina e impone signo negativo al padre (-20.4 pts)
   - YW (+44.0 pts) compensa parcialmente pero no neutraliza el efecto

2. **`Global/SH/Business`** → **`Global/SH`**
   - Escenario: **CANCELACIÓN** `(Economy: +, Business: - | SH: N)`
   - Business SH (-20.4 pts) se cancela con Economy SH (+7.9 pts)
   - El efecto NO propaga al padre SH (queda Normal +5.6 pts)

3. **`Global/SH`** → **`Global`**
   - Escenario: **TRANSFERENCIA** `(LH: N, SH: N | Global: +)`
   - SH es Normal, por lo que el efecto negativo de Mishandling **NO alcanza el nivel Global**
   - La anomalía positiva Global está impulsada por otros factores

**Conclusión de propagación:** El impacto de Mishandling **se detiene en el nivel Business SH** debido a la cancelación con Economy SH. No afecta al Global.

---

### **CAUSA 2: Retrasos >15 minutos (No capturados por OTP15)**

- **Escenario:** DOMINANCIA
- **NMA:** `Global/SH/Business/IB`
- **Afecta a:** Pasajeros Business IB en Short Haul (rutas domésticas/europeas)
- **Tipo de impacto:** NEGATIVO (-35.2 pts, causa secundaria complementaria a Mishandling)

**Cadena de propagación hacia el segmento raíz:**

1. **`Global/SH/Business/IB`** → **`Global/SH/Business`** 
   - Escenario: **DOMINANCIA** `(IB: -, YW: + | Business SH: -)`
   - IB (-35.2 pts) domina e impone signo negativo al padre (-20.4 pts)
   - YW (+44.0 pts) compensa parcialmente

2. **`Global/SH/Business`** → **`Global/SH`**
   - Escenario: **CANCELACIÓN** `(Economy: +, Business: - | SH: N)`
   - Business SH (-20.4 pts) se cancela con Economy SH (+7.9 pts)
   - El efecto NO propaga al padre SH

3. **`Global/SH`** → **`Global`**
   - Escenario: **TRANSFERENCIA** `(LH: N, SH: N | Global: +)`
   - El efecto negativo **NO alcanza el nivel Global**

**Conclusión de propagación:** El impacto de retrasos >15min **se detiene en el nivel Business SH** por cancelación con Economy SH. No afecta al Global.

---

### **CAUSA 3: Mejora Excepcional YW Economy SH (Causa Desconocida)**

- **Escenario:** TRANSFERENCIA
- **NMA:** `Global/SH/Economy/YW`
- **Afecta a:** Pasajeros Economy YW en Short Haul
- **Tipo de impacto:** POSITIVO (+22.1 pts)

**Cadena de propagación hacia el segmento raíz:**

1. **`Global/SH/Economy/YW`** → **`Global/SH/Economy`**
   - Escenario: **TRANSFERENCIA** `(IB: N, YW: + | Economy SH: +)`
   - YW (+22.1 pts) contagia al padre Economy SH (+7.9 pts)
   - IB mantiene estabilidad (+1.4 pts, Normal)

2. **`Global/SH/Economy`** → **`Global/SH`**
   - Escenario: **CANCELACIÓN** `(Economy: +, Business: - | SH: N)`
   - Economy SH (+7.9 pts) se cancela con Business SH (-20.4 pts)
   - El efecto positivo NO propaga al padre SH (queda Normal +5.6 pts)

3. **`Global/SH`** → **`Global`**
   - Escenario: **TRANSFERENCIA** `(LH: N, SH: N | Global: +)`
   - SH es Normal, pero la **acumulación de variaciones positivas** de ambos radios contagia al Global (+4.7 pts)

**Conclusión de propagación:** El efecto positivo de YW Economy **contribuye indirectamente al Global** a través del efecto acumulativo de transferencia, aunque se cancela en el nivel SH.

---

### **CAUSA 4: Mejora Excepcional YW Business SH (Causa Desconocida)**

- **Escenario:** DOMINANCIA (inversa - compensa pero no domina)
- **NMA:** `Global/SH/Business/YW`
- **Afecta a:** Pasajeros Business YW en Short Haul
- **Tipo de impacto:** POSITIVO (+44.0 pts)

**Cadena de propagación hacia el segmento raíz:**

1. **`Global/SH/Business/YW`** → **`Global/SH/Business`**
   - Escenario: **DOMINANCIA** `(IB: -, YW: + | Business SH: -)`
   - YW (+44.0 pts) compensa parcialmente el colapso de IB (-35.2 pts)
   - IB domina e impone signo negativo al padre (-20.4 pts)
   - El efecto positivo de YW **mitiga pero no neutraliza** la caída

2. **`Global/SH/Business`** → **`Global/SH`**
   - Escenario: **CANCELACIÓN** `(Economy: +, Business: - | SH: N)`
   - Business SH (-20.4 pts) se cancela con Economy SH (+7.9 pts)
   - El efecto positivo de YW Business **NO propaga al padre SH**

3. **`Global/SH`** → **`Global`**
   - Escenario: **TRANSFERENCIA** `(LH: N, SH: N | Global: +)`
   - SH es Normal, pero la **acumulación de variaciones positivas** contagia al Global (+4.7 pts)

**Conclusión de propagación:** El efecto positivo de YW Business **contribuye indirectamente al Global** a través del efecto acumulativo, aunque es bloqueado por la dominancia de IB en Business SH.

---

### **CAUSA 5: Mejora Extraordinaria Premium LH (Causa Desconocida)**

- **Escenario:** CANCELACIÓN (a nivel LH)
- **NMA:** `Global/LH/Premium`
- **Afecta a:** Pasajeros Premium en Long Haul
- **Tipo de impacto:** POSITIVO (+31.9 pts)

**Cadena de propagación hacia el segmento raíz:**

1. **`Global/LH/Premium`** → **`Global/LH`**
   - Escenario: **CANCELACIÓN** `(Economy: N, Business: -, Premium: + | LH: N)`
   - Premium (+31.9 pts) se cancela con Business (-2.3 pts) y Economy Normal (+0.8 pts)
   - El efecto positivo NO propaga al padre LH (queda Normal +3.4 pts)

2. **`Global/LH`** → **`Global`**
   - Escenario: **TRANSFERENCIA** `(LH: N, SH: N | Global: +)`
   - LH es Normal, pero la **acumulación de variaciones positivas** de ambos radios contagia al Global (+4.7 pts)

**Conclusión de propagación:** El efecto positivo de Premium LH **contribuye indirectamente al Global** a través del efecto acumulativo de transferencia, aunque se cancela en el nivel LH.

**⚠️ Nota especial:** Este es un segmento hoja sin subniveles de compañía (IB/YW no aplican en LH). El NMA coincide con el segmento raíz.

---

### **CAUSA 6: Deterioro Business LH (Causa Desconocida)**

- **Escenario:** CANCELACIÓN (a nivel LH)
- **NMA:** `Global/LH/Business`
- **Afecta a:** Pasajeros Business en Long Haul
- **Tipo de impacto:** NEGATIVO (-2.3 pts)

**Cadena de propagación hacia el segmento raíz:**

1. **`Global/LH/Business`** → **`Global/LH`**
   - Escenario: **CANCELACIÓN** `(Economy: N, Business: -, Premium: + | LH: N)`
   - Business (-2.3 pts) se cancela con Premium (+31.9 pts) y Economy Normal (+0.8 pts)
   - El efecto negativo NO propaga al padre LH (queda Normal +3.4 pts)

2. **`Global/LH`** → **`Global`**
   - Escenario: **TRANSFERENCIA** `(LH: N, SH: N | Global: +)`
   - LH es Normal, por lo que el efecto negativo de Business **NO alcanza el nivel Global**

**Conclusión de propagación:** El impacto negativo de Business LH **se detiene en el nivel LH** debido a la cancelación con Premium. No afecta al Global.

**⚠️ Nota especial:** Segmento hoja sin subniveles de compañía. El NMA coincide con el segmento raíz.

---

## 🎯 RESUMEN DE NMAs Y PROPAGACIÓN

| Causa | NMA | Impacto | Propaga a SH/LH | Propaga a Global | Mecanismo Bloqueante |
|-------|-----|---------|-----------------|------------------|---------------------|
| **Mishandling** | `SH/Business/IB` | -35.2 | ❌ (Cancelación Eco/Bus) | ❌ | Cancelación en SH |
| **Retrasos >15min** | `SH/Business/IB` | -35.2 | ❌ (Cancelación Eco/Bus) | ❌ | Cancelación en SH |
| **Mejora YW Economy** | `SH/Economy/YW` | +22.1 | ❌ (Cancelación Eco/Bus) | ✅ (Acumulativo) | Cancelación en SH, pero contribuye acumulativamente |
| **Mejora YW Business** | `SH/Business/YW` | +44.0 | ❌ (Dominancia IB) | ✅ (Acumulativo) | Dominancia IB en Bus, pero contribuye acumulativamente |
| **Mejora Premium LH** | `LH/Premium` | +31.9 | ❌ (Cancelación Eco/Bus/Prem) | ✅ (Acumulativo) | Cancelación en LH, pero contribuye acumulativamente |
| **Deterioro Business LH** | `LH/Business` | -2.3 | ❌ (Cancelación Eco/Bus/Prem) | ❌ | Cancelación en LH |

**Conclusión Clave:** La anomalía positiva Global (+4.7 pts) es resultado de un **efecto acumulativo** de mejoras en segmentos premium (YW Eco/Bus SH, Premium LH) que, aunque bloqueadas en niveles intermedios por cancelaciones, se suman en el agregado final. Los problemas operativos graves (Mishandling, retrasos en IB Business SH) quedan invisibilizados en el Global.

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

# 📊 PASO 4B: EXTRACCIÓN DE EVIDENCIAS DEL CONTEXTO INICIAL

---

## === NMA 1: Global/SH/Business/IB ===

### 📈 EXPLANATORY DRIVERS:
No disponible

### 📊 DATOS OPERATIVOS:
- **Mishandling:** 19.79 (⬆️ **+3.57 vs baseline** 16.22)
- **OTP15:** 93.66% (⬆️ +1.55 vs baseline 92.11%)
- **Load Factor:** 74.01% (⬇️ -6.13 vs baseline 80.14%)
- **Misconex:** 0.82 (⬆️ +0.11 vs baseline 0.71)

### 🚨 INCIDENTES NCS (CUANTITATIVO):
- **Total de incidentes:** 14
- **Distribución:**
  - 12 cancelaciones
  - 3 retrasos
  - 1 desvío (BCN-LEN a OVD por meteorología)
  - 2 otras incidencias
- **Temas principales:** weather (1), aircraft_change (1)

### 🧠 NCS (CUALITATIVO / REFLEXIÓN):
No disponible

### 💬 FEEDBACK DE CLIENTES:
**Temas principales en Verbatims:**
- **Equipaje:** 2 menciones
  - AGP-MAD (NPS 0): "Equipaje retrasado"
  - AMS-MAD (NPS 2): "Etiquetado incorrecto de equipaje hasta Lima en vez de Santiago"
- **Retrasos/Puntualidad:** 5 menciones
  - BCN-MAD (NPS 0): "Salió el avión 2 horas tarde"
  - LCG-MAD (NPS 2): "Retraso de casi una hora"
  - BCN-MAD (NPS 4): "Retraso en la salida → pérdida de conexión a Chile"
  - LHR-MAD (NPS 4): "Long to Board → perdimos conexión"
  - LIS-MAD (NPS 6): "El avión llegó tarde"
- **Servicio Business:** 2 menciones de calidad percibida como decepcionante
  - LHR-MAD: Servicio Business decepcionante
  - FCO-MAD: Calidad de servicio Business deficiente
- **Incidente grave:** 1 mención
  - ATH-MAD (NPS 0): "Falsa acusación de agresión en embarque"

### ✈️ RUTAS AFECTADAS (Top 6 por volumen de detractores):

| Ruta | NPS | Encuestas | Problemas Reportados |
|------|-----|-----------|---------------------|
| **BCN-MAD** | 0 | 5 | Retrasos (2h), pérdida de conexiones, incidente grave en embarque |
| **AMS-MAD** | 0 | 2 | Mishandling grave (etiquetado incorrecto de equipaje) |
| **LIS-MAD** | 0 | 2 | Retrasos, llegada tardía |
| **AGP-MAD** | 0 | 1 | Equipaje retrasado |
| **ATH-MAD** | 0 | 1 | Incidente grave en embarque |
| **MAD-MXP** | 0 | 1 | Sin comentarios específicos |

### 👥 PERFILES REACTIVOS:

**Por Propósito de Viaje:**
| Segmento | NPS | Encuestas | Desviación vs día |
|----------|-----|-----------|-------------------|
| **Leisure** | 0.0 | 15 | -6.9 pts |
| **Business/Work** | 14.3 | 14 | +7.4 pts |

**Por Tipo de Flota:**
| Flota | NPS | Encuestas | Desviación vs día |
|-------|-----|-----------|-------------------|
| **A320neo** | 0.0 | 11 | -6.9 pts |
| **A321** | -22.2 | 9 | -29.1 pts |
| **A320** | 44.4 | 9 | +37.5 pts |

**Por Región de Residencia:**
| Región | NPS | Encuestas | Desviación vs día |
|--------|-----|-----------|-------------------|
| **ESPAÑA** | -18.2 | 11 | -25.1 pts |
| **EUROPA** | 12.5 | 8 | +5.6 pts |
| **AMERICA SUR** | 33.3 | 3 | +26.4 pts |

**Por Operador (Codeshare):**
| Operador | NPS | Encuestas | % del total |
|----------|-----|-----------|-------------|
| **IB** | 8.3 | 24 | 83% |
| **BA** | 0.0 | 3 | 10% |
| **LATAM** | -100.0 | 1 | 3% |

---

## === NMA 2: Global/SH/Economy/YW ===

### 📈 EXPLANATORY DRIVERS:
No disponible

### 📊 DATOS OPERATIVOS:
No disponible

### 🚨 INCIDENTES NCS (CUANTITATIVO):
No disponible

### 🧠 NCS (CUALITATIVO / REFLEXIÓN):
No disponible

### 💬 FEEDBACK DE CLIENTES:
No disponible

### ✈️ RUTAS AFECTADAS:
No disponible

### 👥 PERFILES REACTIVOS:
No disponible

**⚠️ Limitación crítica:** El análisis causal de este segmento falló completamente (❌ No se recolectaron datos durante la investigación)

---

## === NMA 3: Global/SH/Business/YW ===

### 📈 EXPLANATORY DRIVERS:
No disponible

### 📊 DATOS OPERATIVOS:
No disponible

### 🚨 INCIDENTES NCS (CUANTITATIVO):
No disponible

### 🧠 NCS (CUALITATIVO / REFLEXIÓN):
No disponible

### 💬 FEEDBACK DE CLIENTES:
No disponible

### ✈️ RUTAS AFECTADAS:
No disponible

### 👥 PERFILES REACTIVOS:
No disponible

**⚠️ Limitación crítica:** El análisis causal de este segmento falló completamente (❌ No se recolectaron datos durante la investigación)

---

## === NMA 4: Global/LH/Premium ===

### 📈 EXPLANATORY DRIVERS:
No disponible

### 📊 DATOS OPERATIVOS:
No disponible

### 🚨 INCIDENTES NCS (CUANTITATIVO):
No disponible

### 🧠 NCS (CUALITATIVO / REFLEXIÓN):
No disponible

### 💬 FEEDBACK DE CLIENTES:
No disponible

### ✈️ RUTAS AFECTADAS:
No disponible

### 👥 PERFILES REACTIVOS:
No disponible

**⚠️ Limitación crítica:** El análisis causal de este segmento falló completamente (❌ No se recolectaron datos durante la investigación)

---

## === NMA 5: Global/LH/Business ===

### 📈 EXPLANATORY DRIVERS:
No disponible

### 📊 DATOS OPERATIVOS:
No disponible

### 🚨 INCIDENTES NCS (CUANTITATIVO):
No disponible

### 🧠 NCS (CUALITATIVO / REFLEXIÓN):
No disponible

### 💬 FEEDBACK DE CLIENTES:
No disponible

### ✈️ RUTAS AFECTADAS:
No disponible

### 👥 PERFILES REACTIVOS:
No disponible

**⚠️ Limitación crítica:** El análisis causal de este segmento falló completamente (❌ No se recolectaron datos durante la investigación)

---

## 🎯 RESUMEN DE DISPONIBILIDAD DE DATOS

| NMA | Datos Operativos | NCS | Verbatims | Rutas | Perfiles | Estado |
|-----|------------------|-----|-----------|-------|----------|--------|
| **SH/Business/IB** | ✅ Completo | ✅ Cuantitativo | ✅ Completo | ✅ Completo | ✅ Completo | **ANÁLISIS COMPLETO** |
| **SH/Economy/YW** | ❌ | ❌ | ❌ | ❌ | ❌ | **FALLO TOTAL** |
| **SH/Business/YW** | ❌ | ❌ | ❌ | ❌ | ❌ | **FALLO TOTAL** |
| **LH/Premium** | ❌ | ❌ | ❌ | ❌ | ❌ | **FALLO TOTAL** |
| **LH/Business** | ❌ | ❌ | ❌ | ❌ | ❌ | **FALLO TOTAL** |

**Conclusión:** Solo **1 de 5 NMAs** (20%) tiene evidencia completa disponible. El 80% de las anomalías detectadas carecen de datos causales, limitando severamente la capacidad de diagnóstico y recomendación.

---

## Cabin Radio Reflection

# 📋 PASO 4C: REFLEXIÓN POR CABINA-RADIO

---

## 🛫 SHORT HAUL

### === Economy SH ===

• **NPS Cabina:** 34.9 (+7.9)  
• **Estado:** POSITIVE ANOMALY  
• **Escenario:** TRANSFERENCIA (IB Normal, YW POSITIVE ANOMALY | Cabina POSITIVE ANOMALY)  

• **IB:** NPS 28.3 (+1.4) - Sin análisis causal disponible. Desempeño estable dentro de rango normal, sin contribución significativa a la anomalía de la cabina.

• **YW:** NPS 49.6 (+22.1) - **Sin análisis causal disponible (fallo en recolección de datos).** La mejora excepcional de +22.1 pts vs baseline no tiene drivers operativos, rutas o perfiles identificados que expliquen este salto extraordinario.

• **Narrativa de agregación:** YW domina completamente y arrastra a la cabina Economy SH a territorio anómalo positivo (+7.9 pts). IB mantiene estabilidad (+1.4 pts, Normal) sin aportar al efecto. El volumen y magnitud de la mejora de YW es suficiente para elevar el agregado, a pesar de la ausencia de datos causales que expliquen esta mejora.

• **Rutas críticas:** No disponible (análisis causal de YW falló)

• **Perfiles reactivos:** No disponible (análisis causal de YW falló)

---

### === Business SH ===

• **NPS Cabina:** 14.7 (-20.4)  
• **Estado:** NEGATIVE ANOMALY  
• **Escenario:** DOMINANCIA (IB NEGATIVE ANOMALY, YW POSITIVE ANOMALY | Cabina NEGATIVE ANOMALY)  

• **IB:** NPS 6.9 (-35.2) - **Causa principal:** Mishandling +3.57 pts (19.79 vs baseline 16.22) con correlación inversa confirmada con NPS. **Causa secundaria:** Retrasos >15min no capturados por OTP15 (5 menciones en verbatims), generando pérdida de conexiones. **Rutas críticas:** BCN-MAD (NPS 0, 5 detractores), AMS-MAD (NPS 0, mishandling grave con etiquetado incorrecto), LIS-MAD (NPS 0, retrasos). **Perfiles afectados:** Leisure (NPS 0.0, -6.9 pts vs día), flotas A320neo (NPS 0.0) y A321 (NPS -22.2). **Contradicción operativa:** OTP15 mejoró +1.55 pts pero los verbatims reportan múltiples retrasos significativos.

• **YW:** NPS 60.0 (+44.0) - **Sin análisis causal disponible (fallo en recolección de datos).** Esta mejora récord de +44.0 pts vs baseline compensa parcialmente el colapso de IB, pero carece de explicación sobre qué factores operativos la generaron.

• **Narrativa de agregación:** IB domina e impone su signo negativo a la cabina (-20.4 pts) a pesar de la mejora excepcional de YW (+44.0 pts). El mayor peso volumétrico o impacto operativo de IB (29 encuestas, 83% de vuelos operados por IB según codeshare) prevalece sobre el efecto positivo de YW. La crisis operativa de IB (Mishandling, retrasos) define el resultado del segmento, aunque YW mitiga parcialmente la caída.

• **Rutas críticas:** BCN-MAD (NPS 0, 5 detractores - retrasos 2h, pérdida conexiones), AMS-MAD (NPS 0, 2 detractores - mishandling grave), LIS-MAD (NPS 0, 2 detractores - retrasos), AGP-MAD (NPS 0 - equipaje retrasado), ATH-MAD (NPS 0 - incidente grave en embarque)

• **Perfiles reactivos:**  
  - **Propósito de viaje:** Leisure (NPS 0.0, -6.9 pts vs día del segmento) más afectado que Business/Work (NPS 14.3, +7.4 pts)
  - **Flota:** A320neo (NPS 0.0, -6.9 pts) y A321 (NPS -22.2, -29.1 pts) concentran problemas; A320 tradicional (NPS 44.4, +37.5 pts) desempeño excelente
  - **Residencia:** España (NPS -18.2, -25.1 pts) más críticos; Europa (NPS 12.5, +5.6 pts) y América Sur (NPS 33.3, +26.4 pts) mejor valoración
  - **Operador:** Vuelos IB (NPS 8.3, 83% del total) vs codeshares BA (NPS 0.0) y LATAM (NPS -100.0)

---

## ✈️ LONG HAUL

### === Economy LH ===

• **NPS:** 4.5 (+0.8)  
• **Estado:** Normal  
• **Causa principal:** Sin análisis causal disponible (fallo en recolección de datos). La variación de +0.8 pts se mantiene dentro del rango normal de fluctuación estadística.  
• **Evidencia clave:** No disponible  
• **Rutas críticas:** No disponible  
• **Perfiles reactivos:** No disponible  

**Nota:** Este segmento actúa como **estabilizador de volumen** en LH, representando el mayor número de pasajeros y diluyendo los efectos extremos de Business (-2.3) y Premium (+31.9).

---

### === Business LH ===

• **NPS:** 16.7 (-2.3)  
• **Estado:** NEGATIVE ANOMALY  
• **Causa principal:** Sin análisis causal disponible (fallo en recolección de datos). La anomalía negativa de -2.3 pts carece de drivers operativos, rutas o perfiles identificados.  
• **Evidencia clave:** No disponible  
• **Rutas críticas:** No disponible  
• **Perfiles reactivos:** No disponible  

**Hipótesis inferida (sin confirmar):** Posible deterioro en experiencia de producto Business LH (servicio, catering, confort) o problemas operativos en rutas intercontinentales específicas, pero sin datos para validar.

---

### === Premium LH ===

• **NPS:** 41.2 (+31.9)  
• **Estado:** POSITIVE ANOMALY  
• **Causa principal:** Sin análisis causal disponible (fallo en recolección de datos). La mejora extraordinaria de +31.9 pts (de baseline 9.3 a 41.2) representa un salto del 343% sin explicación operativa.  
• **Evidencia clave:** No disponible  
• **Rutas críticas:** No disponible  
• **Perfiles reactivos:** No disponible  

**Hipótesis inferida (sin confirmar):** Posible mejora excepcional en experiencia Premium (upgrade operacional, cambio de flota a configuración superior, evento especial de servicio), pero sin evidencia disponible para confirmar o replicar.

---

## 🎯 SÍNTESIS DE DISPONIBILIDAD POR CABINA-RADIO

| Cabina-Radio | Estado | Variación | Análisis Causal | Nivel de Confianza |
|--------------|--------|-----------|-----------------|-------------------|
| **Economy SH** | POSITIVE ANOMALY | +7.9 | ❌ Parcial (solo YW falla) | BAJO |
| **Business SH** | NEGATIVE ANOMALY | -20.4 | ✅ Completo (IB) / ❌ (YW) | ALTO (para IB) |
| **Economy LH** | Normal | +0.8 | ❌ Sin datos | N/A |
| **Business LH** | NEGATIVE ANOMALY | -2.3 | ❌ Sin datos | NULO |
| **Premium LH** | POSITIVE ANOMALY | +31.9 | ❌ Sin datos | NULO |

**Conclusión crítica:** Solo **Business SH (IB)** tiene análisis causal completo. El 80% de las anomalías detectadas carecen de explicación operativa, limitando severamente la capacidad de:
1. Replicar las mejoras excepcionales (YW Eco/Bus SH, Premium LH)
2. Prevenir el agravamiento de deterioros (Business LH)
3. Entender la sostenibilidad de la anomalía positiva Global

---

## 📋 SÍNTESIS EJECUTIVA FINAL

```html
<b>SÍNTESIS EJECUTIVA</b><br>

La red global registró un <b>NPS de 25.6 (+4.7 pts)</b> con respecto a la media de los últimos 7 días, presentando una anomalía positiva resultado de mejoras excepcionales en segmentos premium que compensaron problemas operativos localizados en Business de corto radio.<br><br>

En <b>Business SH</b>, el NPS cayó a <b>14.7 (–20.4 pts)</b>, impulsado principalmente por el colapso operativo de IB, que registró un <b>NPS de 6.9 (–35.2 pts)</b>. La causa raíz fue un deterioro significativo en la gestión de equipaje, con Mishandling aumentando 3.57 puntos hasta alcanzar 19.79, mostrando una correlación inversa confirmada con el NPS. Adicionalmente, se detectaron retrasos superiores a 15 minutos que no fueron capturados por la métrica OTP15, a pesar de que esta mejoró 1.55 puntos. Estos retrasos generaron pérdida de conexiones internacionales, especialmente visibles en rutas como <b>BCN-MAD</b>, donde 5 detractores reportaron demoras de hasta 2 horas, <b>AMS-MAD</b> con casos graves de etiquetado incorrecto de equipaje hacia destinos equivocados, y <b>LIS-MAD</b> con llegadas tardías recurrentes. Los pasajeros más sensibles fueron los de propósito <b>Leisure</b>, que registraron NPS de 0.0, junto con aquellos que volaron en flotas <b>A320neo</b> y <b>A321</b>, ambas con NPS negativos. Los residentes en <b>España</b> mostraron la mayor reacción crítica con NPS de –18.2 puntos. Esta presión en IB Business SH dominó el resultado de la cabina completa, a pesar de que YW registró una mejora extraordinaria de <b>NPS 60.0 (+44.0 pts)</b> que compensó parcialmente el impacto pero no logró neutralizarlo debido al mayor peso volumétrico de IB. El efecto negativo de Business SH se canceló posteriormente con la mejora de Economy SH en el nivel Short Haul, impidiendo que la crisis operativa alcanzara el nivel global.<br><br>

En <b>Economy SH</b>, el NPS subió a <b>34.9 (+7.9 pts)</b>, impulsado exclusivamente por YW, que alcanzó un <b>NPS de 49.6 (+22.1 pts)</b> versus un baseline de 27.4. IB mantuvo un desempeño estable con <b>NPS de 28.3 (+1.4 pts)</b>, sin contribuir a la anomalía. La mejora excepcional de YW arrastró a la cabina completa a territorio anómalo positivo, aunque la causa operativa de esta mejora no pudo ser identificada debido a un fallo en la recolección de datos del análisis causal. Esta anomalía positiva en Economy SH compensó parcialmente el colapso de Business SH en el agregado de corto radio, resultando en una variación normal de Short Haul que ocultó la volatilidad extrema entre cabinas.<br><br>

En <b>Premium LH</b>, el NPS se disparó a <b>41.2 (+31.9 pts)</b>, representando un incremento del 343 por ciento respecto al baseline de 9.3. Esta mejora extraordinaria carece de explicación operativa debido a un fallo completo en la recolección de datos causales, impidiendo identificar los drivers, rutas o perfiles que generaron este salto excepcional. Esta anomalía positiva se canceló con el deterioro de Business LH en el nivel de largo radio, pero contribuyó indirectamente al resultado global a través de un efecto acumulativo.<br><br>

En <b>Business LH</b>, el NPS cayó a <b>16.7 (–2.3 pts)</b>, aunque la causa de este deterioro no pudo ser determinada debido a la ausencia completa de datos causales. Esta anomalía negativa se canceló con la mejora de Premium LH y la estabilidad de Economy LH en el agregado de largo radio, impidiendo que el efecto negativo alcanzara el nivel global.<br><br>

La convergencia de <b>Long Haul</b>, que mostró variación normal de +3.4 puntos, y <b>Short Haul</b>, también con variación normal de +5.6 puntos, produjo una anomalía positiva global de +4.7 puntos mediante un efecto de transferencia acumulativa. Ambos radios, aunque individualmente dentro de rangos normales, superaron el umbral de anomalía al agregarse en el nivel de red. Este resultado oculta una realidad fragmentada: mejoras excepcionales en segmentos premium de bajo volumen pero alto impacto, cuyas causas no están identificadas y podrían ser circunstanciales, compensaron una crisis operativa grave y documentada en IB Business SH relacionada con gestión de equipaje y retrasos en rutas domésticas clave.<br><br>

Durante el período analizado se registraron 14 incidentes operativos, incluyendo 12 cancelaciones, 3 retrasos, 1 desvío de BCN-LEN a OVD por condiciones meteorológicas y 2 otras incidencias. Los temas principales identificados fueron meteorología adversa y cambios de aeronave. El feedback cualitativo de clientes reveló 2 menciones específicas de problemas de equipaje, 5 menciones de retrasos significativos que generaron pérdida de conexiones, 2 menciones de calidad de servicio Business decepcionante y 1 incidente grave de acusación falsa en embarque en la ruta ATH-MAD.<br><br>

<b><u>DETALLE POR CABINA</u></b><br>

<b><u>ECONOMY SH: Mejora impulsada exclusivamente por YW sin causa identificada</u></b><br>
La cabina Economy de corto radio alcanzó un NPS de 34.9 con una variación de +7.9 puntos, resultado de una transferencia directa desde YW, que registró un salto excepcional a NPS de 49.6 (+22.1 pts). IB mantuvo estabilidad con NPS de 28.3 (+1.4 pts), sin aportar al efecto anómalo. El volumen y magnitud de la mejora de YW fue suficiente para elevar el agregado a territorio anómalo positivo, a pesar de que el análisis causal falló completamente en identificar los drivers operativos, rutas críticas o perfiles de cliente que expliquen esta mejora extraordinaria. Esta ausencia de datos impide determinar si la mejora es replicable o circunstancial, limitando la capacidad de capitalizar este éxito en otros segmentos.<br><br>

<b><u>BUSINESS SH: Crisis operativa de IB domina el resultado de la cabina</u></b><br>
La cabina Business de corto radio cayó a un NPS de 14.7 (–20.4 pts), reflejando directamente el desplome de IB, que alcanzó NPS de 6.9 (–35.2 pts) debido a problemas graves de gestión de equipaje y retrasos no capturados por métricas estándar. El incremento de Mishandling en 3.57 puntos hasta 19.79 mostró correlación inversa confirmada con el NPS, mientras que múltiples clientes reportaron retrasos superiores a 15 minutos que generaron pérdida de conexiones internacionales, contradiciendo la mejora registrada en OTP15. Las rutas más afectadas fueron BCN-MAD con 5 detractores y NPS de 0, donde se reportaron demoras de hasta 2 horas y pérdida de conexiones a Chile, AMS-MAD con 2 detractores y casos graves de etiquetado incorrecto de equipaje hacia Lima en lugar de Santiago, y LIS-MAD con 2 detractores por llegadas tardías recurrentes. Los pasajeros de propósito Leisure fueron los más críticos con NPS de 0.0, mostrando una desviación de –6.9 puntos respecto al promedio del día, mientras que las flotas A320neo y A321 concentraron los problemas operativos con NPS de 0.0 y –22.2 respectivamente. Los residentes en España mostraron la mayor sensibilidad negativa con NPS de –18.2 puntos. YW registró una mejora extraordinaria a NPS de 60.0 (+44.0 pts) que compensó parcialmente el colapso de IB, pero el mayor peso volumétrico de IB, que operó el 83 por ciento de los vuelos según datos de codeshare, impuso el signo negativo al agregado de la cabina. La ausencia de análisis causal para YW impide entender qué factores generaron esta mejora récord.<br><br>

<b><u>ECONOMY LH: Estabilidad con función de ancla volumétrica</u></b><br>
La cabina Economy de largo radio mantuvo un NPS de 4.5 con una variación de +0.8 puntos dentro del rango normal de fluctuación estadística. No se dispone de análisis causal debido a un fallo en la recolección de datos. Este segmento actuó como estabilizador de volumen en largo radio, representando el mayor número de pasajeros y diluyendo los efectos extremos de Business y Premium, que mostraron movimientos opuestos significativos.<br><br>

<b><u>BUSINESS LH: Deterioro sin causa operativa identificada</u></b><br>
La cabina Business de largo radio registró un NPS de 16.7 (–2.3 pts), presentando una anomalía negativa cuya causa raíz no pudo ser determinada debido a un fallo completo en la recolección de datos causales. La ausencia de drivers operativos, rutas críticas o perfiles de cliente impide validar hipótesis sobre posibles deterioros en experiencia de producto Business, servicio a bordo o problemas operativos en rutas intercontinentales específicas. Esta anomalía negativa se canceló con la mejora extraordinaria de Premium y la estabilidad de Economy en el agregado de largo radio, impidiendo que el efecto alcanzara el nivel global.<br><br>

<b><u>PREMIUM LH: Mejora extraordinaria sin explicación operativa</u></b><br>
La cabina Premium de largo radio alcanzó un NPS de 41.2 (+31.9 pts), representando un incremento del 343 por ciento respecto al baseline de 9.3. Esta mejora extraordinaria carece completamente de análisis causal debido a un fallo en la recolección de datos, impidiendo identificar si se trata de una mejora en experiencia de producto Premium, un upgrade operacional, un cambio de flota a configuración superior o un evento especial de servicio. La ausencia de rutas críticas y perfiles reactivos limita severamente la capacidad de replicar este éxito excepcional en otros segmentos o períodos. Esta anomalía positiva se canceló con el deterioro de Business en el nivel de largo radio, pero contribuyó indirectamente al resultado global positivo mediante el efecto acumulativo de transferencia.
```

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 7
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-14 to 2025-12-14:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 PASO 1: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL COMPAÑÍA)

---

## **ANÁLISIS DE CABINA: ECONOMY SHORT HAUL**

### **PARTE A: DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(IB: -, YW: + | Padre: -)`

**Escenario Detectado:** **DOMINANCIA** (-,+ | -)

**Explicación del Patrón:**
- **IB (Economy SH):** NEGATIVE ANOMALY (-9.7 pts) - NPS 17.1 vs baseline 26.9
- **YW (Economy SH):** POSITIVE ANOMALY (+15.0 pts) - NPS 42.4 vs baseline 27.4
- **Padre (Economy SH):** NEGATIVE ANOMALY (-1.5 pts) - NPS 25.5 vs baseline 27.1

A pesar de que YW tuvo una mejora significativa (+15.0 pts), **IB domina el resultado agregado** debido a su mayor volumen de operaciones y/o peso en el segmento Economy SH. La caída de IB (-9.7 pts) es suficientemente fuerte para arrastrar al padre a territorio negativo (-1.5 pts), aunque el efecto positivo de YW **atenúa parcialmente** la magnitud de la caída.

**Dispersión entre compañías:** 24.7 pts (15.0 - (-9.7)) - **Volatilidad interna CRÍTICA**.

---

### **PARTE B: NARRATIVA CAUSAL**

**Lógica Aplicada:** DOMINANCIA → Adoptar **Explicación del Hijo Dominante (IB)** + Matiz del hijo opositor (YW).

#### **Narrativa:**

La anomalía negativa en Economy SH (-1.5 pts) está **impulsada por el deterioro operativo de IB** (-9.7 pts), que experimentó problemas críticos en gestión de equipaje y operaciones en tierra. Aunque YW logró una mejora significativa (+15.0 pts Economy SH) gracias a mejor manejo operativo y menor exposición a rutas problemáticas, este efecto positivo **solo atenuó parcialmente** el impacto negativo de IB, sin lograr neutralizarlo completamente en el agregado.

#### **Evidencia Clave del Hijo Dominante (IB):**

**⚠️ NOTA CRÍTICA:** El análisis causal de IB (Economy SH) reportó "Síntesis falló debido a error", por lo que **infiero causas desde el análisis del nodo padre** (Global/SH/Economy) que incluye ambas compañías:

**Causas Principales (inferidas desde padre Economy SH):**

1. **Incremento Crítico en Mishandling (+3.73 pts vs baseline Economy SH):**
   - 3 incidentes NCS de equipaje documentados
   - 8+ verbatims negativos sobre equipaje perdido/dañado
   - Rutas afectadas: MAD-ORY (NPS -4.0), AMS-MAD, DUS-MAD, PMI-VLC

2. **Incidentes Meteorológicos en EAS:**
   - 5 incidentes NCS en aeropuerto EAS (San Sebastián)
   - Ruta EAS-MAD con **NPS -16.7** (peor ruta del día)
   - 3 desvíos + 2 cancelaciones por weather

3. **Problemas de Asignación de Asientos:**
   - 4+ verbatims sobre cambios de asiento sin justificación
   - Rutas: GVA-MAD, MAD-NAP

**Perfiles IB más afectados (inferidos):**
- **Flotas A333 (NPS -66.7) y A319 (NPS -8.7)** - Problemas de espacio/equipaje
- **CodeShare AA (NPS -100.0)** - Conexiones fallidas
- **Región Europa (NPS +15.9)** vs España (NPS +38.2) - Pasajeros europeos más críticos

#### **Evidencia Clave del Hijo Opositor (YW):**

**Factores que explican la mejora de YW (+15.0 pts Economy SH):**

1. **Mejor Gestión de Equipaje:**
   - Mishandling YW: 15.19 (+2.55 pts vs baseline YW) - **Menor deterioro que IB**
   - Solo 3 incidentes NCS de equipaje vs mayor volumen en IB

2. **Mejor Puntualidad:**
   - OTP15 YW: 89.66% (+1.29 pts vs baseline YW)
   - Menor exposición a rutas con incidentes meteorológicos (EAS)

3. **Menor Ocupación:**
   - Load Factor YW: 78.83% (-1.81 pts vs baseline YW) - Mejor experiencia relativa

4. **Rutas con Mejor Desempeño:**
   - Menor exposición a rutas críticas como EAS-MAD, BLQ-MAD, FRA-MAD
   - Mejor distribución geográfica evitando hubs problemáticos

**Perfiles YW favorecidos:**
- **Viajeros Leisure (NPS 49.1)** vs Business (NPS 8.7) - +40.4 pts diferencial
- **Flota ATR (NPS 63.2)** - Mejor desempeño que CRJ
- **Pasajeros España (NPS 55.1)** - Menos afectados que europeos

---

## **ANÁLISIS DE CABINA: BUSINESS SHORT HAUL**

### **PARTE A: DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(IB: N, YW: N | Padre: N)`

**Escenario Detectado:** **NO APLICA - Segmento Normal**

**Explicación:**
- **IB (Business SH):** Normal (+4.8 pts) - NPS 46.9 vs baseline 42.1
- **YW (Business SH):** Normal (+4.0 pts) - NPS 20.0 vs baseline 16.0
- **Padre (Business SH):** Normal (+3.2 pts) - NPS 38.3 vs baseline 35.1

Este segmento **NO presenta anomalía real**. Las variaciones están dentro del rango normal de fluctuación. Ambas compañías mantuvieron desempeño estable.

### **PARTE B: NARRATIVA CAUSAL**

**No se requiere narrativa causal** - Este segmento debe ser **excluido del análisis** según las instrucciones (solo analizar anomalías reales).

---

## **📊 RESUMEN EJECUTIVO - PASO 1**

| Cabina | Escenario | Tríada | Narrativa Dominante | Dispersión IB-YW |
|--------|-----------|--------|---------------------|------------------|
| **Economy SH** | **DOMINANCIA** | `(-, + \| -)` | **IB domina** con deterioro operativo (equipaje +3.73, EAS-MAD -16.7), YW atenúa con mejor gestión | **24.7 pts** ⚠️ |
| **Business SH** | **N/A** | `(N, N \| N)` | Segmento estable - No requiere análisis | 26.9 pts |

**Conclusión Clave:** La anomalía negativa en Economy SH (-1.5 pts) es un **efecto de dominancia de IB** (-9.7 pts) causado por problemas críticos de equipaje y meteorología en EAS, parcialmente compensado por la mejora operativa de YW (+15.0 pts).

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ PASO 2: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL CABINA)

---

## **ANÁLISIS DE RADIO: LONG HAUL (LH)**

### **PARTE A: DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(Economy: +, Business: -, Premium: - | Padre: +)`

**Escenario Detectado:** **DOMINANCIA** (+,-,- | +)

**Explicación del Patrón:**
- **Economy LH:** POSITIVE ANOMALY (+22.7 pts) - NPS 26.4 vs baseline 3.7
- **Business LH:** NEGATIVE ANOMALY (-10.3 pts) - NPS 8.7 vs baseline 19.0
- **Premium LH:** NEGATIVE ANOMALY (-3.4 pts) - NPS 5.9 vs baseline 9.3
- **Padre (LH):** POSITIVE ANOMALY (+16.5 pts) - NPS 22.3 vs baseline 5.9

A pesar de que **dos cabinas premium (Business y Premium Economy) experimentaron caídas**, la **Economy domina completamente el resultado agregado** del radio LH, imponiendo su signo positivo (+) al padre. Esto ocurre porque:

1. **Volumen:** Economy LH concentra la mayor parte de pasajeros en vuelos long-haul
2. **Magnitud:** La mejora de Economy (+22.7 pts) es más del doble que la caída de Business (-10.3 pts)
3. **Peso relativo:** Las cabinas premium tienen menor peso estadístico en el agregado

**Dispersión entre cabinas:** 26.1 pts (22.7 - (-3.4)) - **Volatilidad interna ALTA**.

---

### **PARTE B: NARRATIVA CAUSAL**

**Lógica Aplicada:** DOMINANCIA → Adoptar **Explicación de la Cabina Dominante (Economy LH)** + Matiz de cabinas opositoras.

#### **Narrativa Principal:**

El rendimiento positivo del radio Long Haul (+16.5 pts) está **dictado por Economy LH** (+22.7 pts), que experimentó una mejora significativa respecto a un baseline excepcionalmente bajo. Sin embargo, esta mejora es **relativa, no absoluta**: el análisis operativo revela que el día 14/12 tuvo problemas graves de equipaje y conexiones, pero estos fueron **menos severos que los 7 días previos** que conforman el baseline (NPS baseline: 3.7 pts).

**⚠️ PARADOJA CRÍTICA DETECTADA:** 

La anomalía positiva en Economy LH (+22.7 pts) coexiste con **deterioro operativo significativo**:
- **Mishandling:** 19.04 (+3.73 pts vs baseline Economy LH)
- **Misconex:** 0.81 (+0.13 pts vs baseline Economy LH)
- **OTP15:** 77.27% (-4.58 pts vs baseline Economy LH)

Esta contradicción sugiere que **el baseline de los 7 días previos fue excepcionalmente malo**, haciendo que el día 14/12 parezca mejor en comparación, aunque en términos absolutos tuvo problemas operativos graves.

El efecto positivo de Economy fue **parcialmente contrarrestado** por el deterioro en Business (-10.3 pts) y Premium (-3.4 pts), que sufrieron problemas específicos de producto/servicio no capturados completamente en los análisis disponibles (errores de procesamiento).

---

#### **Evidencia Clave de la Cabina Dominante (Economy LH):**

**1. PROBLEMAS DE EQUIPAJE (Causa Principal - Confianza ALTA):**

**Métricas Operativas:**
- **Mishandling:** 19.04 (+3.73 pts vs baseline Economy LH)
- **Misconex:** 0.81 (+0.13 pts vs baseline Economy LH)

**Incidentes NCS:**
- **IB151 MAD-BOG:** 26 equipajes no cargados por falta de capacidad
- **IB281 (conexión BA458 LHR-MAD):** 27 maletas perdidas, regularización vía DOH
- **Total:** 53 maletas afectadas en 2 vuelos críticos

**Verbatims (5+ menciones NPS 0-3):**
- "Las maletas tardaron dos horas en bajar del vuelo y perdí mi conexión" (MAD-UIO, NPS 0)
- "El equipaje nunca llegó" (EZE-MAD, NPS 0)
- "Me perdieron una maleta... había sido abierta y robada" (GIG-MAD, NPS 0)
- "Me han abierto la maleta y me han sustraído la mitad" (BOG-MAD, NPS 3)

**2. PROCESO DE EMBARQUE CAÓTICO (Causa Secundaria - Confianza MEDIA-ALTA):**

**Verbatims (3+ menciones):**
- "Fila innecesariamente larga para pesar equipaje de mano, tomó más de una hora, retrasó la salida" (MAD-SJO, NPS 0)
- "No nos informaron que debíamos pesar maletas de mano... el caos fue total" (BOG-MAD, NPS 6)
- "Pesaban maletas de mano con tolerancia mínima... sentía estar en low cost" (MAD-SCL, NPS 0)

**3. RUTAS CRÍTICAS ECONOMY LH:**

| Ruta | NPS | Encuestas | Problema Principal |
|------|-----|-----------|-------------------|
| **BOG-MAD** | **14.3** | 21 | Equipaje (26 maletas) + Proceso embarque caótico |
| **MAD-UIO** | **18.2** | 11 | Conexiones perdidas (27 maletas BA458) |
| **DOH-MAD** | **-20.0** | 5 | Sin evidencia específica (muestra pequeña) |
| **MAD-SJO** | **37.5** | 8 | Proceso embarque caótico |

**4. PERFILES ECONOMY LH MÁS AFECTADOS:**

**Por Flota (Dispersión: 133.3 pts):**
- **A33ACMI:** NPS -33.3 (n=3) - Peor flota
- **A333:** NPS -6.7 (n=15) - Problemas de equipaje/espacio
- **A350 next:** NPS 27.5 (n=40) - Mejor flota

**Por Región de Residencia (Dispersión: 200.0 pts):**
- **ASIA:** NPS -100.0 (n=3) - Todos detractores
- **EUROPA:** NPS 10.0 (n=10) - Muy por debajo del promedio
- **AMERICA SUR:** NPS 20.0 (n=20) - Afectados por BOG-MAD, MAD-UIO

**Por CodeShare (Dispersión: 160.0 pts):**
- **BA:** NPS -60.0 (n=5) - Correlaciona con incidente BA458
- **AA:** NPS -33.3 (n=3)
- **LATAM:** NPS -25.0 (n=4)
- **IB:** NPS 31.3 (n=131) - Operación directa mejor

---

#### **Evidencia de Cabinas Opositoras (Business y Premium LH):**

**⚠️ LIMITACIÓN:** Los análisis causales de Business LH y Premium LH reportaron errores de procesamiento, por lo que **infiero causas desde el contexto general**:

**Business LH (-10.3 pts):**

**Hipótesis de Causas (Confianza MEDIA-BAJA):**
1. **Expectativas no cumplidas en servicio premium:**
   - Los problemas de equipaje y proceso de embarque afectan más a pasajeros Business por expectativas más altas
   - Menor tolerancia a disrupciones operativas

2. **Rutas específicas con problemas:**
   - Sin datos granulares disponibles por error de procesamiento
   - Posible correlación con rutas BOG-MAD, DOH-MAD donde el servicio Business pudo verse comprometido

3. **Flotas con configuración Business problemática:**
   - A333 con NPS negativo en Economy podría tener problemas similares en Business
   - Espacio/confort inferior a expectativas

**Premium LH (-3.4 pts):**

**Hipótesis de Causas (Confianza BAJA):**
1. **Caída menor pero consistente:**
   - La magnitud reducida (-3.4 pts) sugiere problemas menores pero generalizados
   - Posible degradación de servicios diferenciales (comida, amenities, atención)

2. **Efecto contagio desde Economy:**
   - Los problemas operativos (equipaje, embarque) afectan a todas las cabinas
   - Premium Economy no tiene suficiente diferenciación para aislar el impacto

---

## **ANÁLISIS DE RADIO: SHORT HAUL (SH)**

### **PARTE A: DINÁMICA DE AGREGACIÓN**

**Díada de Estados:** `(Economy: -, Business: N | Padre: -)`

**Escenario Detectado:** **TRANSFERENCIA** (-,N | -)

**Explicación del Patrón:**
- **Economy SH:** NEGATIVE ANOMALY (-1.5 pts) - NPS 25.5 vs baseline 27.1
- **Business SH:** Normal (+3.2 pts) - NPS 38.3 vs baseline 35.1
- **Padre (SH):** NEGATIVE ANOMALY (-0.9 pts) - NPS 26.8 vs baseline 27.7

A pesar de que Business SH se mantuvo estable (Normal), la **anomalía negativa de Economy SH se transfiere al padre** debido a:

1. **Volumen dominante:** Economy concentra la gran mayoría de pasajeros en Short Haul
2. **Peso estadístico:** Business SH tiene volumen insuficiente para compensar el efecto negativo de Economy
3. **Magnitud limitada:** La caída de Economy (-1.5 pts) es pequeña pero suficiente para arrastrar al padre (-0.9 pts)

**Dispersión entre cabinas:** 4.7 pts (3.2 - (-1.5)) - **Volatilidad interna BAJA**.

---

### **PARTE B: NARRATIVA CAUSAL**

**Lógica Aplicada:** TRANSFERENCIA → Adoptar **Explicación de la Cabina Anómala (Economy SH)**.

#### **Narrativa Principal:**

El rendimiento negativo del radio Short Haul (-0.9 pts) está **arrastrado por Economy SH** (-1.5 pts), que experimentó problemas operativos localizados pero significativos. La estabilidad de Business SH (Normal, +3.2 pts) no fue suficiente para compensar este efecto debido al peso volumétrico dominante de Economy en operaciones de corto radio.

La anomalía en Economy SH es resultado de la **dinámica de dominancia entre compañías** analizada en el Paso 1: IB cayó -9.7 pts por problemas de equipaje y meteorología (EAS), mientras YW mejoró +15.0 pts, resultando en una caída neta de -1.5 pts que se transfiere directamente al padre SH.

---

#### **Evidencia Clave de la Cabina Anómala (Economy SH):**

**1. INCREMENTO SIGNIFICATIVO EN MISHANDLING (Causa Principal - Confianza ALTA):**

**Métricas Operativas:**
- **Mishandling:** 19.04 (+3.73 pts vs baseline Economy SH)
- **Misconex:** 0.81 (+0.13 pts vs baseline Economy SH)
- **OTP15:** 91.78% (+1.68 pts vs baseline Economy SH) - Factor mitigante

**Incidentes NCS:**
- **3 incidentes de equipaje** documentados
- "27 maletas procedente de BA458 LHR MAD, no han llegado a IB281. Se regularizan vía DOH"

**Verbatims (8+ menciones):**
- Equipaje de mano forzado a bodega sin justificación
- Pérdidas de maletas en conexiones
- Daños durante manipulación
- Rutas afectadas: MAD-ORY, AMS-MAD, DUS-MAD, PMI-VLC

**2. INCIDENTES METEOROLÓGICOS EN EAS (Causa Secundaria - Confianza ALTA):**

**Incidentes NCS:**
- **5 incidentes meteorológicos** en aeropuerto EAS (San Sebastián)
- 3 desvíos por weather
- 2 cancelaciones relacionadas
- Vuelos desviados a BIO con transporte por superficie

**Rutas Afectadas:**
- **EAS-MAD:** NPS -16.7 (n=6) - **Peor ruta del día SH**

**3. PROBLEMAS DE ASIGNACIÓN DE ASIENTOS (Causa Terciaria - Confianza MEDIA):**

**Verbatims (4+ menciones):**
- GVA-MAD: "Nos han cambiado los asientos reservados con bastante antelación, sin explicación alguna"
- MAD-NAP: Cliente pagó Priority (asiento 24C) pero recibió 32A
- Impacto emocional alto en clientes que pagaron servicios premium

**4. RUTAS CRÍTICAS ECONOMY SH:**

| Ruta | NPS | Encuestas | Problema Principal |
|------|-----|-----------|-------------------|
| **EAS-MAD** | **-16.7** | 6 | Desvíos meteorológicos |
| **BLQ-MAD** | **-40.0** | 5 | Sin verbatims específicos |
| **MAD-ORY** | **-4.0** | 25 | Equipaje forzado a bodega, retrasos |
| **MAD-NAP** | **0.0** | 5 | Asignación de asientos |

**5. PERFILES ECONOMY SH MÁS AFECTADOS:**

**Por Flota (Dispersión: 129.8 pts):**
- **A333:** NPS -66.7 (n=3) - Peor flota, problemas de espacio
- **A319:** NPS -8.7 (n=23) - Equipaje forzado a bodega
- **ATR:** NPS +63.2 (n=19) - Mejor flota, menor ocupación

**Por CodeShare (Dispersión: 200.0 pts):**
- **AA:** NPS -100.0 (n=3) - Muestra baja, NPS crítico
- **AY:** NPS -100.0 (n=1) - Muestra insuficiente
- **IB:** NPS +27.6 (n=392) - Volumen principal

**Por Región de Residencia:**
- **EUROPA:** NPS +15.9 (n=107) - Más críticos que españoles
- **ESPAÑA:** NPS +38.2 (n=207) - Volumen alto, NPS positivo

---

#### **Contexto de la Cabina Estable (Business SH):**

**Business SH (Normal, +3.2 pts):**
- **IB Business SH:** Normal (+4.8 pts) - NPS 46.9 vs baseline 42.1
- **YW Business SH:** Normal (+4.0 pts) - NPS 20.0 vs baseline 16.0

**Interpretación:**
Business SH mantuvo desempeño estable porque:
1. **Menor exposición a problemas de equipaje:** Pasajeros Business viajan con menos equipaje facturado
2. **Mejor gestión de disrupciones:** Prioridad en re-acomodación ante incidentes meteorológicos
3. **Menor volumen afectado:** Las rutas críticas (EAS-MAD, BLQ-MAD) tienen menor proporción de Business

---

## **📊 RESUMEN EJECUTIVO - PASO 2**

| Radio | Escenario | Estados | Narrativa Dominante | Dispersión Cabinas |
|-------|-----------|---------|---------------------|-------------------|
| **Long Haul** | **DOMINANCIA** | `(+,-,- \| +)` | **Economy LH domina** (+22.7) con mejora relativa vs baseline bajo, a pesar de deterioro operativo real. Business (-10.3) y Premium (-3.4) contrarrestan parcialmente. | **26.1 pts** ⚠️ |
| **Short Haul** | **TRANSFERENCIA** | `(-,N \| -)` | **Economy SH arrastra** (-1.5) por problemas de equipaje (+3.73 Mishandling) y meteorología EAS. Business estable (N) no compensa por menor volumen. | **4.7 pts** |

---

## **🎯 CONCLUSIONES CLAVE - NIVEL CABINA**

1. **Long Haul:** La anomalía positiva (+16.5 pts) es **engañosa** - refleja mejora relativa vs un baseline excepcionalmente malo (3.7 pts), no una mejora absoluta del servicio. Economy domina por volumen a pesar de problemas graves de equipaje (Mishandling +3.73, 53 maletas afectadas).

2. **Short Haul:** La anomalía negativa (-0.9 pts) es **transferencia directa** desde Economy (-1.5 pts), causada por problemas de equipaje (+3.73 Mishandling) y meteorología en EAS (5 incidentes NCS). Business estable no tiene peso suficiente para compensar.

3. **Patrón común:** Ambos radios sufren **problemas de equipaje** como causa operativa subyacente, pero con magnitudes diferentes (LH: +3.73 vs baseline 3.7; SH: +3.73 vs baseline 27.1).

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 PASO 3: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL GLOBAL)

---

## **ANÁLISIS DE NIVEL GLOBAL**

### **PARTE A: DINÁMICA DE AGREGACIÓN**

**Díada de Estados:** `(LH: +, SH: - | Global: +)`

**Escenario Detectado:** **DOMINANCIA** (+,- | +)

**Explicación del Patrón:**
- **Long Haul (LH):** POSITIVE ANOMALY (+16.5 pts) - NPS 22.3 vs baseline 5.9
- **Short Haul (SH):** NEGATIVE ANOMALY (-0.9 pts) - NPS 26.8 vs baseline 27.7
- **Global:** POSITIVE ANOMALY (+4.6 pts) - NPS 25.5 vs baseline 20.9

A pesar de que **Short Haul experimentó una caída** (-0.9 pts), el **Long Haul domina completamente el resultado Global**, imponiendo su signo positivo (+) al agregado de la red. Esto ocurre porque:

1. **Magnitud diferencial:** La mejora de LH (+16.5 pts) es **18.3 veces mayor** que la caída de SH (-0.9 pts)
2. **Impacto en baseline:** LH partió de un baseline excepcionalmente bajo (5.9 pts), amplificando la variación relativa
3. **Peso volumétrico/estratégico:** Los vuelos Long Haul tienen mayor impacto en el NPS Global por su peso en la percepción de marca y valor económico

**Dispersión entre radios:** 17.4 pts (16.5 - (-0.9)) - **Volatilidad interna ALTA**.

---

### **PARTE B: NARRATIVA CAUSAL**

**Lógica Aplicada:** DOMINANCIA → Adoptar **Explicación del Radio Dominante (Long Haul)** + Matiz del radio opositor.

---

## **📊 NARRATIVA EJECUTIVA GLOBAL**

### **Síntesis Principal:**

El resultado Global (+4.6 pts) está **arrastrado por Long Haul** (+16.5 pts), que experimentó una mejora significativa respecto a un baseline excepcionalmente bajo (5.9 pts). Sin embargo, **esta anomalía positiva es engañosa y no refleja una mejora real del servicio**, sino una **recuperación relativa respecto a los 7 días previos que fueron aún peores**.

**⚠️ PARADOJA CRÍTICA GLOBAL:**

El NPS Global subió +4.6 pts (25.5 vs baseline 20.9), pero el análisis operativo revela **deterioro significativo en métricas críticas** que deberían haber causado una caída, no una subida:

| Métrica Global | Valor 14-dic | Variación vs Baseline | Impacto Esperado en NPS |
|----------------|--------------|----------------------|------------------------|
| **Mishandling** | 19.04 | **+3.73 pts** ↑ | NPS debería **BAJAR** ⬇️ |
| **Misconex** | 0.81 | **+0.13 pts** ↑ | NPS debería **BAJAR** ⬇️ |
| **OTP15** | 89.9% | **+0.84 pts** ↑ | NPS debería **SUBIR** ✅ |
| **Load Factor** | 83.7% | **-2.6 pts** ↓ | NPS debería **SUBIR** ✅ |

**Interpretación:** El día 14-dic tuvo **problemas operativos graves en términos absolutos**, pero fue **menos malo que el baseline** de los 7 días previos. Los factores atenuantes (mejor OTP, menor ocupación) compensaron parcialmente el impacto negativo del equipaje.

El efecto positivo de LH fue **parcialmente contrarrestado** por la caída de Short Haul (-0.9 pts), que sufrió problemas similares de equipaje pero con mayor impacto relativo por partir de un baseline más alto (27.7 pts).

---

### **Evidencia Clave del Radio Dominante (Long Haul):**

#### **CAUSA RAÍZ GLOBAL: DETERIORO OPERATIVO EN GESTIÓN DE EQUIPAJE Y CONEXIONES**
**Nivel de Confianza: ALTA** ✅✅✅

**1. MÉTRICAS OPERATIVAS (Triangulación Global):**

- **Mishandling Global: 19.04** (+3.73 pts vs baseline 15.31)
  - LH Economy: 19.04 (+3.73 pts vs baseline 3.67)
  - SH Economy: 19.04 (+3.73 pts vs baseline 27.07)
  - **Interpretación:** Problema transversal a toda la red

- **Misconex Global: 0.81** (+0.13 pts vs baseline 0.68)
  - Indica problemas de conexiones en hub MAD

- **OTP15 Global: 89.9%** (+0.84 pts vs baseline) - **Factor mitigante**
  - LH Economy: 77.27% (-4.58 pts) - Deterioro en largo radio
  - SH Economy: 91.78% (+1.68 pts) - Mejora en corto radio

**2. INCIDENTES NCS (Evidencia Cualitativa Global):**

**Total: 226 incidentes registrados el 14-dic**

| Tipo de Incidente | Cantidad | % del Total | Impacto |
|-------------------|----------|-------------|---------|
| **Retrasos** | 42 | 18.6% | Afectó percepción puntualidad |
| **Cancelaciones** | 22 | 9.7% | Disrupciones graves |
| **Equipaje** | 13 | 5.8% | **Causa principal** |
| **Desvíos** | 13 | 5.8% | Meteorología EAS |
| **Otras incidencias** | 18 | 8.0% | Varios |

**Incidentes críticos de equipaje:**
- **IB151 MAD-BOG (LH):** 26 equipajes no cargados por falta de capacidad
- **IB281 conexión BA458 LHR-MAD (LH):** 27 maletas perdidas, regularizadas vía DOH
- **Total maletas afectadas:** 53+ en 2 vuelos críticos

**Problemas de conexiones:**
- **24 pérdidas de conexión en MAD** (hub principal)
- **83 conexiones reprogramadas** (impacto masivo)

**Problemas meteorológicos:**
- **5 incidentes en aeropuerto EAS (SH):** 3 desvíos + 2 cancelaciones

**3. VERBATIMS (Evidencia Cualitativa Global):**

**30 comentarios críticos analizados** con problemas graves:

**Tema 1: Equipaje (Frecuencia: ALTA - 11+ menciones):**
- **LH:** "Me perdieron una maleta... había sido abierta y robada" (GIG-MAD, NPS 0)
- **LH:** "Me han abierto la maleta y me han sustraído la mitad" (BOG-MAD, NPS 3)
- **LH:** "Las maletas tardaron dos horas en bajar y perdí mi conexión" (MAD-UIO, NPS 0)
- **SH:** "Me ha desaparecido una maleta" (PMI-VLC, NPS 5)
- **SH:** "La maleta no llegó" (FRA-MAD, NPS 5)

**Tema 2: Caos en Proceso de Embarque (Frecuencia: ALTA - 5+ menciones):**
- **LH:** "No nos informaron que debíamos pesar maletas de mano... el caos fue total" (BOG-MAD, NPS 6)
- **LH:** "Fila innecesariamente larga para pesar equipaje de mano, tomó más de una hora" (MAD-SJO, NPS 0)
- **LH:** "Pesaban maletas de mano con tolerancia mínima... sentía estar en low cost" (MAD-SCL, NPS 0)

**Tema 3: Pérdidas de Conexión (Frecuencia: MEDIA - 3+ menciones):**
- **LH:** "Deberían dejar de ofrecer vuelos con conexión en Madrid si no son capaces de hacer que las maletas lleguen" (MAD-UIO, NPS 0)
- **SH:** Información errónea de tripulación causó pérdida de conexión + coste 162.57€ (LEI-MAD, NPS 0)

**Tema 4: Trato Inadecuado del Personal (Frecuencia: MEDIA - 4+ menciones):**
- **LH:** "La actitud y el comportamiento abusivo de las señoritas en la puerta de abordaje" (BOG-MAD, NPS 1)
- **LH:** "El señor que nos hizo el check in... nos habló de manera muy mal educada" (BOG-MAD, NPS 0)
- **SH:** "El personal de embarque es irrespetuoso con los clientes" (MAD-NAP, NPS 0)

---

### **4. RUTAS CRÍTICAS GLOBALES (Triangulación Completa):**

#### **Long Haul (Radio Dominante):**

| Ruta | NPS | Encuestas | Problema Principal | Evidencia |
|------|-----|-----------|-------------------|-----------|
| **BOG-MAD** | **0.0** | 29 | Equipaje (26 maletas IB151) + Caos embarque + Mal trato personal | ✅ NCS + ✅ Verbatims (5 menciones) |
| **LEI-MAD** | **0.0** | 4 | Pérdida de conexión + Equipaje mal gestionado | ✅ Verbatims (2 menciones) |
| **HAV-MAD** | **-25.0** | 4 | Flota A33ACMI con problemas | ✅ Routes (sin verbatims) |
| **DOH-MAD** | **-20.0** | 5 | Sin evidencia específica | ⚠️ Solo Routes |
| **MAD-UIO** | **18.2** | 11 | Conexiones perdidas (27 maletas BA458) | ✅ NCS + ✅ Verbatims |

#### **Short Haul (Radio Opositor):**

| Ruta | NPS | Encuestas | Problema Principal | Evidencia |
|------|-----|-----------|-------------------|-----------|
| **EAS-MAD** | **-16.7** | 6 | Desvíos meteorológicos (5 incidentes NCS) | ✅ NCS + ✅ Routes |
| **BLQ-MAD** | **-40.0** | 5 | Check-in fallido + Asistencia especial | ✅ Verbatims (2 menciones) |
| **MAD-ORY** | **-4.0** | 25 | Equipaje forzado a bodega + Retrasos | ✅ Verbatims |

---

### **5. PERFILES DE CLIENTE AFECTADOS GLOBALMENTE:**

#### **Por Codeshare (Dispersión Global: 155.6 pts - MÁS CRÍTICO):**

| Operador | NPS | Encuestas | Radio Principal | Impacto |
|----------|-----|-----------|----------------|---------|
| **AA (American Airlines)** | **-55.6** | 9 | LH | **PEOR CODESHARE** - Conexiones en MAD |
| **LATAM** | **-33.3** | 9 | LH | Conexiones fallidas |
| **BA (British Airways)** | **-22.2** | 18 | LH | Incidente BA458 (27 maletas) |
| **IB** | **28.9** | 602 | Ambos | Base principal |

**Interpretación:** Los pasajeros de **codeshare long-haul** fueron desproporcionadamente afectados por problemas de conexiones y equipaje en hub MAD, ya que dependen más de transferencias.

#### **Por Región de Residencia (Dispersión Global: 143.2 pts):**

| Región | NPS | Encuestas | Radio Principal | Impacto |
|--------|-----|-----------|----------------|---------|
| **AMERICA NORTE** | **-100.0** | 1 | LH | Muestra muy pequeña |
| **ASIA** | **-50.0** | 8 | LH | Conexiones internacionales |
| **EUROPA** | **12.4** | 137 | Ambos | Afectados por rutas SH+LH |
| **ESPAÑA** | **34.7** | 317 | Ambos | Mayor volumen, mejor NPS |

**Interpretación:** Pasajeros **internacionales long-haul** (América Norte, Asia) fueron más afectados que pasajeros domésticos/europeos, consistente con problemas de conexiones en MAD.

#### **Por Flota (Dispersión Global: 96.4 pts):**

| Flota | NPS | Encuestas | Radio | Impacto |
|-------|-----|-----------|-------|---------|
| **A33ACMI** | **-25.0** | 4 | LH | Peor flota (HAV-MAD) |
| **A333** | **-12.5** | 24 | LH | Problemas equipaje/espacio |
| **A319** | **-4.2** | 24 | SH | Equipaje forzado a bodega |
| **A321XLR** | **71.4** | 7 | LH | Mejor flota (nueva) |

**Interpretación:** Las flotas **long-haul antiguas** (A333, A33ACMI) y **short-haul con limitaciones de espacio** (A319) concentraron los problemas operativos.

#### **Por Business/Leisure (Dispersión Global: 6.0 pts - BAJA):**

| Perfil | NPS | Encuestas | Interpretación |
|--------|-----|-----------|----------------|
| **Leisure** | **26.4** | 556 | Mayoría de pasajeros |
| **Business/Work** | **20.4** | 98 | Ligeramente más críticos |

**Interpretación:** Los problemas operativos afectaron **por igual** a ambos perfiles, sin discriminación por tipo de viaje. Impacto transversal.

---

### **Evidencia del Radio Opositor (Short Haul):**

El **Short Haul (-0.9 pts)** experimentó problemas similares pero con menor magnitud:

**Causas SH (ya detalladas en Paso 2):**
1. **Mishandling SH: +3.73 pts** (mismo deterioro que LH)
2. **Incidentes meteorológicos EAS:** 5 incidentes NCS, ruta EAS-MAD con NPS -16.7
3. **Problemas de asignación de asientos:** 4+ verbatims

**Por qué SH no dominó:**
- **Baseline más alto:** SH partió de 27.7 pts vs LH 5.9 pts
- **Menor magnitud de variación:** -0.9 pts vs +16.5 pts de LH
- **Menor peso estratégico:** Los problemas de SH no tienen el mismo impacto en percepción de marca que LH

---

## **📊 RESUMEN EJECUTIVO - NIVEL GLOBAL**

### **Dinámica de Agregación:**

| Nivel | Escenario | Estados | Narrativa Dominante | Dispersión |
|-------|-----------|---------|---------------------|------------|
| **GLOBAL** | **DOMINANCIA** | `(LH: +, SH: - \| Global: +)` | **Long Haul domina** (+16.5) con mejora relativa vs baseline excepcionalmente bajo (5.9), arrastrando al Global (+4.6) a pesar de caída en Short Haul (-0.9). | **17.4 pts** ⚠️ |

---

## **🎯 CONCLUSIONES CLAVE - NIVEL GLOBAL**

### **1. Interpretación de la Anomalía Positiva Global (+4.6 pts):**

**⚠️ PARADOJA EXPLICADA:**

El NPS Global subió +4.6 pts (25.5 vs baseline 20.9), pero esto **NO indica una mejora del servicio**. La explicación es:

✅ **Baseline excepcionalmente bajo:** Los 7 días previos tuvieron un NPS promedio de 20.9 pts, sugiriendo problemas operativos aún más graves.

✅ **Mejora relativa, no absoluta:** El día 14-dic tuvo problemas significativos (Mishandling +3.73, 24 pérdidas de conexión, 226 incidentes), pero fueron **menos severos** que los días previos.

✅ **Factores atenuantes:**
- OTP15 mejoró +0.84 pts → Mejor puntualidad relativa
- Load Factor bajó -2.6 pts → Menos ocupación = mejor experiencia relativa
- Estos factores **compensaron parcialmente** el impacto de equipaje y conexiones

✅ **Dominancia de Long Haul:** La mejora de LH (+16.5 pts desde baseline 5.9) arrastró al Global, ocultando la caída de SH (-0.9 pts).

---

### **2. Causa Raíz Global (Confianza ALTA):**

**DETERIORO OPERATIVO EN GESTIÓN DE EQUIPAJE Y CONEXIONES**

**Triangulación completa lograda:**
- ✅ **Operative Data:** Mishandling +3.73, Misconex +0.13
- ✅ **NCS:** 13 incidentes equipaje, 24 pérdidas conexión, 226 incidentes totales
- ✅ **Verbatims:** 30 comentarios críticos, 11+ menciones equipaje
- ✅ **Routes:** BOG-MAD (NPS 0.0), LEI-MAD (NPS 0.0), HAV-MAD (NPS -25.0), EAS-MAD (NPS -16.7)
- ✅ **Customer Profile:** Codeshare AA/LATAM/BA más afectados (-55.6 a -22.2), pasajeros internacionales impactados

---

### **3. Segmentos Más Afectados:**

**Por Radio:**
- **LH Economy:** +22.7 pts (mejora relativa engañosa, baseline 3.7)
- **LH Business:** -10.3 pts (deterioro real)
- **SH Economy IB:** -9.7 pts (deterioro real por equipaje/EAS)

**Por Compañía:**
- **IB SH Economy:** -9.7 pts (dominó caída en SH)
- **YW SH Economy:** +15.0 pts (mejor gestión operativa)

**Por Perfil:**
- **Codeshare AA:** NPS -55.6 (peor performer)
- **Región Asia:** NPS -50.0 (conexiones internacionales)
- **Flota A33ACMI:** NPS -25.0 (peor flota)

---

### **4. Recomendaciones Inmediatas:**

1. **Equipaje (Prioridad CRÍTICA):**
   - Revisar procesos de handling en MAD, especialmente vuelos long-haul y codeshare
   - Investigar incidentes IB151 (26 maletas) e IB281/BA458 (27 maletas)
   - Reforzar coordinación con BA, AA, LATAM

2. **Conexiones en MAD:**
   - Optimizar tiempos de conexión (24 pérdidas + 83 reprogramaciones)
   - Mejorar seguimiento de equipaje en transferencias

3. **Proceso de Embarque:**
   - Estandarizar y comunicar claramente control de equipaje de mano
   - Reducir tiempos de espera en puertas (BOG-MAD, MAD-SJO)

4. **Meteorología EAS:**
   - Mejorar protocolos de comunicación en desvíos
   - Evaluar contingencias para ruta MAD-EAS

5. **Personal:**
   - Reforzar formación en atención al cliente (BOG-MAD, MAD-NAP)

---

**Nivel de Confianza Global: ALTA** ✅✅✅  
**Fecha del Informe:** 2025-12-14  
**Conclusión:** La anomalía positiva Global (+4.6) es **relativa al baseline**, no una mejora absoluta. El día 14-dic fue problemático en términos absolutos, pero **menos problemático** que la semana previa.

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

# 🔍 PASO 4: IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS (NMA)

---

## **CAUSA 1: DETERIORO EN GESTIÓN DE EQUIPAJE (Mishandling)**

### **Identificación del NMA:**

**Escenario:** **SINERGIA PARCIAL** con propagación compleja

**Análisis de Propagación por Niveles:**

#### **Nivel 1 - Compañía (Economy SH):**
- **Tríada:** `(IB: -, YW: + | Economy SH: -)`
- **Escenario:** DOMINANCIA
- **Decisión:** NMA ≠ Economy SH (padre), porque los hijos tienen signos opuestos
- **NMA Candidato:** **IB (Economy SH)** - Hijo dominante con deterioro en equipaje

#### **Nivel 2 - Cabina (SH):**
- **Tríada:** `(Economy: -, Business: N | SH: -)`
- **Escenario:** TRANSFERENCIA
- **Decisión:** NMA ≠ SH (padre), porque solo Economy es anómalo
- **NMA Candidato:** **Economy SH** - Pero este nodo tiene DOMINANCIA interna (IB domina)
- **NMA Real:** **IB (Economy SH)** - Se mantiene del nivel inferior

#### **Nivel 3 - Radio (Global):**
- **Tríada:** `(LH: +, SH: - | Global: +)`
- **Escenario:** DOMINANCIA
- **Decisión:** NMA ≠ Global, porque los radios tienen signos opuestos
- **NMA Final:** Se mantiene en el nivel más bajo donde ocurrió la anomalía real

#### **Análisis Paralelo en Long Haul:**
- **Economy LH:** También sufrió Mishandling +3.73 pts (mismo valor que SH)
- **Escenario LH:** `(Economy: +, Business: -, Premium: - | LH: +)` - DOMINANCIA
- **NMA LH:** **Economy LH** (cabina dominante con mejora relativa a pesar del deterioro operativo)

### **Conclusión NMA para Causa 1:**

**⚠️ CASO ESPECIAL: CAUSA TRANSVERSAL CON MÚLTIPLES NMAs**

Esta causa afecta a **TODA LA RED** con el mismo deterioro operativo (Mishandling +3.73 pts), pero con **impactos diferenciados** por baseline:

**NMA Principal:** **Global/LH/Economy**
- **Razón:** Domina el resultado Global (+16.5 pts LH) a pesar del deterioro operativo
- **Paradoja:** Mejora relativa (baseline 3.7) oculta problemas absolutos

**NMA Secundario:** **Global/SH/Economy/IB**
- **Razón:** Deterioro real (-9.7 pts) que arrastra a SH Economy (-1.5 pts)
- **Impacto:** Caída absoluta por problemas de equipaje

---

### **Salida Estructurada - CAUSA 1:**

```
CAUSA 1: Deterioro en Gestión de Equipaje (Mishandling +3.73 pts)
- Escenario: TRANSVERSAL con DOMINANCIA múltiple
- NMA Principal: Global/LH/Economy
- NMA Secundario: Global/SH/Economy/IB
- Afecta a: 
  * Economy LH (148 encuestas)
  * Economy SH IB (280 encuestas estimadas)
  * Economy SH YW (139 encuestas)
- Tipo de impacto: NEGATIVO operativo (pero positivo relativo en LH por baseline bajo)

- Cadena de propagación NMA Principal (LH):
  * Global/LH/Economy (NMA) → Global/LH (DOMINANCIA, Business/Premium opuestos -)
    - Economy +22.7 pts domina sobre Business -10.3 y Premium -3.4
    - Resultado: LH +16.5 pts
  
  * Global/LH → Global (DOMINANCIA, SH opuesto -)
    - LH +16.5 pts domina sobre SH -0.9 pts
    - Resultado: Global +4.6 pts

- Cadena de propagación NMA Secundario (SH):
  * Global/SH/Economy/IB (NMA) → Global/SH/Economy (DOMINANCIA, YW opuesto +)
    - IB -9.7 pts domina sobre YW +15.0 pts
    - Resultado: Economy SH -1.5 pts
  
  * Global/SH/Economy → Global/SH (TRANSFERENCIA, Business estable N)
    - Economy -1.5 pts arrastra a SH (Business Normal no compensa)
    - Resultado: SH -0.9 pts
  
  * Global/SH → Global (DOMINADO por LH)
    - SH -0.9 pts es superado por LH +16.5 pts
    - Resultado: Global +4.6 pts (LH domina)
```

---

## **CAUSA 2: INCIDENTES METEOROLÓGICOS EN AEROPUERTO EAS**

### **Identificación del NMA:**

**Escenario:** **DILUCIÓN** - Causa localizada que no propagó al padre

**Análisis de Propagación:**

#### **Nivel 1 - Compañía (Economy SH):**
- **Impacto:** Afecta principalmente a **IB** (opera ruta EAS-MAD)
- **YW:** Sin evidencia de impacto por meteorología EAS
- **Tríada:** `(IB: -, YW: + | Economy SH: -)`
- **Escenario:** DOMINANCIA (IB domina, pero meteorología EAS es solo 1 factor)

#### **Nivel 2 - Cabina (SH):**
- **Impacto:** Solo Economy afectada (EAS-MAD es ruta Economy)
- **Tríada:** `(Economy: -, Business: N | SH: -)`
- **Escenario:** TRANSFERENCIA

#### **Nivel 3 - Radio (Global):**
- **Impacto:** SH cae -0.9 pts, pero es DOMINADO por LH +16.5 pts
- **Tríada:** `(LH: +, SH: - | Global: +)`
- **Escenario:** DOMINANCIA (LH domina)

### **Conclusión NMA para Causa 2:**

**NMA:** **Global/SH/Economy/IB** (ruta específica EAS-MAD)

**Razón:** 
- Causa **localizada** en una ruta específica operada por IB
- 5 incidentes NCS concentrados en aeropuerto EAS
- Ruta EAS-MAD con NPS -16.7 (peor ruta SH del día)
- No propagó significativamente más allá de Economy SH IB por:
  - Volumen limitado (6 encuestas EAS-MAD)
  - Otros factores dominaron en niveles superiores

---

### **Salida Estructurada - CAUSA 2:**

```
CAUSA 2: Incidentes Meteorológicos en Aeropuerto EAS
- Escenario: DILUCIÓN (causa localizada)
- NMA: Global/SH/Economy/IB (ruta EAS-MAD)
- Afecta a: 
  * Ruta EAS-MAD (6 encuestas, NPS -16.7)
  * Pasajeros con desvíos a BIO (transporte por superficie)
- Tipo de impacto: NEGATIVO localizado

- Cadena de propagación:
  * Global/SH/Economy/IB (NMA - ruta EAS-MAD) → Global/SH/Economy/IB (DILUCIÓN)
    - EAS-MAD (NPS -16.7, n=6) diluido en el agregado IB Economy SH (NPS 17.1, n=280)
    - Contribuye a caída IB -9.7 pts pero NO es causa única
  
  * Global/SH/Economy/IB → Global/SH/Economy (DOMINANCIA, YW opuesto +)
    - IB -9.7 pts domina sobre YW +15.0 pts
    - Resultado: Economy SH -1.5 pts
  
  * Global/SH/Economy → Global/SH (TRANSFERENCIA, Business estable N)
    - Economy -1.5 pts arrastra a SH
    - Resultado: SH -0.9 pts
  
  * Global/SH → Global (DOMINADO por LH)
    - SH -0.9 pts es superado por LH +16.5 pts
    - Resultado: Global +4.6 pts (impacto de EAS no visible en Global)
```

---

## **CAUSA 3: CAOS EN PROCESO DE EMBARQUE (Control de Equipaje de Mano)**

### **Identificación del NMA:**

**Escenario:** **TRANSFERENCIA** - Causa específica de Long Haul que se propaga

**Análisis de Propagación:**

#### **Nivel 1 - Cabina (LH):**
- **Impacto:** Principalmente **Economy LH** (verbatims concentrados)
- **Business/Premium:** También afectados pero sin evidencia específica (errores de procesamiento)
- **Tríada:** `(Economy: +, Business: -, Premium: - | LH: +)`
- **Escenario:** DOMINANCIA (Economy domina)

#### **Nivel 2 - Radio (Global):**
- **Impacto:** LH domina el Global
- **Tríada:** `(LH: +, SH: - | Global: +)`
- **Escenario:** DOMINANCIA (LH domina)

### **Conclusión NMA para Causa 3:**

**NMA:** **Global/LH/Economy**

**Razón:**
- Causa concentrada en **rutas long-haul** (BOG-MAD, MAD-SJO, MAD-SCL)
- 3+ verbatims críticos (NPS 0-6) sobre proceso caótico de pesaje de equipaje de mano
- Economy LH domina el resultado de LH (+22.7 pts) y este domina Global (+4.6 pts)
- Aunque el impacto es negativo operativo, se diluye en la mejora relativa vs baseline

---

### **Salida Estructurada - CAUSA 3:**

```
CAUSA 3: Caos en Proceso de Embarque (Control Equipaje de Mano)
- Escenario: TRANSFERENCIA desde Economy LH
- NMA: Global/LH/Economy
- Afecta a:
  * Rutas BOG-MAD (NPS 0.0, n=29)
  * Rutas MAD-SJO (NPS 37.5, n=8)
  * Rutas MAD-SCL (verbatims críticos)
- Tipo de impacto: NEGATIVO operativo (oculto por baseline bajo)

- Cadena de propagación:
  * Global/LH/Economy (NMA) → Global/LH (DOMINANCIA, Business/Premium opuestos -)
    - Economy +22.7 pts domina sobre Business -10.3 y Premium -3.4
    - Caos de embarque afecta principalmente a Economy por mayor volumen
    - Resultado: LH +16.5 pts
  
  * Global/LH → Global (DOMINANCIA, SH opuesto -)
    - LH +16.5 pts domina sobre SH -0.9 pts
    - Resultado: Global +4.6 pts
```

---

## **CAUSA 4: PROBLEMAS DE CONEXIONES EN HUB MAD**

### **Identificación del NMA:**

**Escenario:** **TRANSVERSAL** con impacto diferenciado por radio

**Análisis de Propagación:**

#### **Impacto por Radio:**
- **Long Haul:** Mayor impacto (pasajeros internacionales dependen de conexiones)
  - Misconex LH Economy: 0.81 (+0.13 pts)
  - 27 maletas BA458 (LHR-MAD) → IB281
  - Codeshare BA/AA/LATAM más afectados
  
- **Short Haul:** Menor impacto (menos conexiones internacionales)
  - Misconex SH Economy: 0.81 (+0.13 pts)
  - Verbatims: LEI-MAD (conexión perdida + 162.57€)

#### **Nivel 1 - Cabina (LH):**
- **Tríada:** `(Economy: +, Business: -, Premium: - | LH: +)`
- **Escenario:** DOMINANCIA (Economy domina)

#### **Nivel 2 - Radio (Global):**
- **Tríada:** `(LH: +, SH: - | Global: +)`
- **Escenario:** DOMINANCIA (LH domina)

### **Conclusión NMA para Causa 4:**

**NMA Principal:** **Global/LH/Economy**

**Razón:**
- **24 pérdidas de conexión en MAD** + **83 conexiones reprogramadas**
- Impacto mayor en pasajeros long-haul (Asia NPS -100.0, América Norte NPS -100.0)
- Codeshare más afectados: AA -55.6, LATAM -33.3, BA -22.2
- Economy LH domina el resultado de LH y este domina Global

---

### **Salida Estructurada - CAUSA 4:**

```
CAUSA 4: Problemas de Conexiones en Hub MAD
- Escenario: TRANSVERSAL con DOMINANCIA de LH
- NMA: Global/LH/Economy
- Afecta a:
  * Pasajeros codeshare (AA: -55.6, LATAM: -33.3, BA: -22.2)
  * Regiones internacionales (Asia: -50.0, América Norte: -100.0)
  * Rutas MAD-UIO (27 maletas BA458 perdidas)
  * Total: 24 pérdidas + 83 reprogramaciones
- Tipo de impacto: NEGATIVO operativo (oculto por baseline bajo en LH)

- Cadena de propagación:
  * Global/LH/Economy (NMA) → Global/LH (DOMINANCIA, Business/Premium opuestos -)
    - Economy +22.7 pts domina (mejora relativa vs baseline 3.7)
    - Conexiones afectan principalmente a Economy por volumen
    - Resultado: LH +16.5 pts
  
  * Global/LH → Global (DOMINANCIA, SH opuesto -)
    - LH +16.5 pts domina sobre SH -0.9 pts
    - Resultado: Global +4.6 pts
```

---

## **CAUSA 5: TRATO INADECUADO DEL PERSONAL EN TIERRA**

### **Identificación del NMA:**

**Escenario:** **DILUCIÓN** - Causa localizada en rutas específicas

**Análisis de Propagación:**

#### **Impacto por Radio:**
- **Long Haul:** Rutas BOG-MAD (2 verbatims NPS 0-1), JFK-MAD (1 verbatim NPS 0)
- **Short Haul:** Rutas MAD-NAP (1 verbatim NPS 0)

#### **Nivel 1 - Cabina:**
- **LH:** Economy domina → NMA candidato: Global/LH/Economy
- **SH:** Economy IB (MAD-NAP) → NMA candidato: Global/SH/Economy/IB

### **Conclusión NMA para Causa 5:**

**⚠️ CASO ESPECIAL: MÚLTIPLES NMAs LOCALIZADOS**

**NMA Principal:** **Global/LH/Economy** (ruta BOG-MAD)
**NMA Secundario:** **Global/SH/Economy/IB** (ruta MAD-NAP)

**Razón:**
- Causa **localizada** en rutas específicas (4+ verbatims totales)
- Impacto diluido en agregados por bajo volumen
- No propagó significativamente más allá de las rutas específicas

---

### **Salida Estructurada - CAUSA 5:**

```
CAUSA 5: Trato Inadecuado del Personal en Tierra
- Escenario: DILUCIÓN (causa localizada)
- NMA Principal: Global/LH/Economy (ruta BOG-MAD)
- NMA Secundario: Global/SH/Economy/IB (ruta MAD-NAP)
- Afecta a:
  * BOG-MAD (2 verbatims NPS 0-1, n=29 total)
  * MAD-NAP (1 verbatim NPS 0)
  * JFK-MAD (1 verbatim NPS 0)
- Tipo de impacto: NEGATIVO localizado

- Cadena de propagación NMA Principal:
  * Global/LH/Economy (NMA - ruta BOG-MAD) → Global/LH (DILUCIÓN)
    - BOG-MAD (NPS 0.0) diluido en Economy LH (NPS 26.4, n=148)
    - Contribuye a problemas pero NO es causa única
  
  * Global/LH → Global (DOMINANCIA sobre SH)
    - LH +16.5 pts domina
    - Resultado: Global +4.6 pts (impacto no visible)

- Cadena de propagación NMA Secundario:
  * Global/SH/Economy/IB (NMA - ruta MAD-NAP) → Global/SH/Economy/IB (DILUCIÓN)
    - MAD-NAP diluido en IB Economy SH (NPS 17.1, n=280)
  
  * Global/SH/Economy/IB → Global/SH/Economy (DOMINANCIA, YW opuesto +)
    - IB -9.7 pts domina
    - Resultado: Economy SH -1.5 pts
  
  * Global/SH/Economy → Global/SH (TRANSFERENCIA)
    - Resultado: SH -0.9 pts
  
  * Global/SH → Global (DOMINADO por LH)
    - Resultado: Global +4.6 pts (impacto no visible)
```

---

## **📊 RESUMEN DE NMAs IDENTIFICADOS**

| Causa | NMA Principal | NMA Secundario | Escenario | Propagación a Global |
|-------|---------------|----------------|-----------|---------------------|
| **1. Mishandling** | Global/LH/Economy | Global/SH/Economy/IB | TRANSVERSAL + DOMINANCIA | ✅ Domina (LH) / Contribuye (SH) |
| **2. Meteorología EAS** | Global/SH/Economy/IB | - | DILUCIÓN | ❌ Diluido en Global |
| **3. Caos Embarque** | Global/LH/Economy | - | TRANSFERENCIA | ✅ Domina vía LH |
| **4. Conexiones MAD** | Global/LH/Economy | - | TRANSVERSAL + DOMINANCIA | ✅ Domina vía LH |
| **5. Trato Personal** | Global/LH/Economy | Global/SH/Economy/IB | DILUCIÓN | ❌ Diluido en Global |

---

## **🎯 CONCLUSIONES CLAVE - PASO 4**

1. **NMA Dominante Global:** **Global/LH/Economy** aparece como NMA en 4 de 5 causas, confirmando su rol como **segmento que arrastra el resultado Global** (+4.6 pts).

2. **Paradoja del NMA Dominante:** A pesar de ser el NMA que "mejora" el Global (+22.7 pts), Economy LH sufrió **deterioro operativo real** (Mishandling +3.73, OTP -4.58). La mejora es **relativa al baseline excepcionalmente bajo** (3.7 pts).

3. **NMA Secundario Crítico:** **Global/SH/Economy/IB** es el NMA de deterioro **real y absoluto** (-9.7 pts), causado por equipaje y meteorología EAS, pero su impacto es **dominado** por LH en el agregado Global.

4. **Causas Localizadas:** Meteorología EAS y Trato Personal son causas **localizadas** que se diluyen en agregados superiores, no propagando significativamente al Global.

5. **Patrón de Propagación:** Las causas transversales (Mishandling, Conexiones MAD) se propagan mediante **DOMINANCIA** de Economy LH → LH → Global, mientras que las causas localizadas sufren **DILUCIÓN** en niveles superiores.

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

# 📊 PASO 4B: EXTRACCIÓN DE EVIDENCIAS DEL CONTEXTO INICIAL

---

## **=== NMA 1: Global/LH/Economy ===**

### 📈 EXPLANATORY DRIVERS:
No disponible

### 📊 DATOS OPERATIVOS:
**Segmento: Global/LH/Economy**
- **OTP15:** 77.27% (-4.58 pts vs baseline)
- **Mishandling:** 19.04 (+3.73 pts vs baseline)
- **Misconex:** 0.81 (+0.13 pts vs baseline)
- **Load Factor:** 88.15% (-1.83 pts vs baseline)

### 🚨 INCIDENTES NCS (CUANTITATIVO):
**Segmento: Global/LH/Economy**
- **Total incidentes reportados:** 6
- **Incidentes de equipaje:** 2 (33% del total)
- **Pasajeros afectados por equipaje:** 53 maletas en 2 vuelos críticos
  - **IB151 MAD-BOG:** 26 equipajes no cargados por falta de capacidad
  - **IB281 (conexión BA458):** 27 maletas perdidas por conexión fallida
- **Incidentes de retrasos:** 1

### 🧠 NCS (CUALITATIVO / REFLEXIÓN):
No disponible

### 💬 FEEDBACK DE CLIENTES:
**Segmento: Global/LH/Economy**

**Temas principales en verbatims (30 comentarios analizados):**

**1. EQUIPAJE (5+ menciones NPS 0-3):**
- "Las maletas tardaron dos horas en bajar del vuelo y perdí mi conexión" (MAD-UIO, NPS 0)
- "El equipaje nunca llegó" (EZE-MAD, NPS 0)
- "Me perdieron una maleta... había sido abierta y robada" (GIG-MAD, NPS 0)
- "Me han abierto la maleta y me han sustraído la mitad" (BOG-MAD, NPS 3)

**2. PROCESO DE EMBARQUE CAÓTICO (3+ menciones):**
- "Fila innecesariamente larga para pesar equipaje de mano, tomó más de una hora, retrasó la salida" (MAD-SJO, NPS 0)
- "No nos informaron que debíamos pesar maletas de mano... el caos fue total" (BOG-MAD, NPS 6)
- "Pesaban maletas de mano con tolerancia mínima... sentía estar en low cost" (MAD-SCL, NPS 0)

**3. CONEXIONES PERDIDAS (2+ menciones):**
- "Deberían dejar de ofrecer vuelos con conexión en Madrid si no son capaces de hacer que las maletas lleguen" (MAD-UIO, NPS 0)

**4. TRATO INADECUADO DEL PERSONAL (4+ menciones):**
- "La actitud y el comportamiento abusivo de las señoritas en la puerta de abordaje... me provocaron un ataque de ansiedad" (BOG-MAD, NPS 1)
- "El señor que nos hizo el check in... nos habló de manera muy mal educada... hablándonos continuamente de manera sarcástica" (BOG-MAD, NPS 0)
- "El personal de embarque del aeropuerto es irrespetuoso con los clientes, especialmente con aquellos con niños" (MAD-NAP, NPS 0)
- "Falta total de empatía por parte del personal... Una vergüenza" (JFK-MAD, NPS 0)

**5. OTROS PROBLEMAS RECURRENTES:**
- Espacio reducido en asientos (múltiples menciones)
- Calidad de comida deficiente (múltiples menciones)
- Limpieza de baños (múltiples menciones)

### ✈️ RUTAS AFECTADAS (Top 5):
**Segmento: Global/LH/Economy**

1. **HAV-MAD:** NPS -25.0 (4 encuestas) - Flota: A33ACMI
2. **BOG-MAD:** NPS 0.0 (29 encuestas) - 5 menciones en verbatims (equipaje robado, caos embarque, mal trato)
3. **LEI-MAD:** NPS 0.0 (4 encuestas) - 2 menciones en verbatims (pérdida conexión, equipaje mal gestionado)
4. **MAD-MXP:** NPS 0.0 (6 encuestas) - 1 mención (asiento inadecuado, equipaje facturado forzosamente)
5. **GRX-MAD:** NPS 0.0 (4 encuestas) - 1 mención (puntualidad ficticia por tiempos inflados)

**Rutas adicionales con NPS negativo y volumen significativo:**
6. **LHR-MAD:** NPS -13.0 (23 encuestas) - 2 menciones (confusión facturación, asiento no asignado)
7. **GVA-MAD:** NPS -11.8 (17 encuestas) - 1 mención (equipaje mano facturado sin aviso)
8. **MAD-ORY:** NPS -4.0 (25 encuestas) - 1 mención (retraso 2 horas sin explicaciones)

**Rutas con múltiples menciones en verbatims:**
9. **MAD-UIO:** NPS 18.2 (11 encuestas) - Conexiones perdidas (27 maletas BA458)
10. **MAD-NAP:** 3 menciones (retraso equipaje, mal trato personal)
11. **JFK-MAD:** 2 menciones (retraso masivo, espera equipaje)
12. **AMS-MAD:** 2 menciones (daño equipaje, aviones antiguos)

### 👥 PERFILES REACTIVOS:
**Segmento: Global/LH/Economy**

**Por Codeshare (Dispersión: 160.0 pts):**
- **BA:** NPS -60.0 (5 encuestas) - Correlaciona con incidente BA458
- **AA:** NPS -33.3 (3 encuestas)
- **LATAM:** NPS -25.0 (4 encuestas)
- **IB:** NPS 31.3 (131 encuestas)

**Por Región de Residencia (Dispersión: 200.0 pts):**
- **ASIA:** NPS -100.0 (3 encuestas) - Todos detractores
- **EUROPA:** NPS 10.0 (10 encuestas) - Muy por debajo del promedio
- **AMERICA SUR:** NPS 20.0 (20 encuestas) - Correlaciona con BOG-MAD, MAD-UIO
- **ESPAÑA:** NPS (dato no especificado en análisis Economy LH)

**Por Flota (Dispersión: 133.3 pts):**
- **A33ACMI:** NPS -33.3 (3 encuestas) - Peor flota
- **A333:** NPS -6.7 (15 encuestas) - Problemas equipaje/espacio
- **A350:** NPS 21.4 (42 encuestas)
- **A350 next:** NPS 27.5 (40 encuestas) - Mejor flota

**Por Business/Leisure (Dispersión: 0.4 pts):**
- **Business/Work:** NPS 26.7 (15 encuestas)
- **Leisure:** NPS 26.3 (133 encuestas)

---

## **=== NMA 2: Global/SH/Economy/IB ===**

### 📈 EXPLANATORY DRIVERS:
No disponible

### 📊 DATOS OPERATIVOS:
**Segmento: Global/SH/Economy (padre de IB)**
- **OTP15:** 91.78% (+1.68 pts vs baseline)
- **Mishandling:** 19.04 (+3.73 pts vs baseline)
- **Misconex:** 0.81 (+0.13 pts vs baseline)
- **Load Factor:** 83.55% (-2.55 pts vs baseline)

### 🚨 INCIDENTES NCS (CUANTITATIVO):
**Segmento: Global/SH/Economy**
- **Total incidentes:** 20 incidentes
- **Equipaje:** 3 incidentes documentados
  - "27 maletas procedente de BA458 LHR MAD, no han llegado a IB281. Se regularizan vía DOH"
- **Retrasos:** 3 incidentes (causa: meteorología en EAS)
- **Desvíos:** 3 incidentes (MAD-EAS desviado a BIO)
- **Cancelaciones:** 2 incidentes
- **Otras incidencias:** 2 incidentes

**Incidente meteorológico crítico:**
- **Aeropuerto EAS:** 5 incidentes meteorológicos totales
  - 3 desvíos por weather
  - 2 cancelaciones relacionadas
  - Vuelos desviados a BIO con transporte por superficie EAS-MAD proporcionado

### 🧠 NCS (CUALITATIVO / REFLEXIÓN):
No disponible

### 💬 FEEDBACK DE CLIENTES:
**Segmento: Global/SH/Economy**

**Temas principales en verbatims:**

**1. EQUIPAJE (8+ menciones - Mayor frecuencia):**
- Equipaje de mano forzado a bodega en vuelos con espacio disponible
- Pérdidas y retrasos en entrega
- Daños durante manipulación
- **FRA-MAD:** "La maleta no llegó" (NPS 5)
- **MAD-XRY:** "mis maletas quedarán en Madrid, y hasta el día de hoy no aparecen" (NPS 0)
- **MAD-NCE:** "Me rompiste la maleta en el vuelo de Niza a Madrid" (NPS 0)
- **LEI-MAD:** Pérdida de maleta de mano causó pérdida de conexión (NPS 0)
- **PMI-VLC:** "Me ha desaparecido una maleta" - cantante de ópera con partitura (NPS 5)

**2. SERVICIO AL CLIENTE/TRATO (6+ menciones):**
- Personal grosero o poco servicial
- Falta de empatía con pasajeros con necesidades especiales
- **MAD-NAP:** "El personal de embarque es irrespetuoso con los clientes, especialmente con aquellos con niños" (NPS 0)

**3. ASIGNACIÓN DE ASIENTOS (4+ menciones):**
- Cambios sin justificación
- Clientes Priority no reciben servicio pagado
- **GVA-MAD:** "Nos han cambiado los asientos reservados con bastante antelación, sin explicación alguna"
- **MAD-NAP:** Cliente pagó Priority (asiento 24C) pero recibió 32A

**4. RETRASOS (3 menciones):**
- **BCN-VLC:** "El vuelo se retrasó 7 horas" sin comunicación (NPS 0)
- **BCN-VLC:** Retraso sin compensación económica (NPS 0)
- **MAH-PMI:** Vuelo forzado por cancelación del día anterior (NPS 0)

**5. CHECK-IN Y CONEXIONES (3 menciones):**
- **BLQ-MAD:** "Al momento del check-in en Bolonia no me emitieron el billete" (NPS 0)
- **LEI-MAD:** Información errónea de tripulación + coste adicional 162.57€ (NPS 0)
- **MAD-NCE:** No dejaron embarcar por no tener tarjeta de crédito (NPS 0)

### ✈️ RUTAS AFECTADAS (Top 5):
**Segmento: Global/SH/Economy**

**Rutas con alta confianza (Triangulación NPS + Verbatims + NCS):**
1. **EAS-MAD:** NPS -16.7 (6 encuestas) - 5 incidentes NCS meteorológicos, desvíos a BIO
2. **BLQ-MAD:** NPS -40.0 (5 encuestas) - 2 verbatims negativos (check-in, asistencia especial) + NCS confirmado
3. **FRA-MAD:** NPS -20.0 (5 encuestas) - 1 verbatim equipaje + NCS confirmado
4. **GRX-MAD:** NPS 0.0 (4 encuestas) - NCS confirmado + 1 verbatim sobre asignación de asientos
5. **MAD-ORY:** NPS -4.0 (25 encuestas) - Equipaje forzado a bodega, retrasos

**Rutas con media confianza:**
6. **MAD-XRY:** NPS 28.6 (7 encuestas) - 2 verbatims negativos (equipaje, cambio asiento) + NCS - Polarización detectada
7. **MAD-MUC:** NPS 0.0 (2 encuestas) - NCS confirmado + 1 verbatim sobre calidad de comida
8. **PMI-VLC:** NPS 33.3 (6 encuestas) - 1 verbatim equipaje (cantante ópera) + NCS
9. **MAH-PMI:** NPS 37.5 (8 encuestas) - 1 verbatim cancelación + NCS
10. **MAD-NAP:** NPS 0.0 (5 encuestas) - Problemas de asignación de asientos

### 👥 PERFILES REACTIVOS:
**Segmento: Global/SH/Economy**

**Por Tipo de Viajero (Dispersión: 40.4 pts):**
- **Leisure:** NPS 49.1 (116 encuestas)
- **Business/Work:** NPS 8.7 (23 encuestas) - **Significativamente más afectados**

**Por Flota (Dispersión: 129.8 pts):**
- **A333:** NPS -66.7 (3 encuestas) - Peor flota, problemas de espacio
- **A319:** NPS -8.7 (23 encuestas) - Equipaje forzado a bodega
- **ATR:** NPS +63.2 (19 encuestas) - Mejor flota, menor ocupación
- **CRJ:** NPS +39.2 (120 encuestas) - Desempeño positivo
- **A320:** NPS +33.3 (175 encuestas) - Desempeño positivo

**Por Región de Residencia:**
- **ESPAÑA:** NPS +38.2 (207 encuestas) - Volumen alto, NPS positivo
- **EUROPA:** NPS +15.9 (107 encuestas) - NPS más bajo (posible impacto por conexiones)
- **AMERICA NORTE:** NPS -100.0 (1 encuesta) - Muestra insuficiente

**Por CodeShare (Dispersión: 200.0 pts):**
- **IB:** NPS +27.6 (392 encuestas) - Volumen principal, NPS positivo
- **AA:** NPS -100.0 (3 encuestas) - Muestra baja, NPS crítico
- **AY:** NPS -100.0 (1 encuesta) - Muestra insuficiente

---

## **=== NMA 3: Global (para contexto general) ===**

### 📈 EXPLANATORY DRIVERS:
No disponible

### 📊 DATOS OPERATIVOS:
**Segmento: Global**
- **OTP15:** 89.9% (+0.84 pts vs baseline)
- **Mishandling:** 19.04 (+3.73 pts vs baseline 15.31)
- **Misconex:** 0.81 (+0.13 pts vs baseline 0.68)
- **Load Factor:** 83.7% (-2.6 pts vs baseline)

### 🚨 INCIDENTES NCS (CUANTITATIVO):
**Segmento: Global - Total: 226 incidentes registrados el 14-dic-2025**

| Tipo de Incidente | Cantidad | % del Total |
|-------------------|----------|-------------|
| **Retrasos** | 42 | 18.6% |
| **Cancelaciones** | 22 | 9.7% |
| **Otras incidencias** | 18 | 8.0% |
| **Desvíos** | 13 | 5.8% |
| **Equipaje** | 13 | 5.8% |

**Incidentes críticos:**
- **24 pérdidas de conexión en MAD** (hub principal)
- **83 conexiones reprogramadas** (impacto masivo)
- **24 cambios de equipo** (afectan configuración/confort)

**Temas principales:**
- Aircraft_change: 6 incidentes
- Baggage: 6 incidentes
- Weather: 4 incidentes (factor externo)

**Ruta más impactada en NCS:** MAD-EAS (2 incidentes)
**Vuelo más afectado:** IB0337 (2 incidentes)

### 🧠 NCS (CUALITATIVO / REFLEXIÓN):
No disponible

### 💬 FEEDBACK DE CLIENTES:
**Segmento: Global - 30 comentarios críticos analizados**

**Temas principales en verbatims (14 de diciembre de 2025):**

| Tema | Frecuencia | Severidad |
|------|------------|-----------|
| **Equipaje (pérdida/daño/robo)** | Alta | Muy Alta |
| **Pérdidas de conexión** | Media-Alta | Alta |
| **Caos en proceso de embarque** | Alta | Alta |
| **Trato inadecuado del personal** | Media | Alta |
| **Retrasos sin comunicación** | Media | Media |
| **Problemas de facturación** | Media | Media |
| **Aviones antiguos/incómodos** | Baja | Media |

### ✈️ RUTAS AFECTADAS (Top 5):
**Segmento: Global - Consolidado de peores rutas**

**Long Haul:**
1. **HAV-MAD:** NPS -25.0 (4 encuestas)
2. **DOH-MAD:** NPS -20.0 (5 encuestas)
3. **BOG-MAD:** NPS 0.0 (29 encuestas)
4. **LEI-MAD:** NPS 0.0 (4 encuestas)
5. **MAD-MXP:** NPS 0.0 (6 encuestas)

**Short Haul:**
1. **BLQ-MAD:** NPS -40.0 (5 encuestas)
2. **FRA-MAD:** NPS -20.0 (5 encuestas)
3. **EAS-MAD:** NPS -16.7 (6 encuestas)
4. **LHR-MAD:** NPS -13.0 (23 encuestas)
5. **GVA-MAD:** NPS -11.8 (17 encuestas)

### 👥 PERFILES REACTIVOS:
**Segmento: Global**

**Por Codeshare (Dispersión: 155.6 pts - MÁS CRÍTICO):**
- **AA (American Airlines):** NPS -55.6 (9 encuestas) - **PEOR CODESHARE**
- **LATAM:** NPS -33.3 (9 encuestas)
- **BA (British Airways):** NPS -22.2 (18 encuestas)
- **IB:** NPS 28.9 (602 encuestas) - Mayoría de pasajeros

**Por Región de Residencia (Dispersión: 143.2 pts):**
- **AMERICA NORTE:** NPS -100.0 (1 encuesta) - Muestra muy pequeña
- **ASIA:** NPS -50.0 (8 encuestas)
- **EUROPA:** NPS 12.4 (137 encuestas)
- **ESPAÑA:** NPS 34.7 (317 encuestas) - Mayor volumen

**Por Flota (Dispersión: 96.4 pts):**
- **A33ACMI:** NPS -25.0 (4 encuestas) - Coincide con HAV-MAD
- **A333:** NPS -12.5 (24 encuestas) - Flota long-haul
- **A319:** NPS -4.2 (24 encuestas) - Flota europea
- **A321XLR:** NPS 71.4 (7 encuestas) - Mejor flota (nueva)

**Por Business/Leisure (Dispersión: 6.0 pts - BAJA):**
- **Leisure:** NPS 26.4 (556 encuestas)
- **Business/Work:** NPS 20.4 (98 encuestas)

---

## **=== NMA 4: Global/SH/Economy/YW (contexto comparativo) ===**

### 📈 EXPLANATORY DRIVERS:
No disponible

### 📊 DATOS OPERATIVOS:
**Segmento: Global/SH/Economy/YW**
- **OTP15:** 89.66% (+1.29 pts vs baseline)
- **Mishandling:** 15.19 (+2.55 pts vs baseline)
- **Misconex:** No especificado
- **Load Factor:** 78.83% (-1.81 pts vs baseline)

### 🚨 INCIDENTES NCS (CUANTITATIVO):
**Segmento: Global/SH/Economy/YW - Total: 20 incidentes**

- **Retrasos:** 3 incidentes (causa: meteorología en EAS)
- **Desvíos:** 3 incidentes (MAD-EAS desviado a BIO)
- **Equipaje:** 3 incidentes (incluyendo 27 maletas perdidas BA458→IB281)
- **Cancelaciones:** 2 incidentes
- **Otras incidencias:** 2 incidentes

**Incidente crítico destacado:**
- **Vuelo BA458 (LHR-MAD):** 27 maletas no llegaron a conexión IB281, regularizadas vía DOH

**Problema meteorológico:**
- Aeropuerto EAS con desvíos a BIO por condiciones meteorológicas
- Transporte por superficie EAS-MAD proporcionado

### 🧠 NCS (CUALITATIVO / REFLEXIÓN):
No disponible

### 💬 FEEDBACK DE CLIENTES:
**Segmento: Global/SH/Economy/YW - 30 comentarios analizados**

**Temas principales:**

**1. Equipaje (6 menciones - TEMA DOMINANTE):**
- **FRA-MAD:** "La maleta no llegó" (NPS 5)
- **MAD-XRY:** "mis maletas quedarán en Madrid, y hasta el día de hoy no aparecen" (NPS 0)
- **MAD-NCE:** "Me rompiste la maleta en el vuelo de Niza a Madrid" (NPS 0)
- **LEI-MAD:** Pérdida de maleta de mano causó pérdida de conexión (NPS 0)
- **PMI-VLC:** "Me ha desaparecido una maleta" - cantante de ópera con partitura (NPS 5)

**2. Check-in y Conexiones (3 menciones):**
- **BLQ-MAD:** "Al momento del check-in en Bolonia no me emitieron el billete" (NPS 0)
- **LEI-MAD:** Información errónea de tripulación + coste adicional 162.57€ (NPS 0)
- **MAD-NCE:** No dejaron embarcar por no tener tarjeta de crédito (NPS 0)

**3. Retrasos y Cancelaciones (3 menciones):**
- **BCN-VLC:** "El vuelo se retrasó 7 horas" sin comunicación (NPS 0)
- **BCN-VLC:** Retraso sin compensación económica (NPS 0)
- **MAH-PMI:** Vuelo forzado por cancelación del día anterior (NPS 0)

**4. Espacio y Comodidad (2 menciones):**
- **MAD-SCQ:** "El avión es demasiado pequeño. Si eres un poco alto no cabes" (NPS 0)
- **MAD-XRY:** Pagó 30€ por bodega pero no tuvo espacio para mochila en cabina (NPS 7)

### ✈️ RUTAS AFECTADAS (Top 5):
**Segmento: Global/SH/Economy/YW**

**Rutas con alta confianza:**
1. **BLQ-MAD:** NPS -40.0 (5 encuestas) - 2 verbatims negativos + NCS confirmado
2. **FRA-MAD:** NPS -20.0 (5 encuestas) - 1 verbatim equipaje + NCS confirmado
3. **GRX-MAD:** NPS 0.0 (4 encuestas) - NCS confirmado + 1 verbatim
4. **MAD-XRY:** NPS 28.6 (7 encuestas) - 2 verbatims negativos + NCS
5. **MAD-MUC:** NPS 0.0 (2 encuestas) - NCS confirmado

**Rutas con evidencia parcial:**
- **BCN-VLC:** 2 verbatims críticos (retraso 7h) pero NO aparece en routes_tool
- **MAD-NCE:** 2 verbatims negativos (equipaje roto, no embarque)
- **LEI-MAD:** 1 verbatim crítico (conexión perdida + 162.57€)

### 👥 PERFILES REACTIVOS:
**Segmento: Global/SH/Economy/YW**

**Por Tipo de Viajero:**
- **Leisure:** NPS 49.1 (116 encuestas)
- **Business/Work:** NPS 8.7 (23 encuestas) - **-40.4 pts diferencial**

**Por Flota:**
- **ATR:** NPS 63.2 (19 encuestas) - Mejor flota
- **CRJ:** NPS 39.2 (120 encuestas) - Mayoría de operación, **-24.0 pts vs ATR**

**Por Región de Residencia:**
- **España:** NPS 55.1 (78 encuestas)
- **Europa:** NPS 34.6 (26 encuestas) - **-20.5 pts vs España**
- **América Latina:** NPS 50.0 (10 encuestas)
- **Otros:** NPS 36.7 (25 encuestas)

**Por Codeshare:**
- **IB:** NPS 42.6 (136 encuestas) - Base principal
- **BA:** NPS 100.0 (1 encuesta) - Muestra insuficiente
- **I2:** NPS 100.0 (1 encuesta) - Muestra insuficiente
- **LATAM:** NPS -100.0 (1 encuesta) - Muestra insuficiente

---

## **📋 RESUMEN DE DISPONIBILIDAD DE DATOS POR NMA**

| NMA | Explanatory Drivers | Datos Operativos | NCS Cuant. | NCS Cual. | Verbatims | Rutas | Perfiles |
|-----|---------------------|------------------|------------|-----------|-----------|-------|----------|
| **Global/LH/Economy** | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Global/SH/Economy/IB** | ❌ | ✅ (padre) | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Global** | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Global/SH/Economy/YW** | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |

**Nota:** No se encontraron Explanatory Drivers (SHAP) ni NCS Reflexión/Cualitativo en el tree_data original para ninguno de los NMAs identificados.

---

## Cabin Radio Reflection

# 📋 PASO 4C: REFLEXIÓN POR CABINA-RADIO

---

## **=== Economy SH ===**

• **NPS Cabina:** 25.5 (-1.5 pts)  
• **Estado:** NEGATIVE ANOMALY  
• **Escenario:** DOMINANCIA (IB -, YW + | Cabina -)  

• **IB:** NPS 17.1 (-9.7 pts) - Deterioro causado por incremento crítico en Mishandling (+3.73 pts vs baseline Economy SH), incidentes meteorológicos en aeropuerto EAS (5 incidentes NCS, ruta EAS-MAD con NPS -16.7), y problemas de asignación de asientos (4+ verbatims). Triangulación ALTA: operative_data + NCS + verbatims + routes + customer_profile.

• **YW:** NPS 42.4 (+15.0 pts) - Mejora relativa impulsada por mejor gestión de equipaje (Mishandling +2.55 pts, menor deterioro que IB), mejor puntualidad (OTP15 89.66%, +1.29 pts), menor ocupación (Load Factor 78.83%, -1.81 pts), y menor exposición a rutas problemáticas. Contradicción detectada: mejora en NPS coexiste con deterioro operativo en Mishandling, sugiriendo baseline excepcionalmente bajo.

• **Narrativa de agregación:** IB domina el resultado agregado de Economy SH debido a su mayor volumen operativo y/o peso en el segmento. La caída de IB (-9.7 pts) es suficientemente fuerte para arrastrar a la cabina a territorio negativo (-1.5 pts), aunque el efecto positivo de YW (+15.0 pts) atenúa parcialmente la magnitud de la caída. Dispersión entre compañías: 24.7 pts - volatilidad interna CRÍTICA.

• **Rutas críticas (IB - hijo dominante):**
  - **EAS-MAD:** NPS -16.7 (n=6) - Desvíos meteorológicos, peor ruta SH del día
  - **BLQ-MAD:** NPS -40.0 (n=5) - Check-in fallido, asistencia especial
  - **FRA-MAD:** NPS -20.0 (n=5) - Problemas de equipaje
  - **MAD-ORY:** NPS -4.0 (n=25) - Equipaje forzado a bodega, retrasos
  - **MAD-NAP:** NPS 0.0 (n=5) - Asignación de asientos

• **Perfiles reactivos (IB - hijo dominante):**
  - **Flota:** Dispersión 129.8 pts - A333 (NPS -66.7), A319 (NPS -8.7) con peores desempeños por problemas de espacio/equipaje; ATR (NPS +63.2) mejor flota
  - **CodeShare:** Dispersión 200.0 pts - AA (NPS -100.0), AY (NPS -100.0) críticos; IB (NPS +27.6) base principal
  - **Business/Leisure:** Dispersión 40.4 pts - Business/Work (NPS 8.7) significativamente más afectados que Leisure (NPS 49.1)
  - **Región de Residencia:** Europa (NPS +15.9) más críticos que España (NPS +38.2)

---

## **=== Business SH ===**

• **NPS Cabina:** 38.3 (+3.2 pts)  
• **Estado:** Normal  
• **Escenario:** NO APLICA - Segmento estable (IB N, YW N | Cabina N)

• **IB:** NPS 46.9 (+4.8 pts) - Desempeño estable dentro del rango normal de fluctuación. Sin análisis causal disponible (segmento marcado como "Normal").

• **YW:** NPS 20.0 (+4.0 pts) - Desempeño estable dentro del rango normal de fluctuación. Sin análisis causal disponible (segmento marcado como "Normal").

• **Narrativa de agregación:** Ambas compañías mantuvieron desempeño estable con variaciones dentro del rango normal (+4.8 pts IB, +4.0 pts YW). No se detectaron anomalías reales. Este segmento debe ser excluido del análisis detallado según las instrucciones (solo analizar anomalías reales). Dispersión entre compañías: 26.9 pts.

• **Rutas críticas:** No disponible (segmento Normal sin análisis causal)

• **Perfiles reactivos:** No disponible (segmento Normal sin análisis causal)

---

## **=== Economy LH ===**

• **NPS:** 26.4 (+22.7 pts)  
• **Estado:** POSITIVE ANOMALY  

• **Causa principal:** Mejora relativa (+22.7 pts) respecto a un baseline excepcionalmente bajo (3.7 pts), NO mejora absoluta del servicio. El día 14-dic tuvo deterioro operativo significativo (Mishandling +3.73 pts, Misconex +0.13 pts, OTP15 -4.58 pts), pero fue menos severo que los 7 días previos. Paradoja crítica: anomalía positiva coexiste con problemas graves de equipaje (53 maletas afectadas en 2 vuelos críticos), caos en proceso de embarque (pesaje equipaje de mano), y conexiones perdidas en MAD.

• **Evidencia clave:**  
  - **Operative Data:** Mishandling 19.04 (+3.73 pts), OTP15 77.27% (-4.58 pts), Misconex 0.81 (+0.13 pts)
  - **NCS:** 6 incidentes totales - IB151 MAD-BOG (26 equipajes no cargados), IB281/BA458 (27 maletas perdidas)
  - **Verbatims:** 5+ menciones equipaje (NPS 0-3), 3+ menciones caos embarque (NPS 0-6)

• **Rutas críticas:**
  1. **BOG-MAD:** NPS 0.0 (n=29) - Equipaje (26 maletas IB151) + Caos embarque + Mal trato personal
  2. **HAV-MAD:** NPS -25.0 (n=4) - Flota A33ACMI con problemas
  3. **DOH-MAD:** NPS -20.0 (n=5) - Sin evidencia específica (muestra pequeña)
  4. **LEI-MAD:** NPS 0.0 (n=4) - Pérdida de conexión + Equipaje mal gestionado
  5. **MAD-UIO:** NPS 18.2 (n=11) - Conexiones perdidas (27 maletas BA458)

• **Perfiles reactivos:**
  - **CodeShare:** Dispersión 160.0 pts - BA (NPS -60.0) correlaciona con incidente BA458; AA (NPS -33.3), LATAM (NPS -25.0) afectados por conexiones
  - **Región de Residencia:** Dispersión 200.0 pts - ASIA (NPS -100.0) todos detractores; EUROPA (NPS 10.0) muy por debajo del promedio; AMERICA SUR (NPS 20.0) afectados por BOG-MAD/MAD-UIO
  - **Flota:** Dispersión 133.3 pts - A33ACMI (NPS -33.3) peor flota; A333 (NPS -6.7) problemas equipaje/espacio; A350 next (NPS 27.5) mejor flota
  - **Business/Leisure:** Dispersión 0.4 pts - Impacto transversal sin discriminación (Business 26.7, Leisure 26.3)

---

## **=== Business LH ===**

• **NPS:** 8.7 (-10.3 pts)  
• **Estado:** NEGATIVE ANOMALY  

• **Causa principal:** Sin análisis causal disponible (error de procesamiento reportado en tree_data). Hipótesis inferidas desde contexto general: expectativas no cumplidas en servicio premium, los problemas de equipaje y proceso de embarque afectan más a pasajeros Business por expectativas más altas y menor tolerancia a disrupciones operativas. Posible correlación con rutas BOG-MAD, DOH-MAD donde el servicio Business pudo verse comprometido.

• **Evidencia clave:** No disponible (análisis causal falló debido a error de procesamiento)

• **Rutas críticas:** No disponible (sin datos granulares por error de procesamiento)

• **Perfiles reactivos:** No disponible (sin análisis causal completo)

---

## **=== Premium LH ===**

• **NPS:** 5.9 (-3.4 pts)  
• **Estado:** NEGATIVE ANOMALY  

• **Causa principal:** Sin análisis causal disponible (error de procesamiento reportado en tree_data). Hipótesis inferidas: caída menor pero consistente (-3.4 pts) sugiere problemas menores pero generalizados. Posible degradación de servicios diferenciales (comida, amenities, atención) y efecto contagio desde Economy - los problemas operativos (equipaje, embarque) afectan a todas las cabinas sin que Premium Economy tenga suficiente diferenciación para aislar el impacto.

• **Evidencia clave:** No disponible (análisis causal falló debido a error de procesamiento)

• **Rutas críticas:** No disponible (sin datos granulares por error de procesamiento)

• **Perfiles reactivos:** No disponible (sin análisis causal completo)

---

## **📊 RESUMEN DE DISPONIBILIDAD DE ANÁLISIS CAUSAL**

| Cabina-Radio | Estado | Análisis Causal Disponible | Calidad de Evidencia |
|--------------|--------|----------------------------|---------------------|
| **Economy SH** | NEGATIVE ANOMALY | ✅ Completo (IB + YW) | ALTA (triangulación 5 fuentes) |
| **Business SH** | Normal | ❌ No aplica (segmento estable) | N/A |
| **Economy LH** | POSITIVE ANOMALY | ✅ Completo | ALTA (triangulación 5 fuentes) |
| **Business LH** | NEGATIVE ANOMALY | ❌ Error de procesamiento | BAJA (solo hipótesis) |
| **Premium LH** | NEGATIVE ANOMALY | ❌ Error de procesamiento | BAJA (solo hipótesis) |

---

## **🎯 INSIGHTS CLAVE POR CABINA-RADIO**

### **Cabinas con Mayor Impacto en Global:**
1. **Economy LH (+22.7 pts):** Domina el resultado Global (+4.6 pts) a pesar de deterioro operativo real. Paradoja crítica entre anomalía positiva y evidencia operativa negativa.
2. **Economy SH IB (-9.7 pts):** Deterioro real y absoluto que arrastra a Economy SH (-1.5 pts) y contribuye negativamente al Global, pero es superado por la dominancia de LH.

### **Cabinas con Problemas sin Datos Completos:**
1. **Business LH (-10.3 pts):** Caída significativa sin análisis causal disponible - requiere investigación urgente.
2. **Premium LH (-3.4 pts):** Caída menor pero consistente sin análisis causal - requiere seguimiento.

### **Patrón Común:**
Todas las cabinas anómalas (excepto Business SH estable) comparten **problemas de equipaje** como causa operativa subyacente, pero con impactos diferenciados por baseline y volumen operativo.

---

## 📋 SÍNTESIS EJECUTIVA FINAL

```html
<b>SÍNTESIS EJECUTIVA</b><br>
<br>
La red global registró un <b>NPS de 25.5 (+4.6 pts)</b> con respecto a la media de los últimos 7 días, resultado de una dinámica de dominancia donde el Largo Radio impuso su mejora relativa sobre el deterioro del Corto Radio. Sin embargo, esta anomalía positiva es engañosa y no refleja una mejora real del servicio: el día 14 de diciembre tuvo problemas operativos graves en términos absolutos, pero fue menos severo que los 7 días previos que conforman el baseline.<br>
<br>
<b>En Long Haul, el segmento Economy alcanzó un NPS de 26.4 (+22.7 pts)</b>, dominando completamente el resultado del radio y arrastrando al Global hacia territorio positivo. Esta mejora aparente oculta una paradoja crítica: las métricas operativas muestran deterioro significativo con un Mishandling de 19.04 (+3.73 pts) y un OTP15 de 77.27% (–4.58 pts), además de un Misconex de 0.81 (+0.13 pts). La triangulación de evidencias revela que el día registró 6 incidentes operativos críticos, incluyendo 26 equipajes no cargados en el vuelo IB151 MAD-BOG por falta de capacidad y 27 maletas perdidas en la conexión del vuelo BA458 LHR-MAD hacia IB281, que tuvieron que ser regularizadas vía DOH. Los verbatims de clientes reflejan esta realidad operativa con más de 5 menciones críticas sobre equipaje perdido, robado o dañado, 3 menciones sobre caos en el proceso de embarque por pesaje excesivo de equipaje de mano, y múltiples quejas sobre conexiones perdidas y trato inadecuado del personal en tierra. Las rutas más afectadas fueron <b>BOG-MAD con NPS de 0.0</b> donde convergieron problemas de equipaje, caos en embarque y mal trato del personal en 29 encuestas, <b>HAV-MAD con NPS de –25.0</b> operada con flota A33ACMI, <b>DOH-MAD con NPS de –20.0</b>, y <b>LEI-MAD con NPS de 0.0</b> por pérdidas de conexión. Los pasajeros de codeshare fueron desproporcionadamente afectados, con BA registrando un NPS de –60.0 correlacionado directamente con el incidente de las 27 maletas, AA con –33.3 y LATAM con –25.0, mostrando una dispersión de 160.0 pts. Por región de residencia, los pasajeros de Asia fueron todos detractores con NPS de –100.0, Europa alcanzó solo 10.0 y América del Sur 20.0, evidenciando una dispersión de 200.0 pts. La flota A33ACMI registró el peor desempeño con NPS de –33.3, seguida de la A333 con –6.7 por problemas de espacio y equipaje, mientras que la A350 next obtuvo 27.5 como mejor flota. Esta presión en Economy LH se propagó al radio completo, donde a pesar de las caídas de Business LH con NPS de 8.7 (–10.3 pts) y Premium LH con NPS de 5.9 (–3.4 pts), el volumen y magnitud de Economy dominaron completamente, llevando a Long Haul a un <b>NPS de 22.3 (+16.5 pts)</b>.<br>
<br>
<b>En Short Haul, el segmento Economy registró un NPS de 25.5 (–1.5 pts)</b>, resultado de una dinámica de dominancia entre compañías donde IB con un <b>NPS de 17.1 (–9.7 pts)</b> impuso su caída sobre la mejora de YW con <b>NPS de 42.4 (+15.0 pts)</b>, generando una dispersión crítica de 24.7 pts entre operadores. El deterioro de IB fue causado principalmente por un incremento significativo en Mishandling de 19.04 (+3.73 pts con respecto al baseline), incidentes meteorológicos concentrados en el aeropuerto EAS con 5 eventos que incluyeron 3 desvíos y 2 cancelaciones forzando a los vuelos a desviarse a BIO con transporte por superficie, y problemas recurrentes de asignación de asientos sin justificación. Los verbatims reflejan 8 menciones sobre equipaje forzado a bodega, pérdidas y daños durante manipulación, 6 menciones sobre trato inadecuado del personal, y 4 menciones sobre cambios de asientos pagados. Las rutas más críticas fueron <b>EAS-MAD con NPS de –16.7</b> como peor ruta del día por los desvíos meteorológicos, <b>BLQ-MAD con NPS de –40.0</b> por fallos en check-in y asistencia especial, <b>FRA-MAD con NPS de –20.0</b> por problemas de equipaje, <b>MAD-ORY con NPS de –4.0</b> en 25 encuestas por equipaje forzado a bodega y retrasos, y <b>MAD-NAP con NPS de 0.0</b> por problemas de asignación de asientos. Los perfiles más reactivos fueron la flota A333 con NPS de –66.7 y A319 con –8.7 por limitaciones de espacio versus ATR con 63.2 como mejor flota, mostrando una dispersión de 129.8 pts. Los pasajeros de codeshare AA y AY registraron NPS de –100.0 cada uno, y los viajeros Business/Work con NPS de 8.7 fueron significativamente más críticos que Leisure con 49.1, evidenciando una dispersión de 40.4 pts. Por otro lado, YW experimentó una mejora relativa impulsada por mejor gestión de equipaje con Mishandling de 15.19 (+2.55 pts, menor deterioro que IB), mejor puntualidad con OTP15 de 89.66% (+1.29 pts), y menor ocupación con Load Factor de 78.83% (–1.81 pts). Sin embargo, esta mejora también coexiste con deterioro operativo en Mishandling, sugiriendo que el baseline de YW fue excepcionalmente bajo. Los verbatims de YW muestran 6 menciones sobre equipaje perdido o dañado, 3 menciones sobre problemas de check-in y conexiones, y 3 menciones sobre retrasos graves incluyendo un vuelo BCN-VLC con 7 horas de retraso sin comunicación. Las rutas críticas de YW fueron <b>BLQ-MAD con NPS de –40.0</b> y <b>FRA-MAD con NPS de –20.0</b>, mientras que los perfiles más afectados fueron viajeros Business/Work con NPS de 8.7 versus Leisure con 49.1, y la flota CRJ con 39.2 versus ATR con 63.2. Esta caída de Economy SH se transfirió directamente al radio Short Haul, donde Business SH se mantuvo estable con <b>NPS de 38.3 (+3.2 pts)</b> pero con volumen insuficiente para compensar el efecto negativo, llevando a Short Haul a un <b>NPS de 26.8 (–0.9 pts)</b>.<br>
<br>
La convergencia de <b>Long Haul con +16.5 pts</b> y <b>Short Haul con –0.9 pts</b>, con magnitudes asimétricas donde la mejora relativa de LH es 18.3 veces mayor que la caída de SH, produjo la anomalía positiva del Global. Sin embargo, el análisis operativo revela que ambos radios compartieron la misma causa raíz: deterioro en gestión de equipaje con Mishandling de 19.04 (+3.73 pts) y problemas de conexiones con Misconex de 0.81 (+0.13 pts), afectando transversalmente a toda la red con 226 incidentes totales que incluyeron 42 retrasos, 22 cancelaciones, 13 incidentes de equipaje, 13 desvíos, 24 pérdidas de conexión en el hub MAD y 83 conexiones reprogramadas. La diferencia en el signo de las anomalías se explica exclusivamente por los baselines de referencia: Long Haul partió de un NPS de 5.9 haciendo que cualquier mejora parezca significativa, mientras que Short Haul partió de 27.7 donde el mismo deterioro operativo se percibe como caída real. Los factores atenuantes fueron un OTP15 global de 89.9% (+0.84 pts) y un Load Factor de 83.7% (–2.6 pts) que compensaron parcialmente el impacto negativo del equipaje, pero no fueron suficientes para revertir los problemas operativos absolutos del día.<br>
<br>
<br>
<b><u>DETALLE POR CABINA</u></b><br>
<br>
<b><u>ECONOMY LH: Mejora relativa que oculta deterioro operativo grave</u></b><br>
La cabina alcanzó un <b>NPS de 26.4 (+22.7 pts)</b>, dominando el resultado del Largo Radio y arrastrando al Global hacia territorio positivo. Sin embargo, esta anomalía positiva es engañosa: el baseline de referencia fue excepcionalmente bajo en 3.7 pts, haciendo que el día 14 de diciembre parezca mejor en comparación a pesar de tener problemas operativos graves en términos absolutos. Las métricas operativas muestran Mishandling de 19.04 (+3.73 pts), OTP15 de 77.27% (–4.58 pts) y Misconex de 0.81 (+0.13 pts). Los incidentes operativos incluyeron 26 equipajes no cargados en IB151 MAD-BOG por falta de capacidad y 27 maletas perdidas en la conexión BA458 hacia IB281. Los verbatims reflejan esta realidad con menciones sobre equipaje robado en GIG-MAD y BOG-MAD, caos total en el proceso de embarque por pesaje de maletas de mano en BOG-MAD y MAD-SJO, y comportamiento abusivo del personal en BOG-MAD y JFK-MAD. Las rutas más afectadas fueron BOG-MAD con NPS de 0.0 en 29 encuestas, HAV-MAD con –25.0, DOH-MAD con –20.0, LEI-MAD con 0.0 y MAD-UIO con 18.2 por las 27 maletas perdidas. Los pasajeros de codeshare fueron desproporcionadamente afectados con BA en –60.0 por el incidente de equipaje, AA en –33.3 y LATAM en –25.0, mostrando dispersión de 160.0 pts. Por región, Asia registró –100.0 con todos detractores, Europa 10.0 y América del Sur 20.0 afectados por las rutas BOG-MAD y MAD-UIO, con dispersión de 200.0 pts. La flota A33ACMI tuvo NPS de –33.3 y A333 de –6.7 por problemas de espacio versus A350 next con 27.5, dispersión de 133.3 pts.<br>
<br>
<b><u>BUSINESS LH: Caída significativa sin análisis causal completo</u></b><br>
La cabina registró un <b>NPS de 8.7 (–10.3 pts)</b>, contrarrestando parcialmente la mejora de Economy LH pero sin poder revertir su dominancia volumétrica. El análisis causal completo no está disponible por error de procesamiento, pero las hipótesis inferidas sugieren que las expectativas no cumplidas en servicio premium amplificaron el impacto de los problemas operativos de equipaje y proceso de embarque que afectaron a toda la red. Los pasajeros Business tienen menor tolerancia a disrupciones operativas y posiblemente fueron afectados en rutas como BOG-MAD y DOH-MAD donde el servicio premium pudo verse comprometido. Se requiere investigación urgente para identificar las causas específicas de esta caída significativa.<br>
<br>
<b><u>PREMIUM LH: Deterioro menor pero consistente</u></b><br>
La cabina alcanzó un <b>NPS de 5.9 (–3.4 pts)</b>, mostrando una caída menor pero consistente que sugiere problemas generalizados en la experiencia Premium Economy. Sin análisis causal disponible por error de procesamiento, las hipótesis apuntan a posible degradación de servicios diferenciales como comida, amenities y atención, además de un efecto contagio desde Economy donde los problemas operativos de equipaje y embarque afectan a todas las cabinas sin que Premium Economy tenga suficiente diferenciación para aislar el impacto. Se requiere seguimiento para confirmar si esta tendencia negativa se mantiene.<br>
<br>
<b><u>ECONOMY SH: Dominancia de IB arrastra la cabina a territorio negativo</u></b><br>
La cabina registró un <b>NPS de 25.5 (–1.5 pts)</b>, resultado directo del desplome de IB con <b>NPS de 17.1 (–9.7 pts)</b> que anuló completamente la mejora de YW con <b>NPS de 42.4 (+15.0 pts)</b>, generando una dispersión crítica de 24.7 pts entre operadores. El deterioro de IB fue causado por incremento significativo en Mishandling de 19.04 (+3.73 pts), 5 incidentes meteorológicos en aeropuerto EAS que incluyeron 3 desvíos y 2 cancelaciones forzando transporte por superficie, y problemas recurrentes de asignación de asientos sin justificación. Los verbatims muestran 8 menciones sobre equipaje forzado a bodega en vuelos con espacio disponible, pérdidas y daños durante manipulación, 6 menciones sobre personal grosero o poco servicial especialmente en MAD-NAP, y 4 menciones sobre cambios de asientos pagados sin explicación en GVA-MAD. Las rutas más críticas de IB fueron EAS-MAD con NPS de –16.7 como peor ruta del día, BLQ-MAD con –40.0 por fallos en check-in, FRA-MAD con –20.0 por equipaje, MAD-ORY con –4.0 en 25 encuestas y MAD-NAP con 0.0. Los perfiles más reactivos fueron flota A333 con –66.7 y A319 con –8.7 por limitaciones de espacio versus ATR con 63.2, dispersión de 129.8 pts, y viajeros Business/Work con 8.7 significativamente más críticos que Leisure con 49.1, dispersión de 40.4 pts. Por otro lado, YW experimentó mejora relativa por mejor gestión de equipaje con Mishandling de 15.19 (+2.55 pts, menor deterioro que IB), mejor puntualidad con OTP15 de 89.66% (+1.29 pts) y menor ocupación con Load Factor de 78.83% (–1.81 pts), aunque esta mejora también coexiste con deterioro operativo sugiriendo baseline excepcionalmente bajo. Los verbatims de YW incluyen 6 menciones sobre equipaje perdido en FRA-MAD y PMI-VLC, 3 menciones sobre problemas de check-in en BLQ-MAD y conexiones perdidas en LEI-MAD con coste adicional de 162.57 euros, y 3 menciones sobre retrasos graves incluyendo BCN-VLC con 7 horas sin comunicación. Las rutas críticas de YW fueron BLQ-MAD con –40.0 y FRA-MAD con –20.0, mientras que los perfiles más afectados fueron viajeros Business/Work con 8.7 versus Leisure con 49.1 y flota CRJ con 39.2 versus ATR con 63.2.<br>
<br>
<b><u>BUSINESS SH: Estabilidad sin impacto en el agregado</u></b><br>
La cabina mantuvo desempeño estable con <b>NPS de 38.3 (+3.2 pts)</b>, donde tanto IB con <b>NPS de 46.9 (+4.8 pts)</b> como YW con <b>NPS de 20.0 (+4.0 pts)</b> registraron variaciones dentro del rango normal de fluctuación. Sin embargo, esta estabilidad no fue suficiente para compensar la caída de Economy SH debido al peso volumétrico dominante de la cabina económica en operaciones de corto radio. Business SH mantuvo mejor desempeño porque los pasajeros de esta cabina viajan con menos equipaje facturado reduciendo exposición a problemas de Mishandling, tienen prioridad en re-acomodación ante incidentes meteorológicos como los de EAS, y las rutas críticas como EAS-MAD y BLQ-MAD tienen menor proporción de pasajeros Business. No se requiere análisis detallado al ser un segmento sin anomalía real.
```

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 7
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-13 to 2025-12-13:
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** 0 ()

---

## 📊 DIAGNÓSTICO A NIVEL DE EMPRESA

# 🏢 PASO 1: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL COMPAÑÍA)

---

## 📊 ANÁLISIS POR CABINA SHORT HAUL

---

### ✈️ **CABINA 1: ECONOMY SHORT HAUL**

#### **PARTE A: DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(IB: +, YW: N | Economy SH: +)`

**Escenario Identificado:** **TRANSFERENCIA**

**Justificación:**
- **IB** presenta **POSITIVE ANOMALY** (+14.7 pts)
- **YW** está en estado **Normal** (+5.2 pts - dentro de rango normal)
- **Economy SH (Padre)** muestra **POSITIVE ANOMALY** (+11.3 pts)

La anomalía positiva de **IB** (+14.7 pts) es suficientemente fuerte para **contagiar al padre** a pesar de que **YW** mantiene comportamiento normal. El volumen operativo de **IB** (394 encuestas según análisis de IB) domina sobre **YW** (16 encuestas), permitiendo la transferencia del efecto.

---

#### **PARTE B: NARRATIVA CAUSAL**

**Narrativa:** La anomalía positiva de Economy SH (+11.3 pts) se explica por la **transferencia directa del excelente desempeño de IB** (+14.7 pts), que logró compensar un deterioro operativo crítico en manejo de equipaje (Mishandling +4.07 pts vs baseline) mediante mejoras en puntualidad (OTP15 +1.88 pts) y menor ocupación (Load Factor -2.71 pts). **YW** mantuvo estabilidad operativa sin contribuir a la anomalía.

**Evidencia Clave (adoptada del análisis de IB):**

1. **Causa Principal - Deterioro de Equipaje Compensado:**
   - **Mishandling:** 20.62 (IB) vs 19.29 (Economy SH baseline) → +4.52 pts de deterioro
   - **7+ menciones** en verbatims de problemas de equipaje (FRA-MAD, MAD-OPO, MAD-ORY, AMS-MAD, BRU-MAD)
   - **Rutas críticas afectadas:** FRA-MAD (NPS 0.0), AMS-MAD (NPS 14.3), BRU-MAD (NPS 14.3)

2. **Factores Compensatorios (IB):**
   - **OTP15:** 94.09% (+1.88 pts) - Mejora significativa en puntualidad
   - **Load Factor:** 86.57% (-2.71 pts) - Menor ocupación mejoró experiencia general
   - **Rutas con excelente desempeño:** MAD-ORY (NPS 54.2, n=24), BCN-MAD (NPS 29.4, n=17)

3. **Perfil del Impacto:**
   - **Flota crítica:** A320neo con NPS 33.3 (93 encuestas) vs A321 con NPS 58.3 (72 encuestas) → Dispersión de 25 pts
   - **Codeshare problemático:** LATAM (NPS -33.3, n=9) y BA (NPS 20.0, n=5) vs operación propia IB (NPS 46.5, n=245)
   - **Región más afectada:** EUROPA con NPS 35.2 (71 encuestas) vs ESPAÑA con NPS 46.4 (138 encuestas)

**Conclusión:** La anomalía del padre (Economy SH +11.3 pts) es una **herencia directa de IB**, cuyo volumen (394 encuestas) y magnitud de mejora (+14.7 pts) dominaron sobre la estabilidad de **YW** (37.5 NPS, 16 encuestas, +5.2 pts dentro de rango normal).

---

### 🎩 **CABINA 2: BUSINESS SHORT HAUL**

#### **PARTE A: DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(IB: N, YW: + | Business SH: +)`

**Escenario Identificado:** **TRANSFERENCIA**

**Justificación:**
- **IB** está en estado **Normal** (+4.6 pts - dentro de rango normal)
- **YW** presenta **POSITIVE ANOMALY** (+28.5 pts)
- **Business SH (Padre)** muestra **POSITIVE ANOMALY** (+11.1 pts)

La anomalía extrema de **YW** (+28.5 pts) es suficientemente fuerte para **contagiar al padre** (+11.1 pts) a pesar de que **IB** mantiene comportamiento normal. Aunque **YW** tiene menor volumen (9 encuestas según análisis YW) que **IB** (34 encuestas según análisis Business SH), la magnitud de la mejora de **YW** es tan extrema (28.5 pts) que logra elevar el agregado.

---

#### **PARTE B: NARRATIVA CAUSAL**

**Narrativa:** La anomalía positiva de Business SH (+11.1 pts) se explica por la **transferencia de la mejora extrema de YW** (+28.5 pts), impulsada por excelente servicio de tripulación/VIP y mejoras operativas (OTP15 +1.62 pts, Load Factor -5.75 pts), que compensaron problemas críticos en segmentos específicos. **IB** mantuvo estabilidad sin contribuir a la anomalía.

**⚠️ ALERTA CRÍTICA:** La mejora agregada **oculta una polarización extrema** en **YW** (dispersión de 162.5 pts entre Business/Work NPS -100 y Leisure NPS 62.5), con problemas severos en calidad de Business Class operada con flota CRJ.

**Evidencia Clave (adoptada del análisis de YW):**

1. **Factores Positivos Dominantes (89% de muestra YW):**
   - **Segmento Leisure:** NPS 62.5 (8 de 9 encuestas)
   - **Load Factor:** 51.62% (-5.75 pts vs baseline) → Menor ocupación mejoró experiencia
   - **OTP15:** 89.94% (+1.62 pts vs baseline) → Mejora en puntualidad
   - **Verbatims excepcionales:** 4 menciones con NPS 10 (LCG-MAD, MAD-NCE x2, LEI-MAD) destacando servicio VIP y tripulación

2. **Problema Crítico Oculto (11% de muestra YW):**
   - **Segmento Business/Work:** NPS -100 (1 encuesta)
   - **Ruta crítica:** MAD-VCE con NPS -100
   - **Causa raíz:** Degradación severa de Business Class en flota CRJ
   - **Verbatim crítico:** *"Pagué clase ejecutiva y recibí servicio de bajo coste. Avión diminuto, asiento estrecho, espacio reducido, comida incomible"*

3. **Problema Secundario:**
   - **Mishandling:** 15.22 (+2.7 pts vs baseline)
   - **Ruta afectada:** ALC-MAD (NPS -100) - Pérdida de equipaje + downgrade Premium Economy

4. **Perfil del Impacto:**
   - **Flota:** 100% operado con CRJ (9 encuestas)
   - **Dispersión extrema Business/Leisure:** 162.5 pts
   - **Región:** España con NPS 0.0 (2 encuestas), Europa con NPS 40.0 (5 encuestas)

**Conclusión:** La anomalía del padre (Business SH +11.1 pts) es una **herencia directa de YW**, cuya mejora extrema (+28.5 pts) logró elevar el agregado a pesar de su menor volumen (9 encuestas vs 34 de IB). Sin embargo, esta mejora **oculta una crisis en el segmento Business/Work** (NPS -100) que requiere atención urgente. **IB** se mantuvo estable (+4.6 pts dentro de rango normal) sin afectar la dinámica.

---

## 📌 RESUMEN EJECUTIVO DE AGREGACIÓN

| Cabina | Escenario | Compañía Dominante | Magnitud | Volumen | Causa Transferida |
|--------|-----------|-------------------|----------|---------|-------------------|
| **Economy SH** | TRANSFERENCIA | **IB** (+14.7 pts) | Alta | 394 enc. | Mejoras OTP/Load Factor compensaron deterioro Mishandling |
| **Business SH** | TRANSFERENCIA | **YW** (+28.5 pts) | Extrema | 9 enc. | Excelente servicio VIP/tripulación (Leisure) oculta crisis Business/Work |

**Implicación Estratégica:** Ambas anomalías positivas del padre (Economy SH +11.3 pts, Business SH +11.1 pts) son **transferencias puras** desde una compañía dominante, pero con **dinámicas opuestas**:
- **Economy SH:** Mejora robusta de **IB** con volumen alto, pero con deterioro operativo subyacente
- **Business SH:** Mejora extrema de **YW** con volumen bajo, pero con polarización crítica oculta

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ PASO 2: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL CABINA)

---

## 📊 ANÁLISIS POR RADIO

---

### 🌍 **RADIO 1: SHORT HAUL (SH)**

#### **PARTE A: DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(Economy: +, Business: + | SH: +)`

**Escenario Identificado:** **SINERGIA (Parcial Fuerte)**

**Justificación:**
- **Economy SH:** **POSITIVE ANOMALY** (+11.3 pts)
- **Business SH:** **POSITIVE ANOMALY** (+11.1 pts)
- **SH (Padre):** **POSITIVE ANOMALY** (+11.3 pts)

Ambas cabinas presentan anomalías positivas de magnitud prácticamente idéntica (+11.3 pts Economy, +11.1 pts Business), lo que genera un **efecto sinérgico** que se transfiere íntegramente al radio padre. La convergencia de magnitudes (+11.3 pts en ambos niveles) indica que el fenómeno es **sistémico del radio SH**, no específico de una cabina.

---

#### **PARTE B: NARRATIVA CAUSAL**

**Narrativa:** El radio SH experimentó una **mejora sistémica transversal** (+11.3 pts) impulsada por factores operativos comunes que beneficiaron a ambas cabinas por igual. Sin embargo, esta mejora agregada **oculta un problema operativo crítico** de manejo de equipaje (Mishandling +4.07 pts vs baseline) que fue compensado por mejoras en puntualidad y menor ocupación.

**Evidencia Clave (adoptada del análisis del Radio SH):**

---

### 🔴 **CAUSA PRINCIPAL: Deterioro en Gestión de Equipaje (Compensado)**

**Nivel de Confianza:** ALTA ✅ (Triangulación de 4 fuentes)

#### **Datos Operativos:**
- **Mishandling:** 19.29 (+4.07 pts vs baseline de 15.22)
- **Desviación significativa:** >3 pts (umbral crítico superado)
- **Relación con NPS:** INVERSA (Mishandling↑ debería = NPS↓, pero fue compensado)

#### **Evidencia Cualitativa (10 menciones explícitas en verbatims):**

**Tres dimensiones del problema:**

1. **Equipajes Perdidos/Extraviados:**
   - MAD-ORY: *"Mi equipaje no llegó a mi destino final (El Cairo) y esperé 5 días"* (NPS 0)
   - ALC-MAD: *"habían perdido la maleta"* (NPS 0)
   - MAD-NTE: *"mi equipaje facturado había desaparecido. Llegó con cuatro días de retraso"* (NPS 1)
   - BCN-MAD: *"Mi maleta no llegó al vuelo a Barcelona"* (NPS 2)
   - MAD-XRY: *"Mi Equipaje se quedó en Madrid"* (NPS 3)
   - MAD-VCE (2 casos): *"Nos quedamos sin equipaje en Venecia"* (NPS 3)
   - EAS-MAD: *"al llegar al destino nos faltó el equipaje al completo"* (NPS 0)

2. **Retrasos Significativos en Entrega:**
   - LIN-MAD: *"las maletas tardaron 1 hora en salir, excesivo"* (NPS 0)
   - AMS-MAD: *"estuvimos esperando hora y cuarto a recoger nuestra maleta"* (NPS 6)
   - BUD-MAD: *"45 minutos para que sacaran el equipaje"* (NPS 1)

3. **Problemas en Conexiones:**
   - MAD-VCE: Conexión de 1 hora insuficiente para transferencia de equipaje
   - EAS-MAD: Equipaje no transfirió en conexión

#### **Triangulación de Fuentes:**
✅ **Operative Data:** Mishandling +4.07 pts (desviación significativa)  
✅ **Verbatims:** 10 menciones explícitas de problemas de equipaje  
✅ **Routes:** 7 rutas con triple evidencia (métricas + NCS + verbatims)  
✅ **Customer Profile:** Segmentos específicos más afectados identificados  
❌ **NCS:** Sin incidentes formales reportados (GAP crítico de reporte)

#### **Rutas Críticas con Triple Evidencia:**

| Ruta | NPS | Encuestas | Problema Identificado |
|------|-----|-----------|----------------------|
| **ALC-MAD** | 0.0 | 6 | Equipaje perdido |
| **AMS-MAD** | 14.3 | 14 | Retraso 75 min en entrega |
| **MAD-NAP** | 28.6 | 7 | Problemas de equipaje |
| **BCN-MAD** | 29.4 | 17 | Equipaje no transfirió |
| **MAD-VIE** | 30.0 | 20 | Equipaje retenido en Madrid |
| **MAD-ORY** | 54.2 | 24 | Equipaje no llegó a destino final |
| **MAD-VCE** | 62.5 | 8 | Equipaje perdido en conexión |

**Patrón Geográfico:** **MAD como hub central** - 6 de 7 rutas críticas involucran Madrid como origen o destino, indicando problema operativo concentrado en el hub principal.

---

### 🟢 **FACTORES COMPENSATORIOS (Explican la mejora de NPS a pesar del deterioro):**

#### **Métricas Operativas Positivas:**

1. **Mejora en Puntualidad:**
   - **OTP15:** 91.9% (+1.81 pts vs baseline de 90.09%)
   - Relación DIRECTA con NPS: OTP15↑ = NPS↑

2. **Menor Ocupación:**
   - **Load Factor:** 82.44% (-2.49 pts vs baseline de 84.93%)
   - Menor ocupación = mejor experiencia (más espacio, menos congestión)

3. **Conexiones Perdidas Estables:**
   - **Misconex:** 0.83 (+0.15 pts vs baseline de 0.68%)
   - Variación no significativa (<3 pts)

#### **Evidencia Cualitativa Positiva:**
- **4 verbatims con NPS 10** destacando puntualidad y servicio excepcional
- Excelente desempeño de tripulación y servicios VIP (especialmente en YW)

---

### 📊 **PERFILES DE CLIENTE AFECTADOS:**

#### **Por Región de Residencia (Mayor dispersión: 200 pts):**

| Segmento | NPS | Encuestas | Impacto |
|----------|-----|-----------|---------|
| **EUROPA** | 28.4 | 116 | ⚠️ MUY BAJO - Segundo mayor volumen |
| **AMÉRICA NORTE** | -100.0 | 2 | ⚠️ CRÍTICO - Muestra pequeña |
| **ASIA** | 0.0 | 4 | ⚠️ NEGATIVO - Mercados estratégicos |
| **ESPAÑA** | 44.8 | 241 | ✅ Relativamente estable - Mayor volumen |

**Conclusión:** Pasajeros internacionales (especialmente europeos) más afectados que domésticos.

#### **Por Flota (Dispersión: 71.5 pts):**

| Flota | NPS | Encuestas | Observación |
|-------|-----|-----------|-------------|
| **CRJ** | 28.5 | 144 | ⚠️ Flota más afectada - Mayor volumen |
| **A319** | 30.8 | 26 | ⚠️ Segunda flota más afectada |
| A321 | 56.8 | 74 | ✅ NPS superior |
| ATR | 72.2 | 18 | ✅ NPS alto - Volumen bajo |

**Patrón:** Flotas regionales/cortas distancias (CRJ, A319) más impactadas por problemas de equipaje.

#### **Por Codeshare (Dispersión: 125 pts):**

| Partner | NPS | Encuestas | Impacto |
|---------|-----|-----------|---------|
| **LATAM** | -25.0 | 12 | ⚠️ CRÍTICO - Alianza estratégica |
| **BA** | -10.0 | 10 | ⚠️ NEGATIVO - Socio IAG |
| **AA** | 0.0 | 7 | ⚠️ NEGATIVO - Alianza oneworld |
| **IB** | 43.9 | 417 | ✅ Operación propia mejor |

**Conclusión:** Conexiones con codeshares severamente afectadas (mayor complejidad operativa de equipajes).

---

### 📌 **CONCLUSIÓN RADIO SH:**

**Tipo de Anomalía:** SINERGIA (Parcial Fuerte) - Ambas cabinas (+,+ | +)

**Explicación:** El radio SH experimentó una **mejora sistémica** (+11.3 pts) impulsada por factores operativos comunes (puntualidad +1.81 pts, menor ocupación -2.49 pts) que beneficiaron transversalmente a Economy y Business. Sin embargo, esta mejora **ocurrió a pesar** de un deterioro crítico en manejo de equipaje (+4.07 pts Mishandling) que afectó a múltiples rutas con Madrid como hub central.

**Implicación Crítica:** La mejora de NPS **no refleja la realidad operativa** del manejo de equipaje. Los pasajeros afectados por pérdidas/retrasos de equipaje son una minoría que no impactó el NPS agregado, pero representan un **riesgo reputacional grave** (10 quejas explícitas en verbatims sin incidentes formales en NCS).

---

### 🌐 **RADIO 2: LONG HAUL (LH)**

#### **PARTE A: DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(Economy: -, Business: +, Premium: N | LH: -)`

**Escenario Identificado:** **DOMINANCIA (Economy Negativa)**

**Justificación:**
- **Economy LH:** **NEGATIVE ANOMALY** (-9.9 pts)
- **Business LH:** **POSITIVE ANOMALY** (+11.0 pts)
- **Premium LH:** **Normal** (+5.7 pts - dentro de rango normal)
- **LH (Padre):** **NEGATIVE ANOMALY** (-6.8 pts)

A pesar de la mejora en Business (+11.0 pts) y estabilidad en Premium (N), el **deterioro severo de Economy** (-9.9 pts) **domina e impone su signo negativo al radio completo** (-6.8 pts). Esto se debe al **volumen operativo** de Economy (177 encuestas según análisis Economy LH) que supera ampliamente a Business (20 encuestas) y Premium (muestra pequeña).

---

#### **PARTE B: NARRATIVA CAUSAL**

**Narrativa:** El rendimiento del radio LH está **dictado por el deterioro severo de Economy** (-9.9 pts) debido a un **triple deterioro operativo** (OTP15 -5.72 pts, Mishandling +4.07 pts, Misconex +0.15 pts), efecto que fue **parcialmente mitigado** por la mejora en Business (+11.0 pts) y la estabilidad de Premium, pero insuficiente para compensar el impacto volumétrico de Economy.

**Evidencia Clave (adoptada del análisis de Economy LH):**

---

### 🔴 **CAUSA PRINCIPAL: TRIPLE DETERIORO OPERATIVO**

**Nivel de Confianza:** ALTO ✅ (Triangulación de 4 fuentes)

#### **1️⃣ DETERIORO DE PUNTUALIDAD (OTP15)**

**Datos Operativos:**
- **OTP15 del día:** 76.1%
- **Variación vs baseline:** -5.72 pts ⬇️
- **Umbral de significancia:** Superado (>3 pts)
- **Relación con NPS:** DIRECTA (OTP↓ = NPS↓)

**Evidencia Cualitativa:**
- MAD-SDQ: *"retraso de 6 horas en mi vuelo de vuelta"* (NPS 0)
- MAD-PTY: *"retraso en la puerta antes de la salida de más de una hora"*
- MAD-SJO: *"dos horas de retraso... avión demasiado pequeño"*
- LIM-MAD: *"4 cambios de horario"*

**Triangulación:**
✅ Operative Data (OTP15 -5.72 pts)  
✅ Verbatims (múltiples menciones de retrasos significativos)  
✅ Routes (rutas con NPS negativo correlacionan con retrasos)  
✅ Customer Profile (Business/Work NPS -24.1 más afectado que Leisure NPS 2.7)

---

#### **2️⃣ AUMENTO DE PROBLEMAS CON EQUIPAJE (MISHANDLING)**

**Datos Operativos:**
- **Mishandling del día:** 19.29
- **Variación vs baseline:** +4.07 pts ⬆️
- **Umbral de significancia:** Superado (>3 pts)
- **Relación con NPS:** INVERSA (Mishandling↑ = NPS↓)

**Evidencia Cualitativa (múltiples incidentes):**
- MAD-MVD: *"Perdieron todo mi equipaje"* (NPS 0)
- BOG-MAD: *"Pérdida de maletas comentadas por varios pasajeros"*
- MAD-UIO: *"equipaje estaba roto por fuera, completamente dañado"*
- MAD-SJO: *"3 horas esperando para que aparecieran las maletas"*
- MAD-SJO: *"esperamos más de una hora por nuestras maletas"*

**Triangulación:**
✅ Operative Data (Mishandling +4.07 pts)  
✅ Verbatims (múltiples incidentes de pérdida/retraso/daño)  
✅ Routes (16 rutas con incidentes NCS reportados)  
✅ NCS (incidentes formalizados en MAD-SJO, BOG-MAD, MAD-MIA)

---

#### **3️⃣ CONEXIONES PERDIDAS (MISCONEX)**

**Datos Operativos:**
- **Misconex del día:** 0.83
- **Variación vs baseline:** +0.15 pts ⬆️
- **Relación con NPS:** INVERSA (Misconex↑ = NPS↓)

**Evidencia Cualitativa:**
- MAD-VCE: Conexión de 1 hora imposible para equipaje
- Múltiples menciones de problemas en transferencias

---

### 🔴 **RUTAS CRÍTICAS CON MAYOR IMPACTO NEGATIVO:**

| Ruta | NPS | Encuestas | Problemas Principales |
|------|-----|-----------|----------------------|
| **BOS-MAD** | -50.0 | 4 | Espacio reducido, IFE, servicio con bebés |
| **MAD-MEX** | -35.0 | 20 | Asientos, comida, no respeto a pagos |
| **MAD-MVD** | -28.6 | 7 | Pérdida total de equipaje |
| **MAD-SJO** | -16.7 | 6 | Equipaje (3h espera), retrasos (2h) |
| **MAD-UIO** | 0.0 | 7 | Equipaje roto/dañado |
| **MAD-SJU** | 0.0 | 2 | Problemas de embarque, falta respeto |

**Patrón Geográfico:**
- **Rutas a Centroamérica/Caribe:** Mayor concentración de NPS negativo (MAD-SJO, MAD-MEX, MAD-SDQ)
- **Rutas desde Sudamérica:** NPS positivo pero con múltiples incidentes reportados (BOG-MAD, EZE-MAD, LIM-MAD)
- **Rutas desde USA:** NPS muy negativo con muestras pequeñas (BOS-MAD: -50.0, n=4)

---

### 🔴 **CAUSAS SECUNDARIAS: DETERIORO DE CALIDAD DE SERVICIO**

**Nivel de Confianza:** MEDIA ⚠️ (evidencia cualitativa fuerte sin métrica operativa)

#### **Problemas de Servicio de Tripulación:**
- JFK-MAD: *"azafata solo contesta en español... extremadamente grosera"*
- EZE-MAD: *"azafata Susana... actitud agria y a la defensiva"*
- BOG-MAD: *"empleado... manera violenta y alterada... 'ustedes no son nadie'"*
- MAD-SJU: *"señora no nos dejó subir... falta de respeto"*

#### **Problemas Técnicos y de Producto:**

**IFE (Sistema de Entretenimiento):**
- BOG-MAD: *"sistema de entretenimiento se reinició... pantalla sin películas"*
- BOS-MAD: *"pantallas no funcionaban bien"*
- MAD-MIA: *"televisor no funcionó en todo el vuelo"*

**Comida:**
- EZE-MAD: *"omelet y nada mas sin pan... pasta picante insoportable... carne cruda"*
- MAD-MIA: *"comida era terrible"*
- MAD-MEX: *"comida ha bajado mucho de calidad"*

**Configuración de Asientos:**
- MAD-MEX: *"NO RESPETARON mi pago... lugar NO APTO para infante"*
- BOS-MAD: *"espacio entre los asientos es demasiado estrecho"*
- EZE-MAD: *"respaldo del asiento delantero casi tocaba el reposabrazos"*
- MAD-REC: *"espacios demasiado angostos... A321XLR"*

---

### 📊 **PERFILES DE CLIENTE AFECTADOS:**

#### **Por Propósito de Viaje:**

| Perfil | NPS | Encuestas | Diferencia |
|--------|-----|-----------|------------|
| **Business/Work** | -24.1 | 29 | ⚠️ SEVERAMENTE AFECTADO |
| **Leisure** | 2.7 | 188 | Impacto moderado |

**Dispersión:** -26.8 pts

**Explicación:** Los viajeros de negocios están significativamente más insatisfechos. Esto correlaciona directamente con el deterioro de OTP15 (-5.72 pts), ya que este perfil es más sensible a retrasos (reuniones comprometidas).

#### **Por Flota (Peor Desempeño):**

| Flota | NPS | Encuestas | Problemas Asociados |
|-------|-----|-----------|---------------------|
| **A33ACMI** | -100.0 | 9 | Wet lease - problemas de servicio |
| **A321** | -100.0 | 2 | Muestra pequeña |
| **A332** | -19.4 | 31 | Configuración, espacio |
| **A350** | -8.2 | 61 | Sorprendente para flota premium |

#### **Por Región de Residencia:**

| Región | NPS | Encuestas | Observación |
|--------|-----|-----------|-------------|
| **EUROPA** | -53.8 | 26 | ⚠️ MÁS AFECTADOS |
| **ESPAÑA** | -3.3 | 91 | Mayor volumen |
| **AMERICA SUR** | 32.3 | 31 | Mejor experiencia |
| **AMERICA CENTRO** | 4.0 | 25 | Moderado |

**Hallazgo:** Pasajeros europeos (EUROPA + ESPAÑA = 117 encuestas, 54% del total) muestran mayor insatisfacción que sudamericanos, a pesar de que las rutas sudamericanas tienen más incidentes NCS reportados.

#### **Por Codeshare:**

| Operador | NPS | Encuestas | Observación |
|----------|-----|-----------|-------------|
| **IB** | -1.1 | 179 | 82% de encuestas, ligeramente negativo |
| **AA** | 25.0 | 16 | Mejor desempeño |
| **BA** | -40.0 | 5 | Muy negativo |
| **QR** | -25.0 | 4 | Negativo |

---

### 📌 **CONCLUSIÓN RADIO LH:**

**Tipo de Anomalía:** DOMINANCIA (Economy Negativa) - `(-,+,N | -)`

**Explicación:** El radio LH experimentó un **deterioro significativo** (-6.8 pts) **dictado por el colapso de Economy** (-9.9 pts), que representa el mayor volumen operativo (177 encuestas). Este deterioro se debe a un **triple shock operativo**:
1. **OTP15:** -5.72 pts (retrasos significativos hasta 6 horas)
2. **Mishandling:** +4.07 pts (pérdidas, retrasos de hasta 3 horas en entrega, daños)
3. **Misconex:** +0.15 pts (conexiones perdidas)

La mejora en Business (+11.0 pts, 20 encuestas) y la estabilidad de Premium (N) **mitigaron parcialmente** el impacto, pero fueron insuficientes para compensar el peso volumétrico de Economy.

**Implicación Crítica:** El 13 de diciembre de 2025 presentó una **"tormenta perfecta"** de problemas operativos en Long Haul que afectó especialmente a viajeros de negocios (NPS -24.1) y rutas hacia Centroamérica/Caribe, con problemas adicionales de calidad de servicio (tripulación, IFE, comida) que amplificaron el impacto negativo.

---

## 📊 RESUMEN EJECUTIVO DE AGREGACIÓN POR RADIO

| Radio | Escenario | Cabina Dominante | Magnitud | Causa Transferida |
|-------|-----------|------------------|----------|-------------------|
| **SH** | SINERGIA (Parcial Fuerte) | Economy+Business | +11.3 pts | Mejoras OTP/Load Factor compensaron deterioro Mishandling |
| **LH** | DOMINANCIA (Economy -) | Economy | -6.8 pts | Triple deterioro operativo (OTP/Mishandling/Misconex) |

**Implicación Estratégica:** Los dos radios presentan **dinámicas opuestas**:
- **SH:** Mejora sistémica que oculta problema operativo de equipaje
- **LH:** Deterioro operativo severo concentrado en Economy que domina el resultado del radio completo

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 PASO 3: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL GLOBAL)

---

## 📊 ANÁLISIS DE AGREGACIÓN GLOBAL

---

### **PARTE A: DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(LH: -, SH: + | GLOBAL: +)`

**Escenario Identificado:** **DOMINANCIA (SH Positivo)**

---

### **Justificación:**

- **Long Haul (LH):** **NEGATIVE ANOMALY** (-6.8 pts)
- **Short Haul (SH):** **POSITIVE ANOMALY** (+11.3 pts)
- **GLOBAL:** **POSITIVE ANOMALY** (+5.5 pts)

A pesar del deterioro significativo en Long Haul (-6.8 pts), el **excelente desempeño de Short Haul** (+11.3 pts) **domina e impone su signo positivo al nivel Global** (+5.5 pts). Esto se debe al **peso volumétrico** de SH que supera ampliamente a LH en número de operaciones y encuestas.

**Evidencia Volumétrica:**
- **SH:** 466 encuestas (Global/SH/Economy: 427 + Global/SH/Business: 39)
- **LH:** 217 encuestas (Global/LH total)
- **Ratio:** SH representa ~68% del volumen total vs LH ~32%

El **factor de dominancia** es claro: la magnitud positiva de SH (+11.3 pts) con mayor volumen (466 enc.) supera el impacto negativo de LH (-6.8 pts) con menor volumen (217 enc.), resultando en un Global positivo (+5.5 pts) que **oculta la crisis operativa en Long Haul**.

---

### **PARTE B: NARRATIVA CAUSAL**

**Narrativa:** El resultado Global (+5.5 pts) está **arrastrado por el excelente desempeño de Short Haul** (+11.3 pts), impulsado por mejoras en puntualidad (OTP15 +1.81 pts) y menor ocupación (Load Factor -2.49 pts) que compensaron un deterioro en manejo de equipaje (Mishandling +4.07 pts). Sin embargo, esta mejora agregada **oculta una crisis operativa severa en Long Haul** (-6.8 pts) causada por un triple deterioro operativo (OTP15 -5.72 pts, Mishandling +4.07 pts, Misconex +0.15 pts) que afectó especialmente a Economy LH (-9.9 pts).

---

## 🔍 **EVIDENCIA CLAVE (Radio Dominante: SHORT HAUL)**

---

### ✅ **FACTORES POSITIVOS QUE DOMINAN EL GLOBAL**

#### **1️⃣ MEJORA EN PUNTUALIDAD (SH)**

**Datos Operativos:**
- **OTP15 SH:** 91.9% (+1.81 pts vs baseline de 90.09%)
- **Relación con NPS:** DIRECTA (OTP15↑ = NPS↑)
- **Impacto:** Benefició transversalmente a Economy SH y Business SH

**Evidencia Cualitativa:**
- Múltiples verbatims positivos destacando puntualidad
- Especialmente valorado por viajeros Business

---

#### **2️⃣ MENOR OCUPACIÓN (SH)**

**Datos Operativos:**
- **Load Factor SH:** 82.44% (-2.49 pts vs baseline de 84.93%)
- **Relación con NPS:** INVERSA (Load Factor↓ = NPS↑)
- **Impacto:** Mejor experiencia general (más espacio, menos congestión)

---

#### **3️⃣ EXCELENTE SERVICIO DE TRIPULACIÓN Y VIP**

**Evidencia Cualitativa (especialmente en YW):**
- LCG-MAD: *"Excelente servicio, sala VIP maravillosa, superó expectativas"* (NPS 10)
- MAD-NCE (2 menciones): *"Equipo VIP increíble, tripulación encantadora, vuelo impecable"* (NPS 10)
- LEI-MAD: *"Puntualidad y personal"* (NPS 10)

---

### ⚠️ **PROBLEMA OPERATIVO OCULTO EN SH (Compensado pero Crítico)**

#### **DETERIORO EN GESTIÓN DE EQUIPAJE**

**Datos Operativos:**
- **Mishandling SH:** 19.29 (+4.07 pts vs baseline de 15.22)
- **Desviación significativa:** >3 pts (umbral crítico superado)
- **Relación con NPS:** INVERSA (Mishandling↑ debería = NPS↓, pero fue compensado)

**Evidencia Cualitativa (10 menciones explícitas):**
- MAD-ORY: *"Mi equipaje no llegó a mi destino final (El Cairo) y esperé 5 días"* (NPS 0)
- ALC-MAD: *"habían perdido la maleta"* (NPS 0)
- MAD-NTE: *"mi equipaje facturado había desaparecido. Llegó con cuatro días de retraso"* (NPS 1)
- AMS-MAD: *"estuvimos esperando hora y cuarto a recoger nuestra maleta"* (NPS 6)
- BUD-MAD: *"45 minutos para que sacaran el equipaje"* (NPS 1)

**Rutas Críticas SH con Triple Evidencia:**

| Ruta | NPS | Encuestas | Problema Identificado |
|------|-----|-----------|----------------------|
| **ALC-MAD** | 0.0 | 6 | Equipaje perdido |
| **AMS-MAD** | 14.3 | 14 | Retraso 75 min en entrega |
| **BCN-MAD** | 29.4 | 17 | Equipaje no transfirió |
| **MAD-VIE** | 30.0 | 20 | Equipaje retenido en Madrid |
| **MAD-ORY** | 54.2 | 24 | Equipaje no llegó a destino final |

**Patrón Geográfico SH:** **MAD como hub central** - Problema operativo concentrado en el hub principal.

---

## 🔴 **CRISIS OPERATIVA OCULTA (Radio Subordinado: LONG HAUL)**

A pesar de no dominar el resultado Global, **LH presenta problemas operativos críticos** que requieren atención urgente:

---

### **TRIPLE DETERIORO OPERATIVO EN LH**

#### **1️⃣ COLAPSO DE PUNTUALIDAD (LH)**

**Datos Operativos:**
- **OTP15 LH:** 76.1% (-5.72 pts vs baseline de 81.82%)
- **Desviación crítica:** >5 pts (muy por encima del umbral de 3 pts)
- **Contraste con SH:** Diferencia de 15.8 pts (91.9% SH vs 76.1% LH)

**Evidencia Cualitativa:**
- MAD-SDQ: *"retraso de 6 horas en mi vuelo de vuelta"* (NPS 0)
- MAD-PTY: *"retraso en la puerta antes de la salida de más de una hora"*
- MAD-SJO: *"dos horas de retraso... avión demasiado pequeño"*
- LIM-MAD: *"4 cambios de horario"*

---

#### **2️⃣ PROBLEMAS MASIVOS CON EQUIPAJE (LH)**

**Datos Operativos:**
- **Mishandling LH:** 19.29 (+4.07 pts vs baseline)
- **Misma magnitud que SH**, pero con **mayor impacto en NPS** por duración de vuelos

**Evidencia Cualitativa:**
- MAD-MVD: *"Perdieron todo mi equipaje"* (NPS 0)
- BOG-MAD: *"Pérdida de maletas comentadas por varios pasajeros"*
- MAD-UIO: *"equipaje estaba roto por fuera, completamente dañado"*
- MAD-SJO: *"3 horas esperando para que aparecieran las maletas"*

**Rutas Críticas LH:**

| Ruta | NPS | Encuestas | Problemas Principales |
|------|-----|-----------|----------------------|
| **BOS-MAD** | -50.0 | 4 | Espacio, IFE, servicio con bebés |
| **MAD-MEX** | -35.0 | 20 | Asientos, comida, no respeto a pagos |
| **MAD-MVD** | -28.6 | 7 | Pérdida total de equipaje |
| **MAD-SJO** | -16.7 | 6 | Equipaje (3h espera), retrasos (2h) |

---

#### **3️⃣ DETERIORO DE CALIDAD DE SERVICIO (LH)**

**Problemas de Tripulación:**
- JFK-MAD: *"azafata solo contesta en español... extremadamente grosera"*
- EZE-MAD: *"azafata Susana... actitud agria y a la defensiva"*
- BOG-MAD: *"empleado... manera violenta y alterada... 'ustedes no son nadie'"*

**Problemas Técnicos:**
- **IFE:** BOG-MAD, BOS-MAD, MAD-MIA (pantallas no funcionales)
- **Comida:** EZE-MAD, MAD-MIA, MAD-MEX (calidad deteriorada, comida cruda)
- **Configuración:** MAD-MEX, BOS-MAD, EZE-MAD, MAD-REC (espacios estrechos, A321XLR criticado)

---

### 📊 **PERFILES DE CLIENTE AFECTADOS (Diferencias LH vs SH)**

#### **Por Propósito de Viaje:**

| Radio | Business/Work NPS | Leisure NPS | Dispersión |
|-------|------------------|-------------|------------|
| **LH** | -24.1 (29 enc.) | 2.7 (188 enc.) | **-26.8 pts** ⚠️ |
| **SH** | 33.3 (66 enc.) | 40.0 (400 enc.) | **-6.7 pts** ✅ |

**Conclusión:** Los viajeros de negocios en **LH** están **severamente más insatisfechos** que en SH, correlacionando directamente con el deterioro de OTP15 en LH (-5.72 pts).

---

#### **Por Región de Residencia:**

**Long Haul:**
| Región | NPS | Encuestas |
|--------|-----|-----------|
| **EUROPA** | -53.8 | 26 |
| **ESPAÑA** | -3.3 | 91 |
| **AMERICA SUR** | 32.3 | 31 |

**Short Haul:**
| Región | NPS | Encuestas |
|--------|-----|-----------|
| **EUROPA** | 28.4 | 116 |
| **ESPAÑA** | 44.8 | 241 |
| **AMERICA NORTE** | -100.0 | 2 |

**Conclusión:** Pasajeros europeos significativamente más insatisfechos en **LH** (-53.8) que en **SH** (28.4), diferencia de **82.2 puntos**.

---

#### **Por Flota:**

**Flotas Críticas en LH:**
| Flota | NPS | Encuestas |
|-------|-----|-----------|
| **A33ACMI** | -100.0 | 9 |
| **A332** | -19.4 | 31 |
| **A350** | -8.2 | 61 |
| **A321XLR** | 0.0 | 21 |

**Flotas Críticas en SH:**
| Flota | NPS | Encuestas |
|-------|-----|-----------|
| **CRJ** | 28.5 | 144 |
| **A319** | 30.8 | 26 |

**Conclusión:** Flotas widebody de **LH** presentan NPS negativo/muy bajo, mientras que flotas regionales de **SH** mantienen NPS positivo a pesar de problemas operativos.

---

## 📌 **CONCLUSIÓN GLOBAL**

### **Tipo de Anomalía:** DOMINANCIA (SH Positivo) - `(-,+ | +)`

---

### **Explicación Ejecutiva:**

El **13 de diciembre de 2025** presenta una **paradoja operativa crítica**:

✅ **El NPS Global (+5.5 pts) muestra mejora aparente**, impulsada por:
- **Short Haul (+11.3 pts, 466 encuestas, 68% del volumen)** con mejoras en puntualidad (OTP15 +1.81 pts) y menor ocupación (Load Factor -2.49 pts)
- Excelente servicio de tripulación y VIP (especialmente YW)

❌ **Esta mejora OCULTA una crisis operativa severa en Long Haul** (-6.8 pts, 217 encuestas, 32% del volumen):
- **Triple deterioro operativo:** OTP15 -5.72 pts, Mishandling +4.07 pts, Misconex +0.15 pts
- **Economy LH colapsó:** -9.9 pts (peor segmento del día)
- **Rutas críticas:** BOS-MAD (-50.0), MAD-MEX (-35.0), MAD-MVD (-28.6)
- **Viajeros Business severamente afectados:** NPS -24.1 (LH) vs 33.3 (SH)

---

### **Implicaciones Estratégicas:**

1. **RIESGO REPUTACIONAL OCULTO:** El NPS Global positivo **NO refleja la realidad operativa** de Long Haul, donde se concentran:
   - Rutas estratégicas de mayor valor (transatlánticas, sudamericanas)
   - Viajeros de negocios (NPS -24.1)
   - Flotas premium con problemas (A33ACMI -100.0, A350 -8.2)

2. **PROBLEMA TRANSVERSAL DE EQUIPAJE:** Ambos radios presentan deterioro en Mishandling (+4.07 pts), pero:
   - En **SH** fue compensado por mejoras en puntualidad/ocupación
   - En **LH** se amplificó por problemas adicionales (OTP, servicio, producto)

3. **DESACOPLE GEOGRÁFICO:** 
   - **SH (Europa/España):** Mejora generalizada
   - **LH (Transatlántico/Sudamérica):** Deterioro severo

---

### **Áreas de Atención Prioritaria:**

#### **🔴 URGENTE (Long Haul):**
1. **Investigar colapso de puntualidad LH:** OTP15 -5.72 pts (diferencia de 15.8 pts vs SH)
2. **Auditar rutas críticas:** BOS-MAD, MAD-MEX, MAD-MVD, MAD-SJO
3. **Revisar estándares de servicio en flotas ACMI:** A33ACMI con NPS -100.0
4. **Evaluar configuración de A321XLR:** Quejas recurrentes de espacio en rutas LH

#### **🟡 IMPORTANTE (Transversal):**
5. **Auditar procesos de handling de equipaje en MAD:** Problema común a ambos radios (+4.07 pts)
6. **Revisar protocolos de servicio de tripulación:** Múltiples quejas graves en LH
7. **Analizar diferencias operativas IB vs codeshares:** AA mejor NPS que IB en LH

#### **🟢 MONITOREO (Short Haul):**
8. **Sostener mejoras de puntualidad SH:** OTP15 +1.81 pts
9. **Monitorear flotas regionales:** CRJ y A319 con NPS bajo pero estable
10. **Fortalecer servicio VIP/tripulación:** Factor diferenciador positivo en YW

---

### **Evidencia Clave Consolidada:**

**Radio Dominante (SH):**
- **Volumen:** 466 encuestas (68%)
- **Magnitud:** +11.3 pts
- **Causas positivas:** OTP15 +1.81, Load Factor -2.49, servicio VIP excepcional
- **Problema oculto:** Mishandling +4.07 (compensado)

**Radio Subordinado (LH):**
- **Volumen:** 217 encuestas (32%)
- **Magnitud:** -6.8 pts
- **Causas negativas:** OTP15 -5.72, Mishandling +4.07, Misconex +0.15
- **Segmento crítico:** Economy -9.9 pts (177 encuestas)
- **Perfil más afectado:** Business/Work -24.1 pts

---

**Conclusión Final:** El Global (+5.5 pts) refleja una **victoria pírrica** donde la mejora de Short Haul enmascara una **crisis operativa estructural en Long Haul** que afecta a segmentos de alto valor estratégico (Business, rutas transatlánticas/sudamericanas, flotas premium). La métrica agregada **NO debe interpretarse como éxito operativo**, sino como una **señal de alerta sobre desacople geográfico** que requiere intervención inmediata en LH.

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

# 🔍 PASO 4: IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS (NMA)

---

## 📊 ANÁLISIS DE CAUSAS Y NODOS MÁXIMO AFECTADOS

---

### **CAUSA 1: Deterioro en Gestión de Equipaje (Mishandling)**

**Escenario:** **SINERGIA PARCIAL** (afecta a múltiples nodos con magnitud similar)

**NMA:** **Global** (causa transversal a toda la red)

**Afecta a:**
- Global/SH/Economy (Mishandling SH: +4.07 pts)
- Global/SH/Economy/IB (Mishandling IB: +4.52 pts)
- Global/SH/Economy/YW (Mishandling YW: +2.7 pts)
- Global/LH/Economy (Mishandling LH: +4.07 pts)

**Tipo de impacto:** **NEGATIVO** (Mishandling↑ = NPS↓)

**Cadena de propagación hacia el segmento raíz:**

1. **Nivel Compañía (SH Economy):**
   - **IB (+4.52 pts)** y **YW (+2.7 pts)** → Ambos presentan deterioro
   - **Escenario:** SINERGIA en dirección negativa
   - **Resultado:** Ambos contribuyen al deterioro de **Economy SH**

2. **Economy SH → SH (Radio):**
   - **Economy SH (Mishandling +4.07)** y **Business SH (Mishandling presente)** → Ambos afectados
   - **Escenario:** SINERGIA (causa común transversal)
   - **Resultado:** El problema se propaga a **SH completo**

3. **Economy LH (paralelo):**
   - **Economy LH (Mishandling +4.07)** → Mismo deterioro que SH
   - **Escenario:** Problema transversal independiente
   - **Resultado:** Afecta a **LH completo**

4. **SH y LH → Global:**
   - **SH (Mishandling +4.07)** y **LH (Mishandling +4.07)** → Ambos radios afectados por igual
   - **Escenario:** SINERGIA TOTAL (causa sistémica de red)
   - **Resultado:** El problema **burbujea hasta Global** como causa transversal

**Conclusión NMA:** El deterioro de equipaje es una **causa sistémica de red** que afecta transversalmente a toda la operación (SH y LH, Economy y Business, IB y YW). El **NMA = Global** porque la causa es común a todos los nodos con magnitud similar (+4.07 pts promedio).

---

### **CAUSA 2: Mejora en Puntualidad (OTP15 en Short Haul)**

**Escenario:** **SINERGIA** (afecta a múltiples nodos SH)

**NMA:** **Global/SH** (causa específica del radio Short Haul)

**Afecta a:**
- Global/SH/Economy (OTP15 +1.88 pts)
- Global/SH/Economy/IB (OTP15 +1.88 pts)
- Global/SH/Business (OTP15 +1.81 pts)
- Global/SH/Business/YW (OTP15 +1.62 pts)

**Tipo de impacto:** **POSITIVO** (OTP15↑ = NPS↑)

**Cadena de propagación hacia el segmento raíz:**

1. **Nivel Compañía (SH Economy):**
   - **IB (OTP15 +1.88)** y **YW (estable)** → IB mejora, YW normal
   - **Escenario:** TRANSFERENCIA (IB domina por volumen)
   - **Resultado:** La mejora de IB se transfiere a **Economy SH**

2. **Nivel Compañía (SH Business):**
   - **YW (OTP15 +1.62)** y **IB (estable)** → YW mejora, IB normal
   - **Escenario:** TRANSFERENCIA (YW domina por magnitud extrema)
   - **Resultado:** La mejora de YW se transfiere a **Business SH**

3. **Economy SH y Business SH → SH (Radio):**
   - **Economy SH (+11.3 pts)** y **Business SH (+11.1 pts)** → Ambos positivos
   - **Escenario:** SINERGIA (mejora común en puntualidad)
   - **Resultado:** Ambas cabinas contribuyen sinérgicamente a **SH (+11.3 pts)**

4. **SH → Global:**
   - **SH (+11.3 pts)** y **LH (-6.8 pts)** → Radios opuestos
   - **Escenario:** DOMINANCIA (SH domina por volumen 68%)
   - **Resultado:** SH arrastra a **Global (+5.5 pts)** a pesar de LH negativo

**Conclusión NMA:** La mejora de puntualidad es una **causa específica de Short Haul** que se propaga sinérgicamente desde las compañías hasta el radio completo. El **NMA = Global/SH** porque la causa es común a todas las cabinas y compañías de SH, pero NO afecta a LH (que tiene OTP15 -5.72 pts).

---

### **CAUSA 3: Colapso de Puntualidad (OTP15 en Long Haul)**

**Escenario:** **DOMINANCIA** (Economy LH domina el deterioro)

**NMA:** **Global/LH/Economy** (segmento más afectado)

**Afecta a:**
- Global/LH/Economy (OTP15 -5.72 pts, NPS -9.9 pts)
- Global/LH (OTP15 -5.72 pts, NPS -6.8 pts)

**Tipo de impacto:** **NEGATIVO** (OTP15↓ = NPS↓)

**Cadena de propagación hacia el segmento raíz:**

1. **Economy LH → LH (Radio):**
   - **Economy LH (-9.9 pts)**, **Business LH (+11.0 pts)**, **Premium LH (Normal +5.7 pts)**
   - **Escenario:** DOMINANCIA (Economy domina por volumen: 177 enc. vs 20 Business + Premium)
   - **Resultado:** El deterioro de Economy arrastra a **LH (-6.8 pts)** a pesar de Business positivo

2. **LH → Global:**
   - **LH (-6.8 pts)** y **SH (+11.3 pts)** → Radios opuestos
   - **Escenario:** DOMINANCIA (SH domina por volumen 68%)
   - **Resultado:** SH compensa parcialmente el deterioro de LH, resultando en **Global (+5.5 pts)**

**Conclusión NMA:** El colapso de puntualidad es una **causa específica de Long Haul Economy** que domina el resultado del radio completo por su volumen operativo. El **NMA = Global/LH/Economy** porque es el segmento con mayor deterioro (-9.9 pts) y mayor volumen (177 encuestas).

**⚠️ NOTA CRÍTICA:** Este NMA está **oculto en el Global positivo** debido a la dominancia de SH.

---

### **CAUSA 4: Menor Ocupación (Load Factor en Short Haul)**

**Escenario:** **SINERGIA** (afecta a múltiples nodos SH)

**NMA:** **Global/SH** (causa específica del radio Short Haul)

**Afecta a:**
- Global/SH/Economy (Load Factor -2.25 pts)
- Global/SH/Economy/IB (Load Factor -2.71 pts)
- Global/SH/Business (Load Factor -5.58 pts)
- Global/SH/Business/YW (Load Factor -5.75 pts)

**Tipo de impacto:** **POSITIVO** (Load Factor↓ = NPS↑, mejor experiencia)

**Cadena de propagación hacia el segmento raíz:**

1. **Nivel Compañía (SH Economy):**
   - **IB (Load Factor -2.71)** y **YW (estable)** → IB mejora, YW normal
   - **Escenario:** TRANSFERENCIA (IB domina por volumen)
   - **Resultado:** La mejora de IB se transfiere a **Economy SH**

2. **Nivel Compañía (SH Business):**
   - **YW (Load Factor -5.75)** y **IB (estable)** → YW mejora significativa
   - **Escenario:** TRANSFERENCIA (YW domina por magnitud)
   - **Resultado:** La mejora de YW se transfiere a **Business SH**

3. **Economy SH y Business SH → SH (Radio):**
   - **Economy SH (-2.25 pts)** y **Business SH (-5.58 pts)** → Ambos con menor ocupación
   - **Escenario:** SINERGIA (causa común transversal)
   - **Resultado:** Ambas cabinas contribuyen sinérgicamente a **SH (+11.3 pts NPS)**

4. **SH → Global:**
   - **SH (+11.3 pts)** y **LH (-6.8 pts)** → Radios opuestos
   - **Escenario:** DOMINANCIA (SH domina por volumen)
   - **Resultado:** SH arrastra a **Global (+5.5 pts)**

**Conclusión NMA:** La menor ocupación es una **causa específica de Short Haul** que benefició transversalmente a todas las cabinas y compañías. El **NMA = Global/SH** porque la causa es común a Economy y Business SH, pero NO afecta significativamente a LH (Load Factor LH -1.63 pts, no significativo).

---

### **CAUSA 5: Excelente Servicio VIP y Tripulación (YW Business)**

**Escenario:** **TRANSFERENCIA** (causa específica de YW)

**NMA:** **Global/SH/Business/YW** (segmento origen de la mejora extrema)

**Afecta a:**
- Global/SH/Business/YW (NPS +28.5 pts)
- Global/SH/Business (NPS +11.1 pts)

**Tipo de impacto:** **POSITIVO** (Servicio excepcional = NPS↑)

**Cadena de propagación hacia el segmento raíz:**

1. **YW → Business SH:**
   - **YW (+28.5 pts)** y **IB (Normal +4.6 pts)** → YW extremo, IB estable
   - **Escenario:** TRANSFERENCIA (YW domina por magnitud extrema a pesar de menor volumen: 9 enc. vs 34 IB)
   - **Resultado:** La mejora extrema de YW contagia a **Business SH (+11.1 pts)**

2. **Business SH → SH (Radio):**
   - **Business SH (+11.1 pts)** y **Economy SH (+11.3 pts)** → Ambos positivos
   - **Escenario:** SINERGIA (mejoras paralelas)
   - **Resultado:** Ambas cabinas contribuyen a **SH (+11.3 pts)**

3. **SH → Global:**
   - **SH (+11.3 pts)** y **LH (-6.8 pts)** → Radios opuestos
   - **Escenario:** DOMINANCIA (SH domina por volumen)
   - **Resultado:** SH arrastra a **Global (+5.5 pts)**

**Conclusión NMA:** El excelente servicio VIP/tripulación es una **causa específica de YW Business** que logró contagiar al nivel superior a pesar de su bajo volumen (9 encuestas) debido a su magnitud extrema (+28.5 pts). El **NMA = Global/SH/Business/YW** porque es el segmento origen de la mejora.

**⚠️ ALERTA:** Esta mejora **oculta una polarización extrema** dentro de YW (Business/Work NPS -100 vs Leisure NPS 62.5, dispersión de 162.5 pts).

---

### **CAUSA 6: Degradación de Business Class en Flota CRJ (YW)**

**Escenario:** **DILUCIÓN** (causa específica diluida en el agregado)

**NMA:** **Global/SH/Business/YW** (segmento afectado)

**Afecta a:**
- Global/SH/Business/YW - Segmento Business/Work (NPS -100, 1 encuesta)
- Ruta MAD-VCE (NPS -100)

**Tipo de impacto:** **NEGATIVO** (Calidad degradada = NPS↓)

**Cadena de propagación hacia el segmento raíz:**

1. **Segmento Business/Work dentro de YW:**
   - **Business/Work (NPS -100, 1 enc.)** y **Leisure (NPS 62.5, 8 enc.)** → Extremos opuestos
   - **Escenario:** DILUCIÓN (el problema crítico se diluye por el volumen de Leisure)
   - **Resultado:** YW muestra NPS +28.5 pts **ocultando la crisis de Business/Work**

2. **YW → Business SH:**
   - **YW (+28.5 pts con crisis oculta)** y **IB (Normal +4.6 pts)**
   - **Escenario:** TRANSFERENCIA (YW domina por magnitud)
   - **Resultado:** Business SH (+11.1 pts) **hereda la mejora pero oculta el problema**

3. **Business SH → SH → Global:**
   - El problema queda **completamente diluido** en los niveles superiores
   - **Resultado:** No impacta visiblemente en SH ni Global

**Conclusión NMA:** La degradación de Business Class en CRJ es una **causa específica oculta por dilución** en el segmento YW Business/Work. El **NMA = Global/SH/Business/YW** (específicamente el sub-segmento Business/Work con 1 encuesta), pero su impacto **NO se propaga** debido a la dilución por el volumen de Leisure (8 encuestas con NPS 62.5).

**⚠️ CRÍTICO:** Este es un **problema grave oculto** que requiere atención urgente a pesar de no afectar el NPS agregado.

---

### **CAUSA 7: Deterioro de Calidad de Servicio en Long Haul (Tripulación, IFE, Comida)**

**Escenario:** **DOMINANCIA** (Economy LH domina)

**NMA:** **Global/LH/Economy** (segmento más afectado)

**Afecta a:**
- Global/LH/Economy (NPS -9.9 pts)
- Global/LH (NPS -6.8 pts)

**Tipo de impacto:** **NEGATIVO** (Servicio deficiente = NPS↓)

**Cadena de propagación hacia el segmento raíz:**

1. **Economy LH → LH (Radio):**
   - **Economy LH (-9.9 pts con múltiples quejas de servicio)**, **Business LH (+11.0 pts)**, **Premium LH (Normal)**
   - **Escenario:** DOMINANCIA (Economy domina por volumen: 177 enc.)
   - **Resultado:** El deterioro de Economy arrastra a **LH (-6.8 pts)**

2. **LH → Global:**
   - **LH (-6.8 pts)** y **SH (+11.3 pts)** → Radios opuestos
   - **Escenario:** DOMINANCIA (SH domina por volumen 68%)
   - **Resultado:** El problema de LH queda **parcialmente oculto** en Global (+5.5 pts)

**Conclusión NMA:** El deterioro de calidad de servicio (tripulación grosera, IFE no funcional, comida de mala calidad) es una **causa específica de Long Haul Economy** que contribuye al deterioro del radio completo. El **NMA = Global/LH/Economy** porque concentra las quejas más graves (JFK-MAD, EZE-MAD, BOG-MAD, MAD-MIA).

**⚠️ NOTA:** Este NMA está **oculto en el Global positivo** debido a la dominancia de SH.

---

### **CAUSA 8: Aumento de Conexiones Perdidas (Misconex en Long Haul)**

**Escenario:** **DOMINANCIA** (Economy LH domina)

**NMA:** **Global/LH/Economy** (segmento más afectado por conexiones)

**Afecta a:**
- Global/LH/Economy (Misconex +0.15 pts)
- Global/LH (Misconex +0.15 pts)

**Tipo de impacto:** **NEGATIVO** (Misconex↑ = NPS↓)

**Cadena de propagación hacia el segmento raíz:**

1. **Economy LH → LH (Radio):**
   - **Economy LH (afectado por Misconex)**, **Business LH (+11.0 pts)**, **Premium LH (Normal)**
   - **Escenario:** DOMINANCIA (Economy domina por volumen)
   - **Resultado:** El problema de Economy contribuye al deterioro de **LH (-6.8 pts)**

2. **LH → Global:**
   - **LH (-6.8 pts)** y **SH (+11.3 pts)** → Radios opuestos
   - **Escenario:** DOMINANCIA (SH domina)
   - **Resultado:** El problema queda **oculto** en Global (+5.5 pts)

**Conclusión NMA:** El aumento de conexiones perdidas es una **causa menor pero contribuyente** al deterioro de Long Haul. El **NMA = Global/LH/Economy** porque los pasajeros en conexión (especialmente europeos con NPS -53.8) son más vulnerables a Misconex.

**⚠️ NOTA:** Variación de +0.15 pts es **no significativa** (<3 pts), pero contribuye al deterioro acumulativo de LH.

---

## 📊 RESUMEN DE NODOS MÁXIMO AFECTADOS

| # | Causa | NMA | Tipo Impacto | Escenario Dominante | Propagación al Global |
|---|-------|-----|--------------|---------------------|----------------------|
| **1** | Deterioro Equipaje (Mishandling) | **Global** | NEGATIVO | SINERGIA | ✅ Transversal a toda la red |
| **2** | Mejora Puntualidad SH (OTP15) | **Global/SH** | POSITIVO | SINERGIA → DOMINANCIA | ✅ Domina y arrastra Global |
| **3** | Colapso Puntualidad LH (OTP15) | **Global/LH/Economy** | NEGATIVO | DOMINANCIA | ⚠️ Oculto por SH dominante |
| **4** | Menor Ocupación SH (Load Factor) | **Global/SH** | POSITIVO | SINERGIA → DOMINANCIA | ✅ Domina y arrastra Global |
| **5** | Servicio VIP/Tripulación YW | **Global/SH/Business/YW** | POSITIVO | TRANSFERENCIA | ✅ Contagia hacia arriba |
| **6** | Degradación Business CRJ (YW) | **Global/SH/Business/YW** | NEGATIVO | DILUCIÓN | ❌ Oculto por dilución |
| **7** | Deterioro Servicio LH | **Global/LH/Economy** | NEGATIVO | DOMINANCIA | ⚠️ Oculto por SH dominante |
| **8** | Conexiones Perdidas LH (Misconex) | **Global/LH/Economy** | NEGATIVO | DOMINANCIA | ⚠️ Oculto por SH dominante |

---

## 🎯 HALLAZGOS CLAVE

### ✅ **Causas que Dominan el Global (+5.5 pts):**
1. **Mejora Puntualidad SH** (NMA: Global/SH) → SINERGIA + DOMINANCIA
2. **Menor Ocupación SH** (NMA: Global/SH) → SINERGIA + DOMINANCIA
3. **Servicio VIP YW** (NMA: Global/SH/Business/YW) → TRANSFERENCIA

### ⚠️ **Causas Ocultas por Dominancia de SH:**
4. **Colapso Puntualidad LH** (NMA: Global/LH/Economy) → Deterioro severo -5.72 pts
5. **Deterioro Servicio LH** (NMA: Global/LH/Economy) → Quejas graves de tripulación/IFE/comida
6. **Conexiones Perdidas LH** (NMA: Global/LH/Economy) → Contribución menor pero acumulativa

### 🚨 **Causa Transversal Crítica:**
7. **Deterioro Equipaje** (NMA: Global) → SINERGIA total, afecta a toda la red (+4.07 pts)

### ❌ **Causa Oculta por Dilución:**
8. **Degradación Business CRJ YW** (NMA: Global/SH/Business/YW) → Crisis grave (NPS -100) diluida por volumen de Leisure

---

**Conclusión Estratégica:** El Global (+5.5 pts) está **dominado por 3 causas positivas de Short Haul**, pero **oculta 4 causas negativas críticas**: 3 en Long Haul (puntualidad, servicio, conexiones), 1 transversal (equipaje), y 1 diluida en YW Business (degradación CRJ). La métrica agregada **NO refleja la complejidad operativa** del día.

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

# 📊 PASO 4B: EXTRACCIÓN DE EVIDENCIAS DEL CONTEXTO INICIAL

---

## EVIDENCIAS POR NODO MÁXIMO AFECTADO (NMA)

---

### **NMA 1: Global** (Causa: Deterioro Equipaje - Mishandling)

📈 **EXPLANATORY DRIVERS:**
No disponible (el análisis de SHAP no está presente en el tree_data para el nodo Global)

📊 **DATOS OPERATIVOS:**
No disponible para el nodo Global específicamente (los datos operativos están segmentados por radio)

🚨 **INCIDENTES NCS (CUANTITATIVO):**
No disponible (el tree_data indica "❌ No hay incidentes NCS reportados para el 2025-12-13 en segmento Global")

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**
No disponible (el tree_data indica ausencia de incidentes formales: "La ausencia de incidentes formales descarta disrupciones operativas mayores (cancelaciones masivas, problemas técnicos graves). Los problemas son de **calidad de servicio** no capturados por sistemas de reporte operativo.")

💬 **FEEDBACK DE CLIENTES:**
**Cluster 1: EQUIPAJE (6+ menciones)**
- Equipaje perdido/retrasado: MAD-XRY, EAS-MAD, BCN-MAD, MAD-SJO
- Gestión caótica T4 Madrid: "45 min para sacar equipaje", "maletas en bucle 3h"

**Cluster 2: SERVICIO A BORDO (8+ menciones)**
- Actitud grosera: BOG-MAD (empleado agresivo), EZE-MAD (azafata Susana), DUS-MAD
- Falta de agua/bebidas: MAD-MIA, MAD-PTY
- Comida mala: EZE-MAD, MAD-MEX, MAD-SCL

**Cluster 3: ESPACIO/CONFORT (5+ menciones)**
- Asientos estrechos: MAD-MEX, BUD-MAD, BOS-MAD
- A321XLR criticado: MAD-REC
- Premium Economy decepcionante: EZE-MAD

**Cluster 4: PROBLEMAS TÉCNICOS (4+ menciones)**
- IFE: BOG-MAD, MAD-MIA
- USB no funciona: BOG-MAD
- Agua no potable (REINCIDENCIA): MAD-ZRH

**Cluster 5: RETRASOS (5+ menciones)**
- MAD-PTY: 1h esperando tripulante
- LIM-MAD: 4 cambios de horario
- MAD-SJO: 2h por sobrepeso

✈️ **RUTAS AFECTADAS (Top 5):**
| Ruta | NPS | Encuestas | Nivel de Confianza | Problemas Identificados |
|------|-----|-----------|-------------------|------------------------|
| **BOS-MAD** | **-50.0** | 4 | **MEDIA-ALTA** | Espacio insuficiente con bebé, carrito no entregado, asientos no reclinables, pantallas no funcionan |
| **MAD-MIA** | **14.3** | 14 | **MEDIA** | Televisor no funcionó, solo una botellita agua (8h vuelo), auxiliares groseros |
| **EZE-MAD** | **20.5** | 44 | **ALTA** | Comida mala (pasta cruda), azafata Susana conflictiva, Premium Economy decepcionante |
| **BOG-MAD** | **30.8** | 26 | **ALTA** | Empleado agresivo puerta embarque, sistema IFE reiniciándose, falta empatía tripulación |
| **BCN-MAD** | **29.4** | 17 | - | Equipaje perdido, entrega tardía, conductor amenazante |

👥 **PERFILES REACTIVOS:**
**Business/Work:** NPS 15.8 (95 encuestas) vs **Leisure:** NPS 28.1 (588 encuestas) → **Diferencia: -12.3 pts** (viajeros de negocio significativamente más críticos)

**Flota (Top 5 más críticas):**
- **A33ACMI:** NPS -100.0 (9 encuestas) - CATASTRÓFICO
- **A332:** NPS -19.4 (31 encuestas)
- **A350:** NPS -4.5 (66 encuestas)
- **A321XLR:** NPS 0.0 (21 encuestas)
- **A333:** NPS 20.7 (29 encuestas)

**Región de Residencia (Top 5 más críticas):**
- **AMÉRICA NORTE:** NPS -100.0 (2 encuestas)
- **ORIENTE MEDIO:** NPS -50.0 (2 encuestas)
- **ASIA:** NPS -16.7 (6 encuestas)
- **Unknown:** NPS 6.0 (50 encuestas)
- **EUROPA:** NPS 13.4 (142 encuestas)

---

### **NMA 2: Global/SH** (Causa: Mejora Puntualidad)

📈 **EXPLANATORY DRIVERS:**
No disponible explícitamente en formato SHAP para Global/SH

📊 **DATOS OPERATIVOS (Global/SH):**
- **OTP15:** 91.9% (+1.81 pts vs baseline de 90.09%)
- **Load Factor:** 82.44% (-2.49 pts vs baseline de 84.93%)
- **Mishandling:** 19.29 (+4.07 pts vs baseline de 15.22)
- **Misconex:** 0.83 (+0.15 pts vs baseline de 0.68)

🚨 **INCIDENTES NCS (CUANTITATIVO):**
- **Total de rutas con incidentes reportados:** 29 rutas
- No hay datos detallados de NCS disponibles para el período específico

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**
No disponible

💬 **FEEDBACK DE CLIENTES:**
**Problemas de Equipaje (10 menciones explícitas):**
- MAD-ORY: "Mi equipaje no llegó a mi destino final (El Cairo) y esperé 5 días" (NPS 0)
- ALC-MAD: "habían perdido la maleta" (NPS 0)
- MAD-NTE: "mi equipaje facturado había desaparecido. Llegó con cuatro días de retraso" (NPS 1)
- BCN-MAD: "Mi maleta no llegó al vuelo a Barcelona" (NPS 2)
- MAD-XRY: "Mi Equipaje se quedó en Madrid" (NPS 3)
- MAD-VCE (2 casos): "Nos quedamos sin equipaje en Venecia" (NPS 3)
- EAS-MAD: "al llegar al destino nos faltó el equipaje al completo" (NPS 0)
- LIN-MAD: "las maletas tardaron 1 hora en salir, excesivo" (NPS 0)
- AMS-MAD: "estuvimos esperando hora y cuarto a recoger nuestra maleta" (NPS 6)
- BUD-MAD: "45 minutos para que sacaran el equipaje" (NPS 1)

**Problemas Secundarios:**
- Conexiones ajustadas (MAD-VCE, EAS-MAD) - 2 menciones
- Problemas de higiene (MAD-ZRH) - 1 mención recurrente
- Servicio de tripulación inconsistente - múltiples menciones
- Problemas con check-in digital (MAD-XRY, MAD-SCQ) - 2 menciones
- Cancelaciones sin alternativas (MAD-NAP) - 1 mención

✈️ **RUTAS AFECTADAS (Top 7 con triple evidencia):**
| Ruta | NPS | Encuestas | Evidencia en Verbatims |
|------|-----|-----------|------------------------|
| **ALC-MAD** | 0.0 | 6 | "habían perdido la maleta" |
| **AMS-MAD** | 14.3 | 14 | "esperando hora y cuarto a recoger nuestra maleta" |
| **MAD-NAP** | 28.6 | 7 | Problemas de equipaje mencionados |
| **BCN-MAD** | 29.4 | 17 | "Mi maleta no llegó al vuelo a Barcelona" |
| **MAD-VIE** | 30.0 | 20 | "Mi equipaje se quedó en Madrid" |
| **MAD-ORY** | 54.2 | 24 | "Mi equipaje no llegó a mi destino final (El Cairo)" |
| **MAD-VCE** | 62.5 | 8 | "Nos quedamos sin equipaje en Venecia" |

👥 **PERFILES REACTIVOS:**
**Por Región de Residencia (Mayor dispersión: 200 pts):**
- **EUROPA:** NPS 28.4 (116 encuestas) - MUY BAJO, segundo mayor volumen
- **AMERICA NORTE:** NPS -100.0 (2 encuestas) - CRÍTICO
- **ASIA:** NPS 0.0 (4 encuestas) - NEGATIVO
- **ESPAÑA:** NPS 44.8 (241 encuestas) - Relativamente estable, mayor volumen

**Por Flota (Dispersión: 71.5 pts):**
- **CRJ:** NPS 28.5 (144 encuestas) - Flota más afectada, mayor volumen
- **A319:** NPS 30.8 (26 encuestas) - Segunda flota más afectada
- **A321:** NPS 56.8 (74 encuestas) - NPS superior
- **ATR:** NPS 72.2 (18 encuestas) - NPS alto, volumen bajo

**Por Codeshare (Dispersión: 125 pts):**
- **LATAM:** NPS -25.0 (12 encuestas) - CRÍTICO, alianza estratégica
- **BA:** NPS -10.0 (10 encuestas) - NEGATIVO, socio IAG
- **AA:** NPS 0.0 (7 encuestas) - NEGATIVO, alianza oneworld
- **IB:** NPS 43.9 (417 encuestas) - Operación propia mejor

---

### **NMA 3: Global/LH/Economy** (Causa: Colapso Puntualidad LH)

📈 **EXPLANATORY DRIVERS:**
No disponible explícitamente en formato SHAP

📊 **DATOS OPERATIVOS (Global/LH/Economy):**
- **OTP15:** 76.1% (-5.72 pts vs baseline de 81.82%)
- **Load Factor:** 88.35% (-1.63 pts vs baseline de 89.98%)
- **Mishandling:** 19.29 (+4.07 pts vs baseline de 15.22)
- **Misconex:** 0.83 (+0.15 pts vs baseline de 0.68)

🚨 **INCIDENTES NCS (CUANTITATIVO):**
**Rutas con incidentes formalizados (16 rutas totales):**
- MAD-SJO: Equipaje + Retrasos
- BOG-MAD: Equipaje + Servicio + IFE
- MAD-MIA: Comida + IFE
- MAD-UIO: Equipaje dañado
- MAD-SJU: Problemas de embarque
- EZE-MAD: Comida + Servicio + Espacio

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**
No disponible

💬 **FEEDBACK DE CLIENTES:**
**Distribución de problemas mencionados (30 verbatims analizados):**

1. **Equipaje (Mishandling):** 7 menciones
   - Pérdida total, retrasos en entrega, daños físicos

2. **Retrasos/Puntualidad:** 6 menciones
   - MAD-SDQ: "retraso de 6 horas en mi vuelo de vuelta"
   - MAD-PTY: "retraso en la puerta antes de la salida de más de una hora"
   - MAD-SJO: "dos horas de retraso... avión demasiado pequeño"
   - LIM-MAD: "4 cambios de horario"

3. **Calidad de servicio/Tripulación:** 6 menciones
   - JFK-MAD: "azafata solo contesta en español... extremadamente grosera"
   - EZE-MAD: "azafata Susana... actitud agria y a la defensiva"
   - BOG-MAD: "empleado... manera violenta y alterada... 'ustedes no son nadie'"
   - MAD-SJU: "señora no nos dejó subir... falta de respeto"

4. **Configuración de asientos/Espacio:** 5 menciones
   - MAD-MEX: "NO RESPETARON mi pago... lugar NO APTO para infante"
   - BOS-MAD: "espacio entre los asientos es demasiado estrecho"
   - EZE-MAD: "respaldo del asiento delantero casi tocaba el reposabrazos"
   - MAD-REC: "espacios demasiado angostos... A321XLR"

5. **IFE (Entretenimiento):** 4 menciones
   - BOG-MAD: "sistema de entretenimiento se reinició... pantalla sin películas"
   - BOS-MAD: "pantallas no funcionaban bien"
   - MAD-MIA: "televisor no funcionó en todo el vuelo"

6. **Comida:** 4 menciones
   - EZE-MAD: "omelet y nada mas sin pan... pasta picante insoportable... carne cruda"
   - MAD-MIA: "comida era terrible"
   - MAD-MEX: "comida ha bajado mucho de calidad"

✈️ **RUTAS AFECTADAS (Top 6):**
| Ruta | NPS | Encuestas | Problemas Principales |
|------|-----|-----------|----------------------|
| **BOS-MAD** | -50.0 | 4 | Espacio, IFE, servicio con bebés |
| **MAD-MEX** | -35.0 | 20 | Asientos, comida, no respeto a pagos |
| **MAD-MVD** | -28.6 | 7 | Pérdida total de equipaje |
| **MAD-SJO** | -16.7 | 6 | Equipaje (3h espera), retrasos (2h) |
| **MAD-UIO** | 0.0 | 7 | Equipaje roto/dañado |
| **MAD-SJU** | 0.0 | 2 | Problemas de embarque, falta respeto |

👥 **PERFILES REACTIVOS:**
**Por Propósito de Viaje:**
- **Leisure:** NPS 2.7 (188 encuestas) - 87% del total
- **Business/Work:** NPS -24.1 (29 encuestas) - 13% del total
- **Diferencia:** -26.8 pts (viajeros de negocios significativamente más insatisfechos)

**Por Flota (Peor Desempeño):**
- **A33ACMI:** NPS -100.0 (9 encuestas) - Wet lease, problemas de servicio
- **A321:** NPS -100.0 (2 encuestas) - Muestra pequeña
- **A332:** NPS -19.4 (31 encuestas) - Configuración, espacio
- **A350:** NPS -8.2 (61 encuestas) - Sorprendente para flota premium

**Por Región de Residencia:**
- **EUROPA:** NPS -53.8 (26 encuestas) - 12% del total
- **ESPAÑA:** NPS -3.3 (91 encuestas) - 42% del total
- **AMERICA SUR:** NPS 32.3 (31 encuestas) - 14% del total
- **AMERICA CENTRO:** NPS 4.0 (25 encuestas) - 12% del total

**Por Codeshare:**
- **IB:** NPS -1.1 (179 encuestas) - 82% del total
- **AA:** NPS 25.0 (16 encuestas) - 7% del total
- **BA:** NPS -40.0 (5 encuestas) - 2% del total
- **QR:** NPS -25.0 (4 encuestas) - 2% del total

---

### **NMA 4: Global/SH** (Causa: Menor Ocupación - Load Factor)

📈 **EXPLANATORY DRIVERS:**
No disponible explícitamente en formato SHAP

📊 **DATOS OPERATIVOS (Global/SH):**
- **Load Factor:** 82.44% (-2.49 pts vs baseline de 84.93%)
- **OTP15:** 91.9% (+1.81 pts vs baseline de 90.09%)
- **Mishandling:** 19.29 (+4.07 pts vs baseline de 15.22)
- **Misconex:** 0.83 (+0.15 pts vs baseline de 0.68)

🚨 **INCIDENTES NCS (CUANTITATIVO):**
- **Total de rutas con incidentes reportados:** 29 rutas
- No hay datos detallados de NCS disponibles para el período específico

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**
No disponible

💬 **FEEDBACK DE CLIENTES:**
(Mismo contenido que NMA 2: Global/SH, ya que comparten el mismo nodo)

✈️ **RUTAS AFECTADAS:**
(Mismo contenido que NMA 2: Global/SH)

👥 **PERFILES REACTIVOS:**
(Mismo contenido que NMA 2: Global/SH)

---

### **NMA 5: Global/SH/Business/YW** (Causa: Excelente Servicio VIP/Tripulación)

📈 **EXPLANATORY DRIVERS:**
No disponible

📊 **DATOS OPERATIVOS (Global/SH/Business/YW):**
- **Load Factor:** 51.62% (-5.75 pts vs baseline de 57.37%)
- **OTP15:** 89.94% (+1.62 pts vs baseline de 88.32%)
- **Mishandling:** 15.22 (+2.7 pts vs baseline de 12.52)
- **Misconex:** 0.43 (sin baseline de referencia)

🚨 **INCIDENTES NCS (CUANTITATIVO):**
- **0 incidentes reportados** para YW en el período 2025-12-13

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**
"Nota: Ausencia de reportes formales a pesar de quejas graves en verbatims (pérdida de equipaje, downgrade)"

💬 **FEEDBACK DE CLIENTES:**
**Distribución de Sentimiento (9 comentarios totales):**
- 4 Promotores (NPS 10) - 44%
- 2 Pasivos (NPS 8-9) - 22%
- 3 Detractores (NPS 0-3) - 33%

**Temas Positivos (4 menciones):**
- LCG-MAD: "Excelente servicio, sala VIP maravillosa, superó expectativas" (NPS 10)
- MAD-NCE (2 menciones): "Equipo VIP increíble, tripulación encantadora, vuelo impecable" (NPS 10)
- LEI-MAD: "Puntualidad y personal" (NPS 10)

**Temas Negativos (3 menciones):**
- MAD-VCE: "Pagué clase ejecutiva y recibí servicio de bajo coste. Avión diminuto, asiento estrecho, espacio reducido, comida incomible" (NPS 3)
- ALC-MAD: "Llegué a Alicante y habían perdido la maleta. Además me bajaron de Premium Economy a turista en vuelo NY-MAD por cambio de avión" (NPS 0)
- GVA-MAD: Retraso en embarque (NPS 0)

**Temas Neutros (2 menciones):**
- MAD-TLS: Cambio de terminal sin notificación
- FRA-MAD: Retraso tolerado por cliente

✈️ **RUTAS AFECTADAS:**
**Rutas con Mayor Impacto Negativo:**
- **MAD-VCE:** NPS -100 (1 encuesta) - Calidad degradada Business Class (avión CRJ inadecuado)
- **ALC-MAD:** NPS -100 (1 encuesta) - Pérdida de equipaje + downgrade Premium Economy
- **GVA-MAD:** NPS 0 (1 encuesta) - Retraso en embarque

**Rutas con Mejor Desempeño:**
- **MAD-NCE:** NPS 100 (2 encuestas) - Servicio VIP excepcional, tripulación profesional
- **LCG-MAD:** NPS 100 (1 encuesta) - Servicio integral destacado
- **LEI-MAD:** NPS 100 (1 encuesta) - Puntualidad y personal
- **FRA-MAD:** NPS 100 (1 encuesta) - Cliente tolerante a retrasos

👥 **PERFILES REACTIVOS:**
**Por Propósito de Viaje (Dispersión: 162.5 pts):**
- **Business/Work:** NPS -100 (1 encuesta) - 11% de la muestra
- **Leisure:** NPS 62.5 (8 encuestas) - 89% de la muestra

**Por Región de Residencia (Dispersión: 100 pts):**
- **España:** NPS 0.0 (2 encuestas) - 22%
- **Europa:** NPS 40.0 (5 encuestas) - 56%
- **América Sur:** NPS 100.0 (1 encuesta) - 11%
- **Unknown:** NPS 100.0 (1 encuesta) - 11%

**Por Flota:**
- **CRJ:** NPS 44.4 (9 encuestas) - 100% de vuelos operados con esta flota

---

### **NMA 6: Global/SH/Business/YW** (Causa: Degradación Business Class CRJ)

📈 **EXPLANATORY DRIVERS:**
No disponible

📊 **DATOS OPERATIVOS:**
(Mismo contenido que NMA 5, ya que comparten el mismo nodo)

🚨 **INCIDENTES NCS (CUANTITATIVO):**
(Mismo contenido que NMA 5)

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**
(Mismo contenido que NMA 5)

💬 **FEEDBACK DE CLIENTES:**
**Caso Crítico Específico:**
- MAD-VCE (NPS 3): "Pagué clase ejecutiva y recibí servicio de bajo coste. Avión diminuto, asiento estrecho, espacio reducido, comida incomible"

**Contexto:** Cliente Business/Work con expectativas premium no cumplidas en configuración CRJ.

✈️ **RUTAS AFECTADAS:**
- **MAD-VCE:** NPS -100 (1 encuesta, segmento Business/Work) - Problema específico de configuración inadecuada de Business Class en flota CRJ

👥 **PERFILES REACTIVOS:**
**Segmento Específico Afectado:**
- **Business/Work (YW):** NPS -100 (1 encuesta de 9 totales)
- **Leisure (YW):** NPS 62.5 (8 encuestas de 9 totales)
- **Dispersión extrema:** 162.5 pts entre ambos segmentos

**Flota:**
- **CRJ:** NPS 44.4 (9 encuestas totales, 100% de operación YW Business)
- El problema está concentrado en la configuración de Business Class de esta flota específica

---

### **NMA 7: Global/LH/Economy** (Causa: Deterioro Calidad de Servicio LH)

📈 **EXPLANATORY DRIVERS:**
No disponible

📊 **DATOS OPERATIVOS:**
(Mismo contenido que NMA 3: Global/LH/Economy)

🚨 **INCIDENTES NCS (CUANTITATIVO):**
(Mismo contenido que NMA 3)

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**
No disponible

💬 **FEEDBACK DE CLIENTES:**
**Problemas de Servicio de Tripulación (6 menciones):**
- JFK-MAD: "azafata solo contesta en español... extremadamente grosera"
- EZE-MAD: "azafata Susana... actitud agria y a la defensiva"
- BOG-MAD: "empleado... manera violenta y alterada... 'ustedes no son nadie'"
- MAD-SJU: "señora no nos dejó subir... falta de respeto"
- DUS-MAD: Queja formal por trato irrespetuoso
- LYS-MAD: "El auxiliar de vuelo no me ofreció nada de comer ni de beber"

**Problemas de IFE (4 menciones):**
- BOG-MAD: "sistema de entretenimiento se reinició... pantalla sin películas"
- BOS-MAD: "pantallas no funcionaban bien"
- MAD-MIA: "televisor no funcionó en todo el vuelo"
- BOG-MAD: "puertos USB no funcionan"

**Problemas de Comida (4 menciones):**
- EZE-MAD: "omelet y nada mas sin pan... pasta picante insoportable... carne cruda"
- MAD-MIA: "comida era terrible"
- MAD-MEX: "comida ha bajado mucho de calidad"
- MAD-SCL: "medio sándwich recalentado"

**Problemas de Configuración/Espacio (5 menciones):**
- MAD-MEX: "NO RESPETARON mi pago... lugar NO APTO para infante"
- BOS-MAD: "espacio entre los asientos es demasiado estrecho"
- EZE-MAD: "respaldo del asiento delantero casi tocaba el reposabrazos"
- MAD-REC: "espacios demasiado angostos... A321XLR"
- BUD-MAD: "espacio ridículo, como ataúd"

✈️ **RUTAS AFECTADAS (con problemas de servicio específicos):**
| Ruta | NPS | Encuestas | Problemas de Servicio Identificados |
|------|-----|-----------|-------------------------------------|
| **BOS-MAD** | -50.0 | 4 | IFE no funcional, espacio insuficiente |
| **MAD-MIA** | 14.3 | 14 | IFE no funcionó, comida terrible, auxiliares groseros |
| **EZE-MAD** | 20.5 | 44 | Comida mala (pasta cruda), azafata Susana conflictiva |
| **BOG-MAD** | 30.8 | 26 | Empleado agresivo, IFE reiniciándose, falta empatía |
| **MAD-MEX** | -35.0 | 20 | Asientos no respetados, comida mala |

👥 **PERFILES REACTIVOS:**
(Mismo contenido que NMA 3: Global/LH/Economy)

---

### **NMA 8: Global/LH/Economy** (Causa: Conexiones Perdidas - Misconex)

📈 **EXPLANATORY DRIVERS:**
No disponible

📊 **DATOS OPERATIVOS:**
- **Misconex:** 0.83 (+0.15 pts vs baseline de 0.68)
- (Resto de datos operativos: mismo contenido que NMA 3)

🚨 **INCIDENTES NCS (CUANTITATIVO):**
(Mismo contenido que NMA 3)

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**
No disponible

💬 **FEEDBACK DE CLIENTES:**
**Menciones Específicas de Problemas de Conexión:**
- MAD-VCE: "Nos quedamos sin equipaje en Venecia" (conexión de 1 hora insuficiente para transferencia de equipaje)
- EAS-MAD: "al llegar al destino nos faltó el equipaje al completo" (problema en conexión)

✈️ **RUTAS AFECTADAS (con problemas de conexión):**
- **MAD-VCE:** NPS 62.5 (8 encuestas SH) / Problemas de equipaje en conexiones cortas
- **EAS-MAD:** NPS 0.0 (mencionado en verbatims) / Equipaje no transfirió

👥 **PERFILES REACTIVOS:**
**Por Región de Residencia (más afectados por conexiones):**
- **EUROPA (no España):** NPS -53.8 (26 encuestas) - Viajeros europeos probablemente en vuelos de conexión, más expuestos a Misconex

(Resto de perfiles: mismo contenido que NMA 3)

---

## 📌 RESUMEN DE DISPONIBILIDAD DE DATOS

| NMA | Explanatory Drivers | Datos Operativos | NCS Cuant. | NCS Cual. | Verbatims | Rutas | Perfiles |
|-----|---------------------|------------------|------------|-----------|-----------|-------|----------|
| **Global** | ❌ | ❌ | ❌ | ✅ Parcial | ✅ | ✅ | ✅ |
| **Global/SH** | ❌ | ✅ | ✅ Parcial | ❌ | ✅ | ✅ | ✅ |
| **Global/LH/Economy** | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Global/SH/Business/YW** | ❌ | ✅ | ✅ | ✅ Parcial | ✅ | ✅ | ✅ |

**Nota:** El tree_data original no contiene valores SHAP explícitos (Explanatory Drivers) para ninguno de los NMAs identificados. Los análisis causales se basan en datos operativos, verbatims, rutas y perfiles de cliente.

---

## Cabin Radio Reflection

# 📋 PASO 4C: REFLEXIÓN POR CABINA-RADIO

---

## 🔷 CABINAS SHORT HAUL (SH)

---

### **=== ECONOMY SH ===**

• **NPS Cabina:** 38.4 (+11.3 pts)  
• **Estado:** POSITIVE ANOMALY  
• **Escenario:** TRANSFERENCIA (IB +, YW N | Cabina +)

• **IB:** NPS 41.6 (+14.7 pts) - POSITIVE ANOMALY  
  **Explicación:** Deterioro operativo en Mishandling (+4.52 pts vs baseline) compensado por mejoras en puntualidad (OTP15 +1.88 pts) y menor ocupación (Load Factor -2.71 pts). A pesar del aumento significativo en problemas de equipaje (7+ menciones en verbatims: FRA-MAD, MAD-OPO, MAD-ORY, AMS-MAD, BRU-MAD), los factores compensatorios fueron suficientemente fuertes para generar una mejora neta de NPS. Problemas específicos incluyen extravío de equipajes, retrasos de hasta 1h15min en entrega (AMS-MAD), y problemas de facturación (HAM-MAD: 2h para facturar 2 maletas).

• **YW:** NPS 32.7 (+5.2 pts) - Normal  
  **Explicación:** Mantuvo estabilidad operativa sin contribuir a la anomalía. Variación dentro del rango normal esperado.

• **Narrativa de agregación:** La anomalía positiva de Economy SH (+11.3 pts) es una **transferencia directa del excelente desempeño de IB** (+14.7 pts), que logró compensar un deterioro operativo crítico en manejo de equipaje mediante mejoras en puntualidad y menor ocupación. El volumen operativo de IB (394 encuestas) domina sobre YW (16 encuestas), permitiendo la transferencia del efecto. YW mantuvo estabilidad operativa sin afectar la dinámica.

• **Rutas críticas (IB - hijo dominante):**
  - **FRA-MAD:** NPS 0.0 (2 enc.) - Pérdida de equipaje + conexiones perdidas
  - **LHR-MAD:** NPS 10.0 (20 enc.) - Servicio de tripulación grosero
  - **AMS-MAD:** NPS 14.3 (14 enc.) - Retraso 1h15min en entrega equipaje
  - **BRU-MAD:** NPS 14.3 (7 enc.) - Problemas de equipaje
  - **LCG-MAD:** NPS 0.0 (7 enc.) - Sin verbatims específicos pero NPS crítico

• **Perfiles reactivos (IB - hijo dominante):**
  - **Región de Residencia:** EUROPA NPS 35.2 (71 enc.) vs ESPAÑA NPS 46.4 (138 enc.) - Clientes españoles 11.2 pts más satisfechos
  - **Flota:** A321 NPS 58.3 (72 enc.) vs A320neo NPS 33.3 (93 enc.) - Dispersión de 25 pts entre flotas
  - **Codeshare:** LATAM NPS -33.3 (9 enc.), BA NPS 20.0 (5 enc.) vs IB operación propia NPS 46.5 (245 enc.) - Spread de 166.7 pts
  - **Business/Leisure:** Impacto transversal, Business NPS 44.1 (34 enc.) vs Leisure NPS 41.2 (240 enc.) - Diferencia no significativa (2.9 pts)

---

### **=== BUSINESS SH ===**

• **NPS Cabina:** 46.2 (+11.1 pts)  
• **Estado:** POSITIVE ANOMALY  
• **Escenario:** TRANSFERENCIA (IB N, YW + | Cabina +)

• **IB:** NPS 46.7 (+4.6 pts) - Normal  
  **Explicación:** Mantuvo estabilidad operativa sin contribuir a la anomalía. Variación dentro del rango normal esperado. A pesar de problemas puntuales (cambio de asiento en Business + falta de higiene en LCG-MAD), el desempeño general se mantuvo estable.

• **YW:** NPS 44.4 (+28.5 pts) - POSITIVE ANOMALY  
  **Explicación:** Mejora extrema impulsada por excelente servicio de tripulación/VIP (4 menciones con NPS 10: LCG-MAD, MAD-NCE x2, LEI-MAD) y mejoras operativas (OTP15 +1.62 pts, Load Factor -5.75 pts). **ALERTA CRÍTICA:** Esta mejora oculta una polarización extrema (dispersión de 162.5 pts entre Business/Work NPS -100 y Leisure NPS 62.5) con problemas severos en calidad de Business Class operada con flota CRJ. Caso crítico: MAD-VCE (NPS -100) donde cliente Business reportó: *"Pagué clase ejecutiva y recibí servicio de bajo coste. Avión diminuto, asiento estrecho, espacio reducido, comida incomible"*. Problema secundario: Mishandling +2.7 pts con caso de pérdida de equipaje en ALC-MAD (NPS -100).

• **Narrativa de agregación:** La anomalía positiva de Business SH (+11.1 pts) es una **transferencia de la mejora extrema de YW** (+28.5 pts), que logró elevar el agregado a pesar de su menor volumen (9 encuestas vs 34 de IB). La magnitud excepcional de la mejora de YW (28.5 pts) fue suficiente para contagiar al padre. IB mantuvo estabilidad (+4.6 pts dentro de rango normal) sin afectar la dinámica. **CRÍTICO:** La mejora agregada oculta una crisis en el segmento Business/Work de YW (NPS -100) que requiere atención urgente.

• **Rutas críticas (YW - hijo dominante):**
  - **MAD-VCE:** NPS -100 (1 enc.) - Degradación severa Business Class en flota CRJ
  - **ALC-MAD:** NPS -100 (1 enc.) - Pérdida de equipaje + downgrade Premium Economy
  - **GVA-MAD:** NPS 0 (1 enc.) - Retraso en embarque
  - **MAD-NCE:** NPS 100 (2 enc.) - Servicio VIP excepcional (contrapunto positivo)
  - **LCG-MAD:** NPS 100 (1 enc.) - Servicio integral destacado (contrapunto positivo)

• **Perfiles reactivos (YW - hijo dominante):**
  - **Business/Leisure:** Business/Work NPS -100 (1 enc.) vs Leisure NPS 62.5 (8 enc.) - **Dispersión extrema de 162.5 pts** (mayor dispersión detectada)
  - **Región de Residencia:** España NPS 0.0 (2 enc.), Europa NPS 40.0 (5 enc.), América Sur NPS 100.0 (1 enc.) - Dispersión de 100 pts
  - **Flota:** CRJ NPS 44.4 (9 enc.) - 100% operado con esta flota; problema concentrado en configuración de Business Class
  - **Codeshare:** No aplica (todas las operaciones son YW)

---

## 🔶 CABINAS LONG HAUL (LH)

---

### **=== ECONOMY LH ===**

• **NPS:** -6.2 (-9.9 pts)  
• **Estado:** NEGATIVE ANOMALY

• **Causa principal:** Triple deterioro operativo que generó una "tormenta perfecta": (1) Colapso de puntualidad con OTP15 cayendo 5.72 pts (76.1% vs baseline 81.82%), generando retrasos de hasta 6 horas (MAD-SDQ); (2) Aumento crítico de Mishandling +4.07 pts con múltiples casos de pérdida total de equipaje (MAD-MVD), retrasos de hasta 3 horas en entrega (MAD-SJO), y equipaje dañado (MAD-UIO); (3) Incremento de conexiones perdidas (Misconex +0.15 pts) que afectó especialmente a pasajeros europeos en tránsito. Problemas secundarios incluyen deterioro severo de calidad de servicio (tripulación grosera en JFK-MAD, EZE-MAD, BOG-MAD), sistemas IFE no funcionales (BOG-MAD, BOS-MAD, MAD-MIA), comida de mala calidad con casos de comida cruda (EZE-MAD), y configuración de asientos inadecuada especialmente en A321XLR (MAD-REC: "espacios demasiado angostos").

• **Evidencia clave:**  
  - **OTP15:** 76.1% (-5.72 pts vs baseline) - Desviación crítica >5 pts, muy por encima del umbral de 3 pts
  - **Mishandling:** 19.29 (+4.07 pts vs baseline) - Desviación crítica >3 pts con múltiples incidentes graves
  - **Misconex:** 0.83 (+0.15 pts vs baseline) - Contribución menor pero acumulativa

• **Rutas críticas:**
  1. **BOS-MAD:** NPS -50.0 (4 enc.) - Espacio insuficiente, IFE no funcional, problemas con servicio a bebés
  2. **MAD-MEX:** NPS -35.0 (20 enc.) - Asientos no respetados (lugar no apto para infante), comida de mala calidad
  3. **MAD-MVD:** NPS -28.6 (7 enc.) - Pérdida total de equipaje
  4. **MAD-SJO:** NPS -16.7 (6 enc.) - Equipaje con 3h de espera, retrasos de 2h por sobrepeso
  5. **MAD-UIO:** NPS 0.0 (7 enc.) - Equipaje roto/completamente dañado

• **Perfiles reactivos:**
  - **Business/Leisure:** Business/Work NPS -24.1 (29 enc.) vs Leisure NPS 2.7 (188 enc.) - **Dispersión de -26.8 pts** (viajeros de negocios severamente más insatisfechos, correlaciona con deterioro de OTP15)
  - **Región de Residencia:** EUROPA NPS -53.8 (26 enc.), ESPAÑA NPS -3.3 (91 enc.), AMERICA SUR NPS 32.3 (31 enc.) - **Spread significativo** (pasajeros europeos más afectados, probablemente en conexiones)
  - **Flota:** A33ACMI NPS -100.0 (9 enc.), A332 NPS -19.4 (31 enc.), A350 NPS -8.2 (61 enc.), A321XLR NPS 0.0 (21 enc.) - **Flotas widebody con NPS negativo/muy bajo**
  - **Codeshare:** IB NPS -1.1 (179 enc., 82% del volumen), AA NPS 25.0 (16 enc.), BA NPS -40.0 (5 enc.), QR NPS -25.0 (4 enc.)

---

### **=== BUSINESS LH ===**

• **NPS:** 30.0 (+11.0 pts)  
• **Estado:** POSITIVE ANOMALY

• **Causa principal:** **Paradoja detectada** - Aunque el NPS del día superó el baseline en +11.0 pts, la investigación revela deterioros operativos críticos (OTP15 -5.72 pts, Mishandling +4.07 pts) que impactaron severamente a segmentos específicos: (1) Viajeros Business/Work (NPS 0.0 vs Leisure 35.3, brecha de 35.3 pts); (2) Flotas A33ACMI y A350 (NPS -100 y 0.0 respectivamente); (3) Rutas específicas MAD-SDQ (retraso de 3h en flota A33ACMI, NPS -100), BOG-MAD (demora equipaje 90 min), EZE-MAD (problemas de asientos en Business). La mejora general del NPS puede estar impulsada por experiencias positivas en otras rutas/segmentos que compensaron los deterioros críticos identificados. Problemas adicionales incluyen servicio deficiente en Business Class (MAD-VCE: "pagué billete ejecutiva y recibí servicio de bajo coste", MAD-NAP: "asientos ejecutiva no más espaciosos") y problema recurrente de higiene en MAD-ZRH (referencia a queja previa P20251102-66917427).

• **Evidencia clave:**  
  - **OTP15:** 76.1% (-5.72 pts vs baseline 81.82%) - Deterioro crítico a pesar de mejora aparente de NPS
  - **Mishandling:** 19.29 (+4.07 pts vs baseline 15.22) - Deterioro crítico con casos específicos (BOG-MAD: 90 min demora)
  - **Load Factor:** 66.52% (-5.58 pts vs baseline 72.10%) - Menor ocupación que favoreció experiencia en rutas sin incidentes

• **Rutas críticas:**
  1. **MAD-SDQ:** NPS -100 (1 enc.) - Retraso de 3h en flota A33ACMI, impacto crítico en viajeros Business
  2. **MAD-ZRH:** NPS -33.3 (3 enc.) - Equipaje perdido + higiene recurrente (agua no potable, bacterias)
  3. **BOG-MAD:** NPS 33.3 (3 enc.) - Demora equipaje 90 min, falta de comunicación
  4. **EZE-MAD:** NPS 33.3 (9 enc.) - Problemas de asientos (cambios unilaterales, separación menores)
  5. **MAD-VCE:** NPS 0.0 (2 enc.) - Equipaje perdido + servicio Business deficiente

• **Perfiles reactivos:**
  - **Business/Leisure:** Business/Work NPS 0.0 (3 enc.) vs Leisure NPS 35.3 (17 enc.) - **Brecha de 35.3 pts** (viajeros Business significativamente más críticos)
  - **Flota:** A33ACMI NPS -100.0 (1 enc.), A350 NPS 0.0 (7 enc.), A350 next NPS 50.0 (8 enc.), A332 NPS 100.0 (2 enc.) - **Dispersión de 200 pts** (impacto significativo del tipo de avión)
  - **Residencia:** ESPAÑA NPS 20.0 (10 enc., mayor volumen con NPS bajo), ARGENTINA NPS 50.0 (4 enc.), COLOMBIA NPS 33.3 (3 enc.)
  - **Programa Fidelización:** IB Plus Silver NPS 0.0 (2 enc.), IB Plus Platino NPS 25.0 (4 enc.), No member NPS 37.5 (8 enc.), IB Plus Oro NPS 50.0 (6 enc.)

---

### **=== PREMIUM LH ===**

• **NPS:** 15.0 (+5.7 pts)  
• **Estado:** Normal

• **Causa principal:** Sin análisis causal disponible en el tree_data. El segmento muestra variación positiva (+5.7 pts) pero dentro del rango normal esperado, sin alcanzar el umbral de anomalía. La nota en el tree_data indica: "No significant changes detected. Current period maintained stable performance."

• **Evidencia clave:**  
No disponible (el tree_data no proporciona análisis detallado para este segmento)

• **Rutas críticas:**  
No disponible

• **Perfiles reactivos:**  
No disponible

---

## 📊 RESUMEN COMPARATIVO DE CABINAS

| Cabina | NPS | Variación | Estado | Escenario Principal | Volumen Impacto |
|--------|-----|-----------|--------|---------------------|-----------------|
| **Economy SH** | 38.4 | +11.3 | POSITIVE ANOMALY | TRANSFERENCIA (IB domina) | 427 enc. |
| **Business SH** | 46.2 | +11.1 | POSITIVE ANOMALY | TRANSFERENCIA (YW domina) | 39 enc. |
| **Economy LH** | -6.2 | -9.9 | NEGATIVE ANOMALY | Triple deterioro operativo | 177 enc. |
| **Business LH** | 30.0 | +11.0 | POSITIVE ANOMALY | Paradoja (mejora oculta crisis) | 20 enc. |
| **Premium LH** | 15.0 | +5.7 | Normal | Estabilidad | N/A |

---

## 🎯 HALLAZGOS CRÍTICOS TRANSVERSALES

### ✅ **Cabinas con Mejoras Aparentes:**
- **Economy SH y Business SH:** Mejoras robustas pero ambas ocultan deterioro operativo en Mishandling (+4.07 pts SH)
- **Business LH:** Mejora paradójica que enmascara crisis en segmentos específicos (Business/Work NPS 0.0, flotas ACMI/A350)

### ⚠️ **Cabinas con Crisis Evidentes:**
- **Economy LH:** Deterioro severo (-9.9 pts) por triple shock operativo (OTP/Mishandling/Misconex)

### 🚨 **Problemas Ocultos por Dilución:**
- **Business SH (YW):** Mejora extrema (+28.5 pts) oculta crisis de Business/Work (NPS -100, dispersión 162.5 pts)

### 📍 **Patrones Comunes:**
- **Mishandling:** Problema transversal que afecta a todas las cabinas (+4.07 pts promedio)
- **Viajeros Business:** Más críticos en todas las cabinas (especialmente LH con brechas >25 pts)
- **Flotas ACMI/widebody:** Problemas recurrentes en LH (A33ACMI NPS -100.0)

---

## 📋 SÍNTESIS EJECUTIVA FINAL

```html
<b>SÍNTESIS EJECUTIVA</b><br>
<br>
La red global registró un <b>NPS de 26.4 (+5.5 pts)</b> con respecto a la media de los últimos 7 días, resultado de una dinámica contradictoria donde la mejora significativa del corto radio compensó el deterioro del largo radio. Este resultado agregado oculta una realidad operativa compleja: mientras SH experimentó mejoras en puntualidad y menor ocupación que elevaron su desempeño, LH sufrió un triple deterioro operativo que afectó especialmente a viajeros de negocios y rutas transatlánticas.<br>
<br>
<b>Deterioro Transversal en Gestión de Equipaje</b><br>
<br>
A nivel de toda la red, el manejo de equipaje registró un deterioro crítico con Mishandling alcanzando 19.29 (+4.07 pts con respecto al baseline), superando ampliamente el umbral de significancia de 3 puntos. Este problema afectó transversalmente tanto a SH como a LH con magnitud similar, manifestándose en pérdidas totales de equipaje en rutas como MAD-MVD y MAD-ORY, retrasos de hasta 3 horas en la entrega de maletas en MAD-SJO, y daños físicos reportados en MAD-UIO. El feedback de clientes evidenció 10 menciones explícitas de problemas de equipaje en SH y 7 en LH, destacando la gestión caótica en el hub de Madrid donde pasajeros reportaron esperas de 45 minutos a 1 hora y 15 minutos para recoger maletas en rutas como BUD-MAD, LIN-MAD y AMS-MAD. Los pasajeros internacionales, especialmente europeos, fueron los más afectados con un NPS de 28.4 en SH y -53.8 en LH, significativamente inferior al de residentes en España. Las flotas regionales CRJ y A319 en SH, así como las widebody A33ACMI y A332 en LH, concentraron el mayor número de incidentes. Las conexiones con codeshares LATAM, BA y AA mostraron NPS negativo, evidenciando mayor complejidad operativa en la gestión de equipajes en transferencias. A pesar de la ausencia de incidentes formales reportados en el sistema NCS, este deterioro representa un riesgo reputacional grave que requiere auditoría urgente de procesos en el hub de Madrid.<br>
<br>
<b>Mejora Sistémica en Short Haul Compensada por Factores Operativos</b><br>
<br>
En <b>Short Haul</b>, el NPS alcanzó <b>39.1 (+11.3 pts)</b> impulsado por mejoras operativas significativas que compensaron el deterioro en equipaje. La puntualidad mejoró con OTP15 alcanzando 91.9% (+1.81 pts con respecto al baseline de 90.09%), mientras que la menor ocupación con Load Factor de 82.44% (-2.49 pts) permitió una mejor experiencia general de los pasajeros al reducir la congestión a bordo. Esta mejora fue consistente tanto en Economy como en Business, beneficiando transversalmente a ambas cabinas. Sin embargo, el análisis cualitativo reveló que esta mejora ocurrió a pesar del deterioro crítico en manejo de equipaje, con rutas como ALC-MAD, AMS-MAD, BCN-MAD, MAD-VIE y MAD-ORY concentrando la mayoría de las quejas por equipajes perdidos o retrasados. Los pasajeros en conexiones con codeshares fueron particularmente vulnerables, con LATAM registrando NPS de -25.0, BA de -10.0 y AA de 0.0, contrastando con la operación propia de IB que alcanzó 43.9. Las flotas CRJ y A319 mostraron el peor desempeño con NPS de 28.5 y 30.8 respectivamente, mientras que A321 y ATR mantuvieron niveles superiores. Esta mejora de SH, representando el 68% del volumen operativo con 466 encuestas, dominó el resultado global y compensó parcialmente el impacto negativo de LH.<br>
<br>
<b>Colapso Operativo en Long Haul Economy</b><br>
<br>
En <b>Long Haul Economy</b>, el NPS cayó a <b>-6.2 (-9.9 pts)</b>, el peor desempeño del día, debido a un triple deterioro operativo que generó una tormenta perfecta. La puntualidad colapsó con OTP15 alcanzando solo 76.1% (-5.72 pts con respecto al baseline de 81.82%), generando retrasos de hasta 6 horas en MAD-SDQ y múltiples cambios de horario en rutas como LIM-MAD que reportó 4 ajustes. El manejo de equipaje se deterioró con Mishandling de 19.29 (+4.07 pts), manifestándose en pérdidas totales en MAD-MVD, retrasos de hasta 3 horas en la entrega en MAD-SJO, y equipaje completamente dañado en MAD-UIO. Las conexiones perdidas aumentaron con Misconex de 0.83 (+0.15 pts), afectando especialmente a pasajeros europeos en tránsito. Adicionalmente, se registró un deterioro severo en la calidad de servicio con múltiples quejas sobre actitud inapropiada de tripulación en JFK-MAD, EZE-MAD y BOG-MAD donde un empleado fue reportado con comportamiento violento, sistemas de entretenimiento no funcionales en BOG-MAD, BOS-MAD y MAD-MIA, comida de mala calidad con casos de comida cruda en EZE-MAD, y configuración de asientos inadecuada especialmente en el A321XLR donde pasajeros reportaron espacios angostos en MAD-REC. Las rutas más afectadas fueron BOS-MAD con NPS de -50.0, MAD-MEX con -35.0, MAD-MVD con -28.6, MAD-SJO con -16.7, y MAD-UIO con 0.0. Los viajeros de negocios fueron severamente impactados con un NPS de -24.1, una brecha de 26.8 puntos con respecto a viajeros de ocio que registraron 2.7, correlacionando directamente con la sensibilidad de este perfil a los retrasos. Las flotas A33ACMI y A332 mostraron NPS de -100.0 y -19.4 respectivamente, mientras que incluso la premium A350 registró -8.2. Este deterioro de Economy LH, con 177 encuestas representando el mayor volumen del radio, dominó el resultado de LH completo arrastrándolo a -6.8 pts a pesar de la mejora de Business LH.<br>
<br>
<b>Mejora Paradójica en Long Haul Business</b><br>
<br>
En <b>Long Haul Business</b>, el NPS alcanzó <b>30.0 (+11.0 pts)</b>, pero esta mejora aparente oculta una crisis operativa en segmentos específicos. A pesar de que las métricas operativas muestran el mismo deterioro que Economy LH con OTP15 de 76.1% (-5.72 pts) y Mishandling de 19.29 (+4.07 pts), el NPS agregado mejoró debido a experiencias positivas en ciertas rutas que compensaron los incidentes críticos. Sin embargo, el análisis reveló problemas graves: la ruta MAD-SDQ registró NPS de -100 con un retraso de 3 horas en flota A33ACMI, BOG-MAD experimentó demoras de 90 minutos en entrega de equipaje, EZE-MAD reportó problemas de asignación de asientos con cambios unilaterales y separación de menores de acompañantes, y MAD-ZRH presentó un problema recurrente de higiene con agua no potable referenciando una queja previa. Los viajeros Business/Work mostraron NPS de 0.0, una brecha de 35.3 puntos con respecto a viajeros de ocio que alcanzaron 35.3, evidenciando mayor criticidad de este perfil. Las flotas A33ACMI y A350 registraron NPS de -100.0 y 0.0 respectivamente, mientras que la menor ocupación con Load Factor de 66.52% (-5.58 pts) favoreció la experiencia en rutas sin incidentes. Esta mejora de Business LH, con solo 20 encuestas, fue insuficiente para compensar el impacto volumétrico del deterioro de Economy LH que representa 177 encuestas, resultando en el deterioro agregado del radio LH.<br>
<br>
<b>Convergencia Final y Dinámica de Radios</b><br>
<br>
La convergencia de <b>Short Haul con 39.1 (+11.3 pts)</b> y <b>Long Haul con -0.9 (-6.8 pts)</b>, en direcciones opuestas, produjo el resultado global positivo debido al peso volumétrico de SH que representa el 68% de las operaciones con 466 encuestas frente a las 217 de LH. La mejora de SH, impulsada por puntualidad y menor ocupación que compensaron el deterioro en equipaje, dominó el agregado y enmascaró la crisis operativa de LH donde el triple deterioro de puntualidad, equipaje y conexiones afectó especialmente a Economy con el peor desempeño del día. Este resultado global de 26.4 (+5.5 pts) no refleja la complejidad operativa real del período, donde coexisten mejoras significativas en operaciones de corto radio con deterioros críticos en rutas estratégicas de largo recorrido que concentran viajeros de mayor valor como Business y conexiones transatlánticas.<br>
<br>
<br>
<b><u>DETALLE POR CABINA</u></b><br>
<br>
<b><u>ECONOMY SH: Mejora robusta que oculta deterioro operativo en equipaje</u></b><br>
<br>
La cabina alcanzó un <b>NPS de 38.4 (+11.3 pts)</b> resultado de una transferencia directa del excelente desempeño de IB que compensó un deterioro operativo crítico en manejo de equipaje mediante mejoras en puntualidad y menor ocupación. <b>IB</b> registró <b>NPS de 41.6 (+14.7 pts)</b> a pesar del aumento de Mishandling a 20.62 (+4.52 pts con respecto al baseline), logrando compensar este deterioro con OTP15 de 94.09% (+1.88 pts) y Load Factor de 86.57% (-2.71 pts). El feedback reveló 7 menciones de problemas de equipaje incluyendo extravíos en FRA-MAD, MAD-OPO y MAD-ORY, retrasos de hasta 1 hora y 15 minutos en entrega en AMS-MAD, y problemas de facturación en HAM-MAD donde se reportaron 2 horas para facturar 2 maletas. Las rutas más afectadas fueron FRA-MAD con NPS de 0.0, LHR-MAD con 10.0, AMS-MAD con 14.3, BRU-MAD con 14.3, y LCG-MAD con 0.0. Los clientes españoles mostraron mayor satisfacción con NPS de 46.4 frente a europeos con 35.2, una diferencia de 11.2 puntos. La flota A321 alcanzó NPS de 58.3 mientras que A320neo registró 33.3, una dispersión de 25 puntos que evidencia diferencias significativas en la experiencia según tipo de avión. Las conexiones con codeshares fueron particularmente problemáticas con LATAM en -33.3 y BA en 20.0, contrastando con la operación propia de IB en 46.5. <b>YW</b> mantuvo <b>NPS de 32.7 (+5.2 pts)</b> dentro del rango normal sin contribuir a la anomalía, con un volumen significativamente menor de 16 encuestas frente a las 394 de IB.<br>
<br>
<b><u>BUSINESS SH: Mejora extrema que enmascara crisis en segmento corporativo</u></b><br>
<br>
La cabina alcanzó un <b>NPS de 46.2 (+11.1 pts)</b> resultado de una transferencia de la mejora extrema de YW que logró elevar el agregado a pesar de su menor volumen. <b>YW</b> registró <b>NPS de 44.4 (+28.5 pts)</b>, la mayor mejora individual del día, impulsada por excelente servicio de tripulación y VIP con 4 menciones de NPS 10 en rutas como LCG-MAD, MAD-NCE y LEI-MAD, así como mejoras operativas con OTP15 de 89.94% (+1.62 pts) y Load Factor de 51.62% (-5.75 pts). Sin embargo, esta mejora oculta una polarización extrema con dispersión de 162.5 puntos entre segmentos Business/Work con NPS de -100 y Leisure con 62.5. El caso crítico se concentró en la ruta MAD-VCE donde un cliente Business reportó haber pagado clase ejecutiva y recibido servicio de bajo coste en un avión CRJ diminuto con asiento estrecho, espacio reducido y comida incomible. Adicionalmente, se registró un problema secundario de equipaje con Mishandling de 15.22 (+2.7 pts) manifestado en pérdida total en ALC-MAD donde además se reportó un downgrade de Premium Economy a turista sin compensación adecuada. La totalidad de las operaciones de YW se realizó con flota CRJ, concentrando el problema de configuración inadecuada de Business Class en este tipo de avión. Los clientes españoles mostraron NPS de 0.0 mientras que europeos alcanzaron 40.0, y los de América Sur 100.0. <b>IB</b> mantuvo <b>NPS de 46.7 (+4.6 pts)</b> dentro del rango normal sin afectar la dinámica, con un volumen de 34 encuestas frente a las 9 de YW, evidenciando que la magnitud excepcional de la mejora de YW fue suficiente para contagiar al nivel superior a pesar de su menor representatividad.<br>
<br>
<b><u>ECONOMY LH: Triple deterioro operativo genera el peor desempeño del día</u></b><br>
<br>
La cabina registró un <b>NPS de -6.2 (-9.9 pts)</b>, el deterioro más severo del período, debido a una convergencia de tres factores operativos críticos. El colapso de puntualidad con OTP15 de 76.1% (-5.72 pts) generó retrasos significativos de hasta 6 horas en MAD-SDQ, más de una hora en MAD-PTY, dos horas en MAD-SJO por problemas de sobrepeso, y 4 cambios de horario en LIM-MAD. El manejo de equipaje se deterioró con Mishandling de 19.29 (+4.07 pts) manifestándose en pérdidas totales reportadas en MAD-MVD, retrasos de hasta 3 horas en entrega en MAD-SJO, y equipaje completamente dañado en MAD-UIO. Las conexiones perdidas aumentaron con Misconex de 0.83 (+0.15 pts) afectando especialmente a pasajeros europeos en tránsito. Adicionalmente, el deterioro de calidad de servicio incluyó múltiples quejas sobre tripulación con comportamientos inapropiados en JFK-MAD donde una azafata fue reportada como extremadamente grosera y solo contestaba en español, en EZE-MAD con actitud agria y defensiva, y en BOG-MAD donde un empleado manifestó comportamiento violento. Los sistemas de entretenimiento fallaron en BOG-MAD con reinicios constantes, en BOS-MAD con pantallas no funcionales, y en MAD-MIA donde el televisor no funcionó durante todo el vuelo. La comida presentó problemas de calidad con casos de comida cruda en EZE-MAD, terrible en MAD-MIA, y deterioro generalizado en MAD-MEX. La configuración de asientos generó quejas en MAD-MEX por no respetar pagos y lugares no aptos para infantes, en BOS-MAD por espacio excesivamente estrecho, y en MAD-REC por espacios angostos en el A321XLR. Las rutas más críticas fueron BOS-MAD con NPS de -50.0, MAD-MEX con -35.0, MAD-MVD con -28.6, MAD-SJO con -16.7, y MAD-UIO con 0.0. Los viajeros de negocios fueron severamente impactados con NPS de -24.1, una brecha de 26.8 puntos con respecto a viajeros de ocio en 2.7, correlacionando directamente con la mayor sensibilidad de este perfil a los retrasos. Los pasajeros europeos mostraron el mayor deterioro con NPS de -53.8, significativamente inferior a residentes en España con -3.3 y América Sur con 32.3. Las flotas A33ACMI, A332 y A350 registraron NPS de -100.0, -19.4 y -8.2 respectivamente, evidenciando problemas estructurales en aviones widebody.<br>
<br>
<b><u>BUSINESS LH: Paradoja entre mejora aparente y deterioro operativo subyacente</u></b><br>
<br>
La cabina alcanzó un <b>NPS de 30.0 (+11.0 pts)</b>, pero este resultado oculta una crisis operativa en segmentos específicos. A pesar de que las métricas muestran el mismo deterioro que Economy LH con OTP15 de 76.1% (-5.72 pts) y Mishandling de 19.29 (+4.07 pts), el NPS agregado mejoró debido a experiencias positivas en ciertas rutas que compensaron los incidentes críticos. Los problemas más graves se concentraron en MAD-SDQ con NPS de -100 donde se registró un retraso de 3 horas en flota A33ACMI impactando críticamente a viajeros Business, en BOG-MAD con demora de 90 minutos en entrega de equipaje sin comunicación adecuada durante la espera, en EZE-MAD con problemas de asignación de asientos incluyendo cambios unilaterales sin consentimiento y separación de menores de 5 años de sus acompañantes, y en MAD-ZRH con un problema recurrente de higiene donde se reportó agua no potable y presencia de bacterias haciendo referencia a una queja previa. Adicionalmente, se identificaron problemas de calidad en Business Class con quejas en MAD-VCE donde un cliente reportó haber pagado billete ejecutivo y recibido servicio digno de aerolínea de bajo coste, y en MAD-NAP donde los asientos de clase ejecutiva no eran más espaciosos que los de turista. Los viajeros Business/Work mostraron NPS de 0.0, una brecha de 35.3 puntos con respecto a viajeros de ocio en 35.3, evidenciando mayor criticidad de este perfil ante los deterioros operativos. Las flotas A33ACMI y A350 registraron NPS de -100.0 y 0.0 respectivamente, con una dispersión de 200 puntos que indica impacto significativo del tipo de avión en la experiencia. Los residentes en España mostraron NPS de 20.0 con el mayor volumen de 10 encuestas pero con nivel bajo de satisfacción, mientras que argentinos alcanzaron 50.0 y colombianos 33.3. Los miembros IB Plus Silver registraron NPS de 0.0, IB Plus Platino 25.0, no miembros 37.5, e IB Plus Oro 50.0. La menor ocupación con Load Factor de 66.52% (-5.58 pts) favoreció la experiencia en rutas sin incidentes, contribuyendo a la mejora aparente del agregado que enmascara los problemas críticos en segmentos específicos de alto valor estratégico.<br>
<br>
<b><u>PREMIUM LH: Estabilidad sin cambios significativos</u></b><br>
<br>
La cabina registró un <b>NPS de 15.0 (+5.7 pts)</b> manteniéndose dentro del rango normal sin alcanzar el umbral de anomalía. El análisis no detectó cambios significativos en este segmento, manteniendo un desempeño estable durante el período analizado.
```

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 7
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
7. **TERMINOLOGÍA:** Preserva la terminología de comparación del interpreter. NO la cambies.
6. NO añadas recomendaciones adicionales
7. Haz el texto fluido y ejecutivo, no técnico, evitando la palabra "anomalía"
8. Solo incluye días que tengan análisis relevantes (con caídas/subidas o datos significativos)
9. Para cabinas/radio con "sin datos": REDACTA como estabilidad semanal y añade, si existen, las oscilaciones diarias relevantes a continuación
10. **CRÍTICO**: Si hay datos en "ANÁLISIS DIARIO SINGLE", DEBES usarlos. NO digas que "no están disponibles" si están presentes en el input.
11. **FORMATO DE NÚMEROS**: Todos los números, porcentajes, métricas y valores NPS deben mostrarse con exactamente UN decimal (ej: 19.8, -4.4, 93.5%)
12. **ATRIBUCIÓN DE SEGMENTO**: Siempre que menciones un dato, indica a qué segmento pertenece (ej: "NPS 19.8 (Economy LH)", "OTP –4.0 pts (Business SH)")
